"""
OmniRay Self-Adaptive Autonomy — Layer 5: Continual Learner
=============================================================

Enables in-deployment learning so the agent never stops improving.
After initial training is complete, this module:

  1. Records episode trajectories in a circular replay buffer
  2. Periodically retrains the PPO policy on recent experience
  3. Checkpoints before retraining for safety rollback
  4. Gates retraining on performance degradation to avoid wasting compute

Usage:
    learner = ContinualLearner(config)
    
    # After each episode:
    learner.record_episode(episode_data)
    learner.maybe_retrain(model, env)
"""

import os
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeRecord:
    """Stores key metrics from a completed episode."""
    total_reward: float = 0.0
    final_coverage: float = 0.0
    steps: int = 0
    timestamp: float = 0.0
    difficulty_score: float = 0.0


class ContinualLearner:
    """
    Post-training continual learning system.
    
    Records episode outcomes and periodically retrains the policy
    on recent experience to keep improving in deployment.
    
    Args:
        replay_buffer_size:   Max episodes to store (default 50)
        retrain_interval:     Episodes between retrain attempts (default 10)
        retrain_steps:        PPO timesteps per retrain cycle (default 2048)
        max_checkpoints:      Number of checkpoint files to keep (default 3)
        performance_gate:     Performance drop % that triggers retrain (default 0.10)
        rollback_threshold:   Performance drop % post-retrain that triggers rollback (default 0.15)
        checkpoint_dir:       Directory for checkpoint files (default "checkpoints")
    """

    def __init__(
        self,
        replay_buffer_size: int = 50,
        retrain_interval: int = 10,
        retrain_steps: int = 2048,
        max_checkpoints: int = 3,
        performance_gate: float = 0.10,
        rollback_threshold: float = 0.15,
        checkpoint_dir: str = "checkpoints",
    ):
        self.buffer_size = replay_buffer_size
        self.retrain_interval = retrain_interval
        self.retrain_steps = retrain_steps
        self.max_checkpoints = max_checkpoints
        self.performance_gate = performance_gate
        self.rollback_threshold = rollback_threshold
        self.checkpoint_dir = checkpoint_dir

        # Episode replay buffer (circular)
        self._buffer: deque[EpisodeRecord] = deque(maxlen=replay_buffer_size)
        self._episode_count: int = 0
        self._peak_reward: float = float("-inf")
        self._checkpoint_paths: deque[str] = deque(maxlen=max_checkpoints)

        # Retrain tracking
        self._total_retrains: int = 0
        self._total_rollbacks: int = 0
        self._last_retrain_episode: int = 0
        self._curriculum_changed: bool = False

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def reset(self):
        """Reset per-deployment state (keeps buffer and learned history)."""
        self._curriculum_changed = False

    def notify_curriculum_change(self):
        """
        Called when the curriculum manager changes difficulty.
        This signals that a retrain may be warranted even if
        performance hasn't dropped yet.
        """
        self._curriculum_changed = True

    def record_episode(
        self,
        total_reward: float,
        final_coverage: float,
        steps: int,
        difficulty_score: float = 0.0,
    ):
        """
        Record a completed episode's key metrics.
        
        Args:
            total_reward:     Cumulative reward over the episode
            final_coverage:   Final coverage ratio [0, 1]
            steps:            Number of steps taken
            difficulty_score: Current curriculum difficulty (0-1)
        """
        self._episode_count += 1

        record = EpisodeRecord(
            total_reward=total_reward,
            final_coverage=final_coverage,
            steps=steps,
            timestamp=time.time(),
            difficulty_score=difficulty_score,
        )
        self._buffer.append(record)

        # Update peak performance (adjusted for difficulty)
        # Higher difficulty should allow lower raw reward to still be "peak"
        adjusted_reward = total_reward / max(1.0 - difficulty_score * 0.3, 0.5)
        if adjusted_reward > self._peak_reward:
            self._peak_reward = adjusted_reward

    def should_retrain(self) -> bool:
        """
        Determine if a retrain cycle should be triggered.
        
        Returns True if:
          - We've accumulated enough episodes since last retrain
          - AND (performance has degraded OR curriculum changed)
        """
        # Check interval
        episodes_since = self._episode_count - self._last_retrain_episode
        if episodes_since < self.retrain_interval:
            return False

        # Need at least some episodes in buffer
        if len(self._buffer) < self.retrain_interval:
            return False

        # Check if curriculum recently changed (always retrain after difficulty shift)
        if self._curriculum_changed:
            return True

        # Check performance gate: has recent performance dropped?
        recent_rewards = [ep.total_reward for ep in list(self._buffer)[-self.retrain_interval:]]
        mean_recent = np.mean(recent_rewards)

        if self._peak_reward > 0:
            drop_ratio = 1.0 - (mean_recent / self._peak_reward)
        else:
            drop_ratio = 0.0

        return drop_ratio > self.performance_gate

    def maybe_retrain(self, model, env) -> dict:
        """
        Check if retraining is needed and execute if so.
        
        Args:
            model: The SB3 PPO model instance
            env:   The Gymnasium environment (or wrapper)
            
        Returns:
            Dict with retrain results (or empty if no retrain occurred)
        """
        if not self.should_retrain():
            return {"retrained": False}

        # Record pre-retrain performance
        recent_rewards = [ep.total_reward for ep in list(self._buffer)[-self.retrain_interval:]]
        pre_retrain_mean = float(np.mean(recent_rewards))

        # 1. Save checkpoint
        checkpoint_name = f"checkpoint_ep{self._episode_count}_{int(time.time())}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        model.save(checkpoint_path)
        self._checkpoint_paths.append(checkpoint_path)

        # Clean up old checkpoints beyond max_checkpoints
        while len(self._checkpoint_paths) > self.max_checkpoints:
            old_path = self._checkpoint_paths.popleft()
            old_file = f"{old_path}.zip"
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except OSError:
                    pass

        # 2. Retrain on the current environment
        try:
            model.learn(
                total_timesteps=self.retrain_steps,
                reset_num_timesteps=False,  # Continue from current timestep counter
            )
            self._total_retrains += 1
            self._last_retrain_episode = self._episode_count
            self._curriculum_changed = False

            result = {
                "retrained": True,
                "retrain_episode": self._episode_count,
                "pre_retrain_reward": pre_retrain_mean,
                "checkpoint_path": checkpoint_path,
                "total_retrains": self._total_retrains,
                "rolled_back": False,
            }

        except Exception as e:
            # If retrain fails, rollback to checkpoint
            result = {
                "retrained": False,
                "error": str(e),
                "checkpoint_path": checkpoint_path,
            }

        return result

    def evaluate_post_retrain(self, model, env, eval_episodes: int = 3) -> bool:
        """
        Run a quick evaluation after retraining to check for degradation.
        If performance dropped too much, triggers a rollback.
        
        Args:
            model:          The retrained SB3 PPO model
            env:            The environment
            eval_episodes:  Number of episodes to evaluate (default 3)
            
        Returns:
            True if performance is acceptable, False if rollback was triggered
        """
        rewards = []
        for _ in range(eval_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)

        post_mean = float(np.mean(rewards))

        # Check against pre-retrain buffer performance
        if len(self._buffer) >= self.retrain_interval:
            recent_pre = [ep.total_reward for ep in list(self._buffer)[-self.retrain_interval:]]
            pre_mean = float(np.mean(recent_pre))

            if pre_mean > 0:
                drop = 1.0 - (post_mean / pre_mean)
                if drop > self.rollback_threshold:
                    # Rollback!
                    return self._rollback(model)

        return True

    def _rollback(self, model) -> bool:
        """
        Rollback to the most recent checkpoint.
        
        Returns:
            True if rollback succeeded, False otherwise
        """
        if len(self._checkpoint_paths) == 0:
            return False

        latest_checkpoint = self._checkpoint_paths[-1]
        checkpoint_file = f"{latest_checkpoint}.zip"

        if os.path.exists(checkpoint_file):
            try:
                from stable_baselines3 import PPO
                # Load checkpoint weights back into the model
                loaded = PPO.load(checkpoint_file)
                model.policy.load_state_dict(loaded.policy.state_dict())
                self._total_rollbacks += 1
                return True
            except Exception:
                return False

        return False

    def get_stats(self) -> dict:
        """Get continual learning statistics."""
        recent_rewards = (
            [ep.total_reward for ep in list(self._buffer)[-10:]]
            if len(self._buffer) > 0
            else [0.0]
        )

        return {
            "continual_episodes_recorded": self._episode_count,
            "continual_buffer_size": len(self._buffer),
            "continual_total_retrains": self._total_retrains,
            "continual_total_rollbacks": self._total_rollbacks,
            "continual_peak_reward": self._peak_reward,
            "continual_recent_mean_reward": float(np.mean(recent_rewards)),
            "continual_should_retrain": self.should_retrain(),
        }
