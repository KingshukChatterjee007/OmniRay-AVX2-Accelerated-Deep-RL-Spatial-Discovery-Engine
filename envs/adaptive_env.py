"""
OmniRay Self-Adaptive Autonomy — Orchestration Wrapper
========================================================

A Gymnasium Wrapper that composes all 5 self-adaptive layers around
the base ActiveSLAMEnv:

  Layer 1: Health Monitor      → Self-awareness (health scoring)
  Layer 2: Adaptive Reward     → Dynamic reward adjustment
  Layer 3: Meta-Policy         → Learned reward weight optimization
  Layer 4: Curriculum          → Auto-difficulty scaling
  Layer 5: Continual Learner   → In-deployment replay + retrain

This wrapper is the SINGLE entry point for adaptive training.
All layers are independently toggleable via config.

Step Flow (one timestep):
    1. base_env.step(action) → base reward + info
    2. health_monitor.update(info) → health score
    3. meta_policy.predict(health) → weight overrides (if enabled)
    4. adaptive_reward.compute(base, health, overrides) → adjusted reward
    5. return (obs, adjusted_reward, terminated, truncated, enriched_info)

Reset Flow (episode boundary):
    1. curriculum.evaluate_and_adjust(episode_stats)
    2. Apply new difficulty params to base env
    3. base_env.reset()
    4. continual_learner.record_episode + maybe_retrain
    5. health_monitor.reset()

Usage:
    from envs.adaptive_env import AdaptiveActiveSLAMEnv
    
    base_env = ActiveSLAMEnv(backend='numpy', ...)
    env = AdaptiveActiveSLAMEnv(base_env, adaptive_config)
    obs, info = env.reset()
"""

import numpy as np
import gymnasium as gym
from typing import Optional

from envs.health_monitor import HealthMonitor
from envs.adaptive_reward import AdaptiveRewardEngine, RewardComponents
from envs.curriculum import CurriculumManager


class AdaptiveActiveSLAMEnv(gym.Wrapper):
    """
    Gymnasium Wrapper composing 5 self-adaptive layers around ActiveSLAMEnv.
    
    Args:
        env:              Base ActiveSLAMEnv instance
        config:           Adaptive config dict (from config.yaml 'adaptive' section)
        enable_meta:      Enable Layer 3 meta-policy (default False)
        enable_curriculum: Enable Layer 4 curriculum (default False)
        enable_continual: Enable Layer 5 continual learning (default False)
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[dict] = None,
        enable_meta: bool = False,
        enable_curriculum: bool = False,
        enable_continual: bool = False,
    ):
        super().__init__(env)
        self.config = config or {}

        # Feature flags
        self._meta_enabled = enable_meta
        self._curriculum_enabled = enable_curriculum
        self._continual_enabled = enable_continual

        # -----------------------------------------------------------
        # Layer 1: Health Monitor (always active)
        # -----------------------------------------------------------
        hm_config = self.config.get("health_monitor", {})
        self.health_monitor = HealthMonitor(
            ema_alpha=hm_config.get("ema_alpha", 0.05),
            entropy_weight=hm_config.get("entropy_weight", 0.35),
            coverage_weight=hm_config.get("coverage_weight", 0.40),
            slam_weight=hm_config.get("slam_weight", 0.25),
            window_size=hm_config.get("window_size", 50),
        )

        # -----------------------------------------------------------
        # Layer 2: Adaptive Reward Engine (always active)
        # -----------------------------------------------------------
        ar_config = self.config.get("adaptive_reward", {})
        self.adaptive_reward = AdaptiveRewardEngine(
            base_exploration=getattr(env, "reward_exploration", 1.0),
            base_frontier=getattr(env, "reward_frontier", 0.1),
            base_time_penalty=getattr(env, "reward_time_penalty", 0.01),
            base_collision_penalty=getattr(env, "reward_collision_penalty", 0.1),
            stuck_frontier_boost=ar_config.get("stuck_frontier_boost", 2.0),
            low_entropy_curiosity=ar_config.get("low_entropy_curiosity", 0.2),
            high_health_explore_reduction=ar_config.get("high_health_explore_reduction", 0.5),
            lost_safety_penalty=ar_config.get("lost_safety_penalty", 0.3),
            collision_rate_threshold=ar_config.get("collision_rate_threshold", 0.2),
            collision_penalty_boost=ar_config.get("collision_penalty_boost", 3.0),
        )

        # -----------------------------------------------------------
        # Layer 3: Meta-Policy (optional)
        # -----------------------------------------------------------
        self.meta_policy = None
        if self._meta_enabled:
            try:
                from envs.meta_policy import MetaPolicy
                mp_config = self.config.get("meta_policy", {})
                self.meta_policy = MetaPolicy(
                    learning_rate=mp_config.get("learning_rate", 1e-4),
                    update_interval=mp_config.get("update_interval", 200),
                    warmup_steps=mp_config.get("warmup_steps", 1000),
                    scale_clamp=tuple(mp_config.get("scale_clamp", [0.1, 5.0])),
                )
            except ImportError as e:
                print(f"  [WARNING] Meta-policy disabled: {e}")
                self._meta_enabled = False

        # -----------------------------------------------------------
        # Layer 4: Curriculum Manager (optional)
        # -----------------------------------------------------------
        self.curriculum = None
        if self._curriculum_enabled:
            cur_config = self.config.get("curriculum", {})
            self.curriculum = CurriculumManager(
                coverage_threshold_up=cur_config.get("coverage_threshold_up", 0.90),
                coverage_threshold_down=cur_config.get("coverage_threshold_down", 0.40),
                evaluation_window=cur_config.get("evaluation_window", 5),
                obstacle_range=tuple(cur_config.get("obstacle_range", [2, 20])),
                arena_range=tuple(cur_config.get("arena_range", [60.0, 200.0])),
                noise_range=tuple(cur_config.get("noise_range", [0.0, 2.0])),
                steps_range=tuple(cur_config.get("steps_range", [100, 500])),
                initial_obstacles=getattr(env, "num_obstacles", 6),
                initial_arena=getattr(env, "arena_size", 100.0),
                initial_noise=1.0,
                initial_steps=getattr(env, "max_steps", 200),
            )

        # -----------------------------------------------------------
        # Layer 5: Continual Learner (optional)
        # -----------------------------------------------------------
        self.continual_learner = None
        if self._continual_enabled:
            from envs.continual_learner import ContinualLearner
            cl_config = self.config.get("continual_learning", {})
            self.continual_learner = ContinualLearner(
                replay_buffer_size=cl_config.get("replay_buffer_size", 50),
                retrain_interval=cl_config.get("retrain_interval", 10),
                retrain_steps=cl_config.get("retrain_steps", 2048),
                max_checkpoints=cl_config.get("max_checkpoints", 3),
                performance_gate=cl_config.get("performance_gate", 0.10),
                rollback_threshold=cl_config.get("rollback_threshold", 0.15),
            )

        # Episode tracking
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._episode_count: int = 0
        self._policy_entropy: float = 1.0  # Updated externally via callback
        self._model_ref = None  # Set by training script for continual learning

        # Previous frontier distance for frontier delta calculation
        self._prev_frontier_dist: float = 0.0

    def set_model(self, model):
        """
        Set the SB3 model reference for continual learning and entropy access.
        Called by the training script after model creation.
        """
        self._model_ref = model

    def set_policy_entropy(self, entropy: float):
        """
        Update the current policy entropy value.
        Called by the AdaptiveCallback during training.
        """
        self._policy_entropy = entropy

    def step(self, action):
        """
        Adaptive step: base step → health → meta → reward adjustment.
        """
        # 1. Execute base environment step
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1

        # Extract components for adaptive reward computation
        components = RewardComponents(
            new_cells=info.get("new_cells", 0),
            frontier_delta=self._compute_frontier_delta(info),
            collision=info.get("collision", False),
            entropy_value=self._policy_entropy,
        )

        # 2. Health Monitor update
        slam_weights = None
        if hasattr(self.env, "use_slam") and self.env.use_slam and hasattr(self.env, "slam"):
            slam_weights = self.env.slam.weights

        health = self.health_monitor.update(
            coverage_ratio=info.get("coverage", 0.0),
            slam_weights=slam_weights,
            policy_entropy=self._policy_entropy,
        )

        # 3. Meta-Policy prediction (if enabled)
        weight_overrides = None
        if self._meta_enabled and self.meta_policy is not None:
            steps_fraction = self._episode_steps / max(getattr(self.env, "max_steps", 200), 1)
            weight_overrides = self.meta_policy.predict(health, steps_fraction)
            if not weight_overrides:  # Empty dict means still in warmup
                weight_overrides = None

            # Feed health back to meta-policy for REINFORCE update
            self.meta_policy.record_health(health.score)

        # 4. Adaptive Reward computation
        adjusted_reward, active_weights = self.adaptive_reward.compute(
            components=components,
            health=health,
            weight_overrides=weight_overrides,
        )

        # Track episode reward
        self._episode_reward += adjusted_reward

        # 5. Enrich info dict with adaptive diagnostics
        info["health_score"] = health.score
        info["health_entropy"] = health.entropy_health
        info["health_coverage"] = health.coverage_health
        info["health_slam"] = health.slam_health
        info["coverage_velocity"] = health.coverage_velocity
        info["is_failing"] = health.is_failing
        info["reward_weights"] = active_weights.to_dict()
        info["adaptive_reward"] = adjusted_reward
        info["base_reward"] = base_reward

        if self._meta_enabled and self.meta_policy is not None:
            info["meta_policy_active"] = self.meta_policy.is_active

        if self._curriculum_enabled and self.curriculum is not None:
            info["curriculum_difficulty"] = self.curriculum.difficulty_score

        return obs, adjusted_reward, terminated, truncated, info

    def _compute_frontier_delta(self, info: dict) -> float:
        """Compute change in frontier distance for reward shaping."""
        # The base env already computes frontier distance internally,
        # but we need the delta for the adaptive reward engine.
        # We approximate from the base env's internal state.
        if hasattr(self.env, "_prev_frontier_dist"):
            current = getattr(self.env, "_prev_frontier_dist", 0.0)
            # _prev_frontier_dist is already updated in the base env step,
            # so we need to track our own previous value
            delta = self._prev_frontier_dist - current
            delta = float(np.clip(delta, -5.0, 5.0))
            self._prev_frontier_dist = current
            return delta
        return 0.0

    def reset(self, *, seed=None, options=None):
        """
        Adaptive reset: curriculum adjust → base reset → record + retrain.
        """
        # Handle end-of-episode recording (skip first call)
        if self._episode_count > 0:
            self._handle_episode_end()

        self._episode_count += 1

        # 1. Curriculum adjustment (if enabled)
        if self._curriculum_enabled and self.curriculum is not None and self._episode_count > 1:
            # Use coverage from the just-completed episode
            coverage = getattr(self, "_last_episode_coverage", 0.0)
            params = self.curriculum.evaluate_and_adjust(coverage)

            # Apply difficulty changes to base env
            if self.curriculum.changed and hasattr(self.env, "set_difficulty_params"):
                self.env.set_difficulty_params(
                    num_obstacles=params.num_obstacles,
                    arena_size=params.arena_size,
                    noise_scale=params.noise_scale,
                    max_steps=params.max_steps,
                )

                # Notify continual learner of curriculum change
                if self._continual_enabled and self.continual_learner is not None:
                    self.continual_learner.notify_curriculum_change()

        # 2. Base environment reset
        obs, info = self.env.reset(seed=seed, options=options)

        # 3. Reset adaptive layers
        self.health_monitor.reset()
        self.adaptive_reward.reset()
        if self._meta_enabled and self.meta_policy is not None:
            self.meta_policy.reset()

        # Reset episode tracking
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._prev_frontier_dist = getattr(self.env, "_prev_frontier_dist", 0.0)

        # Add adaptive info to initial observation info
        info["health_score"] = 0.5
        info["curriculum_difficulty"] = (
            self.curriculum.difficulty_score
            if self._curriculum_enabled and self.curriculum is not None
            else 0.0
        )

        return obs, info

    def _handle_episode_end(self):
        """Process end-of-episode tasks: recording and potential retraining."""
        # Store final coverage for curriculum evaluation
        self._last_episode_coverage = getattr(self.env, "_total_explored", 0) / max(
            getattr(self.env, "map_res", 50) ** 2, 1
        )

        # Record episode in continual learner
        if self._continual_enabled and self.continual_learner is not None:
            difficulty = (
                self.curriculum.difficulty_score
                if self._curriculum_enabled and self.curriculum is not None
                else 0.0
            )
            self.continual_learner.record_episode(
                total_reward=self._episode_reward,
                final_coverage=self._last_episode_coverage,
                steps=self._episode_steps,
                difficulty_score=difficulty,
            )

            # Attempt retrain if conditions are met
            if self._model_ref is not None:
                retrain_result = self.continual_learner.maybe_retrain(
                    model=self._model_ref,
                    env=self.env,
                )
                if retrain_result.get("retrained", False):
                    print(f"  [CONTINUAL] Retrained at episode {self._episode_count} "
                          f"(checkpoint: {retrain_result.get('checkpoint_path', 'N/A')})")

    def get_adaptive_stats(self) -> dict:
        """
        Get comprehensive statistics from all adaptive layers.
        Useful for logging and visualization.
        """
        stats = {
            "episode_count": self._episode_count,
            "health": {
                "score": self.health_monitor.health_score,
                "is_failing": self.health_monitor.is_failing,
                "diagnostics": {
                    "entropy": self.health_monitor.diagnostics.entropy_health,
                    "coverage": self.health_monitor.diagnostics.coverage_health,
                    "slam": self.health_monitor.diagnostics.slam_health,
                    "velocity": self.health_monitor.diagnostics.coverage_velocity,
                },
            },
            "reward": {
                "weights": self.adaptive_reward.current_weights.to_dict(),
                "collision_rate": self.adaptive_reward.collision_rate,
            },
        }

        if self._meta_enabled and self.meta_policy is not None:
            stats["meta_policy"] = self.meta_policy.get_stats()

        if self._curriculum_enabled and self.curriculum is not None:
            stats["curriculum"] = self.curriculum.get_stats()

        if self._continual_enabled and self.continual_learner is not None:
            stats["continual"] = self.continual_learner.get_stats()

        return stats
