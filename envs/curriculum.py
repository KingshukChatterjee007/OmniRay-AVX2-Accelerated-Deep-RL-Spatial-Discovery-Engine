"""
OmniRay Self-Adaptive Autonomy — Layer 4: Curriculum (Self-Difficulty)
======================================================================

Automatically adjusts environment difficulty based on the agent's
sustained performance. Keeps the environment at the edge of the agent's
capability — not too easy (no learning signal), not too hard (catastrophic).

Managed parameters:
  - num_obstacles:  Number of internal wall segments (2–20)
  - arena_size:     Size of the square arena (60–200 units)
  - noise_scale:    Multiplier on base sensor/actuator noise (0.0–2.0)
  - max_steps:      Episode length budget (100–500)

Trigger logic:
  If rolling_avg_coverage > threshold_up for N episodes → increase difficulty
  If rolling_avg_coverage < threshold_down for N episodes → decrease difficulty

Usage:
    curriculum = CurriculumManager(config)
    
    # At episode boundaries:
    params = curriculum.evaluate_and_adjust(episode_coverage)
    env.set_difficulty_params(**params)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DifficultyParams:
    """Current environment difficulty parameters."""
    num_obstacles: int = 6
    arena_size: float = 100.0
    noise_scale: float = 1.0
    max_steps: int = 200

    def to_dict(self) -> dict:
        return {
            "num_obstacles": self.num_obstacles,
            "arena_size": self.arena_size,
            "noise_scale": self.noise_scale,
            "max_steps": self.max_steps,
        }


class CurriculumManager:
    """
    Auto-difficulty system that keeps the environment challenging
    but achievable based on rolling agent performance.
    
    Args:
        coverage_threshold_up:    Coverage above which difficulty increases (default 0.90)
        coverage_threshold_down:  Coverage below which difficulty decreases (default 0.40)
        evaluation_window:        Number of episodes for rolling average (default 5)
        obstacle_range:           Min/max obstacles (default [2, 20])
        arena_range:              Min/max arena size (default [60.0, 200.0])
        noise_range:              Min/max noise multiplier (default [0.0, 2.0])
        steps_range:              Min/max episode steps (default [100, 500])
        initial_obstacles:        Starting obstacle count (default 6)
        initial_arena:            Starting arena size (default 100.0)
        initial_noise:            Starting noise scale (default 1.0)
        initial_steps:            Starting max steps (default 200)
    """

    def __init__(
        self,
        coverage_threshold_up: float = 0.90,
        coverage_threshold_down: float = 0.40,
        evaluation_window: int = 5,
        obstacle_range: tuple[int, int] = (2, 20),
        arena_range: tuple[float, float] = (60.0, 200.0),
        noise_range: tuple[float, float] = (0.0, 2.0),
        steps_range: tuple[int, int] = (100, 500),
        initial_obstacles: int = 6,
        initial_arena: float = 100.0,
        initial_noise: float = 1.0,
        initial_steps: int = 200,
    ):
        self.threshold_up = coverage_threshold_up
        self.threshold_down = coverage_threshold_down
        self.eval_window = evaluation_window

        # Parameter ranges
        self.obstacle_range = obstacle_range
        self.arena_range = arena_range
        self.noise_range = noise_range
        self.steps_range = steps_range

        # Current difficulty state
        self.params = DifficultyParams(
            num_obstacles=initial_obstacles,
            arena_size=initial_arena,
            noise_scale=initial_noise,
            max_steps=initial_steps,
        )

        # Episode tracking
        self._coverage_history: list[float] = []
        self._difficulty_level: int = 0  # Tracks net difficulty changes
        self._total_increases: int = 0
        self._total_decreases: int = 0
        self._last_change_episode: int = 0
        self._episode_count: int = 0
        self._changed_this_eval: bool = False

    def evaluate_and_adjust(self, episode_final_coverage: float) -> DifficultyParams:
        """
        Evaluate agent performance and adjust difficulty if needed.
        
        Called at the end of each episode with the final coverage ratio.
        
        Args:
            episode_final_coverage: Final coverage ratio [0, 1] of the completed episode
            
        Returns:
            Current DifficultyParams (potentially adjusted)
        """
        self._episode_count += 1
        self._coverage_history.append(episode_final_coverage)
        self._changed_this_eval = False

        # Need at least eval_window episodes before adjusting
        if len(self._coverage_history) < self.eval_window:
            return self.params

        # Compute rolling average over the evaluation window
        recent = self._coverage_history[-self.eval_window:]
        avg_coverage = np.mean(recent)

        # Minimum cooldown: don't change difficulty more than once every eval_window episodes
        episodes_since_change = self._episode_count - self._last_change_episode
        if episodes_since_change < self.eval_window:
            return self.params

        # Decision: increase or decrease difficulty
        if avg_coverage > self.threshold_up:
            self._increase_difficulty()
        elif avg_coverage < self.threshold_down:
            self._decrease_difficulty()

        return self.params

    def _increase_difficulty(self):
        """Make the environment harder."""
        changed = False

        # Add obstacles
        new_obs = min(self.params.num_obstacles + 2, self.obstacle_range[1])
        if new_obs != self.params.num_obstacles:
            self.params.num_obstacles = new_obs
            changed = True

        # Expand arena
        new_arena = min(self.params.arena_size + 10.0, self.arena_range[1])
        if new_arena != self.params.arena_size:
            self.params.arena_size = new_arena
            changed = True

        # Increase noise
        new_noise = min(self.params.noise_scale + 0.25, self.noise_range[1])
        if new_noise != self.params.noise_scale:
            self.params.noise_scale = new_noise
            changed = True

        # Optionally reduce step budget to force efficiency
        # Only reduce if agent is doing very well
        recent_avg = np.mean(self._coverage_history[-self.eval_window:])
        if recent_avg > 0.95:
            new_steps = max(self.params.max_steps - 25, self.steps_range[0])
            if new_steps != self.params.max_steps:
                self.params.max_steps = new_steps
                changed = True

        if changed:
            self._difficulty_level += 1
            self._total_increases += 1
            self._last_change_episode = self._episode_count
            self._changed_this_eval = True

    def _decrease_difficulty(self):
        """Make the environment easier."""
        changed = False

        # Remove obstacles
        new_obs = max(self.params.num_obstacles - 2, self.obstacle_range[0])
        if new_obs != self.params.num_obstacles:
            self.params.num_obstacles = new_obs
            changed = True

        # Shrink arena
        new_arena = max(self.params.arena_size - 10.0, self.arena_range[0])
        if new_arena != self.params.arena_size:
            self.params.arena_size = new_arena
            changed = True

        # Reduce noise
        new_noise = max(self.params.noise_scale - 0.25, self.noise_range[0])
        if new_noise != self.params.noise_scale:
            self.params.noise_scale = new_noise
            changed = True

        # Give more steps
        new_steps = min(self.params.max_steps + 25, self.steps_range[1])
        if new_steps != self.params.max_steps:
            self.params.max_steps = new_steps
            changed = True

        if changed:
            self._difficulty_level -= 1
            self._total_decreases += 1
            self._last_change_episode = self._episode_count
            self._changed_this_eval = True

    @property
    def difficulty_score(self) -> float:
        """
        Normalized difficulty score (0.0 = easiest, 1.0 = hardest).
        
        Computed from the current parameter positions within their ranges.
        """
        # Normalize each parameter to [0, 1] within its range
        obs_norm = (self.params.num_obstacles - self.obstacle_range[0]) / max(
            self.obstacle_range[1] - self.obstacle_range[0], 1
        )
        arena_norm = (self.params.arena_size - self.arena_range[0]) / max(
            self.arena_range[1] - self.arena_range[0], 1.0
        )
        noise_norm = (self.params.noise_scale - self.noise_range[0]) / max(
            self.noise_range[1] - self.noise_range[0], 1.0
        )
        # Steps: fewer steps = harder, so invert
        steps_norm = 1.0 - (self.params.max_steps - self.steps_range[0]) / max(
            self.steps_range[1] - self.steps_range[0], 1
        )

        return float(np.mean([obs_norm, arena_norm, noise_norm, steps_norm]))

    @property
    def changed(self) -> bool:
        """Whether difficulty was changed in the last evaluate_and_adjust call."""
        return self._changed_this_eval

    def get_stats(self) -> dict:
        """Get curriculum statistics for logging."""
        return {
            "curriculum_difficulty": self.difficulty_score,
            "curriculum_level": self._difficulty_level,
            "curriculum_increases": self._total_increases,
            "curriculum_decreases": self._total_decreases,
            "curriculum_obstacles": self.params.num_obstacles,
            "curriculum_arena_size": self.params.arena_size,
            "curriculum_noise_scale": self.params.noise_scale,
            "curriculum_max_steps": self.params.max_steps,
            "curriculum_changed": self._changed_this_eval,
        }
