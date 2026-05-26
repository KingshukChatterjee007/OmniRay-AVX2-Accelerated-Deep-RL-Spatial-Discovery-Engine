"""
OmniRay Self-Adaptive Autonomy — Layer 2: Adaptive Reward Engine
==================================================================

Dynamically adjusts reward weights based on the agent's health state.
Can operate in two modes:

  1. Heuristic Mode (default): Uses rule-based adjustments when health
     deteriorates (stuck → boost frontier, lost → add safety penalty, etc.)
     
  2. Meta-Policy Mode: Accepts weight override multipliers from the
     Layer 3 meta-policy network, which has learned optimal weights.

The engine wraps the base reward computation and outputs a modified
reward signal each step.

Usage:
    engine = AdaptiveRewardEngine(base_weights, config)
    
    # Every step:
    adjusted_reward = engine.compute(
        base_reward_components={...},
        health=health_diagnostics,
        weight_overrides=meta_policy_output  # optional
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardComponents:
    """Raw reward components from the base environment step."""
    new_cells: int = 0
    frontier_delta: float = 0.0
    collision: bool = False
    entropy_value: float = 0.0


@dataclass
class RewardWeights:
    """Current active reward weight multipliers."""
    exploration_scale: float = 1.0
    frontier_scale: float = 1.0
    curiosity_bonus: float = 0.0
    time_penalty_scale: float = 1.0
    collision_penalty_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "exploration_scale": self.exploration_scale,
            "frontier_scale": self.frontier_scale,
            "curiosity_bonus": self.curiosity_bonus,
            "time_penalty_scale": self.time_penalty_scale,
            "collision_penalty_scale": self.collision_penalty_scale,
        }


class AdaptiveRewardEngine:
    """
    Dynamic reward modifier that adjusts weights based on agent health.
    
    Args:
        base_exploration:         Base exploration reward per cell (default 1.0)
        base_frontier:            Base frontier shaping weight (default 0.1)
        base_time_penalty:        Base time penalty per step (default 0.01)
        base_collision_penalty:   Base collision penalty (default 0.1)
        stuck_frontier_boost:     Frontier multiplier when stuck (default 2.0)
        low_entropy_curiosity:    Curiosity bonus when entropy is low (default 0.2)
        high_health_explore_reduction: Exploration reduction when healthy (default 0.5)
        lost_safety_penalty:      Extra time penalty when SLAM is lost (default 0.3)
        collision_rate_threshold: Rolling collision rate that triggers boost (default 0.2)
        collision_penalty_boost:  Collision penalty multiplier when rate is high (default 3.0)
    """

    def __init__(
        self,
        base_exploration: float = 1.0,
        base_frontier: float = 0.1,
        base_time_penalty: float = 0.01,
        base_collision_penalty: float = 0.1,
        stuck_frontier_boost: float = 2.0,
        low_entropy_curiosity: float = 0.2,
        high_health_explore_reduction: float = 0.5,
        lost_safety_penalty: float = 0.3,
        collision_rate_threshold: float = 0.2,
        collision_penalty_boost: float = 3.0,
    ):
        # Base weights (from env config)
        self.base_exploration = base_exploration
        self.base_frontier = base_frontier
        self.base_time_penalty = base_time_penalty
        self.base_collision_penalty = base_collision_penalty

        # Heuristic tuning parameters
        self.stuck_frontier_boost = stuck_frontier_boost
        self.low_entropy_curiosity = low_entropy_curiosity
        self.high_health_explore_reduction = high_health_explore_reduction
        self.lost_safety_penalty = lost_safety_penalty
        self.collision_rate_threshold = collision_rate_threshold
        self.collision_penalty_boost = collision_penalty_boost

        # Internal tracking
        self._collision_history: list[bool] = []
        self._collision_window = 50  # Rolling window for collision rate
        self._current_weights = RewardWeights()

    def reset(self):
        """Reset internal state at episode boundaries."""
        self._collision_history = []
        self._current_weights = RewardWeights()

    def _compute_heuristic_weights(self, health) -> RewardWeights:
        """
        Apply rule-based reward weight adjustments based on health state.
        
        Args:
            health: HealthDiagnostics from the health monitor
            
        Returns:
            RewardWeights with adjusted multipliers
        """
        weights = RewardWeights()

        # Rule 1: If coverage velocity is near zero (stuck), boost frontier
        if health.coverage_velocity < 0.001 and health.coverage_health < 0.4:
            weights.frontier_scale = self.stuck_frontier_boost
            # Also slightly boost exploration to incentivize any new cells
            weights.exploration_scale = 1.3

        # Rule 2: If entropy is too low (policy not exploring enough), add curiosity
        if health.entropy_health < 0.3:
            weights.curiosity_bonus = self.low_entropy_curiosity
            # Also boost frontier to push toward unexplored
            weights.frontier_scale = max(weights.frontier_scale, 1.5)

        # Rule 3: If doing really well (health > 0.8), focus on efficiency
        if health.score > 0.8:
            weights.exploration_scale *= self.high_health_explore_reduction
            # Keep frontier high to push toward remaining unknowns
            weights.frontier_scale = max(weights.frontier_scale, 1.2)

        # Rule 4: If SLAM confidence is low (localization lost), add safety
        if health.slam_health < 0.3:
            weights.time_penalty_scale = 1.0 + self.lost_safety_penalty
            # Don't penalize exploration though — still need to map
            weights.exploration_scale = max(weights.exploration_scale, 1.0)

        # Rule 5: If collision rate is too high, triple collision penalty
        collision_rate = self._get_collision_rate()
        if collision_rate > self.collision_rate_threshold:
            weights.collision_penalty_scale = self.collision_penalty_boost

        return weights

    def _get_collision_rate(self) -> float:
        """Get rolling collision rate over the recent window."""
        if len(self._collision_history) == 0:
            return 0.0
        window = self._collision_history[-self._collision_window:]
        return sum(window) / len(window)

    def compute(
        self,
        components: RewardComponents,
        health,
        weight_overrides: Optional[dict] = None,
    ) -> tuple[float, RewardWeights]:
        """
        Compute the adaptive reward for this step.
        
        Args:
            components:      Raw reward components from the environment
            health:          HealthDiagnostics from the health monitor
            weight_overrides: Optional dict from meta-policy with scale values
            
        Returns:
            Tuple of (adjusted_reward, active_weights)
        """
        # Track collision history
        self._collision_history.append(components.collision)
        if len(self._collision_history) > self._collision_window * 2:
            self._collision_history = self._collision_history[-self._collision_window:]

        # Determine weights: meta-policy overrides take priority
        if weight_overrides is not None:
            weights = RewardWeights(
                exploration_scale=float(weight_overrides.get("exploration_scale", 1.0)),
                frontier_scale=float(weight_overrides.get("frontier_scale", 1.0)),
                curiosity_bonus=float(weight_overrides.get("curiosity_bonus", 0.0)),
                time_penalty_scale=float(weight_overrides.get("time_penalty_scale", 1.0)),
                collision_penalty_scale=float(weight_overrides.get("collision_penalty_scale", 1.0)),
            )
        else:
            weights = self._compute_heuristic_weights(health)

        self._current_weights = weights

        # Compute final reward
        reward = 0.0

        # Exploration bonus (new cells discovered)
        reward += components.new_cells * self.base_exploration * weights.exploration_scale

        # Frontier shaping (distance change to nearest frontier)
        reward += components.frontier_delta * self.base_frontier * weights.frontier_scale

        # Curiosity bonus (entropy-based intrinsic motivation)
        if weights.curiosity_bonus > 0.0:
            reward += weights.curiosity_bonus * components.entropy_value

        # Time penalty
        reward -= self.base_time_penalty * weights.time_penalty_scale

        # Collision penalty
        if components.collision:
            reward -= self.base_collision_penalty * weights.collision_penalty_scale

        return reward, weights

    @property
    def current_weights(self) -> RewardWeights:
        """The currently active reward weight multipliers."""
        return self._current_weights

    @property
    def collision_rate(self) -> float:
        """Current rolling collision rate."""
        return self._get_collision_rate()
