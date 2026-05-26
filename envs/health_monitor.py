"""
OmniRay Self-Adaptive Autonomy — Layer 1: Health Monitor
==========================================================

Computes a real-time scalar health score (0.0–1.0) every simulation step
by evaluating three sub-metrics:

  1. Entropy Health:    Is the policy exploring enough? (moderate entropy = healthy)
  2. Coverage Velocity: Is the agent making mapping progress? (positive delta = healthy)
  3. SLAM Confidence:   Does the particle filter agree on the pose? (low variance = healthy)

The combined score uses exponential moving average (EMA) smoothing to
reduce noise and provide a stable signal for downstream adaptive layers.

Usage:
    monitor = HealthMonitor(config)
    monitor.reset()
    
    # Every step:
    health = monitor.update(
        coverage_ratio=0.45,
        slam_weights=env.slam.weights,
        policy_entropy=1.2
    )
    print(health.score, health.is_failing)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HealthDiagnostics:
    """Container for all health sub-metrics and the final score."""
    score: float = 0.5
    entropy_health: float = 0.5
    coverage_health: float = 0.5
    slam_health: float = 0.5
    coverage_velocity: float = 0.0
    raw_entropy: float = 1.0
    raw_slam_variance: float = 0.0
    is_failing: bool = False
    step_count: int = 0


class HealthMonitor:
    """
    Self-awareness layer: monitors agent performance in real-time.
    
    Produces a scalar health score (0.0–1.0) from three orthogonal signals:
    entropy behavior, coverage progress, and SLAM localization confidence.
    
    Args:
        ema_alpha:        EMA smoothing factor (lower = smoother, default 0.05)
        entropy_weight:   Weight for entropy sub-score in final health (default 0.35)
        coverage_weight:  Weight for coverage sub-score in final health (default 0.40)
        slam_weight:      Weight for SLAM confidence sub-score (default 0.25)
        window_size:      Rolling window for coverage velocity computation (default 50)
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        entropy_weight: float = 0.35,
        coverage_weight: float = 0.40,
        slam_weight: float = 0.25,
        window_size: int = 50,
    ):
        self.ema_alpha = ema_alpha
        self.entropy_weight = entropy_weight
        self.coverage_weight = coverage_weight
        self.slam_weight = slam_weight
        self.window_size = window_size

        # Normalize weights to sum to 1.0
        total = self.entropy_weight + self.coverage_weight + self.slam_weight
        if total > 0:
            self.entropy_weight /= total
            self.coverage_weight /= total
            self.slam_weight /= total

        # Internal state
        self._ema_health: float = 0.5
        self._coverage_history: list[float] = []
        self._step_count: int = 0
        self._diagnostics = HealthDiagnostics()

    def reset(self):
        """Reset monitor state at episode boundaries."""
        self._ema_health = 0.5
        self._coverage_history = []
        self._step_count = 0
        self._diagnostics = HealthDiagnostics()

    def update(
        self,
        coverage_ratio: float,
        slam_weights: Optional[np.ndarray] = None,
        policy_entropy: Optional[float] = None,
    ) -> HealthDiagnostics:
        """
        Compute health score from current step observations.
        
        Args:
            coverage_ratio:  Fraction of map explored [0, 1]
            slam_weights:    Particle filter weight array (num_particles,)
            policy_entropy:  Current PPO policy entropy (from SB3 logger)
            
        Returns:
            HealthDiagnostics with all sub-scores and final health
        """
        self._step_count += 1

        # -----------------------------------------------------------
        # Sub-Metric 1: Entropy Health
        # -----------------------------------------------------------
        # Moderate entropy (0.3–1.5) is healthy: the policy is exploring
        # but not randomly. Collapsed (<0.1) or exploding (>3.0) is bad.
        if policy_entropy is not None:
            entropy_val = float(policy_entropy)
            if entropy_val < 0.1:
                # Entropy collapsed → policy is deterministic, not exploring
                entropy_health = max(0.0, entropy_val / 0.1 * 0.3)
            elif entropy_val <= 1.5:
                # Sweet spot: linearly scale from 0.3 at 0.1 to 1.0 at ~0.8
                entropy_health = min(1.0, 0.3 + (entropy_val - 0.1) / 0.7 * 0.7)
            elif entropy_val <= 3.0:
                # Getting high: linearly decrease from 1.0 to 0.4
                entropy_health = max(0.4, 1.0 - (entropy_val - 1.5) / 1.5 * 0.6)
            else:
                # Exploding entropy: very unhealthy
                entropy_health = max(0.1, 0.4 - (entropy_val - 3.0) / 3.0 * 0.3)
        else:
            # No entropy signal available yet — assume neutral
            entropy_health = 0.5

        # -----------------------------------------------------------
        # Sub-Metric 2: Coverage Velocity
        # -----------------------------------------------------------
        # Measures how fast the agent is discovering new cells.
        # Positive and steady = healthy. Flat/declining = stuck.
        self._coverage_history.append(coverage_ratio)

        if len(self._coverage_history) >= 2:
            # Use the last `window_size` entries for velocity
            window = self._coverage_history[-self.window_size:]
            if len(window) >= 2:
                # Linear regression slope over the window
                x = np.arange(len(window), dtype=np.float64)
                y = np.array(window, dtype=np.float64)
                # Least-squares slope: Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
                x_mean = x.mean()
                y_mean = y.mean()
                slope = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12)
                coverage_velocity = float(slope)
            else:
                coverage_velocity = 0.0
        else:
            coverage_velocity = 0.0

        # Normalize velocity to health score:
        # slope > 0.002 is great (actively exploring)
        # slope ≈ 0 means stuck
        # slope < 0 shouldn't happen but means coverage tracking error
        if coverage_velocity > 0.005:
            coverage_health = 1.0
        elif coverage_velocity > 0.0:
            coverage_health = min(1.0, 0.3 + coverage_velocity / 0.005 * 0.7)
        elif coverage_velocity > -0.001:
            # Essentially zero — agent is stuck
            coverage_health = 0.2
        else:
            coverage_health = 0.1

        # Late-game adjustment: if coverage is already very high (>85%),
        # don't penalize for low velocity — there's little left to explore
        if coverage_ratio > 0.85:
            coverage_health = max(coverage_health, 0.7)

        # -----------------------------------------------------------
        # Sub-Metric 3: SLAM Confidence
        # -----------------------------------------------------------
        # Low weight variance = particles agree = confident localization
        # High variance = particles disagree = lost
        if slam_weights is not None and len(slam_weights) > 1:
            weight_variance = float(np.var(slam_weights))
            # Effective sample size: lower = more particle degeneracy
            ess = 1.0 / max(np.sum(slam_weights ** 2), 1e-12)
            ess_ratio = ess / len(slam_weights)  # 1.0 = perfect, 0.0 = degenerate

            # Combine variance and ESS
            # Low variance + high ESS = confident
            if ess_ratio > 0.5:
                slam_health = 1.0
            elif ess_ratio > 0.1:
                slam_health = 0.3 + (ess_ratio - 0.1) / 0.4 * 0.7
            else:
                slam_health = max(0.05, ess_ratio / 0.1 * 0.3)
        else:
            slam_health = 0.5
            weight_variance = 0.0

        # -----------------------------------------------------------
        # Combine into final health score
        # -----------------------------------------------------------
        raw_health = (
            self.entropy_weight * entropy_health
            + self.coverage_weight * coverage_health
            + self.slam_weight * slam_health
        )
        raw_health = float(np.clip(raw_health, 0.0, 1.0))

        # Apply EMA smoothing
        self._ema_health = self.ema_alpha * raw_health + (1.0 - self.ema_alpha) * self._ema_health

        # Build diagnostics
        self._diagnostics = HealthDiagnostics(
            score=self._ema_health,
            entropy_health=entropy_health,
            coverage_health=coverage_health,
            slam_health=slam_health,
            coverage_velocity=coverage_velocity,
            raw_entropy=policy_entropy if policy_entropy is not None else 0.0,
            raw_slam_variance=weight_variance,
            is_failing=self._ema_health < 0.5,
            step_count=self._step_count,
        )

        return self._diagnostics

    @property
    def health_score(self) -> float:
        """Current EMA-smoothed health score."""
        return self._ema_health

    @property
    def is_failing(self) -> bool:
        """True if health is below the failure threshold (0.5)."""
        return self._ema_health < 0.5

    @property
    def diagnostics(self) -> HealthDiagnostics:
        """Full diagnostic breakdown from the last update."""
        return self._diagnostics
