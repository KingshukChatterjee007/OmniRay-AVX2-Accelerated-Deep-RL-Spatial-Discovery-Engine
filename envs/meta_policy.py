"""
OmniRay Self-Adaptive Autonomy — Layer 3: Meta-Policy (Self-Tuning Network)
============================================================================

A lightweight neural network that learns the mapping:
    health_metrics → optimal_reward_weights

Instead of hand-coded heuristic rules (Layer 2), the meta-policy learns
from experience which reward weight configurations lead to health improvement.

Architecture:
    Input  (6):  [entropy_health, coverage_health, slam_health,
                  coverage_velocity, health_score, steps_fraction]
    Hidden:      Linear(6,32) → ReLU → Linear(32,32) → ReLU
    Output (5):  Linear(32,5) → Softplus (ensures positive)
                 [exploration_scale, frontier_scale, curiosity_bonus,
                  time_penalty_scale, collision_penalty_scale]

Training:
    Uses REINFORCE-style gradient updates on a simple scalar reward:
    Δhealth = current_health - health_at_last_update
    
    This is a contextual bandit — not nested RL — keeping it fast and stable.

Usage:
    meta = MetaPolicy(config)
    weights = meta.predict(health_diagnostics, steps_fraction=0.5)
    meta.record_health(current_health)
    meta.maybe_update()  # REINFORCE step every update_interval
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetaPolicyNetwork(nn.Module):
    """Small MLP that maps health metrics to reward weight multipliers."""

    def __init__(self, input_dim: int = 6, output_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),  # Ensures all outputs are positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaPolicy:
    """
    Self-tuning meta-policy that learns optimal reward weights from health feedback.
    
    Args:
        learning_rate:     Optimizer learning rate (default 1e-4)
        update_interval:   Steps between weight updates (default 200)
        warmup_steps:      Steps before meta-policy activates (default 1000)
        scale_clamp:       Min/max output clamp range (default [0.1, 5.0])
        device:            PyTorch device ('cpu' or 'cuda')
    """

    # Output indices mapping
    OUTPUT_KEYS = [
        "exploration_scale",
        "frontier_scale",
        "curiosity_bonus",
        "time_penalty_scale",
        "collision_penalty_scale",
    ]

    def __init__(
        self,
        learning_rate: float = 1e-4,
        update_interval: int = 200,
        warmup_steps: int = 1000,
        scale_clamp: tuple[float, float] = (0.1, 5.0),
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "MetaPolicy requires PyTorch. Install with: pip install torch"
            )

        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        self.scale_clamp = scale_clamp
        self.device = device

        # Network and optimizer
        self.network = MetaPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Training state
        self._step_count: int = 0
        self._health_at_last_update: float = 0.5
        self._last_prediction: dict = {}
        self._log_probs: list[torch.Tensor] = []
        self._health_deltas: list[float] = []
        self._active: bool = False

        # Tracking for logging
        self._total_updates: int = 0
        self._cumulative_delta: float = 0.0

    def reset(self):
        """Reset episode-level state (keeps learned weights)."""
        self._step_count = 0
        self._health_at_last_update = 0.5
        self._log_probs = []
        self._health_deltas = []
        self._active = False

    @property
    def is_active(self) -> bool:
        """Whether the meta-policy is past warmup and actively producing weights."""
        return self._active

    def predict(self, health_diagnostics, steps_fraction: float = 0.0) -> dict:
        """
        Predict optimal reward weights given current health state.
        
        Args:
            health_diagnostics: HealthDiagnostics from the health monitor
            steps_fraction:     Current step / max_steps (0-1), provides temporal context
            
        Returns:
            Dict of reward weight multipliers (or empty dict if in warmup)
        """
        self._step_count += 1

        # During warmup, don't override — let heuristic rules handle it
        if self._step_count < self.warmup_steps:
            self._active = False
            return {}

        self._active = True

        # Build input tensor from health diagnostics
        input_vec = torch.tensor([
            health_diagnostics.entropy_health,
            health_diagnostics.coverage_health,
            health_diagnostics.slam_health,
            health_diagnostics.coverage_velocity * 100.0,  # Scale up for network
            health_diagnostics.score,
            steps_fraction,
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass (with gradient tracking for REINFORCE)
        self.network.train()
        raw_output = self.network(input_vec)

        # Add small noise for exploration (log-normal style)
        noise = torch.randn_like(raw_output) * 0.1
        noisy_output = raw_output + noise

        # Clamp to safe range
        clamped = torch.clamp(noisy_output, self.scale_clamp[0], self.scale_clamp[1])

        # Store log probability for REINFORCE update
        # Treat as Gaussian with learned mean and fixed std
        log_prob = -0.5 * ((clamped - raw_output) ** 2).sum()
        self._log_probs.append(log_prob)

        # Convert to dict
        weights = clamped.detach().cpu().numpy().flatten()
        self._last_prediction = {
            key: float(weights[i]) for i, key in enumerate(self.OUTPUT_KEYS)
        }

        return self._last_prediction

    def record_health(self, current_health: float):
        """
        Record current health for computing the REINFORCE reward signal.
        Called every step after the health monitor updates.
        
        Args:
            current_health: The current EMA health score (0-1)
        """
        if not self._active:
            return

        # Check if it's time for an update
        if self._step_count % self.update_interval == 0 and len(self._log_probs) > 0:
            # Compute health improvement since last update
            delta_health = current_health - self._health_at_last_update
            self._health_deltas.append(delta_health)
            self._cumulative_delta += delta_health

            self._update(delta_health)
            self._health_at_last_update = current_health

    def _update(self, delta_health: float):
        """
        Perform REINFORCE-style gradient update.
        
        The reward signal is the health improvement (delta_health).
        Positive delta → reinforce the weight choices that led to it.
        Negative delta → push away from those choices.
        """
        if len(self._log_probs) == 0:
            return

        # Combine all log probs from this window
        total_log_prob = torch.stack(self._log_probs).sum()

        # REINFORCE loss: -log_prob * reward
        # We want to maximize health improvement, so loss = -log_prob * delta
        loss = -total_log_prob * delta_health

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clear stored log probs
        self._log_probs = []
        self._total_updates += 1

    def get_stats(self) -> dict:
        """Get meta-policy training statistics."""
        return {
            "meta_updates": self._total_updates,
            "meta_cumulative_delta": self._cumulative_delta,
            "meta_active": self._active,
            "meta_last_weights": self._last_prediction.copy(),
        }
