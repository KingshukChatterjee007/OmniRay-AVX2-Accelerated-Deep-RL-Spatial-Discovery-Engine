"""
OmniRay Active SLAM Environment
=================================

A Gymnasium-compatible environment for training Deep RL agents on
active exploration / SLAM in 2D gridworlds with LiDAR observations.

Features:
  - Configurable raycasting backend (numpy / pymunk / simd)
  - 2D occupancy map with exploration reward
  - LiDAR observation → occupancy grid mapping
  - Continuous action space (linear vel, angular vel)
  - Exploration coverage as primary reward signal

Usage:
    from envs.active_slam_env import ActiveSLAMEnv

    env = ActiveSLAMEnv(backend='numpy', num_rays=360)
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.raycaster_backends import create_raycaster


class ActiveSLAMEnv(gym.Env):
    """
    2D Active SLAM environment with LiDAR-based observation.

    Observation: dict with
        - 'lidar': (num_rays,) normalized distances [0, 1]
        - 'pose': (3,) robot [x, y, theta] normalized
        - 'coverage_map': (map_res, map_res) binary occupancy

    Action: (2,) continuous
        - [0] linear velocity ∈ [-1, 1]
        - [1] angular velocity ∈ [-1, 1]

    Reward:
        - +1.0 per newly explored cell
        - -0.1 collision penalty
        - -0.01 time penalty (encourages efficiency)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        backend: str = "numpy",
        num_rays: int = 360,
        max_range: float = 30.0,
        arena_size: float = 100.0,
        map_resolution: int = 50,
        max_steps: int = 1000,
        num_obstacles: int = 6,
        render_mode: str | None = None,
        use_slam: bool = True,
        slam_num_particles: int = 50,
    ):
        super().__init__()

        self.backend_name = backend
        self.num_rays = num_rays
        self.max_range = max_range
        self.arena_size = arena_size
        self.map_res = map_resolution
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.use_slam = use_slam
        self.slam_num_particles = slam_num_particles

        # Create raycaster
        self.raycaster = create_raycaster(backend, num_rays, max_range)

        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation space
        obs_dict = {
            "lidar": spaces.Box(0.0, 1.0, shape=(num_rays,), dtype=np.float32),
            "pose": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
            "coverage_map": spaces.Box(
                0.0, 1.0, shape=(map_resolution, map_resolution), dtype=np.float32
            ),
        }

        # Initialize SLAM if enabled
        if self.use_slam:
            from envs.vector_slam import VectorSLAM
            self.slam = VectorSLAM(
                map_size=arena_size,
                map_resolution=arena_size / map_resolution,
                num_particles=slam_num_particles,
                max_range=max_range,
            )
            # Add SLAM-specific observations
            obs_dict["slam_pose"] = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
            obs_dict["slam_map"] = spaces.Box(
                -5.0, 5.0, shape=(map_resolution, map_resolution), dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Internal state
        self._walls = []
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0
        self._coverage = None
        self._step_count = 0
        self._total_explored = 0
        self._scan_time_ms = 0.0
        self._slam_time_ms = 0.0
        self.fig = None
        self.axes = None

    def _generate_walls(self, rng: np.random.Generator):
        """Generate arena boundary + random internal obstacles."""
        s = self.arena_size
        walls = [
            (0, 0, s, 0),      # bottom
            (s, 0, s, s),      # right
            (s, s, 0, s),      # top
            (0, s, 0, 0),      # left
        ]
        for _ in range(self.num_obstacles):
            x1 = rng.uniform(10, s - 10)
            y1 = rng.uniform(10, s - 10)
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(5, 25)
            x2 = x1 + length * np.cos(angle)
            y2 = y1 + length * np.sin(angle)
            # Clip to arena
            x2 = np.clip(x2, 1, s - 1)
            y2 = np.clip(y2, 1, s - 1)
            walls.append((x1, y1, x2, y2))
        return walls

    def _get_obs(self, lidar: np.ndarray) -> dict:
        """Build observation dict."""
        obs = {
            "lidar": lidar / self.max_range,  # normalize to [0, 1]
            "pose": np.array([
                self._robot_x / self.arena_size,
                self._robot_y / self.arena_size,
                (self._robot_theta % (2 * np.pi)) / (2 * np.pi),
            ], dtype=np.float32),
            "coverage_map": self._coverage.astype(np.float32),
        }
        if self.use_slam:
            best_idx = np.argmax(self.slam.weights)
            best_pose = self.slam.particles[best_idx]
            obs["slam_pose"] = np.array([
                best_pose[0] / self.arena_size,
                best_pose[1] / self.arena_size,
                (best_pose[2] % (2 * np.pi)) / (2 * np.pi),
            ], dtype=np.float32)
            obs["slam_map"] = self.slam.map.copy()
        return obs

    def _update_coverage(self, lidar: np.ndarray):
        """Mark cells along rays as explored."""
        new_cells = 0
        cell_size = self.arena_size / self.map_res
        for i in range(self.num_rays):
            angle = self._robot_theta + i * 2 * np.pi / self.num_rays
            dist = lidar[i]
            # Mark cells along the ray
            num_steps = max(1, int(dist / cell_size))
            for s in range(num_steps + 1):
                d = s * cell_size
                if d > dist:
                    break
                px = self._robot_x + d * np.cos(angle)
                py = self._robot_y + d * np.sin(angle)
                gx = int(np.clip(px / cell_size, 0, self.map_res - 1))
                gy = int(np.clip(py / cell_size, 0, self.map_res - 1))
                if self._coverage[gy, gx] == 0:
                    self._coverage[gy, gx] = 1
                    new_cells += 1
        return new_cells

    def _is_collision(self, x: float, y: float) -> bool:
        """Check if position is inside a wall (simple boundary check)."""
        margin = 1.0
        if x < margin or x > self.arena_size - margin:
            return True
        if y < margin or y > self.arena_size - margin:
            return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Generate world
        self._walls = self._generate_walls(rng)
        self.raycaster.set_walls(self._walls)

        # Place robot at center
        self._robot_x = self.arena_size / 2
        self._robot_y = self.arena_size / 2
        self._robot_theta = rng.uniform(0, 2 * np.pi)

        # Reset coverage
        self._coverage = np.zeros((self.map_res, self.map_res), dtype=np.float32)
        self._step_count = 0
        self._total_explored = 0

        # Reset SLAM
        if self.use_slam:
            self.slam.reset(self._robot_x, self._robot_y, self._robot_theta)

        # Initial scan
        import time
        t0 = time.perf_counter()
        lidar = self.raycaster.scan(self._robot_x, self._robot_y, self._robot_theta)
        self._scan_time_ms = (time.perf_counter() - t0) * 1000

        new_cells = self._update_coverage(lidar)
        self._total_explored += new_cells

        obs = self._get_obs(lidar)
        info = {
            "scan_time_ms": self._scan_time_ms,
            "slam_time_ms": 0.0,
            "coverage": self._total_explored / (self.map_res ** 2),
            "new_cells": new_cells,
        }
        return obs, info

    def step(self, action):
        import time

        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Kinematics: differential drive
        linear_vel = action[0] * 2.0   # max 2 units/step
        angular_vel = action[1] * 0.3  # max 0.3 rad/step

        self._robot_theta += angular_vel
        new_x = self._robot_x + linear_vel * np.cos(self._robot_theta)
        new_y = self._robot_y + linear_vel * np.sin(self._robot_theta)

        # Collision check
        collision = self._is_collision(new_x, new_y)
        if not collision:
            self._robot_x = new_x
            self._robot_y = new_y

        # LiDAR scan
        t0 = time.perf_counter()
        lidar = self.raycaster.scan(self._robot_x, self._robot_y, self._robot_theta)
        self._scan_time_ms = (time.perf_counter() - t0) * 1000

        # SLAM update
        self._slam_time_ms = 0.0
        if self.use_slam:
            slam_t0 = time.perf_counter()
            self.slam.update(linear_vel, angular_vel, lidar)
            self._slam_time_ms = (time.perf_counter() - slam_t0) * 1000

        # Coverage update
        new_cells = self._update_coverage(lidar)
        self._total_explored += new_cells
        coverage_ratio = self._total_explored / (self.map_res ** 2)

        # Reward
        reward = 0.0
        reward += new_cells * 1.0          # exploration bonus
        reward -= 0.01                      # time penalty
        if collision:
            reward -= 0.1                   # collision penalty

        # Termination
        terminated = coverage_ratio >= 0.95  # 95% explored
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs(lidar)
        info = {
            "scan_time_ms": self._scan_time_ms,
            "slam_time_ms": self._slam_time_ms,
            "coverage": coverage_ratio,
            "new_cells": new_cells,
            "collision": collision,
            "steps": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (matplotlib-based)."""
        if self.render_mode is None:
            return None

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Create persistent figure once
        if self.fig is None:
            # Set background color style
            plt.style.use('dark_background')
            self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
            # Enable interactive mode so window stays responsive
            plt.ion()
            if self.render_mode == "human":
                plt.show(block=False)

        ax = self.axes[0]
        ax2 = self.axes[1]

        # Clear axes to redraw
        ax.cla()
        ax2.cla()

        # --- Left: Arena with robot + rays ---
        ax.set_xlim(-5, self.arena_size + 5)
        ax.set_ylim(-5, self.arena_size + 5)
        ax.set_aspect("equal")
        ax.set_facecolor("#0a0a1a")

        # Draw walls
        for x1, y1, x2, y2 in self._walls:
            ax.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=2.5, alpha=0.9)

        # Draw LiDAR rays (subset for clarity)
        lidar = self.raycaster.scan(self._robot_x, self._robot_y, self._robot_theta)
        for i in range(0, self.num_rays, max(1, self.num_rays // 36)):
            angle = self._robot_theta + i * 2 * np.pi / self.num_rays
            end_x = self._robot_x + lidar[i] * np.cos(angle)
            end_y = self._robot_y + lidar[i] * np.sin(angle)
            ax.plot(
                [self._robot_x, end_x], [self._robot_y, end_y],
                color="#00f0ff", alpha=0.25, linewidth=0.6,
            )

        # Draw SLAM particle cloud if enabled
        if self.use_slam:
            ax.scatter(
                self.slam.particles[:, 0], self.slam.particles[:, 1],
                color="#bd93f9", s=4, alpha=0.5, zorder=4, label="Particles"
            )

        # Draw robot pose marker
        ax.plot(self._robot_x, self._robot_y, "o", color="#00ff88", markersize=10, zorder=5)
        # Draw head heading line
        hx = self._robot_x + 3.0 * np.cos(self._robot_theta)
        hy = self._robot_y + 3.0 * np.sin(self._robot_theta)
        ax.plot([self._robot_x, hx], [self._robot_y, hy], color="#00ff88", linewidth=2.5, zorder=6)

        coverage_pct = self._total_explored / (self.map_res ** 2) * 100
        ax.set_title(f"Arena & Particles (Explored: {coverage_pct:.1f}%)", fontsize=11, fontweight="bold")

        # --- Right: SLAM or Coverage Map ---
        if self.use_slam:
            # Convert log-odds representation to occupancy probabilities [0, 1] for display
            # occupancy_prob = 1 / (1 + exp(-log_odds))
            slam_prob = 1.0 / (1.0 + np.exp(-self.slam.map))
            
            im = ax2.imshow(
                slam_prob, origin="lower", cmap="inferno",
                extent=[0, self.arena_size, 0, self.arena_size],
                vmin=0.0, vmax=1.0,
            )
            ax2.set_title(f"VectorSLAM Real-time Grid Map", fontsize=11, fontweight="bold")
        else:
            im = ax2.imshow(
                self._coverage, origin="lower", cmap="inferno",
                extent=[0, self.arena_size, 0, self.arena_size],
                vmin=0.0, vmax=1.0,
            )
            ax2.set_title(f"Occupancy Coverage Map", fontsize=11, fontweight="bold")

        ax2.set_aspect("equal")

        self.fig.tight_layout()

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            return data[:, :, :3]
        elif self.render_mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        """Close visualization figure."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.axes = None
