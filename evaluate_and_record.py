"""
OmniRay Active SLAM — Robust Agent Evaluation and Trajectory Recording
========================================================================

This script runs the trained RL policy inside the ActiveSLAMEnv under
real-world physical sensor and actuator noise. It records:
  - Ground truth trajectory
  - Dead-reckoning (odometry-only integration)
  - VectorSLAM particle filter estimation

It saves diagnostic plots comparing these three trajectories to showcase
the drift and particle filter correction, and saves the final occupancy grid.
"""

import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.active_slam_env import ActiveSLAMEnv

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_agent(model_path: str, save_dir: str, num_rays: int, map_res: int, steps: int):
    print("=" * 75)
    print("  OmniRay Active SLAM — Robust Agent Evaluation Under Real-World Noise")
    print("=" * 75)
    print(f"  Model Path:       {model_path}")
    print(f"  Output Directory: {save_dir}")
    print(f"  LiDAR rays:       {num_rays}")
    print(f"  Steps:            {steps}")
    print("-" * 75)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialize environment with noise enabled
    print("  Initializing Gymnasium active SLAM environment with physical noise...")
    env = ActiveSLAMEnv(
        backend="numpy",
        num_rays=num_rays,
        map_resolution=map_res,
        max_steps=steps,
        render_mode=None,  # Running headless for quantitative evaluation
        use_slam=True,
        real_world_noise=True,  # Crucial: enable physical wheel slip and sensor dropout
    )

    # 2. Load trained model
    print(f"  Loading trained PPO model from: {model_path}...")
    try:
        model = PPO.load(model_path, env=env)
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  [ERROR] Failed to load PPO model: {e}")
        env.close()
        return

    # 3. Reset and prepare logging
    obs, info = env.reset(seed=42)
    
    # Store trajectories
    gt_trajectory = []
    odom_trajectory = []
    slam_trajectory = []
    coverage_history = []
    reward_history = []
    
    # Initial states
    gt_x, gt_y, gt_theta = env._robot_x, env._robot_y, env._robot_theta
    odom_x, odom_y, odom_theta = gt_x, gt_y, gt_theta
    
    # Record initial step
    gt_trajectory.append((gt_x, gt_y))
    odom_trajectory.append((odom_x, odom_y))
    
    best_idx = np.argmax(env.slam.weights)
    slam_pose = env.slam.particles[best_idx]
    slam_trajectory.append((slam_pose[0], slam_pose[1]))
    
    coverage_history.append(info["coverage"])

    step_count = 0
    total_reward = 0.0

    print("  Running simulation episode...")
    while True:
        # Get action from the policy (deterministic for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Retrieve action values to reconstruct dead-reckoning
        linear_vel = action[0] * 2.0
        angular_vel = action[1] * 0.3
        
        # Update dead-reckoning (odometry only integration - what the robot THINKS it did)
        odom_theta += angular_vel
        odom_x += linear_vel * np.cos(odom_theta)
        odom_y += linear_vel * np.sin(odom_theta)
        
        # Retrieve actual ground truth pose
        gt_x, gt_y, gt_theta = env._robot_x, env._robot_y, env._robot_theta
        
        # Retrieve SLAM estimated pose
        best_idx = np.argmax(env.slam.weights)
        slam_pose = env.slam.particles[best_idx]
        
        # Record trajectories
        gt_trajectory.append((gt_x, gt_y))
        odom_trajectory.append((odom_x, odom_y))
        slam_trajectory.append((slam_pose[0], slam_pose[1]))
        coverage_history.append(info["coverage"])
        reward_history.append(reward)

        if terminated or truncated:
            print(f"    Finished in {step_count} steps.")
            print(f"    Final coverage achieved: {info['coverage'] * 100:.2f}%")
            print(f"    Total cumulative reward: {total_reward:.2f}")
            break

    # Convert to NumPy arrays for easier slicing and plotting
    gt_arr = np.array(gt_trajectory)
    odom_arr = np.array(odom_trajectory)
    slam_arr = np.array(slam_trajectory)
    
    # Calculate drift metrics
    final_odom_drift = np.hypot(odom_arr[-1, 0] - gt_arr[-1, 0], odom_arr[-1, 1] - gt_arr[-1, 1])
    final_slam_drift = np.hypot(slam_arr[-1, 0] - gt_arr[-1, 0], slam_arr[-1, 1] - gt_arr[-1, 1])
    
    print("\n  Diagnostic Drift Metrics:")
    print(f"    Raw Odometry (Dead-Reckoning) final position error: {final_odom_drift:.2f} units")
    print(f"    VectorSLAM (Particle Filter) final position error:   {final_slam_drift:.2f} units")
    print(f"    SLAM Error reduction:                              {(1.0 - final_slam_drift/final_odom_drift)*100:.1f}%")

    # 4. Generate beautiful diagnostic plots
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0d0d1a')
    
    # Setup styling parameters
    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#8888aa')
        ax.spines['bottom'].set_color('#333355')
        ax.spines['top'].set_color('#333355')
        ax.spines['left'].set_color('#333355')
        ax.spines['right'].set_color('#333355')
        ax.grid(color='#222244', linestyle='--', alpha=0.5)

    # Plot 1: Trajectory Comparison (Left)
    ax_traj = axes[0]
    ax_traj.set_xlim(-10, env.arena_size + 10)
    ax_traj.set_ylim(-10, env.arena_size + 10)
    ax_traj.set_aspect("equal")
    
    # Draw Arena boundary walls and obstacles
    for x1, y1, x2, y2 in env._walls:
        ax_traj.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=2.5, alpha=0.8)
        
    # Plot trajectories
    ax_traj.plot(odom_arr[:, 0], odom_arr[:, 1], color="#ff9f43", linestyle="--", linewidth=1.8, label="Dead-Reckoning (Uncorrected)")
    ax_traj.plot(slam_arr[:, 0], slam_arr[:, 1], color="#bd93f9", linestyle="-.", linewidth=2.0, label="VectorSLAM Estimated Path")
    ax_traj.plot(gt_arr[:, 0], gt_arr[:, 1], color="#00ff88", linestyle="-", linewidth=2.5, label="Ground Truth (Actual Path)")
    
    # Draw start and end markers
    ax_traj.scatter(gt_arr[0, 0], gt_arr[0, 1], color="#00ff88", s=120, edgecolors='white', zorder=5, label="Spawn Point")
    ax_traj.scatter(gt_arr[-1, 0], gt_arr[-1, 1], color="#00ff88", marker="X", s=150, zorder=5, label="Robot Final Pose")
    ax_traj.scatter(odom_arr[-1, 0], odom_arr[-1, 1], color="#ff9f43", marker="o", s=100, zorder=5, label="Odom Final Pose (Drifted)")

    ax_traj.set_title("Physical Trajectory and Drift Correction", fontsize=13, color='white', fontweight="bold", pad=15)
    ax_traj.legend(loc="upper left", framealpha=0.2, facecolor='#0d0d1a', labelcolor='white')
    
    # Plot 2: VectorSLAM Probability Map (Right)
    ax_map = axes[1]
    slam_prob = 1.0 / (1.0 + np.exp(-env.slam.map))
    
    im = ax_map.imshow(
        slam_prob, origin="lower", cmap="inferno",
        extent=[0, env.arena_size, 0, env.arena_size],
        vmin=0.0, vmax=1.0,
    )
    
    # Overlay the Ground Truth and SLAM Trajectory on map
    ax_map.plot(gt_arr[:, 0], gt_arr[:, 1], color="#00ff88", linewidth=1.5, alpha=0.7, label="GT Path")
    ax_map.plot(slam_arr[:, 0], slam_arr[:, 1], color="#bd93f9", linewidth=1.5, alpha=0.7, label="SLAM Path")
    
    ax_map.set_title(f"Reconstructed Occupancy Map (Explored: {coverage_history[-1]*100:.1f}%)", fontsize=13, color='white', fontweight="bold", pad=15)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#8888aa', labelcolor='white')
    ax_map.legend(loc="upper left", framealpha=0.2, facecolor='#0d0d1a', labelcolor='white')

    plt.suptitle(f"OmniRay Active SLAM under Real-World Noise (LiDAR + Actuator Drift)\nSLAM Position Error: {final_slam_drift:.2f} vs Odometry Drift: {final_odom_drift:.2f}", 
                 fontsize=15, color='white', fontweight="bold", y=0.98)
    
    fig.tight_layout()
    
    # Save the diagnostic plot
    plot_path = os.path.join(save_dir, "robust_evaluation_report.png")
    plt.savefig(plot_path, dpi=150, facecolor='#0d0d1a', bbox_inches='tight')
    plt.close()
    
    # Also save a plot of exploration rate and reward progression
    fig_prog, ax_prog = plt.subplots(figsize=(10, 5), facecolor='#0d0d1a')
    ax_prog.set_facecolor('#0d0d1a')
    ax_prog.tick_params(colors='#8888aa')
    ax_prog.spines['bottom'].set_color('#333355')
    ax_prog.spines['top'].set_color('#333355')
    ax_prog.spines['left'].set_color('#333355')
    ax_prog.spines['right'].set_color('#333355')
    ax_prog.grid(color='#222244', linestyle='--', alpha=0.5)
    
    steps_range = np.arange(len(coverage_history))
    ax_prog.plot(steps_range, np.array(coverage_history) * 100, color='#00ff88', linewidth=2.5, label="Exploration Coverage (%)")
    
    ax_prog2 = ax_prog.twinx()
    ax_prog2.set_facecolor('#0d0d1a')
    ax_prog2.tick_params(colors='#8888aa')
    ax_prog2.spines['right'].set_color('#333355')
    cum_rewards = np.cumsum(reward_history)
    ax_prog2.plot(steps_range[:-1], cum_rewards, color='#00f0ff', linewidth=2.0, linestyle='--', label="Cumulative Reward")
    
    # Alignment of legends
    lines, labels = ax_prog.get_legend_handles_labels()
    lines2, labels2 = ax_prog2.get_legend_handles_labels()
    ax_prog.legend(lines + lines2, labels + labels2, loc='upper left', framealpha=0.2, facecolor='#0d0d1a', labelcolor='white')
    
    ax_prog.set_xlabel("Simulation Steps", color='white', labelpad=10)
    ax_prog.set_ylabel("Exploration Coverage (%)", color='#00ff88', labelpad=10)
    ax_prog2.set_ylabel("Cumulative Reward", color='#00f0ff', labelpad=10)
    ax_prog.set_title("Exploration and Reward Progression Under Physical Noise", fontsize=13, color='white', fontweight="bold", pad=15)
    
    prog_path = os.path.join(save_dir, "robust_exploration_progression.png")
    plt.savefig(prog_path, dpi=150, facecolor='#0d0d1a', bbox_inches='tight')
    plt.close()
    
    env.close()
    print(f"\n  [SUCCESS] Evaluation complete! Diagnostics saved to:")
    print(f"    - Trajectory Report:   {plot_path}")
    print(f"    - Progression Report:  {prog_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniRay PPO Robust Evaluation and Diagnostic Recorder")
    parser.add_argument("--model-path", type=str, default="active_slam_ppo.zip", help="Path to the trained PPO zip file")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save diagnostic reports")
    parser.add_argument("--num-rays", type=int, default=128, help="Number of rays for LiDAR scan")
    parser.add_argument("--map-res", type=int, default=50, help="Resolution of the mapping grid")
    parser.add_argument("--steps", type=int, default=150, help="Number of steps in the evaluation episode")
    args = parser.parse_args()

    # Automatically add file extension if missing
    model_path = args.model_path
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        model_path += ".zip"

    if not os.path.exists(model_path):
        print(f"  [ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    evaluate_agent(
        model_path=model_path,
        save_dir=args.save_dir,
        num_rays=args.num_rays,
        map_res=args.map_res,
        steps=args.steps,
    )
