"""
OmniRay Active SLAM — Trained Agent Visualizer
===============================================

This script loads a trained RL policy (from a zip file) and runs it inside the
Gymnasium environment with real-time rendering. You can watch the robot use its
LiDAR rays and VectorSLAM filter to explore the arena, avoid collisions, 
and map the occupancy grid!

Usage:
    py -3.11 visualize_agent.py --model-path active_slam_ppo.zip --num-rays 128
"""

import argparse
import time
import sys
import numpy as np
from stable_baselines3 import PPO
from envs.active_slam_env import ActiveSLAMEnv

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def visualize(model_path: str, num_rays: int, map_res: int, episodes: int, max_steps: int):
    print("=" * 70)
    print("  OmniRay Active SLAM — Trained Agent Visualizer")
    print("=" * 70)
    print(f"  Model Path:       {model_path}")
    print(f"  LiDAR rays:       {num_rays}")
    print(f"  Map Resolution:   {map_res}x{map_res}")
    print(f"  Episodes:         {episodes}")
    print("-" * 70)

    # 1. Create the render-enabled environment
    print("  Initializing Gymnasium active SLAM environment...")
    env = ActiveSLAMEnv(
        backend="numpy",
        num_rays=num_rays,
        map_resolution=map_res,
        max_steps=max_steps,
        render_mode="human",  # Enables interactive matplotlib plots
        use_slam=True,
    )

    # 2. Load the trained model
    print(f"  Loading PPO policy from: {model_path}...")
    try:
        model = PPO.load(model_path, env=env)
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  [ERROR] Failed to load PPO model: {e}")
        env.close()
        return

    # 3. Evaluation Loop
    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        step_count = 0
        
        print(f"\n  Starting Episode {ep + 1}/{episodes}")
        print(f"    Initial exploration coverage: {info['coverage'] * 100:.1f}%")

        while True:
            # Predict the best action from the trained model policy (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

            # Render frame (draws LiDAR rays, robot pose, SLAM map, and coverage)
            env.render()
            
            # Slow down slightly for human visual tracking
            time.sleep(0.02)

            if terminated or truncated:
                print(f"    Episode finished in {step_count} steps.")
                print(f"    Final exploration coverage:   {info['coverage'] * 100:.1f}%")
                print(f"    Total cumulative reward:      {ep_reward:.2f}")
                time.sleep(1.0)  # Pause before next episode
                break

    print("\n  Visualization completed. Closing environment...")
    env.close()
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniRay PPO Agent Visualizer")
    parser.add_argument("--model-path", type=str, default="active_slam_ppo.zip", help="Path to the trained PPO zip file")
    parser.add_argument("--num-rays", type=int, default=128, help="Number of rays for LiDAR scan")
    parser.add_argument("--map-res", type=int, default=50, help="Resolution of the mapping grid")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    args = parser.parse_args()

    # Automatically add file extension if missing
    model_path = args.model_path
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        model_path += ".zip"

    import os
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    visualize(
        model_path=model_path,
        num_rays=args.num_rays,
        map_res=args.map_res,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
