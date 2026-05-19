"""
OmniRay Deep RL Active SLAM Training Pipeline
=============================================

A highly optimized training script utilizing Stable-Baselines3 PPO and a custom
convolutional-MLP feature extractor designed specifically for the ActiveSLAMEnv.

Features:
  - Custom Multi-Input SLAM Feature Extractor (CNN + MLP fusion).
  - Pluggable support for VectorSLAM observations.
  - CUDA GPU-accelerated training automatically enabled if available.
  - Fully configurable via command line flags.
"""

import argparse
import os
import time
import sys
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from envs.active_slam_env import ActiveSLAMEnv

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class SLAMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for ActiveSLAMEnv observations.
    
    Channels:
      - coverage_map (2D): CNN branch
      - slam_map (2D, optional): CNN branch
      - lidar (1D): MLP branch
      - pose (1D): MLP branch
      - slam_pose (1D, optional): MLP branch
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        extractors = {}
        total_concat_dim = 0

        # 1. 2D Map Branch (Coverage Map)
        cov_shape = observation_space.spaces["coverage_map"].shape
        # Input shape: (1, H, W) for CNN
        extractors["coverage_map"] = th.nn.Sequential(
            th.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        # Compute conv output size dynamically
        with th.no_grad():
            dummy = th.zeros(1, 1, cov_shape[0], cov_shape[1])
            conv_out_size = extractors["coverage_map"](dummy).shape[1]
        
        extractors["coverage_map"].add_module("fc", th.nn.Linear(conv_out_size, 128))
        extractors["coverage_map"].add_module("fc_relu", th.nn.ReLU())
        total_concat_dim += 128

        # 2. 2D Map Branch (SLAM Map, if available)
        if "slam_map" in observation_space.spaces:
            slam_map_shape = observation_space.spaces["slam_map"].shape
            extractors["slam_map"] = th.nn.Sequential(
                th.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Flatten(),
            )
            with th.no_grad():
                dummy = th.zeros(1, 1, slam_map_shape[0], slam_map_shape[1])
                conv_out_size = extractors["slam_map"](dummy).shape[1]
            
            extractors["slam_map"].add_module("fc", th.nn.Linear(conv_out_size, 128))
            extractors["slam_map"].add_module("fc_relu", th.nn.ReLU())
            total_concat_dim += 128

        # 3. 1D Vector Branches (LiDAR and Poses)
        lidar_dim = observation_space.spaces["lidar"].shape[0]
        extractors["lidar"] = th.nn.Sequential(
            th.nn.Linear(lidar_dim, 64),
            th.nn.ReLU(),
        )
        total_concat_dim += 64

        pose_dim = observation_space.spaces["pose"].shape[0]
        extractors["pose"] = th.nn.Sequential(
            th.nn.Linear(pose_dim, 32),
            th.nn.ReLU(),
        )
        total_concat_dim += 32

        if "slam_pose" in observation_space.spaces:
            slam_pose_dim = observation_space.spaces["slam_pose"].shape[0]
            extractors["slam_pose"] = th.nn.Sequential(
                th.nn.Linear(slam_pose_dim, 32),
                th.nn.ReLU(),
            )
            total_concat_dim += 32

        self.extractors = th.nn.ModuleDict(extractors)

        # Final projection head
        self.fc_head = th.nn.Sequential(
            th.nn.Linear(total_concat_dim, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # Run each input through its branch
        for key, extractor in self.extractors.items():
            obs = observations[key]
            # CNNs expect (Batch, Channel, Height, Width)
            if key in ["coverage_map", "slam_map"]:
                if len(obs.shape) == 3:
                    obs = obs.unsqueeze(1)  # Add channel dim
                elif len(obs.shape) == 2:
                    obs = obs.unsqueeze(0).unsqueeze(0)
            
            encoded_tensor_list.append(extractor(obs))

        # Concatenate and project
        features = th.cat(encoded_tensor_list, dim=1)
        return self.fc_head(features)


def train():
    parser = argparse.ArgumentParser(description="OmniRay PPO Active SLAM Training")
    parser.add_argument("--total-steps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--num-rays", type=int, default=128, help="Number of rays for LiDAR scan")
    parser.add_argument("--map-res", type=int, default=50, help="Resolution of the mapping grid")
    parser.add_argument("--disable-slam", action="store_true", help="Disable the VectorSLAM matching engine")
    parser.add_argument("--save-path", type=str, default="active_slam_ppo", help="Path to save trained agent")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
    args = parser.parse_args()

    print("=" * 70)
    print("  OmniRay Active SLAM Deep RL Trainer")
    print("=" * 70)
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"  Device:           {device.upper()}")
    print(f"  LiDAR rays:       {args.num_rays}")
    print(f"  Map Resolution:   {args.map_res}x{args.map_res}")
    print(f"  SLAM engine:      {'DISABLED' if args.disable_slam else 'ENABLED'}")
    print(f"  Total Steps:      {args.total_steps}")
    print(f"  Learning Rate:    {args.lr}")
    print("-" * 70)

    # Initialize environment
    env = ActiveSLAMEnv(
        backend="numpy",
        num_rays=args.num_rays,
        map_resolution=args.map_res,
        use_slam=not args.disable_slam,
        max_steps=200,
    )

    # Setup custom features extractor arguments
    policy_kwargs = dict(
        features_extractor_class=SLAMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),  # compact networks for policy and value
    )

    # Instantiate PPO Agent
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )

    # Train model
    print("\n  Starting PPO Training Pipeline...")
    t0 = time.time()
    
    try:
        model.learn(total_timesteps=args.total_steps)
        duration = time.time() - t0
        print("\n" + "=" * 70)
        print("  Training Completed Successfully! [SUCCESS]")
        print(f"  Total Duration:   {duration:.1f} seconds")
        print(f"  Save Path:        {args.save_path}.zip")
        print("=" * 70)
        
        # Save model
        model.save(args.save_path)
    except KeyboardInterrupt:
        print("\n  [WARNING] Training interrupted by user. Saving checkpoint...")
        model.save(f"{args.save_path}_interrupted")
        print(f"  Saved checkpoint to {args.save_path}_interrupted.zip")
    
    env.close()


if __name__ == "__main__":
    train()
