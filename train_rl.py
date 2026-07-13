"""
OmniRay Deep RL Active SLAM Training Pipeline
=============================================

A highly optimized training script utilizing Stable-Baselines3 PPO and a custom
convolutional-MLP feature extractor designed specifically for the ActiveSLAMEnv.

Features:
  - Custom Multi-Input SLAM Feature Extractor (CNN + MLP fusion).
  - Pluggable support for VectorSLAM observations.
  - CUDA GPU-accelerated training automatically enabled if available.
  - Configuration hyperparams fully loaded from config.yaml.
  - 5-Layer Self-Adaptive Autonomy System (--adaptive flag).
"""

import argparse
import os
import time
import sys
import yaml
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from envs.active_slam_env import ActiveSLAMEnv

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class AdaptiveCallback(BaseCallback):
    """
    SB3 training callback that bridges the PPO training loop with the
    adaptive environment wrapper. Feeds real-time policy entropy to the
    health monitor every training step.
    """

    def __init__(self, adaptive_env=None, verbose=0):
        super().__init__(verbose)
        self.adaptive_env = adaptive_env
        self._last_entropy = 1.0

    def _on_step(self) -> bool:
        """Called after every environment step during training."""
        if self.adaptive_env is None:
            return True

        # Extract policy entropy from the SB3 logger
        # SB3 logs 'entropy_loss' which is the mean entropy of the policy
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                # During rollout collection, entropy isn't directly available
                # but we can compute it from the policy distribution
                if hasattr(self.model.policy, 'action_dist') and self.model.policy.action_dist is not None:
                    try:
                        entropy = self.model.policy.action_dist.entropy()
                        if entropy is not None:
                            self._last_entropy = float(entropy.mean().item())
                    except Exception:
                        pass
            except Exception:
                pass

        # Feed entropy to the adaptive wrapper
        self.adaptive_env.set_policy_entropy(self._last_entropy)
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout collection."""
        if self.adaptive_env is None:
            return

        # Log adaptive stats periodically
        try:
            stats = self.adaptive_env.get_adaptive_stats()
            health = stats.get("health", {})
            self.logger.record("adaptive/health_score", health.get("score", 0.0))
            self.logger.record("adaptive/is_failing", health.get("is_failing", False))

            diagnostics = health.get("diagnostics", {})
            self.logger.record("adaptive/entropy_health", diagnostics.get("entropy", 0.0))
            self.logger.record("adaptive/coverage_health", diagnostics.get("coverage", 0.0))
            self.logger.record("adaptive/slam_health", diagnostics.get("slam", 0.0))
            self.logger.record("adaptive/coverage_velocity", diagnostics.get("velocity", 0.0))

            reward_info = stats.get("reward", {})
            weights = reward_info.get("weights", {})
            self.logger.record("adaptive/exploration_scale", weights.get("exploration_scale", 1.0))
            self.logger.record("adaptive/frontier_scale", weights.get("frontier_scale", 1.0))
            self.logger.record("adaptive/curiosity_bonus", weights.get("curiosity_bonus", 0.0))
            self.logger.record("adaptive/collision_rate", reward_info.get("collision_rate", 0.0))

            if "curriculum" in stats:
                cur = stats["curriculum"]
                self.logger.record("adaptive/difficulty", cur.get("curriculum_difficulty", 0.0))
                self.logger.record("adaptive/obstacles", cur.get("curriculum_obstacles", 6))
                self.logger.record("adaptive/arena_size", cur.get("curriculum_arena_size", 100.0))
                self.logger.record("adaptive/noise_scale", cur.get("curriculum_noise_scale", 1.0))

            if "continual" in stats:
                cl = stats["continual"]
                self.logger.record("adaptive/retrains", cl.get("continual_total_retrains", 0))
                self.logger.record("adaptive/rollbacks", cl.get("continual_total_rollbacks", 0))
        except Exception:
            pass  # Don't crash training on logging errors


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

    def __init__(self, observation_space, features_dim: int = 256, cnn_config: dict = None, mlp_config: dict = None):
        super().__init__(observation_space, features_dim)
        
        extractors = {}
        total_concat_dim = 0

        # Use defaults if config sections are missing
        if cnn_config is None:
            cnn_config = {
                "coverage_channels": [16, 32],
                "coverage_kernel": 3,
                "coverage_stride": 2,
                "coverage_padding": 1,
                "slam_channels": [16, 32],
                "slam_kernel": 3,
                "slam_stride": 2,
                "slam_padding": 1
            }
        if mlp_config is None:
            mlp_config = {
                "lidar_dim": 64,
                "pose_dim": 32,
                "slam_pose_dim": 32
            }

        # 1. 2D Map Branch (Coverage Map)
        cov_shape = observation_space.spaces["coverage_map"].shape
        cov_ch = cnn_config.get("coverage_channels", [16, 32])
        cov_k = cnn_config.get("coverage_kernel", 3)
        cov_s = cnn_config.get("coverage_stride", 2)
        cov_p = cnn_config.get("coverage_padding", 1)

        extractors["coverage_map"] = th.nn.Sequential(
            th.nn.Conv2d(1, cov_ch[0], kernel_size=cov_k, stride=cov_s, padding=cov_p),
            th.nn.ReLU(),
            th.nn.Conv2d(cov_ch[0], cov_ch[1], kernel_size=cov_k, stride=cov_s, padding=cov_p),
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
            slam_ch = cnn_config.get("slam_channels", [16, 32])
            slam_k = cnn_config.get("slam_kernel", 3)
            slam_s = cnn_config.get("slam_stride", 2)
            slam_p = cnn_config.get("slam_padding", 1)

            extractors["slam_map"] = th.nn.Sequential(
                th.nn.Conv2d(1, slam_ch[0], kernel_size=slam_k, stride=slam_s, padding=slam_p),
                th.nn.ReLU(),
                th.nn.Conv2d(slam_ch[0], slam_ch[1], kernel_size=slam_k, stride=slam_s, padding=slam_p),
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
            th.nn.Linear(lidar_dim, mlp_config.get("lidar_dim", 64)),
            th.nn.ReLU(),
        )
        total_concat_dim += mlp_config.get("lidar_dim", 64)

        pose_dim = observation_space.spaces["pose"].shape[0]
        extractors["pose"] = th.nn.Sequential(
            th.nn.Linear(pose_dim, mlp_config.get("pose_dim", 32)),
            th.nn.ReLU(),
        )
        total_concat_dim += mlp_config.get("pose_dim", 32)

        if "slam_pose" in observation_space.spaces:
            slam_pose_dim = observation_space.spaces["slam_pose"].shape[0]
            extractors["slam_pose"] = th.nn.Sequential(
                th.nn.Linear(slam_pose_dim, mlp_config.get("slam_pose_dim", 32)),
                th.nn.ReLU(),
            )
            total_concat_dim += mlp_config.get("slam_pose_dim", 32)

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
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps to train (overrides config)")
    parser.add_argument("--num-rays", type=int, default=None, help="Number of rays for LiDAR scan (overrides config)")
    parser.add_argument("--map-res", type=int, default=None, help="Resolution of the mapping grid (overrides config)")
    parser.add_argument("--disable-slam", action="store_true", help="Disable the VectorSLAM matching engine")
    parser.add_argument("--save-path", type=str, default="active_slam_ppo", help="Path to save trained agent")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for PPO (overrides config)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient for PPO (overrides config)")
    
    # Ablation-specific overrides
    parser.add_argument("--no-noise", action="store_true", help="Disable physical environment noise for ablation")
    parser.add_argument("--reward-exploration", type=float, default=None, help="Override exploration reward weight")
    parser.add_argument("--reward-time", type=float, default=None, help="Override time penalty reward weight")
    parser.add_argument("--reward-collision", type=float, default=None, help="Override collision penalty weight")
    parser.add_argument("--reward-frontier", type=float, default=None, help="Override frontier exploration shaping weight")
    
    # Self-Adaptive Autonomy System flags
    parser.add_argument("--adaptive", action="store_true", help="Enable the 5-layer self-adaptive autonomy system")
    parser.add_argument("--no-health", action="store_true", help="Disable Layer 1 health monitor (requires --adaptive)")
    parser.add_argument("--no-adaptive-reward", action="store_true", help="Disable Layer 2 adaptive reward (requires --adaptive)")
    parser.add_argument("--meta-policy", action="store_true", help="Enable Layer 3 meta-policy (requires --adaptive)")
    parser.add_argument("--curriculum", action="store_true", help="Enable Layer 4 curriculum auto-difficulty (requires --adaptive)")
    parser.add_argument("--continual", action="store_true", help="Enable Layer 5 continual learning (requires --adaptive)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (seeds PyTorch, NumPy, env)")
    args = parser.parse_args()

    # Load configuration file
    config = {}
    if os.path.exists(args.config):
        print(f"  Loading hyperparameters from config file: {args.config}...")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"  [WARNING] Config file not found at: {args.config}. Using hardcoded default parameters.")

    # Extract configs or fallback to defaults
    env_config = config.get("env", {})
    network_config = config.get("network", {})
    ppo_config = config.get("ppo", {})

    # Apply command line overrides if provided, otherwise use config/default
    num_rays = args.num_rays if args.num_rays is not None else env_config.get("num_rays", 128)
    map_res = args.map_res if args.map_res is not None else env_config.get("map_resolution", 50)
    use_slam = not args.disable_slam if args.disable_slam else env_config.get("use_slam", True)
    total_steps = args.total_steps if args.total_steps is not None else ppo_config.get("total_timesteps", 50000)
    lr = args.lr if args.lr is not None else ppo_config.get("learning_rate", 3e-4)
    ent_coef = args.ent_coef if args.ent_coef is not None else ppo_config.get("ent_coef", 0.01)
    
    # Environment noise & reward weights overrides
    real_world_noise = False if args.no_noise else env_config.get("real_world_noise", True)
    rew_exp = args.reward_exploration if args.reward_exploration is not None else env_config.get("reward_exploration", 1.0)
    rew_time = args.reward_time if args.reward_time is not None else env_config.get("reward_time_penalty", 0.01)
    rew_col = args.reward_collision if args.reward_collision is not None else env_config.get("reward_collision_penalty", 0.1)
    rew_front = args.reward_frontier if args.reward_frontier is not None else env_config.get("reward_frontier", 0.1)

    # Adaptive system configuration
    adaptive_config = config.get("adaptive", {})
    use_adaptive = args.adaptive or adaptive_config.get("enabled", False)
    use_health = not args.no_health  # Layer 1: default ON when adaptive is enabled
    use_adaptive_reward = not args.no_adaptive_reward  # Layer 2: default ON when adaptive is enabled
    use_meta = args.meta_policy or adaptive_config.get("meta_policy", {}).get("enabled", False)
    use_curriculum = args.curriculum or adaptive_config.get("curriculum", {}).get("enabled", False)
    use_continual = args.continual or adaptive_config.get("continual_learning", {}).get("enabled", False)
    
    # Seed initialization for reproducibility
    seed = args.seed
    if seed is not None:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
        print(f"  [SEED] Random seed set to {seed} (PyTorch, NumPy, env)")

    print("=" * 70)
    print("  OmniRay Active SLAM Deep RL Trainer")
    print("=" * 70)
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"  Device:           {device.upper()}")
    print(f"  LiDAR rays:       {num_rays}")
    print(f"  Map Resolution:   {map_res}x{map_res}")
    print(f"  SLAM engine:      {'ENABLED' if use_slam else 'DISABLED'}")
    print(f"  Total Steps:      {total_steps}")
    print(f"  Learning Rate:    {lr}")
    print(f"  Entropy Coeff:    {ent_coef}")
    print(f"  Physical Noise:   {'ENABLED' if real_world_noise else 'DISABLED'}")
    print(f"  Explor Reward:    {rew_exp}")
    print(f"  Time Penalty:     {rew_time}")
    print(f"  Colli Penalty:    {rew_col}")
    print(f"  Front Reward:     {rew_front}")
    if use_adaptive:
        print(f"  Adaptive System:  ENABLED")
        print(f"    ├─ Health Monitor:   {'ACTIVE' if use_health else 'DISABLED'}")
        print(f"    ├─ Adaptive Reward:  {'ACTIVE' if use_adaptive_reward else 'DISABLED'}")
        print(f"    ├─ Meta-Policy:      {'ACTIVE' if use_meta else 'DISABLED'}")
        print(f"    ├─ Curriculum:       {'ACTIVE' if use_curriculum else 'DISABLED'}")
        print(f"    └─ Continual Learn:  {'ACTIVE' if use_continual else 'DISABLED'}")
    else:
        print(f"  Adaptive System:  DISABLED (use --adaptive to enable)")
    if seed is not None:
        print(f"  Seed:             {seed}")
    print("-" * 70)

    # Initialize base environment
    base_env = ActiveSLAMEnv(
        backend="simd",
        num_rays=num_rays,
        map_resolution=map_res,
        use_slam=use_slam,
        max_steps=env_config.get("max_steps", 200),
        real_world_noise=real_world_noise,
        reward_exploration=rew_exp,
        reward_time_penalty=rew_time,
        reward_collision_penalty=rew_col,
        reward_frontier=rew_front,
    )

    # Wrap with adaptive system if enabled
    adaptive_env = None
    if use_adaptive:
        from envs.adaptive_env import AdaptiveActiveSLAMEnv
        adaptive_env = AdaptiveActiveSLAMEnv(
            env=base_env,
            config=adaptive_config,
            enable_health=use_health,
            enable_adaptive_reward=use_adaptive_reward,
            enable_meta=use_meta,
            enable_curriculum=use_curriculum,
            enable_continual=use_continual,
        )
        env = adaptive_env
        print("  [ADAPTIVE] 5-Layer Self-Adaptive Autonomy System initialized.")
    else:
        env = base_env

    # Setup custom features extractor arguments
    policy_kwargs = dict(
        features_extractor_class=SLAMFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=network_config.get("features_dim", 256),
            cnn_config=network_config.get("cnn", None),
            mlp_config=network_config.get("mlp", None),
        ),
        net_arch=dict(
            pi=network_config.get("pi_arch", [128, 64]),
            vf=network_config.get("vf_arch", [128, 64]),
        ),
    )

    # Instantiate PPO Agent
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=lr,
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )

    # Connect adaptive system to the model (for entropy access + continual learning)
    callbacks = []
    if use_adaptive and adaptive_env is not None:
        adaptive_env.set_model(model)
        adaptive_callback = AdaptiveCallback(adaptive_env=adaptive_env, verbose=1)
        callbacks.append(adaptive_callback)
        print("  [ADAPTIVE] AdaptiveCallback registered for entropy + health logging.")

    # Train model
    print("\n  Starting PPO Training Pipeline...")
    if use_adaptive:
        print("  [ADAPTIVE] Health monitor, adaptive reward, and all enabled layers are LIVE.")
    t0 = time.time()
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks if callbacks else None,
        )
        duration = time.time() - t0
        print("\n" + "=" * 70)
        print("  Training Completed Successfully! [SUCCESS]")
        print(f"  Total Duration:   {duration:.1f} seconds")
        print(f"  Save Path:        {args.save_path}.zip")

        # Print adaptive summary if enabled
        if use_adaptive and adaptive_env is not None:
            stats = adaptive_env.get_adaptive_stats()
            print("\n  --- Adaptive System Final Summary ---")
            health = stats.get("health", {})
            print(f"  Final Health Score:    {health.get('score', 0.0):.3f}")
            print(f"  System Failing:        {health.get('is_failing', False)}")
            
            if "curriculum" in stats:
                cur = stats["curriculum"]
                print(f"  Difficulty Level:      {cur.get('curriculum_level', 0)}")
                print(f"  Difficulty Increases:  {cur.get('curriculum_increases', 0)}")
                print(f"  Difficulty Decreases:  {cur.get('curriculum_decreases', 0)}")
                print(f"  Final Obstacles:       {cur.get('curriculum_obstacles', 6)}")
                print(f"  Final Arena Size:      {cur.get('curriculum_arena_size', 100.0)}")
                print(f"  Final Noise Scale:     {cur.get('curriculum_noise_scale', 1.0)}")
            
            if "continual" in stats:
                cl = stats["continual"]
                print(f"  Continual Retrains:    {cl.get('continual_total_retrains', 0)}")
                print(f"  Continual Rollbacks:   {cl.get('continual_total_rollbacks', 0)}")
                print(f"  Peak Reward:           {cl.get('continual_peak_reward', 0.0):.2f}")
            
            if "meta_policy" in stats:
                mp = stats["meta_policy"]
                print(f"  Meta-Policy Updates:   {mp.get('meta_updates', 0)}")
                print(f"  Meta Cumulative Δ:     {mp.get('meta_cumulative_delta', 0.0):.4f}")
            
            print("  ------------------------------------")

        print("=" * 70)
        
        # Save model
        model.save(args.save_path)
        
        # Write structured JSON result file for ablation analysis
        import json
        result_data = {
            "config": {
                "seed": seed,
                "total_steps": total_steps,
                "num_rays": num_rays,
                "map_resolution": map_res,
                "use_slam": use_slam,
                "learning_rate": lr,
                "ent_coef": ent_coef,
                "real_world_noise": real_world_noise,
                "reward_exploration": rew_exp,
                "reward_time_penalty": rew_time,
                "reward_collision_penalty": rew_col,
                "reward_frontier": rew_front,
                "backend": "simd",
                "adaptive": use_adaptive,
                "layers": {
                    "health_monitor": use_health if use_adaptive else False,
                    "adaptive_reward": use_adaptive_reward if use_adaptive else False,
                    "meta_policy": use_meta if use_adaptive else False,
                    "curriculum": use_curriculum if use_adaptive else False,
                    "continual": use_continual if use_adaptive else False,
                },
            },
            "results": {
                "wall_clock_seconds": duration,
                "save_path": f"{args.save_path}.zip",
            },
        }
        
        # Add adaptive stats if available
        if use_adaptive and adaptive_env is not None:
            result_data["results"]["adaptive_stats"] = adaptive_env.get_adaptive_stats()
        
        result_json_path = f"{args.save_path}_results.json"
        with open(result_json_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        print(f"  📊 Results saved to: {result_json_path}")

    except KeyboardInterrupt:
        print("\n  [WARNING] Training interrupted by user. Saving checkpoint...")
        model.save(f"{args.save_path}_interrupted")
        print(f"  Saved checkpoint to {args.save_path}_interrupted.zip")
    
    env.close()


if __name__ == "__main__":
    train()
