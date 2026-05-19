"""
OmniRay Active SLAM — Ablation Study Orchestrator
==================================================

This script coordinates and executes ablation studies for the Active SLAM Deep RL system.
It trains PPO agents under varied conditions to study hyperparameter sensitivity,
entropy incentives, and sim-to-real noise robustness.

Experiments Configured:
  1. Entropy Impact:       Train with (--ent-coef 0.01) vs without (--ent-coef 0.0) entropy reward.
  2. Reward Shaping Sensitivity: Train with high frontier shaping (--reward-frontier 0.5) vs none (--reward-frontier 0.0).
  3. Physical Noise Impact:      Train with physical slip/sensor dropout vs ideal environment (--no-noise).

Usage:
  - Run all ablations sequentially (10,000 steps per test as a quick benchmark):
      python run_ablation_study.py --experiment all --steps 10000

  - Run single target ablation (e.g. Entropy):
      python run_ablation_study.py --experiment entropy --steps 50000
"""

import argparse
import subprocess
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_command(command_args):
    """Utility to launch a subprocess shell command safely."""
    cmd_str = " ".join(command_args)
    print(f"\n🚀 Running: {cmd_str}")
    
    try:
        # Run subprocess and stream output live
        result = subprocess.run(
            [sys.executable] + command_args[1:] if command_args[0].lower() in ["python", "py"] else command_args,
            check=True
        )
        print(f"✅ Success: {cmd_str}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {cmd_str}\nError: {e}")
        return False


def run_entropy_ablation(steps):
    print("=" * 80)
    print(f"🧪 [EXPERIMENT 1/3] Entropy Rewards Ablation Study ({steps} steps)")
    print("=" * 80)
    print("  Purpose: Compares coverage exploration convergence rates with and without")
    print("           policy entropy regularizers to analyze exploration/exploitation behavior.")
    print("-" * 80)
    
    # 1. PPO with Entropy regularizer (Baseline)
    print("\n🔹 Step 1a: Training WITH Entropy regularizer (ent_coef = 0.01)...")
    cmd_with = ["train_rl.py", "--total-steps", str(steps), "--ent-coef", "0.01", "--save-path", "results/ablation_entropy_with"]
    
    # 2. PPO without Entropy regularizer
    print("\n🔹 Step 1b: Training WITHOUT Entropy regularizer (ent_coef = 0.00)...")
    cmd_without = ["train_rl.py", "--total-steps", str(steps), "--ent-coef", "0.00", "--save-path", "results/ablation_entropy_without"]
    
    success_with = run_command(cmd_with)
    success_without = run_command(cmd_without)
    
    if success_with and success_without:
        print("🎉 Entropy Ablation study runs completed! Models saved under results/")
        print("   - WITH Entropy:    results/ablation_entropy_with.zip")
        print("   - WITHOUT Entropy: results/ablation_entropy_without.zip")
    return success_with and success_without


def run_reward_shaping_ablation(steps):
    print("=" * 80)
    print(f"🧪 [EXPERIMENT 2/3] Reward Weights Shaping Sensitivity ({steps} steps)")
    print("=" * 80)
    print("  Purpose: Measures policy sensitivity to the frontier exploration shaping reward.")
    print("           Compares High frontier pull (0.5) against pure cell exploration (0.0).")
    print("-" * 80)
    
    # 1. High Frontier Shaping Pull
    print("\n🔹 Step 2a: Training with HIGH Frontier Shaping Pull (weight = 0.5)...")
    cmd_high = ["train_rl.py", "--total-steps", str(steps), "--reward-frontier", "0.5", "--save-path", "results/ablation_rewards_shaping_high"]
    
    # 2. No Frontier Shaping Pull (Pure Cell Exploration)
    print("\n🔹 Step 2b: Training with NO Frontier Shaping Pull (weight = 0.0)...")
    cmd_none = ["train_rl.py", "--total-steps", str(steps), "--reward-frontier", "0.0", "--save-path", "results/ablation_rewards_shaping_none"]
    
    success_high = run_command(cmd_high)
    success_none = run_command(cmd_none)
    
    if success_high and success_none:
        print("🎉 Reward Shaping Ablation study runs completed! Models saved under results/")
        print("   - HIGH Frontier Shaping: results/ablation_rewards_shaping_high.zip")
        print("   - NO Frontier Shaping:   results/ablation_rewards_shaping_none.zip")
    return success_high and success_none


def run_noise_ablation(steps):
    print("=" * 80)
    print(f"🧪 [EXPERIMENT 3/3] Sim-to-Real Robustness Ablation ({steps} steps)")
    print("=" * 80)
    print("  Purpose: Evaluates robustness of policy navigation inside highly noisy physical")
    print("           environments vs ideal, zero-noise kinematic simulations.")
    print("-" * 80)
    
    # 1. Training WITH Physical Noise (Standard)
    print("\n🔹 Step 3a: Training WITH physical kinodynamic slip and sensor noise...")
    cmd_with = ["train_rl.py", "--total-steps", str(steps), "--save-path", "results/ablation_noise_with"]
    
    # 2. Training WITHOUT Noise (Ideal Physics)
    print("\n🔹 Step 3b: Training WITHOUT noise (Ideal physical kinematics)...")
    cmd_without = ["train_rl.py", "--total-steps", str(steps), "--no-noise", "--save-path", "results/ablation_noise_without"]
    
    success_with = run_command(cmd_with)
    success_without = run_command(cmd_without)
    
    if success_with and success_without:
        print("🎉 Noise Ablation study runs completed! Models saved under results/")
        print("   - WITH Noise:    results/ablation_noise_with.zip")
        print("   - WITHOUT Noise: results/ablation_noise_without.zip")
    return success_with and success_without


def main():
    parser = argparse.ArgumentParser(description="OmniRay Active SLAM PPO Ablation Study Orchestrator")
    parser.add_argument(
        "--experiment", 
        type=str, 
        choices=["entropy", "rewards", "noise", "all"], 
        default="all", 
        help="Select which ablation study to run"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50000, 
        help="Total training steps per experiment (default: 50,000)"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 80)
    print("        🤖 OmniRay Active SLAM — PPO Ablation Study Orchestrator 🧪")
    print("=" * 80)
    print(f"  Target Experiment: {args.experiment.upper()}")
    print(f"  Steps per Model:   {args.steps:,}")
    print("-" * 80)
    print("  🚦 READY TO START! (Subprocess triggers will call train_rl.py on launch)")
    print("=" * 80)

    if args.experiment == "entropy":
        run_entropy_ablation(args.steps)
    elif args.experiment == "rewards":
        run_reward_shaping_ablation(args.steps)
    elif args.experiment == "noise":
        run_noise_ablation(args.steps)
    elif args.experiment == "all":
        entropy_ok = run_entropy_ablation(args.steps)
        rewards_ok = run_reward_shaping_ablation(args.steps)
        noise_ok = run_noise_ablation(args.steps)
        
        print("=" * 80)
        print("🏁 ABLATION STUDY SEQUENCER STATUS SUMMARY:")
        print(f"  - Entropy Ablation Run:         {'✅ COMPLETED' if entropy_ok else '❌ FAILED'}")
        print(f"  - Reward Weight Ablation Run:    {'✅ COMPLETED' if rewards_ok else '❌ FAILED'}")
        print(f"  - Noise Robustness Ablation Run: {'✅ COMPLETED' if noise_ok else '❌ FAILED'}")
        print("=" * 80)


if __name__ == "__main__":
    main()
