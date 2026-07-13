"""
OmniRay Active SLAM — Paper-Ready Ablation Study Orchestrator
=============================================================

Executes a systematic ablation study isolating each of the 5 self-adaptive
autonomy layers. Produces reproducible, multi-seed results suitable for
a peer-reviewed paper's results section.

Ablation Matrix (6 configs):
    a. Full System:      All 5 layers ON  (health, adaptive_reward, meta, curriculum, continual)
    b. No Health:        Layer 1 OFF only
    c. No Adaptive Rew:  Layer 2 OFF only
    d. No Meta-Policy:   Layer 3 OFF only  (since meta is OFF by default, this is layers 1+2+4+5)
    e. No Curriculum:    Layer 4 OFF only
    f. No Continual:     Layer 5 OFF only

Baseline Comparisons (2 configs):
    g. No Adaptive:      All 5 layers OFF  (pure PPO with base reward)
    h. No Noise:         Full system + no physical noise  (ideal kinematics)

Legacy Hyperparameter Ablations (3 configs, preserved from Phase 6):
    i. Entropy ON/OFF:   ent_coef 0.01 vs 0.0
    j. Frontier HIGH/NONE: reward_frontier 0.5 vs 0.0
    k. Noise ON/OFF:     physical noise vs ideal

Each config is run across multiple seeds (default 3) for statistical significance.

Usage:
    # Run the full adaptive layer ablation matrix (3 seeds, 50k steps each)
    python run_ablation_study.py --experiment adaptive --steps 50000 --seeds 3

    # Run only baseline comparisons
    python run_ablation_study.py --experiment baselines --steps 50000 --seeds 3

    # Run everything (adaptive + baselines + legacy)
    python run_ablation_study.py --experiment all --steps 50000 --seeds 5

    # Quick smoke test (1 seed, 5000 steps)
    python run_ablation_study.py --experiment adaptive --steps 5000 --seeds 1
"""

import argparse
import subprocess
import os
import sys
import json
import time
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────
# Ablation Matrix Configuration Definitions
# ─────────────────────────────────────────────────────────────────

# Each config defines the layer flags to pass to train_rl.py
# Base flags: --adaptive enables the wrapper; --no-health / --no-adaptive-reward
# disable layers 1/2; layers 3/4/5 are opt-in via --meta-policy / --curriculum / --continual.

ADAPTIVE_CONFIGS = {
    "full_system": {
        "label": "Full System (All 5 Layers)",
        "short": "full",
        "flags": ["--adaptive", "--meta-policy", "--curriculum", "--continual"],
        "description": "All 5 adaptive layers active: health + adaptive_reward + meta + curriculum + continual",
    },
    "no_health": {
        "label": "No Health Monitor (Layer 1 OFF)",
        "short": "no_health",
        "flags": ["--adaptive", "--no-health", "--meta-policy", "--curriculum", "--continual"],
        "description": "Layer 1 (health_monitor) disabled, all others active",
    },
    "no_adaptive_reward": {
        "label": "No Adaptive Reward (Layer 2 OFF)",
        "short": "no_adrew",
        "flags": ["--adaptive", "--no-adaptive-reward", "--meta-policy", "--curriculum", "--continual"],
        "description": "Layer 2 (adaptive_reward) disabled, all others active",
    },
    "no_meta_policy": {
        "label": "No Meta-Policy (Layer 3 OFF)",
        "short": "no_meta",
        "flags": ["--adaptive", "--curriculum", "--continual"],
        "description": "Layer 3 (meta_policy) disabled, layers 1+2+4+5 active",
    },
    "no_curriculum": {
        "label": "No Curriculum (Layer 4 OFF)",
        "short": "no_curric",
        "flags": ["--adaptive", "--meta-policy", "--continual"],
        "description": "Layer 4 (curriculum) disabled, layers 1+2+3+5 active",
    },
    "no_continual": {
        "label": "No Continual Learner (Layer 5 OFF)",
        "short": "no_contin",
        "flags": ["--adaptive", "--meta-policy", "--curriculum"],
        "description": "Layer 5 (continual_learner) disabled, layers 1+2+3+4 active",
    },
}

BASELINE_CONFIGS = {
    "no_adaptive": {
        "label": "No Adaptive System (Pure PPO)",
        "short": "baseline_ppo",
        "flags": [],  # No --adaptive flag → pure PPO with base reward
        "description": "All adaptive layers OFF — pure PPO with fixed reward weights",
    },
    "no_noise_full": {
        "label": "Full System + No Noise (Ideal Kinematics)",
        "short": "ideal_noise",
        "flags": ["--adaptive", "--meta-policy", "--curriculum", "--continual", "--no-noise"],
        "description": "Full adaptive system + ideal zero-noise kinematics (no tire slip, yaw drift, LiDAR dropout)",
    },
}

LEGACY_CONFIGS = {
    "entropy_with": {
        "label": "Entropy ON (ent_coef=0.01)",
        "short": "ent_on",
        "flags": ["--ent-coef", "0.01"],
        "description": "PPO with entropy regularizer (Phase 6 ablation)",
    },
    "entropy_without": {
        "label": "Entropy OFF (ent_coef=0.0)",
        "short": "ent_off",
        "flags": ["--ent-coef", "0.0"],
        "description": "PPO without entropy regularizer (Phase 6 ablation)",
    },
    "frontier_high": {
        "label": "High Frontier Shaping (weight=0.5)",
        "short": "front_hi",
        "flags": ["--reward-frontier", "0.5"],
        "description": "High frontier exploration shaping pull (Phase 6 ablation)",
    },
    "frontier_none": {
        "label": "No Frontier Shaping (weight=0.0)",
        "short": "front_off",
        "flags": ["--reward-frontier", "0.0"],
        "description": "No frontier shaping — pure cell exploration (Phase 6 ablation)",
    },
    "noise_with": {
        "label": "Physical Noise ON",
        "short": "noise_on",
        "flags": [],
        "description": "Standard physical noise (Phase 6 ablation)",
    },
    "noise_without": {
        "label": "Physical Noise OFF",
        "short": "noise_off",
        "flags": ["--no-noise"],
        "description": "Ideal zero-noise kinematics (Phase 6 ablation)",
    },
}


# ─────────────────────────────────────────────────────────────────
# Execution Engine
# ─────────────────────────────────────────────────────────────────

def run_single_config(config_name, config, steps, seed, results_dir):
    """
    Launch a single training run for one config + one seed.
    Returns (success: bool, result_json_path: str or None).
    """
    save_name = f"{config['short']}_seed{seed}"
    save_path = os.path.join(results_dir, save_name)

    cmd = [
        sys.executable, "train_rl.py",
        "--total-steps", str(steps),
        "--seed", str(seed),
        "--save-path", save_path,
    ] + config["flags"]

    cmd_str = " ".join(cmd)
    print(f"\n{'─' * 70}")
    print(f"  🚀 [{config_name}] seed={seed}")
    print(f"     {config['label']}")
    print(f"     Command: {cmd_str}")
    print(f"{'─' * 70}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  ✅ [{config_name}] seed={seed} completed in {elapsed:.1f}s")

        result_json = f"{save_path}_results.json"
        if os.path.exists(result_json):
            return True, result_json
        else:
            print(f"  ⚠️  Results JSON not found at {result_json}")
            return True, None

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  ❌ [{config_name}] seed={seed} FAILED after {elapsed:.1f}s: {e}")
        return False, None


def run_config_set(config_set, set_label, steps, seeds, results_dir):
    """
    Run all configs in a set across all seeds.
    Returns list of (config_name, seed, success, result_path) tuples.
    """
    print("\n" + "=" * 80)
    print(f"  🧪 {set_label}")
    print(f"     Configs: {len(config_set)}  |  Seeds: {seeds}  |  Steps/run: {steps:,}")
    print(f"     Total runs: {len(config_set) * len(seeds)}")
    print("=" * 80)

    all_results = []
    for config_name, config in config_set.items():
        for seed in seeds:
            success, result_path = run_single_config(
                config_name, config, steps, seed, results_dir
            )
            all_results.append({
                "config": config_name,
                "label": config["label"],
                "description": config["description"],
                "seed": seed,
                "success": success,
                "result_path": result_path,
            })
    return all_results


def print_summary(all_results, results_dir, timestamp):
    """Print and save a final summary of all runs."""
    print("\n" + "=" * 80)
    print("  🏁 ABLATION STUDY FINAL SUMMARY")
    print("=" * 80)

    total = len(all_results)
    passed = sum(1 for r in all_results if r["success"])
    failed = total - passed

    print(f"\n  Total runs:  {total}")
    print(f"  ✅ Passed:    {passed}")
    print(f"  ❌ Failed:    {failed}")

    # Group results by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    print(f"\n  {'Config':<25} {'Seeds':>6}  {'Pass':>5}  {'Fail':>5}")
    print(f"  {'─' * 45}")
    for config_name, runs in by_config.items():
        n_pass = sum(1 for r in runs if r["success"])
        n_fail = len(runs) - n_pass
        print(f"  {config_name:<25} {len(runs):>6}  {n_pass:>5}  {n_fail:>5}")

    print("=" * 80)

    # Save the summary manifest
    manifest = {
        "timestamp": timestamp,
        "total_runs": total,
        "passed": passed,
        "failed": failed,
        "runs": all_results,
    }
    manifest_path = os.path.join(results_dir, "ablation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  📋 Manifest saved to: {manifest_path}")


def estimate_wall_clock(num_configs, num_seeds, steps_per_run):
    """
    Estimate total wall-clock time based on SIMD-backend per-step timing.
    SIMD per-step ≈ 0.015ms raycaster + ~2ms total step overhead = ~2.015ms/step.
    Conservative estimate: ~3ms/step including PPO update overhead.
    """
    ms_per_step = 3.0  # Conservative estimate
    total_steps = num_configs * num_seeds * steps_per_run
    total_seconds = (total_steps * ms_per_step) / 1000.0
    total_minutes = total_seconds / 60.0
    total_hours = total_minutes / 60.0
    return total_steps, total_seconds, total_minutes, total_hours


# ─────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OmniRay Active SLAM — Paper-Ready Ablation Study Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["adaptive", "baselines", "legacy", "all"],
        default="adaptive",
        help="Which experiment set to run (default: adaptive layer matrix)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total training steps per run (default: 50,000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds per config (default: 3)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First seed value (subsequent seeds are +1 increments, default: 42)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/ablation_<timestamp>)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only print wall-clock time estimate, do not run any training",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or os.path.join("results", f"ablation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Build seed list
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Determine which config sets to run
    config_sets = []
    if args.experiment in ("adaptive", "all"):
        config_sets.append(("Adaptive Layer Ablation Matrix (6 configs)", ADAPTIVE_CONFIGS))
    if args.experiment in ("baselines", "all"):
        config_sets.append(("Baseline Comparisons (2 configs)", BASELINE_CONFIGS))
    if args.experiment in ("legacy", "all"):
        config_sets.append(("Legacy Hyperparameter Ablations (6 configs)", LEGACY_CONFIGS))

    total_configs = sum(len(cs[1]) for cs in config_sets)

    # Print header
    print("=" * 80)
    print("  🤖 OmniRay Active SLAM — Paper-Ready Ablation Study Orchestrator 🧪")
    print("=" * 80)
    print(f"  Experiment Set:   {args.experiment.upper()}")
    print(f"  Total Configs:    {total_configs}")
    print(f"  Seeds per Config: {args.seeds}  ({seeds})")
    print(f"  Steps per Run:    {args.steps:,}")
    print(f"  Total Runs:       {total_configs * args.seeds}")
    print(f"  Results Dir:      {results_dir}")
    print(f"  Backend:          C++ SIMD (AVX2) — default for all runs")

    # Wall-clock estimate
    total_steps, total_secs, total_mins, total_hrs = estimate_wall_clock(
        total_configs, args.seeds, args.steps
    )
    print(f"\n  ⏱️  Estimated Wall-Clock Time:")
    print(f"     Total Steps:   {total_steps:,}")
    print(f"     Estimate:      {total_mins:.1f} min ({total_hrs:.2f} hrs)")
    print(f"     (Based on ~3ms/step conservative estimate with SIMD backend)")
    print("=" * 80)

    if args.estimate_only:
        print("\n  [ESTIMATE ONLY] No training runs will be executed.")
        return

    # Execute all config sets
    all_results = []
    run_start = time.time()

    for set_label, config_set in config_sets:
        results = run_config_set(config_set, set_label, args.steps, seeds, results_dir)
        all_results.extend(results)

    total_duration = time.time() - run_start

    # Print final summary
    print_summary(all_results, results_dir, timestamp)
    print(f"\n  ⏱️  Actual Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")


if __name__ == "__main__":
    main()
