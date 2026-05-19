"""
OmniRay Environment Smoke Test
================================

Quick validation that the ActiveSLAMEnv works with all available backends.
Run this to verify your setup before training.

Usage:
    python test_env.py
    python test_env.py --backend numpy --episodes 3 --render
"""

import argparse
import time
import numpy as np
import sys

# Force UTF-8 output on Windows (cp1252 can't handle Unicode symbols)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_smoke_test(backend: str, episodes: int = 3, max_steps: int = 200, render: bool = False):
    """Run a few episodes with random actions and report performance."""
    from envs.active_slam_env import ActiveSLAMEnv

    print(f"\n{'=' * 60}")
    print(f"  ActiveSLAMEnv Smoke Test — Backend: {backend}")
    print(f"{'=' * 60}")

    try:
        env = ActiveSLAMEnv(
            backend=backend,
            num_rays=360,
            max_range=30.0,
            arena_size=100.0,
            map_resolution=50,
            max_steps=max_steps,
            num_obstacles=6,
            render_mode="human" if render else None,
        )
    except Exception as e:
        print(f"  ❌ Failed to create environment: {e}")
        return None

    all_scan_times = []
    all_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_scan_times = []

        print(f"\n  Episode {ep + 1}/{episodes}")
        print(f"    Initial coverage: {info['coverage'] * 100:.1f}%")
        print(f"    Initial scan time: {info['scan_time_ms']:.3f} ms")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_scan_times.append(info["scan_time_ms"])

            if render:
                env.render()

            if terminated or truncated:
                break

        mean_scan = np.mean(ep_scan_times)
        all_scan_times.extend(ep_scan_times)
        all_rewards.append(ep_reward)

        print(f"    Steps: {step + 1}")
        print(f"    Coverage: {info['coverage'] * 100:.1f}%")
        print(f"    Reward: {ep_reward:.2f}")
        print(f"    Avg scan time: {mean_scan:.3f} ms")

    env.close()

    # Summary
    total_mean = np.mean(all_scan_times)
    total_p99 = np.percentile(all_scan_times, 99)
    est_100k = total_mean * 100_000 / 1000 / 60

    print(f"\n{'=' * 60}")
    print(f"  Summary ({backend} backend)")
    print(f"{'=' * 60}")
    print(f"  Scan time:  mean={total_mean:.3f}ms  p99={total_p99:.3f}ms")
    print(f"  Reward:     mean={np.mean(all_rewards):.2f}")
    print(f"  100K steps: ≈ {est_100k:.0f} min (scan time only)")
    
    # Verdict
    if total_mean < 5.0:
        print(f"\n  ✅ FAST ENOUGH — {backend} backend is suitable for training.")
    elif total_mean < 20.0:
        print(f"\n  ⚠️  MODERATE — Consider reducing ray count or trying SIMD backend.")
    else:
        print(f"\n  🔴 SLOW — C++ SIMD backend strongly recommended.")
    
    print()
    return total_mean


def main():
    parser = argparse.ArgumentParser(description="OmniRay Environment Smoke Test")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "pymunk", "simd"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--all", action="store_true", help="Test all available backends")
    args = parser.parse_args()

    if args.all:
        results = {}
        for backend in ["numpy", "pymunk", "simd"]:
            try:
                t = run_smoke_test(backend, args.episodes, args.max_steps, args.render)
                results[backend] = t
            except Exception as e:
                print(f"\n  ⏭️  {backend} skipped: {e}")
                results[backend] = None

        print("\n" + "=" * 60)
        print("  All Backends Comparison")
        print("=" * 60)
        for name, t in results.items():
            if t is not None:
                print(f"  {name:<12} {t:.3f} ms/scan")
            else:
                print(f"  {name:<12} UNAVAILABLE")
        print()
    else:
        run_smoke_test(args.backend, args.episodes, args.max_steps, args.render)


if __name__ == "__main__":
    main()
