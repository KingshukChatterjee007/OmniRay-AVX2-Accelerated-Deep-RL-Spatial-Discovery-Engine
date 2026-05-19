"""
OmniRay Bottleneck Profiler — Phase 0: Measure Before You Optimize
===================================================================

Run this FIRST before touching any C++ code.
Profiles three raycasting backends on YOUR hardware:
  1. Pure Python (for-loop baseline)
  2. NumPy vectorized (batch math, no physics engine)
  3. PyMunk segment queries (physics-engine-backed)

Decision matrix printed at the end tells you exactly what to build next.

Usage:
    python -m profiling.benchmark_bottleneck
    python -m profiling.benchmark_bottleneck --rays 360 --iterations 1000
"""

import time
import argparse
import statistics
import sys
import numpy as np

# Force UTF-8 output on Windows (cp1252 can't handle Unicode symbols)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_box_walls(width: float = 100.0, height: float = 100.0):
    """Returns list of wall segments [(x1, y1, x2, y2), ...]"""
    return [
        (0, 0, width, 0),        # bottom
        (width, 0, width, height),  # right
        (width, height, 0, height),  # top
        (0, height, 0, 0),        # left
    ]


def make_obstacle_walls(n: int = 6, width: float = 100.0, height: float = 100.0):
    """Adds random internal wall segments for complexity."""
    rng = np.random.default_rng(42)
    walls = make_box_walls(width, height)
    for _ in range(n):
        x1, y1 = rng.uniform(10, width - 10), rng.uniform(10, height - 10)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(5, 20)
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        walls.append((x1, y1, x2, y2))
    return walls


# ---------------------------------------------------------------------------
# Backend 1: Pure Python Raycasting (worst-case baseline)
# ---------------------------------------------------------------------------

def ray_segment_intersection_python(
    ox: float, oy: float, dx: float, dy: float,
    x1: float, y1: float, x2: float, y2: float,
    max_range: float,
) -> float:
    """Ray-line-segment intersection via parameterized algebra. Pure Python."""
    sx = x2 - x1
    sy = y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return max_range  # parallel
    t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if 0 <= t <= max_range and 0 <= u <= 1:
        return t
    return max_range


def raytrace_pure_python(
    robot_x: float, robot_y: float,
    walls: list, num_rays: int = 360, max_range: float = 30.0,
) -> np.ndarray:
    """Scan all rays via pure Python loops. No vectorization."""
    distances = []
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        dx = np.cos(angle)
        dy = np.sin(angle)
        min_d = max_range
        for x1, y1, x2, y2 in walls:
            d = ray_segment_intersection_python(robot_x, robot_y, dx, dy,
                                                 x1, y1, x2, y2, max_range)
            if d < min_d:
                min_d = d
        distances.append(min_d)
    return np.array(distances, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backend 2: NumPy Vectorized Raycasting (batch math)
# ---------------------------------------------------------------------------

def raytrace_numpy(
    robot_x: float, robot_y: float,
    walls: list, num_rays: int = 360, max_range: float = 30.0,
) -> np.ndarray:
    """
    Fully vectorized ray-segment intersection using NumPy broadcasting.
    Traces ALL rays against ALL walls in a single batch.
    """
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=np.float64)
    dx = np.cos(angles)  # (R,)
    dy = np.sin(angles)  # (R,)

    walls_arr = np.array(walls, dtype=np.float64)  # (W, 4)
    sx = walls_arr[:, 2] - walls_arr[:, 0]  # (W,)
    sy = walls_arr[:, 3] - walls_arr[:, 1]  # (W,)

    # Broadcasting: (R, 1) vs (1, W) → (R, W)
    denom = dx[:, None] * sy[None, :] - dy[:, None] * sx[None, :]

    diffx = walls_arr[:, 0][None, :] - robot_x  # (1, W)
    diffy = walls_arr[:, 1][None, :] - robot_y  # (1, W)

    # Avoid division by zero
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    t = (diffx * sy[None, :] - diffy * sx[None, :]) / safe_denom  # (R, W)
    u = (diffx * dy[:, None] - diffy * dx[:, None]) / safe_denom  # (R, W)

    # Valid hits: t >= 0, t <= max_range, 0 <= u <= 1
    valid = (t >= 0) & (t <= max_range) & (u >= 0) & (u <= 1) & (np.abs(denom) >= 1e-12)
    t_valid = np.where(valid, t, max_range)

    distances = np.min(t_valid, axis=1).astype(np.float32)
    return distances


# ---------------------------------------------------------------------------
# Backend 3: PyMunk Segment Query
# ---------------------------------------------------------------------------

def raytrace_pymunk(
    robot_x: float, robot_y: float,
    space, num_rays: int = 360, max_range: float = 30.0,
) -> np.ndarray:
    """Raycasting via PyMunk's segment_query_first (C-backed physics engine)."""
    distances = np.full(num_rays, max_range, dtype=np.float32)
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        end_x = robot_x + max_range * np.cos(angle)
        end_y = robot_y + max_range * np.sin(angle)
        import pymunk
        query = space.segment_query_first(
            (robot_x, robot_y), (end_x, end_y), 0.0, pymunk.ShapeFilter()
        )
        if query is not None and query.shape is not None:
            distances[i] = query.alpha * max_range
    return distances


def build_pymunk_space(walls):
    """Create a PyMunk space with static wall segments."""
    import pymunk
    space = pymunk.Space()
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    for x1, y1, x2, y2 in walls:
        seg = pymunk.Segment(body, (x1, y1), (x2, y2), 1.0)
        space.add(body, seg)
    # Need separate bodies for each segment to avoid duplicate body add
    return space


def build_pymunk_space_v2(walls):
    """Create a PyMunk space — one static body per wall segment."""
    import pymunk
    space = pymunk.Space()
    for x1, y1, x2, y2 in walls:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        seg = pymunk.Segment(body, (x1, y1), (x2, y2), 1.0)
        seg.elasticity = 0.0
        space.add(body, seg)
    return space


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def benchmark(func, args, iterations: int, label: str):
    """Run func(*args) for `iterations` times, return timing stats."""
    # Warmup
    for _ in range(min(10, iterations)):
        func(*args)

    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p99_ms = sorted(times_ms)[int(0.99 * len(times_ms))]
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    return {
        "label": label,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p99_ms": p99_ms,
        "std_ms": std_ms,
        "total_s": sum(times_ms) / 1000,
        "iterations": iterations,
        "sample_result_shape": result.shape if hasattr(result, 'shape') else len(result),
    }


def print_results(results: list):
    """Pretty-print benchmark table and decision."""
    print("\n" + "=" * 78)
    print("  OmniRay Bottleneck Profiler — Results")
    print("=" * 78)
    
    header = f"{'Backend':<28} {'Mean':>8} {'Median':>8} {'P99':>8} {'StdDev':>8} {'100K Steps':>12}"
    print(header)
    print("-" * 78)

    for r in results:
        est_100k = r["mean_ms"] * 100_000 / 1000 / 60  # minutes
        print(
            f"  {r['label']:<26} {r['mean_ms']:>7.3f}ms {r['median_ms']:>7.3f}ms "
            f"{r['p99_ms']:>7.3f}ms {r['std_ms']:>7.3f}ms {est_100k:>9.1f} min"
        )

    print("=" * 78)

    # Decision
    best = min(results, key=lambda r: r["mean_ms"])
    print(f"\n  ★ Fastest backend: {best['label']} ({best['mean_ms']:.3f} ms/scan)")
    print()

    numpy_result = next((r for r in results if "NumPy" in r["label"]), None)
    if numpy_result:
        ms = numpy_result["mean_ms"]
        if ms < 5.0:
            print("  ✅ VERDICT: NumPy vectorized is fast enough (< 5 ms).")
            print("     → Skip C++ SIMD. Use NumPy backend. Go to Phase 2 (RL training).")
            print("     → Estimated 100K training steps: {:.0f} min".format(ms * 100_000 / 1000 / 60))
        elif ms < 20.0:
            print("  ⚠️  VERDICT: NumPy is moderate (5–20 ms).")
            print("     → Try reducing ray count (128 instead of 360).")
            print("     → Or optimize PyMunk with spatial hashing + batch queries.")
            print("     → C++ SIMD is optional but would help.")
        else:
            print("  🔴 VERDICT: NumPy is slow (> 20 ms).")
            print("     → C++ SIMD with AVX2 is STRONGLY recommended.")
            print("     → Expected speedup: 50–100× over Pure Python.")
    
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniRay Bottleneck Profiler — measure before you optimize"
    )
    parser.add_argument("--rays", type=int, default=360,
                        help="Number of LiDAR rays per scan (default: 360)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of scan iterations for benchmarking (default: 500)")
    parser.add_argument("--max-range", type=float, default=30.0,
                        help="Max ray distance (default: 30.0)")
    parser.add_argument("--obstacles", type=int, default=6,
                        help="Number of internal obstacles (default: 6)")
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip pure Python benchmark (very slow)")
    args = parser.parse_args()

    print("=" * 78)
    print("  OmniRay Bottleneck Profiler")
    print(f"  Rays: {args.rays} | Iterations: {args.iterations} | "
          f"Max Range: {args.max_range} | Obstacles: {args.obstacles}")
    print("=" * 78)

    # Setup
    walls = make_obstacle_walls(args.obstacles)
    robot_x, robot_y = 50.0, 50.0  # center of 100×100 arena
    results = []

    # --- Backend 1: Pure Python ---
    if not args.skip_python:
        python_iters = min(args.iterations, 50)  # Cap it — it's very slow
        print(f"\n  [1/3] Pure Python raycasting ({python_iters} iterations)...")
        r = benchmark(
            raytrace_pure_python,
            (robot_x, robot_y, walls, args.rays, args.max_range),
            python_iters, "Pure Python (for-loops)"
        )
        results.append(r)
        print(f"        → {r['mean_ms']:.3f} ms/scan")
    else:
        print("\n  [1/3] Pure Python — SKIPPED")

    # --- Backend 2: NumPy Vectorized ---
    print(f"\n  [2/3] NumPy vectorized raycasting ({args.iterations} iterations)...")
    r = benchmark(
        raytrace_numpy,
        (robot_x, robot_y, walls, args.rays, args.max_range),
        args.iterations, "NumPy Vectorized"
    )
    results.append(r)
    print(f"        → {r['mean_ms']:.3f} ms/scan")

    # --- Backend 3: PyMunk ---
    print(f"\n  [3/3] PyMunk segment_query ({args.iterations} iterations)...")
    try:
        space = build_pymunk_space_v2(walls)
        r = benchmark(
            raytrace_pymunk,
            (robot_x, robot_y, space, args.rays, args.max_range),
            args.iterations, "PyMunk segment_query"
        )
        results.append(r)
        print(f"        → {r['mean_ms']:.3f} ms/scan")
    except Exception as e:
        print(f"        → PyMunk failed: {e}")

    # Print results & decision
    print_results(results)

    # --- Bonus: Test reduced ray counts ---
    print("\n" + "-" * 78)
    print("  Bonus: Ray count sensitivity (NumPy backend)")
    print("-" * 78)
    for n_rays in [64, 128, 256, 360, 720]:
        r = benchmark(
            raytrace_numpy,
            (robot_x, robot_y, walls, n_rays, args.max_range),
            100, f"NumPy @ {n_rays} rays"
        )
        est_100k = r["mean_ms"] * 100_000 / 1000 / 60
        print(f"  {n_rays:>4} rays → {r['mean_ms']:.3f} ms/scan  (100K steps ≈ {est_100k:.0f} min)")

    print("\n  Done! Use these numbers to decide your optimization path.\n")


if __name__ == "__main__":
    main()
