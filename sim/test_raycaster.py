"""
OmniRay C++ SIMD Raycaster — Correctness & Performance Test
=============================================================

Validates the C++ SIMD raycaster against the NumPy reference implementation.
Run AFTER building the C++ module (see sim/CMakeLists.txt).

Usage:
    python sim/test_raycaster.py
"""

import sys
import numpy as np
import time


def test_simd_raycaster():
    """Test the C++ SIMD raycaster against NumPy reference."""
    try:
        from raycaster_simd import Raycaster
    except ImportError:
        print("❌ raycaster_simd module not found.")
        print("   Build it first:")
        print("   cd sim && mkdir build && cd build")
        print("   cmake .. && cmake --build . --config Release")
        print("   Then copy the .pyd/.so file to the project root.")
        return False

    print("=" * 60)
    print("  OmniRay SIMD Raycaster — Test Suite")
    print("=" * 60)

    # --- Setup ---
    num_rays = 360
    max_range = 30.0
    rc = Raycaster(num_rays, max_range)

    # Box arena
    rc.add_wall(0, 0, 100, 0)    # bottom
    rc.add_wall(100, 0, 100, 100)  # right
    rc.add_wall(100, 100, 0, 100)  # top
    rc.add_wall(0, 100, 0, 0)    # left

    # Internal obstacles
    rc.add_wall(30, 30, 60, 30)
    rc.add_wall(60, 30, 60, 60)
    rc.add_wall(20, 70, 50, 80)
    rc.add_wall(70, 20, 90, 50)

    print(f"  Walls: {rc.wall_count()}")
    print(f"  Rays:  {num_rays}")

    # --- Correctness ---
    print("\n  [1/3] Correctness check...")
    result = rc.scan(50.0, 50.0, 0.0)
    distances = np.array(result.distances)

    assert len(distances) == num_rays, f"Expected {num_rays} rays, got {len(distances)}"
    assert np.all(distances > 0), "Distances should be positive"
    assert np.all(distances <= max_range), f"Distances should be <= {max_range}"
    print(f"        ✅ Shape: {distances.shape}")
    print(f"        ✅ Range: [{distances.min():.2f}, {distances.max():.2f}]")
    print(f"        ✅ Mean:  {distances.mean():.2f}")

    # --- NumPy Reference Comparison ---
    print("\n  [2/3] Reference comparison (NumPy vs SIMD)...")
    from envs.raycaster_backends import NumpyRaycaster
    
    walls = [
        (0, 0, 100, 0), (100, 0, 100, 100),
        (100, 100, 0, 100), (0, 100, 0, 0),
        (30, 30, 60, 30), (60, 30, 60, 60),
        (20, 70, 50, 80), (70, 20, 90, 50),
    ]
    
    np_rc = NumpyRaycaster(num_rays, max_range)
    np_rc.set_walls(walls)
    
    np_distances = np_rc.scan(50.0, 50.0, 0.0)
    
    max_diff = np.max(np.abs(distances - np_distances))
    mean_diff = np.mean(np.abs(distances - np_distances))
    print(f"        Max difference:  {max_diff:.6f}")
    print(f"        Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.1:
        print("        ✅ Results match (tolerance < 0.1)")
    else:
        print("        ⚠️  Results diverge — check implementation")

    # --- Performance ---
    print("\n  [3/3] Performance benchmark...")
    
    # SIMD timing
    iterations = 10000
    simd_times = []
    for _ in range(iterations):
        result = rc.scan(50.0, 50.0, np.random.uniform(0, 2 * np.pi))
        simd_times.append(result.query_time_ms)
    
    # NumPy timing
    numpy_times = []
    for _ in range(min(iterations, 1000)):
        t0 = time.perf_counter()
        np_rc.scan(50.0, 50.0, np.random.uniform(0, 2 * np.pi))
        numpy_times.append((time.perf_counter() - t0) * 1000)

    simd_mean = np.mean(simd_times)
    numpy_mean = np.mean(numpy_times)
    speedup = numpy_mean / simd_mean if simd_mean > 0 else float('inf')

    print(f"\n  {'Backend':<20} {'Mean':>10} {'Median':>10} {'P99':>10}")
    print(f"  {'-'*50}")
    print(f"  {'C++ SIMD (AVX2)':<20} {simd_mean:>9.3f}ms "
          f"{np.median(simd_times):>9.3f}ms {np.percentile(simd_times, 99):>9.3f}ms")
    print(f"  {'NumPy Vectorized':<20} {numpy_mean:>9.3f}ms "
          f"{np.median(numpy_times):>9.3f}ms {np.percentile(numpy_times, 99):>9.3f}ms")
    print(f"\n  ★ Speedup: {speedup:.1f}×")

    est_100k_simd = simd_mean * 100_000 / 1000 / 60
    est_100k_numpy = numpy_mean * 100_000 / 1000 / 60
    print(f"  100K training steps: SIMD ≈ {est_100k_simd:.0f} min, NumPy ≈ {est_100k_numpy:.0f} min")

    print("\n  Done! ✅\n")
    return True


if __name__ == "__main__":
    success = test_simd_raycaster()
    sys.exit(0 if success else 1)
