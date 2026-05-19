#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * OmniRay SIMD Raycaster
 * =======================
 * 
 * AVX2-accelerated 2D raycasting engine.
 * Processes 8 rays simultaneously using 256-bit SIMD registers.
 * 
 * Each ray is tested against all wall segments via vectorized
 * ray-line-segment intersection (parametric form).
 * 
 * Performance target: < 0.5 ms for 360 rays × 20 walls
 * on Intel i7-1355U (Raptor Lake, AVX2).
 */

struct Wall {
    float x1, y1, x2, y2;
    // Pre-computed deltas for SIMD
    float dx, dy;
};

struct RaycastResult {
    std::vector<float> distances;   // num_rays distances
    float query_time_ms;            // scan timing
};

class Raycaster {
public:
    /**
     * @param num_rays  Number of LiDAR rays per scan (should be multiple of 8)
     * @param max_range Maximum ray distance
     */
    Raycaster(int num_rays = 360, float max_range = 30.0f);
    ~Raycaster() = default;

    /** Add a wall segment to the scene. */
    void add_wall(float x1, float y1, float x2, float y2);

    /** Remove all walls. */
    void clear_walls();

    /** Get number of walls. */
    int wall_count() const { return static_cast<int>(walls_.size()); }

    /**
     * Perform a full 360° LiDAR scan from (robot_x, robot_y) 
     * facing robot_angle radians.
     * Returns distances for num_rays evenly-spaced rays.
     */
    RaycastResult scan(float robot_x, float robot_y, float robot_angle) const;

private:
    int num_rays_;
    float max_range_;
    std::vector<Wall> walls_;

    /** 
     * Scalar fallback: single ray vs all walls.
     * Used when AVX2 is not available or for remainder rays.
     */
    float raytrace_scalar(float ox, float oy, float dx, float dy) const;

#ifdef __AVX2__
    /**
     * AVX2 batch: 8 rays vs all walls simultaneously.
     * Writes results to out[0..7].
     */
    void raytrace_avx2_batch(
        float ox, float oy,
        const float* cos_angles,  // 8 cosines
        const float* sin_angles,  // 8 sines
        float* out                // 8 distances
    ) const;
#endif
};
