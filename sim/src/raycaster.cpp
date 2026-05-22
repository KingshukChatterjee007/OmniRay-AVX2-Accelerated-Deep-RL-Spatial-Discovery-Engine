/**
 * OmniRay SIMD Raycaster — Core Implementation
 * ==============================================
 * 
 * Ray-line-segment intersection using parametric form:
 * 
 *   Ray:     P = O + t * D           (t >= 0)
 *   Segment: Q = A + u * (B - A)     (0 <= u <= 1)
 * 
 *   Intersection when:
 *     t = ((A - O) × S) / (D × S)
 *     u = ((A - O) × D) / (D × S)
 *   where S = B - A, × is 2D cross product (a.x*b.y - a.y*b.x)
 * 
 * AVX2 version processes 8 rays against each wall in parallel:
 *   - 8 different directions (cos/sin) in __m256 registers
 *   - Same wall tested against all 8 simultaneously
 *   - ~8x throughput vs scalar
 * 
 * Compile with: /arch:AVX2 (MSVC) or -mavx2 (GCC/Clang)
 */

#include "raycaster.h"

#include <cmath>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER)
// MSVC uses /arch:AVX2 flag, immintrin.h is available regardless
#include <immintrin.h>
#define __AVX2__ 1
#elif defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// ============================================================================
// Constructor / Wall Management
// ============================================================================

Raycaster::Raycaster(int num_rays, float max_range)
    : num_rays_(num_rays), max_range_(max_range) {
    walls_.reserve(64);
}

void Raycaster::add_wall(float x1, float y1, float x2, float y2) {
    Wall w;
    w.x1 = x1; w.y1 = y1;
    w.x2 = x2; w.y2 = y2;
    w.dx = x2 - x1;
    w.dy = y2 - y1;
    walls_.push_back(w);
}

void Raycaster::clear_walls() {
    walls_.clear();
}

// ============================================================================
// Scalar Fallback
// ============================================================================

float Raycaster::raytrace_scalar(float ox, float oy, float dx, float dy) const {
    float min_t = max_range_;

    for (const auto& w : walls_) {
        float sx = w.dx;
        float sy = w.dy;

        float denom = dx * sy - dy * sx;
        if (std::abs(denom) < 1e-12f) continue;

        float inv_denom = 1.0f / denom;

        float diffx = w.x1 - ox;
        float diffy = w.y1 - oy;

        float t = (diffx * sy - diffy * sx) * inv_denom;
        float u = (diffx * dy - diffy * dx) * inv_denom;

        if (t >= 0.0f && t < min_t && u >= 0.0f && u <= 1.0f) {
            min_t = t;
        }
    }

    return min_t;
}

// ============================================================================
// AVX2 Batch (8 rays at once)
// ============================================================================

#ifdef __AVX2__

void Raycaster::raytrace_avx2_batch(
    float ox, float oy,
    const float* cos_angles,
    const float* sin_angles,
    float* out
) const {
    // Load 8 ray directions
    __m256 dx = _mm256_loadu_ps(cos_angles);
    __m256 dy = _mm256_loadu_ps(sin_angles);

    // Initialize min distances to max_range
    __m256 min_t = _mm256_set1_ps(max_range_);

    // Constants
    __m256 zero = _mm256_setzero_ps();
    __m256 one  = _mm256_set1_ps(1.0f);
    __m256 eps  = _mm256_set1_ps(1e-10f);

    for (const auto& w : walls_) {
        // Wall direction (broadcast to all 8 lanes)
        __m256 sx = _mm256_set1_ps(w.dx);
        __m256 sy = _mm256_set1_ps(w.dy);

        // denom = dx * sy - dy * sx  (2D cross product: ray × wall)
        __m256 denom = _mm256_sub_ps(
            _mm256_mul_ps(dx, sy),
            _mm256_mul_ps(dy, sx)
        );

        // |denom| > eps  (not parallel)
        __m256 abs_denom = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), denom);
        __m256 valid_denom = _mm256_cmp_ps(abs_denom, eps, _CMP_GT_OQ);

        // inv_denom = 1 / denom (safe: masked later)
        __m256 inv_denom = _mm256_div_ps(one, _mm256_blendv_ps(one, denom, valid_denom));

        // diff = wall_start - ray_origin
        __m256 diffx = _mm256_set1_ps(w.x1 - ox);
        __m256 diffy = _mm256_set1_ps(w.y1 - oy);

        // t = (diff × wall_dir) * inv_denom
        __m256 t = _mm256_mul_ps(
            _mm256_sub_ps(
                _mm256_mul_ps(diffx, sy),
                _mm256_mul_ps(diffy, sx)
            ),
            inv_denom
        );

        // u = (diff × ray_dir) * inv_denom
        __m256 u = _mm256_mul_ps(
            _mm256_sub_ps(
                _mm256_mul_ps(diffx, dy),
                _mm256_mul_ps(diffy, dx)
            ),
            inv_denom
        );

        // Valid hit: t >= 0 && t < min_t && u >= 0 && u <= 1 && |denom| > eps
        __m256 valid = _mm256_and_ps(
            _mm256_and_ps(
                _mm256_cmp_ps(t, zero, _CMP_GE_OQ),
                _mm256_cmp_ps(t, min_t, _CMP_LT_OQ)
            ),
            _mm256_and_ps(
                _mm256_and_ps(
                    _mm256_cmp_ps(u, zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(u, one, _CMP_LE_OQ)
                ),
                valid_denom
            )
        );

        // Update min_t where valid
        min_t = _mm256_blendv_ps(min_t, t, valid);
    }

    // Store results
    _mm256_storeu_ps(out, min_t);
}

#endif  // __AVX2__

// ============================================================================
// ARM NEON Batch (4 rays at once)
// ============================================================================

#ifdef __ARM_NEON__

void Raycaster::raytrace_neon_batch(
    float ox, float oy,
    const float* cos_angles,
    const float* sin_angles,
    float* out
) const {
    // Load 4 ray directions
    float32x4_t dx = vld1q_f32(cos_angles);
    float32x4_t dy = vld1q_f32(sin_angles);

    // Initialize min distances to max_range
    float32x4_t min_t = vdupq_n_f32(max_range_);

    // Constants
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one  = vdupq_n_f32(1.0f);
    float32x4_t eps  = vdupq_n_f32(1e-10f);

    for (const auto& w : walls_) {
        // Wall direction (broadcast to all 4 lanes)
        float32x4_t sx = vdupq_n_f32(w.dx);
        float32x4_t sy = vdupq_n_f32(w.dy);

        // denom = dx * sy - dy * sx
        float32x4_t denom = vsubq_f32(
            vmulq_f32(dx, sy),
            vmulq_f32(dy, sx)
        );

        // |denom|
        float32x4_t abs_denom = vabsq_f32(denom);

        // |denom| > eps
        uint32x4_t valid_denom = vcgtq_f32(abs_denom, eps);

        // inv_denom = 1 / denom (safeguarded against parallel lanes)
        float32x4_t safe_denom = vbslq_f32(valid_denom, denom, one);
        float32x4_t inv_denom = vdivq_f32(one, safe_denom);

        // diff = wall_start - ray_origin
        float32x4_t diffx = vdupq_n_f32(w.x1 - ox);
        float32x4_t diffy = vdupq_n_f32(w.y1 - oy);

        // t = (diff × wall_dir) * inv_denom
        float32x4_t t = vmulq_f32(
            vsubq_f32(
                vmulq_f32(diffx, sy),
                vmulq_f32(diffy, sx)
            ),
            inv_denom
        );

        // u = (diff × ray_dir) * inv_denom
        float32x4_t u = vmulq_f32(
            vsubq_f32(
                vmulq_f32(diffx, dy),
                vmulq_f32(diffy, dx)
            ),
            inv_denom
        );

        // Valid hit: t >= 0 && t < min_t && u >= 0 && u <= 1 && |denom| > eps
        uint32x4_t t_ge_zero = vcgeq_f32(t, zero);
        uint32x4_t t_lt_min  = vcltq_f32(t, min_t);
        uint32x4_t u_ge_zero = vcgeq_f32(u, zero);
        uint32x4_t u_le_one  = vcleq_f32(u, one);

        uint32x4_t valid = vandq_u32(
            vandq_u32(t_ge_zero, t_lt_min),
            vandq_u32(
                vandq_u32(u_ge_zero, u_le_one),
                valid_denom
            )
        );

        // Update min_t where valid
        min_t = vbslq_f32(valid, t, min_t);
    }

    // Store results
    vst1q_f32(out, min_t);
}

#endif  // __ARM_NEON__

// ============================================================================
// Main Scan
// ============================================================================

RaycastResult Raycaster::scan(float robot_x, float robot_y, float robot_angle) const {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> distances(num_rays_);

    // Pre-compute all angles
    std::vector<float> cos_angles(num_rays_);
    std::vector<float> sin_angles(num_rays_);
    
    float angle_step = 2.0f * static_cast<float>(M_PI) / num_rays_;
    for (int i = 0; i < num_rays_; ++i) {
        float angle = robot_angle + i * angle_step;
        cos_angles[i] = std::cos(angle);
        sin_angles[i] = std::sin(angle);
    }

#ifdef __AVX2__
    // Process 8 rays at a time
    int simd_count = (num_rays_ / 8) * 8;
    for (int i = 0; i < simd_count; i += 8) {
        raytrace_avx2_batch(
            robot_x, robot_y,
            &cos_angles[i], &sin_angles[i],
            &distances[i]
        );
    }
    // Handle remaining rays (if num_rays not multiple of 8)
    for (int i = simd_count; i < num_rays_; ++i) {
        distances[i] = raytrace_scalar(robot_x, robot_y, cos_angles[i], sin_angles[i]);
    }
#elif defined(__ARM_NEON__)
    // Process 4 rays at a time using NEON
    int simd_count = (num_rays_ / 4) * 4;
    for (int i = 0; i < simd_count; i += 4) {
        raytrace_neon_batch(
            robot_x, robot_y,
            &cos_angles[i], &sin_angles[i],
            &distances[i]
        );
    }
    // Handle remaining rays (if num_rays not multiple of 4)
    for (int i = simd_count; i < num_rays_; ++i) {
        distances[i] = raytrace_scalar(robot_x, robot_y, cos_angles[i], sin_angles[i]);
    }
#else
    // Scalar fallback
    for (int i = 0; i < num_rays_; ++i) {
        distances[i] = raytrace_scalar(robot_x, robot_y, cos_angles[i], sin_angles[i]);
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return {distances, elapsed_ms};
}
