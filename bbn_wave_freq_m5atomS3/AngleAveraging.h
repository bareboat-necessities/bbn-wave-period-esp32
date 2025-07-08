#ifndef AngleAveraging_h
#define AngleAveraging_h

#include <math.h>

/*
  AngleAverager - Stateful angle averaging filter with adaptive or fixed alpha smoothing.

  - Automatically handles circular wraparound (e.g., 350° and 10° average to 0°)
  - Alpha < 0 enables adaptive smoothing using circular variance
  - Tracks previous angle and estimated variance
  - Suitable for embedded use (no heap or STL required)

  Usage:
      AngleAverager avg(0.1f);         // fixed alpha smoothing (10%)
      AngleAverager avg(-1.0f);        // adaptive alpha smoothing
      avg.reset(heading_start);        // optional explicit reset
      auto result = avg.average360(new_heading);  // returns filtered result
*/

class AngleAverager {
public:
    // Structure for filtered angle and quality metrics
    struct AngleEstimate {
        float angle = 0.0f;                    // Filtered angle in degrees
        float magnitude = 1e-12f;              // Confidence (0–1)
        float variance = M_PI * M_PI / 4.0f;   // Estimated circular variance (rad²)
        float consistency = 0.0f;              // Dot product similarity [0–1]
    };

    // Constructor: alpha < 0 enables adaptive smoothing based on circular variance
    explicit AngleAverager(float alpha_init = 0.004f)
        : alpha(alpha_init) {}

    // Explicitly set initial angle and reset variance
    void reset(float angle_deg) {
        angle_prev = angle_deg;
        variance_prev = M_PI * M_PI / 4.0f;
        initialized = true;
    }

    // Averaging in [0, 360) space (e.g., compass heading)
    AngleEstimate average360(float new_angle_deg) {
        if (!initialized) reset(new_angle_deg);

        float cx = cosf(deg2rad(angle_prev));
        float cy = sinf(deg2rad(angle_prev));
        float nx = cosf(deg2rad(new_angle_deg));
        float ny = sinf(deg2rad(new_angle_deg));

        float new_mag = magnitude(nx, ny);
        float new_var = estimate_variance(new_mag);
        float eff_alpha = (alpha >= 0.0f) ? alpha : adaptive_alpha(variance_prev, new_var);

        float fx = (1.0f - eff_alpha) * cx + eff_alpha * nx;
        float fy = (1.0f - eff_alpha) * cy + eff_alpha * ny;

        float filtered_angle = rad2deg(atan2f(fy, fx));
        if (filtered_angle < 0.0f) filtered_angle += 360.0f;

        float mag = magnitude(fx, fy);
        float var = estimate_variance(mag);
        float cons = consistency(cx, cy, nx, ny);

        angle_prev = filtered_angle;
        variance_prev = var;

        return {filtered_angle, mag, var, cons};
    }

    // Averaging in [0, 180) space (e.g., directional angle without sign)
    AngleEstimate average180(float new_angle_deg) {
        if (!initialized) reset(new_angle_deg);

        float cd = 2.0f * angle_prev;
        float nd = 2.0f * new_angle_deg;

        float cx = cosf(deg2rad(cd));
        float cy = sinf(deg2rad(cd));
        float nx = cosf(deg2rad(nd));
        float ny = sinf(deg2rad(nd));

        float new_mag = magnitude(nx, ny);
        float new_var = estimate_variance(new_mag);
        float eff_alpha = (alpha >= 0.0f) ? alpha : adaptive_alpha(variance_prev, new_var);

        float fx = (1.0f - eff_alpha) * cx + eff_alpha * nx;
        float fy = (1.0f - eff_alpha) * cy + eff_alpha * ny;

        float filtered_angle_doubled = rad2deg(atan2f(fy, fx));
        float filtered_angle = 0.5f * filtered_angle_doubled;
        if (filtered_angle < 0.0f) filtered_angle += 180.0f;

        float mag = magnitude(fx, fy);
        float var = estimate_variance(mag) / 4.0f;  // compensate for doubling
        float cons = consistency(cx, cy, nx, ny);

        angle_prev = filtered_angle;
        variance_prev = var;

        AngleEstimate result;
        result.angle = filtered_angle;
        result.magnitude = mag;
        result.variance = var;
        result.consistency = cons;
        return result;
    }

    static inline float deg2rad(float angle_deg) {
        return angle_deg * (M_PI / 180.0f);
    }

    static inline float rad2deg(float angle_rad) {
        return angle_rad * (180.0f / M_PI);
    }

    static inline float magnitude(float x, float y) {
        return sqrtf(x * x + y * y);
    }

    // Get last angle estimate (raw)
    float get_angle() const { return angle_prev; }

    // Get last variance estimate
    float get_variance() const { return variance_prev; }

    // Check if filter has been initialized
    bool is_initialized() const { return initialized; }

private:
    float alpha;
    bool initialized = false;
    float angle_prev = 0.0f;
    float variance_prev = M_PI * M_PI / 4.0f;

    static inline float estimate_variance(float mag) {
        if (mag <= 0.0f) return M_PI * M_PI;
        if (mag > 0.999f) return 0.0f;
        return -2.0f * logf(mag);
    }

    static inline float consistency(float x1, float y1, float x2, float y2) {
        float dot = fmaxf(-1.0f, fminf(1.0f, x1 * x2 + y1 * y2));
        return 0.5f * (1.0f + dot);
    }

    static inline float adaptive_alpha(float var_current, float var_new) {
        constexpr float eps = 1e-6f;
        float a = var_new / (var_current + var_new + eps);
        return fminf(fmaxf(a, 0.0f), 1.0f);
    }
};

typedef AngleAverager::AngleEstimate AngleEstimate; 

#endif
