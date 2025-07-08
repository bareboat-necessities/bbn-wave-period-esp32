#ifndef AngleAveraging_h
#define AngleAveraging_h

#include <math.h>

// Convert degrees to radians
#define DEG_TO_RAD_UTIL(angle_deg) ((angle_deg) * M_PI / 180.0)

// Convert radians to degrees
#define RAD_TO_DEG_UTIL(angle_rad) ((angle_rad) * 180.0 / M_PI)

// Structure to hold angle estimate and quality metrics
typedef struct {
    float angle = 0.0f;                    // Filtered angle estimate
    float magnitude = 1e-12f;              // Magnitude of the resultant vector (0-1, higher is better)
    float variance = M_PI * M_PI / 4.0f;   // Estimated variance of the measurement (radians^2)
    float consistency = 0.0f;              // Consistency between current and new measurement (0-1, higher is better)
} AngleEstimate;

// Helper function to calculate vector magnitude
static inline float calculate_magnitude(const float x, const float y) {
    return sqrtf(x * x + y * y);
}

// Helper function to estimate circular variance from magnitude
static inline float estimate_variance(const float magnitude) {
    if (magnitude > 0.999f) {
        return 0.0f;  // Virtually no variance
    }
    return -2.0f * logf(magnitude);
}

// Helper function to calculate consistency between two vectors
static inline float calculate_consistency(const float x1, const float y1, const float x2, const float y2) {
    float dot_product = x1 * x2 + y1 * y2;
    return 0.5f * (1.0f + dot_product);  // Maps [-1,1] to [0,1]
}

// Calculate adaptive alpha based on current variance and new measurement variance
static inline float calculate_adaptive_alpha(float current_variance, float new_variance) {
    // Kalman-like adaptive alpha: alpha = new_variance / (current_variance + new_variance)
    // Add small epsilon to avoid division by zero
    const float epsilon = 1e-6f;
    return new_variance / (current_variance + new_variance + epsilon);
}

// Low-pass filter for averaging angles (0°-360°) with quality metrics. Smoothing factor (smaller alpha = smoother)
// Negative current_variance - use fixed alpha. Otherwise alpha is adaptive and recalculated
AngleEstimate low_pass_angle_average_360(float current_angle, float new_angle, float alpha, float current_variance) {
    // Convert angles to unit vectors
    float current_x = cosf(DEG_TO_RAD_UTIL(current_angle));
    float current_y = sinf(DEG_TO_RAD_UTIL(current_angle));
    
    float new_x = cosf(DEG_TO_RAD_UTIL(new_angle));
    float new_y = sinf(DEG_TO_RAD_UTIL(new_angle));
    
    // Calculate new measurement variance from the new vector
    float new_magnitude = calculate_magnitude(new_x, new_y);
    float new_variance = estimate_variance(new_magnitude);
    
    // Determine which alpha to use based on current_variance
    float effective_alpha = (current_variance < 0.0f) ? alpha : calculate_adaptive_alpha(current_variance, new_variance);
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0f - effective_alpha) * current_x + effective_alpha * new_x;
    float filtered_y = (1.0f - effective_alpha) * current_y + effective_alpha * new_y;
    
    // Compute the resulting angle (using atan2)
    float filtered_angle_rad = atan2f(filtered_y, filtered_x);

    AngleEstimate result;
    result.angle = RAD_TO_DEG_UTIL(filtered_angle_rad);
    
    // Ensure the result is in [0, 360)
    if (result.angle < 0.0f) {
        result.angle += 360.0f;
    }
    
    // Calculate quality metrics using helper functions
    result.magnitude = calculate_magnitude(filtered_x, filtered_y);
    result.variance = estimate_variance(result.magnitude);
    result.consistency = calculate_consistency(current_x, current_y, new_x, new_y);
    
    return result;
}

// Low-pass filter for angles in 0-180° range (rollover at 180°) with quality metrics. Smoothing factor (smaller alpha = smoother)
// Negative current_variance - use fixed alpha. Otherwise alpha is adaptive and recalculated
AngleEstimate low_pass_angle_average_180(float current_angle, float new_angle, float alpha, float current_variance) {
    // Double the angles to convert 180° wrap-around to 360°
    float current_doubled = 2.0f * current_angle;
    float new_doubled = 2.0f * new_angle;
    
    // Convert doubled angles to unit vectors
    float current_x = cosf(DEG_TO_RAD_UTIL(current_doubled));
    float current_y = sinf(DEG_TO_RAD_UTIL(current_doubled));
    
    float new_x = cosf(DEG_TO_RAD_UTIL(new_doubled));
    float new_y = sinf(DEG_TO_RAD_UTIL(new_doubled));
    
    // Calculate new measurement variance from the new vector
    float new_magnitude = calculate_magnitude(new_x, new_y);
    float new_variance = estimate_variance(new_magnitude);
    
    // Determine which alpha to use based on current_variance
    float effective_alpha = (current_variance < 0.0f) ? alpha : calculate_adaptive_alpha(current_variance, new_variance);
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0f - effective_alpha) * current_x + effective_alpha * new_x;
    float filtered_y = (1.0f - effective_alpha) * current_y + effective_alpha * new_y;
    
    // Compute the resulting angle in doubled space
    float filtered_angle_doubled_rad = atan2f(filtered_y, filtered_x);
    float filtered_angle_doubled_deg = RAD_TO_DEG_UTIL(filtered_angle_doubled_rad);

    AngleEstimate result; 
    // Halve the angle to return to original range [0, 180)
    result.angle = 0.5f * filtered_angle_doubled_deg;
    
    // Ensure the result is in [0, 180)
    if (result.angle < 0.0f) {
        result.angle += 180.0f;
    }
    
    // Calculate quality metrics using helper functions (with doubled angles)
    result.magnitude = calculate_magnitude(filtered_x, filtered_y);
    result.variance = estimate_variance(result.magnitude) / 4.0f; // Adjust for angle doubling
    result.consistency = calculate_consistency(current_x, current_y, new_x, new_y);
    
    return result;
}

#endif
