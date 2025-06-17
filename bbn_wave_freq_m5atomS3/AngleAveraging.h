#ifndef AngleAveraging_h
#define AngleAveraging_h

#include <math.h>

// Convert degrees to radians
#define DEG_TO_RAD_UTIL(angle_deg) ((angle_deg) * M_PI / 180.0)

// Convert radians to degrees
#define RAD_TO_DEG_UTIL(angle_rad) ((angle_rad) * 180.0 / M_PI)

// Structure to hold angle estimate and quality metrics
typedef struct {
    float angle;            // Filtered angle estimate
    float magnitude;        // Magnitude of the resultant vector (0-1, higher is better)
    float variance;         // Estimated variance of the measurement (radians^2)
    float consistency;      // Consistency between current and new measurement (0-1, higher is better)
} AngleEstimate;

// Helper function to calculate vector magnitude
static inline float calculate_magnitude(float x, float y) {
    return sqrtf(x * x + y * y);
}

// Helper function to estimate circular variance from magnitude
static inline float estimate_variance(float magnitude) {
    if (magnitude > 0.999f) {
        return 0.0f;  // Virtually no variance
    }
    return -2.0f * logf(magnitude);
}

// Helper function to calculate consistency between two vectors
static inline float calculate_consistency(float x1, float y1, float x2, float y2) {
    float dot_product = x1 * x2 + y1 * y2;
    return 0.5f * (1.0f + dot_product);  // Maps [-1,1] to [0,1]
}

// Low-pass filter for averaging angles (0-360 degrees) with quality metrics
AngleEstimate low_pass_angle_average_360(float current_angle, float new_angle, float alpha, float current_variance) {
    AngleEstimate result;
    
    // Convert angles to unit vectors
    float current_x = cosf(DEG_TO_RAD_UTIL(current_angle));
    float current_y = sinf(DEG_TO_RAD_UTIL(current_angle));
    
    float new_x = cosf(DEG_TO_RAD_UTIL(new_angle));
    float new_y = sinf(DEG_TO_RAD_UTIL(new_angle));
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0 - alpha) * current_x + alpha * new_x;  // Smoothing factor (smaller alpha = smoother)
    float filtered_y = (1.0 - alpha) * current_y + alpha * new_y;
    
    // Compute the resulting angle (using atan2)
    float filtered_angle_rad = atan2f(filtered_y, filtered_x);
    result.angle = RAD_TO_DEG_UTIL(filtered_angle_rad);
    
    // Ensure the result is in [0, 360)
    if (result.angle < 0) {
        result.angle += 360.0;
    }
    
    // Calculate quality metrics using helper functions
    result.magnitude = calculate_magnitude(filtered_x, filtered_y);
    result.variance = estimate_variance(result.magnitude);
    result.consistency = calculate_consistency(current_x, current_y, new_x, new_y);
    
    return result;
}

// Low-pass filter for angles in 0-180° range (rollover at 180°) with quality metrics
AngleEstimate low_pass_angle_average_180(float current_angle, float new_angle, float alpha, float current_variance) {
    AngleEstimate result;
    
    // Double the angles to convert 180° wrap-around to 360°
    float current_doubled = 2.0 * current_angle;
    float new_doubled = 2.0 * new_angle;
    
    // Convert doubled angles to unit vectors
    float current_x = cosf(DEG_TO_RAD_UTIL(current_doubled));
    float current_y = sinf(DEG_TO_RAD_UTIL(current_doubled));
    
    float new_x = cosf(DEG_TO_RAD_UTIL(new_doubled));
    float new_y = sinf(DEG_TO_RAD_UTIL(new_doubled));
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0 - alpha) * current_x + alpha * new_x;  // Smoothing factor (smaller alpha = smoother)
    float filtered_y = (1.0 - alpha) * current_y + alpha * new_y;
    
    // Compute the resulting angle in doubled space
    float filtered_angle_doubled_rad = atan2f(filtered_y, filtered_x);
    float filtered_angle_doubled_deg = RAD_TO_DEG_UTIL(filtered_angle_doubled_rad);
    
    // Halve the angle to return to original range [0, 180)
    result.angle = 0.5 * filtered_angle_doubled_deg;
    
    // Ensure the result is in [0, 180)
    if (result.angle < 0) {
        result.angle += 180.0;
    }
    
    // Calculate quality metrics using helper functions (with doubled angles)
    result.magnitude = calculate_magnitude(filtered_x, filtered_y);
    result.variance = estimate_variance(result.magnitude) / 4.0f; // Adjust for angle doubling
    result.consistency = calculate_consistency(current_x, current_y, new_x, new_y);
    
    return result;
}

#endif
