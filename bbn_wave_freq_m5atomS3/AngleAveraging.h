#ifndef AngleAveraging_h
#define AngleAveraging_h

// Convert degrees to radians
#define DEG_TO_RAD(angle_deg) ((angle_deg) * M_PI / 180.0)

// Convert radians to degrees
#define RAD_TO_DEG(angle_rad) ((angle_rad) * 180.0 / M_PI)

// Low-pass filter for averaging angles (0-360 degrees)
float low_pass_angle_average_360(float current_angle, float new_angle, float alpha) {
    // Convert angles to unit vectors
    float current_x = cosf(DEG_TO_RAD(current_angle));
    float current_y = sinf(DEG_TO_RAD(current_angle));
    
    float new_x = cosf(DEG_TO_RAD(new_angle));
    float new_y = sinf(DEG_TO_RAD(new_angle));
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0 - alpha) * current_x + alpha * new_x;  // Smoothing factor (smaller alpha = smoother)
    float filtered_y = (1.0 - alpha) * current_y + alpha * new_y;
    
    // Compute the resulting angle (using atan2)
    float filtered_angle_rad = atan2f(filtered_y, filtered_x);
    float filtered_angle_deg = RAD_TO_DEG(filtered_angle_rad);
    
    // Ensure the result is in [0, 360)
    if (filtered_angle_deg < 0) {
        filtered_angle_deg += 360.0;
    }
    
    return filtered_angle_deg;
}

// Low-pass filter for angles in 0-180° range (rollover at 180°)
float low_pass_angle_average_180(float current_angle, float new_angle, float alpha) {
    // Double the angles to convert 180° wrap-around to 360°
    float current_doubled = 2.0 * current_angle;
    float new_doubled = 2.0 * new_angle;
    
    // Convert doubled angles to unit vectors
    float current_x = cosf(DEG_TO_RAD(current_doubled));
    float current_y = sinf(DEG_TO_RAD(current_doubled));
    
    float new_x = cosf(DEG_TO_RAD(new_doubled));
    float new_y = sinf(DEG_TO_RAD(new_doubled));
    
    // Apply low-pass filtering (weighted average)
    float filtered_x = (1.0 - alpha) * current_x + alpha * new_x;  // Smoothing factor (smaller alpha = smoother)
    float filtered_y = (1.0 - alpha) * current_y + alpha * new_y;
    
    // Compute the resulting angle in doubled space
    float filtered_angle_doubled_rad = atan2f(filtered_y, filtered_x);
    float filtered_angle_doubled_deg = RAD_TO_DEG(filtered_angle_doubled_rad);
    
    // Halve the angle to return to original range [0, 180)
    float filtered_angle_deg = 0.5 * filtered_angle_doubled_deg;
    
    // Ensure the result is in [0, 180)
    if (filtered_angle_deg < 0) {
        filtered_angle_deg += 180.0;
    }
    
    return filtered_angle_deg;
}

#endif
