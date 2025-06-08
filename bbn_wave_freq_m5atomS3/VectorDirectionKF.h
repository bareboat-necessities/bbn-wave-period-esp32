#pragma once

#include <ArduinoEigenDense.h>

/*

  Kalman filter which estimates vector direction from X and Y components

*/

class VectorDirectionKF {
private:
    float theta;    // Estimated angle (radians)
    float P;        // Estimation error variance (scalar)
    float Q;        // Process noise variance
    float R;        // Measurement noise variance

public:
    VectorDirectionKF(float initial_angle = 0.0f, 
                     float initial_uncertainty = 1.0f,
                     float process_noise = 0.01f,
                     float measurement_noise = 1.0f) :
        theta(initial_angle),
        P(initial_uncertainty),
        Q(process_noise),
        R(measurement_noise) {}

    void predict() {
        P += Q;
    }

    void update(float x, float y) {
        // Since we're estimating a scalar (angle), we need to handle the H matrix properly
        float H_x = -sin(theta);  // Partial derivative of x measurement
        float H_y = cos(theta);   // Partial derivative of y measurement
        
        // Residuals
        float y_x = x - cos(theta);
        float y_y = y - sin(theta);
        
        // Innovation covariance (scalar)
        float S = (H_x * P * H_x) + (H_y * P * H_y) + R;
        
        // Kalman gain (scalar for our 1D state)
        float K = P * (H_x + H_y) / S;
        
        // State update
        theta += K * (y_x * H_x + y_y * H_y);
        
        // Covariance update (Joseph form for stability)
        P = (1 - K * (H_x + H_y)) * P;
        
        // Normalize angle to [-π, π]
        theta = atan2(sin(theta), cos(theta));
    }

    float getAngle() const { return theta; }
    float getAngleDegrees() const { return theta * 180.0f / M_PI; }

    // Set process noise (affects smoothing)
    void setProcessNoise(float q) {
        Q = q;
    }

    // Set measurement noise (affects smoothing)
    void setMeasurementNoise(float r) {
        R = r;
    }
};
