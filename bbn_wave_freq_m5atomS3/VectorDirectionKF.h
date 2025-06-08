#pragma once

#include <ArduinoEigenDense.h>

/*

  Kalman filter which estimates vector direction from X and Y components

*/

class VectorDirectionKF {
private:
    float theta;    // Estimated angle (radians)
    float P;        // Estimation error variance
    float Q;        // Process noise variance
    float R;        // Measurement noise variance

public:
    // Constructor
    VectorDirectionKF(float initial_angle = 0.0f, 
                      float initial_uncertainty = 1.0f,
                      float process_noise = 0.01f,
                      float measurement_noise = 1.0f) :
        theta(initial_angle),
        P(initial_uncertainty),
        Q(process_noise),
        R(measurement_noise) {
    }

    // Prediction step
    void predict() {
        P += Q;  // Uncertainty grows with process noise
    }

    // Update step with x,y measurements
    void update(float x, float y) {
        // Compute expected measurements
        float h_x = cos(theta);
        float h_y = sin(theta);
        
        // Measurement residuals
        float y_x = x - h_x;
        float y_y = y - h_y;
        
        // Linearized measurement matrix H
        Eigen::Matrix<float, 1, 2> H;
        H << -sin(theta), cos(theta);
        
        // Residual covariance (scalar in this case)
        float S = (H * P * H.transpose())(0) + R;
        
        // Kalman gain (2x1 matrix)
        Eigen::Matrix<float, 2, 1> K = (P * H.transpose()) / S;
        
        // Update state estimate
        theta += (K(0) * y_x + K(1) * y_y);
        
        // Update estimate uncertainty
        P = (1 - (K.transpose() * H)(0)) * P;
        
        // Normalize angle to [-π, π]
        theta = atan2(sin(theta), cos(theta));
    }

    // Get current estimate in radians
    float getAngle() const {
        return theta;
    }

    // Get current estimate in degrees
    float getAngleDegrees() const {
        return theta * 180.0f / M_PI;
    }

    // Set process noise (affects smoothing)
    void setProcessNoise(float q) {
        Q = q;
    }

    // Set measurement noise (affects smoothing)
    void setMeasurementNoise(float r) {
        R = r;
    }
};
