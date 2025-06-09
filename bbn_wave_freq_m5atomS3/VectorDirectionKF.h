#pragma once

/*

  Kalman filter which estimates vector direction from X and Y components

*/

#include <math.h>

class VectorDirectionKF {
private:
    float sin_theta;    // sin(θ)
    float cos_theta;    // cos(θ)
    float P;            // Variance estimate
    float Q;            // Process noise
    float R;            // Measurement noise
    float alpha;        // Adaptive filter gain

    void update_gain() {
        alpha = Q / (Q + R);
    }

public:
    VectorDirectionKF(float initial_angle = 0.0f,
                      float process_noise = 0.01f,
                      float meas_noise = 0.1f) {
        reset(initial_angle);
        setProcessNoise(process_noise);
        setMeasurementNoise(meas_noise);
    }

    // Reset filter with new angle
    void reset(float angle = 0.0f) {
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        P = 1.0f;  // Initial uncertainty
    }

    void setProcessNoise(float process_noise) {
        Q = process_noise;
        update_gain();
    }

    void setMeasurementNoise(float meas_noise) {
        R = meas_noise;
        update_gain();
    }

    // Prediction step
    void predict() {
        P += Q;  // Increase uncertainty
    }

    // Update step
    void update(float x, float y) {
        // Normalize measurement vector
        float mag = sqrt(x*x + y*y);
        if (mag < 1e-9f) return;  // Avoid division by zero
        
        float meas_sin = y/mag;
        float meas_cos = x/mag;
        
        // Update estimates (weighted average)
        sin_theta = (1-alpha)*sin_theta + alpha*meas_sin;
        cos_theta = (1-alpha)*cos_theta + alpha*meas_cos;
        
        // Normalize the estimate
        mag = sqrt(sin_theta*sin_theta + cos_theta*cos_theta);
        sin_theta /= mag;
        cos_theta /= mag;
        
        // Update variance estimate
        P = (1-alpha)*P + alpha*R;
    }

    // Get current estimates
    float getAngle() const { return atan2(sin_theta, cos_theta); }
    float getAngleDeg() const { return atan2(sin_theta, cos_theta) * 57.2957795f; }
};
