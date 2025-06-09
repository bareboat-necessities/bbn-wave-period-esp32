#pragma once

/*

  Kalman filter which estimates vector direction from X and Y components

*/

#include <math.h>

class VectorDirectionKF {
private:
    float theta;        // Estimated angle (radians)
    float theta_rate;   // Angular velocity (rad/s)
    float P[2][2];      // Covariance matrix
    float Q_angle;      // Process noise - angle
    float Q_rate;       // Process noise - rate
    float R_measure;    // Measurement noise
    
    // Safety limits
    const float MAX_RATE = 6.28319f;    // 2π rad/s (1 full rotation per second)
    const float MIN_COVARIANCE = 1e-6f; // Prevent covariance collapse

public:
    VectorDirectionKF(float initial_angle = 0.0f,
                      float angle_noise = 0.01f,
                      float rate_noise = 0.03f,
                      float meas_noise = 0.1f) :
        theta(initial_angle),
        theta_rate(0),
        Q_angle(angle_noise),
        Q_rate(rate_noise),
        R_measure(meas_noise) {
        reset(initial_angle);
    }

    void reset(float angle = 0.0f) {
        theta = angle;
        theta_rate = 0;
        // Initialize with reasonable uncertainty
        P[0][0] = 1.0f;
        P[0][1] = 0.5f;
        P[1][0] = 0.5f;
        P[1][1] = 1.0f;
    }

    void predict(float dt) {
        // Apply rate limiting
        float limited_rate = constraint(theta_rate, -MAX_RATE, MAX_RATE);
        
        // State prediction
        theta += limited_rate * dt;
        
        // Normalize angle first to prevent error accumulation
        theta = atan2(sin(theta), cos(theta));
        
        // Covariance prediction with safeguards
        float dt2 = dt * dt;
        P[0][0] += dt * (P[0][1] + P[1][0] + dt * P[1][1]) + fmaxf(Q_angle * dt, MIN_COVARIANCE);
        P[0][1] += dt * P[1][1];
        P[1][0] += dt * P[1][1];
        P[1][1] += fmaxf(Q_rate * dt, MIN_COVARIANCE);
        
        // Ensure positive definiteness
        P[0][0] = fmaxf(P[0][0], MIN_COVARIANCE);
        P[1][1] = fmaxf(P[1][1], MIN_COVARIANCE);
    }

    void update(float x, float y) {
        // Convert measurement to angle (-π to π)
        float meas_angle = atan2(y, x);
        
        // Calculate smallest angular difference
        float angle_diff = atan2(sin(meas_angle - theta), cos(meas_angle - theta));
        
        // Innovation covariance with safeguard
        float S = fmaxf(P[0][0] + R_measure, MIN_COVARIANCE);
        
        // Kalman gain
        float K[2];
        K[0] = P[0][0] / S;
        K[1] = P[1][0] / S;
        
        // State update with rate limiting
        theta = atan2(sin(theta + K[0] * angle_diff), cos(theta + K[0] * angle_diff));
        theta_rate = constraint(theta_rate + K[1] * angle_diff, -MAX_RATE, MAX_RATE);
        
        // Stabilized covariance update
        float P00 = P[0][0];
        float P01 = P[0][1];
        float P10 = P[1][0];
        float P11 = P[1][1];
        
        P[0][0] = (1 - K[0]) * P00;
        P[0][1] = (1 - K[0]) * P01;
        P[1][0] = -K[1] * P00 + P10;
        P[1][1] = -K[1] * P01 + P11;
        
        // Force minimum covariance and symmetry
        P[0][0] = fmaxf(P[0][0], MIN_COVARIANCE);
        P[1][1] = fmaxf(P[1][1], MIN_COVARIANCE);
        P[0][1] = P[1][0] = 0.5f * (P[0][1] + P[1][0]); // Maintain symmetry
    }

    void setAngleNoise(float noise) { Q_angle = noise; }
    void setRateNoise(float noise) { Q_rate = noise; }
    void setMeasNoise(float noise) { R_measure = noise; }

    float getAngle() const { return theta; }
    float getAngleDeg() const { return theta * 57.2957795f; }
    float getRate() const { return theta_rate; }

private:
    float constraint(float value, float min, float max) {
        return (value < min) ? min : ((value > max) ? max : value);
    }
};
