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
    const float MAX_RATE = 3.14159f * 2; // 2π rad/s max rotation

public:
    VectorDirectionKF(float initial_angle = 0.0f,
                      float angle_noise = 0.001f,
                      float rate_noise = 0.003f,
                      float meas_noise = 0.03f) :
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
        P[0][1] = 0;
        P[1][0] = 0;
        P[1][1] = 1.0f;
    }

    void predict(float dt) {
        // State prediction with rate limiting
        theta += constrain(theta_rate, -MAX_RATE, MAX_RATE) * dt;
        
        // Covariance prediction (prevent collapse)
        P[0][0] += dt * (dt*P[1][1] + P[0][1] + P[1][0] + fmaxf(Q_angle, 0.000001f));
        P[0][1] += dt * P[1][1];
        P[1][0] += dt * P[1][1];
        P[1][1] += dt * fmaxf(Q_rate, 0.000001f);
        
        // Normalize angle to [-π, π]
        theta = atan2(sin(theta), cos(theta));
    }

    void update(float x, float y) {
        // Convert measurement to angle
        float meas_angle = atan2(y, x);
        
        // Calculate smallest angle difference
        float angle_diff = atan2(sin(meas_angle - theta), cos(meas_angle - theta));
        
        // Prevent covariance collapse
        float S = fmaxf(P[0][0] + R_measure, 0.000001f);
        
        // Kalman gain
        float K[2];
        K[0] = P[0][0] / S;
        K[1] = P[1][0] / S;
        
        // State update with rate limiting
        theta += K[0] * angle_diff;
        theta_rate = constrain(theta_rate + K[1] * angle_diff, -MAX_RATE, MAX_RATE);
        
        // Joseph-form covariance update (more stable)
        float P00 = P[0][0];
        float P01 = P[0][1];
        P[0][0] = (1 - K[0]) * P00;
        P[0][1] = (1 - K[0]) * P01;
        P[1][0] = -K[1] * P00 + P[1][0];
        P[1][1] = -K[1] * P01 + P[1][1];
        
        // Force minimum covariance
        P[0][0] = fmaxf(P[0][0], 0.000001f);
        P[1][1] = fmaxf(P[1][1], 0.000001f);
    }

    void setAngleNoise(float noise) { Q_angle = noise; }
    void setRateNoise(float noise) { Q_rate = noise; }
    void setMeasNoise(float noise) { R_measure = noise; }

    float getAngle() const { return theta; }
    float getAngleDeg() const { return theta * 57.2957795f; }
    float getRate() const { return theta_rate; }
    
    // Utility function
    float constrain(float value, float min, float max) {
        return (value < min) ? min : (value > max) ? max : value;
    }
};
