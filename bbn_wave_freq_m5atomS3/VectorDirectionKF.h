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
        // Initialize covariance matrix
        P[0][0] = 0;
        P[0][1] = 0;
        P[1][0] = 0;
        P[1][1] = 0;
    }

    // Noise parameter setters
    void setAngleNoise(float noise) { Q_angle = noise; }
    void setRateNoise(float noise) { Q_rate = noise; }
    void setMeasNoise(float noise) { R_measure = noise; }

    // Reset the filter
    void reset(float angle = 0.0f, float rate = 0.0f) {
        theta = angle;
        theta_rate = rate;
        P[0][0] = P[0][1] = P[1][0] = P[1][1] = 0;
    }

    // Prediction with explicit time step
    void predict(float dt) {
        // State prediction
        theta += theta_rate * dt;
        
        // Covariance prediction
        P[0][0] += dt * (dt*P[1][1] + P[0][1] + P[1][0] + Q_angle);
        P[0][1] += dt * P[1][1];
        P[1][0] += dt * P[1][1];
        P[1][1] += dt * Q_rate;
        
        // Normalize angle
        theta = atan2(sin(theta), cos(theta));
    }

    void update(float x, float y) {
        // Convert measurement to angle
        float meas_angle = atan2(y, x);
        
        // Calculate angle difference (handles wrap-around)
        float angle_diff = atan2(sin(meas_angle - theta), cos(meas_angle - theta));
        
        // Innovation covariance
        float S = P[0][0] + R_measure;
        
        // Kalman gain
        float K[2];
        K[0] = P[0][0] / S;
        K[1] = P[1][0] / S;
        
        // State update
        theta += K[0] * angle_diff;
        theta_rate += K[1] * angle_diff;
        
        // Covariance update
        float P00_temp = P[0][0];
        float P01_temp = P[0][1];
        
        P[0][0] -= K[0] * P00_temp;
        P[0][1] -= K[0] * P01_temp;
        P[1][0] -= K[1] * P00_temp;
        P[1][1] -= K[1] * P01_temp;
        
        // Normalize angle
        theta = atan2(sin(theta), cos(theta));
    }

    // Getters
    float getAngle() const { return theta; }
    float getAngleDeg() const { return theta * 57.2957795f; }  // 180/π
    float getRate() const { return theta_rate; }
};
