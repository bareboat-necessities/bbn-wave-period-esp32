#pragma once

/*

  Kalman filter which estimates vector direction from X and Y components

*/

#include <math.h>

class VectorDirectionKF {
private:
    // State variables
    float sin_theta;    // sin(θ)
    float cos_theta;    // cos(θ)
    float omega;        // Angular velocity (rad/s)
    
    // Covariance matrix [P_θθ, P_θω; P_ωθ, P_ωω]
    float P[2][2];
    
    // Noise parameters
    float Q_angle;      // Process noise - angle
    float Q_omega;      // Process noise - angular velocity
    float R;            // Measurement noise
    
    // Constraints
    const float MAX_OMEGA = 6.2832f; // 2π rad/s (1 full rotation per second)
    const float MIN_COV = 1e-8f;     // Minimum covariance

public:
    VectorDirectionKF(float initial_angle = 0.0f,
                      float angle_noise = 0.01f,
                      float omega_noise = 0.1f,
                      float meas_noise = 0.1f) {
        reset(initial_angle);
        setNoises(angle_noise, omega_noise, meas_noise);
    }

    void reset(float angle = 0.0f) {
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        omega = 0;
        P[0][0] = 1.0f; P[0][1] = 0;
        P[1][0] = 0;    P[1][1] = 1.0f;
    }

    // Noise parameter setters
    void setAngleNoise(float noise) { Q_angle = noise; }
    void setOmegaNoise(float noise) { Q_omega = noise; }
    void setMeasNoise(float noise) { R = noise; }

    void setNoises(float angle_noise, float omega_noise, float meas_noise) {
        Q_angle = angle_noise;
        Q_omega = omega_noise;
        R = meas_noise;
    }

    void predict(float dt) {
        // State prediction (include angular velocity)
        float predicted_theta = atan2(sin_theta, cos_theta) + omega * dt;
        sin_theta = sin(predicted_theta);
        cos_theta = cos(predicted_theta);
        
        // Covariance prediction
        float dt2 = dt * dt;
        P[0][0] += dt * (2 * P[0][1] + dt * P[1][1] + Q_angle);
        P[0][1] += dt * P[1][1];
        P[1][0] = P[0][1];
        P[1][1] += dt * Q_omega;
        
        // Ensure minimum covariance
        P[0][0] = fmaxf(P[0][0], MIN_COV);
        P[1][1] = fmaxf(P[1][1], MIN_COV);
    }

    void update(float x, float y) {
        // Normalize measurement
        float mag = sqrt(x*x + y*y);
        if (mag < 1e-7f) return;
        float meas_sin = y/mag;
        float meas_cos = x/mag;
        
        // Calculate angle difference
        float angle_diff = atan2(
            meas_sin * cos_theta - meas_cos * sin_theta,
            meas_cos * cos_theta + meas_sin * sin_theta
        );
        
        // Kalman gain calculation
        float S = P[0][0] + R;
        float K[2] = {P[0][0]/S, P[1][0]/S};
        
        // State update
        float new_theta = atan2(sin_theta, cos_theta) + K[0] * angle_diff;
        sin_theta = sin(new_theta);
        cos_theta = cos(new_theta);
        
        omega = constraint(omega + K[1] * angle_diff, -MAX_OMEGA, MAX_OMEGA);
        
        // Covariance update (Joseph form)
        float P00 = P[0][0];
        float P01 = P[0][1];
        P[0][0] = (1 - K[0]) * P00;
        P[0][1] = (1 - K[0]) * P01;
        P[1][0] = -K[1] * P00 + P[1][0];
        P[1][1] = -K[1] * P01 + P[1][1];
        
        // Ensure symmetry and minimum covariance
        P[0][0] = fmaxf(P[0][0], MIN_COV);
        P[1][1] = fmaxf(P[1][1], MIN_COV);
        P[1][0] = P[0][1] = 0.5f * (P[0][1] + P[1][0]);
    }

    // Getters
    float getAngle() const { return atan2(sin_theta, cos_theta); }
    float getAngleDeg() const { return atan2(sin_theta, cos_theta) * 57.2958f; }
    float getOmega() const { return omega; }
    float getOmegaDeg() const { return omega * 57.2958f; }

private:
    float constraint(float val, float min, float max) {
        return val < min ? min : (val > max ? max : val);
    }
};
