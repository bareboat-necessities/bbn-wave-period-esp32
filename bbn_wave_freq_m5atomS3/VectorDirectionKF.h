#pragma once

/*

  Kalman filter which estimates vector direction from X and Y components

*/

#include <math.h>

class VectorDirectionKF {
private:
    // State variables
    float sin_theta, cos_theta;  // Unit vector components
    float omega;                 // Angular velocity (rad/s)
    float P[2][2];               // Covariance matrix
    
    // Noise parameters
    float Q_theta;               // Angle process noise
    float Q_omega;               // Omega process noise
    float R_angle;               // Angle measurement noise
    float R_omega;               // Omega pseudo-measurement noise
    
    // Constraints
    const float MAX_OMEGA = 6.2832f; // 2π rad/s max
    const float MIN_COV = 1e-8f;

public:
    VectorDirectionKF(float initial_angle = 0.0f,
                      float theta_noise = 1e-2f,
                      float omega_noise = 1e-4f,
                      float angle_meas_noise = 0.09f,
                      float omega_meas_noise = 1.0f) {
        reset(initial_angle);
        setNoises(theta_noise, omega_noise, angle_meas_noise, omega_meas_noise);
    }

    void reset(float angle = 0.0f) {
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        omega = 0;
        P[0][0] = 1.0f; P[0][1] = 0;
        P[1][0] = 0;    P[1][1] = 1.0f;
    }

    void setNoises(float q_th, float q_om, float r_th, float r_om) {
        Q_theta = q_th;
        Q_omega = q_om;
        R_angle = r_th;
        R_omega = r_om;
    }

    void predict(float dt) {
        // State prediction
        float new_theta = atan2(sin_theta, cos_theta) + omega * dt;
        sin_theta = sin(new_theta);
        cos_theta = cos(new_theta);
        
        // Covariance prediction
        float dt2 = dt*dt;
        P[0][0] += dt*(2*P[0][1] + dt*P[1][1] + Q_theta);
        P[0][1] += dt*P[1][1];
        P[1][0] = P[0][1];
        P[1][1] += dt*Q_omega;
        
        // Apply minimum covariance
        P[0][0] = fmaxf(P[0][0], MIN_COV);
        P[1][1] = fmaxf(P[1][1], MIN_COV);
    }

    void update(float x, float y, bool use_omega_pseudo = true) {
        // Normalize measurement
        float mag = sqrt(x*x + y*y);
        if (mag < 1e-9f) return;
        float meas_sin = y/mag;
        float meas_cos = x/mag;
        
        // Angle update
        float angle_diff = atan2(meas_sin, meas_cos) - atan2(sin_theta, cos_theta);
        angle_diff = atan2(sin(angle_diff), cos(angle_diff));  // Normalize to [-π, π]
        
        // Combined measurement vector [angle_diff; 0] (pseudo-measurement)
        float y_vec[2] = {angle_diff, -omega}; // omega pseudo-measurement
        
        // Measurement matrix
        float H[2][2] = {{1, 0}, {0, 1}};
        
        // Residual covariance
        float S[2][2] = {
            {P[0][0] + R_angle,    P[0][1]},
            {P[1][0],           P[1][1] + R_omega}
        };
        
        // Kalman gain
        float detS = S[0][0]*S[1][1] - S[0][1]*S[1][0];
        float K[2][2] = {
            {(P[0][0]*S[1][1] - P[0][1]*S[1][0])/detS, (P[0][1]*S[0][0] - P[0][0]*S[0][1])/detS},
            {(P[1][0]*S[1][1] - P[1][1]*S[1][0])/detS, (P[1][1]*S[0][0] - P[1][0]*S[0][1])/detS}
        };
        
        // State update
        float new_theta = atan2(sin_theta, cos_theta) + K[0][0]*y_vec[0] + K[0][1]*y_vec[1];
        sin_theta = sin(new_theta);
        cos_theta = cos(new_theta);
        omega += K[1][0]*y_vec[0] + K[1][1]*y_vec[1];
        omega = constraint(omega, -MAX_OMEGA, MAX_OMEGA);
        
        // Covariance update (Joseph form)
        float IKH[2][2] = {
            {1 - K[0][0], -K[0][1]},
            {-K[1][0], 1 - K[1][1]}
        };
        float P_new[2][2];
        for(int i=0; i<2; i++) {
            for(int j=0; j<2; j++) {
                P_new[i][j] = IKH[i][0]*P[0][j] + IKH[i][1]*P[1][j];
            }
        }
        P[0][0] = fmaxf(P_new[0][0], MIN_COV);
        P[0][1] = P_new[0][1];
        P[1][0] = P_new[1][0];
        P[1][1] = fmaxf(P_new[1][1], MIN_COV);
    }

    float getAngle() const { return atan2(sin_theta, cos_theta); }
    float getAngleDeg() const { return getAngle() * 57.2958f; }
    float getOmega() const { return omega; }

private:
    float constraint(float val, float min, float max) {
        return val < min ? min : (val > max ? max : val);
    }
};
