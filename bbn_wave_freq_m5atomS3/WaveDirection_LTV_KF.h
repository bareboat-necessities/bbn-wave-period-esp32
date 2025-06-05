#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter for wave direction estimation.

  Wave direction is given by the plane in which we observe oscillations of horizontal acceleration.
  In case of trochoidal wave model those oscillations are harmonic.

  This model assumes x, y axis acceleration measurements have constant biases and Gaussian noise.
  True x, y accelerations without bias and noise are harmonic and have same phase (or move in counter phase). 
  Frequency is considered known and is a parameter on each step.

  See details in:  https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/30

*/

#include <ArduinoEigenDense.h>  // Eigen for matrix operations

using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector2f = Eigen::Matrix<float, 2, 1>;
using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Matrix2f = Eigen::Matrix<float, 2, 2>;
using Matrix26f = Eigen::Matrix<float, 2, 6>;
using Matrix62f = Eigen::Matrix<float, 6, 2>;

class WaveDirection_LTV_KF {
public:
    WaveDirection_LTV_KF(float omega, const Vector6f& initial_state,
                         const Matrix6f& initial_covariance,
                         const Matrix6f& process_noise,
                         const Matrix2f& measurement_noise)
        : omega_(omega), state_(initial_state), P_(initial_covariance),
          Q_(process_noise), R_(measurement_noise) {}

    // Prediction step (state is constant)
    void predict() {
        // P = F * P * F^T + Q (F is identity, so just add Q)
        P_ += Q_;
    }

    // Update step with new measurements
    void update(float t, float omega, float x_meas, float y_meas) {
        omega_ = omega;
      
        // Precompute trig values
        const float wt = omega_ * t;
        const float cos_wt = cos(wt);
        const float sin_wt = sin(wt);
        
        // Measurement matrix H (2x6)
        Matrix26f H;
        H << cos_wt, sin_wt, 0, 0, 1, 0,
             0, 0, cos_wt, sin_wt, 0, 1;
        
        // Predicted measurements
        Vector2f z_pred = H * state_;
        
        // Measurement vector
        Vector2f z;
        z << x_meas, y_meas;
        
        // Measurement residual
        Vector2f residual = z - z_pred;
        
        // Kalman update equations
        Matrix2f S = H * P_ * H.transpose() + R_;
        Matrix62f K = P_ * H.transpose() * S.inverse();
        
        // State update
        state_ += K * residual;
        
        // Covariance update
        P_ -= K * H * P_;
    }

    // Get estimated parameters
    float get_A() const {
        return sqrtf(state_[0]*state_[0] + state_[1]*state_[1]);
    }

    float get_B() const {
        return sqrtf(state_[2]*state_[2] + state_[3]*state_[3]);
    }

    float get_phase() const {
        return atan2f(state_[1], state_[0]);  // φ = atan2(A_Q, A_I)
    }

    float get_abs_theta() const {
        return atan2f(get_A(), get_B());  // θ = atan2(A, B)
    }

    float get_bias_x() const { return state_[4]; }
    float get_bias_y() const { return state_[5]; }

    // Get estimated state
    Vector6f getState() const { return state_; }

private:
    float omega_;    // Known angular frequency (rad/s)
    Vector6f state_; // [A_I, A_Q, B_I, B_Q, b_x, b_y]
    Matrix6f P_;     // Covariance matrix
    Matrix6f Q_;     // Process noise covariance
    Matrix2f R_;     // Measurement noise covariance
};

// Example usage
void test_setup() {
    Serial.begin(115200);
    delay(2000);  // Wait for serial monitor

    // System parameters
    const float freq = 0.3f;              // Signal frequency (Hz)
    const float omega = 2 * M_PI * freq;  // Angular frequency (rad/s)
    
    // True parameters (for simulation)
    const float true_A = 1.0f;
    const float true_B = -1.5f;
    const float true_phi = -0.5f * M_PI;  // Phase (rad)
    const float true_bx = 0.1f;
    const float true_by = -0.2f;
    
    // Noise parameters
    const float measurement_noise = 0.3f;  // Standard deviation

    Vector6f initial_state = [] {
      Vector6f tmp;
      tmp << true_A * cosf(true_phi),  // A_I
             true_A * sinf(true_phi),  // A_Q
             true_B * cosf(true_phi),  // B_I
             true_B * sinf(true_phi),  // B_Q
             0.0f, 0.0f;               // Initial bias estimates (0)
      return tmp;
    }(); // Initial state: [A_I, A_Q, B_I, B_Q, b_x, b_y]                             
    
    Matrix6f initial_cov = [] {
      Matrix6f tmp = Matrix6f::Identity() * 100.0f;
      return tmp;
    }(); // Initial covariance (high uncertainty)
  
    // Process noise (small values for constant parameters)
    Matrix6f Q = Matrix6f::Identity() * 1e-6f;
    
    // Measurement noise
    Matrix2f R = Matrix2f::Identity() * (measurement_noise * measurement_noise);
    
    // Initialize Kalman filter
    WaveDirection_LTV_KF kf(omega, initial_state, initial_cov, Q, R);
    
    // Simulation parameters
    const float dt = 0.004f;  // Time step (4 ms)
    const int steps = 5000;  
    float t = 0.0f;
    
    Serial.println("Starting Kalman Filter...");
    Serial.println("Time\tA_true\tA_est\tB_true\tB_est\tPhi_true\tPhi_est\tTheta_est");
    
    for (int i = 0; i < steps; i++, t += dt) {
        // Generate true signals (simulation only)
        float true_angle = omega * t + true_phi;
        float x_true = true_A * sinf(true_angle);
        float y_true = true_B * sinf(true_angle);
        
        // Add measurement noise
        float x_meas = x_true + measurement_noise * (rand() % 100 - 50) / 50.0f + true_bx;
        float y_meas = y_true + measurement_noise * (rand() % 100 - 50) / 50.0f + true_by;
        
        // Kalman filter steps
        kf.predict();
        kf.update(t, omega + 0.001 * (rand() % 100 - 50) / 50.0f, x_meas, y_meas);
        
        // Log results every 100 steps (1 second)
        if (i % 100 == 0) {
            Serial.print(t, 3);
            Serial.print("\t");
            Serial.print(true_A, 3);
            Serial.print("\t");
            Serial.print(kf.get_A(), 3);
            Serial.print("\t");
            Serial.print(true_B, 3);
            Serial.print("\t");
            Serial.print(kf.get_B(), 3);
            Serial.print("\t");
            Serial.print(true_phi, 3);
            Serial.print("\t");
            Serial.print(kf.get_phase(), 3);
            Serial.print("\t");
            Serial.println(kf.get_abs_theta(), 3);
        }
    }
    
    // Final results
    Serial.println("\nFinal Estimates:");
    Serial.print("A: "); Serial.println(kf.get_A(), 4);
    Serial.print("B: "); Serial.println(kf.get_B(), 4);
    Serial.print("Phase: "); Serial.println(kf.get_phase(), 4);
    Serial.print("Theta: "); Serial.println(kf.get_abs_theta(), 4);
    Serial.print("b_x: "); Serial.println(kf.get_bias_x(), 4);
    Serial.print("b_y: "); Serial.println(kf.get_bias_y(), 4);
}

void test_loop() {
    // Empty - everything runs in setup()
}
