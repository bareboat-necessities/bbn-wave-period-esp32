#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  Extended Kalman filter for wave direction estimation.

  Wave direction is given by the plane in which we observe oscillations of horizontal acceleration.
  In case of trochoidal wave model those oscillations are harmonic.

  This model assumes x, y axis acceleration measurements have constant biases and Gaussian noise.
  True x, y accelerations without bias and noise are harmonic and have same phase. Phase is unknown and estimated by the filter.
  Frequency is considered known and is a parameter on each step.

  See details in:  https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/30#issuecomment-2931856187

*/

#include <ArduinoEigenDense.h>  // Eigen for matrix operations

// Define matrix types for 5 states and 2 measurements
using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector2f = Eigen::Matrix<float, 2, 1>;
using Matrix5f = Eigen::Matrix<float, 5, 5>;
using Matrix2f = Eigen::Matrix<float, 2, 2>;
using Matrix25f = Eigen::Matrix<float, 2, 5>;
using Matrix52f = Eigen::Matrix<float, 5, 2>;

class WaveDirectionEKF {
public:
    // Constructor
    WaveDirectionEKF(float omega, const Vector5f& initial_state, 
                     const Matrix5f& initial_covariance, 
                     const Matrix5f& process_noise, 
                     const Matrix2f& measurement_noise)
        : omega_(omega), state_(initial_state), P_(initial_covariance),
          Q_(process_noise), R_(measurement_noise) {}

    // Prediction step
    void predict() {
        P_ += Q_;  // Simplified: F = identity matrix
    }

    // Update step
    void update(float t, float omega, float x_meas, float y_meas) {
        omega_ = omega;

        // Extract states
        float A = state_(0), B = state_(1), phi = state_(2), 
              bx = state_(3), by = state_(4);
        
        // Compute argument and trig functions
        float arg = omega_ * t + phi;
        float sin_arg = sin(arg);
        float cos_arg = cos(arg);
        
        // Predicted measurements
        Vector2f z_pred;
        z_pred << A * sin_arg + bx, 
                  B * sin_arg + by;
        
        // Measurement residual
        Vector2f z;
        z << x_meas, y_meas;
        Vector2f residual = z - z_pred;
        
        // Jacobian matrix H (2x5)
        Matrix25f H;
        H << sin_arg, 0, A * cos_arg, 1, 0,
             0, sin_arg, B * cos_arg, 0, 1;
        
        // Kalman gain calculation
        Matrix2f S = H * P_ * H.transpose() + R_;
        Matrix52f K = P_ * H.transpose() * S.inverse();
        
        // State and covariance update
        state_ += K * residual;
        P_ -= K * H * P_;
        
        // Phase wrapping to [-π, π]
        state_(2) = fmod(state_(2) + M_PI, 2 * M_PI) - M_PI;
    }

    // Get estimated state
    Vector5f getState() const { return state_; }

    // Compute θ = atan2(A, B)
    float getTheta() const {
        return atan2(state_(0), state_(1));
    }

private:
    float omega_;    // Known angular frequency
    Vector5f state_; // [A, B, φ, b_x, b_y]
    Matrix5f P_;     // Covariance matrix
    Matrix5f Q_;     // Process noise covariance
    Matrix2f R_;     // Measurement noise covariance
};

/*
  
// Example usage
void setup() {
    Serial.begin(9600);
    delay(1000);  // Wait for serial monitor

    // Known angular frequency
    const float omega = 2 * M_PI * 0.3f; // 0.3 Hz

    // Initial state: [A, B, φ, b_x, b_y]
    Vector5f initial_state;
    initial_state << 1.0f, 1.0f, 0.0f, 0.0f, 0.0f;

    // Initial covariance (high uncertainty)
    Matrix5f initial_covariance = Matrix5f::Identity() * 100.0f;
    initial_covariance(2, 2) = 4 * M_PI * M_PI;  // Large phase uncertainty

    // Process noise covariance (small values)
    Matrix5f Q = Matrix5f::Identity() * 1e-6f;

    // Measurement noise covariance
    Matrix2f R;
    R << 0.01f, 0.0f,   // σ_x^2 = 0.01 (std dev 0.1)
         0.0f,  0.01f;  // σ_y^2 = 0.01

    // Initialize EKF
    WaveDirectionEKF ekf(omega, initial_state, initial_covariance, Q, R);

    // Simulate measurements
    const float true_A = 1.0f, true_B = 1.5f, true_phi = 0.5f;
    const float true_bx = 0.1f, true_by = -0.2f;
    const int num_steps = 5000;
    const float dt = 0.004f;  // Time step (4ms)

    for (int i = 0; i < num_steps; ++i) {
        float t = i * dt;
        
        // Generate true signals
        float arg = omega * t + true_phi;
        float x_true = true_A * sin(arg) + true_bx;
        float y_true = true_B * sin(arg) + true_by;
        
        // Add noise (simulated measurements)
        float x_meas = x_true + 0.1f * (rand() % 100 - 50) / 50.0f;
        float y_meas = y_true + 0.1f * (rand() % 100 - 50) / 50.0f;
        
        // EKF steps
        ekf.predict();
        ekf.update(t, omega, x_meas, y_meas);
        
        // Periodically log results
        if (i % 100 == 0) {
            Vector5f state = ekf.getState();
            Serial.print("t: "); Serial.print(t, 3);
            Serial.print(" | A: "); Serial.print(state(0), 3);
            Serial.print(" | B: "); Serial.print(state(1), 3);
            Serial.print(" | φ: "); Serial.print(state(2), 3);
            Serial.print(" | θ: "); Serial.print(ekf.getTheta(), 3);
            Serial.print(" | b_x: "); Serial.print(state(3), 3);
            Serial.print(" | b_y: "); Serial.println(state(4), 3);
        }
    }
}

void loop() {}  // Empty loop

*/
