#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  Extended Kalman filter for wave direction estimation.

  Wave direction is given by the plane in which we observe oscillations of horizontal acceleration.
  In case of trochoidal wave model those oscillations are harmonic.

  This model assumes x, y axis acceleration measurements have constant biases and Gaussian noise.
  True x, y accelerations without bias and noise are harmonic and have same phase. Phase is unknown and estimated by the filter.
  Frequency is considered known and is a parameter on each step.

  See details in:  https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/32

*/

#include <ArduinoEigenDense.h>  // Eigen for matrix operations

// Type shortcuts
using Matrix5f = Eigen::Matrix<float, 5, 5>;
using Vector5f = Eigen::Matrix<float, 5, 1>;
using Matrix2f = Eigen::Matrix<float, 2, 2>;
using Vector2f = Eigen::Matrix<float, 2, 1>;
using Matrix2x5f = Eigen::Matrix<float, 2, 5>;
using Matrix5x2f = Eigen::Matrix<float, 5, 2>;

class WaveDirectionEKF {
private:
    Vector5f z_hat_;      // State vector: [log(a), b, phi, bias_x, bias_y]
    Matrix5f P_;          // Covariance matrix
    Matrix5f Q_;          // Process noise
    Matrix2f R_;          // Measurement noise

public:
    WaveDirectionEKF(float a_init, float b_init, float phi_init,
                     float bias_x_init, float bias_y_init) {
        // Initialize state
        z_hat_ << logf(a_init), b_init, phi_init, bias_x_init, bias_y_init;
        
        // Initialize covariance
        P_ = Matrix5f::Identity() * 100.0f;  // High uncertainty
        P_(0, 0) = 10.0f;
        P_(2, 2) = 4 * M_PI * M_PI; // Large phase uncertainty
        P_(3, 3) = 1.0f; // Lower uncertainty for biases
        P_(4, 4) = 1.0f; // Lower uncertainty for biases
        
        // Configure process noise
        Q_ = Matrix5f::Identity() * 1e-6f;
        Q_(2, 2) = 1e-4f * M_PI * M_PI;  // Higher noise for phase
        
        // Configure measurement noise
        R_ = Matrix2f::Identity() * 0.09f;
    }

    void predict(float t, float omega, float dt) {
        // State transition matrix (identity for constant states)
        Matrix5f F = Matrix5f::Identity();

        if (t < 0.0) {
            // Predict state (phase integrates omega)
            z_hat_(2) += omega * dt;
        } else {
            // Use spot frequency
            const float phi = z_hat_(2);
            z_hat_(2) = omega * t + phi;
        }
        
        // Predict covariance
        P_ = F * P_ * F.transpose() + Q_;
    }

    // Update step now takes omega as parameter
    void update(float t, float omega, float x_measured, float y_measured) {
        // Extract current state
        const float log_a = z_hat_(0);
        const float b = z_hat_(1);
        const float phi = z_hat_(2);
        const float bias_x = z_hat_(3);
        const float bias_y = z_hat_(4);
        
        // Compute trigonometric terms
        const float theta = omega * t + phi;
        const float sin_theta = sinf(theta);
        const float cos_theta = cosf(theta);
        
        // Compute predicted measurements
        const float a = expf(log_a);
        Vector2f y_pred;
        y_pred << a * sin_theta + bias_x,
                  b * sin_theta + bias_y;
        
        // Measurement Jacobian
        Matrix2x5f H;
        H << a * sin_theta, 0, a * cos_theta, 1, 0,
             0, sin_theta, b * cos_theta, 0, 1;
        
        // Actual measurements
        Vector2f y_meas;
        y_meas << x_measured, y_measured;
        
        // Innovation (measurement residual)
        Vector2f error = y_meas - y_pred;
        
        // Kalman gain
        Matrix2f S = H * P_ * H.transpose() + R_;
        Matrix5x2f K = P_ * H.transpose() * S.inverse();
        
        // Update state and covariance
        z_hat_ += K * error;
        P_ = (Matrix5f::Identity() - K * H) * P_;
        
        // Normalize phase to [-π, π]
        float& phi_adj = z_hat_(2);
        phi_adj = std::fmod(phi_adj, 2 * M_PI);
        if (phi_adj < -M_PI) {
            phi_adj += 2 * M_PI;
        } else if (phi_adj > M_PI) {
            phi_adj -= 2 * M_PI;
        }
    }

    float getA() const { return expf(z_hat_(0)); }
    float getB() const { return z_hat_(1); }
    float getPhase() const { return z_hat_(2); }
    float getBiasX() const { return z_hat_(3); }
    float getBiasY() const { return z_hat_(4); }
    float getAtanAB() const { return atan2(getA(), getB()); }  // Compute θ = atan2(A, B)

    float getAmplitude() const { 
        float A = getA(), B = getB();
        return sqrtf(A * A + B * B);
    }

    // Configuration methods
    void setProcessNoise(float log_a_noise, float b_noise, float phi_noise,
                        float bias_x_noise, float bias_y_noise) {
        Q_.diagonal() << log_a_noise, b_noise, phi_noise,
                        bias_x_noise, bias_y_noise;
    }
    
    void setMeasurementNoise(float x_noise, float y_noise) {
        R_.diagonal() << x_noise, y_noise;
    }
};

// Example usage
WaveDirectionEKF ekf(1.0f, 1.0f, 0.0f, 0.0f, 0.0f);

void test_WaveDirectionEKF_setup() {
    Serial.begin(115200);
    // Optional noise tuning
    ekf.setProcessNoise(1e-6f, 1e-6f, 1e-4f, 1e-6f, 1e-6f);
    ekf.setMeasurementNoise(1e-2f, 1e-2f);
}

void test_WaveDirectionEKF_loop() {
    static float t = 0.0f;
    const float omega = 2*M_PI*1.0f;  // 1 Hz signal
    const float dt = 0.01f;           // 10 ms sampling
    
    // Simulated measurements
    float x = 1.5f * sinf(omega*t + 0.1f) + 0.2f + 0.01f * random(-100, 100)/100.0f;
    float y = 0.8f * sinf(omega*t + 0.1f) - 0.3f + 0.01f * random(-100, 100)/100.0f;
    
    // EKF steps with parameters passed in
    ekf.predict(-1.0, omega, dt);
    ekf.update(t, omega, x, y);
    
    // Output results
    Serial.print("atanAB: ");
    Serial.print(ekf.getAtanAB());
    Serial.print(" | A: ");
    Serial.print(ekf.getA());
    Serial.print(" | B: ");
    Serial.println(ekf.getB());
    
    t += dt;
    delay(dt * 1000);
}
