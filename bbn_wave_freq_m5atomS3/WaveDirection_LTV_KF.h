#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter for wave direction estimation.

  Wave direction is given by the plane in which we observe oscillations of horizontal acceleration.
  In case of trochoidal wave model those oscillations are harmonic.

  This model assumes x, y axis acceleration measurements have constant biases and Gaussian noise.
  True x, y accelerations without bias and noise are harmonic and have same phase (or move in counter phase). 
  Frequency is considered known and is a parameter on each step.

  See details in: https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/31

*/

#include <ArduinoEigenDense.h>  // Eigen for matrix operations

// Define matrix types for 6 states and 2 measurements
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector2f = Eigen::Matrix<float, 2, 1>;
using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Matrix2f = Eigen::Matrix<float, 2, 2>;
using Matrix26f = Eigen::Matrix<float, 2, 6>;
using Matrix62f = Eigen::Matrix<float, 6, 2>;
using Matrix16f = Eigen::Matrix<float, 1, 6>;

class WaveDirection_LTV_KF {
public:
    // Constructor
    WaveDirection_LTV_KF() {
        // Default initialization
        x_hat.setZero();
        P.setIdentity();
        Q.setIdentity();
        R.setIdentity();
    };

    // Initialize the filter
    void init(
        const Matrix6f& Q,  // Process noise
        const Matrix2f& R,  // Measurement noise
        const Matrix6f& P0  // Initial covariance
    ) {
        Q = Q_init;
        R = R_init;
        P = P0;
        x_hat.setZero();
    }

    // Update the filter with new measurements
    void update(
        float t,        // Current time
        float omega,    // Oscillation frequency (rad/s)
        float x_meas,   // Measured x value
        float y_meas    // Measured y value
    ) {
        // Measurement matrix H(t)
        float cos_wt = cos(omega * t);
        float sin_wt = sin(omega * t);
    
        Matrix26f H;
        H << cos_wt, 0, -sin_wt, 0, 1, 0,
             0, cos_wt, 0, -sin_wt, 0, 1;
    
        // Measurement vector
        Vector2f z;
        z << x_meas, y_meas;
    
        // ===== Kalman Filter Update =====
        // Innovation: z - H * x_hat
        Vector2f y = z - H * x_hat;
    
        // Innovation covariance: S = H * P * H^T + R
        Matrix2f S = H * P * H.transpose() + R;
    
        // Kalman gain: K = P * H^T * S^-1
        Matrix62f K = P * H.transpose() * S.inverse();
    
        // State update: x_hat += K * y
        x_hat += K * y;
    
        // Covariance update: P = (I - K * H) * P
        P = (Matrix6f::Identity() - K * H) * P;
    
        // ===== Projection Step =====
        projectState();
        projectCovariance();
    }

    // Get the current state estimate
    const Vector6f& getState() const { return x_hat; }

    // Get the current covariance
    const Matrix6f& getCovariance() const { return P; }

    const float getTheta() const { return atan2(x_hat(1), x_hat(0));  // atan2(I_y, I_x) }

private:
    // State: [I_x, I_y, Q_x, Q_y, b_x, b_y]
    Vector6f x_hat;

    // Covariance matrix
    Matrix6f P;

    // Process noise covariance
    Matrix6f Q;

    // Measurement noise covariance
    Matrix2f R;

    // Project state to enforce I_y Q_x = I_x Q_y
    void projectState() {
        float I_x = x_hat(0), I_y = x_hat(1);
        float Q_x = x_hat(2), Q_y = x_hat(3);
    
        // Constraint residual: r = I_y Q_x - I_x Q_y
        float r = I_y * Q_x - I_x * Q_y;
    
        // Gradient of constraint: G = [ -Q_y, Q_x, I_y, -I_x, 0, 0 ]
        Matrix16f G;
        G << -Q_y, Q_x, I_y, -I_x, 0, 0;
    
        // Project state: x_hat -= (G^T * r) / (G * G^T)
        if (G.squaredNorm() > 1e-6f) {
            x_hat -= (G.transpose() * r) / G.squaredNorm();
        }
    }

    // Project covariance to maintain consistency
    void projectCovariance() {
        float I_x = x_hat(0), I_y = x_hat(1);
        float Q_x = x_hat(2), Q_y = x_hat(3);
    
        // Gradient of constraint: G = [ -Q_y, Q_x, I_y, -I_x, 0, 0 ]
        Matrix16f G;
        G << -Q_y, Q_x, I_y, -I_x, 0, 0;
    
        // Projection matrix: (I - G^T (G G^T)^-1 G)
        Matrix6f K_c = P * G.transpose() * (G * P * G.transpose()).inverse();
        Matrix6f I = Matrix6f::Identity();
        P = (I - K_c * G) * P * (I - K_c * G).transpose();
    }
};

