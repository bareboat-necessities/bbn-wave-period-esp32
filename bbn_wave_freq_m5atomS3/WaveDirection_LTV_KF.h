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

class KalmanFilterIQ {
public:
    // Constructor
    KalmanFilterIQ();

    // Initialize the filter
    void init(
        const Eigen::Matrix<float, 6, 6>& Q,  // Process noise
        const Eigen::Matrix<float, 2, 2>& R,  // Measurement noise
        const Eigen::Matrix<float, 6, 6>& P0  // Initial covariance
    );

    // Update the filter with new measurements
    void update(
        float t,        // Current time
        float omega,    // Oscillation frequency (rad/s)
        float x_meas,   // Measured x value
        float y_meas    // Measured y value
    );

    // Get the current state estimate
    const Eigen::Vector<float, 6>& getState() const { return x_hat; }

    // Get the current covariance
    const Eigen::Matrix<float, 6, 6>& getCovariance() const { return P; }

private:
    // State: [I_x, I_y, Q_x, Q_y, b_x, b_y]
    Eigen::Vector<float, 6> x_hat;

    // Covariance matrix
    Eigen::Matrix<float, 6, 6> P;

    // Process noise covariance
    Eigen::Matrix<float, 6, 6> Q;

    // Measurement noise covariance
    Eigen::Matrix<float, 2, 2> R;

    // Project state to enforce I_y Q_x = I_x Q_y
    void projectState();

    // Project covariance to maintain consistency
    void projectCovariance();
};

KalmanFilterIQ::KalmanFilterIQ() {
    // Default initialization
    x_hat.setZero();
    P.setIdentity();
    Q.setIdentity();
    R.setIdentity();
}

void KalmanFilterIQ::init(
    const Eigen::Matrix<float, 6, 6>& Q_init,
    const Eigen::Matrix<float, 2, 2>& R_init,
    const Eigen::Matrix<float, 6, 6>& P0
) {
    Q = Q_init;
    R = R_init;
    P = P0;
    x_hat.setZero();
}

void KalmanFilterIQ::update(float t, float omega, float x_meas, float y_meas) {
    // Measurement matrix H(t)
    float cos_wt = cos(omega * t);
    float sin_wt = sin(omega * t);

    Eigen::Matrix<float, 2, 6> H;
    H << cos_wt, 0, -sin_wt, 0, 1, 0,
         0, cos_wt, 0, -sin_wt, 0, 1;

    // Measurement vector
    Eigen::Vector<float, 2> z;
    z << x_meas, y_meas;

    // ===== Kalman Filter Update =====
    // Innovation: z - H * x_hat
    Eigen::Vector<float, 2> y = z - H * x_hat;

    // Innovation covariance: S = H * P * H^T + R
    Eigen::Matrix<float, 2, 2> S = H * P * H.transpose() + R;

    // Kalman gain: K = P * H^T * S^-1
    Eigen::Matrix<float, 6, 2> K = P * H.transpose() * S.inverse();

    // State update: x_hat += K * y
    x_hat += K * y;

    // Covariance update: P = (I - K * H) * P
    P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;

    // ===== Projection Step =====
    projectState();
    projectCovariance();
}

void KalmanFilterIQ::projectState() {
    float I_x = x_hat(0), I_y = x_hat(1);
    float Q_x = x_hat(2), Q_y = x_hat(3);

    // Constraint residual: r = I_y Q_x - I_x Q_y
    float r = I_y * Q_x - I_x * Q_y;

    // Gradient of constraint: G = [ -Q_y, Q_x, I_y, -I_x, 0, 0 ]
    Eigen::Matrix<float, 1, 6> G;
    G << -Q_y, Q_x, I_y, -I_x, 0, 0;

    // Project state: x_hat -= (G^T * r) / (G * G^T)
    if (G.squaredNorm() > 1e-6f) {
        x_hat -= (G.transpose() * r) / G.squaredNorm();
    }
}

void KalmanFilterIQ::projectCovariance() {
    float I_x = x_hat(0), I_y = x_hat(1);
    float Q_x = x_hat(2), Q_y = x_hat(3);

    // Gradient of constraint: G = [ -Q_y, Q_x, I_y, -I_x, 0, 0 ]
    Eigen::Matrix<float, 1, 6> G;
    G << -Q_y, Q_x, I_y, -I_x, 0, 0;

    // Projection matrix: (I - G^T (G G^T)^-1 G)
    Eigen::Matrix<float, 6, 6> K_c = P * G.transpose() * (G * P * G.transpose()).inverse();
    Eigen::Matrix<float, 6, 6> I = Eigen::Matrix<float, 6, 6>::Identity();
    P = (I - K_c * G) * P * (I - K_c * G).transpose();
}
