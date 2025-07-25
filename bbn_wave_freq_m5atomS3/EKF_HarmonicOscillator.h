#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

template<int M>
class EKF_HarmonicOscillator {
public:
    static constexpr int N_STATE = 2 * M + 2;
    using Vec = Eigen::Matrix<float, N_STATE, 1>;
    using Mat = Eigen::Matrix<float, N_STATE, N_STATE>;
    using Row = Eigen::Matrix<float, 1, N_STATE>;

    EKF_HarmonicOscillator(float dt)
        : T(dt)
    {
        x.setZero();
        x(2 * M) = 1.0f; // Initial ω estimate
        P.setIdentity(); P *= 0.1f;
        Q.setIdentity(); Q *= 1e-4f;
        Q(2 * M, 2 * M) = 1e-5f;
        Q(2 * M + 1, 2 * M + 1) = 1e-4f;
        R(0, 0) = 0.01f;

        H.setZero();
        for (int k = 0; k < M; ++k)
            H(0, 2 * k) = 1.0f; // cosine terms only
        H(0, N_STATE - 1) = 1.0f; // bias
    }

    void setProcessNoise(float q_osc, float q_omega, float q_bias) {
        Q.setIdentity(); Q *= q_osc;
        Q(2 * M, 2 * M) = q_omega;
        Q(2 * M + 1, 2 * M + 1) = q_bias;
    }

    void setMeasurementNoise(float r) {
        R(0, 0) = r;
    }

    // EKF update given measured acceleration
    void update(float y_meas) {
        float omega = x(2 * M);

        // Predict step
        Vec x_pred = Vec::Zero();
        Mat F = Mat::Identity();

        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            float theta = k * omega * T;
            float c = cosf(theta), s = sinf(theta);
            Eigen::Matrix2f Rk;
            Rk << c, -s,
                  s,  c;
            x_pred.segment<2>(i) = Rk * x.segment<2>(i);
            F.block<2,2>(i, i) = Rk;

            // ∂R/∂ω ⋅ x
            float dtheta = k * T;
            float x1 = x(i), x2 = x(i+1);
            Eigen::Vector2f dR_omega;
            dR_omega << -dtheta * (x1 * s + x2 * c),
                         dtheta * (x1 * c - x2 * s);
            F.block<2,1>(i, 2*M) = dR_omega;
        }

        x_pred(2*M) = x(2*M);       // ω
        x_pred(2*M+1) = x(2*M+1);   // b
        P = F * P * F.transpose() + Q;

        // Update step
        float y_pred = (H * x_pred)(0);
        float y_err = y_meas - y_pred;

        float S = (H * P * H.transpose())(0,0) + R(0,0);
        Eigen::Matrix<float, N_STATE, 1> K = P * H.transpose() * (1.0f / S);

        x = x_pred + K * y_err;
        P = (Mat::Identity() - K * H) * P;
    }

    float estimatedAccel() const {
        return (H * x)(0);
    }

    float estimatedHeave() const {
        float heave = 0.0f;
        float omega = x(2 * M);
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            float denom = k * omega;
            heave -= x(i) / (denom * denom);
        }
        return heave;
    }

    float getFrequency() const { return x(2 * M); }
    float getBias() const { return x(2 * M + 1); }

private:
    const float T;

    Vec x;
    Mat P, Q;
    Eigen::Matrix<float, 1, 1> R;
    Row H;
};
