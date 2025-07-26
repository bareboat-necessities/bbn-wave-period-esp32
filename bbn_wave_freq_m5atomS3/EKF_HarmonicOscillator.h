#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <cmath>

// State vector layout:
// x = [ a1_cos, a1_sin, a2_cos, a2_sin, ..., aM_cos, aM_sin, ω, bias ]
template<int M, typename Real = float>
class EKF_HarmonicOscillator {
public:
    static constexpr int N_STATE = 2 * M + 2;
    using Vec = Eigen::Matrix<Real, N_STATE, 1>;
    using Mat = Eigen::Matrix<Real, N_STATE, N_STATE>;
    using Row = Eigen::Matrix<Real, 1, N_STATE>;

    EKF_HarmonicOscillator()
    {
        x.setZero();
        for (int k = 0; k < M; ++k) {
            x(2 * k) = Real(0.01); 
            x(2 * k + 1) = Real(0); 
        }
        x(0) = Real(0.01); 
        x(1) = Real(0.01); 
        x(2 * M) = Real(2 * M_PI * 0.3 /*FREQ_GUESS*/);       // Initial ω estimate
        x(2 * M + 1) = Real(0);   // Initial bias estimate
        P.setIdentity(); P *= Real(10.0);
        Q.setIdentity(); Q *= Real(1e-3);
        Q(2 * M, 2 * M) = Real(1e-5);
        Q(2 * M + 1, 2 * M + 1) = Real(1e-4);
        R.setZero();
        R(0, 0) = Real(0.01);

        H.setZero();
        for (int k = 0; k < M; ++k) {
            H(0, 2 * k) = Real(1); // cosine terms only
        }
        H(0, N_STATE - 1) = Real(1); // bias
    }

    void setProcessNoise(Real q_osc, Real q_omega, Real q_bias) {
        Q.setIdentity(); Q *= q_osc;
        Q(2 * M, 2 * M) = q_omega;
        Q(2 * M + 1, 2 * M + 1) = q_bias;
    }

    void setMeasurementNoise(Real r) {
        R.setZero();
        R(0, 0) = r;
    }

    // EKF update given measured acceleration
    void update(Real y_meas, Real dt) {
        Real omega = std::max(x(2 * M), Real(1e-4));

        // Predict step
        Vec x_pred = Vec::Zero();
        Mat F = Mat::Identity();

        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real theta = k * omega * dt;
            Real c = std::cos(theta), s = std::sin(theta);
            Eigen::Matrix<Real, 2, 2> Rk;
            Rk << c, -s,
                  s,  c;
            x_pred.segment(i, 2) = Rk * x.segment(i, 2);
            F.block(i, i, 2, 2) = Rk;

            // ∂R/∂ω ⋅ x
            Real dtheta = k * dt;
            Real x1 = x(i), x2 = x(i+1);
            Eigen::Matrix<Real, 2, 1> dR_omega;
            dR_omega << -dtheta * (x1 * s + x2 * c),
                         dtheta * (x1 * c - x2 * s);
            F.block(i, 2 * M, 2, 1) = dR_omega;
        }

        x_pred(2 * M) = x(2 * M);           // ω
        x_pred(2 * M + 1) = x(2 * M + 1);   // b
        P = F * P * F.transpose() + Q;

        // Update step
        Real y_pred = (H * x_pred)(0);
        Real y_err = y_meas - y_pred;

        Real S = (H * P * H.transpose())(0, 0) + R(0, 0);
        Eigen::Matrix<Real, N_STATE, 1> K = P * H.transpose() * (Real(1) / S);

        x = x_pred + K * y_err;
        P = (Mat::Identity() - K * H) * P;
    }

    Real estimatedAccel() const {
        return (H * x)(0);
    }

    Real estimatedHeave() const {
        Real heave = Real(0);
        Real omega = std::max(x(2 * M), Real(1e-4));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real denom = k * omega;
            if (std::abs(denom) < Real(1e-7)) denom = Real(1e-7);
            heave -= x(i) / (denom * denom);
        }
        return heave;
    }

    Real estimatedVelocity() const {
        Real vel = Real(0);
        Real omega = std::max(x(2 * M), Real(1e-4));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real denom = k * omega;
            if (std::abs(denom) < Real(1e-7)) denom = Real(1e-7);
            vel -= x(i + 1) / denom;
        }
        return vel;
    }

    Real estimatedPhase() const {
        Real a1_cos = x(0);
        Real a1_sin = x(1);
        return std::atan2(a1_sin, a1_cos);
    }

    Real getFrequency() const { return x(2 * M) / (2 * M_PI); }
    Real getBias() const { return x(2 * M + 1); }

private:

    Vec x;
    Mat P, Q;
    Eigen::Matrix<Real, 1, 1> R;
    Row H;
};
