#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <cmath>

template<int M, typename Real = float>
class UKF_HarmonicOscillator {
public:
    static constexpr int N_STATE = 2 * M + 3; // a_k, b_k, omega, bias, phase
    static constexpr int SIG_CNT = 2 * N_STATE + 1;
    using SigmaMat = Eigen::Matrix<Real, N_STATE, SIG_CNT>;
    using Vec = Eigen::Matrix<Real, N_STATE, 1>;
    using Mat = Eigen::Matrix<Real, N_STATE, N_STATE>;

    // UKF tuning parameters
    static constexpr Real alpha = 0.9;
    static constexpr Real beta = 2.0;
    static constexpr Real kappa = 2.0;
    static constexpr Real lambda = alpha * alpha * (N_STATE + kappa) - N_STATE;

    UKF_HarmonicOscillator() {
        x.setZero();
        for (int k = 0; k < M; ++k)
            x(2 * k) = Real(0.2 - 0.03 * k); // a_k
        x(2 * M) = Real(2 * M_PI * 0.3); // omega
        x(2 * M + 1) = Real(0);          // bias
        x(2 * M + 2) = Real(0);          // phase

        P.setIdentity(); P *= Real(1.0);
        P(2 * M, 2 * M) = Real(5e-1);         // omega 
        P(2 * M + 1, 2 * M + 1) = Real(1e-2); // bias
        P(2 * M + 2, 2 * M + 2) = Real(1e-1); // phase
        
        Q.setIdentity(); Q *= Real(1e-5);
        Q(2 * M, 2 * M) = Real(1e-4);         // omega process noise
        Q(2 * M + 1, 2 * M + 1) = Real(1e-6); // bias
        Q(2 * M + 2, 2 * M + 2) = Real(1e-6); // phase

        R.setZero(); R(0, 0) = Real(0.1);

        calculateWeights();
    }

    void setProcessNoise(Real q_osc, Real q_omega, Real q_bias, Real q_phase) {
        Q.setIdentity(); Q *= q_osc;
        Q(2 * M, 2 * M) = q_omega;
        Q(2 * M + 1, 2 * M + 1) = q_bias;
        Q(2 * M + 2, 2 * M + 2) = q_phase;
    }

    void setMeasurementNoise(Real r) {
        R.setZero(); R(0, 0) = r;
    }

    void update(Real y_meas, Real dt) {
        SigmaMat sigma_points = generateSigmaPoints();
        SigmaMat sigma_pred = predictSigmaPoints(sigma_points, dt);
        predictMeanAndCovariance(sigma_pred);
        updateWithMeasurement(sigma_pred, y_meas);

        // Clamp omega and wrap phase
        x(2 * M) = std::clamp(x(2 * M), Real(2 * M_PI * 0.0001), Real(2 * M_PI * 100.0));
        x(2 * M + 2) = wrapPhase(x(2 * M + 2));
    }

    Real estimatedAccel() const {
        Real y = 0;
        Real phase = x(2 * M + 2);
        Real omega = std::max(x(2 * M), Real(1e-4));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real theta = k * phase;
            Real a = x(i), b = x(i + 1);
            Real factor = -(k * omega) * (k * omega);
            y += factor * (a * std::cos(theta) + b * std::sin(theta));
        }
        return y + x(2 * M + 1); // add bias
    }

    Real estimatedVelocity() const {
        Real velocity = Real(0);
        Real omega = std::max(x(2 * M), Real(1e-4));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real harmonic_freq = k * omega;
            harmonic_freq = std::max(harmonic_freq, Real(1e-7));
            velocity -= x(i + 1) / harmonic_freq; // -b_k / (k*omega)
        }
        return velocity;
    }

    Real estimatedHeave() const {
        Real heave = Real(0);
        Real phase = x(2 * M + 2);
        Real omega = std::max(x(2 * M), Real(1e-4));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real denom = k * omega;
            denom = denom * denom;
            denom = std::max(denom, Real(1e-6));
            Real theta = k * phase;
            heave += -x(i)     / denom * std::cos(theta); // -a_k / (kω)^2 * cos(kφ)
            heave += -x(i + 1) / denom * std::sin(theta); // -b_k / (kω)^2 * sin(kφ)
        }
        return heave;
    }

    Real estimatedPhase() const { return x(2 * M + 2); }
    Real getFrequency()   const { return x(2 * M) / (2 * M_PI); }
    Real getBias()        const { return x(2 * M + 1); }

private:
    Vec x;
    Mat P, Q, R;
    Eigen::Matrix<Real, 1, SIG_CNT> weights_m, weights_c;

    void calculateWeights() {
        const Real denom = lambda + N_STATE;
        if (denom <= Real(0)) {
            // fallback safeguard: choose alpha/kappa so denom > 0
            throw std::runtime_error("Invalid UKF scaling: lambda + N_STATE <= 0");
        }
        weights_m(0) = lambda / denom;
        weights_c(0) = weights_m(0) + (1 - alpha * alpha + beta);
        Real w = Real(1) / (2 * denom);
        for (int i = 1; i < SIG_CNT; ++i)
            weights_m(i) = weights_c(i) = w;
    }

    SigmaMat generateSigmaPoints() {
        SigmaMat sigma(N_STATE, SIG_CNT);
        Real scale = std::sqrt(std::max(lambda + N_STATE, Real(1e-5)));
    
        // Regularize and symmetrize
        Mat P_sym = (P + P.transpose()) * Real(0.5);
        Eigen::LLT<Mat> llt(P_sym);
        if (llt.info() != Eigen::Success) {
            P_sym += Mat::Identity() * Real(1e-6);
            llt.compute(P_sym);
        }
        Mat sqrtP = llt.matrixL();
    
        sigma.col(0) = x;
        for (int i = 0; i < N_STATE; ++i) {
            sigma.col(i + 1)         = x + scale * sqrtP.col(i);
            sigma.col(i + 1 + N_STATE) = x - scale * sqrtP.col(i);
        }
        return sigma;
    }

    SigmaMat predictSigmaPoints(const SigmaMat& sigma, Real dt) {
        SigmaMat sigma_pred(N_STATE, SIG_CNT);
        for (int i = 0; i < SIG_CNT; ++i) {
            Vec xi = sigma.col(i);
            Vec xp = xi;

            Real omega = xi(2 * M); //std::max(xi(2 * M), Real(1e-4));
            for (int k = 1; k <= M; ++k) {
                int j = 2 * (k - 1);
                Real theta = k * omega * dt;
                Real c = std::cos(theta), s = std::sin(theta);
                xp(j)     = c * xi(j) - s * xi(j + 1);
                xp(j + 1) = s * xi(j) + c * xi(j + 1);
            }
            // omega, bias unchanged
            xp(2 * M)     = xi(2 * M);
            xp(2 * M + 1) = xi(2 * M + 1);
            xp(2 * M + 2) = wrapPhase(xi(2 * M + 2) + dt * omega); // phase += omega * dt

            sigma_pred.col(i) = xp;
        }
        return sigma_pred;
    }

    void predictMeanAndCovariance(const SigmaMat& sigma_pred) {
        x.setZero();
        // Circular mean for phase
        Real sin_sum = 0, cos_sum = 0;
        for (int i = 0; i < SIG_CNT; ++i) {
            for (int j = 0; j < N_STATE; ++j) {
                if (j == 2 * M + 2) continue; // skip phase
                x(j) += weights_m(i) * sigma_pred(j, i);
            }
            Real phi = sigma_pred(2 * M + 2, i);
            sin_sum += weights_m(i) * std::sin(phi);
            cos_sum += weights_m(i) * std::cos(phi);
        }
        x(2 * M + 2) = std::atan2(sin_sum, cos_sum); // circular mean for phase
    
        // Covariance
        P.setZero();
        for (int i = 0; i < SIG_CNT; ++i) {
            Vec dx = sigma_pred.col(i) - x;
    
            // Correct wrapped phase residual
            Real dphi = sigma_pred(2 * M + 2, i) - x(2 * M + 2);
            dx(2 * M + 2) = std::atan2(std::sin(dphi), std::cos(dphi));
            P += weights_c(i) * dx * dx.transpose();
        }
        P += Q;
        symmetrize(P);
    }
    
    void updateWithMeasurement(const SigmaMat& sigma_pred, Real y_meas) {
        Eigen::Matrix<Real, 1, SIG_CNT> y_sigma;
        for (int i = 0; i < SIG_CNT; ++i)
            y_sigma(i) = measurementModel(sigma_pred.col(i));
    
        Real y_pred = 0;
        for (int i = 0; i < SIG_CNT; ++i)
            y_pred += weights_m(i) * y_sigma(i);
    
        Real Pyy = R(0, 0);
        Vec Pxy = Vec::Zero();
        for (int i = 0; i < SIG_CNT; ++i) {
            Real dy = y_sigma(i) - y_pred;
            Vec dx = sigma_pred.col(i) - x;
    
            // Correct wrapped phase residual
            Real dphi = sigma_pred(2 * M + 2, i) - x(2 * M + 2);
            dx(2 * M + 2) = std::atan2(std::sin(dphi), std::cos(dphi));
    
            Pyy += weights_c(i) * dy * dy;
            Pxy += weights_c(i) * dx * dy;
        }
        Vec K = Pxy / std::max(Pyy, Real(1e-6));
        x += K * (y_meas - y_pred);
        P -= K * K.transpose() * Pyy;
        symmetrize(P);
    }
    
    Real measurementModel(const Vec& xi) const {
        Real phase = xi(2 * M + 2);
        Real omega = xi(2 * M); //std::max(xi(2 * M), Real(1e-3));
        Real y = 0;
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real a = xi(i), b = xi(i + 1);
            Real theta = k * phase;
            Real factor = -(k * omega) * (k * omega);
            y += factor * (a * std::cos(theta) + b * std::sin(theta));
        }
        y += xi(2 * M + 1); // bias
        return y;
    }

    static Real wrapPhase(Real phase) {
        return std::atan2(std::sin(phase), std::cos(phase));
    }

    void symmetrize(Mat& P) {
        P = (P + P.transpose()) * Real(0.5);
        if (!P.allFinite() || P.trace() <= Real(0))
            P = Mat::Identity() * 1e-3;
    }
};
