#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <cmath>

template<int M, typename Real = float>
class EKF_HarmonicOscillator {
public:
    static constexpr int N_STATE = 2 * M + 2;
    static constexpr int SIG_CNT = 2 * N_STATE + 1;
    using SigmaMat = Eigen::Matrix<Real, N_STATE, SIG_CNT>;
    using Vec = Eigen::Matrix<Real, N_STATE, 1>;
    using Mat = Eigen::Matrix<Real, N_STATE, N_STATE>;
    using Row = Eigen::Matrix<Real, 1, N_STATE>;

    // UKF parameters
    static constexpr Real alpha = 0.5;
    static constexpr Real beta = 2.0;
    static constexpr Real kappa = 3.0 - N_STATE;
    static constexpr Real lambda = alpha * alpha * (N_STATE + kappa) - N_STATE;
    
    EKF_HarmonicOscillator() {
        // Initialize state
        x.setZero();
        for (int k = 0; k < M; ++k) {
            x(2 * k) = Real(0.01); 
            x(2 * k + 1) = Real(0.001);
        }
        x(2 * M) = Real(2 * M_PI * 0.3);  // Initial ω estimate (0.3 Hz)
        x(2 * M + 1) = Real(0);           // Initial bias estimate
        
        // Initialize covariance
        P.setIdentity(); 
        P *= Real(10.0);
        
        // Process noise
        Q.setIdentity(); 
        Q *= Real(1e-3);
        Q(2 * M, 2 * M) = Real(1e0);      // Frequency process noise
        Q(2 * M + 1, 2 * M + 1) = Real(1e-5); // Bias process noise
        
        // Measurement noise
        R.setZero();
        R(0, 0) = Real(0.5);
        
        // Calculate weights
        calculateWeights();
    }

    void setProcessNoise(Real q_osc, Real q_omega, Real q_bias) {
        Q.setIdentity(); 
        Q *= q_osc;
        Q(2 * M, 2 * M) = q_omega;
        Q(2 * M + 1, 2 * M + 1) = q_bias;
    }

    void setMeasurementNoise(Real r) {
        R.setZero();
        R(0, 0) = r;
    }

    void update(Real y_meas, Real dt) {
        // 1. Generate sigma points
        SigmaMat sigma_points = generateSigmaPoints();
        
        // 2. Predict step (time update)
        SigmaMat sigma_points_pred = predictSigmaPoints(sigma_points, dt);
        predictMeanAndCovariance(sigma_points_pred);
        
        // 3. Measurement update
        updateWithMeasurement(sigma_points_pred, y_meas);
        
        // Ensure frequency stays within reasonable bounds
        x(2 * M) = std::max(x(2 * M), Real(2 * M_PI * 0.02));  // min 0.02 Hz
        x(2 * M) = std::min(x(2 * M), Real(2 * M_PI * 10.0));  // max 10 Hz
    }

    // Measurement prediction (for innovation calculation)
    Real estimatedAccel() const {
        Real y = 0;
        for (int k = 0; k < M; ++k) {
            y += x(2 * k);  // Sum of cosine terms
        }
        y += x(2 * M + 1);  // Add bias
        return y;
    }

    Real estimatedHeave() const {
        Real heave = Real(0);
        Real omega = x(2 * M);
        omega = std::clamp(omega, Real(2 * M_PI * 0.02), Real(2 * M_PI * 10.0));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real denom = k * omega;
            denom = std::abs(denom) < Real(1e-7) ? Real(1e-7) : denom;
            heave -= x(i) / (denom * denom);
        }
        return heave;
    }

    Real estimatedVelocity() const {
        Real vel = Real(0);
        Real omega = x(2 * M);
        omega = std::clamp(omega, Real(2 * M_PI * 0.02), Real(2 * M_PI * 10.0));
        for (int k = 1; k <= M; ++k) {
            int i = 2 * (k - 1);
            Real denom = k * omega;
            denom = std::abs(denom) < Real(1e-7) ? Real(1e-7) : denom;
            vel -= x(i + 1) / denom;
        }
        return vel;
    }

    Real estimatedPhase() const {
        return std::atan2(x(1), x(0));
    }

    Real getFrequency() const { return x(2 * M) / (2 * M_PI); }
    Real getBias() const { return x(2 * M + 1); }

private:
    Vec x;      // State vector
    Mat P;      // State covariance
    Mat Q;      // Process noise covariance
    Mat R;      // Measurement noise covariance
    
    // Weights for sigma points
    Eigen::Matrix<Real, 1, SIG_CNT> weights_m;  // Mean weights
    Eigen::Matrix<Real, 1, SIG_CNT> weights_c;  // Covariance weights

    void calculateWeights() {
        weights_m(0) = lambda / (N_STATE + lambda);
        weights_c(0) = weights_m(0) + (1 - alpha * alpha + beta);
        
        Real w = Real(1) / (2 * (N_STATE + lambda));
        for (int i = 1; i < SIG_CNT; ++i) {
            weights_m(i) = w;
            weights_c(i) = w;
        }
    }

    SigmaMat generateSigmaPoints() {
        SigmaMat sigma_points(N_STATE, SIG_CNT);
        const Real scale = sqrt(N_STATE + lambda);
        
        // Matrix square root of P
        Eigen::LLT<Mat> lltOfP(P);
        if (lltOfP.info() != Eigen::Success) {
            // Handle fallback (e.g. add jitter, or fallback to identity)
            P += Mat::Identity() * Real(1e-6);
            lltOfP.compute(P);
        }
        Mat sqrtP = lltOfP.matrixL();
        
        sigma_points.col(0) = x;
        for (int i = 0; i < N_STATE; ++i) {
            sigma_points.col(i + 1) = x + scale * sqrtP.col(i);
            sigma_points.col(i + 1 + N_STATE) = x - scale * sqrtP.col(i);
        }
        return sigma_points;
    }

    SigmaMat predictSigmaPoints(const SigmaMat& sigma_points, Real dt) {
        SigmaMat sigma_points_pred(N_STATE, SIG_CNT);
        
        for (int i = 0; i < SIG_CNT; ++i) {
            Vec x_sigma = sigma_points.col(i);
            Vec x_pred = Vec::Zero();
            x_sigma(2 * M) = std::clamp(x_sigma(2 * M), Real(2 * M_PI * 0.02), Real(2 * M_PI * 10.0));
            Real omega = x_sigma(2 * M);
            // Predict harmonic components
            for (int k = 1; k <= M; ++k) {
                int idx = 2 * (k - 1);
                Real theta = k * omega * dt;
                Real c = cos(theta), s = sin(theta);
                x_pred(idx) = c * x_sigma(idx) - s * x_sigma(idx + 1);
                x_pred(idx + 1) = s * x_sigma(idx) + c * x_sigma(idx + 1);
            }
            
            // Frequency and bias are assumed constant (with noise handled by Q)
            x_pred(2 * M) = x_sigma(2 * M);
            x_pred(2 * M + 1) = x_sigma(2 * M + 1);
            
            sigma_points_pred.col(i) = x_pred;
        }
        return sigma_points_pred;
    }

    void predictMeanAndCovariance(const SigmaMat& sigma_points_pred) {
        // Calculate predicted state mean
        x.setZero();
        for (int i = 0; i < SIG_CNT; ++i) {
            x += weights_m(i) * sigma_points_pred.col(i);
        }
        
        // Calculate predicted state covariance
        P.setZero();
        for (int i = 0; i < SIG_CNT; ++i) {
            Vec dx = sigma_points_pred.col(i) - x;
            P += weights_c(i) * dx * dx.transpose();
        }
        P += Q;  // Add process noise
    }

    void updateWithMeasurement(const SigmaMat& sigma_points_pred, Real y_meas) {
        // Transform sigma points through measurement model
        Eigen::Matrix<Real, 1, SIG_CNT> y_sigma;
        for (int i = 0; i < SIG_CNT; ++i) {
            y_sigma(i) = measurementModel(sigma_points_pred.col(i));
        }
        
        // Calculate mean measurement
        Real y_pred = 0;
        for (int i = 0; i < SIG_CNT; ++i) {
            y_pred += weights_m(i) * y_sigma(i);
        }
        
        // Calculate innovation covariance
        Real Pyy = R(0, 0);
        Vec Pxy = Vec::Zero();
        for (int i = 0; i < SIG_CNT; ++i) {
            Real dy = y_sigma(i) - y_pred;
            Vec dx = sigma_points_pred.col(i) - x;
            Pyy += weights_c(i) * dy * dy;
            Pxy += weights_c(i) * dx * dy;
        }
        
        // Kalman update
        if (std::abs(Pyy) < 1e-6 || std::isnan(Pyy)) {
            Pyy = Real(1e-6);
        }
        Vec K = Pxy / Pyy;
        x += K * (y_meas - y_pred);
        P -= K * Pyy * K.transpose();
        for (int i = 0; i < N_STATE; ++i) {
            if (P(i, i) < 1e-9f) { 
                P(i, i) = 1e-9f;
            }
        }
    }

    Real measurementModel(const Vec& x_sigma) {
        Real y = 0;
        Real clamped = std::clamp(x_sigma(2 * M), Real(2 * M_PI * 0.02), Real(2 * M_PI * 10.0));
        x_sigma(2 * M) = clamped;
        Real omega = x_sigma(2 * M);
        for (int k = 1; k <= M; ++k) {
            int idx = 2 * (k - 1);
            Real cos_term = x_sigma(idx);
            Real kw = k * omega;
            kw = std::clamp(kw, Real(1e-5), Real(1e+5));
            Real term = -kw * kw * cos_term;
            y += term;
        }
        y += x_sigma(2 * M + 1);  // Add bias     
        return y;
    }
};
