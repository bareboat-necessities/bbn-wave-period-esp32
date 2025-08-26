#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Stokes-corrected waves simulation (surface, deep-water).
  - 1st order: Airy (linear) components
  - 2nd order: simplified deep-water sum-frequency bound harmonics + simple
    estimate of surface Stokes drift (monochromatic approximation)
*/

// JonswapSpectrum (unchanged from your version)
template<int N_FREQ = 256>
class JonswapSpectrum {
public:
    JonswapSpectrum(double Hs, double Tp,
                    double f_min = 0.02, double f_max = 0.8,
                    double gamma = 2.0, double g = 9.81)
        : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max), gamma_(gamma), g_(g)
    {
        if (N_FREQ < 2) throw std::runtime_error("N_FREQ must be >= 2");
        if (!(Hs_ > 0.0)) throw std::runtime_error("Hs must be > 0");
        if (!(Tp_ > 0.0)) throw std::runtime_error("Tp must be > 0");
        if (!(f_min_ > 0.0) || !(f_max_ > f_min_)) throw std::runtime_error("Invalid frequency range");
        if (!(1.0/Tp >= f_min_ && 1.0/Tp <= f_max_)) throw std::runtime_error("1/Tp must be within [f_min, f_max]");

        frequencies_.setZero();
        S_.setZero();
        A_.setZero();
        df_.setZero();

        computeLogFrequencySpacing();
        computeFrequencyIncrements();
        computeJonswapSpectrumFromHs();
    }

    // Accessors (fixed-size Eigen vectors)
    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum() const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes() const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df() const { return df_; }

    // Diagnostics
    double integratedVariance() const {
        return (S_.cwiseProduct(df_)).sum();
    }

private:
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
        double log_f_min = std::log(f_min_);
        double log_f_max = std::log(f_max_);
        for (int i = 0; i < N_FREQ; ++i)
            frequencies_(i) = std::exp(log_f_min + (log_f_max - log_f_min) * i / (N_FREQ - 1));
    }

    void computeFrequencyIncrements() {
        if (N_FREQ < 2) {
            df_.setZero();
            return;
        }
        df_(0) = frequencies_(1) - frequencies_(0); // forward diff
        for (int i = 1; i < N_FREQ - 1; ++i) {       // central diff
            df_(i) = 0.5 * (frequencies_(i + 1) - frequencies_(i - 1));
        }
        df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2); // backward diff
    }

    void computeJonswapSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        Eigen::Matrix<double, N_FREQ, 1> S0;
        for (int i = 0; i < N_FREQ; ++i) {
            double f = frequencies_(i);
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(f - fp, 2) / (2.0 * sigma * sigma * fp * fp));
            double base = (g_ * g_) / std::pow(2.0 * M_PI, 4.0) * std::pow(f, -5.0)
                          * std::exp(-1.25 * std::pow(fp / f, 4.0));
            S0(i) = base * std::pow(gamma_, r);
        }

        double variance_unit = (S0.cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("JonswapSpectrum: computed zero/negative variance (check frequency grid)");

        double variance_target = (Hs_ * Hs_) / 16.0;
        double alpha = variance_target / variance_unit;

        S_ = S0 * alpha;
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

        // sanity: tiny relative mismatch should be corrected
        double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est <= 0.0) throw std::runtime_error("JonswapSpectrum: Hs_est <= 0 after amplitude computation");
        double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            // correct numerical mismatch
            A_ *= (Hs_ / Hs_est);
            // recompute S from A for internal consistency
            for (int i = 0; i < N_FREQ; ++i) {
                double dfi = df_(i) > 0.0 ? df_(i) : 1e-12;
                S_(i) = (A_(i) * A_(i)) / (2.0 * dfi);
            }
        }
    }
};

// Jonswap3dStokesWaves (replacement for Gerstner)
template<int N_FREQ = 256>
class Jonswap3dStokesWaves {
public:
    struct WaveState {
        Eigen::Vector3d displacement;   // (x,y,z) Lagrangian/Eulerian depending on method
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02,
                         double f_max = 0.8,
                         double gamma = 2.0,
                         double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u)
        : spectrum_(Hs, Tp, f_min, f_max, gamma, g),
          Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          gamma_(gamma), g_(g), spreading_exponent_(spreading_exponent),
          seed_(seed)
    {
        // copy spectrum data
        frequencies_ = spectrum_.frequencies();
        S_ = spectrum_.spectrum();
        A_ = spectrum_.amplitudes();
        df_ = spectrum_.df();

        // derived
        omega_.setZero(); k_.setZero();
        phi_.setZero();
        dir_x_.setZero(); dir_y_.setZero();
        kx_.setZero(); ky_.setZero();

        omega_ = 2.0 * M_PI * frequencies_;
        // deep-water dispersion relation: k = omega^2 / g
        k_ = omega_.array().square() / g_;

        initializeRandomPhases();
        initializeDirectionalSpreadRejection();
        computeWaveDirectionComponents();

        computePerComponentStokesDriftEstimate();

        checkSteepness();
    }

    // same API as before
    WaveState getLagrangianState(double x0, double y0, double t) const {
        // For simplicity, this returns Eulerian-position-based kinematics plus
        // an estimate of mean Stokes drift added to the horizontal velocity.
        // Full Lagrangian tracking of particles requires integrating velocities.
        Eigen::Vector3d disp = evaluateDisplacement(x0, y0, t);
        Eigen::Vector3d vel = evaluateVelocity(x0, y0, t);
        Eigen::Vector3d acc = evaluateAcceleration(x0, y0, t);
        return {disp, vel, acc};
    }

    WaveState getEulerianState(double x, double y, double t) const {
        return getLagrangianState(x, y, t);
    }

    Eigen::Matrix<double, N_FREQ, 3> exportSpectrum() const {
        Eigen::Matrix<double, N_FREQ, 3> result;
        for (int i = 0; i < N_FREQ; ++i) {
            double dir_angle = std::atan2(dir_y_(i), dir_x_(i));
            result(i, 0) = frequencies_(i);
            result(i, 1) = A_(i);
            result(i, 2) = dir_angle;
        }
        return result;
    }

private:
    // Owned spectrum
    JonswapSpectrum<N_FREQ> spectrum_;

    // original parameters (kept for API/logic)
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;

    // Arrays (copied/viewed from spectrum where relevant)
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, S_, A_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;

    // Precomputed per-component Stokes drift (surface) vector (mean Eulerian velocity)
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
    Eigen::Vector2d stokes_drift_mean_xy_ = Eigen::Vector2d::Zero();

    // Randomization
    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
        for (int i = 0; i < N_FREQ; ++i)
            phi_(i) = dist(gen);
    }

    void initializeDirectionalSpreadRejection() {
        std::mt19937 gen(seed_ + 1);
        std::uniform_real_distribution<double> u_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> y_dist(0.0, 1.0);

        for (int i = 0; i < N_FREQ; ++i) {
            double theta = 0.0;
            while (true) {
                double candidate = u_dist(gen);
                double base = std::cos(candidate - mean_dir_rad_);
                // clamp base to [0,1]
                double clamped = std::max(0.0, base);
                double pdf_val = std::pow(clamped, spreading_exponent_);
                if (y_dist(gen) <= pdf_val) {
                    theta = candidate;
                    break;
                }
            }
            dir_x_(i) = std::cos(theta);
            dir_y_(i) = std::sin(theta);
        }
    }

    void computeWaveDirectionComponents() {
        for (int i = 0; i < N_FREQ; ++i) {
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    void computePerComponentStokesDriftEstimate() {
        // Deep-water monochromatic approximation:
        // U_s_i = 0.5 * a_i^2 * k_i * omega_i  (surface)
        // We'll project onto direction to obtain vector contribution.
        stokes_drift_scalar_.setZero();
        stokes_drift_mean_xy_.setZero();

        for (int i = 0; i < N_FREQ; ++i) {
            double a = A_(i);                          // linear amplitude per component
            double ki = k_(i);
            double wi = omega_(i);
            double Usi = 0.5 * a * a * ki * wi;       // deep-water approx
            stokes_drift_scalar_(i) = Usi;
            stokes_drift_mean_xy_.x() += Usi * dir_x_(i);
            stokes_drift_mean_xy_.y() += Usi * dir_y_(i);
        }
    }

    // --- Evaluations ---

    // First-order (Airy/linear) surface elevation
    double eta1(double x, double y, double t) const {
        double eta = 0.0;
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            eta += A_(i) * std::cos(th);
        }
        return eta;
    }

    // Second-order simplified bound-wave vertical elevation (sum-frequency only)
    // B_ij = (k_i · k_j) / (2 g) * a_i * a_j   (deep-water simplified)
    double eta2_sumfreq(double x, double y, double t) const {
        double eta2 = 0.0;
        // note: double-loop is O(N^2); acceptable for moderate N_FREQ but heavy for large N
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = 0; j < N_FREQ; ++j) {
                double th = (kx_(i) + kx_(j)) * x + (ky_(i) + ky_(j)) * y
                            - (omega_(i) + omega_(j)) * t + (phi_(i) + phi_(j));
                double kdot = kx_(i) * kx_(j) + ky_(i) * ky_(j); // = k_i k_j cos(theta_ij)
                double Bij = (kdot) / (2.0 * g_) * (A_(i) * A_(j));
                eta2 += Bij * std::cos(th);
            }
        }
        return eta2;
    }

    // Combined surface elevation (η = η1 + η2)
    double surfaceElevation(double x, double y, double t) const {
        return eta1(x, y, t) + eta2_sumfreq(x, y, t);
    }

    // Velocity (Eulerian) at surface:
    // Linear (u_x,u_y) contributions: u_x^1 = sum a_i * ω_i * cos(th) * (k_x / k)
    // Vertical velocity w^1 = sum a_i * ω_i * sin(th)
    // Second-order bound contributions obtained by time-derivative of eta2 (vertical)
    Eigen::Vector3d evaluateVelocity(double x, double y, double t) const {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();

        // 1st-order horizontal and vertical oscillatory components
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            double sin_th = std::sin(th);
            double factor = A_(i) * omega_(i);
            double k_mag = k_(i) > 0.0 ? k_(i) : 1e-12;
            double dirx = (k_mag > 0.0) ? (kx_(i) / k_mag) : dir_x_(i);
            double diry = (k_mag > 0.0) ? (ky_(i) / k_mag) : dir_y_(i);

            v.x() += factor * cos_th * dirx;   // u_x (oscillatory)
            v.y() += factor * cos_th * diry;   // u_y (oscillatory)
            v.z() += factor * sin_th;          // w (oscillatory)
        }

        // 2nd-order vertical velocity: time derivative of eta2_sumfreq (w2 = d/dt eta2)
        // For cos(theta_sum) term, derivative is - (omega_i+omega_j) * sin(theta_sum)
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = 0; j < N_FREQ; ++j) {
                double th = (kx_(i) + kx_(j)) * x + (ky_(i) + ky_(j)) * y
                            - (omega_(i) + omega_(j)) * t + (phi_(i) + phi_(j));
                double kdot = kx_(i) * kx_(j) + ky_(i) * ky_(j);
                double Bij = (kdot) / (2.0 * g_) * (A_(i) * A_(j));
                double sum_omega = (omega_(i) + omega_(j));
                // vertical velocity contribution from bound sum-frequency
                v.z() += -Bij * sum_omega * std::sin(th);
                // horizontal components: approximate by projecting vertical time-derivative
                // into horizontal direction using (k_x / (k_i + k_j)) factor (very approximate)
                double ksum = std::sqrt((kx_(i) + kx_(j))*(kx_(i) + kx_(j)) + (ky_(i) + ky_(j))*(ky_(i) + ky_(j)));
                if (ksum > 1e-12) {
                    double hx = (kx_(i) + kx_(j)) / ksum;
                    double hy = (ky_(i) + ky_(j)) / ksum;
                    // assume horizontal velocity related: u_h ~ (1/ksum) * d/dt(eta2) * ksum => use Bij * sum_omega * sin * hx
                    v.x() += -Bij * sum_omega * std::sin(th) * hx;
                    v.y() += -Bij * sum_omega * std::sin(th) * hy;
                }
            }
        }

        // Add mean Stokes drift (steady Eulerian contribution) projected to x,y
        v.x() += stokes_drift_mean_xy_.x();
        v.y() += stokes_drift_mean_xy_.y();

        return v;
    }

    // Acceleration: time derivative of velocity (mostly -omega^2 times amplitudes)
    Eigen::Vector3d evaluateAcceleration(double x, double y, double t) const {
        Eigen::Vector3d a = Eigen::Vector3d::Zero();

        // 1st-order accelerations
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double sin_th = std::sin(th);
            double cos_th = std::cos(th);
            double fac = A_(i) * omega_(i) * omega_(i);
            double k_mag = k_(i) > 0.0 ? k_(i) : 1e-12;
            double dirx = (k_mag > 0.0) ? (kx_(i) / k_mag) : dir_x_(i);
            double diry = (k_mag > 0.0) ? (ky_(i) / k_mag) : dir_y_(i);

            a.x() += -fac * sin_th * dirx;    // du_x/dt
            a.y() += -fac * sin_th * diry;
            a.z() += -fac * cos_th;          // dw/dt
        }

        // 2nd-order accelerations from bound sum-frequency: time derivative of v contributions
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = 0; j < N_FREQ; ++j) {
                double th = (kx_(i) + kx_(j)) * x + (ky_(i) + ky_(j)) * y
                            - (omega_(i) + omega_(j)) * t + (phi_(i) + phi_(j));
                double kdot = kx_(i) * kx_(j) + ky_(i) * ky_(j);
                double Bij = (kdot) / (2.0 * g_) * (A_(i) * A_(j));
                double sum_omega = (omega_(i) + omega_(j));
                double sum_omega2 = sum_omega * sum_omega;
                // vertical acceleration contribution
                a.z() += -Bij * (-sum_omega2) * std::cos(th); // d/dt(-Bij*sum_omega*sin) = -Bij*sum_omega^2*cos
                // horizontal approx:
                double ksum = std::sqrt((kx_(i) + kx_(j))*(kx_(i) + kx_(j)) + (ky_(i) + ky_(j))*(ky_(i) + ky_(j)));
                if (ksum > 1e-12) {
                    double hx = (kx_(i) + kx_(j)) / ksum;
                    double hy = (ky_(i) + ky_(j)) / ksum;
                    a.x() += -Bij * (-sum_omega2) * std::cos(th) * hx;
                    a.y() += -Bij * (-sum_omega2) * std::cos(th) * hy;
                }
            }
        }

        // Stokes drift is steady; acceleration of mean drift is zero (we ignore slow variations)
        return a;
    }

    Eigen::Vector3d evaluateDisplacement(double x, double y, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        // horizontal displacement: approximate from linear potential displacement (not exact Lagrangian)
        // We'll produce Eulerian-like displacement: horizontal displacement ~ - sum (a * cos(th)) * dir
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            d.x() += -A_(i) * cos_th * dir_x_(i);
            d.y() += -A_(i) * cos_th * dir_y_(i);
            d.z() +=  A_(i) * std::sin(th);  // vertical (linear)
        }
        // add second-order vertical displacement from eta2_sumfreq
        d.z() += eta2_sumfreq(x, y, t);
        // add mean drift displacement (small) as a linear-in-time shift would be Lagrangian; we omit here.
        return d;
    }

    void checkSteepness() const {
        double max_steepness = (A_.array() * k_.array()).maxCoeff();
        if (max_steepness > 0.2)
            throw std::runtime_error("Wave steepness exceeds 0.2");
    }
};

#ifdef JONSWAP_TEST
void generateWaveJonswapCSV(const std::string& filename,
                            double Hs, double Tp, double mean_dir_deg,
                            double duration = 40.0, double dt = 0.005) {

    Jonswap3dStokesWaves<256> waveModel(
        Hs, Tp, mean_dir_deg, 0.02, 0.8, 2.0, 9.81, 10.0
    );

    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

    const double x0 = 0.0, y0 = 0.0;
    for (double t = 0; t <= duration; t += dt) {
        auto state = waveModel.getLagrangianState(x0, y0, t);
        file << t << ","
             << state.displacement.x() << ","
             << state.displacement.y() << ","
             << state.displacement.z() << ","
             << state.velocity.x() << ","
             << state.velocity.y() << ","
             << state.velocity.z() << ","
             << state.acceleration.x() << ","
             << state.acceleration.y() << ","
             << state.acceleration.z() << "\n";
    }
}

void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
