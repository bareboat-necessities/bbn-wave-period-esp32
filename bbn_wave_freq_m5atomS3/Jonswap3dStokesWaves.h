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

// JonswapSpectrum 
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

// Jonswap3dStokesWaves
template<int N_FREQ = 256>
class Jonswap3dStokesWaves {
public:
    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02,
                         double f_max = 0.8,
                         double gamma = 2.0,
                         double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 239u)
        : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max),
          gamma_(gamma), g_(g),
          spreading_exponent_(spreading_exponent),
          mean_direction_rad_(mean_direction_deg * M_PI / 180.0),
          seed_(seed)
    {
        initializeSpectrum();
        precomputePairwise(); // prepare all O(N²) data once
    }

    // free surface elevation (first + second order)
    double surfaceElevation(double x, double y, double t) const {
        return eta_sumfreq(x, y, t) + eta2_sumfreq(x, y, t);
    }

    // particle velocities (u,v,w) at depth z
    Eigen::Vector3d evaluateVelocity(double x, double y, double z, double t) const {
        Eigen::Vector3d vel(0,0,0);

        // First-order velocities
        for (int i = 0; i < N_FREQ; ++i) {
            double kx = kx_(i), ky = ky_(i), k = k_(i);
            double omega = omega_(i);
            double phase = kx * x + ky * y - omega * t + phi_(i);
            double expz = std::exp(k * z);

            vel(0) += A_(i) * omega * expz * std::cos(phase) * (kx / k);
            vel(1) += A_(i) * omega * expz * std::cos(phase) * (ky / k);
            vel(2) += A_(i) * omega * expz * std::sin(phase);
        }

        // Second-order corrections (symmetric sums)
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double Bij = Bij_(i,j);
                double kx_sum = kx_sum_(i,j);
                double ky_sum = ky_sum_(i,j);
                double k_sum = std::sqrt(kx_sum*kx_sum + ky_sum*ky_sum);
                double omega_sum = omega_sum_(i,j);
                double phi_sum = phi_sum_(i,j);

                double th = kx_sum*x + ky_sum*y - omega_sum*t + phi_sum;
                double expz = std::exp(k_sum*z);

                double cos_th = std::cos(th);
                double sin_th = std::sin(th);

                double factor = (i == j) ? 1.0 : 2.0;

                vel(0) += factor * Bij * omega_sum * expz * cos_th * (kx_sum / k_sum);
                vel(1) += factor * Bij * omega_sum * expz * cos_th * (ky_sum / k_sum);
                vel(2) += factor * Bij * omega_sum * expz * sin_th;
            }
        }

        return vel;
    }

    // particle accelerations (ax, ay, az) at depth z
    Eigen::Vector3d evaluateAcceleration(double x, double y, double z, double t) const {
        Eigen::Vector3d acc(0,0,0);

        // First-order accelerations
        for (int i = 0; i < N_FREQ; ++i) {
            double kx = kx_(i), ky = ky_(i), k = k_(i);
            double omega = omega_(i);
            double phase = kx * x + ky * y - omega * t + phi_(i);
            double expz = std::exp(k * z);

            acc(0) -= A_(i) * omega * omega * expz * std::sin(phase) * (kx / k);
            acc(1) -= A_(i) * omega * omega * expz * std::sin(phase) * (ky / k);
            acc(2) += A_(i) * omega * omega * expz * std::cos(phase);
        }

        // Second-order accelerations
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double Bij = Bij_(i,j);
                double kx_sum = kx_sum_(i,j);
                double ky_sum = ky_sum_(i,j);
                double k_sum = std::sqrt(kx_sum*kx_sum + ky_sum*ky_sum);
                double omega_sum = omega_sum_(i,j);
                double phi_sum = phi_sum_(i,j);

                double th = kx_sum*x + ky_sum*y - omega_sum*t + phi_sum;
                double expz = std::exp(k_sum*z);

                double sin_th = std::sin(th);
                double cos_th = std::cos(th);

                double factor = (i == j) ? 1.0 : 2.0;

                acc(0) -= factor * Bij * omega_sum*omega_sum * expz * sin_th * (kx_sum / k_sum);
                acc(1) -= factor * Bij * omega_sum*omega_sum * expz * sin_th * (ky_sum / k_sum);
                acc(2) += factor * Bij * omega_sum*omega_sum * expz * cos_th;
            }
        }

        return acc;
    }

private:
    // --- Core parameters ---
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    double spreading_exponent_;
    double mean_direction_rad_;
    unsigned int seed_;

    // --- First-order state ---
    Eigen::Matrix<double, N_FREQ, 1> A_, k_, omega_, phi_, theta_;
    Eigen::Matrix<double, N_FREQ, 1> kx_, ky_;

    // --- Pairwise precomputed matrices ---
    Eigen::Matrix<double, N_FREQ, N_FREQ> Bij_;
    Eigen::Matrix<double, N_FREQ, N_FREQ> kx_sum_, ky_sum_;
    Eigen::Matrix<double, N_FREQ, N_FREQ> omega_sum_, phi_sum_;

    // initialize first-order spectrum
    void initializeSpectrum() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> unif(0.0, 2.0 * M_PI);
        std::normal_distribution<double> normal(0.0, 1.0);

        double df = (f_max_ - f_min_) / N_FREQ;

        for (int i = 0; i < N_FREQ; ++i) {
            double f = f_min_ + (i + 0.5) * df;
            double S = jonswapSpectrum(f);

            // Amplitude from spectrum (Rayleigh distributed)
            double amp = std::sqrt(2.0 * S * df);
            A_(i) = amp * std::fabs(normal(gen));

            omega_(i) = 2.0 * M_PI * f;
            k_(i) = omega_(i) * omega_(i) / g_;
            phi_(i) = unif(gen);

            // spreading: cos^{2s}((θ-θ0)/2)
            double theta_rel = std::acos(2.0 * ((i+0.5)/N_FREQ) - 1.0);
            theta_(i) = mean_direction_rad_ + theta_rel;
            kx_(i) = k_(i) * std::cos(theta_(i));
            ky_(i) = k_(i) * std::sin(theta_(i));
        }
    }

    // precompute pairwise sums (Bij, kx+ky, omega+phi)
    void precomputePairwise() {
        Bij_.setZero();
        kx_sum_.setZero();
        ky_sum_.setZero();
        omega_sum_.setZero();
        phi_sum_.setZero();

        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double kx_sum = kx_(i) + kx_(j);
                double ky_sum = ky_(i) + ky_(j);
                double kdot = kx_(i)*kx_(j) + ky_(i)*ky_(j);
                double Bij = (kdot)/(2.0*g_) * A_(i) * A_(j);
                double omega_sum = omega_(i) + omega_(j);
                double phi_sum = phi_(i) + phi_(j);

                Bij_(i,j) = Bij_(j,i) = Bij;
                kx_sum_(i,j) = kx_sum_(j,i) = kx_sum;
                ky_sum_(i,j) = ky_sum_(j,i) = ky_sum;
                omega_sum_(i,j) = omega_sum_(j,i) = omega_sum;
                phi_sum_(i,j) = phi_sum_(j,i) = phi_sum;
            }
        }
    }

    // first-order elevation
    double eta_sumfreq(double x, double y, double t) const {
        double eta = 0.0;
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i)*x + ky_(i)*y - omega_(i)*t + phi_(i);
            eta += A_(i) * std::cos(th);
        }
        return eta;
    }

    // second-order elevation
    double eta2_sumfreq(double x, double y, double t) const {
        double eta2 = 0.0;
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double th = kx_sum_(i,j)*x + ky_sum_(i,j)*y - omega_sum_(i,j)*t + phi_sum_(i,j);
                double contrib = Bij_(i,j) * std::cos(th);
                eta2 += (i == j) ? contrib : 2.0 * contrib;
            }
        }
        return eta2;
    }

    // JONSWAP spectrum (Hz-based)
    double jonswapSpectrum(double f) const {
        double fp = 1.0 / Tp_;
        double alpha = 0.076 * std::pow(Hs_*Hs_ * fp*fp*fp*fp / (g_*g_), -0.22);
        double sigma = (f <= fp) ? 0.07 : 0.09;
        double r = std::exp(-(f-fp)*(f-fp) / (2.0*sigma*sigma*fp*fp));
        double gamma_term = std::pow(gamma_, r);
        double S = alpha * g_*g_ * std::pow(2.0*M_PI, -4.0) * std::pow(f, -5.0) *
                   std::exp(-1.25 * std::pow(fp/f, 4.0)) * gamma_term;
        return S;
    }
};

#ifdef JONSWAP_TEST
void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
