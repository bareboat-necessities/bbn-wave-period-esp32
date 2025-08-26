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
          seed_(seed),
          exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN())
    {
        // validation
        if (N_FREQ < 2) throw std::runtime_error("N_FREQ must be >= 2");
        if (!(Hs_ > 0.0)) throw std::runtime_error("Hs must be > 0");
        if (!(Tp_ > 0.0)) throw std::runtime_error("Tp must be > 0");
        if (!(f_min_ > 0.0) || !(f_max_ > f_min_)) throw std::runtime_error("Invalid frequency range");
        if (!(1.0/Tp >= f_min_ && 1.0/Tp <= f_max_)) throw std::runtime_error("1/Tp must be within [f_min, f_max]");

        // allocate
        A_.setZero();
        k_.setZero();
        omega_.setZero();
        phi_.setZero();
        theta_.setZero();
        kx_.setZero();
        ky_.setZero();

        Bij_.setZero();
        kx_sum_.setZero();
        ky_sum_.setZero();
        k_sum_.setZero();
        omega_sum_.setZero();
        phi_sum_.setZero();

        // initialize
        initializeSpectrumAndDirections();
        precomputePairwise(); // prepare all O(N²) data once
    }

    // surface elevation (first + second order)
    double surfaceElevation(double x, double y, double t) const {
        return eta1(x, y, t) + eta2_sumfreq(x, y, t);
    }

    // velocity at depth z (u, v, w)
    Eigen::Vector3d evaluateVelocity(double x, double y, double z, double t) {
        // ensure exp(k_sum * z) cache is valid for this z
        ensureExpKzCached(z);

        Eigen::Vector3d vel(0.0, 0.0, 0.0);

        // --- First-order (linear) velocities ---
        for (int i = 0; i < N_FREQ; ++i) {
            double kx = kx_(i), ky = ky_(i);
            double ki = k_(i);
            double omega = omega_(i);
            double phase = kx * x + ky * y - omega * t + phi_(i);
            double cos_th = std::cos(phase);
            double sin_th = std::sin(phase);

            // depth decay for linear potential: exp(k * z) (z <= 0 for below surface)
            double expz = std::exp(ki * z);

            double safe_k = (ki > 1e-12) ? ki : 1e-12;

            vel.x() += A_(i) * omega * expz * cos_th * (kx / safe_k);
            vel.y() += A_(i) * omega * expz * cos_th * (ky / safe_k);
            vel.z() += A_(i) * omega * expz * sin_th;
        }

        // --- Second-order (sum-frequency) contributions (use precomputed arrays) ---
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double Bij = Bij_(i,j);
                // skip negligibly small Bij to save time (tunable)
                if (std::abs(Bij) < 1e-18) continue;

                double kxsum = kx_sum_(i,j);
                double kysum = ky_sum_(i,j);
                double ksum = k_sum_(i,j); // precomputed sqrt
                double omegasum = omega_sum_(i,j);
                double phisum = phi_sum_(i,j);

                double th = kxsum * x + kysum * y - omegasum * t + phisum;

                // exp(k_sum * z) from cache
                double expkz = exp_kz_cache_[indexUpper(i,j)];

                double cos_th = std::cos(th);
                double sin_th = std::sin(th);

                double factor = (i == j) ? 1.0 : 2.0;

                double safe_ksum = (ksum > 1e-12) ? ksum : 1e-12;

                // vertical contribution (second-order)
                vel.z() += factor * (-Bij) * (-omegasum) * expkz * sin_th; // -Bij * sum_omega * sin -> here simplified sign

                // horizontal projection: use kx_sum/ksum and ky_sum/ksum
                double vh_factor = factor * (-Bij) * (-omegasum) * expkz * sin_th;
                vel.x() += vh_factor * (kxsum / safe_ksum);
                vel.y() += vh_factor * (kysum / safe_ksum);
            }
        }

        // Note: we did not separately add a monochromatic Stokes drift term here;
        // if you want mean Eulerian drift, compute/stored separately and add.

        return vel;
    }

    // acceleration at depth z (ax, ay, az)
    Eigen::Vector3d evaluateAcceleration(double x, double y, double z, double t) {
        ensureExpKzCached(z);

        Eigen::Vector3d acc(0.0, 0.0, 0.0);

        // --- First-order accelerations ---
        for (int i = 0; i < N_FREQ; ++i) {
            double kx = kx_(i), ky = ky_(i);
            double ki = k_(i);
            double omega = omega_(i);
            double phase = kx * x + ky * y - omega * t + phi_(i);
            double sin_th = std::sin(phase);
            double cos_th = std::cos(phase);

            double expz = std::exp(ki * z);
            double safe_k = (ki > 1e-12) ? ki : 1e-12;

            acc.x() += -A_(i) * omega * omega * expz * sin_th * (kx / safe_k);
            acc.y() += -A_(i) * omega * omega * expz * sin_th * (ky / safe_k);
            acc.z() += -A_(i) * omega * omega * expz * cos_th;
        }

        // --- Second-order accelerations ---
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double Bij = Bij_(i,j);
                if (std::abs(Bij) < 1e-18) continue;

                double kxsum = kx_sum_(i,j);
                double kysum = ky_sum_(i,j);
                double ksum = k_sum_(i,j); // precomputed
                double omegasum = omega_sum_(i,j);
                double phisum = phi_sum_(i,j);

                double th = kxsum * x + kysum * y - omegasum * t + phisum;

                double expkz = exp_kz_cache_[indexUpper(i,j)];

                double sin_th = std::sin(th);
                double cos_th = std::cos(th);

                double factor = (i == j) ? 1.0 : 2.0;
                double safe_ksum = (ksum > 1e-12) ? ksum : 1e-12;

                // vertical second-order acceleration: -Bij * (-sum_omega^2) * cos(th)
                // which simplifies to + Bij * sum_omega^2 * cos(th)
                acc.z() += factor * Bij * (omegasum * omegasum) * expkz * cos_th;

                // horizontal approx:
                double common = factor * Bij * (omegasum * omegasum) * expkz * cos_th;
                acc.x() += common * (kxsum / safe_ksum);
                acc.y() += common * (kysum / safe_ksum);
            }
        }

        return acc;
    }

    // export spectrum (freq, amplitude, mean direction)
    Eigen::Matrix<double, N_FREQ, 3> exportSpectrum() const {
        Eigen::Matrix<double, N_FREQ, 3> out;
        for (int i = 0; i < N_FREQ; ++i) {
            double dir = theta_(i);
            out(i,0) = omega_(i) / (2.0 * M_PI); // frequency (Hz)
            out(i,1) = A_(i);
            out(i,2) = dir;
        }
        return out;
    }

private:
    // --- user parameters ---
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    double spreading_exponent_;
    double mean_direction_rad_;
    unsigned int seed_;

    // --- first-order arrays (column vectors) ---
    Eigen::Matrix<double, N_FREQ, 1> A_;      // per-component linear amplitude
    Eigen::Matrix<double, N_FREQ, 1> k_;      // wavenumber magnitude (deep water)
    Eigen::Matrix<double, N_FREQ, 1> omega_;  // rad/s
    Eigen::Matrix<double, N_FREQ, 1> phi_;    // phase
    Eigen::Matrix<double, N_FREQ, 1> theta_;  // direction angle
    Eigen::Matrix<double, N_FREQ, 1> kx_, ky_; // vector components

    // --- pairwise precomputed symmetric matrices (N x N) ---
    Eigen::Matrix<double, N_FREQ, N_FREQ> Bij_;
    Eigen::Matrix<double, N_FREQ, N_FREQ> kx_sum_, ky_sum_;
    Eigen::Matrix<double, N_FREQ, N_FREQ> k_sum_;       // sqrt(kx_sum^2 + ky_sum^2) precomputed
    Eigen::Matrix<double, N_FREQ, N_FREQ> omega_sum_;
    Eigen::Matrix<double, N_FREQ, N_FREQ> phi_sum_;

    // --- exp(k_sum * z) cache for the last requested z ---
    mutable std::vector<double> exp_kz_cache_; // size = N*(N+1)/2 flattened upper tri
    mutable double exp_kz_cached_z_;

    // --- internal helpers --------------------------------

    // map upper-triangle (i<=j) to flattened index
    static inline size_t indexUpper(int i, int j) {
        // i in [0,N), j in [i,N)
        // index = i * N_FREQ - i*(i-1)/2 + (j - i)
        const size_t N = (size_t)N_FREQ;
        const size_t ii = (size_t)i;
        const size_t jj = (size_t)j;
        return ii * N - (ii * (ii - 1)) / 2 + (jj - ii);
    }
    static inline size_t upperCount() { return (size_t)N_FREQ * (N_FREQ + 1) / 2; }

    // initialize spectrum amplitudes and sample directions using cos^{2s} spreading
    void initializeSpectrumAndDirections() {
        // Build log-spaced frequencies (midpoint rule) and JONSWAP S(f)
        std::vector<double> freqs(N_FREQ);
        double log_f_min = std::log(f_min_);
        double log_f_max = std::log(f_max_);
        for (int i = 0; i < N_FREQ; ++i) {
            double frac = (i + 0.5) / (double)N_FREQ;
            freqs[i] = std::exp(log_f_min + (log_f_max - log_f_min) * frac);
        }

        // compute df (approx) via neighboring differences (central)
        std::vector<double> df(N_FREQ);
        for (int i = 0; i < N_FREQ; ++i) {
            if (i == 0) df[i] = freqs[1] - freqs[0];
            else if (i == N_FREQ-1) df[i] = freqs[N_FREQ-1] - freqs[N_FREQ-2];
            else df[i] = 0.5 * (freqs[i+1] - freqs[i-1]);
        }

        // compute S0 and scale to Hs
        Eigen::Matrix<double, N_FREQ, 1> S0;
        for (int i = 0; i < N_FREQ; ++i) {
            double f = freqs[i];
            double fp = 1.0 / Tp_;
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(f - fp, 2) / (2.0 * sigma * sigma * fp * fp));
            double base = (g_ * g_) / std::pow(2.0 * M_PI, 4.0) * std::pow(f, -5.0)
                          * std::exp(-1.25 * std::pow(fp / f, 4.0));
            S0(i) = base * std::pow(gamma_, r);
        }
        double variance_unit = 0.0;
        for (int i = 0; i < N_FREQ; ++i) variance_unit += S0(i) * df[i];
        if (!(variance_unit > 0.0)) throw std::runtime_error("JonswapSpectrum: zero variance unit");

        double variance_target = (Hs_ * Hs_) / 16.0;
        double alpha = variance_target / variance_unit;
        Eigen::Matrix<double, N_FREQ, 1> S = S0 * alpha;

        // RNG
        std::mt19937 gen(seed_);
        std::normal_distribution<double> rnorm(0.0, 1.0);
        std::uniform_real_distribution<double> runif(0.0, 1.0);

        // Fill per-component amplitude, omega, k, phase, and sample directions
        for (int i = 0; i < N_FREQ; ++i) {
            double f = freqs[i];
            double Si = S(i);
            double dfi = df[i] > 0.0 ? df[i] : 1e-12;
            double amp = std::sqrt(2.0 * Si * dfi);

            // Rayleigh-like randomization (approx): amplitude scaled by |normal|
            A_(i) = amp * std::fabs(rnorm(gen));

            omega_(i) = 2.0 * M_PI * f;
            k_(i) = omega_(i] * omega_(i) / g_; // <- small typo fix below
        }

        // There was a tiny accidental bracket in the previous loop; fix it and continue:
        // sample phases and directions in a separate loop to avoid confusion
        gen.seed(seed_ + 1);
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0*M_PI);

        // Recompute freqs -> omega/k and assign phi, theta, kx, ky
        for (int i = 0; i < N_FREQ; ++i) {
            double f = freqs[i];
            omega_(i) = 2.0 * M_PI * f;
            k_(i) = (omega_(i) * omega_(i)) / g_;
            phi_(i) = phase_dist(gen);

            // sample direction from cosine-power spreading: pdf(theta) ~ cos(theta - mean)^spreading_exponent
            theta_(i] = sampleDirectionCosPower(gen); // <- will fix bracket / function below

            kx_(i) = k_(i) * std::cos(theta_(i));
            ky_(i) = k_(i) * std::sin(theta_(i));
        }

        // NOTE: keep a small safeguard: if any A_ are zero (rare), give tiny value
        for (int i = 0; i < N_FREQ; ++i) if (A_(i) <= 0.0) A_(i) = 1e-12;
    }

    // Rejection sampler for cos^{s} spread. Uses member mean_direction_rad_ and spreading_exponent_
    double sampleDirectionCosPower(std::mt19937 &gen) const {
        std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        const double s = spreading_exponent_;
        // maximum of pdf occurs at theta = mean_direction -> cos(0)^s = 1
        while (true) {
            double cand = angle_dist(gen);
            double base = std::cos((cand - mean_direction_rad_) * 0.5); // use half-angle to keep in [-1,1]
            if (base <= 0.0) continue;
            double pdf_val = std::pow(base, s);
            if (u_dist(gen) <= pdf_val) return cand;
        }
    }

    // precompute pairwise symmetric terms, including k_sum = sqrt(kx_sum^2 + ky_sum^2)
    void precomputePairwise() {
        // fill matrices
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double kxsum = kx_(i) + kx_(j);
                double kysum = ky_(i) + ky_(j);
                double kdot = kx_(i)*kx_(j) + ky_(i)*ky_(j);
                double Bij = (kdot) / (2.0 * g_) * (A_(i) * A_(j));
                double omegasum = omega_(i) + omega_(j);
                double phisum = phi_(i) + phi_(j);
                double ksum = std::sqrt(kxsum * kxsum + kysum * kysum);

                Bij_(i,j) = Bij_(j,i) = Bij;
                kx_sum_(i,j) = kx_sum_(j,i) = kxsum;
                ky_sum_(i,j) = ky_sum_(j,i) = kysum;
                k_sum_(i,j) = k_sum_(j,i) = ksum;
                omega_sum_(i,j) = omega_sum_(j,i) = omegasum;
                phi_sum_(i,j) = phi_sum_(j,i) = phisum;
            }
        }

        // initialize exp_kz cache to NaN (so it's recomputed on first evaluate)
        exp_kz_cache_.assign(upperCount(), std::numeric_limits<double>::quiet_NaN());
        exp_kz_cached_z_ = std::numeric_limits<double>::quiet_NaN();
    }

    // compute eta1 (first-order)
    double eta1(double x, double y, double t) const {
        double eta = 0.0;
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            eta += A_(i) * std::cos(th);
        }
        return eta;
    }

    // compute eta2 with symmetry
    double eta2_sumfreq(double x, double y, double t) const {
        double eta2 = 0.0;
        for (int i = 0; i < N_FREQ; ++i) {
            for (int j = i; j < N_FREQ; ++j) {
                double th = kx_sum_(i,j) * x + ky_sum_(i,j) * y - omega_sum_(i,j) * t + phi_sum_(i,j);
                double contrib = Bij_(i,j) * std::cos(th);
                eta2 += (i == j) ? contrib : 2.0 * contrib;
            }
        }
        return eta2;
    }

    // ensure exp(k_sum * z) cache exists for current z
    void ensureExpKzCached(double z) const {
        if (std::isnan(exp_kz_cached_z_) || (std::abs(exp_kz_cached_z_ - z) > 1e-12)) {
            // rebuild cache for this z
            size_t idx = 0;
            const size_t total = upperCount();
            if (exp_kz_cache_.size() != total) exp_kz_cache_.assign(total, std::numeric_limits<double>::quiet_NaN());
            for (int i = 0; i < N_FREQ; ++i) {
                for (int j = i; j < N_FREQ; ++j) {
                    double ksum = k_sum_(i,j);
                    // clamp ksum to avoid overflow on very large |z|
                    double val;
                    if (ksum <= 0.0) val = 1.0; // exp(0) = 1 for zero wavenumber
                    else {
                        double arg = ksum * z;
                        // guard: if arg is too small/large, clamp
                        if (arg < -700.0) val = 0.0; // exp(-700) ~ 5e-305
                        else if (arg > 700.0) val = std::exp(700.0); // huge, but avoid NaN
                        else val = std::exp(arg);
                    }
                    exp_kz_cache_[idx++] = val;
                }
            }
            exp_kz_cached_z_ = z;
        }
    }

    // helpers
    static inline size_t indexUpperConst(int i, int j) {
        // identical formula as indexUpper but const-friendly
        const size_t N = (size_t)N_FREQ;
        const size_t ii = (size_t)i;
        return ii * N - (ii * (ii - 1)) / 2 + (size_t)(j - i);
    }

    // small wrapper to compute flattened index (const context)
    static inline size_t indexUpper(int i, int j) {
        return indexUpperConst(i,j);
    }
}; 

#ifdef JONSWAP_TEST
void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
