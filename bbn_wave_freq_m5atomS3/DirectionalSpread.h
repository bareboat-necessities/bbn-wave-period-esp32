#pragma once
#pragma GCC optimize ("no-fast-math")

/*
  Copyright 2025, Mikhail Grushinskiy
*/

#include <random>
#include <cmath>
#include <vector>
#include <memory>

#ifndef PI
static constexpr double PI = 3.14159265358979323846264338327950288;
#else
static constexpr double PI = M_PI;
#endif

// Directional Distribution Interface
class DirectionalDistribution {
public:
    virtual ~DirectionalDistribution() = default;

    // --- Theoretical spectrum interface ---
    // Evaluate continuous D(θ; f) (normalized so ∫ D dθ = 1).
    virtual double operator()(double theta, double f) const = 0;

    // Precompute discrete angular weights (for exporting spectra).
    virtual std::vector<double> weights(int M) const = 0;

    // Principal (mean) direction of travel [rad].
    virtual double principal_direction_rad() const = 0;

    // --- Realization interface ---
    // Default: no per-frequency randomization → all waves go principal dir.
    virtual std::vector<double> sample_directions(int N_freq) {
        return std::vector<double>(N_freq, principal_direction_rad());
    }
};

// Cosine-2s Distribution (default oceanographic spreading)
// Implements the commonly used directional spreading function in oceanography:
//
//   D(θ; θ₀, s) = C_s · cos^(2s) ((θ - θ₀)/2),   for |θ - θ₀| ≤ π
//
// where θ₀ is the mean (principal) direction and s ≥ 0 is the spreading parameter.
// - Normalization constant: C_s = Γ(s+0.5) / (√π Γ(s+1))
// - Larger s → narrower distribution (s → ∞ collapses to a delta at θ₀).
// - Typical ocean values: s ≈ 2–10.
class Cosine2sRandomizedDistribution : public DirectionalDistribution {
public:
    Cosine2sRandomizedDistribution(double mean_dir_rad, double s, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s_(s), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        // Normalization factor: Γ(s+0.5)/(√π Γ(s+1))
        double norm = std::tgamma(s_ + 0.5) / (std::sqrt(PI) * std::tgamma(s_ + 1.0));
        return norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta)), 2.0 * s_);
    }

    std::vector<double> weights(int M) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        double norm = std::tgamma(s_ + 0.5) / (std::sqrt(PI) * std::tgamma(s_ + 1.0));
        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            double dtheta_rel = theta - mean_dir_rad_;
            spread[m] = norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta_rel)), 2.0 * s_);
        }
        return spread;
    }

    double principal_direction_rad() const override {
        return mean_dir_rad_;
    }

    std::vector<double> sample_directions(int N_freq) override {
        // Monte Carlo sampling using rejection method
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        // Max density is at mean_dir, value = norm
        double max_val = operator()(mean_dir_rad_, 0.0);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            double y = u01(rng_) * max_val;
            if (y <= operator()(theta, 0.0)) {
                dirs.push_back(theta);
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;

    // Spreading parameter: larger s_ → narrower directional distribution
    // (s_ ≈ 2–10 typical in oceanography, s_ → ∞ gives a delta at mean_dir_rad_)
    double s_;
    mutable std::mt19937 rng_;
};

// Mitsuyasu-type spreading (Mitsuyasu et al. 1975)
class MitsuyasuDistribution : public DirectionalDistribution {
public:
    MitsuyasuDistribution(double mean_dir_rad, double s0, double f0, double m = 5.0, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s0_(s0), f0_(f0), m_(m), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double s_f = s0_ * std::pow(f / f0_, m_);
        double norm = std::tgamma(s_f + 0.5) / (std::sqrt(PI) * std::tgamma(s_f + 1.0));
        double dtheta = theta - mean_dir_rad_;
        return norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta)), 2.0 * s_f);
    }

    std::vector<double> weights(int M) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        double s_f = s0_;
        double norm = std::tgamma(s_f + 0.5) / (std::sqrt(PI) * std::tgamma(s_f + 1.0));
        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            double dtheta_rel = theta - mean_dir_rad_;
            spread[m] = norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta_rel)), 2.0 * s_f);
        }
        return spread;
    }

    double principal_direction_rad() const override {
        return mean_dir_rad_;
    }

    std::vector<double> sample_directions(int N_freq) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        double max_val = operator()(mean_dir_rad_, f0_); // peak value at mean direction

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            double y = u01(rng_) * max_val;
            if (y <= operator()(theta, f0_)) {
                dirs.push_back(theta);
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double s0_, f0_, m_;
    mutable std::mt19937 rng_;
};

// Donelan spreading (Donelan et al. 1985)
class DonelanDistribution : public DirectionalDistribution {
public:
    DonelanDistribution(double mean_dir_rad, double s0, double fp, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s0_(s0), fp_(fp), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double ratio = f / fp_;
        double s_f = (f < fp_) ? s0_ * std::pow(ratio, 2.0)
                               : s0_ * std::pow(ratio, -2.0);
        double norm = std::tgamma(s_f + 0.5) / (std::sqrt(PI) * std::tgamma(s_f + 1.0));
        double dtheta = theta - mean_dir_rad_;
        return norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta)), 2.0 * s_f);
    }

    std::vector<double> weights(int M) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        double s_f = s0_;
        double norm = std::tgamma(s_f + 0.5) / (std::sqrt(PI) * std::tgamma(s_f + 1.0));
        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            double dtheta_rel = theta - mean_dir_rad_;
            spread[m] = norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta_rel)), 2.0 * s_f);
        }
        return spread;
    }

    double principal_direction_rad() const override {
        return mean_dir_rad_;
    }

    std::vector<double> sample_directions(int N_freq) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        double max_val = operator()(mean_dir_rad_, fp_);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            double y = u01(rng_) * max_val;
            if (y <= operator()(theta, fp_)) {
                dirs.push_back(theta);
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double s0_, fp_;
    mutable std::mt19937 rng_;
};

// Sech² Distribution (Longuet-Higgins type)
class Sech2Distribution : public DirectionalDistribution {
public:
    Sech2Distribution(double mean_dir_rad, double beta, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), beta_(beta), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        double x = beta_ * dtheta;
        double val = 1.0 / std::cosh(x);
        return val * val; // not normalized perfectly, but standard form
    }

    std::vector<double> weights(int M) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            double dtheta_rel = theta - mean_dir_rad_;
            double x = beta_ * dtheta_rel;
            double val = 1.0 / std::cosh(x);
            spread[m] = val * val;
        }
        return spread;
    }

    double principal_direction_rad() const override {
        return mean_dir_rad_;
    }

    std::vector<double> sample_directions(int N_freq) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        double max_val = operator()(mean_dir_rad_, 0.0);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            double y = u01(rng_) * max_val;
            if (y <= operator()(theta, 0.0)) {
                dirs.push_back(theta);
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double beta_;
    mutable std::mt19937 rng_;
};

// Gaussian spreading
class GaussianDistribution : public DirectionalDistribution {
public:
    GaussianDistribution(double mean_dir_rad, double sigma, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), sigma_(sigma), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        double norm = 1.0 / (sigma_ * std::sqrt(2.0 * PI));
        return norm * std::exp(-0.5 * (dtheta / sigma_) * (dtheta / sigma_));
    }

    std::vector<double> weights(int M) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        double norm = 1.0 / (sigma_ * std::sqrt(2.0 * PI));
        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            double dtheta_rel = theta - mean_dir_rad_;
            spread[m] = norm * std::exp(-0.5 * (dtheta_rel / sigma_) * (dtheta_rel / sigma_));
        }
        return spread;
    }

    double principal_direction_rad() const override {
        return mean_dir_rad_;
    }

    std::vector<double> sample_directions(int N_freq) override {
        std::normal_distribution<double> normal(mean_dir_rad_, sigma_);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        for (int i = 0; i < N_freq; ++i) {
            double theta = normal(rng_);
            // wrap to [-π, π]
            theta = std::fmod(theta + PI, 2*PI);
            if (theta < 0) theta += 2*PI;
            dirs.push_back(theta - PI);
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double sigma_;
    mutable std::mt19937 rng_;
};

