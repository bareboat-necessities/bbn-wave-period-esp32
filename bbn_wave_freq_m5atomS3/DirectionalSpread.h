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
    double s_;
    mutable std::mt19937 rng_;
};

