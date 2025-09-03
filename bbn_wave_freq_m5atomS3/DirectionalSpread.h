#pragma once
#pragma GCC optimize ("no-fast-math")

/*
  Directional Wave Spreading Distributions

  Implements several common directional spreading functions used in oceanography:
  - Cosine-2s (cos^(2s) law, frequency-independent)
  - Mitsuyasu (frequency-dependent cos^(2s(f)) law)
  - Donelan (frequency-dependent cos^(2s(f)) law, different scaling)
  - Sech² (hyperbolic secant squared)
  - Gaussian (wrapped normal approximation)

  Copyright: Mikhail Grushinskiy, 2025
*/

#include <random>
#include <cmath>
#include <vector>
#include <memory>
#include <numeric>
#include <utility>

#ifndef PI
static constexpr double PI = 3.14159265358979323846264338327950288;
#else
static constexpr double PI = M_PI;
#endif

// Base class for directional distributions
class DirectionalDistribution {
public:
    virtual ~DirectionalDistribution() = default;

    // Theoretical spectrum interface
    //
    // Evaluate continuous density D(θ; f), normalized so that:
    //   ∫ D(θ; f) dθ = 1   over [-π, π]
    //
    // θ in radians, f = frequency [Hz]
    virtual double operator()(double theta, double f) const = 0;

    // Return unnormalized discrete weights at frequency f
    virtual std::vector<double> weights(int M, double f) const = 0;

    // Return normalized weights (trapezoidal integration)
    // Ensures ∑ wᵢ Δθ ≈ 1
    std::vector<double> normalized_weights(int M, double f) const {
        std::vector<double> w = weights(M, f);
        const double dtheta = 2.0 * PI / M;
        normalize_weights(w, dtheta);
        return w;
    }

    // Return (θ, weight) pairs for direct export/plotting
    std::vector<std::pair<double,double>> angle_weight_pairs(int M, double f) const {
        const double dtheta = 2.0 * PI / M;
        auto w = normalized_weights(M, f);

        std::vector<std::pair<double,double>> result;
        result.reserve(M);

        for (int m = 0; m < M; ++m) {
            double theta = -PI + m * dtheta;
            result.emplace_back(theta, w[m]);
        }
        return result;
    }

    // Principal (mean) direction [rad]
    virtual double principal_direction_rad() const = 0;

    // Realization interface
    // Default: all samples go in principal direction
    virtual std::vector<double> sample_directions(int N_freq, double f) {
        return std::vector<double>(N_freq, principal_direction_rad());
    }

    // Convenience: sample one direction per frequency in a batch
    virtual std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) 
    {
        std::vector<double> dirs;
        dirs.reserve(freqs.size());
        for (double f : freqs) {
            auto d = sample_directions(1, f);
            dirs.push_back(d[0]);
        }
        return dirs;
    }

protected:
    // Normalize weights with trapezoidal rule:
    // ∫ D(θ) dθ ≈ Δθ [½w₀ + w₁ + … + wₙ₋₂ + ½wₙ₋₁]
    static void normalize_weights(std::vector<double>& w, double dtheta) {
        if (w.empty()) return;

        double sum = 0.0;
        sum += 0.5 * w.front();
        sum += 0.5 * w.back();
        for (size_t i = 1; i + 1 < w.size(); ++i) {
            sum += w[i];
        }
        sum *= dtheta;

        if (sum > 0.0) {
            for (auto &val : w) {
                val /= sum;
            }
        }
    }

    // Wrap an angle into [-π, π]
    static double wrap_to_pi(double theta) noexcept {
        theta = std::fmod(theta + PI, 2.0 * PI);
        if (theta < 0.0) theta += 2.0 * PI;
        return theta - PI;
    }

    // Numerically stable cos^(exp) for cosval ∈ [−1, 1]
    static double stable_pow_cos(double cosval, double exp) noexcept {
        if (cosval <= 0.0) return 0.0;
        return std::exp(exp * std::log(cosval));
    }
};

// Cosine-2s Distribution
//
// Formula:
//   D(θ; s) = Cₛ · cos^(2s)((θ − θ₀)/2)
//
// Normalization constant:
//   Cₛ = Γ(s+½) / (√π · Γ(s+1))
//
// Larger s → narrower directional spreading.
//
class Cosine2sRandomizedDistribution : public DirectionalDistribution {
public:
    Cosine2sRandomizedDistribution(double mean_dir_rad, double s, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s_(s), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        // Normalization constant:
        //   Cₛ = Γ(s+1) / (2 √π Γ(s+½))
        double norm = std::exp(std::lgamma(s_ + 1.0)
                             - std::lgamma(s_ + 0.5)
                             - 0.5 * std::log(PI)
                             - std::log(2.0));
        return norm * stable_pow_cos(std::cos(0.5 * dtheta), 2.0 * s_);
    }

    std::vector<double> weights(int M, double f) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m)
            spread[m] = operator()(-PI + m * dtheta, f);
        return spread;
    }

    double principal_direction_rad() const override { return mean_dir_rad_; }

    // Monte Carlo rejection sampling
    std::vector<double> sample_directions(int N_freq, double f) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);
        double max_val = operator()(mean_dir_rad_, f);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            if (u01(rng_) * max_val <= operator()(theta, f))
                dirs.push_back(theta);
        }
        return dirs;
    }

    std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) override 
    {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(freqs.size());

        double max_val = operator()(mean_dir_rad_, 0.0); // independent of f

        for (size_t i = 0; i < freqs.size(); ++i) {
            while (true) {
                double theta = angle(rng_);
                if (u01(rng_) * max_val <= operator()(theta, freqs[i])) {
                    dirs.push_back(theta);
                    break;
                }
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double s_;
    mutable std::mt19937 rng_;
};

// Mitsuyasu Distribution (1975)
//
// Frequency-dependent spreading exponent:
//   s(f) = s₀ (f / f₀)ᵐ
//
// With typical m ≈ 5.
//
class MitsuyasuDistribution : public DirectionalDistribution {
public:
    MitsuyasuDistribution(double mean_dir_rad, double s0, double f0, double m = 5.0, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s0_(s0), f0_(f0), m_(m), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double s_f = s0_ * std::pow(f / f0_, m_);
        double norm = std::exp(std::lgamma(s_f + 0.5)
                             - std::lgamma(s_f + 1.0)
                             - 0.5 * std::log(PI));
        double dtheta = theta - mean_dir_rad_;
        return norm * stable_pow_cos(std::cos(0.5 * dtheta), 2.0 * s_f);     
    }

    std::vector<double> weights(int M, double f) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m)
            spread[m] = operator()(-PI + m * dtheta, f);
        return spread;
    }

    double principal_direction_rad() const override { return mean_dir_rad_; }

    std::vector<double> sample_directions(int N_freq, double f) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);
        double max_val = operator()(mean_dir_rad_, f);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            if (u01(rng_) * max_val <= operator()(theta, f))
                dirs.push_back(theta);
        }
        return dirs;
    }

    std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) override 
    {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(freqs.size());

        for (double f : freqs) {
            // peak value of PDF at θ₀
            double max_val = operator()(mean_dir_rad_, f);

            // rejection sample one θ
            while (true) {
                double theta = angle(rng_);
                if (u01(rng_) * max_val <= operator()(theta, f)) {
                    dirs.push_back(theta);
                    break;
                }
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double s0_, f0_, m_;
    mutable std::mt19937 rng_;
};

// Donelan Distribution (1985)
//
// Frequency-dependent spreading exponent:
//   s(f) = s₀ (f / fₚ)²   for f < fₚ
//   s(f) = s₀ (f / fₚ)⁻²  for f ≥ fₚ
//
class DonelanDistribution : public DirectionalDistribution {
public:
    DonelanDistribution(double mean_dir_rad, double s0, double fp, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), s0_(s0), fp_(fp), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double ratio = f / fp_;
        double s_f = (f < fp_) ? s0_ * std::pow(ratio, 2.0)
                               : s0_ * std::pow(ratio, -2.0);
        double norm = std::exp(std::lgamma(s_f + 0.5)
                             - std::lgamma(s_f + 1.0)
                             - 0.5 * std::log(PI));
        double dtheta = theta - mean_dir_rad_;
        return norm * stable_pow_cos(std::cos(0.5 * dtheta), 2.0 * s_f);
    }    

    std::vector<double> weights(int M, double f) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m)
            spread[m] = operator()(-PI + m * dtheta, f);
        return spread;
    }

    double principal_direction_rad() const override { return mean_dir_rad_; }

    std::vector<double> sample_directions(int N_freq, double f) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);
        double max_val = operator()(mean_dir_rad_, f);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            if (u01(rng_) * max_val <= operator()(theta, f))
                dirs.push_back(theta);
        }
        return dirs;
    }

    std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) override 
    {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(freqs.size());

        for (double f : freqs) {
            double max_val = operator()(mean_dir_rad_, f); // peak at θ₀
            while (true) {
                double theta = angle(rng_);
                if (u01(rng_) * max_val <= operator()(theta, f)) {
                    dirs.push_back(theta);
                    break;
                }
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
//
// Formula:
//   D(θ) ∝ sech²(β (θ − θ₀))
//
class Sech2Distribution : public DirectionalDistribution {
public:
    Sech2Distribution(double mean_dir_rad, double beta, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), beta_(beta), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        double val = 1.0 / std::cosh(beta_ * dtheta);
        return 0.5 * beta_ * val * val;  // normalized form
    }

    std::vector<double> weights(int M, double f) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m)
            spread[m] = operator()(-PI + m * dtheta, f);
        return spread;
    }

    double principal_direction_rad() const override { return mean_dir_rad_; }

    std::vector<double> sample_directions(int N_freq, double f) override {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(N_freq);
        double max_val = operator()(mean_dir_rad_, f);

        while (dirs.size() < static_cast<size_t>(N_freq)) {
            double theta = angle(rng_);
            if (u01(rng_) * max_val <= operator()(theta, f))
                dirs.push_back(theta);
        }
        return dirs;
    }

    std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) override 
    {
        std::uniform_real_distribution<double> angle(-PI, PI);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        std::vector<double> dirs;
        dirs.reserve(freqs.size());

        // peak of PDF is always at θ₀
        double max_val = operator()(mean_dir_rad_, 0.0);

        for (double f : freqs) {
            (void)f; // frequency argument not used, but keeps interface consistent
            while (true) {
                double theta = angle(rng_);
                if (u01(rng_) * max_val <= operator()(theta, f)) {
                    dirs.push_back(theta);
                    break;
                }
            }
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double beta_;
    mutable std::mt19937 rng_;
};

// Gaussian Distribution
//
// Formula:
//   D(θ) = (1 / (σ √(2π))) · exp(−(θ − θ₀)² / (2σ²))
//
class GaussianDistribution : public DirectionalDistribution {
public:
    GaussianDistribution(double mean_dir_rad, double sigma, unsigned int seed = 1234)
        : mean_dir_rad_(mean_dir_rad), sigma_(sigma), rng_(seed) {}

    double operator()(double theta, double f) const override {
        double dtheta = theta - mean_dir_rad_;
        double norm = 1.0 / (sigma_ * std::sqrt(2.0 * PI));
        return norm * std::exp(-0.5 * (dtheta / sigma_)
                                      * (dtheta / sigma_));
    }

    std::vector<double> weights(int M, double f) const override {
        std::vector<double> spread(M);
        const double dtheta = 2.0 * PI / M;
        for (int m = 0; m < M; ++m)
            spread[m] = operator()(-PI + m * dtheta, f);
        return spread;
    }

    double principal_direction_rad() const override { return mean_dir_rad_; }

    std::vector<double> sample_directions(int N_freq, double f) override {
        std::normal_distribution<double> normal(mean_dir_rad_, sigma_);

        std::vector<double> dirs;
        dirs.reserve(N_freq);

        for (int i = 0; i < N_freq; ++i) {
            double theta = normal(rng_);
            // wrap into [−π, π]
            dirs.push_back(wrap_to_pi(theta));
        }
        return dirs;
    }

    std::vector<double> sample_directions_for_frequencies(
        const std::vector<double>& freqs) override 
    {
        std::normal_distribution<double> normal(mean_dir_rad_, sigma_);

        std::vector<double> dirs;
        dirs.reserve(freqs.size());

        for (size_t i = 0; i < freqs.size(); ++i) {
            double theta = normal(rng_);
            dirs.push_back(wrap_to_pi(theta));
        }
        return dirs;
    }

private:
    double mean_dir_rad_;
    double sigma_;
    mutable std::mt19937 rng_;
};
