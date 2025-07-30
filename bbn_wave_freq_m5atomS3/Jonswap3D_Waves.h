#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Gerstner waves simulation. Surface
  
 */

template<int N_FREQ = 512>
class Jonswap3dGerstnerWaves {
public:
    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    Jonswap3dGerstnerWaves(double Hs, double Tp,
                           double f_min = 0.05,
                           double f_max = 2.5,
                           double gamma = 3.3,
                           double g = 9.81,
                           double mean_direction_deg = 0.0,
                           double spreading_exponent = 10.0,
                           unsigned int seed = 42)
        : Hs_(Hs), Tp_(Tp), gamma_(gamma), g_(g),
          mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          spreading_exponent_(spreading_exponent),
          seed_(seed)
    {
        frequencies_.setZero(); omega_.setZero(); k_.setZero(); S_.setZero(); A_.setZero(); phi_.setZero(); 
        df_.setZero(); dir_x_.setZero(); dir_y_.setZero(); kx_.setZero(); ky_.setZero();

        computeLogFrequencySpacing(f_min, f_max);
        computeFrequencyIncrements();
        omega_ = 2.0 * M_PI * frequencies_;
        k_ = omega_.array().square() / g_;

        computeJonswapSpectrum();
        initializeRandomPhases();
        initializeDirectionalSpread();
        computeWaveDirectionComponents();

        normalizeAmplitudeToMatchHs();
        checkSteepness();
    }

    WaveState getLagrangianState(double x0, double y0, double t) const {
        Eigen::Vector3d displacement = evaluateDisplacement(x0, y0, t);
        Eigen::Vector3d particle = Eigen::Vector3d(x0, y0, 0.0) + displacement;
        Eigen::Vector3d velocity = evaluateVelocity(particle.x(), particle.y(), t);
        Eigen::Vector3d localAccel = evaluateLocalAcceleration(particle.x(), particle.y(), t);
        Eigen::Matrix3d velGrad = computeVelocityGradient(particle.x(), particle.y(), t);
        Eigen::Vector3d convective = velGrad.transpose() * velocity;

        return {displacement, velocity, localAccel + convective};
    }

    WaveState getEulerianState(double x, double y, double t) const {
        return {
            evaluateDisplacement(x, y, t),
            evaluateVelocity(x, y, t),
            evaluateLocalAcceleration(x, y, t)
        };
    }

    double getSurfaceElevation(double x, double y, double t) const {
        double eta = 0.0;
        for (int i = 0; i < N_FREQ; ++i)
            eta += A_(i) * std::sin(kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i));
        return eta;
    }

private:
    double Hs_, Tp_, gamma_, g_;
    double mean_dir_rad_, spreading_exponent_;
    unsigned int seed_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, S_, A_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;

    void computeLogFrequencySpacing(double f_min, double f_max) {
        for (int i = 0; i < N_FREQ; ++i)
            frequencies_(i) = f_min * std::pow(f_max / f_min, double(i) / (N_FREQ - 1));
    }

    void computeFrequencyIncrements() {
        for (int i = 1; i < N_FREQ - 1; ++i)
            df_(i) = 0.5 * (frequencies_(i + 1) - frequencies_(i - 1));
        df_(0) = frequencies_(1) - frequencies_(0);
        df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2);
    }

    void computeJonswapSpectrum() {
        double fp = 1.0 / Tp_;
        for (int i = 0; i < N_FREQ; ++i) {
            double f = frequencies_(i);
            double sigma = (f <= fp) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(f - fp, 2) / (2.0 * sigma * sigma * fp * fp));
            double alpha = 0.076 * std::pow(Hs_, 2) / std::pow(Tp_, 4);
            double val = alpha * std::pow(g_, 2) / std::pow(2 * M_PI, 4)
                       * std::pow(f, -5.0)
                       * std::exp(-1.25 * std::pow(fp / f, 4.0))
                       * std::pow(gamma_, r);
            S_(i) = val;
        }
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();
    }

    void normalizeAmplitudeToMatchHs() {
        double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est > 1e-6)
            A_ *= (Hs_ / Hs_est);
    }

    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
        for (int i = 0; i < N_FREQ; ++i)
            phi_(i) = dist(gen);
    }

    void initializeDirectionalSpread() {
        std::mt19937 gen(seed_ + 1);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N_FREQ; ++i) {
            double u = dist(gen);
            double theta = sampleDirectionalAngleCosine(u);
            dir_x_(i) = std::cos(theta);
            dir_y_(i) = std::sin(theta);
        }
    }

    double sampleDirectionalAngleCosine(double u) const {
        // Inverse CDF sampling for D(θ) ∝ cos^s(θ - θ_m)
        double theta_max = M_PI / 2.0;
        double exponent = 1.0 / (spreading_exponent_ + 1.0);
        double shifted = 2.0 * u - 1.0;
        shifted = std::clamp(shifted, -1.0, 1.0);
        double angle = std::asin(std::pow(std::abs(shifted), exponent)) * theta_max;
        return mean_dir_rad_ + (shifted < 0 ? -angle : angle);
    }

    void computeWaveDirectionComponents() {
        for (int i = 0; i < N_FREQ; ++i) {
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    Eigen::Vector3d evaluateDisplacement(double x, double y, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            double sin_th = std::sin(th);
            d[0] += -A_(i) * cos_th * dir_x_(i);
            d[1] += -A_(i) * cos_th * dir_y_(i);
            d[2] +=  A_(i) * sin_th;
        }
        return d;
    }

    Eigen::Vector3d evaluateVelocity(double x, double y, double t) const {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w = omega_(i);
            double th = kx_(i) * x + ky_(i) * y - w * t + phi_(i);
            double sin_th = std::sin(th);
            double cos_th = std::cos(th);
            v[0] += A_(i) * w * sin_th * dir_x_(i);
            v[1] += A_(i) * w * sin_th * dir_y_(i);
            v[2] += A_(i) * w * cos_th;
        }
        return v;
    }

    Eigen::Vector3d evaluateLocalAcceleration(double x, double y, double t) const {
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w2 = omega_(i) * omega_(i);
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            double sin_th = std::sin(th);
            a[0] += A_(i) * w2 * cos_th * dir_x_(i);
            a[1] += A_(i) * w2 * cos_th * dir_y_(i);
            a[2] += -A_(i) * w2 * sin_th;
        }
        return a;
    }

    Eigen::Matrix3d computeVelocityGradient(double x, double y, double t) const {
        Eigen::Matrix3d grad = Eigen::Matrix3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w = omega_(i);
            double th = kx_(i) * x + ky_(i) * y - w * t + phi_(i);
            double cos_th = std::cos(th), sin_th = std::sin(th);
            double A = A_(i), kx = kx_(i), ky = ky_(i);

            grad(0, 0) += A * w * cos_th * kx * dir_x_(i);
            grad(0, 1) += A * w * cos_th * ky * dir_x_(i);
            grad(1, 0) += A * w * cos_th * kx * dir_y_(i);
            grad(1, 1) += A * w * cos_th * ky * dir_y_(i);
            grad(2, 0) += -A * w * sin_th * kx;
            grad(2, 1) += -A * w * sin_th * ky;
        }
        return grad;
    }

    void checkSteepness() const {
        double max_steepness = (A_.array() * k_.array()).maxCoeff();
        if (max_steepness > 0.3)
            std::cerr << "[Warning] Wave steepness exceeds 0.3: " << max_steepness << "\n";
    }
};
