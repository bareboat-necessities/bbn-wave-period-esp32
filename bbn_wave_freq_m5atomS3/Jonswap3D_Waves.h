#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D waves simulation. Surface
  
 */

template<int N_FREQ = 512>
class Jonswap3dWaves {
public:
    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    Jonswap3dWaves(double Hs, double Tp,
                   double f_min = 0.05,
                   double f_max = 2.5,
                   double gamma = 3.3,
                   double g = 9.81,
                   double mean_direction_deg = 0.0,
                   double spreading_exponent = 10.0)
        : Hs_(Hs), Tp_(Tp), gamma_(gamma), g_(g),
          mean_dir_rad_(mean_direction_deg * M_PI / 180.0),
          spreading_exponent_(spreading_exponent) {

        computeLogFrequencySpacing(f_min, f_max);
        computeFrequencyIncrements();
        omega_ = 2.0 * M_PI * frequencies_;
        k_ = omega_.array().square() / g_;  // deep water dispersion

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
            evaluateSurfaceElevation(x, y, t),
            evaluateVelocity(x, y, t),
            evaluateLocalAcceleration(x, y, t)
        };
    }

private:
    double Hs_, Tp_, gamma_, g_;
    double mean_dir_rad_, spreading_exponent_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, S_, A_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;

    void computeLogFrequencySpacing(double f_min, double f_max) {
        for (int i = 0; i < N_FREQ; ++i)
            frequencies_(i) = f_min * std::pow(f_max / f_min, double(i) / (N_FREQ - 1));
    }

    void computeFrequencyIncrements() {
        df_.setZero();
        for (int i = 1; i < N_FREQ - 1; ++i)
            df_(i) = 0.5 * (frequencies_(i+1) - frequencies_(i-1));
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
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
        for (int i = 0; i < N_FREQ; ++i)
            phi_(i) = dist(gen);
    }

    void initializeDirectionalSpread() {
        std::mt19937 gen(1337);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N_FREQ; ++i) {
            double theta = sampleCosineSpread(dist(gen));
            dir_x_(i) = std::cos(theta);
            dir_y_(i) = std::sin(theta);
        }
    }

    double sampleCosineSpread(double u) const {
        // Inverse CDF sampling from cos^n(θ - θ₀) using power of cosine
        double angle_range = M_PI;
        double angle = std::acos(std::pow(u, 1.0 / (spreading_exponent_ + 1.0))) * (u < 0.5 ? -1 : 1);
        return mean_dir_rad_ + angle;
    }

    void computeWaveDirectionComponents() {
        for (int i = 0; i < N_FREQ; ++i) {
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    double theta(int i, double x, double y, double t) const {
        return kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
    }

    Eigen::Vector3d evaluateDisplacement(double x, double y, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double th = theta(i, x, y, t);
            d[0] += -A_(i) * std::cos(th) * dir_x_(i);
            d[1] += -A_(i) * std::cos(th) * dir_y_(i);
            d[2] +=  A_(i) * std::sin(th);
        }
        return d;
    }

    Eigen::Vector3d evaluateSurfaceElevation(double x, double y, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i)
            d[2] += A_(i) * std::sin(theta(i, x, y, t));
        return d;
    }

    Eigen::Vector3d evaluateVelocity(double x, double y, double t) const {
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w = omega_(i);
            double th = theta(i, x, y, t);
            v[0] += A_(i) * w * std::sin(th) * dir_x_(i);
            v[1] += A_(i) * w * std::sin(th) * dir_y_(i);
            v[2] += A_(i) * w * std::cos(th);
        }
        return v;
    }

    Eigen::Vector3d evaluateLocalAcceleration(double x, double y, double t) const {
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w2 = omega_(i) * omega_(i);
            double th = theta(i, x, y, t);
            a[0] += A_(i) * w2 * std::cos(th) * dir_x_(i);
            a[1] += A_(i) * w2 * std::cos(th) * dir_y_(i);
            a[2] += -A_(i) * w2 * std::sin(th);
        }
        return a;
    }

    Eigen::Matrix3d computeVelocityGradient(double x, double y, double t) const {
        Eigen::Matrix3d grad = Eigen::Matrix3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w = omega_(i);
            double th = theta(i, x, y, t);
            double cos_th = std::cos(th), sin_th = std::sin(th);
            double A = A_(i), kx = kx_(i), ky = ky_(i);

            // ∂vx/∂x, ∂vx/∂y
            grad(0, 0) += A * w * cos_th * kx * dir_x_(i);
            grad(0, 1) += A * w * cos_th * ky * dir_x_(i);

            // ∂vy/∂x, ∂vy/∂y
            grad(1, 0) += A * w * cos_th * kx * dir_y_(i);
            grad(1, 1) += A * w * cos_th * ky * dir_y_(i);

            // ∂vz/∂x, ∂vz/∂y
            grad(2, 0) += -A * w * sin_th * kx;
            grad(2, 1) += -A * w * sin_th * ky;
        }
        return grad;
    }

    void checkSteepness() const {
        auto steepness = (A_.array() * k_.array()).maxCoeff();
        if (steepness > 0.3)
            std::cerr << "[Warning] Wave steepness exceeds 0.3: " << steepness << "\n";
    }
};
