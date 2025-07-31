#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Gerstner waves simulation. Surface
  
 */

template<int N_FREQ = 256>
class Jonswap3dGerstnerWaves {
public:
    struct WaveState {
        Eigen::Vector3d displacement;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
    };

    Jonswap3dGerstnerWaves(double Hs, double Tp,
                           double mean_direction_deg = 0.0,
                           double f_min = 0.2,
                           double f_max = 1.5,
                           double gamma = 3.3,
                           double g = 9.81,
                           double spreading_exponent = 25.0,
                           unsigned int seed = 42)
        : Hs_(Hs), Tp_(Tp), mean_dir_rad_(mean_direction_deg * M_PI / 180.0), gamma_(gamma), g_(g),
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
        initializeDirectionalSpreadRejection(); 
        computeWaveDirectionComponents();

        normalizeAmplitudeToMatchHs();
        checkSteepness();
    }

    WaveState getLagrangianState(double x0, double y0, double t) const {
        Eigen::Vector3d displacement = evaluateDisplacement(x0, y0, t);
        Eigen::Vector3d particle = Eigen::Vector3d(x0, y0, 0.0) + displacement;
        Eigen::Vector3d velocity = evaluateVelocity(x0, y0, t);
        Eigen::Vector3d acceleration = evaluateAcceleration(x0, y0, t);
        return {displacement, velocity, acceleration};
    }

    WaveState getEulerianState(double x, double y, double t) const {
        return {
            evaluateDisplacement(x, y, t),
            evaluateVelocity(x, y, t),
            evaluateAcceleration(x, y, t)
        };
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
    double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
    unsigned int seed_;

    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, S_, A_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;

    void computeLogFrequencySpacing(double f_min, double f_max) {
        double log_f_min = std::log(f_min);
        double log_f_max = std::log(f_max);
        for (int i = 0; i < N_FREQ; ++i)
            frequencies_(i) = std::exp(log_f_min + (log_f_max - log_f_min) * i / (N_FREQ - 1));
    }

    void computeFrequencyIncrements() {
        for (int i = 0; i < N_FREQ - 1; ++i)
            df_(i) = frequencies_(i + 1) - frequencies_(i);
        df_(N_FREQ - 1) = df_(N_FREQ - 2);
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

    void initializeDirectionalSpreadRejection() {
        std::mt19937 gen(seed_ + 1);
        std::uniform_real_distribution<double> u_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> y_dist(0.0, 1.0);

        for (int i = 0; i < N_FREQ; ++i) {
            double theta = 0.0;
            while (true) {
                double candidate = u_dist(gen);
                double base = std::cos(candidate - mean_dir_rad_);
                double pdf_val = std::pow(std::clamp(base, 0.0, 1.0), spreading_exponent_);
                double y = y_dist(gen);
                if (y <= pdf_val) {
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

    Eigen::Vector3d evaluateDisplacement(double x, double y, double t) const {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            double sin_th = std::sin(th);
            d[0] += A_(i) * cos_th * dir_x_(i);  // horizontal circular motion
            d[1] += A_(i) * cos_th * dir_y_(i);
            d[2] += A_(i) * sin_th;              // vertical motion
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
            v[0] += -A_(i) * w * sin_th * dir_x_(i);
            v[1] += -A_(i) * w * sin_th * dir_y_(i);
            v[2] +=  A_(i) * w * cos_th;
        }
        return v;
    }

    Eigen::Vector3d evaluateAcceleration(double x, double y, double t) const {
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        for (int i = 0; i < N_FREQ; ++i) {
            double w2 = omega_(i) * omega_(i);
            double th = kx_(i) * x + ky_(i) * y - omega_(i) * t + phi_(i);
            double cos_th = std::cos(th);
            double sin_th = std::sin(th);
            a[0] += -A_(i) * w2 * cos_th * dir_x_(i);
            a[1] += -A_(i) * w2 * cos_th * dir_y_(i);
            a[2] += -A_(i) * w2 * sin_th;
        }
        return a;
    }

    void checkSteepness() const {
        double max_steepness = (A_.array() * k_.array()).maxCoeff();
        if (max_steepness > 0.3)
            std::cerr << "[Warning] Wave steepness exceeds 0.3: " << max_steepness << "\n";
    }
};

#ifdef JONSWAP_TEST
void generateWaveJonswapCSV(const std::string& filename, 
                            double Hs, double Tp, double mean_dir_deg,
                            double duration = 40.0, double dt = 0.005) {
    // Initialize wave model with realistic parameters
    Jonswap3dGerstnerWaves<256> waveModel(
        Hs, Tp, mean_dir_deg, 0.2, 1.5, 3.1, 9.81, 25.0
    );

    // Open CSV file
    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

    // Simulate at fixed point (0,0)
    const double x0 = 0.0, y0 = 0.0;
    for (double t = 0; t <= duration; t += dt) {
        Jonswap3dGerstnerWaves<>::WaveState state = waveModel.getLagrangianState(x0, y0, t);
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
    // Short waves (choppy, wind-driven)
    generateWaveJonswapCSV("short_waves.csv", 0.5, 3.0, 30.0);
    // Medium waves (typical sea state)
    generateWaveJonswapCSV("medium_waves.csv", 2.0, 7.0, 30.0);
    // Long waves (swell)
    generateWaveJonswapCSV("long_waves.csv", 4.0, 12.0, 30.0);
}
#endif
