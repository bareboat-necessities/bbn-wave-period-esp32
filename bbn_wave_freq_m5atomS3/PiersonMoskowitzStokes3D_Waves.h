#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <vector>

/*
  - Stochastic linear combination of Stokes-N harmonics (deep water) driven
    by a Pierson–Moskowitz spectrum.
  - ORDER: 1..5 (number of Stokes terms)
  - WaveState reports particle kinematics in global Cartesian coordinates:
      displacement = [x, y, z]  (meters)
      velocity     = [u, v, w]  (m/s)
      acceleration = [ax, ay, az] (m/s^2)
    Horizontal and vertical components included.
  - Eulerian fields at depth z <= 0: harmonics multiplied by exp(k z)

  Copyright 2025, Mikhail Grushinskiy
*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct IMUReadingsBody {
    Eigen::Vector3d accel_body;  // linear acceleration in IMU frame
    Eigen::Vector3d gyro_body;   // angular velocity in IMU frame (rad/s)
};

template<int N_FREQ = 256>
class EIGEN_ALIGN_MAX PiersonMoskowitzSpectrum {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PiersonMoskowitzSpectrum(double Hs, double Tp,
                             double f_min = 0.02, double f_max = 0.8,
                             double g = 9.81)
        : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max), g_(g)
    {
        if (N_FREQ < 2) throw std::runtime_error("N_FREQ must be >= 2");
        if (!(Hs_ > 0.0)) throw std::runtime_error("Hs must be > 0");
        if (!(Tp_ > 0.0)) throw std::runtime_error("Tp must be > 0");
        if (!(f_min_ > 0.0) || !(f_max_ > f_min_)) throw std::runtime_error("Invalid frequency range");
        if (!((1.0 / Tp_) >= f_min_ && (1.0 / Tp_) <= f_max_))
            throw std::runtime_error("1/Tp must be within [f_min, f_max]");

        frequencies_.setZero(); S_.setZero(); A_.setZero(); df_.setZero();

        computeLogFrequencySpacing();
        computeFrequencyIncrements();
        computePMSpectrumFromHs();
    }

    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum()    const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes()  const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df()          const { return df_; }

    double integratedVariance() const { return (S_.cwiseProduct(df_)).sum(); }

private:
    double Hs_, Tp_, f_min_, f_max_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
        double log_f_min = std::log(f_min_);
        double log_f_max = std::log(f_max_);
        for (int i = 0; i < N_FREQ; ++i) {
            double u = (N_FREQ == 1) ? 0.0 : double(i) / double(N_FREQ - 1);
            frequencies_(i) = std::exp(log_f_min + (log_f_max - log_f_min) * u);
        }
    }

    void computeFrequencyIncrements() {
        if (N_FREQ < 2) { df_.setZero(); return; }
        df_(0) = std::max(1e-12, frequencies_(1) - frequencies_(0));
        for (int i = 1; i < N_FREQ - 1; ++i)
            df_(i) = std::max(1e-12, 0.5 * (frequencies_(i + 1) - frequencies_(i - 1)));
        df_(N_FREQ - 1) = std::max(1e-12, frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2));
    }

    void computePMSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        Eigen::Matrix<double, N_FREQ, 1> S0;

        for (int i = 0; i < N_FREQ; ++i) {
            const double f = frequencies_(i);
            const double base = (g_ * g_) / std::pow(2.0 * M_PI, 4.0);
            S0(i) = base * std::pow(f, -5.0) * std::exp(-1.25 * std::pow(fp / f, 4.0));
        }

        const double variance_unit   = (S0.cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("PM: zero/negative variance");

        const double variance_target = (Hs_ * Hs_) / 16.0;
        const double alpha           = variance_target / variance_unit;

        S_ = S0 * alpha;
        A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

        const double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (Hs_est <= 0.0) throw std::runtime_error("PM: Hs_est <= 0");
        const double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            A_ *= (Hs_ / Hs_est);
            for (int i = 0; i < N_FREQ; ++i) {
                const double dfi = df_(i) > 0.0 ? df_(i) : 1e-12;
                S_(i) = (A_(i) * A_(i)) / (2.0 * dfi);
            }
        }
    }
};

// PMStokesN3dWaves
//
// Stochastic linear combination of Stokes-N harmonics (deep water) driven
// by a Pierson–Moskowitz spectrum.
//
// ORDER: 1..5 (number of Stokes terms)
//
// WaveState reports particle kinematics in global Cartesian coordinates:
//   displacement = [x, y, z]  (meters)
//   velocity     = [u, v, w]  (m/s)
//   acceleration = [ax, ay, az] (m/s²)
//
// Horizontal and vertical components included.
// Eulerian fields at depth z ≤ 0: harmonics multiplied by exp(k z).
//
template<int N_FREQ = 256, int ORDER = 5>
class EIGEN_ALIGN_MAX PMStokesN3dWaves {
    static_assert(ORDER >= 1 && ORDER <= 5, "ORDER supported range is 1..5");

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct EIGEN_ALIGN_MAX WaveState {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d displacement;  // [x, y, z] meters
        Eigen::Vector3d velocity;      // [u, v, w] m/s
        Eigen::Vector3d acceleration;  // [ax, ay, az] m/s²
    };

    PMStokesN3dWaves(double Hs, double Tp,
                     std::shared_ptr<DirectionalDistribution> dirDist,
                     double f_min = 0.02,
                     double f_max = 0.8,
                     double g = 9.81,
                     unsigned int seed = 239u)
        : spectrum_(Hs, Tp, f_min, f_max, g),
          Hs_(Hs), Tp_(Tp), g_(g), seed_(seed),
          directional_dist_(std::move(dirDist))
    {
        if (!directional_dist_) {
            throw std::runtime_error("DirectionalDistribution must not be null");
        }

        frequencies_ = spectrum_.frequencies();
        A1_          = spectrum_.amplitudes();
        df_          = spectrum_.df();

        omega_ = 2.0 * M_PI * frequencies_;
        k_     = omega_.array().square() / g_;

        phi_.setZero();
        dir_x_.setZero(); dir_y_.setZero();
        kx_.setZero(); ky_.setZero();

        initializeRandomPhases();
        initializeDirectionsFromDistribution();
        computeWaveDirectionComponents();
        checkSteepness();
    }

    // Lagrangian particle at surface (x₀ = y₀ = 0)
    WaveState getLagrangianState(double t) const {
        return computeWaveState(0.0, 0.0, 0.0, t, WaveFrame::Lagrangian);
    }

    // Eulerian field at (x, y, z)
    WaveState getEulerianState(double x, double y, double z, double t) const {
        return computeWaveState(x, y, z, t, WaveFrame::Eulerian);
    }

    // Surface slopes (∂η/∂x, ∂η/∂y) at z = 0, including up to ORDER terms
    Eigen::Vector2d getSurfaceSlopes(double x, double y, double t) const {
        double slope_x = 0.0;
        double slope_y = 0.0;

        for (int i = 0; i < N_FREQ; ++i) {
            const double k_val = k_(i);
            const double w_val = omega_(i);
            const double ka    = k_val * A1_(i);

            // θᵢ(x,y,t) = kₓᵢ x + k_yᵢ y − ωᵢ t + φᵢ
            const double theta = kx_(i) * x + ky_(i) * y - w_val * t + phi_(i);

            for (int n = 1; n <= ORDER; ++n) {
                // cₙ = coeff[n] · A₁ · (k a)^(n−1)
                const double cn  = stokesCoeff(n) * A1_(i) * std::pow(ka, n - 1);
                const double arg = n * theta;

                // ∂η/∂x = Σ cₙ · n kₓᵢ cos(arg)
                // ∂η/∂y = Σ cₙ · n k_yᵢ cos(arg)
                slope_x += n * cn * kx_(i) * std::cos(arg);
                slope_y += n * cn * ky_(i) * std::cos(arg);
            }
        }

        return Eigen::Vector2d(slope_x, slope_y);
    }

    // IMU readings at (x,y,t,z), consistent with ORDER Stokes slopes
    IMUReadingsBody getIMUReadings(double x, double y, double t, double z = 0.0,
                                   double dt = 1e-3) const {
        IMUReadingsBody imu;

        // --- accelerations ---
        auto state  = getEulerianState(x, y, z, t);
        auto slopes = getSurfaceSlopes(x, y, t);
        Eigen::Matrix3d R_WI = orientationFromSlopes(slopes);

        Eigen::Vector3d g_world(0, 0, -g_);
        imu.accel_body = R_WI * (state.acceleration + g_world);

        // --- gyro angular velocity ---
        auto slopes_next = getSurfaceSlopes(x, y, t + dt);
        Eigen::Matrix3d R1 = orientationFromSlopes(slopes);
        Eigen::Matrix3d R2 = orientationFromSlopes(slopes_next);

        Eigen::Matrix3d dR = (R2 - R1) / dt;
        Eigen::Matrix3d Omega = dR * R1.transpose();

        // vee map: skew-symmetric → vector
        imu.gyro_body = Eigen::Vector3d(Omega(2,1), Omega(0,2), Omega(1,0));
        return imu;
    }

    // Build local wave IMU orientation from slopes
    Eigen::Matrix3d orientationFromSlopes(const Eigen::Vector2d &slopes) const {
        Eigen::Vector3d n(-slopes.x(), -slopes.y(), 1.0);
        n.normalize();

        // project global X onto tangent plane for x-axis
        Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX();
        x_axis -= n * (x_axis.dot(n));
        if (x_axis.norm() < 1e-6) x_axis = Eigen::Vector3d::UnitY(); // fallback
        x_axis.normalize();

        Eigen::Vector3d y_axis = n.cross(x_axis);

        Eigen::Matrix3d R_WI; // world->IMU
        R_WI.row(0) = x_axis.transpose();
        R_WI.row(1) = y_axis.transpose();
        R_WI.row(2) = n.transpose();
        return R_WI;
    }

private:
    enum class WaveFrame { Lagrangian, Eulerian };

    WaveState computeWaveState(double x, double y, double z, double t,
                               WaveFrame frame) const {
        WaveState state;
        state.displacement.setZero();
        state.velocity.setZero();
        state.acceleration.setZero();

        for (int i = 0; i < N_FREQ; ++i) {
            const double k_val = k_(i);
            const double w_val = omega_(i);
            const double ka    = k_val * A1_(i);

            // θᵢ(x,y,t) = kₓᵢ x + k_yᵢ y − ωᵢ t + φᵢ
            const double theta = kx_(i) * x + ky_(i) * y - w_val * t + phi_(i);

            for (int n = 1; n <= ORDER; ++n) {
                // Stokes coefficient: cₙ = ctable[n] · A₁ · (k a)^(n−1)
                const double cn = stokesCoeff(n) * A1_(i) * std::pow(ka, n - 1);
                const double arg = n * theta;
                const double depthFactor = (frame == WaveFrame::Eulerian)
                    ? std::exp(n * k_val * z) : 1.0;

                // --- Displacement ---
                if (frame == WaveFrame::Lagrangian) {
                    state.displacement.x() += -cn * std::cos(arg) * dir_x_(i);
                    state.displacement.y() += -cn * std::cos(arg) * dir_y_(i);
                    state.displacement.z() +=  cn * std::sin(arg);
                } else {
                    // Eulerian: only vertical displacement relevant
                    state.displacement.z() += cn * std::sin(arg) * depthFactor;
                }

                // --- Velocity ---
                // u,v = n ω cₙ sin(arg) dir;  w = n ω cₙ cos(arg)
                const double velFactor = n * w_val * cn * depthFactor;
                state.velocity.x() += velFactor * std::sin(arg) * dir_x_(i);
                state.velocity.y() += velFactor * std::sin(arg) * dir_y_(i);
                state.velocity.z() += velFactor * std::cos(arg);

                // --- Acceleration ---
                // aₓ,a_y = n² ω² cₙ cos(arg) dir;  a_z = −n² ω² cₙ sin(arg)
                const double accFactor = n * n * w_val * w_val * cn * depthFactor;
                state.acceleration.x() += accFactor * std::cos(arg) * dir_x_(i);
                state.acceleration.y() += accFactor * std::cos(arg) * dir_y_(i);
                state.acceleration.z() -= accFactor * std::sin(arg);
            }
        }

        return state;
    }

    // Stokes coefficients for expansion order n
    //
    // c₁ = 1
    // c₂ = 1/2
    // c₃ = 3/8
    // c₄ = 1/3
    // c₅ = 125/384
    //
    static double stokesCoeff(int n) {
        static const double c[6] = {0.0, 1.0, 0.5, 3.0/8.0, 1.0/3.0, 125.0/384.0};
        return (n >= 1 && n <= 5) ? c[n] : 0.0;
    }

    void initializeRandomPhases() {
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
        for (int i = 0; i < N_FREQ; ++i) phi_(i) = dist(gen);
    }

    // Initialize directions from active directional distribution
    void initializeDirectionsFromDistribution() {
        auto dirs = directional_dist_->sample_directions_for_frequencies(
            std::vector<double>(spectrum_.frequencies().data(),
                                spectrum_.frequencies().data() + N_FREQ));
        for (int i = 0; i < N_FREQ; ++i) {
            dir_x_(i) = std::cos(dirs[i]);
            dir_y_(i) = std::sin(dirs[i]);
        }
    }

    void computeWaveDirectionComponents() {
        for (int i = 0; i < N_FREQ; ++i) {
            kx_(i) = k_(i) * dir_x_(i);
            ky_(i) = k_(i) * dir_y_(i);
        }
    }

    void checkSteepness() const {
        // max steepness k a ≤ ~0.20 for validity of Stokes expansion
        const double max_ka = (A1_.array() * k_.array()).maxCoeff();
        if (max_ka > 0.20)
            throw std::runtime_error("Stokes: max steepness k·a exceeds ~0.20 (validity warning)");
    }

    // Spectrum
    PiersonMoskowitzSpectrum<N_FREQ> spectrum_;

    // Parameters
    double Hs_, Tp_, g_;
    unsigned int seed_;

    // Directional distribution
    std::shared_ptr<DirectionalDistribution> directional_dist_;

    // Arrays
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, omega_, k_, A1_, phi_, df_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
};

#ifdef PM_STOKES_TEST
#include <fstream>
#include <string>

// CSV generator for testing Pierson–Moskowitz Stokes waves
static void generateWavePMStokesCSV(const std::string& filename,
                                    double Hs, double Tp, double mean_dir_deg,
                                    double duration = 40.0, double dt = 0.005) {
    // Use a cosine-2s directional distribution by default
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(
        mean_dir_deg * M_PI / 180.0, 15.0, 239u);

    PMStokesN3dWaves<256, 5> waveModel(
        Hs, Tp, dirDist, 0.02, 0.8, 9.81, 239u
    );

    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

    const double x0 = 0.0, y0 = 0.0;
    for (double t = 0.0; t <= duration; t += dt) {
        auto state = waveModel.getLagrangianState(t);
        file << t << ","
             << state.displacement.x() << ","
             << state.displacement.y() << ","
             << state.displacement.z() << ","
             << state.velocity.x()     << ","
             << state.velocity.y()     << ","
             << state.velocity.z()     << ","
             << state.acceleration.x() << ","
             << state.acceleration.y() << ","
             << state.acceleration.z() << "\n";
    }
}

// Batch generator for typical PM test cases
static void PMStokes_testWavePatterns() {
    generateWavePMStokesCSV("short_pms_waves.csv",  0.5,  3.0, 30.0);
    generateWavePMStokesCSV("medium_pms_waves.csv", 2.0,  7.0, 30.0);
    generateWavePMStokesCSV("long_pms_waves.csv",   7.4, 14.3, 30.0);
}
#endif
