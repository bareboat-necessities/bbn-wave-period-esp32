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

#ifdef PM_STOKES_TEST
#include <iostream>
#include <fstream>
#endif

#include "DirectionalSpread.h"

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
template<int N_FREQ = 256, int ORDER = 3>
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
                     unsigned int seed = 42u)
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
        renormalizeForStokesElevationVariance();
        computePerComponentStokesDriftEstimate();  // Lagrangian mean drift (2nd order)
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
    // Lagrangian mode with Stokes drift:
    //   - Uses Lagrangian particle kinematics (includes U_s drift in velocity).
    //   - Acceleration is oscillatory + gravity; mean drift is steady ⇒ no direct accel term.
    //   - Orientation from ORDER-consistent surface slopes (wave-following buoy).
    // By design, getIMUReadings() uses the Lagrangian surface case, consistent with
    // a buoy-mounted IMU.
    IMUReadingsBody getIMUReadings(double x, double y, double t, double z = 0.0,
                                   double dt = 1e-3) const {
        IMUReadingsBody imu;

        // Use actual sensor depth z for Lagrangian state (attenuates oscillations)
        auto state = computeWaveState(x, y, z, t, WaveFrame::Lagrangian);

        // Advected surface position for slope/orientation (buoy location)
        const double px = x + state.displacement.x();
        const double py = y + state.displacement.y();

        // Orientation from ORDER-consistent slopes at the *advected* position
        const auto slopes = getSurfaceSlopes(px, py, t);
        const Eigen::Matrix3d R_WI = orientationFromSlopes(slopes);

        // World gravity + particle acceleration → body-frame accelerometer
        const Eigen::Vector3d g_world(0, 0, -g_);

        // IMU specific force: f_body = R_WI * (a_world - g_world).
        imu.accel_body = R_WI * (state.acceleration - g_world);

        // Predict advected position at t+dt for gyro (1-step kinematic extrapolation)
        const double px_next = px + state.velocity.x() * dt;
        const double py_next = py + state.velocity.y() * dt;

        // Gyro from orientation finite-difference (body-at-t frame)
        auto slopes_next = getSurfaceSlopes(px_next, py_next, t + dt);
        Eigen::Matrix3d R1 = orientationFromSlopes(slopes);       // W->B at t
        Eigen::Matrix3d R2 = orientationFromSlopes(slopes_next);  // W->B at t+dt

        // Relative rotation from t to t+dt
        Eigen::Matrix3d Rdelta = R2 * R1.transpose();
        Eigen::AngleAxisd aa(Rdelta);

        // Angular velocity in IMU/body frame at time t
        imu.gyro_body = (aa.axis() * aa.angle()) / dt;

        return imu;
    }

    // Build local wave IMU orientation from slopes
    Eigen::Matrix3d orientationFromSlopes(const Eigen::Vector2d &slopes) const {
        Eigen::Vector3d n(-slopes.x(), -slopes.y(), 1.0);
        n.normalize();

        // project global X onto tangent plane for x-axis
        Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX();
        x_axis -= n * (x_axis.dot(n));
        if (x_axis.norm() < 1e-6) {
            x_axis = Eigen::Vector3d::UnitY(); // fallback
            x_axis -= n * (x_axis.dot(n));
        }
        x_axis.normalize();

        Eigen::Vector3d y_axis = n.cross(x_axis);
        y_axis.normalize();

        Eigen::Matrix3d R_WI; // world->IMU
        R_WI.row(0) = x_axis.transpose();
        R_WI.row(1) = y_axis.transpose();
        R_WI.row(2) = n.transpose();
        return R_WI;
    }

    // Accessors for spectral data (mirror Jonswap3dStokesWaves API)
    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { 
        return spectrum_.frequencies(); 
    }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum() const { 
        return spectrum_.spectrum(); 
    }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes() const { 
        return spectrum_.amplitudes(); 
    }
    const Eigen::Matrix<double, N_FREQ, 1>& df() const { 
        return spectrum_.df(); 
    }

    // Discrete directional spectrum, size N_FREQ × M
    // If normalize = true, weights are normalized so that ∑ D(θ; f) Δθ ≈ 1.
    Eigen::MatrixXd getDirectionalSpectrum(int M, bool normalize = true) const {
        Eigen::MatrixXd E(N_FREQ, M);
        for (int i = 0; i < N_FREQ; ++i) {
            double f = spectrum_.frequencies()(i);
            std::vector<double> weights = normalize
                ? directional_dist_->normalized_weights(M, f)
                : directional_dist_->weights(M, f);
            double S_f = spectrum_.spectrum()(i);
            for (int m = 0; m < M; ++m) {
                E(i, m) = S_f * weights[m];
            }
        }
        return E;
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
                // Depth decay for both Eulerian and Lagrangian (z ≤ 0)
                const double depthFactor = std::exp(n * k_val * z);

                // Displacement
                if (frame == WaveFrame::Lagrangian) {
                    state.displacement.x() += -cn * std::cos(arg) * dir_x_(i) * depthFactor;
                    state.displacement.y() += -cn * std::cos(arg) * dir_y_(i) * depthFactor;
                    state.displacement.z() +=  cn * std::sin(arg) * depthFactor;
                } else {
                    // Eulerian: only vertical displacement relevant
                    state.displacement.z() += cn * std::sin(arg) * depthFactor;
                }

                // Velocity
                // u,v = n ω cₙ sin(arg) dir;  w = n ω cₙ cos(arg)
                const double velFactor = n * w_val * cn * depthFactor;
                state.velocity.x() += velFactor * std::sin(arg) * dir_x_(i);
                state.velocity.y() += velFactor * std::sin(arg) * dir_y_(i);
                state.velocity.z() += velFactor * std::cos(arg);

                // Acceleration
                // aₓ,a_y = n² ω² cₙ cos(arg) dir;  a_z = −n² ω² cₙ sin(arg)
                const double accFactor = n * n * w_val * w_val * cn * depthFactor;
                state.acceleration.x() += accFactor * std::cos(arg) * dir_x_(i);
                state.acceleration.y() += accFactor * std::cos(arg) * dir_y_(i);
                state.acceleration.z() -= accFactor * std::sin(arg);
            }
        }

        // Add Lagrangian mean Stokes drift (2nd order, steady)
        // Vector form (deep water): U_s(z) = Σ [ (ω_i k_i a_i²) e^{2 k_i z} ] 𝚑𝚊𝚝{k}_i
        // This affects Lagrangian velocity (and integrated displacement), but not (instantaneous)
        // oscillatory acceleration; ∂/∂t of the mean is ~0.
        if (frame == WaveFrame::Lagrangian) {
            const Eigen::Array<double, N_FREQ, 1> exp2 = (2.0 * k_.array() * z).exp(); // z=0 → 1
            const double Usx = (stokes_drift_scalar_.array() * exp2 * dir_x_.array()).sum();
            const double Usy = (stokes_drift_scalar_.array() * exp2 * dir_y_.array()).sum();
            state.velocity.x() += Usx;
            state.velocity.y() += Usy;
            /// include mean-drift translation in displacement:
            state.displacement.x() += Usx * t;
            state.displacement.y() += Usy * t;
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

    // Deep-water Stokes drift estimate per component (monochromatic approximation)
    // U_s(z) = (ω k a²) e^{2 k z} 𝚑𝚊𝚝{k},  z ≤ 0.  Here we store U_s0 = ω k a².
    void computePerComponentStokesDriftEstimate() {
        stokes_drift_scalar_ = omega_.array() * k_.array() * A1_.array().square();
    }

    // Normalization of Hs
    void renormalizeForStokesElevationVariance() {
        if (ORDER <= 1) return; // nothing to do if only linear

        const double m0_target = (Hs_ * Hs_) / 16.0;

        auto m0_from_beta = [&](double beta) {
            double sum_sq = 0.0;
            for (int i = 0; i < N_FREQ; ++i) {
                const double a  = beta * A1_(i);
                const double ka = k_(i) * a;

                // n=1 term
                double cn1 = a;
                sum_sq += cn1 * cn1;

                // higher-order terms
                for (int n = 2; n <= ORDER; ++n) {
                    const double c = stokesCoeff(n);
                    const double cn = c * a * std::pow(ka, n - 1);
                    sum_sq += cn * cn;
                }
            }
            return 0.5 * sum_sq; // variance
        };

        const double m0_initial = m0_from_beta(1.0);
        if (m0_initial <= 0.0) return;

        if (m0_initial > m0_target * 1.0001) {
            // bisection search for scaling factor beta
            double lo = 0.0, hi = 1.0;
            for (int it = 0; it < 60; ++it) {
                double mid = 0.5 * (lo + hi);
                if (m0_from_beta(mid) > m0_target) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            double beta = 0.5 * (lo + hi);
            A1_ *= beta;
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

    // Per-component deep-water Stokes drift scalar (surface): U_s0,i = ω_i k_i a_i²
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
};

#ifdef PM_STOKES_TEST
// CSV generator for testing Pierson–Moskowitz Stokes waves
static void generateWavePMStokesCSV(const std::string& filename,
                                    double Hs, double Tp, double mean_dir_deg,
                                    double duration = 40.0, double dt = 0.005) {
    constexpr int N = 256;
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(mean_dir_deg * M_PI / 180.0, 10.0, 42u);
    PMStokesN3dWaves<N, 3> waveModel(Hs, Tp, dirDist, 0.02, 0.8, 9.81, 42u);

    const int N_time = static_cast<int>(duration / dt) + 1;
    Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N_time, 0.0, duration);

    Eigen::ArrayXXd disp(3, N_time), vel(3, N_time), acc(3, N_time);
    Eigen::ArrayXXd accel_body(3, N_time), gyro_body(3, N_time);
    Eigen::ArrayXXd euler_deg(3, N_time); // roll, pitch, yaw (yaw = 0)

    for (int i = 0; i < N_time; ++i) {
        double t = time(i);

        // global kinematics (Lagrangian)
        auto state = waveModel.getLagrangianState(t);

        // IMU
        auto imu = waveModel.getIMUReadings(0.0, 0.0, t);

        // store
        for (int j = 0; j < 3; ++j) {
            disp(j, i)       = state.displacement(j);
            vel(j, i)        = state.velocity(j);
            acc(j, i)        = state.acceleration(j);
            accel_body(j, i) = imu.accel_body(j);
            gyro_body(j, i)  = imu.gyro_body(j);
        }

        // slopes → roll/pitch
        auto slopes = waveModel.getSurfaceSlopes(0.0, 0.0, t);
        double slope_x = slopes.x();
        double slope_y = slopes.y();

        double roll  = std::atan2(slope_y, 1.0) * 180.0 / M_PI;
        double pitch = std::atan2(-slope_x, 1.0) * 180.0 / M_PI;
        double yaw   = 0.0;

        euler_deg(0, i) = roll;
        euler_deg(1, i) = pitch;
        euler_deg(2, i) = yaw;
    }

    // CSV output aligned with Jonswap3dStokesWaves
    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,"
         << "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,roll_deg,pitch_deg,yaw_deg\n";
  
    for (int i = 0; i < N_time; ++i) {
        file << time(i) << ","
             << disp(0, i) << "," << disp(1, i) << "," << disp(2, i) << ","
             << vel(0, i)  << "," << vel(1, i)  << "," << vel(2, i)  << ","
             << acc(0, i)  << "," << acc(1, i)  << "," << acc(2, i)  << ","
             << accel_body(0, i) << "," << accel_body(1, i) << "," << accel_body(2, i) << ","
             << gyro_body(0, i)  << "," << gyro_body(1, i)  << "," << gyro_body(2, i) << ","
             << euler_deg(0, i) << "," << euler_deg(1, i) << "," << euler_deg(2, i) << "\n";
    }
}

// Export PM directional spectrum to CSV, aligned with Jonswap3dStokesWaves
static void generatePMStokesDirSpectrumCSV(const std::string& filename,
                                           double Hs, double Tp,
                                           double mean_dir_deg = 0.0,
                                           int N_freq = 256, int N_theta = 72) {
    auto dirDist = std::make_shared<Cosine2sRandomizedDistribution>(mean_dir_deg * M_PI / 180.0, 10.0, 42u);
    PMStokesN3dWaves<256, 3> waveModel(Hs, Tp, dirDist, 0.02, 0.8, 9.81, 42u);

    auto freqs = waveModel.frequencies();
    Eigen::MatrixXd E = waveModel.getDirectionalSpectrum(N_theta);

    std::ofstream file(filename);
    file << "f_Hz,theta_deg,E\n";

    const double dtheta = 360.0 / N_theta;
    for (int i = 0; i < N_freq; ++i) {
        for (int m = 0; m < N_theta; ++m) {
            double theta_deg = -180.0 + m * dtheta;
            file << freqs(i) << "," << theta_deg << "," << E(i, m) << "\n";
        }
    }
}

// Batch generator for directional spectra
static void PMStokes_testWaveSpectrum() {
    generatePMStokesDirSpectrumCSV("short_pms_spectrum.csv",  0.5,  3.0, 30.0);
    generatePMStokesDirSpectrumCSV("medium_pms_spectrum.csv", 2.0,  7.0, 30.0);
    generatePMStokesDirSpectrumCSV("long_pms_spectrum.csv",   7.4, 14.3, 30.0);
}

// Batch generator
static void PMStokes_testWavePatterns() {
    generateWavePMStokesCSV("short_pms_waves.csv",  0.5,  3.0, 30.0);
    generateWavePMStokesCSV("medium_pms_waves.csv", 2.0,  7.0, 30.0);
    generateWavePMStokesCSV("long_pms_waves.csv",   7.4, 14.3, 30.0);
}
#endif
