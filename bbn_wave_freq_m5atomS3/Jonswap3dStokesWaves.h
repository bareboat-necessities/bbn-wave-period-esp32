#pragma once
#pragma GCC optimize ("no-fast-math")

/*
  JONSWAP-spectrum 3D Stokes-corrected waves simulation (surface, deep-water).
  - 1st order: Airy (linear) components
  - 2nd order: simplified deep-water sum-frequency bound harmonics + simple
    estimate of surface Stokes drift (monochromatic approximation)

  Conventions:
  - z = 0 at the free surface; z < 0 is below the surface (into water).
  - First-order horizontal velocity signs are negative to match disp = -A cos(theta).
  - Second-order velocity/acceleration signs consistent with d/dt cos(theta - omega t).

*/

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
#include <limits>
#include <vector>
#include <memory>

#ifndef PI
static constexpr double PI = 3.14159265358979323846;
#else
static constexpr double PI = M_PI;
#endif

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

// Portable fast_sincos helper
inline void fast_sincos(double x, double &s, double &c) {
#if defined(__GNUC__) || defined(__clang__)
    #if defined(__GLIBC__) || defined(_GNU_SOURCE)
        ::sincos(x, &s, &c);
    #else
        s = std::sin(x);
        c = std::cos(x);
    #endif
#elif defined(_MSC_VER)
    double cs;
    _sincos(x, &s, &cs);
    c = cs;
#else
    s = std::sin(x);
    c = std::cos(x);
#endif
}

inline void robust_sincos(double theta, double omega, double t, double &s, double &c) {
    constexpr double LONG_WAVE_THRESHOLD = 1e-4;
    double arg = theta - omega * t;
    if (std::abs(omega) < LONG_WAVE_THRESHOLD) {
        s = std::sin(arg);
        c = std::cos(arg);
    } else {
        fast_sincos(arg, s, c);
    }
}

// ====================== JonswapSpectrum (vectorized) ======================
template<int N_FREQ = 128>
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

        computeLogFrequencySpacing();
        computeFrequencyIncrements();
        computeJonswapSpectrumFromHs();
    }

    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum()    const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes()  const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df()          const { return df_; }

    double integratedVariance() const {
        return (S_.cwiseProduct(df_)).sum();
    }

private:
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
        const double log_f_min = std::log(f_min_);
        const double log_f_max = std::log(f_max_);
        const double step = (log_f_max - log_f_min) / (N_FREQ - 1);
        Eigen::Array<double, N_FREQ, 1> idx = Eigen::Array<double, N_FREQ, 1>::LinSpaced(N_FREQ, 0.0, double(N_FREQ-1));
        frequencies_ = ((log_f_min + step * idx).exp()).matrix();
    }

    void computeFrequencyIncrements() {
        df_(0) = frequencies_(1) - frequencies_(0);
        df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2);
        if constexpr (N_FREQ > 2) {
            df_.segment(1, N_FREQ - 2) =
                0.5 * (frequencies_.segment(2, N_FREQ - 2) - frequencies_.segment(0, N_FREQ - 2));
        }
    }

    void computeJonswapSpectrumFromHs() {
        const double fp = 1.0 / Tp_;
        const Eigen::Array<double, N_FREQ, 1> f  = frequencies_.array();
        const Eigen::Array<double, N_FREQ, 1> f2 = f.square();
        const Eigen::Array<double, N_FREQ, 1> f4 = f2.square();
        const Eigen::Array<double, N_FREQ, 1> inv_f5 = 1.0 / (f * f4);

        Eigen::Array<double, N_FREQ, 1> sigma = Eigen::Array<double, N_FREQ, 1>::Constant(0.09);
        sigma = (f <= fp).select(0.07, sigma);

        const Eigen::Array<double, N_FREQ, 1> dfreq  = f - fp;
        const Eigen::Array<double, N_FREQ, 1> denom  = 2.0 * sigma.square() * fp * fp;
        const Eigen::Array<double, N_FREQ, 1> r      = (- (dfreq.square()) / denom).exp();

        const double g4_over_2pi4 = (g_*g_) / std::pow(2.0 * PI, 4);
        const Eigen::Array<double, N_FREQ, 1> ratio4 = ((fp*fp) / f2).square();
        const Eigen::Array<double, N_FREQ, 1> base   = g4_over_2pi4 * inv_f5 * (-1.25 * ratio4).exp();

        const Eigen::Array<double, N_FREQ, 1> gamma_r = (r * std::log(gamma_)).exp();
        Eigen::Array<double, N_FREQ, 1> S0 = base * gamma_r;

        double variance_unit = (S0.matrix().cwiseProduct(df_)).sum();
        if (!(variance_unit > 0.0)) throw std::runtime_error("JonswapSpectrum: zero/negative variance");

        const double variance_target = (Hs_ * Hs_) / 16.0;
        const double alpha = variance_target / variance_unit;

        S_ = (S0 * alpha).matrix();
        A_ = (2.0 * S_.array() * df_.array()).sqrt().matrix();

        const double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
        if (!(Hs_est > 0.0)) throw std::runtime_error("JonswapSpectrum: Hs_est <= 0");
        const double rel_err = std::abs(Hs_est - Hs_) / Hs_;
        if (rel_err > 1e-3) {
            A_ *= (Hs_ / Hs_est);
            S_ = (0.5 * A_.array().square() / df_.array()).matrix();
        }
    }
};

// =================== Jonswap3dStokesWaves (vectorized + tuned) ====================
template<int N_FREQ = 128>
class Jonswap3dStokesWaves {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct WaveState {
    Eigen::Vector3d displacement;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
  };

  Jonswap3dStokesWaves(double Hs, double Tp,
                       double mean_direction_deg = 0.0,
                       double f_min = 0.02, double f_max = 0.8,
                       double gamma = 2.0, double g = 9.81,
                       double spreading_exponent = 15.0,
                       unsigned int seed = 239u,
                       double cutoff_tol = 1e-8)
    : Hs_(Hs),
      Tp_(Tp),
      mean_dir_rad_(mean_direction_deg * PI / 180.0),
      gamma_(gamma),
      g_(g),
      spreading_exponent_(spreading_exponent),
      seed_(seed),
      cutoff_tol_(cutoff_tol),
      pairwise_size_(static_cast<size_t>(N_FREQ) * (N_FREQ + 1) / 2),
      spectrum_(Hs, Tp, f_min, f_max, gamma, g),
      exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN()),
      stokes_drift_mean_xy_cache_(0.0, 0.0),
      stokes_drift_mean_xy_cache_valid_(false),
      trig_last_t_(std::numeric_limits<double>::quiet_NaN())
  {
    // Spectrum data
    frequencies_ = spectrum_.frequencies();
    A_           = spectrum_.amplitudes();
    df_          = spectrum_.df();

    // Dispersion (deep water)
    omega_ = 2.0 * PI * frequencies_;
    k_     = omega_.array().square() / g_;

    // Direction & phase
    dir_x_.setZero(); dir_y_.setZero();
    kx_.setZero();    ky_.setZero();
    phi_.setZero();

    // First-order derived arrays
    Aomega_  = A_.array() * omega_.array();
    Aomega2_ = A_.array() * omega_.array().square();

    stokes_drift_scalar_.setZero();

    // Allocate per-pair arrays and caches
    allocatePairArrays();

    // Initialize content
    initializeRandomPhases();
    initializeDirectionalSpread();
    computeWaveDirectionComponents();
    computePerComponentStokesDriftEstimate();
    precomputePairwise();
    checkSteepness();
  }

  // Get displacement/velocity/acceleration at (x,y,z,t)
  // z = 0 at surface; z < 0 below surface
  WaveState getLagrangianState(double x, double y, double t, double z = 0.0) const {
    // ---- depth caches ----
    if (std::isnan(exp_kz_cached_z_) || exp_kz_cached_z_ != z) {
      exp_kz_       = (k_.array() * z).exp();
      exp_kz_pairs_ = (k_sum_ * z).exp();

      if (cutoff_tol_ > 0.0) {
        pair_mask_ = ((Bij_.abs() * exp_kz_pairs_) >= cutoff_tol_).select(1.0, 0.0);
      } else {
        pair_mask_.setOnes();
      }

      exp_kz_cached_z_ = z;
      stokes_drift_mean_xy_cache_valid_ = false; // depth-dependent
    }

    // ---- first-order ----
    const Eigen::Array<double, N_FREQ, 1> theta0 = kx_.array() * x + ky_.array() * y + phi_.array();
    const Eigen::Array<double, N_FREQ, 1> arg0   = theta0 - omega_.array() * t;
    const Eigen::Array<double, N_FREQ, 1> sin0   = arg0.sin() * exp_kz_;
    const Eigen::Array<double, N_FREQ, 1> cos0   = arg0.cos() * exp_kz_;

    Eigen::Vector3d disp = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel  = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc  = Eigen::Vector3d::Zero();

    // Displacement (1st)
    disp.x() -= (A_.array() * cos0 * dir_x_.array()).sum();
    disp.y() -= (A_.array() * cos0 * dir_y_.array()).sum();
    disp.z() += (A_.array() * sin0).sum();

    // Velocity (1st)
    vel.x()  -= (Aomega_  * sin0 * dir_x_.array()).sum();
    vel.y()  -= (Aomega_  * sin0 * dir_y_.array()).sum();
    vel.z()  -= (Aomega_  * cos0).sum();

    // Acceleration (1st)
    acc.x()  += (Aomega2_ * cos0 * dir_x_.array()).sum();
    acc.y()  += (Aomega2_ * cos0 * dir_y_.array()).sum();
    acc.z()  -= (Aomega2_ * sin0).sum();

    // ---- second-order theta cache ----
    if (std::isnan(x_cached_) || std::isnan(y_cached_) || x_cached_ != x || y_cached_ != y) {
      Eigen::Matrix<double, 2, 1> xy; xy << x, y;
      theta2_cache_ = (Ksum2_ * xy).array() + phi_sum_;
      x_cached_ = x; y_cached_ = y;
    }

    // ---- second-order trig cache ----
    if (std::isnan(trig_last_t_) || trig_last_t_ != t) {
      const Eigen::ArrayXd arg2 = theta2_cache_ - omega_sum_ * t;
      sin2_ = arg2.sin() * pair_mask_;
      cos2_ = arg2.cos() * pair_mask_;
      trig_last_t_ = t;
    }

    // ---- second-order contributions ----
    const Eigen::ArrayXd coeff = (factor_ * Bij_) * exp_kz_pairs_;
    const Eigen::ArrayXd C     = coeff * cos2_;
    const Eigen::ArrayXd S     = coeff * sin2_;

    const double dz2 = C.sum();
    const double dx2 = -(C * hx_).sum();
    const double dy2 = -(C * hy_).sum();

    const double vz2 = (S * omega_sum_).sum();
    const double vx2 = (S * omega_sum_ * hx_).sum();
    const double vy2 = (S * omega_sum_ * hy_).sum();

    const double az2 = -(C * omega_sum2_).sum();
    const double ax2 = -(C * omega_sum2_ * hx_).sum();
    const double ay2 = -(C * omega_sum2_ * hy_).sum();

    // Accumulate 2nd order
    disp.x() += dx2; disp.y() += dy2; disp.z() += dz2;
    vel.x()  += vx2; vel.y()  += vy2; vel.z()  += vz2;
    acc.x()  += ax2; acc.y()  += ay2; acc.z()  += az2;

    // ---- Stokes drift (depth-dependent, cached per z) ----
    if (!stokes_drift_mean_xy_cache_valid_) {
      const Eigen::Array<double, N_FREQ, 1> exp2 = exp_kz_ * exp_kz_;
      const Eigen::Array<double, N_FREQ, 1> Us_z = stokes_drift_scalar_.array() * exp2;
      stokes_drift_mean_xy_cache_.x() = (Us_z * dir_x_.array()).sum();
      stokes_drift_mean_xy_cache_.y() = (Us_z * dir_y_.array()).sum();
      stokes_drift_mean_xy_cache_valid_ = true;
    }
    vel.x() += stokes_drift_mean_xy_cache_.x();
    vel.y() += stokes_drift_mean_xy_cache_.y();

    return {disp, vel, acc};
  }

private:
  // ===================== members =====================

  // Inputs / parameters
  JonswapSpectrum<N_FREQ> spectrum_;
  double Hs_, Tp_, mean_dir_rad_, gamma_, g_, spreading_exponent_;
  unsigned int seed_;
  double cutoff_tol_;
  size_t pairwise_size_;

  // Per-frequency (size N_FREQ)
  Eigen::Matrix<double, N_FREQ, 1> frequencies_;
  Eigen::Matrix<double, N_FREQ, 1> A_, df_;
  Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
  Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
  Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
  Eigen::Array<double, N_FREQ, 1>  Aomega_, Aomega2_;

  // Per-pair (size pairwise_size_)
  Eigen::ArrayXd Bij_, kx_sum_, ky_sum_, k_sum_;
  Eigen::ArrayXd omega_sum_, omega_sum2_, phi_sum_, factor_;
  Eigen::ArrayXd hx_, hy_;
  Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::ColMajor> Ksum2_;

  // Caches (mutable)
  mutable Eigen::Array<double, N_FREQ, 1> exp_kz_;     // e^{k z}
  mutable Eigen::ArrayXd exp_kz_pairs_;                 // e^{(k_i+k_j) z}
  mutable Eigen::ArrayXd theta2_cache_;                 // kx_sum*x + ky_sum*y + phi_sum
  mutable Eigen::ArrayXd sin2_, cos2_;                  // sin/cos(theta2 - wsum t)
  mutable Eigen::ArrayXd pair_mask_;                    // numeric {0,1}

  mutable Eigen::Vector2d stokes_drift_mean_xy_cache_;
  mutable bool   stokes_drift_mean_xy_cache_valid_;

  mutable double exp_kz_cached_z_;
  mutable double x_cached_, y_cached_;
  mutable double trig_last_t_;

  // ===================== helpers =====================

  void allocatePairArrays() {
    const ptrdiff_t P = static_cast<ptrdiff_t>(pairwise_size_);
    Bij_.resize(P);
    kx_sum_.resize(P);
    ky_sum_.resize(P);
    k_sum_.resize(P);
    omega_sum_.resize(P);
    omega_sum2_.resize(P);
    phi_sum_.resize(P);
    factor_.resize(P);
    hx_.resize(P);
    hy_.resize(P);
    Ksum2_.resize(P, 2);

    exp_kz_pairs_.resize(P);
    theta2_cache_.resize(P);
    sin2_.resize(P);
    cos2_.resize(P);
    pair_mask_.resize(P);

    // Initialize to zero for safety
    Bij_.setZero(); kx_sum_.setZero(); ky_sum_.setZero(); k_sum_.setZero();
    omega_sum_.setZero(); omega_sum2_.setZero();
    phi_sum_.setZero(); factor_.setZero();
    hx_.setZero(); hy_.setZero(); Ksum2_.setZero();

    exp_kz_pairs_.setZero(); theta2_cache_.setZero();
    sin2_.setZero(); cos2_.setZero(); pair_mask_.setZero();

    exp_kz_.setZero();
    x_cached_ = y_cached_ = std::numeric_limits<double>::quiet_NaN();
  }

  void initializeRandomPhases() {
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<double> dist(0.0, 2.0 * PI);

    for (int i = 0; i < N_FREQ / 2; ++i) {
      const double ph = dist(rng);
      phi_(i) = ph;
      phi_(N_FREQ - 1 - i) = -ph;
    }
    if (N_FREQ % 2 == 1) {
      phi_(N_FREQ / 2) = dist(rng);
    }
  }

  void initializeDirectionalSpread() {
    std::mt19937 rng_dir(seed_ + 1234567u);
    std::uniform_real_distribution<double> u01(-1.0, 1.0);

    const Eigen::Array<double, N_FREQ, 1> amplitude_ratio = A_.array() / A_.maxCoeff();
    const Eigen::Array<double, N_FREQ, 1> spread_scale    = amplitude_ratio.pow(spreading_exponent_);
    constexpr double max_spread = PI * 0.5;

    for (int i = 0; i < N_FREQ / 2; ++i) {
      const double dev   = u01(rng_dir);
      const double delta = dev * (spread_scale(i) * max_spread);
      const double angle = mean_dir_rad_ + delta;

      dir_x_(i) = std::cos(angle);
      dir_y_(i) = std::sin(angle);

      const int j = N_FREQ - 1 - i;
      const double angle_j = angle + PI;
      dir_x_(j) = std::cos(angle_j);
      dir_y_(j) = std::sin(angle_j);
    }

    if (N_FREQ % 2 == 1) {
      const int mid = N_FREQ / 2;
      const double dev   = u01(rng_dir);
      const double delta = dev * (spread_scale(mid) * max_spread);
      const double angle = mean_dir_rad_ + delta;
      dir_x_(mid) = std::cos(angle);
      dir_y_(mid) = std::sin(angle);
    }

    // Normalize direction vectors
    for (int i = 0; i < N_FREQ; ++i) {
      const double norm = std::hypot(dir_x_(i), dir_y_(i));
      if (norm > 0.0) {
        dir_x_(i) /= norm;
        dir_y_(i) /= norm;
      } else {
        dir_x_(i) = std::cos(mean_dir_rad_);
        dir_y_(i) = std::sin(mean_dir_rad_);
      }
    }
  }

  void computeWaveDirectionComponents() {
    for (int i = 0; i < N_FREQ; ++i) {
      kx_(i) = k_(i) * dir_x_(i);
      ky_(i) = k_(i) * dir_y_(i);
    }
  }

  void computePerComponentStokesDriftEstimate() {
    // Deep-water Stokes drift at surface: U_s0 = omega * k * a^2
    for (int i = 0; i < N_FREQ; ++i) {
      stokes_drift_scalar_(i) = omega_(i) * k_(i) * A_(i) * A_(i);
    }
  }

  void precomputePairwise() {
    constexpr double tiny = 1e-18;
    ptrdiff_t idx = 0;

    for (int i = 0; i < N_FREQ; ++i) {
      const double ki = k_(i);
      for (int j = i; j < N_FREQ; ++j) {
        const double kj   = k_(j);
        const double ksum = ki + kj;
        const double kij  = ki * kj;

        // Simplified deep-water T^+ kernel
        const double T_plus = (ksum > tiny) ? (kij / ksum) : 0.0;

        const double kxsum  = kx_(i) + kx_(j);
        const double kysum  = ky_(i) + ky_(j);
        const double wsum   = omega_(i) + omega_(j);

        Bij_(idx)        = T_plus * A_(i) * A_(j);
        kx_sum_(idx)     = kxsum;
        ky_sum_(idx)     = kysum;
        k_sum_(idx)      = ksum;
        omega_sum_(idx)  = wsum;
        omega_sum2_(idx) = wsum * wsum;
        phi_sum_(idx)    = phi_(i) + phi_(j);
        factor_(idx)     = (i == j) ? 1.0 : 2.0;

        if (ksum > tiny) {
          hx_(idx) = kxsum / ksum;
          hy_(idx) = kysum / ksum;
        } else {
          hx_(idx) = 0.0;
          hy_(idx) = 0.0;
        }

        // Fused matrix for theta2
        Ksum2_(idx, 0) = kxsum;
        Ksum2_(idx, 1) = kysum;

        ++idx;
      }
    }
  }

  void checkSteepness() {
    const double max_steepness = (k_.array() * A_.array()).maxCoeff();
    if (max_steepness > 0.4) {
      throw std::runtime_error("Jonswap3dStokesWaves: wave too steep (>0.4), unstable");
    }
  }
};

#ifdef JONSWAP_TEST
static void generateWaveJonswapCSV(const std::string& filename,
                                   double Hs, double Tp, double mean_dir_deg,
                                   double duration = 40.0, double dt = 0.005) {
    constexpr int N_FREQ = 128;
    auto waveModel = std::make_unique<Jonswap3dStokesWaves<N_FREQ>>(Hs, Tp, mean_dir_deg, 0.02, 0.8, 2.0, 9.81, 10.0);

    const int N_time = static_cast<int>(duration / dt) + 1;
    Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N_time, 0.0, duration);

    Eigen::ArrayXXd disp(3, N_time), vel(3, N_time), acc(3, N_time);

    for (int i = 0; i < N_time; ++i) {
        auto state = waveModel->getLagrangianState(0.0, 0.0, time(i));
        for (int j = 0; j < 3; ++j) {
            disp(j, i) = state.displacement(j);
            vel(j, i)  = state.velocity(j);
            acc(j, i)  = state.acceleration(j);
        }
    }

    std::ofstream file(filename);
    file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";
    for (int i = 0; i < N_time; ++i) {
        file << time(i) << ","
             << disp(0, i) << "," << disp(1, i) << "," << disp(2, i) << ","
             << vel(0, i)  << "," << vel(1, i)  << "," << vel(2, i)  << ","
             << acc(0, i)  << "," << acc(1, i)  << "," << acc(2, i)  << "\n";
    }
}

static void Jonswap_testWavePatterns() {
    generateWaveJonswapCSV("short_waves_stokes.csv", 0.5, 3.0, 30.0);
    generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0, 7.0, 30.0);
    generateWaveJonswapCSV("long_waves_stokes.csv", 4.0, 12.0, 30.0);
}
#endif
