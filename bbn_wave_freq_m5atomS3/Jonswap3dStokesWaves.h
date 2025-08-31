#pragma once
#pragma GCC optimize ("no-fast-math")

/*
  Copyright 2025, Mikhail Grushinskiy

  JONSWAP-spectrum 3D Stokes-corrected waves simulation (surface, deep-water).
  - 1st order: Airy (linear) components
  - 2nd order: simplified deep-water sum-frequency bound harmonics + simple
    estimate of surface Stokes drift (monochromatic approximation)

  Conventions:
  - z = 0 at the free surface; z < 0 is below the surface (into water).
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
#include <cassert>

#ifndef PI
static constexpr double PI = 3.14159265358979323846264338327950288;
#else
static constexpr double PI = M_PI;
#endif

#ifdef JONSWAP_TEST
#include <iostream>
#include <fstream>
#endif

// robust sincos
inline void fast_sincos(double x, double &s, double &c) {
#if defined(__GNUC__) || defined(__clang__)
# if defined(__GLIBC__) || defined(_GNU_SOURCE)
  ::sincos(x, &s, &c);
# else
  s = std::sin(x); c = std::cos(x);
# endif
#elif defined(_MSC_VER)
  _sincos(x, &s, &c);
#else
  s = std::sin(x); c = std::cos(x);
#endif
}

inline void robust_sincos(double theta, double omega, double t, double &s, double &c) {
  constexpr double LONG_WAVE_THRESHOLD = 1e-4; // small omega threshold
  const double arg = theta - omega * t;
  if (std::abs(omega) < LONG_WAVE_THRESHOLD) {
    s = std::sin(arg);
    c = std::cos(arg);
  }
  else {
    fast_sincos(arg, s, c);
  }
}

using VecD = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, Eigen::Dynamic, 1>;

// JonswapSpectrum
template<int N_FREQ = 128>
class JonswapSpectrum {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    JonswapSpectrum(double Hs, double Tp,
                    double f_min = 0.02, double f_max = 0.8,
                    double gamma = 2.0, double g = 9.81)
      : Hs_(Hs), Tp_(Tp), f_min_(f_min), f_max_(f_max), gamma_(gamma), g_(g)
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
      computeJonswapSpectrumFromHs();
    }

    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return frequencies_; }
    const Eigen::Matrix<double, N_FREQ, 1>& spectrum()    const { return S_; }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes()  const { return A_; }
    const Eigen::Matrix<double, N_FREQ, 1>& df()          const { return df_; }
    double integratedVariance() const { return (S_.cwiseProduct(df_)).sum(); }

  private:
    double Hs_, Tp_, f_min_, f_max_, gamma_, g_;
    Eigen::Matrix<double, N_FREQ, 1> frequencies_, S_, A_, df_;

    void computeLogFrequencySpacing() {
      const double log_f_min = std::log(f_min_);
      const double log_f_max = std::log(f_max_);
      Eigen::ArrayXd idx = Eigen::ArrayXd::LinSpaced(N_FREQ, 0, N_FREQ - 1);
      frequencies_ = (log_f_min + (log_f_max - log_f_min) * idx / (N_FREQ - 1)).exp().matrix();
    }

    void computeFrequencyIncrements() {
      df_.head(N_FREQ - 1) = (frequencies_.segment(1, N_FREQ - 1) - frequencies_.head(N_FREQ - 1));
      df_.segment(1, N_FREQ - 2) = 0.5 * (frequencies_.segment(2, N_FREQ - 2) - frequencies_.head(N_FREQ - 2));
      df_(N_FREQ - 1) = frequencies_(N_FREQ - 1) - frequencies_(N_FREQ - 2);
    }

    void computeJonswapSpectrumFromHs() {
      const double fp = 1.0 / Tp_;
      Eigen::Matrix<double, N_FREQ, 1> S0;

      for (int i = 0; i < N_FREQ; ++i) {
        const double f = frequencies_(i);
        const double sigma = (f <= fp) ? 0.07 : 0.09;
        const double dfreq = f - fp;
        const double denom = 2.0 * sigma * sigma * fp * fp;
        const double r = std::exp(-(dfreq * dfreq) / denom);

        const double f2 = f * f;
        const double f4 = f2 * f2;
        const double inv_f5 = 1.0 / (f * f4);

        const double fp2 = fp * fp;
        const double ratio2 = fp2 / f2;
        const double ratio4 = ratio2 * ratio2;

        const double base = (g_ * g_) / std::pow(2.0 * PI, 4)
                            * inv_f5 * std::exp(-1.25 * ratio4);

        const double gamma_r = std::exp(r * std::log(gamma_));
        const double val = base * gamma_r;

        S0(i) = std::isfinite(val) ? val : 0.0;
      }

      const double variance_unit = (S0.cwiseProduct(df_)).sum();
      if (!(variance_unit > 0.0))
        throw std::runtime_error("JonswapSpectrum: computed zero/negative variance");

      const double variance_target = (Hs_ * Hs_) / 16.0;
      const double alpha = variance_target / variance_unit;

      S_ = S0 * alpha;
      A_ = (2.0 * S_.cwiseProduct(df_)).cwiseSqrt();

      const double Hs_est = 4.0 * std::sqrt(0.5 * A_.squaredNorm());
      if (Hs_est <= 0.0) throw std::runtime_error("JonswapSpectrum: Hs_est <= 0");
      const double rel_err = std::abs(Hs_est - Hs_) / Hs_;
      if (rel_err > 1e-3) {
        A_ *= (Hs_ / Hs_est);
        for (int i = 0; i < N_FREQ; ++i) S_(i) = (A_(i) * A_(i)) / (2.0 * df_(i));
      }
    }
};

// Jonswap3dStokesWaves
template<int N_FREQ = 128>
class Jonswap3dStokesWaves {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct WaveState {
      Eigen::Vector3d displacement, velocity, acceleration;
    };

    Jonswap3dStokesWaves(double Hs, double Tp,
                         double mean_direction_deg = 0.0,
                         double f_min = 0.02, double f_max = 0.8,
                         double gamma = 2.0, double g = 9.81,
                         double spreading_exponent = 15.0,
                         unsigned int seed = 42u,
                         double cutoff_tol = 1e-8)
      : spectrum_(Hs, Tp, f_min, f_max, gamma, g),
        mean_dir_rad_(mean_direction_deg * PI / 180.0),
        g_(g), spreading_exponent_(spreading_exponent), cutoff_tol_(cutoff_tol),
        pairwise_size_(size_t(N_FREQ) * (N_FREQ + 1) / 2),
        exp_kz_cached_z_(std::numeric_limits<double>::quiet_NaN()),
        x_cached_(std::numeric_limits<double>::quiet_NaN()),
        y_cached_(std::numeric_limits<double>::quiet_NaN()),
        stokes_drift_mean_xy_valid_(false),
        stokes_drift_surface_valid_(false)
    {
      // Deep-water dispersion (use spectrum frequencies directly)
      omega_ = 2.0 * PI * spectrum_.frequencies();
      k_     = omega_.array().square() / g_;

      // Direction & phase
      dir_x_.setZero(); dir_y_.setZero();
      kx_.setZero();    ky_.setZero();
      phi_.setZero();
      stokes_drift_scalar_.setZero();

      // Derived arrays
      Aomega_  = spectrum_.amplitudes().array() * omega_.array();
      Aomega2_ = spectrum_.amplitudes().array() * omega_.array().square();

      // Allocate & init
      allocatePairArrays();
      initializeRandomPhases(seed);
      initializeDirectionalSpread(seed);
      computeWaveDirectionComponents();
      computePerComponentStokesDriftEstimate();

      // Precompute products
      A_dirx_       = (spectrum_.amplitudes().array() * dir_x_.array()).eval();
      A_diry_       = (spectrum_.amplitudes().array() * dir_y_.array()).eval();
      Aomega_dirx_  = (Aomega_ * dir_x_.array()).eval();
      Aomega_diry_  = (Aomega_ * dir_y_.array()).eval();
      Aomega2_dirx_ = (Aomega2_ * dir_x_.array()).eval();
      Aomega2_diry_ = (Aomega2_ * dir_y_.array()).eval();

      precomputePairwise();
      precomputeSurfaceConstants();
      checkSteepness();

      // Initialize caches (per-frequency fixed-size, pairwise dynamic)
      exp_kz_.setOnes();
      exp_kz_pairs_ = VecD::Ones(pairwise_size_);
      pair_mask_    = VecD::Ones(pairwise_size_);
      theta2_cache_ = VecD::Zero(pairwise_size_);

      trig_cache_.sin_second = VecD::Zero(pairwise_size_);
      trig_cache_.cos_second = VecD::Zero(pairwise_size_);
      trig_cache_.last_t = std::numeric_limits<double>::quiet_NaN();

      // First-order slope helpers: A * kx, A * ky
      Akx_ = (spectrum_.amplitudes().array() * kx_.array()).eval();
      Aky_ = (spectrum_.amplitudes().array() * ky_.array()).eval();
    }

    // Surface (z = 0)
    WaveState getSurfaceState(double x, double y, double t) const {
      return computeState(x, y, t,
                          exp_kz_surface_, exp_kz_pairs_surface_, pair_mask_surface_,
                          stokes_drift_surface_xy_, stokes_drift_surface_valid_);
    }

    // Any depth
    WaveState getLagrangianState(double x, double y, double t, double z = 0.0) const {
      if (!std::isfinite(exp_kz_cached_z_) || exp_kz_cached_z_ != z) {
        // exp(k z) per-frequency
        for (int i = 0; i < N_FREQ; ++i) exp_kz_(i) = std::exp(k_(i) * z);

        // exp((k_i + k_j) z) pairwise (vectorized)
        exp_kz_pairs_ = (k_sum_.array() * z).exp().matrix();

        // attenuation mask
        if (cutoff_tol_ > 0.0) {
          pair_mask_ = ((Bij_.array().abs() * exp_kz_pairs_.array()) >= cutoff_tol_).cast<double>().matrix();
        } else {
          pair_mask_.setOnes();
        }

        exp_kz_cached_z_ = z;
        stokes_drift_mean_xy_valid_ = false;
        trig_cache_.last_t = std::numeric_limits<double>::quiet_NaN();
      }
      return computeState(x, y, t,
                          exp_kz_, exp_kz_pairs_, pair_mask_,
                          stokes_drift_mean_xy_, stokes_drift_mean_xy_valid_);
    }

    // Surface slopes (∂η/∂x, ∂η/∂y) at z = 0, including 1st + 2nd order
    Eigen::Vector2d getSurfaceSlopes(double x, double y, double t) const {
      // First-order part: η1(x,y,t) = Σ A_i sin(θ_i), θ_i = kx_i x + ky_i y + φ_i - ω_i t
      // ∂η1/∂x = Σ A_i kx_i cos(θ_i), ∂η1/∂y = Σ A_i ky_i cos(θ_i)
      const Eigen::Array<double, N_FREQ, 1> arg0 =
        (kx_.array() * x + ky_.array() * y + phi_.array() - omega_.array() * t).eval();
      const Eigen::Array<double, N_FREQ, 1> cos0 = arg0.cos();

      // surface => exp(k z) = 1
      double slope_x = (Akx_ * cos0).sum();
      double slope_y = (Aky_ * cos0).sum();

      // Prepare second-order angle cache θ2 = (k_i + k_j)·(x,y) + (φ_i + φ_j)
      // Recompute if (x,y) changed
      if (!std::isfinite(x_cached_) || !std::isfinite(y_cached_) || x_cached_ != x || y_cached_ != y) {
        Eigen::Vector2d xy; xy << x, y;
        const VecD Kxy = Ksum2_ * xy; // P×1
        theta2_cache_ = Kxy + phi_sum_;
        x_cached_ = x; y_cached_ = y;
        trig_cache_.last_t = std::numeric_limits<double>::quiet_NaN();
      }

      // Second-order: η2 = Σ coeff_ij cos(θ2_ij - (ω_i+ω_j)t)
      // where coeff_ij = factor * Bij * exp((k_i+k_j)z) ; at surface z=0 => exp(...) = 1
      // ∂η2/∂x = - Σ coeff_ij sin(θ2_ij - wsum t) * (kx_i + kx_j)
      // ∂η2/∂y = - Σ coeff_ij sin(θ2_ij - wsum t) * (ky_i + ky_j)
      
      const Eigen::ArrayXd arg2  = (theta2_cache_.array() - omega_sum_.array() * t);
      const Eigen::ArrayXd sin2  = (arg2.sin() * pair_mask_surface_.array());
      const Eigen::ArrayXd coeff = (factor_.array() * Bij_.array()); // exp(...) = 1 at surface

      // add second-order slope contributions
      // note the minus sign from d/dx cos(·) = -sin(·) * d(·)/dx
      slope_x += -((coeff * sin2) * kx_sum_.array()).sum();
      slope_y += -((coeff * sin2) * ky_sum_.array()).sum();

      return Eigen::Vector2d(slope_x, slope_y);
    }

    // Directional Spectrum API

    // Normalization Γ(s+1/2)/(√π Γ(s+1)) for cos^{2s}
    // Ensures ∫_0^{2π} D(θ; f) dθ = 1
    static double spreadingNormalization(double s) {
      return std::tgamma(s + 0.5) / (std::sqrt(PI) * std::tgamma(s + 1.0));
    }

    // Compute directional spectrum at a given frequency f and angle θ
    double directionalSpectrumValue(double f, double theta) const {
      // nearest frequency bin
      auto &freqs = spectrum_.frequencies();
      int idx = int(std::lower_bound(freqs.data(), freqs.data() + N_FREQ, f) - freqs.data());
      if (idx < 0 || idx >= N_FREQ) return 0.0;

      const double S_f = spectrum_.spectrum()(idx);

      // spreading around mean_dir_rad_
      const double dtheta = theta - mean_dir_rad_;

      // Cosine spreading
      const double s = spreading_exponent_;
      const double spread = spreadingNormalization(s) *
                            std::pow(std::max(0.0, std::cos(0.5 * dtheta)), 2.0 * s);

      return S_f * spread;
    }

    // Discrete directional spectrum, size N_FREQ × M
    Eigen::MatrixXd getDirectionalSpectrum(int M) const {
      Eigen::MatrixXd E(N_FREQ, M);
      const double dtheta = 2.0 * PI / M;

      const double s = spreading_exponent_;
      const double norm = spreadingNormalization(s);

      // Precompute spreading weights
      std::vector<double> spread(M);
      for (int m = 0; m < M; ++m) {
        double theta = -PI + m * dtheta;
        double dtheta_rel = theta - mean_dir_rad_;
        spread[m] = norm * std::pow(std::max(0.0, std::cos(0.5 * dtheta_rel)), 2.0 * s);
      }

      for (int i = 0; i < N_FREQ; ++i) {
        double S_f = spectrum_.spectrum()(i);
        for (int m = 0; m < M; ++m) {
          E(i, m) = S_f * spread[m];
        }
      }
      return E;
    }

    const Eigen::Matrix<double, N_FREQ, 1>& spectrum() const    { return spectrum_.spectrum(); }
    const Eigen::Matrix<double, N_FREQ, 1>& frequencies() const { return spectrum_.frequencies(); }
    const Eigen::Matrix<double, N_FREQ, 1>& amplitudes() const  { return spectrum_.amplitudes(); }
    const Eigen::Matrix<double, N_FREQ, 1>& df() const          { return spectrum_.df(); }

  private:
    using IndexT = Eigen::Index;

    // Spectrum & parameters
    JonswapSpectrum<N_FREQ> spectrum_;

    double mean_dir_rad_, g_, spreading_exponent_, cutoff_tol_;
    size_t pairwise_size_;

    // Per-frequency (fixed-size, aligned)
    Eigen::Matrix<double, N_FREQ, 1> omega_, k_, phi_;
    Eigen::Matrix<double, N_FREQ, 1> dir_x_, dir_y_, kx_, ky_;
    Eigen::Matrix<double, N_FREQ, 1> stokes_drift_scalar_;
    Eigen::Array<double, N_FREQ, 1>  Aomega_, Aomega2_;

    // Precomputed products
    Eigen::Array<double, N_FREQ, 1> A_dirx_, A_diry_;
    Eigen::Array<double, N_FREQ, 1> Aomega_dirx_, Aomega_diry_;
    Eigen::Array<double, N_FREQ, 1> Aomega2_dirx_, Aomega2_diry_;

    // Per-pair (dynamic Eigen, alignment-safe)
    VecD Bij_, kx_sum_, ky_sum_, k_sum_;
    VecD omega_sum_, omega_sum2_, phi_sum_, factor_;
    VecD hx_, hy_;
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::ColMajor, Eigen::Dynamic, 2> Ksum2_;

    // Depth caches
    mutable Eigen::Array<double, N_FREQ, 1> exp_kz_;
    mutable VecD exp_kz_pairs_, theta2_cache_, pair_mask_;
    mutable Eigen::Vector2d stokes_drift_mean_xy_;
    mutable bool   stokes_drift_mean_xy_valid_;
    mutable double exp_kz_cached_z_, x_cached_, y_cached_;

    // Trig cache
    struct TrigCache {
      VecD sin_second, cos_second;
      double last_t;
    };
    mutable TrigCache trig_cache_;

    // Surface caches
    Eigen::Array<double, N_FREQ, 1> exp_kz_surface_;
    VecD exp_kz_pairs_surface_, pair_mask_surface_;
    mutable Eigen::Vector2d stokes_drift_surface_xy_;
    mutable bool stokes_drift_surface_valid_;

    // amplitude*wavenumber components (for first-order slopes)
    Eigen::Array<double, N_FREQ, 1> Akx_, Aky_;

    // Core compute
    WaveState computeState(double x, double y, double t,
                           const Eigen::Array<double, N_FREQ, 1> &exp_kz_arr,
                           const VecD &exp_kz_pairs_v,
                           const VecD &pair_mask_v,
                           Eigen::Vector2d &stokes_xy_cache,
                           bool &stokes_xy_valid) const
    {
      // First-order
      const Eigen::Array<double, N_FREQ, 1> arg0 =
        (kx_.array() * x + ky_.array() * y + phi_.array() - omega_.array() * t).eval();
      const Eigen::Array<double, N_FREQ, 1> sin0 = (arg0.sin() * exp_kz_arr).eval();
      const Eigen::Array<double, N_FREQ, 1> cos0 = (arg0.cos() * exp_kz_arr).eval();

      Eigen::Vector3d disp = Eigen::Vector3d::Zero();
      Eigen::Vector3d vel  = Eigen::Vector3d::Zero();
      Eigen::Vector3d acc  = Eigen::Vector3d::Zero();

      disp.x() -= (A_dirx_       * cos0).sum();
      disp.y() -= (A_diry_       * cos0).sum();
      disp.z() += (spectrum_.amplitudes().array() * sin0).sum();

      vel.x()  -= (Aomega_dirx_  * sin0).sum();
      vel.y()  -= (Aomega_diry_  * sin0).sum();
      vel.z()  -= (Aomega_       * cos0).sum();

      acc.x()  += (Aomega2_dirx_ * cos0).sum();
      acc.y()  += (Aomega2_diry_ * cos0).sum();
      acc.z()  -= (Aomega2_      * sin0).sum();

      // Recompute θ2 if (x,y) changed
      if (!std::isfinite(x_cached_) || !std::isfinite(y_cached_) ||
          x_cached_ != x || y_cached_ != y) {
        Eigen::Vector2d xy; xy << x, y;
        const VecD Kxy = Ksum2_ * xy; // P×1
        theta2_cache_ = Kxy + phi_sum_;
        x_cached_ = x; y_cached_ = y;
        trig_cache_.last_t = std::numeric_limits<double>::quiet_NaN();
      }

      // Update second-order trig cache if time changed
      if (!std::isfinite(trig_cache_.last_t) || std::fabs(trig_cache_.last_t - t) > 1e-12) {
        const Eigen::ArrayXd arg2 = (theta2_cache_.array() - omega_sum_.array() * t);
        trig_cache_.sin_second = (arg2.sin() * pair_mask_v.array()).matrix();
        trig_cache_.cos_second = (arg2.cos() * pair_mask_v.array()).matrix();
        trig_cache_.last_t = t;
      }

      // Second-order contributions
      const Eigen::ArrayXd coeff = (factor_.array() * Bij_.array()) * exp_kz_pairs_v.array();
      const Eigen::ArrayXd C     = coeff * trig_cache_.cos_second.array();
      const Eigen::ArrayXd S     = coeff * trig_cache_.sin_second.array();

      disp.x() += -(C * hx_.array()).sum();
      disp.y() += -(C * hy_.array()).sum();
      disp.z() +=   C.sum();

      vel.x()  += (S * omega_sum_.array() * hx_.array()).sum();
      vel.y()  += (S * omega_sum_.array() * hy_.array()).sum();
      vel.z()  += (S * omega_sum_.array()).sum();

      acc.x()  += -(C * omega_sum2_.array() * hx_.array()).sum();
      acc.y()  += -(C * omega_sum2_.array() * hy_.array()).sum();
      acc.z()  += -(C * omega_sum2_.array()).sum();

      // Stokes drift (lazy, depth-dependent)
      if (!stokes_xy_valid) {
        const Eigen::Array<double, N_FREQ, 1> exp2 = exp_kz_arr * exp_kz_arr;
        const auto Us_z = stokes_drift_scalar_.array() * exp2;

        stokes_xy_cache[0] = (Us_z * dir_x_.array()).sum();
        stokes_xy_cache[1] = (Us_z * dir_y_.array()).sum();
        stokes_xy_valid = true;
      }
      vel.x() += stokes_xy_cache[0];
      vel.y() += stokes_xy_cache[1];
      return {disp, vel, acc};
    }

    // Helpers
    void allocatePairArrays() {
      const IndexT P = static_cast<IndexT>(pairwise_size_);
      Bij_.resize(P); kx_sum_.resize(P); ky_sum_.resize(P); k_sum_.resize(P);
      omega_sum_.resize(P); omega_sum2_.resize(P); phi_sum_.resize(P);
      factor_.resize(P); hx_.resize(P); hy_.resize(P);
      Ksum2_.resize(P, 2);

      exp_kz_surface_.setOnes();                           // size = N_FREQ
      exp_kz_pairs_surface_ = VecD::Ones(P);    // size = P
      pair_mask_surface_     = VecD::Ones(P);

      exp_kz_.setOnes();                                   // size = N_FREQ
      exp_kz_pairs_.resize(P);
      theta2_cache_.resize(P);
      pair_mask_.resize(P);
    }

    void initializeRandomPhases(unsigned int seed) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> dist(0.0, 2.0 * PI);
      for (int i = 0; i < N_FREQ / 2; ++i) {
        const double ph = dist(rng);
        phi_(i) = ph;
        phi_(N_FREQ - 1 - i) = -ph;
      }
      if (N_FREQ % 2 == 1) phi_(N_FREQ / 2) = dist(rng);
    }

    void initializeDirectionalSpread(unsigned int seed) {
      std::mt19937 rng_dir(seed + 1u);
      std::uniform_real_distribution<double> u01(-1.0, 1.0);

      const double amax = spectrum_.amplitudes().maxCoeff();
      Eigen::Array<double, N_FREQ, 1> spread_scale = Eigen::Array<double, N_FREQ, 1>::Ones();
      if (amax > 0.0) {
        const auto amp_ratio = (spectrum_.amplitudes().array() / amax).eval();
        spread_scale = amp_ratio.pow(spreading_exponent_);
      }

      constexpr double max_spread = PI * 0.5;
      for (int i = 0; i < N_FREQ / 2; ++i) {
        const double delta = u01(rng_dir) * (spread_scale(i) * max_spread);
        const double angle = mean_dir_rad_ + delta;
        dir_x_(i) = std::cos(angle); dir_y_(i) = std::sin(angle);
        const int j = N_FREQ - 1 - i;
        const double angle_j = angle + PI;
        dir_x_(j) = std::cos(angle_j); dir_y_(j) = std::sin(angle_j);
      }
      if (N_FREQ % 2 == 1) {
        const int mid = N_FREQ / 2;
        const double delta = u01(rng_dir) * (spread_scale(mid) * max_spread);
        const double angle = mean_dir_rad_ + delta;
        dir_x_(mid) = std::cos(angle); dir_y_(mid) = std::sin(angle);
      }
      for (int i = 0; i < N_FREQ; ++i) {
        const double n = std::hypot(dir_x_(i), dir_y_(i));
        if (n > 0.0) {
          dir_x_(i) /= n;
          dir_y_(i) /= n;
        } else {
          dir_x_(i) = std::cos(mean_dir_rad_);
          dir_y_(i) = std::sin(mean_dir_rad_);
        }
      }
    }

    void computeWaveDirectionComponents() {
      kx_ = k_.array() * dir_x_.array();
      ky_ = k_.array() * dir_y_.array();
    }

    void computePerComponentStokesDriftEstimate() {
      stokes_drift_scalar_ = omega_.array() * k_.array() * spectrum_.amplitudes().array().square();
    }

    void precomputePairwise() {
      constexpr double tiny = 1e-18;
      IndexT idx = 0;
      const IndexT P = static_cast<IndexT>(pairwise_size_);
      for (int i = 0; i < N_FREQ; ++i) {
        const double ki = k_(i);
        for (int j = i; j < N_FREQ; ++j) {
          const double kj   = k_(j);
          const double ksum = ki + kj;
          const double kij  = ki * kj;
          const double T_plus = (ksum > tiny) ? (kij / ksum) : 0.0;

          const double kxsum = kx_(i) + kx_(j);
          const double kysum = ky_(i) + ky_(j);
          const double wsum  = omega_(i) + omega_(j);

          Bij_(idx)        = T_plus * spectrum_.amplitudes()(i) * spectrum_.amplitudes()(j);
          kx_sum_(idx)     = kxsum; ky_sum_(idx) = kysum; k_sum_(idx) = ksum;
          omega_sum_(idx)  = wsum;  omega_sum2_(idx) = wsum * wsum;
          phi_sum_(idx)    = phi_(i) + phi_(j);
          factor_(idx)     = (i == j) ? 1.0 : 2.0;
          hx_(idx)         = (ksum > tiny) ? (kxsum / ksum) : 0.0;
          hy_(idx)         = (ksum > tiny) ? (kysum / ksum) : 0.0;

          Ksum2_(idx, 0)    = kxsum; Ksum2_(idx, 1) = kysum;
          ++idx;
        }
      }
      assert(idx == P && "pairwise fill mismatch");
    }

    void precomputeSurfaceConstants() {
      exp_kz_surface_.setOnes();
      exp_kz_pairs_surface_.setOnes();

      if (cutoff_tol_ > 0.0) {
        pair_mask_surface_ = (Bij_.array().abs() >= cutoff_tol_).cast<double>().matrix();
      } else {
        pair_mask_surface_.setOnes();
      }

      // Stokes drift at the surface
      stokes_drift_surface_xy_.setZero();
      for (int i = 0; i < N_FREQ; ++i) {
        const double Us0 = stokes_drift_scalar_(i); // e^{0}^2 = 1
        stokes_drift_surface_xy_[0] += Us0 * dir_x_(i);
        stokes_drift_surface_xy_[1] += Us0 * dir_y_(i);
      }
      stokes_drift_surface_valid_ = true;
    }

    void checkSteepness() {
      const double max_steep = (k_.array() * spectrum_.amplitudes().array()).maxCoeff();
      if (max_steep > 0.4) throw std::runtime_error("Jonswap3dStokesWaves: wave too steep (>0.4), unstable");
    }
};

#ifdef JONSWAP_TEST
// CSV generator for testing
static void generateWaveJonswapCSV(const std::string& filename,
                                   double Hs, double Tp, double mean_dir_deg,
                                   double duration = 40.0, double dt = 0.005) {
  constexpr int N = 128;
  auto waveModel = std::make_unique<Jonswap3dStokesWaves<N>>(Hs, Tp, mean_dir_deg, 0.02, 0.8, 2.0, 9.81, 10.0);
  const int N_time = static_cast<int>(duration / dt) + 1;
  Eigen::ArrayXd time = Eigen::ArrayXd::LinSpaced(N_time, 0.0, duration);
  Eigen::ArrayXXd disp(3, N_time), vel(3, N_time), acc(3, N_time);
  Eigen::ArrayXXd slopes(2, N_time);

  for (int i = 0; i < N_time; ++i) {
    auto state  = waveModel->getSurfaceState(0.0, 0.0, time(i));
    auto sxy    = waveModel->getSurfaceSlopes(0.0, 0.0, time(i));
    for (int j = 0; j < 3; ++j) {
      disp(j, i) = state.displacement(j);
      vel(j, i)  = state.velocity(j);
      acc(j, i)  = state.acceleration(j);
    }
    slopes(0, i) = sxy.x();
    slopes(1, i) = sxy.y();
  }
  std::ofstream file(filename);
  file << "time,disp_x,disp_y,disp_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,slope_x,slope_y\n";
  for (int i = 0; i < N_time; ++i) {
    file << time(i) << ","
         << disp(0, i) << "," << disp(1, i) << "," << disp(2, i) << ","
         << vel(0, i)  << "," << vel(1, i)  << "," << vel(2, i)  << ","
         << acc(0, i)  << "," << acc(1, i)  << "," << acc(2, i)  << ","
         << slopes(0, i) << "," << slopes(1, i) << "\n";
  }
}

static void exportDirectionalSpectrumCSV(const std::string& filename,
    double Hs, double Tp,
    double mean_dir_deg = 0.0,
    int N_freq = 128, int N_theta = 72) {
  auto waveModel = std::make_unique<Jonswap3dStokesWaves<128>>(Hs, Tp, mean_dir_deg);
  auto freqs = waveModel->frequencies();
  Eigen::MatrixXd E = waveModel->getDirectionalSpectrum(N_theta);

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

static void Jonswap_testWavePatterns() {
  generateWaveJonswapCSV("short_waves_stokes.csv",  0.5,  3.0, 30.0);
  generateWaveJonswapCSV("medium_waves_stokes.csv", 2.0,  7.0, 30.0);
  generateWaveJonswapCSV("long_waves_stokes.csv",   4.0, 12.0, 30.0);
}

static void Jonswap_testWaveSpectrum() {
  exportDirectionalSpectrumCSV("short_waves_jonswap_spectrum.csv",  0.5,  3.0, 30.0, 128, 72);
  exportDirectionalSpectrumCSV("medium_waves_jonswap_spectrum.csv", 2.0,  7.0, 30.0, 128, 72);
  exportDirectionalSpectrumCSV("long_waves_jonswap_spectrum.csv",   4.0, 12.0, 30.0, 128, 72);
}
#endif
