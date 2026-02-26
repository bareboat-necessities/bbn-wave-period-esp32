#pragma once
/*
  Copyright (c) 2025  Mikhail Grushinskiy
  Released under the MIT License

  SeaStateFusionFilter2 

*/

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

#include <cmath>
#include <memory>
#include <algorithm>
#include <array>

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "FirstOrderIIRSmoother.h"
#include "SeaStateAutoTuner.h"
#include "MagAutoTuner.h"
#include "Kalman3D_Wave_2.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"
#include "WaveSpectrumEstimator2.h"

// Shared constants

constexpr float ACC_NOISE_FLOOR_SIGMA_DEFAULT = 0.12f;

constexpr float MIN_FREQ_HZ = 0.12f;
constexpr float MAX_FREQ_HZ = 5.0f;

constexpr float MIN_TAU_S   = 0.02f;
constexpr float MAX_TAU_S   = 4.5f;
constexpr float MAX_SIGMA_A = 6.0f;

constexpr float ADAPT_TAU_SEC          = 1.5f;
constexpr float ADAPT_EVERY_SECS       = 0.1f;
constexpr float ONLINE_TUNE_WARMUP_SEC = 5.0f;
constexpr float MAG_DELAY_SEC          = 8.0f;

// Frequency smoother dt (SeaStateFusionFilter2 is designed for 240 Hz)
constexpr float FREQ_SMOOTHER_DT = 1.0f / 240.0f;

// Tune state

struct TuneState {
  float tau_applied   = 1.1f;   // s
  float sigma_applied = 1e-2f;  // m/s^2
};

// Tracker policy traits

template<TrackerType>
struct TrackerPolicy;

// Aranovskiy
template<>
struct TrackerPolicy<TrackerType::ARANOVSKIY> {
  using Tracker = AranovskiyFilter<double>;
  Tracker t;

  TrackerPolicy() : t() {
    double omega_up   = (FREQ_GUESS * 2.0) * (2.0 * M_PI);
    double k_gain     = 20.0;
    double x1_0       = 0.0;
    double omega_init = (FREQ_GUESS / 1.5) * 2.0 * M_PI;
    double theta_0    = -(omega_init * omega_init);
    double sigma_0    = theta_0;
    t.setParams(omega_up, k_gain);
    t.setState(x1_0, theta_0, sigma_0);
  }

  double run(float a, float dt) {
    t.update(static_cast<double>(a) / g_std, static_cast<double>(dt));
    return t.getFrequencyHz();
  }
};

// KalmANF
template<>
struct TrackerPolicy<TrackerType::KALMANF> {
  using Tracker = KalmANF<double>;
  Tracker t = Tracker();

  double run(float a, float dt) {
    double e;
    double freq = t.process(static_cast<double>(a) / g_std, static_cast<double>(dt), &e);
    return freq;
  }
};

// ZeroCross
#define ZERO_CROSSINGS_HYSTERESIS  0.04f
#define ZERO_CROSSINGS_PERIODS     1

template<>
struct TrackerPolicy<TrackerType::ZEROCROSS> {
  using Tracker = SchmittTriggerFrequencyDetector;
  Tracker t = Tracker(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

  double run(float a, float dt) {
    float f_byZeroCross = t.update(a / g_std,
                                   ZERO_CROSSINGS_SCALE,
                                   ZERO_CROSSINGS_DEBOUNCE_TIME,
                                   ZERO_CROSSINGS_STEEPNESS_TIME,
                                   dt);
    double freq =
      (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT ||
       f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ)
      ? FREQ_GUESS
      : static_cast<double>(f_byZeroCross);
    return freq;
  }
};

//  Unified SeaState fusion filter

template<TrackerType trackerT>
class SeaStateFusionFilter2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using TrackingPolicy = TrackerPolicy<trackerT>;

  enum class StartupStage { Cold, TunerWarm, Live };

  explicit SeaStateFusionFilter2(bool with_mag = true)
    : with_mag_(with_mag),
      time_(0.0),
      last_adapt_time_sec_(0.0),
      freq_hz_(FREQ_GUESS),
      freq_hz_slow_(FREQ_GUESS)
  {
    freq_input_lpf_.setCutoff(max_freq_hz_);
    freq_stillness_.setTargetFreqHz(min_freq_hz_);
    startup_stage_   = StartupStage::Cold;
    startup_stage_t_ = 0.0f;
  }

  StartupStage getStartupStage() const noexcept { return startup_stage_; }
  bool isAdaptiveLive() const noexcept { return startup_stage_ == StartupStage::Live; }

  void tune_for_wave_RMS_() {
    mekf_->set_wave_Q_scale(1.10f);
  
    // much smaller bias learning so bias doesn't absorb wave residual
    mekf_->set_accel_bias_update_scale(0.02f);
  
    // Clamp closer to expected truth, with some room
    mekf_->set_accel_bias_abs_max(0.06f);
  
    // Much smaller accel bias RW (especially Z)
    mekf_->set_Q_bacc_rw(Eigen::Vector3f(7.0e-5f, 7.0e-5f, 3.0e-5f));
  }

  void initialize(const Eigen::Vector3f& sigma_a,
                  const Eigen::Vector3f& sigma_g,
                  const Eigen::Vector3f& sigma_m)
  {
    mekf_ = std::make_unique<Kalman3D_Wave_2<float>>(sigma_a, sigma_g, sigma_m);
    init_spectrum_adapter_();
    tune_for_wave_RMS_();
    
    enterCold_();
    apply_oscillators_tune_();
    mekf_->set_exact_att_bias_Qd(true);
  }

  void initialize_ext(const Eigen::Vector3f& sigma_a,
                      const Eigen::Vector3f& sigma_g,
                      const Eigen::Vector3f& sigma_m,
                      float Pq0, float Pb0,
                      float b0,
                      float gravity_magnitude)
  {
    mekf_ = std::make_unique<Kalman3D_Wave_2<float>>(sigma_a, sigma_g, sigma_m,
                                                    Pq0, Pb0, b0, gravity_magnitude);
    init_spectrum_adapter_();
    tune_for_wave_RMS_();
    
    enterCold_();
    apply_oscillators_tune_();
    mekf_->set_exact_att_bias_Qd(true);
  }

  // BODY-frame accel
  void initialize_from_acc(const Eigen::Vector3f& acc_body_ned) {
    if (mekf_ && acc_body_ned.allFinite()) {
      mekf_->initialize_from_acc(acc_body_ned);
    }
  }

  void updateTime(float dt,
                  const Eigen::Vector3f& gyro_body_ned,
                  const Eigen::Vector3f& acc_body_ned,
                  float tempC = 35.0f)
  {
    if (!mekf_) return;
    if (!(dt > 0.0f) || !std::isfinite(dt)) return;
    if (!gyro_body_ned.allFinite() || !acc_body_ned.allFinite()) return;
  
    time_ += dt;
    startup_stage_t_ += dt;
  
    if (mekf_->warmup_mode()) {
      // mag is unused in update_initialization() right now, so pass zero.
      mekf_->update_initialization(acc_body_ned, gyro_body_ned,
                                   Eigen::Vector3f::Zero(), dt);
    }
  
    const float a_x_body = acc_body_ned.x();
    const float a_y_body = acc_body_ned.y();
  
    // In NED, acc.z is "down" specific force. a_z_inertial_down = acc_z + g
    const float a_z_inertial_down = acc_body_ned.z() + g_std;
  
    // MEKF time update
    mekf_->time_update(gyro_body_ned, dt);
  
    // Grace period after wave enable: temporarily inflate accel noise
    // to avoid the first-enable kick.
    if (wave_enable_grace_sec_ > 0.0f) {
      wave_enable_grace_sec_ = std::max(0.0f, wave_enable_grace_sec_ - dt);
  
      if (Racc_nominal_hold_.allFinite() && Racc_nominal_hold_.maxCoeff() > 0.0f) {
        mekf_->set_Racc((1.25f * Racc_nominal_hold_).eval());
      }
      if (wave_enable_grace_sec_ <= 0.0f) {
        mekf_->set_Racc(Racc_nominal_hold_.eval()); // restore
      }
    }
  
    // Adaptive accel measurement noise in Live (must be set BEFORE accel update).
    // set_Racc expects std-dev, not variance.
    if (startup_stage_ == StartupStage::Live && wave_enable_grace_sec_ <= 0.0f) {
      Eigen::Vector3f sig_nom = Racc_nominal_;
      if (!sig_nom.allFinite() || sig_nom.maxCoeff() <= 0.0f) {
        sig_nom = Eigen::Vector3f::Constant(std::max(0.12f, acc_noise_floor_sigma_));
      }
  
      const float sea = std::max(0.0f, tune_.sigma_applied);
  
      // Dynamic-motion indicator: if |a|-g or gyro is high, trust accel less for attitude
      const float an = acc_body_ned.norm();
      const float gyro_n = gyro_body_ned.norm();
      const float accel_dev_g = (std::isfinite(an) ? std::fabs(an - g_std) / g_std : 0.0f);
      const float gyro_dps    = gyro_n * 57.295779513f;
      
      // Make gating a bit more aggressive for violent motion
      const float dyn = std::clamp(std::max(accel_dev_g / 0.12f, gyro_dps / 18.0f), 0.0f, 1.0f);
      
      // Protect attitude via XY inflation
      const float infl_xy_base = std::clamp(1.20f + 0.85f * std::min(sea, 3.0f), 1.20f, 3.80f);
      const float infl_xy = std::clamp(infl_xy_base * (1.0f + 2.5f * dyn), 1.0f, 10.0f);
      
      const float infl_z = std::clamp(1.0f + 0.60f * dyn, 1.0f, 2.5f);
      Eigen::Vector3f sig_live(sig_nom.x() * infl_xy,
                               sig_nom.y() * infl_xy,
                               sig_nom.z() * infl_z);
      
      // Never go below nominal sensor sigma 
      sig_live.x() = std::max(sig_live.x(), sig_nom.x());
      sig_live.y() = std::max(sig_live.y(), sig_nom.y());
      sig_live.z() = std::max(sig_live.z(), sig_nom.z());
      
      const float floor = std::max(0.12f, acc_noise_floor_sigma_);
      sig_live.x() = std::max(sig_live.x(), floor);
      sig_live.y() = std::max(sig_live.y(), floor);
      sig_live.z() = std::max(sig_live.z(), floor);
      
      mekf_->set_Racc(sig_live);
    }
  
    // Accel update (uses whatever Racc was set above)
    mekf_->measurement_update_acc_only(acc_body_ned, tempC);
  
    // Tilt reset gate
    {
      Eigen::Quaternionf q_bw = mekf_->quaternion_boat();
      q_bw.normalize();
  
      const Eigen::Vector3f z_body_down_world = q_bw * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
      const Eigen::Vector3f z_world_down(0.0f, 0.0f, 1.0f);
  
      float cos_tilt = z_body_down_world.normalized().dot(z_world_down);
      cos_tilt = std::max(-1.0f, std::min(1.0f, cos_tilt));
      const float tilt_deg = std::acos(cos_tilt) * 57.295779513f;
  
      constexpr float TILT_RESET_DEG          = 70.0f;
      constexpr float TILT_RESET_HOLD_SEC     = 0.35f;
      constexpr float TILT_RESET_COOLDOWN_SEC = 3.0f;
  
      if (tilt_reset_cooldown_sec_ > 0.0f) {
        tilt_reset_cooldown_sec_ = std::max(0.0f, tilt_reset_cooldown_sec_ - dt);
      }
  
      if (tilt_deg > TILT_RESET_DEG) {
        tilt_over_limit_sec_ += dt;
      } else {
        tilt_over_limit_sec_ = std::max(0.0f, tilt_over_limit_sec_ - 2.0f * dt);
      }
  
      if (tilt_over_limit_sec_ >= TILT_RESET_HOLD_SEC && tilt_reset_cooldown_sec_ <= 0.0f) {
        mekf_->initialize_from_acc(acc_body_ned);
  
        if (startup_stage_ != StartupStage::Live) {
          enterCold_();
          resetTrackingState_();
        }
  
        tilt_over_limit_sec_ = 0.0f;
        tilt_reset_cooldown_sec_ = TILT_RESET_COOLDOWN_SEC;
      }
    }
  
    // vertical (up positive) = -a_inertial_down
    a_vert_up_ = -a_z_inertial_down;

    bool spectrum_new_block = false;
    if (spectral_mode_matching_enable_) {
      // Feed RAW vertical inertial accel; estimator has its own HP/LP/decimation.
      spectrum_new_block = spectrum_.processSample(static_cast<double>(a_vert_up_));
    }
    
    // LPF for tracker/tuner input
    const float a_vert_lp = freq_input_lpf_.step(a_vert_up_, dt);
  
    const float f_tracker = static_cast<float>(tracker_policy_.run(a_vert_lp, dt));
    f_raw_ = f_tracker;
  
    const float f_after_still = freq_stillness_.step(a_vert_lp, dt, f_tracker);
  
    float f_fast = freq_fast_smoother_.update(f_after_still);
    float f_slow = freq_slow_smoother_.update(f_fast);
  
    f_fast = std::min(std::max(f_fast, min_freq_hz_), max_freq_hz_);
    f_slow = std::min(std::max(f_slow, min_freq_hz_), max_freq_hz_);
  
    freq_hz_      = f_fast;
    freq_hz_slow_ = f_slow;

    {
      const float fr = std::max(1e-4f, f_raw_);
      const float fs = std::max(1e-4f, freq_hz_slow_);
      const float e  = std::log(fr) - std::log(fs);
      const float e2 = e * e;
      base_logvar_ema_ = (1.0f - base_logvar_alpha_) * base_logvar_ema_ + base_logvar_alpha_ * e2;
    }

    // Spectral fusion (log-domain), using CURRENT freq_hz_slow_
    if (spectrum_new_block && spectrum_.ready()) {
      const auto st = spectrum_.estimateLogFreqStats();
    
      if (std::isfinite(st.f_center_hz) && st.f_center_hz > 0.0 && std::isfinite(st.sig_logf)) {
        const float f_spec = std::clamp((float)st.f_center_hz, min_freq_hz_, max_freq_hz_);
    
        const float R_spec = std::max(1e-4f, (float)(st.sig_logf * st.sig_logf));
        const float R_base = std::max(1e-4f, base_logvar_ema_);
    
        const float lf_base = std::log(std::max(1e-4f, freq_hz_slow_));
        const float lf_spec = std::log(std::max(1e-4f, f_spec));
    
        const float w_base = 1.0f / R_base;
        const float w_spec = 1.0f / R_spec;
    
        const float lf_fused = (w_base * lf_base + w_spec * lf_spec) / (w_base + w_spec);
        const float f_fused  = std::clamp(std::exp(lf_fused), min_freq_hz_, max_freq_hz_);
    
        spectral_fp_hz_smth_ = f_fused;
        spectral_fp_valid_   = true;
      }
    }
    
    // wave_freq_hz_ is what we trust for wave/tau (spectral if valid, else tracker slow)
    wave_freq_hz_ = spectral_fp_valid_ && std::isfinite(spectral_fp_hz_smth_) && spectral_fp_hz_smth_ > 0.0f
                  ? std::clamp(spectral_fp_hz_smth_, min_freq_hz_, max_freq_hz_)
                  : std::clamp(freq_hz_slow_,        min_freq_hz_, max_freq_hz_);
    
    if (enable_tuner_) {
      // Use LPF'ed vertical accel here too (reduces ringing / freq overshoot)
      update_tuner_(dt, a_vert_lp, wave_freq_hz_);
    }

    if (spectrum_new_block && spectral_mode_matching_enable_ &&
        startup_stage_ == StartupStage::Live && (time_ - last_spectral_apply_time_sec_) >= spectral_apply_every_sec_)
    {
      apply_spectral_mode_matching_();
      last_spectral_apply_time_sec_ = time_;
    }
    
    // Direction filters
    const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz_;
    dir_filter_.update(a_x_body, a_y_body, omega, dt);
    dir_sign_state_ = dir_sign_.update(a_x_body, a_y_body, a_vert_up_, dt);

    if (time_ >= next_debug_print_time_sec_) {
      next_debug_print_time_sec_ = time_ + debug_print_every_sec_;
      printf(
        "f_slow:%6.3f  f_wave:%6.3f  fp_disp:%6.3f  f0_app:%6.3f\n",
        (double)freq_hz_slow_,
        (double)wave_freq_hz_,
        (double)(spectral_fp_valid_ ? spectral_fp_hz_smth_ : NAN),
        (double)(std::isfinite(broadband_f0_applied_hz_) ? broadband_f0_applied_hz_ : NAN)
      );
    }
  }

  void updateMag(const Eigen::Vector3f& mag_body_ned) {
    if (!with_mag_ || !mekf_) return;
    if (time_ < mag_delay_sec_) return;
    if (!mag_body_ned.allFinite()) return;

    mekf_->measurement_update_mag_only(mag_body_ned);
    mag_updates_applied_++;

    if (!std::isfinite(first_mag_update_time_)) {
      first_mag_update_time_ = static_cast<float>(time_);
    }

    if (accel_bias_locked_ &&
        startup_stage_ == StartupStage::Live &&
        mag_updates_applied_ >= MAG_UPDATES_TO_UNLOCK &&
        std::isfinite(first_mag_update_time_) &&
        (static_cast<float>(time_) - first_mag_update_time_) > 1.0f)
    {
      accel_bias_locked_ = false;

      if (freeze_acc_bias_until_live_ && startup_stage_ == StartupStage::Live) {
        mekf_->set_acc_bias_updates_enabled(true);

        if (warmup_Racc_active_) {
          if (Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
            mekf_->set_Racc(Racc_nominal_.eval());
          }
          warmup_Racc_active_ = false;
        }
      }
    }
  }

  void setWithMag(bool with_mag) { with_mag_ = with_mag; }

  void setTauCoeff(float c)   { if (std::isfinite(c) && c > 0.0f) tau_coeff_   = c; }
  void setSigmaCoeff(float c) { if (std::isfinite(c) && c > 0.0f) sigma_coeff_ = c; }

  void setAccNoiseFloorSigma(float s) { if (std::isfinite(s) && s > 0.0f) acc_noise_floor_sigma_ = s; }
  float getAccNoiseFloorSigma() const noexcept { return acc_noise_floor_sigma_; }

  void setFreqInputCutoffHz(float fc) { freq_input_lpf_.setCutoff(fc); }

  void enableClamp(bool flag = true) { enable_clamp_ = flag; }
  void enableTuner(bool flag = true) { enable_tuner_ = flag; }

  void enableLinearBlock(bool flag = true) {
    enable_linear_block_ = flag;
    if (mekf_) {
      const bool on_now = flag && (startup_stage_ == StartupStage::Live);
      mekf_->set_wave_block_enabled(on_now);
    }
  }

  void enableSpectralModeMatching(bool en = true) { spectral_mode_matching_enable_ = en; }

  void setSpectralModeQGain(float g) {
    if (std::isfinite(g) && g > 0.0f) spectral_q_gain_ = g;
  }

  void setSpectralHorizQRatio(float r) {
    if (std::isfinite(r) && r >= 0.0f) spectral_horiz_q_ratio_ = r;
  }

  void setSpectralModeFreqBounds(float fmin, float fmax) {
    if (std::isfinite(fmin) && std::isfinite(fmax) && fmin > 0.0f && fmax > fmin) {
      spectral_mode_fmin_hz_ = fmin;
      spectral_mode_fmax_hz_ = fmax;
    }
  }

  void setFreqBounds(float min_hz, float max_hz) {
    if (!std::isfinite(min_hz) || !std::isfinite(max_hz)) return;
    if (min_hz <= 0.0f || max_hz <= min_hz) return;
    min_freq_hz_ = min_hz;
    max_freq_hz_ = max_hz;
    freq_stillness_.setTargetFreqHz(min_freq_hz_);
  }

  void setTauBounds(float min_tau_s, float max_tau_s) {
    if (!std::isfinite(min_tau_s) || !std::isfinite(max_tau_s)) return;
    if (min_tau_s <= 0.0f || max_tau_s <= min_tau_s) return;
    min_tau_s_ = min_tau_s;
    max_tau_s_ = max_tau_s;
  }

  void setMaxSigmaA(float max_sigma_a) {
    if (std::isfinite(max_sigma_a) && max_sigma_a > 0.0f) max_sigma_a_ = max_sigma_a;
  }

  void setAdaptationTimeConstants(float tau_sec) { if (std::isfinite(tau_sec) && tau_sec > 0.0f) adapt_tau_sec_ = tau_sec; }
  void setAdaptationUpdatePeriod(float every_sec) { if (std::isfinite(every_sec) && every_sec > 0.0f) adapt_every_secs_ = every_sec; }
  void setOnlineTuneWarmupSec(float warmup_sec) { if (std::isfinite(warmup_sec) && warmup_sec >= 0.0f) online_tune_warmup_sec_ = warmup_sec; }
  void setMagDelaySec(float delay_sec) { if (std::isfinite(delay_sec) && delay_sec >= 0.0f) mag_delay_sec_ = delay_sec; }

  void setFreezeAccBiasUntilLive(bool en) { freeze_acc_bias_until_live_ = en; }
  void setWarmupRacc(float r) { if (std::isfinite(r) && r > 0.0f) Racc_warmup_ = r; }
  void setNominalRacc(const Eigen::Vector3f& r) { Racc_nominal_ = r; }

  inline float getFreqHz()       const noexcept { return freq_hz_; }
  inline float getFreqSlowHz()   const noexcept { return freq_hz_slow_; }
  inline float getFreqRawHz()    const noexcept { return f_raw_; }
  inline float getTauApplied()   const noexcept { return tune_.tau_applied; }
  inline float getSigmaApplied() const noexcept { return tune_.sigma_applied; }
  inline float getTauTarget()    const noexcept { return tau_target_; }
  inline float getSigmaTarget()  const noexcept { return sigma_target_; }

  inline float getPeriodSec() const noexcept {
    return (freq_hz_slow_ > 1e-6f) ? 1.0f / freq_hz_slow_ : NAN;
  }

  inline float getAccelVariance() const noexcept { return tuner_.getAccelVariance(); }
  inline float getAccelVertical() const noexcept { return a_vert_up_; }

  inline float getHeaveAbs() const noexcept {
    if (!mekf_) return NAN;
    return std::fabs(mekf_->get_position().z());
  }

  inline float getDisplacementScale(bool /*smoothed*/ = true) const noexcept {
    // "Scale" = Longuet–Higgins mean envelope amplitude E[R] (meters)
    if (spectral_mode_matching_enable_ && spectrum_.ready()) {
      const auto st = spectrum_.estimateLogFreqStats();
      const float m0 = std::max(0.0f, (float)st.m0);          // variance [m^2]
      const float sigma_eta = std::sqrt(m0);                  // std-dev [m]
      return std::sqrt(float(M_PI) / 2.0f) * sigma_eta;        // mean Rayleigh envelope
    }
    return NAN; // before spectrum is warm, don't lie
  }
  
  static float loglog_interp_extrap_(float x, const float* xs, const float* ys, int n)
  {
    if (!(x > 0.0f) || !std::isfinite(x)) return NAN;
  
    auto lerp_log = [&](int i0, int i1, float xv) -> float {
      const float x0 = std::max(xs[i0], 1e-6f);
      const float x1 = std::max(xs[i1], 1e-6f);
      const float y0 = std::max(ys[i0], 1e-6f);
      const float y1 = std::max(ys[i1], 1e-6f);
  
      const float lx0 = std::log(x0), lx1 = std::log(x1);
      const float ly0 = std::log(y0), ly1 = std::log(y1);
      const float lx  = std::log(std::max(xv, 1e-6f));
  
      const float t = (std::fabs(lx1 - lx0) > 1e-9f) ? ((lx - lx0) / (lx1 - lx0)) : 0.0f;
      return std::exp(ly0 + t * (ly1 - ly0));
    };
  
    if (x <= xs[0])       return lerp_log(0, 1, x);
    if (x >= xs[n - 1])   return lerp_log(n - 2, n - 1, x);
  
    for (int i = 0; i < n - 1; ++i) {
      if (x >= xs[i] && x <= xs[i + 1]) {
        return lerp_log(i, i + 1, x);
      }
    }
    return ys[n - 1];
  }

  inline WaveDirection getDirSignState() const noexcept { return dir_sign_state_; }

  Eigen::Vector3f getEulerNautical() const {
    if (!mekf_) return {NAN, NAN, NAN};

    Eigen::Quaternionf q_bw = mekf_->quaternion_boat();
    q_bw.normalize();

    const float x = q_bw.x();
    const float y = q_bw.y();
    const float z = q_bw.z();
    const float w = q_bw.w();
    const float two = 2.0f;

    const float s_yaw = two * std::fma(w, z,  x * y);
    const float c_yaw = 1.0f - two * std::fma(y, y,  z * z);
    float yaw         = std::atan2(s_yaw, c_yaw);

    float s_pitch     = two * std::fma(w, y, -z * x);
    s_pitch           = std::max(-1.0f, std::min(1.0f, s_pitch));
    float pitch       = std::asin(s_pitch);

    const float s_roll = two * std::fma(w, x,  y * z);
    const float c_roll = 1.0f - two * std::fma(x, x,  y * y);
    float roll         = std::atan2(s_roll, c_roll);

    float rn = roll, pn = pitch, yn = yaw;
    aero_to_nautical(rn, pn, yn);

    constexpr float RAD2DEG = 57.29577951308232f;
    return { rn * RAD2DEG, pn * RAD2DEG, yn * RAD2DEG };
  }

  inline auto& mekf() noexcept { return *mekf_; }
  inline const auto& mekf() const noexcept { return *mekf_; }

  inline KalmanWaveDirection& dir() noexcept { return dir_filter_; }
  inline const KalmanWaveDirection& dir() const noexcept { return dir_filter_; }

  inline WaveDirectionDetector<float>& dir_sign() noexcept { return dir_sign_; }
  inline const WaveDirectionDetector<float>& dir_sign() const noexcept { return dir_sign_; }

private:
  struct FreqInputLPF {
    float state       = 0.0f;
    float fc_hz       = 1.0f;
    bool  initialized = false;

    void setCutoff(float fc) {
      if (std::isfinite(fc) && fc > 0.0f) fc_hz = fc;
    }

    float step(float x, float dt) {
      const float alpha = std::exp(-2.0f * static_cast<float>(M_PI) * fc_hz * dt);
      if (!initialized) {
        state = x;
        initialized = true;
        return state;
      }
      state = (1.0f - alpha) * x + alpha * state;
      return state;
    }
  };

  struct StillnessAdapter {
    float energy_ema     = 0.0f;
    float energy_alpha   = 0.05f;
    float energy_thresh  = 8e-4f;

    float still_time_sec = 0.0f;
    float still_thresh_s = 2.0f;

    float relax_tau_sec  = 1.0f;
    float target_freq_hz = MIN_FREQ_HZ;

    bool  freq_init      = false;
    float freq_state     = FREQ_GUESS;

    bool  last_is_still  = false;

    void setTargetFreqHz(float f) { if (std::isfinite(f) && f > 0.0f) target_freq_hz = f; }

    float step(float a_z_inertial_lp, float dt, float freq_in) {
      if (!(dt > 0.0f) || !std::isfinite(freq_in)) return freq_in;

      if (!freq_init || !std::isfinite(freq_state)) {
        freq_state = freq_in;
        freq_init  = true;
      }

      const float a_norm      = a_z_inertial_lp / g_std;
      const float inst_energy = a_norm * a_norm;

      energy_ema = (1.0f - energy_alpha) * energy_ema + energy_alpha * inst_energy;
      const bool is_still = (energy_ema < energy_thresh);
      last_is_still = is_still;

      if (is_still) {
        still_time_sec += dt;
        if (still_time_sec > 60.0f) still_time_sec = 60.0f;

        if (still_time_sec > still_thresh_s) {
          const float relax_alpha = 1.0f - std::exp(-dt / relax_tau_sec);
          freq_state += relax_alpha * (target_freq_hz - freq_state);
        } else {
          freq_state = freq_in;
        }
      } else {
        still_time_sec = 0.0f;
        freq_state = freq_in;
      }
      return freq_state;
    }

    bool  isStill()      const { return last_is_still; }
    float getStillTime() const { return still_time_sec; }
    float getEnergyEma() const { return energy_ema; }
  };
  
  void apply_oscillators_tune_() {
    if (!mekf_) return;

    // Once spectrum is ready and we're doing spectral mode matching,
    // do NOT call set_broadband_params(), it will overwrite per-mode f/q.
    if (spectral_mode_matching_enable_ && spectrum_.ready()) {
      return;
    }
    
    // Wave accel std-dev (m/s^2). Keep a small floor so math doesn't go singular.
    const float sigma_a = std::max(std::max(0.05f, acc_noise_floor_sigma_), tune_.sigma_applied);
  
    // Use the frequency you decided is the "wave/displacement" frequency.
    float f_hz = wave_freq_hz_;
    if (!std::isfinite(f_hz) || f_hz <= 0.0f) f_hz = freq_hz_slow_;
    f_hz = std::clamp(f_hz, min_freq_hz_, max_freq_hz_);
  
    // Longuet–Higgins-consistent elevation/heave scale:
    // σ_η = σ_a / ω², Hs ≈ 4σ_η
    const float sigma_eta = sigmaEtaFromSigmaA_F_(sigma_a, f_hz);
    float Hs_m = NAN;
    if (spectral_mode_matching_enable_ && spectrum_.ready()) {
      Hs_m = (float)spectrum_.computeHs();   // Hs = 4*sqrt(m0) from S_eta
    } else {
      Hs_m = NAN; // keep whatever was set at init; don't inject wrong amplitude
    }
  
    // Keep your existing damping/horizontal heuristics for now (these don't cause the blow-up).
    const float sea = std::max(0.0f, tune_.sigma_applied);
    const float zeta_mid     = std::clamp(0.017f + 0.003f * std::min(sea, 3.0f), 0.016f, 0.026f);
    const float horiz_scale  = std::clamp(0.18f + 0.04f * std::min(sea, 2.0f), 0.18f, 0.28f);
  
    // BASE f0: tracker/tuner driven
    float f0_base_hz = freq_hz_slow_;
    if (tuner_.isFreqReady()) {
      const float ft = tuner_.getFrequencyHz();
      if (std::isfinite(ft) && ft > 0.0f) {
        f0_base_hz = 0.20f * freq_hz_slow_ + 0.80f * ft;
      }
    }
    f0_base_hz = std::clamp(f0_base_hz, min_freq_hz_, max_freq_hz_);
  
    // Spectral-corrected command for broadband center frequency
    float f0_cmd_hz = f0_base_hz;
    if (spectral_fp_valid_ && std::isfinite(spectral_fp_hz_smth_) && spectral_fp_hz_smth_ > 0.0f) {
      f0_cmd_hz = std::clamp(spectral_fp_hz_smth_, min_freq_hz_, max_freq_hz_);
    }
  
    // Smooth applied f0
    if (!std::isfinite(broadband_f0_applied_hz_)) broadband_f0_applied_hz_ = f0_cmd_hz;
    else {
      const float a = expBlendAlpha_(adapt_every_secs_, broadband_f0_tau_sec_);
      broadband_f0_applied_hz_ += a * (f0_cmd_hz - broadband_f0_applied_hz_);
    }

    if (!std::isfinite(Hs_m)) return;
    mekf_->set_broadband_params(broadband_f0_applied_hz_, Hs_m, zeta_mid, horiz_scale);
  }

  // Adaptive knobs that depend on current sea-state energy.
  // Call this right after you update tune_.tau_applied / tune_.sigma_applied
  // and right before/after apply_oscillators_tune_().
  void apply_adaptive_rms_tuning_() {
    if (!mekf_) return;
  
    const float sea = std::max(0.15f, tune_.sigma_applied);
  
    // scale wave process freedom by expected displacement scale
    float disp_scale_m = getDisplacementScale(true); // ~ C_HS * sigma * tau^2
    if (!std::isfinite(disp_scale_m ) || disp_scale_m  <= 0.0f) {
      constexpr float C_HS = 2.0f * std::sqrt(2.0f) / (float(M_PI) * float(M_PI));
      const float tau = std::clamp(tune_.tau_applied, min_tau_s_, max_tau_s_);
      disp_scale_m  = C_HS * sea * tau * tau;
    }
  
    // Linear law: q_scale ∝ displacement scale
    // Tune this single coefficient.
    constexpr float WAVE_Q_PER_M = 3.8f;   // start here; increase to 3.5..5.0 if still too stiff on 4m/8.5m
    const float q_scale = WAVE_Q_PER_M * disp_scale_m;
    
    if (!(spectral_mode_matching_enable_ && spectrum_.ready())) {
      mekf_->set_wave_Q_scale(q_scale);   // fallback before spectrum is ready
    } else {
      mekf_->set_wave_Q_scale(1.0f);      // spectral mode q_k is the source of truth
    }
  
    // bias tuning
    const float ba_gain = std::clamp(0.06f - 0.015f * std::min(sea, 2.0f), 0.02f, 0.06f);
    mekf_->set_accel_bias_update_scale(ba_gain);
    mekf_->set_accel_bias_abs_max(0.06f);
  
    const float rw_xy = std::clamp(5.0e-5f + 0.8e-5f * std::min(sea, 2.0f), 5.0e-5f, 6.6e-5f);
    const float rw_z  = std::clamp(1.8e-5f + 0.5e-5f * std::min(sea, 2.0f), 1.8e-5f, 2.8e-5f);
    mekf_->set_Q_bacc_rw(Eigen::Vector3f(rw_xy, rw_xy, rw_z));
  }

  void update_tuner_(float dt, float a_vert_inertial_up, float wave_freq_hz) {
    tuner_.update(dt, a_vert_inertial_up, wave_freq_hz);
  
    switch (startup_stage_) {
      case StartupStage::Cold:
        if (startup_stage_t_ >= online_tune_warmup_sec_) {
          startup_stage_   = StartupStage::TunerWarm;
          startup_stage_t_ = 0.0f;
        }
        return;
  
      case StartupStage::TunerWarm:
        if (!tuner_.isFreqReady()) return;
        if (tuner_.isReady()) enterLive_();
        break;
  
      case StartupStage::Live:
        break;
    }
  
    // sigma from variance
    float var_total = acc_noise_floor_sigma_ * acc_noise_floor_sigma_;
    if (tuner_.isVarReady()) var_total = std::max(0.0f, tuner_.getAccelVariance());
  
    const float var_noise = acc_noise_floor_sigma_ * acc_noise_floor_sigma_;
    float var_wave = var_total - var_noise;
    if (var_wave < 0.0f) var_wave = 0.0f;
  
    if (freq_stillness_.isStill()) {
      const float still_t = std::max(0.0f, freq_stillness_.getStillTime());
      constexpr float STILL_VAR_DECAY_SEC = 1.0f;
      float atten = std::exp(-still_t / STILL_VAR_DECAY_SEC);
      atten = std::clamp(atten, 0.0f, 1.0f);
      var_wave *= atten;
    }
  
    var_wave = std::max(var_wave, 1e-6f);
    const float sigma_wave = std::sqrt(var_wave);
  
    // tau from wave_freq_hz
    float f_used = wave_freq_hz;
    if (!std::isfinite(f_used) || f_used <= 0.0f) f_used = tuner_.getFrequencyHz();
    if (!std::isfinite(f_used) || f_used < min_freq_hz_) f_used = min_freq_hz_;
    if (f_used > max_freq_hz_) f_used = max_freq_hz_;
  
    const float tau_raw = tau_coeff_ * 0.5f / f_used;
  
    if (enable_clamp_) {
      tau_target_   = std::clamp(tau_raw, min_tau_s_, max_tau_s_);
      sigma_target_ = std::min(sigma_wave * sigma_coeff_, max_sigma_a_);
    } else {
      tau_target_   = tau_raw;
      sigma_target_ = sigma_wave;
    }
  
    if (!tuner_.isVarReady()) {
      sigma_target_ = std::max(sigma_target_, std::max(0.05f, acc_noise_floor_sigma_));
    }
  
    tau_target_   = std::clamp(tau_target_,   min_tau_s_, max_tau_s_);
    sigma_target_ = std::clamp(sigma_target_, 0.05f,      max_sigma_a_);
  
    adapt_mekf_(dt, tau_target_, sigma_target_);
  }

  void adapt_mekf_(float dt, float tau_t, float sigma_t) {
    const float alpha = 1.0f - std::exp(-dt / adapt_tau_sec_);

    tune_.tau_applied   += alpha * (tau_t   - tune_.tau_applied);
    tune_.sigma_applied += alpha * (sigma_t - tune_.sigma_applied);

    if (time_ - last_adapt_time_sec_ > adapt_every_secs_) {
      if (!(spectral_mode_matching_enable_ && spectrum_.ready())) {
        if (tuner_.isFreqReady() || spectral_fp_valid_) apply_oscillators_tune_();
      }
      apply_adaptive_rms_tuning_();
      last_adapt_time_sec_ = time_;
    }
  }

  void resetTrackingState_() {
    tracker_policy_ = TrackingPolicy{};
    freq_input_lpf_ = FreqInputLPF{};
    freq_stillness_ = StillnessAdapter{};
    freq_input_lpf_.setCutoff(max_freq_hz_);
    freq_stillness_.setTargetFreqHz(min_freq_hz_);

    tuner_.reset();

    freq_fast_smoother_ = FirstOrderIIRSmoother<float>(FREQ_SMOOTHER_DT, 3.5f);
    freq_slow_smoother_ = FirstOrderIIRSmoother<float>(FREQ_SMOOTHER_DT, 10.0f);

    freq_hz_      = FREQ_GUESS;
    freq_hz_slow_ = FREQ_GUESS;
    f_raw_        = FREQ_GUESS;
    wave_freq_hz_ = FREQ_GUESS;

    dir_filter_ = KalmanWaveDirection(2.0f * static_cast<float>(M_PI) * FREQ_GUESS);
    dir_sign_state_ = UNCERTAIN;

    reset_spectrum_adapter_();
    
    last_adapt_time_sec_ = time_;
  }

  void enterCold_() {
    startup_stage_   = StartupStage::Cold;
    startup_stage_t_ = 0.0f;

    if (!mekf_) return;

    mekf_->set_warmup_mode(true);
    mekf_->set_wave_block_enabled(false);

    accel_bias_locked_     = with_mag_;
    mag_updates_applied_   = 0;
    first_mag_update_time_ = NAN;

    if (freeze_acc_bias_until_live_) {
      mekf_->set_acc_bias_updates_enabled(false);
      if (std::isfinite(Racc_warmup_) && Racc_warmup_ > 0.0f) {
        mekf_->set_Racc(Eigen::Vector3f::Constant(Racc_warmup_).eval());
        warmup_Racc_active_ = true;
      } else {
        warmup_Racc_active_ = false;
      }
    }

    reset_spectrum_adapter_();
  }

  void enterLive_() {
    startup_stage_   = StartupStage::Live;
    startup_stage_t_ = 0.0f;

    if (!mekf_) return;

    mekf_->set_warmup_mode(false);
    mekf_->set_wave_block_enabled(enable_linear_block_);
    wave_enable_grace_sec_ = 4.0f;             
    Racc_nominal_hold_ = Racc_nominal_;        // keep a copy
    
    if (freeze_acc_bias_until_live_) {
      const bool allow_bias = !accel_bias_locked_;
      mekf_->set_acc_bias_updates_enabled(allow_bias);

      if (warmup_Racc_active_ && Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
        mekf_->set_Racc(Racc_nominal_);
      }
      warmup_Racc_active_ = false;
    }
    
    apply_oscillators_tune_();
    apply_adaptive_rms_tuning_();
    last_adapt_time_sec_ = time_;
  }

private:
  StartupStage startup_stage_   = StartupStage::Cold;
  float        startup_stage_t_ = 0.0f;

  float wave_enable_grace_sec_ = 0.0f;
  Eigen::Vector3f Racc_nominal_hold_ = Eigen::Vector3f::Constant(NAN);

  bool  freeze_acc_bias_until_live_ = true;
  float Racc_warmup_                = 0.5f;
  bool  warmup_Racc_active_         = false;
  Eigen::Vector3f Racc_nominal_     = Eigen::Vector3f::Constant(0.0f);

  bool accel_bias_locked_ = true;
  int  mag_updates_applied_ = 0;
  static constexpr int MAG_UPDATES_TO_UNLOCK = 40;

  bool   with_mag_;
  double time_;
  double last_adapt_time_sec_;

  float first_mag_update_time_ = NAN;

  float tilt_over_limit_sec_ = 0.0f;
  float tilt_reset_cooldown_sec_ = 0.0f;

  float freq_hz_      = FREQ_GUESS;
  float freq_hz_slow_ = FREQ_GUESS;
  float f_raw_        = FREQ_GUESS;
  float wave_freq_hz_ = FREQ_GUESS;   // the frequency we trust for wave/tau (spectral if available)

  float a_vert_up_ = 0.0f;

  bool enable_clamp_ = true;
  bool enable_tuner_ = true;

  bool enable_linear_block_ = true;

  float min_freq_hz_            = MIN_FREQ_HZ;
  float max_freq_hz_            = MAX_FREQ_HZ;
  float min_tau_s_              = MIN_TAU_S;
  float max_tau_s_              = MAX_TAU_S;
  float max_sigma_a_            = MAX_SIGMA_A;
  float adapt_tau_sec_          = ADAPT_TAU_SEC;
  float adapt_every_secs_       = ADAPT_EVERY_SECS;
  float online_tune_warmup_sec_ = ONLINE_TUNE_WARMUP_SEC;
  float mag_delay_sec_          = MAG_DELAY_SEC;

  TrackingPolicy               tracker_policy_{};
  FirstOrderIIRSmoother<float> freq_fast_smoother_{FREQ_SMOOTHER_DT, 3.5f};
  FirstOrderIIRSmoother<float> freq_slow_smoother_{FREQ_SMOOTHER_DT, 10.0f};

  SeaStateAutoTuner tuner_;
  TuneState         tune_;

  float tau_target_   = NAN;
  float sigma_target_ = NAN;

  float acc_noise_floor_sigma_ = ACC_NOISE_FLOOR_SIGMA_DEFAULT;

  float tau_coeff_   = 1.4f;
  float sigma_coeff_ = 0.9f;

  std::unique_ptr<Kalman3D_Wave_2<float>> mekf_;

  KalmanWaveDirection dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

  FreqInputLPF     freq_input_lpf_;
  StillnessAdapter freq_stillness_;

  WaveDirectionDetector<float> dir_sign_{0.002f, 0.005f};
  WaveDirection                dir_sign_state_ = UNCERTAIN;

  // Spectral mode matching
  WaveSpectrumEstimator2<28, 512> spectrum_{};

  bool  spectral_mode_matching_enable_ = true;

  // Displacement-spectrum peak
  float spectral_fp_hz_       = NAN;   // raw peak frequency from latest spectrum block
  float spectral_fp_hz_smth_  = NAN;   // smoothed peak
  bool  spectral_fp_valid_    = false;

  float spectral_fp_tau_sec_  = 3.0f;  // peak smoothing (seconds)

  // Feed peak into broadband center frequency
  float spectral_f0_blend_      = 0.85f; // 0=ignore spectral, 1=use only spectral
  float broadband_f0_applied_hz_= NAN;   // smoothed f0 actually passed to set_broadband_params
  float broadband_f0_tau_sec_   = 2.0f;  // smoothing for applied f0

  // Debug print cadence
  float  debug_print_every_sec_      = 5.0f;
  double next_debug_print_time_sec_  = 0.0;

  // How often to apply to Kalman after a new spectrum block is available
  float spectral_apply_every_sec_ = 1.0f;
  double last_spectral_apply_time_sec_ = -1e9;

  // Spectrum -> q_k conversion knobs
  float spectral_q_gain_         = 1.0f;    // main knob
  float spectral_q_floor_        = 1e-6f;
  float spectral_q_cap_          = 25.0f;   // per-mode cap (pre/post normalize clamp)
  float spectral_horiz_q_ratio_  = 0.22f;   // XY q = ratio * Z q

  // Allowed analysis range for spectral fitting (not applied as mode centers anymore)
  float spectral_mode_fmin_hz_   = 0.06f;
  float spectral_mode_fmax_hz_   = 1.00f;

  float base_logvar_ema_   = 0.25f * 0.25f; // initial guess in (log Hz)^2
  float base_logvar_alpha_ = 0.01f;         // slow EMA

  // Q-only smoothing / budget control
  float spectral_q_apply_tau_sec_   = 3.0f;   // smoothing of per-mode q (seconds)
  float spectral_q_budget_tau_sec_  = 6.0f;   // smoothing of total q budget (seconds)
  float spectral_q_step_up_ratio_   = 1.35f;  // per-apply max growth (before EMA)
  float spectral_q_step_down_ratio_ = 0.55f;  // per-apply max shrink (before EMA)

  bool spectral_applied_initialized_ = false;
  std::array<float, Kalman3D_Wave_2<float>::kWaveModes> spectral_qz_applied_{};

  // Total-q budget (anchored to tuner displacement scale)
  bool  spectral_q_budget_initialized_   = false;
  float spectral_q_budget_base_sum_      = NAN;  // captured baseline total q when spectral starts
  float spectral_q_budget_sum_           = NAN;  // smoothed current budget
  float spectral_q_budget_ref_disp_scale_m_ = NAN;

  static float omegaFromHz_(float f_hz) {
    return 2.0f * float(M_PI) * f_hz;
  }
  
  // σ_η = σ_a / ω²  (narrowband assumption)
  static float sigmaEtaFromSigmaA_F_(float sigma_a, float f_hz) {
    if (!(sigma_a > 0.0f) || !std::isfinite(sigma_a)) return NAN;
    if (!(f_hz > 0.0f) || !std::isfinite(f_hz)) return NAN;
    const float w = omegaFromHz_(f_hz);
    return sigma_a / std::max(1e-9f, w * w);
  }
  
  // Longuet–Higgins mean envelope: E[R] = √(π/2) σ_η
  static float lhMeanEnvelope_(float sigma_eta) {
    if (!(sigma_eta > 0.0f) || !std::isfinite(sigma_eta)) return NAN;
    return std::sqrt(float(M_PI) / 2.0f) * sigma_eta;
  }
  
  // Significant height: Hs ≈ 4 σ_η
  static float lhHs_(float sigma_eta) {
    if (!(sigma_eta > 0.0f) || !std::isfinite(sigma_eta)) return NAN;
    return 4.0f * sigma_eta;
  }

  void init_spectrum_adapter_() {
    reset_spectrum_adapter_();
  }
  
  void reset_spectrum_adapter_() {
    spectrum_ = WaveSpectrumEstimator2<28, 512>{};
  
    spectral_fp_hz_      = NAN;
    spectral_fp_hz_smth_ = NAN;
    spectral_fp_valid_   = false;
  
    broadband_f0_applied_hz_ = NAN;
  
    spectral_applied_initialized_ = false;
    spectral_qz_applied_.fill(0.0f);
  
    spectral_q_budget_initialized_ = false;
    spectral_q_budget_base_sum_ = NAN;
    spectral_q_budget_sum_ = NAN;
    spectral_q_budget_ref_disp_scale_m_ = NAN;
  
    last_spectral_apply_time_sec_ = -1e9;
  }

  static float expBlendAlpha_(float dt, float tau_s) {
    if (!(dt > 0.0f) || !std::isfinite(dt)) return 1.0f;
    if (!(tau_s > 0.0f) || !std::isfinite(tau_s)) return 1.0f;
    return 1.0f - std::exp(-dt / tau_s);
  }

  float spectral_q_budget_target_from_tuner_() const {
    if (!spectral_q_budget_initialized_) return NAN;
    if (!(spectral_q_budget_base_sum_ > 0.0f) || !std::isfinite(spectral_q_budget_base_sum_)) {
      return NAN;
    }

    // Tie the spectral total-q budget to the tuner's displacement-scale estimate,
    // but preserve the baseline budget captured at spectral activation.
    float disp_now = getDisplacementScale(true);
    if (!std::isfinite(disp_now) || disp_now <= 0.0f) {
      return spectral_q_budget_base_sum_;
    }

    float disp_ref = spectral_q_budget_ref_disp_scale_m_;
    if (!std::isfinite(disp_ref) || disp_ref <= 0.0f) {
      disp_ref = disp_now;
    }

    float ratio = disp_now / std::max(1e-5f, disp_ref);
    ratio = std::clamp(ratio, 0.35f, 3.0f); // allow sea-state changes, avoid runaway

    const float tgt = spectral_q_budget_base_sum_ * ratio;
    const float min_budget = float(Kalman3D_Wave_2<float>::kWaveModes) * spectral_q_floor_;
    const float max_budget = float(Kalman3D_Wave_2<float>::kWaveModes) * spectral_q_cap_;

    return std::clamp(tgt, min_budget, max_budget);
  }

  void apply_spectral_mode_matching_() {
    if (!mekf_) return;
    if (!spectral_mode_matching_enable_) return;
    if (!spectrum_.ready()) return;
    if (!enable_linear_block_) return;
  
    constexpr int K = Kalman3D_Wave_2<float>::kWaveModes;
  
    // Get current damping ratios (used by q = 4*zeta*omega^3*var_p)
    std::array<float, K> zeta_k{};
    mekf_->get_wave_mode_zetas(zeta_k);
    for (int k = 0; k < K; ++k) {
      float z = zeta_k[k];
      if (!std::isfinite(z) || z <= 0.0f) z = 0.02f;
      zeta_k[k] = std::clamp(z, 0.003f, 0.20f);
    }
  
    // Spectrum -> (mode centers f_k) + (qz_k) using displacement PSD S_eta
    std::array<float, K> f_mode_hz{};
    std::array<float, K> qz_mode{};
  
    spectrum_.template estimateModeQz<K>(
        f_mode_hz,
        qz_mode,
        zeta_k,
        spectral_mode_fmin_hz_,
        spectral_mode_fmax_hz_,
        spectral_q_gain_,
        spectral_q_floor_,
        spectral_q_cap_);
  
    // Horizontal ratio (constant is fine; physics is in vertical here)
    const float hr = std::clamp(spectral_horiz_q_ratio_, 0.0f, 1.0f);
  
    // When driving per-mode q directly, global wave_Q_scale must be neutral
    mekf_->set_wave_Q_scale(1.0f);
  
    // Push BOTH frequencies and q (preserves existing zeta_k inside mekf)
    mekf_->set_wave_mode_freqs_and_qz(f_mode_hz, qz_mode, hr, spectral_q_floor_);
  }
};

// SeaStateFusion2 wrapper

template<TrackerType trackerT>
class SeaStateFusion2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    bool with_mag = true;

    float mag_delay_sec          = MAG_DELAY_SEC;
    float online_tune_warmup_sec = ONLINE_TUNE_WARMUP_SEC;

    bool  use_fixed_mag_world_ref = false;
    Eigen::Vector3f mag_world_ref = Eigen::Vector3f(0,0,0);

    bool  freeze_acc_bias_until_live = true;
    float Racc_warmup = 0.5f;

    Eigen::Vector3f sigma_a = Eigen::Vector3f(0.2f,0.2f,0.2f);
    Eigen::Vector3f sigma_g = Eigen::Vector3f(0.01f,0.01f,0.01f);
    Eigen::Vector3f sigma_m = Eigen::Vector3f(0.3f,0.3f,0.3f);

    float mag_ref_timeout_sec = 4.5f;
    float mag_odr_guess_hz = 80.0f;

    bool use_custom_mag_tuner_cfg = false;
    MagAutoTuner::Config mag_tuner_cfg{};
  };

  void begin(const Config& cfg) {
    cfg_ = cfg;

    begun_   = true;
    stage_   = Stage::Uninitialized;
    t_       = 0.0f;
    stage_t_ = 0.0f;

    mag_ref_set_ = false;
    mag_body_hold_.setZero();
    last_mag_time_sec_ = NAN;
    dt_mag_sec_ = NAN;
    mag_ref_deadline_sec_ = cfg_.mag_delay_sec + cfg_.mag_ref_timeout_sec;

    fallback_acc_mean_.setZero();
    fallback_mag_mean_.setZero();
    fallback_mean_count_ = 0;

    if (cfg_.use_custom_mag_tuner_cfg) {
      mag_auto_.setConfig(cfg_.mag_tuner_cfg);
    } else {
      mag_auto_.reset();
      cfg_.mag_tuner_cfg = makeDefaultMagInitCfg(cfg_.mag_odr_guess_hz);
      mag_auto_.setConfig(cfg_.mag_tuner_cfg);
    }

    last_acc_body_ned_.setZero();
    last_gyro_body_ned_.setZero();
    last_imu_dt_ = NAN;
    have_last_imu_ = false;

    impl_.setWithMag(cfg.with_mag);
    impl_.setFreezeAccBiasUntilLive(cfg.freeze_acc_bias_until_live);
    impl_.setWarmupRacc(cfg.Racc_warmup);
    impl_.setMagDelaySec(cfg.mag_delay_sec);
    impl_.setOnlineTuneWarmupSec(cfg.online_tune_warmup_sec);

    impl_.initialize(cfg.sigma_a, cfg.sigma_g, cfg.sigma_m);
    last_impl_startup_stage_ = impl_.getStartupStage();

    // keep for warmup->live restore
    impl_.setNominalRacc(cfg.sigma_a);
  }

  void update(float dt,
              const Eigen::Vector3f& gyro_body_ned,
              const Eigen::Vector3f& acc_body_ned,
              float tempC = 35.0f)
  {
    if (!begun_) return;
    if (!(dt > 0.0f) || !std::isfinite(dt)) return;

    t_ += dt;

    if (stage_ == Stage::Uninitialized) {
      impl_.initialize_from_acc(acc_body_ned);
      stage_ = Stage::Warming;
      stage_t_ = 0.0f;
    } else {
      stage_t_ += dt;
    }

    last_acc_body_ned_  = acc_body_ned;
    last_gyro_body_ned_ = gyro_body_ned;
    last_imu_dt_        = dt;
    have_last_imu_      = true;

    impl_.updateTime(dt, gyro_body_ned, acc_body_ned, tempC);

    const auto cur_stage = impl_.getStartupStage();
    
    if (cur_stage != last_impl_startup_stage_) {
      // Entered Cold (e.g. after tilt reset): reset mag-init accumulators ONCE
      if (cur_stage == SeaStateFusionFilter2<trackerT>::StartupStage::Cold) {
        mag_ref_set_ = false;
        mag_auto_.reset();
    
        last_mag_time_sec_ = NAN;
        dt_mag_sec_ = NAN;
    
        // Reset fallback means too, so we don't mix pre/post-reset attitude regimes
        fallback_acc_mean_.setZero();
        fallback_mag_mean_.setZero();
        fallback_mean_count_ = 0;
    
        // Start a fresh deadline window from now (or from mag delay if still before it)
        mag_ref_deadline_sec_ = std::max(t_, cfg_.mag_delay_sec) + cfg_.mag_ref_timeout_sec;
      }
    
      last_impl_startup_stage_ = cur_stage;
    }

    if (stage_ == Stage::Warming && impl_.isAdaptiveLive()) {
      stage_ = Stage::Live;
    }
  }

  void updateMag(const Eigen::Vector3f& mag_body_ned) {
    if (!begun_ || !cfg_.with_mag) return;

    mag_body_hold_ = mag_body_ned;

    if (std::isfinite(last_mag_time_sec_)) {
      dt_mag_sec_ = t_ - last_mag_time_sec_;
    }
    last_mag_time_sec_ = t_;

    if (have_last_imu_) {
      float dtm = dt_mag_sec_;
      if (!std::isfinite(dtm) || dtm <= 0.0f) dtm = 1.0f / std::max(1.0f, cfg_.mag_odr_guess_hz);
      (void)mag_auto_.addMagSample(dtm, last_acc_body_ned_, mag_body_ned, last_gyro_body_ned_);
    }

    // fallback means for ref acquisition
    if (have_last_imu_) {
      const float an = last_acc_body_ned_.norm();
      const float mn = mag_body_ned.norm();
      const float g  = 9.80665f;
      const bool accel_ok = last_acc_body_ned_.allFinite() && std::fabs(an - g) < 0.12f * g;
      const bool mag_ok   = mag_body_ned.allFinite() && (mn > 1e-3f);
      const float gyro_n = last_gyro_body_ned_.norm();
      const bool gyro_ok = last_gyro_body_ned_.allFinite() &&
                           (gyro_n < (60.0f * float(M_PI) / 180.0f)); // 60 deg/s
      if (accel_ok && mag_ok && gyro_ok) {
        fallback_mean_count_++;
        const float invN = 1.0f / static_cast<float>(fallback_mean_count_);
        fallback_acc_mean_ += (last_acc_body_ned_ - fallback_acc_mean_) * invN;
        fallback_mag_mean_ += (mag_body_ned      - fallback_mag_mean_) * invN;
      }
    }

    if (t_ < cfg_.mag_delay_sec) return;

    if (!mag_ref_set_) {
      if (cfg_.use_fixed_mag_world_ref) {
        impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
        mag_ref_set_ = true;
      } else {
        if (cfg_.mag_world_ref.allFinite() && cfg_.mag_world_ref.norm() > 1e-3f) {
          impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
          mag_ref_set_ = true;
        }

        Eigen::Vector3f mag_body_for_ref = mag_body_ned;
        Eigen::Vector3f acc_for_ref      = last_acc_body_ned_;
        bool have_ref_candidate = false;

        Eigen::Vector3f acc_mean, mag_raw_mean, mag_unit_mean;
        if (!mag_ref_set_ && mag_auto_.getResult(acc_mean, mag_raw_mean, mag_unit_mean)) {
          if (acc_mean.allFinite() && acc_mean.norm() > 1e-3f &&
              mag_raw_mean.allFinite() && mag_raw_mean.norm() > 1e-3f)
          {
            acc_for_ref      = acc_mean;
            mag_body_for_ref = mag_raw_mean;
            have_ref_candidate = true;
          }
        }

        if (!have_ref_candidate &&
            std::isfinite(mag_ref_deadline_sec_) && t_ >= mag_ref_deadline_sec_)
        {
          if (cfg_.mag_world_ref.allFinite() && cfg_.mag_world_ref.norm() > 1e-3f) {
            impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
            mag_ref_set_ = true;
          } else if (fallback_mean_count_ >= 200 &&
                     fallback_acc_mean_.allFinite() && fallback_acc_mean_.norm() > 1e-3f &&
                     fallback_mag_mean_.allFinite() && fallback_mag_mean_.norm() > 1e-3f)
          {
            acc_for_ref      = fallback_acc_mean_;
            mag_body_for_ref = fallback_mag_mean_;
            have_ref_candidate = true;
          } else {
            mag_ref_deadline_sec_ = t_ + cfg_.mag_ref_timeout_sec;
          }
        }

        if (!mag_ref_set_ && have_ref_candidate) {
          Eigen::Quaternionf q_tilt = tiltOnlyQuatFromAccel_(acc_for_ref);
          q_tilt.normalize();
          Eigen::Vector3f mag_world_ref_uT = q_tilt * mag_body_for_ref;
          if (mag_world_ref_uT.allFinite() && mag_world_ref_uT.norm() > 1e-3f) {
            impl_.mekf().set_mag_world_ref(mag_world_ref_uT);
            mag_ref_set_ = true;
          }
        }
      }
    }
    
    if (mag_ref_set_) {
      impl_.updateMag(mag_body_ned);
    }
  }

  bool  isLive() const { return stage_ == Stage::Live; }
  float freqHz() const { return impl_.getFreqHz(); }
  Eigen::Vector3f eulerNauticalDeg() const { return impl_.getEulerNautical(); }

  SeaStateFusionFilter2<trackerT>& raw() { return impl_; }

private:
  enum class Stage { Uninitialized, Warming, Live };

  static Eigen::Quaternionf tiltOnlyQuatFromAccel_(const Eigen::Vector3f& acc_body_ned) {
    Eigen::Vector3f a = acc_body_ned;
    const float an = a.norm();
    if (!(an > 1e-6f) || !a.allFinite()) return Eigen::Quaternionf::Identity();

    Eigen::Vector3f body_down = (-a / an);
    const Eigen::Vector3f world_down(0.0f, 0.0f, 1.0f);

    const float d = std::max(-1.0f, std::min(1.0f, body_down.dot(world_down)));
    Eigen::Vector3f axis = body_down.cross(world_down);
    const float axis_n = axis.norm();

    if (axis_n < 1e-6f) {
      if (d > 0.0f) return Eigen::Quaternionf::Identity();
      Eigen::Vector3f ortho = std::fabs(body_down.z()) < 0.9f
        ? Eigen::Vector3f(0,0,1).cross(body_down)
        : Eigen::Vector3f(0,1,0).cross(body_down);
      ortho.normalize();
      return Eigen::Quaternionf(Eigen::AngleAxisf(float(M_PI), ortho));
    }

    axis /= axis_n;
    const float angle = std::acos(d);
    Eigen::Quaternionf q(Eigen::AngleAxisf(angle, axis));
    q.normalize();
    return q;
  }

private:
  Config cfg_{};
  SeaStateFusionFilter2<trackerT> impl_{false};

  bool begun_ = false;

  Stage stage_ = Stage::Uninitialized;
  float t_ = 0.0f;
  float stage_t_ = 0.0f;
  typename SeaStateFusionFilter2<trackerT>::StartupStage last_impl_startup_stage_ =
           SeaStateFusionFilter2<trackerT>::StartupStage::Cold;

  bool mag_ref_set_ = false;
  Eigen::Vector3f mag_body_hold_ = Eigen::Vector3f::Zero();

  float last_mag_time_sec_ = NAN;
  float dt_mag_sec_ = NAN;
  float mag_ref_deadline_sec_ = NAN;

  Eigen::Vector3f last_acc_body_ned_  = Eigen::Vector3f::Zero();
  Eigen::Vector3f last_gyro_body_ned_ = Eigen::Vector3f::Zero();
  float last_imu_dt_ = NAN;
  bool  have_last_imu_ = false;

  Eigen::Vector3f fallback_acc_mean_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f fallback_mag_mean_ = Eigen::Vector3f::Zero();
  int fallback_mean_count_ = 0;

  MagAutoTuner mag_auto_;
};
