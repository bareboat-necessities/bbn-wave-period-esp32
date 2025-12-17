#pragma once

/*
  Copyright (c) 2025  Mikhail Grushinskiy  
  Released under the MIT License 

  SeaStateFusionFilter
  
  Marine Inertial Navigational System (INS) Filter for IMU

  Combines multiple real-time estimators into a cohesive ocean-state tracker:

    • Quaternion-based attitude and linear motion estimation via Kalman3D_Wave  

    • Dominant frequency tracking using one of:
          – AranovskiyFilter     (frequency estimator)
          – KalmANF              (adaptive notch / Kalman frequency tracker)
          – SchmittTrigger       (zero-cross event detector)

    • Dual-stage frequency smoothing:
          – Fast 1st-order IIR (≈ few s, ~90% step) for demodulation / direction
          – Slow 1st-order IIR (≈ longer s, ~90% step) for auto-tuning / moments

    • Online auto-tuning of Kalman filter parameters (τ, σₐ, Rₛ) through
      SeaStateAutoTuner, which estimates acceleration variance and applies the
      σₐ·τ³ regularization law to stabilize displacement drift correction.

  Where
  – τ (tau):  OU process time constant ≈ ½ · T  (half the dominant period of acceleration)
  – σₐ:       Stationary acceleration standard deviation, EWMA-tracked online
  – Rₛ:       Pseudo-measurement noise controlling integral drift suppression
  – Rₛ_xy:    Reduced in X/Y (anisotropic weighting for vertical-dominant seas)
  
  Adaptive update:  exponential smoothing toward targets over ADAPT_TAU_SEC

  Features
  • Modular tracker selection via TrackerPolicy template
  • Quaternion-consistent Euler conversion (aerospace → nautical, ENU frame)
  • Magnetometer yaw correction with configurable startup delay
  • Fully compatible with Arduino or native Eigen builds
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <cmath>
#include <memory>
#include <algorithm>

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "FirstOrderIIRSmoother.h"
#include "SeaStateAutoTuner.h"
#include "Kalman3D_Wave.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"

// Shared constants

// Estimated vertical accel noise floor (1σ), m/s².
// Tweak from bench data with IMU sitting still.
constexpr float ACC_NOISE_FLOOR_SIGMA_DEFAULT = 0.15f; 

constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 12.0f;

constexpr float MIN_TAU_S   = 0.02f;
constexpr float MAX_TAU_S   = 3.0f;
constexpr float MAX_SIGMA_A = 6.0f;
constexpr float MIN_R_S     = 0.5f;
constexpr float MAX_R_S     = 35.0f;

constexpr float ADAPT_TAU_SEC            = 1.5f;
constexpr float ADAPT_R_S_SEC            = 8.0f;
constexpr float ADAPT_EVERY_SECS         = 0.1f;
constexpr float ONLINE_TUNE_WARMUP_SEC   = 5.0f;
constexpr float MAG_DELAY_SEC            = 8.0f;

// Frequency smoother dt (SeaStateFusionFilter is designed for 240 Hz)
constexpr float FREQ_SMOOTHER_DT = 1.0f / 240.0f;

struct TuneState {
    float tau_applied   = 0.97f;    // s
    float sigma_applied = 1e-3f;    // m/s²
    float RS_applied    = 0.2f;     // m*s
};

//  Tracker policy traits
template<TrackerType>
struct TrackerPolicy; // primary template (undefined)

// Aranovskiy
template<>
struct TrackerPolicy<TrackerType::ARANOVSKIY> {
    using Tracker = AranovskiyFilter<double>;
    Tracker t;

    TrackerPolicy() : t() {
        double omega_up   = (FREQ_GUESS * 2.0) * (2.0 * M_PI);  // upper angular frequency
        double k_gain     = 20.0;                               // higher = faster, but risk overflow if too high
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
#define ZERO_CROSSINGS_HYSTERESIS     0.04f
#define ZERO_CROSSINGS_PERIODS        1

template<>
struct TrackerPolicy<TrackerType::ZEROCROSS> {
    using Tracker = SchmittTriggerFrequencyDetector;
    Tracker t = Tracker(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);
    double run(float a, float dt) {
        float f_byZeroCross = t.update(a / g_std,
                                       ZERO_CROSSINGS_SCALE /* max g */,
                                       ZERO_CROSSINGS_DEBOUNCE_TIME,
                                       ZERO_CROSSINGS_STEEPNESS_TIME,
                                       dt);
        double freq = (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT ||
                       f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ)
                      ? FREQ_GUESS
                      : static_cast<double>(f_byZeroCross);
        return freq;
    }
};

//  Unified SeaState fusion filter
template<TrackerType trackerT>
class SeaStateFusionFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using TrackingPolicy = TrackerPolicy<trackerT>;

    enum class StartupStage {
        Cold,        // just booted or just had a big tilt reset
        TunerWarm,   // MEKF + freq running, tuner collecting stats
        Live         // tuner is trusted; full adaptation & extras allowed
    };

    explicit SeaStateFusionFilter(bool with_mag = true)
        : with_mag_(with_mag),
          time_(0.0),
          last_adapt_time_sec_(0.0),
          freq_hz_(FREQ_GUESS),
          freq_hz_slow_(FREQ_GUESS)
    {
        // Default cutoff ~max_freq_hz_ Hz: passes waves, kills 8–37 Hz engine band
        freq_input_lpf_.setCutoff(max_freq_hz_);
        freq_stillness_.setTargetFreqHz(min_freq_hz_);
        startup_stage_   = StartupStage::Cold;
        startup_stage_t_ = 0.0f;
    }

    StartupStage getStartupStage() const noexcept { return startup_stage_; }
    bool isAdaptiveLive() const noexcept { return startup_stage_ == StartupStage::Live; }

    void initialize(const Eigen::Vector3f& sigma_a,
                    const Eigen::Vector3f& sigma_g,
                    const Eigen::Vector3f& sigma_m)
    {
        mekf_ = std::make_unique<Kalman3D_Wave<float>>(sigma_a, sigma_g, sigma_m);
        enterCold_();      // applies freeze + warmup Racc + disables linear block
        apply_tune();      // ok to preload
        mekf_->set_exact_att_bias_Qd(true);
    }

    void initialize_ext(const Eigen::Vector3f& sigma_a,
                        const Eigen::Vector3f& sigma_g,
                        const Eigen::Vector3f& sigma_m,
                        float Pq0, float Pb0,
                        float b0, float R_S_noise,
                        float gravity_magnitude) 
    {
        mekf_ = std::make_unique<Kalman3D_Wave<float>>(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0, R_S_noise, gravity_magnitude);
        enterCold_();      // applies freeze + warmup Racc + disables linear block
        apply_tune();      // ok to preload
        mekf_->set_exact_att_bias_Qd(true);
    } 

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) {
            mekf_->initialize_from_acc(acc_world);
        }
    }

    // Time update (IMU integration + frequency tracking)
    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc,
                    float tempC = 35.0f)
    {
        if (!mekf_) return;
        time_ += dt;
    
        // Track time spent in current startup stage
        if (dt > 0.0f && std::isfinite(dt)) {
            startup_stage_t_ += dt;
        }
    
        // Keep BODY components around for direction/sign
        const float a_x_body = acc.x();
        const float a_y_body = acc.y();
        const float a_z_inertial = acc.z() + g_std;
    
        // MEKF updates first (attitude + latent a_w)
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc, tempC);

        if (!with_mag_) {
            Eigen::Quaternionf q_bw = mekf_->quaternion_boat();
            q_bw.normalize();
        
            const Eigen::Vector3f z_body_down_world = q_bw * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            const Eigen::Vector3f z_world_down(0.0f, 0.0f, 1.0f);
        
            float cos_tilt = z_body_down_world.normalized().dot(z_world_down);
            cos_tilt = std::max(-1.0f, std::min(1.0f, cos_tilt));
            const float tilt_deg = std::acos(cos_tilt) * 57.295779513f;
        
            constexpr float TILT_RESET_DEG = 45.0f;
            if (tilt_deg > TILT_RESET_DEG) {
                // Re-lock attitude to gravity
                mekf_->initialize_from_acc(acc);
            
                // Return to Cold stage (this disables linear block + applies warmup bias policy)
                enterCold_();
            
                // Reset *all* slow/statistical machinery (not just some of it)
                freq_input_lpf_       = FreqInputLPF{};
                freq_stillness_       = StillnessAdapter{};
                tuner_.reset();
            
                freq_fast_smoother_   = FirstOrderIIRSmoother<float>(FREQ_SMOOTHER_DT, 3.5f);
                freq_slow_smoother_   = FirstOrderIIRSmoother<float>(FREQ_SMOOTHER_DT, 10.0f);
            
                freq_hz_      = FREQ_GUESS;
                freq_hz_slow_ = FREQ_GUESS;
                f_raw         = FREQ_GUESS;
            
                // Optional but recommended: reset direction state too
                dir_filter_      = KalmanWaveDirection(2.0f * static_cast<float>(M_PI) * FREQ_GUESS);
                // dir_sign_ is not re-assigned here because WaveDirectionDetector has const members
                // and is not assignable; if you want a logical reset, add a reset() method to
                // WaveDirectionDetector and call it instead.
                dir_sign_state_  = UNCERTAIN;
            
                // Optional: avoid immediate adapt burst after reset
                last_adapt_time_sec_ = time_;
            
                // Done: do NOT also set startup_stage_ manually here
            }
        }    
        // vertical (up positive)
        a_vert_up = -a_z_inertial;
    
        // LPF on vertical accel for tracker input
        const float a_vert_lp = freq_input_lpf_.step(a_vert_up, dt);
    
        // Raw freq from tracker
        const float f_tracker = static_cast<float>(tracker_policy_.run(a_vert_lp, dt));
        f_raw = f_tracker;
    
        // Stillness detector also sees vertical Z accel
        const float f_after_still = freq_stillness_.step(a_vert_lp, dt, f_tracker);
    
        // Fast & slow smoothed frequencies
        float f_fast = freq_fast_smoother_.update(f_after_still);
        float f_slow = freq_slow_smoother_.update(f_fast);
    
        f_fast = std::min(std::max(f_fast, min_freq_hz_), max_freq_hz_);
        f_slow = std::min(std::max(f_slow, min_freq_hz_), max_freq_hz_);
    
        freq_hz_      = f_fast;   // demod / direction
        freq_hz_slow_ = f_slow;   // tuner / moments
    
        // Tuner gets vertical accel
        if (enable_tuner_) {
            update_tuner(dt, a_vert_up, f_after_still);
        }
    
        const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz_;
    
        if (enable_extra_drift_correction_) {
            const float f_for_drift     = freq_hz_;
            const float f_min_for_drift = 0.15f;              // > MIN_FREQ_HZ, tuned
            const float sigma_a         = tune_.sigma_applied;
        
            // Only use extra drift when clearly in wave regime
            const float SIGMA_A_MIN_FOR_DRIFT = 0.25f;        // ~2.5% g; tune on bench
        
            const bool wavey_enough =
                std::isfinite(f_for_drift) &&
                f_for_drift > f_min_for_drift &&
                std::isfinite(sigma_a) &&
                sigma_a > SIGMA_A_MIN_FOR_DRIFT &&
                !freq_stillness_.isStill() &&
                (startup_stage_ == StartupStage::Live);
        
            if (wavey_enough) {
                const float tau = std::max(tune_.tau_applied,  min_tau_s_);
        
                // Vertical displacement std ~ k * σ_a * τ²
                // 
                // σ_a * τ² is decent estimate of amplitude
                // For noisy or biased sensors this can be used
                // to drop R_S when displacement estimate from
                // filter exceeds it. This drop needs to be relatively smooth and fast.
                // When displacement returns to normal range R_S needs to
                // be restored to normal value predicted by adaptation
                // also relatively fast and smooth. 
                float sigma_disp_vert  = extra_drift_gain_ * getDisplacementScale();
                float sigma_disp_horiz = sigma_disp_vert * S_factor_;
        
                // Never let this pseudo-measurement get *too* confident
                const float SIGMA_P_MIN = 0.05f;  // 5 cm
                const float SIGMA_P_MAX = 5.0f;   // 5 m, safety upper bound
                sigma_disp_vert  = std::min(std::max(sigma_disp_vert,  SIGMA_P_MIN), SIGMA_P_MAX);
                sigma_disp_horiz = std::min(std::max(sigma_disp_horiz, SIGMA_P_MIN), SIGMA_P_MAX);
        
                Eigen::Vector3f sigma_disp_meas(sigma_disp_horiz, sigma_disp_horiz, sigma_disp_vert);   
                Eigen::Vector3f acc_in(0, 0, a_z_inertial);
                updatePositionFromAccOmega(acc_in, omega, sigma_disp_meas);
            }
        }
      
        // Direction filters run on BODY accel, "sign" uses vertical acceleration
        dir_filter_.update(a_x_body, a_y_body, omega, dt);
        dir_sign_state_ = dir_sign_.update(a_x_body, a_y_body, a_vert_up, dt);
    }

    //  Magnetometer correction
    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (with_mag_ && mekf_ && time_ >= mag_delay_sec_) {
            mekf_->measurement_update_mag_only(mag_body_ned);
        }
    }

    // Convenience wrapper: infer p_meas from inertial acceleration and
    // angular frequency ω [rad/s] via p ≈ -a/ω² on all 3 axes.
    //
    // a: inertial acceleration (NED), m/s²
    // omega  : angular frequency, rad/s (use 2π f from frequency tracker)
    // sigma_disp_meas: per-axis std of the resulting displacement measurement [m]
    // omega_min: minimum |ω| to avoid insane amplification at very low freq
    // 
    // This correction assumes very narrowband regime. 
    // Use it as very soft correction with very high sigma_disp_meas or when
    // you know that the sea state is close to harrowband harmonic. 
    void updatePositionFromAccOmega(
        const Eigen::Vector3f& a, float omega, const Eigen::Vector3f& sigma_disp_meas, float omega_min = (2.0f * M_PI * 0.06f)) 
    {
        if (!std::isfinite(omega)) {
            return;
        }
    
        float abs_omega = std::abs(omega);
        if (abs_omega < omega_min) {
            // Too low frequency: 1/ω² would blow up
            abs_omega = omega_min;
        }
        const float w2 = abs_omega * abs_omega;
    
        // First-order approximation: p ≈ -a/ω² on all axes
        Eigen::Vector3f p_meas = -a / w2;
    
        if (!p_meas.allFinite()) {
            return;
        }
        mekf_->measurement_update_position_pseudo(p_meas, sigma_disp_meas);
    }

    void setWithMag(bool with_mag) {
        with_mag_ = with_mag;
    }

    // Anisotropy configuration (runtime)
    // S-factor scales horizontal vs vertical stationary std of a_w.
    // RS XY factor scales pseudo-measurement noise in X/Y vs Z.
    void setSFactor(float s) {
        if (std::isfinite(s) && s > 0.0f) {
            S_factor_ = s;
        }
    }
    void setRSXYFactor(float k) {
        if (std::isfinite(k)) {
            R_S_xy_factor_ = std::min(std::max(k, 0.0f), 1.0f);
        }
    }

    void setTauCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            tau_coeff_ = c;
        }
    }
    void setSigmaCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            sigma_coeff_ = c;
        }
    }
    void setRSCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            R_S_coeff_ = c;
        }
    }

    void setAccNoiseFloorSigma(float s) {
        if (std::isfinite(s) && s > 0.0f) {
            acc_noise_floor_sigma_ = s;
        }
    }
    float getAccNoiseFloorSigma() const noexcept {
        return acc_noise_floor_sigma_;
    }

    // Configure engine-reject LPF on vertical accel for tracker input
    void setFreqInputCutoffHz(float fc) {
        freq_input_lpf_.setCutoff(fc);
    }

    void enableClamp(bool flag = true) {
        enable_clamp_ = flag;
    }
    void enableTuner(bool flag = true) {
        enable_tuner_ = flag;
    }
    void enableHeaveRSGating(bool flag = true) {
        enable_heave_RS_gating_ = flag;
    }
    void enableExtraDriftCorrection(bool flag = true) {
        enable_extra_drift_correction_ = flag;
    }

    // Enable/disable use of the extended linear block [v,p,S,a_w] in Kalman3D_Wave.
    //
    // When flag == false:
    //   • Kalman3D_Wave::set_linear_block_enabled(false) is enforced,
    //   • startup stages still progress (Cold → TunerWarm → Live),
    //   • tuner still runs and exposes tau/sigma/R_S targets & metrics,
    //   • but OU / v/p/S/a_w propagation and S-pseudo-measurements are never used.
    //
    // When flag == true (default):
    //   • the linear block is enabled once the tuner is ready in TunerWarm.
    void enableLinearBlock(bool flag = true) {
        enable_linear_block_ = flag;
        if (mekf_) {
            // Only actually *use* the block in Live stage; Cold/TunerWarm are QMEKF-only.
            const bool on_now = flag && (startup_stage_ == StartupStage::Live);
            mekf_->set_linear_block_enabled(on_now);
        }
    }

    // Tunable adaptation bounds and time constants
    void setFreqBounds(float min_hz, float max_hz) {
        if (!std::isfinite(min_hz) || !std::isfinite(max_hz)) return;
        if (min_hz <= 0.0f || max_hz <= min_hz) return;
        min_freq_hz_ = min_hz;
        max_freq_hz_ = max_hz;
        // stillness target is conceptually "relax toward min freq"
        freq_stillness_.setTargetFreqHz(min_freq_hz_);
    }

    void setTauBounds(float min_tau_s, float max_tau_s) {
        if (!std::isfinite(min_tau_s) || !std::isfinite(max_tau_s)) return;
        if (min_tau_s <= 0.0f || max_tau_s <= min_tau_s) return;
        min_tau_s_ = min_tau_s;
        max_tau_s_ = max_tau_s;
    }

    void setMaxSigmaA(float max_sigma_a) {
        if (!std::isfinite(max_sigma_a) || max_sigma_a <= 0.0f) return;
        max_sigma_a_ = max_sigma_a;
    }

    void setRSBounds(float min_RS, float max_RS) {
        if (!std::isfinite(min_RS) || !std::isfinite(max_RS)) return;
        if (min_RS <= 0.0f || max_RS <= min_RS) return;
        min_R_S_ = min_RS;
        max_R_S_ = max_RS;
    }

    void setAdaptationTimeConstants(float tau_sec, float RS_sec) {
        if (std::isfinite(tau_sec) && tau_sec > 0.0f)   adapt_tau_sec_   = tau_sec;
        if (std::isfinite(RS_sec)  && RS_sec  > 0.0f)   adapt_R_S_sec_   = RS_sec;
    }

    void setAdaptationUpdatePeriod(float every_sec) {
        if (std::isfinite(every_sec) && every_sec > 0.0f) {
            adapt_every_secs_ = every_sec;
        }
    }

    void setOnlineTuneWarmupSec(float warmup_sec) {
        if (std::isfinite(warmup_sec) && warmup_sec >= 0.0f) {
            online_tune_warmup_sec_ = warmup_sec;
        }
    }

    void setMagDelaySec(float delay_sec) {
        if (std::isfinite(delay_sec) && delay_sec >= 0.0f) {
            mag_delay_sec_ = delay_sec;
        }
    }

    // Scale factor for extra drift correction: σ_p ≈ k * σ_a * τ²
    void setExtraDriftGain(float k) {
        if (std::isfinite(k) && k > 0.0f) {
            extra_drift_gain_ = k;
        }
    }

    void setFreezeAccBiasUntilLive(bool en) { freeze_acc_bias_until_live_ = en; }
    void setWarmupRacc(float r) { if (std::isfinite(r) && r > 0.0f) Racc_warmup_ = r; }

    // For SeaStateFusionFilter to restore Racc automatically
    void setNominalRacc(const Eigen::Vector3f& r) { Racc_nominal_ = r; }

    //  Exposed getters
    inline float getFreqHz()        const noexcept { return freq_hz_; }        // fast branch
    inline float getFreqSlowHz()    const noexcept { return freq_hz_slow_; }   // slow branch
    inline float getFreqRawHz()     const noexcept { return f_raw; }
    inline float getTauApplied()    const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied()  const noexcept { return tune_.sigma_applied; }
    inline float getRSApplied()     const noexcept { return tune_.RS_applied; }
    inline float getTauTarget()     const noexcept { return tau_target_;   }
    inline float getSigmaTarget()   const noexcept { return sigma_target_; }
    inline float getRSTarget()      const noexcept { return RS_target_;    }

    // Use slow frequency as a more stable "period" proxy
    inline float getPeriodSec() const noexcept {
        return (freq_hz_slow_ > 1e-6f) ? 1.0f / freq_hz_slow_ : NAN;
    }

    inline float getAccelVariance() const noexcept { return tuner_.getAccelVariance(); }
    inline float getAccelVertical() const noexcept { return a_vert_up; }

    inline float getHeaveAbs() const noexcept { return std::fabs(mekf_->get_position().z()); }

    inline float getDisplacementScale() const noexcept {                 // Longuet-Higgins / Rayleigh
        constexpr float C_HS  = 2.0f * std::sqrt(2.0f) / (M_PI * M_PI);  // ≈ 0.28658
        return C_HS * sigma_target_ * tau_target_ * tau_target_;
    }

    inline WaveDirection getDirSignState() const noexcept { return dir_sign_state_; }

    Eigen::Vector3f getEulerNautical() const {
        if (!mekf_) return {NAN, NAN, NAN};

        // q_bw: body→world 
        Eigen::Quaternionf q_bw = mekf_->quaternion_boat();
        q_bw.normalize();

        const float x = q_bw.x();
        const float y = q_bw.y();
        const float z = q_bw.z();
        const float w = q_bw.w();
        const float two = 2.0f;

        // ZYX (aerospace) from q_bw — radians
        const float s_yaw = two * std::fma(w, z,  x * y);
        const float c_yaw = 1.0f - two * std::fma(y, y,  z * z);
        float yaw         = std::atan2(s_yaw, c_yaw);

        float s_pitch     = two * std::fma(w, y, -z * x);
        s_pitch           = std::max(-1.0f, std::min(1.0f, s_pitch));
        float pitch       = std::asin(s_pitch);

        const float s_roll = two * std::fma(w, x,  y * z);
        const float c_roll = 1.0f - two * std::fma(x, x,  y * y);
        float roll         = std::atan2(s_roll, c_roll);

        // Aerospace/NED → Nautical/ENU (expects radians)
        float rn = roll;
        float pn = pitch;
        float yn = yaw;
        aero_to_nautical(rn, pn, yn);

        // Radians → degrees
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
    // Simple first-order low-pass filter for vertical accel → tracker input
    struct FreqInputLPF {
        float state       = 0.0f;
        float fc_hz       = 1.0f;   // cutoff in Hz
        bool  initialized = false;

        void setCutoff(float fc) {
            if (std::isfinite(fc) && fc > 0.0f) {
                fc_hz = fc;
            }
        }

        float step(float x, float dt) {
            // y' = -2π fc (y - x)
            // discrete: y_n = (1 - alpha)*x_n + alpha*y_{n-1}
            const float alpha = std::exp(-2.0f * static_cast<float>(M_PI) * fc_hz * dt);

            if (!initialized) {
                state       = x;
                initialized = true;
                return state;
            }
            state = (1.0f - alpha) * x + alpha * state;
            return state;
        }
    };

    // Detect “stillness” from vertical accel and relax frequency when still.
    struct StillnessAdapter {
        // Stillness detection
        float energy_ema      = 0.0f;   // EMA of (a_z/g)^2
        float energy_alpha    = 0.05f;  // smoothing for energy EMA
        float energy_thresh   = 8e-4f;  // below this ⇒ effectively still

        float still_time_sec  = 0.0f;   // accumulated still time
        float still_thresh_s  = 2.0f;   // seconds of low energy before relaxing

        // Relaxation behaviour
        float relax_tau_sec   = 1.0f;   // time constant for freq relaxation
        float target_freq_hz  = MIN_FREQ_HZ; // relaxed target (could also be FREQ_GUESS)

        // Internal frequency state
        bool  freq_init       = false;
        float freq_state      = FREQ_GUESS;

        // Last stillness flag (for external inspection)
        bool  last_is_still   = false;

        void setTargetFreqHz(float f) {
            if (std::isfinite(f) && f > 0.0f) {
                target_freq_hz = f;
            }
        }

        // a_z_inertial_lp: vertical inertial accel (m/s²), low-passed
        // dt             : timestep (s)
        // freq_in        : raw tracker freq (Hz)
        //
        // Returns adjusted frequency:
        //   – follows tracker when not still
        //   – decays toward target_freq_hz when still for long enough
        float step(float a_z_inertial_lp, float dt, float freq_in) {
            if (!(dt > 0.0f) || !std::isfinite(freq_in)) {
                return freq_in;
            }

            // Initialize internal state from tracker on first valid call
            if (!freq_init || !std::isfinite(freq_state)) {
                freq_state = freq_in;
                freq_init  = true;
            }

            // Normalize by g → dimensionless
            const float a_norm      = a_z_inertial_lp / g_std;
            const float inst_energy = a_norm * a_norm; // (a_z/g)^2

            // EWMA of energy
            energy_ema = (1.0f - energy_alpha) * energy_ema + energy_alpha * inst_energy;
            const bool is_still = (energy_ema < energy_thresh);
            last_is_still = is_still;

            if (is_still) {
                // Count how long we've been still
                still_time_sec += dt;
                if (still_time_sec > 60.0f) still_time_sec = 60.0f;

                // After some time still, relax INTERNAL freq_state toward target
                if (still_time_sec > still_thresh_s) {
                    const float relax_alpha = 1.0f - std::exp(-dt / relax_tau_sec);
                    freq_state += relax_alpha * (target_freq_hz - freq_state);
                } else {
                    // before threshold, just track the tracker
                    freq_state = freq_in;
                }
            } else {
                // Not still: reset still timer and follow tracker tightly
                still_time_sec = 0.0f;
                freq_state     = freq_in;
            }
            return freq_state;
        }

        // Expose stillness info to tuner
        bool  isStill()       const { return last_is_still; }
        float getStillTime()  const { return still_time_sec; }
        float getEnergyEma()  const { return energy_ema; }
    };

    float adjustRSWithHeave(float RS_in) const noexcept {
        // Base clamp into global RS range
        float RS_base = std::min(std::max(RS_in, min_R_S_), max_R_S_);

        // Expected displacement scale from (σ, τ)
        const float disp_scale = getDisplacementScale();
        if (!std::isfinite(disp_scale)) {
            return RS_base;
        }

        // Vertical heave magnitude 
        const float heave_abs = getHeaveAbs();
        if (!std::isfinite(heave_abs)) {
            return RS_base;
        }

        // If we're within scale, don't touch R_S
        const float ratio = heave_abs / disp_scale;
        if (ratio <= 1.7f) {
            return RS_base;
        }

        // Excess factor above "expected" envelope
        const float r = ratio - 1.0f;

        // Smooth, aggressive shrinkage:
        //   gate(r) = 1 / (1 + k * r²), with a floor.
        //   r = 0  → gate = 1
        //   r ↑    → gate ↓ smoothly and more aggressively at large r
        const float k        = 0.9f;   // aggressiveness
        const float min_gate = 0.10f;  // don't go totally insane
        float gate = 1.0f / (1.0f + k * r * r);
        if (gate < min_gate) gate = min_gate;

        float RS_adj = RS_base * gate;

        // Ensure we stay in the usual [min_R_S_, max_R_S_] range
        RS_adj = std::min(std::max(RS_adj, min_R_S_), max_R_S_);
        return RS_adj;
    }

void apply_ou_tune_() {
    if (!mekf_) return;

    mekf_->set_aw_time_constant(tune_.tau_applied);

    // In attitude-only mode (linear frozen), Σ_aw still matters (marginalization path),
    // so don’t let sigma collapse to ~0.
    const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
    const float sZ = std::max(sigma_floor, tune_.sigma_applied);
    const float sH = sZ * S_factor_;
    mekf_->set_aw_stationary_std(Eigen::Vector3f(sH, sH, sZ));
}

void apply_RS_tune_() {
    if (!mekf_) return;

    const float RSb = enable_heave_RS_gating_
        ? adjustRSWithHeave(tune_.RS_applied)
        : std::min(std::max(tune_.RS_applied, min_R_S_), max_R_S_);

    mekf_->set_RS_noise(Eigen::Vector3f(
        RSb * R_S_xy_factor_,
        RSb * R_S_xy_factor_,
        RSb
    ));
}

    //  Internal tuning and adaptation
    void apply_tune() {
        if (!mekf_) return;
    
        // OU time constant
        mekf_->set_aw_time_constant(tune_.tau_applied);
    
        // WORLD-frame stationary covariance for a_w (XY equal, Z separate).
        const float sZ = std::max(1e-5f, tune_.sigma_applied);
        const float sH = sZ * S_factor_;
        Eigen::Vector3f a_w_std(sH, sH, sZ);
        mekf_->set_aw_stationary_std(a_w_std); 
    
        // WORLD-frame pseudo-measurement noise for S (anisotropic diagonal).
        const float RSb = enable_heave_RS_gating_ ?
            adjustRSWithHeave(tune_.RS_applied) :
            std::min(std::max(tune_.RS_applied, min_R_S_), max_R_S_);
        mekf_->set_RS_noise(Eigen::Vector3f(
            RSb * R_S_xy_factor_,   // world X
            RSb * R_S_xy_factor_,   // world Y
            RSb                     // world Z
        ));
    }

    void update_tuner(float dt, float a_vert_inertial, float freq_hz_for_tuner) {
        tuner_.update(dt, a_vert_inertial, freq_hz_for_tuner);
    
        // Startup stage logic
        switch (startup_stage_) {
           case StartupStage::Cold:
               // Only track time; no adaptation, no linear block.
               if (startup_stage_t_ >= online_tune_warmup_sec_) {
                   startup_stage_   = StartupStage::TunerWarm;
                   startup_stage_t_ = 0.0f;
               }
               return; // remain QMEKF-only
       
 case StartupStage::TunerWarm:
    // Start tuning as soon as frequency smoothing is "ready"
    // (tau depends only on f).
    if (!tuner_.isFreqReady()) return;

    // Only enter Live once BOTH freq + variance are ready
    // (so bias unfreeze / linear enable still waits for a sane sigma estimate).
    if (tuner_.isReady()) {
        enterLive_();
        // fallthrough to Live adaptation below
    }
    break;
          
          // now Live, continue with adaptation
   
           case StartupStage::Live:
               break;
        }

        // From here on, we are in Live stage.
        // Frequency as seen by the tuner
        float f_tune = tuner_.getFrequencyHz();
        if (!std::isfinite(f_tune) || f_tune < min_freq_hz_) {
            f_tune = min_freq_hz_;
        }
        if (f_tune > max_freq_hz_) {
            f_tune = max_freq_hz_;
        }

// If variance isn't ready yet, treat it as "noise floor only" so sigma doesn't go crazy.
float var_total = acc_noise_floor_sigma_ * acc_noise_floor_sigma_;
if (tuner_.isVarReady()) {
    var_total = std::max(0.0f, tuner_.getAccelVariance());
}

      
        const float var_noise = acc_noise_floor_sigma_ * acc_noise_floor_sigma_;
        float var_wave = var_total - var_noise;
        if (var_wave < 0.0f) var_wave = 0.0f;

        if (freq_stillness_.isStill()) {
            const float still_t = std::max(0.0f, freq_stillness_.getStillTime());
            constexpr float STILL_VAR_DECAY_SEC = 1.0f;
            float atten = std::exp(-still_t / STILL_VAR_DECAY_SEC);
            atten = std::min(std::max(atten, 0.0f), 1.0f);
            var_wave *= atten;
        }

        var_wave = std::max(var_wave, 1e-6f);
        float sigma_wave = std::sqrt(var_wave);

        float tau_raw = tau_coeff_ * 0.5f / f_tune;

        if (enable_clamp_) {
            tau_target_   = std::min(std::max(tau_raw,  min_tau_s_), max_tau_s_);
            sigma_target_ = std::min(sigma_wave * sigma_coeff_,      max_sigma_a_);
        } else {
            tau_target_   = tau_raw;
            sigma_target_ = sigma_wave;
        }

        float RS_raw = R_S_coeff_ * sigma_target_
                       * tau_target_ * tau_target_ * tau_target_;

        if (enable_clamp_) {
            RS_target_ = std::min(std::max(RS_raw, min_R_S_), max_R_S_);
        } else {
            RS_target_ = RS_raw;
        }

        adapt_mekf(dt, tau_target_, sigma_target_, RS_target_);
    }
    
    void adapt_mekf(float dt, float tau_t, float sigma_t, float RS_t) {
        const float alpha    = 1.0f - std::exp(-dt / adapt_tau_sec_);
        const float alpha_RS = 1.0f - std::exp(-dt / adapt_R_S_sec_);

        tune_.tau_applied   += alpha    * (tau_t   - tune_.tau_applied);
        tune_.sigma_applied += alpha    * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha_RS * (RS_t    - tune_.RS_applied);

if (time_ - last_adapt_time_sec_ > adapt_every_secs_) {
    // Push OU τ/σ whenever frequency has been learned by the tuner,
    // even in attitude-only mode (linear block OFF).
    if (tuner_.isFreqReady()) {
        apply_ou_tune_();
    }

    // Push R_S only when S pseudo-measurements are actually in play.
    if (startup_stage_ == StartupStage::Live && enable_linear_block_) {
        apply_RS_tune_();
    }

    last_adapt_time_sec_ = time_;
}

    }

    void enterCold_() {
        startup_stage_   = StartupStage::Cold;
        startup_stage_t_ = 0.0f;
    
        if (!mekf_) return;
    
        mekf_->set_linear_block_enabled(false);
    
        if (freeze_acc_bias_until_live_) {
            mekf_->set_acc_bias_updates_enabled(false);
            // Big Racc so early motion doesn't back-drive attitude too hard either
            mekf_->set_Racc(Eigen::Vector3f::Constant(Racc_warmup_));
            warmup_Racc_active_ = true;
        }
    }
    
    void enterLive_() {
        startup_stage_   = StartupStage::Live;
        startup_stage_t_ = 0.0f;
    
        if (!mekf_) return;
    
        // Enable linear block only if user wants it
        mekf_->set_linear_block_enabled(enable_linear_block_);
    
        if (freeze_acc_bias_until_live_) {
            mekf_->set_acc_bias_updates_enabled(true);
    
            // Restore nominal Racc if provided, otherwise leave whatever caller set
            if (warmup_Racc_active_ && Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
                mekf_->set_Racc(Racc_nominal_);
            }
            warmup_Racc_active_ = false;
        }
    
        // Push the latest τ/σ/R_S 
        apply_ou_tune_();                 // always useful
        if (enable_linear_block_) apply_RS_tune_();
     }

    StartupStage startup_stage_    = StartupStage::Cold;
    float        startup_stage_t_  = 0.0f;   // seconds since entering this stage

    // Warmup behavior
    bool  freeze_acc_bias_until_live_ = true;
    float Racc_warmup_               = 0.5f;   // big accel noise during warmup
    bool  warmup_Racc_active_         = false;
    Eigen::Vector3f Racc_nominal_     = Eigen::Vector3f::Constant(0.0f); // 0 => don't touch
   
    //  Members
    bool   with_mag_;
    double time_;
    double last_adapt_time_sec_;

    float freq_hz_       = FREQ_GUESS; // fast branch
    float freq_hz_slow_  = FREQ_GUESS; // slow branch
    float f_raw          = FREQ_GUESS;

    float a_vert_up = 0.0f; // accel vertical (Z-up)

    bool enable_clamp_ = true;
    bool enable_tuner_ = true;
    bool enable_heave_RS_gating_ = false; 
    bool enable_extra_drift_correction_ = false;

    // Controls whether the extended linear block [v,p,S,a_w] of Kalman3D_Wave
    // is ever enabled. When false, the underlying filter runs as a pure
    // attitude/bias QMEKF (linear states frozen, no OU, no S pseudo-measurements),
    // while all frequency tracking / tuner / direction logic still operates.
    bool enable_linear_block_ = true;

    // Dimensionless gain for extra drift correction (vertical).
    // Rough guideline: 0.2–1.0. Start modest and adjust by looking at heave drift vs noise.
    float extra_drift_gain_ = 0.5f;

    // Tunable adaptation parameters (initialized from global constexpr defaults)
    float min_freq_hz_            = MIN_FREQ_HZ;
    float max_freq_hz_            = MAX_FREQ_HZ;
    float min_tau_s_              = MIN_TAU_S;
    float max_tau_s_              = MAX_TAU_S;
    float max_sigma_a_            = MAX_SIGMA_A;
    float min_R_S_                = MIN_R_S;
    float max_R_S_                = MAX_R_S;
    float adapt_tau_sec_          = ADAPT_TAU_SEC;
    float adapt_R_S_sec_          = ADAPT_R_S_SEC;
    float adapt_every_secs_       = ADAPT_EVERY_SECS;
    float online_tune_warmup_sec_ = ONLINE_TUNE_WARMUP_SEC;
    float mag_delay_sec_          = MAG_DELAY_SEC;

    // Runtime-configurable anisotropy knobs
    float R_S_xy_factor_ = 0.17f;  // [0..1] scales XY pseudo-meas vs Z
    float S_factor_      = 1.7f;   // (>0) scales Σ_aw horizontal std vs vertical

    TrackingPolicy                  tracker_policy_{};
    FirstOrderIIRSmoother<float>    freq_fast_smoother_{FREQ_SMOOTHER_DT, 3.5f};   // ~3.5 s to 90% step
    FirstOrderIIRSmoother<float>    freq_slow_smoother_{FREQ_SMOOTHER_DT, 10.0f};  // ~10 s to 90% step
    SeaStateAutoTuner               tuner_;
    TuneState                       tune_;

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    // Runtime-configurable accel noise floor (1σ), m/s²
    float acc_noise_floor_sigma_ = ACC_NOISE_FLOOR_SIGMA_DEFAULT;

    float R_S_coeff_    = 1.2f;
    float tau_coeff_    = 1.4f;
    float sigma_coeff_  = 0.9f;  // Real noise inflates estimated sigma, to get more realistic sigma for OU we reduce it.

    std::unique_ptr<Kalman3D_Wave<float>>  mekf_;
    KalmanWaveDirection                              dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

    FreqInputLPF        freq_input_lpf_;   // LPF used only for tracker input
    StillnessAdapter    freq_stillness_;   // Detector of "still" mode

    WaveDirectionDetector<float> dir_sign_{0.002f, 0.005f};   // smoothing, sensitivity
    WaveDirection                dir_sign_state_ = UNCERTAIN;
};

template<TrackerType trackerT>
class SeaStateFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
    struct Config {
        bool with_mag = true;
    
        // Init / staging
        float mag_delay_sec          = MAG_DELAY_SEC;
        float online_tune_warmup_sec = ONLINE_TUNE_WARMUP_SEC;
    
        // If you want: fixed world mag ref (WMM), or “learn from measurement”
        bool  use_fixed_mag_world_ref = false;
        Eigen::Vector3f mag_world_ref = Eigen::Vector3f(0,0,0); // set to WMM result by caller if desired
    
        // Bias freeze behavior
        bool  freeze_acc_bias_until_live = true;
        float Racc_warmup = 0.5f;   // large accel noise during warmup to avoid eating motion as bias
    
        // Sensor noise
        Eigen::Vector3f sigma_a = Eigen::Vector3f(0.2f,0.2f,0.2f);
        Eigen::Vector3f sigma_g = Eigen::Vector3f(0.01f,0.01f,0.01f);
        Eigen::Vector3f sigma_m = Eigen::Vector3f(0.3f,0.3f,0.3f);
    };
  
    void begin(const Config& cfg) {
        cfg_ = cfg;
    
        // Reconfigure existing impl_ instead of reassigning it
        impl_.setWithMag(cfg.with_mag);
        impl_.setFreezeAccBiasUntilLive(cfg.freeze_acc_bias_until_live);
        impl_.setWarmupRacc(cfg.Racc_warmup);
        impl_.setMagDelaySec(cfg.mag_delay_sec);
        impl_.setOnlineTuneWarmupSec(cfg.online_tune_warmup_sec);
        impl_.initialize(cfg.sigma_a, cfg.sigma_g, cfg.sigma_m);
    
        // Optional: auto-restore Racc
        // impl_.setNominalRacc(Eigen::Vector3f(/* normal accel R */));
    
        t_ = 0.0f;
        stage_ = Stage::Uninitialized;
        mag_ref_set_ = false;
    }
  
    // One IMU sample
    void update(float dt,
                const Eigen::Vector3f& gyro_body_ned,
                const Eigen::Vector3f& acc_body_ned,
                float tempC = 35.0f)
    {
        if (!implReady_()) return;
        t_ += dt;
    
        // auto tilt-init on first IMU sample (hidden from client)
        if (stage_ == Stage::Uninitialized) {
            impl_.initialize_from_acc(acc_body_ned);
            stage_ = Stage::Warming;
            stage_t_ = 0.0f;
        } else {
            stage_t_ += dt;
        }
    
        // run normal IMU fusion (time + accel + tracker + tuner + direction, etc.)
        impl_.updateTime(dt, gyro_body_ned, acc_body_ned, tempC);
    
        if (stage_ == Stage::Warming && impl_.isAdaptiveLive()) {
            stage_ = Stage::Live;
        }
    
        // set mag world ref once (hidden)
        if (cfg_.with_mag && !mag_ref_set_ && t_ >= cfg_.mag_delay_sec) {
            if (cfg_.use_fixed_mag_world_ref) {
                impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
                mag_ref_set_ = true;
            } else {
                // learn-from-measurement: only possible once we’ve seen a mag sample
                if (mag_body_hold_.squaredNorm() > 1e-6f) {
                    const Eigen::Quaternionf q_bw = impl_.mekf().quaternion(); // body->world
                    impl_.mekf().set_mag_world_ref(q_bw * mag_body_hold_);
                    mag_ref_set_ = true;
                }
            }
        }
    }
  
    // Mag sample (can be called at different ODR)
    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (!implReady_()) return;
        mag_body_hold_ = mag_body_ned;
        // Only feed mag once reference is set AND delay passed
        if (cfg_.with_mag && mag_ref_set_) {
            impl_.updateMag(mag_body_ned);
        }
    }
  
    // Minimal getters client likely needs
    bool isLive() const { return stage_ == Stage::Live; }
    float freqHz() const { return impl_.getFreqHz(); }
    Eigen::Vector3f eulerNauticalDeg() const { return impl_.getEulerNautical(); }
  
    SeaStateFusionFilter<trackerT>& raw() { return impl_; } // optional “expert escape hatch”

private:
    enum class Stage { Uninitialized, Warming, Live };
  
    bool implReady_() const { return true; } // to guard begin()
  
    Config cfg_{};
    SeaStateFusionFilter<trackerT> impl_{false};
  
    Stage stage_ = Stage::Uninitialized;
    float t_ = 0.0f;
    float stage_t_ = 0.0f;
  
    bool mag_ref_set_ = false;
    Eigen::Vector3f mag_body_hold_ = Eigen::Vector3f::Zero();
};
