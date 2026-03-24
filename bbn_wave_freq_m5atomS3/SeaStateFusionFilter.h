#pragma once

/*
  Copyright (c) 2025  Mikhail Grushinskiy  
  Released under the MIT License 

  SeaStateFusionFilter
  
  Marine Inertial Navigational System (INS) Filter for IMU

  Combines multiple real-time estimators into a cohesive ocean-state tracker:

    • Quaternion-based attitude and linear motion estimation via Kalman3D_Wave_OU_III  

    • Dominant frequency tracking using one of:
          – AranovskiyFreqTracker     (frequency estimator)
          – KalmANFFreqTracker              (adaptive notch / Kalman frequency tracker)
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

#include "FirstOrderIIRSmoother.h"
#include "FrequencyTrackerPolicy.h"
#include "SeaStateAutoTuner.h"
#include "MagAutoTuner.h"
#include "Kalman3D_Wave_OU_III.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"
#include "TimeAwareSpikeFilter.h"

// Shared constants

extern const float g_std;

#ifndef FREQ_GUESS
#define FREQ_GUESS 0.3f
#endif

#ifndef ZERO_CROSSINGS_SCALE
#define ZERO_CROSSINGS_SCALE 1.0f
#endif

#ifndef ZERO_CROSSINGS_DEBOUNCE_TIME
#define ZERO_CROSSINGS_DEBOUNCE_TIME 0.12f
#endif

#ifndef ZERO_CROSSINGS_STEEPNESS_TIME
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.21f
#endif

// Estimated vertical accel noise floor (1σ), m/s².
// Tweak from bench data with IMU sitting still.
constexpr float ACC_NOISE_FLOOR_SIGMA_DEFAULT = 0.12f; 

constexpr float MIN_FREQ_HZ = 0.2f;
constexpr float MAX_FREQ_HZ = 6.0f;

constexpr float MIN_TAU_S   = 0.02f;
constexpr float MAX_TAU_S   = 3.0f;
constexpr float MAX_SIGMA_A = 6.0f;
constexpr float MIN_R_S     = 0.4f;
constexpr float MAX_R_S     = 35.0f;

constexpr float ADAPT_TAU_SEC              = 1.5f;
constexpr float ADAPT_EVERY_SECS           = 0.1f;
constexpr float ADAPT_RS_MULT              = 5.0f;   // dimensionless 
constexpr float ONLINE_TUNE_WARMUP_SEC     = 5.0f;
constexpr float MAG_DELAY_SEC              = 8.0f;

// Frequency smoother dt (SeaStateFusionFilter is designed for 200 Hz)
constexpr float FREQ_SMOOTHER_DT = 1.0f / 200.0f;

struct TuneState {
    float tau_applied   = 1.1f;    // s
    float sigma_applied = 1e-2f;    // m/s²
    float RS_applied    = 0.5f;     // m*s
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
        mekf_ = std::make_unique<Kalman3D_Wave_OU_III<float>>(sigma_a, sigma_g, sigma_m);
        enterCold_();      // applies freeze + warmup Racc + disables linear block
        apply_ou_tune_();
        mekf_->set_exact_att_bias_Qd(true);
    }

    void initialize_ext(const Eigen::Vector3f& sigma_a,
                        const Eigen::Vector3f& sigma_g,
                        const Eigen::Vector3f& sigma_m,
                        float Pq0, float Pb0,
                        float b0, float R_S_noise,
                        float gravity_magnitude) 
    {
        mekf_ = std::make_unique<Kalman3D_Wave_OU_III<float>>(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0, R_S_noise, gravity_magnitude);
        enterCold_();      // applies freeze + warmup Racc + disables linear block
        apply_ou_tune_();
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
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;
        time_ += dt;
        // Track time spent in current startup stage
        startup_stage_t_ += dt;
    
        // Keep BODY components around for direction/sign
        const float a_x_body = acc.x();
        const float a_y_body = acc.y();
        const float a_z_inertial = acc.z() + g_std;
    
        // MEKF updates first (attitude + latent a_w)
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc, tempC);

        {
            Eigen::Quaternionf q_bw = mekf_->quaternion_boat();
            q_bw.normalize();
        
            const Eigen::Vector3f z_body_down_world = q_bw * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            const Eigen::Vector3f z_world_down(0.0f, 0.0f, 1.0f);
        
            float cos_tilt = z_body_down_world.normalized().dot(z_world_down);
            cos_tilt = std::max(-1.0f, std::min(1.0f, cos_tilt));
            const float tilt_deg = std::acos(cos_tilt) * 57.295779513f;
        
            constexpr float TILT_RESET_DEG = 70.0f;
            constexpr float TILT_RESET_HOLD_SEC = 0.35f;
            constexpr float TILT_RESET_COOLDOWN_SEC = 3.0f;

            if (tilt_reset_cooldown_sec_ > 0.0f) {
                tilt_reset_cooldown_sec_ = std::max(0.0f, tilt_reset_cooldown_sec_ - dt);
            }

            if (tilt_deg > TILT_RESET_DEG) {
                tilt_over_limit_sec_ += dt;
            } else {
                // decay quickly on recovery so brief transients do not trigger resets
                tilt_over_limit_sec_ = std::max(0.0f, tilt_over_limit_sec_ - 2.0f * dt);
            }

            if (tilt_over_limit_sec_ >= TILT_RESET_HOLD_SEC && tilt_reset_cooldown_sec_ <= 0.0f) {
                // Re-lock attitude to gravity.
                mekf_->initialize_from_acc(acc);

                // Only force full Cold re-entry while not yet fully live. In Live, keep
                // linear states running to avoid long "frozen heave" plateaus.
                if (startup_stage_ != StartupStage::Live) {
                    enterCold_();
                    resetTrackingState_();
                }

                tilt_over_limit_sec_ = 0.0f;
                tilt_reset_cooldown_sec_ = TILT_RESET_COOLDOWN_SEC;
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

        // Keep linear-block R_S tuning responsive in Live mode instead of
        // waiting for slow adaptation cadence.
        if (startup_stage_ == StartupStage::Live && enable_linear_block_) {
            apply_RS_tune_();
            applyDriftPseudoMeasurements_(dt, acc, a_vert_up); 
        }
    
        const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz_;
      
        // Direction filters run on BODY accel, "sign" uses vertical acceleration
        dir_filter_.update(a_x_body, a_y_body, omega, dt);
        dir_sign_state_ = dir_sign_.update(a_x_body, a_y_body, a_vert_up, dt);
    }

    //  Magnetometer correction
    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (!with_mag_ || !mekf_) return;
        if (time_ < mag_delay_sec_) return;
    
        mekf_->measurement_update_mag_only(mag_body_ned);
        mag_updates_applied_++;
    
        if (!std::isfinite(first_mag_update_time_)) {
            first_mag_update_time_ = static_cast<float>(time_);
        }
     
        // We can "unlock" once mag has had a few updates, but we DO NOT
        // enable accel-bias learning or restore Racc unless we're already Live.
        if (accel_bias_locked_ &&
            startup_stage_ == StartupStage::Live &&
            mag_updates_applied_ >= MAG_UPDATES_TO_UNLOCK &&
            std::isfinite(first_mag_update_time_) &&
            (static_cast<float>(time_) - first_mag_update_time_) > 1.0f) // 1s guard
        {
            accel_bias_locked_ = false;
    
            // Only allow accel bias to start learning once the system is Live.
            if (freeze_acc_bias_until_live_ && startup_stage_ == StartupStage::Live) {
                mekf_->set_acc_bias_updates_enabled(true);
    
                // Restore nominal Racc only when bias learning is allowed.
                if (warmup_Racc_active_) {
                    if (Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
                        mekf_->set_Racc(Racc_nominal_);
                        warmup_Racc_active_ = false;
                    }
                }
            }
        }
    }

    // Drift pseudo-measurements bundle
    //
    // All std dev are derived from current (sigma_a, tau) coming from the tuner:
    //   pos std  ~  (sigma_a * tau^2) * sigma_mult
    //   vel std  ~  (sigma_a * tau)   * sigma_mult
    //
    // Each pseudo-meas has:
    //   enabled, period_steps, sigma_mult, gate_scale (threshold vs envelope).
    //
    struct DriftPseudoCfg {
        bool  enabled      = false;
        int   period_steps = 6;      // cadence in IMU cycles
        float sigma_mult   = 10.0f;  // dimensionless multiplier (default HIGH = light touch)
        float sigma_min    = 0.02f;  // absolute floor (units depend on measurement)
        float gate_scale   = 1.2f;   // envelope gate (e.g. 1.2 => breach at 120% envelope)
    };
    
    void setPseudoVzZeroOnPosBreachCfg(const DriftPseudoCfg& c) { pm_vz_zero_on_pos_breach_ = c; }
    void setPseudoPosZeroCfg(const DriftPseudoCfg& c)           { pm_pos_zero_              = c; }
    void setPseudoVzClampCfg(const DriftPseudoCfg& c)           { pm_vz_clamp_              = c; }
    void setPseudoHarmonicPosCfg(const DriftPseudoCfg& c)       { pm_harmonic_pos_          = c; }
    
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
            const float prev = R_S_coeff_;
            R_S_coeff_ = c;

            // Keep runtime behavior responsive when the coefficient is changed online.
            // Otherwise RS_applied can remain near its previous value for several
            // adaptation time constants and look like this setter has no effect.
            if (std::isfinite(prev) && prev > 0.0f) {
                const float scale = c / prev;

                if (std::isfinite(tune_.RS_applied) && tune_.RS_applied > 0.0f) {
                    tune_.RS_applied *= scale;
                }
                if (std::isfinite(RS_target_) && RS_target_ > 0.0f) {
                    RS_target_ *= scale;
                }

                if (enable_linear_block_) {
                    apply_RS_tune_();
                }
            }
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

    // Configure LPF on vertical accel for tracker input
    void setFreqInputCutoffHz(float fc) {
        freq_input_lpf_.setCutoff(fc);
    }

    void enableClamp(bool flag = true) {
        enable_clamp_ = flag;
    }
    void enableTuner(bool flag = true) {
        enable_tuner_ = flag;
    }

    void setHarmonicPositionDespikeConfig(int window_size, float threshold) {
        if (window_size < 3) return;
        if (!std::isfinite(threshold) || threshold <= 0.0f) return;
        harmonic_despike_window_ = window_size;
        harmonic_despike_threshold_ = threshold;
        initHarmonicDespikeFilters_();
    }

    // Enable/disable use of the extended linear block [v,p,S,a_w] in Kalman3D_Wave_OU_III.
    //
    // When flag == false:
    //   • Kalman3D_Wave_OU_III::set_linear_block_enabled(false) is enforced,
    //   • startup stages still progress (Cold → TunerWarm → Live),
    //   • tuner still runs and exposes tau/sigma/R_S targets & metrics,
    //   • but OU / v/p/S/a_w propagation and S-pseudo-measurements are never used.
    //
    // When flag == true (default):
    //   • the linear block is enabled 
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

    void setAdaptationTimeConstants(float tau_sec) {
        if (std::isfinite(tau_sec) && tau_sec > 0.0f)   adapt_tau_sec_   = tau_sec;
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

    inline float getHeaveAbs() const noexcept { if (!mekf_) return NAN; return std::fabs(mekf_->get_position().z()); }

    inline float getDisplacementScale(bool smoothed = true) const noexcept {
        const float tau = smoothed ? tune_.tau_applied : tau_target_;
        const float sigma = smoothed ? tune_.sigma_applied : sigma_target_;
        if (!std::isfinite(sigma) || !std::isfinite(tau)) return NAN;
        constexpr float C_HS  = 2.0f * std::sqrt(2.0f) / (M_PI * M_PI);  // Longuet–Higgins envelope for wave height
        return C_HS * sigma * tau * tau / 2.0f;
    }

    float getVerticalSpeedEnvelopeMps(bool smoothed = true) const noexcept {
        const float tau   = smoothed ? tune_.tau_applied   : tau_target_;
        const float sigma = smoothed ? tune_.sigma_applied : sigma_target_;
        if (!(tau > 1e-6f) || !std::isfinite(tau) || !std::isfinite(sigma)) return NAN;
        // RMS of Rayleigh envelope amplitude for narrowband Gaussian v(t):
        // v_env_rms = sqrt(2) * sigma_v,  sigma_v ≈ sigma_a / omega,  omega = pi/tau
        constexpr float K = std::sqrt(2.0f) / M_PI;
        const float v_env = speed_env_mult_ * K * sigma * tau;
        return std::isfinite(v_env) ? v_env : NAN;
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
    static inline float sgnf_(float x) noexcept { return (x >= 0.0f) ? 1.0f : -1.0f; }

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

    void apply_RS_tune_(float rs_scale = 1.0f) {
        if (!mekf_) return;
        const float s = (std::isfinite(rs_scale) && rs_scale > 0.0f)
                        ? std::min(rs_scale, 1.0f)
                        : 1.0f;
        const float RSb = std::min(std::max(tune_.RS_applied, min_R_S_), max_R_S_);
        const float rs_xy = RSb * s * R_S_xy_factor_;
        mekf_->set_RS_noise(Eigen::Vector3f(
            rs_xy,
            rs_xy,
            RSb * s
        ));
    }

    void initHarmonicDespikeFilters_() {
        despike_ax_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
        despike_ay_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
        despike_az_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
    }

    void resetDriftCorrections_() {
        pm_ctr_vz_zero_  = 0;
        pm_ctr_pos_zero_ = 0;
        pm_ctr_vz_clamp_ = 0;
        pm_ctr_harmonic_pos_ = 0;
        initHarmonicDespikeFilters_();
    }

    void applyHarmonicPositionCorrection_(float dt,
                                          const Eigen::Vector3f& acc_body_ned,
                                          float a_vert_up_osc)
    {
        if (!mekf_) return;
        if (!pm_harmonic_pos_.enabled) return;
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;
    
        // Envelope gate (drift-risk only)
        float env_scale = getDisplacementScale(true);
        if (!std::isfinite(env_scale) || env_scale <= 0.0f) env_scale = 1.0f;
    
        const float absz = std::fabs(mekf_->get_position().z());
        const float gate = std::max(0.0f, pm_harmonic_pos_.gate_scale) * env_scale;
        if (!(absz > gate)) return;
    
        // Cadence (same pattern as other pm_* corrections)
        if (++pm_ctr_harmonic_pos_ < std::max(1, pm_harmonic_pos_.period_steps)) return;
        pm_ctr_harmonic_pos_ = 0;
    
        // Frequency for harmonic proxy
        const float harmonic_freq_hz =
            (std::isfinite(freq_hz_) && freq_hz_ > 0.0f) ? freq_hz_ : freq_hz_slow_;
    
        const float omega = 2.0f * static_cast<float>(M_PI) *
                            std::max(harmonic_freq_hz, min_freq_hz_);
        const float omega_sq = omega * omega;
        if (!(omega_sq > 1e-4f) || !std::isfinite(omega_sq)) return;
    
        if (!acc_body_ned.allFinite()) return;
    
        // “no backfeed”: use raw IMU accel (no attitude rotation), but use oscillatory vertical for z.
        const Eigen::Vector3f a_world_ned = acc_body_ned;
    
        if (!despike_ax_ || !despike_ay_ || !despike_az_) {
            initHarmonicDespikeFilters_();
        }
    
        // NED-down oscillatory vertical accel
        const float a_z_ned_osc = -a_vert_up_osc;
    
        const Eigen::Vector3f a_despiked(
            despike_ax_->filterWithDelta(a_world_ned.x(), dt),
            despike_ay_->filterWithDelta(a_world_ned.y(), dt),
            despike_az_->filterWithDelta(a_z_ned_osc, dt)
        );
        if (!a_despiked.allFinite()) return;
    
        // p ≈ -a / ω² (componentwise, in NED)
        const Eigen::Vector3f p_meas = -a_despiked / omega_sq;
        if (!p_meas.allFinite()) return;
    
        // position std ≈ (sigma_a * tau^2) * sigma_mult, with anisotropy via S_factor_.
        Eigen::Vector3f sig = sigmaP_fromSigmaTau_(pm_harmonic_pos_.sigma_mult);
    
        const float smin = std::max(1e-6f, pm_harmonic_pos_.sigma_min);
        sig.x() = std::max(sig.x(), smin);
        sig.y() = std::max(sig.y(), smin);
        sig.z() = std::max(sig.z(), smin);
    
        mekf_->measurement_update_position_pseudo(p_meas, sig);
    }

    static Eigen::Vector3f harmonicPseudoPositionFromAccel_(const Eigen::Vector3f& a_ned, float omega_sq) {
        return -a_ned / omega_sq;
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

        // From here on, we are in TunerWarm or Live: compute targets and adapt.
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
        // Keep published sigma_target_ sane before variance is ready
        if (!tuner_.isVarReady()) {
            sigma_target_ = std::max(sigma_target_, std::max(0.05f, acc_noise_floor_sigma_));
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
        const float alpha = 1.0f - std::exp(-dt / adapt_tau_sec_);
    
        // R_S smoothing depends on tau (tau_t is already clamped upstream)
        const float RS_sec   = ADAPT_RS_MULT * tau_t;     // or tune_.tau_applied if preferred
        const float alpha_RS = 1.0f - std::exp(-dt / RS_sec);
    
        tune_.tau_applied   += alpha    * (tau_t   - tune_.tau_applied);
        tune_.sigma_applied += alpha    * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha_RS * (RS_t    - tune_.RS_applied);
    
        if (time_ - last_adapt_time_sec_ > adapt_every_secs_) {
            if (tuner_.isFreqReady()) {
                apply_ou_tune_();
            }
            if (startup_stage_ == StartupStage::Live && enable_linear_block_) {
                apply_RS_tune_();
            }
            last_adapt_time_sec_ = time_;
        }
    }

    void resetTrackingState_() {
        // Reset *all* slow/statistical machinery (not just some of it)
        tracker_policy_       = TrackingPolicy{};
        freq_input_lpf_       = FreqInputLPF{};
        freq_stillness_       = StillnessAdapter{};
        freq_input_lpf_.setCutoff(max_freq_hz_);
        freq_stillness_.setTargetFreqHz(min_freq_hz_);

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

        resetDriftCorrections_();
    }
    
    void enterCold_() {
        startup_stage_   = StartupStage::Cold;
        startup_stage_t_ = 0.0f;

        if (!mekf_) return;
        mekf_->set_linear_block_enabled(false);
    
        accel_bias_locked_   = with_mag_;
        mag_updates_applied_ = 0;  
        first_mag_update_time_  = NAN;
      
        // optionally: warmup_Racc_active_ 
        if (freeze_acc_bias_until_live_) {
            mekf_->set_acc_bias_updates_enabled(false);
            mekf_->set_Racc(Eigen::Vector3f::Constant(Racc_warmup_));
            warmup_Racc_active_ = true;
        }

        resetDriftCorrections_();
    }

    void enterLive_() {
        startup_stage_   = StartupStage::Live;
        startup_stage_t_ = 0.0f;
    
        if (!mekf_) return;
        mekf_->set_linear_block_enabled(enable_linear_block_);
    
        if (freeze_acc_bias_until_live_) {
            // Bias learning may remain locked until magnetometer has stabilized,
            // but accel measurement noise should still return to nominal in Live.
            const bool allow_bias = !accel_bias_locked_;
            mekf_->set_acc_bias_updates_enabled(allow_bias);

            if (warmup_Racc_active_ && Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
                mekf_->set_Racc(Racc_nominal_);
            }
            warmup_Racc_active_ = false;
        }
        apply_ou_tune_();
        if (enable_linear_block_) apply_RS_tune_();
      
        resetDriftCorrections_();
    }

    StartupStage startup_stage_    = StartupStage::Cold;
    float        startup_stage_t_  = 0.0f;   // seconds since entering this stage

    // Warmup behavior
    bool  freeze_acc_bias_until_live_ = true;
    float Racc_warmup_               = 0.5f;   // big accel noise during warmup
    bool  warmup_Racc_active_         = false;
    Eigen::Vector3f Racc_nominal_     = Eigen::Vector3f::Constant(0.0f); // 0 => don't touch

    bool accel_bias_locked_ = true;
    int  mag_updates_applied_ = 0;
    static constexpr int MAG_UPDATES_TO_UNLOCK = 40;  // tune as you like

    //  Members
    bool   with_mag_;
    double time_;
    double last_adapt_time_sec_;

    float first_mag_update_time_ = NAN;

    // Tilt-reset gate persistence/cooldown to avoid false transient resets.
    float tilt_over_limit_sec_ = 0.0f;
    float tilt_reset_cooldown_sec_ = 0.0f;

    float freq_hz_       = FREQ_GUESS; // fast branch
    float freq_hz_slow_  = FREQ_GUESS; // slow branch
    float f_raw          = FREQ_GUESS;

    float a_vert_up = 0.0f; // accel vertical (Z-up)

    bool enable_clamp_ = true;
    bool enable_tuner_ = true;

    float harmonic_despike_threshold_ = 4.0f;
    int harmonic_despike_window_ = 5;
    std::unique_ptr<TimeAwareSpikeFilter> despike_ax_;
    std::unique_ptr<TimeAwareSpikeFilter> despike_ay_;
    std::unique_ptr<TimeAwareSpikeFilter> despike_az_;

    // drift pseudo-measurement configs
    DriftPseudoCfg pm_vz_zero_on_pos_breach_ {
        /*enabled*/ true, /*period*/ 3, /*sigma_mult*/ 2.0f, /*sigma_min*/ 0.03f, /*gate*/ 0.8f
    };
    DriftPseudoCfg pm_pos_zero_ {
        /*enabled*/ true, /*period*/ 5, /*sigma_mult*/ 8.0f, /*sigma_min*/ 0.05f, /*gate*/ 0.3f
    };
    DriftPseudoCfg pm_vz_clamp_ {
        /*enabled*/ true, /*period*/ 3, /*sigma_mult*/ 2.0f, /*sigma_min*/ 0.03f, /*gate*/ 0.9f
    };
    DriftPseudoCfg pm_harmonic_pos_ {
        /*enabled*/ true, /*period*/ 5, /*sigma_mult*/ 6.0f, /*sigma_min*/ 0.05f, /*gate*/ 0.10f
    };
      
    float speed_env_mult_ = 1.0f;   // v_env ≈ speed_env_mult * ω * z_env

    int pm_ctr_vz_zero_ = 0;
    int pm_ctr_pos_zero_ = 0;
    int pm_ctr_vz_clamp_ = 0;
    int pm_ctr_harmonic_pos_ = 0;

    // Controls whether the extended linear block [v,p,S,a_w] of Kalman3D_Wave_OU_III
    // is ever enabled. When false, the underlying filter runs as a pure
    // attitude/bias QMEKF (linear states frozen, no OU, no S pseudo-measurements),
    // while all frequency tracking / tuner / direction logic still operates.
    bool enable_linear_block_ = true;

    // Tunable adaptation parameters (initialized from global constexpr defaults)
    float min_freq_hz_            = MIN_FREQ_HZ;
    float max_freq_hz_            = MAX_FREQ_HZ;
    float min_tau_s_              = MIN_TAU_S;
    float max_tau_s_              = MAX_TAU_S;
    float max_sigma_a_            = MAX_SIGMA_A;
    float min_R_S_                = MIN_R_S;
    float max_R_S_                = MAX_R_S;
    float adapt_tau_sec_          = ADAPT_TAU_SEC;
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

    std::unique_ptr<Kalman3D_Wave_OU_III<float>>  mekf_;
    KalmanWaveDirection                    dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

    FreqInputLPF        freq_input_lpf_;   // LPF used only for tracker input
    StillnessAdapter    freq_stillness_;   // Detector of "still" mode

    WaveDirectionDetector<float> dir_sign_{0.002f, 0.005f};   // smoothing, sensitivity
    WaveDirection                dir_sign_state_ = UNCERTAIN;

    void applyDriftPseudoMeasurements_(float dt,
                                       const Eigen::Vector3f& acc_body_ned,
                                       float a_vert_up_osc)
    {
        if (!mekf_) return;
        if (startup_stage_ != StartupStage::Live) return;
        if (!enable_linear_block_) return;
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;
    
        // Displacement envelope (your existing model)
        float z_env = getDisplacementScale(true);
        if (!std::isfinite(z_env) || z_env <= 0.0f) z_env = 0.3f;
    
        const float pz = mekf_->get_position().z();     // NED down
        const float absz = std::fabs(pz);

        // Harmonic approximation
        if (pm_harmonic_pos_.enabled) {
            applyHarmonicPositionCorrection_(dt, acc_body_ned, a_vert_up_osc);
        }
    
        // On displacement envelope breach: pseudo-measure v_z -> 0
        if (pm_vz_zero_on_pos_breach_.enabled) {
            if (++pm_ctr_vz_zero_ >= std::max(1, pm_vz_zero_on_pos_breach_.period_steps)) {
                pm_ctr_vz_zero_ = 0;
                const float gate = std::max(0.05f, pm_vz_zero_on_pos_breach_.gate_scale) * z_env;
                if (absz > gate) {
                    const float vz = mekf_->get_velocity().z(); // NED down
                    // Apply only if moving AWAY from sea level (z=0): pz and vz same sign => |pz| increasing
                    // (If pz>0 and vz>0 => going down further; pz<0 and vz<0 => going up further.)
                    if (std::isfinite(vz) && (pz * vz > 0.0f)) {
                        const float sigma_vz = std::max(
                            sigmaV_fromSigmaTau_(pm_vz_zero_on_pos_breach_.sigma_mult).z(),
                            pm_vz_zero_on_pos_breach_.sigma_min
                        );
                        mekf_->measurement_update_vert_velocity_pseudo(0.0f, sigma_vz);
                    }
                }
            }
        }
    
        // Direct displacement zero (z only), gentle, high sigma by default
        if (pm_pos_zero_.enabled) {
            if (++pm_ctr_pos_zero_ >= std::max(1, pm_pos_zero_.period_steps)) {
                pm_ctr_pos_zero_ = 0;  
                const float gate = std::max(0.05f, pm_pos_zero_.gate_scale) * z_env;
                if (absz > gate) {
                    const float sigma_z = std::max(
                        sigmaP_fromSigmaTau_(pm_pos_zero_.sigma_mult).z(),
                        pm_pos_zero_.sigma_min
                    );
                    // Only affect z: keep x/y "measured" at their current predicted values
                    Eigen::Vector3f p_meas = mekf_->get_position();
                    p_meas.z() = 0.0f;
                    const float BIG = 1e5f;
                    Eigen::Vector3f sig(BIG, BIG, sigma_z);
                    mekf_->measurement_update_position_pseudo(p_meas, sig);
                }
            }
        }
    
        // Speed envelope estimate + clamp v_z when breached
        if (pm_vz_clamp_.enabled) {
            if (++pm_ctr_vz_clamp_ >= std::max(1, pm_vz_clamp_.period_steps)) {
                pm_ctr_vz_clamp_ = 0;
                const float v_env = getVerticalSpeedEnvelopeMps(true);
                if (std::isfinite(v_env) && v_env > 0.0f) {
                    const float vz = mekf_->get_velocity().z();  // NED down
                    const float abs_vz = std::fabs(vz);
                    const float gate_v = std::max(0.05f, pm_vz_clamp_.gate_scale) * v_env;
                    if (abs_vz > gate_v) {
                        const float vz_clamped = (vz >= 0.0f) ? std::min(vz, v_env) : std::max(vz, -v_env);
                        const float sigma_vz = std::max(
                            sigmaV_fromSigmaTau_(pm_vz_clamp_.sigma_mult).z(),
                            pm_vz_clamp_.sigma_min
                        );
                        // Only affect z velocity (keep x/y at predicted)
                        Eigen::Vector3f v_meas = mekf_->get_velocity();
                        v_meas.z() = vz_clamped;
                        const float BIG = 1e6f;
                        Eigen::Vector3f sig(BIG, BIG, sigma_vz);
                        mekf_->measurement_update_velocity_pseudo(v_meas, sig);
                    }
                }
            }
        }
    }
      
    static inline float clampf_(float x, float lo, float hi) {
        return std::max(lo, std::min(hi, x));
    }
    
    Eigen::Vector3f sigmaP_fromSigmaTau_(float sigma_mult) const {
        const float tau = std::max(getTauApplied(), 1e-3f);
        const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
        const float sZ = std::max(sigma_floor, getSigmaApplied());
        const float sH = sZ * S_factor_;
    
        const float k = std::max(0.0f, sigma_mult) * (tau * tau);
        return Eigen::Vector3f(sH * k, sH * k, sZ * k); // meters
    }
    
    Eigen::Vector3f sigmaV_fromSigmaTau_(float sigma_mult) const {
        const float tau = std::max(getTauApplied(), 1e-3f);
        const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
        const float sZ = std::max(sigma_floor, getSigmaApplied());
        const float sH = sZ * S_factor_;
    
        const float k = std::max(0.0f, sigma_mult) * tau;
        return Eigen::Vector3f(sH * k, sH * k, sZ * k); // m/s
    }
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
        Eigen::Vector3f mag_world_ref = Eigen::Vector3f(0,0,0); // caller sets (e.g. WMM)

        // Bias freeze behavior
        bool  freeze_acc_bias_until_live = true;
        float Racc_warmup = 0.5f;

        // Sensor noise
        Eigen::Vector3f sigma_a = Eigen::Vector3f(0.2f,0.2f,0.2f);
        Eigen::Vector3f sigma_g = Eigen::Vector3f(0.01f,0.01f,0.01f);
        Eigen::Vector3f sigma_m = Eigen::Vector3f(0.3f,0.3f,0.3f);

        // How long after mag_delay we’re willing to wait for MagAutoTuner
        float mag_ref_timeout_sec = 4.5f; // fallback guard

        // Used only if dt_mag can’t be inferred
        float mag_odr_guess_hz = 80.0f;

        // MagAutoTuner config override (optional)
        bool use_custom_mag_tuner_cfg = false;
        MagAutoTuner::Config mag_tuner_cfg{};
    };

    void begin(const Config& cfg) {
        cfg_ = cfg;

        // Reset wrapper state
        begun_ = true;
        stage_ = Stage::Uninitialized;
        t_ = 0.0f;
        stage_t_ = 0.0f;

        mag_ref_set_ = false;
        mag_body_hold_.setZero();
        last_mag_time_sec_ = NAN;
        dt_mag_sec_ = NAN;
        mag_ref_deadline_sec_ = cfg_.mag_delay_sec + cfg_.mag_ref_timeout_sec;

        // Fallback running means (used only if MagAutoTuner is not ready by timeout)
        fallback_acc_mean_.setZero();
        fallback_mag_mean_.setZero();
        fallback_mean_count_ = 0;

        // Reset tuner
        if (cfg_.use_custom_mag_tuner_cfg) {
            mag_auto_.setConfig(cfg_.mag_tuner_cfg);    
        } else {
            mag_auto_.reset();
            cfg_.mag_tuner_cfg = makeDefaultMagInitCfg(cfg_.mag_odr_guess_hz);   
            mag_auto_.setConfig(cfg_.mag_tuner_cfg);
        }
            
        // Track last IMU samples for gating
        last_acc_body_ned_.setZero();
        last_gyro_body_ned_.setZero();
        last_imu_dt_ = NAN;
        have_last_imu_ = false;

        // Configure internal impl without reassign
        impl_.setWithMag(cfg.with_mag);
        impl_.setFreezeAccBiasUntilLive(cfg.freeze_acc_bias_until_live);
        impl_.setWarmupRacc(cfg.Racc_warmup);
        impl_.setMagDelaySec(cfg.mag_delay_sec);
        impl_.setOnlineTuneWarmupSec(cfg.online_tune_warmup_sec);

        impl_.initialize(cfg.sigma_a, cfg.sigma_g, cfg.sigma_m);
        last_impl_startup_stage_ = impl_.getStartupStage();

        // IMPORTANT: allow warmup to restore nominal accel measurement noise
        impl_.setNominalRacc(cfg.sigma_a);
    }

    // One IMU sample
    void update(float dt,
                const Eigen::Vector3f& gyro_body_ned,
                const Eigen::Vector3f& acc_body_ned,
                float tempC = 35.0f)
    {
        if (!begun_) return;
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;

        t_ += dt;

        // auto tilt-init on first IMU sample
        if (stage_ == Stage::Uninitialized) {
            impl_.initialize_from_acc(acc_body_ned);
            stage_ = Stage::Warming;
            stage_t_ = 0.0f;
        } else {
            stage_t_ += dt;
        }

        // Store last IMU samples for MagAutoTuner gating
        last_acc_body_ned_  = acc_body_ned;
        last_gyro_body_ned_ = gyro_body_ned;
        last_imu_dt_        = dt;
        have_last_imu_      = true;

        // Normal IMU fusion
        impl_.updateTime(dt, gyro_body_ned, acc_body_ned, tempC);

        // If internal filter fell back to Cold (tilt reset), force mag ref re-learn
const auto cur_stage = impl_.getStartupStage();

if (cur_stage != last_impl_startup_stage_) {
    if (cur_stage == SeaStateFusionFilter<trackerT>::StartupStage::Cold) {
        // Entered Cold (startup or non-Live tilt reset): reset mag-init ONCE
        mag_ref_set_ = false;
        mag_auto_.reset();

        last_mag_time_sec_ = NAN;
        dt_mag_sec_ = NAN;

        fallback_acc_mean_.setZero();
        fallback_mag_mean_.setZero();
        fallback_mean_count_ = 0;

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

        // Track effective magnetometer sample period.
        if (std::isfinite(last_mag_time_sec_)) {
            dt_mag_sec_ = t_ - last_mag_time_sec_;
        }
        last_mag_time_sec_ = t_;

        // Learning path: accumulate only stable acc+mag+gyro samples.
        if (have_last_imu_) {
            float dtm = dt_mag_sec_;
            if (!std::isfinite(dtm) || dtm <= 0.0f) {
                dtm = 1.0f / std::max(1.0f, cfg_.mag_odr_guess_hz);
            }
            (void)mag_auto_.addMagSample(dtm, last_acc_body_ned_, mag_body_ned, last_gyro_body_ned_);
        }

        // Keep robust running means as a timeout fallback (method-level safety net).
        // This avoids latching world magnetic reference from one instantaneous sample.
        if (have_last_imu_) {
            const float an = last_acc_body_ned_.norm();
            const float mn = mag_body_ned.norm();
            const float g = 9.80665f;
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

        // Respect mag delay for actual mag fusion, but keep learning active
        // from startup so the first post-delay reference is better conditioned.
        if (t_ < cfg_.mag_delay_sec) return;

        if (!mag_ref_set_) {
            if (cfg_.use_fixed_mag_world_ref) {
                impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
                mag_ref_set_ = true;
            } else {
                // If a valid world-field prior exists (e.g., WMM), use it immediately
                // once mag fusion window opens. This avoids prolonged yaw drift while
                // waiting for learned alignment under rough sea/noisy startup.
                if (cfg_.mag_world_ref.allFinite() && cfg_.mag_world_ref.norm() > 1e-3f) {
                    impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
                    mag_ref_set_ = true;
                }

                Eigen::Vector3f mag_body_for_ref = mag_body_ned;
                Eigen::Vector3f acc_for_ref = last_acc_body_ned_;
                bool have_ref_candidate = false;

                // Preferred: learned relation from stable samples only.
                Eigen::Vector3f acc_mean, mag_raw_mean, mag_unit_mean;
                if (!mag_ref_set_ && mag_auto_.getResult(acc_mean, mag_raw_mean, mag_unit_mean)) {
                    if (acc_mean.allFinite() && acc_mean.norm() > 1e-3f &&
                        mag_raw_mean.allFinite() && mag_raw_mean.norm() > 1e-3f)
                    {
                        acc_for_ref = acc_mean;
                        mag_body_for_ref = mag_raw_mean;
                        have_ref_candidate = true;
                    }
                }

                // Timeout fallback policy:
                //  1) Prefer caller-provided world-field prior if valid.
                //  2) Else use robust running means (many samples), never one-shot sample.
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
                        acc_for_ref = fallback_acc_mean_;
                        mag_body_for_ref = fallback_mag_mean_;
                        have_ref_candidate = true;
                    } else {
                        // Keep waiting; extend deadline to avoid busy re-checking.
                        mag_ref_deadline_sec_ = t_ + cfg_.mag_ref_timeout_sec;
                    }
                }

                if (!mag_ref_set_ && have_ref_candidate) {
                    Eigen::Quaternionf q_tilt = tiltOnlyQuatFromAccel_(acc_for_ref);
                    q_tilt.normalize();

                    Eigen::Vector3f mag_world_ref_uT = q_tilt * mag_body_for_ref; // keep raw uT
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

    // Minimal getters
    bool  isLive() const { return stage_ == Stage::Live; }
    float freqHz() const { return impl_.getFreqHz(); }
    Eigen::Vector3f eulerNauticalDeg() const { return impl_.getEulerNautical(); }

    SeaStateFusionFilter<trackerT>& raw() { return impl_; }

private:
    enum class Stage { Uninitialized, Warming, Live };

    // Tilt-only (yaw-free) quaternion body->world from accel.
    // We assume “rest accel” direction represents gravity (up to sign convention),
    // so we align BODY measured down with WORLD down (NED: +Z down).
    static Eigen::Quaternionf tiltOnlyQuatFromAccel_(const Eigen::Vector3f& acc_body_ned) {
        // At rest, specific force typically points ~ -g in world coordinates.
        // But your code already deals with sign mismatches elsewhere, so we do:
        //   body_down ≈ -(acc / |acc|)
        Eigen::Vector3f a = acc_body_ned;
        const float an = a.norm();
        if (!(an > 1e-6f) || !a.allFinite()) {
            return Eigen::Quaternionf::Identity();
        }

        Eigen::Vector3f body_down = (-a / an);              // body frame "down" direction
        const Eigen::Vector3f world_down(0.0f, 0.0f, 1.0f); // NED down

        // Quaternion from vector u -> v (shortest arc)
        const float d = std::max(-1.0f, std::min(1.0f, body_down.dot(world_down)));
        Eigen::Vector3f axis = body_down.cross(world_down);
        const float axis_n = axis.norm();

        if (axis_n < 1e-6f) {
            // parallel or anti-parallel
            if (d > 0.0f) {
                return Eigen::Quaternionf::Identity();
            } else {
                // 180 deg flip around any axis perpendicular to body_down
                Eigen::Vector3f ortho = std::fabs(body_down.z()) < 0.9f
                    ? Eigen::Vector3f(0,0,1).cross(body_down)
                    : Eigen::Vector3f(0,1,0).cross(body_down);
                ortho.normalize();
                return Eigen::Quaternionf(Eigen::AngleAxisf(float(M_PI), ortho));
            }
        }

        axis /= axis_n;
        const float angle = std::acos(d);
        Eigen::Quaternionf q(Eigen::AngleAxisf(angle, axis));
        q.normalize();
        return q;
    }

private:
    Config cfg_{};
    SeaStateFusionFilter<trackerT> impl_{false};

    bool begun_ = false;

    Stage stage_ = Stage::Uninitialized;
    float t_ = 0.0f;
    float stage_t_ = 0.0f;
    typename SeaStateFusionFilter<trackerT>::StartupStage last_impl_startup_stage_ =
             SeaStateFusionFilter<trackerT>::StartupStage::Cold;

    // Mag init state
    bool mag_ref_set_ = false;
    Eigen::Vector3f mag_body_hold_ = Eigen::Vector3f::Zero();

    float last_mag_time_sec_ = NAN;
    float dt_mag_sec_ = NAN;
    float mag_ref_deadline_sec_ = NAN;

    // Last IMU samples (for MagAutoTuner gating)
    Eigen::Vector3f last_acc_body_ned_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f last_gyro_body_ned_ = Eigen::Vector3f::Zero();
    float last_imu_dt_ = NAN;
    bool  have_last_imu_ = false;

    // Robust running means used only as timeout fallback for mag-world ref init.
    Eigen::Vector3f fallback_acc_mean_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f fallback_mag_mean_ = Eigen::Vector3f::Zero();
    int fallback_mean_count_ = 0;

    MagAutoTuner mag_auto_;
};
