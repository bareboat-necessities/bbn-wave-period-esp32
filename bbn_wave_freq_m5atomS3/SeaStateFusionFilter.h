#pragma once

/*

  SeaStateFusionFilter
  
  Marine Inertial Navigational System (INS) Filter for IMU

  Combines multiple real-time estimators into a cohesive ocean-state tracker:

    • Quaternion-based attitude and linear motion estimation via Kalman3D_Wave  

    • Dominant frequency tracking using one of:
          – AranovskiyFilter     (frequency estimator)
          – KalmANF              (adaptive notch / Kalman frequency tracker)
          – SchmittTrigger       (zero-cross event detector)

    • Dual-stage frequency smoothing:
          – Fast 1st-order IIR (≈ 1 s, ~90% step) for demodulation / direction
          – Slow 1st-order IIR (≈ 10 s, ~90% step) for auto-tuning / moments

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

  Copyright (c) 2025  Mikhail Grushinskiy  
  Released under the MIT License 
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
constexpr float ACC_NOISE_FLOOR_SIGMA = 0.03f; // e.g. ≈3 mg

constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 2.0f;

constexpr float MIN_TAU_S   = 0.5f;
constexpr float MAX_TAU_S   = 8.5f;
constexpr float MAX_SIGMA_A = 8.0f;
constexpr float MAX_R_S     = 120.0f;

constexpr float ADAPT_TAU_SEC            = 3.0f;
constexpr float ADAPT_EVERY_SECS         = 0.1f;
constexpr float ONLINE_TUNE_WARMUP_SEC   = 35.0f;
constexpr float MAG_DELAY_SEC            = 5.0f;

// Frequency smoother dt (SeaStateFusionFilter is designed for 240 Hz)
constexpr float FREQ_SMOOTHER_DT = 1.0f / 240.0f;

struct TuneState {
    float tau_applied   = 0.97f;  // s
    float sigma_applied = 0.0f;   // m/s²
    float RS_applied    = 0.0f;   // m*s
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

    explicit SeaStateFusionFilter(bool with_mag = true)
        : with_mag_(with_mag),
          time_(0.0),
          last_adapt_time_sec_(0.0),
          freq_hz_(FREQ_GUESS),
          freq_hz_slow_(FREQ_GUESS)
    {
        // Default cutoff ~MAX_FREQ_HZ Hz: passes waves, kills 8–37 Hz engine band
        freq_input_lpf_.setCutoff(MAX_FREQ_HZ);
    }

    void initialize(const Eigen::Vector3f& sigma_a,
                    const Eigen::Vector3f& sigma_g,
                    const Eigen::Vector3f& sigma_m)
    {
        mekf_ = std::make_unique<Kalman3D_Wave<float,true,true>>(sigma_a, sigma_g, sigma_m);
        mekf_->set_exact_att_bias_Qd(true);
        apply_tune();
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) {
            mekf_->initialize_from_acc(acc_world);
        }
    }

    // Time update (IMU integration + frequency tracking)
    void updateTime(float dt,
                    const Eigen::Vector3f& gyro,
                    const Eigen::Vector3f& acc,
                    float tempC = 35.0f)
    {
        if (!mekf_) return;
        time_ += dt;

        // Tracker input: vertical inertial (BODY, m/s^2)
        const float a_z_inertial = acc.z() + g_std;
        const float a_x = acc.x();
        const float a_y = acc.y();

        // LPF to suppress engine band (8–37 Hz) before tracker
        const float a_z_inertial_lp = freq_input_lpf_.step(a_z_inertial, dt);

        // MEKF updates
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc, tempC);

        // Raw freq from tracker
        const float f_tracker = static_cast<float>(tracker_policy_.run(a_z_inertial_lp, dt));
        f_raw = f_tracker;

        // Adjust for stillness (same logic for all trackers)
        const float f_after_still = freq_stillness_.step(a_z_inertial_lp, dt, f_tracker);

        // Fast frequency smoother (≈1 s to ~90% of a step)
        float f_fast = freq_fast_smoother_.update(f_after_still);
        // Slow frequency smoother (≈10 s to ~90% of a step)
        float f_slow = freq_slow_smoother_.update(f_fast);

        // Clamp both
        f_fast = std::min(std::max(f_fast, MIN_FREQ_HZ), MAX_FREQ_HZ);
        f_slow = std::min(std::max(f_slow, MIN_FREQ_HZ), MAX_FREQ_HZ);

        // Fast branch used for demod / direction
        freq_hz_       = f_fast;
        // Slow branch used for adaptation / moment-like quantities
        freq_hz_slow_  = f_slow;

        // Tuner uses the SLOW branch
        if (enable_tuner_) {
            update_tuner(dt, a_z_inertial, freq_hz_slow_);
        }

        // Direction filter uses fast frequency (ω = 2πf_fast)
        const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz_;
        dir_filter_.update(a_x, a_y, omega, dt);
        dir_sign_state_ = dir_sign_.update(a_x, a_y, a_z_inertial, dt);
    }

    //  Magnetometer correction
    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (with_mag_ && mekf_ && time_ >= MAG_DELAY_SEC) {
            mekf_->measurement_update_mag_only(mag_body_ned);
        }
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

    void setRSCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            R_S_coeff_ = c;
        }
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

    inline WaveDirection getDirSignState() const noexcept { return dir_sign_state_; }

    Eigen::Vector3f getEulerNautical() const {
        if (!mekf_) return {NAN, NAN, NAN};

        // q_bw: body→world (Kalman3D_Wave::quaternion() already returns qref.conjugate())
        Eigen::Quaternionf q_bw = mekf_->quaternion();
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
    struct FreqStillnessAdapter {
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
            energy_ema = (1.0f - energy_alpha) * energy_ema
                       + energy_alpha * inst_energy;

            const bool is_still = (energy_ema < energy_thresh);

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
    };

    //  Internal tuning and adaptation
    void apply_tune() {
        if (!mekf_) return;
    
        // OU time constant
        mekf_->set_aw_time_constant(tune_.tau_applied);
    
        // WORLD-frame stationary covariance for a_w (XY equal, Z separate).
        const float sZ = std::max(1e-6f, tune_.sigma_applied);
        const float sH = sZ * S_factor_;
        Eigen::Vector3f a_w_std(sH, sH, sZ);
        mekf_->set_aw_stationary_std(a_w_std); 
    
        // WORLD-frame pseudo-measurement noise for S (anisotropic diagonal).
        const float RSb = std::min(std::max(tune_.RS_applied, 0.0f), MAX_R_S);
        mekf_->set_RS_noise(Eigen::Vector3f(
            RSb * R_S_xy_factor_,   // world X
            RSb * R_S_xy_factor_,   // world Y
            RSb                     // world Z
        ));
    }

    void update_tuner(float dt, float a_vert_inertial, float freq_hz_slow) {
        tuner_.update(dt, a_vert_inertial, freq_hz_slow);
    
        if (time_ < ONLINE_TUNE_WARMUP_SEC) return;

        const float f_tune    = tuner_.getFrequencyHz();
        const float var_total = std::max(0.0f, tuner_.getAccelVariance());

        // Fixed noise floor variance
        const float var_noise = ACC_NOISE_FLOOR_SIGMA * ACC_NOISE_FLOOR_SIGMA;
        // Wave-only variance (never negative)
        const float var_wave  = std::max(0.0f, var_total - var_noise);

        // Wave-only sigma; if var_wave ~ 0, this goes to 0 → flat sea mode
        float sigma_wave = (var_wave > 0.0f) ? std::sqrt(var_wave) : 0.0f;

        // τ target from frequency
        float tau_raw = tau_coeff_ * 0.5f / f_tune;

        if (enable_clamp_) {
            tau_target_   = std::min(std::max(tau_raw,  MIN_TAU_S), MAX_TAU_S);
            sigma_target_ = std::min(sigma_wave,        MAX_SIGMA_A);
        } else {
            tau_target_   = tau_raw;
            sigma_target_ = sigma_wave;
        }

        // Rₛ from (wave-only) σ and τ³
        float RS_raw = R_S_coeff_ * sigma_target_
                     * tau_target_ * tau_target_ * tau_target_;

        if (enable_clamp_) {
            RS_target_ = std::min(RS_raw, MAX_R_S);
        } else {
            RS_target_ = RS_raw;
        }

        adapt_mekf(dt, tau_target_, sigma_target_, RS_target_);
    }
    
    void adapt_mekf(float dt, float tau_t, float sigma_t, float RS_t) {
        const float alpha = 1.0f - std::exp(-dt / ADAPT_TAU_SEC);

        tune_.tau_applied   += alpha * (tau_t   - tune_.tau_applied);
        tune_.sigma_applied += alpha * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha * (RS_t    - tune_.RS_applied);

        if (time_ - last_adapt_time_sec_ > ADAPT_EVERY_SECS) {
            apply_tune();
            last_adapt_time_sec_ = time_;
        }
    }

    //  Members
    bool   with_mag_;
    double time_;
    double last_adapt_time_sec_;

    float freq_hz_       = FREQ_GUESS; // fast branch
    float freq_hz_slow_  = FREQ_GUESS; // slow branch
    float f_raw          = FREQ_GUESS;

    bool  enable_clamp_  = true;
    bool  enable_tuner_  = true;

    // Runtime-configurable anisotropy knobs
    float R_S_xy_factor_ = 0.07f;  // [0..1] scales XY pseudo-meas vs Z
    float S_factor_      = 1.3f;   // (>0) scales Σ_aw horizontal std vs vertical

    TrackingPolicy                  tracker_policy_{};
    FirstOrderIIRSmoother<float>    freq_fast_smoother_{FREQ_SMOOTHER_DT, 1.0f};   // ~1 s to 90% step
    FirstOrderIIRSmoother<float>    freq_slow_smoother_{FREQ_SMOOTHER_DT, 10.0f};  // ~10 s to 90% step
    SeaStateAutoTuner               tuner_;
    TuneState                       tune_;

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    float R_S_coeff_    = 2.4f;
    float tau_coeff_    = 1.6f;

    std::unique_ptr<Kalman3D_Wave<float,true,true>>  mekf_;
    KalmanWaveDirection                              dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

    FreqInputLPF            freq_input_lpf_;   // LPF used only for tracker input
    FreqStillnessAdapter    freq_stillness_;   // Detector of "still" mode

    WaveDirectionDetector<float> dir_sign_{0.002f, 0.005f};   // smoothing, sensitivity
    WaveDirection                dir_sign_state_ = UNCERTAIN;
};
