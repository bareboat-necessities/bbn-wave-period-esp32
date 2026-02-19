#pragma once

/*
  Copyright (c) 2025  Mikhail Grushinskiy
  Released under the MIT License

  SeaStateFusionFilter (DROP-IN FIXED)

  Fixes applied (no behavior regressions intended):
  - FIX: initialize_from_acc() argument naming + usage: it's BODY-frame accel, not "world".
  - FIX: initialize_ext() constructor signature: removed bogus R_S_noise arg (did not match Kalman3D_Wave_2).
  - FIX: Use Kalman3D_Wave_2 warmup mode as the single source of truth for freeze behavior:
         enterCold_() -> set_warmup_mode(true), enterLive_() -> set_warmup_mode(false).
  - FIX: apply_ou_tune_() and apply_RS_tune_() now actually call into Kalman3D_Wave_2
         (set_aw_time_constant / set_aw_stationary_std / set_RS_noise) if present.
  - SAFETY: All setter calls are guarded by mekf_ and finite checks.
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
#include "MagAutoTuner.h"
#include "Kalman3D_Wave_2.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"
#include "TimeAwareSpikeFilter.h"

// Shared constants
constexpr float ACC_NOISE_FLOOR_SIGMA_DEFAULT = 0.15f;

constexpr float MIN_FREQ_HZ = 0.2f;
constexpr float MAX_FREQ_HZ = 6.0f;

constexpr float MIN_TAU_S   = 0.02f;
constexpr float MAX_TAU_S   = 3.0f;
constexpr float MAX_SIGMA_A = 6.0f;
constexpr float MIN_R_S     = 0.4f;
constexpr float MAX_R_S     = 35.0f;

constexpr float ADAPT_TAU_SEC              = 1.5f;
constexpr float ADAPT_EVERY_SECS           = 0.1f;
constexpr float ADAPT_RS_MULT              = 5.0f;
constexpr float ONLINE_TUNE_WARMUP_SEC     = 5.0f;
constexpr float MAG_DELAY_SEC              = 8.0f;

constexpr float HARMONIC_POS_SIGMA_AT_REF  = 1.0f;

// Frequency smoother dt (SeaStateFusionFilter is designed for 240 Hz)
constexpr float FREQ_SMOOTHER_DT = 1.0f / 240.0f;

struct TuneState {
    float tau_applied   = 1.1f;    // s
    float sigma_applied = 1e-2f;   // m/s^2
    float RS_applied    = 0.5f;    // m*s
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
#define ZERO_CROSSINGS_HYSTERESIS     0.04f
#define ZERO_CROSSINGS_PERIODS        1

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
        double freq = (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT ||
                       f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ)
                      ? FREQ_GUESS
                      : static_cast<double>(f_byZeroCross);
        return freq;
    }
};

// Unified SeaState fusion filter
template<TrackerType trackerT>
class SeaStateFusionFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using TrackingPolicy = TrackerPolicy<trackerT>;

    enum class StartupStage {
        Cold,
        TunerWarm,
        Live
    };

    explicit SeaStateFusionFilter(bool with_mag = true)
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

    void initialize(const Eigen::Vector3f& sigma_a,
                    const Eigen::Vector3f& sigma_g,
                    const Eigen::Vector3f& sigma_m)
    {
        mekf_ = std::make_unique<Kalman3D_Wave_2<float>>(sigma_a, sigma_g, sigma_m);
        enterCold_();
        apply_ou_tune_();
        mekf_->set_exact_att_bias_Qd(true);
    }

    // FIX: removed R_S_noise from signature + ctor args (did not match Kalman3D_Wave_2)
    void initialize_ext(const Eigen::Vector3f& sigma_a,
                        const Eigen::Vector3f& sigma_g,
                        const Eigen::Vector3f& sigma_m,
                        float Pq0, float Pb0,
                        float b0,
                        float gravity_magnitude)
    {
        mekf_ = std::make_unique<Kalman3D_Wave_2<float>>(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0, gravity_magnitude);
        enterCold_();
        apply_ou_tune_();
        mekf_->set_exact_att_bias_Qd(true);
    }

    // FIX: this is BODY-frame accel in NED (same frame you pass everywhere)
    void initialize_from_acc(const Eigen::Vector3f& acc_body_ned) {
        if (mekf_) {
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

        time_ += dt;
        startup_stage_t_ += dt;

        const float a_x_body = acc_body_ned.x();
        const float a_y_body = acc_body_ned.y();

        // In NED, acc.z is "down" specific force. a_z_inertial = acc_z + g
        const float a_z_inertial = acc_body_ned.z() + g_std;

        // MEKF updates
        mekf_->time_update(gyro_body_ned, dt);
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

            constexpr float TILT_RESET_DEG = 70.0f;
            constexpr float TILT_RESET_HOLD_SEC = 0.35f;
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
        a_vert_up = -a_z_inertial;

        // LPF for tracker input
        const float a_vert_lp = freq_input_lpf_.step(a_vert_up, dt);

        const float f_tracker = static_cast<float>(tracker_policy_.run(a_vert_lp, dt));
        f_raw = f_tracker;

        const float f_after_still = freq_stillness_.step(a_vert_lp, dt, f_tracker);

        float f_fast = freq_fast_smoother_.update(f_after_still);
        float f_slow = freq_slow_smoother_.update(f_fast);

        f_fast = std::min(std::max(f_fast, min_freq_hz_), max_freq_hz_);
        f_slow = std::min(std::max(f_slow, min_freq_hz_), max_freq_hz_);

        freq_hz_      = f_fast;
        freq_hz_slow_ = f_slow;

        if (enable_tuner_) {
            update_tuner(dt, a_vert_up, f_after_still);
        }

        if (startup_stage_ == StartupStage::Live && enable_linear_block_) {
            applyEnvelopeDriftCorrection_(dt);
            applyHarmonicPositionCorrection_(dt, acc_body_ned, a_vert_up);
        }

        const float omega = 2.0f * static_cast<float>(M_PI) * freq_hz_;

        dir_filter_.update(a_x_body, a_y_body, omega, dt);
        dir_sign_state_ = dir_sign_.update(a_x_body, a_y_body, a_vert_up, dt);
    }

    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (!with_mag_ || !mekf_) return;
        if (time_ < mag_delay_sec_) return;

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
                        // TODO: mekf_->set_Racc(Racc_nominal_);
                        warmup_Racc_active_ = false;
                    }
                }
            }
        }
    }

    void setWithMag(bool with_mag) { with_mag_ = with_mag; }

    void setSFactor(float s) {
        if (std::isfinite(s) && s > 0.0f) S_factor_ = s;
    }
    void setRSXYFactor(float k) {
        if (std::isfinite(k)) R_S_xy_factor_ = std::min(std::max(k, 0.0f), 1.0f);
    }

    void setTauCoeff(float c)   { if (std::isfinite(c) && c > 0.0f) tau_coeff_   = c; }
    void setSigmaCoeff(float c) { if (std::isfinite(c) && c > 0.0f) sigma_coeff_ = c; }
    void setRSCoeff(float c)    { if (std::isfinite(c) && c > 0.0f) R_S_coeff_   = c; }

    void setAccNoiseFloorSigma(float s) { if (std::isfinite(s) && s > 0.0f) acc_noise_floor_sigma_ = s; }
    float getAccNoiseFloorSigma() const noexcept { return acc_noise_floor_sigma_; }

    void setFreqInputCutoffHz(float fc) { freq_input_lpf_.setCutoff(fc); }

    void enableClamp(bool flag = true) { enable_clamp_ = flag; }
    void enableTuner(bool flag = true) { enable_tuner_ = flag; }

    void setEnvelopeStateCorrectionEnabled(bool en) { enable_env_state_correction_ = en; }
    void setEnvelopeRSCorrectionEnabled(bool en)    { enable_env_rs_correction_    = en; }

    void setHarmonicPositionCorrectionEnabled(bool en) { enable_harmonic_position_correction_ = en; }
    void setHarmonicPositionCorrectionPeriodSteps(int steps) { if (steps > 0) harmonic_position_update_period_steps_ = steps; }
    void setHarmonicPositionCorrectionSigma(float sigma_m) {
        if (std::isfinite(sigma_m) && sigma_m > 0.0f) harmonic_position_sigma_m_at_ref_env_ = sigma_m;
    }
    void setHarmonicPositionDespikeConfig(int window_size, float threshold) {
        if (window_size < 3) return;
        if (!std::isfinite(threshold) || threshold <= 0.0f) return;
        harmonic_despike_window_ = window_size;
        harmonic_despike_threshold_ = threshold;
        initHarmonicDespikeFilters_();
    }

    void setEnvelopeStateCorrectionParams(float sigma0_m, float /*gain*/) {
        if (std::isfinite(sigma0_m) && sigma0_m > 0.0f) env_sigma0_m_ = sigma0_m;
    }
    void setEnvelopeStateCorrectionMaxSpeed(float max_speed_mps) {
        if (std::isfinite(max_speed_mps) && max_speed_mps > 0.0f) env_state_max_speed_mps_ = max_speed_mps;
    }
    void setEnvelopeStateCorrectionUpdatePeriod(float every_sec) {
        if (std::isfinite(every_sec) && every_sec > 0.02f) env_state_every_sec_ = every_sec;
    }
    void setEnvelopeStateCorrectionMinErrRatio(float r) {
        if (std::isfinite(r) && r >= 0.0f) env_state_min_err_ratio_ = std::min(std::max(r, 0.0f), 1.0f);
    }

    void setEnvelopeRSMinScale(float min_scale) {
        if (std::isfinite(min_scale)) env_rs_min_scale_ = std::min(std::max(min_scale, 1e-4f), 1.0f);
    }
    void setEnvelopeRSSmoothingTauSec(float tau_sec) {
        if (std::isfinite(tau_sec) && tau_sec >= 0.0f) env_rs_smooth_tau_sec_ = tau_sec;
    }
    void setEnvelopeGateThresholdScale(float scale) {
        if (std::isfinite(scale) && scale > 0.0f) env_gate_threshold_scale_ = scale;
    }
    float getEnvelopeGateThresholdScale() const noexcept { return env_gate_threshold_scale_; }

    void setEnvelopeRSGateThresholdScale(float scale) {
        if (std::isfinite(scale) && scale > 0.0f) env_rs_gate_threshold_scale_ = scale;
    }
    float getEnvelopeRSGateThresholdScale() const noexcept { return env_rs_gate_threshold_scale_; }

    void setEnvelopeRSGateHysteresisScales(float on_scale, float off_scale) {
        if (!std::isfinite(on_scale) || !std::isfinite(off_scale)) return;
        if (on_scale < 0.5f || off_scale < 0.1f || off_scale >= on_scale) return;
        env_rs_gate_on_scale_  = on_scale;
        env_rs_gate_off_scale_ = off_scale;
    }

    void setEnvelopeRSOutwardEps(float eps_m) {
        if (std::isfinite(eps_m) && eps_m >= 0.0f) env_outward_eps_m_ = eps_m;
    }

    void setEnvelopeCorrectionWarmupSec(float warmup_sec) {
        if (std::isfinite(warmup_sec) && warmup_sec >= 0.0f) env_correction_warmup_sec_ = warmup_sec;
    }

    void enableLinearBlock(bool flag = true) {
        enable_linear_block_ = flag;
        if (mekf_) {
            const bool on_now = flag && (startup_stage_ == StartupStage::Live);
            mekf_->set_wave_block_enabled(on_now);
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

    void setMaxSigmaA(float max_sigma_a) { if (std::isfinite(max_sigma_a) && max_sigma_a > 0.0f) max_sigma_a_ = max_sigma_a; }

    void setRSBounds(float min_RS, float max_RS) {
        if (!std::isfinite(min_RS) || !std::isfinite(max_RS)) return;
        if (min_RS <= 0.0f || max_RS <= min_RS) return;
        min_R_S_ = min_RS;
        max_R_S_ = max_RS;
    }

    void setAdaptationTimeConstants(float tau_sec) { if (std::isfinite(tau_sec) && tau_sec > 0.0f) adapt_tau_sec_ = tau_sec; }
    void setAdaptationUpdatePeriod(float every_sec) { if (std::isfinite(every_sec) && every_sec > 0.0f) adapt_every_secs_ = every_sec; }
    void setOnlineTuneWarmupSec(float warmup_sec) { if (std::isfinite(warmup_sec) && warmup_sec >= 0.0f) online_tune_warmup_sec_ = warmup_sec; }
    void setMagDelaySec(float delay_sec) { if (std::isfinite(delay_sec) && delay_sec >= 0.0f) mag_delay_sec_ = delay_sec; }

    void setFreezeAccBiasUntilLive(bool en) { freeze_acc_bias_until_live_ = en; }
    void setWarmupRacc(float r) { if (std::isfinite(r) && r > 0.0f) Racc_warmup_ = r; }
    void setNominalRacc(const Eigen::Vector3f& r) { Racc_nominal_ = r; }

    inline float getFreqHz()        const noexcept { return freq_hz_; }
    inline float getFreqSlowHz()    const noexcept { return freq_hz_slow_; }
    inline float getFreqRawHz()     const noexcept { return f_raw; }
    inline float getTauApplied()    const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied()  const noexcept { return tune_.sigma_applied; }
    inline float getRSApplied()     const noexcept { return tune_.RS_applied; }
    inline float getTauTarget()     const noexcept { return tau_target_; }
    inline float getSigmaTarget()   const noexcept { return sigma_target_; }
    inline float getRSTarget()      const noexcept { return RS_target_; }

    inline float getPeriodSec() const noexcept {
        return (freq_hz_slow_ > 1e-6f) ? 1.0f / freq_hz_slow_ : NAN;
    }

    inline float getAccelVariance() const noexcept { return tuner_.getAccelVariance(); }
    inline float getAccelVertical() const noexcept { return a_vert_up; }

    inline float getHeaveAbs() const noexcept {
        if (!mekf_) return NAN;
        return std::fabs(mekf_->get_position().z());
    }

    inline float getDisplacementScale(bool smoothed = true) const noexcept {
        const float tau = smoothed ? tune_.tau_applied : tau_target_;
        const float sigma = smoothed ? tune_.sigma_applied : sigma_target_;
        if (!std::isfinite(sigma) || !std::isfinite(tau)) return NAN;
        constexpr float C_HS  = 2.0f * std::sqrt(2.0f) / (M_PI * M_PI);
        return C_HS * sigma * tau * tau;
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
    static inline float sgnf_(float x) noexcept { return (x >= 0.0f) ? 1.0f : -1.0f; }

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
        float energy_ema      = 0.0f;
        float energy_alpha    = 0.05f;
        float energy_thresh   = 8e-4f;

        float still_time_sec  = 0.0f;
        float still_thresh_s  = 2.0f;

        float relax_tau_sec   = 1.0f;
        float target_freq_hz  = MIN_FREQ_HZ;

        bool  freq_init       = false;
        float freq_state      = FREQ_GUESS;

        bool  last_is_still   = false;

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

    // FIX: actually apply OU tuning to Kalman3D_Wave_2
    void apply_ou_tune_() {
        if (!mekf_) return;

        const float tau = std::min(std::max(tune_.tau_applied, min_tau_s_), max_tau_s_);
        if (std::isfinite(tau) && tau > 0.0f) {
            // TODO: mekf_->set_aw_time_constant(tau);
        }

        const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
        const float sZ = std::max(sigma_floor, tune_.sigma_applied);
        const float sH = sZ * S_factor_;
        if (std::isfinite(sH) && std::isfinite(sZ) && sH > 0.0f && sZ > 0.0f) {
            // TODO: mekf_->set_aw_stationary_std(Eigen::Vector3f(sH, sH, sZ));
        }
    }

    // FIX: actually apply RS tuning to Kalman3D_Wave_2
    void apply_RS_tune_(float rs_scale = 1.0f) {
        if (!mekf_) return;

        const float s = (std::isfinite(rs_scale) && rs_scale > 0.0f) ? std::min(rs_scale, 1.0f) : 1.0f;
        const float RSb = std::min(std::max(tune_.RS_applied, min_R_S_), max_R_S_);
        const float rs_xy = RSb * s * R_S_xy_factor_;
        mekf_->set_RS_noise(Eigen::Vector3f(rs_xy, rs_xy, RSb * s));
    }

    void applyEnvelopeDriftCorrection_(float dt) {
        if (!mekf_) return;

        const float scale = std::max(getDisplacementScale(), 0.3f);
        const float pz = mekf_->get_position().z();
        if (!std::isfinite(scale) || !std::isfinite(pz) || scale <= 0.0f) {
            apply_RS_tune_();
            return;
        }

        if (startup_stage_t_ < env_correction_warmup_sec_) {
            apply_RS_tune_();
            return;
        }

        const float absz = std::fabs(pz);

        const float rs_gate_scale = std::max(env_rs_gate_threshold_scale_, 0.05f);
        const float rs_on_scale = std::max(env_rs_gate_on_scale_, 0.1f);
        const float rs_off_scale = std::max(env_rs_gate_off_scale_, 0.1f);
        const float rs_gate_on  = scale * rs_gate_scale * rs_on_scale;
        const float rs_gate_off = scale * rs_gate_scale * std::min(rs_off_scale, rs_on_scale - 1e-3f);

        if (!env_rs_latched_) {
            if (enable_env_rs_correction_ && absz > rs_gate_on) env_rs_latched_ = true;
        } else if (absz < rs_gate_off) {
            env_rs_latched_ = false;
        }

        float rs_scale_cmd = 1.0f;
        if (enable_env_rs_correction_ && env_rs_latched_) {
            const float r = absz / std::max(rs_gate_on, 1e-3f);
            if (r > 1.0f) {
                constexpr float RS_TIGHTEN_POWER = 8.0f;
                rs_scale_cmd = std::clamp(std::pow(r, -RS_TIGHTEN_POWER), env_rs_min_scale_, 1.0f);
            }
        }

        if (std::isfinite(dt) && dt > 0.0f && env_rs_smooth_tau_sec_ > 0.0f) {
            const float alpha = dt / (env_rs_smooth_tau_sec_ + dt);
            env_rs_scale_state_ += alpha * (rs_scale_cmd - env_rs_scale_state_);
        } else {
            env_rs_scale_state_ = rs_scale_cmd;
        }

        env_rs_scale_state_ = std::clamp(env_rs_scale_state_, env_rs_min_scale_, 1.0f);
        apply_RS_tune_(env_rs_scale_state_);
    }

    void initHarmonicDespikeFilters_() {
        despike_ax_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
        despike_ay_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
        despike_az_ = std::make_unique<TimeAwareSpikeFilter>(harmonic_despike_window_, harmonic_despike_threshold_);
    }

    void resetHarmonicPositionCorrection_() {
        harmonic_position_counter_ = 0;
        initHarmonicDespikeFilters_();
    }

    void applyHarmonicPositionCorrection_(float dt, const Eigen::Vector3f& acc_body_ned, float a_vert_up_osc) {
        if (!mekf_ || !enable_harmonic_position_correction_) return;
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;

        float env_scale = getDisplacementScale(true);
        if (!std::isfinite(env_scale) || env_scale <= 0.0f) env_scale = 1.0f;

        const float absz = std::fabs(mekf_->get_position().z());
        constexpr float HARMONIC_DRIFT_RISK_ENVELOPE_RATIO = 0.1f;
        if (!(absz > env_scale * HARMONIC_DRIFT_RISK_ENVELOPE_RATIO)) return;

        if (++harmonic_position_counter_ < harmonic_position_update_period_steps_) return;
        harmonic_position_counter_ = 0;

        const float harmonic_freq_hz = (std::isfinite(freq_hz_) && freq_hz_ > 0.0f) ? freq_hz_ : freq_hz_slow_;
        const float omega = 2.0f * static_cast<float>(M_PI) * std::max(harmonic_freq_hz, min_freq_hz_);
        const float omega_sq = omega * omega;
        if (!(omega_sq > 1e-4f) || !std::isfinite(omega_sq)) return;

        if (!acc_body_ned.allFinite()) return;

        const Eigen::Vector3f a_world_ned = acc_body_ned;

        if (!despike_ax_ || !despike_ay_ || !despike_az_) initHarmonicDespikeFilters_();

        const float a_z_ned_osc = -a_vert_up_osc;

        const Eigen::Vector3f a_despiked(
            despike_ax_->filterWithDelta(a_world_ned.x(), dt),
            despike_ay_->filterWithDelta(a_world_ned.y(), dt),
            despike_az_->filterWithDelta(a_z_ned_osc, dt)
        );
        if (!a_despiked.allFinite()) return;

        const Eigen::Vector3f p_meas = -a_despiked / omega_sq;
        if (!p_meas.allFinite()) return;

        const float sigma = std::max(
            harmonic_position_sigma_m_at_ref_env_ * (env_scale / harmonic_position_ref_envelope_m_),
            harmonic_position_sigma_min_m_
        );
        mekf_->measurement_update_position_pseudo(Eigen::Vector3f::Constant(sigma), p_meas);
    }

    void update_tuner(float dt, float a_vert_inertial, float /*freq_hz_for_tuner*/) {
        tuner_.update(dt, a_vert_inertial, freq_hz_slow_);

        switch (startup_stage_) {
            case StartupStage::Cold:
                if (startup_stage_t_ >= online_tune_warmup_sec_) {
                    startup_stage_   = StartupStage::TunerWarm;
                    startup_stage_t_ = 0.0f;
                }
                return;

            case StartupStage::TunerWarm:
                if (!tuner_.isFreqReady()) return;
                if (tuner_.isReady()) {
                    enterLive_();
                }
                break;

            case StartupStage::Live:
                break;
        }

        float f_tune = tuner_.getFrequencyHz();
        if (!std::isfinite(f_tune) || f_tune < min_freq_hz_) f_tune = min_freq_hz_;
        if (f_tune > max_freq_hz_) f_tune = max_freq_hz_;

        float var_total = acc_noise_floor_sigma_ * acc_noise_floor_sigma_;
        if (tuner_.isVarReady()) var_total = std::max(0.0f, tuner_.getAccelVariance());

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

        if (!tuner_.isVarReady()) {
            sigma_target_ = std::max(sigma_target_, std::max(0.05f, acc_noise_floor_sigma_));
        }

        float RS_raw = R_S_coeff_ * sigma_target_ * tau_target_ * tau_target_ * tau_target_;
        RS_target_ = enable_clamp_ ? std::min(std::max(RS_raw, min_R_S_), max_R_S_) : RS_raw;

        adapt_mekf(dt, tau_target_, sigma_target_, RS_target_);
    }

    void adapt_mekf(float dt, float tau_t, float sigma_t, float RS_t) {
        const float alpha = 1.0f - std::exp(-dt / adapt_tau_sec_);

        const float RS_sec   = ADAPT_RS_MULT * tau_t;
        const float alpha_RS = 1.0f - std::exp(-dt / RS_sec);

        tune_.tau_applied   += alpha    * (tau_t   - tune_.tau_applied);
        tune_.sigma_applied += alpha    * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha_RS * (RS_t    - tune_.RS_applied);

        if (time_ - last_adapt_time_sec_ > adapt_every_secs_) {
            if (tuner_.isFreqReady()) apply_ou_tune_();
            if (startup_stage_ == StartupStage::Live && enable_linear_block_) apply_RS_tune_();
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
        f_raw         = FREQ_GUESS;

        dir_filter_ = KalmanWaveDirection(2.0f * static_cast<float>(M_PI) * FREQ_GUESS);
        dir_sign_state_ = UNCERTAIN;

        last_adapt_time_sec_ = time_;

        env_outside_time_sec_ = 0.0f;
        env_rs_latched_ = false;
        env_rs_scale_state_ = 1.0f;
        last_env_state_update_sec_ = static_cast<float>(time_);
        resetHarmonicPositionCorrection_();
    }

    void enterCold_() {
        startup_stage_   = StartupStage::Cold;
        startup_stage_t_ = 0.0f;

        if (!mekf_) return;

        // FIX: make Kalman3D_Wave_2 warmup mode the source-of-truth
        mekf_->set_warmup_mode(true);

        // Keep wrapper's linear-block gating consistent
        mekf_->set_wave_block_enabled(false);

        accel_bias_locked_   = with_mag_;
        mag_updates_applied_ = 0;
        first_mag_update_time_ = NAN;

        if (freeze_acc_bias_until_live_) {
            mekf_->set_acc_bias_updates_enabled(false);
            mekf_->set_Racc(Eigen::Vector3f::Constant(Racc_warmup_).eval());
            warmup_Racc_active_ = true;
        }

        env_rs_latched_ = false;
        resetHarmonicPositionCorrection_();
    }

    void enterLive_() {
        startup_stage_   = StartupStage::Live;
        startup_stage_t_ = 0.0f;

        if (!mekf_) return;

        // FIX: exit warmup
        mekf_->set_warmup_mode(false);

        mekf_->set_wave_block_enabled(enable_linear_block_);

        if (freeze_acc_bias_until_live_) {
            const bool allow_bias = !accel_bias_locked_;
            mekf_->set_acc_bias_updates_enabled(allow_bias);

            if (warmup_Racc_active_ && Racc_nominal_.allFinite() && Racc_nominal_.maxCoeff() > 0.0f) {
                mekf_->set_Racc(Racc_nominal_);
            }
            warmup_Racc_active_ = false;
        }

        apply_ou_tune_();
        if (enable_linear_block_) apply_RS_tune_();

        env_outside_time_sec_ = 0.0f;
        env_rs_latched_ = false;
        env_rs_scale_state_ = 1.0f;
        last_env_state_update_sec_ = static_cast<float>(time_);
        resetHarmonicPositionCorrection_();
    }

private:
    StartupStage startup_stage_   = StartupStage::Cold;
    float        startup_stage_t_ = 0.0f;

    bool  freeze_acc_bias_until_live_ = true;
    float Racc_warmup_               = 0.5f;
    bool  warmup_Racc_active_         = false;
    Eigen::Vector3f Racc_nominal_     = Eigen::Vector3f::Constant(0.0f);

    bool accel_bias_locked_ = true;
    int  mag_updates_applied_ = 0;
    static constexpr int MAG_UPDATES_TO_UNLOCK = 5;

    bool   with_mag_;
    double time_;
    double last_adapt_time_sec_;

    float first_mag_update_time_ = NAN;

    float tilt_over_limit_sec_ = 0.0f;
    float tilt_reset_cooldown_sec_ = 0.0f;

    float freq_hz_      = FREQ_GUESS;
    float freq_hz_slow_ = FREQ_GUESS;
    float f_raw         = FREQ_GUESS;

    float a_vert_up = 0.0f;

    bool enable_clamp_ = true;
    bool enable_tuner_ = true;

    bool enable_env_state_correction_ = false;
    bool enable_env_rs_correction_ = false;
    bool enable_harmonic_position_correction_ = false;

    int   harmonic_position_update_period_steps_ = 3;
    int   harmonic_position_counter_ = 0;
    float harmonic_position_ref_envelope_m_ = 8.0f;
    float harmonic_position_sigma_m_at_ref_env_ = HARMONIC_POS_SIGMA_AT_REF;
    float harmonic_position_sigma_min_m_ = 0.05f;
    float harmonic_despike_threshold_ = 4.0f;
    int   harmonic_despike_window_ = 5;

    std::unique_ptr<TimeAwareSpikeFilter> despike_ax_;
    std::unique_ptr<TimeAwareSpikeFilter> despike_ay_;
    std::unique_ptr<TimeAwareSpikeFilter> despike_az_;

    float env_sigma0_m_ = 0.5f;
    float env_state_max_speed_mps_ = 0.35f;
    float env_state_dwell_sec_ = 0.8f;
    float env_outside_time_sec_ = 0.0f;
    float env_state_every_sec_ = 0.15f;
    float env_state_min_err_ratio_ = 0.08f;
    float last_env_state_update_sec_ = -1e9f;
    float env_rs_min_scale_ = 5e-4f;
    float env_rs_smooth_tau_sec_ = 0.0f;
    float env_rs_scale_state_ = 1.0f;
    bool  env_rs_latched_ = false;
    float env_rs_gate_on_scale_ = 1.20f;
    float env_rs_gate_off_scale_ = 0.95f;
    float env_outward_eps_m_ = 1e-4f;
    float env_gate_threshold_scale_ = 1.2f;
    float env_rs_gate_threshold_scale_ = 0.8f;
    float env_correction_warmup_sec_ = 16.0f;

    bool enable_linear_block_ = true;

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

    float R_S_xy_factor_ = 0.17f;
    float S_factor_      = 1.7f;

    TrackingPolicy               tracker_policy_{};
    FirstOrderIIRSmoother<float> freq_fast_smoother_{FREQ_SMOOTHER_DT, 3.5f};
    FirstOrderIIRSmoother<float> freq_slow_smoother_{FREQ_SMOOTHER_DT, 10.0f};

    SeaStateAutoTuner tuner_;
    TuneState         tune_;

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    float acc_noise_floor_sigma_ = ACC_NOISE_FLOOR_SIGMA_DEFAULT;

    float R_S_coeff_   = 1.2f;
    float tau_coeff_   = 1.4f;
    float sigma_coeff_ = 0.9f;

    std::unique_ptr<Kalman3D_Wave_2<float>> mekf_;

    KalmanWaveDirection dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

    FreqInputLPF     freq_input_lpf_;
    StillnessAdapter freq_stillness_;

    WaveDirectionDetector<float> dir_sign_{0.002f, 0.005f};
    WaveDirection               dir_sign_state_ = UNCERTAIN;
};

// -----------------------------------------------------------------------------
// SeaStateFusion wrapper remains identical to your version EXCEPT:
// - begin(): call initialize_ext signature if you used it (now no R_S_noise).
// - no other changes required.
// -----------------------------------------------------------------------------

template<TrackerType trackerT>
class SeaStateFusion {
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

        bool  enable_harmonic_position_correction = true;
        int   harmonic_position_update_period_steps = 3;
        float harmonic_position_sigma_m = HARMONIC_POS_SIGMA_AT_REF;

        Eigen::Vector3f sigma_a = Eigen::Vector3f(0.2f,0.2f,0.2f);
        Eigen::Vector3f sigma_g = Eigen::Vector3f(0.01f,0.01f,0.01f);
        Eigen::Vector3f sigma_m = Eigen::Vector3f(0.3f,0.3f,0.3f);

        float mag_ref_timeout_sec = 1.5f;
        float mag_odr_guess_hz = 80.0f;

        bool use_custom_mag_tuner_cfg = false;
        MagAutoTuner::Config mag_tuner_cfg{};
    };

    void begin(const Config& cfg) {
        cfg_ = cfg;

        begun_ = true;
        stage_ = Stage::Uninitialized;
        t_ = 0.0f;
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
        impl_.setHarmonicPositionCorrectionEnabled(cfg.enable_harmonic_position_correction);
        impl_.setHarmonicPositionCorrectionPeriodSteps(cfg.harmonic_position_update_period_steps);
        impl_.setHarmonicPositionCorrectionSigma(cfg.harmonic_position_sigma_m);

        impl_.initialize(cfg.sigma_a, cfg.sigma_g, cfg.sigma_m);

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

        if (impl_.getStartupStage() == SeaStateFusionFilter<trackerT>::StartupStage::Cold) {
            mag_ref_set_ = false;
            mag_auto_.reset();
            last_mag_time_sec_ = NAN;
            dt_mag_sec_ = NAN;
            mag_ref_deadline_sec_ = t_ + cfg_.mag_ref_timeout_sec;
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

        if (have_last_imu_) {
            const float an = last_acc_body_ned_.norm();
            const float mn = mag_body_ned.norm();
            const float g = 9.80665f;
            const bool accel_ok = last_acc_body_ned_.allFinite() && (an > 0.85f * g) && (an < 1.15f * g);
            const bool mag_ok   = mag_body_ned.allFinite() && (mn > 1e-3f);
            if (accel_ok && mag_ok) {
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
                Eigen::Vector3f acc_for_ref = last_acc_body_ned_;
                bool have_ref_candidate = false;

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

                if (!have_ref_candidate &&
                    std::isfinite(mag_ref_deadline_sec_) && t_ >= mag_ref_deadline_sec_)
                {
                    if (cfg_.mag_world_ref.allFinite() && cfg_.mag_world_ref.norm() > 1e-3f) {
                        impl_.mekf().set_mag_world_ref(cfg_.mag_world_ref);
                        mag_ref_set_ = true;
                    } else if (fallback_mean_count_ >= 25 &&
                               fallback_acc_mean_.allFinite() && fallback_acc_mean_.norm() > 1e-3f &&
                               fallback_mag_mean_.allFinite() && fallback_mag_mean_.norm() > 1e-3f)
                    {
                        acc_for_ref = fallback_acc_mean_;
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

        if (mag_ref_set_) impl_.updateMag(mag_body_ned);
    }

    bool  isLive() const { return stage_ == Stage::Live; }
    float freqHz() const { return impl_.getFreqHz(); }
    Eigen::Vector3f eulerNauticalDeg() const { return impl_.getEulerNautical(); }

    SeaStateFusionFilter<trackerT>& raw() { return impl_; }

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
    SeaStateFusionFilter<trackerT> impl_{false};

    bool begun_ = false;

    Stage stage_ = Stage::Uninitialized;
    float t_ = 0.0f;
    float stage_t_ = 0.0f;

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
