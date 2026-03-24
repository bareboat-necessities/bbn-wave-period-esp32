#pragma once

/*
  Copyright (c) 2025-2026 Mikhail Grushinskiy
  Released under the MIT License

  SeaStateFusionFilter_II

  Marine Inertial Navigational System (INS) Filter for IMU

  Combines multiple real-time estimators into a cohesive ocean-state tracker:

    • Quaternion-based attitude and linear motion estimation via Kalman3D_Wave_II
    • Dominant frequency tracking
    • Dual-stage frequency smoothing
    • Online sea-state tuning for wrapper tau/sigma/R_p0/R_v0

  Wave_5 notes:
    - Kalman3D_Wave_II has no latent OU a_w state and no tau_aw in the core filter.
    - Wrapper tau is retained as a tuning / envelope / pseudo-measurement heuristic.
    - Wrapper sigma is retained as a sea-state amplitude parameter.
    - Core filter is tuned through:
        • Racc (accel measurement noise)
        • b_aw RW scaling (world-frame residual acceleration bias / command correction)
    - The linear block is [v, p, b_aw].
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
#include "Kalman3D_Wave_II.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"

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

constexpr float ACC_NOISE_FLOOR_SIGMA_DEFAULT = 0.12f;

constexpr float MIN_FREQ_HZ = 0.2f;
constexpr float MAX_FREQ_HZ = 6.0f;

constexpr float MIN_TAU_S     = 0.02f;
constexpr float MAX_TAU_S     = 10.0f;
constexpr float MAX_SIGMA_A   = 6.0f;
constexpr float MIN_R_p0_std  = 0.05f;
constexpr float MAX_R_p0_std  = 2000.0f;
constexpr float MIN_R_v0_std  = 0.01f;
constexpr float MAX_R_v0_std  = 2000.0f;

constexpr float ADAPT_TAU_SEC              = 1.5f;
constexpr float ADAPT_EVERY_SECS           = 0.1f;
constexpr float ADAPT_R_p0_MULT            = 5.0f;
constexpr float ADAPT_R_v0_MULT            = 5.0f;
constexpr float ONLINE_TUNE_WARMUP_SEC     = 5.0f;
constexpr float MAG_DELAY_SEC              = 8.0f;

constexpr float FREQ_SMOOTHER_DT = 1.0f / 200.0f;

struct TuneState {
    float tau_applied       = 1.1f;   // wrapper-side heuristic tau [s]
    float sigma_applied     = 1e-2f;  // wrapper-side sea-state sigma [m/s²]
    float R_p0_std_applied  = 0.1f;   // [m]
    float R_v0_std_applied  = 0.1f;   // [m/s]
};

template<TrackerType trackerT>
class SeaStateFusionFilter_II {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using TrackingPolicy = TrackerPolicy<trackerT>;
    using MekfT = Kalman3D_Wave_II<float>;

    enum class StartupStage {
        Cold,
        TunerWarm,
        Live
    };

    explicit SeaStateFusionFilter_II(bool with_mag = true)
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
        mekf_ = std::make_unique<MekfT>(sigma_a, sigma_g, sigma_m);

        Racc_nominal_std_   = sigma_a;
        tune_.sigma_applied = std::max(acc_noise_floor_sigma_, sigma_a.z());

        enterCold_();
        apply_wave_tune_();
        mekf_->set_exact_att_bias_Qd(true);
    }

    void initialize_ext(const Eigen::Vector3f& sigma_a,
                        const Eigen::Vector3f& sigma_g,
                        const Eigen::Vector3f& sigma_m,
                        float Pq0, float Pb0,
                        float b0, float R_p0_var_init, float R_v0_var_init,
                        float gravity_magnitude)
    {
        mekf_ = std::make_unique<MekfT>(
            sigma_a, sigma_g, sigma_m,
            Pq0, Pb0, b0, R_p0_var_init, R_v0_var_init, gravity_magnitude);

        Racc_nominal_std_      = sigma_a;
        tune_.sigma_applied    = std::max(acc_noise_floor_sigma_, sigma_a.z());
        tune_.R_p0_std_applied = std::sqrt(std::max(0.0f, R_p0_var_init));
        tune_.R_v0_std_applied = std::sqrt(std::max(0.0f, R_v0_var_init));

        enterCold_();
        apply_wave_tune_();
        mekf_->set_exact_att_bias_Qd(true);
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_body) {
        if (mekf_) {
            mekf_->initialize_from_acc(acc_body);
        }
    }

    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc)
    {
        if (!mekf_) return;
        if (!(dt > 0.0f) || !std::isfinite(dt)) return;

        time_ += dt;
        startup_stage_t_ += dt;

        const float a_x_body = acc.x();
        const float a_y_body = acc.y();
        const float a_z_inertial = acc.z() + g_std;

        // Predict first
        mekf_->time_update(gyro, acc, dt);

        // Update motion metric before accel correction so this step's Racc is used
        update_motion_state_(dt, gyro, acc);

        // In Live, make Racc motion-adaptive every step
        if (startup_stage_ == StartupStage::Live) {
            apply_wave_tune_();
        }

        mekf_->measurement_update_acc_only(acc);

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
                mekf_->initialize_from_acc(acc);

                if (startup_stage_ != StartupStage::Live) {
                    enterCold_();
                    resetTrackingState_();
                }

                tilt_over_limit_sec_ = 0.0f;
                tilt_reset_cooldown_sec_ = TILT_RESET_COOLDOWN_SEC;
            }
        }

        a_vert_up = -a_z_inertial;

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
            apply_R_p0_tune_();
            apply_R_v0_tune_();
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
                    if (Racc_nominal_std_.allFinite() && Racc_nominal_std_.maxCoeff() > 0.0f) {
                        mekf_->set_Racc_std(Racc_nominal_std_);
                        warmup_Racc_active_ = false;
                    }
                }
            }
        }
    }

    void setWithMag(bool with_mag) {
        with_mag_ = with_mag;
    }

    void setPFactor(float p) {
        if (std::isfinite(p) && p > 0.0f) {
            P_factor_ = p;
            if (mekf_) apply_wave_tune_();
        }
    }

    void setR_p0_XYFactor(float k) {
        if (std::isfinite(k)) {
            R_p0_xy_factor_ = std::min(std::max(k, 0.0f), 1.0f);
            if (mekf_ && enable_linear_block_) apply_R_p0_tune_();
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

    void setR_p0_Coeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            const float prev = R_p0_coeff_;
            R_p0_coeff_ = c;

            if (std::isfinite(prev) && prev > 0.0f) {
                const float scale = c / prev;

                if (std::isfinite(tune_.R_p0_std_applied) && tune_.R_p0_std_applied > 0.0f) {
                    tune_.R_p0_std_applied *= scale;
                }
                if (std::isfinite(R_p0_std_target_) && R_p0_std_target_ > 0.0f) {
                    R_p0_std_target_ *= scale;
                }
                if (enable_linear_block_) {
                    apply_R_p0_tune_();
                }
            }
        }
    }

    void setR_v0_Coeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            const float prev = R_v0_coeff_;
            R_v0_coeff_ = c;

            if (std::isfinite(prev) && prev > 0.0f) {
                const float scale = c / prev;

                if (std::isfinite(tune_.R_v0_std_applied) && tune_.R_v0_std_applied > 0.0f) {
                    tune_.R_v0_std_applied *= scale;
                }
                if (std::isfinite(R_v0_std_target_) && R_v0_std_target_ > 0.0f) {
                    R_v0_std_target_ *= scale;
                }
                if (enable_linear_block_) {
                    apply_R_v0_tune_();
                }
            }
        }
    }

    void setAccNoiseFloorSigma(float s) {
        if (std::isfinite(s) && s > 0.0f) {
            acc_noise_floor_sigma_ = s;
            if (mekf_) apply_wave_tune_();
        }
    }

    float getAccNoiseFloorSigma() const noexcept {
        return acc_noise_floor_sigma_;
    }

    void setFreqInputCutoffHz(float fc) {
        freq_input_lpf_.setCutoff(fc);
    }

    void enableClamp(bool flag = true) {
        enable_clamp_ = flag;
    }

    void enableTuner(bool flag = true) {
        enable_tuner_ = flag;
    }

    // Wave_5 linear block = [v, p, b_aw]
    void enableLinearBlock(bool flag = true) {
        enable_linear_block_ = flag;
        if (mekf_) {
            const bool on_now = flag && (startup_stage_ == StartupStage::Live);
            mekf_->set_linear_block_enabled(on_now);
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
        if (!std::isfinite(max_sigma_a) || max_sigma_a <= 0.0f) return;
        max_sigma_a_ = max_sigma_a;
    }

    void setR_p0_Bounds(float min_R_p0_std, float max_R_p0_std) {
        if (!std::isfinite(min_R_p0_std) || !std::isfinite(max_R_p0_std)) return;
        if (min_R_p0_std <= 0.0f || max_R_p0_std <= min_R_p0_std) return;
        MIN_R_p0_std_ = min_R_p0_std;
        MAX_R_p0_std_ = max_R_p0_std;
    }

    void setR_v0_Bounds(float min_R_v0_std, float max_R_v0_std) {
        if (!std::isfinite(min_R_v0_std) || !std::isfinite(max_R_v0_std)) return;
        if (min_R_v0_std <= 0.0f || max_R_v0_std <= min_R_v0_std) return;
        MIN_R_v0_std_ = min_R_v0_std;
        MAX_R_v0_std_ = max_R_v0_std;
    }

    void setAdaptationTimeConstants(float tau_sec) {
        if (std::isfinite(tau_sec) && tau_sec > 0.0f) {
            adapt_tau_sec_ = tau_sec;
        }
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

    void setFreezeAccBiasUntilLive(bool en) {
        freeze_acc_bias_until_live_ = en;
    }

    void setWarmupRaccStd(float r) {
        if (std::isfinite(r) && r > 0.0f) {
            Racc_warmup_std_ = r;
        }
    }

    // Baseline accel noise used for nominal axis ratios and Live scaling reference
    void setNominalRaccStd(const Eigen::Vector3f& r) {
        if (!r.allFinite()) return;
        if (!(r.minCoeff() > 0.0f)) return;
        Racc_nominal_std_ = r;
        if (mekf_ && !warmup_Racc_active_) {
            apply_wave_tune_();
        }
    }

    // Live Racc tuning knobs
    void setLiveRaccBaseMaxScale(float s) {
        if (std::isfinite(s) && s >= 1.0f) {
            live_racc_base_max_scale_ = s;
            if (mekf_ && !warmup_Racc_active_) apply_wave_tune_();
        }
    }

    void setLiveRaccBaseScalePower(float p) {
        if (std::isfinite(p) && p > 0.0f) {
            live_racc_base_scale_power_ = p;
            if (mekf_ && !warmup_Racc_active_) apply_wave_tune_();
        }
    }

    void setLiveMotionAdaptiveRaccEnabled(bool en) {
        motion_adaptive_racc_enabled_ = en;
        if (mekf_ && !warmup_Racc_active_) apply_wave_tune_();
    }

    void setLiveMotionAdaptiveRaccScales(float xy_max_scale, float z_max_scale) {
        if (std::isfinite(xy_max_scale) && xy_max_scale >= 1.0f) {
            live_racc_xy_motion_max_scale_ = xy_max_scale;
        }
        if (std::isfinite(z_max_scale) && z_max_scale >= 1.0f) {
            live_racc_z_motion_max_scale_ = z_max_scale;
        }
        if (mekf_ && !warmup_Racc_active_) apply_wave_tune_();
    }

    void setLiveMotionAdaptiveRaccPower(float p) {
        if (std::isfinite(p) && p > 0.0f) {
            live_motion_scale_power_ = p;
            if (mekf_ && !warmup_Racc_active_) apply_wave_tune_();
        }
    }

    void setMotionEmaTauSec(float tau_sec) {
        if (std::isfinite(tau_sec) && tau_sec > 0.0f) {
            motion_ema_tau_sec_ = tau_sec;
        }
    }

    void setMotionAccelRefG(float ref_g) {
        if (std::isfinite(ref_g) && ref_g > 0.0f) {
            motion_acc_ref_g_ = ref_g;
        }
    }

    void setMotionGyroRefDegPerSec(float degps) {
        if (std::isfinite(degps) && degps > 0.0f) {
            motion_gyro_ref_radps_ = degps * float(M_PI) / 180.0f;
        }
    }

    void setBiasRwBaseGain(float g) {
        if (std::isfinite(g) && g > 0.0f) {
            baw_gain_base_ = g;
            if (mekf_) apply_wave_tune_();
        }
    }

    void setBiasRwFloor(float s) {
        if (std::isfinite(s) && s > 0.0f) {
            baw_rw_floor_ = s;
            if (mekf_) apply_wave_tune_();
        }
    }

    inline float getFreqHz()            const noexcept { return freq_hz_; }
    inline float getFreqSlowHz()        const noexcept { return freq_hz_slow_; }
    inline float getFreqRawHz()         const noexcept { return f_raw; }

    // Wrapper sea-state quantities
    inline float getTauApplied()        const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied()      const noexcept { return tune_.sigma_applied; }

    // Actual core filter accel noise
    inline float getRaccAppliedZ()      const noexcept {
        return mekf_ ? mekf_->get_Racc_std().z() : NAN;
    }

    inline Eigen::Vector3f getRaccAppliedStd() const noexcept {
        return mekf_ ? mekf_->get_Racc_std() : Eigen::Vector3f::Constant(NAN);
    }

    inline float getMotionAccelEmaG() const noexcept { return motion_acc_ema_g_; }
    inline float getMotionGyroEmaDegPerSec() const noexcept {
        return motion_gyro_ema_radps_ * 180.0f / float(M_PI);
    }

    inline float getR_p0_std_applied()  const noexcept {
        return mekf_ ? mekf_->get_Rp0_noise_std().z() : NAN;
    }

    inline float getR_v0_std_applied()  const noexcept {
        return mekf_ ? mekf_->get_Rv0_noise_std().z() : NAN;
    }

    inline float getTauTarget()         const noexcept { return tau_target_; }
    inline float getSigmaTarget()       const noexcept { return sigma_target_; }
    inline float getR_p0_std_target()   const noexcept { return R_p0_std_target_; }
    inline float getR_v0_std_target()   const noexcept { return R_v0_std_target_; }

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
        const float tau   = smoothed ? tune_.tau_applied   : tau_target_;
        const float sigma = smoothed ? tune_.sigma_applied : sigma_target_;
        if (!std::isfinite(sigma) || !std::isfinite(tau)) return NAN;
        constexpr float C_HS  = 2.0f * std::sqrt(2.0f) / (M_PI * M_PI);
        return C_HS * sigma * tau * tau / 2.0f;
    }

    float getVerticalSpeedEnvelopeMps(bool smoothed = true) const noexcept {
        const float tau   = smoothed ? tune_.tau_applied   : tau_target_;
        const float sigma = smoothed ? tune_.sigma_applied : sigma_target_;
        if (!(tau > 1e-6f) || !std::isfinite(tau) || !std::isfinite(sigma)) return NAN;
        constexpr float K = std::sqrt(2.0f) / M_PI;
        const float v_env = speed_env_mult_ * K * sigma * tau;
        return std::isfinite(v_env) ? v_env : NAN;
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

        float rn = roll;
        float pn = pitch;
        float yn = yaw;
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
            if (std::isfinite(fc) && fc > 0.0f) {
                fc_hz = fc;
            }
        }

        float step(float x, float dt) {
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

        void setTargetFreqHz(float f) {
            if (std::isfinite(f) && f > 0.0f) {
                target_freq_hz = f;
            }
        }

        float step(float a_z_inertial_lp, float dt, float freq_in) {
            if (!(dt > 0.0f) || !std::isfinite(freq_in)) {
                return freq_in;
            }

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
                freq_state     = freq_in;
            }
            return freq_state;
        }

        bool  isStill()      const { return last_is_still; }
        float getStillTime() const { return still_time_sec; }
        float getEnergyEma() const { return energy_ema; }
    };

    static inline float clamp01_(float x) {
        return std::min(std::max(x, 0.0f), 1.0f);
    }

    void update_motion_state_(float dt,
                              const Eigen::Vector3f& gyro_body,
                              const Eigen::Vector3f& acc_body)
    {
        if (!(dt > 0.0f)) return;

        const float tau = std::max(1e-3f, motion_ema_tau_sec_);
        const float alpha = 1.0f - std::exp(-dt / tau);

        float acc_dev_g = 0.0f;
        if (acc_body.allFinite()) {
            acc_dev_g = std::fabs(acc_body.norm() - g_std) / std::max(g_std, 1e-6f);
            if (!std::isfinite(acc_dev_g)) acc_dev_g = 0.0f;
        }

        float gyro_norm = 0.0f;
        if (gyro_body.allFinite()) {
            gyro_norm = gyro_body.norm();
            if (!std::isfinite(gyro_norm)) gyro_norm = 0.0f;
        }

        motion_acc_ema_g_     += alpha * (acc_dev_g - motion_acc_ema_g_);
        motion_gyro_ema_radps_ += alpha * (gyro_norm - motion_gyro_ema_radps_);
    }

    float motion_level_() const {
        const float acc_norm  = motion_acc_ema_g_ / std::max(motion_acc_ref_g_, 1e-6f);
        const float gyro_norm = motion_gyro_ema_radps_ / std::max(motion_gyro_ref_radps_, 1e-6f);

        const float m = std::max(acc_norm, gyro_norm);
        return clamp01_(m);
    }

    Eigen::Vector3f compute_live_racc_cmd_(float sigma_hint) const {
        const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
        const float sigma_eff   = std::max(sigma_floor, sigma_hint);

        Eigen::Vector3f base;
        if (Racc_nominal_std_.allFinite() && Racc_nominal_std_.maxCoeff() > 0.0f) {
            base = Racc_nominal_std_;
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(base(i)) || !(base(i) > 0.0f)) {
                    base(i) = sigma_floor;
                }
            }
        } else {
            base = Eigen::Vector3f::Constant(sigma_floor);
        }

        // Mild sea-state driven isotropic baseline inflation
        const float ref   = std::max(base.z(), 1e-6f);
        const float ratio = std::max(1.0f, sigma_eff / ref);

        float base_scale = std::pow(ratio, live_racc_base_scale_power_);
        base_scale = std::min(std::max(base_scale, 1.0f), live_racc_base_max_scale_);

        Eigen::Vector3f out = base * base_scale;

        // Motion-adaptive anisotropic inflation only in Live
        if (startup_stage_ == StartupStage::Live && motion_adaptive_racc_enabled_) {
            const float m = motion_level_();
            const float shaped = std::pow(m, live_motion_scale_power_);

            const float xy_scale = 1.0f + (live_racc_xy_motion_max_scale_ - 1.0f) * shaped;
            const float z_scale  = 1.0f + (live_racc_z_motion_max_scale_  - 1.0f) * shaped;

            out.x() *= xy_scale;
            out.y() *= xy_scale;
            out.z() *= z_scale;
        }

        return out;
    }

    void apply_wave_tune_() {
        if (!mekf_) return;
    
        // During warmup preserve explicit warmup Racc
        if (warmup_Racc_active_) {
            return;
        }
    
        const float sigma_floor = std::max(0.05f, acc_noise_floor_sigma_);
        const float sigma_eff   = std::max(sigma_floor, tune_.sigma_applied);
    
        const Eigen::Vector3f racc_cmd = compute_live_racc_cmd_(sigma_eff);
        mekf_->set_Racc_std(racc_cmd);
    
        // Moderate fixed b_aw RW.
        // Decoupled from Racc, but not so small that p/v drift cannot be absorbed.
        const float baw_xy_std = std::max(baw_rw_floor_, baw_gain_base_ * P_factor_);
        const float baw_z_std  = std::max(baw_rw_floor_, baw_gain_base_);
    
        mekf_->set_world_accel_bias_rw_std(
            Eigen::Vector3f(baw_xy_std, baw_xy_std, baw_z_std));
    }

    void apply_R_p0_tune_(float rp_scale = 1.0f) {
        if (!mekf_) return;
        const float p = (std::isfinite(rp_scale) && rp_scale > 0.0f)
                        ? std::min(rp_scale, 1.0f) : 1.0f;
        const float R_p0_b = std::min(std::max(tune_.R_p0_std_applied, MIN_R_p0_std_), MAX_R_p0_std_);
        const float rp_xy = R_p0_b * p * R_p0_xy_factor_;
        mekf_->set_Rp0_noise_std(Eigen::Vector3f(rp_xy, rp_xy, R_p0_b * p));
    }

    void apply_R_v0_tune_(float rv_scale = 1.0f) {
        if (!mekf_) return;
        const float p = (std::isfinite(rv_scale) && rv_scale > 0.0f)
                        ? std::min(rv_scale, 1.0f) : 1.0f;
        const float R_v0_b = std::min(std::max(tune_.R_v0_std_applied, MIN_R_v0_std_), MAX_R_v0_std_);
        mekf_->set_Rv0_noise_std(Eigen::Vector3f::Constant(R_v0_b * p));
    }

    void update_tuner(float dt, float a_vert_inertial, float freq_hz_for_tuner) {
        tuner_.update(dt, a_vert_inertial, freq_hz_for_tuner);

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
        if (!std::isfinite(f_tune) || f_tune < min_freq_hz_) {
            f_tune = min_freq_hz_;
        }
        if (f_tune > max_freq_hz_) {
            f_tune = max_freq_hz_;
        }

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
        const float sigma_wave = std::sqrt(var_wave);
        const float tau_raw = tau_coeff_ * 0.5f / f_tune;

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

        float R_p0_raw = R_p0_coeff_ * sigma_target_ * tau_target_ * tau_target_;
        float R_v0_raw = R_v0_coeff_ * sigma_target_ * tau_target_;

        if (enable_clamp_) {
            R_p0_std_target_ = std::min(std::max(R_p0_raw, MIN_R_p0_std_), MAX_R_p0_std_);
            R_v0_std_target_ = std::min(std::max(R_v0_raw, MIN_R_v0_std_), MAX_R_v0_std_);
        } else {
            R_p0_std_target_ = R_p0_raw;
            R_v0_std_target_ = R_v0_raw;
        }

        adapt_mekf(dt, tau_target_, sigma_target_, R_p0_std_target_, R_v0_std_target_);
    }

    void adapt_mekf(float dt, float tau_t, float sigma_t, float R_p0_t, float R_v0_t) {
        const float alpha = 1.0f - std::exp(-dt / adapt_tau_sec_);

        const float R_p0_sec   = ADAPT_R_p0_MULT * tau_t;
        const float R_v0_sec   = ADAPT_R_v0_MULT * tau_t;
        const float alpha_R_p0 = 1.0f - std::exp(-dt / R_p0_sec);
        const float alpha_R_v0 = 1.0f - std::exp(-dt / R_v0_sec);

        tune_.tau_applied       += alpha      * (tau_t   - tune_.tau_applied);
        tune_.sigma_applied     += alpha      * (sigma_t - tune_.sigma_applied);
        tune_.R_p0_std_applied  += alpha_R_p0 * (R_p0_t  - tune_.R_p0_std_applied);
        tune_.R_v0_std_applied  += alpha_R_v0 * (R_v0_t  - tune_.R_v0_std_applied);

        if (time_ - last_adapt_time_sec_ > adapt_every_secs_) {
            apply_wave_tune_();

            if (startup_stage_ == StartupStage::Live && enable_linear_block_) {
                apply_R_p0_tune_();
                apply_R_v0_tune_();
            }
            last_adapt_time_sec_ = time_;
        }
    }

    void resetTrackingState_() {
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

        dir_filter_      = KalmanWaveDirection(2.0f * static_cast<float>(M_PI) * FREQ_GUESS);
        dir_sign_state_  = UNCERTAIN;

        motion_acc_ema_g_ = 0.0f;
        motion_gyro_ema_radps_ = 0.0f;

        last_adapt_time_sec_ = time_;
    }

    void enterCold_() {
        startup_stage_   = StartupStage::Cold;
        startup_stage_t_ = 0.0f;

        if (!mekf_) return;

        mekf_->set_linear_block_enabled(false);

        accel_bias_locked_   = with_mag_;
        mag_updates_applied_ = 0;
        first_mag_update_time_ = NAN;

        motion_acc_ema_g_ = 0.0f;
        motion_gyro_ema_radps_ = 0.0f;

        if (freeze_acc_bias_until_live_) {
            mekf_->set_acc_bias_updates_enabled(false);
            mekf_->set_Racc_std(Eigen::Vector3f::Constant(Racc_warmup_std_));
            warmup_Racc_active_ = true;
        }
    }

    void enterLive_() {
        startup_stage_   = StartupStage::Live;
        startup_stage_t_ = 0.0f;

        if (!mekf_) return;

        mekf_->set_linear_block_enabled(enable_linear_block_);

        if (freeze_acc_bias_until_live_) {
            const bool allow_bias = !accel_bias_locked_;
            mekf_->set_acc_bias_updates_enabled(allow_bias);

            if (warmup_Racc_active_ &&
                Racc_nominal_std_.allFinite() &&
                Racc_nominal_std_.maxCoeff() > 0.0f) {
                mekf_->set_Racc_std(Racc_nominal_std_);
            }
            warmup_Racc_active_ = false;
        }

        apply_wave_tune_();

        if (enable_linear_block_) {
            apply_R_p0_tune_();
            apply_R_v0_tune_();
        }
    }

    StartupStage startup_stage_   = StartupStage::Cold;
    float        startup_stage_t_ = 0.0f;

    bool  freeze_acc_bias_until_live_ = true;
    float Racc_warmup_std_            = 0.5f;
    bool  warmup_Racc_active_         = false;
    Eigen::Vector3f Racc_nominal_std_ = Eigen::Vector3f::Constant(0.0f);

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
    float f_raw         = FREQ_GUESS;

    float a_vert_up = 0.0f;

    bool enable_clamp_ = true;
    bool enable_tuner_ = true;

    float speed_env_mult_ = 1.0f;
    bool  enable_linear_block_ = true;

    float min_freq_hz_            = MIN_FREQ_HZ;
    float max_freq_hz_            = MAX_FREQ_HZ;
    float min_tau_s_              = MIN_TAU_S;
    float max_tau_s_              = MAX_TAU_S;
    float max_sigma_a_            = MAX_SIGMA_A;
    float MIN_R_p0_std_           = MIN_R_p0_std;
    float MAX_R_p0_std_           = MAX_R_p0_std;
    float MIN_R_v0_std_           = MIN_R_v0_std;
    float MAX_R_v0_std_           = MAX_R_v0_std;
    float adapt_tau_sec_          = ADAPT_TAU_SEC;
    float adapt_every_secs_       = ADAPT_EVERY_SECS;
    float online_tune_warmup_sec_ = ONLINE_TUNE_WARMUP_SEC;
    float mag_delay_sec_          = MAG_DELAY_SEC;

    float R_p0_xy_factor_ = 0.23f;
    float P_factor_       = 1.5f;

    // Wave_5 retune controls
    float live_racc_base_max_scale_   = 1.35f;  // isotropic sea-state inflation cap
    float live_racc_base_scale_power_ = 0.25f;  // sublinear sigma->Racc baseline
    float baw_gain_base_              = 0.035f; // interpreted now as base RW std / sqrt(s)
    float baw_rw_floor_               = 0.006f; // floor std / sqrt(s)

    // Motion-adaptive Live Racc
    bool  motion_adaptive_racc_enabled_  = true;
    float live_racc_xy_motion_max_scale_ = 2.2f; // stronger XY inflation
    float live_racc_z_motion_max_scale_  = 1.2f; // weaker Z inflation
    float live_motion_scale_power_       = 1.0f;

    float motion_ema_tau_sec_       = 0.25f;
    float motion_acc_ref_g_         = 0.14f;                     // 0.10 g accel-deviation reference
    float motion_gyro_ref_radps_    = 35.0f * float(M_PI) / 180.0f; // 35 deg/s reference
    float motion_acc_ema_g_         = 0.0f;
    float motion_gyro_ema_radps_    = 0.0f;

    TrackingPolicy                  tracker_policy_{};
    FirstOrderIIRSmoother<float>    freq_fast_smoother_{FREQ_SMOOTHER_DT, 3.5f};
    FirstOrderIIRSmoother<float>    freq_slow_smoother_{FREQ_SMOOTHER_DT, 10.0f};
    SeaStateAutoTuner               tuner_;
    TuneState                       tune_;

    float tau_target_      = NAN;
    float sigma_target_    = NAN;
    float R_p0_std_target_ = NAN;
    float R_v0_std_target_ = NAN;

    float acc_noise_floor_sigma_ = ACC_NOISE_FLOOR_SIGMA_DEFAULT;

    float R_p0_coeff_   = 16.0f;
    float R_v0_coeff_   = 32.0f;
    float tau_coeff_    = 1.7f;
    float sigma_coeff_  = 0.85f;

    std::unique_ptr<MekfT>          mekf_;
    KalmanWaveDirection             dir_filter_{2.0f * static_cast<float>(M_PI) * FREQ_GUESS};

    FreqInputLPF                    freq_input_lpf_;
    StillnessAdapter                freq_stillness_;

    WaveDirectionDetector<float>    dir_sign_{0.002f, 0.005f};
    WaveDirection                   dir_sign_state_ = UNCERTAIN;
};

template<TrackerType trackerT>
class SeaStateFusion_II {
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
        float mag_odr_guess_hz = 25.0f;

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
        have_last_imu_ = false;

        impl_.setWithMag(cfg.with_mag);
        impl_.setFreezeAccBiasUntilLive(cfg.freeze_acc_bias_until_live);
        impl_.setWarmupRaccStd(cfg.Racc_warmup);
        impl_.setMagDelaySec(cfg.mag_delay_sec);
        impl_.setOnlineTuneWarmupSec(cfg.online_tune_warmup_sec);

        impl_.initialize(cfg.sigma_a, cfg.sigma_g, cfg.sigma_m);
        last_impl_startup_stage_ = impl_.getStartupStage();

        impl_.setNominalRaccStd(cfg.sigma_a);
    }

    void update(float dt, const Eigen::Vector3f& gyro_body_ned,
                const Eigen::Vector3f& acc_body_ned)
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
        have_last_imu_      = true;

        impl_.updateTime(dt, gyro_body_ned, acc_body_ned);

        const auto cur_stage = impl_.getStartupStage();

        if (cur_stage != last_impl_startup_stage_) {
            if (cur_stage == SeaStateFusionFilter_II<trackerT>::StartupStage::Cold) {
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

        if (std::isfinite(last_mag_time_sec_)) {
            dt_mag_sec_ = t_ - last_mag_time_sec_;
        }
        last_mag_time_sec_ = t_;

        if (have_last_imu_) {
            float dtm = dt_mag_sec_;
            if (!std::isfinite(dtm) || dtm <= 0.0f) {
                dtm = 1.0f / std::max(1.0f, cfg_.mag_odr_guess_hz);
            }
            (void)mag_auto_.addMagSample(dtm, last_acc_body_ned_, mag_body_ned, last_gyro_body_ned_);
        }

        if (have_last_imu_) {
            const float an = last_acc_body_ned_.norm();
            const float mn = mag_body_ned.norm();
            const float g = 9.80665f;
            const bool accel_ok = last_acc_body_ned_.allFinite() && std::fabs(an - g) < 0.12f * g;
            const bool mag_ok   = mag_body_ned.allFinite() && (mn > 1e-3f);
            const float gyro_n = last_gyro_body_ned_.norm();
            const bool gyro_ok = last_gyro_body_ned_.allFinite() &&
                                 (gyro_n < (60.0f * float(M_PI) / 180.0f));
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
                    } else if (fallback_mean_count_ >= 200 &&
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

        if (mag_ref_set_) {
            impl_.updateMag(mag_body_ned);
        }
    }

    bool  isLive() const { return stage_ == Stage::Live; }
    float freqHz() const { return impl_.getFreqHz(); }
    Eigen::Vector3f eulerNauticalDeg() const { return impl_.getEulerNautical(); }

    SeaStateFusionFilter_II<trackerT>& raw() { return impl_; }

private:
    enum class Stage { Uninitialized, Warming, Live };

    static Eigen::Quaternionf tiltOnlyQuatFromAccel_(const Eigen::Vector3f& acc_body_ned) {
        Eigen::Vector3f a = acc_body_ned;
        const float an = a.norm();
        if (!(an > 1e-6f) || !a.allFinite()) {
            return Eigen::Quaternionf::Identity();
        }

        Eigen::Vector3f body_down = (-a / an);
        const Eigen::Vector3f world_down(0.0f, 0.0f, 1.0f);

        const float d = std::max(-1.0f, std::min(1.0f, body_down.dot(world_down)));
        Eigen::Vector3f axis = body_down.cross(world_down);
        const float axis_n = axis.norm();

        if (axis_n < 1e-6f) {
            if (d > 0.0f) {
                return Eigen::Quaternionf::Identity();
            } else {
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
    SeaStateFusionFilter_II<trackerT> impl_{false};

    bool begun_ = false;

    Stage stage_ = Stage::Uninitialized;
    float t_ = 0.0f;
    float stage_t_ = 0.0f;

    typename SeaStateFusionFilter_II<trackerT>::StartupStage last_impl_startup_stage_ =
        SeaStateFusionFilter_II<trackerT>::StartupStage::Cold;

    bool mag_ref_set_ = false;
    Eigen::Vector3f mag_body_hold_ = Eigen::Vector3f::Zero();

    float last_mag_time_sec_ = NAN;
    float dt_mag_sec_ = NAN;
    float mag_ref_deadline_sec_ = NAN;

    Eigen::Vector3f last_acc_body_ned_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f last_gyro_body_ned_ = Eigen::Vector3f::Zero();
    bool  have_last_imu_ = false;

    Eigen::Vector3f fallback_acc_mean_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f fallback_mag_mean_ = Eigen::Vector3f::Zero();
    int fallback_mean_count_ = 0;

    MagAutoTuner mag_auto_;
};
