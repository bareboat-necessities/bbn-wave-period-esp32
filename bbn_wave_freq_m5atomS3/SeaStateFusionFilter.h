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
#include "FrequencySmoother.h"
#include "SeaStateAutoTuner.h"
#include "Kalman3D_Wave.h"
#include "FrameConversions.h"
#include "KalmanWaveDirection.h"
#include "WaveDirectionDetector.h"

// Shared constants
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 5.0f;

constexpr float MIN_TAU_S   = 0.5f;
constexpr float MAX_TAU_S   = 8.5f;
constexpr float MIN_SIGMA_A = 0.3f;
constexpr float MAX_SIGMA_A = 8.0f;
constexpr float MIN_R_S     = 0.1f;
constexpr float MAX_R_S     = 35.0f;

constexpr float ADAPT_TAU_SEC = 3.0f;
constexpr float ADAPT_EVERY_SECS = 0.1f;
constexpr float ONLINE_TUNE_WARMUP_SEC = 35.0f;
constexpr float MAG_DELAY_SEC = 5.0f;

struct TuneState {
    float tau_applied   = 0.97f;              // s
    float sigma_applied = 0.65f;              // m/s²
    float RS_applied    = 0.87f;              // m*s
    //float tau_applied   = 0.92f;
    //float sigma_applied = 0.57f;
    //float RS_applied    = 1.17f;
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
        double omega_up = (FREQ_GUESS * 2) * (2 * M_PI);  // upper angular frequency
        double k_gain = 20.0; // Higher = faster, but risk overflow if too high
        double x1_0 = 0.0;
        double omega_init = (FREQ_GUESS / 1.5) * 2 * M_PI;
        double theta_0 = -(omega_init * omega_init);
        double sigma_0 = theta_0;
        t.setParams(omega_up, k_gain);
        t.setState(x1_0, theta_0, sigma_0);
    }

    double run(float a, float dt) {
        t.update((double)a / g_std, (double)dt);
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
        double freq = t.process((double)a / g_std, (double)dt, &e);
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
        float f_byZeroCross = t.update(a / g_std, ZERO_CROSSINGS_SCALE /* max g */,
                              ZERO_CROSSINGS_DEBOUNCE_TIME, ZERO_CROSSINGS_STEEPNESS_TIME, dt);
        double freq = (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT || f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ) ?
           FREQ_GUESS : f_byZeroCross;
        return freq;
    }
};

//  Unified SeaState fusion filter
template<TrackerType trackerT>
class SeaStateFusionFilter {
public:
    using TrackingPolicy  = TrackerPolicy<trackerT>;

    explicit SeaStateFusionFilter(bool with_mag = true)
        : with_mag_(with_mag), tuner_(),
          time_(0.0), last_adapt_time_sec_(0.0), freq_hz_(NAN)
    {}

    void initialize(const Eigen::Vector3f& sigma_a,
                    const Eigen::Vector3f& sigma_g,
                    const Eigen::Vector3f& sigma_m)
    {
        mekf_ = std::make_unique<Kalman3D_Wave<float,true,true>>(sigma_a, sigma_g, sigma_m);
        mekf_->set_exact_att_bias_Qd(true);
        apply_tune();
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) mekf_->initialize_from_acc(acc_world);
    }

    // Time update (IMU integration + frequency tracking)
    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc, float tempC = 35.0f) {
        if (!mekf_) return;
        time_ += dt;
    
        // MEKF
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc, tempC);
    
        // Tracker input: vertical inertial (BODY)
        const float a_z_inertial = acc.z() + g_std;
    
        // Raw freq from tracker, clamp
        const double f_raw = tracker_policy_.run(a_z_inertial, dt);
        const float  f_clamped = std::min(std::max(static_cast<float>(f_raw), MIN_FREQ_HZ), MAX_FREQ_HZ);
    
        // Smooth ONCE here
        if (!freq_init_) { freqSmoother.setInitial(f_clamped); freq_init_ = true; }
        const float f_smooth = freqSmoother.update(f_clamped);
    
        // Store the smoothed “truth” for everyone to reuse
        freq_hz_ = f_smooth;
    
        // Tuner uses the SAME smoothed freq
        if (enable_tuner) {
            update_tuner(dt, a_z_inertial, f_smooth);
        }
    
        // Direction filter also uses the SAME smoothed freq (ω = 2πf)
        const float omega = 2.0f * static_cast<float>(M_PI) * f_smooth;
        dir_filter_.update(acc.x(), acc.y(), omega, dt);
        dir_sign_state_ = dir_sign_.update(acc.x(), acc.y(), a_z_inertial, dt);  
    }

    //  Magnetometer correction
    void updateMag(const Eigen::Vector3f& mag_body_ned) {
        if (with_mag_ && mekf_ && time_ >= MAG_DELAY_SEC)
            mekf_->measurement_update_mag_only(mag_body_ned);
    }

    // Anisotropy configuration (runtime)
    // S-factor scales horizontal vs vertical stationary std of a_w.
    // RS XY factor scales pseudo-measurement noise in X/Y vs Z.
    void setSFactor(float s) {
        if (std::isfinite(s) && s > 0.0f) {
            S_factor = s;
        }
    }
    void setRSXYFactor(float k) {
        if (std::isfinite(k)) {
            R_S_xy_factor = std::min(std::max(k, 0.0f), 1.0f);
        }
    }

    void setTauCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            tau_coeff = c;
        }
    }

    void setRSCoeff(float c) {
        if (std::isfinite(c) && c > 0.0f) {
            R_S_coeff = c;
        }
    }

    void enableClamp(bool flag = true) {
        enable_clamp = flag;
    }
    void enableTuner(bool flag = true) {
        enable_tuner = flag;
    }

    //  Exposed getters
    inline float getFreqHz()        const noexcept { return freq_hz_; }
    inline float getTauApplied()    const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied()  const noexcept { return tune_.sigma_applied; }
    inline float getRSApplied()     const noexcept { return tune_.RS_applied; }
    inline float getTauTarget()     const noexcept { return tau_target_;   }
    inline float getSigmaTarget()   const noexcept { return sigma_target_; }
    inline float getRSTarget()      const noexcept { return RS_target_;    }
    inline float getPeriodSec()     const noexcept { return (freq_hz_ > 1e-6f) ? 1.0f / freq_hz_ : NAN; }
    inline float getAccelVariance() const noexcept { return tuner_.getAccelVariance(); }

    inline WaveDirection getDirSignState() const noexcept { return dir_sign_state_; }

    Eigen::Vector3f getEulerNautical() const {
        if (!mekf_) return {NAN, NAN, NAN};

        // q_bw: body→world (Kalman3D_Wave::quaternion() already returns qref.conjugate())
        Eigen::Quaternionf q_bw = mekf_->quaternion();
        q_bw.normalize();

        const float x = q_bw.x(), y = q_bw.y(), z = q_bw.z(), w = q_bw.w();
        const float two = 2.0f;

        // ZYX (aerospace) from q_bw — radians
        const float s_yaw   = two * std::fma(w, z,  x * y);
        const float c_yaw   = 1.0f - two * std::fma(y, y,  z * z);
        float yaw           = std::atan2(s_yaw, c_yaw);

        float s_pitch       = two * std::fma(w, y, -z * x);
        s_pitch             = std::max(-1.0f, std::min(1.0f, s_pitch));
        float pitch         = std::asin(s_pitch);

        const float s_roll  = two * std::fma(w, x,  y * z);
        const float c_roll  = 1.0f - two * std::fma(x, x,  y * y);
        float roll          = std::atan2(s_roll, c_roll);

        // Aerospace/NED → Nautical/ENU (expects radians)
        float rn = roll, pn = pitch, yn = yaw;
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
    //  Internal tuning and adaptation
    void apply_tune() {
        if (!mekf_) return;
    
        // OU time constant
        mekf_->set_aw_time_constant(tune_.tau_applied);
    
        // WORLD-frame stationary covariance for a_w (XY equal, Z separate).
        const float sZ = std::max(1e-6f, tune_.sigma_applied);
        const float sH = sZ * S_factor; 
        Eigen::Vector3f a_w_std = Eigen::Vector3f(sH, sH, sZ);
        mekf_->set_aw_stationary_std(a_w_std); 
    
        // WORLD-frame pseudo-measurement noise for S (anisotropic diagonal).
        // Clamp to configured bounds for robustness.
        const float RSb = std::min(std::max(tune_.RS_applied, MIN_R_S), MAX_R_S);
        mekf_->set_RS_noise(Eigen::Vector3f(
            RSb * R_S_xy_factor,   // world X
            RSb * R_S_xy_factor,   // world Y
            RSb                    // world Z
        ));
    }

    void update_tuner(float dt, float a_vert_inertial, float freq_hz_smooth) {
        // No smoothing here – it already happened in updateTime()
        tuner_.update(dt, a_vert_inertial, freq_hz_smooth);
    
        if (time_ < ONLINE_TUNE_WARMUP_SEC) return;

        if (enable_clamp) {
            tau_target_   = std::min(std::max(tau_coeff * 0.5f / tuner_.getFrequencyHz(), MIN_TAU_S), MAX_TAU_S);
            sigma_target_ = std::min(std::max(std::sqrt(std::max(0.0f, tuner_.getAccelVariance())), MIN_SIGMA_A), MAX_SIGMA_A);
            RS_target_    = std::min(std::max(R_S_coeff * sigma_target_ * tau_target_ * tau_target_ * tau_target_, MIN_R_S), MAX_R_S);
        } else {
            tau_target_   = tau_coeff * 0.5f / tuner_.getFrequencyHz();
            sigma_target_ = std::sqrt(std::max(0.0f, tuner_.getAccelVariance()));
            RS_target_    = R_S_coeff * sigma_target_ * tau_target_ * tau_target_ * tau_target_;
        }      
    
        adapt_mekf(dt, tau_target_, sigma_target_, RS_target_);
    }
    
    void adapt_mekf(float dt, float tau_t, float sigma_t, float RS_t) {
        const float alpha = 1.0f - std::exp(-dt / ADAPT_TAU_SEC);
        tune_.tau_applied   += alpha * (tau_t - tune_.tau_applied);
        tune_.sigma_applied += alpha * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha * (RS_t - tune_.RS_applied);

        if (time_ - last_adapt_time_sec_ > ADAPT_EVERY_SECS) {
            apply_tune();
            last_adapt_time_sec_ = time_;
        }
    }

    //  Members
    bool with_mag_;
    double time_, last_adapt_time_sec_;
    float freq_hz_ = FREQ_GUESS;
    bool freq_init_ = false;
    bool enable_clamp = true;    
    bool enable_tuner = true;

    // Runtime-configurable anisotropy knobs
    float R_S_xy_factor = 0.07f;  // [0..1] scales XY pseudo-meas vs Z
    float S_factor = 1.3f;       // (>0) scales Σ_aw horizontal std vs vertical

    TrackingPolicy tracker_policy_{};  // one instance of frequency tracker per filter
    FrequencySmoother<float> freqSmoother;
    SeaStateAutoTuner tuner_;
    TuneState tune_;

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    float R_S_coeff   = 2.4f;
    float tau_coeff   = 1.6f;

    std::unique_ptr<Kalman3D_Wave<float,true,true>> mekf_;
    KalmanWaveDirection dir_filter_{ 2.0f * static_cast<float>(M_PI) * FREQ_GUESS };  // FREQ_GUESS in Hz → ω0

    WaveDirectionDetector<float> dir_sign_{ 0.002f, 0.005f }; // smoothing, sensitivity
    WaveDirection                dir_sign_state_ = UNCERTAIN;
};
