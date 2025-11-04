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
  • Sensor-only Mahony tilt proxy for adaptation (no MEKF feedback)
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

// Mahony tilt-only observer
struct TiltMahony {
    float g_std    = 9.80665f; // m/s^2
    float kp       = 2.8f;     // base P gain
    float ki       = 0.05f;    // I gain (gyro bias)
    float gate_rel = 0.18f;    // trust window: accept accel ~ g*(1±gate_rel)

    Eigen::Vector3f zhat_b = {0,0,1}; // world +Z (down) in BODY
    Eigen::Vector3f bg     = {0,0,0}; // local gyro bias (not shared with MEKF)

    static inline Eigen::Matrix3f skew(const Eigen::Vector3f& v){
        Eigen::Matrix3f S;
        S << 0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0;
        return S;
    }

    inline float trust_from_norm(float a_norm) const {
        // linear ramp: 1 inside g*(1±gate), fades to 0 outside
        const float gmin = g_std * (1.0f - gate_rel);
        const float gmax = g_std * (1.0f + gate_rel);
        if (a_norm <= gmin || a_norm >= gmax) return 0.0f;
        const float d = std::min(a_norm - gmin, gmax - a_norm);
        return std::max(0.0f, std::min(1.0f, d / (g_std * gate_rel)));
    }

    inline void update(float dt,
                       const Eigen::Vector3f& gyr_b_rad_s,
                       const Eigen::Vector3f& acc_b_mps2)
    {
        if (!(dt > 0)) return;

        const float a = acc_b_mps2.norm();
        const float w = trust_from_norm(a);      // 0…1 trust weight

        // "Measured" world-down in BODY when believable
        Eigen::Vector3f m = (a > 1e-6f) ? (-acc_b_mps2 / a) : zhat_b;

        // Tilt-only error
        Eigen::Vector3f e = zhat_b.cross(m);

        // PI gyro with TRUST on the correction/bias
        const float kpw = kp * w;
        const float kiw = ki * w;

        Eigen::Vector3f omega = (gyr_b_rad_s - bg) + kpw * e;

        // Integrate zhat_b: v̇ = -ω × v
        const float th = omega.norm() * dt;
        if (th < 1e-6f) {
            zhat_b += dt * ( -omega.cross(zhat_b) );
        } else {
            const Eigen::Vector3f u = omega / (th + 1e-12f);
            const Eigen::Matrix3f K = skew(u);
            const Eigen::Matrix3f R = Eigen::Matrix3f::Identity()
                                    - std::sin(th)*K + (1-std::cos(th))*(K*K);
            zhat_b = R * zhat_b;
        }
        zhat_b.normalize();

        // Bias only integrates when accel is trustworthy
        if (kiw > 0.0f) bg += kiw * e * dt;
    }

    inline Eigen::Vector3f g_body() const { return g_std * zhat_b; }

    // inertial accel in BODY:  a_b(inertial) = f_b + g_b
    inline Eigen::Vector3f a_body_inertial(const Eigen::Vector3f& acc_b) const {
        return acc_b + g_body();
    }

    // yaw-free tilt frame (Z=down)
    inline Eigen::Matrix3f R_tilt_to_body() const {
        const Eigen::Vector3f z = zhat_b;
        Eigen::Vector3f x = (Eigen::Matrix3f::Identity() - z*z.transpose()) * Eigen::Vector3f::UnitX();
        if (x.squaredNorm() < 1e-8f)
            x = (Eigen::Matrix3f::Identity() - z*z.transpose()) * Eigen::Vector3f::UnitY();
        x.normalize();
        Eigen::Vector3f y = z.cross(x); y.normalize();
        Eigen::Matrix3f R; R.col(0)=x; R.col(1)=y; R.col(2)=z; // tilt→body
        return R;
    }

    inline Eigen::Vector3f inertial_accel_tilt(const Eigen::Vector3f& acc_b) const {
        const Eigen::Vector3f a_b = a_body_inertial(acc_b);
        return R_tilt_to_body().transpose() * a_b; // body→tilt
    }
};

struct CorrXZEstimator {
    float tau = 30.0f;     // EWMA horizon [s] for covariance
    float last_dt = -1.0f;
    float alpha = 0.0f;
    float w = 0.0f;        // debias weight accumulator

    Eigen::Vector3f m = Eigen::Vector3f::Zero();   // E[x]
    Eigen::Matrix3f Exx = Eigen::Matrix3f::Zero(); // E[xxᵀ]

    inline void setTau(float t){ tau = std::max(1e-3f, t); last_dt = -1.0f; }
    inline void reset(){ last_dt=-1.0f; alpha=0.0f; w=0.0f; m.setZero(); Exx.setZero(); }

    inline void update(float dt, const Eigen::Vector3f& a_tilt) {
        if (!(dt > 0)) return;
        if (dt != last_dt) { alpha = 1.0f - std::exp(-dt / tau); last_dt = dt; }
        m   = (1.0f - alpha) * m   + alpha * a_tilt;
        Exx = (1.0f - alpha) * Exx + alpha * (a_tilt * a_tilt.transpose());
        w   = (1.0f - alpha) * w   + alpha;
    }

    inline bool ready() const { return w > 1e-3f; }

    inline Eigen::Vector3f var() const {
        Eigen::Vector3f v = Exx.diagonal() - m.array().square().matrix();
        return v.cwiseMax(0.0f);
    }

    inline float cov_xz() const { return Exx(0,2) - m.x()*m.z(); }
    inline float cov_yz() const { return Exx(1,2) - m.y()*m.z(); }

    inline float rho_avg(float eps=1e-6f) const {
        auto v = var();
        const float sx = std::sqrt(v.x() + eps);
        const float sy = std::sqrt(v.y() + eps);
        const float sz = std::sqrt(v.z() + eps);
        const float rx = cov_xz() / (sx*sz + eps);
        const float ry = cov_yz() / (sy*sz + eps);
        // clip for PSD safety
        const float r  = 0.5f * (rx + ry);
        return std::max(-0.95f, std::min(0.95f, r));
    }
};

// Shared constants
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 5.0f;

constexpr float MIN_TAU_S   = 0.5f;
constexpr float MAX_TAU_S   = 11.5f;
constexpr float MIN_SIGMA_A = 0.3f;
constexpr float MAX_SIGMA_A = 8.0f;
constexpr float MIN_R_S     = 0.1f;
constexpr float MAX_R_S     = 40.0f;

constexpr float R_S_coeff   = 2.5f;
constexpr float tau_coeff   = 1.6f;

constexpr float ADAPT_TAU_SEC = 3.0f;
constexpr float ADAPT_EVERY_SECS = 0.1f;
constexpr float ONLINE_TUNE_WARMUP_SEC = 40.0f;
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
        apply_tune();
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) mekf_->initialize_from_acc(acc_world);
    }

    //  Time update (IMU integration + frequency tracking)
    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc) {
        if (!mekf_) return;
        time_ += dt;

        // MEKF updates (independent)
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc);

        // Sensor-only tilt proxy → inertial accel in yaw-free tilt frame
        tilt_.update(dt, gyro, acc);
        const Eigen::Vector3f a_tilt = tilt_.inertial_accel_tilt(acc);

        // Feed tracker with world-like vertical acceleration (tilt Z)
        const double f = tracker_policy_.run(a_tilt.z(), dt);
        if (!std::isnan(f)) {
            freq_hz_ = std::min(std::max(static_cast<float>(f), MIN_FREQ_HZ), MAX_FREQ_HZ);
            // Keep scalar tuner API, but use tilt-vertical instead of acc.z()+g
            update_tuner(dt, a_tilt.z(), freq_hz_);
        } else {
            freq_hz_ = FREQ_GUESS;
        }
    }

    //  Magnetometer correction
    void updateMag(const Eigen::Vector3f& mag_world) {
        if (with_mag_ && mekf_ && time_ >= MAG_DELAY_SEC)
            mekf_->measurement_update_mag_only(mag_world);
    }

    //  Exposed getters
    inline float getFreqHz()       const noexcept { return freq_hz_; }
    inline float getTauApplied()   const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied() const noexcept { return tune_.sigma_applied; }
    inline float getRSApplied()    const noexcept { return tune_.RS_applied; }
    inline float getTauTarget()    const noexcept { return tau_target_;   }
    inline float getSigmaTarget()  const noexcept { return sigma_target_; }
    inline float getRSTarget()     const noexcept { return RS_target_;    }
    inline float getPeriodSec()    const noexcept { return (freq_hz_ > 1e-6f) ? 1.0f / freq_hz_ : NAN; }
    inline float getAccelVariance()const noexcept { return tuner_.getAccelVariance(); }

    Eigen::Vector3f getEulerNautical() const {
        if (!mekf_) return Eigen::Vector3f::Zero();

        // Fetch quaternion in Eigen coeff order (x, y, z, w)
        const auto coeffs = mekf_->quaternion().coeffs();
        Eigen::Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2)); // w,x,y,z

        // Convert from aerospace (body-to-world, NED) to nautical (Z-up ENU)
        float roll_a, pitch_a, yaw_a;
        quat_to_euler_aero(q, roll_a, pitch_a, yaw_a);

        float roll_n = roll_a;
        float pitch_n = pitch_a;
        float yaw_n = yaw_a;
        aero_to_nautical(roll_n, pitch_n, yaw_n);

        return Eigen::Vector3f(roll_n, pitch_n, yaw_n);
    }

    inline const auto& mekf() const noexcept { return *mekf_; }
    inline auto& mekf() noexcept { return *mekf_; }

private:
    //  Internal tuning and adaptation
    void apply_tune() {
        if (!mekf_) return;
        mekf_->set_aw_time_constant(tune_.tau_applied);

        // Keep anisotropic Σ (XY boosted) with neutral ρ=0 for now
        mekf_->set_aw_stationary_corr_std(
            Eigen::Vector3f(tune_.sigma_applied * S_factor,
                            tune_.sigma_applied * S_factor,
                            tune_.sigma_applied),
            0.0f);

        // Keep anisotropic R_S (XY reduced)
        mekf_->set_RS_noise(Eigen::Vector3f(tune_.RS_applied * R_S_xy_factor,
                                            tune_.RS_applied * R_S_xy_factor,
                                            tune_.RS_applied));
    }

    void update_tuner(float dt, float a_vert_tilt_frame, float freq_hz) {
        if (!std::isfinite(freq_hz)) {
            freqSmoother.setInitial(FREQ_GUESS);
            return;
        }
        if (!freq_init_) {
            freqSmoother.setInitial(freq_hz);
            freq_init_ = true;
        }

        float smoothFreq = freqSmoother.update(freq_hz);
        if (time_ < ONLINE_TUNE_WARMUP_SEC)  {
            return;
        }

        // Scalar tuner fed with tilt-vertical inertial acceleration
        tuner_.update(dt, a_vert_tilt_frame, smoothFreq);

        tau_target_   = std::min(std::max(tau_coeff * 0.5f / tuner_.getFrequencyHz(), MIN_TAU_S), MAX_TAU_S);
        sigma_target_ = std::min(std::max(
            std::sqrt(std::max(0.0f, tuner_.getAccelVariance())), MIN_SIGMA_A), MAX_SIGMA_A);
        RS_target_    = std::min(std::max(
            R_S_coeff * sigma_target_ * tau_target_ * tau_target_ * tau_target_, MIN_R_S), MAX_R_S);

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

    static constexpr float R_S_xy_factor = 0.05f;
    static constexpr float S_factor = 1.2f;

    TrackingPolicy tracker_policy_{};  // one instance of frequency tracker per filter
    FrequencySmoother<float> freqSmoother;
    SeaStateAutoTuner tuner_;
    TuneState tune_;
    TiltMahony tilt_;                   // <<< sensor-only tilt proxy (no MEKF feedback)

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    // correlation estimation state
    CorrXZEstimator corr_;
    float rho_target_  = 0.0f;
    float rho_applied_ = 0.0f;   // smoothed value we actually apply

    std::unique_ptr<Kalman3D_Wave<float,true,true>> mekf_;
};

