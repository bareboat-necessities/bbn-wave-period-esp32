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

struct CorrXZEstimator {
    float tau = 1.5f;   // EWMA horizon [s]
    float last_dt = -1.0f;
    float alpha = 0.0f;
    float w = 0.0f;

    Eigen::Vector3f m = Eigen::Vector3f::Zero();   // E[x]
    Eigen::Matrix3f Exx = Eigen::Matrix3f::Zero(); // E[xxᵀ]

    inline void setTau(float t){ tau = std::max(1e-3f, t); last_dt = -1.0f; }
    inline void reset(){ last_dt=-1.0f; alpha=0.0f; w=0.0f; m.setZero(); Exx.setZero(); }

    inline void update(float dt, const Eigen::Vector3f& a_) {
        if (!(dt > 0)) return;
        if (dt != last_dt) { alpha = 1.0f - std::exp(-dt / tau); last_dt = dt; }
        m   = (1.0f - alpha) * m   + alpha * a_;
        Exx = (1.0f - alpha) * Exx + alpha * (a_ * a_.transpose());
        w   = (1.0f - alpha) * w   + alpha;
    }

    inline bool ready() const { return w > 1e-3f; }

    inline Eigen::Vector3f var() const {
        Eigen::Vector3f v = Exx.diagonal() - m.array().square().matrix();
        return v.cwiseMax(0.0f);
    }

    inline float cov_xy() const { return Exx(0,1) - m.x()*m.y(); }
    inline float cov_xz() const { return Exx(0,2) - m.x()*m.z(); }
    inline float cov_yz() const { return Exx(1,2) - m.y()*m.z(); }

    // 2×2 XY covariance
    inline Eigen::Matrix2f covXY() const {
        const auto v = var();
        Eigen::Matrix2f C;
        C << v.x(), cov_xy(),
             cov_xy(), v.y();
        // avoid negative tiny numerics
        C(0,0) = std::max(C(0,0), 0.0f);
        C(1,1) = std::max(C(1,1), 0.0f);
        return C;
    }

    // PCA of XY: returns theta (rad) of dominant eigenvector (in body XY),
    // lambda_max, lambda_min. Returns false if not ready or degenerate.
    inline bool pcaXY(float& theta, float& lambda_max, float& lambda_min) const {
        if (!ready()) return false;
        Eigen::Matrix2f C = covXY();
        // closed-form eig for 2x2 SPD
        const float tr = C(0,0) + C(1,1);
        const float det = C(0,0)*C(1,1) - C(0,1)*C(1,0);
        const float disc = std::sqrt(std::max(0.0f, tr*tr*0.25f - det));
        lambda_max = tr*0.5f + disc;
        lambda_min = tr*0.5f - disc;
        if (!(lambda_max > 0.0f)) return false;

        // dominant eigenvector direction
        Eigen::Vector2f v;
        if (std::abs(C(0,1)) > 1e-12f) {
            v << lambda_max - C(1,1), C(0,1);
        } else {
            v << 1.0f, 0.0f;
        }
        if (v.squaredNorm() < 1e-20f) return false;
        v.normalize();
        theta = std::atan2(v.y(), v.x());
        return true;
    }

    inline bool rhos(float& rho_xz, float& rho_yz) const {
        if (!ready()) return false;
        const auto v = var();
        const float sx = std::sqrt(std::max(0.0f, v.x()));
        const float sy = std::sqrt(std::max(0.0f, v.y()));
        const float sz = std::sqrt(std::max(0.0f, v.z()));
        const float eps = 1e-9f;
        rho_xz = (sx > eps && sz > eps) ? cov_xz()/(sx*sz) : 0.0f;
        rho_yz = (sy > eps && sz > eps) ? cov_yz()/(sy*sz) : 0.0f;
        rho_xz = std::clamp(rho_xz, -0.99f, 0.99f);
        rho_yz = std::clamp(rho_yz, -0.99f, 0.99f);
        return true;
    }
};

// Shared constants
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 5.0f;

constexpr float MIN_TAU_S   = 0.5f;
constexpr float MAX_TAU_S   = 8.5f;
constexpr float MIN_SIGMA_A = 0.3f;
constexpr float MAX_SIGMA_A = 8.0f;
constexpr float MIN_R_S     = 0.1f;
constexpr float MAX_R_S     = 35.0f;

constexpr float R_S_coeff   = 2.4f;
constexpr float tau_coeff   = 1.5f;

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
        mekf_->set_exact_att_bias_Qd(true);
        apply_tune();
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) mekf_->initialize_from_acc(acc_world);
    }

    // Time update (IMU integration + frequency tracking)
    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc) {
        if (!mekf_) return;
        time_ += dt;

        // MEKF updates (independent)
        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc);

        const float a_z = acc.z() + g_std;
        const float a_norm = a_z / g_std;

        // accumulate covariance for ρ 
        corr_.update(dt, acc);

        // Feed tracker 
        const double f = tracker_policy_.run(a_z, dt);
        if (!std::isnan(f)) {
            freq_hz_ = std::min(std::max(static_cast<float>(f), MIN_FREQ_HZ), MAX_FREQ_HZ);
            update_tuner(dt, a_z, freq_hz_);
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

    inline auto& mekf() noexcept { return *mekf_; }

private:
    //  Internal tuning and adaptation
    void apply_tune() {
        if (!mekf_) return;
    
        // OU time constant
        mekf_->set_aw_time_constant(tune_.tau_applied);
    
        // WORLD-frame stationary covariance for a_w (XY equal, Z separate).
        const float sZ = std::max(1e-6f, tune_.sigma_applied);
        const float sH = sZ * S_factor; 
        Eigen::Matrix3f Sigma_world = Eigen::Matrix3f::Zero();
        Sigma_world(0,0) = sH * sH;   // world X
        Sigma_world(1,1) = sH * sH;   // world Y
        Sigma_world(2,2) = sZ * sZ;   // world Z
        mekf_->set_aw_stationary_cov_full(Sigma_world);  // internally SPD-projects & merges
    
        // WORLD-frame pseudo-measurement noise for S (anisotropic diagonal).
        // Clamp to configured bounds for robustness.
        const float RSb = std::min(std::max(tune_.RS_applied, MIN_R_S), MAX_R_S);
        mekf_->set_RS_noise(Eigen::Vector3f(
            RSb * R_S_xy_factor,   // world X
            RSb * R_S_xy_factor,   // world Y
            RSb                    // world Z
        ));
    }

    void update_tuner(float dt, float a_vert, float freq_hz) {
        if (!std::isfinite(freq_hz)) {
            freqSmoother.setInitial(FREQ_GUESS);
            return;
        }
        if (!freq_init_) {
            freqSmoother.setInitial(freq_hz);
            freq_init_ = true;
        }

        float smoothFreq = freqSmoother.update(freq_hz);

        // Warm-up: we still collect covariance/ρ, but skip parameter application
        if (time_ < ONLINE_TUNE_WARMUP_SEC)  {
            return;
        }

        tuner_.update(dt, a_vert, smoothFreq);

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

    static constexpr float R_S_xy_factor = 0.07f;
    static constexpr float S_factor = 1.3f;

    TrackingPolicy tracker_policy_{};  // one instance of frequency tracker per filter
    FrequencySmoother<float> freqSmoother;
    SeaStateAutoTuner tuner_;
    TuneState tune_;

    float tau_target_   = NAN;
    float sigma_target_ = NAN;
    float RS_target_    = NAN;

    // correlation estimation state
    CorrXZEstimator corr_;

    std::unique_ptr<Kalman3D_Wave<float,true,true>> mekf_;
};
