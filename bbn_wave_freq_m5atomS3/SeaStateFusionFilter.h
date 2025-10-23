#pragma once

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
#include "SeaStateAutoTuner.h"
#include "Kalman3D_Wave.h"
#include "FrameConversions.h"

// Shared constants (synchronized with main)
constexpr float MIN_FREQ_HZ = 0.1f;
constexpr float MAX_FREQ_HZ = 6.0f;

constexpr float MIN_TAU_S   = 0.1f;
constexpr float MAX_TAU_S   = 11.5f;
constexpr float MIN_SIGMA_A = 0.1f;
constexpr float MAX_SIGMA_A = 20.0f;
constexpr float MIN_R_S     = 0.01f;
constexpr float MAX_R_S     = 20.0f;

constexpr float R_S_coeff   = 2.1f;

constexpr float ADAPT_TAU_SEC = 10.0f;
constexpr float ONLINE_TUNE_WARMUP_SEC = 20.0f;
constexpr float MAG_DELAY_SEC = 5.0f;

struct TuneState {
    float tau_applied   = 1.15f;
    float sigma_applied = 1.22f;
    float RS_applied    = 8.17f;
};

//  Tracker policy traits

template<TrackerType>
struct TrackerPolicy; // primary template (undefined)

// Aranovskiy
template<>
struct TrackerPolicy<TrackerType::ARANOVSKIY> {
    using Tracker = AranovskiyFilter<double>;
    static double run(Tracker& t, float a, float dt) {
        t.update((double)a, (double)dt);
        double freq = t.getFrequencyHz();
        return freq;
    }
};

// KalmANF
template<>
struct TrackerPolicy<TrackerType::KALMANF> {
    using Tracker = KalmANF<double>;
    static double run(Tracker& t, float a, float dt) {
        double e;
        double freq = t.process((double)a, (double)dt, &e);
        return freq;
    }
};

// ZeroCross
template<>
struct TrackerPolicy<TrackerType::ZEROCROSS> {
    using Tracker = SchmittTriggerFrequencyDetector;
    static double run(Tracker& t, float a, float dt) {
        float f_byZeroCross = t.update(a, ZERO_CROSSINGS_SCALE /* max fractions of g */,
                              ZERO_CROSSINGS_DEBOUNCE_TIME, ZERO_CROSSINGS_STEEPNESS_TIME, dt);
        double freq;
        if (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT || f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ) {
           freq = FREQ_GUESS;
        } else {
           freq = f_byZeroCross;
        }
        return freq;
    }
};

//  Unified SeaState fusion filter
template<TrackerType trackerT>
class SeaStateFusionFilter {
public:
    using Policy  = TrackerPolicy<trackerT>;
    using Tracker = typename Policy::Tracker;

    explicit SeaStateFusionFilter(bool with_mag)
        : with_mag_(with_mag),
          tuner_(),
          time_(0.0),
          freq_hz_(NAN)
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
    void updateTime(float dt, const Eigen::Vector3f& gyro, const Eigen::Vector3f& acc)
    {
        if (!mekf_) return;
        time_ += dt;

        mekf_->time_update(gyro, dt);
        mekf_->measurement_update_acc_only(acc);

        const float a_z = acc.z() - g_std;
        const float a_norm = a_z / g_std;

        const double f = Policy::run(tracker_, a_z, dt);
        if (!std::isnan(f)) {
            freq_hz_ = std::min(std::max(static_cast<float>(f), MIN_FREQ_HZ), MAX_FREQ_HZ);
            update_tuner(dt, a_z);
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
    inline float getPeriodSec()    const noexcept { return (freq_hz_ > 1e-6f) ? 1.0f / freq_hz_ : NAN; }
    inline float getAccelVariance()const noexcept { return tuner_.getAccelVariance(); }

Eigen::Vector3f getEulerNautical() const {
    // 1. Fetch quaternion in Eigen coeff order (x, y, z, w)
    const auto coeffs = mekf_.quaternion().coeffs();
    Eigen::Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2)); // w,x,y,z

    // 2. Convert from aerospace (body-to-world, NED) to nautical (Z-up ENU)
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
        mekf_->set_aw_stationary_corr_std(Eigen::Vector3f::Constant(tune_.sigma_applied));
        mekf_->set_RS_noise(Eigen::Vector3f::Constant(tune_.RS_applied));
    }

    void update_tuner(float dt, float a_z) {
        if (!std::isfinite(freq_hz_) || time_ < ONLINE_TUNE_WARMUP_SEC)
            return;

        tuner_.update(dt, a_z, freq_hz_);
        const float tau_target   = std::min(std::max(0.5f / freq_hz_, MIN_TAU_S), MAX_TAU_S);
        const float sigma_target = std::min(std::max(
            std::sqrt(std::max(0.0f, tuner_.getAccelVariance())),
            MIN_SIGMA_A), MAX_SIGMA_A);
        const float RS_target    = std::min(std::max(
            R_S_coeff * sigma_target * tau_target * tau_target * tau_target,
            MIN_R_S), MAX_R_S);

        adapt_mekf(dt, tau_target, sigma_target, RS_target);
    }

    void adapt_mekf(float dt, float tau_t, float sigma_t, float RS_t) {
        const float alpha = 1.0f - std::exp(-dt / ADAPT_TAU_SEC);
        tune_.tau_applied   += alpha * (tau_t - tune_.tau_applied);
        tune_.sigma_applied += alpha * (sigma_t - tune_.sigma_applied);
        tune_.RS_applied    += alpha * (RS_t - tune_.RS_applied);
        apply_tune();
    }

    //  Members
    bool with_mag_;
    double time_;
    float freq_hz_;

    Tracker tracker_{};  // one instance per filter
    SeaStateAutoTuner tuner_;
    TuneState tune_;
    std::unique_ptr<Kalman3D_Wave<float,true,true>> mekf_;
};
