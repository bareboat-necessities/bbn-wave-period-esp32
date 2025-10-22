#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "FrequencySmoother.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "SeaStateAutoTuner.h"
#include "Kalman3D_Wave.h"
#include "FrameConversions.h"   // for quat_to_euler_aero() and aero_to_nautical()

/*
  ============================================================================
   SeaStateFusionFilter  —  Unified adaptive 3-D wave estimator
  ============================================================================
   Combines:
     • Frequency tracker (compile-time selectable: Aranovskiy, KalmANF, or ZeroCross)
     • Online smoother for stable frequency output
     • 3-D quaternion-based Kalman filter (Kalman3D_Wave)
     • SeaStateAutoTuner for adaptive σₐ, τ, and Rₛ tuning

   Provides:
     • updateTime(dt, gyro, accel)  →  propagation + accel correction + tuner update
     • updateMag(mag)               →  optional yaw correction
     • Getters for frequency, τ, σₐ, Rₛ, Tₚ, and Euler angles (nautical, °)

   Intended as the single front-end interface for real-time IMU-based sea-state
   estimation and simulation analysis.
  ============================================================================
*/

enum class TrackerType { ARANOVSKIY, KALMANF, ZEROCROSS };

template<TrackerType trackerT>
class SeaStateFusionFilter {
public:
    struct TuneState {
        float tau_applied   = 1.15f;   // s
        float sigma_applied = 1.22f;   // m/s²
        float RS_applied    = 8.17f;   // m·s
    };

    explicit SeaStateFusionFilter(bool with_mag)
        : with_mag_(with_mag)
    {
        freq_detector_.reset();
    }

    // --- Initialization ---
    void initialize(const Eigen::Vector3f& sigma_a,
                    const Eigen::Vector3f& sigma_g,
                    const Eigen::Vector3f& sigma_m)
    {
        mekf_ = std::make_unique<Kalman3D_Wave<float, true, true>>(sigma_a, sigma_g, sigma_m);
        mekf_->set_aw_time_constant(tune_.tau_applied);
        mekf_->set_aw_stationary_corr_std(Eigen::Vector3f::Constant(tune_.sigma_applied));
        mekf_->set_RS_noise(Eigen::Vector3f::Constant(tune_.RS_applied));
    }

    void initialize_from_acc(const Eigen::Vector3f& acc_world) {
        if (mekf_) mekf_->initialize_from_acc(acc_world);
    }

    // --- TIME UPDATE (gyro + accel, runs every IMU step) ---
    void updateTime(float dt,
                    const Eigen::Vector3f& gyro_world,
                    const Eigen::Vector3f& acc_world)
    {
        if (!mekf_) return;
        time_ += dt;

        // 1. Propagate + accel correction
        mekf_->time_update(gyro_world, dt);
        mekf_->measurement_update_acc_only(acc_world);

        // 2. Frequency tracking
        float a_z = acc_world.z() - g_std;
        float a_norm = a_z / g_std;
        auto [est_freq, ok] = run_tracker(a_norm, a_z, dt);
        if (ok) {
            if (first_freq_) {
                freq_smoother_.setInitial(est_freq);
                first_freq_ = false;
            } else {
                est_freq = freq_smoother_.update(est_freq);
            }
            freq_hz_ = std::clamp(est_freq, MIN_FREQ_HZ, MAX_FREQ_HZ);
        }

        // 3. SeaStateAutoTuner adaptation
        if (std::isfinite(freq_hz_) && time_ >= ONLINE_TUNE_WARMUP_SEC) {
            tuner_.update(dt, a_z, static_cast<float>(freq_hz_));
            Tp_tuner_  = tuner_.getPeriodSec();
            accel_var_ = tuner_.getAccelVariance();

            // Targets
            float tau_target   = std::clamp(0.5f / freq_hz_, MIN_TAU_S, MAX_TAU_S);
            float sigma_target = std::clamp(std::sqrt(std::max(0.0f, accel_var_)),
                                            MIN_SIGMA_A, MAX_SIGMA_A);
            float RS_target    = std::clamp(R_S_coeff *
                                            sigma_target * std::pow(tau_target, 3),
                                            MIN_R_S, MAX_R_S);

            adapt_mekf(dt, tau_target, sigma_target, RS_target);
        }
    }

    // --- MAGNETOMETER UPDATE (yaw correction) ---
    void updateMag(const Eigen::Vector3f& mag_world)
    {
        if (!with_mag_ || !mekf_) return;
        if (time_ < MAG_DELAY_SEC) return;
        mekf_->measurement_update_mag_only(mag_world);
    }

    // === GETTERS ===
    inline float getFreqHz()        const noexcept { return freq_hz_; }
    inline float getTauApplied()    const noexcept { return tune_.tau_applied; }
    inline float getSigmaApplied()  const noexcept { return tune_.sigma_applied; }
    inline float getRSApplied()     const noexcept { return tune_.RS_applied; }
    inline float getPeriodSec()     const noexcept { return Tp_tuner_; }
    inline float getAccelVariance() const noexcept { return accel_var_; }
    inline bool  tunerReady()       const noexcept { return tuner_.isReady(); }
    inline const Kalman3D_Wave<float, true, true>& mekf() const { return *mekf_; }

    // --- Euler angles (nautical, degrees) ---
    inline Eigen::Vector3f getEulerNautical() const {
        Eigen::Vector3f eul_deg = Eigen::Vector3f::Zero();
        if (!mekf_) return eul_deg;

        auto coeffs = mekf_->quaternion().coeffs();  // (x, y, z, w)
        Eigen::Quaternionf q(coeffs(3), coeffs(0), coeffs(1), coeffs(2)); // (w, x, y, z)

        float roll_aero, pitch_aero, yaw_aero;
        quat_to_euler_aero(q, roll_aero, pitch_aero, yaw_aero);
        aero_to_nautical(roll_aero, pitch_aero, yaw_aero);

        eul_deg << roll_aero, pitch_aero, yaw_aero;
        eul_deg *= 180.0f / static_cast<float>(M_PI);
        return eul_deg;
    }

private:
    bool with_mag_;
    bool first_freq_ = true;
    double time_ = 0.0;

    float freq_hz_   = NAN;
    float Tp_tuner_  = NAN;
    float accel_var_ = NAN;

    AranovskiyFilter<double> aran_;
    KalmANF<double> kalmANF_;
    FrequencySmoother<float> freq_smoother_;
    SchmittTriggerFrequencyDetector freq_detector_{ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS};

    SeaStateAutoTuner tuner_;
    TuneState tune_;
    std::unique_ptr<Kalman3D_Wave<float, true, true>> mekf_;

    // === Compile-time tracker dispatch ===
    std::pair<double,bool> run_tracker(float a_norm, float a_raw, float dt) {
        double freq = FREQ_GUESS;
        bool valid = true;

        if constexpr (trackerT == TrackerType::ARANOVSKIY) {
            freq = estimate_freq(Aranovskiy, &aran_, &kalmANF_, &freq_detector_,
                                 a_norm, a_raw, dt, now_us());
        } else if constexpr (trackerT == TrackerType::KALMANF) {
            freq = estimate_freq(Kalm_ANF, &aran_, &kalmANF_, &freq_detector_,
                                 a_norm, a_raw, dt, now_us());
        } else if constexpr (trackerT == TrackerType::ZEROCROSS) {
            freq = estimate_freq(ZeroCrossing, &aran_, &kalmANF_, &freq_detector_,
                                 a_norm, a_raw, dt, now_us());
        }

        valid = !std::isnan(freq);
        return {freq, valid};
    }

    void adapt_mekf(float dt, float tau_target, float sigma_target, float RS_target) {
        const float alpha = 1.0f - std::exp(-dt / ADAPT_TAU_SEC);
        tune_.tau_applied   += alpha * (tau_target   - tune_.tau_applied);
        tune_.sigma_applied += alpha * (sigma_target - tune_.sigma_applied);
        tune_.RS_applied    += alpha * (RS_target    - tune_.RS_applied);

        mekf_->set_aw_time_constant(tune_.tau_applied);
        mekf_->set_aw_stationary_corr_std(Eigen::Vector3f::Constant(tune_.sigma_applied));
        mekf_->set_RS_noise(Eigen::Vector3f::Constant(tune_.RS_applied));
    }

    static uint32_t now_us() {
        return static_cast<uint32_t>(sim_time_us_++);
    }

    inline static uint64_t sim_time_us_ = 0;
};
