#pragma once

/*

  Wrapper / scheduler around VerticalPIIObserver

  What this class does
  
  1) Owns a VerticalPIIObserver<T, WithBias>
  2) Tracks acceleration sigma (std dev / RMS-like scale) online
  3) Optionally owns a WaveFrequencyTracker<T> for acceleration-frequency fallback
  4) Calls VerticalPIIObserver::update_adaptation(...) using:
       - displacement frequency estimate (preferred), OR
       - acceleration frequency estimate (fallback proxy)

  This wrapper adds the outer adaptation logic:
      - signal statistics
      - optional frequency tracker integration
      - scheduling cadence
      - preferred source selection for frequency updates

  Preferred adaptation source

  BEST:
      displacement-frequency estimate from your external wave/displacement tracker

  FALLBACK:
      acceleration-frequency estimate from WaveFrequencyTracker

  Typical usage

      marine_obs::AdaptiveVerticalPII<float, true> heave;

      // Every IMU sample:
      float z = heave.update(vertical_world_accel, dt_imu);

      // Whenever displacement-frequency estimate updates (preferred):
      heave.updateAdaptationFromDisplacementFrequency(f_disp_hz, dt_track, confidence);

  If no displacement-frequency estimate is available:
      cfg.auto_schedule_from_accel_freq = true;
      heave.update(vertical_world_accel, dt_imu); // wrapper will periodically schedule from internal accel tracker

  Notes

  - This wrapper is header-only and embedded-friendly.
  - No dynamic allocation.
  - No exceptions.
  - No std::vector / std::string / RTTI required.
  - Requires VerticalPIIObserver.h and WaveFrequencyTracker.h.
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "VerticalPIIObserver.h"
#include "WaveFrequencyTracker.h"

namespace marine_obs {

template<typename T = float, bool WithBias = true, typename AccelFreqTrackerT = WaveFrequencyTracker<T>>
class AdaptiveVerticalPII {
    static_assert(std::is_floating_point<T>::value, "AdaptiveVerticalPII<T>: T must be a floating-point type.");

public:
    using Observer = VerticalPIIObserver<T, WithBias>;
    using ObserverConfig = typename Observer::Config;
    using ObserverAdaptConfig = typename Observer::AdaptConfig;
    using ObserverSnapshot = typename Observer::Snapshot;
    using AccelFreqTracker = AccelFreqTrackerT;
    using AccelFreqTrackerConfig = typename AccelFreqTracker::Config;

    struct Config {
        // Core observer config
        ObserverConfig observer{};
        ObserverAdaptConfig adaptation{};

        // Internal acceleration-frequency tracker config
        AccelFreqTrackerConfig accel_freq_tracker{};

        // Online acceleration sigma estimation:
        //   mean_lp tracks slow mean / drift
        //   var_lp tracks variance of (a - mean_lp)
        T sigma_mean_tau_s = T(20.0);
        T sigma_var_tau_s  = T(6.0);
        T sigma_floor      = T(1e-4); // lower bound returned by accelSigma()

        // If true, update() will periodically call observer.update_adaptation()
        // using the internal acceleration-frequency tracker as a fallback proxy.
        //
        // Recommended:
        //   false when you have displacement-frequency estimates
        //   true only as fallback
        bool auto_schedule_from_accel_freq = true;
        T auto_schedule_period_s = T(0.25); // cadence of fallback adaptation calls

        // If the caller omits confidence or passes NaN, this is used.
        T default_external_confidence = T(1.0);
    };

    struct Snapshot {
        ObserverSnapshot observer{};
        T accel_mean = T(0);
        T accel_var  = T(0);
        T accel_sigma = T(0);

        T last_sched_dt = T(0);
        T sched_accum_s = T(0);

        bool auto_schedule_from_accel_freq = true;
        T accel_freq_hz = T(0);
        T accel_freq_confidence = T(0);
    };

public:
    explicit AdaptiveVerticalPII(const Config& cfg = Config())
        : observer_(cfg.observer, cfg.adaptation),
          accel_freq_tracker_(cfg.accel_freq_tracker) {
        configure(cfg);
        reset();
    }

    void configure(const Config& cfg) {
        cfg_ = sanitizeConfig_(cfg);

        observer_.configure(cfg_.observer);
        observer_.configure_adaptation(cfg_.adaptation);
        accel_freq_tracker_.configure(cfg_.accel_freq_tracker);

        // keep current state, but reset outer adaptation helpers
        resetSchedulerState_();
    }

    void reset(T p0 = T(0),
               T v0 = T(0),
               T a_f0 = T(0),
               T S0 = T(0),
               T d0 = T(0),
               T b0 = T(0))
    {
        observer_.reset(p0, v0, a_f0, S0, d0, b0);

        accel_mean_  = T(0);
        accel_var_   = T(0);
        accel_sigma_ = cfg_.sigma_floor;

        accel_freq_tracker_.reset(cfg_.accel_freq_tracker.f_init_hz);

        resetSchedulerState_();
    }

    // Per-sample update
    //
    // Call this on every IMU sample with already gravity-compensated vertical
    // world-frame inertial acceleration.
    //
    // If cfg.auto_schedule_from_accel_freq is enabled, this also periodically
    // calls observer.update_adaptation() using:
    //   f_disp_hz <- internal acceleration-frequency tracker output (proxy)
    //   sigma_a   <- internal sigma estimate
    //   confidence<- internal tracker confidence
    //
    // Preferred real use:
    //   keep auto_schedule_from_accel_freq = false
    //   call updateAdaptationFromDisplacementFrequency(...) from your external
    //   displacement-frequency tracker instead.
    T update(T a_meas, T dt) {
        if (!(std::isfinite(dt) && dt > T(0))) {
            return observer_.displacement();
        }

        // 1) Update outer statistics / trackers first
        updateAccelSigma_(a_meas, dt);
        accel_freq_tracker_.update(a_meas, dt);

        // 2) Optional fallback scheduling from acceleration frequency proxy
        if (cfg_.auto_schedule_from_accel_freq && observer_.adaptation_enabled()) {
            sched_accum_s_ += dt;
            if (sched_accum_s_ >= cfg_.auto_schedule_period_s) {
                const T dt_sched = sched_accum_s_;
                sched_accum_s_ = T(0);
                last_sched_dt_ = dt_sched;
                updateAdaptationFromAccelFrequencyProxy(dt_sched);
            }
        }

        // 3) Propagate the core observer
        return observer_.update(a_meas, dt);
    }

    // Preferred adaptation hook:
    // use externally estimated DISPLACEMENT frequency + internally estimated sigma
    //
    // Call this whenever your displacement-frequency tracker updates.
    // dt_est is the elapsed time since the previous adaptation update from that tracker.
    void updateAdaptationFromDisplacementFrequency(T f_disp_hz,
                                                   T dt_est,
                                                   T confidence = std::numeric_limits<T>::quiet_NaN())
    {
        if (!std::isfinite(confidence)) {
            confidence = cfg_.default_external_confidence;
        }
        confidence = clamp01_(confidence);

        last_sched_dt_ = dt_est;
        observer_.update_adaptation(
            f_disp_hz,
            accel_sigma_,
            confidence,
            dt_est
        );
    }

    // Explicit external hook:
    // use externally provided displacement frequency AND externally provided sigma
    //
    // Useful if you already estimate sigma elsewhere and don't want the wrapper's
    // internal sigma estimator.
    void updateAdaptationExternal(T f_disp_hz,
                                  T sigma_a,
                                  T dt_est,
                                  T confidence = std::numeric_limits<T>::quiet_NaN())
    {
        if (!std::isfinite(confidence)) {
            confidence = cfg_.default_external_confidence;
        }
        confidence = clamp01_(confidence);

        last_sched_dt_ = dt_est;
        observer_.update_adaptation(
            f_disp_hz,
            sigma_a,
            confidence,
            dt_est
        );
    }

    // Fallback adaptation hook:
    // use internal ACCELERATION frequency tracker as a proxy for displacement frequency
    //
    // This is weaker / less preferred than displacement-frequency scheduling,
    // but useful when no displacement-frequency estimate is available yet.
    void updateAdaptationFromAccelFrequencyProxy(T dt_est) {
        observer_.update_adaptation(
            accel_freq_tracker_.getFrequencyHz(),
            accel_sigma_,
            accel_freq_tracker_.getConfidence(),
            dt_est
        );
    }

    // Convenience controls
    void setAutoScheduleFromAccelFreq(bool on) {
        cfg_.auto_schedule_from_accel_freq = on;
    }

    void setAutoSchedulePeriod(T period_s) {
        if (std::isfinite(period_s) && period_s > T(0)) {
            cfg_.auto_schedule_period_s = period_s;
        }
    }

    // Accessors
    Observer& observer() { return observer_; }
    const Observer& observer() const { return observer_; }

    AccelFreqTracker& accelFreqTracker() { return accel_freq_tracker_; }
    const AccelFreqTracker& accelFreqTracker() const { return accel_freq_tracker_; }

    T displacement() const { return observer_.displacement(); }
    T velocity() const     { return observer_.velocity(); }
    T accelFiltered() const { return observer_.accel_filtered(); }

    T accelMean() const  { return accel_mean_; }
    T accelVar() const   { return accel_var_; }
    T accelSigma() const { return accel_sigma_; }

    T accelFrequencyHz() const { return accel_freq_tracker_.getFrequencyHz(); }
    T accelFrequencyConfidence() const { return accel_freq_tracker_.getConfidence(); }

    T lastSchedulerDt() const { return last_sched_dt_; }

    Snapshot snapshot() const {
        Snapshot s;
        s.observer = observer_.snapshot();
        s.accel_mean = accel_mean_;
        s.accel_var = accel_var_;
        s.accel_sigma = accel_sigma_;
        s.last_sched_dt = last_sched_dt_;
        s.sched_accum_s = sched_accum_s_;
        s.auto_schedule_from_accel_freq = cfg_.auto_schedule_from_accel_freq;
        s.accel_freq_hz = accel_freq_tracker_.getFrequencyHz();
        s.accel_freq_confidence = accel_freq_tracker_.getConfidence();
        return s;
    }

private:
    static constexpr T eps_() {
        return T(1e-9);
    }

    static T clamp01_(T x) {
        return std::clamp(x, T(0), T(1));
    }

    static T finite_or_default_(T x, T def) {
        return std::isfinite(x) ? x : def;
    }

    static T one_pole_alpha_(T dt, T tau) {
        if (!(std::isfinite(dt) && dt > T(0))) return T(0);
        if (!(std::isfinite(tau) && tau > T(0))) return T(1);
        const T a = dt / (tau + dt);
        return std::clamp(a, T(0), T(1));
    }

    static Config sanitizeConfig_(Config cfg) {
        cfg.sigma_mean_tau_s = std::max(finite_or_default_(cfg.sigma_mean_tau_s, T(20)), eps_());
        cfg.sigma_var_tau_s  = std::max(finite_or_default_(cfg.sigma_var_tau_s,  T(6)),  eps_());
        cfg.sigma_floor      = std::max(finite_or_default_(cfg.sigma_floor, T(1e-4)), T(0));
        cfg.auto_schedule_period_s =
            std::max(finite_or_default_(cfg.auto_schedule_period_s, T(0.25)), eps_());
        cfg.default_external_confidence =
            clamp01_(finite_or_default_(cfg.default_external_confidence, T(1)));
        return cfg;
    }

    void resetSchedulerState_() {
        sched_accum_s_ = T(0);
        last_sched_dt_ = T(0);
    }

    // Online sigma estimator for vertical acceleration.
    //
    // We estimate sigma from variance of (a - mean_lp), where mean_lp is a very slow
    // one-pole mean estimate. This is intentionally simple and embedded-friendly.
    void updateAccelSigma_(T a_meas, T dt) {
        if (!std::isfinite(a_meas) || !(std::isfinite(dt) && dt > T(0))) {
            return;
        }

        const T alpha_mean = one_pole_alpha_(dt, cfg_.sigma_mean_tau_s);
        const T alpha_var  = one_pole_alpha_(dt, cfg_.sigma_var_tau_s);

        accel_mean_ += alpha_mean * (a_meas - accel_mean_);
        const T e = a_meas - accel_mean_;

        accel_var_ += alpha_var * (e * e - accel_var_);
        if (!(accel_var_ >= T(0)) || !std::isfinite(accel_var_)) {
            accel_var_ = T(0);
        }

        accel_sigma_ = std::sqrt(accel_var_);
        if (!(std::isfinite(accel_sigma_) && accel_sigma_ >= cfg_.sigma_floor)) {
            accel_sigma_ = cfg_.sigma_floor;
        }
    }

private:
    Config cfg_{};

    Observer observer_;
    AccelFreqTracker accel_freq_tracker_;

    // outer stats
    T accel_mean_  = T(0);
    T accel_var_   = T(0);
    T accel_sigma_ = T(1e-4);

    // fallback scheduler cadence
    T sched_accum_s_ = T(0);
    T last_sched_dt_ = T(0);
};

} // namespace marine_obs


/*
EXAMPLE USAGE

#include "AdaptiveVerticalPII.h"

// Bias-enabled
using HeaveAdaptive = marine_obs::AdaptiveVerticalPII<float, true>;

HeaveAdaptive::Config cfg;
cfg.observer.r = 0.16f;
cfg.observer.tau_a = 0.60f;
cfg.observer.tau_d = 40.0f;
cfg.observer.kb = 1e-4f;
cfg.observer.lambda_b = 1e-2f;

// Core adaptation behavior (same hooks already supported by VerticalPIIObserver)
cfg.adaptation.enabled = true;
cfg.adaptation.f_disp_ref_hz = 0.17f;
cfg.adaptation.sigma_a_ref = 0.30f;

// Optional acceleration-frequency fallback
cfg.auto_schedule_from_accel_freq = false; // preferred OFF if you have displacement frequency
cfg.auto_schedule_period_s = 0.25f;

HeaveAdaptive heave(cfg);

// Every IMU sample:
//   a_world_z is vertical world-frame inertial acceleration (gravity removed)
float z = heave.update(a_world_z, dt_imu);

// Whenever your DISPLACEMENT-frequency tracker updates (preferred path):
heave.updateAdaptationFromDisplacementFrequency(
    f_disp_hz,     // preferred scheduling frequency input
    dt_tracker,    // elapsed time since last tracker update
    confidence     // [0..1], or omit if unavailable
);

// If you do NOT yet have displacement frequency:
// enable fallback and let the wrapper schedule from internal accel-frequency
// tracker automatically.
//
// cfg.auto_schedule_from_accel_freq = true;
//
// Then heave.update(...) will periodically call:
//   observer.update_adaptation(accel_freq, accel_sigma, accel_freq_confidence, dt_sched)
//
// This is less preferred than displacement-frequency scheduling, but useful as
// a fallback.
//
// If you already estimate sigma elsewhere and want full external control:
heave.updateAdaptationExternal(
    f_disp_hz,
    sigma_a_external,
    dt_tracker,
    confidence
);

*/
