#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "VerticalPIIObserver.h"
#include "WaveFrequencyTracker.h"

namespace marine_obs {

template<typename T = float, bool WithBias = true, typename AccelFreqTrackerT = WaveFrequencyTracker<T>>
class AdaptiveVerticalPII {
    static_assert(std::is_floating_point<T>::value,
                  "AdaptiveVerticalPII<T>: T must be a floating-point type.");

public:
    using Observer = VerticalPIIObserver<T, WithBias>;
    using ObserverConfig = typename Observer::Config;
    using ObserverAdaptConfig = typename Observer::AdaptConfig;
    using ObserverSnapshot = typename Observer::Snapshot;
    using AccelFreqTracker = AccelFreqTrackerT;
    using AccelFreqTrackerConfig = typename AccelFreqTracker::Config;

    struct Config {
        ObserverConfig observer{};
        ObserverAdaptConfig adaptation{};
        AccelFreqTrackerConfig accel_freq_tracker{};

        T sigma_mean_tau_s = T(20.0);
        T sigma_var_tau_s  = T(6.0);
        T sigma_floor      = T(1e-4);

        bool auto_schedule_from_accel_freq = true;
        T auto_schedule_period_s = T(0.25);

        // When auto scheduling is enabled, also force observer adaptation on.
        bool force_enable_adaptation_when_auto_schedule = true;

        // External default confidence for explicit user-provided scheduling calls.
        T default_external_confidence = T(1.0);

        // Fallback confidence shaping for the INTERNAL accel-frequency path.
        //
        // Raw tracker confidence can stay too low for too long, which makes the
        // auto scheduler look "dead" because VerticalPIIObserver holds params
        // below AdaptConfig::min_confidence.
        //
        // So:
        //   - if coarse estimate exists, raise confidence at least to this floor
        //   - if the tracker says it is locked, raise confidence at least to this
        //
        // These do NOT force bad frequencies into the observer from t=0; they only
        // help the auto path actually schedule once the tracker has some structure.
        T fallback_confidence_floor = T(0.35);
        T fallback_confidence_when_locked = T(0.70);
    };

    struct Snapshot {
        ObserverSnapshot observer{};

        T accel_mean = T(0);
        T accel_var  = T(0);
        T accel_sigma = T(0);

        T last_sched_dt = T(0);
        T sched_accum_s = T(0);

        bool auto_schedule_from_accel_freq = true;
        bool observer_adaptation_enabled = false;

        T accel_freq_hz = T(0);
        T accel_freq_confidence_raw = T(0);
        T accel_freq_confidence_used = T(0);

        bool accel_freq_locked = false;
        bool accel_freq_has_coarse = false;
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

        if (cfg_.auto_schedule_from_accel_freq &&
            cfg_.force_enable_adaptation_when_auto_schedule) {
            observer_.set_adaptation_enabled(true);
        }

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

    T update(T a_meas, T dt) {
        if (!(std::isfinite(dt) && dt > T(0))) {
            return observer_.displacement();
        }

        updateAccelSigma_(a_meas, dt);
        accel_freq_tracker_.update(a_meas, dt);

        if (cfg_.auto_schedule_from_accel_freq) {
            sched_accum_s_ += dt;

            if (sched_accum_s_ >= cfg_.auto_schedule_period_s) {
                const T dt_sched = sched_accum_s_;
                sched_accum_s_ = T(0);
                last_sched_dt_ = dt_sched;

                if (observer_.adaptation_enabled()) {
                    updateAdaptationFromAccelFrequencyProxy(dt_sched);
                }
            }
        }

        return observer_.update(a_meas, dt);
    }

    void updateAdaptationFromDisplacementFrequency(T f_disp_hz,
                                                   T dt_est,
                                                   T confidence = std::numeric_limits<T>::quiet_NaN())
    {
        if (!std::isfinite(confidence)) {
            confidence = cfg_.default_external_confidence;
        }
        confidence = clamp01_(confidence);

        last_sched_dt_ = dt_est;
        last_auto_confidence_used_ = confidence;

        observer_.update_adaptation(
            f_disp_hz,
            accel_sigma_,
            confidence,
            dt_est
        );
    }

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
        last_auto_confidence_used_ = confidence;

        observer_.update_adaptation(
            f_disp_hz,
            sigma_a,
            confidence,
            dt_est
        );
    }

    void updateAdaptationFromAccelFrequencyProxy(T dt_est) {
        const T f_hz = accel_freq_tracker_.getFrequencyHz();
        if (!(std::isfinite(f_hz) && f_hz > T(0))) {
            return;
        }

        const T conf_used = computeFallbackConfidence_();
        last_auto_confidence_used_ = conf_used;
        last_auto_tracker_locked_ = accel_freq_tracker_.isLocked();
        last_auto_tracker_has_coarse_ = accel_freq_tracker_.hasCoarseEstimate();

        observer_.update_adaptation(
            f_hz,
            accel_sigma_,
            conf_used,
            dt_est
        );
    }

    void setAutoScheduleFromAccelFreq(bool on) {
        cfg_.auto_schedule_from_accel_freq = on;

        if (on && cfg_.force_enable_adaptation_when_auto_schedule) {
            observer_.set_adaptation_enabled(true);
        }
    }

    void setAutoSchedulePeriod(T period_s) {
        if (std::isfinite(period_s) && period_s > T(0)) {
            cfg_.auto_schedule_period_s = period_s;
        }
    }

    void setFallbackConfidenceFloor(T c) {
        if (std::isfinite(c)) {
            cfg_.fallback_confidence_floor = clamp01_(c);
        }
    }

    void setFallbackConfidenceWhenLocked(T c) {
        if (std::isfinite(c)) {
            cfg_.fallback_confidence_when_locked = clamp01_(c);
        }
    }

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
    T autoScheduledConfidenceUsed() const { return last_auto_confidence_used_; }

    Snapshot snapshot() const {
        Snapshot s;
        s.observer = observer_.snapshot();

        s.accel_mean = accel_mean_;
        s.accel_var = accel_var_;
        s.accel_sigma = accel_sigma_;

        s.last_sched_dt = last_sched_dt_;
        s.sched_accum_s = sched_accum_s_;

        s.auto_schedule_from_accel_freq = cfg_.auto_schedule_from_accel_freq;
        s.observer_adaptation_enabled = observer_.adaptation_enabled();

        s.accel_freq_hz = accel_freq_tracker_.getFrequencyHz();
        s.accel_freq_confidence_raw = accel_freq_tracker_.getConfidence();
        s.accel_freq_confidence_used = last_auto_confidence_used_;

        s.accel_freq_locked = accel_freq_tracker_.isLocked();
        s.accel_freq_has_coarse = accel_freq_tracker_.hasCoarseEstimate();
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

        cfg.fallback_confidence_floor =
            clamp01_(finite_or_default_(cfg.fallback_confidence_floor, T(0.35)));

        cfg.fallback_confidence_when_locked =
            clamp01_(finite_or_default_(cfg.fallback_confidence_when_locked, T(0.70)));

        if (cfg.auto_schedule_from_accel_freq &&
            cfg.force_enable_adaptation_when_auto_schedule) {
            cfg.adaptation.enabled = true;
        }

        return cfg;
    }

    void resetSchedulerState_() {
        sched_accum_s_ = T(0);
        last_sched_dt_ = T(0);
        last_auto_confidence_used_ = T(0);
        last_auto_tracker_locked_ = false;
        last_auto_tracker_has_coarse_ = false;
    }

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

    T computeFallbackConfidence_() const {
        T conf = accel_freq_tracker_.getConfidence();
        if (!std::isfinite(conf)) {
            conf = T(0);
        }

        // If the tracker has at least coarse structure, do not leave the fallback
        // path stuck forever below the observer's min_confidence gate.
        if (accel_freq_tracker_.hasCoarseEstimate()) {
            conf = std::max(conf, cfg_.fallback_confidence_floor);
        }

        if (accel_freq_tracker_.isLocked()) {
            conf = std::max(conf, cfg_.fallback_confidence_when_locked);
        }

        return clamp01_(conf);
    }

private:
    Config cfg_{};

    Observer observer_;
    AccelFreqTracker accel_freq_tracker_;

    T accel_mean_  = T(0);
    T accel_var_   = T(0);
    T accel_sigma_ = T(1e-4);

    T sched_accum_s_ = T(0);
    T last_sched_dt_ = T(0);

    T last_auto_confidence_used_ = T(0);
    bool last_auto_tracker_locked_ = false;
    bool last_auto_tracker_has_coarse_ = false;
};

} // namespace marine_obs
