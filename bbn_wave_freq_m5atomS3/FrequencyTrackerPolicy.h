#pragma once

#include <cmath>

#include "WaveFilesSupport.h"
#include "AranovskiyFreqTracker.h"
#include "KalmANFFreqTracker.h"
#include "PLLFreqTracker.h"
#include "SchmittTriggerZCFreqTracker.h"

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

#ifndef ZERO_CROSSINGS_HYSTERESIS
#define ZERO_CROSSINGS_HYSTERESIS 0.04f
#endif

#ifndef ZERO_CROSSINGS_PERIODS
#define ZERO_CROSSINGS_PERIODS 1
#endif

// Tracker policy traits
template<TrackerType>
struct TrackerPolicy; // primary template (undefined)

// Aranovskiy
template<>
struct TrackerPolicy<TrackerType::ARANOVSKIY> {
    using Tracker = AranovskiyFreqTracker<double>;
    Tracker t;

    TrackerPolicy() : t() {
        const double omega_up   = (FREQ_GUESS * 2.0) * (2.0 * M_PI);
        const double k_gain     = 20.0;
        const double x1_0       = 0.0;
        const double omega_init = (FREQ_GUESS / 1.5) * 2.0 * M_PI;
        const double theta_0    = -(omega_init * omega_init);
        const double sigma_0    = theta_0;

        t.setParams(omega_up, k_gain);
        t.setState(x1_0, theta_0, sigma_0);
    }

    double run(float a, float dt) {
        t.update(static_cast<double>(a) / static_cast<double>(g_std), static_cast<double>(dt));
        return getFrequencyHz();
    }

    double getFrequencyHz() const { return t.getFrequencyHz(); }
    double getRawFrequencyHz() const { return t.getRawFrequencyHz(); }
    double getConfidence() const { return t.getConfidence(); }
    bool isLocked() const { return t.isLocked(); }
    bool hasCoarseEstimate() const { return t.hasCoarseEstimate(); }
    double getCoarseFrequencyHz() const { return t.getCoarseFrequencyHz(); }
};

// KalmANF
template<>
struct TrackerPolicy<TrackerType::KALMANF> {
    using Tracker = KalmANFFreqTracker<double>;
    Tracker t{};

    double run(float a, float dt) {
        double e = 0.0;
        t.process(static_cast<double>(a) / static_cast<double>(g_std),
                  static_cast<double>(dt), &e);
        return getFrequencyHz();
    }

    double getFrequencyHz() const { return t.getFrequencyHz(); }
    double getRawFrequencyHz() const { return t.getRawFrequencyHz(); }
    double getConfidence() const { return t.getConfidence(); }
    bool isLocked() const { return t.isLocked(); }
    bool hasCoarseEstimate() const { return t.hasCoarseEstimate(); }
    double getCoarseFrequencyHz() const { return t.getCoarseFrequencyHz(); }
};

// PLL
template<>
struct TrackerPolicy<TrackerType::PLLFREQTRACKER> {
    using Tracker = PLLFreqTracker<double>;
    using Config = typename Tracker::Config;

    Tracker t{};

    void configure(const Config& cfg) {
        t.configure(cfg);
    }

    void reset(double f_init_hz) {
        t.reset(f_init_hz);
    }

    void update(float a, float dt) {
        t.update(static_cast<double>(a) / static_cast<double>(g_std), static_cast<double>(dt));
    }

    double run(float a, float dt) {
        update(a, dt);
        return getFrequencyHz();
    }

    double getFrequencyHz() const { return t.getFrequencyHz(); }
    double getRawFrequencyHz() const { return t.getRawFrequencyHz(); }
    double getConfidence() const { return t.getConfidence(); }
    bool isLocked() const { return t.isLocked(); }
    bool hasCoarseEstimate() const { return t.hasCoarseEstimate(); }
    double getCoarseFrequencyHz() const { return t.getCoarseFrequencyHz(); }
};

// ZeroCross
template<>
struct TrackerPolicy<TrackerType::ZEROCROSS> {
    using Tracker = SchmittTriggerZCFreqTracker;
    Tracker t{ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS};

    double run(float a, float dt) {
        t.update(a / g_std,
                 ZERO_CROSSINGS_SCALE /* max g */,
                 ZERO_CROSSINGS_DEBOUNCE_TIME,
                 ZERO_CROSSINGS_STEEPNESS_TIME, dt);
        return getFrequencyHz();
    }

    double getFrequencyHz() const {
        const float raw = t.getFrequencyHz();
        return isZeroCrossFallback_(raw) ? static_cast<double>(FREQ_GUESS) : static_cast<double>(raw);
    }
    double getRawFrequencyHz() const { return static_cast<double>(t.getRawFrequencyHz()); }
    double getConfidence() const { return static_cast<double>(t.getConfidence()); }
    bool isLocked() const { return t.isLocked(); }
    bool hasCoarseEstimate() const { return t.hasCoarseEstimate(); }
    double getCoarseFrequencyHz() const { return static_cast<double>(t.getCoarseFrequencyHz()); }

private:
    static bool isZeroCrossFallback_(float f) {
        return (f == SCHMITT_TRIGGER_FREQ_INIT || f == SCHMITT_TRIGGER_FALLBACK_FREQ);
    }
};
