#pragma once

#include <cmath>

#include "WaveFilesSupport.h"
#include "AranovskiyFreqTracker.h"
#include "KalmANFFreqTracker.h"
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

// KalmANFFreqTracker
template<>
struct TrackerPolicy<TrackerType::KALMANF> {
    using Tracker = KalmANFFreqTracker<double>;
    Tracker t = Tracker();

    double run(float a, float dt) {
        double e;
        double freq = t.process(static_cast<double>(a) / g_std, static_cast<double>(dt), &e);
        return freq;
    }
};

// ZeroCross
template<>
struct TrackerPolicy<TrackerType::ZEROCROSS> {
    using Tracker = SchmittTriggerZCFreqTracker;
    Tracker t = Tracker(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);

    double run(float a, float dt) {
        float f_byZeroCross = t.update(a / g_std,
                                       ZERO_CROSSINGS_SCALE /* max g */,
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
