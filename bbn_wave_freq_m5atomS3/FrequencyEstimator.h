
#ifndef FREQUENCY_ESTIMATOR_H
#define FREQUENCY_ESTIMATOR_H

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"

enum class FrequencyTracker {
    Aranovskiy,
    Kalm_ANF,
    ZeroCrossing
};

// Frequency range and fallback guess
constexpr float FREQ_LOWER = 0.04f;
constexpr float FREQ_UPPER = 2.0f;
constexpr float FREQ_GUESS = 0.3f;

namespace wave_utils {
    template <typename T>
    constexpr T clamp(T val, T min, T max) {
        return (val < min) ? min : (val > max) ? max : val;
    }
}

// Base template (undefined)
template<FrequencyTracker TrackerType>
class FrequencyEstimator;

// ----------------------------------------
// Specialization: Aranovskiy Filter
// ----------------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::Aranovskiy> {
public:
    void init() {
        double omega_init = (FREQ_GUESS * 2) * (2 * M_PI);
        double k_gain     = 8.0;
        double theta_0    = - (omega_init * omega_init / 4.0);
        ar_filter_.setParams(omega_init, k_gain);
        ar_filter_.setState(0.0, theta_0, theta_0);
    }

    float estimate(float /*a_noisy*/, float a_no_spikes, float delta_t, float /*t*/) {
        ar_filter_.update(a_no_spikes, delta_t);
        return wave_utils::clamp(ar_filter_.getFrequencyHz(), FREQ_LOWER, FREQ_UPPER);
    }

private:
    AranovskiyFilter<double> ar_filter_;
};

// ----------------------------------------
// Specialization: Kalman ANF
// ----------------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::Kalm_ANF> {
public:
    void init() {
        kalman_anf_.init(0.985f, 1e-5f, 5e+4f, 1.0f, 0.0f, 0.0f, 1.9999f);
    }

    float estimate(float a_noisy, float /*a_no_spikes*/, float delta_t, float /*t*/) {
        float e;
        float f = kalman_anf_.process(a_noisy, delta_t, &e);
        return wave_utils::clamp(f, FREQ_LOWER, FREQ_UPPER);
    }

private:
    KalmANF<float> kalman_anf_;
};

// ----------------------------------------
// Specialization: Zero Crossing Detector
// ----------------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::ZeroCrossing> {
public:
    void init() {
        // No specific init needed
    }

    float estimate(float a_noisy, float /*a_no_spikes*/, float delta_t, float /*t*/) {
        float f = schmitt_.update(a_noisy, kScale, kDebounce, kSteepness, delta_t);
        if (f == SCHMITT_TRIGGER_FREQ_INIT || f == SCHMITT_TRIGGER_FALLBACK_FREQ)
            f = FREQ_GUESS;

        return wave_utils::clamp(f, FREQ_LOWER, FREQ_UPPER);
    }

private:
    SchmittTriggerFrequencyDetector schmitt_;

    static constexpr float kScale     = 1.0f;
    static constexpr float kDebounce  = 0.12f;
    static constexpr float kSteepness = 0.16f;
};

#endif // FREQUENCY_ESTIMATOR_H

