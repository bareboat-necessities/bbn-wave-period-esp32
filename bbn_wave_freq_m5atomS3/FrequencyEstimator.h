#ifndef FREQUENCY_ESTIMATOR_H
#define FREQUENCY_ESTIMATOR_H

#include "KalmanForWaveBasic.h"
#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "SchmittTriggerFrequencyDetector.h"
#include <cmath>
#include <algorithm>

//----------------------------------
// Utility Clamp
//----------------------------------
namespace wave_utils {
template<typename T>
T clamp(T val, T lo, T hi) {
    return std::max(lo, std::min(val, hi));
}
}

//----------------------------------
// Enum: Tracker Selection
//----------------------------------
enum class FrequencyTracker {
    Aranovskiy,
    Kalm_ANF,
    ZeroCrossing
};

//----------------------------------
// Constants
//----------------------------------
constexpr float FREQ_LOWER = 0.03f;  // Hz
constexpr float FREQ_UPPER = 1.0f;   // Hz
constexpr float FREQ_GUESS = 0.12f;  // Hz

constexpr float SCHMITT_TRIGGER_FREQ_INIT     = -1.0f;
constexpr float SCHMITT_TRIGGER_FALLBACK_FREQ = -2.0f;

//----------------------------------
// CRTP Base Class
//----------------------------------
template<typename Derived>
class FrequencyEstimatorBase {
public:
    void init() {
        kalman_.initialize(5.0f, 1e-4f, 1e-2f, 1e-5f);      // p, q, r, b
        kalman_.initMeasurementNoise(1e-3f);               // measurement noise
        derived()->initImpl();           // delegate
    }

    float estimate(float a_noisy, float a_no_spikes, float delta_t, float t) {
        float heave = kalman_.update(a_noisy, delta_t);    // step 1: heave estimation
        float f = derived()->estimateFromHeave(heave, delta_t, t); // step 2
        return wave_utils::clamp(f, FREQ_LOWER, FREQ_UPPER);
    }

    const KalmanForWaveBasic& heaveFilter() const {
        return kalman_;
    }

protected:
    KalmanForWaveBasic kalman_;

private:
    Derived& derived() {
        return static_cast<Derived&>(*this);
    }
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
};

//----------------------------------
// Aranovskiy Specialization
//----------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::Aranovskiy>
    : public FrequencyEstimatorBase<FrequencyEstimator<FrequencyTracker::Aranovskiy>> {
public:
    void initImpl() {
        double omega_init = (FREQ_GUESS * 2) * (2 * M_PI);
        double k_gain = 8.0;
        double theta_0 = -0.09; // -(omega_init * omega_init / 4.0);
        ar_filter_.setParams(omega_init, k_gain);
        ar_filter_.setState(0.0, theta_0, theta_0);
    }

    float estimateFromHeave(float heave, float delta_t, float /*t*/) {
        ar_filter_.update(heave, delta_t);
        return ar_filter_.getFrequencyHz();
    }

private:
    AranovskiyFilter<double> ar_filter_;
};

//----------------------------------
// KalmANF Specialization
//----------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::Kalm_ANF>
    : public FrequencyEstimatorBase<FrequencyEstimator<FrequencyTracker::Kalm_ANF>> {
public:
    void initImpl() {
        kalman_anf_.init(0.985, 1e-5, 5e+4, 1.0, 0.0, 0.0, 1.9999);
    }

    double estimateFromHeave(float double, double delta_t, double /*t*/) {
        double error;
        return kalman_anf_.process(heave, delta_t, &error);
    }

private:
    KalmANF<double> kalman_anf_;
};

//----------------------------------
// ZeroCrossing (Schmitt) Specialization
//----------------------------------
template<>
class FrequencyEstimator<FrequencyTracker::ZeroCrossing>
    : public FrequencyEstimatorBase<FrequencyEstimator<FrequencyTracker::ZeroCrossing>> {
public:
    void initImpl() {
        // No stateful initialization required
    }

    float estimateFromHeave(float heave, float delta_t, float /*t*/) {
        float f = schmitt_.update(heave, kScale, kDebounce, kSteepness, delta_t);
        return (f == SCHMITT_TRIGGER_FREQ_INIT || f == SCHMITT_TRIGGER_FALLBACK_FREQ)
                 ? FREQ_GUESS : f;
    }

private:
    SchmittTriggerFrequencyDetector schmitt_;

    static constexpr float kScale     = 1.0f;
    static constexpr float kDebounce  = 0.12f;
    static constexpr float kSteepness = 0.16f;
};

#endif // FREQUENCY_ESTIMATOR_H


