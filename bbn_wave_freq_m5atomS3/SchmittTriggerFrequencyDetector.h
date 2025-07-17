#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

#include <cmath>
#include <algorithm>

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/

#define SCHMITT_TRIGGER_FREQ_INIT         1e-3f
#define SCHMITT_TRIGGER_FREQ_MAX          1e4f
#define SCHMITT_TRIGGER_FALLBACK_FREQ     1e-2f
#define SCHMITT_TRIGGER_FALLBACK_TIME     60.0f

class SchmittTriggerFrequencyDetector {
public:
    // Quality metrics for frequency estimation
    struct QualityMetrics {
        float confidence;       // 0.0 (no confidence) to 1.0 (full confidence)
        float jitter;           // Time jitter in seconds (standard deviation of period measurements)
        float amplitude_ratio;  // Signal amplitude relative to hysteresis threshold
        bool  is_fallback;      // Whether we're using fallback frequency
    };

    // Constructor: sets hysteresis threshold (default: 0.1)
    // hysteresis must be positive and typically between 0.01 and 0.5
    // periodsInCycle must be positive integer
    // fallbackToLowFreqTime must be positive time in seconds
    explicit SchmittTriggerFrequencyDetector(
        float hysteresis = 0.1f,
        unsigned int periodsInCycle = 1,
        float fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME
    )
      : hysteresisThreshold(std::fabs(hysteresis)),
        upperThreshold(hysteresisThreshold),
        lowerThreshold(-hysteresisThreshold),
        currentState(State::WAS_NOT_SET),
        frequencyHz(SCHMITT_TRIGGER_FREQ_INIT),
        fallbackToLowFreqTime(fallbackToLowFreqTime),
        lastPeriodEstimate(0.0f),
        periodVariance(0.0f),
        amplitudeRatio(0.0f),
        fallbackActive(true),
        periodsPerCycle(periodsInCycle),
        cycleTime(0.0f),
        lastLowCrossTime(0.0f),
        lastHighCrossTime(0.0f),
        lastCrossingTime(0.0f),
        cycleStartCrossingTime(0.0f),
        crossingCount(0),
        historyIndex(0),
        historyCount(0)
    {
        for (size_t i = 0; i < PERIOD_HISTORY_SIZE; ++i) {
            periodHistory[i] = 0.0f;
        }
    }

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz).
    // signalValue      - raw signal value (can be positive or negative)
    // signalMagnitude  - must be positive (absolute amplitude of the signal)
    // debounceTime     - in seconds, must be positive
    // steepnessTime    - in seconds, must be positive (min time for a valid crossing)
    // dt               - elapsed time since last update (in seconds)
    float update(
        float signalValue,
        float signalMagnitude,
        float debounceTime,
        float steepnessTime,
        float dt
    ) {
        if (dt <= 0.0f || signalMagnitude == 0.0f) {
            return frequencyHz;
        }

        // normalize to hysteresis range [-1,1]
        float scaled = signalValue / std::fabs(signalMagnitude);
        amplitudeRatio = std::fabs(signalMagnitude) / hysteresisThreshold;

        switch (currentState) {
            case State::WAS_NOT_SET:
                // initial state: detect first crossing
                initializeState(scaled);
                break;

            case State::WAS_LOW:
                // we were below the lower threshold: look for upward crossing
                processLowState(scaled, debounceTime, steepnessTime, dt);
                break;

            case State::WAS_HIGH:
                // we were above the upper threshold: look for downward crossing
                processHighState(scaled, debounceTime, steepnessTime, dt);
                break;
        }

        return frequencyHz;
    }

    // Get latest computed frequency (Hz)
    float getFrequency() const {
        return frequencyHz;
    }

    // Get phase of sine wave in rad
    float getPhaseEstimate() const {
        if (frequencyHz <= 0.0f || fallbackActive) {
            return 0.0f;
        }
        float period = 1.0f / frequencyHz;
        float timeSinceLast = cycleTime - lastCrossingTime;
        float phase = 2.0f * M_PI * std::fmod(timeSinceLast / period, 1.0f);

        if (currentState == State::WAS_LOW) {
            phase = std::fmod(phase + M_PI, 2.0f * M_PI);
        }
        return (phase < 0.0f ? phase + 2.0f*M_PI : phase);
    }

    // Get quality metrics for the current frequency estimate
    QualityMetrics getQualityMetrics() const {
        QualityMetrics m;
        if (historyCount < 2) {
            m.confidence      = 0.1f;
        } else {
            float stddev = std::sqrt(periodVariance);
            float norm   = stddev / lastPeriodEstimate;
            float raw    = 1.0f - norm;
            // clamp between 0 and 1
            m.confidence      = std::max(0.0f, std::min(raw, 1.0f));
        }
        m.jitter          = std::sqrt(periodVariance);
        m.amplitude_ratio = amplitudeRatio;
        m.is_fallback     = fallbackActive;
        return m;
    }

    // Reset the detector (clears history and returns to initial state)
    void reset() {
        currentState           = State::WAS_NOT_SET;
        frequencyHz            = SCHMITT_TRIGGER_FREQ_INIT;
        fallbackToLowFreqTime  = SCHMITT_TRIGGER_FALLBACK_TIME;
        periodsPerCycle        = 1;
        cycleTime              = 0.0f;
        lastLowCrossTime       = 0.0f;
        lastHighCrossTime      = 0.0f;
        lastCrossingTime       = 0.0f;
        cycleStartCrossingTime = 0.0f;
        crossingCount          = 0;
        lastPeriodEstimate     = 0.0f;
        periodVariance         = 0.0f;
        amplitudeRatio         = 0.0f;
        fallbackActive         = true;
        historyIndex           = 0;
        historyCount           = 0;
        for (size_t i = 0; i < PERIOD_HISTORY_SIZE; ++i) {
            periodHistory[i] = 0.0f;
        }
    }

private:
    enum class State {
        WAS_NOT_SET,
        WAS_LOW,
        WAS_HIGH,
    };

    void initializeState(float scaled) {
        if (scaled > upperThreshold) {
            currentState = State::WAS_HIGH;
        } else if (scaled < lowerThreshold) {
            currentState = State::WAS_LOW;
        }
        // reset cycle timers & counters
        cycleTime              = 0.0f;
        lastLowCrossTime       = 0.0f;
        lastHighCrossTime      = 0.0f;
        lastCrossingTime       = 0.0f;
        cycleStartCrossingTime = 0.0f;
        crossingCount          = 0;
        fallbackActive         = true;
    }

    void processLowState(float scaled, float debounceTime, float steepnessTime, float dt) {
        cycleTime += dt;  // advance by actual elapsed time
        float sinceLow  = cycleTime - lastLowCrossTime;
        float thisCross = cycleTime - sinceLow / 2.0f;

        // valid rising edge?
        if (scaled > upperThreshold
            && sinceLow > steepnessTime
            && (crossingCount == 0 || (thisCross - lastCrossingTime) > debounceTime))
        {
            currentState      = State::WAS_HIGH;
            lastHighCrossTime = cycleTime;       // time we crossed high
            lastCrossingTime  = thisCross;       // mark crossing

            if (crossingCount == 0) {
                cycleStartCrossingTime = thisCross;  // first crossing in cycle
            }
            ++crossingCount;
            computeFrequencyOnCrossing(thisCross);
        }
        // update last low-cross time for hysteresis
        else if (scaled < lowerThreshold) {
            lastLowCrossTime = cycleTime;
        }
        checkFallbackCondition();
    }

    void processHighState(float scaled, float debounceTime, float steepnessTime, float dt) {
        cycleTime += dt;  // advance by actual elapsed time
        float sinceHigh = cycleTime - lastHighCrossTime;
        float thisCross = cycleTime - sinceHigh / 2.0f;

        // valid falling edge?
        if (scaled < lowerThreshold
            && sinceHigh > steepnessTime
            && (crossingCount == 0 || (thisCross - lastCrossingTime) > debounceTime))
        {
            currentState     = State::WAS_LOW;
            lastLowCrossTime = cycleTime;        // time we crossed low
            lastCrossingTime = thisCross;        // mark crossing

            if (crossingCount == 0) {
                cycleStartCrossingTime = thisCross; // first crossing in cycle
            }
            ++crossingCount;
            computeFrequencyOnCrossing(thisCross);
        }
        // update last high-cross time for hysteresis
        else if (scaled > upperThreshold) {
            lastHighCrossTime = cycleTime;
        }
        checkFallbackCondition();
    }

    void computeFrequencyOnCrossing(float thisCross) {
        // initial estimate when we see at least two crossings
        if (crossingCount > 1
            && (frequencyHz == SCHMITT_TRIGGER_FREQ_INIT
                || frequencyHz == SCHMITT_TRIGGER_FALLBACK_FREQ))
        {
            float cycleDur = thisCross - cycleStartCrossingTime;
            float period   = 2.0f * cycleDur / (crossingCount - 1);
            updatePeriodStatistics(period);
            frequencyHz    = 1.0f / period;
            fallbackActive = false;
        }

        // full-cycle complete?
        if (crossingCount == (2 * periodsPerCycle + 1)) {
            float cycleDur = thisCross - cycleStartCrossingTime;
            float period   = 2.0f * cycleDur / (crossingCount - 1);
            updatePeriodStatistics(period);
            frequencyHz    = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
            fallbackActive = false;

            // restart cycle
            crossingCount           = 1;
            lastCrossingTime        = -cycleDur / (crossingCount - 1);
            cycleStartCrossingTime  = lastCrossingTime;
            cycleTime               = 0.0f;
            lastHighCrossTime       = 0.0f;
            lastLowCrossTime        = 0.0f;
        }
    }

    void checkFallbackCondition() {
        // apply fallback if no crossing within timeout
        if ((cycleTime - lastCrossingTime) > fallbackToLowFreqTime) {
            frequencyHz    = SCHMITT_TRIGGER_FALLBACK_FREQ;
            fallbackActive = true;
        }
    }

    void updatePeriodStatistics(float period) {
        lastPeriodEstimate              = period;             // store last raw period
        periodHistory[historyIndex]     = period;             // circular buffer
        historyIndex                    = (historyIndex + 1) % PERIOD_HISTORY_SIZE;
        if (historyCount < PERIOD_HISTORY_SIZE) ++historyCount;
        calculatePeriodVariance();
    }

    void calculatePeriodVariance() {
        if (historyCount < 2) {
            periodVariance = 0.0f;
            return;
        }
        float sum   = 0.0f, sumSq = 0.0f;
        for (size_t i = 0; i < historyCount; ++i) {
            float v   = periodHistory[i];
            sum      += v;
            sumSq    += v * v;
        }
        float mean = sum / historyCount;
        periodVariance = (sumSq / historyCount) - (mean * mean);
    }

    // hysteresis thresholds
    float hysteresisThreshold;            // Hysteresis threshold
    float upperThreshold;                 // Upper threshold
    float lowerThreshold;                 // Lower threshold

    State       currentState;             // Tracks states
    float       frequencyHz;              // Latest frequency estimate (Hz)
    float       fallbackToLowFreqTime;    // Time to fallback if no crossing detected

    float       lastPeriodEstimate;
    float       periodVariance;
    float       amplitudeRatio;
    bool        fallbackActive;

    unsigned int periodsPerCycle;
    float       cycleTime;
    float       lastLowCrossTime;
    float       lastHighCrossTime;
    float       lastCrossingTime;
    float       cycleStartCrossingTime;
    unsigned int crossingCount;

    static constexpr size_t PERIOD_HISTORY_SIZE = 10;
    float       periodHistory[PERIOD_HISTORY_SIZE];
    size_t      historyIndex;
    size_t      historyCount;
};

#endif // SCHMITT_TRIGGER_FREQ_DETECTOR_H
