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
        float amplitudeRatio;   // Signal amplitude relative to hysteresis threshold
        bool  isFallback;       // Whether we're using fallback frequency
    };

    // Constructor: sets hysteresis threshold (default: 0.1),
    // periodsPerCycle (default: 1),
    // fallbackToLowFreqTime (default: SCHMITT_TRIGGER_FALLBACK_TIME)
    // hysteresisThresholdArg must be positive and typically between 0.01 and 0.5
    // periodsPerCycleArg must be positive integer
    // fallbackToLowFreqTimeArg must be positive time in seconds
    explicit SchmittTriggerFrequencyDetector(
        float hysteresisThresholdArg = 0.1f,
        unsigned int periodsPerCycleArg = 1,
        float fallbackToLowFreqTimeArg = SCHMITT_TRIGGER_FALLBACK_TIME
    )
      : hysteresisThreshold(std::fabs(hysteresisThresholdArg)),
        upperThreshold(hysteresisThreshold),
        lowerThreshold(-hysteresisThreshold),
        currentState(State::NotSet),
        frequencyHz(SCHMITT_TRIGGER_FREQ_INIT),
        fallbackToLowFreqTime(fallbackToLowFreqTimeArg),
        lastPeriodEstimate(0.0f),
        periodVariance(0.0f),
        amplitudeRatio(0.0f),
        fallbackActive(true),
        periodsPerCycle(periodsPerCycleArg),
        cycleTime(0.0f),
        lastLowCrossTime(0.0f),
        lastHighCrossTime(0.0f),
        lastCrossingTime(0.0f),
        cycleStartCrossingTime(0.0f),
        crossingCount(0),
        historyIndex(0),
        historyCount(0)
    {
        for (auto& p : periodHistory) {
            p = 0.0f;
        }
    }

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz).
    // signalValue      - raw signal value (can be positive or negative)
    // signalAmplitude  - must be positive (absolute amplitude of the signal)
    // debounceTime     - in seconds, must be positive
    // steepnessTime    - in seconds, must be positive (min time for a valid crossing)
    // dt               - elapsed time since last update (in seconds)
    float update(
        float signalValue,
        float signalAmplitude,
        float debounceTime,
        float steepnessTime,
        float dt
    ) {
        if (dt <= 0.0f || signalAmplitude == 0.0f) {
            return frequencyHz;
        }

        // normalize signal to hysteresis range [-1,1]
        float scaled = signalValue / std::fabs(signalAmplitude);
        amplitudeRatio = std::fabs(signalAmplitude) / hysteresisThreshold;

        switch (currentState) {
            case State::NotSet:
                // initial state: detect first crossing
                initializeState(scaled);
                break;

            case State::Low:
                // we were below the lower threshold: look for upward crossing
                processLowState(scaled, debounceTime, steepnessTime);
                break;

            case State::High:
                // we were above the upper threshold: look for downward crossing
                processHighState(scaled, debounceTime, steepnessTime);
                break;
        }

        return frequencyHz;
    }

    // Get latest computed frequency (Hz)
    float getFrequency() const {
        return frequencyHz;
    }

    // Get phase of sine wave in radians
    float getPhaseEstimate() const {
        if (frequencyHz <= 0.0f || fallbackActive) {
            return 0.0f;
        }
        // fraction of period since last crossing
        float period = 1.0f / frequencyHz;
        float timeSince = cycleTime - lastCrossingTime;
        float phase = 2.0f * M_PI * std::fmod(timeSince / period, 1.0f);

        // shift by π if last state was Low
        if (currentState == State::Low) {
            phase = std::fmod(phase + M_PI, 2.0f * M_PI);
        }
        return (phase < 0.0f ? phase + 2.0f*M_PI : phase);
    }

    // Get quality metrics for the current frequency estimate
    QualityMetrics getQualityMetrics() const {
        QualityMetrics m;
        if (historyCount < 2) {
            m.confidence = 0.1f;
        } else {
            float stddev = std::sqrt(periodVariance);
            float norm = stddev / lastPeriodEstimate;
            m.confidence = std::clamp(1.0f - norm, 0.0f, 1.0f);
        }
        m.jitter = std::sqrt(periodVariance);
        m.amplitudeRatio = amplitudeRatio;
        m.isFallback = fallbackActive;
        return m;
    }

    // Reset the detector (clears history and returns to initial state)
    void reset() {
        currentState = State::NotSet;
        frequencyHz = SCHMITT_TRIGGER_FREQ_INIT;
        fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME;
        periodsPerCycle = 1;
        cycleTime = 0.0f;
        lastLowCrossTime = 0.0f;
        lastHighCrossTime = 0.0f;
        lastCrossingTime = 0.0f;
        cycleStartCrossingTime = 0.0f;
        crossingCount = 0;
        lastPeriodEstimate = 0.0f;
        periodVariance = 0.0f;
        amplitudeRatio = 0.0f;
        fallbackActive = true;
        historyIndex = 0;
        historyCount = 0;
        for (auto& p : periodHistory) {
            p = 0.0f;
        }
    }

private:
    enum class State { NotSet, Low, High };

    // Initialize after reset or on first call
    void initializeState(float scaled) {
        if (scaled > upperThreshold) {
            currentState = State::High;
        } else if (scaled < lowerThreshold) {
            currentState = State::Low;
        }
        // reset timers & counters
        cycleTime = 0.0f;
        lastLowCrossTime = 0.0f;
        lastHighCrossTime = 0.0f;
        lastCrossingTime = 0.0f;
        cycleStartCrossingTime = 0.0f;
        crossingCount = 0;
        fallbackActive = true;
    }

    // Handle transitions from Low state
    void processLowState(float scaled, float debounceTime, float steepnessTime) {
        cycleTime += steepnessTime;  // advance time in cycle
        float sinceLow = cycleTime - lastLowCrossTime;
        float thisCrossing = cycleTime - sinceLow / 2.0f;

        // valid rising edge?
        if (scaled > upperThreshold
            && sinceLow > steepnessTime
            && (crossingCount == 0 || (thisCrossing - lastCrossingTime) > debounceTime))
        {
            currentState = State::High;
            lastHighCrossTime = cycleTime;           // update last time we crossed high
            lastCrossingTime = thisCrossing;         // mark crossing time

            if (crossingCount == 0) {
                cycleStartCrossingTime = thisCrossing; // first crossing in this cycle
            }
            ++crossingCount;
            computeFrequencyOnCrossing(thisCrossing);
        }
        // update last low-cross time for hysteresis
        else if (scaled < lowerThreshold) {
            lastLowCrossTime = cycleTime;
        }
        checkFallbackCondition();
    }

    // Handle transitions from High state
    void processHighState(float scaled, float debounceTime, float steepnessTime) {
        cycleTime += steepnessTime;  // advance time in cycle
        float sinceHigh = cycleTime - lastHighCrossTime;
        float thisCrossing = cycleTime - sinceHigh / 2.0f;

        // valid falling edge?
        if (scaled < lowerThreshold
            && sinceHigh > steepnessTime
            && (crossingCount == 0 || (thisCrossing - lastCrossingTime) > debounceTime))
        {
            currentState = State::Low;
            lastLowCrossTime = cycleTime;           // update last time we crossed low
            lastCrossingTime = thisCrossing;        // mark crossing time

            if (crossingCount == 0) {
                cycleStartCrossingTime = thisCrossing; // first crossing in this cycle
            }
            ++crossingCount;
            computeFrequencyOnCrossing(thisCrossing);
        }
        // update last high-cross time for hysteresis
        else if (scaled > upperThreshold) {
            lastHighCrossTime = cycleTime;
        }
        checkFallbackCondition();
    }

    // Calculate frequency when enough crossings have occurred
    void computeFrequencyOnCrossing(float thisCrossing) {
        // initial estimate when we see at least two crossings
        if (crossingCount > 1
            && (frequencyHz == SCHMITT_TRIGGER_FREQ_INIT || frequencyHz == SCHMITT_TRIGGER_FALLBACK_FREQ))
        {
            float cycleDuration = thisCrossing - cycleStartCrossingTime;
            float period = 2.0f * cycleDuration / (crossingCount - 1);
            updatePeriodStatistics(period);
            frequencyHz = 1.0f / period;
            fallbackActive = false;
        }

        // full-cycle: reset counters after (2*periodsPerCycle + 1) crossings
        if (crossingCount == (2 * periodsPerCycle + 1)) {
            float cycleDuration = thisCrossing - cycleStartCrossingTime;
            float period = 2.0f * cycleDuration / (crossingCount - 1);
            updatePeriodStatistics(period);
            frequencyHz = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
            fallbackActive = false;

            // prepare for next cycle
            crossingCount = 1;
            lastCrossingTime = - cycleDuration / (crossingCount - 1);
            cycleStartCrossingTime = lastCrossingTime;
            cycleTime = 0.0f;
            lastHighCrossTime = 0.0f;
            lastLowCrossTime = 0.0f;
        }
    }

    // Apply fallback frequency if no crossing within timeout
    void checkFallbackCondition() {
        if ((cycleTime - lastCrossingTime) > fallbackToLowFreqTime) {
            frequencyHz = SCHMITT_TRIGGER_FALLBACK_FREQ;
            fallbackActive = true;
        }
    }

    // Add new period to history and update variance
    void updatePeriodStatistics(float period) {
        lastPeriodEstimate = period;               // store last raw period
        periodHistory[historyIndex] = period;      // circular buffer
        historyIndex = (historyIndex + 1) % kHistorySize;
        if (historyCount < kHistorySize) {
            ++historyCount;
        }
        calculatePeriodVariance();
    }

    // Compute variance of recorded periods
    void calculatePeriodVariance() {
        if (historyCount < 2) {
            periodVariance = 0.0f;
            return;
        }
        float sum = 0.0f, sumSq = 0.0f;
        for (size_t i = 0; i < historyCount; ++i) {
            float v = periodHistory[i];
            sum += v;
            sumSq += v * v;
        }
        float mean = sum / historyCount;
        periodVariance = (sumSq / historyCount) - (mean * mean);
    }

    // hysteresis thresholds
    float hysteresisThreshold;            // hysteresis amplitude
    float upperThreshold;                 // normalized upper crossing threshold
    float lowerThreshold;                 // normalized lower crossing threshold

    State currentState;                   // current Schmitt trigger state
    float frequencyHz;                    // latest frequency estimate (Hz)
    float fallbackToLowFreqTime;          // time without crossing before fallback

    float lastPeriodEstimate;             // last measured period (s)
    float periodVariance;                 // variance of period measurements
    float amplitudeRatio;                 // signal amplitude / hysteresisThreshold
    bool  fallbackActive;                 // true if using fallback frequency

    unsigned int periodsPerCycle;         // number of half-cycles per full cycle
    float cycleTime;                      // elapsed time within the current cycle
    float lastLowCrossTime;               // time of last low-to-high crossing
    float lastHighCrossTime;              // time of last high-to-low crossing
    float lastCrossingTime;               // time of last counted crossing
    float cycleStartCrossingTime;         // time of first crossing in this cycle
    unsigned int crossingCount;           // how many crossings since cycle start

    static constexpr size_t kHistorySize = 10;
    float periodHistory[kHistorySize];    // buffer of recent period measurements
    size_t historyIndex;                  // next index to write in periodHistory
    size_t historyCount;                  // how many valid entries in periodHistory
};

#endif // SCHMITT_TRIGGER_FREQ_DETECTOR_H
