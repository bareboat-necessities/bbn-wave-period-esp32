#ifndef SCHMITT_TRIGGER_ZC_FREQ_TRACKER_H
#define SCHMITT_TRIGGER_ZC_FREQ_TRACKER_H

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/

#define SCHMITT_TRIGGER_FREQ_INIT     1e-3f
#define SCHMITT_TRIGGER_FREQ_MAX      1e4f
#define SCHMITT_TRIGGER_FALLBACK_FREQ 1e-2f
#define SCHMITT_TRIGGER_FALLBACK_TIME 60.0f

class SchmittTriggerZCFreqTracker {
  public:
    // Quality metrics for frequency estimation
    struct QualityMetrics {
      float confidence;       // 0.0 (no confidence) to 1.0 (full confidence)
      float jitter;           // Time jitter in seconds (standard deviation of period measurements)
      float amplitude_ratio;  // Signal amplitude relative to hysteresis threshold
      bool is_fallback;       // Whether we're using fallback frequency
    };

    // Constructor: sets hysteresis threshold (default: 0.1)
    // hysteresis must be positive and typically between 0.01 and 0.5
    // periodsInCycle must be positive integer
    // fallbackToLowFreqTime must be positive time in seconds
    explicit SchmittTriggerZCFreqTracker(
      float hysteresis = 0.1f,
      unsigned int periodsInCycle = 1,
      float fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME)
      : hysteresis(std::fabs(hysteresis)),
        upperThreshold(std::fabs(hysteresis)),
        lowerThreshold(-std::fabs(hysteresis)),
        state(State::WAS_NOT_SET),
        frequency(SCHMITT_TRIGGER_FREQ_INIT),
        fallbackToLowFreqTime(fallbackToLowFreqTime),
        lastPeriodEstimate(0.0f),
        periodVariance(0.0f),
        amplitudeRatio(0.0f),
        isFallback(true),
        periodsInCycle(periodsInCycle),
        timeInCycle(0.0f),
        lastLowTime(0.0f),
        lastHighTime(0.0f),
        lastCrossingInCycleTime(0.0f),
        beginningCrossingInCycleTime(0.0f),
        crossingsCounter(0),
        periodHistoryIndex(0),
        periodHistoryCount(0),
        lockConfidenceThreshold(0.60f),
        lockAmplitudeRatioMin(1.0f)
    {
      for (auto& p : periodHistory) p = 0.0f;
    }

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz).
    // signalMagnitude - must be positive (absolute amplitude of the signal).
    // debounceTime (in seconds) - must be positive.
    float update(float signalValue,
                 float signalMagnitude,
                 float debounceTime,
                 float steepnessTime,
                 float dt)
    {
      if (dt <= 0.0f || std::fabs(signalMagnitude) <= 0.0f) {
        return frequency;
      }

      const float scaledValue = signalValue / std::fabs(signalMagnitude);
      amplitudeRatio = std::fabs(signalMagnitude) / std::max(hysteresis, 1e-12f);

      switch (state) {
        case State::WAS_NOT_SET: {
            if (scaledValue > upperThreshold) {
              state = State::WAS_HIGH;
            } else if (scaledValue < lowerThreshold) {
              state = State::WAS_LOW;
            }
            lastLowTime = 0.0f;
            lastHighTime = 0.0f;
            crossingsCounter = 0;
            lastCrossingInCycleTime = 0.0f;
            beginningCrossingInCycleTime = 0.0f;
            timeInCycle = 0.0f;
            isFallback = true;
          }
          break;

        case State::WAS_LOW: {
            timeInCycle += dt;
            float timeSinceLow = timeInCycle - lastLowTime;
            float thisCrossingTime = timeInCycle - timeSinceLow / 2.0f;

            if (scaledValue > upperThreshold &&
                (timeSinceLow > steepnessTime) &&
                (crossingsCounter == 0 ||
                 (thisCrossingTime - lastCrossingInCycleTime) > debounceTime))
            {
              state = State::WAS_HIGH;
              lastHighTime = timeInCycle;
              lastCrossingInCycleTime = thisCrossingTime;

              if (crossingsCounter == 0) {
                beginningCrossingInCycleTime = thisCrossingTime;
              }

              crossingsCounter++;

              if (crossingsCounter > 1 &&
                  (frequency == SCHMITT_TRIGGER_FREQ_INIT ||
                   frequency == SCHMITT_TRIGGER_FALLBACK_FREQ))
              {
                float cycleTime = thisCrossingTime - beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / static_cast<float>(crossingsCounter - 1);
                updatePeriodStatistics(period);
                frequency = 1.0f / period;
                isFallback = false;
              }

              if (crossingsCounter == (2 * periodsInCycle + 1)) {
                float cycleTime = thisCrossingTime - beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / static_cast<float>(crossingsCounter - 1);
                updatePeriodStatistics(period);
                frequency = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
                isFallback = false;

                crossingsCounter = 1;
                lastCrossingInCycleTime = -timeSinceLow / 2.0f;
                beginningCrossingInCycleTime = lastCrossingInCycleTime;
                timeInCycle = 0.0f;
                lastHighTime = timeInCycle;
              }
            } else if (scaledValue < lowerThreshold) {
              lastLowTime = timeInCycle;
            }

            if ((timeInCycle - lastCrossingInCycleTime) > fallbackToLowFreqTime) {
              frequency = SCHMITT_TRIGGER_FALLBACK_FREQ;
              isFallback = true;
            }
          }
          break;

        case State::WAS_HIGH: {
            timeInCycle += dt;
            float timeSinceHigh = timeInCycle - lastHighTime;
            float thisCrossingTime = timeInCycle - timeSinceHigh / 2.0f;

            if (scaledValue < lowerThreshold &&
                (timeSinceHigh > steepnessTime) &&
                (crossingsCounter == 0 ||
                 (thisCrossingTime - lastCrossingInCycleTime) > debounceTime))
            {
              state = State::WAS_LOW;
              lastLowTime = timeInCycle;
              lastCrossingInCycleTime = thisCrossingTime;

              if (crossingsCounter == 0) {
                beginningCrossingInCycleTime = thisCrossingTime;
              }

              crossingsCounter++;

              if (crossingsCounter > 1 &&
                  (frequency == SCHMITT_TRIGGER_FREQ_INIT ||
                   frequency == SCHMITT_TRIGGER_FALLBACK_FREQ))
              {
                float cycleTime = thisCrossingTime - beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / static_cast<float>(crossingsCounter - 1);
                updatePeriodStatistics(period);
                frequency = 1.0f / period;
                isFallback = false;
              }

              if (crossingsCounter == (2 * periodsInCycle + 1)) {
                float cycleTime = thisCrossingTime - beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / static_cast<float>(crossingsCounter - 1);
                updatePeriodStatistics(period);
                frequency = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
                isFallback = false;

                crossingsCounter = 1;
                lastCrossingInCycleTime = -timeSinceHigh / 2.0f;
                beginningCrossingInCycleTime = lastCrossingInCycleTime;
                timeInCycle = 0.0f;
                lastLowTime = timeInCycle;
              }
            } else if (scaledValue > upperThreshold) {
              lastHighTime = timeInCycle;
            }

            if ((timeInCycle - lastCrossingInCycleTime) > fallbackToLowFreqTime) {
              frequency = SCHMITT_TRIGGER_FALLBACK_FREQ;
              isFallback = true;
            }
          }
          break;

        default:
          timeInCycle += dt;
      }

      return frequency;
    }

    // Get latest computed frequency (Hz)
    float getFrequency() const {
      return frequency;
    }

    // Uniform tracker API
    float getFrequencyHz() const {
      return frequency;
    }

    float getRawFrequencyHz() const {
      return frequency;
    }

    float getConfidence() const {
      return getQualityMetrics().confidence;
    }

    bool isLocked() const {
      const QualityMetrics q = getQualityMetrics();
      return !q.is_fallback &&
             periodHistoryCount >= 2 &&
             q.confidence >= lockConfidenceThreshold &&
             q.amplitude_ratio >= lockAmplitudeRatioMin;
    }

    bool hasCoarseEstimate() const {
      return false;
    }

    float getCoarseFrequencyHz() const {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
      throw std::logic_error("SchmittTriggerZCFreqTracker: coarse frequency estimate is not implemented");
#else
      std::abort();
#endif
    }

    // Optional tuning for synthesized lock logic
    void setLockParams(float confidenceThreshold, float amplitudeRatioMin = 1.0f) {
      lockConfidenceThreshold = std::clamp(confidenceThreshold, 0.0f, 1.0f);
      lockAmplitudeRatioMin = std::max(amplitudeRatioMin, 0.0f);
    }

    // Get phase of sine wave in rad
    float getPhaseEstimate() const {
      if (frequency <= 0.0f || isFallback) {
        return 0.0f;
      }
      float period = 1.0f / frequency;
      float timeSinceLastCrossing = timeInCycle - lastCrossingInCycleTime;
      float phase = 2.0f * static_cast<float>(M_PI) *
                    std::fmod(timeSinceLastCrossing / period, 1.0f);
      if (state == State::WAS_LOW) {
        phase = std::fmod(phase + static_cast<float>(M_PI),
                          2.0f * static_cast<float>(M_PI));
      }
      if (phase < 0.0f) phase += 2.0f * static_cast<float>(M_PI);
      return phase;
    }

    // Get quality metrics for the current frequency estimate
    QualityMetrics getQualityMetrics() const {
      QualityMetrics metrics;
      if (periodHistoryCount < 2 || lastPeriodEstimate <= 0.0f) {
        metrics.confidence = isFallback ? 0.0f : 0.1f;
      } else {
        float stddev = std::sqrt(std::max(periodVariance, 0.0f));
        float normalizedStddev = stddev / std::max(lastPeriodEstimate, 1e-12f);

        // Base confidence from timing stability.
        float conf = std::max(0.0f, std::min(1.0f, 1.0f - normalizedStddev));

        // Mild amplitude shaping: if the signal barely clears hysteresis,
        // confidence should not look "fully locked".
        const float ampScore = std::max(0.0f, std::min(1.0f, amplitudeRatio / 2.0f));
        conf *= (0.35f + 0.65f * ampScore);

        if (isFallback) {
          conf = 0.0f;
        }
        metrics.confidence = conf;
      }

      metrics.jitter = std::sqrt(std::max(periodVariance, 0.0f));
      metrics.amplitude_ratio = amplitudeRatio;
      metrics.is_fallback = isFallback;
      return metrics;
    }

    // Reset the detector (clears history)
    void reset() {
      state = State::WAS_NOT_SET;
      frequency = SCHMITT_TRIGGER_FREQ_INIT;
      fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME;
      periodsInCycle = 1;
      timeInCycle = 0.0f;
      lastLowTime = 0.0f;
      lastHighTime = 0.0f;
      lastCrossingInCycleTime = 0.0f;
      beginningCrossingInCycleTime = 0.0f;
      crossingsCounter = 0;
      lastPeriodEstimate = 0.0f;
      periodVariance = 0.0f;
      amplitudeRatio = 0.0f;
      isFallback = true;
      periodHistoryIndex = 0;
      periodHistoryCount = 0;
      for (auto& p : periodHistory) p = 0.0f;
    }

  private:
    enum class State {
      WAS_NOT_SET,
      WAS_LOW,
      WAS_HIGH,
    };

    void updatePeriodStatistics(float period) {
      lastPeriodEstimate = period;
      periodHistory[periodHistoryIndex] = period;
      periodHistoryIndex = (periodHistoryIndex + 1) % PERIOD_HISTORY_SIZE;
      if (periodHistoryCount < PERIOD_HISTORY_SIZE) periodHistoryCount++;
      calculatePeriodVariance();
    }

    void calculatePeriodVariance() {
      if (periodHistoryCount < 2) {
        periodVariance = 0.0f;
        return;
      }

      float sum = 0.0f;
      float sumSq = 0.0f;
      for (size_t i = 0; i < periodHistoryCount; ++i) {
        sum += periodHistory[i];
        sumSq += periodHistory[i] * periodHistory[i];
      }

      float mean = sum / static_cast<float>(periodHistoryCount);
      periodVariance = (sumSq / static_cast<float>(periodHistoryCount)) - (mean * mean);
      if (periodVariance < 0.0f) {
        periodVariance = 0.0f;
      }
    }

    float hysteresis;                  // Hysteresis threshold
    float upperThreshold;              // Upper threshold
    float lowerThreshold;              // Lower threshold
    State state;                       // Tracks states
    float frequency;                   // Latest frequency estimate (Hz)
    float fallbackToLowFreqTime;       // Time to fallback if no crossing detected

    float lastPeriodEstimate;
    float periodVariance;
    float amplitudeRatio;
    bool isFallback;

    unsigned int periodsInCycle;
    float timeInCycle;
    float lastLowTime;
    float lastHighTime;
    float lastCrossingInCycleTime;
    float beginningCrossingInCycleTime;
    unsigned int crossingsCounter;

    static constexpr size_t PERIOD_HISTORY_SIZE = 10;
    float periodHistory[PERIOD_HISTORY_SIZE];
    size_t periodHistoryIndex;
    size_t periodHistoryCount;

    // Uniform-API lock tuning
    float lockConfidenceThreshold;
    float lockAmplitudeRatioMin;
};

#endif // SCHMITT_TRIGGER_ZC_FREQ_TRACKER_H
