
#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

#include <cmath>
#include <algorithm>

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/

#define SCHMITT_TRIGGER_FREQ_INIT     1e-3f
#define SCHMITT_TRIGGER_FREQ_MAX      1e4f
#define SCHMITT_TRIGGER_FALLBACK_FREQ 1e-2f
#define SCHMITT_TRIGGER_FALLBACK_TIME 60.0f

class SchmittTriggerFrequencyDetector {
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
    explicit SchmittTriggerFrequencyDetector(
      float hysteresis = 0.1f, unsigned int periodsInCycle = 1, float fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME)
      : _hysteresis(fabs(hysteresis)),
        _upperThreshold(_hysteresis),
        _lowerThreshold(-_hysteresis),
        _state(State::WAS_NOT_SET),
        _frequency(SCHMITT_TRIGGER_FREQ_INIT),
        _fallbackToLowFreqTime(fallbackToLowFreqTime),
        _lastPeriodEstimate(0.0f),
        _periodVariance(0.0f),
        _amplitudeRatio(0.0f),
        _isFallback(true),
        _periodsInCycle(periodsInCycle),
        _timeInCycle(0.0f),
        _lastLowTime(0.0f),
        _lastHighTime(0.0f),
        _lastCrossingInCycleTime(0.0f),
        _beginningCrossingInCycleTime(0.0f),
        _crossingsCounter(0),
        _periodHistoryIndex(0),
        _periodHistoryCount(0)
    {
      for (auto& p : _periodHistory) p = 0.0f;
    }

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz).
    // signalMagnitude - must be positive (absolute amplitude of the signal).
    // debounceTime (in seconds) - must be positive.
    float update(float signalValue, float signalMagnitude, float debounceTime, float steepnessTime, float dt) {
      if (dt <= 0.0f || fabs(signalMagnitude) <= 0.0f) {
        return _frequency;
      }

      const float scaledValue = signalValue / fabs(signalMagnitude);
      _amplitudeRatio = fabs(signalMagnitude) / _hysteresis;

      switch (_state) {
        case State::WAS_NOT_SET: {
            if (scaledValue > _upperThreshold) {
              _state = State::WAS_HIGH;
            } else if (scaledValue < _lowerThreshold) {
              _state = State::WAS_LOW;
            }
            _lastLowTime = 0.0f;
            _lastHighTime = 0.0f;
            _crossingsCounter = 0;
            _lastCrossingInCycleTime = 0.0f;
            _beginningCrossingInCycleTime = 0.0f;
            _timeInCycle = 0.0f;
            _isFallback = true;
          }
          break;

        case State::WAS_LOW: {
            _timeInCycle += dt;
            float timeSinceLow = _timeInCycle - _lastLowTime;
            float thisCrossingTime = _timeInCycle - timeSinceLow / 2.0f;

            if (scaledValue > _upperThreshold && (timeSinceLow > steepnessTime)
                && (_crossingsCounter == 0 || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {
              _state = State::WAS_HIGH;
              _lastHighTime = _timeInCycle;
              _lastCrossingInCycleTime = thisCrossingTime;

              if (_crossingsCounter == 0) {
                _beginningCrossingInCycleTime = thisCrossingTime;
              }
              _crossingsCounter++;
              if (_crossingsCounter > 1 && (_frequency == SCHMITT_TRIGGER_FREQ_INIT || _frequency == SCHMITT_TRIGGER_FALLBACK_FREQ)) {
                float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / (_crossingsCounter - 1);
                updatePeriodStatistics(period);
                _frequency = 1.0f / period;
                _isFallback = false;
              }
              if (_crossingsCounter == (2 * _periodsInCycle + 1)) {
                float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / (_crossingsCounter - 1);
                updatePeriodStatistics(period);
                _frequency = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
                _isFallback = false;
                _crossingsCounter = 1;
                _lastCrossingInCycleTime = - timeSinceLow / 2.0f;
                _beginningCrossingInCycleTime = _lastCrossingInCycleTime;
                _timeInCycle = 0.0f;
                _lastHighTime = _timeInCycle;
              }
            } else if (scaledValue < _lowerThreshold) {
              _lastLowTime = _timeInCycle;
            }
            if ((_timeInCycle - _lastCrossingInCycleTime) > _fallbackToLowFreqTime) {
              _frequency = SCHMITT_TRIGGER_FALLBACK_FREQ;
              _isFallback = true;
            }
          }
          break;

        case State::WAS_HIGH: {
            _timeInCycle += dt;
            float timeSinceHigh = _timeInCycle - _lastHighTime;
            float thisCrossingTime = _timeInCycle - timeSinceHigh / 2.0f;

            if (scaledValue < _lowerThreshold && (timeSinceHigh > steepnessTime)
                && (_crossingsCounter == 0 || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {
              _state = State::WAS_LOW;
              _lastLowTime = _timeInCycle;
              _lastCrossingInCycleTime = thisCrossingTime;

              if (_crossingsCounter == 0) {
                _beginningCrossingInCycleTime = thisCrossingTime;
              }
              _crossingsCounter++;
              if (_crossingsCounter > 1 && (_frequency == SCHMITT_TRIGGER_FREQ_INIT || _frequency == SCHMITT_TRIGGER_FALLBACK_FREQ)) {
                float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / (_crossingsCounter - 1);
                updatePeriodStatistics(period);
                _frequency = 1.0f / period;
                _isFallback = false;
              }
              if (_crossingsCounter == (2 * _periodsInCycle + 1)) {
                float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
                float period = 2.0f * cycleTime / (_crossingsCounter - 1);
                updatePeriodStatistics(period);
                _frequency = std::min(1.0f / period, SCHMITT_TRIGGER_FREQ_MAX);
                _isFallback = false;
                _crossingsCounter = 1;
                _lastCrossingInCycleTime = - timeSinceHigh / 2.0f;
                _beginningCrossingInCycleTime = _lastCrossingInCycleTime;
                _timeInCycle = 0.0f;
                _lastLowTime = _timeInCycle;
              }
            } else if (scaledValue > _upperThreshold) {
              _lastHighTime = _timeInCycle;
            }
            if ((_timeInCycle - _lastCrossingInCycleTime) > _fallbackToLowFreqTime) {
              _frequency = SCHMITT_TRIGGER_FALLBACK_FREQ;
              _isFallback = true;
            }
          }
          break;

        default:
          _timeInCycle += dt;
      }

      return _frequency;
    }

    // Get latest computed frequency (Hz)
    float getFrequency() const {
      return _frequency;
    }

    // Get phase of sine wave in rad
    float getPhaseEstimate() const {
      if (_frequency <= 0.0f || _isFallback) {
        return 0.0f;
      }

      float period = 1.0f / _frequency;
      float timeSinceLastCrossing = _timeInCycle - _lastCrossingInCycleTime;
      float phase = 2.0f * M_PI * fmodf(timeSinceLastCrossing / period, 1.0f);

      if (_state == State::WAS_LOW) {
        phase = fmodf(phase + M_PI, 2.0f * M_PI);
      }

      if (phase < 0.0f) phase += 2.0f * M_PI;
      return phase;
    }

    // Get quality metrics for the current frequency estimate
    QualityMetrics getQualityMetrics() const {
      QualityMetrics metrics;

      if (_periodHistoryCount < 2) {
        metrics.confidence = 0.1f;
      } else {
        float stddev = sqrtf(_periodVariance);
        float normalizedStddev = stddev / _lastPeriodEstimate;
        metrics.confidence = std::max(0.0f, std::min(1.0f, 1.0f - normalizedStddev));
      }

      metrics.jitter = sqrtf(_periodVariance);
      metrics.amplitude_ratio = _amplitudeRatio;
      metrics.is_fallback = _isFallback;

      return metrics;
    }

    // Reset the detector (clears history)
    void reset() {
      _state = State::WAS_NOT_SET;
      _frequency = SCHMITT_TRIGGER_FREQ_INIT;
      _fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME;
      _periodsInCycle = 1;
      _timeInCycle = 0.0f;
      _lastLowTime = 0.0f;
      _lastHighTime = 0.0f;
      _lastCrossingInCycleTime = 0.0f;
      _beginningCrossingInCycleTime = 0.0f;
      _crossingsCounter = 0;
      _lastPeriodEstimate = 0.0f;
      _periodVariance = 0.0f;
      _amplitudeRatio = 0.0f;
      _isFallback = true;
      _periodHistoryIndex = 0;
      _periodHistoryCount = 0;
      for (auto& p : _periodHistory) p = 0.0f;
    }

  private:
    enum class State {
      WAS_NOT_SET,
      WAS_LOW,
      WAS_HIGH,
    };

    void updatePeriodStatistics(float period) {
      _lastPeriodEstimate = period;
      _periodHistory[_periodHistoryIndex] = period;
      _periodHistoryIndex = (_periodHistoryIndex + 1) % PERIOD_HISTORY_SIZE;
      if (_periodHistoryCount < PERIOD_HISTORY_SIZE) _periodHistoryCount++;
      calculatePeriodVariance();
    }

    void calculatePeriodVariance() {
      if (_periodHistoryCount < 2) {
        _periodVariance = 0.0f;
        return;
      }

      float sum = 0.0f, sumSq = 0.0f;
      for (size_t i = 0; i < _periodHistoryCount; ++i) {
        sum += _periodHistory[i];
        sumSq += _periodHistory[i] * _periodHistory[i];
      }
      float mean = sum / _periodHistoryCount;
      _periodVariance = (sumSq / _periodHistoryCount) - (mean * mean);
    }

    float _hysteresis;                  // Hysteresis threshold
    float _upperThreshold;              // Upper threshold
    float _lowerThreshold;              // Lower threshold
    State _state;                       // Tracks states
    float _frequency;                   // Latest frequency estimate (Hz)
    float _fallbackToLowFreqTime;       // Time to fallback if no crossing detected

    float _lastPeriodEstimate;
    float _periodVariance;
    float _amplitudeRatio;
    bool _isFallback;

    unsigned int _periodsInCycle;
    float _timeInCycle;
    float _lastLowTime;
    float _lastHighTime;
    float _lastCrossingInCycleTime;
    float _beginningCrossingInCycleTime;
    unsigned int _crossingsCounter;

    static constexpr size_t PERIOD_HISTORY_SIZE = 10;
    float _periodHistory[PERIOD_HISTORY_SIZE];
    size_t _periodHistoryIndex;
    size_t _periodHistoryCount;
};

#endif // SCHMITT_TRIGGER_FREQ_DETECTOR_H
