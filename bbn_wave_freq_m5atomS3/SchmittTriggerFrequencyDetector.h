#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

#define SCHMITT_TRIGGER_FREQ_INIT     1e-3f
#define SCHMITT_TRIGGER_FREQ_MAX      1e4f
#define SCHMITT_TRIGGER_FALLBACK_FREQ 1e-2f
#define SCHMITT_TRIGGER_FALLBACK_TIME 60.0f

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/
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
      float hysteresis = 0.1f, unsigned int periodsInCycle = 1, float fallbackToLowFreqTime = SCHMITT_TRIGGER_FALLBACK_TIME);

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz)
    // signalMagnitude must be positive (absolute amplitude of the signal)
    // debounceTime (in seconds) must be positive
    float update(float signalValue, float signalMagnitude, float debounceTime, float steepnessTime, float dt);

    // Get latest computed frequency (Hz)
    float getFrequency() const;

    // Get quality metrics for the current frequency estimate
    QualityMetrics getQualityMetrics() const;

    // Reset the detector (clears history)
    void reset();

  private:
    enum class State {
      WAS_NOT_SET,        // Initial undefined
      WAS_LOW,            // Below lower threshold
      WAS_HIGH,           // Above upper threshold
    };

    // Update period statistics with new period measurement
    void updatePeriodStatistics(float period);
    
    // Calculate current variance of period measurements
    void calculatePeriodVariance();

    float _hysteresis;            // Hysteresis threshold
    float _upperThreshold;        // Upper threshold
    float _lowerThreshold;        // Lower threshold
    State _state;                 // Tracks states
    float _frequency;             // Latest frequency estimate (Hz)
    float _fallbackToLowFreqTime; // Time to fallback if no crossing detected  

    // Quality tracking
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
    
    // For period statistics
    static constexpr size_t PERIOD_HISTORY_SIZE = 10;
    float _periodHistory[PERIOD_HISTORY_SIZE];
    size_t _periodHistoryIndex;
    size_t _periodHistoryCount;
};

SchmittTriggerFrequencyDetector::SchmittTriggerFrequencyDetector(
  float hysteresis, unsigned int periodsInCycle, float fallbackToLowFreqTime)
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

void SchmittTriggerFrequencyDetector::updatePeriodStatistics(float period) {
    _lastPeriodEstimate = period;
    _periodHistory[_periodHistoryIndex] = period;
    _periodHistoryIndex = (_periodHistoryIndex + 1) % PERIOD_HISTORY_SIZE;
    if (_periodHistoryCount < PERIOD_HISTORY_SIZE) _periodHistoryCount++;
    calculatePeriodVariance();
}

void SchmittTriggerFrequencyDetector::calculatePeriodVariance() {
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

float SchmittTriggerFrequencyDetector::update(
  float signalValue, float signalMagnitude, float debounceTime, float steepnessTime, float dt) {
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
        float thisCrossingTime = _timeInCycle - timeSinceLow / 2.0f; // or (_timeInCycle + _lastLowTime) / 2.0f

        if (scaledValue > _upperThreshold && (timeSinceLow > steepnessTime)
            && (_crossingsCounter == 0 || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {  // found crossing
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
            _lastCrossingInCycleTime = - timeSinceLow / 2.0f; // negative
            _beginningCrossingInCycleTime = _lastCrossingInCycleTime; // negative
            _timeInCycle = 0.0f;
            _lastHighTime = _timeInCycle;
          }
        } else if (scaledValue < _lowerThreshold) {
          // still LOW
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
        float thisCrossingTime = _timeInCycle - timeSinceHigh / 2.0f; // or (_timeInCycle + _lastHighTime) / 2.0f

        if (scaledValue < _lowerThreshold && (timeSinceHigh > steepnessTime)
            && (_crossingsCounter == 0 || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {  // found crossing
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
            _lastCrossingInCycleTime = - timeSinceHigh / 2.0f; // negative
            _beginningCrossingInCycleTime = _lastCrossingInCycleTime; // negative
            _timeInCycle = 0.0f;
            _lastLowTime = _timeInCycle;
          }
        } else if (scaledValue > _upperThreshold) {
          // still HIGH
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

float SchmittTriggerFrequencyDetector::getFrequency() const {
  return _frequency;
}

SchmittTriggerFrequencyDetector::QualityMetrics SchmittTriggerFrequencyDetector::getQualityMetrics() const {
    QualityMetrics metrics;
    
    // Confidence based on variance and number of samples
    if (_periodHistoryCount < 2) {
        metrics.confidence = 0.1f;  // Minimal confidence with few samples
    } else {
        // Confidence decreases with higher variance and increases with more samples
        float stddev = sqrtf(_periodVariance);
        float normalizedStddev = stddev / _lastPeriodEstimate;  // Relative standard deviation
        metrics.confidence = std::max(0.0f, std::min(1.0f, 1.0f - normalizedStddev));
    }
    
    // Jitter is the standard deviation of period measurements
    metrics.jitter = sqrtf(_periodVariance);
    
    // Signal-to-threshold ratio
    metrics.amplitude_ratio = _amplitudeRatio;
    
    // Whether we're using fallback frequency
    metrics.is_fallback = _isFallback;
    
    return metrics;
}

void SchmittTriggerFrequencyDetector::reset() {
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

#endif
