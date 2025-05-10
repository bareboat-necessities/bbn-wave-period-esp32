#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

#define SCHMITT_TRIGGER_FREQ_INIT     1e-7f
#define SCHMITT_TRIGGER_FALLBACK_FREQ 1e-2f
#define SCHMITT_TRIGGER_FALLBACK_TIME 60.0f

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/
class SchmittTriggerFrequencyDetector {
  public:
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

    // Reset the detector (clears history)
    void reset();

  private:
    enum class State {
      WAS_NOT_SET,        // Initial undefined
      WAS_LOW,            // Below lower threshold
      WAS_HIGH,           // Above upper threshold
    };

    float _hysteresis;            // Hysteresis threshold
    float _upperThreshold;        // Upper threshold
    float _lowerThreshold;        // Lower threshold
    State _state;                 // Tracks states
    float _frequency;             // Latest frequency estimate (Hz)
    float _fallbackToLowFreqTime; // Time to fallback if no crossing detected  

    unsigned int _periodsInCycle;
    float _timeInCycle;
    float _lastLowTime;
    float _lastHighTime;
    float _lastCrossingInCycleTime;
    float _beginningCrossingInCycleTime;
    unsigned int _crossingsCounter;

    bool doNotDebounce();
};

SchmittTriggerFrequencyDetector::SchmittTriggerFrequencyDetector(
  float hysteresis, unsigned int periodsInCycle, float fallbackToLowFreqTime)
  : _hysteresis(fabs(hysteresis)),
    _upperThreshold(_hysteresis),
    _lowerThreshold(-_hysteresis),
    _state(State::WAS_NOT_SET),
    _frequency(SCHMITT_TRIGGER_FREQ_INIT),
    _periodsInCycle(periodsInCycle),
    _timeInCycle(0.0f),
    _lastLowTime(0.0f),
    _lastHighTime(0.0f),
    _lastCrossingInCycleTime(0.0f),
    _beginningCrossingInCycleTime(0.0f),
    _fallbackToLowFreqTime(SCHMITT_TRIGGER_FALLBACK_TIME),
    _crossingsCounter(0)
{}

float SchmittTriggerFrequencyDetector::update(
  float signalValue, float signalMagnitude, float debounceTime, float steepnessTime, float dt) {
  if (dt <= 0.0f || fabs(signalMagnitude) <= 0.0f) {
    return _frequency;
  }
  const float scaledValue = signalValue / fabs(signalMagnitude);

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
      }
      break;

    case State::WAS_LOW: {
        _timeInCycle += dt;
        float timeSinceLow = _timeInCycle - _lastLowTime;
        float thisCrossingTime = _timeInCycle - timeSinceLow / 2.0f; // or (_timeInCycle + _lastLowTime) / 2.0f

        if (scaledValue > _upperThreshold && (doNotDebounce() || timeSinceLow > steepnessTime)
            && (doNotDebounce() || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {  // found crossing
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
            _frequency = 1.0 / period;
          }
          if (_crossingsCounter == (2 * _periodsInCycle + 1)) {
            float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
            float period = 2.0f * cycleTime / (_crossingsCounter - 1);
            _frequency = 1.0 / period;
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
        }
      }
      break;

    case State::WAS_HIGH: {
        _timeInCycle += dt;
        float timeSinceHigh = _timeInCycle - _lastHighTime;
        float thisCrossingTime = _timeInCycle - timeSinceHigh / 2.0f; // or (_timeInCycle + _lastHighTime) / 2.0f

        if (scaledValue < _lowerThreshold && (doNotDebounce() || timeSinceHigh > steepnessTime)
            && (doNotDebounce() || (thisCrossingTime - _lastCrossingInCycleTime) > debounceTime)) {  // found crossing
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
            _frequency = 1.0 / period;
          }
          if (_crossingsCounter == (2 * _periodsInCycle + 1)) {
            float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
            float period = 2.0f * cycleTime / (_crossingsCounter - 1);
            _frequency = 1.0 / period;
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
        }
      }
      break;

    default:
      _timeInCycle += dt;
  }

  return _frequency;
}

bool SchmittTriggerFrequencyDetector::doNotDebouce() {
  return _crossingsCounter == 0 || _frequency == SCHMITT_TRIGGER_FREQ_INIT || _frequency == SCHMITT_TRIGGER_FALLBACK_FREQ;
}

float SchmittTriggerFrequencyDetector::getFrequency() const {
  return _frequency;
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
}


#endif
