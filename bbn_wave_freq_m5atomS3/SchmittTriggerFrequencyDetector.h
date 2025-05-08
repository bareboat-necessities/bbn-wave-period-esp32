#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

#define SCHMITT_TRIGGER_FREQ_INIT  1e-7f

/*
   Copyright 2025, Mikhail Grushinskiy

   Zero crossing frequency detector with hysteresis and debouncing
*/
class SchmittTriggerFrequencyDetector {
  public:
    // Constructor: sets hysteresis threshold (default: 0.1)
    // hysteresis must be positive and typically between 0.01 and 0.5
    // halfPeriodsInCycle must be positive integer
    explicit SchmittTriggerFrequencyDetector(float hysteresis = 0.1f, unsigned int halfPeriodsInCycle = 2);

    // Update with new signal sample and time since last update (dt in seconds)
    // Returns frequency (Hz)
    // signalMagnitude must be positive (absolute amplitude of the signal)
    // debounceTime (in seconds) must be positive
    float update(float signalValue, float signalMagnitude, float debounceTime, float dt);

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

    float _hysteresis;       // Hysteresis threshold
    float _upperThreshold;   // Upper threshold
    float _lowerThreshold;   // Lower threshold
    State _state;            // Tracks states
    float _frequency;        // Latest frequency estimate (Hz)
    unsigned int _halfPeriodsInCycle;

    float _timeInCycle;
    float _lastLowTime;
    float _lastHighTime;
    float _beginningCrossingInCycleTime;
    unsigned int _crossingsCounter;
};

SchmittTriggerFrequencyDetector::SchmittTriggerFrequencyDetector(float hysteresis, unsigned int halfPeriodsInCycle)
  : _hysteresis(fabs(hysteresis)),
    _upperThreshold(_hysteresis),
    _lowerThreshold(-_hysteresis),
    _state(State::WAS_NOT_SET),
    _frequency(SCHMITT_TRIGGER_FREQ_INIT),
    _halfPeriodsInCycle(halfPeriodsInCycle),
    _timeInCycle(0.0f),
    _lastLowTime(0.0f),
    _lastHighTime(0.0f),
    _beginningCrossingInCycleTime(0.0f),
    _crossingsCounter(0)
{}

float SchmittTriggerFrequencyDetector::update(float signalValue, float signalMagnitude, float debounceTime, float dt) {
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
        _beginningCrossingInCycleTime = 0;
        _timeInCycle = 0.0f;
      }
      break;

    case State::WAS_LOW: {
        _timeInCycle += dt;
        float timeSinceLow = _timeInCycle - _lastLowTime;
        if (scaledValue > _upperThreshold && timeSinceLow > debounceTime) {  // found crossing
          _state = State::WAS_HIGH;
          _lastHighTime = _timeInCycle;
          float thisCrossingTime = _timeInCycle - timeSinceLow / 2.0f;
          if (_crossingsCounter == 0) {
            _beginningCrossingInCycleTime = thisCrossingTime;
          }
          _crossingsCounter++;
          if (_crossingsCounter == (_halfPeriodsInCycle + 1)) {
            float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
            float period = 2.0f * cycleTime / _halfPeriodsInCycle;
            _frequency = 1.0 / period;
            _crossingsCounter = 0;
            _timeInCycle = 0.0f;
            _lastHighTime = _timeInCycle;
          }
        } else if (scaledValue < _lowerThreshold) {
          // still LOW
          _lastLowTime = _timeInCycle;
        }
      }
      break;

    case State::WAS_HIGH: {
        _timeInCycle += dt;
        float timeSinceHigh = _timeInCycle - _lastHighTime;
        if (scaledValue < _lowerThreshold && timeSinceHigh > debounceTime) {  // found crossing
          _state = State::WAS_LOW;
          _lastLowTime = _timeInCycle;
          float thisCrossingTime = _timeInCycle - timeSinceHigh / 2.0f;
          if (_crossingsCounter == 0) {
            _beginningCrossingInCycleTime = thisCrossingTime;
          }
          _crossingsCounter++;
          if (_crossingsCounter == (_halfPeriodsInCycle + 1)) {
            float cycleTime = thisCrossingTime - _beginningCrossingInCycleTime;
            float period = 2.0f * cycleTime / _halfPeriodsInCycle;
            _frequency = 1.0 / period;
            _crossingsCounter = 0;
            _timeInCycle = 0.0f;
            _lastLowTime = _timeInCycle;
          }
        } else if (scaledValue > _upperThreshold) {
          // still HIGH
          _lastHighTime = _timeInCycle;
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

void SchmittTriggerFrequencyDetector::reset() {
  _state = State::WAS_NOT_SET;
  _frequency = SCHMITT_TRIGGER_FREQ_INIT;
  _halfPeriodsInCycle = 2;
  _timeInCycle = 0.0f;
  _lastLowTime = 0.0f;
  _lastHighTime = 0.0f;
  _beginningCrossingInCycleTime = 0.0f;
  _crossingsCounter = 0;
}


#endif
