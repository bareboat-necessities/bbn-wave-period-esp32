#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

class SchmittTriggerFrequencyDetector {
public:
  // Constructor: sets hysteresis threshold (default: 0.1) and debounce time (default: 0.01s)
  // hysteresis should be positive and typically between 0.01 and 0.5
  // debounceTime should be positive and represents minimum time between valid transitions
  explicit SchmittTriggerFrequencyDetector(float hysteresis = 0.1f, float debounceTime = 0.01f);

  // Update with new signal sample and time since last update (dt in seconds)
  // Returns frequency (Hz), or 0 if no complete cycle detected yet
  // signalMagnitude should be positive (absolute amplitude of the signal)
  float update(float signalValue, float signalMagnitude, float dt);

  // Get latest computed frequency (Hz)
  float getFrequency() const;

  // Reset the detector (clears history)
  void reset();

  // Set new debounce time (seconds)
  void setDebounceTime(float debounceTime);

private:
  enum class State {
        LOW,            // Below lower threshold
        HIGH,           // Above upper threshold
        RISING_EDGE,    // Between thresholds, coming from low
        FALLING_EDGE    // Between thresholds, coming from high
  };

  float _hysteresis;       // Hysteresis threshold
  float _upperThreshold;   // Upper threshold
  float _lowerThreshold;   // Lower threshold
  float _debounceTime;     // Minimum time between valid transitions (seconds)
  float _debounceCounter;  // Time accumulated since last valid transition
  State _state;            // Tracks states
  float _lastCrossingTime; // Accumulated time since the last zero-crossing
  float _frequency;        // Latest frequency estimate (Hz)
  bool _hasCompleteCycle;  // Tracks if we've completed at least one full cycle
};

SchmittTriggerFrequencyDetector::SchmittTriggerFrequencyDetector(float hysteresis, float debounceTime) 
    : _hysteresis(fabs(hysteresis)),
    _upperThreshold(_hysteresis),
    _lowerThreshold(-_hysteresis),
    _debounceTime(std::max(0.0f, debounceTime)),
    _debounceCounter(0.0f),
    _state(State::LOW),
    _lastCrossingTime(0.0f),
    _frequency(0.0f),
    _hasCompleteCycle(false) {}


float SchmittTriggerFrequencyDetector::update(float signalValue, float signalMagnitude, float dt) {
  if (dt <= 0.0f || signalMagnitude <= 0.0f) {
    return _frequency;  // Invalid input, return last known frequency
  }

  // Update debounce counter if we're in a pending state
  if (_transitionPending) {
    _debounceCounter += dt;
    if (_debounceCounter >= _debounceTime) {
      _transitionPending = false;
      _debounceCounter = 0.0f;
    } else {
      _lastCrossingTime += dt;
      return _frequency;  // Still in debounce period
    }
  }

  // Schmitt Trigger logic
  if (_wasAboveUpper) {
    if (signalValue < _lowerThreshold * signalMagnitude) {
      // Potential zero-crossing detected (high → low transition)
      if (!_transitionPending && _debounceTime > 0.0f) {
        _transitionPending = true;
        _debounceCounter = 0.0f;
        _lastCrossingTime += dt;
        return _frequency;
      }
      
      _wasAboveUpper = false;
      
      // Compute frequency if we've completed at least one full cycle
      if (_hasCompleteCycle) {
        _frequency = 1.0f / (2.0f * _lastCrossingTime);
      } else {
        _hasCompleteCycle = true;
      }
      _lastCrossingTime = 0.0f; // Reset for next half-cycle
    }
  } else {
    if (signalValue > _upperThreshold * signalMagnitude) {
      // Potential transition to high state
      if (!_transitionPending && _debounceTime > 0.0f) {
        _transitionPending = true;
        _debounceCounter = 0.0f;
        _lastCrossingTime += dt;
        return _frequency;
      }
      
      _wasAboveUpper = true;
    }
  }

  // Accumulate time since last crossing
  _lastCrossingTime += dt;

  return _frequency;
}

float SchmittTriggerFrequencyDetector::getFrequency() const {
  return _frequency;
}

void SchmittTriggerFrequencyDetector::reset() {
  _wasAboveUpper = false;
  _lastCrossingTime = 0.0f;
  _frequency = 1.0f;
  _hasCompleteCycle = false;
  _transitionPending = false;
  _debounceCounter = 0.0f;
}

void SchmittTriggerFrequencyDetector::setDebounceTime(float debounceTime) {
  _debounceTime = std::max(0.0f, debounceTime); // Ensure non-negative
}

#endif
