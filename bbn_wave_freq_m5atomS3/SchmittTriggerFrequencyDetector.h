#ifndef SCHMITT_TRIGGER_FREQ_DETECTOR_H
#define SCHMITT_TRIGGER_FREQ_DETECTOR_H

class SchmittTriggerFrequencyDetector {
public:
  // Constructor: sets hysteresis threshold (default: 0.1)
  SchmittTriggerFrequencyDetector(float hysteresis = 0.1f);

  // Update with new signal sample and time since last update (dt in seconds)
  // Returns frequency (Hz), or 0 if no crossings detected yet
  float update(float signalValue, float dt);

  // Get latest computed frequency (Hz)
  float getFrequency() const;

  // Reset the detector (clears history)
  void reset();

private:
  float _hysteresis;       // Hysteresis threshold (e.g., 0.1 for 10% of amplitude)
  float _upperThreshold;   // Upper threshold (must cross this first)
  float _lowerThreshold;   // Lower threshold (must cross this next)
  bool _wasAboveUpper;     // Tracks if signal was last above upper threshold
  float _lastCrossingTime; // Time (in seconds) of the last zero-crossing
  float _frequency;        // Latest frequency estimate (Hz)
};

SchmittTriggerFrequencyDetector::SchmittTriggerFrequencyDetector(float hysteresis) 
  : _hysteresis(hysteresis),
    _upperThreshold(hysteresis),
    _lowerThreshold(-hysteresis),
    _wasAboveUpper(false),
    _lastCrossingTime(0.0f),
    _frequency(0.0f) {}

float SchmittTriggerFrequencyDetector::update(float signalValue, float dt) {
  // Schmitt Trigger logic
  if (_wasAboveUpper) {
    if (signalValue < _lowerThreshold) {
      // Zero-crossing detected (high → low transition)
      _wasAboveUpper = false;
      
      // Compute time since last crossing (skip first crossing)
      if (_lastCrossingTime > 0.0f) {
        float crossingInterval = dt; // Time since last update
        _frequency = 1.0f / (2.0f * crossingInterval); // Frequency = 1/(2*T)
      }
      _lastCrossingTime = 0.0f; // Reset for next half-cycle
    }
  } else {
    if (signalValue > _upperThreshold) {
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
  _frequency = 0.0f;
}

#endif
