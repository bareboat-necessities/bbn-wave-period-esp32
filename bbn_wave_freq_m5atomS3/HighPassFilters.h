#ifndef HighPassFilters_h
#define HighPassFilters_h

class HighPassFirstOrderFilter {
private:
  float timeConstant; // Time constant (tau = 1/(2*pi*fc)) in seconds
  float prevInput;
  float prevOutput;
  
public:
  // Constructor takes cutoff PERIOD in seconds (1/frequency)
  DeltaTimeHighPassFilter(float cutoffPeriod) {
    // Convert period to time constant (tau = RC = 1/(2*pi*fc) = period/(2*pi))
    timeConstant = cutoffPeriod / (2 * PI);
    reset();
  }
  
  void reset() {
    prevInput = 0;
    prevOutput = 0;
  }
  
  // Alternative update where you provide deltaTime yourself
  float update(float input, float deltaTime) {
    // Calculate alpha for this step
    float alpha = timeConstant / (timeConstant + deltaTime);
    
    // Apply high-pass filter formula
    float output = alpha * (prevOutput + input - prevInput);
    
    // Store current values for next iteration
    prevInput = input;
    prevOutput = output;
    
    return output;
  }
};

#endif
