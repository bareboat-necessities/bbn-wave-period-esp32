#ifndef HighPassFilters_h
#define HighPassFilters_h

class HighPassFirstOrderFilter {
private:
  float timeConstant; // Time constant (tau = 1/(2*pi*fc)) in seconds
  float prevInput;
  float prevOutput;
  
public:
  // Constructor takes cutoff PERIOD in seconds (1/frequency)
  HighPassFirstOrderFilter(float cutoffPeriod) {
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

class HighPassSecondOrderFilter {
private:
  // Filter coefficients
  float a0, a1, a2;
  float b1, b2;
  
  // Previous values
  float x1 = 0, x2 = 0; // Input history (n-1, n-2)
  float y1 = 0, y2 = 0; // Output history (n-1, n-2)
  
  float cutoffPeriod; // Stored cutoff period (seconds)
  
  // Recalculate coefficients based on current cutoff and sample time
  void calculateCoefficients(float deltaTime) {
    
    float omega = (2 * PI) / cutoffPeriod; // Convert period to angular frequency
    float T = deltaTime;
    
    // Bilinear transform with pre-warping
    float omega_prewarp = 2/T * tan(omega * T/2);
    
    // Second-order Butterworth coefficients
    const float sqrt2 = sqrt(2);
    float alpha = (2/T) * (2/T) + omega_prewarp * omega_prewarp + (2/T) * omega_prewarp * sqrt2;
    
    a0 = (2/T) * (2/T) / alpha;
    a1 = -2 * a0;
    a2 = a0;
    
    b1 = (2 * ((2/T) * (2/T) - omega_prewarp * omega_prewarp)) / alpha;
    b2 = ((2/T) * (2/T) + omega_prewarp * omega_prewarp - (2/T) * omega_prewarp * sqrt2) / alpha;
  }

public:
  // Constructor takes cutoff period in seconds
  HighPassSecondOrderFilter(float period) : cutoffPeriod(fabs(period)) {
    reset();
  }
  
  // Reset filter state
  void reset() {
    x1 = x2 = 0;
    y1 = y2 = 0;
  }
  
  // Update filter with provided delta-time
  float update(float input, float deltaTime) {
    calculateCoefficients(deltaTime);
    
    // Apply difference equation
    float output = a0 * input + a1 * x1 + a2 * x2 - b1 * y1 - b2 * y2;
    
    // Update state variables
    x2 = x1;
    x1 = input;
    y2 = y1;
    y1 = output;
    
    return output;
  }
  
  // Change cutoff period (in seconds)
  void setCutoffPeriod(float period) {
    cutoffPeriod = fabs(period);
  }
};

#endif
