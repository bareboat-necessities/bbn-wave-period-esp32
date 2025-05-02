#ifndef TIME_AWARE_BANDPASS_FILTER_H
#define TIME_AWARE_BANDPASS_FILTER_H

class TimeAwareBandpassFilter {
  private:
    // Filter coefficients (current)
    float a0, a1, a2, b1, b2;

    // Filter state variables
    float w1, w2;

    // Filter parameters
    float centerFreq;
    float bandwidth;

    // Last sample time (microseconds)
    unsigned long lastSampleTime;

    // Calculate coefficients for given delta time (in seconds)
    void updateCoefficients(float deltaTimeSec) {
      if (deltaTimeSec <= 0) return; // Prevent division by zero

      float effectiveSampleRate = 1.0 / deltaTimeSec;
      float omega = 2.0 * PI * centerFreq / effectiveSampleRate;
      float alpha = sin(omega) * sinh(log(2.0) / 2.0 * bandwidth * omega / sin(omega));

      float b0 = alpha;
      float b1 = 0.0;
      float b2 = -alpha;
      float a0_temp = 1.0 + alpha;
      float a1_temp = -2.0 * cos(omega);
      float a2_temp = 1.0 - alpha;

      // Normalize coefficients
      b0 /= a0_temp;
      b1 /= a0_temp;
      b2 /= a0_temp;
      a1_temp /= a0_temp;
      a2_temp /= a0_temp;

      this->a0 = b0;
      this->a1 = b1;
      this->a2 = b2;
      this->b1 = a1_temp;
      this->b2 = a2_temp;
    }

  public:
    // Constructor
    TimeAwareBandpassFilter(float centerFreq, float bandwidth, unsigned long startTimeMicros)
      : centerFreq(centerFreq), bandwidth(bandwidth), lastSampleTime(startTimeMicros), w1(0), w2(0) {
    }

    // Process a single sample with provided time delta (in seconds)
    float processWithDelta(float input, float deltaTimeSec) {
      // Update coefficients based on actual time step
      updateCoefficients(deltaTimeSec);

      // Direct Form II implementation
      float w0 = input - b1 * w1 - b2 * w2;
      float output = a0 * w0 + a1 * w1 + a2 * w2;

      // Update state variables
      w2 = w1;
      w1 = w0;

      return output;
    }

    // Update filter parameters
    void setParameters(float newCenterFreq, float newBandwidth) {
      centerFreq = newCenterFreq;
      bandwidth = newBandwidth;
    }
};

#endif
