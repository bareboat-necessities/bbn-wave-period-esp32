#ifndef TIME_AWARE_BANDPASS_FILTER_H
#define TIME_AWARE_BANDPASS_FILTER_H

#include <cmath>  // For sin, cos, sinh, log

class TimeAwareBandpassFilter {
private:
    // Filter coefficients (standard naming convention)
    float b0, b1, b2;  // Numerator (feedforward) coefficients
    float a1, a2;      // Denominator (feedback) coefficients

    // Filter state variables
    float w1, w2;

    // Filter parameters
    float centerFreq;
    float bandwidth;

    // Last sample time (microseconds)
    unsigned long lastSampleTime;

    // Calculate coefficients for given delta time (in seconds)
    void updateCoefficients(float deltaTimeSec) {
        if (deltaTimeSec <= 1e-6f) {  // Prevent division by zero and handle very small deltas
            deltaTimeSec = 1e-6f;
        }

        float effectiveSampleRate = 1.0f / deltaTimeSec;
        float omega = 2.0f * M_PI * centerFreq / effectiveSampleRate;
        float sinOmega = sin(omega);
        float alpha = sinOmega * sinh(log(2.0f) / 2.0f * bandwidth * omega / sinOmega);

        // Raw coefficients
        float raw_b0 = alpha;
        float raw_b1 = 0.0f;  // b1 is always 0 for bandpass
        float raw_b2 = -alpha;
        float raw_a0 = 1.0f + alpha;
        float raw_a1 = -2.0f * cos(omega);
        float raw_a2 = 1.0f - alpha;

        // Normalize coefficients (divide all by a0)
        float inv_a0 = 1.0f / raw_a0;
        this->b0 = raw_b0 * inv_a0;
        this->b1 = raw_b1 * inv_a0;  
        this->b2 = raw_b2 * inv_a0;
        this->a1 = raw_a1 * inv_a0;
        this->a2 = raw_a2 * inv_a0;
    }

public:
    // Constructor
    TimeAwareBandpassFilter(float centerFreq, float bandwidth, unsigned long startTimeMicros)
        : centerFreq(centerFreq), 
          bandwidth(bandwidth), 
          lastSampleTime(startTimeMicros), 
          w1(0), w2(0),
          b0(0), b1(0), b2(0), a1(0), a2(0) {
    }

    // Process a single sample with provided time delta (in seconds)
    float processWithDelta(float input, float deltaTimeSec) {
        updateCoefficients(deltaTimeSec);

        // Direct Form II implementation
        float w0 = input - a1 * w1 - a2 * w2;
        float output = b0 * w0 + b1 * w1 + b2 * w2;

        // Update state variables
        w2 = w1;
        w1 = w0;

        return output;
    }

    // Process a single sample with automatic time calculation (microseconds)
    float process(float input, unsigned long currentTimeMicros) {
        float deltaTimeSec = (currentTimeMicros - lastSampleTime) / 1e6f;
        lastSampleTime = currentTimeMicros;
        return processWithDelta(input, deltaTimeSec);
    }

    // Update filter parameters
    void setParameters(float newCenterFreq, float newBandwidth) {
        centerFreq = newCenterFreq;
        bandwidth = newBandwidth;
    }

    // Reset filter state
    void reset() {
        w1 = w2 = 0.0f;
    }

    // Reset filter state with new time
    void reset(unsigned long newTimeMicros) {
        w1 = w2 = 0.0f;
        lastSampleTime = newTimeMicros;
    }
};

#endif
