#ifndef FOURTH_ORDER_LOWPASS_H
#define FOURTH_ORDER_LOWPASS_H

#include <math.h>

/*
  Cutoff frequency needs to be less than sampleFrequency/5
*/
class FourthOrderLowPass {
public:
    FourthOrderLowPass(float cutoffFreq);
    float process(float input, float deltaTime);
    void setCutoffFrequency(float cutoffFreq);
    void reset();

private:
    struct Biquad {
        float a1, a2; // Denominator coefficients
        float b0, b1, b2; // Numerator coefficients
        float v1, v2; // State variables
    };

    Biquad stage1, stage2;
    float cutoffFreq;
    
    void calculateCoefficients(float deltaTime);
};

FourthOrderLowPass::FourthOrderLowPassDT(float cutoffFreq) {
    this->cutoffFreq = cutoffFreq;
    reset();
}

void FourthOrderLowPass::reset() {
    stage1.v1 = stage1.v2 = 0.0f;
    stage2.v1 = stage2.v2 = 0.0f;
}

void FourthOrderLowPass::setCutoffFrequency(float cutoffFreq) {
    this->cutoffFreq = cutoffFreq;
}

void FourthOrderLowPass::calculateCoefficients(float deltaTime) {
    // Design a 4th-order Butterworth low-pass filter
    // Using bilinear transform with pre-warping
    
    // Only proceed if we have a valid delta time
    if (deltaTime <= 0) return;
    
    float sampleRate = 1.0f / deltaTime;
    
    // First compute the analog prototype frequencies
    float omega = 2.0f * PI * cutoffFreq;
    float T = deltaTime;
    float warped = (2.0f/T) * tan(omega * T/2.0f);
    
    // Butterworth coefficients (2nd order)
    float sqrt2 = sqrt(2.0f);
    float k = warped / sqrt2;
    float a1_analog = 2.0f * k;
    float a2_analog = k * k;
    
    // Bilinear transform
    float C = 2.0f * sampleRate;
    float D = (warped*warped) + a1_analog * warped * C + a2_analog * C*C;
    
    // First stage coefficients
    stage1.b0 = (warped*warped) / D;
    stage1.b1 = 2.0f * stage1.b0;
    stage1.b2 = stage1.b0;
    stage1.a1 = (2.0f*warped*warped - 2.0f*a2_analog*C*C) / D;
    stage1.a2 = (warped*warped - a1_analog*warped*C + a2_analog*C*C) / D;
    
    // Second stage (same coefficients for Butterworth)
    stage2.b0 = stage1.b0;
    stage2.b1 = stage1.b1;
    stage2.b2 = stage1.b2;
    stage2.a1 = stage1.a1;
    stage2.a2 = stage1.a2;
}

float FourthOrderLowPass::process(float input, float deltaTime) {
    // Update coefficients based on current delta time
    calculateCoefficients(deltaTime);
    
    // Process first biquad stage
    float v = input - stage1.a1 * stage1.v1 - stage1.a2 * stage1.v2;
    float output = stage1.b0 * v + stage1.b1 * stage1.v1 + stage1.b2 * stage1.v2;
    stage1.v2 = stage1.v1;
    stage1.v1 = v;
    
    // Process second biquad stage
    v = output - stage2.a1 * stage2.v1 - stage2.a2 * stage2.v2;
    output = stage2.b0 * v + stage2.b1 * stage2.v1 + stage2.b2 * stage2.v2;
    stage2.v2 = stage2.v1;
    stage2.v1 = v;
    
    return output;
}

#endif
