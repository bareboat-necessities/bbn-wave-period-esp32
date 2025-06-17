#ifndef FOURTH_ORDER_LOWPASS_H
#define FOURTH_ORDER_LOWPASS_H

#include <math.h>
#include <limits>

/*
  Cutoff frequency needs to be less than sampleFrequency / 2 (Nyquist limit)
*/
class FourthOrderLowPass {
public:
    FourthOrderLowPass(float cutoffFreq);
    float process(float input, float deltaTime);
    void setCutoffFrequency(float cutoffFreq);
    void reset();

private:
    struct Biquad {
        float a0, a1, a2; // Denominator coefficients
        float b0, b1, b2; // Numerator coefficients
        float x1, x2, y1, y2; // State variables (direct form II)
    };

    Biquad stage1, stage2;
    float cutoffFreq;
    float lastSampleRate;
    bool coeffsDirty;
    
    void calculateCoefficients(float sampleRate);
};

FourthOrderLowPass::FourthOrderLowPass(float cutoffFreq) 
    : cutoffFreq(cutoffFreq), lastSampleRate(0), coeffsDirty(true) {
    reset();
}

void FourthOrderLowPass::reset() {
    stage1.x1 = stage1.x2 = stage1.y1 = stage1.y2 = 0.0f;
    stage2.x1 = stage2.x2 = stage2.y1 = stage2.y2 = 0.0f;
}

void FourthOrderLowPass::setCutoffFrequency(float cutoffFreq) {
    this->cutoffFreq = cutoffFreq;
    coeffsDirty = true;
}

void FourthOrderLowPass::calculateCoefficients(float sampleRate) {
    // Only recalculate if needed
    if (!coeffsDirty && fabs(sampleRate - lastSampleRate) < std::numeric_limits<float>::epsilon()) {
        return;
    }
    
    coeffsDirty = false;
    lastSampleRate = sampleRate;
    
    // Validate parameters
    if (cutoffFreq <= 0 || sampleRate <= 0) return;
    if (cutoffFreq >= sampleRate/2) {
        cutoffFreq = sampleRate/2 * 0.99f; // Force below Nyquist
    }

    // 4th-order Butterworth filter parameters
    const float sqrt2 = sqrt(2.0f);
    const float q1 = 1.0f / (2.0f * cos(5.0f * M_PI / 8.0f));  // Q ≈ 0.541
    const float q2 = 1.0f / (2.0f * cos(7.0f * M_PI / 8.0f));  // Q ≈ 1.306
    
    // Pre-warping for bilinear transform
    const float omega = 2.0f * M_PI * cutoffFreq;
    const float T = 1.0f / sampleRate;
    const float warped = (2.0f/T) * tan(omega * T/2.0f);
    
    // Calculate coefficients for each stage
    auto calculateStage = [&](Biquad& stage, float q) {
        const float alpha = sin(warped * T/2.0f) / (2.0f * q);
        const float cosw0 = cos(warped * T/2.0f);
        
        const float a0 = 1.0f + alpha;
        
        stage.b0 = ((1.0f - cosw0)/2.0f) / a0;
        stage.b1 = (1.0f - cosw0) / a0;
        stage.b2 = stage.b0;
        stage.a0 = 1.0f;
        stage.a1 = (-2.0f * cosw0) / a0;
        stage.a2 = (1.0f - alpha) / a0;
    };
    
    calculateStage(stage1, q1);
    calculateStage(stage2, q2);
}

float FourthOrderLowPass::process(float input, float deltaTime) {
    if (deltaTime <= 0) return input; // skip invalid timesteps
    
    float currentSampleRate = 1.0f / deltaTime;
    calculateCoefficients(currentSampleRate);
    
    // Process first stage (direct form II)
    float y = stage1.b0 * input + stage1.b1 * stage1.x1 + stage1.b2 * stage1.x2
            - stage1.a1 * stage1.y1 - stage1.a2 * stage1.y2;
    stage1.x2 = stage1.x1;
    stage1.x1 = input;
    stage1.y2 = stage1.y1;
    stage1.y1 = y;
    
    // Process second stage
    float output = stage2.b0 * y + stage2.b1 * stage2.x1 + stage2.b2 * stage2.x2
                 - stage2.a1 * stage2.y1 - stage2.a2 * stage2.y2;
    stage2.x2 = stage2.x1;
    stage2.x1 = y;
    stage2.y2 = stage2.y1;
    stage2.y1 = output;
    
    return output;
}

#endif
