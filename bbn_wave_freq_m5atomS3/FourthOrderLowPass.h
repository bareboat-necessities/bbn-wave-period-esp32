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
        
        Biquad() : a0(1), a1(0), a2(0), b0(1), b1(0), b2(0), x1(0), x2(0), y1(0), y2(0) {}
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
    if (cutoffFreq <= 0 || sampleRate <= 0) {
        // Default to passthrough if invalid
        stage1.b0 = 1.0f; stage1.b1 = 0.0f; stage1.b2 = 0.0f;
        stage1.a0 = 1.0f; stage1.a1 = 0.0f; stage1.a2 = 0.0f;
        stage2 = stage1;
        return;
    }
    
    if (cutoffFreq >= sampleRate/2) {
        cutoffFreq = sampleRate/2 * 0.99f; // Force below Nyquist
    }

    // 4th-order Butterworth filter parameters
    const float q1 = 0.54119610f;  // 1/(2*cos(5π/8))
    const float q2 = 1.3065630f;   // 1/(2*cos(7π/8))
    
    // Pre-warping for bilinear transform
    const float omega = 2.0f * M_PI * cutoffFreq;
    const float T = 1.0f / sampleRate;
    const float warped = (2.0f/T) * tan(omega * T/2.0f);
    
    // Calculate coefficients for each stage
    auto calculateStage = [&](Biquad& stage, float q) {
        const float alpha = sin(warped * T/2.0f) / (2.0f * q);
        const float cosw0 = cos(warped * T/2.0f);
        
        const float a0 = 1.0f + alpha;
        
        // Protect against division by zero
        if (fabs(a0) < std::numeric_limits<float>::epsilon()) {
            stage.b0 = 1.0f; stage.b1 = 0.0f; stage.b2 = 0.0f;
            stage.a0 = 1.0f; stage.a1 = 0.0f; stage.a2 = 0.0f;
            return;
        }
        
        const float inv_a0 = 1.0f / a0;
        
        stage.b0 = ((1.0f - cosw0)/2.0f) * inv_a0;
        stage.b1 = (1.0f - cosw0) * inv_a0;
        stage.b2 = stage.b0;
        stage.a0 = 1.0f;
        stage.a1 = (-2.0f * cosw0) * inv_a0;
        stage.a2 = (1.0f - alpha) * inv_a0;
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
    
    // Prevent NaN propagation
    if (!std::isfinite(output)) {
        output = 0.0f;
        reset();
    }
    
    return output;
}

#endif
