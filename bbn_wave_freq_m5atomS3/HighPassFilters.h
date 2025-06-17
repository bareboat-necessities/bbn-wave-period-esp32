#ifndef HighPassFilters_h
#define HighPassFilters_h

#include <cmath>  // For PI, sqrt, tan, fabs

class HighPassFirstOrderFilter {
private:
    float timeConstant;  // τ = 1/(2πf_c) = cutoffPeriod / (2π)
    float prevInput = 0;
    float prevOutput = 0;

public:
    // Constructor: cutoffPeriod = 1/f_c (seconds)
    HighPassFirstOrderFilter(float cutoffPeriod) {
        timeConstant = cutoffPeriod / (2 * M_PI);
        reset();
    }

    // Reset filter state
    void reset() {
        prevInput = 0;
        prevOutput = 0;
    }

    // Update filter (deltaTime = time since last update in seconds)
    float update(float input, float deltaTime) {
        // Compute alpha = τ / (τ + Δt)
        float alpha = timeConstant / (timeConstant + deltaTime);

        // Correct 1st-order high-pass difference equation
        float output = alpha * (prevOutput + input - prevInput);

        // Update state
        prevInput = input;
        prevOutput = output;

        return output;
    }
};

class HighPassSecondOrderFilter {
private:
    // Filter coefficients
    float a0 = 0, a1 = 0, a2 = 0;  // Feedforward (numerator)
    float b1 = 0, b2 = 0;          // Feedback (denominator)

    // State variables
    float x1 = 0, x2 = 0;  // Previous inputs (n-1, n-2)
    float y1 = 0, y2 = 0;  // Previous outputs (n-1, n-2)

    float cutoffPeriod;  // Cutoff period (1/f_c) in seconds

    // Recompute coefficients using bilinear transform + pre-warping
    void calculateCoefficients(float deltaTime) {
        float T = deltaTime;
        float omega_c = (2 * M_PI) / cutoffPeriod;  // Cutoff frequency (rad/s)
        float omega_prewarp = (2.0f / T) * tan(omega_c * T / 2.0f);  // Pre-warping

        // Butterworth Q = 1/√2 (maximally flat)
        const float sqrt2 = sqrt(2.0f);
        float alpha = 4.0f + 2.0f * sqrt2 * omega_prewarp * T + omega_prewarp * omega_prewarp * T * T;

        // Numerator (high-pass coefficients)
        a0 =  4.0f / alpha;
        a1 = -8.0f / alpha;
        a2 =  4.0f / alpha;

        // Denominator (feedback terms)
        b1 = (2.0f * (omega_prewarp * omega_prewarp * T * T - 4.0f)) / alpha;
        b2 = (4.0f - 2.0f * sqrt2 * omega_prewarp * T + omega_prewarp * omega_prewarp * T * T) / alpha;
    }

public:
    // Constructor: cutoffPeriod = 1/f_c (seconds)
    HighPassSecondOrderFilter(float period) : cutoffPeriod(fabs(period)) {
        reset();
    }

    // Reset filter state
    void reset() {
        x1 = x2 = 0;
        y1 = y2 = 0;
    }

    // Update filter with new input
    float update(float input, float deltaTime) {
        calculateCoefficients(deltaTime);

        // Apply difference equation (feedback terms ADDED, not subtracted)
        float output = a0 * input + a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2;

        // Update state
        x2 = x1;
        x1 = input;
        y2 = y1;
        y1 = output;

        return output;
    }

    // Change cutoff frequency (1/f_c in seconds)
    void setCutoffPeriod(float period) {
        cutoffPeriod = fabs(period);
    }
};

#endif
