#ifndef ENGINE_VIBRATION_SIM_H
#define ENGINE_VIBRATION_SIM_H

#include <Arduino.h>

class EngineVibrationSim {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor with engine RPM and optional gravity (default Earth's gravity)
    EngineVibrationSim(float engineRPM, float gravity = 9.80665f)
      : rpm(engineRPM), g(gravity), time(0.0f) {
        fundamentalFreq = rpm / 60.0f;
    }

    // Call every sample with sample period dt in seconds.
    // Returns simulated vibration acceleration in m/s² (SI units).
    float update(float dt) {
        float g_signal = 0.0f;
        for (int i = 0; i < numHarmonics; ++i) {
            float freq = fundamentalFreq * (i + 1);
            g_signal += amps[i] * sinf(2.0f * PI * freq * time);
        }
        time += dt;
        return g_signal * g;  // convert g units to m/s² using provided gravity
    }

    // Set amplitude of harmonic (index 0..3), amplitude in g
    void setHarmonicAmplitude(int index, float amplitude) {
        if (index >= 0 && index < numHarmonics) {
            amps[index] = amplitude;
        }
    }

private:
    float rpm;              // Engine RPM
    float fundamentalFreq;  // Fundamental frequency in Hz
    float time;             // Internal time tracker
    float g;                // Gravity constant to convert g to m/s²

    static const int numHarmonics = 4;
    float amps[numHarmonics] = {0.02f, 0.01f, 0.005f, 0.003f}; // Default amplitudes in g
};

#endif // ENGINE_VIBRATION_SIM_H
