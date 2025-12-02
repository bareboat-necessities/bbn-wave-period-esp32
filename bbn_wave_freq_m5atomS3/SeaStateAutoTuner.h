#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

/*
  SeaStateAutoTuner — minimal online acceleration stats

  Purpose:
    • Estimate acceleration variance σ_a² (time-domain EWMA)
    • Smooth externally provided frequency f_in (Hz) via EMA
    • Compute R_S estimate = σ_a τ³ where τ = 1 / (2f) (half period)
*/

struct DebiasedEMA {
    float value  = 0.0f;
    float weight = 0.0f;
    inline void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float alpha) {
        value  = (1.0f - alpha) * value + alpha * x;
        weight = (1.0f - alpha) * weight + alpha;
    }
    inline float get() const {
        return (weight > 1e-12f) ? value / weight : 0.0f;
    }
    inline bool isReady() const { return weight > 1e-6f; }
};

class SeaStateAutoTuner {
public:
    explicit SeaStateAutoTuner(float tau_var_sec = 15.0f,   // variance EWMA horizon
                               float tau_freq_sec = 0.5f)   // frequency smoothing horizon
    : tau_var(tau_var_sec), tau_freq(tau_freq_sec) {
        reset();
    }

    inline void reset() {
        last_dt = -1.0f;
        alpha_var = alpha_freq = 0.0f;
        A_mean.reset(); A_sq.reset(); A_var.reset();
        Freq_smoothed.reset();
    }

    // main update
    inline void update(float dt_s, float accel, float f_input_hz) {
        if (!(dt_s > 0.0f) || !std::isfinite(accel) || !std::isfinite(f_input_hz))
            return;

        updateAlphas(dt_s);

        // time-domain EWMA variance
        A_mean.update(accel, alpha_var);
        A_sq.update(accel * accel, alpha_var);
        const float mu  = A_mean.get();
        const float var = std::max(0.0f, A_sq.get() - mu * mu);
        A_var.update(var, alpha_var);

        // smooth incoming frequency
        Freq_smoothed.update(f_input_hz, alpha_freq);
    }

    // accessors
    inline float getAccelVariance() const { return A_var.get(); }       // σ_a²
    inline float getAccelStd()      const { return std::sqrt(std::max(0.0f, A_var.get())); }
    inline float getFrequencyHz()   const { return Freq_smoothed.get(); }
    inline float getPeriodSec()     const {
        const float f = Freq_smoothed.get();
        return (f > 1e-9f) ? (1.0f / f) : 0.0f;
    }

    // R_S estimate = σ_a τ³ where τ = 1 / (2f) (half period)
    inline float getR_S_est() const {
        const float sigma_a = getAccelStd();
        const float f = Freq_smoothed.get();
        if (f <= 1e-6f || sigma_a <= 0.0f)
            return 0.0f;
        const float tau = 0.5f / f;
        return sigma_a * (tau * tau * tau);
    }

    inline bool isReady() const { return A_var.isReady() && Freq_smoothed.isReady(); }

    // Optional runtime tuning
    inline void setTauVar(float t)  { tau_var  = std::max(1e-3f, t); last_dt = -1.0f; }
    inline void setTauFreq(float t) { tau_freq = std::max(1e-3f, t); last_dt = -1.0f; }

private:
    float tau_var  = 15.0f;
    float tau_freq = 0.5f;
    float last_dt  = -1.0f;
    float alpha_var = 0.0f;
    float alpha_freq = 0.0f;

    DebiasedEMA A_mean, A_sq, A_var;
    DebiasedEMA Freq_smoothed;

    inline void updateAlphas(float dt_s) {
        if (dt_s == last_dt) return;
        last_dt = dt_s;
        alpha_var  = 1.0f - std::exp(-dt_s / tau_var);
        alpha_freq = 1.0f - std::exp(-dt_s / tau_freq);
    }
};

#ifdef SEA_STATE_TUNER_TEST
#include <iostream>

static inline void SeaStateAutoTuner_test() {
    constexpr float Fs = 240.0f;
    constexpr float DT = 1.0f / Fs;
    constexpr float F_HZ = 0.5f;
    const float omega = 2.0f * 3.14159265358979323846f * F_HZ;

    SeaStateAutoTuner tuner(30.0f, 3.0f);

    float t = 0.0f;
    for (int n = 0; n < int(20.0f / DT); ++n) {
        t += DT;
        const float a = -std::sin(omega * t) * omega * omega;
        tuner.update(DT, a, F_HZ);
    }

    std::cerr << "[AutoTuner] σ_a=" << tuner.getAccelStd()
              << " m/s², f=" << tuner.getFrequencyHz()
              << " Hz, R_S=" << tuner.getR_S_est() << "\n";
}
#endif
