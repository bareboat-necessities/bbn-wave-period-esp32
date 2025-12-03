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

  Notes:
    • K_periods is a dimensionless factor controlling the variance horizon:
        τ_var_dyn ≈ K_periods * T_eff,
      where T_eff = 1 / f_eff and f_eff is the smoothed frequency.
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
    // K_periods: dimensionless horizon in periods.
    //   τ_var_dyn ≈ K_periods * T_eff,  T_eff = 1 / f_eff
    // Default K_periods ≈ 1.5 → ~4.5 periods for ~95% response.
    explicit SeaStateAutoTuner(float K_periods_    = 1.5f,
                               float tau_freq_sec = 0.5f)   // frequency smoothing horizon (seconds)
    : K_periods(K_periods_), tau_freq(tau_freq_sec) {
        reset();
    }

    inline void reset() {
        last_dt_freq = -1.0f;
        alpha_var = alpha_freq = 0.0f;
        A_mean.reset(); A_sq.reset(); A_var.reset();
        Freq_smoothed.reset();
    }

    // main update
    inline void update(float dt_s, float accel, float f_input_hz) {
        if (!(dt_s > 0.0f) || !std::isfinite(accel) || !std::isfinite(f_input_hz))
            return;

        // 1) Update alpha for frequency smoothing (fixed horizon in seconds)
        updateAlphaFreq(dt_s);

        // 2) Smooth incoming frequency first
        Freq_smoothed.update(f_input_hz, alpha_freq);

        // Effective frequency for variance horizon (use smoothed if ready)
        float f_eff = Freq_smoothed.isReady() ? Freq_smoothed.get() : f_input_hz;
        // Clamp to avoid insane horizons at tiny/zero frequency
        const float F_MIN = 0.05f;   // 20 s period max
        const float F_MAX = 5.0f;    // avoid crazy high freq
        if (!std::isfinite(f_eff)) f_eff = F_MIN;
        f_eff = std::max(F_MIN, std::min(F_MAX, f_eff));

        const float T_eff = 1.0f / f_eff;

        // 3) Dynamic variance time constant: a few periods
        const float TAU_MIN = 0.3f;       // seconds: don't go *too* twitchy
        const float TAU_MAX = 60.0f;      // seconds: don't be glacial
        const float tau_var_dyn = std::max(TAU_MIN,
                                           std::min(TAU_MAX, K_periods * T_eff));

        // Compute alpha for variance based on dynamic tau
        alpha_var = 1.0f - std::exp(-dt_s / tau_var_dyn);

        // 4) Time-domain EWMA variance
        A_mean.update(accel, alpha_var);
        A_sq.update(accel * accel, alpha_var);
        const float mu       = A_mean.get();
        const float var_inst = std::max(0.0f, A_sq.get() - mu * mu);
        A_var.update(var_inst, alpha_var);
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
    inline void setKPeriods(float k) { K_periods = std::max(0.1f, k); }   // >= 0.1 periods

    // Backward-compat alias if you still call setTauVar(...) somewhere
    inline void setTauVar(float t) { setKPeriods(t); }

    inline void setTauFreq(float t) {
        tau_freq = std::max(1e-3f, t);
        last_dt_freq = -1.0f;  // force recompute on next update
    }

private:
    // K_periods: dimensionless factor such that τ_var_dyn ≈ K_periods * T_eff
    float K_periods = 1.5f;
    float tau_freq  = 0.5f;   // seconds
    float last_dt_freq  = -1.0f;
    float alpha_var  = 0.0f;
    float alpha_freq = 0.0f;

    DebiasedEMA A_mean, A_sq, A_var;
    DebiasedEMA Freq_smoothed;

    inline void updateAlphaFreq(float dt_s) {
        if (dt_s == last_dt_freq) return;
        last_dt_freq = dt_s;
        alpha_freq = 1.0f - std::exp(-dt_s / tau_freq);
    }
};

#ifdef SEA_STATE_TUNER_TEST
#include <iostream>

static inline void SeaStateAutoTuner_test() {
    constexpr float Fs   = 240.0f;
    constexpr float DT   = 1.0f / Fs;
    constexpr float F_HZ = 0.5f;
    const float omega = 2.0f * 3.14159265358979323846f * F_HZ;

    // K_periods ≈ 1.5 → τ_var_dyn ≈ 1.5 * T = 3 s at 0.5 Hz
    // ~9 s (~4.5 periods) for 95% response
    SeaStateAutoTuner tuner(1.5f, 3.0f);

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
