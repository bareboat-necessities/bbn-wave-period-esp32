#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

/*
  SeaStateAutoTuner — minimal online stats from acceleration input

  Computes:
    • σ_a^2  — acceleration variance (EWMA, bias-corrected)
    • f_a    — average acceleration frequency (Hz) via constant-Q band analyzer
*/

struct DebiasedEMA {
    float value  = 0.0f;
    float weight = 0.0f;
    inline void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float alpha) {
        value  = (1.0f - alpha) * value + alpha * x;
        weight = (1.0f - alpha) * weight + alpha;
    }
    inline float get()     const { return (weight > 1e-12f) ? value / weight : 0.0f; }
    inline bool  isReady() const { return weight > 1e-6f; }
};

class SeaStateAutoTuner {
public:
    // --- Public knobs (safe defaults) ---
    static constexpr float FMIN_HZ      = 0.03f;
    static constexpr float FMAX_HZ      = 4.00f;
    static constexpr int   N_BINS       = 60;    // log grid
    static constexpr float CONST_Q_R    = 0.12f; // ENBW = r * f_k
    static constexpr float PI_          = 3.14159265358979323846f;
    static constexpr float TWO_PI_      = 2.0f * PI_;

    explicit SeaStateAutoTuner(float tau_var_sec  = 60.0f,   // variance EWMA horizon
                               float tau_spec_sec = 20.0f,   // per-bin analyzer smoothing
                               float tau_out_sec  = 5.0f)    // output frequency smoothing
    : tau_var(tau_var_sec),
      tau_spec(tau_spec_sec),
      tau_out((tau_out_sec > 1e-3f) ? tau_out_sec : 1e-3f) {
        reset();
    }

    inline void reset() {
        last_dt  = -1.0f;
        alpha_var = alpha_out = 0.0f;
        A_mean.reset(); A_sq.reset(); A_var.reset();
        Freq_out.reset();
        ready_spec = false;

        for (int i = 0; i < N_BINS; ++i) {
            freq_hz[i] = 0.0f; domega[i] = 0.0f; enbw_hz[i] = 0.0f;
            c[i] = 1.0; s[i] = 0.0; cos_dphi[i] = 1.0f; sin_dphi[i] = 0.0f;
            zr[i] = zi[i] = 0.0;
        }
        buildGrid();
    }

    // --- Main update ---
    inline void update(float dt_s, float accel) {
        if (!(dt_s > 0.0f) || !std::isfinite(accel)) return;
        updateAlphas(dt_s);
        precomputePerDt(dt_s);

        // Acceleration variance (debiaised)
        A_mean.update(accel, alpha_var);
        A_sq.update(accel * accel, alpha_var);
        const float mu  = A_mean.get();
        const float var = std::max(0.0f, A_sq.get() - mu * mu);
        A_var.update(var, alpha_var);

        // Constant-Q analyzer → spectral centroid of acceleration
        double Sa0 = 0.0; // ∫ S_a(ω) dω
        double Sa1 = 0.0; // ∫ ω S_a(ω) dω

        const double a_now = static_cast<double>(accel);

        for (int i = 0; i < N_BINS; ++i) {
            // rotate unit phasor
            const double cn = c[i] * cos_dphi[i] - s[i] * sin_dphi[i];
            const double sn = s[i] * cos_dphi[i] + c[i] * sin_dphi[i];
            c[i] = cn; s[i] = sn;

            // baseband demod of acceleration at ω_i
            const double y_r = a_now * c[i];
            const double y_i = -a_now * s[i];

            // one-pole IIR (EWMA) with α_k fixed by ENBW
            const double a = static_cast<double>(alpha_k[i]);
            zr[i] = (1.0 - a) * zr[i] + a * y_r;
            zi[i] = (1.0 - a) * zi[i] + a * y_i;

            // power & ENBW normalization
            const double Pbb     = zr[i]*zr[i] + zi[i]*zi[i];
            const double ENBW_Hz = std::max(double(enbw_hz[i]), 1e-12);
            const double S_a_Hz  = Pbb / ENBW_Hz;            // m²/s⁴ per Hz
            const double S_a_rad = S_a_Hz / double(TWO_PI_); // m²/s⁴ per (rad/s)

            // integrate with bin width (2·Δω_i)
            const double w   = double(omega[i]);
            const double dwi = double(2.0f * domega[i]);

            Sa0 += S_a_rad * dwi;
            Sa1 += S_a_rad * w   * dwi;
        }

        if (Sa0 > 0.0) {
            const float w_cent = float(Sa1 / Sa0);              // rad/s
            const float f_cent = std::max(0.0f, w_cent / TWO_PI_);
            Freq_out.update(f_cent, alpha_out);
            ready_spec = true;
        }
    }

    // --- Accessors ---
    inline float getAccelVariance()   const { return A_var.get(); }               // σ_a²
    inline float getAccelFrequencyHz() const { return Freq_out.get(); }           // f_a
    inline float getAccelPeriodSec()   const {
        const float f = getAccelFrequencyHz();
        return (f > 1e-9f) ? (1.0f / f) : 0.0f;
    }
    inline bool  isReady()            const { return A_var.isReady() && ready_spec; }

    // Optional: tweak horizons at runtime
    inline void setTauVar(float t)  { tau_var  = std::max(1e-3f, t); last_dt = -1.0f; }
    inline void setTauSpec(float t) { tau_spec = std::max(1e-3f, t); last_dt = -1.0f; }
    inline void setTauOut(float t)  { tau_out  = std::max(1e-3f, t); last_dt = -1.0f; }

private:
    // --- Grid + analyzer state ---
    float freq_hz[N_BINS];
    float omega[N_BINS];
    float domega[N_BINS];   // half-widths in rad/s (Voronoi)
    float alpha_k[N_BINS];  // per-bin EWMA
    float enbw_hz[N_BINS];  // per-bin exact ENBW (Hz)

    // rotators & IIR states
    double c[N_BINS], s[N_BINS];
    float  cos_dphi[N_BINS], sin_dphi[N_BINS];
    double zr[N_BINS], zi[N_BINS];

    // horizons and alphas
    float tau_var  = 60.0f;
    float tau_spec = 20.0f;
    float tau_out  = 5.0f;

    float last_dt  = -1.0f;
    float alpha_var = 0.0f;
    float alpha_out = 0.0f;

    // outputs
    DebiasedEMA A_mean, A_sq, A_var;
    DebiasedEMA Freq_out;
    bool        ready_spec = false;

    // --- Helpers ---
    inline void buildGrid() {
        // geometric spacing in Hz
        const float ratio = std::exp(std::log(FMAX_HZ / FMIN_HZ) / float(N_BINS - 1));
        freq_hz[0] = FMIN_HZ;
        for (int i = 1; i < N_BINS; ++i) freq_hz[i] = freq_hz[i - 1] * ratio;

        // to rad/s & Voronoi half-widths
        for (int i = 0; i < N_BINS; ++i) {
            omega[i] = TWO_PI_ * freq_hz[i];
        }
        for (int i = 0; i < N_BINS; ++i) {
            const float w   = omega[i];
            const float wL  = (i > 0) ? omega[i - 1] : w;
            const float wR  = (i < N_BINS - 1) ? omega[i + 1] : w;
            float dW = 0.5f * (wR - wL);
            const float dW_min_rel = 0.0025f * std::max(w, 0.0f); // 0.25% of ω
            const float dW_min_abs = 1e-5f;
            if (dW < std::max(dW_min_abs, dW_min_rel))
                dW = std::max(dW_min_abs, dW_min_rel);
            domega[i] = dW;
        }
    }

    inline void updateAlphas(float dt_s) {
        if (dt_s == last_dt) return;
        last_dt   = dt_s;
        alpha_var = 1.0f - std::exp(-dt_s / tau_var);
        alpha_out = 1.0f - std::exp(-dt_s / tau_out);
    }

    inline void precomputePerDt(float dt_s) {
        // per-dt update of analyzer constants (α_k, ENBW, rotator step)
        const float Fs = 1.0f / dt_s;
        for (int i = 0; i < N_BINS; ++i) {
            const float fk = freq_hz[i];

            // Set α_k to hit an exact ENBW = r * f_k (Hz)
            const float ENBW_target = CONST_Q_R * std::max(fk, 1e-9f);
            float alpha = (2.0f * ENBW_target) / (ENBW_target + 0.5f * Fs);
            if (alpha < 0.0f) alpha = 0.0f; else if (alpha > 1.0f) alpha = 1.0f;
            alpha_k[i] = alpha;

            enbw_hz[i] = (alpha / std::max(2.0f - alpha, 1e-12f)) * (0.5f * Fs);

            const float dphi = omega[i] * dt_s;
            if (std::fabs(dphi) < 1e-3f) {
                cos_dphi[i] = 1.0f - 0.5f * dphi * dphi;
                sin_dphi[i] = dphi;
            } else {
                cos_dphi[i] = std::cos(dphi);
                sin_dphi[i] = std::sin(dphi);
            }
        }
    }
};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

static inline void SeaStateAutoTuner_sine_test() {
    constexpr float Fs = 240.0f, DT = 1.0f / Fs;
    constexpr float A = 1.0f, F_HZ = 0.7f; // accel from z = sin → a = −ω^2 sin
    const float omega = SeaStateAutoTuner::TWO_PI_ * F_HZ;

    SeaStateAutoTuner tun(30.0f, 10.0f, 2.0f);

    float t = 0.0f;
    for (int n = 0; n < int(15.0f / DT); ++n) {
        t += DT;
        const float a = -A * omega * omega * std::sin(omega * t);
        tun.update(DT, a);
    }

    const float f_est = tun.getAccelFrequencyHz();
    const float var_a = tun.getAccelVariance();

    std::cerr << "[AutoTuner] f_est=" << f_est << " Hz (true " << F_HZ
              << "), sigma_a=" << std::sqrt(std::max(0.0f, var_a)) << " m/s^2\n";

    if (!(std::fabs(f_est - F_HZ) < 0.05f * F_HZ))
        throw std::runtime_error("Frequency estimate off by >5%");
    if (!(var_a > 0.0f))
        throw std::runtime_error("Variance should be positive");
}
#endif
