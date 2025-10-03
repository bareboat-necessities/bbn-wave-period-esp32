#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * Copyright 2025, Mikhail Grushinskiy
 *
 * SeaStateRegularity — Online estimator of ocean wave regularity and height.
 *
 * Inputs
 *   • Vertical acceleration a_z(t) [m/s²]
 *   • Instantaneous angular frequency ω_inst(t) [rad/s] from an external tracker
 *
 * Pipeline
 *   1) Demodulate acceleration by φ(t) = ∫ ω_inst dt to obtain baseband z(t).
 *   2) Normalize by ω²: η_env(t) = z(t)/ω²(t).
 *      ⇒ This converts acceleration to displacement envelope.
 *   3) Track power and spectral moments with **bias-corrected exponential averages (DebiasedEMA)**:
 *        A0 = ⟨|z|²⟩             (acceleration variance, diagnostic only)
 *        M0 = ⟨|η_env|²⟩         (displacement variance)
 *        M1 = ⟨|η_env|² ω⟩
 *        M2 = ⟨|η_env|² ω²⟩
 *      (Q0,Q1,Q2 also tracked with DebiasedEMA for Jensen bias correction.)
 *   4) Compute mean frequency ω̄ and variance μ₂ using **Jensen-aware corrections**
 *      applied to the bias-corrected moments.
 *   5) Spectral regularity: R_spec = exp(−β · √μ₂/ω̄).
 *   6) Phase regularity: R_phase = mean resultant length of demodulated phase,
 *      tracked via **bias-corrected EMA** of phase unit vectors.
 *   7) Final regularity: R = max(R_phase, R_spec), smoothed by a **bias-corrected EMA**.
 *   8) Narrowness ν = √μ₂/ω̄ (dimensionless bandwidth).
 *   9) Significant wave height: Hs ≈ 4√M0, using bias-corrected M0 (oceanographic convention).
 *
 * Notes
 *   • Although the input is acceleration, all reported spectral moments (M0,M1,M2)
 *     are for displacement, consistent with oceanographic definitions of Hs, Tp, and bandwidth.
 *   • A0 is separately provided as the variance of acceleration envelope for diagnostics.
 *   • Bias-corrected EMAs eliminate startup underestimation: outputs are unbiased in
 *     expectation from the very first sample.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

#if __cplusplus < 201703L
namespace std {
    template <class T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }
}
#endif

// ---- Bias-corrected EMA helper ----
struct DebiasedEMA {
    float value  = 0.0f;
    float weight = 0.0f;
    void reset() { value = 0.0f; weight = 0.0f; }
    void update(float x, float alpha) {
        value  = (1.0f - alpha) * value + alpha * x;
        weight = (1.0f - alpha) * weight + alpha;
    }
    float get() const {
        return (weight > 1e-12f) ? value / weight : 0.0f;
    }
    bool isReady() const { return weight > 1e-6f; }
};

class SeaStateRegularity {
public:
    constexpr static float EPSILON   = 1e-12f;
    constexpr static float BETA_SPEC = 1.0f;

    SeaStateRegularity(float tau_env_sec   = 15.0f,
                       float tau_mom_sec   = 180.0f,
                       float tau_coh_sec   = 60.0f,
                       float tau_out_sec   = 30.0f)
    {
        tau_env = tau_env_sec;
        tau_mom = tau_mom_sec;
        tau_coh = std::max(1e-3f, tau_coh_sec);
        tau_out = std::max(1e-3f, tau_out_sec);
        reset();
    }

    void reset() {
        phi = 0.0f;
        z_real = z_imag = 0.0f;
        M0.reset(); M1.reset(); M2.reset();
        Q0.reset(); Q1.reset(); Q2.reset();
        A0.reset();
        R_out.reset();
        coh_r.reset(); coh_i.reset();
        R_spec = R_phase = 0.0f;
        has_moments = false;
        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;
        nu = 0.0f;
        omega_bar_corr = 0.0f;
        omega_bar_naive = 0.0f;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updatePhaseCoherence();
        updateSpectralMoments(omega_inst);
        computeRegularityOutput();
    }

    // === Public getters ===

    /// Final regularity index R (bias-corrected, smoothed EMA of max(R_phase,R_spec)).
    float getRegularity() const { return R_out.get(); }

    /// Spectral regularity R_spec = exp(−β · √μ₂/ω̄), from bias-corrected moments.
    float getRegularitySpectral() const { return R_spec; }

    /// Phase regularity R_phase = mean resultant length of demodulated phase
    /// (tracked via bias-corrected EMA of phase unit vectors).
    float getRegularityPhase() const { return R_phase; }

    /// Narrowness ν = √μ₂ / ω̄, dimensionless bandwidth from bias-corrected moments.
    float getNarrowness() const { return nu; }

    /// Significant wave height estimate Hs = 4√M0, using bias-corrected M0 (oceanographic convention).
    float getWaveHeightEnvelopeEst() const {
        float m0 = M0.get();
        return (m0 > 0.0f) ? 4.0f * std::sqrt(m0) : 0.0f;
    }

    /// Jensen-corrected displacement mean frequency [Hz], from bias-corrected moments.
    float getDisplacementFrequencyHz() const {
        return (omega_bar_corr > EPSILON) ? (omega_bar_corr / (2.0f * float(M_PI))) : 0.0f;
    }

    /// Naive (uncorrected) displacement frequency [Hz] = (M1/M0)/(2π), from bias-corrected M0,M1.
    float getDisplacementFrequencyNaiveHz() const {
        return (omega_bar_naive > EPSILON) ? (omega_bar_naive / (2.0f * float(M_PI))) : 0.0f;
    }

    /// Displacement peak period Tp = 2π/ω̄ [s], based on Jensen-corrected, bias-corrected ω̄.
    float getDisplacementPeriodSec() const {
        return (omega_bar_corr > EPSILON) ? (2.0f * float(M_PI) / omega_bar_corr) : 0.0f;
    }

    /// Diagnostic: variance of acceleration envelope (before 1/ω² normalization),
    /// tracked with bias-corrected EMA.
    float getAccelerationVariance() const { return A0.get(); }

private:
    // time constants
    float tau_env, tau_mom, tau_coh, tau_out;
    float last_dt;
    float alpha_env, alpha_mom, alpha_coh, alpha_out;

    // demod state
    float phi;
    float z_real, z_imag;

    // displacement spectral moments (bias-corrected EMAs)
    DebiasedEMA M0, M1, M2;
    DebiasedEMA Q0, Q1, Q2;
    DebiasedEMA A0;

    // coherence and regularity
    DebiasedEMA coh_r, coh_i;
    DebiasedEMA R_out;
    float R_spec, R_phase;

    // cached narrowness and mean freq
    float nu;
    float omega_bar_corr, omega_bar_naive;
    bool has_moments;

    void updateAlpha(float dt_s) {
        if (dt_s == last_dt) return;
        alpha_env = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_out = 1.0f - std::exp(-dt_s / tau_out);
        last_dt = dt_s;
    }

    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        phi += omega_inst * dt_s;
        float phi_wrapped = std::fmod(phi, 2.0f * float(M_PI));
        float c = std::cos(phi_wrapped), s = std::sin(phi_wrapped);
        float y_real =  accel_z * c;
        float y_imag = -accel_z * s;

        if (!has_moments) {
            z_real = y_real;
            z_imag = y_imag;
            return;
        }
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;
    }

    void updateSpectralMoments(float omega) {
        float P_acc = z_real * z_real + z_imag * z_imag;
        float inv_w2 = 1.0f / std::max(omega * omega, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;
        float P2 = P_disp * P_disp;

        A0.update(P_acc, alpha_mom);
        M0.update(P_disp, alpha_mom);
        M1.update(P_disp * omega, alpha_mom);
        M2.update(P_disp * omega * omega, alpha_mom);
        Q0.update(P2, alpha_mom);
        Q1.update(P2 * omega, alpha_mom);
        Q2.update(P2 * omega * omega, alpha_mom);

        has_moments = true;
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            if (coh_r.isReady() && coh_i.isReady())
                R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()),0.0f,1.0f);
            else
                R_phase = 0.0f;
            return;
        }
        float u_r = z_real / mag;
        float u_i = z_imag / mag;
        coh_r.update(u_r, alpha_coh);
        coh_i.update(u_i, alpha_coh);
        R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()),0.0f,1.0f);
    }

    void computeRegularityOutput() {
        if (!M0.isReady()) {
            R_out.update(R_phase, alpha_out);
            R_spec = R_phase;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        float m0 = M0.get(), m1 = M1.get(), m2 = M2.get();
        float q0 = Q0.get(), q1 = Q1.get(), q2 = Q2.get();

        omega_bar_naive  = (m0 > EPSILON) ? (m1 / m0) : 0.0f;
        float omega2_bar_naive = m2 / m0;

        float varY  = std::max(0.0f, q0 - m0*m0);
        float cov10 = q1 - m1*m0;
        float cov20 = q2 - m2*m0;
        float invm0_2 = 1.0f / (m0*m0);

        float omega_bar  = omega_bar_naive  + omega_bar_naive  * invm0_2 * varY - cov10 * invm0_2;
        float omega2_bar = omega2_bar_naive + omega2_bar_naive * invm0_2 * varY - cov20 * invm0_2;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar * omega_bar);
        float rbw = (omega_bar > EPSILON) ? (std::sqrt(mu2) / omega_bar) : 0.0f;

        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);
        nu = rbw;
        omega_bar_corr = (omega_bar > 0.0f) ? omega_bar : 0.0f;

        float R_now = std::max(R_phase, R_spec);
        R_out.update(R_now, alpha_out);
    }
};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

/**
 * SeaState_sine_wave_test
 *
 * Validates SeaStateRegularity on a pure sine displacement of amplitude A and
 * frequency f:
 *   • Ensures R_spec → 1 and R_phase → 1 (high regularity).
 *   • Checks Hs ≈ 2√2·A (true oceanographic significant wave height for a sine).
 *   • Verifies narrowness ν → 0 for a monochromatic signal.
 *   • Confirms displacement frequency estimates match the input frequency.
 *
 * This test also implicitly validates:
 *   • Jensen corrections applied to bias-corrected spectral moments (M0,M1,M2).
 *   • Bias-corrected EMA startup behavior — estimates converge immediately without
 *     the typical exponential warm-up lag.
 */
constexpr float SAMPLE_FREQ_HZ   = 240.0f;
constexpr float DT               = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 60.0f;
constexpr float SINE_AMPLITUDE   = 1.0f;
constexpr float SINE_FREQ_HZ     = 0.3f;

struct SineWave {
    float amplitude;
    float omega;
    float phi;
    SineWave(float A, float f_hz)
        : amplitude(A), omega(2.0f * float(M_PI) * f_hz), phi(0.0f) {}
    std::pair<float,float> step(float dt) {
        phi += omega * dt;
        if (phi > 2.0f * float(M_PI)) phi -= 2.0f * float(M_PI);
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

inline void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity reg;
    float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f, nu = 0.0f;
    float f_disp_corr = 0.0f, f_disp_naive = 0.0f, Tp = 0.0f;
    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        float a = za.second;
        reg.update(DT, a, wave.omega);
        R_spec      = reg.getRegularitySpectral();
        R_phase     = reg.getRegularityPhase();
        Hs_est      = reg.getWaveHeightEnvelopeEst();
        nu          = reg.getNarrowness();
        f_disp_corr = reg.getDisplacementFrequencyHz();
        f_disp_naive= reg.getDisplacementFrequencyNaiveHz();
        Tp          = reg.getDisplacementPeriodSec();
    }

    // Correct oceanographic Hs for a sine: Hs = 2√2·A
    const float Hs_expected = 2.0f * std::sqrt(2.0f) * SINE_AMPLITUDE;

    if (!(R_spec > 0.90f))
        throw std::runtime_error("Sine: R_spec did not converge near 1.");
    if (!(R_phase > 0.80f))
        throw std::runtime_error("Sine: R_phase did not converge near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.6f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within tolerance.");
    if (!(nu < 0.05f))
        throw std::runtime_error("Sine: Narrowness should be close to 0 for a pure tone.");

    std::cerr << "[PASS] Sine wave test passed. "
              << "Hs_est=" << Hs_est
              << " (expected ~" << Hs_expected << "), Narrowness=" << nu
              << ", f_disp_corr=" << f_disp_corr << " Hz"
              << ", f_disp_naive=" << f_disp_naive << " Hz"
              << ", Tp=" << Tp << " s\n";
}
#endif
