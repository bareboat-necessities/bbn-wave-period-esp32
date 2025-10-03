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
 *   3) Exponentially average spectral moments:
 *        M0 = ⟨|η_env|²⟩,
 *        M1 = ⟨|η_env|² ω⟩,
 *        M2 = ⟨|η_env|² ω²⟩.
 *      Also accumulate Q0,Q1,Q2 for bias correction.
 *   4) Mean frequency ω̄ and variance μ₂ computed with Jensen-aware corrections.
 *   5) Spectral regularity: R_spec = exp(−β · √μ₂/ω̄).
 *   6) Phase regularity: R_phase = mean resultant length of demodulated phase.
 *   7) Final regularity: R = max(R_phase, R_spec).
 *   8) Significant wave height: Hs ≈ 2√M0 (monochromatic assumption).
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
        M0 = M1 = M2 = 0.0f;
        Q0 = Q1 = Q2 = 0.0f;
        R_spec = R_phase = R_out = 0.0f;
        coh_r = coh_i = 0.0f;
        has_coh = false;
        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;
        has_moments = false;
        has_R_out = false;
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

    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    float getWaveHeightEnvelopeEst() const {
        if (M0 <= 0.0f) return 0.0f;
        return 2.0f * std::sqrt(M0); // monochromatic assumption
    }

    float getDisplacementFrequencyHz() const {
        return (M0 > EPSILON) ? (M1 / M0) / (2.0f * float(M_PI)) : 0.0f;
    }

private:
    // time constants
    float tau_env, tau_mom, tau_coh, tau_out;
    float last_dt;

    // EMAs
    float alpha_env, alpha_mom, alpha_coh, alpha_out;

    // demod state
    float phi;
    float z_real, z_imag;

    // moments
    float M0, M1, M2;
    // second-order helpers for Jensen-aware correction
    float Q0, Q1, Q2;
    bool has_moments;

    // regularity
    float R_spec, R_phase, R_out;
    bool  has_R_out;

    // phase coherence
    float coh_r, coh_i;
    bool  has_coh;

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
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(phi), s = std::sin(phi);
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
        float inv_w2 = 1.0f / std::max(omega * omega, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;
        float P2 = P_disp * P_disp;

        if (!has_moments) {
            M0 = P_disp;
            M1 = P_disp * omega;
            M2 = P_disp * omega * omega;
            Q0 = P2;
            Q1 = P2 * omega;
            Q2 = P2 * omega * omega;
            has_moments = true;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * omega;
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * omega * omega;
            Q0 = (1.0f - alpha_mom) * Q0 + alpha_mom * P2;
            Q1 = (1.0f - alpha_mom) * Q1 + alpha_mom * P2 * omega;
            Q2 = (1.0f - alpha_mom) * Q2 + alpha_mom * P2 * omega * omega;
        }
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            if (!has_coh) { coh_r = 1.0f; coh_i = 0.0f; has_coh = true; }
            R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
            return;
        }

        float u_r = z_real / mag;
        float u_i = z_imag / mag;

        if (!has_coh) { coh_r = u_r; coh_i = u_i; has_coh = true; }
        else {
            coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
            coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;
        }
        R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
    }

    void computeRegularityOutput() {
        if (!(M0 > EPSILON)) { R_spec = R_out = R_phase; return; }

        float invM0   = 1.0f / M0;
        float invM0_2 = invM0 * invM0;
        float varY    = std::max(0.0f, Q0 - M0 * M0);
        float cov10   = Q1 - M1 * M0;
        float cov20   = Q2 - M2 * M0;

        float omega_bar_naive  = M1 * invM0;
        float omega2_bar_naive = M2 * invM0;

        // delta-method corrections
        float omega_bar  = omega_bar_naive  + omega_bar_naive  * invM0_2 * varY - cov10 * invM0_2;
        float omega2_bar = omega2_bar_naive + omega2_bar_naive * invM0_2 * varY - cov20 * invM0_2;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar * omega_bar);

        float rbw = (omega_bar > EPSILON) ? (std::sqrt(mu2) / omega_bar) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);

        R_out = std::max(R_phase, R_spec);
        has_R_out = true;
    }
};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

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
    float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f;
    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        float a = za.second;
        reg.update(DT, a, wave.omega);
        R_spec  = reg.getRegularitySpectral();
        R_phase = reg.getRegularityPhase();
        Hs_est  = reg.getWaveHeightEnvelopeEst();
    }
    const float Hs_expected = 2.0f * SINE_AMPLITUDE;
    if (!(R_spec > 0.90f))
        throw std::runtime_error("Sine: R_spec did not converge near 1.");
    if (!(R_phase > 0.80f))
        throw std::runtime_error("Sine: R_phase did not converge near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.6f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within tolerance.");
    std::cerr << "[PASS] Sine wave test passed. Hs_est=" << Hs_est
              << " (expected ~" << Hs_expected << ")\n";
}
#endif
