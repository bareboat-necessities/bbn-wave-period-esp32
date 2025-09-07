#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

#ifdef SEA_STATE_TEST
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#endif

/**
 * Copyright 2025, Mikhail Grushinskiy
 *
 * @brief Online estimator of ocean wave regularity from vertical acceleration.
 *
 * Computes:
 *  - Spectral regularity R_spec (from spectral moments)
 *  - Phase coherence regularity R_phase
 *  - Harmonic-safe regularity R_safe
 *  - Smoothed output R_out
 *  - Approximate significant wave height estimate
 *  - Displacement frequency estimate
 *
 */

class SeaStateRegularity {
public:
    // Configurable constants
    constexpr static float EPSILON = 1e-12f;
    constexpr static float HEIGHT_R_HI = 0.98f;
    constexpr static float HEIGHT_R_LO = 0.50f;
    constexpr static float BROADBAND_WAVE_THRESHOLD = 0.3f;
    constexpr static float BROADBAND_WAVE_REDUCTION_MAX = 0.08f;
    constexpr static float LARGE_WAVE_THRESHOLD = 0.5f;
    constexpr static float LARGE_WAVE_BOOST_MAX = 0.12f;

    // Safe defaults for typical ocean waves (PM/JONSWAP)
    SeaStateRegularity(float tau_env_sec   = 15.0f,   // envelope I/Q smoother (~2–3 waves)
                       float tau_mom_sec   = 180.0f,  // moments horizon (minutes-scale)
                       float omega_min_hz  = 0.03f,   // cutoff (~33 s period)
                       float tau_coh_sec   = 60.0f,   // steadies R_phase
                       float tau_out_sec   = 30.0f,   // final R_out smoother
                       float tau_omega_sec = 45.0f)   // protects 1/omega^2 normalization
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_out   = std::max(1e-3f, tau_out_sec);
        tau_omega = std::max(0.0f,  tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz; // Hz → rad/s
        reset();
    }

    void reset() {
        // Phase / envelope state
        phi = 0.0f;
        z_real = z_imag = 0.0f;

        // Spectral moments and regularities
        M0 = M1 = M2 = 0.0f;
        nu = 0.0f;
        R_spec = R_phase = R_safe = R_out = 0.0f;

        // Alphas and timing
        alpha_env = alpha_mom = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

        // Phase coherence avg
        coh_r = coh_i = 0.0f;

        // Frequency tracking
        omega_lp = omega_disp_lp = 0.0f;
        omega_last = mu_w = 0.0f;

        // Variance trackers for omega stability compensation
        var_fast = 0.0f;
        var_slow = 0.0f;

        // Flags
        has_omega_lp = false;
        has_omega_disp_lp = false;
        has_moments = false;
        has_coh = false;
        has_R_out = false;

        // Gating smoother & hysteresis state
        P_disp_gate = 0.0f;
        has_gate = false;
        bb_active = false;
        lg_active = false;

        // Demod-driving omega slew limiter memory
        omega_phi_last = 0.0f;
    }

    // Main update: dt (s), vertical accel (m/s^2), instantaneous omega (rad/s)
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z != accel_z) accel_z = 0.0f;             // NaN guard
        if (omega_inst != omega_inst) omega_inst = omega_min;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityOutput();
    }

    // Readouts
    float getNarrowness() const { return nu; }
    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    float getWaveHeightEnvelopeEst() const {
        if (M0 <= 0.0f || !has_R_out) return 0.0f;
        return 2.0f * std::sqrt(M0) * heightFactorFromR(R_out);
    }

    float getDisplacementFrequencyHz() const {
        return omega_disp_lp / (2.0f * float(M_PI));
    }

    // Optional: scale time constants from dominant period Tp (s)
    void set_safe_defaults_from_peak_period(float Tp) {
        // Clamp Tp to plausible ocean periods
        Tp = std::min(std::max(Tp, 5.0f), 20.0f);
        tau_env   = std::clamp(2.0f  * Tp,  8.0f,  40.0f);
        tau_omega = std::clamp(5.0f  * Tp, 20.0f, 120.0f);
        tau_mom   = std::clamp(12.0f * Tp, 60.0f, 300.0f);
        tau_coh   = std::clamp(6.0f  * Tp, 30.0f, 120.0f);
        tau_out   = std::clamp(4.0f  * Tp, 15.0f,  60.0f);
        // Alphas will refresh on next updateAlpha() call (e.g., when dt changes).
    }

private:
    // Time constants (s)
    float tau_env, tau_mom, tau_coh, tau_out, tau_omega;
    float omega_min;
    float last_dt;

    // EMA alphas
    float alpha_env, alpha_mom, alpha_coh, alpha_out, alpha_omega;

    // Demod/envelope state
    float phi;
    float z_real, z_imag;

    // Spectral moments & regularity
    float M0, M1, M2;
    float nu;
    float R_spec, R_phase, R_safe, R_out;

    // Phase coherence
    float coh_r, coh_i;

    // Frequency tracking
    float omega_lp, omega_disp_lp;
    float omega_last, mu_w;
    float var_fast, var_slow;

    // Flags
    bool has_omega_lp;
    bool has_omega_disp_lp;
    bool has_moments;
    bool has_coh;
    bool has_R_out;

    // Variance adaptation (gentler)
    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    // Gate smoothing/hysteresis
    float P_disp_gate;
    bool has_gate;
    bool bb_active;
    bool lg_active;

    // Demod omega slew limiter memory
    float omega_phi_last;

    // ----------------------------------------------------
    // Implementation
    // ----------------------------------------------------
    void updateAlpha(float dt_s) {
        if (dt_s == last_dt) return;
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_out   = 1.0f - std::exp(-dt_s / tau_out);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
        last_dt = dt_s;
    }

    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        // Use smoothed omega with slew limit to rotate demod basis
        float omega_phi = has_omega_lp ? omega_lp : std::max(omega_inst, omega_min);
        const float domega_max = 0.5f; // rad/s per update (tune 0.2..1.0)
        if (!has_omega_lp) omega_phi_last = omega_phi;
        float domega = omega_phi - omega_phi_last;
        if (domega >  domega_max) omega_phi = omega_phi_last + domega_max;
        if (domega < -domega_max) omega_phi = omega_phi_last - domega_max;
        omega_phi_last = omega_phi;

        phi += omega_phi * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        // Rotate accel into baseband I/Q
        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        // First-time init
        if (!has_omega_lp) {
            z_real = y_real;
            z_imag = y_imag;
            omega_lp = std::max(omega_inst, omega_min);
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        // Envelope EMA (controls per-cycle smoothness)
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // Low-pass the instantaneous omega from external estimator
        float w_inst = std::max(omega_inst, omega_min);
        if (tau_omega > 0.0f) {
            omega_lp = (1.0f - alpha_omega) * omega_lp + alpha_omega * w_inst;
        } else {
            omega_lp = w_inst;
        }

        // Variance trackers for omega (used to stabilize M2)
        float delta = omega_lp - mu_w;
        mu_w += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
        // Blended omega for normalization: mostly omega_lp with a touch of moment-based omega
        float omega_norm = omega_lp;
        if (has_omega_disp_lp) omega_norm = 0.7f * omega_lp + 0.3f * omega_disp_lp;
        omega_norm = std::max(omega_norm, omega_min);

        // Convert accel-envelope → displacement-envelope via 1/omega^2
        float inv_w2 = 1.0f / std::max(omega_norm * omega_norm, EPSILON);
        float disp_real_corr = z_real * inv_w2;
        float disp_imag_corr = z_imag * inv_w2;
        float P_disp_corr = disp_real_corr * disp_real_corr + disp_imag_corr * disp_imag_corr;

        // Exponential averaging of spectral moments
        if (!has_moments) {
            M0 = P_disp_corr;
            M1 = P_disp_corr * omega_norm;
            M2 = P_disp_corr * omega_norm * omega_norm;
            has_moments = true;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp_corr;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp_corr * omega_norm;

            // Stabilize M2 with slow variance compensation on omega
            float M2_candidate = P_disp_corr * omega_norm * omega_norm - var_slow * M0;
            if (M2_candidate < 0.0f) M2_candidate = 0.0f;
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * M2_candidate;
        }

        // Moment-based mean frequency
        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;

        // Low-pass the moment-derived omega as well
        if (!has_omega_disp_lp) {
            omega_disp_lp = omega_disp;
            has_omega_disp_lp = true;
        } else {
            omega_disp_lp = (1.0f - alpha_omega) * omega_disp_lp + alpha_omega * omega_disp;
        }
    }

    void updatePhaseCoherence() {
        float mag = std::sqrt(z_real * z_real + z_imag * z_imag);
        float u_r = (mag > EPSILON) ? z_real / mag : 1.0f;
        float u_i = (mag > EPSILON) ? z_imag / mag : 0.0f;

        if (!has_coh) {
            coh_r = u_r;
            coh_i = u_i;
            has_coh = true;
        } else {
            coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
            coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;
        }

        R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
    }

    void computeRegularityOutput() {
        // Spectral narrowness → R_spec
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu = std::sqrt(ratio);
            R_spec = std::clamp(std::exp(-nu), 0.0f, 1.0f);
        } else {
            nu = 0.0f;
            R_spec = 0.0f;
        }

        R_safe = std::max(R_spec, R_phase);

        // Use blended omega for displacement normalization here too
        float omega_norm = 0.7f * omega_lp + 0.3f * omega_disp_lp;
        omega_norm = std::max(omega_norm, omega_min);
        float inv_w2 = 1.0f / std::max(omega_norm * omega_norm, EPSILON);

        float disp_real = z_real * inv_w2;
        float disp_imag = z_imag * inv_w2;
        float P_disp = disp_real * disp_real + disp_imag * disp_imag;

        // Tiny pre-LP for gating (about 2 s) to avoid flicker near thresholds
        const float tau_gate = 2.0f;
        const float alpha_gate = 1.0f - std::exp(-last_dt / std::max(1e-3f, tau_gate));
        if (!has_gate) { P_disp_gate = P_disp; has_gate = true; }
        else           { P_disp_gate = (1.0f - alpha_gate) * P_disp_gate + alpha_gate * P_disp; }

                // Smooth ν-based adjustment instead of P_disp thresholds
        // nu ≈ 0.0 (narrowband swell) → factor ≈ 1.0 (no change)
        // nu ≈ 0.5+ (broadband stormy) → factor ≈ 0.9 (10% reduction)
        const float nu_max  = 0.5f;   // treat this ν as "fully broadband"
        const float k_broad = 0.10f;  // maximum reduction fraction
        float broad_frac = std::clamp(nu / nu_max, 0.0f, 1.0f);
        float adj_factor = 1.0f - k_broad * broad_frac;

        float R_target = R_safe * adj_factor;

        // Final output smoothing (unchanged)
        if (!has_R_out) {
            R_out = R_target;
            has_R_out = true;
        } else {
            R_out = (1.0f - alpha_out) * R_out + alpha_out * R_target;
        }
    }

    // Gentler mapping from R to height factor to avoid twitchiness
    static float heightFactorFromR(float R_val) {
        if (R_val >= HEIGHT_R_HI) return 1.0f;
        if (R_val <= HEIGHT_R_LO) return std::sqrt(2.0f);
        float x = (HEIGHT_R_HI - R_val) / (HEIGHT_R_HI - HEIGHT_R_LO);
        const float p = 1.1f; // softer than 1.5
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::pow(x, p);
    }
};

#ifdef SEA_STATE_TEST
// ------------------- Simple sine-wave test harness -------------------
constexpr float SAMPLE_FREQ_HZ = 240.0f;
constexpr float DT = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 60.0f;
constexpr float SINE_AMPLITUDE = 1.0f;    // meters
constexpr float SINE_FREQ_HZ = 0.3f;      // Hz

struct SineWave {
    float amplitude;
    float freq_hz;
    float omega;
    float phi;

    SineWave(float A, float f_hz)
        : amplitude(A), freq_hz(f_hz),
          omega(2.0f * float(M_PI) * f_hz), phi(0.0f) {}

    std::pair<float,float> step(float dt) {
        phi += omega * dt;
        if (phi > 2.0f * float(M_PI)) phi -= 2.0f * float(M_PI);
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

// Run: compile with -DSEA_STATE_TEST and call SeaState_sine_wave_test() from main().
inline void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity reg; // uses safe defaults

    float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f;
    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        float z = za.first;
        float a = za.second;
        // Feed noisy-ish omega if you want: e.g., wave.omega + noise
        reg.update(DT, a, wave.omega);
        R_spec = reg.getRegularitySpectral();
        R_phase = reg.getRegularityPhase();
        Hs_est = reg.getWaveHeightEnvelopeEst();
        (void)z; // unused in this simple test
    }
    const float Hs_expected = 2.0f * SINE_AMPLITUDE;
    if (!(R_spec > 0.90f))
        throw std::runtime_error("Sine: R_spec did not converge near 1.");
    if (!(R_phase > 0.80f))
        throw std::runtime_error("Sine: R_phase did not converge near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.6f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within tolerance.");
    std::cerr << "[PASS] Sine wave test passed. Hs_est=" << Hs_est << " (expected ~" << Hs_expected << ")\n";
}
#endif // SEA_STATE_TEST
