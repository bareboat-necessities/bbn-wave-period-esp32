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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Configurable constants
    constexpr static float EPSILON = 1e-12f;
    constexpr static float HEIGHT_R_HI = 0.98f;
    constexpr static float HEIGHT_R_LO = 0.50f;
    constexpr static float BROADBAND_WAVE_THRESHOLD = 0.3f;
    constexpr static float BROADBAND_WAVE_REDUCTION_MAX = 0.08f;
    constexpr static float LARGE_WAVE_THRESHOLD = 0.5f;
    constexpr static float LARGE_WAVE_BOOST_MAX = 0.12f;

    SeaStateRegularity(float tau_env_sec   = 1.0f,
                       float tau_mom_sec   = 60.0f,
                       float omega_min_hz  = 0.03f,
                       float tau_coh_sec   = 20.0f,
                       float tau_out_sec   = 15.0f,
                       float tau_omega_sec = 0.0f)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_out   = std::max(1e-3f, tau_out_sec);
        tau_omega = std::max(0.0f, tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz; // Hz → rad/s
        reset();
    }

    void reset() {
        phi = z_real = z_imag = 0.0f;

        M0 = M1 = M2 = 0.0f;
        nu = 0.0f;
        R_spec = R_phase = R_safe = R_out = 0.0f;

        alpha_env = alpha_mom = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

        coh_r = coh_i = 0.0f;

        omega_lp = omega_disp_lp = 0.0f;
        omega_last = mu_w = 0.0f;

        var_fast = var_slow = 0.0f;

        // explicit flags (replace NaN sentinels)
        has_omega_lp = false;
        has_omega_disp_lp = false;
        has_moments = false;
        has_coh = false;
        has_R_out = false;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        // robust NaN checks: x!=x is true iff NaN
        if (!(dt_s > 0.0f)) return; // also filters NaN
        if (accel_z != accel_z) accel_z = 0.0f;
        if (omega_inst != omega_inst) omega_inst = omega_min;

        updateAlpha(dt_s);

        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityOutput();
    }

    float getNarrowness() const { return nu; }
    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    float getWaveHeightEnvelopeEst() const {
        if (M0 <= 0.0f || !has_R_out) return 0.0f;
        return 2.0f * std::sqrt(M0) * heightFactorFromR(R_out);
    }

    float getDisplacementFrequencyHz() const { return omega_disp_lp / (2.0f * M_PI); }

private:
    float tau_env, tau_mom, tau_coh, tau_out, tau_omega;
    float omega_min;
    float last_dt;

    float alpha_env, alpha_mom, alpha_coh, alpha_out, alpha_omega;

    float phi;
    float z_real, z_imag;

    float M0, M1, M2;
    float nu;
    float R_spec, R_phase, R_safe, R_out;

    float coh_r, coh_i;
    float omega_lp, omega_disp_lp;
    float omega_last, mu_w;
    float var_fast, var_slow;

    bool has_omega_lp;
    bool has_omega_disp_lp;
    bool has_moments;
    bool has_coh;
    bool has_R_out;

    constexpr static float ALPHA_FAST = 0.1f;
    constexpr static float ALPHA_SLOW = 0.01f;

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
        phi += omega_inst * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        if (!has_omega_lp) {
            z_real = y_real;
            z_imag = y_imag;
            omega_lp = std::max(omega_inst, omega_min);
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        float w_inst = std::max(omega_inst, omega_min);
        if (tau_omega > 0.0f) {
            omega_lp = (1.0f - alpha_omega) * omega_lp + alpha_omega * w_inst;
        } else {
            omega_lp = w_inst;
        }

        float delta = omega_lp - mu_w;
        mu_w += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
        float inv_w2 = 1.0f / std::max(omega_lp * omega_lp, EPSILON);
        float disp_real_corr = z_real * inv_w2;
        float disp_imag_corr = z_imag * inv_w2;
        float P_disp_corr = disp_real_corr * disp_real_corr + disp_imag_corr * disp_imag_corr;

        if (!has_moments) {
            M0 = P_disp_corr;
            M1 = P_disp_corr * omega_lp;
            M2 = P_disp_corr * omega_lp * omega_lp;
            has_moments = true;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp_corr;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp_corr * omega_lp;
            float M2_candidate = P_disp_corr * omega_lp * omega_lp - var_slow * M0;
            M2_candidate = std::max(M2_candidate, 0.0f);
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * M2_candidate;
        }

        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
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
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            ratio = std::max(0.0f, ratio);
            nu = std::sqrt(ratio);
            R_spec = std::clamp(std::exp(-nu), 0.0f, 1.0f);
        } else {
            nu = 0.0f;
            R_spec = 0.0f;
        }

        R_safe = std::max(R_spec, R_phase);

        float inv_w2 = 1.0f / std::max(omega_lp * omega_lp, EPSILON);
        float disp_real = z_real * inv_w2;
        float disp_imag = z_imag * inv_w2;
        float P_disp = disp_real * disp_real + disp_imag * disp_imag;

        float R_target = R_safe;

        if (P_disp < BROADBAND_WAVE_THRESHOLD) {
            float reduce = BROADBAND_WAVE_REDUCTION_MAX * (1.0f - P_disp / BROADBAND_WAVE_THRESHOLD);
            R_target = std::max(R_target - reduce, 0.0f);
        }

        if (P_disp > LARGE_WAVE_THRESHOLD) {
            float boost = LARGE_WAVE_BOOST_MAX * std::min(P_disp / LARGE_WAVE_THRESHOLD, 2.0f);
            R_target = std::min(R_target + boost, 1.0f);
        }

        if (!has_R_out) {
            R_out = R_target;
            has_R_out = true;
        } else {
            R_out = (1.0f - alpha_out) * R_out + alpha_out * R_target;
        }
    }

    static float heightFactorFromR(float R_val) {
        if (R_val >= HEIGHT_R_HI) return 1.0f;
        if (R_val <= HEIGHT_R_LO) return std::sqrt(2.0f);
        float x = (HEIGHT_R_HI - R_val) / (HEIGHT_R_HI - HEIGHT_R_LO);
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::pow(x, 1.5f);
    }
};

#ifdef SEA_STATE_TEST
constexpr float SAMPLE_FREQ_HZ = 240.0f;
constexpr float DT = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 300.0f;
constexpr float SINE_AMPLITUDE = 1.0f;
constexpr float SINE_FREQ_HZ = 0.3f;

struct SineWave {
    float amplitude;
    float freq_hz;
    float omega;
    float phi;

    SineWave(float A, float f_hz)
        : amplitude(A), freq_hz(f_hz),
          omega(2.0f * M_PI * f_hz), phi(0.0f) {}

    std::pair<float,float> step(float dt) {
        phi += omega * dt;
        if (phi > 2 * M_PI) phi -= 2 * M_PI;
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity reg;
    float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f;
    for (int i = 0; i < SIM_DURATION_SEC / DT; i++) {
        auto [z, a] = wave.step(DT);
        reg.update(DT, a, wave.omega);
        R_spec = reg.getRegularitySpectral();
        R_phase = reg.getRegularityPhase();
        Hs_est = reg.getWaveHeightEnvelopeEst();
    }
    const float Hs_expected = 2.0f * SINE_AMPLITUDE;
    if (!(R_spec > 0.95f))
        throw std::runtime_error("Sine: R_spec did not converge to ~1.");
    if (!(R_phase > 0.85f))
        throw std::runtime_error("Sine: R_phase did not converge to ~1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.5f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within 50%.");
    std::cerr << "[PASS] Sine wave test passed.\n";
}
#endif // SEA_STATE_TEST
