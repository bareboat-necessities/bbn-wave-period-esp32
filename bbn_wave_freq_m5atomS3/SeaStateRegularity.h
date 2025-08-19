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
 * This class computes:
 *  - Spectral regularity R_spec (from spectral moments)
 *  - Phase coherence regularity R_phase
 *  - Harmonic-safe regularity R_safe
 *  - Smoothed output R_out
 *  - Approximate significant wave height estimate
 *  - Displacement frequency estimate
 *
 * The implementation is inspired by classical ocean wave theory:
 *  - Demodulate vertical acceleration to approximate displacement:
 *      z_real = a_z * cos(phi), z_imag = a_z * sin(phi)
 *      where phi = integral of instantaneous angular frequency
 *  - Compute spectral moments of displacement:
 *      M0 = <x^2>, M1 = <x^2 * omega>, M2 = <x^2 * omega^2>
 *  - Spectral regularity:
 *      nu = sqrt( (M0*M2)/(M1*M1) - 1 )
 *      R_spec = exp(-nu)
 *  - Phase coherence:
 *      R_phase = |<z_real + i*z_imag>|
 *  - Output smoothing and selective boost/reduction to handle
 *    nonlinear waves and moderate JONSWAP-like waves.
 */
class SeaStateRegularity {
public:
    // Configurable constants
    constexpr static float EPSILON = 1e-12f;                      // small number for stability
    constexpr static float HEIGHT_R_HI = 0.98f;                   // upper bound for height factor
    constexpr static float HEIGHT_R_LO = 0.50f;                   // lower bound for height factor
    constexpr static float BROADBAND_WAVE_THRESHOLD = 0.3f;       // broadband wave threshold
    constexpr static float BROADBAND_WAVE_REDUCTION_MAX = 0.08f;  // max reduction for broadband waves
    constexpr static float LARGE_WAVE_THRESHOLD = 0.5f;           // threshold for large nonlinear waves
    constexpr static float LARGE_WAVE_BOOST_MAX = 0.12f;          // max R boost

    /**
     * @brief Constructor
     * @param tau_env_sec  Envelope smoothing time (sec)
     * @param tau_mom_sec  Spectral moment smoothing time (sec)
     * @param omega_min_hz Minimum angular frequency (Hz) to avoid division by zero
     * @param tau_coh_sec  Phase coherence smoothing time (sec)
     * @param tau_out_sec  Output smoothing time (sec)
     * @param tau_omega_sec Optional frequency smoothing time (sec)
     */
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
        omega_min = 2.0f * float(M_PI) * omega_min_hz; // convert Hz to rad/s
        reset();
    }

    /**
     * @brief Reset internal state
     */
    void reset() {
        phi = z_real = z_imag = 0.0f;
        M0 = M1 = M2 = std::numeric_limits<float>::quiet_NaN();
        nu = R_spec = R_phase = R_safe = R_out = std::numeric_limits<float>::quiet_NaN();

        alpha_env = alpha_mom = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

        coh_r = 1.0f;
        coh_i = 0.0f;

        // frequency-related -> start as NaN to allow seeding on first update
        omega_lp      = std::numeric_limits<float>::quiet_NaN();
        omega_disp_lp = std::numeric_limits<float>::quiet_NaN();
    }

    /**
     * @brief Main update function (call every sample)
     * @param dt_s Time step in seconds
     * @param accel_z Vertical acceleration measurement (m/s^2)
     * @param omega_inst Instantaneous angular frequency estimate (rad/s)
     */
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!std::isfinite(dt_s) || dt_s <= 0.0f) return;
        if (!std::isfinite(accel_z)) accel_z = 0.0f;
        if (!std::isfinite(omega_inst)) omega_inst = omega_min;

        updateAlpha(dt_s);

        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityOutput();
    }

    // Getters
    float getNarrowness() const { return nu; }
    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    /**
     * @brief Estimate significant wave height from M0 and R
     */
    float getWaveHeightEnvelopeEst() const {
        if (M0 <= 0.0f || !std::isfinite(R_out)) return 0.0f;
        // Hs ≈ 2 * sqrt(M0) * heightFactor(R)
        return 2.0f * std::sqrt(M0) * heightFactorFromR(R_out);
    }

    /**
     * @brief Smoothed displacement frequency in Hz
     */
    float getDisplacementFrequencyHz() const { return omega_disp_lp / (2.0f * M_PI); }

private:
    float tau_env, tau_mom, tau_coh, tau_out, tau_omega;
    float omega_min;
    float last_dt;

    float alpha_env, alpha_mom, alpha_coh, alpha_out, alpha_omega;

    float phi;        // phase of demodulation
    float z_real;     // demodulated displacement real part
    float z_imag;     // demodulated displacement imaginary part

    float M0, M1, M2;  // spectral moments
    float nu;          // spectral narrowness
    float R_spec;      // spectral regularity
    float R_phase;     // phase-coherence regularity
    float R_safe;      // max(R_spec, R_phase)
    float R_out;       // smoothed output regularity

    float coh_r, coh_i;   // phase coherence accumulator
    float omega_lp;       // smoothed instantaneous frequency
    float omega_disp_lp;  // smoothed displacement frequency

    // Helper: compute smoothing coefficients
    void updateAlpha(float dt_s) {
        if (dt_s == last_dt) return;
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_out   = 1.0f - std::exp(-dt_s / tau_out);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
        last_dt = dt_s;
    }

    // Helper: demodulate acceleration to approximate displacement
    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        phi += omega_inst * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        // first-sample seeding
        if (!std::isfinite(omega_lp)) {
            z_real = y_real;
            z_imag = y_imag;
            omega_lp = std::max(omega_inst, omega_min);
            return; // skip EMA update on first sample
        }

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        float w_inst = std::max(omega_inst, omega_min);
        omega_lp = (tau_omega > 0.0f)
                   ? ((std::isfinite(omega_lp) ? (1.0f - alpha_omega) * omega_lp : 0.0f) + alpha_omega * w_inst)
                   : w_inst;
    }

    // Helper: update spectral moments (M0, M1, M2)
    void updateSpectralMoments() {
        float inv_w2 = 1.0f / (omega_lp * omega_lp);
        float disp_real = z_real * inv_w2;
        float disp_imag = z_imag * inv_w2;
        float P_disp = disp_real*disp_real + disp_imag*disp_imag;

        // first-sample seeding
        if (std::isnan(M0)) {
            M0 = P_disp;
            M1 = P_disp * omega_lp;
            M2 = P_disp * omega_lp * omega_lp;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * omega_lp;
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * omega_lp * omega_lp;
        }

        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (!std::isfinite(omega_disp_lp)) omega_disp_lp = omega_disp;
        else omega_disp_lp = (1.0f - alpha_out) * omega_disp_lp + alpha_out * omega_disp;
    }

    // Helper: compute phase coherence
    void updatePhaseCoherence() {
        float mag = std::sqrt(z_real*z_real + z_imag*z_imag);
        float u_r = (mag > EPSILON) ? z_real / mag : 1.0f;
        float u_i = (mag > EPSILON) ? z_imag / mag : 0.0f;

        coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
        coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;

        R_phase = std::clamp(std::sqrt(coh_r*coh_r + coh_i*coh_i), 0.0f, 1.0f);
    }

    // Helper: compute final regularity with boosts/reductions
    void computeRegularityOutput() {
        // spectral R
        if (M1 > EPSILON && M0>0.0f && M2>0.0f) {
            float ratio = (M0*M2)/(M1*M1) - 1.0f;
            ratio = std::max(0.0f, ratio);
            nu = std::sqrt(ratio);
            R_spec = std::clamp(std::exp(-nu), 0.0f, 1.0f);
        } else {
            nu = 0.0f;
            R_spec = 0.0f;
        }

        R_safe = std::max(R_spec, R_phase);

        float inv_w2 = 1.0f / (omega_lp * omega_lp);
        float disp_real = z_real * inv_w2;
        float disp_imag = z_imag * inv_w2;
        float P_disp = disp_real*disp_real + disp_imag*disp_imag;

        float R_target = R_safe;

        // reduce R for moderate waves (JONSWAP)
        if (P_disp < BROADBAND_WAVE_THRESHOLD) {
            float reduce = BROADBAND_WAVE_REDUCTION_MAX * (1.0f - P_disp / BROADBAND_WAVE_THRESHOLD);
            R_target = std::max(R_target - reduce, 0.0f);
        }

        // boost R for large nonlinear waves
        if (P_disp > LARGE_WAVE_THRESHOLD) {
            float boost = LARGE_WAVE_BOOST_MAX * std::min(P_disp / LARGE_WAVE_THRESHOLD, 2.0f);
            R_target = std::min(R_target + boost, 1.0f);
        }

        // first-sample seeding for R_out
        if (!std::isfinite(R_out)) R_out = R_target;
        else R_out = (1.0f - alpha_out) * R_out + alpha_out * R_target;
    }

    // Helper: map regularity R to height factor
    static float heightFactorFromR(float R_val) {
        if (!std::isfinite(R_val) || R_val >= HEIGHT_R_HI) return 1.0f;
        if (R_val <= HEIGHT_R_LO) return std::sqrt(2.0f);
        float x = (HEIGHT_R_HI - R_val) / (HEIGHT_R_HI - HEIGHT_R_LO);
        return 1.0f + (std::sqrt(2.0f)-1.0f) * std::pow(x, 1.5f);
    }
};

#ifdef SEA_STATE_TEST
// Constants
constexpr float SAMPLE_FREQ_HZ = 240.0f;
constexpr float DT = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 300.0f;
constexpr float SINE_AMPLITUDE = 1.0f;
constexpr float SINE_FREQ_HZ = 0.1f;

// Simple sine-wave generator
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
        if (phi > 2*M_PI) phi -= 2*M_PI;
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

// Test: pure sine wave
void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity reg;  // defaults
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
    if (!(R_phase > 0.95f))
        throw std::runtime_error("Sine: R_phase did not converge to ~1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.1f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within 10%.");
    std::cout << "[PASS] Sine wave test passed.\n";
}
#endif // SEA_STATE_TEST
