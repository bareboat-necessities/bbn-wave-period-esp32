#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * SeaStateRegularity — Online estimator of ocean wave regularity and height.
 *
 * Inputs
 *   • Vertical acceleration a_z(t) [m/s²]
 *   • Instantaneous angular frequency ω_inst(t) [rad/s] from an external tracker
 *
 * Pipeline
 *   1) Demodulate acceleration by φ(t) = ∫ ω_inst dt to obtain baseband z(t).
 *   2) Normalize by ω²: η_env(t) = z(t)/ω²(t) to approximate displacement envelope.
 *   3) Track power and displacement spectral moments with bias-corrected EMAs (DebiasedEMA):
 *        A0 = ⟨|z|²⟩                (acceleration envelope variance, diagnostic)
 *        M0 = ⟨Y⟩                   (displacement variance proxy, scaled — see K_ENV)
 *        M1 = ⟨Y·ω⟩
 *        M2 = ⟨Y·ω²⟩
 *      (Q0,Q1,Q2 = ⟨Y²⟩, ⟨Y²·ω⟩, ⟨Y²·ω²⟩ retained, but NOT used in the current output
 *       because Jensen delta-method is disabled pending re-verification.)
 *   4) Mean frequency ω̄ and variance μ₂ are computed from the NAIVE moments:
 *        ω̄ = M1/M0,   μ₂ = M2/M0 − (M1/M0)²
 *   5) Spectral regularity (spectral-only): R_spec = exp(−β · ν),  ν = √μ₂/ω̄.
 *   6) Phase regularity (diagnostic): mean resultant length of demodulated phase.
 *   7) Final regularity: R_out = EMA(R_spec)  (no fusion with phase).
 *   8) Significant wave height: Hs ≈ 4√M0, using the scaled M0.
 *
 * Notes
 *   • K_ENV=2 accounts for I/Q envelope-to-variance conversion so that M0 is on
 *     the same scale as displacement variance for narrowband content.
 *   • To reduce tracker dependence, ω_inst is used for DEMODULATION only.
 *     Moments use a blended ω_used that favors our own mean (ω̄) when available.
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

// ---------------- Debiased EMA ----------------
struct DebiasedEMA {
    float value  = 0.0f;
    float weight = 0.0f;
    void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float alpha) {
        value  = (1.0f - alpha) * value + alpha * x;
        weight = (1.0f - alpha) * weight + alpha;
    }
    inline float get() const { return (weight > 1e-12f) ? value / weight : 0.0f; }
    inline bool  isReady() const { return weight > 1e-6f; }
};

// ---------------- SeaStateRegularity ----------------
class SeaStateRegularity {
public:
    // Numerics / mapping
    constexpr static float EPSILON    = 1e-12f;
    constexpr static float BETA_SPEC  = 1.0f;     // linear exponent in ν
    constexpr static float K_ENV      = 2.0f;     // envelope->variance power scale

    // Tracker-robust ω used for moments (blend + clamp)
    constexpr static float GAMMA_BLEND = 0.80f;   // weight for our ω̂ vs. instant ω
    constexpr static float OMEGA_MIN_HZ = 0.03f;  // 0.03 Hz
    constexpr static float OMEGA_MAX_HZ = 1.50f;  // 1.5 Hz

    SeaStateRegularity(float tau_env_sec = 15.0f,
                       float tau_mom_sec = 180.0f,
                       float tau_coh_sec = 60.0f,
                       float tau_out_sec = 30.0f)
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
        nu = 0.0f;

        omega_bar_corr = 0.0f;
        omega_bar_naive = 0.0f;
        omega_used_prev = 0.0f;

        has_moments = false;
        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;
    }

    // Main update
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updatePhaseCoherence();
        updateSpectralMoments(omega_inst);   // robust ω_used inside
        computeRegularityOutput();           // NAIVE μ₂/ω̄ mapping (no Jensen)
    }

    // Getters
    float getRegularity() const { return R_out.get(); }              // spectral-only, smoothed
    float getRegularitySpectral() const { return R_spec; }           // instantaneous spectral R
    float getRegularityPhase() const { return R_phase; }             // diagnostic only
    float getNarrowness() const { return nu; }                       // ν = √μ₂/ω̄

    float getWaveHeightEnvelopeEst() const {                        // Hs ≈ 4√M0
        float m0 = M0.get();
        return (m0 > 0.0f) ? 4.0f * std::sqrt(m0) : 0.0f;
    }

    float getDisplacementFrequencyHz() const {                      // Jensen OFF: returns ω̄_naive/(2π)
        return (omega_bar_corr > EPSILON) ? (omega_bar_corr / (2.0f * float(M_PI))) : 0.0f;
    }

    float getDisplacementFrequencyNaiveHz() const {                 // same as above while Jensen OFF
        return (omega_bar_naive > EPSILON) ? (omega_bar_naive / (2.0f * float(M_PI))) : 0.0f;
    }

    float getDisplacementPeriodSec() const {                        // Tp = 2π/ω̄
        return (omega_bar_corr > EPSILON) ? (2.0f * float(M_PI) / omega_bar_corr) : 0.0f;
    }

    float getAccelerationVariance() const { return A0.get(); }      // diagnostic

private:
    // time constants and alphas
    float tau_env, tau_mom, tau_coh, tau_out;
    float last_dt;
    float alpha_env, alpha_mom, alpha_coh, alpha_out;

    // demod state
    float phi;
    float z_real, z_imag;

    // moments (bias-corrected EMAs)
    DebiasedEMA M0, M1, M2;
    DebiasedEMA Q0, Q1, Q2;   // retained for future Jensen re-enable
    DebiasedEMA A0;

    // coherence + output
    DebiasedEMA coh_r, coh_i;
    DebiasedEMA R_out;
    float R_spec, R_phase;

    // cached
    float nu;                 // √μ₂/ω̄
    float omega_bar_corr;     // currently equals ω̄_naive (Jensen off)
    float omega_bar_naive;    // M1/M0
    float omega_used_prev;    // for debugging / potential future smoothing
    bool  has_moments;

    // helpers
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

        // Quadrature baseband (acceleration envelope)
        float y_real =  accel_z * c;
        float y_imag = -accel_z * s;

        if (!has_moments) {
            z_real = y_real;
            z_imag = y_imag;
            return;
        }
        // EMA for envelope
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;
    }

    void updateSpectralMoments(float omega_inst) {
        // --- Build ω_used (robust to tracker idiosyncrasies) ---
        const float OMEGA_MIN = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
        const float OMEGA_MAX = 2.0f * float(M_PI) * OMEGA_MAX_HZ;

        // Use our own mean when available; otherwise fall back to instant ω
        float omega_hat = (omega_bar_corr > 0.0f) ? omega_bar_corr : omega_inst;
        float omega_used = GAMMA_BLEND * omega_hat + (1.0f - GAMMA_BLEND) * omega_inst;
        omega_used = std::clamp(omega_used, OMEGA_MIN, OMEGA_MAX);
        omega_used_prev = omega_used;

        // --- Accel envelope power (diagnostic) ---
        float P_acc = z_real * z_real + z_imag * z_imag;
        A0.update(P_acc, alpha_mom);

        // --- Displacement envelope via 1/ω² and scale to variance proxy ---
        float inv_w2 = 1.0f / std::max(omega_used * omega_used, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;

        // Envelope power; scale so that M0 tracks displacement variance
        float P_disp = disp_r * disp_r + disp_i * disp_i;
        float Y      = K_ENV * P_disp;            // first order (variance proxy)
        float Y2     = (K_ENV * K_ENV) * P_disp * P_disp; // second order

        if (!has_moments) has_moments = true;

        // --- Debiased EMA updates for moments ---
        M0.update(Y,                   alpha_mom);
        M1.update(Y * omega_used,      alpha_mom);
        M2.update(Y * omega_used * omega_used, alpha_mom);

        // Keep Q* updated for future Jensen correction (currently not used)
        Q0.update(Y2,                         alpha_mom);
        Q1.update(Y2 * omega_used,            alpha_mom);
        Q2.update(Y2 * omega_used * omega_used, alpha_mom);
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            // Keep prior running estimate if present; else 0
            if (coh_r.isReady() && coh_i.isReady())
                R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()), 0.0f, 1.0f);
            else
                R_phase = 0.0f;
            return;
        }
        float u_r = z_real / mag;
        float u_i = z_imag / mag;
        coh_r.update(u_r, alpha_coh);
        coh_i.update(u_i, alpha_coh);
        R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()), 0.0f, 1.0f);
    }

    void computeRegularityOutput() {
        if (!M0.isReady()) {
            // Early startup: just smooth diagnostic phase
            R_out.update(R_phase, alpha_out);
            R_spec = R_phase;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        // --- Naive moments (Jensen disabled) ---
        float m0 = M0.get();
        float m1 = M1.get();
        float m2 = M2.get();

        omega_bar_naive  = (m0 > EPSILON) ? (m1 / m0) : 0.0f;
        float omega2_bar = (m0 > EPSILON) ? (m2 / m0) : 0.0f;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar_naive * omega_bar_naive);
        omega_bar_corr = omega_bar_naive; // while Jensen is OFF

        float rbw = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2) / omega_bar_corr) : 0.0f;

        // Spectral-only, linear exponent
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);
        nu = rbw;

        R_out.update(R_spec, alpha_out);
    }
};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

/**
 * Simple sine test (monochromatic):
 *  - R_spec → ~1, R_phase → ~1
 *  - Hs ≈ 2√2·A (oceanographic)
 *  - ν → ~0
 *  - f and Tp correct
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
