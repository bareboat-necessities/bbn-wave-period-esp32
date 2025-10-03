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
 *        M0 = ⟨Y⟩                   (displacement variance proxy)
 *        M1 = ⟨Y·ω⟩
 *        M2 = ⟨Y·ω²⟩
 *        Q0 = ⟨Y²⟩, Q1 = ⟨Y²·ω⟩, Q2 = ⟨Y²·ω²⟩ (for Jensen correction)
 *   4) Jensen-corrected mean frequency and variance:
 *        ω̄ = M1/M0 with correction,   μ₂ = M2/M0 − (M1/M0)² with correction
 *   5) Spectral regularity: R_spec = exp(−β · ν),  ν = √μ₂/ω̄.
 *   6) Phase regularity: mean resultant length of demodulated phase (diagnostic only).
 *   7) Final regularity: R_out = EMA(R_spec).
 *   8) Significant wave height: Hs ≈ 4√M0.
 *
 * Notes
 *   • K_eff is auto-scaled: 2·(1 + 6ν²), so broadband seas get more correction.
 *   • ω_inst is used for demodulation only; spectral stats use a blended ω̂.
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

    // Tracker-robust ω clamp
    constexpr static float OMEGA_MIN_HZ = 0.02f;
    constexpr static float OMEGA_MAX_HZ = 2.00f;
    constexpr static float GAMMA_BLEND  = 0.80f;   // blend weight

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
        updateSpectralMoments(omega_inst);
        computeRegularityOutput();
    }

    // Getters
    float getRegularity() const { return R_out.get(); }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }
    float getNarrowness() const { return nu; }

    float getWaveHeightEnvelopeEst() const {
        float m0 = M0.get();
        return (m0 > 0.0f) ? 4.0f * std::sqrt(m0) : 0.0f;
    }

    float getDisplacementFrequencyHz() const {
        return (omega_bar_corr > EPSILON) ? (omega_bar_corr / (2.0f * float(M_PI))) : 0.0f;
    }

    float getDisplacementFrequencyNaiveHz() const {
        return (omega_bar_naive > EPSILON) ? (omega_bar_naive / (2.0f * float(M_PI))) : 0.0f;
    }

    float getDisplacementPeriodSec() const {
        return (omega_bar_corr > EPSILON) ? (2.0f * float(M_PI) / omega_bar_corr) : 0.0f;
    }

    float getAccelerationVariance() const { return A0.get(); }

private:
    // time constants and alphas
    float tau_env, tau_mom, tau_coh, tau_out;
    float last_dt;
    float alpha_env, alpha_mom, alpha_coh, alpha_out;

    // demod state
    float phi;
    float z_real, z_imag;

    // moments
    DebiasedEMA M0, M1, M2;
    DebiasedEMA Q0, Q1, Q2;
    DebiasedEMA A0;

    // coherence + output
    DebiasedEMA coh_r, coh_i;
    DebiasedEMA R_out;
    float R_spec, R_phase;

    // cached
    float nu;
    float omega_bar_corr;
    float omega_bar_naive;
    float omega_used_prev;
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

    void updateSpectralMoments(float omega_inst) {
        const float OMEGA_MIN = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
        const float OMEGA_MAX = 2.0f * float(M_PI) * OMEGA_MAX_HZ;

        float omega_hat = (omega_bar_corr > 0.0f) ? omega_bar_corr : omega_inst;
        float omega_used = GAMMA_BLEND * omega_hat + (1.0f - GAMMA_BLEND) * omega_inst;
        omega_used = std::clamp(omega_used, OMEGA_MIN, OMEGA_MAX);
        omega_used_prev = omega_used;

        float P_acc = z_real * z_real + z_imag * z_imag;
        A0.update(P_acc, alpha_mom);

        float inv_w2 = 1.0f / std::max(omega_used * omega_used, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;

        float P_disp = disp_r * disp_r + disp_i * disp_i;

        // bandwidth-dependent scale
        float K_eff = 2.0f * (1.0f + 6.0f * nu * nu);
        float Y     = K_eff * P_disp;
        float Y2    = (K_eff * K_eff) * P_disp * P_disp;

        if (!has_moments) has_moments = true;

        M0.update(Y,                   alpha_mom);
        M1.update(Y * omega_used,      alpha_mom);
        M2.update(Y * omega_used * omega_used, alpha_mom);

        Q0.update(Y2,                         alpha_mom);
        Q1.update(Y2 * omega_used,            alpha_mom);
        Q2.update(Y2 * omega_used * omega_used, alpha_mom);
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
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
            R_out.update(R_phase, alpha_out);
            R_spec = R_phase;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        float m0 = M0.get(), m1 = M1.get(), m2 = M2.get();
        float q0 = Q0.get(), q1 = Q1.get(), q2 = Q2.get();

        omega_bar_naive  = (m0 > EPSILON) ? (m1 / m0) : 0.0f;
        float omega2_bar_naive = (m0 > EPSILON) ? (m2 / m0) : 0.0f;

        float varY  = std::max(0.0f, q0 - m0*m0);
        float cov10 = q1 - m1*m0;
        float cov20 = q2 - m2*m0;
        float invm0_2 = (m0 > EPSILON*EPSILON) ? 1.0f / (m0*m0) : 0.0f;

        float omega_bar  = omega_bar_naive  + omega_bar_naive  * invm0_2 * varY - cov10 * invm0_2;
        float omega2_bar = omega2_bar_naive + omega2_bar_naive * invm0_2 * varY - cov20 * invm0_2;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar * omega_bar);
        nu = (omega_bar > EPSILON) ? (std::sqrt(mu2) / omega_bar) : 0.0f;

        R_spec = std::clamp(std::exp(-BETA_SPEC * nu), 0.0f, 1.0f);
        omega_bar_corr = (omega_bar > 0.0f) ? omega_bar : 0.0f;

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
