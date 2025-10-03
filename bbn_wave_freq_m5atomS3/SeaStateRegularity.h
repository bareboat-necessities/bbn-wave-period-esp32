#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * SeaStateRegularity — Online estimator of ocean wave regularity and height.
 *
 * Inputs
 *   • Vertical acceleration a_z(t) [m/s²]
 *   • Instantaneous angular frequency ω_inst(t) [rad/s] (used ONLY for demodulation)
 *
 * Pipeline
 *   1) Demodulate acceleration by φ(t) = ∫ ω_inst dt → baseband z(t) (accel envelope, complex).
 *   2) Convert to displacement envelope with ω̂_prev:
 *        η_env ≈ z / ω̂_prev²  (tracker-independent conversion)
 *   3) Track power with bias-corrected EMAs:
 *        A0 = ⟨|z|²⟩               (acceleration envelope variance proxy)
 *        M0 = ⟨K_ENV·|η_env|²⟩     (displacement variance; K_ENV=2 for I/Q→variance)
 *   4) Self-estimate mean frequency from ratio (tracker-free):
 *        ω̂ = (A0/M0)^{1/4},  clamped to [2π·f_min, 2π·f_max]
 *      Track E[ω̂] and E[ω̂²] via debiased EMA → Var[ω̂].
 *   5) Bandwidth ν = sqrt(Var[ω̂]) / E[ω̂];  Spectral regularity: R_spec = exp(−β·ν).
 *   6) Output smoothing (bias-corrected EMA) on R_spec.
 *   7) Significant wave height: Hs ≈ 4√M0.
 *
 * Notes
 *   • ω_inst is NOT used in Hs/Tp/ν after demod; it only creates a stable baseband.
 *   • This removes tracker dependence, stabilizes Tp, and lowers R for PM/JONSWAP
 *     without per-sea tuning.
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

class SeaStateRegularity {
public:
    // Numerics / mapping
    constexpr static float EPSILON   = 1e-12f;
    constexpr static float BETA_SPEC = 1.0f;   // linear exponent in ν
    constexpr static float K_ENV     = 2.0f;   // I/Q envelope → variance (constant)

    // Frequency bounds (Hz) for ω̂ clamping; keep sane ocean band
    constexpr static float OMEGA_MIN_HZ = 0.02f; // 20 s
    constexpr static float OMEGA_MAX_HZ = 2.00f; // 0.5 s

    SeaStateRegularity(float tau_env_sec = 15.0f,
                       float tau_mom_sec = 180.0f,
                       float tau_coh_sec = 60.0f,
                       float tau_out_sec = 30.0f)
    {
        tau_env = tau_env_sec;
        tau_mom = tau_mom_sec;   // used for A0/M0 and ω̂ stats
        tau_coh = std::max(1e-3f, tau_coh_sec);
        tau_out = std::max(1e-3f, tau_out_sec);
        reset();
    }

    void reset() {
        phi = 0.0f;
        z_real = z_imag = 0.0f;

        // Power EMAs
        A0.reset();
        M0.reset();

        // ω̂ stats
        w_mean.reset();
        w_sq.reset();

        // Phase coherence (diagnostic)
        coh_r.reset(); coh_i.reset();

        // Output smoothing
        R_out.reset();

        // Scalars
        R_spec = R_phase = 0.0f;
        nu = 0.0f;

        omega_hat = 0.0f;          // current ω̂ (rad/s)
        omega_hat_prev = 0.0f;     // previous ω̂ used in demod→disp conversion

        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;

        // For API compatibility
        omega_bar_corr  = 0.0f;
        omega_bar_naive = 0.0f;
    }

    // Main update
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updatePhaseCoherence();

        // Displacement envelope using *previous* ω̂ (tracker-independent)
        float w_use = omega_hat_prev;
        if (w_use <= 0.0f) {
            // Bootstrap once with ω_inst (clamped) until we get our own ω̂
            const float WMIN = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
            const float WMAX = 2.0f * float(M_PI) * OMEGA_MAX_HZ;
            w_use = std::clamp(omega_inst, WMIN, WMAX);
        }
        updatePowerMomentsWithOmega(w_use);

        // Self-estimate ω̂ from ratio A0/M0 (tracker-free), then smooth & clamp
        updateSelfOmega();

        // Prepare next-step conversion: use smoothed ω̂
        omega_hat_prev = omega_hat;

        // Regularity and outputs
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

    // demod state (acceleration baseband)
    float phi;
    float z_real, z_imag;

    // power EMAs
    DebiasedEMA A0;   // ⟨|z|²⟩
    DebiasedEMA M0;   // ⟨K_ENV · |η_env|²⟩

    // ω̂ stats
    DebiasedEMA w_mean;   // ⟨ω̂⟩
    DebiasedEMA w_sq;     // ⟨ω̂²⟩
    float omega_hat;      // current ω̂ (smoothed, clamped)
    float omega_hat_prev; // used to form η_env

    // phase coherence (diagnostic)
    DebiasedEMA coh_r, coh_i;
    float R_phase;

    // output
    DebiasedEMA R_out;
    float R_spec;
    float nu;

    // API compatibility
    float omega_bar_corr;   // expose ω̂ here
    float omega_bar_naive;  // same as ω̂ (no Jensen moments now)

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

        // Accel baseband (I/Q)
        float y_real =  accel_z * c;
        float y_imag = -accel_z * s;

        // EMA of envelope
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // Phase coherence (diagnostic)
        updatePhaseCoherence();
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            if (coh_r.isReady() && coh_i.isReady()) {
                R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()), 0.0f, 1.0f);
            } else {
                R_phase = 0.0f;
            }
            return;
        }
        float u_r = z_real / mag;
        float u_i = z_imag / mag;
        coh_r.update(u_r, alpha_coh);
        coh_i.update(u_i, alpha_coh);
        R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()), 0.0f, 1.0f);
    }

    void updatePowerMomentsWithOmega(float w_use) {
        // Acceleration envelope power
        float P_acc = z_real * z_real + z_imag * z_imag;
        A0.update(P_acc, alpha_mom);

        // Displacement envelope via 1/w_use^2, then power
        float inv_w2 = 1.0f / std::max(w_use * w_use, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;

        // Constant envelope→variance scale
        float Y = K_ENV * P_disp;
        M0.update(Y, alpha_mom);
    }

    void updateSelfOmega() {
        // ω̂_raw from ratio (A0/M0)^{1/4}
        float a0 = A0.get();
        float m0 = M0.get();
        if (a0 <= EPSILON || m0 <= EPSILON) {
            // Keep prior ω̂ if not ready
            if (!w_mean.isReady()) {
                omega_hat = 0.0f;
                omega_bar_corr = omega_bar_naive = 0.0f;
            }
            return;
        }

        float w_raw = std::pow(std::max(a0 / m0, EPSILON), 0.25f);
        const float WMIN = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
        const float WMAX = 2.0f * float(M_PI) * OMEGA_MAX_HZ;
        w_raw = std::clamp(w_raw, WMIN, WMAX);

        // Smooth ω̂ and track its variance (debiased EMAs)
        w_mean.update(w_raw, alpha_mom);
        w_sq.update(w_raw * w_raw, alpha_mom);

        float mu_w  = w_mean.get();
        float Ew2   = w_sq.get();
        float var_w = std::max(0.0f, Ew2 - mu_w * mu_w);

        omega_hat = mu_w;
        omega_bar_corr = omega_hat;   // expose via API
        omega_bar_naive = omega_hat;  // same here
        nu = (omega_hat > EPSILON) ? (std::sqrt(var_w) / omega_hat) : 0.0f;
    }

    void computeRegularityOutput() {
        // Spectral-only mapping from ν
        R_spec = std::clamp(std::exp(-BETA_SPEC * std::max(0.0f, nu)), 0.0f, 1.0f);
        R_out.update(R_spec, alpha_out);
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
              << ", f_disp=" << f_disp_corr << " Hz"
              << ", Tp=" << Tp << " s\n";
}
#endif
