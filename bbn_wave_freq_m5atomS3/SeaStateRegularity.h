#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * SeaStateRegularity — Online estimator of ocean wave regularity and height.
 *
 * (1) Demod with a single, heavily-smoothed ω_used (tracker-only EMA + clamp).
 * (2) Convert acceleration envelope to displacement proxy via 1/ω_used²,
 *     so power scales as 1/ω_used⁴ (physics-consistent).
 * (3) Track M0,M1,M2 with debiased EMAs; Jensen OFF for robustness.
 * (4) R_spec = exp(-β·ν), ν = sqrt(μ2)/ω̄ ; Hs = 4√M0 ; Tp = 2π/ω̄.
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
    constexpr static float EPSILON   = 1e-12f;
    constexpr static float BETA_SPEC = 1.0f;    // linear exponent in ν
    constexpr static float K_EFF     = 2.0f;    // envelope→variance (narrowband analytic)

    // Tracker smoothing (hardcoded)
    constexpr static float OMEGA_MIN_HZ = 0.03f;
    constexpr static float OMEGA_MAX_HZ = 1.50f;
    constexpr static float TAU_W_SEC    = 30.0f; // strong smoothing for ω_used

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
        A0.reset();

        R_out.reset();
        coh_r.reset(); coh_i.reset();

        R_spec = R_phase = 0.0f;
        nu = 0.0f;

        omega_bar_corr = 0.0f;
        omega_bar_naive = 0.0f;

        omega_used = 0.0f;
        alpha_w    = 0.0f;

        has_moments = false;
        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        // Update all alphas, incl. ω EMA
        updateAlpha(dt_s);

        // Build ω_used (EMA of clamped tracker) BEFORE demod and use it everywhere
        const float OMEGA_MIN = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
        const float OMEGA_MAX = 2.0f * float(M_PI) * OMEGA_MAX_HZ;
        float w_obs = std::clamp(omega_inst, OMEGA_MIN, OMEGA_MAX);
        if (omega_used <= 0.0f) omega_used = w_obs;
        else                    omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

        // DEMOD with ω_used (not ω_inst) to avoid baseband leakage
        demodulateAcceleration_withUsedOmega(accel_z, omega_used, dt_s);

        // Phase coherence (diagnostic only)
        updatePhaseCoherence();

        // Moments using the same ω_used for 1/ω² and weights
        updateSpectralMoments_withUsedOmega();

        // Regularity / Hs / Tp from naive moments
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
    float last_dt{};
    float alpha_env{}, alpha_mom{}, alpha_coh{}, alpha_out{};
    float alpha_w{}; // ω EMA

    // ω_used (single source of truth)
    float omega_used{};

    // demod state
    float phi{};
    float z_real{}, z_imag{};

    // moments
    DebiasedEMA M0, M1, M2; // displacement moments
    DebiasedEMA A0;         // accel envelope power (diagnostic)

    // coherence + output
    DebiasedEMA coh_r, coh_i;
    DebiasedEMA R_out;
    float R_spec{}, R_phase{};

    // cached
    float nu{};
    float omega_bar_corr{};
    float omega_bar_naive{};
    bool  has_moments{false};

    void updateAlpha(float dt_s) {
        if (dt_s == last_dt) return;
        alpha_env = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_out = 1.0f - std::exp(-dt_s / tau_out);
        alpha_w   = 1.0f - std::exp(-dt_s / TAU_W_SEC);
        last_dt = dt_s;
    }

    void demodulateAcceleration_withUsedOmega(float accel_z, float w_used, float dt_s) {
        phi += w_used * dt_s;                                  // phase from the same ω_used
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

    void updateSpectralMoments_withUsedOmega() {
        // accel envelope power (diagnostic)
        float P_acc = z_real * z_real + z_imag * z_imag;
        A0.update(P_acc, alpha_mom);

        // displacement envelope via 1/ω_used²  → power scales as 1/ω_used⁴
        float inv_w2 = 1.0f / std::max(omega_used * omega_used, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;

        float Y  = K_EFF * P_disp;                 // variance proxy (narrowband-correct)
        if (!has_moments) has_moments = true;

        // same ω_used for weights
        M0.update(Y,                          alpha_mom);
        M1.update(Y * omega_used,             alpha_mom);
        M2.update(Y * omega_used * omega_used,alpha_mom);
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            R_phase = (coh_r.isReady() && coh_i.isReady())
                      ? std::clamp(std::sqrt(coh_r.get()*coh_r.get() + coh_i.get()*coh_i.get()), 0.0f, 1.0f)
                      : 0.0f;
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

        // Naive moments (Jensen OFF) with tiny floor on m0 to avoid spikes
        float m0 = M0.get();
        float m1 = M1.get();
        float m2 = M2.get();

        const float m0_floor = 1e-10f; // small, just to prevent ratio blow-ups in very calm/skip phases
        if (m0 < m0_floor) {
            R_out.update(0.0f, alpha_out);
            R_spec = 0.0f;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        omega_bar_naive  = m1 / m0;
        float omega2_bar = m2 / m0;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar_naive * omega_bar_naive);
        omega_bar_corr = omega_bar_naive; // Jensen OFF

        nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2) / omega_bar_corr) : 0.0f;

        R_spec = std::clamp(std::exp(-BETA_SPEC * nu), 0.0f, 1.0f);
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
    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        reg.update(DT, za.second, wave.omega);
    }
    // smoke checks
    float R_spec = reg.getRegularitySpectral();
    float Hs_est = reg.getWaveHeightEnvelopeEst();
    float nu     = reg.getNarrowness();
    float f_disp = reg.getDisplacementFrequencyHz();
    float Tp     = reg.getDisplacementPeriodSec();

    const float Hs_expected = 2.0f * std::sqrt(2.0f) * SINE_AMPLITUDE;
    if (!(R_spec > 0.90f)) throw std::runtime_error("Sine: R_spec too low.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.6f * Hs_expected)) throw std::runtime_error("Sine: Hs bad.");
    if (!(nu < 0.05f)) throw std::runtime_error("Sine: ν not ~0.");
    (void)f_disp; (void)Tp;
}
#endif

