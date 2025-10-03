#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

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
    constexpr static float BETA_SPEC  = 1.0f;     // exponent in ν
    constexpr static float K_EFF      = 1.0f;     // fixed envelope→variance scale

    // Tracker-robust ω clamp and smoothing
    constexpr static float OMEGA_MIN_HZ = 0.03f;
    constexpr static float OMEGA_MAX_HZ = 1.50f;
    constexpr static float TAU_W_SEC    = 15.0f;  // EMA time-constant for ω_used

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
        t_abs = 0.0f;
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

        omega_used = 0.0f;
        alpha_w    = 0.0f;

        has_moments = false;
        last_dt = -1.0f;
        alpha_env = alpha_mom = alpha_coh = alpha_out = 0.0f;
    }

    // Main update
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        t_abs += dt_s; // accumulate absolute time
        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updatePhaseCoherence();
        updateSpectralMoments(omega_inst);   // now adaptive multi-bin
        computeRegularityOutput();
    }

    // Getters
    float getRegularity() const { return R_out.get(); }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }
    float getNarrowness() const { return nu; }

    float getWaveHeightEnvelopeEst() const { // Hs ≈ 4√M0
        float m0 = M0.get();
        return (m0 > 0.0f) ? 4.0f * std::sqrt(m0) : 0.0f;
    }

    float getDisplacementFrequencyHz() const {
        return (omega_bar_corr > EPSILON) ? (omega_bar_corr / (2.0f * float(M_PI))) : 0.0f;
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

    // ω_used smoothing
    float omega_used;
    float alpha_w;

    // demod state
    float phi;
    float z_real, z_imag;
    float t_abs; // NEW: absolute time accumulator

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
    bool  has_moments;

    // helpers
    void updateAlpha(float dt_s) {
        if (dt_s == last_dt) return;
        alpha_env = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_out = 1.0f - std::exp(-dt_s / tau_out);
        alpha_w   = 1.0f - std::exp(-dt_s / TAU_W_SEC);
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

        float w_obs = std::clamp(omega_inst, OMEGA_MIN, OMEGA_MAX);
        if (omega_used <= 0.0f) omega_used = w_obs;
        else                    omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

        // Outlier gate
        if (omega_used > 0.0f) {
            float ratio = w_obs / omega_used;
            if (ratio < 0.7f || ratio > 1.3f) {
                float P_acc_skip = z_real * z_real + z_imag * z_imag;
                A0.update(P_acc_skip, alpha_mom);
                return;
            }
        }

        // Diagnostic acceleration variance
        float P_acc = z_real * z_real + z_imag * z_imag;
        A0.update(P_acc, alpha_mom);

        // === Adaptive multi-bin integration ===
        int   K    = 0;
        float STEP = 0.0f;

        if (nu < 0.05f) {
            K = 0; STEP = 0.0f;        // pure-tone
        } else if (nu < 0.15f) {
            K = 2; STEP = 0.10f;       // moderate
        } else {
            K = 4; STEP = 0.05f;       // broadband
        }

        float Y_sum  = 0.0f;
        float Y2_sum = 0.0f;
        int count    = 0;

        for (int k = -K; k <= K; k++) {
            float omega_k = omega_used * (1.0f + STEP * k);
            if (omega_k <= EPSILON) continue;

            float phi_shift = (omega_k - omega_used) * t_abs;
            float c = std::cos(phi_shift);
            float s = std::sin(phi_shift);

            float z_r_k =  z_real * c - z_imag * s;
            float z_i_k =  z_real * s + z_imag * c;

            float inv_w2 = 1.0f / (omega_k * omega_k);
            float disp_r = -z_r_k * inv_w2;
            float disp_i = -z_i_k * inv_w2;

            float P_disp = disp_r * disp_r + disp_i * disp_i;
            float Y      = K_EFF * P_disp;
            float Y2     = (K_EFF * K_EFF) * P_disp * P_disp;

            Y_sum  += Y;
            Y2_sum += Y2;
            count++;
        }

        float Y_final  = (count > 0) ? (Y_sum  / count) : 0.0f;
        float Y2_final = (count > 0) ? (Y2_sum / count) : 0.0f;

        if (!has_moments) has_moments = true;

        M0.update(Y_final, alpha_mom);
        M1.update(Y_final * omega_used, alpha_mom);
        M2.update(Y_final * omega_used * omega_used, alpha_mom);

        Q0.update(Y2_final, alpha_mom);
        Q1.update(Y2_final * omega_used, alpha_mom);
        Q2.update(Y2_final * omega_used * omega_used, alpha_mom);
    }

    void updatePhaseCoherence() {
        float mag = std::hypot(z_real, z_imag);
        if (mag <= EPSILON) {
            if (coh_r.isReady() && coh_i.isReady())
                R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() +
                                               coh_i.get()*coh_i.get()), 0.0f, 1.0f);
            else
                R_phase = 0.0f;
            return;
        }
        float u_r = z_real / mag;
        float u_i = z_imag / mag;
        coh_r.update(u_r, alpha_coh);
        coh_i.update(u_i, alpha_coh);
        R_phase = std::clamp(std::sqrt(coh_r.get()*coh_r.get() +
                                       coh_i.get()*coh_i.get()), 0.0f, 1.0f);
    }

    void computeRegularityOutput() {
        if (!M0.isReady()) {
            R_out.update(R_phase, alpha_out);
            R_spec = R_phase;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        float m0 = M0.get();
        float m1 = M1.get();
        float m2 = M2.get();

        if (!(m0 > EPSILON)) {
            R_out.update(0.0f, alpha_out);
            R_spec = 0.0f;
            nu = 0.0f;
            omega_bar_corr = omega_bar_naive = 0.0f;
            return;
        }

        omega_bar_naive  =  m1 / m0;
        float omega2_bar =  m2 / m0;

        float mu2 = std::max(0.0f, omega2_bar - omega_bar_naive * omega_bar_naive);
        omega_bar_corr = omega_bar_naive;

        nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2) / omega_bar_corr) : 0.0f;
        nu = std::max(0.0f, nu);

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

    const float Hs_expected = 4.0f * SINE_AMPLITUDE;

    if (!(R_spec > 0.90f))
        throw std::runtime_error("Sine: R_spec did not converge near 1.");
    if (!(R_phase > 0.80f))
        throw std::runtime_error("Sine: R_phase did not converge near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.25f * Hs_expected))
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

