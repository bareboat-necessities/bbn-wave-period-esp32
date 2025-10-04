#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/*
    Copyright 2025, Mikhail Grushinskiy
*/

// Debiased EMA (bias-corrected by tracking weight)
struct DebiasedEMA {
    float value  = 0.0f;
    float weight = 0.0f;
    void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float alpha) {
        value  = (1.0f - alpha) * value + alpha * x;
        weight = (1.0f - alpha) * weight + alpha;
    }
    inline void decay(float alpha) {              // <-- add this
        value  = (1.0f - alpha) * value;
        weight = (1.0f - alpha) * weight;
    }
    inline float get() const { return (weight > 1e-12f) ? value / weight : 0.0f; }
    inline bool  isReady() const { return weight > 1e-6f; }
};

// SeaStateRegularity
class SeaStateRegularity {
public:
    // Numerics / mapping
    constexpr static float EPSILON    = 1e-12f;
    constexpr static float BETA_SPEC  = 1.0f;     // exponent in ν
    constexpr static float K_EFF_MIX  = 2.0f;     // extra scale (calibration)

    // Tracker-robust ω clamp and smoothing
    constexpr static float OMEGA_MIN_HZ = 0.03f;
    constexpr static float OMEGA_MAX_HZ = 1.50f;
    constexpr static float TAU_W_SEC    = 15.0f;  // EMA time-constant for ω_used

    // Multi-bin params
    constexpr static int   MAX_K       = 12;       // up to ±12 bins
    constexpr static int   NBINS       = 2*MAX_K + 1;
    constexpr static float STEP_NARROW = 0.10f;
    constexpr static float STEP_BROAD  = 0.05f;
    constexpr static float MIN_FC_HZ   = 0.02f;

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
        A0.reset();

        // For Jensen correction
        Q00.reset();  // ⟨S0^2⟩
        Q10.reset();  // ⟨S0*S1⟩

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

        for (int i=0;i<NBINS;i++){ bin_zr[i]=bin_zi[i]=0.0f; bin_c[i]=1.0f; bin_s[i]=0.0f; }
        bins_init=false;
        last_accel=0.0f;
    }

    // Main update
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) return;

        t_abs += dt_s;
        last_accel = accel_z;

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
        return (m0 > 0.0f) ? 4.0f * std::sqrt(std::max(0.0f, m0)) : 0.0f;
    }

    float getDisplacementFrequencyHz() const {
        return (omega_bar_corr > EPSILON) ? (omega_bar_corr / (2.0f * float(M_PI))) : 0.0f;
    }

    float getDisplacementFrequencyNaiveHz() const {
        return (omega_bar_naive > EPSILON)
             ? (omega_bar_naive / (2.0f * float(M_PI)))
             : 0.0f;
    }

    float getDisplacementPeriodSec() const {
        return (omega_bar_corr > EPSILON) ? (2.0f * float(M_PI) / omega_bar_corr) : 0.0f;
    }

    float getAccelerationVariance() const { return A0.get(); }

private:
    // Precomputed rad/s clamps
    static constexpr float OMEGA_MIN_RAD = 2.0f * float(M_PI) * OMEGA_MIN_HZ;
    static constexpr float OMEGA_MAX_RAD = 2.0f * float(M_PI) * OMEGA_MAX_HZ;

    // time constants and alphas
    float tau_env, tau_mom, tau_coh, tau_out;
    float last_dt;
    float alpha_env, alpha_mom, alpha_coh, alpha_out;

    // ω_used smoothing
    float omega_used;
    float alpha_w;

    // demod state (narrow diagnostic)
    float phi;
    float z_real, z_imag;
    float t_abs;

    // raw accel cache for multi-bin
    float last_accel;

    // per-bin demod states
    float bin_zr[NBINS], bin_zi[NBINS];
    float bin_c[NBINS],  bin_s[NBINS];
    bool  bins_init;

    // moments (primary)
    DebiasedEMA M0, M1, M2;
    DebiasedEMA A0;

    // moments for Jensen correction
    // Q00 ≈ ⟨S0^2⟩, Q10 ≈ ⟨S0*S1⟩, where S0=ΣYk, S1=Σ(Yk*ωk)
    DebiasedEMA Q00, Q10;

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
        // Robust wrapping without fmod (avoid cumulative growth)
        const float TWO_PI_ = 2.0f * float(M_PI);
        if (phi >= TWO_PI_) phi -= TWO_PI_;
        if (phi <  0.0f)    phi += TWO_PI_;

        float c = std::cos(phi);
        float s = std::sin(phi);

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
        float w_obs = std::clamp(omega_inst, OMEGA_MIN_RAD, OMEGA_MAX_RAD);

        // Smooth ω_used
        if (omega_used <= 0.0f) omega_used = w_obs;
        else                    omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

        // Outlier gate: skip updates if tracker jumps too far
        if (omega_used > 0.0f) {
            float ratio = w_obs / omega_used;
            if (ratio < 0.7f || ratio > 1.3f) {
                float P_acc_skip = z_real*z_real + z_imag*z_imag;
                A0.update(P_acc_skip, alpha_mom);
                return;
            }
        }

        // Multi-bin extent
        int   K    = 0;
        float STEP = 0.0f;
        if      (nu < 0.05f) { K = 0; STEP = 0.0f; }
        else if (nu < 0.15f) { K = 2; STEP = STEP_NARROW; }
        else                 { K = MAX_K; STEP = STEP_BROAD; }

        if (!bins_init) {
            for (int i = 0; i < NBINS; i++) {
                bin_c[i]  = 1.0f;
                bin_s[i]  = 0.0f;
                bin_zr[i] = 0.0f;
                bin_zi[i] = 0.0f;
            }
            bins_init = true;
        }

        // Compute bin centers (linear ratio grid)
        float omega_k_arr[NBINS];
        for (int k = -K; k <= K; ++k) {
            int idx = k + MAX_K;
            omega_k_arr[idx] = omega_used * (1.0f + STEP * k);
        }

        // Compute per-bin widths Δω (central diff, one-sided at edges)
        float domega_k_arr[NBINS];
        for (int k = -K; k <= K; ++k) {
            int idx = k + MAX_K;
            float wL, wR;
            if (k == -K) {
                wL = omega_k_arr[idx];
                wR = omega_k_arr[idx + 1];
                domega_k_arr[idx] = std::max(EPSILON, wR - wL);
            } else if (k == K) {
                wL = omega_k_arr[idx - 1];
                wR = omega_k_arr[idx];
                domega_k_arr[idx] = std::max(EPSILON, wR - wL);
            } else {
                wL = omega_k_arr[idx - 1];
                wR = omega_k_arr[idx + 1];
                domega_k_arr[idx] = std::max(EPSILON, 0.5f * (wR - wL));
            }
        }

        // Baseband LPF alpha (shared across bins; constant ENBW factor)
        float f_used_hz     = omega_used / (2.0f * float(M_PI));
        float fc_hz         = std::max(0.05f, STEP * f_used_hz);
        float alpha_env_bin = 1.0f - std::exp(-last_dt * 2.0f * float(M_PI) * fc_hz);
        alpha_env_bin       = std::clamp(alpha_env_bin, 0.0f, 1.0f);

        float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
        float A_var = 0.0f; // acceleration variance accumulator

        // Loop over bins
        for (int k = -K; k <= K; ++k) {
            int idx = k + MAX_K;
            float omega_k = omega_k_arr[idx];
            if (omega_k <= EPSILON) continue;

            // Advance oscillator
            float dphi = omega_k * last_dt;
            float cd = std::cos(dphi), sd = std::sin(dphi);
            float c0 = bin_c[idx], s0 = bin_s[idx];
            float c1 =  c0*cd - s0*sd;
            float s1 =  c0*sd + s0*cd;
            bin_c[idx] = c1; bin_s[idx] = s1;

            // Mix to baseband
            float y_r =  last_accel * c1;
            float y_i = -last_accel * s1;

            // LP filter
            bin_zr[idx] = (1.0f - alpha_env_bin) * bin_zr[idx] + alpha_env_bin * y_r;
            bin_zi[idx] = (1.0f - alpha_env_bin) * bin_zi[idx] + alpha_env_bin * y_i;

            // Accel → displacement (z = −a/ω²)
            float inv_w2 = 1.0f / std::max(omega_k*omega_k, EPSILON);
            float dr = bin_zr[idx] * inv_w2;
            float di = bin_zi[idx] * inv_w2;

            float P_disp = dr*dr + di*di;
            float Yk = K_EFF_MIX * P_disp; // displacement spectrum proxy

            // Weighted displacement moments
            float dω = domega_k_arr[idx];
            S0 += Yk * dω;
            S1 += Yk * omega_k * dω;
            S2 += Yk * omega_k * omega_k * dω;

            // Acceleration variance contribution: ω^4 * S_eta(ω) * Δω
            A_var += (omega_k*omega_k*omega_k*omega_k) * Yk;
        }

        if (!has_moments) has_moments = true;

        // Primary moment EMAs
        M0.update(S0, alpha_mom);
        M1.update(S1, alpha_mom);
        M2.update(S2, alpha_mom);

        // Jensen correction helpers
        Q00.update(S0 * S0, alpha_mom);
        Q10.update(S0 * S1, alpha_mom);

        // Update broadband acceleration variance (σ²[a])
        A0.update(A_var, alpha_mom);
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

        // Naive mean and variance of ω
        omega_bar_naive  =  m1 / m0;
        float omega2_bar =  m2 / m0;
        float mu2 = std::max(0.0f, omega2_bar - omega_bar_naive * omega_bar_naive);

        // Jensen correction for ratio E[M1/M0]
        // Var[M0] ≈ E[S0^2] - M0^2,  Cov[M1,M0] ≈ E[S0*S1] - M1*M0
        float q00 = Q00.get();   // ⟨S0^2⟩
        float q10 = Q10.get();   // ⟨S0*S1⟩
        float varM0  = std::max(0.0f, q00 - m0*m0);
        float cov10  = q10 - m1*m0;
        float invM0_2 = 1.0f / std::max(m0*m0, EPSILON);

        omega_bar_corr = omega_bar_naive + (omega_bar_naive * varM0 - cov10) * invM0_2;

        // Narrowness ν and spectral regularity
        nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2) / omega_bar_corr) : 0.0f;
        nu = std::max(0.0f, nu);
        R_spec = std::clamp(std::exp(-BETA_SPEC * nu), 0.0f, 1.0f);

        // Output = max of phase vs spectral
        float R_combined = R_spec; //std::max(R_phase, R_spec);
        R_out.update(R_combined, alpha_out);
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


