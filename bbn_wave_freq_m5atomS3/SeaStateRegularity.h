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
 * SeaStateRegularity — Online estimator of ocean wave regularity from vertical acceleration.
 *
 * Inputs
 *   • Vertical acceleration a_z(t) [m/s²]
 *   • Instantaneous angular frequency ω_inst(t) [rad/s] from an external tracker
 *
 * Method (sketch)
 *
 * 1) Demodulate acceleration into baseband I/Q
 *      y(t) = a_z(t) · e^(−jφ(t)),     φ̇(t) = ω_φ(t) ≈ LP(ω_inst(t))
 *    Low-pass ⇒ z(t) = (z_r,z_i) (complex envelope of acceleration).
 *
 * 2) Form a displacement-envelope proxy using the small-slope relation
 *      d(t) = z(t) / ω_norm(t)²,       ω_norm ≈ blend( ω_lp,  ω̄_disp )
 *    Power in displacement envelope:
 *      P_disp(t) = |d(t)|² .
 *
 * 3) Track spectral moments online (exponential moving averages)
 *      M₀ = ⟨ P_disp ⟩,
 *      M₁ = ⟨ P_disp · ω ⟩,
 *      M₂ = ⟨ P_disp · ω² ⟩.
 *    From these,
 *      ω̄ = M₁ / M₀                            (mean frequency),
 *      μ₂ = (M₀M₂ − M₁²) / M₀  ≥ 0            (central variance of frequency).
 *    Define relative bandwidth (dimensionless)
 *      rbw = √μ₂ / ω̄ .
 *
 * 4) Regularity components (both ∈ [0,1])
 *      R_spec  = exp( −β · rbw )               (spectral narrowness; smooth & monotone),
 *      R_phase = ‖ ⟨ u ⟩ ‖,  with  u = z / ‖z‖ (phase coherence; mean resultant length).
 *
 * 5) Fusion (phase-forward geometric mean; scale-free)
 *      R_safe = ( R_phase^w ) · ( R_spec^(1−w) ),   w ∈ [0,1] (default w≈0.75)
 *      Optional mild softening tied to phase variance:
 *      R_safe ← R_safe · ( 1 − k · (1 − R_phase) ).
 *
 * 6) Significant wave height (smooth blend of edge cases)
 *      Hs^reg ≈ 2√M₀                  (regular single-tone),
 *      Hs^irr ≈ 2√2 · √M₀             (Gaussian narrowband, random phase).
 *    We use
 *      Hs ≈ 2√M₀ · f(R_out),   where  f : [0,1] → [√2, 1]
 *    is a cubic-smoothstep mapping (threshold-free, differentiable).
 *
 */

class SeaStateRegularity {
public:
    // Small constants and weights
    constexpr static float EPSILON      = 1e-12f;
    constexpr static float PHASE_WEIGHT = 0.75f;   // fusion weight w in R = R_phase^w · R_spec^(1−w)
    constexpr static float BETA_SPEC    = 1.00f;   // R_spec = exp(−β · rbw),  rbw = √μ₂/ω̄
    constexpr static float K_CIRC_VAR   = 0.20f;   // mild softening: 1 − k · (1 − R_phase)
    constexpr static float HEIGHT_GAMMA = 1.00f;   // shape for smoothstep in f(R)

    // Safe defaults for typical ocean waves (PM/JONSWAP)
    SeaStateRegularity(float tau_env_sec   = 15.0f,   // envelope I/Q smoother (~2–3 waves)
                       float tau_mom_sec   = 180.0f,  // moments horizon (minutes-scale)
                       float omega_min_hz  = 0.03f,   // cutoff (~33 s period)
                       float tau_coh_sec   = 60.0f,   // steadies R_phase
                       float tau_out_sec   = 30.0f,   // final R_out smoother
                       float tau_omega_sec = 45.0f)   // protects 1/ω² normalization
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

        // Spectral moments and regularities (no M4)
        M0 = M1 = M2 = 0.0f;
        nu = 0.0f;
        R_spec = R_phase = R_safe = R_out = 0.0f;

        // Alphas and timing
        alpha_env = alpha_mom = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

        // Phase coherence avg
        coh_r = coh_i = 0.0f;
        has_coh = false;

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
        has_R_out = false;

        // Demod-driving omega slew limiter memory
        omega_phi_last = 0.0f;
    }

    // Main update: dt (s), vertical accel (m/s^2), instantaneous omega (rad/s)
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z   != accel_z)   accel_z   = 0.0f;            // NaN guard
        if (omega_inst!= omega_inst)omega_inst= omega_min;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityOutput();
    }

    // Readouts
    float getNarrowness() const { return nu; }                  // ν = √((M0 M2 / M1²) − 1)
    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    float getCircularVariance() const { return 1.0f - R_phase; } // 0 for perfect coherence, 1 for uniform

    float getWaveHeightEnvelopeEst() const {
        if (M0 <= 0.0f || !has_R_out) return 0.0f;
        // Hs ≈ 2√M0 · f(R), where f(R): smoothstep from √2 (irregular) to 1 (regular)
        return 2.0f * std::sqrt(M0) * heightFactorFromR(R_out);
    }

    float getDisplacementFrequencyHz() const {
        return omega_disp_lp / (2.0f * float(M_PI));            // f_disp = ω̄/(2π)
    }

    // Optional: scale time constants from dominant period Tp (s)
    void set_safe_defaults_from_peak_period(float Tp) {
        Tp = std::min(std::max(Tp, 5.0f), 20.0f);
        tau_env   = std::clamp(2.0f  * Tp,  8.0f,  40.0f);
        tau_omega = std::clamp(5.0f  * Tp, 20.0f, 120.0f);
        tau_mom   = std::clamp(12.0f * Tp, 60.0f, 300.0f);
        tau_coh   = std::clamp(6.0f  * Tp, 30.0f, 120.0f);
        tau_out   = std::clamp(4.0f  * Tp, 15.0f,  60.0f);
        // Alphas refresh in updateAlpha() when dt changes.
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

    // Spectral moments & regularity (M4 removed)
    float M0, M1, M2;
    float nu;
    float R_spec, R_phase, R_safe, R_out;

    // Phase coherence
    float coh_r, coh_i;
    bool  has_coh;

    // Frequency tracking
    float omega_lp, omega_disp_lp;
    float omega_last, mu_w;
    float var_fast, var_slow;

    // Flags
    bool has_omega_lp;
    bool has_omega_disp_lp;
    bool has_moments;
    bool has_R_out;

    // Variance adaptation
    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    // Demod omega slew limiter memory
    float omega_phi_last;

    // Impl
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
        // Use smoothed ω with a conservative slew limit to rotate demod basis.
        // This keeps φ(t) well-behaved if the tracker ω_inst jitters.
        float omega_phi = has_omega_lp ? omega_lp : std::max(omega_inst, omega_min);
        const float domega_max = 0.5f; // rad/s per update
        if (!has_omega_lp) omega_phi_last = omega_phi;
        float domega = omega_phi - omega_phi_last;
        if (domega >  domega_max) omega_phi = omega_phi_last + domega_max;
        if (domega < -domega_max) omega_phi = omega_phi_last - domega_max;
        omega_phi_last = omega_phi;

        // Integrate demodulation phase
        phi += omega_phi * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        // Rotate acceleration to baseband I/Q: y = a · e^(−jφ)
        float c = std::cos(-phi), s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        // First-time init
        if (!has_omega_lp) {
            z_real = y_real; z_imag = y_imag;
            omega_lp = std::max(omega_inst, omega_min);
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        // Envelope EMA (per-cycle smoothness)
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // LP the instantaneous ω from external estimator (stabilizes 1/ω² below)
        float w_inst = std::max(omega_inst, omega_min);
        omega_lp = (tau_omega > 0.0f)
                 ? (1.0f - alpha_omega) * omega_lp + alpha_omega * w_inst
                 : w_inst;

        // Track slow variance of ω_lp to stabilize M2 (reduces rbw bias from ω noise)
        float delta = omega_lp - mu_w;
        mu_w    += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
        // ω used for accel→disp normalization: ω_norm ~ 0.7·ω_lp + 0.3·ω̄_disp (if available)
        float omega_norm = omega_lp;
        if (has_omega_disp_lp) omega_norm = 0.7f * omega_lp + 0.3f * omega_disp_lp;
        omega_norm = std::max(omega_norm, omega_min);

        // Convert accel-envelope → displacement-envelope via 1/ω_norm²
        float inv_w2 = 1.0f / std::max(omega_norm * omega_norm, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;

        // Exponential averaging of spectral moments (M0..M2)
        if (!has_moments) {
            M0 = P_disp;
            M1 = P_disp * omega_norm;
            // Stabilize M2 by compensating slow ω variance; keep ≥ 0
            float M2_c = P_disp * omega_norm * omega_norm - var_slow * M0;
            M2 = (M2_c > 0.0f) ? M2_c : 0.0f;
            has_moments = true;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * omega_norm;
            float M2_c = P_disp * omega_norm * omega_norm - var_slow * M0;
            if (M2_c < 0.0f) M2_c = 0.0f;
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * M2_c;
        }

        // Moment-based mean frequency ω̄_disp = M1/M0  (LP for readout/feedback)
        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (!has_omega_disp_lp) {
            omega_disp_lp = omega_disp;
            has_omega_disp_lp = true;
        } else {
            omega_disp_lp = (1.0f - alpha_omega) * omega_disp_lp + alpha_omega * omega_disp;
        }
    }

    void updatePhaseCoherence() {
        // Unit vector u = (z_r, z_i)/‖z‖; R_phase = ‖E[u]‖ (mean resultant length in [0,1])
        float mag = std::sqrt(z_real * z_real + z_imag * z_imag);
        float u_r = (mag > EPSILON) ? (z_real / mag) : 1.0f;
        float u_i = (mag > EPSILON) ? (z_imag / mag) : 0.0f;

        if (!has_coh) { coh_r = u_r; coh_i = u_i; has_coh = true; }
        else {
            coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
            coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;
        }
        R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
    }

    void computeRegularityOutput() {
        // Spectral regularity from relative bandwidth rbw = √μ₂ / ω̄
        //    ω̄ = M1/M0,  μ₂ = (M0*M2 - M1²)/M0 ≥ 0,  R_spec = exp(−β·rbw)
        if (M0 > EPSILON) {
            float omega_bar = M1 / M0;
            float num = M0 * M2 - M1 * M1;                       // = M0² Var(ω)
            if (num < 0.0f) num = 0.0f;                          // numeric guard
            float mu2 = num / std::max(M0, EPSILON);             // Var(ω)
            float rbw = (omega_bar > 0.0f)
                      ? std::sqrt(mu2) / std::max(omega_bar, omega_min)
                      : 0.0f;
            R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);
        } else {
            R_spec = 0.0f;
        }

        // Fusion with phase coherence:
        //    R_base = R_phase^w · R_spec^(1−w)  (geometric mean; phase-forward)
        float Rp = std::max(R_phase, 1e-6f);
        float Rs = std::max(R_spec,  1e-6f);
        float w  = std::clamp(PHASE_WEIGHT, 0.0f, 1.0f);
        float R_base = std::exp(w * std::log(Rp) + (1.0f - w) * std::log(Rs));

        // Optional mild softening tied to phase variance:
        // adj_var = 1 − k · (1 − R_phase)
        float adj_var = 1.0f - K_CIRC_VAR * (1.0f - R_phase);

        // “Harmonic-safe regularity”: expose fused value (no thresholds)
        R_safe = std::clamp(R_base * adj_var, 0.0f, 1.0f);

        // Final output smoothing
        float R_target = R_safe;
        if (!has_R_out) { R_out = R_target; has_R_out = true; }
        else            { R_out = (1.0f - alpha_out) * R_out + alpha_out * R_target; }

        // Legacy narrowness diagnostic: ν = √((M0*M2 / M1²) − 1)  when M1>0
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            nu = (ratio > 0.0f) ? std::sqrt(ratio) : 0.0f;
        } else {
            nu = 0.0f;
        }
    }

    // Smooth height factor f(R): maps R∈[0,1] to f∈[√2, 1] using cubic smoothstep
    static float heightFactorFromR(float R) {
        float x = std::clamp(R, 0.0f, 1.0f);
        float s = x * x * (3.0f - 2.0f * x);                 // smoothstep ∈ [0,1]
        if (HEIGHT_GAMMA != 1.0f) s = std::pow(s, std::max(0.0f, HEIGHT_GAMMA));
        const float C_reg = 1.0f, C_irr = std::sqrt(2.0f);
        return C_irr - (C_irr - C_reg) * s;                  // f(0)=√2 (irreg), f(1)=1 (reg)
    }
};

#ifdef SEA_STATE_TEST
// Simple sine-wave test harness
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
        reg.update(DT, a, wave.omega);    // feed perfect ω to isolate filter behavior
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
