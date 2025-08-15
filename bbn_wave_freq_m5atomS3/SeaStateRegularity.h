#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec   = 1.0f,   // envelope LP (z)
                       float tau_mom_sec   = 60.0f,  // spectral moments LP
                       float omega_min_hz  = 0.03f,  // min usable freq (Hz)
                       float tau_ref_sec   = 60.0f,  // slow mean-freq tracker for jitter
                       float tau_coh_sec   = 10.0f,  // coherence avg for e^{j dtheta}
                       float tau_out_sec   = 10.0f,  // final R output smoother
                       float tau_omega_sec = 0.0f)   // smoothing for 1/ω² only (0 = off)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_ref   = std::max(1e-3f, tau_ref_sec);
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_out   = std::max(1e-3f, tau_out_sec);
        tau_omega = std::max(0.0f,   tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz;
        reset();
    }

    void reset() {
        // signal state
        phi = 0.0f;
        z_real = 0.0f;
        z_imag = 0.0f;

        // displacement moments
        M0 = M1 = M2 = 0.0f;

        // regularities
        nu      = std::numeric_limits<float>::quiet_NaN();
        R_spec  = std::numeric_limits<float>::quiet_NaN();
        R_phase = std::numeric_limits<float>::quiet_NaN();
        R_out   = std::numeric_limits<float>::quiet_NaN();

        // step alphas
        alpha_env = alpha_mom = alpha_ref = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt   = -1.0f;

        // jitter path
        omega_ref = omega_min;
        coh_r = 1.0f;  // start fully coherent to avoid initial dip
        coh_i = 0.0f;

        // ω for 1/ω² conversion
        omega_lp = omega_min;
    }

    // dt_s: seconds, accel_z: m/s^2 (up is +), omega_inst: rad/s instantaneous dominant frequency
    void update(float dt_s, float accel_z, float omega_inst) {
        // guard inputs
        if (!std::isfinite(dt_s) || dt_s <= 0.0f) return;
        if (!std::isfinite(accel_z)) accel_z = 0.0f;
        if (!std::isfinite(omega_inst)) omega_inst = omega_min;

        // update alphas only when dt changes (saves expf cost)
        if (dt_s != last_dt) {
            alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
            alpha_ref   = 1.0f - std::exp(-dt_s / tau_ref);
            alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
            alpha_out   = 1.0f - std::exp(-dt_s / tau_out);
            alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
            last_dt     = dt_s;
        }

        // ------------------ Demodulate accel by instantaneous phase ------------------
        phi += omega_inst * dt_s;
        // keep phase bounded each step to reduce trig drift
        phi = std::fmod(phi, 2.0f * float(M_PI));

        const float c = std::cos(-phi);
        const float s = std::sin(-phi);
        const float y_real = accel_z * c;
        const float y_imag = accel_z * s;

        // Envelope LP (complex one-pole)
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // ------------------ 1/ω² conversion (use smoothed ω to avoid spikes) ------------------
        const float w_inst = std::max(omega_inst, omega_min);
        omega_lp = (tau_omega > 0.0f) ? ((1.0f - alpha_omega) * omega_lp + alpha_omega * w_inst)
                                      : w_inst;
        const float w = omega_lp;
        const float inv_w2 = 1.0f / (w * w);

        const float disp_real = z_real * inv_w2;
        const float disp_imag = z_imag * inv_w2;
        const float P_disp    = disp_real * disp_real + disp_imag * disp_imag;

        // ------------------ Displacement spectral moments ------------------
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * w;
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * w * w;

        // R_spec from bandwidth (same definition you used)
        if (M1 > 1e-12f && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu     = std::sqrt(ratio);
            R_spec = std::exp(-nu);
        } else {
            nu     = std::numeric_limits<float>::quiet_NaN();
            R_spec = 0.0f; // treat as low regularity when moments are invalid
        }
        R_spec = std::clamp(R_spec, 0.0f, 1.0f);

        // ------------------ Phase coherence from frequency jitter ------------------
        // slow mean frequency (very slow so it doesn't cancel jitter)
        omega_ref = (1.0f - alpha_ref) * omega_ref + alpha_ref * w_inst;

        const float dtheta = (w_inst - omega_ref) * dt_s;  // radians per step
        const float cd = std::cos(dtheta);
        const float sd = std::sin(dtheta);

        // exponential average of unit phasor e^{j dtheta}
        coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * cd;
        coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * sd;

        float R_phase_inst = std::sqrt(coh_r * coh_r + coh_i * coh_i);
        if (R_phase_inst > 1.0f) R_phase_inst = 1.0f;
        R_phase = R_phase_inst; // expose latest value (already smoothed by tau_coh)

        // ------------------ Smooth, stable blending ------------------
        // Make the phase rescue matter only when spectral regularity is already decent.
        // Use smoothstep window to avoid handoff jumps.
        const float R_lo = 0.75f;   // below this, trust spectral only
        const float R_hi = 0.92f;   // above this, trust phase strongly

        float w_phase = smoothstep(R_spec, R_lo, R_hi);     // in [0,1]
        // geometric blend is smoother and avoids overshoot:
        float R_blend = std::pow(std::max(R_spec, 1e-6f), (1.0f - w_phase))
                      * std::pow(std::max(R_phase, 1e-6f), w_phase);

        // final output smoothing to kill residual twitch
        if (!std::isfinite(R_out)) {
            R_out = R_blend; // init
        } else {
            R_out = (1.0f - alpha_out) * R_out + alpha_out * R_blend;
        }
        R_out = std::clamp(R_out, 0.0f, 1.0f);
    }

    // --- Public getters ---
    float getNarrowness() const         { return nu; }      // spectral narrowness
    float getRegularity() const         { return R_out; }   // stable, blended, smoothed
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const    { return R_phase; }

    // Significant wave height using harmonic-safe, smoothed R
    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f || !std::isfinite(R_out)) return 0.0f;
        const float factor = heightFactorFromR(R_out);
        return 2.0f * std::sqrt(M0) * factor;
    }

private:
    // Tunables
    float tau_env, tau_mom, tau_ref, tau_coh, tau_out, tau_omega;
    float omega_min;

    // Cached per-dt alphas
    float last_dt = -1.0f;
    float alpha_env = 0.0f, alpha_mom = 0.0f, alpha_ref = 0.0f, alpha_coh = 0.0f, alpha_out = 0.0f, alpha_omega = 0.0f;

    // Demod + envelope
    float phi = 0.0f;
    float z_real = 0.0f, z_imag = 0.0f;

    // Moments (displacement domain)
    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f;

    // Spectral regularity path
    float nu   = std::numeric_limits<float>::quiet_NaN();
    float R_spec = 0.0f;

    // Phase-regularity path
    float omega_ref = 0.0f;
    float coh_r = 1.0f, coh_i = 0.0f; // complex accumulator of e^{j dtheta}
    float R_phase = 1.0f;

    // Output
    float omega_lp = 0.0f; // for 1/ω² conversion
    float R_out = 0.0f;

    static float smoothstep(float x, float edge0, float edge1) {
        if (edge1 <= edge0) return (x >= edge1) ? 1.0f : 0.0f;
        x = (x - edge0) / (edge1 - edge0);
        x = std::clamp(x, 0.0f, 1.0f);
        return x * x * (3.0f - 2.0f * x);
    }

    static float heightFactorFromR(float R_val) {
        // 1.0 for very regular, up to ~sqrt(2) as regularity falls
        const float R_hi = 0.98f;
        const float R_lo = 0.50f;
        if (!std::isfinite(R_val) || R_val >= R_hi) return 1.0f;
        if (R_val <= R_lo) return std::sqrt(2.0f);

        float x = (R_hi - R_val) / (R_hi - R_lo);  // 0→regular, 1→irregular
        // gentle slope near R≈1 to avoid twitch in Hs
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::pow(x, 1.5f);
    }
};
