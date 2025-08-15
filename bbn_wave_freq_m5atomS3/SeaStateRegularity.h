#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec  = 1.0f,
                       float tau_mom_sec  = 60.0f,
                       float omega_min_hz = 0.03f,
                       float tau_phase_sec = -1.0f,   // <=0 -> use tau_mom_sec
                       float tau_omega_sec = 0.0f,    // 0 -> no smoothing of omega_inst
                       bool  use_max_blend = true,    // true: R = max(Rspec, Rphi)
                       float beta_blend    = 0.5f)    // used only if use_max_blend=false
    : use_max_blend_(use_max_blend), beta_blend_(std::clamp(beta_blend, 0.0f, 1.0f))
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_phase = (tau_phase_sec > 0.0f) ? tau_phase_sec : tau_mom_sec;
        tau_omega = std::max(0.0f, tau_omega_sec);
        omega_min = 2.0f * M_PI * omega_min_hz; // rad/s
        reset();
    }

    void reset() {
        phi = 0.0f;
        z_real = 0.0f;
        z_imag = 0.0f;
        M0 = M1 = M2 = 0.0f;

        nu        = std::numeric_limits<float>::quiet_NaN();
        R_spec    = std::numeric_limits<float>::quiet_NaN();
        R_phase   = std::numeric_limits<float>::quiet_NaN();
        R_safe    = std::numeric_limits<float>::quiet_NaN();

        alpha_env = 0.0f;
        alpha_mom = 0.0f;
        alpha_phase = 0.0f;
        alpha_omega = 0.0f;
        last_dt   = -1.0f;

        // Phase-coherence accumulator
        u_prev_real = 1.0f;
        u_prev_imag = 0.0f;
        have_u_prev = false;
        coh_acc_real = 0.0f;
        coh_acc_imag = 0.0f;

        // Smoothed omega used only for displacement conversion
        omega_lp = omega_min;
    }

    // Online update with (dt, vertical acceleration, instantaneous omega)
    void update(float dt_s, float accel_z, float omega_inst) {
        // Update alphas only when dt changes (cheap and stable)
        if (dt_s != last_dt) {
            alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
            alpha_phase = 1.0f - std::exp(-dt_s / tau_phase);
            alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f; // if tau_omega==0 -> no smoothing, alpha=1
            last_dt     = dt_s;
        }

        // Phase advance and wrap to avoid float drift
        phi += omega_inst * dt_s;
        if (phi > 1e6f || phi < -1e6f) {
            phi = std::fmod(phi, 2.0f * float(M_PI));
        }

        // Mix down accel into complex envelope (baseband around omega_inst)
        const float c = std::cos(-phi);
        const float s = std::sin(-phi);
        const float y_real = accel_z * c;
        const float y_imag = accel_z * s;

        // Envelope low-pass (one-pole)
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // Optionally smooth omega for displacement conversion to reduce spikes
        if (tau_omega > 0.0f) {
            float w_target = std::max(omega_inst, omega_min);
            omega_lp = (1.0f - alpha_omega) * omega_lp + alpha_omega * w_target;
        } else {
            omega_lp = std::max(omega_inst, omega_min);
        }
        const float w = omega_lp;

        // Convert acceleration to displacement-equivalent at current dominant frequency
        const float inv_w2 = 1.0f / (w * w);
        const float disp_real = z_real * inv_w2;
        const float disp_imag = z_imag * inv_w2;
        const float P_disp = disp_real * disp_real + disp_imag * disp_imag;

        // Spectral-moment path (same as yours)
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * w;
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * w * w;

        // Compute spectral bandwidth nu and R_spec
        if (M1 > 1e-12f && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu     = std::sqrt(ratio);
            R_spec = std::exp(-nu);
        } else {
            nu     = std::numeric_limits<float>::quiet_NaN();
            R_spec = std::numeric_limits<float>::quiet_NaN();
        }

        // ---- Harmonic-insensitive regularity via phase coherence ----
        // Unit phasor of the (filtered) complex envelope
        float mag = std::sqrt(z_real * z_real + z_imag * z_imag);
        float u_r = (mag > 1e-12f) ? (z_real / mag) : u_prev_real;
        float u_i = (mag > 1e-12f) ? (z_imag / mag) : u_prev_imag;

        if (!have_u_prev) {
            u_prev_real = u_r;
            u_prev_imag = u_i;
            have_u_prev = true;
        }

        // Delta step on the unit circle: conj(u_prev) * u
        float delta_r =  u_prev_real * u_r + u_prev_imag * u_i;
        float delta_i = -u_prev_imag * u_r + u_prev_real * u_i;

        // Exponential average of delta (complex). Magnitude in [0,1] measures stability.
        coh_acc_real = (1.0f - alpha_phase) * coh_acc_real + alpha_phase * delta_r;
        coh_acc_imag = (1.0f - alpha_phase) * coh_acc_imag + alpha_phase * delta_i;
        R_phase = std::sqrt(coh_acc_real * coh_acc_real + coh_acc_imag * coh_acc_imag);
        if (R_phase > 1.0f) R_phase = 1.0f;  // numeric guard

        // Prepare for next step
        u_prev_real = u_r;
        u_prev_imag = u_i;

        // ---- Blend to get harmonic-safe R ----
        float Rspec_bounded  = (std::isfinite(R_spec)) ? std::clamp(R_spec, 0.0f, 1.0f) : 0.0f;
        float Rphase_bounded = (std::isfinite(R_phase)) ? std::clamp(R_phase, 0.0f, 1.0f) : 0.0f;

        if (use_max_blend_) {
            // Conservative rescue for nonlinear narrowband: don't let harmonics drag R down
            R_safe = std::max(Rspec_bounded, Rphase_bounded);
        } else {
            // Smooth blend if you prefer continuity with R_spec
            R_safe = beta_blend_ * Rspec_bounded + (1.0f - beta_blend_) * Rphase_bounded;
        }
    }

    // --- Public getters ---
    float getNarrowness() const           { return nu; }           // spectral measure (unchanged)
    float getRegularity() const           { return R_safe; }       // harmonic-safe regularity
    float getRegularitySpectral() const   { return R_spec; }       // original R = exp(-nu)
    float getRegularityPhase() const      { return R_phase; }      // phase-coherence R in [0,1]

    // Significant wave height estimate based on harmonic-safe R
    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f || !std::isfinite(R_safe)) return 0.0f;
        // factor(R): 1 for very regular (R≈1), up to ~sqrt(2) for very irregular
        const float factor = significantHeightFactorFromR(R_safe);
        return 2.0f * std::sqrt(M0) * factor;
    }

private:
    // Time constants
    float tau_env;
    float tau_mom;
    float tau_phase;
    float tau_omega;

    // Frequencies
    float omega_min;

    // Cached step parameters
    float last_dt     = -1.0f;
    float alpha_env   = 0.0f;
    float alpha_mom   = 0.0f;
    float alpha_phase = 0.0f;
    float alpha_omega = 0.0f;

    // Mixing state
    float phi;
    float z_real, z_imag;

    // Spectral moments (displacement domain)
    float M0, M1, M2;

    // Spectral bandwidth path
    float nu;        // narrowness
    float R_spec;    // exp(-nu)

    // Phase coherence path
    float u_prev_real, u_prev_imag;
    bool  have_u_prev;
    float coh_acc_real, coh_acc_imag;
    float R_phase;   // in [0,1]

    // Blended, harmonic-safe regularity
    float R_safe;

    // Smoothed omega for 1/ω² conversion
    float omega_lp;

    // Blend configuration
    bool  use_max_blend_;
    float beta_blend_;

    // Height factor based on harmonic-safe R
    static float significantHeightFactorFromR(float R_val) {
        if (!std::isfinite(R_val)) return 1.0f;

        // Map R in [0.5, 0.98] to factor in [sqrt(2), 1]
        // Very regular (R >= 0.98): factor = 1
        // Very irregular (R <= 0.5): factor ≈ sqrt(2)
        const float R_hi = 0.98f;
        const float R_lo = 0.50f;

        if (R_val >= R_hi) return 1.0f;
        if (R_val <= R_lo) return std::sqrt(2.0f);

        float x = (R_hi - R_val) / (R_hi - R_lo); // 0 → regular, 1 → irregular
        // Smooth, monotone transition; tunable sharpness
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::tanh(3.0f * x);
    }
};
