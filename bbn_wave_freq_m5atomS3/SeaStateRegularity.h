#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec   = 1.0f,
                       float tau_mom_sec   = 60.0f,
                       float omega_min_hz  = 0.03f,
                       float tau_ref_sec   = 60.0f,
                       float tau_coh_sec   = 10.0f,
                       float tau_out_sec   = 10.0f,
                       float tau_omega_sec = 0.0f)
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
        phi = z_real = z_imag = 0.0f;
        M0 = M1 = M2 = 0.0f;
        nu = R_spec = R_phase = R_out = std::numeric_limits<float>::quiet_NaN();

        alpha_env = alpha_mom = alpha_ref = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

        omega_ref = omega_min;
        coh_r = 1.0f;
        coh_i = 0.0f;
        omega_lp = omega_min;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (!std::isfinite(dt_s) || dt_s <= 0.0f) return;
        if (!std::isfinite(accel_z)) accel_z = 0.0f;
        if (!std::isfinite(omega_inst)) omega_inst = omega_min;

        if (dt_s != last_dt) {
            alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
            alpha_ref   = 1.0f - std::exp(-dt_s / tau_ref);
            alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
            alpha_out   = 1.0f - std::exp(-dt_s / tau_out);
            alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
            last_dt     = dt_s;
        }

        // --- demodulate accel ---
        phi += omega_inst * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // --- displacement ---
        float w_inst = std::max(omega_inst, omega_min);
        omega_lp = (tau_omega > 0.0f) ? ((1.0f - alpha_omega) * omega_lp + alpha_omega * w_inst) : w_inst;
        float inv_w2 = 1.0f / (omega_lp * omega_lp);

        float disp_real = z_real * inv_w2;
        float disp_imag = z_imag * inv_w2;
        float P_disp = disp_real*disp_real + disp_imag*disp_imag;

        // --- spectral moments ---
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * omega_lp;
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * omega_lp * omega_lp;

        // --- spectral regularity ---
        if (M1 > 1e-12f && M0>0.0f && M2>0.0f) {
            float ratio = (M0*M2)/(M1*M1) - 1.0f;
            ratio = std::max(0.0f, ratio);
            nu = std::sqrt(ratio);
            R_spec = std::clamp(std::exp(-nu), 0.0f, 1.0f);
        } else {
            nu = 0.0f;
            R_spec = 0.0f;
        }

        // --- phase regularity ---
        omega_ref = (1.0f - alpha_ref)*omega_ref + alpha_ref*omega_lp;
        float dtheta = (omega_lp - omega_ref) * dt_s;
        float cd = std::cos(dtheta), sd = std::sin(dtheta);
        coh_r = (1.0f - alpha_coh)*coh_r + alpha_coh*cd;
        coh_i = (1.0f - alpha_coh)*coh_i + alpha_coh*sd;
        R_phase = std::sqrt(coh_r*coh_r + coh_i*coh_i);
        R_phase = std::clamp(R_phase, 0.0f, 1.0f);

        // --- conservative blending ---
        float R_lo = 0.85f;
        float R_hi = 0.98f;
        float w_phase = smoothstep(R_spec, R_lo, R_hi);
        // limit the max phase rescue to 0.1
        float delta = (R_phase - R_spec) * w_phase;
        if (delta > 0.1f) delta = 0.1f;

        float R_blend = R_spec + delta;

        // --- output smoothing ---
        if (!std::isfinite(R_out)) R_out = R_blend;
        else R_out = (1.0f - alpha_out)*R_out + alpha_out*R_blend;
        R_out = std::clamp(R_out, 0.0f, 1.0f);
    }

    float getNarrowness() const { return nu; }
    float getRegularity() const { return R_out; }
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }

    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f || !std::isfinite(R_out)) return 0.0f;
        return 2.0f * std::sqrt(M0) * heightFactorFromR(R_out);
    }

private:
    float tau_env, tau_mom, tau_ref, tau_coh, tau_out, tau_omega;
    float omega_min;

    float last_dt;
    float alpha_env, alpha_mom, alpha_ref, alpha_coh, alpha_out, alpha_omega;

    float phi, z_real, z_imag;
    float M0, M1, M2;
    float nu;
    float R_spec, R_phase, R_out;

    float omega_ref, coh_r, coh_i, omega_lp;

    static float smoothstep(float x, float edge0, float edge1) {
        if (edge1 <= edge0) return (x >= edge1) ? 1.0f : 0.0f;
        x = (x - edge0)/(edge1 - edge0);
        x = std::clamp(x,0.0f,1.0f);
        return x*x*(3.0f - 2.0f*x);
    }

    static float heightFactorFromR(float R_val) {
        const float R_hi = 0.98f;
        const float R_lo = 0.50f;
        if (!std::isfinite(R_val) || R_val >= R_hi) return 1.0f;
        if (R_val <= R_lo) return std::sqrt(2.0f);
        float x = (R_hi - R_val)/(R_hi - R_lo);
        return 1.0f + (std::sqrt(2.0f)-1.0f)*std::pow(x, 1.5f);
    }
};
