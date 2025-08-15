#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec   = 1.0f,
                       float tau_mom_sec   = 60.0f,
                       float omega_min_hz  = 0.03f,
                       float tau_coh_sec   = 20.0f,
                       float tau_out_sec   = 15.0f,
                       float tau_omega_sec = 0.0f)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_out   = std::max(1e-3f, tau_out_sec);
        tau_omega = std::max(0.0f, tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz;
        reset();
    }

    void reset() {
        phi = z_real = z_imag = 0.0f;
        M0 = M1 = M2 = 0.0f;
        nu = R_spec = R_phase = R_safe = R_out = std::numeric_limits<float>::quiet_NaN();

        alpha_env = alpha_mom = alpha_coh = alpha_out = alpha_omega = 0.0f;
        last_dt = -1.0f;

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
            alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
            alpha_out   = 1.0f - std::exp(-dt_s / tau_out);
            alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
            last_dt     = dt_s;
        }

        // --- demodulate acceleration ---
        phi += omega_inst * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // --- displacement conversion ---
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

        // --- spectral R ---
        if (M1 > 1e-12f && M0>0.0f && M2>0.0f) {
            float ratio = (M0*M2)/(M1*M1) - 1.0f;
            ratio = std::max(0.0f, ratio);
            nu = std::sqrt(ratio);
            R_spec = std::clamp(std::exp(-nu), 0.0f, 1.0f);
        } else {
            nu = 0.0f;
            R_spec = 0.0f;
        }

        // --- phase-coherence path ---
        float mag = std::sqrt(z_real*z_real + z_imag*z_imag);
        float u_r = (mag>1e-12f) ? z_real/mag : 1.0f;
        float u_i = (mag>1e-12f) ? z_imag/mag : 0.0f;

        coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
        coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;

        R_phase = std::clamp(std::sqrt(coh_r*coh_r + coh_i*coh_i), 0.0f, 1.0f);

        // --- harmonic-safe R ---
        R_safe = std::max(R_spec, R_phase);

        // --- target R with selective boost/reduction ---
        float R_target = R_safe;

        // Pull down JONSWAP-like moderate waves for separation
        const float P_jonswap = 0.3f;  // typical JONSWAP threshold
        const float reduction_max = 0.08f; // 8% max reduction
        if (P_disp < P_jonswap) {
            float reduce = reduction_max * (1.0f - P_disp/P_jonswap);
            R_target = std::max(R_target - reduce, 0.0f);
        }

        // Gradually boost large nonlinear waves
        const float P_thr = 0.5f;       // threshold for "large wave"
        const float boost_max = 0.12f;  // 12% max R boost
        if (P_disp > P_thr) {
            float boost = boost_max * std::min(P_disp / P_thr, 2.0f);
            R_target = std::min(R_target + boost, 1.0f);
        }

        // --- smooth output ---
        if (!std::isfinite(R_out)) R_out = R_target;
        else R_out = (1.0f - alpha_out) * R_out + alpha_out * R_target;
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
    float tau_env, tau_mom, tau_coh, tau_out, tau_omega;
    float omega_min;

    float last_dt;
    float alpha_env, alpha_mom, alpha_coh, alpha_out, alpha_omega;

    float phi, z_real, z_imag;
    float M0, M1, M2;
    float nu;
    float R_spec, R_phase, R_safe, R_out;

    float coh_r, coh_i;
    float omega_lp;

    static float heightFactorFromR(float R_val) {
        const float R_hi = 0.98f;
        const float R_lo = 0.50f;
        if (!std::isfinite(R_val) || R_val >= R_hi) return 1.0f;
        if (R_val <= R_lo) return std::sqrt(2.0f);
        float x = (R_hi - R_val)/(R_hi - R_lo);
        return 1.0f + (std::sqrt(2.0f)-1.0f)*std::pow(x, 1.5f);
    }
};
