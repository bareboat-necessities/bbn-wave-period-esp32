#pragma once
#include <cmath>
#include <limits>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec   = 1.0f,
                       float tau_mom_sec   = 60.0f,
                       float tau_phase_sec = 20.0f,
                       float omega_min_hz  = 0.03f)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_phase = tau_phase_sec;
        omega_min = 2.0f * M_PI * omega_min_hz; // rad/s
        reset();
    }

    void reset() {
        phi = 0.0f;
        z_real = 0.0f;
        z_imag = 0.0f;
        M0 = M1 = M2 = 0.0f;
        nu = std::numeric_limits<float>::quiet_NaN();
        R  = std::numeric_limits<float>::quiet_NaN();
        alpha_env = 0.0f;
        alpha_mom = 0.0f;
        alpha_phase = 0.0f;
        last_dt = -1.0f;

        omega_ref = omega_min;
        coh_acc_real = 0.0f;
        coh_acc_imag = 0.0f;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (dt_s != last_dt) {
            alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
            alpha_phase = 1.0f - std::exp(-dt_s / tau_phase);
            last_dt     = dt_s;
        }

        // Rotate acceleration into phase
        phi += omega_inst * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // Convert acceleration to displacement-equivalent
        float w = (omega_inst > omega_min) ? omega_inst : omega_min;
        float disp_real = z_real / (w * w);
        float disp_imag = z_imag / (w * w);

        float P_disp = disp_real * disp_real + disp_imag * disp_imag;

        // Update spectral moments using displacement
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * w;
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * w * w;

        // --- Regularity from spectral moments ---
        float R_spec;
        if (M1 > 1e-12f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu = std::sqrt(ratio);
            R_spec = std::exp(-nu);
        } else {
            nu = std::numeric_limits<float>::quiet_NaN();
            R_spec = std::numeric_limits<float>::quiet_NaN();
        }

        // --- Harmonic-insensitive regularity via frequency jitter coherence ---
        const float w_inst_clamped = std::isfinite(omega_inst) ? std::max(omega_inst, omega_min) : omega_min;
        omega_ref = (1.0f - alpha_phase) * omega_ref + alpha_phase * w_inst_clamped;

        float dtheta = (w_inst_clamped - omega_ref) * dt_s;
        float cdt = std::cos(dtheta);
        float sdt = std::sin(dtheta);
        coh_acc_real = (1.0f - alpha_phase) * coh_acc_real + alpha_phase * cdt;
        coh_acc_imag = (1.0f - alpha_phase) * coh_acc_imag + alpha_phase * sdt;

        float R_phase = std::sqrt(coh_acc_real * coh_acc_real + coh_acc_imag * coh_acc_imag);
        if (R_phase > 1.0f) R_phase = 1.0f;

        // Combine metrics: keep high R for regular nonlinear waves
        R = std::max(R_spec, R_phase);
    }

    float getNarrowness() const { return nu; }
    float getRegularity() const { return R; }

    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f) return 0.0f;

        float factor = (R > 0.95f) ? 1.0f : significantHeightFactorFromR(R);
        return 2.0f * std::sqrt(M0) * factor;
    }

private:
    float tau_env;
    float tau_mom;
    float tau_phase;
    float omega_min;

    float last_dt = -1.0f;
    float alpha_env = 0.0f;
    float alpha_mom = 0.0f;
    float alpha_phase = 0.0f;

    float phi;
    float z_real, z_imag;
    float M0, M1, M2;
    float nu;
    float R;

    float omega_ref;
    float coh_acc_real, coh_acc_imag;

    static float significantHeightFactorFromR(float R_val) {
        constexpr float R_min = 0.3f;
        constexpr float R_max = 0.95f;
        if (!std::isfinite(R_val) || R_val >= R_max) return 1.0f;

        float clipped_R = std::max(R_val, R_min);
        float x = (R_max - clipped_R) / (R_max - R_min);
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::tanh(3.0f * x);
    }
};
