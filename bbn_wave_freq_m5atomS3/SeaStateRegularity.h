#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

class SeaStateRegularity {
public:
    SeaStateRegularity(float tau_env_sec = 1.0f,
                       float tau_mom_sec = 60.0f,
                       float omega_min_hz = 0.03f)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
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
        last_dt = -1.0f;

        // Phase-regularity state
        omega_ref = std::numeric_limits<float>::quiet_NaN();
        phase_dev_accum = 0.0f;
        phase_dev_count = 0;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (dt_s != last_dt) {
            alpha_env = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
            last_dt   = dt_s;
        }

        // Rotate acceleration into phase frame
        phi += omega_inst * dt_s;
        if (phi > 1e6f) phi -= 1e6f;

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

        // --- Spectral regularity (narrowness) ---
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

        // --- Phase regularity ---
        float R_phase = phaseRegularity(omega_inst);

        // --- Blend instead of max ---
        const float R_thresh_low  = 0.7f;
        const float R_thresh_high = 0.9f;
        float w_blend = smoothstep(R_spec, R_thresh_low, R_thresh_high);

        R = (1.0f - w_blend) * R_spec + w_blend * R_phase;
    }

    float getNarrowness() const { return nu; }
    float getRegularity() const { return R; }

    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f) return 0.0f;
        float factor = (R > 0.9f) ? 1.0f : significantHeightFactorFromR(R);
        return 2.0f * std::sqrt(M0) * factor;
    }

private:
    // Core state
    float tau_env;
    float tau_mom;
    float omega_min;
    float last_dt;
    float alpha_env;
    float alpha_mom;

    // Rotated signal state
    float phi;
    float z_real, z_imag;

    // Spectral moments
    float M0, M1, M2;
    float nu;  // narrowness
    float R;   // blended regularity

    // Phase-regularity tracking
    float omega_ref;
    float phase_dev_accum;
    int   phase_dev_count;

    static float smoothstep(float x, float edge0, float edge1) {
        x = (x - edge0) / (edge1 - edge0);
        x = std::clamp(x, 0.0f, 1.0f);
        return x * x * (3.0f - 2.0f * x);
    }

    float phaseRegularity(float omega_inst) {
        if (!std::isfinite(omega_ref))
            omega_ref = omega_inst;

        // Slow track of mean frequency
        float alpha_ref = 0.02f;
        omega_ref = (1.0f - alpha_ref) * omega_ref + alpha_ref * omega_inst;

        // Relative deviation
        float dev = std::fabs(omega_inst - omega_ref) / omega_ref;
        phase_dev_accum += dev;
        phase_dev_count++;

        // Update every ~1s worth of samples (assuming ~dt_s << 1s)
        if (phase_dev_count > 20) {
            float avg_dev = phase_dev_accum / phase_dev_count;
            phase_dev_accum = 0.0f;
            phase_dev_count = 0;

            float phase_nu = avg_dev; // proxy for spectral spread
            float R_phase = std::exp(-phase_nu);
            return R_phase;
        }

        return std::exp(-(phase_dev_accum / std::max(phase_dev_count,1)));
    }

    static float significantHeightFactorFromR(float R_val) {
        constexpr float R_min = 0.5f;
        constexpr float R_max = 0.92f;
        if (!std::isfinite(R_val) || R_val <= R_min) return 1.0f;

        float clipped_R = std::min(R_val, R_max);
        float x = (clipped_R - R_min) / (R_max - R_min);
        return 1.0f + (std::sqrt(2.0f) - 1.0f) * std::tanh(3.0f * x);
    }
};
