#pragma once
#include <cmath>
#include <limits>

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
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (dt_s != last_dt) {
            alpha_env = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
            last_dt   = dt_s;
        }

        phi += omega_inst * dt_s;
        if (phi > 1e6f) phi -= 1e6f;

        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        float P = z_real * z_real + z_imag * z_imag;

        float w = (omega_inst > omega_min) ? omega_inst : omega_min;
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P * std::pow(w, -4.0f);
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P * std::pow(w, -3.0f);
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P * std::pow(w, -2.0f);

        if (M1 > 1e-12f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu = std::sqrt(ratio);
            R  = std::exp(-nu);
        } else {
            nu = std::numeric_limits<float>::quiet_NaN();
            R  = std::numeric_limits<float>::quiet_NaN();
        }
    }

    float getNarrowness() const { return nu; }
    float getRegularity() const { return R; }

    float getSignificantWaveHeightEnvelope() const {
        if (M0 <= 0.0f) return 0.0f;
        return 4.0f * std::sqrt(2.0f * M0);
    }

    float getSignificantWaveHeightEst() const {
        if (M0 <= 0.0f) return 0.0f;
        float factor = significantHeightFactor(nu);
        return 2.0f * std::sqrt(M0) * factor;
    }

private:
    float tau_env;
    float tau_mom;
    float omega_min;

    float last_dt = -1.0f;
    float alpha_env = 0.0f;
    float alpha_mom = 0.0f;

    float phi;
    float z_real, z_imag;
    float M0, M1, M2;
    float nu;
    float R;

    // Separate function to calculate correction factor based on bandwidth
    static float significantHeightFactor(float nu_val) {
        const nu_lim = 0.5f;
        if (!std::isfinite(nu_val) || nu_val <= nu_lim) return 1.0f; // pure sine
        // Interpolation from 1.0 to sqrt(2)
        float factor = 1.0f + (std::sqrt(2.0f) - 1.0f) * std::tanh(3.0f * (nu_val - nu_lim));
        return factor;
    }
};
