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

    /**
     * Update filter with one new acceleration sample and instantaneous frequency.
     *
     * @param dt_s       Sample time step (seconds)
     * @param accel_z    Vertical acceleration (m/s^2)
     * @param omega_inst Instantaneous angular frequency (rad/s)
     */
    void update(float dt_s, float accel_z, float omega_inst) {
        // update alphas if dt changed
        if (dt_s != last_dt) {
            alpha_env = 1.0f - std::exp(-dt_s / tau_env);
            alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
            last_dt   = dt_s;
        }

        // integrate phase
        phi += omega_inst * dt_s;
        if (phi > 1e6f) phi -= 1e6f; // avoid overflow (no need for strict wrapping)

        // demodulate
        float c = std::cos(-phi);
        float s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        // envelope filter (complex one-pole LPF)
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // instantaneous power
        float P = z_real * z_real + z_imag * z_imag;

        // moment updates
        float w = (omega_inst > omega_min) ? omega_inst : omega_min;
        M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P * std::pow(w, -4.0f);
        M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P * std::pow(w, -3.0f);
        M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P * std::pow(w, -2.0f);

        // compute narrowness nu
        if (M1 > 1e-12f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            nu = std::sqrt(ratio);
            R  = std::exp(-nu); // optional mapping to 0–1
        } else {
            nu = std::numeric_limits<float>::quiet_NaN();
            R  = std::numeric_limits<float>::quiet_NaN();
        }
    }

    float getNarrowness() const { return nu; }   // Longuet–Higgins ν
    float getRegularity() const { return R; }    // 0–1 score (1=very regular)

private:
    float tau_env;    // envelope smoothing time constant [s]
    float tau_mom;    // moment smoothing time constant [s]
    float omega_min;  // lower limit for ω to avoid blow-up

    float last_dt = -1.0f;
    float alpha_env = 0.0f;
    float alpha_mom = 0.0f;

    float phi;        // reference phase
    float z_real, z_imag; // complex envelope state
    float M0, M1, M2; // spectral moment estimates
    float nu;         // current bandwidth estimate
    float R;          // regularity score
};
