#pragma once
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>  // for std::logic_error

/**
 * Copyright 2025, Mikhail Grushinskiy
 *
 * SeaMetrics — Online estimator of ocean wave spectral metrics
 * from vertical acceleration and an external frequency tracker.
 *
 * Core outputs (always available):
 *   • Spectral moments (raw, exponential averages):
 *        M0 = ⟨P_disp⟩
 *        M1 = ⟨P_disp·ω⟩
 *        M2 = ⟨P_disp·ω²⟩
 *   • Mean frequency:       ω̄ = M1 / M0
 *   • Relative bandwidth:   rbw = √μ₂ / ω̄, with μ₂ from decoupled two-pole stats
 *   • Regularities:
 *        R_spec = exp(−β·rbw)
 *        R_phase = ‖⟨ z/‖z‖ ⟩‖
 *   • Narrowness diagnostic: ν = √((M0·M2 / M1²) − 1)
 *   • Height estimates:
 *        Hs_reg ≈ 2√M0        (regular, monochromatic)
 *        Hs_irr ≈ 2√2√M0      (irregular, Rayleigh assumption)
 *
 * Extended outputs (only if constructed with enable_extended=true):
 *   • Higher-order raw moments:
 *        M3 = ⟨P_disp·ω³⟩
 *        M4 = ⟨P_disp·ω⁴⟩
 *
 *   • Central moments (about ω̄ = M1/M0):
 *        μ₂ = M2/M0 − (M1/M0)²
 *        μ₃ = M3/M0 − 3·μ·(M2/M0) + 2·μ³
 *        μ₄ = M4/M0 − 4·μ·(M3/M0) + 6·μ²·(M2/M0) − 3·μ⁴
 *        where μ = M1/M0
 *
 *   • Shape diagnostics:
 *        Spectral skewness        = μ₃ / μ₂^(3/2)
 *        Spectral kurtosis        = μ₄ / μ₂²
 *        Spectral excess kurtosis = (μ₄ / μ₂²) − 3
 *
 *   Notes:
 *     – For Gaussian-like spectra: skew ≈ 0, excess kurtosis ≈ 0.
 *     – Narrow, peaky spectra (e.g. JONSWAP) → positive excess kurtosis.
 *     – Skewness indicates asymmetry of the spectrum about ω̄.
 */

// Portability guard
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// std::clamp polyfill for pre-C++17
#if __cplusplus < 201703L
namespace std {
    template <class T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
        return (v < lo) ? lo : (hi < v) ? hi : v;
    }
}
#endif

class SeaMetrics {
public:
    constexpr static float EPSILON   = 1e-12f;
    constexpr static float BETA_SPEC = 1.0f;

    SeaMetrics(float tau_env_sec   = 15.0f,
               float tau_mom_sec   = 180.0f,
               float omega_min_hz  = 0.03f,
               float tau_coh_sec   = 60.0f,
               float tau_omega_sec = 45.0f,
               bool enable_extended = false)
        : extended_metrics(enable_extended)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_omega = std::max(0.0f,  tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz;
        reset();
    }

    void reset() {
        phi = 0.0f; z_real = z_imag = 0.0f;
        M0 = M1 = M2 = nu = 0.0f;
        R_spec = R_phase = 0.0f;
        rbw = 0.0f;

        if (extended_metrics) { M3 = M4 = mu2 = mu3 = mu4 = 0.0f; }

        coh_r = coh_i = 0.0f; has_coh = false;

        omega_lp = omega_disp_lp = 0.0f;
        omega_last = mu_w = 0.0f;
        var_fast = var_slow = 0.0f;

        has_omega_lp = false;
        has_omega_disp_lp = false;
        has_moments = false;

        omega_phi_last = 0.0f;
        wbar_ema = w2bar_ema = 0.0f;
    }

    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;
        if (omega_inst != omega_inst) omega_inst = omega_min;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityMetrics();
    }

    // === Raw moments ===
    float getMoment0() const { return M0; }
    float getMoment1() const { return M1; }
    float getMoment2() const { return M2; }
    float getMoment3() const {
        if (!extended_metrics) throw std::logic_error("M3 not enabled");
        return M3;
    }
    float getMoment4() const {
        if (!extended_metrics) throw std::logic_error("M4 not enabled");
        return M4;
    }

    // === Central moments ===
    float getCentralMoment2() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        return mu2;
    }
    float getCentralMoment3() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        return mu3;
    }
    float getCentralMoment4() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        return mu4;
    }

    // === Shape diagnostics ===
    float getSpectralSkewness() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return mu3 / std::pow(mu2, 1.5f);
    }
    float getSpectralKurtosis() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return mu4 / (mu2*mu2);
    }
    float getSpectralExcessKurtosis() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return (mu4 / (mu2*mu2)) - 3.0f;
    }

    // === Core metrics ===
    float getMeanFrequencyRad() const { return (M0 > EPSILON) ? (M1 / M0) : omega_min; }
    float getMeanFrequencyHz()  const { return getMeanFrequencyRad() / (2.0f * float(M_PI)); }

    float getRBW() const { return rbw; }

    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase()    const { return R_phase; }

    float getNarrowness() const { return nu; }

    float getRMSDisplacement() const { return (M0 > EPSILON) ? std::sqrt(M0) : 0.0f; }
    float getHs_reg() const { return 2.0f * getRMSDisplacement(); }
    float getHs_irr() const { return 2.0f * std::sqrt(2.0f) * getRMSDisplacement(); }

private:
    // Flag
    bool extended_metrics;

    // Time constants
    float tau_env, tau_mom, tau_coh, tau_omega;
    float omega_min;

    // Alphas
    float alpha_env, alpha_mom, alpha_coh, alpha_omega;

    // Demod state
    float phi, z_real, z_imag;

    // Raw moments
    float M0, M1, M2;
    float M3, M4;   // only valid if extended_metrics
    float nu;

    // Central moments (cached if extended)
    float mu2, mu3, mu4;

    // Metrics
    float R_spec, R_phase;
    float rbw;

    // Phase coherence
    float coh_r, coh_i;
    bool  has_coh;

    // Frequency tracking
    float omega_lp, omega_disp_lp;
    float omega_last, mu_w;
    float var_fast, var_slow;
    bool  has_omega_lp, has_omega_disp_lp, has_moments;

    // Demod ω limiter
    float omega_phi_last;

    // Two-pole RBW stats
    float wbar_ema, w2bar_ema;

    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    void updateAlpha(float dt_s) {
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
    }

    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        // Time-scaled slew limiting on ω (rad/s²) for demodulation basis
        float w_target = std::max(omega_inst, omega_min);
        if (!has_omega_lp) omega_phi_last = w_target;
        float dw = w_target - omega_phi_last;
        float dw_clamped = std::clamp(dw, -30.0f * dt_s, 30.0f * dt_s);
        float omega_phi = omega_phi_last + dw_clamped;
        omega_phi_last  = omega_phi;

        // Integrate demodulation phase
        phi += omega_phi * dt_s;
        phi = std::fmod(phi, 2.0f * float(M_PI));

        // Rotate acceleration to baseband I/Q
        float c = std::cos(-phi), s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;

        // First-time init
        if (!has_omega_lp) {
            z_real = y_real; z_imag = y_imag;
            omega_lp = w_target;
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        // Envelope EMA
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // LP the instantaneous ω
        omega_lp = (tau_omega > 0.0f)
                 ? (1.0f - alpha_omega) * omega_lp + alpha_omega * w_target
                 : w_target;

        // Track variance of ω_lp
        float delta = omega_lp - mu_w;
        mu_w    += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
        float omega_norm = omega_lp;
        float inv_w2 = 1.0f / std::max(omega_norm * omega_norm, EPSILON);
        float disp_r = z_real * inv_w2;
        float disp_i = z_imag * inv_w2;
        float P_disp = disp_r * disp_r + disp_i * disp_i;

        if (!has_moments) {
            M0 = P_disp;
            M1 = P_disp * omega_norm;
            M2 = P_disp * omega_norm * omega_norm;
            if (extended_metrics) {
                M3 = P_disp * omega_norm * omega_norm * omega_norm;
                M4 = P_disp * omega_norm * omega_norm * omega_norm * omega_norm;
            }
            has_moments = true;
        } else {
            M0 = (1.0f - alpha_mom) * M0 + alpha_mom * P_disp;
            M1 = (1.0f - alpha_mom) * M1 + alpha_mom * P_disp * omega_norm;
            M2 = (1.0f - alpha_mom) * M2 + alpha_mom * P_disp * omega_norm * omega_norm;
            if (extended_metrics) {
                M3 = (1.0f - alpha_mom) * M3 + alpha_mom * P_disp * omega_norm * omega_norm * omega_norm;
                M4 = (1.0f - alpha_mom) * M4 + alpha_mom * P_disp * omega_norm * omega_norm * omega_norm * omega_norm;
            }
        }

        // Decoupled RBW stats
        wbar_ema  = (1.0f - alpha_mom) * wbar_ema  + alpha_mom * omega_norm;
        w2bar_ema = (1.0f - alpha_mom) * w2bar_ema + alpha_mom * omega_norm * omega_norm;
    }

    void updatePhaseCoherence() {
        float mag = std::sqrt(z_real * z_real + z_imag * z_imag);
        if (mag <= 1e-6f) {
            if (!has_coh) { coh_r = 1.0f; coh_i = 0.0f; has_coh = true; }
            R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
            return;
        }

        float u_r = z_real / mag;
        float u_i = z_imag / mag;

        if (!has_coh) { coh_r = u_r; coh_i = u_i; has_coh = true; }
        else {
            coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
            coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;
        }
        R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
    }

    void computeRegularityMetrics() {
        float omega_bar = getMeanFrequencyRad();
        float mu2_tmp = std::max(0.0f, w2bar_ema - wbar_ema*wbar_ema - std::max(var_slow, 0.0f));
        rbw = (omega_bar > 0.0f) ? (std::sqrt(mu2_tmp) / omega_bar) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);

        // Legacy narrowness diagnostic
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0*M2) / (M1*M1) - 1.0f;
            nu = (ratio > 0.0f) ? std::sqrt(ratio) : 0.0f;
        } else {
            nu = 0.0f;
        }

        // Extended central moments
        if (extended_metrics && M0 > EPSILON) {
            float mu = M1 / M0;
            float m2 = M2 / M0;
            float m3 = M3 / M0;
            float m4 = M4 / M0;

            mu2 = m2 - mu*mu;
            mu3 = m3 - 3*mu*m2 + 2*mu*mu*mu;
            mu4 = m4 - 4*mu*m3 + 6*mu*mu*m2 - 3*mu*mu*mu*mu;
        } else {
            mu2 = mu3 = mu4 = 0.0f;
        }
    }
};
