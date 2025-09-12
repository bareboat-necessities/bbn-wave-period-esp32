#pragma once
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>  // std::logic_error

#pragma once
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>  // std::logic_error

/**
 * Copyright 2025, Mikhail Grushinskiy
 *
 * SeaMetrics — Online estimator of ocean wave spectral metrics
 * from vertical acceleration and an external frequency tracker.
 *
 * === Raw spectral moments ===
 *   • M0 = ⟨P_disp⟩
 *   • M1 = ⟨P_disp·ω⟩
 *   • M2 = ⟨P_disp·ω²⟩
 *   • M3 = ⟨P_disp·ω³⟩   [extended only]
 *   • M4 = ⟨P_disp·ω⁴⟩   [extended only]
 *   • M_{−1} = ⟨P_disp·ω^{−1}⟩   [negative moments only]
 *
 * === Central moments (extended only) ===
 *   • μ₂ = M2/M0 − (M1/M0)²
 *   • μ₃ = M3/M0 − 3·μ·(M2/M0) + 2·μ³
 *   • μ₄ = M4/M0 − 4·μ·(M3/M0) + 6·μ²·(M2/M0) − 3·μ⁴
 *     where μ = M1/M0
 *
 * === Frequency metrics ===
 *   • Mean frequency (rad/s): ω̄ = M1/M0
 *   • Mean frequency (Hz):    f̄ = ω̄ / (2π)
 *   • Relative bandwidth (RBW): √μ₂ / ω̄   (μ₂ from decoupled two-pole stats)
 *
 * === Regularity metrics ===
 *   • R_spec = exp(−β·RBW)
 *   • R_phase = ‖⟨ z/‖z‖ ⟩‖
 *   • Narrowness ν = √((M0 M2 / M1²) − 1)
 *
 * === Shape diagnostics (extended only) ===
 *   • Spectral skewness        = μ₃ / μ₂^(3/2)
 *   • Spectral kurtosis        = μ₄ / μ₂²
 *   • Spectral excess kurtosis = (μ₄ / μ₂²) − 3
 *   • Ochi peakedness Q        = (M0 M4) / M2²
 *   • Benassai parameter B     = (M0 M4) / M2²
 *
 * === Period summaries (s) ===
 *   • T_z, T_zup, T_zdown
 *   • ν_up, ν_down
 *   • Wave count + Garwood CI
 *   • T_1, T_m02, T_e
 *   • T_{m0,-1}, T_{m1,-1}
 *   • Mean group period
 *
 * === Heights & steepness ===
 *   • RMS displacement
 *   • Hs (regular, Rayleigh)
 *   • Wave steepness (Hs/L0, deep water)
 *
 * === Probability metrics ===
 *   • Crest exceedance (Rayleigh): P(Hc > h)
 *
 * === Bandwidths ===
 *   • CLH, Goda, Kuik
 *   • Longuet–Higgins width
 */


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// std::clamp polyfill for pre-C++17 (kept harmless on C++17+)
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
    // Numeric constants
    constexpr static float EPSILON      = 1e-12f;
    constexpr static float BETA_SPEC    = 1.0f;     // R_spec = exp(−β·RBW)
    constexpr static float OMEGA_DD_MAX = 30.0f;    // rad/s² max slew for demod basis

    struct WaveCountEstimate {
        float expected;   // expected count
        float ci_lower;   // lower bound (Garwood)
        float ci_upper;   // upper bound (Garwood)
        float confidence; // confidence level used
    };

    // Constructor
    SeaMetrics(float tau_env_sec   = 15.0f,
               float tau_mom_sec   = 180.0f,
               float omega_min_hz  = 0.03f,
               float tau_coh_sec   = 60.0f,
               float tau_omega_sec = 45.0f,
               bool enable_extended = false,
               bool enable_negative = false)
        : extended_metrics(enable_extended),
          negative_moments(enable_negative)
    {
        tau_env   = tau_env_sec;
        tau_mom   = tau_mom_sec;
        tau_coh   = std::max(1e-3f, tau_coh_sec);
        tau_omega = std::max(0.0f,  tau_omega_sec);
        omega_min = 2.0f * float(M_PI) * omega_min_hz; // Hz → rad/s
        reset();
    }

    void reset() {
        // Phase / envelope
        phi = 0.0f; z_real = z_imag = 0.0f;

        // Raw moments & metrics
        M0 = M1 = M2 = 0.0f;
        nu = 0.0f;
        R_spec = R_phase = 0.0f;
        rbw = 0.0f;

        if (extended_metrics) { M3 = M4 = mu2 = mu3 = mu4 = 0.0f; }
        if (negative_moments) { M_neg1 = 0.0f; }

        // Coherence
        coh_r = coh_i = 0.0f;
        has_coh = false;

        // Frequency trackers
        omega_lp = omega_disp_lp = 0.0f;
        omega_last = mu_w = 0.0f;
        var_fast = var_slow = 0.0f;

        // Flags
        has_omega_lp = false;
        has_omega_disp_lp = false;
        has_moments = false;

        // Demod ω limiter
        omega_phi_last = 0.0f;

        // Two-pole RBW stats
        wbar_ema = w2bar_ema = 0.0f;
    }

    // Main update: dt (s), vertical acceleration (m/s²), instantaneous omega (rad/s)
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;   // NaN guards
        if (omega_inst != omega_inst) omega_inst = omega_min;

        updateAlpha(dt_s);
        demodulateAcceleration(accel_z, omega_inst, dt_s);
        updateSpectralMoments();
        updatePhaseCoherence();
        computeRegularityMetrics();
    }

    // === Raw spectral moments ===
    float getMomentMinus1() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        return M_neg1;
    }
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

    // === Central moments (extended only) ===
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

    // === Frequency metrics ===
    float getMeanFrequencyRad() const { return (M0 > EPSILON) ? (M1 / M0) : omega_min; }
    float getMeanFrequencyHz()  const { return getMeanFrequencyRad() / (2.0f * float(M_PI)); }
    float getRBW() const { return rbw; }

    // === Regularity metrics ===
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase()    const { return R_phase; }
    float getNarrowness() const { return nu; }

    // === Shape diagnostics (extended only) ===
    float getSpectralSkewness() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return mu3 / std::pow(mu2, 1.5f);
    }
    float getSpectralKurtosis() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return mu4 / (mu2 * mu2);
    }
    float getSpectralExcessKurtosis() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (mu2 <= EPSILON) return 0.0f;
        return (mu4 / (mu2 * mu2)) - 3.0f;
    }

    // === Period summaries (s) & rates (Hz) ===
    float getMeanPeriod_Tz() const {
        if (M2 <= EPSILON) return 0.0f;
        return std::sqrt(2.0f * float(M_PI) * float(M_PI) * (M0 / M2));
    }
    float getMeanPeriod_TzDown() const { return getMeanPeriod_Tz(); }
    float getMeanPeriod_TzUp()   const { return getMeanPeriod_Tz(); }

    float getUpcrossingRate() const {
        if (M0 <= EPSILON) return 0.0f;
        return (1.0f / (2.0f * float(M_PI))) * std::sqrt(M2 / M0);
    }
    float getDowncrossingRate() const { return getUpcrossingRate(); }

    float estimateWaveCount(float duration_s) const {
        if (duration_s <= 0.0f) return 0.0f;
        return getUpcrossingRate() * duration_s;
    }

    WaveCountEstimate estimateWaveCountWithCI(float duration_s, float confidence = 0.95f) const {
        WaveCountEstimate out{0.0f, 0.0f, 0.0f, confidence};
        if (duration_s <= 0.0f) return out;

        // Use expected count as Poisson 'observation' for Garwood CI
        float Nexp = estimateWaveCount(duration_s);
        int   N    = (int)std::round(Nexp);

        float alpha = std::max(0.0f, std::min(1.0f, 1.0f - confidence));
        float lower, upper;

        if (N == 0) {
            lower = 0.0f;
            upper = 0.5f * chi2Quantile(1.0f - alpha / 2.0f, 2 * (N + 1));
        } else {
            lower = 0.5f * chi2Quantile(alpha / 2.0f,            2 * N);
            upper = 0.5f * chi2Quantile(1.0f - alpha / 2.0f, 2 * (N + 1));
        }

        out.expected = Nexp;
        out.ci_lower = lower;
        out.ci_upper = upper;
        return out;
    }

    float getMeanPeriod_T1() const {
        if (M1 <= EPSILON) return 0.0f;
        return (2.0f * float(M_PI) * M0) / M1;
    }
    float getMeanPeriod_Tm02() const {
        if (M2 <= EPSILON) return 0.0f;
        return 2.0f * float(M_PI) * std::sqrt(M0 / M2);
    }
    float getMeanPeriod_Te() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        if (M0 <= EPSILON) return 0.0f;
        return (2.0f * float(M_PI) * M_neg1) / M0;
    }

    // === Displacement & wave heights ===
    float getRMSDisplacement() const {
        return (M0 > EPSILON) ? std::sqrt(M0) : 0.0f;
    }
    float getSignificantWaveHeightRegular() const {
        return 2.0f * getRMSDisplacement(); // 2√M0
    }
    float getSignificantWaveHeightRayleigh() const {
        return 2.0f * std::sqrt(2.0f) * getRMSDisplacement(); // 2√2√M0
    }

    // === Classical bandwidth measures ===
    float getBandwidthCLH() const {
        if (M0 <= EPSILON || M2 <= EPSILON) return 0.0f;
        float ratio = (M1 * M1) / (M0 * M2);
        ratio = std::clamp(ratio, 0.0f, 1.0f);
        return std::sqrt(1.0f - ratio);
    }
    float getBandwidthGoda() const {
        if (M0 <= EPSILON || M1 <= EPSILON) return 0.0f;
        float ratio = (M0 * M2) / (M1 * M1);
        return (ratio > 1.0f) ? std::sqrt(ratio - 1.0f) : 0.0f;
    }
    float getBandwidthKuik() const {
        if (M1 <= EPSILON) return 0.0f;
        float val = (M0 * M2) - (M1 * M1);
        return (val > 0.0f) ? std::sqrt(val) / M1 : 0.0f;
    }

private:
    // Flags
    bool extended_metrics;   // enable M3, M4 and central moments/skew/kurt
    bool negative_moments;   // enable M_{-1} and T_e

    // Time constants
    float tau_env, tau_mom, tau_coh, tau_omega;
    float omega_min;

    // EMA alphas
    float alpha_env = 0.0f, alpha_mom = 0.0f, alpha_coh = 0.0f, alpha_omega = 0.0f;

    // Demod/envelope state
    float phi = 0.0f;
    float z_real = 0.0f, z_imag = 0.0f;

    // Raw moments & metrics
    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f;
    float M3 = 0.0f, M4 = 0.0f;         // extended
    float M_neg1 = 0.0f;                // negative
    float mu2 = 0.0f, mu3 = 0.0f, mu4 = 0.0f; // central moments (cached when extended)
    float nu = 0.0f;
    float R_spec = 0.0f, R_phase = 0.0f;
    float rbw = 0.0f;

    // Phase coherence
    float coh_r = 0.0f, coh_i = 0.0f;
    bool  has_coh = false;

    // Frequency tracking
    float omega_lp = 0.0f, omega_disp_lp = 0.0f;
    float omega_last = 0.0f, mu_w = 0.0f;
    float var_fast = 0.0f, var_slow = 0.0f;
    bool  has_omega_lp = false, has_omega_disp_lp = false, has_moments = false;

    // Demod ω slew limiter memory
    float omega_phi_last = 0.0f;

    // Decoupled running means for RBW (two-pole)
    float wbar_ema = 0.0f;   // ⟨ω⟩
    float w2bar_ema = 0.0f;  // ⟨ω²⟩

    // Variance adaptation for ω tracker (fast/slow)
    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    // ---- internals ----
    void updateAlpha(float dt_s) {
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
    }

    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        // Slew-limited demodulation basis ω (rad/s)
        float w_target = std::max(omega_inst, omega_min);
        if (!has_omega_lp) omega_phi_last = w_target;
        float dw = w_target - omega_phi_last;
        float dw_clamped = std::clamp(dw, -OMEGA_DD_MAX * dt_s, OMEGA_DD_MAX * dt_s);
        float omega_phi = omega_phi_last + dw_clamped;
        omega_phi_last  = omega_phi;

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
            omega_lp = w_target;
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        // Envelope EMA
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // LP the instantaneous ω from external estimator
        omega_lp = (tau_omega > 0.0f)
                 ? (1.0f - alpha_omega) * omega_lp + alpha_omega * w_target
                 : w_target;

        // Track slow variance of ω_lp (stabilizes RBW)
        float delta = omega_lp - mu_w;
        mu_w    += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
        // accel→disp envelope via 1/ω_lp² (lean normalization)
        float omega_norm = std::max(omega_lp, omega_min);
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
            if (negative_moments) {
                M_neg1 = P_disp / omega_norm; // guarded by omega_min
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
            if (negative_moments) {
                M_neg1 = (1.0f - alpha_mom) * M_neg1 + alpha_mom * (P_disp / omega_norm);
            }
        }

        // RBW stats (decoupled two-pole)
        wbar_ema  = (1.0f - alpha_mom) * wbar_ema  + alpha_mom * omega_norm;
        w2bar_ema = (1.0f - alpha_mom) * w2bar_ema + alpha_mom * omega_norm * omega_norm;

        // Moment-based mean ω for readouts (LP)
        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (!has_omega_disp_lp) {
            omega_disp_lp = omega_disp;
            has_omega_disp_lp = true;
        } else {
            omega_disp_lp = (1.0f - alpha_omega) * omega_disp_lp + alpha_omega * omega_disp;
        }
    }

    void updatePhaseCoherence() {
        // R_phase = ‖E[u]‖ with u = z / ‖z‖
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
        // RBW via μ2_two_pole / ω̄
        float omega_bar = getMeanFrequencyRad();
        float mu2_two_pole = std::max(0.0f, w2bar_ema - wbar_ema * wbar_ema - std::max(var_slow, 0.0f));
        rbw = (omega_bar > 0.0f) ? (std::sqrt(mu2_two_pole) / omega_bar) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);

        // Narrowness (legacy)
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            nu = (ratio > 0.0f) ? std::sqrt(ratio) : 0.0f;
        } else {
            nu = 0.0f;
        }

        // Extended: cache central moments from raw moments
        if (extended_metrics && M0 > EPSILON) {
            float mu = M1 / M0;
            float m2 = M2 / M0;
            float m3 = M3 / M0;
            float m4 = M4 / M0;

            mu2 = m2 - mu * mu;
            mu3 = m3 - 3.0f * mu * m2 + 2.0f * mu * mu * mu;
            mu4 = m4 - 4.0f * mu * m3 + 6.0f * mu * mu * m2 - 3.0f * mu * mu * mu * mu;
        } else {
            mu2 = mu3 = mu4 = 0.0f;
        }
    }

    // ---- math helpers ----

    // Fast inverse error function approximation (C++17 compatible).
    // Winitzki initial approximation + one Halley refinement.
    static float erfinv_approx(float x) {
        // clamp domain
        x = std::clamp(x, -0.999999f, 0.999999f);

        // Winitzki (2008) initial guess
        const float a = 0.147f;
        float ln = std::log(1.0f - x * x);
        float tt1 = 2.0f / (float(M_PI) * a) + 0.5f * ln;
        float tt2 = 1.0f / a * ln;
        float sign = (x < 0.0f) ? -1.0f : 1.0f;
        float y = sign * std::sqrt(std::sqrt(tt1 * tt1 - tt2) - tt1);

        // Halley's method refinement on erf(y) - x = 0
        auto erf_approx = [](float t)->float { return std::erf(t); };
        for (int i = 0; i < 2; ++i) {
            float ey = erf_approx(y) - x;
            float dy = (2.0f / std::sqrt(float(M_PI))) * std::exp(-y * y);      // derivative
            float d2y = -2.0f * y * dy;                                         // second derivative of erf
            float denom = 2.0f * dy * dy - ey * d2y;
            if (std::fabs(denom) < 1e-12f) break;
            y -= (2.0f * ey * dy) / denom;
        }
        return y;
    }

    static float normalQuantile(float p) {
        // z = sqrt(2) * erfinv(2p - 1)
        p = std::clamp(p, 1e-7f, 1.0f - 1e-7f);
        return std::sqrt(2.0f) * erfinv_approx(2.0f * p - 1.0f);
    }

    // Chi-square quantile via Wilson–Hilferty transform
    static float chi2Quantile(float p, int k) {
        if (p <= 0.0f) return 0.0f;
        if (p >= 1.0f) return std::numeric_limits<float>::infinity();
        float z = normalQuantile(p);
        float v = (float)k;
        float h = 2.0f / (9.0f * v);
        return v * std::pow(1.0f - h + z * std::sqrt(h), 3.0f);
    }
};
