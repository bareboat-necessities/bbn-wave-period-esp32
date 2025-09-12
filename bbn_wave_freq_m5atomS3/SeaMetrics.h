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
 *   • Phase-increment RBW: variance of Δφ/dt normalized by ω̄
 *
 * === Regularity metrics ===
 *   • R_spec  = exp(−β·RBW)
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
 *   • Crest exceedance (Tayfun):  nonlinear approximation
 *
 * === Bandwidths ===
 *   • CLH, Goda, Kuik
 *   • Longuet–Higgins width
 *
 * === Extremes & groupiness ===
 *   • H1/10 crest height
 *   • H1/100 crest height
 *   • Tayfun exceedance probability
 *   • Groupiness factor (Tg/Tz)
 *   • Benjamin–Feir index (BFI)
 *
 * === Energy & wave power ===
 *   • Energy flux period (Te_flux)
 *   • Wave power per crest length (kW/m)
 *
 * === Bias-corrected metrics ===
 *
 * Apply first-order Jensen corrections for ω-tracker jitter:
 *
 *   E[1/ω^n] ≈ (1/ω̄^n)(1 + c_n σ²/ω̄²),   with coefficients
 *     n = 1 → c = 1
 *     n = 2 → c = 3
 *     n = 3 → c = 6
 *     n = 4 → c = 10
 *
 * Implemented by scaling raw spectral moments:
 *     M0c      = M0 / (1 + 10 σ²/ω̄²)
 *     M1c      = M1 / (1 +  6 σ²/ω̄²)
 *     M2c      = M2 / (1 +  3 σ²/ω̄²)
 *     M3c      = M3 / (1 +  1 σ²/ω̄²)
 *     M4c      = M4  (no correction needed)
 *     M_{−1}c  = M_{−1} / (1 + 6 σ²/ω̄²)
 *
 * where ω̄ = mu_w (mean ω) and σ² = var_slow (slow variance of ω).
 *
 * All higher-level bias-corrected getters (heights, periods, bandwidths,
 * counts, central moments, skew/kurtosis, extremes, power) are computed
 * consistently from these corrected moments. Phase-based metrics
 * (e.g. R_phase, RBW_PhaseIncrement) are unaffected.
 *
 * Notes:
 *   – Corrections reduce upward bias of energy, bandwidth, and moment-based
 *     metrics when frequency estimates jitter or acceleration carries DC.
 *   – Bias-corrected and raw getters are provided side-by-side.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    constexpr static float EPSILON      = 1e-12f;
    constexpr static float BETA_SPEC    = 1.0f;
    constexpr static float OMEGA_DD_MAX = 30.0f;

    struct WaveCountEstimate {
        float expected;
        float ci_lower;
        float ci_upper;
        float confidence;
    };

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
        omega_min = 2.0f * float(M_PI) * omega_min_hz;
        reset();
    }

    void reset() {
        phi = 0.0f; z_real = z_imag = 0.0f;
        M0 = M1 = M2 = 0.0f;
        nu = 0.0f; R_spec = R_phase = rbw = 0.0f;
        if (extended_metrics) { M3 = M4 = mu2 = mu3 = mu4 = 0.0f; }
        if (negative_moments) { M_neg1 = 0.0f; }
        coh_r = coh_i = 0.0f; has_coh = false;
        omega_lp = omega_disp_lp = 0.0f; omega_last = mu_w = 0.0f;
        var_fast = var_slow = 0.0f;
        has_omega_lp = has_omega_disp_lp = has_moments = false;
        omega_phi_last = 0.0f;
        wbar_ema = w2bar_ema = 0.0f;
        dphi_mean = dphi_var = 0.0f;
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

    float getRBW_PhaseIncrement() const {
        float omega_bar = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (omega_bar <= EPSILON) return 0.0f;
        return std::sqrt(std::max(0.0f, dphi_var)) / omega_bar;
    }

    // === Regularity metrics ===
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase()    const { return R_phase; }
    float getNarrowness()         const { return nu; }

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
    float getPeakednessOchi() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (M2 <= EPSILON) return 0.0f;
        return (M0 * M4) / (M2 * M2);
    }
    float getBenassaiParameter() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (M2 <= EPSILON) return 0.0f;
        return (M0 * M4) / (M2 * M2);
    }

    // === Period summaries & rates ===
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
    float getMeanPeriod_Tm0m1() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        if (M_neg1 <= EPSILON) return 0.0f;
        return M0 / M_neg1;
    }
    float getMeanPeriod_Tm1m1() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        if (M_neg1 <= EPSILON) return 0.0f;
        return M1 / M_neg1;
    }
    float getMeanGroupPeriod() const { return getMeanPeriod_Tz(); }

    // === Heights & steepness ===
    float getRMSDisplacement() const { return (M0 > EPSILON) ? std::sqrt(M0) : 0.0f; }
    float getSignificantWaveHeightRegular() const { return 2.0f * getRMSDisplacement(); }
    float getSignificantWaveHeightRayleigh() const { return 2.0f * std::sqrt(2.0f) * getRMSDisplacement(); }
    float getWaveSteepness() const {
        float Tz = getMeanPeriod_Tz();
        if (Tz <= EPSILON) return 0.0f;
        float L0 = 9.80665f * Tz * Tz / (2.0f * float(M_PI)); // deep-water wavelength
        return getSignificantWaveHeightRayleigh() / L0;
    }

    // === Probability ===
    float getExceedanceProbRayleigh(float crest_height) const {
        if (M0 <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0);
        return std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
    }

    // === Bandwidths ===
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
    float getWidthLonguetHiggins() const {
        if (M1 <= EPSILON) return 0.0f;
        float val = (M0 * M2) / (M1 * M1);
        return (val > 1.0f) ? std::sqrt(val - 1.0f) : 0.0f;
    }

// === Extremes & groupiness ===

// H1/10 crest height (linear Rayleigh model)
float getCrestHeight_H1over10() const {
    if (M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    // Rayleigh 90th percentile crest amplitude
    return sigma * std::sqrt(-2.0f * std::log(0.10f));
}

// H1/100 crest height
float getCrestHeight_H1over100() const {
    if (M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    return sigma * std::sqrt(-2.0f * std::log(0.01f));
}

// Nonlinear crest exceedance (Tayfun 1980 2nd-order approximation)
// P(Hc > h) ≈ exp(−h²/(2σ²)) * exp(Λ h³/(σ³)), where Λ ~ steepness factor
float getExceedanceProbTayfun(float crest_height) const {
    if (M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    float Hs = 2.0f * std::sqrt(2.0f) * sigma;
    float steep = getWaveSteepness();
    float Lambda = 0.5f * steep; // crude param
    float term1 = std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
    float term2 = std::exp(Lambda * std::pow(crest_height / sigma, 3));
    return term1 * term2;
}

// Groupiness factor = mean group period / mean zero-crossing period
float getGroupinessFactor() const {
    float Tg = getMeanGroupPeriod();
    float Tz = getMeanPeriod_Tz();
    return (Tz > EPSILON) ? (Tg / Tz) : 0.0f;
}

// Benjamin–Feir Index (deep water, linear est.)
// BFI = √2 * Hs / (Δf / fp)
float getBenjaminFeirIndex() const {
    float Hs = getSignificantWaveHeightRayleigh();
    float fp = getMeanFrequencyHz();
    float bw = getRBW();
    if (fp <= EPSILON || bw <= EPSILON) return 0.0f;
    float df = bw * fp;
    return (std::sqrt(2.0f) * Hs) / (df / fp);
}


// === Energy & wave power ===

// Energy flux period Te_flux = M_{-1} / M0 (s)
float getEnergyFluxPeriod() const {
    if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
    if (M0 <= EPSILON) return 0.0f;
    return M_neg1 / M0;
}

// Wave power per unit crest length (deep water, kW/m if SI units)
// P = (ρ g^2 / 64π) * Hs^2 * Te
float getWavePower(float rho = 1025.0f) const {
    float Hs = getSignificantWaveHeightRayleigh();
    float Te = getMeanPeriod_Te();
    if (Te <= EPSILON) return 0.0f;
    return (rho * 9.80665f * 9.80665f / (64.0f * float(M_PI))) * (Hs * Hs) * Te;
}

// === Bias-corrected metrics ===
//
// Apply first-order Jensen corrections for inverse ω powers:
//   E[1/ω^n] ≈ (1/ω̄^n)(1 + c σ²/ω̄²)
// with coefficients: n=1→c=1, n=2→c=3, n=3→c=6, n=4→c=10
// Correction factors: 1/(1 + c σ²/ω̄²)

bool isBiasCorrectionSignificant(float threshold = 0.01f) const {
    if (mu_w <= EPSILON) return false;
    return (std::max(var_slow, 0.0f) / (mu_w * mu_w)) > threshold;
}

float getMoment0_BiasCorrected() const {
    if (M0 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M0;
    float corr = 1.0f / (1.0f + 10.0f * var / (omega_bar * omega_bar));
    return M0 * corr;
}

float getMoment1_BiasCorrected() const {
    if (M1 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M1;
    float corr = 1.0f / (1.0f + 6.0f * var / (omega_bar * omega_bar));
    return M1 * corr;
}

float getMoment2_BiasCorrected() const {
    if (M2 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M2;
    float corr = 1.0f / (1.0f + 3.0f * var / (omega_bar * omega_bar));
    return M2 * corr;
}

float getMoment3_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("M3 not enabled");
    if (M3 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M3;
    float corr = 1.0f / (1.0f + 1.0f * var / (omega_bar * omega_bar));
    return M3 * corr;
}

float getMoment4_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("M4 not enabled");
    if (M4 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M4;
    float corr = 1.0f / (1.0f + 10.0f * var / (omega_bar * omega_bar));
    return M4 * corr;
}

float getMomentMinus1_BiasCorrected() const {
    if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
    if (M_neg1 <= EPSILON) return 0.0f;
    float omega_bar = mu_w;
    float var = std::max(var_slow, 0.0f);
    if (omega_bar <= EPSILON) return M_neg1;
    float corr = 1.0f / (1.0f + 6.0f * var / (omega_bar * omega_bar));
    return M_neg1 * corr;
}

// --- Derived bias-corrected metrics ---

float getRMSDisplacement_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    return (M0c > EPSILON) ? std::sqrt(M0c) : 0.0f;
}

float getSignificantWaveHeightRegular_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    return 2.0f * std::sqrt(M0c);
}

float getSignificantWaveHeightRayleigh_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    return 2.0f * std::sqrt(2.0f) * std::sqrt(M0c);
}

// Frequencies
float getMeanFrequencyRad_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    return (M0c > EPSILON) ? (M1c / M0c) : omega_min;
}

float getMeanFrequencyHz_BiasCorrected() const {
    return getMeanFrequencyRad_BiasCorrected() / (2.0f * float(M_PI));
}

// Periods
float getMeanPeriod_Tz_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    if (M2c <= EPSILON) return 0.0f;
    return std::sqrt(2.0f * float(M_PI) * float(M_PI) * (M0c / M2c));
}

float getMeanPeriod_TzUp_BiasCorrected() const { return getMeanPeriod_Tz_BiasCorrected(); }
float getMeanPeriod_TzDown_BiasCorrected() const { return getMeanPeriod_Tz_BiasCorrected(); }

float getMeanPeriod_Tm02_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    if (M2c <= EPSILON) return 0.0f;
    return 2.0f * float(M_PI) * std::sqrt(M0c / M2c);
}

float getMeanPeriod_T1_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M1c <= EPSILON) return 0.0f;
    return (2.0f * float(M_PI) * M0c) / M1c;
}

float getMeanPeriod_Te_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float Mneg1c = getMomentMinus1_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    return (2.0f * float(M_PI) * Mneg1c) / M0c;
}

float getMeanPeriod_Tm0m1_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float Mneg1c = getMomentMinus1_BiasCorrected();
    if (Mneg1c <= EPSILON) return 0.0f;
    return M0c / Mneg1c;
}

float getMeanPeriod_Tm1m1_BiasCorrected() const {
    float M1c = getMoment1_BiasCorrected();
    float Mneg1c = getMomentMinus1_BiasCorrected();
    if (Mneg1c <= EPSILON) return 0.0f;
    return M1c / Mneg1c;
}

// Rates & counts
float getUpcrossingRate_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    return (1.0f / (2.0f * float(M_PI))) * std::sqrt(M2c / M0c);
}

float getDowncrossingRate_BiasCorrected() const {
    return getUpcrossingRate_BiasCorrected();
}

float estimateWaveCount_BiasCorrected(float duration_s) const {
    if (duration_s <= 0.0f) return 0.0f;
    return getUpcrossingRate_BiasCorrected() * duration_s;
}

WaveCountEstimate estimateWaveCountWithCI_BiasCorrected(
        float duration_s, float confidence = 0.95f) const 
{
    WaveCountEstimate out{0.0f, 0.0f, 0.0f, confidence};
    if (duration_s <= 0.0f) return out;

    // Expected count from bias-corrected upcrossing rate
    float Nexp = estimateWaveCount_BiasCorrected(duration_s);
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

// Bandwidths
float getBandwidthCLH_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
    float ratio = (M1c * M1c) / (M0c * M2c);
    ratio = std::clamp(ratio, 0.0f, 1.0f);
    return std::sqrt(1.0f - ratio);
}

float getBandwidthGoda_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M0c <= EPSILON || M1c <= EPSILON) return 0.0f;
    float ratio = (M0c * M2c) / (M1c * M1c);
    return (ratio > 1.0f) ? std::sqrt(ratio - 1.0f) : 0.0f;
}

float getBandwidthKuik_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M1c <= EPSILON) return 0.0f;
    float val = (M0c * M2c) - (M1c * M1c);
    return (val > 0.0f) ? std::sqrt(val) / M1c : 0.0f;
}

float getWidthLonguetHiggins_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M1c <= EPSILON) return 0.0f;
    float val = (M0c * M2c) / (M1c * M1c);
    return (val > 1.0f) ? std::sqrt(val - 1.0f) : 0.0f;
}

// === Bias-corrected central moments & shape metrics ===
float getCentralMoment2_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Central moments not enabled");
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float mu = M1c / M0c;
    float m2 = M2c / M0c;
    return m2 - mu * mu;
}

float getCentralMoment3_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Central moments not enabled");
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M3c = getMoment3_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float mu = M1c / M0c;
    float m2 = M2c / M0c;
    float m3 = M3c / M0c;
    return m3 - 3.0f * mu * m2 + 2.0f * mu * mu * mu;
}

float getCentralMoment4_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Central moments not enabled");
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M3c = getMoment3_BiasCorrected();
    float M4c = getMoment4_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float mu = M1c / M0c;
    float m2 = M2c / M0c;
    float m3 = M3c / M0c;
    float m4 = M4c / M0c;
    return m4 - 4.0f * mu * m3 + 6.0f * mu * mu * m2 - 3.0f * mu * mu * mu * mu;
}

// === Bias-corrected shape diagnostics ===
float getSpectralSkewness_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
    float mu2c = getCentralMoment2_BiasCorrected();
    float mu3c = getCentralMoment3_BiasCorrected();
    if (mu2c <= EPSILON) return 0.0f;
    return mu3c / std::pow(mu2c, 1.5f);
}

float getSpectralKurtosis_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
    float mu2c = getCentralMoment2_BiasCorrected();
    float mu4c = getCentralMoment4_BiasCorrected();
    if (mu2c <= EPSILON) return 0.0f;
    return mu4c / (mu2c * mu2c);
}

float getSpectralExcessKurtosis_BiasCorrected() const {
    if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
    float mu2c = getCentralMoment2_BiasCorrected();
    float mu4c = getCentralMoment4_BiasCorrected();
    if (mu2c <= EPSILON) return 0.0f;
    return (mu4c / (mu2c * mu2c)) - 3.0f;
}

// === Bias-corrected extremes & groupiness ===

float getCrestHeight_H1over10_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    return sigma * std::sqrt(-2.0f * std::log(0.10f));
}

float getCrestHeight_H1over100_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    return sigma * std::sqrt(-2.0f * std::log(0.01f));
}

float getExceedanceProbTayfun_BiasCorrected(float crest_height) const {
    float M0c = getMoment0_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    float Hs = 2.0f * std::sqrt(2.0f) * sigma;
    float steep = getWaveSteepness_BiasCorrected();
    float Lambda = 0.5f * steep;
    float term1 = std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
    float term2 = std::exp(Lambda * std::pow(crest_height / sigma, 3));
    return term1 * term2;
}

float getGroupinessFactor_BiasCorrected() const {
    float Tg = getMeanPeriod_Tz_BiasCorrected(); // group period = Tz in deep water
    float Tz = getMeanPeriod_Tz_BiasCorrected();
    return (Tz > EPSILON) ? (Tg / Tz) : 0.0f;
}

float getBenjaminFeirIndex_BiasCorrected() const {
    float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
    float fp = getMeanFrequencyHz_BiasCorrected();
    float bw = getRBW(); // RBW itself not bias-corrected, phase-based
    if (fp <= EPSILON || bw <= EPSILON) return 0.0f;
    float df = bw * fp;
    return (std::sqrt(2.0f) * Hs) / (df / fp);
}

// === Bias-corrected energy & power ===
float getEnergyFluxPeriod_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float Mneg1c = getMomentMinus1_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    return Mneg1c / M0c;
}

float getWavePower_BiasCorrected(float rho = 1025.0f) const {
    float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
    float Te = getMeanPeriod_Te_BiasCorrected();
    if (Te <= EPSILON) return 0.0f;
    return (rho * 9.80665f * 9.80665f / (64.0f * float(M_PI))) * (Hs * Hs) * Te;
}

private:
    // Flags
    bool extended_metrics;
    bool negative_moments;

    // Time constants & alphas
    float tau_env = 0.0f, tau_mom = 0.0f, tau_coh = 0.0f, tau_omega = 0.0f;
    float omega_min = 0.0f;
    float alpha_env = 0.0f, alpha_mom = 0.0f, alpha_coh = 0.0f, alpha_omega = 0.0f;

    // Demod/envelope state
    float phi = 0.0f;
    float z_real = 0.0f, z_imag = 0.0f;

    // Moments & metrics
    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f, M3 = 0.0f, M4 = 0.0f, M_neg1 = 0.0f;
    float mu2 = 0.0f, mu3 = 0.0f, mu4 = 0.0f;
    float nu = 0.0f, R_spec = 0.0f, R_phase = 0.0f, rbw = 0.0f;

    // Coherence
    float coh_r = 0.0f, coh_i = 0.0f; bool has_coh = false;

    // Frequency tracking
    float omega_lp = 0.0f, omega_disp_lp = 0.0f;
    float omega_last = 0.0f, mu_w = 0.0f;
    float var_fast = 0.0f, var_slow = 0.0f;
    bool  has_omega_lp = false, has_omega_disp_lp = false, has_moments = false;

    // Demod ω limiter
    float omega_phi_last = 0.0f;

    // Two-pole RBW stats
    float wbar_ema = 0.0f, w2bar_ema = 0.0f;

    // Phase-increment RBW tracking
    float dphi_mean = 0.0f;     // running mean of Δφ/dt
    float dphi_var  = 0.0f;     // running variance of Δφ/dt

    // Variance adaptation for ω tracker
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

        // Instantaneous frequency estimate from phase increment
        float inst_freq = omega_phi; // since Δφ/dt = ω_phi

        // Update running mean/variance of inst_freq
        float delta_f = inst_freq - dphi_mean;
        dphi_mean += ALPHA_FAST * delta_f;
        dphi_var  = (1.0f - ALPHA_FAST) * dphi_var + ALPHA_FAST * delta_f * delta_f;

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
        // accel→disp envelope via 1/ω_lp²
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
        // Spectral regularity via RBW = √μ₂ / ω̄ using decoupled two-pole μ₂
        float omega_bar = getMeanFrequencyRad();
        float mu2_two_pole = std::max(0.0f, w2bar_ema - wbar_ema * wbar_ema - std::max(var_slow, 0.0f));

        rbw   = (omega_bar > 0.0f) ? (std::sqrt(mu2_two_pole) / std::max(omega_bar, omega_min)) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);

        // Narrowness ν (legacy)
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

    // Fast inverse error function approximation (C++17).
    // Winitzki initial approximation + two Halley refinements.
    static float erfinv_approx(float x) {
        x = std::clamp(x, -0.999999f, 0.999999f);
        const float a = 0.147f;
        float ln = std::log(1.0f - x * x);
        float tt1 = 2.0f / (float(M_PI) * a) + 0.5f * ln;
        float tt2 = 1.0f / a * ln;
        float sign = (x < 0.0f) ? -1.0f : 1.0f;
        float y = sign * std::sqrt(std::sqrt(tt1 * tt1 - tt2) - tt1);

        for (int i = 0; i < 2; ++i) {
            float ey = std::erf(y) - x;
            float dy = (2.0f / std::sqrt(float(M_PI))) * std::exp(-y * y);
            float d2y = -2.0f * y * dy;
            float denom = 2.0f * dy * dy - ey * d2y;
            if (std::fabs(denom) < 1e-12f) break;
            y -= (2.0f * ey * dy) / denom;
        }
        return y;
    }

    static float normalQuantile(float p) {
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
