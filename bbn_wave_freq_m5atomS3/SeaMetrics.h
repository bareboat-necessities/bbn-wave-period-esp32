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
    // ---- public constants & PODs ----
    constexpr static float EPSILON      = 1e-12f;
    constexpr static float BETA_SPEC    = 1.0f;
    constexpr static float OMEGA_DD_MAX = 30.0f;

    struct WaveCountEstimate {
        float expected;
        float ci_lower;
        float ci_upper;
        float confidence;
    };

    // ---- lifecycle ----
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
        // phasor / envelope
        phi = 0.0f; z_real = z_imag = 0.0f;

        // raw moments & shape cache
        M0 = M1 = M2 = 0.0f;
        if (extended_metrics) { M3 = M4 = 0.0f; }
        if (negative_moments) { M_neg1 = 0.0f; }
        mu2 = mu3 = mu4 = 0.0f;

        // regularity & bandwidth
        nu = 0.0f; R_spec = R_phase = rbw = 0.0f;

        // coherence
        coh_r = coh_i = 0.0f; has_coh = false;

        // frequency tracking
        omega_lp = omega_disp_lp = 0.0f; omega_last = mu_w = 0.0f;
        var_fast = var_slow = 0.0f;
        has_omega_lp = has_omega_disp_lp = has_moments = false;
        omega_phi_last = 0.0f;

        // two-pole stats for RBW
        wbar_ema = w2bar_ema = 0.0f;

        // phase-increment stats
        dphi_mean = dphi_var = 0.0f;
    }

    // ---- main update ----
    void update(float dt_s, float accel_z, float omega_inst) {
        if (!(dt_s > 0.0f)) return;
        if (accel_z    != accel_z)    accel_z    = 0.0f;      // NaN guard
        if (omega_inst != omega_inst) omega_inst = omega_min; // NaN guard
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
    float getMeanFrequencyRad() const {
        return (M0 > EPSILON) ? (M1 / M0) : omega_min;
    }
    float getMeanFrequencyHz()  const { return getMeanFrequencyRad() / (2.0f * float(M_PI)); }
    float getRBW() const { return rbw; }

    float getRBW_PhaseIncrement() const {
        const float omega_bar = (M0 > EPSILON) ? (M1 / M0) : omega_min;
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
        return std::sqrt(two_pi_sq() * (M0 / M2));
    }
    float getMeanPeriod_TzDown() const { return getMeanPeriod_Tz(); }
    float getMeanPeriod_TzUp()   const { return getMeanPeriod_Tz(); }

    float getUpcrossingRate() const {
        if (M0 <= EPSILON) return 0.0f;
        return inv_two_pi() * std::sqrt(M2 / M0);
    }
    float getDowncrossingRate() const { return getUpcrossingRate(); }

    float estimateWaveCount(float duration_s) const {
        if (duration_s <= 0.0f) return 0.0f;
        return getUpcrossingRate() * duration_s;
    }

    WaveCountEstimate estimateWaveCountWithCI(float duration_s, float confidence = 0.95f) const {
        WaveCountEstimate out{0.0f, 0.0f, 0.0f, confidence};
        if (duration_s <= 0.0f) return out;

        const float Nexp = estimateWaveCount(duration_s);
        const int   N    = (int)std::round(Nexp);
        const float alpha = std::clamp(1.0f - confidence, 0.0f, 1.0f);

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
        return two_pi() * (M0 / M1);
    }
    float getMeanPeriod_Tm02() const {
        if (M2 <= EPSILON) return 0.0f;
        return two_pi() * std::sqrt(M0 / M2);
    }
    float getMeanPeriod_Te() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        if (M0 <= EPSILON) return 0.0f;
        return two_pi() * (M_neg1 / M0);
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
    float getRMSDisplacement() const {
        return (M0 > EPSILON) ? std::sqrt(M0) : 0.0f;
    }
    float getSignificantWaveHeightRegular()  const { return 2.0f * getRMSDisplacement(); }
    float getSignificantWaveHeightRayleigh() const { return 2.0f * std::sqrt(2.0f) * getRMSDisplacement(); }
    float getWaveSteepness() const {
        const float Tz = getMeanPeriod_Tz();
        if (Tz <= EPSILON) return 0.0f;
        const float L0 = deep_water_wavelength(Tz);
        return (L0 > EPSILON) ? (getSignificantWaveHeightRayleigh() / L0) : 0.0f;
    }

    // === Probability ===
    float getExceedanceProbRayleigh(float crest_height) const {
        if (M0 <= EPSILON) return 0.0f;
        const float sigma = std::sqrt(M0);
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
        const float ratio = (M0 * M2) / (M1 * M1);
        return (ratio > 1.0f) ? std::sqrt(ratio - 1.0f) : 0.0f;
    }
    float getBandwidthKuik() const {
        if (M1 <= EPSILON) return 0.0f;
        const float val = (M0 * M2) - (M1 * M1);
        return (val > 0.0f) ? std::sqrt(val) / M1 : 0.0f;
    }
    float getWidthLonguetHiggins() const {
        if (M1 <= EPSILON) return 0.0f;
        const float val = (M0 * M2) / (M1 * M1);
        return (val > 1.0f) ? std::sqrt(val - 1.0f) : 0.0f;
    }

    // === Extremes & groupiness ===
    float getCrestHeight_H1over10() const {
        if (M0 <= EPSILON) return 0.0f;
        const float sigma = std::sqrt(M0);
        return sigma * std::sqrt(-2.0f * std::log(0.10f));
    }
    float getCrestHeight_H1over100() const {
        if (M0 <= EPSILON) return 0.0f;
        const float sigma = std::sqrt(M0);
        return sigma * std::sqrt(-2.0f * std::log(0.01f));
    }
    float getExceedanceProbTayfun(float crest_height) const {
        if (M0 <= EPSILON) return 0.0f;
        const float sigma  = std::sqrt(M0);
        const float steep  = getWaveSteepness();
        const float Lambda = 0.5f * steep; // coarse param
        const float term1  = std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
        const float term2  = std::exp(Lambda * std::pow(crest_height / sigma, 3));
        return term1 * term2;
    }
// === Extremes & groupiness ===

// Benjamin–Feir Index (modulational instability criterion)
// BFI = (√2 * Hs / L0) / (Δf/fp), with Δf ≈ RBW * fp
float getBenjaminFeirIndex() const {
    const float Hs = getSignificantWaveHeightRayleigh();
    const float Tz = getMeanPeriod_Tz();
    if (Hs <= EPSILON || Tz <= EPSILON) return 0.0f;

    // deep-water wavelength
    const float L0 = g() * Tz * Tz / (2.0f * float(M_PI));
    if (L0 <= EPSILON) return 0.0f;

    // peak frequency proxy from mean period
    const float fp = 1.0f / Tz;
    const float df = rbw * fp; // bandwidth in Hz

    if (df <= EPSILON) return 0.0f;
    return (std::sqrt(2.0f) * Hs / L0) / (df / fp);
}

// Groupiness factor (Longuet–Higgins): Tg/Tz
// Tg = 2π M0 / M2
float getGroupinessFactor() const {
    if (M0 <= EPSILON || M2 <= EPSILON) return 0.0f;
    const float Tg = (2.0f * float(M_PI)) * (M0 / M2);
    const float Tz = getMeanPeriod_Tz();
    return (Tz > EPSILON) ? (Tg / Tz) : 0.0f;
}

    // === Energy & wave power ===
    float getEnergyFluxPeriod() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        if (M0 <= EPSILON) return 0.0f;
        return M_neg1 / M0;
    }
    float getWavePower(float rho = 1025.0f) const {
        const float Hs = getSignificantWaveHeightRayleigh();
        const float Te = getMeanPeriod_Te();
        if (Te <= EPSILON) return 0.0f;
        return (rho * g() * g() / (64.0f * float(M_PI))) * (Hs * Hs) * Te;
    }

// ================== ADDITIONS (no-bias-corrected) ==================
//
// Paste these methods inside the `public:` section of SeaMetrics.
// They use existing members/fields only (M_PI, EPSILON, R_phase, rbw, etc.).

// --- 1) Spectral width parameters ---

// Cartwright–Longuet-Higgins style spectral bandwidth (≈ √μ2)
float getSpectralBandwidth() const {
    if (M0 <= EPSILON) return 0.0f;
    float mu2_est = (M2 / M0) - (M1 * M1) / (M0 * M0);
    return (mu2_est > 0.0f) ? std::sqrt(mu2_est) : 0.0f;
}

// Spectral narrowness parameter (M1^2 / (M0 M2))
float getSpectralNarrowness() const {
    if (M0 <= EPSILON || M2 <= EPSILON) return 0.0f;
    return (M1 * M1) / (M0 * M2);
}


// --- 2) Wave group statistics (thresholded runs; Rayleigh model) ---

// Mean run length (number of successive waves with crest > threshold_sigma·σ)
float getMeanRunLength(float threshold_sigma = 1.0f) const {
    if (M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    float p = getExceedanceProbRayleigh(threshold_sigma * sigma);
    return (p > EPSILON) ? (1.0f / p) : 0.0f;
}

// Mean group duration (s) for above-threshold waves
float getGroupDuration(float threshold_sigma = 1.0f) const {
    return getMeanRunLength(threshold_sigma) * getMeanPeriod_Tz();
}


// --- 3) Nonlinear wave parameters ---

// Ursell number (long-wave/shallow-water nonlinearity): Ur = H * L^2 / h^3
// Uses deep-water proxy for L from Tz (L0 = g Tz^2 / (2π)).
float getUrsellNumber(float depth) const {
    if (depth <= EPSILON) return 0.0f;
    float Tz = getMeanPeriod_Tz();
    if (Tz <= EPSILON) return 0.0f;
    float L0 = 9.80665f * Tz * Tz / (2.0f * float(M_PI));
    float Hs = getSignificantWaveHeightRayleigh();
    return (Hs * L0 * L0) / (depth * depth * depth);
}

// Simple deep-water nonlinearity parameter: (Hs * k) / 2
float getNonlinearityParameter() const {
    float Tz = getMeanPeriod_Tz();
    if (Tz <= EPSILON) return 0.0f;
    float L0 = 9.80665f * Tz * Tz / (2.0f * float(M_PI));
    if (L0 <= EPSILON) return 0.0f;
    float k  = 2.0f * float(M_PI) / L0;
    float Hs = getSignificantWaveHeightRayleigh();
    return 0.5f * Hs * k;
}


// --- 4) Wave age (wind-sea development) ---

// Wave age = c_p / U10, with c_p ≈ g Tz / (2π) as a proxy
float getWaveAge(float wind_speed) const {
    if (wind_speed <= EPSILON) return 0.0f;
    float Tz = getMeanPeriod_Tz();
    if (Tz <= EPSILON) return 0.0f;
    float cp = 9.80665f * Tz / (2.0f * float(M_PI)); // deep-water phase speed proxy
    return cp / wind_speed;
}


// --- 5) Spectral shape parameters ---

// JONSWAP peak enhancement factor (very rough proxy from Ochi peakedness)
float getPeakEnhancementFactor() const {
    if (!extended_metrics || M2 <= EPSILON) return 0.0f;
    float Qp = getPeakednessOchi(); // uses M0,M2,M4; throws if extended disabled
    // Clamp to a reasonable [1, 10] envelope and compress via sqrt
    float guess = 0.25f + 0.75f * std::sqrt(std::max(0.0f, Qp));
    if (guess < 1.0f)  guess = 1.0f;
    if (guess > 10.0f) guess = 10.0f;
    return guess;
}

// Goda-style peakedness (proxy via Ochi; requires M4 → extended)
float getPeakednessGoda() const {
    if (!extended_metrics) return 0.0f;
    // Many definitions exist; we map via Ochi as a simple proxy
    return 2.0f * (getPeakednessOchi() - 1.0f);
}


// --- 6) Wave breaking indicators ---

// Crude breaking probability (Battjes–Janssen-inspired proxy)
float getBreakingProbability(float depth = 0.0f) const {
    float Hs = getSignificantWaveHeightRayleigh();
    if (depth > EPSILON) {
        // Finite-depth: probability rises as Hs approaches ~0.78 h
        if (Hs <= EPSILON) return 0.0f;
        float ratio = 0.78f * depth / Hs;
        // Map ratio<1 → high probability; ratio≫1 → low probability
        float x = std::max(0.0f, std::min(5.0f, ratio));
        return std::exp(-0.5f * x * x); // heuristic decay
    } else {
        // Deep water: relate to overall steepness
        float steepness = getWaveSteepness();
        float p = 10.0f * steepness; // heuristic scaling
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        return p;
    }
}

// Maximum wave height before depth-limited breaking (Miche-type)
float getBreakingWaveHeight(float depth = 0.0f) const {
    if (depth > EPSILON) {
        return 0.78f * depth;
    }
    return 0.0f; // deep water: no simple hard cap without kinematics
}


// --- 7) Statistical extremes (linear/Rayleigh proxies) ---

// Most probable maximum wave height over duration (N waves)
float getMostProbableMaxHeight(float duration_s) const {
    float N = estimateWaveCount(duration_s);
    if (N <= EPSILON || M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    // For Rayleigh-like crest amplitudes: ~ σ √(2 ln N)
    return sigma * std::sqrt(2.0f * std::log(N));
}

// Expected maximum crest height (Gumbel-type refinement)
float getExpectedMaxCrestHeight(float duration_s) const {
    float N = estimateWaveCount(duration_s);
    if (N <= EPSILON || M0 <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0);
    float aN = std::sqrt(2.0f * std::log(N));
    if (aN <= EPSILON) return 0.0f;
    const float gamma = 0.5772f; // Euler–Mascheroni
    return sigma * (aN + gamma / aN);
}


// --- 8) Wave energy metrics ---

// Total wave energy density per unit surface area (J/m^2): E = ρ g m0
float getWaveEnergy(float rho = 1025.0f) const {
    return rho * 9.80665f * std::max(0.0f, M0);
}

// Spectral mean period Tm01 (= 2π M0 / M1); same as getMeanPeriod_T1()
float getSpectralMeanPeriod() const {
    if (M1 <= EPSILON) return 0.0f;
    return (2.0f * float(M_PI)) * (M0 / M1);
}


// --- 9) Simple quality-control metrics ---

// Aggregate data quality indicator (0–1): coherence × bandwidth penalty × bias flag
float getDataQuality() const {
    float quality = 1.0f;
    // Favor high phase coherence
    quality *= std::max(0.0f, std::min(1.0f, R_phase));
    // Penalize broad relative bandwidth
    quality *= std::exp(-0.5f * std::max(0.0f, rbw));
    // Penalize if bias corrections would likely be significant (uses internal var_slow/mu_w)
    if (isBiasCorrectionSignificant(0.05f)) {
        quality *= 0.8f;
    }
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 1.0f) quality = 1.0f;
    return quality;
}

// Crude SNR estimate (dB) using tracked slow variance as a noise proxy
float getSNR() const {
    if (M0 <= EPSILON) return 0.0f;
    float noise = std::max(var_slow, EPSILON);
    return 10.0f * std::log10(M0 / noise);
}

    // === Bias-corrected metrics ===
    // Significance flag
    bool isBiasCorrectionSignificant(float threshold = 0.01f) const {
        if (mu_w <= EPSILON) return false;
        return (std::max(var_slow, 0.0f) / (mu_w * mu_w)) > threshold;
    }

    // Bias-corrected raw moments
    float getMoment0_BiasCorrected() const  { return applyMomentCorrection(M0, 10.0f); } // ~1/ω^4
    float getMoment1_BiasCorrected() const  { return applyMomentCorrection(M1,  6.0f); } // ~1/ω^3
    float getMoment2_BiasCorrected() const  { return applyMomentCorrection(M2,  3.0f); } // ~1/ω^2
    float getMoment3_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("M3 not enabled");
        return applyMomentCorrection(M3, 1.0f); // ~1/ω^1
    }
    float getMoment4_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("M4 not enabled");
        return M4; // no correction (c=0)
    }
    float getMomentMinus1_BiasCorrected() const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        return applyMomentCorrection(M_neg1, 6.0f); // ~1/ω^3
    }

    // Derived bias-corrected metrics
    float getRMSDisplacement_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        return (M0c > EPSILON) ? std::sqrt(M0c) : 0.0f;
    }
    float getSignificantWaveHeightRegular_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        return 2.0f * std::sqrt(M0c);
    }
    float getSignificantWaveHeightRayleigh_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        return 2.0f * std::sqrt(2.0f) * std::sqrt(M0c);
    }
    float getWaveSteepness_BiasCorrected() const {
        const float Tz = getMeanPeriod_Tz_BiasCorrected();
        if (Tz <= EPSILON) return 0.0f;
        const float L0 = deep_water_wavelength(Tz);
        return (L0 > EPSILON) ? (getSignificantWaveHeightRayleigh_BiasCorrected() / L0) : 0.0f;
    }

    // Frequencies (bias-corrected)
    float getMeanFrequencyRad_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        return (M0c > EPSILON) ? (M1c / M0c) : omega_min;
    }
    float getMeanFrequencyHz_BiasCorrected() const {
        return getMeanFrequencyRad_BiasCorrected() / (2.0f * float(M_PI));
    }

    // Periods (bias-corrected)
    float getMeanPeriod_Tz_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        if (M2c <= EPSILON) return 0.0f;
        return std::sqrt(two_pi_sq() * (M0c / M2c));
    }
    float getMeanPeriod_TzUp_BiasCorrected() const   { return getMeanPeriod_Tz_BiasCorrected(); }
    float getMeanPeriod_TzDown_BiasCorrected() const { return getMeanPeriod_Tz_BiasCorrected(); }

    float getMeanPeriod_Tm02_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        if (M2c <= EPSILON) return 0.0f;
        return two_pi() * std::sqrt(M0c / M2c);
    }
    float getMeanPeriod_T1_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        if (M1c <= EPSILON) return 0.0f;
        return two_pi() * (M0c / M1c);
    }
    float getMeanPeriod_Te_BiasCorrected() const {
        const float M0c    = getMoment0_BiasCorrected();
        const float Mneg1c = getMomentMinus1_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        return two_pi() * (Mneg1c / M0c);
    }
    float getMeanPeriod_Tm0m1_BiasCorrected() const {
        const float M0c    = getMoment0_BiasCorrected();
        const float Mneg1c = getMomentMinus1_BiasCorrected();
        if (Mneg1c <= EPSILON) return 0.0f;
        return M0c / Mneg1c;
    }
    float getMeanPeriod_Tm1m1_BiasCorrected() const {
        const float M1c    = getMoment1_BiasCorrected();
        const float Mneg1c = getMomentMinus1_BiasCorrected();
        if (Mneg1c <= EPSILON) return 0.0f;
        return M1c / Mneg1c;
    }

    // Rates & counts (bias-corrected)
    float getUpcrossingRate_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        return inv_two_pi() * std::sqrt(M2c / M0c);
    }
    float getDowncrossingRate_BiasCorrected() const { return getUpcrossingRate_BiasCorrected(); }

    float estimateWaveCount_BiasCorrected(float duration_s) const {
        if (duration_s <= 0.0f) return 0.0f;
        return getUpcrossingRate_BiasCorrected() * duration_s;
    }

    WaveCountEstimate estimateWaveCountWithCI_BiasCorrected(float duration_s,
                                                            float confidence = 0.95f) const
    {
        WaveCountEstimate out{0.0f, 0.0f, 0.0f, confidence};
        if (duration_s <= 0.0f) return out;

        const float Nexp  = estimateWaveCount_BiasCorrected(duration_s);
        const int   N     = (int)std::round(Nexp);
        const float alpha = std::clamp(1.0f - confidence, 0.0f, 1.0f);

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

    // Bandwidths (bias-corrected)
    float getBandwidthCLH_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        float ratio = (M1c * M1c) / (M0c * M2c);
        ratio = std::clamp(ratio, 0.0f, 1.0f);
        return std::sqrt(1.0f - ratio);
    }
    float getBandwidthGoda_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        if (M0c <= EPSILON || M1c <= EPSILON) return 0.0f;
        const float ratio = (M0c * M2c) / (M1c * M1c);
        return (ratio > 1.0f) ? std::sqrt(ratio - 1.0f) : 0.0f;
    }
    float getBandwidthKuik_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        if (M1c <= EPSILON) return 0.0f;
        const float val = (M0c * M2c) - (M1c * M1c);
        return (val > 0.0f) ? std::sqrt(val) / M1c : 0.0f;
    }
    float getWidthLonguetHiggins_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        if (M1c <= EPSILON) return 0.0f;
        const float val = (M0c * M2c) / (M1c * M1c);
        return (val > 1.0f) ? std::sqrt(val - 1.0f) : 0.0f;
    }

    // Central moments & shape (bias-corrected)
    float getCentralMoment2_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        const float M0c = getMoment0_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float mu  = M1c / M0c;
        const float m2  = M2c / M0c;
        return m2 - mu * mu;
    }
    float getCentralMoment3_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        const float M0c = getMoment0_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M3c = getMoment3_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float mu  = M1c / M0c;
        const float m2  = M2c / M0c;
        const float m3  = M3c / M0c;
        return m3 - 3.0f * mu * m2 + 2.0f * mu * mu * mu;
    }
    float getCentralMoment4_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        const float M0c = getMoment0_BiasCorrected();
        const float M1c = getMoment1_BiasCorrected();
        const float M2c = getMoment2_BiasCorrected();
        const float M3c = getMoment3_BiasCorrected();
        const float M4c = getMoment4_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float mu  = M1c / M0c;
        const float m2  = M2c / M0c;
        const float m3  = M3c / M0c;
        const float m4  = M4c / M0c;
        return m4 - 4.0f * mu * m3 + 6.0f * mu * mu * m2 - 3.0f * mu * mu * mu * mu;
    }
    float getSpectralSkewness_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        const float mu2c = getCentralMoment2_BiasCorrected();
        const float mu3c = getCentralMoment3_BiasCorrected();
        if (mu2c <= EPSILON) return 0.0f;
        return mu3c / std::pow(mu2c, 1.5f);
    }
    float getSpectralKurtosis_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        const float mu2c = getCentralMoment2_BiasCorrected();
        const float mu4c = getCentralMoment4_BiasCorrected();
        if (mu2c <= EPSILON) return 0.0f;
        return mu4c / (mu2c * mu2c);
    }
    float getSpectralExcessKurtosis_BiasCorrected() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        const float mu2c = getCentralMoment2_BiasCorrected();
        const float mu4c = getCentralMoment4_BiasCorrected();
        if (mu2c <= EPSILON) return 0.0f;
        return (mu4c / (mu2c * mu2c)) - 3.0f;
    }

    // Extremes, groupiness, power (bias-corrected)
    float getCrestHeight_H1over10_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float sigma = std::sqrt(M0c);
        return sigma * std::sqrt(-2.0f * std::log(0.10f));
    }
    float getCrestHeight_H1over100_BiasCorrected() const {
        const float M0c = getMoment0_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float sigma = std::sqrt(M0c);
        return sigma * std::sqrt(-2.0f * std::log(0.01f));
    }
    float getExceedanceProbTayfun_BiasCorrected(float crest_height) const {
        const float M0c = getMoment0_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        const float sigma  = std::sqrt(M0c);
        const float steep  = getWaveSteepness_BiasCorrected();
        const float Lambda = 0.5f * steep;
        const float term1  = std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
        const float term2  = std::exp(Lambda * std::pow(crest_height / sigma, 3));
        return term1 * term2;
    }
    float getGroupinessFactor_BiasCorrected() const {
        const float Tg = getMeanPeriod_Tz_BiasCorrected(); // deep-water: Tg ~ Tz
        const float Tz = getMeanPeriod_Tz_BiasCorrected();
        return (Tz > EPSILON) ? (Tg / Tz) : 0.0f;
    }
    float getBenjaminFeirIndex_BiasCorrected() const {
        const float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
        const float fp = getMeanFrequencyHz_BiasCorrected();
        const float bw = getRBW(); // RBW is phase/two-pole based; unchanged
        if (fp <= EPSILON || bw <= EPSILON) return 0.0f;
        const float df = bw * fp;
        return (std::sqrt(2.0f) * Hs) / (df / fp);
    }
    float getEnergyFluxPeriod_BiasCorrected() const {
        const float M0c    = getMoment0_BiasCorrected();
        const float Mneg1c = getMomentMinus1_BiasCorrected();
        if (M0c <= EPSILON) return 0.0f;
        return Mneg1c / M0c;
    }
    float getWavePower_BiasCorrected(float rho = 1025.0f) const {
        const float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
        const float Te = getMeanPeriod_Te_BiasCorrected();
        if (Te <= EPSILON) return 0.0f;
        return (rho * g() * g() / (64.0f * float(M_PI))) * (Hs * Hs) * Te;
    }

// ================== ADDITIONS (bias-corrected versions) ==================
//
// Place inside the `public:` section of SeaMetrics.
// All methods reuse the bias-corrected moments already implemented.

// --- 1) Spectral width parameters ---

float getSpectralBandwidth_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float mu2_est = (M2c / M0c) - (M1c * M1c) / (M0c * M0c);
    return (mu2_est > 0.0f) ? std::sqrt(mu2_est) : 0.0f;
}

float getSpectralNarrowness_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M2c = getMoment2_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
    return (M1c * M1c) / (M0c * M2c);
}


// --- 2) Wave group statistics ---

float getMeanRunLength_BiasCorrected(float threshold_sigma = 1.0f) const {
    float M0c = getMoment0_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    float p = getExceedanceProbRayleigh(threshold_sigma * sigma);
    return (p > EPSILON) ? (1.0f / p) : 0.0f;
}

float getGroupDuration_BiasCorrected(float threshold_sigma = 1.0f) const {
    return getMeanRunLength_BiasCorrected(threshold_sigma) * getMeanPeriod_Tz_BiasCorrected();
}


// --- 3) Nonlinear wave parameters ---

float getUrsellNumber_BiasCorrected(float depth) const {
    if (depth <= EPSILON) return 0.0f;
    float Tz = getMeanPeriod_Tz_BiasCorrected();
    if (Tz <= EPSILON) return 0.0f;
    float L0 = 9.80665f * Tz * Tz / (2.0f * float(M_PI));
    float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
    return (Hs * L0 * L0) / (depth * depth * depth);
}

float getNonlinearityParameter_BiasCorrected() const {
    float Tz = getMeanPeriod_Tz_BiasCorrected();
    if (Tz <= EPSILON) return 0.0f;
    float L0 = 9.80665f * Tz * Tz / (2.0f * float(M_PI));
    if (L0 <= EPSILON) return 0.0f;
    float k  = 2.0f * float(M_PI) / L0;
    float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
    return 0.5f * Hs * k;
}


// --- 4) Wave age ---

float getWaveAge_BiasCorrected(float wind_speed) const {
    if (wind_speed <= EPSILON) return 0.0f;
    float Tz = getMeanPeriod_Tz_BiasCorrected();
    if (Tz <= EPSILON) return 0.0f;
    float cp = 9.80665f * Tz / (2.0f * float(M_PI));
    return cp / wind_speed;
}


// --- 5) Spectral shape parameters ---

float getPeakEnhancementFactor_BiasCorrected() const {
    if (!extended_metrics) return 0.0f;
    float Qp = getPeakednessOchi(); // Ochi already bias-robust by construction
    float guess = 0.25f + 0.75f * std::sqrt(std::max(0.0f, Qp));
    if (guess < 1.0f)  guess = 1.0f;
    if (guess > 10.0f) guess = 10.0f;
    return guess;
}

float getPeakednessGoda_BiasCorrected() const {
    if (!extended_metrics) return 0.0f;
    return 2.0f * (getPeakednessOchi() - 1.0f);
}


// --- 6) Wave breaking indicators ---

float getBreakingProbability_BiasCorrected(float depth = 0.0f) const {
    float Hs = getSignificantWaveHeightRayleigh_BiasCorrected();
    if (depth > EPSILON) {
        if (Hs <= EPSILON) return 0.0f;
        float ratio = 0.78f * depth / Hs;
        float x = std::max(0.0f, std::min(5.0f, ratio));
        return std::exp(-0.5f * x * x);
    } else {
        float steepness = getWaveSteepness_BiasCorrected();
        float p = 10.0f * steepness;
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        return p;
    }
}

float getBreakingWaveHeight_BiasCorrected(float depth = 0.0f) const {
    if (depth > EPSILON) {
        return 0.78f * depth;
    }
    return 0.0f;
}


// --- 7) Statistical extremes ---

float getMostProbableMaxHeight_BiasCorrected(float duration_s) const {
    float N = estimateWaveCount_BiasCorrected(duration_s);
    float M0c = getMoment0_BiasCorrected();
    if (N <= EPSILON || M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    return sigma * std::sqrt(2.0f * std::log(N));
}

float getExpectedMaxCrestHeight_BiasCorrected(float duration_s) const {
    float N = estimateWaveCount_BiasCorrected(duration_s);
    float M0c = getMoment0_BiasCorrected();
    if (N <= EPSILON || M0c <= EPSILON) return 0.0f;
    float sigma = std::sqrt(M0c);
    float aN = std::sqrt(2.0f * std::log(N));
    if (aN <= EPSILON) return 0.0f;
    const float gamma = 0.5772f;
    return sigma * (aN + gamma / aN);
}


// --- 8) Wave energy metrics ---

float getWaveEnergy_BiasCorrected(float rho = 1025.0f) const {
    float M0c = getMoment0_BiasCorrected();
    return rho * 9.80665f * std::max(0.0f, M0c);
}

float getSpectralMeanPeriod_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    float M1c = getMoment1_BiasCorrected();
    if (M1c <= EPSILON) return 0.0f;
    return (2.0f * float(M_PI)) * (M0c / M1c);
}


// --- 9) Quality-control metrics ---

float getDataQuality_BiasCorrected() const {
    float quality = 1.0f;
    quality *= std::max(0.0f, std::min(1.0f, R_phase));
    quality *= std::exp(-0.5f * std::max(0.0f, rbw));
    if (isBiasCorrectionSignificant(0.05f)) {
        quality *= 0.8f;
    }
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 1.0f) quality = 1.0f;
    return quality;
}

float getSNR_BiasCorrected() const {
    float M0c = getMoment0_BiasCorrected();
    if (M0c <= EPSILON) return 0.0f;
    float noise = std::max(var_slow, EPSILON);
    return 10.0f * std::log10(M0c / noise);
}

private:
    // ---- configuration flags ----
    bool extended_metrics;
    bool negative_moments;

    // ---- time constants & alphas ----
    float tau_env = 0.0f, tau_mom = 0.0f, tau_coh = 0.0f, tau_omega = 0.0f;
    float omega_min = 0.0f;
    float alpha_env = 0.0f, alpha_mom = 0.0f, alpha_coh = 0.0f, alpha_omega = 0.0f;

    // ---- demod/envelope state ----
    float phi = 0.0f;
    float z_real = 0.0f, z_imag = 0.0f;

    // ---- moments & metrics ----
    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f, M3 = 0.0f, M4 = 0.0f, M_neg1 = 0.0f;
    float mu2 = 0.0f, mu3 = 0.0f, mu4 = 0.0f;           // cached central moments
    float nu  = 0.0f, R_spec = 0.0f, R_phase = 0.0f, rbw = 0.0f;

    // ---- coherence (vector average of phase unit vector) ----
    float coh_r = 0.0f, coh_i = 0.0f; bool has_coh = false;

    // ---- frequency tracking ----
    float omega_lp = 0.0f, omega_disp_lp = 0.0f;
    float omega_last = 0.0f, mu_w = 0.0f;      // mean ω for bias correction
    float var_fast = 0.0f, var_slow = 0.0f;    // running variances for ω
    bool  has_omega_lp = false, has_omega_disp_lp = false, has_moments = false;

    // ---- demod ω limiter ----
    float omega_phi_last = 0.0f;

    // ---- two-pole RBW stats ----
    float wbar_ema = 0.0f, w2bar_ema = 0.0f;

    // ---- phase-increment RBW tracking ----
    float dphi_mean = 0.0f;
    float dphi_var  = 0.0f;

    // ---- ω variance adaptation ----
    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    // ---- small math helpers ----
    static constexpr float g()            { return 9.80665f; }
    static constexpr float two_pi()       { return 2.0f * float(M_PI); }
    static constexpr float inv_two_pi()   { return 1.0f / (2.0f * float(M_PI)); }
    static constexpr float two_pi_sq()    { return 2.0f * float(M_PI) * float(M_PI); }

    static float deep_water_wavelength(float T) {
        // L0 = g T^2 / (2π)
        return (T > 0.0f) ? (g() * T * T / two_pi()) : 0.0f;
    }

    // ---- alpha updates ----
    void updateAlpha(float dt_s) {
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
    }

    // ---- demodulation & ω tracking ----
    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        // Slew-limited demodulation basis ω (rad/s)
        const float w_target = std::max(omega_inst, omega_min);
        if (!has_omega_lp) omega_phi_last = w_target;

        const float dw          = w_target - omega_phi_last;
        const float dw_clamped  = std::clamp(dw, -OMEGA_DD_MAX * dt_s, OMEGA_DD_MAX * dt_s);
        const float omega_phi   = omega_phi_last + dw_clamped;
        omega_phi_last          = omega_phi;

        // integrate demodulation phase
        phi += omega_phi * dt_s;
        phi = std::fmod(phi, two_pi());

        // "instantaneous" frequency from phase increment
        const float inst_freq = omega_phi;

        // running mean/variance for Δφ/dt
        const float delta_f = inst_freq - dphi_mean;
        dphi_mean += ALPHA_FAST * delta_f;
        dphi_var  = (1.0f - ALPHA_FAST) * dphi_var + ALPHA_FAST * delta_f * delta_f;

        // rotate acceleration to baseband I/Q: y = a · e^(−jφ)
        const float c = std::cos(-phi), s = std::sin(-phi);
        const float y_real = accel_z * c;
        const float y_imag = accel_z * s;

        // first-time init
        if (!has_omega_lp) {
            z_real = y_real; z_imag = y_imag;
            omega_lp = w_target;
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }

        // envelope EMA
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;

        // LP the external ω estimate
        omega_lp = (tau_omega > 0.0f)
                 ? (1.0f - alpha_omega) * omega_lp + alpha_omega * w_target
                 : w_target;

        // Track slow variance of ω_lp (stabilizes RBW & bias correction)
        const float delta = omega_lp - mu_w;
        mu_w    += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    // ---- spectral moments update ----
    void updateSpectralMoments() {
        // accel→disp envelope via 1/ω_lp²
        const float omega_norm = std::max(omega_lp, omega_min);
        const float inv_w2 = 1.0f / std::max(omega_norm * omega_norm, EPSILON);
        const float disp_r = z_real * inv_w2;
        const float disp_i = z_imag * inv_w2;
        const float P_disp = disp_r * disp_r + disp_i * disp_i;

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
            const float a = alpha_mom, b = (1.0f - alpha_mom);
            M0 = b * M0 + a * P_disp;
            M1 = b * M1 + a * (P_disp * omega_norm);
            M2 = b * M2 + a * (P_disp * omega_norm * omega_norm);
            if (extended_metrics) {
                M3 = b * M3 + a * (P_disp * omega_norm * omega_norm * omega_norm);
                M4 = b * M4 + a * (P_disp * omega_norm * omega_norm * omega_norm * omega_norm);
            }
            if (negative_moments) {
                M_neg1 = b * M_neg1 + a * (P_disp / omega_norm);
            }
        }

        // RBW stats (decoupled two-pole on ω)
        wbar_ema  = (1.0f - alpha_mom) * wbar_ema  + alpha_mom * omega_norm;
        w2bar_ema = (1.0f - alpha_mom) * w2bar_ema + alpha_mom * omega_norm * omega_norm;

        // moment-based mean ω for readouts (LP)
        const float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (!has_omega_disp_lp) {
            omega_disp_lp = omega_disp;
            has_omega_disp_lp = true;
        } else {
            omega_disp_lp = (1.0f - alpha_omega) * omega_disp_lp + alpha_omega * omega_disp;
        }
    }

    // ---- coherence ----
    void updatePhaseCoherence() {
        const float mag = std::sqrt(z_real * z_real + z_imag * z_imag);
        if (mag <= 1e-6f) {
            if (!has_coh) { coh_r = 1.0f; coh_i = 0.0f; has_coh = true; }
            R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
            return;
        }

        const float u_r = z_real / mag;
        const float u_i = z_imag / mag;

        if (!has_coh) { coh_r = u_r; coh_i = u_i; has_coh = true; }
        else {
            coh_r = (1.0f - alpha_coh) * coh_r + alpha_coh * u_r;
            coh_i = (1.0f - alpha_coh) * coh_i + alpha_coh * u_i;
        }
        R_phase = std::clamp(std::sqrt(coh_r * coh_r + coh_i * coh_i), 0.0f, 1.0f);
    }

    // ---- regularity & cached central moments ----
    void computeRegularityMetrics() {
        // Spectral regularity via RBW = √μ₂ / ω̄ using decoupled two-pole μ₂
        const float omega_bar = getMeanFrequencyRad();
        const float mu2_two_pole = std::max(0.0f, w2bar_ema - wbar_ema * wbar_ema - std::max(var_slow, 0.0f));

        rbw   = (omega_bar > 0.0f) ? (std::sqrt(mu2_two_pole) / std::max(omega_bar, omega_min)) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);

        // Narrowness ν (legacy)
        if (M1 > EPSILON && M0 > 0.0f && M2 > 0.0f) {
            const float ratio = (M0 * M2) / (M1 * M1) - 1.0f;
            nu = (ratio > 0.0f) ? std::sqrt(ratio) : 0.0f;
        } else {
            nu = 0.0f;
        }

        // Extended: cache central moments from raw moments
        if (extended_metrics && M0 > EPSILON) {
            const float mu = M1 / M0;
            const float m2 = M2 / M0;
            const float m3 = M3 / M0;
            const float m4 = M4 / M0;

            mu2 = m2 - mu * mu;
            mu3 = m3 - 3.0f * mu * m2 + 2.0f * mu * mu * mu;
            mu4 = m4 - 4.0f * mu * m3 + 6.0f * mu * mu * m2 - 3.0f * mu * mu * mu * mu;
        } else {
            mu2 = mu3 = mu4 = 0.0f;
        }
    }

    // ---- bias-correction helper ----
    float applyMomentCorrection(float Mraw, float c_coeff) const {
        if (Mraw <= EPSILON) return 0.0f;
        const float omega_bar = mu_w;
        const float var = std::max(var_slow, 0.0f);
        if (omega_bar <= EPSILON || c_coeff <= 0.0f) return Mraw;
        const float corr = 1.0f / (1.0f + c_coeff * var / (omega_bar * omega_bar));
        return Mraw * corr;
    }

    // ---- math helpers (quantiles) ----

    // Fast inverse error function approximation
    static float erfinv_approx(float x) {
        x = std::clamp(x, -0.999999f, 0.999999f);
        const float a = 0.147f;
        const float ln = std::log(1.0f - x * x);
        const float tt1 = 2.0f / (float(M_PI) * a) + 0.5f * ln;
        const float tt2 = 1.0f / a * ln;
        const float sign = (x < 0.0f) ? -1.0f : 1.0f;
        float y = sign * std::sqrt(std::sqrt(tt1 * tt1 - tt2) - tt1);

        // two Halley refinements
        for (int i = 0; i < 2; ++i) {
            const float ey  = std::erf(y) - x;
            const float dy  = (2.0f / std::sqrt(float(M_PI))) * std::exp(-y * y);
            const float d2y = -2.0f * y * dy;
            const float denom = 2.0f * dy * dy - ey * d2y;
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
        const float z = normalQuantile(p);
        const float v = (float)k;
        const float h = 2.0f / (9.0f * v);
        return v * std::pow(1.0f - h + z * std::sqrt(h), 3.0f);
    }
};
