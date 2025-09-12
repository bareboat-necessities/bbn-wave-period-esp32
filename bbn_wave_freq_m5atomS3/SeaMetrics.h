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
 * Raw spectral moments:
 *   • M0      = ⟨P_disp⟩
 *   • M1      = ⟨P_disp·ω⟩
 *   • M2      = ⟨P_disp·ω²⟩
 *   • M3      = ⟨P_disp·ω³⟩   [extended only]
 *   • M4      = ⟨P_disp·ω⁴⟩   [extended only]
 *   • M_{−1}  = ⟨P_disp·ω^{−1}⟩   [negative moments only]
 *
 * Central moments (extended only):
 *   • μ₂ = M2/M0 − (M1/M0)²
 *   • μ₃ = M3/M0 − 3·μ·(M2/M0) + 2·μ³
 *   • μ₄ = M4/M0 − 4·μ·(M3/M0) + 6·μ²·(M2/M0) − 3·μ⁴
 *     where μ = M1/M0
 *
 * Frequency metrics:
 *   • Mean frequency (rad/s): ω̄ = M1/M0
 *   • Mean frequency (Hz):    f̄ = ω̄ / (2π)
 *   • Relative bandwidth (RBW): √μ₂ / ω̄   (μ₂ from decoupled two-pole stats)
 *   • Phase-increment RBW: variance of Δφ/dt normalized by ω̄
 *
 * Regularity metrics:
 *   • R_spec  = exp(−β·RBW)
 *   • R_phase = ‖⟨ z/‖z‖ ⟩‖
 *   • Narrowness ν = √((M0 M2 / M1²) − 1)
 *
 * Shape diagnostics (extended only):
 *   • Spectral skewness        = μ₃ / μ₂^(3/2)
 *   • Spectral kurtosis        = μ₄ / μ₂²
 *   • Spectral excess kurtosis = (μ₄ / μ₂²) − 3
 *   • Ochi peakedness Q        = (M0 M4) / M2²
 *   • Benassai parameter B     = (M0 M4) / M2²
 *
 * Period summaries (s):
 *   • T_z, T_zup, T_zdown
 *   • ν_up, ν_down
 *   • Wave count + Garwood CI
 *   • T_1, T_m02, T_e
 *   • T_{m0,−1}, T_{m1,−1}
 *   • Mean group period
 *
 * Heights & steepness:
 *   • RMS displacement
 *   • Hs (regular, Rayleigh)
 *   • Wave steepness (Hs/L0, deep water)
 *
 * Probability metrics:
 *   • Crest exceedance (Rayleigh): P(Hc > h)
 *   • Crest exceedance (Tayfun):  nonlinear approximation
 *
 * Bandwidths:
 *   • CLH, Goda, Kuik
 *   • Longuet–Higgins width
 *
 * Extremes & groupiness:
 *   • H1/10 crest height
 *   • H1/100 crest height
 *   • Tayfun exceedance probability
 *   • Groupiness factor G = Tg / Tz   (Longuet–Higgins, 1957)
 *   • Benjamin–Feir index (BFI) = (√2 Hs / L0) / (Δf/fp)
 *
 * Energy & wave power:
 *   • Energy flux period (Te_flux)
 *   • Wave power per crest length (kW/m)
 *
 * Nonlinear & development:
 *   • Ursell number (Ur = H L² / h³)
 *   • Nonlinearity parameter (Hs·k/2)
 *   • Wave age (c_p / U10)
 *
 * Breaking indicators:
 *   • Probability of breaking (depth- or steepness-based proxy)
 *   • Depth-limited breaking height (≈ 0.78 h)
 *
 * Comfort & motion sickness:
 *   • Motion Sickness Dose Value (MSDV)
 *   • Seasickness incidence (%) vs. exposure
 *   • Comfort level rating (0–100)
 *   • Vertical motion intensity
 *   • Time to onset of sickness
 *
 * Bias-correction API:
 *   All getters accept a boolean argument `bias_corrected` (default = true).
 *   When true, raw spectral moments (M0, M1, M2, M3, M−1) are adjusted by
 *   first-order Jensen corrections for ω-tracker jitter:
 *
 *     E[1/ω^n] ≈ (1/ω̄^n)(1 + c_n σ²/ω̄²), with coefficients:
 *       n = 1 → c = 1
 *       n = 2 → c = 3
 *       n = 3 → c = 6
 *
 *   — Correction is applied only up to M3. —
 *   Higher-order moments (e.g. M4) are left uncorrected, because literature
 *   shows they are dominated by high-frequency noise and rarely used in
 *   operational metrics. (See Holthuijsen 2007; Kuik 1988; Ochi 1976.)
 *
 *   Derived metrics (periods, heights, bandwidths, skew/kurtosis,
 *   extremes, power) are computed consistently from these corrected
 *   moments. Phase-based metrics (R_phase, RBW_PhaseIncrement) are
 *   unaffected.
 *
 * Notes:
 *   – Bias correction reduces upward bias of energy, bandwidth, and
 *     moment-based metrics when frequency estimates jitter or acceleration
 *     carries DC.
 *   – Raw (uncorrected) values can be obtained by passing
 *     `bias_corrected = false`.
 *
 * Limitations & assumptions:
 *   – Deep-water dispersion relation is assumed (ω² = gk); finite-depth
 *     effects are not included unless depth is explicitly provided in
 *     formulas (e.g. Ursell number, breaking probability).
 *   – All metrics are scalar (1-D) and nondirectional; no spreading or
 *     directional spectra are modeled.
 *   – Acceleration is assumed to come from a vertical axis sensor with
 *     Z-up convention; biases and sensor errors must be pre-filtered.
 *   – Bias correction applies only to frequency-tracker jitter (first-order
 *     Jensen expansion). It does not address amplitude noise, sensor bias,
 *     or nonstationarity.
 *   – Comfort and seasickness models use empirical constants (ISO-based
 *     approximations) and should be treated as heuristic indicators rather
 *     than clinical predictions.
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
        if (extended_metrics) { M3 = M4 = 0.0f; }
        if (negative_moments) { M_neg1 = 0.0f; }
        mu2 = mu3 = mu4 = 0.0f;
        nu = R_spec = R_phase = rbw = 0.0f;
        coh_r = coh_i = 0.0f; has_coh = false;
        omega_lp = omega_disp_lp = omega_last = mu_w = 0.0f;
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

    // Spectral moments
    float getMomentMinus1(bool bias_corrected = true) const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        return bias_corrected ? applyMomentCorrection(M_neg1, 6.0f) : M_neg1;
    }
    float getMoment0(bool bias_corrected = true) const {
        return bias_corrected ? applyMomentCorrection(M0, 10.0f) : M0;
    }
    float getMoment1(bool bias_corrected = true) const {
        return bias_corrected ? applyMomentCorrection(M1, 6.0f) : M1;
    }
    float getMoment2(bool bias_corrected = true) const {
        return bias_corrected ? applyMomentCorrection(M2, 3.0f) : M2;
    }
    float getMoment3(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("M3 not enabled");
        return bias_corrected ? applyMomentCorrection(M3, 1.0f) : M3;
    }
    float getMoment4(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("M4 not enabled");
        return M4;
    }

    // Central moments & shape (extended only)
    float getCentralMoment2(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float mu  = M1c / M0c;
        float m2  = M2c / M0c;
        return m2 - mu * mu;
    }
    float getCentralMoment3(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        float M3c = getMoment3(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float mu  = M1c / M0c;
        float m2  = M2c / M0c;
        float m3  = M3c / M0c;
        return m3 - 3.0f * mu * m2 + 2.0f * mu * mu * mu;
    }
    float getCentralMoment4(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("Central moments not enabled");
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        float M3c = getMoment3(bias_corrected);
        float M4c = getMoment4(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float mu  = M1c / M0c;
        float m2  = M2c / M0c;
        float m3  = M3c / M0c;
        float m4  = M4c / M0c;
        return m4 - 4.0f * mu * m3 + 6.0f * mu * mu * m2 - 3.0f * mu * mu * mu * mu;
    }
    float getSpectralSkewness(bool bias_corrected = true) const {
        float mu2c = getCentralMoment2(bias_corrected);
        float mu3c = getCentralMoment3(bias_corrected);
        if (mu2c <= EPSILON) return 0.0f;
        return mu3c / std::pow(mu2c, 1.5f);
    }
    float getSpectralKurtosis(bool bias_corrected = true) const {
        float mu2c = getCentralMoment2(bias_corrected);
        float mu4c = getCentralMoment4(bias_corrected);
        if (mu2c <= EPSILON) return 0.0f;
        return mu4c / (mu2c * mu2c);
    }
    float getSpectralExcessKurtosis(bool bias_corrected = true) const {
        float mu2c = getCentralMoment2(bias_corrected);
        float mu4c = getCentralMoment4(bias_corrected);
        if (mu2c <= EPSILON) return 0.0f;
        return (mu4c / (mu2c * mu2c)) - 3.0f;
    }
    float getPeakednessOchi() const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        if (M2 <= EPSILON) return 0.0f;
        return (M0 * M4) / (M2 * M2);
    }
    float getBenassaiParameter() const { return getPeakednessOchi(); }
    float getPeakEnhancementFactor() const {
        if (!extended_metrics) return 0.0f;
        float Qp = getPeakednessOchi();
        float guess = 0.25f + 0.75f * std::sqrt(std::max(0.0f, Qp));
        return std::clamp(guess, 1.0f, 10.0f);
    }
    float getPeakednessGoda() const {
        if (!extended_metrics) return 0.0f;
        return 2.0f * (getPeakednessOchi() - 1.0f);
    }

    // Frequency metrics
    float getMeanFrequencyRad(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        return (M0c > EPSILON) ? (M1c / M0c) : omega_min;
    }
    float getMeanFrequencyHz(bool bias_corrected = true) const {
        return getMeanFrequencyRad(bias_corrected) / (2.0f * float(M_PI));
    }
    float getRBW() const { return rbw; }
    float getRBW_PhaseIncrement() const {
        float omega_bar = getMeanFrequencyRad(false);
        if (omega_bar <= EPSILON) return 0.0f;
        return std::sqrt(std::max(0.0f, dphi_var)) / omega_bar;
    }

    // Regularity
    float getRegularitySpectral() const { return R_spec; }
    float getRegularityPhase() const { return R_phase; }
    float getNarrowness(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M1c <= EPSILON || M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        float ratio = (M0c * M2c) / (M1c * M1c) - 1.0f;
        return (ratio > 0.0f) ? std::sqrt(ratio) : 0.0f;
    }

    // Periods & rates
    float getMeanPeriod_Tz(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M2c <= EPSILON) return 0.0f;
        return std::sqrt(two_pi_sq() * (M0c / M2c));
    }
    float getMeanPeriod_T1(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        if (M1c <= EPSILON) return 0.0f;
        return two_pi() * (M0c / M1c);
    }
    float getMeanPeriod_Tm02(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M2c <= EPSILON) return 0.0f;
        return two_pi() * std::sqrt(M0c / M2c);
    }
    float getMeanPeriod_Te(bool bias_corrected = true) const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        float M0c = getMoment0(bias_corrected);
        float Mneg1c = getMomentMinus1(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        return two_pi() * (Mneg1c / M0c);
    }
    float getMeanPeriod_Tm0m1(bool bias_corrected = true) const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        float M0c = getMoment0(bias_corrected);
        float Mneg1c = getMomentMinus1(bias_corrected);
        if (Mneg1c <= EPSILON) return 0.0f;
        return M0c / Mneg1c;
    }
    float getMeanPeriod_Tm1m1(bool bias_corrected = true) const {
        if (!negative_moments) throw std::logic_error("M_{-1} not enabled");
        float M1c = getMoment1(bias_corrected);
        float Mneg1c = getMomentMinus1(bias_corrected);
        if (Mneg1c <= EPSILON) return 0.0f;
        return M1c / Mneg1c;
    }
    float getMeanGroupPeriod(bool bias_corrected = true) const { return getMeanPeriod_Tz(bias_corrected); }
    float getUpcrossingRate(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        return inv_two_pi() * std::sqrt(M2c / M0c);
    }
    float getDowncrossingRate(bool bias_corrected = true) const { return getUpcrossingRate(bias_corrected); }
    float estimateWaveCount(float duration_s, bool bias_corrected = true) const {
        if (duration_s <= 0.0f) return 0.0f;
        return getUpcrossingRate(bias_corrected) * duration_s;
    }
    WaveCountEstimate estimateWaveCountWithCI(float duration_s, float confidence = 0.95f,
                                              bool bias_corrected = true) const {
        WaveCountEstimate out{0.0f, 0.0f, 0.0f, confidence};
        if (duration_s <= 0.0f) return out;
        float Nexp = estimateWaveCount(duration_s, bias_corrected);
        int N = (int)std::round(Nexp);
        float alpha = std::clamp(1.0f - confidence, 0.0f, 1.0f);
        float lower, upper;
        if (N == 0) {
            lower = 0.0f;
            upper = 0.5f * chi2Quantile(1.0f - alpha / 2.0f, 2 * (N + 1));
        } else {
            lower = 0.5f * chi2Quantile(alpha / 2.0f, 2 * N);
            upper = 0.5f * chi2Quantile(1.0f - alpha / 2.0f, 2 * (N + 1));
        }
        out.expected = Nexp;
        out.ci_lower = lower;
        out.ci_upper = upper;
        return out;
    }

    // Heights & steepness
    float getRMSDisplacement(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        return (M0c > EPSILON) ? std::sqrt(M0c) : 0.0f;
    }
    float getSignificantWaveHeightRegular(bool bias_corrected = true) const {
        return 2.0f * getRMSDisplacement(bias_corrected);
    }
    float getSignificantWaveHeightRayleigh(bool bias_corrected = true) const {
        return 2.0f * std::sqrt(2.0f) * getRMSDisplacement(bias_corrected);
    }
    float getWaveSteepness(bool bias_corrected = true) const {
        float Tz = getMeanPeriod_Tz(bias_corrected);
        if (Tz <= EPSILON) return 0.0f;
        float L0 = deep_water_wavelength(Tz);
        return (L0 > EPSILON) ? (getSignificantWaveHeightRayleigh(bias_corrected) / L0) : 0.0f;
    }

    // Probabilities & extremes
    float getExceedanceProbRayleigh(float crest_height, bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        return std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
    }
    float getExceedanceProbTayfun(float crest_height, bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        float steep = getWaveSteepness(bias_corrected);
        float Lambda = 0.5f * steep;
        float term1 = std::exp(-(crest_height * crest_height) / (2.0f * sigma * sigma));
        float term2 = std::exp(Lambda * std::pow(crest_height / sigma, 3));
        return term1 * term2;
    }
    float getCrestHeight_H1over10(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        return sigma * std::sqrt(-2.0f * std::log(0.10f));
    }
    float getCrestHeight_H1over100(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        return sigma * std::sqrt(-2.0f * std::log(0.01f));
    }
    float getMostProbableMaxHeight(float duration_s, bool bias_corrected = true) const {
        float N = estimateWaveCount(duration_s, bias_corrected);
        float M0c = getMoment0(bias_corrected);
        if (N <= EPSILON || M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        return sigma * std::sqrt(2.0f * std::log(N));
    }
    float getExpectedMaxCrestHeight(float duration_s, bool bias_corrected = true) const {
        float N = estimateWaveCount(duration_s, bias_corrected);
        float M0c = getMoment0(bias_corrected);
        if (N <= EPSILON || M0c <= EPSILON) return 0.0f;
        float sigma = std::sqrt(M0c);
        float aN = std::sqrt(2.0f * std::log(N));
        if (aN <= EPSILON) return 0.0f;
        const float gamma = 0.5772f;
        return sigma * (aN + gamma / aN);
    }

    // Bandwidths
    float getBandwidthCLH(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        float ratio = (M1c * M1c) / (M0c * M2c);
        return std::sqrt(std::max(0.0f, 1.0f - ratio));
    }
    float getBandwidthGoda(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON || M1c <= EPSILON) return 0.0f;
        float ratio = (M0c * M2c) / (M1c * M1c);
        return (ratio > 1.0f) ? std::sqrt(ratio - 1.0f) : 0.0f;
    }
    float getBandwidthKuik(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M1c <= EPSILON) return 0.0f;
        float val = (M0c * M2c) - (M1c * M1c);
        return (val > 0.0f) ? std::sqrt(val) / M1c : 0.0f;
    }
    float getWidthLonguetHiggins(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M1c <= EPSILON) return 0.0f;
        float val = (M0c * M2c) / (M1c * M1c);
        return (val > 1.0f) ? std::sqrt(val - 1.0f) : 0.0f;
    }
    float getSpectralBandwidth(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float mu2_est = (M2c / M0c) - (M1c * M1c) / (M0c * M0c);
        return (mu2_est > 0.0f) ? std::sqrt(mu2_est) : 0.0f;
    }
    float getSpectralNarrowness(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        return (M1c * M1c) / (M0c * M2c);
    }

    // Groupiness & instability
    float getGroupinessFactor(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        float Tg = two_pi() * (M0c / M2c);
        float Tz = getMeanPeriod_Tz(bias_corrected);
        return (Tz > EPSILON) ? (Tg / Tz) : 0.0f;
    }
    float getBenjaminFeirIndex(bool bias_corrected = true) const {
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        float Tz = getMeanPeriod_Tz(bias_corrected);
        if (Hs <= EPSILON || Tz <= EPSILON) return 0.0f;
        float L0 = g() * Tz * Tz / (2.0f * float(M_PI));
        if (L0 <= EPSILON) return 0.0f;
        float fp = 1.0f / Tz;
        float df = rbw * fp;
        if (df <= EPSILON) return 0.0f;
        return (std::sqrt(2.0f) * Hs / L0) / (df / fp);
    }

    // Energy & power
    float getEnergyFluxPeriod(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float Mneg1c = getMomentMinus1(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        return Mneg1c / M0c;
    }
    float getWavePower(bool bias_corrected = true, float rho = 1025.0f) const {
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        float Te = getMeanPeriod_Te(bias_corrected);
        if (Te <= EPSILON) return 0.0f;
        return (rho * g() * g() / (64.0f * float(M_PI))) * (Hs * Hs) * Te;
    }
    float getWaveEnergy(bool bias_corrected = true, float rho = 1025.0f) const {
        float M0c = getMoment0(bias_corrected);
        return rho * g() * std::max(0.0f, M0c);
    }
    float getSpectralMeanPeriod(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        if (M1c <= EPSILON) return 0.0f;
        return two_pi() * (M0c / M1c);
    }

    // Nonlinear & development
    float getUrsellNumber(float depth, bool bias_corrected = true) const {
        if (depth <= EPSILON) return 0.0f;
        float Tz = getMeanPeriod_Tz(bias_corrected);
        if (Tz <= EPSILON) return 0.0f;
        float L0 = g() * Tz * Tz / (2.0f * float(M_PI));
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        return (Hs * L0 * L0) / (depth * depth * depth);
    }
    float getNonlinearityParameter(bool bias_corrected = true) const {
        float Tz = getMeanPeriod_Tz(bias_corrected);
        if (Tz <= EPSILON) return 0.0f;
        float L0 = g() * Tz * Tz / (2.0f * float(M_PI));
        if (L0 <= EPSILON) return 0.0f;
        float k = two_pi() / L0;
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        return 0.5f * Hs * k;
    }
    float getWaveAge(float wind_speed, bool bias_corrected = true) const {
        if (wind_speed <= EPSILON) return 0.0f;
        float Tz = getMeanPeriod_Tz(bias_corrected);
        if (Tz <= EPSILON) return 0.0f;
        float cp = g() * Tz / (2.0f * float(M_PI));
        return cp / wind_speed;
    }

    // Breaking indicators
    float getBreakingProbability(float depth = 0.0f, bool bias_corrected = true) const {
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        if (depth > EPSILON) {
            if (Hs <= EPSILON) return 0.0f;
            float ratio = 0.78f * depth / Hs;
            float x = std::clamp(ratio, 0.0f, 5.0f);
            return std::exp(-0.5f * x * x);
        } else {
            float steepness = getWaveSteepness(bias_corrected);
            float p = 10.0f * steepness;
            return std::clamp(p, 0.0f, 1.0f);
        }
    }
    float getBreakingWaveHeight(float depth = 0.0f) const {
        if (depth > EPSILON) return 0.78f * depth;
        return 0.0f;
    }

    // Quality control
    float getDataQuality(bool bias_corrected = true) const {
        float quality = 1.0f;
        quality *= std::clamp(R_phase, 0.0f, 1.0f);
        quality *= std::exp(-0.5f * std::max(0.0f, rbw));
        if (isBiasCorrectionSignificant(0.05f)) quality *= 0.8f;
        return std::clamp(quality, 0.0f, 1.0f);
    }
    float getSNR(bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float noise = std::max(var_slow, EPSILON);
        return 10.0f * std::log10(M0c / noise);
    }

    // Comfort & seasickness
    float getSeasicknessIncidence(float exposure_hours, float susceptibility = 0.5f,
                                  bool bias_corrected = true) const {
        if (exposure_hours <= 0.0f) return 0.0f;
        float M2c = getMoment2(bias_corrected);
        float M0c = getMoment0(bias_corrected);
        float M1c = getMoment1(bias_corrected);
        if (M0c <= EPSILON || M2c <= EPSILON) return 0.0f;
        float accel_rms = std::sqrt(M2c);
        float f_dom = (M0c > EPSILON) ? (M1c / M0c) / two_pi() : 0.0f;
        if (f_dom <= EPSILON) return 0.0f;
        const float a = 0.5f, b = 2.0f, c = 0.4f, k = 0.7f;
        float msdv = accel_rms * std::pow(f_dom, b / a) * std::pow(exposure_hours, c);
        float incidence = 100.0f * (1.0f - std::exp(-k * msdv * susceptibility));
        return std::clamp(incidence, 0.0f, 100.0f);
    }
    float getMotionSicknessDoseValue(float exposure_hours, bool bias_corrected = true) const {
        if (exposure_hours <= 0.0f) return 0.0f;
        float M2c = getMoment2(bias_corrected);
        if (M2c <= EPSILON) return 0.0f;
        return std::sqrt(M2c) * std::sqrt(exposure_hours);
    }
    float getMotionComfortLevel(float exposure_hours = 1.0f, bool bias_corrected = true) const {
        float msdv = getMotionSicknessDoseValue(exposure_hours, bias_corrected);
        if (msdv <= 0.1f) return 100.0f;
        if (msdv <= 0.2f) return 80.0f;
        if (msdv <= 0.3f) return 60.0f;
        if (msdv <= 0.4f) return 40.0f;
        if (msdv <= 0.5f) return 20.0f;
        return 0.0f;
    }
    float getVerticalMotionIntensity(bool bias_corrected = true) const {
        float M2c = getMoment2(bias_corrected);
        return (M2c > EPSILON) ? std::sqrt(M2c) : 0.0f;
    }
    float getMotionCharacter() const {
        float f_dom = getMeanFrequencyHz(false);
        if (f_dom <= EPSILON) return 0.0f;
        float optimal_freq = 0.2f;
        float freq_factor = std::exp(-std::pow((f_dom - optimal_freq) / 0.1f, 2.0f));
        return std::clamp(freq_factor, 0.0f, 1.0f);
    }
    float getTimeToOnset(float susceptibility = 0.5f, bool bias_corrected = true) const {
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        if (M0c <= EPSILON) return std::numeric_limits<float>::infinity();
        float accel_rms = std::sqrt(M2c);
        float f_dom = getMeanFrequencyHz(bias_corrected);
        if (f_dom <= EPSILON || accel_rms <= EPSILON)
            return std::numeric_limits<float>::infinity();
        float t_onset = 30.0f / (accel_rms * std::pow(f_dom, 0.7f) * susceptibility);
        return std::max(1.0f, t_onset);
    }

    // Spectral Shape Descriptors
    float getBandwidthEpsilon(bool bias_corrected = true) const {
        if (!extended_metrics) throw std::logic_error("Extended metrics not enabled");
        float M0c = getMoment0(bias_corrected);
        float M2c = getMoment2(bias_corrected);
        float M4c = getMoment4(false); // left uncorrected
        if (M0c <= EPSILON || M4c <= EPSILON) return 0.0f;
        float val = 1.0f - (M2c * M2c) / (M0c * M4c);
        return (val > 0.0f) ? std::sqrt(val) : 0.0f;
    }
    float classifySpectrumType(bool bias_corrected = true) const {
        // Heuristic: 0 = swell, 1 = mixed, 2 = wind-sea
        float gamma = getPeakEnhancementFactor();
        float Tz = getMeanPeriod_Tz(bias_corrected);
        float Tp = 1.0f / std::max(getMeanFrequencyHz(bias_corrected), EPSILON);
        float ratio = (Tz > EPSILON) ? Tp / Tz : 1.0f;
        if (gamma < 1.5f && ratio > 1.2f) return 0.0f; // swell
        if (gamma > 3.0f && ratio < 0.9f) return 2.0f; // wind sea
        return 1.0f; // mixed
    }

    // Wave Group Statistics
    float getMeanGroupLength(bool bias_corrected = true) const {
        // Approximate: inverse of RBW
        float rbw_val = getRBW();
        return (rbw_val > EPSILON) ? (1.0f / rbw_val) : 0.0f;
    }
    float getGroupHeightFactor(bool bias_corrected = true) const {
        // Hs_group / Hs_total ~ inverse of RBW scaling
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        float rbw_val = getRBW();
        return (rbw_val > EPSILON) ? (Hs / (1.0f + rbw_val)) : Hs;
    }
    float getRunLengthAboveThreshold(float alpha = 1.0f, bool bias_corrected = true) const {
        // Expected number of consecutive waves with crest > alpha·Hs
        float p_exceed = getExceedanceProbRayleigh(alpha * getSignificantWaveHeightRayleigh(bias_corrected),
                                                  bias_corrected);
        if (p_exceed <= EPSILON) return 0.0f;
        return 1.0f / p_exceed;
    }

    // Extreme Value Analysis (short/medium-term)
    float getWeibullReturnHeight(float T_return, float duration_s,
                                 bool bias_corrected = true) const {
        // Fit simple 2-parameter Weibull to block maxima
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        if (Hs <= EPSILON || duration_s <= 0.0f) return 0.0f;
        float N = estimateWaveCount(duration_s, bias_corrected);
        if (N <= 0.0f) return 0.0f;
        // Shape parameter (k) ~ 2, scale λ ~ Hs / sqrt(log 2)
        float k = 2.0f;
        float lambda = Hs / std::sqrt(std::log(2.0f));
        float p = 1.0f - 1.0f / std::max(T_return, 1.0f);
        return lambda * std::pow(-std::log(1.0f - p), 1.0f / k);
    }
    float getPOTMeanExcess(float threshold, bool bias_corrected = true) const {
        // Peak-over-threshold: mean excess above given crest threshold
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        if (Hs <= EPSILON) return 0.0f;
        if (threshold <= EPSILON) threshold = Hs;
        float sigma = Hs / std::sqrt(2.0f);
        float p = std::exp(-threshold * threshold / (2.0f * sigma * sigma));
        return (p > EPSILON) ? sigma * p / (1.0f - p) : 0.0f;
    }

    // Shallow-Water Extensions
    float getWavenumber(float omega, float depth) const {
        if (depth <= EPSILON) return omega * omega / g(); // deep-water approx
        // Newton iteration for dispersion ω² = gk tanh(kh)
        float k = omega * omega / g();
        for (int i = 0; i < 3; ++i) {
            float f = g() * k * std::tanh(k * depth) - omega * omega;
            float df = g() * (std::tanh(k * depth) + k * depth / std::cosh(k * depth) / std::cosh(k * depth));
            k -= f / std::max(df, EPSILON);
        }
        return k;
    }
    float getShoalingCoefficient(float omega, float depth) const {
        float k = getWavenumber(omega, depth);
        float cg = 0.5f * g() / omega * (1.0f + 2.0f * k * depth / std::sinh(2.0f * k * depth));
        float cg0 = 0.5f * g() / omega; // deep water group velocity
        return std::sqrt(cg0 / std::max(cg, EPSILON));
    }
    float getRefractionSnell(float theta0, float omega, float depth0, float depth) const {
        float k0 = getWavenumber(omega, depth0);
        float k = getWavenumber(omega, depth);
        float c0 = omega / k0;
        float c = omega / k;
        float val = std::sin(theta0) * c0 / c;
        return std::asin(std::clamp(val, -1.0f, 1.0f));
    }
    float getBattjesJanssenBreakingProb(float gamma, float depth, bool bias_corrected = true) const {
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        if (depth <= EPSILON || Hs <= EPSILON) return 0.0f;
        float Hb = gamma * depth;
        return std::clamp(Hs / Hb, 0.0f, 1.0f);
    }

    // Data Quality & Diagnostics
    float getTemporalStability(float window_var) const {
        // Proxy: compare variance of Hs against supplied reference
        if (window_var <= EPSILON) return 1.0f;
        float rel_var = var_slow / window_var;
        return std::clamp(1.0f - rel_var, 0.0f, 1.0f);
    }
    float getBandSNR(float f_low, float f_high, bool bias_corrected = true) const {
        // Approximate: ratio of M0 in-band vs. total
        float M0c = getMoment0(bias_corrected);
        if (M0c <= EPSILON) return 0.0f;
        float omega_low = 2.0f * float(M_PI) * f_low;
        float omega_high = 2.0f * float(M_PI) * f_high;
        float inband = (omega_last >= omega_low && omega_last <= omega_high) ? M0c : 0.0f;
        return (inband > 0.0f) ? (inband / M0c) : 0.0f;
    }
    float getDataGapFraction() const {
        // Based on missing input samples count (approximate from NaN handling)
        // For now, return 0 (no internal counter implemented)
        return 0.0f;
    }

    // Integration with Environmental Inputs
    float classifyWaveAge(float U10, bool bias_corrected = true) const {
        if (U10 <= EPSILON) return 0.0f;
        float age = getWaveAge(U10, bias_corrected);
        if (age < 0.8f) return 0.0f; // young
        if (age > 1.2f) return 2.0f; // old swell
        return 1.0f; // mature
    }
    float getSeaSwellPartition(float U10, bool bias_corrected = true) const {
        // Return fraction of energy considered "swell"
        if (U10 <= EPSILON) return 1.0f;
        float age = getWaveAge(U10, bias_corrected);
        return (age > 1.2f) ? 1.0f : 0.5f; // heuristic
    }
    float getRadiationStressXX(bool bias_corrected = true) const {
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        return 0.25f * rho() * g() * Hs * Hs;
    }
    float getRadiationStressYY(bool bias_corrected = true) const {
        return getRadiationStressXX(bias_corrected);
    }
    float getRadiationStressXY() const { return 0.0f; } // no directionality

    // Application-Specific Metrics
    float getBottomOrbitalVelocity(float depth, bool bias_corrected = true) const {
        if (depth <= EPSILON) return 0.0f;
        float Hs = getSignificantWaveHeightRayleigh(bias_corrected);
        float Tz = getMeanPeriod_Tz(bias_corrected);
        float omega = two_pi() / std::max(Tz, EPSILON);
        float k = getWavenumber(omega, depth);
        return (M_PI * Hs / Tz) / std::sinh(k * depth);
    }
    float getWECCaptureWidthRatio(float extracted_kWm, bool bias_corrected = true) const {
        float Pwave = getWavePower(bias_corrected);
        if (Pwave <= EPSILON) return 0.0f;
        return extracted_kWm / Pwave;
    }

private:
    static constexpr float rho() { return 1025.0f; } // seawater density

    bool extended_metrics;
    bool negative_moments;

    float tau_env = 0.0f, tau_mom = 0.0f, tau_coh = 0.0f, tau_omega = 0.0f;
    float omega_min = 0.0f;
    float alpha_env = 0.0f, alpha_mom = 0.0f, alpha_coh = 0.0f, alpha_omega = 0.0f;

    float phi = 0.0f;
    float z_real = 0.0f, z_imag = 0.0f;

    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f, M3 = 0.0f, M4 = 0.0f, M_neg1 = 0.0f;
    float mu2 = 0.0f, mu3 = 0.0f, mu4 = 0.0f;
    float nu = 0.0f, R_spec = 0.0f, R_phase = 0.0f, rbw = 0.0f;

    float coh_r = 0.0f, coh_i = 0.0f; bool has_coh = false;

    float omega_lp = 0.0f, omega_disp_lp = 0.0f;
    float omega_last = 0.0f, mu_w = 0.0f;
    float var_fast = 0.0f, var_slow = 0.0f;
    bool has_omega_lp = false, has_omega_disp_lp = false, has_moments = false;
    float omega_phi_last = 0.0f;

    float wbar_ema = 0.0f, w2bar_ema = 0.0f;
    float dphi_mean = 0.0f, dphi_var = 0.0f;

    constexpr static float ALPHA_FAST = 0.05f;
    constexpr static float ALPHA_SLOW = 0.02f;

    static constexpr float g() { return 9.80665f; }
    static constexpr float two_pi() { return 2.0f * float(M_PI); }
    static constexpr float inv_two_pi() { return 1.0f / (2.0f * float(M_PI)); }
    static constexpr float two_pi_sq() { return 2.0f * float(M_PI) * float(M_PI); }
    static float deep_water_wavelength(float T) { return (T > 0.0f) ? (g() * T * T / two_pi()) : 0.0f; }

    void updateAlpha(float dt_s) {
        alpha_env   = 1.0f - std::exp(-dt_s / tau_env);
        alpha_mom   = 1.0f - std::exp(-dt_s / tau_mom);
        alpha_coh   = 1.0f - std::exp(-dt_s / tau_coh);
        alpha_omega = (tau_omega > 0.0f) ? (1.0f - std::exp(-dt_s / tau_omega)) : 1.0f;
    }

    void demodulateAcceleration(float accel_z, float omega_inst, float dt_s) {
        float w_target = std::max(omega_inst, omega_min);
        if (!has_omega_lp) omega_phi_last = w_target;
        float dw = w_target - omega_phi_last;
        float dw_clamped = std::clamp(dw, -OMEGA_DD_MAX * dt_s, OMEGA_DD_MAX * dt_s);
        float omega_phi = omega_phi_last + dw_clamped;
        omega_phi_last = omega_phi;
        phi += omega_phi * dt_s;
        phi = std::fmod(phi, two_pi());
        float inst_freq = omega_phi;
        float delta_f = inst_freq - dphi_mean;
        dphi_mean += ALPHA_FAST * delta_f;
        dphi_var  = (1.0f - ALPHA_FAST) * dphi_var + ALPHA_FAST * delta_f * delta_f;
        float c = std::cos(-phi), s = std::sin(-phi);
        float y_real = accel_z * c;
        float y_imag = accel_z * s;
        if (!has_omega_lp) {
            z_real = y_real; z_imag = y_imag;
            omega_lp = w_target;
            omega_last = mu_w = omega_lp;
            var_fast = var_slow = 0.0f;
            has_omega_lp = true;
            return;
        }
        z_real = (1.0f - alpha_env) * z_real + alpha_env * y_real;
        z_imag = (1.0f - alpha_env) * z_imag + alpha_env * y_imag;
        omega_lp = (tau_omega > 0.0f)
                 ? (1.0f - alpha_omega) * omega_lp + alpha_omega * w_target
                 : w_target;
        float delta = omega_lp - mu_w;
        mu_w    += ALPHA_FAST * delta;
        var_fast = (1.0f - ALPHA_FAST) * var_fast + ALPHA_FAST * delta * delta;
        var_slow = (1.0f - ALPHA_SLOW) * var_slow + ALPHA_SLOW * delta * delta;
        omega_last = omega_lp;
    }

    void updateSpectralMoments() {
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
                M_neg1 = P_disp / omega_norm;
            }
            has_moments = true;
        } else {
            float a = alpha_mom, b = (1.0f - alpha_mom);
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
        wbar_ema  = (1.0f - alpha_mom) * wbar_ema  + alpha_mom * omega_norm;
        w2bar_ema = (1.0f - alpha_mom) * w2bar_ema + alpha_mom * omega_norm * omega_norm;
        float omega_disp = (M0 > EPSILON) ? (M1 / M0) : omega_min;
        if (!has_omega_disp_lp) {
            omega_disp_lp = omega_disp;
            has_omega_disp_lp = true;
        } else {
            omega_disp_lp = (1.0f - alpha_omega) * omega_disp_lp + alpha_omega * omega_disp;
        }
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
        float omega_bar = getMeanFrequencyRad(false);
        float mu2_two_pole = std::max(0.0f, w2bar_ema - wbar_ema * wbar_ema - std::max(var_slow, 0.0f));
        rbw   = (omega_bar > 0.0f) ? (std::sqrt(mu2_two_pole) / std::max(omega_bar, omega_min)) : 0.0f;
        R_spec = std::clamp(std::exp(-BETA_SPEC * rbw), 0.0f, 1.0f);
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

    float applyMomentCorrection(float Mraw, float c_coeff) const {
        if (Mraw <= EPSILON) return 0.0f;
        float omega_bar = mu_w;
        float var = std::max(var_slow, 0.0f);
        if (omega_bar <= EPSILON || c_coeff <= 0.0f) return Mraw;
        float corr = 1.0f / (1.0f + c_coeff * var / (omega_bar * omega_bar));
        return Mraw * corr;
    }

    bool isBiasCorrectionSignificant(float threshold = 0.01f) const {
        if (mu_w <= EPSILON) return false;
        return (std::max(var_slow, 0.0f) / (mu_w * mu_w)) > threshold;
    }

    static float erfinv_approx(float x) {
        x = std::clamp(x, -0.999999f, 0.999999f);
        const float a = 0.147f;
        float ln = std::log(1.0f - x * x);
        float tt1 = 2.0f / (float(M_PI) * a) + 0.5f * ln;
        float tt2 = 1.0f / a * ln;
        float sign = (x < 0.0f) ? -1.0f : 1.0f;
        float y = sign * std::sqrt(std::sqrt(tt1 * tt1 - tt2) - tt1);
        for (int i = 0; i < 2; ++i) {
            float ey  = std::erf(y) - x;
            float dy  = (2.0f / std::sqrt(float(M_PI))) * std::exp(-y * y);
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
    static float chi2Quantile(float p, int k) {
        if (p <= 0.0f) return 0.0f;
        if (p >= 1.0f) return std::numeric_limits<float>::infinity();
        float z = normalQuantile(p);
        float v = (float)k;
        float h = 2.0f / (9.0f * v);
        return v * std::pow(1.0f - h + z * std::sqrt(h), 3.0f);
    }
};
