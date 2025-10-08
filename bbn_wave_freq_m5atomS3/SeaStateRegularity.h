#pragma once

#include <cmath>
#include <limits>
#include <algorithm>

/*
    Copyright 2025, Mikhail Grushinskiy

    SeaStateRegularity — Online estimator of ocean wave regularity from vertical acceleration.

    Inputs
      • Vertical acceleration a_z(t) [m/s²]
      • Instantaneous angular frequency ω_inst(t) [rad/s] from an external tracker

    Physics & Spectral Relations (solid math, not heuristics)
      a_z(t) = d²η/dt² = −ω² η(t)
      ⇒ S_a(ω) = ω⁴ S_η(ω),   S_η(ω) = S_a(ω) / ω⁴

    Moments of the displacement spectrum (continuous):
      M₀ = ∫ S_η(ω) dω       (variance of η)
      M₁ = ∫ ω S_η(ω) dω
      M₂ = ∫ ω² S_η(ω) dω
      Narrowness: ν = sqrt( M₂/M₀ − (M₁/M₀)² ) / (M₁/M₀)
      Oceanographic significant height: H_s ≈ 4√M₀
      Mean/“displacement” frequency: \bar{ω} = M₁/M₀  (Hz = \bar{ω}/2π)

    Discrete, per-bin estimator used here
      • Demodulate a_z at candidate ω_k → baseband Y_k (I/Q), then 1st-order LPF with cutoff f_c,k.
      • Convert to displacement envelope via η̂_k = −Y_k / ω_k².
      • Per-bin captured power P_k = |η̂_k|² (m²).
      • 1st-order LPF ENBW in rad/s: ENBW_k = π² f_c,k  (with f_c,k in Hz).
      • PSD estimate Ŝ_η(ω_k) ≈ (K_EFF_MIX * P_k) / ENBW_k, where K_EFF_MIX≈2 compensates the I/Q halving.
      • Integrate moments with each bin’s Voronoi width Δω_k in linear ω:
            M_n ≈ Σ_k Ŝ_η(ω_k) · ω_k^n · Δω_k

    Jensen correction for ratio bias:
      Let S0=∫S_η dω, S1=∫ω S_η dω. We track ⟨S0²⟩ and ⟨S0·S1⟩ to approximate
      Var[M0] and Cov[M1,M0]. Then
        \bar{ω}_naive = M1/M0
        \bar{ω}_corr  = \bar{ω}_naive + ( \bar{ω}_naive Var[M0] − Cov[M1,M0] ) / M0²

    Regularity score:
      • Spectral (bandwidth-based): R_spec = exp(−β ν), β≈1
      • Phase coherence R_phase from unit envelope vector averaging
      • Final: R_out = EMA{ max(R_phase, R_spec) }

    NOTE: All ω here are radians per second. Your caller already passes ω_inst = 2π·freq(Hz).
*/

// Debiased EMA
struct DebiasedEMA {
  float value  = 0.0f;
  float weight = 0.0f;
  void reset() {
    value = 0.0f;
    weight = 0.0f;
  }
  inline void update(float x, float alpha) {
    value  = (1.0f - alpha) * value + alpha * x;
    weight = (1.0f - alpha) * weight + alpha;
  }
  inline void decay(float alpha) {         // optional decay hook
    value  = (1.0f - alpha) * value;
    weight = (1.0f - alpha) * weight;
  }
  inline float get() const {
    return (weight > 1e-12f) ? value / weight : 0.0f;
  }
  inline bool  isReady() const {
    return weight > 1e-6f;
  }
};

// SeaStateRegularity
class SeaStateRegularity {
  public:
    // Numerics / mapping
    constexpr static float EPSILON    = 1e-12f;
    constexpr static float BETA_SPEC  = 1.0f;     // exponent in ν
    constexpr static float K_EFF_MIX  = 2.0f;     // amplitude calibration (I/Q → variance)

    // Tracker-robust ω clamp and smoothing (Hz range widened for real seas)
    constexpr static float OMEGA_MIN_HZ = 0.01f;  // 100 s swell
    constexpr static float OMEGA_MAX_HZ = 3.00f;  // 0.33 s wind chop
    constexpr static float TAU_W_SEC    = 15.0f;  // EMA time-constant for ω_used

    // Multi-bin params (ratio spacing)
    constexpr static int   MAX_K       = 25;      // up to ±25 bins → 51 bins total
    constexpr static int   NBINS       = 2 * MAX_K + 1;
    constexpr static float MIN_FC_HZ   = 0.02f;

    SeaStateRegularity(float tau_mom_sec = 180.0f,
                       float tau_coh_sec = 10.0f,
                       float tau_out_sec = 30.0f)
    {
      tau_mom = tau_mom_sec;
      tau_coh = std::max(1e-3f, tau_coh_sec);
      tau_out = std::max(1e-3f, tau_out_sec);
      reset();
    }

    void reset() {

      M0.reset(); M1.reset(); M2.reset();
      A0.reset(); A1_mean.reset(); A2_second.reset();

      // For Jensen correction
      Q00.reset();  // ⟨S0^2⟩
      Q10.reset();  // ⟨S0*S1⟩
      Q20.reset();

      R_out.reset();

      R_spec = R_phase = 0.0f;
      nu = 0.0f;

      omega_bar_corr = 0.0f;
      omega_bar_naive = 0.0f;

      omega_used = 0.0f;
      alpha_w    = 0.0f;

      has_moments = false;
      last_dt = -1.0f;
      alpha_mom = alpha_coh = alpha_out = 0.0f;

      for (int i = 0; i < NBINS; i++) {
        bin_zr[i] = bin_zi[i] = 0.0f;
        bin_c[i] = 1.0f;
        bin_s[i] = 0.0f;
        coh_r_k[i] = 0.0f;
        coh_i_k[i] = 0.0f;
      }
      bins_init = false;
      last_accel = 0.0f;

      for (int i = 0; i < NBINS; ++i) last_S_eta_hat[i] = 0.0f;
    }

    // Main update
    void update(float dt_s, float accel_z, float omega_inst) {
      if (!(dt_s > 0.0f)) return;
      if (!std::isfinite(accel_z) || !std::isfinite(omega_inst)) return;

      last_accel = accel_z;

      updateAlpha(dt_s);

// --- Direct time-domain acceleration variance ---
A1_mean.update(accel_z,           alpha_mom);
A2_second.update(accel_z*accel_z, alpha_mom);

float a_mean = A1_mean.get();
float a_var  = std::max(0.0f, A2_second.get() - a_mean * a_mean);

// Optional: subtract known sensor noise floor if available
// const float NOISE_VAR = n_a * n_a * B_eff;
// a_var = std::max(0.0f, a_var - NOISE_VAR);

A0.update(a_var, alpha_mom);   // keep A0 as your variance cache

      updateSpectralMoments(omega_inst);
        updatePhaseCoherence();
      computeRegularityOutput();
    }

    // Getters
    float getRegularity() const {
      return R_out.get();
    }
    float getRegularitySpectral() const {
      return R_spec;
    }
    float getRegularityPhase() const {
      return R_phase;
    }
    float getNarrowness() const {
      return nu;
    }

    float getDisplacementFrequencyNaiveHz() const {
      return (omega_bar_naive > EPSILON) ? (omega_bar_naive / (2.0f * PI)) : 0.0f;
    }

// Wave height envelope and frequency blending
float getWaveHeightEnvelopeEst() const {
    float m0 = M0.get();
    if (!(m0 > 0.0f)) return 0.0f;

    // Random-sea (Rayleigh) significant height
    float Hs_rand = 4.0f * std::sqrt(m0);

    // Deterministic single-wave height (2A = 2√(2M₀))
    float Hs_mono = 2.0f * std::sqrt(2.0f * m0);

    // Correction for harmonic over-amplification under strong coherence
    float R = std::clamp(R_phase, 0.0f, 1.0f);
    float R2 = R * R;
    float correction = 1.0f / (1.0f + 2.0f * R2);  // softened suppression (was 4.0f)
    Hs_mono *= correction;

    // Blend: coherent → corrected mono-wave, random → oceanographic
    return R * Hs_mono + (1.0f - R) * Hs_rand;
}

float getDisplacementFrequencyHz() const {
    float m0 = M0.get();
    float m2 = M2.get();
    if (!(m0 > EPSILON)) return 0.0f;

    float q00 = Q00.get();
    float q20 = Q20.get();
    float varM0  = std::max(0.0f, q00 - m0 * m0);
    float cov20  = q20 - m2 * m0;
    float invM0_2 = 1.0f / std::max(m0 * m0, EPSILON);

    float r2_naive = m2 / m0;
    float r2_corr  = r2_naive + (r2_naive * varM0 - cov20) * invM0_2;
    r2_corr = std::max(r2_corr, 0.0f);

    float omega_z = std::sqrt(r2_corr);
    return omega_z / (2.0f * PI);
}

float getDisplacementPeriodSec() const {
    float fz = getDisplacementFrequencyHz();
    return (fz > EPSILON) ? (1.0f / fz) : 0.0f;
}

    float getAccelerationVariance() const {
      return A0.get();
    }

  private:
    // Constants
    static constexpr float PI             = 3.14159265358979323846f;
    static constexpr float TWO_PI_  = 2.0f * PI;  
    static constexpr float OMEGA_MIN_RAD  = TWO_PI_ * OMEGA_MIN_HZ;
    static constexpr float OMEGA_MAX_RAD  = TWO_PI_ * OMEGA_MAX_HZ;

    // time constants and alphas
    float tau_mom, tau_coh, tau_out;
    float last_dt;
    float alpha_mom, alpha_coh, alpha_out;

    // ω_used smoothing
    float omega_used;
    float alpha_w;

    // raw accel cache for multi-bin
    float last_accel;

    // per-bin demod states
    float bin_zr[NBINS], bin_zi[NBINS];
    float bin_c[NBINS],  bin_s[NBINS];
    bool  bins_init;

    // Per-bin phase-coherence EMAs (unit-phasor averages)
    float coh_r_k[NBINS], coh_i_k[NBINS];

    // moments (primary)
    DebiasedEMA M0, M1, M2;
    DebiasedEMA A0;

    // Direct time-domain acceleration variance helpers
    DebiasedEMA A1_mean;   // mean of acceleration
    DebiasedEMA A2_second; // mean of a^2

    // moments for Jensen correction
    // Q00 ≈ ⟨S0^2⟩, Q10 ≈ ⟨S0*S1⟩, where S0=ΣYk, S1=Σ(Yk*ωk)
    DebiasedEMA Q00, Q10;
    DebiasedEMA Q20;  // ⟨S0*S2⟩ for Jensen correction of M2/M0

    // output
    DebiasedEMA R_out;
    float R_spec, R_phase;

    // cached
    float nu;
    float omega_bar_corr;
    float omega_bar_naive;
    bool  has_moments;

    float last_S_eta_hat[NBINS] = {0.0f};    // PSD per bin from last update
    float last_P_acc[NBINS] = {0.0f};   // baseband power in acceleration units

    // Helpers
    void updateAlpha(float dt_s) {
      if (dt_s == last_dt) return;
      alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
      alpha_coh = 1.0f - std::exp(-dt_s / tau_coh);
      alpha_out = 1.0f - std::exp(-dt_s / tau_out);
      alpha_w   = 1.0f - std::exp(-dt_s / TAU_W_SEC);
      last_dt = dt_s;
    }

void updatePhaseCoherence() {
    float smax = 0.0f;
    for (int i = 0; i < NBINS; ++i)
        smax = std::max(smax, last_S_eta_hat[i]);
    const float THRESH = 0.005f * smax;

    // --- temporal update per bin ---
    for (int i = 0; i < NBINS; ++i) {
        float w = last_S_eta_hat[i];
        if (!(w > THRESH)) continue;

        float zr = bin_zr[i];
        float zi = bin_zi[i];
        float mag = std::hypot(zr, zi);
        if (mag <= EPSILON) continue;

        float ur = zr / mag;
        float ui = zi / mag;

        coh_r_k[i] = (1.0f - alpha_coh) * coh_r_k[i] + alpha_coh * ur;
        coh_i_k[i] = (1.0f - alpha_coh) * coh_i_k[i] + alpha_coh * ui;
    }

    // --- global coherence across bins ---
    float sum_cr = 0.0f, sum_ci = 0.0f, sum_w = 0.0f;
    for (int i = 0; i < NBINS; ++i) {
        float w = last_S_eta_hat[i];
        if (!(w > THRESH)) continue;
        sum_cr += w * coh_r_k[i];
        sum_ci += w * coh_i_k[i];
        sum_w  += w;
    }

    float R_now = (sum_w > EPSILON) ? (std::hypot(sum_cr, sum_ci) / sum_w) : 0.0f;
    R_phase = (1.0f - alpha_coh) * R_phase + alpha_coh * R_now;
}

    // Spectral moments: physically correct a→η conversion (1/ω⁴)
    void updateSpectralMoments(float omega_inst) {
      float w_obs = std::clamp(omega_inst, OMEGA_MIN_RAD, OMEGA_MAX_RAD);

      // Smooth ω_used
      if (omega_used <= 0.0f) omega_used = w_obs;
      else                    omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

      // Outlier gate: skip updates if tracker jumps too far
      if (omega_used > 0.0f) {
        float ratio = w_obs / omega_used;
        if (ratio < 0.7f || ratio > 1.3f) return;
      }

      int   K = MAX_K;                   // keep full span
      // Choose target upward span to capture Fenton/cnoidal harmonics (≥5× recommended)
      constexpr float TARGET_SPAN_UP = 6.0f;   // cover to ~6·ω_used (includes 5th harmonic)
      const float r = std::exp(std::log(TARGET_SPAN_UP) / float(K));  // ratio step
        
      if (!bins_init) {
        for (int i = 0; i < NBINS; i++) {
          bin_c[i]  = 1.0f;
          bin_s[i]  = 0.0f;
          bin_zr[i] = 0.0f;
          bin_zi[i] = 0.0f;
          coh_r_k[i] = 0.0f;
          coh_i_k[i] = 0.0f;
        }
        bins_init = true;
      }

      // Ratio-spaced ω grid around ω_used
      const int left  = MAX_K - K;
      const int right = MAX_K + K;

      float omega_k_arr[NBINS] = {};
      omega_k_arr[MAX_K] = omega_used;
      for (int k = 1; k <= K; ++k) {
        omega_k_arr[MAX_K + k] = omega_k_arr[MAX_K + k - 1] * r;
        omega_k_arr[MAX_K - k] = omega_k_arr[MAX_K - k + 1] / r;
      }
      for (int idx = left; idx <= right; ++idx)
        omega_k_arr[idx] = std::clamp(omega_k_arr[idx], OMEGA_MIN_RAD, OMEGA_MAX_RAD);

      // Voronoi Δω_k in linear ω
      float domega_k_arr[NBINS] = {};
      if (K == 0) {
        domega_k_arr[MAX_K] = 0.0f;  // will use ENBW later
      } else {
        for (int idx = left; idx <= right; ++idx) {
          if (idx == left) {
            float w0 = omega_k_arr[idx], w1 = omega_k_arr[idx + 1];
            domega_k_arr[idx] = std::max(EPSILON, w1 - w0);
          } else if (idx == right) {
            float wL = omega_k_arr[idx - 1], w0 = omega_k_arr[idx];
            domega_k_arr[idx] = std::max(EPSILON, w0 - wL);
          } else {
            float wL = omega_k_arr[idx - 1], wR = omega_k_arr[idx + 1];
            domega_k_arr[idx] = std::max(EPSILON, 0.5f * (wR - wL));
          }
        }
      }

      float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;

      // Bin loop
      for (int idx = left; idx <= right; ++idx) {
        float omega_k = omega_k_arr[idx];
        if (omega_k <= EPSILON) continue;

        // Advance oscillator for this bin
        float dphi = omega_k * last_dt;
        float cd = std::cos(dphi), sd = std::sin(dphi);
        float c0 = bin_c[idx], s0 = bin_s[idx];
        float c1 =  c0 * cd - s0 * sd;
        float s1 =  c0 * sd + s0 * cd;
        bin_c[idx] = c1; bin_s[idx] = s1;

        // Mix acceleration to baseband
        float y_r =  last_accel * c1;
        float y_i = -last_accel * s1;

        // Per-bin LPF and ENBW (Hz→rad/s)
        float f_k_hz  = omega_k / TWO_PI_;
        float fc_k_hz = std::max(MIN_FC_HZ, (K > 0 ? (r - 1.0f) * f_k_hz : MIN_FC_HZ));
        float alpha_k = 1.0f - std::exp(-last_dt * TWO_PI_ * fc_k_hz);
        float enbw_k  = PI * PI * fc_k_hz; // [rad/s]

        // Low-pass baseband envelope
        bin_zr[idx] = (1.0f - alpha_k) * bin_zr[idx] + alpha_k * y_r;
        bin_zi[idx] = (1.0f - alpha_k) * bin_zi[idx] + alpha_k * y_i;

        // Physically correct acceleration→displacement normalization
        // a = −ω²η ⇒ S_η = S_a / ω⁴
        float w2 = omega_k * omega_k;
        float inv_w4 = 1.0f / std::max(w2 * w2, EPSILON);
        float P_disp = (bin_zr[idx] * bin_zr[idx] + bin_zi[idx] * bin_zi[idx]) * inv_w4;

        // PSD estimate: P ≈ S * ENBW ⇒ S ≈ P / ENBW
        float S_eta_hat = K_EFF_MIX * P_disp / std::max(enbw_k, EPSILON);
        last_S_eta_hat[idx] = S_eta_hat;

        // Integrate moments
        float domega = (K == 0) ? std::max(EPSILON, enbw_k) : domega_k_arr[idx];
        S0 += S_eta_hat * domega;
        S1 += S_eta_hat * omega_k * domega;
        S2 += S_eta_hat * omega_k * omega_k * domega;
      }

      has_moments = true;

      // Update EMAs
      M0.update(S0, alpha_mom);
      M1.update(S1, alpha_mom);
      M2.update(S2, alpha_mom);

      // Jensen correction helpers
      Q00.update(S0 * S0, alpha_mom);
      Q10.update(S0 * S1, alpha_mom); 
      Q20.update(S0 * S2, alpha_mom);
    }

void computeRegularityOutput() {
    if (!M0.isReady()) {
        R_out.update(R_phase, alpha_out);
        R_spec = R_phase;
        nu = 0.0f;
        omega_bar_corr = omega_bar_naive = 0.0f;
        return;
    }

    float m0 = M0.get();
    float m1 = M1.get();
    float m2 = M2.get();

    if (!(m0 > EPSILON)) {
        R_out.update(0.0f, alpha_out);
        R_spec = 0.0f;
        nu = 0.0f;
        omega_bar_corr = omega_bar_naive = 0.0f;
        return;
    }

    // --- Raw means and variances ---
    omega_bar_naive  = m1 / m0;
    float omega2_bar_naive = m2 / m0;

    // --- Jensen corrections ---
    float q00 = Q00.get();
    float q10 = Q10.get();
    float q20 = Q20.get();

    float varM0  = std::max(0.0f, q00 - m0 * m0);
    float cov10  = q10 - m1 * m0;
    float cov20  = q20 - m2 * m0;
    float invM0_2 = 1.0f / std::max(m0 * m0, EPSILON);

    omega_bar_corr  = omega_bar_naive  + (omega_bar_naive  * varM0 - cov10) * invM0_2;
    float omega2_bar_corr = omega2_bar_naive + (omega2_bar_naive * varM0 - cov20) * invM0_2;

    // --- Spectral narrowness ν (all debiased) ---
    float mu2_corr = std::max(0.0f, omega2_bar_corr - omega_bar_corr * omega_bar_corr);
    nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2_corr) / omega_bar_corr) : 0.0f;
    if (!std::isfinite(nu) || nu < 0.0f) nu = 0.0f;

    R_spec = std::clamp(std::exp(-BETA_SPEC * nu), 0.0f, 1.0f);

    // --- Fusion: stronger component dominates ---
    float R_combined = std::max(R_phase, R_spec);
    R_out.update(R_combined, alpha_out);
}

};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

constexpr float SAMPLE_FREQ_HZ   = 240.0f;
constexpr float DT               = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 60.0f;
constexpr float SINE_AMPLITUDE   = 1.0f;
constexpr float SINE_FREQ_HZ     = 0.3f;

struct SineWave {
  float amplitude;
  float omega;
  float phi;
  SineWave(float A, float f_hz)
    : amplitude(A), omega(2.0f * float(M_PI) * f_hz), phi(0.0f) {}
  std::pair<float, float> step(float dt) {
    phi += omega * dt;
    if (phi > 2.0f * float(M_PI)) phi -= 2.0f * float(M_PI);
    float z = amplitude * std::sin(phi);
    float a = -amplitude * omega * omega * std::sin(phi);
    return {z, a};
  }
};

inline void SeaState_sine_wave_test() {
  SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
  SeaStateRegularity reg;
  float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f, nu = 0.0f;
  float f_disp_corr = 0.0f, f_disp_naive = 0.0f, Tp = 0.0f;
  for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
    auto za = wave.step(DT);
    float a = za.second;
    reg.update(DT, a, wave.omega);
    R_spec      = reg.getRegularitySpectral();
    R_phase     = reg.getRegularityPhase();
    Hs_est      = reg.getWaveHeightEnvelopeEst();
    nu          = reg.getNarrowness();
    f_disp_corr = reg.getDisplacementFrequencyHz();
    f_disp_naive = reg.getDisplacementFrequencyNaiveHz();
    Tp          = reg.getDisplacementPeriodSec();
  }

  const float Hs_expected = 4.0f * SINE_AMPLITUDE;

  if (!(R_spec > 0.90f))
    throw std::runtime_error("Sine: R_spec did not converge near 1.");
  if (!(R_phase > 0.80f))
    throw std::runtime_error("Sine: R_phase did not converge near 1.");
  if (!(std::fabs(Hs_est - Hs_expected) < 0.25f * Hs_expected))
    throw std::runtime_error("Sine: Hs estimate not within tolerance.");
  if (!(nu < 0.05f))
    throw std::runtime_error("Sine: Narrowness should be close to 0 for a pure tone.");

  std::cerr << "[PASS] Sine wave test passed. "
            << "Hs_est=" << Hs_est
            << " (expected ~" << Hs_expected << "), Narrowness=" << nu
            << ", f_disp_corr=" << f_disp_corr << " Hz"
            << ", f_disp_naive=" << f_disp_naive << " Hz"
            << ", Tp=" << Tp << " s\n";
}
#endif
