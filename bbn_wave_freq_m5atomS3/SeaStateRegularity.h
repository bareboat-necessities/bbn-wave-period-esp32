#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/*
    Copyright 2025, Mikhail Grushinskiy

    SeaStateRegularity — Online estimator of ocean wave regularity
    --------------------------------------------------------------
    Momentum-based version (no phase coherence or heuristic blending).

    Purpose
      Estimate ocean wave regularity and spectral properties directly
      from vertical acceleration measurements a_z(t).

    Physical relations
      a_z(t) = d²η/dt² = −ω² η(t)
      ⇒ S_a(ω) = ω⁴ S_η(ω)
         S_η(ω) = S_a(ω) / ω⁴

    Spectral moments (displacement spectrum)
      M₀ = ∫ S_η(ω) dω         — variance of displacement η
      M₁ = ∫ ω S_η(ω) dω
      M₂ = ∫ ω² S_η(ω) dω

    Derived quantities
      • Mean angular frequency      ω̄ = M₁ / M₀
      • Spectral narrowness (ν)     ν = sqrt(M₂/M₀ − (M₁/M₀)²) / (M₁/M₀)
      • Oceanographic Hs            Hs = 4√M₀

    Jensen bias correction
      To remove ratio bias from M₁/M₀ and M₂/M₀ estimators:

        ω̄_corr = ω̄_naive + (ω̄_naive Var[M₀] − Cov[M₁,M₀]) / M₀²
        similarly for M₂/M₀ → ω̄²_corr

    Regularity score (moment-based only)
      R_spec = exp(−β·ν), with β≈1–2 (empirical sharpness factor)

    Implementation overview
      • Multi-bin demodulation of acceleration at log-spaced ωₖ around ω_inst.
      • Each bin has 1st-order LPF with cutoff f_c,k and ENBW = π² f_c,k.
      • Convert acceleration power to displacement PSD via division by ωₖ⁴.
      • Accumulate M₀, M₁, M₂, and Jensen correction terms.
*/

//
// === Debiased Exponential Moving Average ==============================
//
struct DebiasedEMA {
  float value  = 0.0f;
  float weight = 0.0f;

  void reset() { value = 0.0f; weight = 0.0f; }

  inline void update(float x, float alpha) {
    value  = (1.0f - alpha) * value + alpha * x;
    weight = (1.0f - alpha) * weight + alpha;
  }

  inline float get() const {
    return (weight > 1e-12f) ? value / weight : 0.0f;
  }

  inline bool isReady() const { return weight > 1e-6f; }
};

//
// === SeaStateRegularity ===============================================
//
class SeaStateRegularity {
public:
  // Numerical constants
  constexpr static float EPSILON    = 1e-12f;
  constexpr static float BETA_SPEC  = 1.7f;    // spectral regularity exponent
  constexpr static float K_EFF_MIX  = 2.0f;    // compensates I/Q demod halving

  // Tracker and frequency limits
  constexpr static float OMEGA_MIN_HZ = 0.02f; // ~50 s swell
  constexpr static float OMEGA_MAX_HZ = 4.00f; // ~0.25 s chop
  constexpr static float TAU_W_SEC    = 30.0f; // smoothing time for ω_used

  // Multi-bin spectral grid parameters
  constexpr static int   MAX_K     = 25;      // ±25 bins → 51 total
  constexpr static int   NBINS     = 2 * MAX_K + 1;
  constexpr static float MIN_FC_HZ = 0.04f;   // minimum LPF cutoff per bin

  //--------------------------------------------------------------------
  // Constructor
  //--------------------------------------------------------------------
  SeaStateRegularity(float tau_mom_sec = 180.0f, float tau_out_sec = 60.0f) {
    tau_mom = tau_mom_sec;
    tau_out = std::max(1e-3f, tau_out_sec);
    reset();
  }

  //--------------------------------------------------------------------
  // Reset all EMAs and state
  //--------------------------------------------------------------------
  void reset() {
    M0.reset(); M1.reset(); M2.reset();
    A0.reset(); A1_mean.reset(); A2_second.reset();
    Q00.reset(); Q10.reset(); Q20.reset();
    R_out.reset();

    R_spec = 0.0f;
    nu = 0.0f;
    omega_bar_corr = omega_bar_naive = 0.0f;

    omega_used = 0.0f;
    alpha_w = 0.0f;
    last_dt = -1.0f;
    alpha_mom = alpha_out = 0.0f;

    has_moments = false;
    last_accel = 0.0f;
    bins_init = false;

    // Clear bin memories
    for (int i = 0; i < NBINS; ++i) {
      bin_zr[i] = bin_zi[i] = 0.0;
      bin_c[i] = 1.0;
      bin_s[i] = 0.0;
      last_S_eta_hat[i] = 0.0f;
      n_harm[i] = 1;
    }
    omega_peak_smooth = 0.0f;
  }

  //--------------------------------------------------------------------
  // Main update per sample
  //--------------------------------------------------------------------
  void update(float dt_s, float accel_z, float omega_inst) {
    if (!(dt_s > 0.0f)) return;
    if (!std::isfinite(accel_z) || !std::isfinite(omega_inst)) return;

    last_accel = accel_z;
    updateAlpha(dt_s);

    // --- Direct acceleration variance estimation ---
    A1_mean.update(accel_z, alpha_mom);
    A2_second.update(accel_z * accel_z, alpha_mom);
    float a_mean = A1_mean.get();
    float a_var  = std::max(0.0f, A2_second.get() - a_mean * a_mean);
    A0.update(a_var, alpha_mom);

    // --- Spectral integration and moment update ---
    updateSpectralMoments(omega_inst);

    // --- Moment-based regularity computation ---
    computeRegularityOutput();
  }

  //--------------------------------------------------------------------
  // === Getters ========================================================
  //--------------------------------------------------------------------
  float getRegularity() const            { return R_out.get(); }
  float getRegularitySpectral() const    { return R_spec; }
  float getNarrowness() const            { return nu; }

  float getDisplacementFrequencyNaiveHz() const {
    return (omega_bar_naive > EPSILON) ? (omega_bar_naive / (2.0f * PI)) : 0.0f;
  }

  float getDisplacementFrequencyHz() const {
    float m0 = M0.get();
    float m2 = M2.get();
    if (!(m0 > EPSILON)) return 0.0f;

    float q00 = Q00.get(), q20 = Q20.get();
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

  // --- Oceanographic significant wave height (Hs = 4√M0) ---
  float getWaveHeightEnvelopeEst() const {
    const float m0 = M0.get();
    if (!(m0 > EPSILON) || !std::isfinite(m0))
      return 0.0f;
    return 4.0f * std::sqrt(m0);
  }

  float getAccelerationVariance() const { return A0.get(); }

private:
  //--------------------------------------------------------------------
  // Internal constants
  //--------------------------------------------------------------------
  static constexpr float PI             = 3.14159265358979323846f;
  static constexpr float TWO_PI_        = 2.0f * PI;
  static constexpr float OMEGA_MIN_RAD  = TWO_PI_ * OMEGA_MIN_HZ;
  static constexpr float OMEGA_MAX_RAD  = TWO_PI_ * OMEGA_MAX_HZ;

  //--------------------------------------------------------------------
  // State variables
  //--------------------------------------------------------------------
  float tau_mom, tau_out;
  float last_dt;
  float alpha_mom, alpha_out;
  float omega_used, alpha_w;
  float last_accel;
  bool  bins_init;
  bool  has_moments;

  // Per-bin demod states
  double bin_zr[NBINS], bin_zi[NBINS];
  double bin_c[NBINS],  bin_s[NBINS];

  // Moment EMAs
  DebiasedEMA M0, M1, M2;
  DebiasedEMA A0, A1_mean, A2_second;
  DebiasedEMA Q00, Q10, Q20;
  DebiasedEMA R_out;

  // Outputs and cached quantities
  float R_spec, nu;
  float omega_bar_corr, omega_bar_naive;

  // Misc buffers
  float last_S_eta_hat[NBINS];
  int   n_harm[NBINS];
  float omega_peak_smooth;

  //--------------------------------------------------------------------
  // Update exponential smoothing coefficients
  //--------------------------------------------------------------------
  void updateAlpha(float dt_s) {
    if (dt_s == last_dt) return;
    alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
    alpha_out = 1.0f - std::exp(-dt_s / tau_out);
    alpha_w   = 1.0f - std::exp(-dt_s / TAU_W_SEC);
    last_dt   = dt_s;
  }

  //--------------------------------------------------------------------
  // Spectral demodulation and moment accumulation
  //--------------------------------------------------------------------
  void updateSpectralMoments(float omega_inst) {
    float w_obs = std::clamp(omega_inst, OMEGA_MIN_RAD, OMEGA_MAX_RAD);

    // Smooth ω_used to prevent jitter
    if (omega_used <= 0.0f) omega_used = w_obs;
    else omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

    // Reject large ω jumps (>±30%)
    if (omega_used > 0.0f) {
      float ratio = w_obs / omega_used;
      if (ratio < 0.7f || ratio > 1.3f) return;
    }

    // === Log-spaced ω grid around ω_used ===
    const int K = MAX_K;
    constexpr float TARGET_SPAN_UP = 6.0f;
    const float r = std::exp(std::log(TARGET_SPAN_UP) / float(K));

    float omega_k_arr[NBINS] = {};
    omega_k_arr[MAX_K] = omega_used;
    for (int k = 1; k <= K; ++k) {
      omega_k_arr[MAX_K + k] = omega_k_arr[MAX_K + k - 1] * r;
      omega_k_arr[MAX_K - k] = omega_k_arr[MAX_K - k + 1] / r;
    }

    // Voronoi bin widths Δω
    float domega_k_arr[NBINS];
    for (int i = 0; i < NBINS; ++i) {
      float wL = (i > 0) ? omega_k_arr[i - 1] : omega_k_arr[i];
      float wR = (i < NBINS - 1) ? omega_k_arr[i + 1] : omega_k_arr[i];
      domega_k_arr[i] = std::max(EPSILON, 0.5f * (wR - wL));
    }

    // Accumulators
    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;

    // === Bin loop: demodulate, LPF, accumulate ===
    for (int i = 0; i < NBINS; ++i) {
      float omega_k = omega_k_arr[i];
      float f_k_hz  = omega_k / TWO_PI_;
      float fc_k_hz = std::max(MIN_FC_HZ, (r - 1.0f) * f_k_hz);
      float alpha_k = 1.0f - std::exp(-last_dt * TWO_PI_ * fc_k_hz);
      float enbw_k  = PI * PI * fc_k_hz;

      // Baseband demodulation of acceleration
      float y_r = last_accel * std::cos(omega_k * last_dt);
      float y_i = -last_accel * std::sin(omega_k * last_dt);

      // Low-pass filtering
      bin_zr[i] = (1.0f - alpha_k) * bin_zr[i] + alpha_k * y_r;
      bin_zi[i] = (1.0f - alpha_k) * bin_zi[i] + alpha_k * y_i;

      // Acceleration power per bin
      float P_acc = bin_zr[i]*bin_zr[i] + bin_zi[i]*bin_zi[i];

      // Convert to displacement PSD
      float w2 = omega_k * omega_k;
      float inv_w4 = 1.0f / std::max(w2 * w2, EPSILON);
      float P_disp = P_acc * inv_w4;

      // PSD estimate of displacement
      float S_eta_hat = K_EFF_MIX * P_disp / std::max(enbw_k, EPSILON);
      last_S_eta_hat[i] = S_eta_hat;

      // Accumulate spectral moments
      float domega = domega_k_arr[i];
      S0 += S_eta_hat * domega;
      S1 += S_eta_hat * omega_k * domega;
      S2 += S_eta_hat * omega_k * omega_k * domega;
    }

    // === Update EMAs and Jensen helper moments ===
    M0.update(S0, alpha_mom);
    M1.update(S1, alpha_mom);
    M2.update(S2, alpha_mom);
    Q00.update(S0 * S0, alpha_mom);
    Q10.update(S0 * S1, alpha_mom);
    Q20.update(S0 * S2, alpha_mom);
    has_moments = true;
  }

  //--------------------------------------------------------------------
  // Compute regularity and corrected mean frequency
  //--------------------------------------------------------------------
  void computeRegularityOutput() {
    if (!M0.isReady()) {
      R_out.update(0.0f, alpha_out);
      R_spec = 0.0f;
      nu = 0.0f;
      return;
    }

    const float m0 = M0.get(), m1 = M1.get(), m2 = M2.get();
    if (!(m0 > EPSILON)) { R_out.update(0.0f, alpha_out); return; }

    // Naive means
    omega_bar_naive = m1 / m0;
    const float omega2_bar_naive = m2 / m0;

    // Jensen bias correction
    const float q00 = Q00.get(), q10 = Q10.get(), q20 = Q20.get();
    const float varM0 = std::max(0.0f, q00 - m0 * m0);
    const float cov10 = q10 - m1 * m0;
    const float cov20 = q20 - m2 * m0;
    const float invM0_2 = 1.0f / std::max(m0 * m0, EPSILON);

    omega_bar_corr = omega_bar_naive + (omega_bar_naive * varM0 - cov10) * invM0_2;
    const float omega2_bar_corr = omega2_bar_naive + (omega2_bar_naive * varM0 - cov20) * invM0_2;

    // Spectral narrowness ν
    const float mu2_corr = std::max(0.0f, omega2_bar_corr - omega_bar_corr * omega_bar_corr);
    nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2_corr) / omega_bar_corr) : 0.0f;
    if (!std::isfinite(nu) || nu < 0.0f) nu = 0.0f;

    // Regularity score (0–1)
    R_spec = std::clamp(std::exp(-BETA_SPEC * nu), 0.0f, 1.0f);
    R_out.update(R_spec, alpha_out);
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
