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
      Estimate ocean wave spectral properties directly from vertical
      acceleration a_z(t). Everything here is moment-based.

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

    Jensen bias correction for ratios
      ω̄_corr = ω̄_naive + (ω̄_naive Var[M₀] − Cov[M₁,M₀]) / M₀²
      and similarly for M₂/M₀ in the ν computation.

    Implementation overview
      • Multi-bin demodulation of acceleration at log-spaced ω_k around ω_inst.
      • Each bin uses a 1st-order LPF with cutoff f_c,k (Hz) and ENBW = π² f_c,k (rad/s).
      • Convert acceleration baseband power to displacement PSD via ω_k⁻⁴.
      • Accumulate {M₀,M₁,M₂} and Jensen correction helpers each step.
*/

struct DebiasedEMA {
  float value  = 0.0f;
  float weight = 0.0f;

  void reset() { value = 0.0f; weight = 0.0f; }
  inline void update(float x, float alpha) {
    value  = (1.0f - alpha) * value + alpha * x;
    weight = (1.0f - alpha) * weight + alpha;
  }
  inline float get() const { return (weight > 1e-12f) ? value / weight : 0.0f; }
  inline bool  isReady() const { return weight > 1e-6f; }
};

template <int MAX_K_ = 25>
class SeaStateRegularity {
public:
  // ---------------------------------------------
  // Constants
  // ---------------------------------------------
  constexpr static float EPSILON    = 1e-12f;
  constexpr static float BETA_SPEC  = 1.5f;
  constexpr static float K_EFF_MIX  = 1.6f;
  constexpr static float OMEGA_MIN_HZ = 0.02f;
  constexpr static float OMEGA_MAX_HZ = 4.00f;
  constexpr static float TAU_W_SEC    = 30.0f;
  constexpr static int   MAX_K     = MAX_K_;
  constexpr static int   NBINS     = 2 * MAX_K + 1;

  // ---------------------------------------------
  // Nested Spectrum struct (encapsulates grid + PSD)
  // ---------------------------------------------
  struct Spectrum {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr int MAX_BINS = NBINS;

    bool  ready = false;
    float omega_center = 0.0f;
    float ratio_r = 1.0f;
    float omega[MAX_BINS]{};
    float domega[MAX_BINS]{};
    float S_eta_rad[MAX_BINS]{};
    float freq[MAX_BINS]{};
    float df[MAX_BINS]{};
    float S_eta_hz[MAX_BINS]{};

    void clear() { ready = false; }

    // Build a symmetric log grid around ω_center
    void buildGrid(float omega_used, float omega_min, float omega_max) {
      constexpr float TARGET_SPAN_UP = 6.0f;
      const int K = MAX_K;
      ratio_r = std::exp(std::log(TARGET_SPAN_UP) / float(K));
      omega_center = omega_used;

      omega[MAX_K] = omega_used;
      for (int k = 1; k <= K; ++k) {
        omega[MAX_K + k] = omega[MAX_K + k - 1] * ratio_r;
        omega[MAX_K - k] = omega[MAX_K - k + 1] / ratio_r;
      }
      for (int i = 0; i < NBINS; ++i)
        omega[i] = std::clamp(omega[i], omega_min, omega_max);

      for (int i = 0; i < NBINS; ++i) {
        const float wL = (i > 0) ? omega[i - 1] : omega[i];
        const float wR = (i < NBINS - 1) ? omega[i + 1] : omega[i];
        domega[i] = std::max(1e-12f, 0.5f * (wR - wL));
        freq[i]   = omega[i] / (2.0f * PI);
        df[i]     = domega[i] / (2.0f * PI);
      }
    }

    // Compute instantaneous m_n = ∑ ωⁿ Sη(ω) Δω
    float integrateMoment(int n) const {
      if (!ready) return 0.0f;
      double acc = 0.0;
      for (int i = 0; i < MAX_BINS; ++i)
        acc += std::pow(double(omega[i]), n) * double(S_eta_rad[i]) * double(domega[i]);
      return float(acc);
    }

    // Convenience: integrate M₀
    float integrateM0() const { return integrateMoment(0); }
  };

  // ---------------------------------------------
  // Constructor / Reset
  // ---------------------------------------------
  SeaStateRegularity(float tau_mom_sec = 180.0f, float tau_out_sec = 60.0f)
  {
    tau_mom = tau_mom_sec;
    tau_out = std::max(1e-3f, tau_out_sec);
    reset();
  }

  void reset() {
    M0.reset(); M1.reset(); M2.reset();
    A0.reset(); A1_mean.reset(); A2_second.reset();
    Q00.reset(); Q10.reset(); Q20.reset();
    R_out.reset();
    R_spec = 0.0f; nu = 0.0f;
    omega_bar_corr = omega_bar_naive = 0.0f;
    omega_used = 0.0f; alpha_w = 0.0f;
    last_dt = -1.0f; alpha_mom = alpha_out = 0.0f;
    has_moments = false; last_accel = 0.0f;
    for (int i = 0; i < NBINS; ++i) { phi_k[i] = 0.0f; zr[i] = zi[i] = 0.0; }
    spectrum_.clear();
  }

  // ---------------------------------------------
  // Main update
  // ---------------------------------------------
  void update(float dt_s, float accel_z, float omega_inst)
  {
    if (!(dt_s > 0.0f)) return;
    if (!std::isfinite(accel_z) || !std::isfinite(omega_inst)) return;

    last_accel = accel_z;
    updateAlpha(dt_s);
    A1_mean.update(accel_z, alpha_mom);
    A2_second.update(accel_z * accel_z, alpha_mom);
    const float a_mean = A1_mean.get();
    const float a_var  = std::max(0.0f, A2_second.get() - a_mean * a_mean);
    A0.update(a_var, alpha_mom);

    updateSpectralMoments(omega_inst);
    computeRegularityOutput();
  }

  // ---------------------------------------------
  // Public Getters
  // ---------------------------------------------
  float getRegularity() const         { return R_out.get(); }
  float getRegularitySpectral() const { return R_spec; }
  float getNarrowness() const         { return nu; }

  float getDisplacementFrequencyHz() const {
    return getDisplacementFrequencyM01Hz();
  }

  float getDisplacementFrequencyM01Hz() const {
    const float m0 = M0.get(), m1 = M1.get();
    if (!(m0 > EPSILON)) return 0.0f;
    const float q00 = Q00.get(), q10 = Q10.get();
    const float varM0 = std::max(0.0f, q00 - m0 * m0);
    const float cov10 = q10 - m1 * m0;
    const float invM0_2 = 1.0f / std::max(m0 * m0, EPSILON);
    const float omega_bar_corr = (m1 / m0) + ((m1 / m0) * varM0 - cov10) * invM0_2;
    return (omega_bar_corr > 0.0f) ? (omega_bar_corr / (2.0f * Spectrum::PI)) : 0.0f;
  }

  float getDisplacementPeriodSec() const {
    const float fz = getDisplacementFrequencyHz();
    return (fz > EPSILON) ? (1.0f / fz) : 0.0f;
  }

  float getWaveHeightEnvelopeEst() const {
    const float m0 = M0.get();
    return (m0 > EPSILON) ? 4.0f * std::sqrt(m0) : 0.0f;
  }

  float getAccelerationVariance() const { return A0.get(); }

  // New: Access spectrum object
  const Spectrum& getSpectrum() const { return spectrum_; }
  bool spectrumReady() const { return spectrum_.ready; }

private:
  // ---------------------------------------------
  // Constants
  // ---------------------------------------------
  static constexpr float PI = Spectrum::PI;
  static constexpr float TWO_PI_ = 2.0f * PI;
  static constexpr float OMEGA_MIN_RAD = TWO_PI_ * OMEGA_MIN_HZ;
  static constexpr float OMEGA_MAX_RAD = TWO_PI_ * OMEGA_MAX_HZ;

  // ---------------------------------------------
  // State variables
  // ---------------------------------------------
  float tau_mom, tau_out;
  float last_dt = -1.0f;
  float alpha_mom = 0.0f, alpha_out = 0.0f, alpha_w = 0.0f;
  float omega_used = 0.0f, last_accel = 0.0f;
  bool  has_moments = false;

  float  phi_k[NBINS];
  double zr[NBINS], zi[NBINS];

  DebiasedEMA M0, M1, M2;
  DebiasedEMA A0, A1_mean, A2_second;
  DebiasedEMA Q00, Q10, Q20;
  DebiasedEMA R_out;

  float R_spec = 0.0f, nu = 0.0f;
  float omega_bar_corr = 0.0f, omega_bar_naive = 0.0f;

  Spectrum spectrum_;

  // ---------------------------------------------
  // Internal helpers
  // ---------------------------------------------
  void updateAlpha(float dt_s) {
    if (dt_s == last_dt) return;
    alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
    alpha_out = 1.0f - std::exp(-dt_s / tau_out);
    alpha_w   = 1.0f - std::exp(-dt_s / TAU_W_SEC);
    last_dt   = dt_s;
  }

  void updateSpectralMoments(float omega_inst)
  {
    const float w_obs = std::clamp(omega_inst, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
    if (omega_used <= 0.0f) omega_used = w_obs;
    else omega_used = (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

    const float ratio = (omega_used > 0.0f) ? w_obs / omega_used : 1.0f;
    if (ratio < 0.7f || ratio > 1.3f) return;

    spectrum_.buildGrid(omega_used, OMEGA_MIN_RAD, OMEGA_MAX_RAD);

    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;

    for (int i = 0; i < NBINS; ++i) {
      const float wk = spectrum_.omega[i];
      const float enbw  = spectrum_.domega[i];
      const float fc_hz = enbw / (PI * PI);
      const float alpha = 1.0f - std::exp(-last_dt * TWO_PI_ * fc_hz);

      phi_k[i] += wk * last_dt;
      phi_k[i] = std::fmod(phi_k[i], 2.0f * PI);
      if (phi_k[i] >  PI) phi_k[i] -= 2.0f * PI;
      if (phi_k[i] < -PI) phi_k[i] += 2.0f * PI;

      const float c = std::cos(phi_k[i]);
      const float s = std::sin(phi_k[i]);
      const float y_r = last_accel * c;
      const float y_i = -last_accel * s;

      zr[i] = (1.0 - alpha) * zr[i] + alpha * y_r;
      zi[i] = (1.0 - alpha) * zi[i] + alpha * y_i;

      const float P_acc = float(zr[i]*zr[i] + zi[i]*zi[i]);
      const float inv_w4 = 1.0f / std::max(wk*wk*wk*wk, EPSILON);
      const float P_disp = P_acc * inv_w4;
      const float S_eta_hat = K_EFF_MIX * P_disp / std::max(enbw, EPSILON);

      spectrum_.S_eta_rad[i] = S_eta_hat;
      spectrum_.S_eta_hz[i]  = S_eta_hat * (2.0f * PI);

      const float domega = spectrum_.domega[i];
      S0 += S_eta_hat * domega;
      S1 += S_eta_hat * wk * domega;
      S2 += S_eta_hat * wk * wk * domega;
    }

    M0.update(S0, alpha_mom);
    M1.update(S1, alpha_mom);
    M2.update(S2, alpha_mom);
    Q00.update(S0 * S0, alpha_mom);
    Q10.update(S0 * S1, alpha_mom);
    Q20.update(S0 * S2, alpha_mom);

    spectrum_.ready = true;
    has_moments = true;
  }

  void computeRegularityOutput() {
    if (!M0.isReady()) { R_out.update(0.0f, alpha_out); R_spec = 0.0f; nu = 0.0f; return; }
    const float m0 = M0.get(), m1 = M1.get(), m2 = M2.get();
    if (!(m0 > EPSILON)) { R_out.update(0.0f, alpha_out); return; }

    omega_bar_naive = m1 / m0;
    const float omega2_bar_naive = m2 / m0;
    const float q00 = Q00.get(), q10 = Q10.get(), q20 = Q20.get();
    const float varM0 = std::max(0.0f, q00 - m0 * m0);
    const float cov10 = q10 - m1 * m0;
    const float cov20 = q20 - m2 * m0;
    const float invM0_2 = 1.0f / std::max(m0 * m0, EPSILON);

    omega_bar_corr = omega_bar_naive + (omega_bar_naive * varM0 - cov10) * invM0_2;
    const float omega2_bar_corr = omega2_bar_naive + (omega2_bar_naive * varM0 - cov20) * invM0_2;

    const float mu2_corr = std::max(0.0f, omega2_bar_corr - omega_bar_corr * omega_bar_corr);
    nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2_corr) / omega_bar_corr) : 0.0f;
    if (!std::isfinite(nu) || nu < 0.0f) nu = 0.0f;

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

  float Hs_est = 0.0f, nu = 0.0f;
  float f_disp_corr = 0.0f, f_disp_naive = 0.0f, Tp = 0.0f, R_spec = 0.0f;

  for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
    auto za = wave.step(DT);
    float a = za.second;
    reg.update(DT, a, wave.omega);

    R_spec       = reg.getRegularitySpectral();
    Hs_est       = reg.getWaveHeightEnvelopeEst();
    nu           = reg.getNarrowness();
    f_disp_corr  = reg.getDisplacementFrequencyHz();
    f_disp_naive = reg.getDisplacementFrequencyNaiveHz();
    Tp           = reg.getDisplacementPeriodSec();
  }

  const float Hs_expected = 4.0f * SINE_AMPLITUDE;

  if (!(R_spec > 0.85f))
    throw std::runtime_error("Sine: R_spec (moment-based) should be high for a narrowband tone.");
  if (!(std::fabs(Hs_est - Hs_expected) < 0.30f * Hs_expected))
    throw std::runtime_error("Sine: Hs estimate not within tolerance.");
  if (!(nu < 0.10f))
    throw std::runtime_error("Sine: Narrowness should be small for a pure tone.");

  std::cerr << "[PASS] Sine wave test (moment-only) — "
            << "Hs_est=" << Hs_est
            << " (~" << Hs_expected << "), ν=" << nu
            << ", f_disp_corr=" << f_disp_corr << " Hz"
            << ", f_disp_naive=" << f_disp_naive << " Hz"
            << ", Tp=" << Tp << " s, R_spec=" << R_spec << "\n";

  // Example of using the spectrum snapshot (optional)
  if (reg.spectrumReady()) {
    const auto& S = reg.getSpectrum();
    float m0_snap = S.integrateM0();
    (void)m0_snap; // for debugging/validation if needed
  }
}
#endif
