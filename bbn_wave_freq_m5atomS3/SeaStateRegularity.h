#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

/*
    Copyright 2025, Mikhail Grushinskiy

    SeaStateRegularity — Online estimator of ocean wave regularity

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

    Embedded optimizations (hot path)
      • Rotator recurrence per bin (no per-sample sin/cos/fmod).
      • Per-bin α_k, 1/ENBW_k and 1/ω_k⁴ precomputed at grid build.
      • Float state with double accumulators only for moment sums.
      • Hysteresis on grid rebuild to avoid thrashing.
*/

struct DebiasedEMA {
  float value  = 0.0f;
  float weight = 0.0f;
  inline void reset() { value = 0.0f; weight = 0.0f; }
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
  // Constants
  constexpr static int   MAX_K   = MAX_K_;
  constexpr static int   NBINS   = 2 * MAX_K + 1;
  constexpr static float EPSILON = 1e-12f;
  constexpr static float PI_     = 3.14159265358979323846f;
  constexpr static float TWO_PI_ = 2.0f * PI_;

  // Behavior knobs
  constexpr static float BETA_SPEC     = 1.5f;
  constexpr static float K_EFF_MIX     = 1.6f;
  constexpr static float OMEGA_MIN_HZ  = 0.02f;
  constexpr static float OMEGA_MAX_HZ  = 4.00f;

  // Nested Spectrum struct (encapsulates grid + PSD)
  struct Spectrum {
    // Grid parameters
    bool  ready        = false;
    float omega_center = 0.0f;
    float ratio_r      = 1.0f;

    // Grid arrays (structure-of-arrays, cache friendly)
    float omega[NBINS]{};
    float domega[NBINS]{};

    // Per-bin precomputes from grid
    float inv_w4[NBINS]{};     // 1 / ω_k^4
    float inv_enbw[NBINS]{};   // 1 / ENBW_k  (rad/s)

    // Per-sample precomputes (depend on dt)
    float alpha_k[NBINS]{};    // LPF α per bin
    float cos_dphi[NBINS]{};
    float sin_dphi[NBINS]{};

    // Rotator state (cos,sin) and IIR state
    float c[NBINS]{};
    float s[NBINS]{};
    float zr[NBINS]{};
    float zi[NBINS]{};

    // Live PSD snapshot (for inspection/plotting)
    float S_eta_rad[NBINS]{};
    float S_eta_hz[NBINS]{};

    // API expected by caller
    inline void clear() { ready = false; }

    // Build a symmetric log grid around ω_center; compute Δω, inv_w4, inv_enbw.
    inline void buildGrid(float omega_ctr, float omega_min, float omega_max) {
      constexpr float TARGET_SPAN_UP = 6.0f;                   // multiplicative span upward
      omega_center = omega_ctr;
      ratio_r = std::exp(std::log(TARGET_SPAN_UP) / float(MAX_K));

      omega[MAX_K] = omega_center;
      for (int k = 1; k <= MAX_K; ++k) {
        omega[MAX_K + k] = omega[MAX_K + k - 1] * ratio_r;
        omega[MAX_K - k] = omega[MAX_K - k + 1] / ratio_r;
      }
      // clamp to bounds
      for (int i = 0; i < NBINS; ++i) {
        float w = omega[i];
        if (w < omega_min) w = omega_min;
        if (w > omega_max) w = omega_max;
        omega[i] = w;
      }
      // Δω and per-bin invariants from the grid
      for (int i = 0; i < NBINS; ++i) {
        const float wL = (i > 0) ? omega[i - 1] : omega[i];
        const float wR = (i < NBINS - 1) ? omega[i + 1] : omega[i];
        const float domeg = 0.5f * (wR - wL);
        domega[i]    = (domeg > 1e-12f) ? domeg : 1e-12f;

        const float w  = omega[i];
        const float w2 = w * w;
        inv_w4[i]   = (w2 > 0.0f) ? 1.0f / (w2 * w2) : 0.0f;

        // ENBW (rad/s) ~ Δω; store inverse for hot loop
        inv_enbw[i] = 1.0f / domega[i];
      }
      // rotator/IIR init stays gentle (keep states if rebuilds are frequent)
      for (int i = 0; i < NBINS; ++i) {
        c[i]  = 1.0f;  // aligned phase seed
        s[i]  = 0.0f;
        zr[i] *= 0.95f;
        zi[i] *= 0.95f;
      }
      ready = true;
    }

    // Prepare per-bin α_k and rotator step for a given dt (call when dt changes or after buildGrid()).
    inline void precomputeForDt(float dt) {
      const float PI_ = 3.14159265358979323846f;
      const float TWO_PI_ = 2.0f * PI_;
      for (int i = 0; i < NBINS; ++i) {
        // LPF alpha per bin: fc = ENBW/π² (Hz); α = 1 − exp(−dt·2π·fc)
        const float fc_hz = domega[i] / (PI_ * PI_);
        float a = 1.0f - std::exp(-dt * TWO_PI_ * fc_hz);
        if (a < 0.0f) a = 0.0f; else if (a > 1.0f) a = 1.0f;
        alpha_k[i] = a;

        // Rotator step Δφ = ω·dt; small-angle fallback
        const float dphi = omega[i] * dt;
        if (std::fabs(dphi) < 1e-3f) {
          cos_dphi[i] = 1.0f - 0.5f * dphi * dphi;
          sin_dphi[i] = dphi;
        } else {
          cos_dphi[i] = std::cos(dphi);
          sin_dphi[i] = std::sin(dphi);
        }
      }
    }

    // Compute instantaneous m_n = ∑ ωⁿ Sη(ω) Δω from the snapshot
    inline float integrateMoment(int n) const {
      if (!ready) return 0.0f;
      double acc = 0.0;
      for (int i = 0; i < NBINS; ++i) {
        acc += std::pow(double(omega[i]), n) * double(S_eta_rad[i]) * double(domega[i]);
      }
      return float(acc);
    }
  };

  // Constructor / Reset
  explicit SeaStateRegularity(float tau_mom_sec = 180.0f,
                              float tau_out_sec = 60.0f,
                              float tau_w_sec   = 30.0f)
  : tau_mom(tau_mom_sec),
    tau_out((tau_out_sec > 1e-3f) ? tau_out_sec : 1e-3f),
    tau_w(tau_w_sec) {
    reset();
  }

  inline void reset() {
    M0.reset(); M1.reset(); M2.reset();
    A0.reset(); A1_mean.reset(); A2_second.reset();
    Q00.reset(); Q10.reset(); Q20.reset();
    R_out.reset();

    R_spec = 0.0f; nu = 0.0f;
    omega_bar_corr = omega_bar_naive = 0.0f;
    omega_used = 0.0f; last_accel = 0.0f;

    last_dt = -1.0f;
    alpha_mom = alpha_out = alpha_w = 0.0f;

    has_moments = false;
    spectrum_.clear();
    grid_valid = false;
  }

  // Main update
  inline void update(float dt_s, float accel_z, float omega_inst) {
    if (!(dt_s > 0.0f) || !std::isfinite(accel_z) || !std::isfinite(omega_inst)) return;

    last_accel = accel_z;
    updateGlobalAlphas(dt_s);

    // running accel stats (kept from your version)
    A1_mean.update(accel_z, alpha_mom);
    A2_second.update(accel_z * accel_z, alpha_mom);
    const float a_mean = A1_mean.get();
    const float a_var  = A2_second.get() - a_mean * a_mean;
    A0.update((a_var > 0.0f) ? a_var : 0.0f, alpha_mom);

    // smoothed center ω
    const float w_obs = clampf(omega_inst, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
    omega_used = (omega_used <= 0.0f) ? w_obs : (1.0f - alpha_w) * omega_used + alpha_w * w_obs;

    // rebuild Spectrum grid if drifted; then prep per-dt constants
    const float ratio = (omega_used > 0.0f) ? (w_obs / omega_used) : 1.0f;
    if (!grid_valid || ratio < 0.995f || ratio > 1.005f) {
      spectrum_.buildGrid(omega_used, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
      grid_valid = true;
    }
    if (dt_s != last_bins_dt) {
      spectrum_.precomputeForDt(dt_s);
      last_bins_dt = dt_s;
    }

    // hot loop: rotator + 1st-order IIR using precomputed constants
    double S0 = 0.0, S1 = 0.0, S2 = 0.0;  // double only for accumulation
    for (int i = 0; i < NBINS; ++i) {
      // rotate (c,s) by Δφ
      const float c_next = spectrum_.c[i] * spectrum_.cos_dphi[i] - spectrum_.s[i] * spectrum_.sin_dphi[i];
      const float s_next = spectrum_.s[i] * spectrum_.cos_dphi[i] + spectrum_.c[i] * spectrum_.sin_dphi[i];
      spectrum_.c[i] = c_next;
      spectrum_.s[i] = s_next;

      const float y_r = last_accel * c_next;
      const float y_i = -last_accel * s_next;

      // 1st-order IIR with α_k
      const float a = spectrum_.alpha_k[i];
      spectrum_.zr[i] = (1.0f - a) * spectrum_.zr[i] + a * y_r;
      spectrum_.zi[i] = (1.0f - a) * spectrum_.zi[i] + a * y_i;

      // convert to S_η via ω⁻⁴ and ENBW compensation
      const float P_acc  = spectrum_.zr[i] * spectrum_.zr[i] + spectrum_.zi[i] * spectrum_.zi[i];
      const float P_disp = P_acc * spectrum_.inv_w4[i];
      const float S_hat  = K_EFF_MIX * P_disp * spectrum_.inv_enbw[i];

      spectrum_.S_eta_rad[i] = S_hat;
      spectrum_.S_eta_hz[i]  = S_hat * (2.0f * PI_);

      const float w  = spectrum_.omega[i];
      const float dw = spectrum_.domega[i];

      S0 += double(S_hat) * double(dw);
      S1 += double(S_hat) * double(w)  * double(dw);
      S2 += double(S_hat) * double(w)  * double(w) * double(dw);
    }

    // moments + Jensen helpers
    M0.update(float(S0), alpha_mom);
    M1.update(float(S1), alpha_mom);
    M2.update(float(S2), alpha_mom);

    Q00.update(float(S0 * S0), alpha_mom);
    Q10.update(float(S0 * S1), alpha_mom);
    Q20.update(float(S0 * S2), alpha_mom);

    spectrum_.ready = true;
    has_moments = true;

    computeRegularityOutput();
  }

  // Public Getters
  inline float getRegularity()         const { return R_out.get(); }
  inline float getRegularitySpectral() const { return R_spec; }
  inline float getNarrowness()         const { return nu; }

  inline float getDisplacementFrequencyHz() const {
    // Jensen-corrected ω̄
    const float m0 = M0.get(), m1 = M1.get();
    if (!(m0 > EPSILON)) return 0.0f;
    const float q00 = Q00.get(), q10 = Q10.get();
    const float varM0  = fmaxf(0.0f, q00 - m0 * m0);
    const float cov10  = q10 - m1 * m0;
    const float invM0_2 = 1.0f / fmaxf(m0 * m0, EPSILON);
    const float wbar_corr = (m1 / m0) + ((m1 / m0) * varM0 - cov10) * invM0_2;
    return (wbar_corr > 0.0f) ? (wbar_corr / (2.0f * PI_)) : 0.0f;
  }

  inline float getDisplacementFrequencyNaiveHz() const {
    // Uncorrected ω̄ = M1/M0 (handy for debugging)
    const float m0 = M0.get(), m1 = M1.get();
    if (!(m0 > EPSILON) || !(m1 > 0.0f)) return 0.0f;
    return (m1 / m0) / (2.0f * PI_);
  }

  inline float getDisplacementPeriodSec() const {
    const float fz = getDisplacementFrequencyHz();
    return (fz > EPSILON) ? (1.0f / fz) : 0.0f;
  }

  inline float getWaveHeightEnvelopeEst() const {
    const float m0 = M0.get();
    return (m0 > EPSILON) ? 4.0f * std::sqrt(m0) : 0.0f;
  }

  inline float getAccelerationVariance() const { return A0.get(); }

  // Spectrum snapshot access
  inline const Spectrum& getSpectrum() const { return spectrum_; }
  inline bool spectrumReady() const { return spectrum_.ready; }

private:
  // Internal constants
  constexpr static float OMEGA_MIN_RAD = TWO_PI_ * OMEGA_MIN_HZ;
  constexpr static float OMEGA_MAX_RAD = TWO_PI_ * OMEGA_MAX_HZ;

  // State variables
  float tau_mom = 180.0f, tau_out = 60.0f, tau_w = 30.0f;
  float last_dt = -1.0f;
  float last_bins_dt = -1.0f;
  float alpha_mom = 0.0f, alpha_out = 0.0f, alpha_w = 0.0f;
  float omega_used = 0.0f, last_accel = 0.0f;
  bool  has_moments = false, grid_valid = false;

  DebiasedEMA M0, M1, M2;
  DebiasedEMA A0, A1_mean, A2_second;
  DebiasedEMA Q00, Q10, Q20;
  DebiasedEMA R_out;

  float R_spec = 0.0f, nu = 0.0f;
  float omega_bar_corr = 0.0f, omega_bar_naive = 0.0f;

  Spectrum spectrum_;

  // Internal helpers
  static inline float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
  }

  inline void updateGlobalAlphas(float dt_s) {
    if (dt_s == last_dt) return;
    last_dt   = dt_s;
    alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
    alpha_out = 1.0f - std::exp(-dt_s / tau_out);
    alpha_w   = 1.0f - std::exp(-dt_s / tau_w);
  }

  inline void computeRegularityOutput() {
    if (!M0.isReady()) {
      R_out.update(0.0f, alpha_out);
      R_spec = 0.0f; nu = 0.0f;
      return;
    }
    const float m0 = M0.get(), m1 = M1.get(), m2 = M2.get();
    if (!(m0 > EPSILON)) { R_out.update(0.0f, alpha_out); return; }

    // naive and corrected frequency moments
    omega_bar_naive = m1 / m0;
    const float omega2_bar_naive = m2 / m0;

    const float q00 = Q00.get(), q10 = Q10.get(), q20 = Q20.get();
    const float varM0  = fmaxf(0.0f, q00 - m0 * m0);
    const float cov10  = q10 - m1 * m0;
    const float cov20  = q20 - m2 * m0;
    const float invM0_2 = 1.0f / fmaxf(m0 * m0, EPSILON);

    omega_bar_corr = omega_bar_naive + (omega_bar_naive * varM0 - cov10) * invM0_2;
    const float omega2_bar_corr = omega2_bar_naive + (omega2_bar_naive * varM0 - cov20) * invM0_2;

    // narrowness ν from corrected central moment μ₂
    const float mu2_corr = fmaxf(0.0f, omega2_bar_corr - omega_bar_corr * omega_bar_corr);
    nu = (omega_bar_corr > EPSILON) ? (std::sqrt(mu2_corr) / omega_bar_corr) : 0.0f;
    if (!std::isfinite(nu) || nu < 0.0f) nu = 0.0f;

    // spectral regularity
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
    float m0_snap = S.integrateMoment(0);
    (void)m0_snap; // for debugging/validation if needed
  }
}
#endif
