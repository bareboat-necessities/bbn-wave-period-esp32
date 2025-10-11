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
      a_z(t) = d^2 eta/dt^2 = -omega^2 eta(t)
      => S_a(omega) = omega^4 S_eta(omega)
         S_eta(omega) = S_a(omega) / omega^4

    Spectral moments (displacement spectrum)
      M0 = integral S_eta(omega) d omega        — variance of displacement eta
      M1 = integral omega S_eta(omega) d omega
      M2 = integral omega^2 S_eta(omega) d omega

    Derived quantities
      • Mean angular frequency      omega_bar = M1 / M0
      • Spectral narrowness (nu)    nu = sqrt(M2/M0 - (M1/M0)^2) / (M1/M0)
      • Oceanographic Hs            Hs = 4*sqrt(M0)

    Jensen bias correction for ratios
      omega_bar_corr = omega_bar_naive + (omega_bar_naive Var[M0] - Cov[M1,M0]) / M0^2
      and similarly for M2/M0 in the nu computation.

    Implementation overview
      • Multi-bin demodulation of acceleration at log-spaced omega_k around omega_inst.
      • Each bin uses a 1st-order LPF with cutoff f_c,k (Hz) and ENBW derived from the discrete α.
      • Convert acceleration baseband power to displacement PSD via omega_k^-4 (with a small ω floor).
      • Accumulate {M0,M1,M2} and Jensen correction helpers each step.
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
  constexpr static float K_EFF_MIX     = 2.0f;
  constexpr static float OMEGA_MIN_HZ  = 0.03f;
  constexpr static float OMEGA_MAX_HZ  = 4.00f;

  // Nested Spectrum struct (grid + demod state)
  struct Spectrum {
    bool  ready        = false;
    float omega_center = 0.0f;

    float omega[NBINS]{};
    float domega[NBINS]{};

    float inv_w4[NBINS]{};

    float alpha_k[NBINS]{};
    float cos_dphi[NBINS]{};
    float sin_dphi[NBINS]{};

    float c[NBINS]{};
    float s[NBINS]{};
    float zr[NBINS]{};
    float zi[NBINS]{};

    float S_eta_rad[NBINS]{};

    inline void clear() { ready = false; }

    // Local clamp (nested classes are NOT friends by default)
    static inline float clampf_(float x, float lo, float hi) {
      return (x < lo) ? lo : ((x > hi) ? hi : x);
    }

    // Cosine-tapered geometric spacing (denser near center)
    inline void buildGrid(float omega_ctr, float omega_min, float omega_max) {
      constexpr float TARGET_SPAN_UP = 6.0f;
      omega_center = clampf_(omega_ctr, omega_min, omega_max);

      // precompute tapers and their sum
      float taper[MAX_K + 1];
      float sum_w = 0.0f;
      taper[0] = 0.0f;
      for (int k = 1; k <= MAX_K; ++k) {
        float t = 0.5f * (1.0f + std::cos((float(k) / float(MAX_K)) * PI_));
        taper[k] = t;
        sum_w += t;
      }

      float span_up = std::min(TARGET_SPAN_UP, omega_max / std::max(omega_center, 1e-20f));
      float span_dn = std::min(TARGET_SPAN_UP, std::max(omega_center, 1e-20f) / std::max(omega_min, 1e-20f));

      float base_up = (span_up  > 1.0f && sum_w > 0.0f) ? std::exp(std::log(span_up) / sum_w) : 1.0f;
      float base_dn = (span_dn  > 1.0f && sum_w > 0.0f) ? std::exp(std::log(span_dn) / sum_w) : 1.0f;

      omega[MAX_K] = omega_center;

      float w_up = omega_center;
      for (int k = 1; k <= MAX_K; ++k) {
        w_up *= std::pow(base_up, taper[k]);
        omega[MAX_K + k] = clampf_(w_up, omega_min, omega_max);
      }

      float w_dn = omega_center;
      for (int k = 1; k <= MAX_K; ++k) {
        w_dn /= std::pow(base_dn, taper[k]);
        omega[MAX_K - k] = clampf_(w_dn, omega_min, omega_max);
      }

      // Voronoi half-widths (rad/s) and stabilized ω^-4
      // Add a tiny ω floor inside ω^4 to prevent blow-ups at very low ω.
      constexpr float W_FLOOR = 2e-2f; // rad/s (tiny; << typical ω)
      for (int i = 0; i < NBINS; ++i) {
        const float w  = omega[i];
        const float wL = (i > 0)         ? omega[i - 1] : w;
        const float wR = (i < NBINS - 1) ? omega[i + 1] : w;

        float dW = 0.5f * (wR - wL);
        // adaptive floor: prevent tiny ω bins from dominating
        const float dW_min_rel = 0.01f * std::max(w, 0.0f); // 0.25% of ω
        const float dW_min_abs = 1e-5f;
        if (dW < std::max(dW_min_abs, dW_min_rel))
          dW = std::max(dW_min_abs, dW_min_rel);
        domega[i] = dW;

        const float w2 = w * w;
        const float w4 = (w2 + W_FLOOR * W_FLOOR) * (w2 + W_FLOOR * W_FLOOR);
        inv_w4[i] = (w4 > 0.0f) ? 1.0f / w4 : 0.0f;
      }

      if (!ready) {
        for (int i = 0; i < NBINS; ++i) { c[i] = 1.0f; s[i] = 0.0f; zr[i] = zi[i] = 0.0f; }
        ready = true;
      }
    }

    // LPF alphas and rotator steps — (ENBW = pi^2 fc)
    inline void precomputeForDt(float dt) {
        for (int i = 0; i < NBINS; ++i) {
            const float fc_hz = domega[i] / (PI_ * PI_);  // fc = ENBW/pi^2 (analog-calibrated)
            float a = 1.0f - std::exp(-dt * TWO_PI_ * fc_hz);
            if (a < 0.0f) a = 0.0f; else if (a > 1.0f) a = 1.0f;
            alpha_k[i] = a;

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

    inline float integrateMoment(int n) const {
      if (!ready) return 0.0f;
      double acc = 0.0;
      for (int i = 0; i < NBINS; ++i) {
        // Use full bin width = wR - wL (energy-conserving)
        const float w   = omega[i];
        const float wL  = (i > 0)         ? omega[i - 1] : w;
        const float wR  = (i < NBINS - 1) ? omega[i + 1] : w;
        const double width = double(wR) - double(wL);
        acc += std::pow(double(w), n) * double(S_eta_rad[i]) * width;
      }
      return float(acc);
    }
  };

  // Fixed-grid online averaged spectrum (absolute Hz grid)
  struct FixedGridAvg {
    static constexpr int N_BINS = 80;
    static constexpr float FMIN_HZ = 0.01f;
    static constexpr float FMAX_HZ = 4.0f;

    // persistent state
    float freq_hz[N_BINS];
    float domega[N_BINS];     // bin half-widths (rad/s)
    float S_avg[N_BINS];      // exponentially averaged PSD (rad/s)
    float weight[N_BINS];
    bool  initialized = false;

    // time constant for averaging (seconds)
    float tau_spec = 120.0f;

    // grid setup / housekeeping
    inline void reset() {
      initialized = false;
      for (int i = 0; i < N_BINS; ++i) {
        freq_hz[i] = domega[i] = 0.0f;
        S_avg[i] = weight[i] = 0.0f;
      }
    }

    inline void buildGrid() {
      float ratio = std::exp(std::log(FMAX_HZ / FMIN_HZ) / (N_BINS - 1));
      freq_hz[0] = FMIN_HZ;
      for (int i = 1; i < N_BINS; ++i)
        freq_hz[i] = freq_hz[i - 1] * ratio;
      for (int i = 0; i < N_BINS; ++i) {
        float fL = (i > 0) ? freq_hz[i - 1] : freq_hz[i];
        float fR = (i < N_BINS - 1) ? freq_hz[i + 1] : freq_hz[i];
        domega[i] = 2.0f * float(M_PI) * 0.5f * (fR - fL);  // Δω half-width
      }
      initialized = true;
    }

    // helpers
    inline static float clampf(float x, float lo, float hi) {
      return (x < lo) ? lo : ((x > hi) ? hi : x);
    }

    // main accumulation (energy-conserving rebinning with full widths)
    template <int NK>
    inline void accumulate(const typename SeaStateRegularity<NK>::Spectrum& S, float dt_s) {
      if (!initialized) buildGrid();
      if (!(tau_spec > 1e-3f)) return;

      const float alpha = 1.0f - std::exp(-dt_s / tau_spec);

      for (int i = 0; i < SeaStateRegularity<NK>::NBINS; ++i) {
        const float Srad = S.S_eta_rad[i];
        if (!(Srad > 0.0f)) continue;

        const float w_c  = S.omega[i];
        const float dw_c = S.domega[i];
        const float wL_i = w_c - dw_c;
        const float wR_i = w_c + dw_c;

        // --- Correct source bin full width and energy ---
        const float width_i = std::max(wR_i - wL_i, 1e-12f); // full width
        const float E_src   = Srad * width_i;                 // total energy in source bin

        for (int j = 0; j < N_BINS; ++j) {
          const float f_c   = freq_hz[j];
          const float dw_j  = domega[j];
          const float w_c_j = TWO_PI_ * f_c;
          const float wL_j  = w_c_j - dw_j;
          const float wR_j  = w_c_j + dw_j;

          const float overlap = std::max(0.0f,
              std::min(wR_i, wR_j) - std::max(wL_i, wL_j));
          if (overlap <= 0.0f) continue;

          // Fraction of source energy going into target bin j
          const float frac    = overlap / width_i;
          const float E_part  = E_src * frac;

          // Convert overlapped energy back to PSD using TARGET full width
          const float width_j = std::max(wR_j - wL_j, 1e-12f);
          const float S_part  = E_part / width_j;

          S_avg[j]  = (1.0f - alpha) * S_avg[j]  + alpha * S_part;
          weight[j] = (1.0f - alpha) * weight[j] + alpha;
        }
      }
    }

    // accessors
    inline float valueRad(int k) const {
      if (k < 0 || k >= N_BINS) return 0.0f;
      return (weight[k] > 1e-6f) ? S_avg[k] / weight[k] : 0.0f;
    }

    inline float valueHz(int k) const {
      // convert m²/(rad/s) → m²/Hz by multiplying 2π
      return valueRad(k) * (2.0f * float(M_PI));
    }

    inline int size() const { return N_BINS; }
  };

  // Constructor / Reset
  explicit SeaStateRegularity(float tau_mom_sec = 180.0f,
                              float tau_a_mom_sec = 90.0f,
                              float tau_out_sec = 30.0f,
                              float tau_w_sec   = 30.0f)
  : tau_mom(tau_mom_sec), tau_a_mom(tau_a_mom_sec),
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
    last_bins_dt = -1.0f;
    alpha_mom = alpha_a_mom = alpha_out = alpha_w = 0.0f;

    has_moments = false;
    spectrum_.clear();
    fixed_avg_.reset();
    grid_valid = false;
  }

  // Main update
  inline void update(float dt_s, float accel_z, float omega_inst) {
    if (!(dt_s > 0.0f) || !std::isfinite(accel_z) || !std::isfinite(omega_inst)) return;

    last_accel = accel_z;
    updateGlobalAlphas(dt_s);

    // running accel stats
    A1_mean.update(accel_z, alpha_a_mom);
    A2_second.update(accel_z * accel_z, alpha_a_mom);
    const float a_mean = A1_mean.get();
    const float a_var  = A2_second.get() - a_mean * a_mean;
    A0.update((a_var > 0.0f) ? a_var : 0.0f, alpha_a_mom);

    // smoothed center omega
    const float w_obs = omega_inst;
    omega_used = (omega_used <= 0.0f) ? w_obs : (1.0f - alpha_w) * omega_used + alpha_w * w_obs;
    const float a_demean = accel_z - a_mean;

    // Handle update on large ω jumps 
    if (omega_used > 0.0f) {
      const float ratio = w_obs / omega_used;
      if (ratio < 0.67f || ratio > 1.50f) {
        omega_used = w_obs;
        spectrum_.clear();
        spectrum_.buildGrid(omega_used, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
        spectrum_.precomputeForDt(dt_s);
      }
    }
      
    spectrum_.buildGrid(omega_used, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
    spectrum_.precomputeForDt(dt_s);
    grid_valid = true;
    last_bins_dt = dt_s;

    // hot loop: rotator + 1st-order IIR using precomputed constants
    double S0 = 0.0, S1 = 0.0, S2 = 0.0;
    for (int i = 0; i < NBINS; ++i) {
      // rotate (c,s) by dphi
      const float c_next = spectrum_.c[i] * spectrum_.cos_dphi[i] - spectrum_.s[i] * spectrum_.sin_dphi[i];
      const float s_next = spectrum_.s[i] * spectrum_.cos_dphi[i] + spectrum_.c[i] * spectrum_.sin_dphi[i];
      spectrum_.c[i] = c_next;
      spectrum_.s[i] = s_next;

      // keep rotator on unit circle
      float nrm = std::sqrt(spectrum_.c[i]*spectrum_.c[i] + spectrum_.s[i]*spectrum_.s[i]);
      if (nrm > 0.0f) {
        spectrum_.c[i] /= nrm;
        spectrum_.s[i] /= nrm;
      }

      const float y_r = a_demean * spectrum_.c[i];
      const float y_i = -a_demean * spectrum_.s[i];

      // 1st-order IIR with alpha_k
      const float a = spectrum_.alpha_k[i];
      spectrum_.zr[i] = (1.0f - a) * spectrum_.zr[i] + a * y_r;
      spectrum_.zi[i] = (1.0f - a) * spectrum_.zi[i] + a * y_i;

      // convert to S_eta via omega^-4 and ENBW-like compensation using Δω (half-width)
      const float P_acc  = spectrum_.zr[i] * spectrum_.zr[i] + spectrum_.zi[i] * spectrum_.zi[i];
      const float P_disp = P_acc * spectrum_.inv_w4[i];

      // Correct energy normalization: use actual full bin width = 2 * Δω for internal bins
      const float dw = spectrum_.domega[i];
      const float width = (i == 0 || i == NBINS - 1) ? dw : (2.0f * dw);
      float S_hat = K_EFF_MIX * P_disp / std::max(width, 1e-12f);

      spectrum_.S_eta_rad[i] = S_hat;

      const float w = spectrum_.omega[i];

      // width contribution to moments (use the same 'width' for consistency)
      S0 += double(S_hat) * double(width);
      S1 += double(S_hat) * double(w)  * double(width);
      S2 += double(S_hat) * double(w)  * double(w) * double(width); 
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

    // accumulate into fixed-grid averaged spectrum
    fixed_avg_.template accumulate<MAX_K>(spectrum_, dt_s); 

    computeRegularityOutput();
  }

  // Public Getters
  inline float getRegularity()         const { return R_out.get(); }
  inline float getRegularitySpectral() const { return R_spec; }
  inline float getNarrowness()         const { return nu; }

  inline float getDisplacementFrequencyHz() const {
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

  inline const Spectrum& getSpectrum() const { return spectrum_; }
  inline bool spectrumReady() const { return spectrum_.ready; }
  inline const FixedGridAvg& getAveragedSpectrum() const { return fixed_avg_; }

private:
  // Internal constants
  constexpr static float OMEGA_MIN_RAD = TWO_PI_ * OMEGA_MIN_HZ;
  constexpr static float OMEGA_MAX_RAD = TWO_PI_ * OMEGA_MAX_HZ;

  // State variables
  float tau_mom = 180.0f, tau_a_mom = 60.0f, tau_out = 60.0f, tau_w = 30.0f;
  float last_dt = -1.0f;
  float last_bins_dt = -1.0f;
  float alpha_mom = 0.0f, alpha_a_mom = 0.0f, alpha_out = 0.0f, alpha_w = 0.0f;
  float omega_used = 0.0f, last_accel = 0.0f;
  bool  has_moments = false, grid_valid = false;

  DebiasedEMA M0, M1, M2;
  DebiasedEMA A0, A1_mean, A2_second;
  DebiasedEMA Q00, Q10, Q20;
  DebiasedEMA R_out;

  float R_spec = 0.0f, nu = 0.0f;
  float omega_bar_corr = 0.0f, omega_bar_naive = 0.0f;

  struct Spectrum spectrum_;
  struct FixedGridAvg fixed_avg_;

  // Internal helpers
  inline void updateGlobalAlphas(float dt_s) {
    if (dt_s == last_dt) return;
    last_dt   = dt_s;
    alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
    alpha_a_mom = 1.0f - std::exp(-dt_s / tau_a_mom);
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

    // narrowness nu from corrected central moment mu2
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

  float Hs_est = 0.0f, nu_val = 0.0f;
  float f_disp_corr = 0.0f, f_disp_naive = 0.0f, Tp = 0.0f, R_spec_val = 0.0f;

  for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
    auto za = wave.step(DT);
    float a = za.second;
    reg.update(DT, a, wave.omega);

    R_spec_val   = reg.getRegularitySpectral();
    Hs_est       = reg.getWaveHeightEnvelopeEst();
    nu_val       = reg.getNarrowness();
    f_disp_corr  = reg.getDisplacementFrequencyHz();
    f_disp_naive = reg.getDisplacementFrequencyNaiveHz();
    Tp           = reg.getDisplacementPeriodSec();
  }

  const float Hs_expected = 4.0f * SINE_AMPLITUDE;

  if (!(R_spec_val > 0.85f))
    throw std::runtime_error("Sine: R_spec (moment-based) should be high for a narrowband tone.");
  if (!(std::fabs(Hs_est - Hs_expected) < 0.30f * Hs_expected))
    throw std::runtime_error("Sine: Hs estimate not within tolerance.");
  if (!(nu_val < 0.10f))
    throw std::runtime_error("Sine: Narrowness should be small for a pure tone.");

  std::cerr << "[PASS] Sine wave test (moment-only) — "
            << "Hs_est=" << Hs_est
            << " (~" << Hs_expected << "), nu=" << nu_val
            << ", f_disp_corr=" << f_disp_corr << " Hz"
            << ", f_disp_naive=" << f_disp_naive << " Hz"
            << ", Tp=" << Tp << " s, R_spec=" << R_spec_val << "\n";

  if (reg.spectrumReady()) {
    const auto& S = reg.getSpectrum();
    float m0_snap = S.integrateMoment(0);
    (void)m0_snap;
  }
}
#endif

