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
  constexpr static float BETA_SPEC     = 1.4f;
  constexpr static float K_EFF_MIX     = 2.0f;
  constexpr static float OMEGA_MIN_HZ  = 0.03f;
  constexpr static float OMEGA_MAX_HZ  = 4.00f;

  // Nested Spectrum struct (grid + demod state)
  struct Spectrum {
    bool  ready        = false;
    float omega_center = 0.0f;

    float omega[NBINS]{};
    float domega[NBINS]{};

    float alpha_k[NBINS]{};
    float cos_dphi[NBINS]{};
    float sin_dphi[NBINS]{};

    double c[NBINS]{};
    double s[NBINS]{};
    double zr[NBINS]{};
    double zi[NBINS]{};
double zr_prev[NBINS]{};
double zi_prev[NBINS]{};
double omega_eff[NBINS]{};


    float S_eta_rad[NBINS]{};

    float enbw_hz[NBINS]{};  // exact ENBW (Hz) for each IIR analyzer bin

inline void clear() {
    ready = false;
    for (int i = 0; i < NBINS; ++i) {
        c[i] = 1.0f; s[i] = 0.0f;
        zr[i] = zi[i] = zr_prev[i] = zi_prev[i] = 0.0f;
        omega_eff[i] = omega[i];
        S_eta_rad[i] = 0.0f;
    }
}

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
      constexpr float W_FLOOR = 1e-2f; // rad/s (tiny; << typical ω)
      for (int i = 0; i < NBINS; ++i) {
        const double w = double(omega[i]);
        const float wL = (i > 0)         ? omega[i - 1] : w;
        const float wR = (i < NBINS - 1) ? omega[i + 1] : w;

        float dW = 0.5f * (wR - wL);
        // adaptive floor: prevent tiny ω bins from dominating
        const float dW_min_rel = 0.0025f * std::max(w, 0.0); // 0.25% of ω
        const float dW_min_abs = 1e-5f;
        if (dW < std::max(dW_min_abs, dW_min_rel))
          dW = std::max(dW_min_abs, dW_min_rel);
        domega[i] = dW;

        const float w2 = w * w;
      }

if (!ready) {
    for (int i = 0; i < NBINS; ++i) {
        c[i] = 1.0f; s[i] = 0.0f;
        zr[i] = zi[i] = zr_prev[i] = zi_prev[i] = 0.0f;
    }
    ready = true;
}
        for (int i = 0; i < NBINS; ++i) {
            omega_eff[i] = omega[i];    
        }
    }

inline void precomputeForDt(float dt) {
    const float Fs = 1.0f / dt;
    for (int i = 0; i < NBINS; ++i) {
        const float fk = omega[i] / TWO_PI_;   // bin center frequency (Hz)

        // --- Constant-Q target (relative ENBW = r * fk) ---
        const float r = 0.12f;                 // tune: 0.08..0.20 typical
        const float ENBW_target = r * std::max(fk, 1e-9f);

        // --- Solve for exact EWMA α with this ENBW ---
        float alpha = (2.0f * ENBW_target) / (ENBW_target + 0.5f * Fs);
        if (alpha < 0.0f) alpha = 0.0f;
        else if (alpha > 1.0f) alpha = 1.0f;
        alpha_k[i] = alpha;

        // --- Store the actual ENBW for later PSD normalization ---
        enbw_hz[i] = (alpha / std::max(2.0f - alpha, 1e-12f)) * (0.5f * Fs);

        // --- Rotator step ---
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
   const double w = double(omega_eff[i]);
    const double S = double(S_eta_rad[i]);
    const double dw2 = double(2.0f * domega[i]);

    double term;
    switch (n) {
      case -1: term = (w > 0.0) ? S / w : 0.0; break;
      case 0:  term = S;           break;
      case 1:  term = S * w;       break;
      case 2:  term = S * w * w;   break;
      case 3:  term = S * w * w * w; break;
      case 4:  term = S * w * w * w * w; break;
      default: term = S * std::pow(w, n); break; // fallback for rare cases
    }

    acc += term * dw2;
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

    // main accumulation (unchanged math; already uses full 2·dω)
    template <int NK>
    inline void accumulate(const typename SeaStateRegularity<NK>::Spectrum& S, float dt_s) {
      if (!initialized) buildGrid();
      if (!(tau_spec > 1e-3f)) return;

      const float alpha = 1.0f - std::exp(-dt_s / tau_spec);

      for (int i = 0; i < SeaStateRegularity<NK>::NBINS; ++i) {
        const float Srad = S.S_eta_rad[i];
        if (!(Srad > 0.0f)) continue;

        const float w_c  = S.omega_eff[i];
        const float dw_c = S.domega[i];
        const float wL_i = w_c - dw_c;
        const float wR_i = w_c + dw_c;
        const float E_src = Srad * (2.0f * dw_c);

        for (int j = 0; j < N_BINS; ++j) {
          const float f_c  = freq_hz[j];
          const float dw_j = domega[j];
          const float w_center = TWO_PI_ * f_c;
          const float wL_j = w_center - dw_j;
          const float wR_j = w_center + dw_j;

          const float overlap = std::max(0.0f, std::min(wR_i, wR_j) - std::max(wL_i, wL_j));
          if (overlap <= 0.0f) continue;

          const float frac   = overlap / (wR_i - wL_i);
          const float E_part = E_src * frac;
          const float S_part = E_part / std::max(2.0f * dw_j, 1e-12f);

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
                              float tau_a_mom_sec = 60.0f,
                              float tau_out_sec = 30.0f,
                              float tau_w_sec   = 5.0f)
  : tau_mom(tau_mom_sec), tau_a_mom(tau_a_mom_sec),
    tau_out((tau_out_sec > 1e-3f) ? tau_out_sec : 1e-3f),
    tau_w(tau_w_sec) {
    reset();
  }

  inline void reset() {
    elapsed_s = 0.0f;
    warmup_s  = 0.0f;
      
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
// --- Regularization frequency from averaging horizon (moved before HP) ---
{
    const float Teff = std::max(fixed_avg_.tau_spec, tau_mom);  // seconds
    const float w0 = (Teff > 1e-3f) ? (TWO_PI_ / Teff)
                                    : (2.0f * TWO_PI_ * 0.01f);  // fallback 0.02 Hz
    const float w0_2 = w0 * w0;
    w0_4 = w0_2 * w0_2;
}
      
// --- Textbook one-pole high-pass differentiator (discrete-time) ---
// Purpose: eliminate any DC or sub-ω₀ components before ω⁻⁴ inversion.
// It follows H_HP(z) = (1 - z⁻¹) / (1 - (1 - α) z⁻¹), α = e^(−2π f_c dt)

{
    // Tie cutoff frequency to Tikhonov regularizer (~inverse horizon)
    const double fc_hp = std::max(std::sqrt(std::sqrt(double(w0_4))) / double(TWO_PI_), 0.02); // Hz
    const double alpha_hp = std::exp(-2.0 * M_PI * fc_hp * dt_s);

    // Discrete-time differentiator form:
    // y[n] = α (y[n−1] + x[n] − x[n−1])
    const double x_now = static_cast<double>(accel_z);
    const double y_hp  = alpha_hp * (dc_y1_ + x_now - dc_x1_);

    // update memory
    dc_x1_ = x_now;
    dc_y1_ = y_hp;

    // return filtered acceleration
    const float a_hp = static_cast<float>(y_hp);

    // === Warm-up estimation and ramp (leave unchanged below) ===
    const float tau_hp = 1.0f / std::max(2.0f * float(M_PI) * float(fc_hp), 2e-2f);
    const float f_c    = std::max(omega_used / TWO_PI_, 2e-2f);
    const float r_Q    = 0.12f;
    const float tau_bin = 1.0f / std::max(r_Q * f_c, 2e-2f);
    const float tau_m   = tau_mom;
    warmup_s = 3.0f * std::max({tau_hp, tau_bin, 0.5f * tau_m});

    // Advance time and compute warm factor [0..1]
    elapsed_s += dt_s;
    const float warm = (warmup_s > 1e-6f)
                         ? std::clamp(elapsed_s / warmup_s, 0.0f, 1.0f)
                         : 1.0f;

}
      
// Advance time and compute warm factor [0..1]
elapsed_s += dt_s;
const float warm = (warmup_s > 1e-6f)
                     ? std::clamp(elapsed_s / warmup_s, 0.0f, 1.0f)
                     : 1.0f;
      
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

  // --- Regularization shaping constants ---
const float w0_reg  = (w0_4 > 0.0f) ? std::sqrt(std::sqrt(w0_4))
                                   : (2.0f * TWO_PI_ * 0.01f);
const float beta_reg = 1.4f;  // shape factor for low-ω steepness    

    // hot loop: rotator + 1st-order IIR using precomputed constants
    double S0 = 0.0, S1 = 0.0, S2 = 0.0;
double Sa0 = 0.0, Sa1 = 0.0;
    for (int i = 0; i < NBINS; ++i) {
        
// --- Numerically stable double-precision rotation and IIR update ---
const double c_next = spectrum_.c[i] * spectrum_.cos_dphi[i] - spectrum_.s[i] * spectrum_.sin_dphi[i];
const double s_next = spectrum_.s[i] * spectrum_.cos_dphi[i] + spectrum_.c[i] * spectrum_.sin_dphi[i];
spectrum_.c[i] = c_next;
spectrum_.s[i] = s_next;

// keep rotator on unit circle
const double nrm = std::sqrt(spectrum_.c[i]*spectrum_.c[i] + spectrum_.s[i]*spectrum_.s[i]);
if (nrm > 0.0)
{
    spectrum_.c[i] /= nrm;
    spectrum_.s[i] /= nrm;
}

// use double for demodulation, but cast accel to double once
const double y_r = static_cast<double>(a_hp) * spectrum_.c[i];
const double y_i = -static_cast<double>(a_hp) * spectrum_.s[i];

const double a = static_cast<double>(spectrum_.alpha_k[i]);
spectrum_.zr[i] = (1.0 - a) * spectrum_.zr[i] + a * y_r;
spectrum_.zi[i] = (1.0 - a) * spectrum_.zi[i] + a * y_i;
        
// --- Estimate residual rotation for reassignment (Auger–Flandrin style) ---
const float zr_old = spectrum_.zr_prev[i];
const float zi_old = spectrum_.zi_prev[i];

const float dot = spectrum_.zr[i]*zr_old + spectrum_.zi[i]*zi_old;
const float crs = spectrum_.zi[i]*zr_old - spectrum_.zr[i]*zi_old;
const float dphi = std::atan2(crs, std::max(dot, 1e-24f));
const float delta_omega = dphi / dt_s;

// --- Smoothed reassignment to suppress phase jitter ---
const float alpha_reassign = 0.2f;  // 0.1–0.3 typical; adjust for responsiveness
const float w_target = Spectrum::clampf_(spectrum_.omega[i] + delta_omega,
                                         OMEGA_MIN_RAD, OMEGA_MAX_RAD);
spectrum_.omega_eff[i] = (1.0f - alpha_reassign) * spectrum_.omega_eff[i]
                       + alpha_reassign * w_target;

// --- Save for next-step phase derivative ---
spectrum_.zr_prev[i] = spectrum_.zr[i];
spectrum_.zi_prev[i] = spectrum_.zi[i];
        
// --- Baseband power ---
const float P_bb = spectrum_.zr[i] * spectrum_.zr[i] + spectrum_.zi[i] * spectrum_.zi[i];

// --- Normalize by ENBW (Hz) of this analyzer ---
const float ENBW_Hz = std::max(spectrum_.enbw_hz[i], 1e-12f);
const float S_a_Hz  = P_bb / ENBW_Hz;
const float S_a_rad = S_a_Hz / TWO_PI_;

// --- Use reassigned ω (critical!) ---
const float w  = spectrum_.omega_eff[i];   // ✅ not spectrum_.omega[i]
const float w2 = w * w;
        
const float dw = spectrum_.domega[i];
const float fk = w / TWO_PI_;

// use a relative mask around current center to avoid floor/skirts
const float f_center = std::max(omega_used, 1e-6f) / TWO_PI_;
if (fk >= 0.7f * f_center && fk <= 1.4f * f_center) {
    Sa0 += double(S_a_rad)      * double(2.0f * dw);
    Sa1 += double(S_a_rad) * double(w) * double(2.0f * dw);
}

// --- Physically motivated pre-whitening HP window ---
// This suppresses artificial low-ω lift caused by ω⁻⁴ inversion of noise.
const float w_hp = 2.0f * w0_reg;  // cutoff tied to Tikhonov regularizer
const float hp_gain = (w2 / (w2 + w_hp * w_hp));  // 2nd-order HP envelope      
        
// --- Acceleration → displacement PSD per (rad/s) with Tikhonov regularization ---
const float denom = (w2 * w2) + (beta_reg * w0_reg * w0_reg) * w2 + w0_4;
        
// --- Apply high-pass envelope + regularization + warm-up fade ---
const float S_eta_rad_i = hp_gain * (S_a_rad / std::max(denom, 1e-24f)) * warm;
        
// --- Store ---
spectrum_.S_eta_rad[i] = S_eta_rad_i;
        
      // width contribution to moments
S0 += double(S_eta_rad_i) * double(2.0f * dw);
S1 += double(S_eta_rad_i) * double(w)  * double(2.0f * dw);
S2 += double(S_eta_rad_i) * double(w)  * double(w) * double(2.0f * dw);
        
    }

 if (Sa0 > 0.0) {
    const float w_cent = float(Sa1 / Sa0);             // rad/s (accel centroid)
    const float w_obs_blend = (1.0f - 0.35f) * omega_inst + 0.35f * w_cent;
    // fast complementary correction toward blended observation
    omega_used = (omega_used <= 0.0f)
                   ? w_obs_blend
                   : (1.0f - alpha_wc) * omega_used + alpha_wc * w_obs_blend;

    // tighten jump reset to avoid long mis-centering tails
    const float ratio = w_obs_blend / std::max(omega_used, 1e-6f);
    if (ratio < 0.80f || ratio > 1.25f) {
        omega_used = w_obs_blend;
        spectrum_.clear();
        spectrum_.buildGrid(omega_used, OMEGA_MIN_RAD, OMEGA_MAX_RAD);
        spectrum_.precomputeForDt(dt_s);
    }
}
      
    // moments + Jensen helpers
// --- Fade-in moment accumulation during warm-up ---
const float alpha_mom_eff = alpha_mom * warm;

M0.update(float(S0), alpha_mom_eff);
M1.update(float(S1), alpha_mom_eff);
M2.update(float(S2), alpha_mom_eff);

Q00.update(float(S0 * S0), alpha_mom_eff);
Q10.update(float(S0 * S1), alpha_mom_eff);
Q20.update(float(S0 * S2), alpha_mom_eff);

    spectrum_.ready = true;
    has_moments = true;

    // accumulate into fixed-grid averaged spectrum
// --- Fade-in averaged spectrum accumulation ---
if (warm > 0.0f)
    fixed_avg_.template accumulate<MAX_K>(spectrum_, dt_s * warm);
      
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

  float w0_4 = 0.0f;       // regularization for ω⁻⁴
  float hp_state = 0.0f;   // high-pass memory
  float hp_prev_in = 0.0f;

float tau_wc = 3.0f;   // centroid follow (s)  — fast
float alpha_wc = 0.0f; // computed from dt

// --- Warmup state ---
float elapsed_s = 0.0f;     // time since last reset
float warmup_s  = 0.0f;     // target burn-in horizon (computed online)

// --- DC blocker state (double precision) ---
double dc_x1_ = 0.0;   // x[n-1]
double dc_y1_ = 0.0;   // y[n-1]

  // Internal helpers
  inline void updateGlobalAlphas(float dt_s) {
    if (dt_s == last_dt) return;
    last_dt   = dt_s;
    alpha_mom = 1.0f - std::exp(-dt_s / tau_mom);
    alpha_a_mom = 1.0f - std::exp(-dt_s / tau_a_mom);
    alpha_out = 1.0f - std::exp(-dt_s / tau_out);
    alpha_w   = 1.0f - std::exp(-dt_s / tau_w);

alpha_wc = 1.0f - std::exp(-dt_s / tau_wc);      
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
