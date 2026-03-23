#ifndef KALMANF_FREQ_TRACKER_H
#define KALMANF_FREQ_TRACKER_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

/*
   See: https://github.com/randyaliased/KalmANF/

   and

   See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter
   update of a single parameter adaptive notch filter",
   Proceedings of the 26th International Conference on Digital Audio Effects
   (DAFx23), Copenhagen, Denmark, September 2023

   Conceptual model

   The algorithm tracks the dominant sinusoidal component of a real-valued signal y[n]
   by adapting the coefficient a in a second-order resonator / notch filter. The
   resonator is parameterized as

       s[n] = y[n] + ρ · a · s[n−1] − ρ² · s[n−2],

   where a = 2 cos(ω_d), and ω_d is the (digital) radian frequency per sample.

   The scalar state a is updated by a 1-D Kalman filter driven by the notch error e[n]:

       e[n] = s[n] − a[n−1] · s[n−1] + s[n−2],
       a[n] = a[n−1] + K[n] · e[n],

   with K[n] chosen according to a scalar Kalman update.

   Parameters (conceptual roles)

   • ρ (rho) – pole radius of the resonator, 0 < ρ < 1
       – ρ → 1.0  : high-Q, narrowband, long memory, more selective.
       – smaller ρ: more damping, broader bandwidth, less selective, more robust.

   • a – adaptive notch coefficient, a = 2 cos(ω_d)
       – Encodes the tracked digital frequency ω_d.
       – a ≈  2  ⇒ very low frequency (near DC).
       – a ≈  0  ⇒ mid-band (around f_s / 4).
       – a ≈ −2  ⇒ near Nyquist (f_s / 2).

   • q – process noise variance on a (Q ≈ q)
       – Controls how quickly the filter “forgets” its previous estimate of a.
       – Larger q: p grows faster between samples → larger Kalman gain → a adapts
         quickly (forgets history faster), but the frequency can become noisier.
       – Smaller q: p grows slowly → smaller Kalman gain → very smooth and
         “sticky” behaviour (remembers history longer), but slower to follow
         genuine frequency shifts.

   • r – measurement noise variance on e[n] (R ≈ r)
       – Models how noisy / unreliable the error e[n] is.
       – Larger r: smaller Kalman gain, smoother and slower updates.
       – Smaller r: larger Kalman gain, more aggressive and jitter-prone.

   • p – p_cov, Kalman error covariance on a
       – Internal state tracking the uncertainty in a.
       – Larger p → larger gain K (we believe a is uncertain).
       – Smaller p → smaller gain K (we believe a is well known).

   Mapping back to Hz

   Once the updated a is available, the instantaneous digital frequency is

       ω̂_d = arccos(a / 2),

   and the physical frequency in Hz is

       f̂ = (ω̂_d / Δt) / (2π),

   where Δt is the effective sample period.

   Notes:
     - Confidence / lock are synthesized heuristics. Native KalmANF does not
       directly provide them.
     - hasCoarseEstimate() returns false.
     - getCoarseFrequencyHz() throws (or aborts if exceptions are disabled).
*/

template <typename Real = double>
class KalmANFFreqTracker {
private:
  static constexpr Real defaultRho = Real(0.995);
  static constexpr Real default_a  = Real(1.9999);

  class ANFResonator {
  public:
    Real s_prev1 = Real(0);
    Real s_prev2 = Real(0);
    Real a       = default_a;
    Real rho     = defaultRho;
    Real rho_sq  = defaultRho * defaultRho;

    void init(Real rho_init, Real a_init, Real s1, Real s2) {
      rho     = rho_init;
      rho_sq  = rho * rho;
      a       = a_init;
      s_prev1 = s1;
      s_prev2 = s2;
    }

    Real compute_s(Real y) const {
      return y + rho * s_prev1 * a - rho_sq * s_prev2;
    }

    void update_state(Real s) {
      s_prev2 = s_prev1;
      s_prev1 = s;
    }

    Real get_phase() const {
      return std::atan2(s_prev1, s_prev2);
    }
  };

  ANFResonator res;

  // Kalman parameters
  Real p_cov = Real(1);
  Real q     = Real(1e-5);
  Real r     = Real(1e+3);

  // Latest frequency estimate
  Real f_est_hz = Real(0);

  // Synthesized confidence / lock tuning
  Real rms_tau_s            = Real(8.0);
  Real err_tau_s            = Real(5.0);
  Real confidence_tau_s     = Real(6.0);
  Real freq_smooth_tau_s    = Real(3.0);
  Real lock_rms_min         = Real(0.012);
  Real lock_conf_threshold  = Real(0.65);
  Real f_lock_min_hz        = Real(0.03);
  Real f_lock_max_hz        = Real(2.0);
  Real rel_err_lock_soft    = Real(1.25);

  // Synthesized confidence / lock state
  Real signal_power         = Real(0);
  Real signal_rms           = Real(0);
  Real err_power            = Real(0);
  Real err_rms              = Real(0);
  Real last_error           = Real(0);
  Real f_smooth_hz          = Real(0);
  Real confidence           = Real(0);
  bool locked               = false;

public:
  void init(Real rho      = defaultRho,
            Real q_       = Real(1e-5),
            Real r_       = Real(1e+3),
            Real p_cov_   = Real(1),
            Real s_prev1_ = Real(0),
            Real s_prev2_ = Real(0),
            Real a_       = default_a)
  {
    q     = q_;
    r     = r_;
    p_cov = std::max(p_cov_, Real(1e-12));
    res.init(rho, a_, s_prev1_, s_prev2_);
    f_est_hz = Real(0);
    resetConfidenceState_();
  }

  // Convenience initializer: seed a from a frequency guess in Hz.
  // f_guess_hz : initial frequency guess (Hz)
  // dt         : sample period (s)
  // rho, q_, r_, p_cov_ behave as in init().
  void initFromFreqGuess(Real f_guess_hz,
                         Real dt,
                         Real rho      = defaultRho,
                         Real q_       = Real(1e-6),
                         Real r_       = Real(1e+3),
                         Real p_cov_   = Real(1))
  {
    const Real dt_safe = (std::isfinite(dt) && dt > Real(0)) ? dt : Real(1e-3);

    // ω_d (rad/sample) = 2π f / f_s = 2π f · dt
    const Real omega_d = Real(2) * Real(M_PI) * f_guess_hz * dt_safe;
    const Real a_init  = Real(2) * std::cos(omega_d);
    init(rho, q_, r_, p_cov_, Real(0), Real(0), a_init);

    f_est_hz = std::max(f_guess_hz, Real(0));
    f_smooth_hz = f_est_hz;
  }

  // Optional tuning for synthesized confidence / lock logic.
  void setConfidenceParams(Real rms_tau_s_in,
                           Real err_tau_s_in,
                           Real confidence_tau_s_in,
                           Real freq_smooth_tau_s_in,
                           Real lock_rms_min_in,
                           Real lock_conf_threshold_in,
                           Real f_lock_min_hz_in = Real(0.03),
                           Real f_lock_max_hz_in = Real(2.0),
                           Real rel_err_lock_soft_in = Real(1.25))
  {
    rms_tau_s           = std::max(rms_tau_s_in, Real(1e-6));
    err_tau_s           = std::max(err_tau_s_in, Real(1e-6));
    confidence_tau_s    = std::max(confidence_tau_s_in, Real(1e-6));
    freq_smooth_tau_s   = std::max(freq_smooth_tau_s_in, Real(1e-6));
    lock_rms_min        = std::max(lock_rms_min_in, Real(0));
    lock_conf_threshold = clamp01(lock_conf_threshold_in);
    f_lock_min_hz       = std::max(f_lock_min_hz_in, Real(1e-6));
    f_lock_max_hz       = std::max(f_lock_max_hz_in, f_lock_min_hz + Real(1e-6));
    rel_err_lock_soft   = std::max(rel_err_lock_soft_in, Real(1e-6));
  }

  void resetConfidenceState() {
    resetConfidenceState_();
  }

  // y: input sample, in *whatever* units (but tune q,r for that scale)
  // dt: actual sample period (seconds)
  Real process(Real y, Real dt, Real* e_out = nullptr) {
    if (!std::isfinite(y)) {
      if (e_out) *e_out = last_error;
      return f_est_hz;
    }
    if (!(std::isfinite(dt) && dt > Real(0))) {
      if (e_out) *e_out = last_error;
      return f_est_hz;
    }

    const Real delta_t = dt;

    // 1. resonator
    const Real s = res.compute_s(y);

    // 2. prediction
    p_cov += q;
    p_cov = std::max(p_cov, Real(1e-12));

    // 3. Kalman gain (original form)
    const Real signal_power_inst = res.s_prev1 * res.s_prev1;
    const Real denom = signal_power_inst + r / (p_cov + std::numeric_limits<Real>::epsilon());
    const Real K = res.s_prev1 / (denom + Real(1e-12));

    // 4. error
    const Real e = s - res.s_prev1 * res.a + res.s_prev2;
    last_error = e;

    // 5. update a
    Real a = res.a + K * e;

    // 6. keep a in acos domain
    if (a > Real(2) || a < Real(-2)) {
      a = (a > Real(2)) ? Real(1.99999) : Real(-1.99999);
    }

    // 7. covariance update
    p_cov = (Real(1) - K * res.s_prev1) * p_cov;
    p_cov = std::max(p_cov, Real(1e-12));

    // 8. frequency estimate
    const Real omega_hat = std::acos(a / Real(2));  // rad/sample
    Real f_now = (omega_hat / delta_t) / (Real(2) * Real(M_PI)); // Hz
    if (!(std::isfinite(f_now) && f_now >= Real(0))) {
      f_now = Real(0);
    }
    f_est_hz = f_now;

    // 9. state update
    res.a = a;
    res.update_state(s);

    if (e_out) *e_out = e;

    updateConfidenceAndLock_(y, e, f_now, dt);
    return f_est_hz;
  }

  Real get_phase() const { return res.get_phase(); }
  Real get_a() const     { return res.a; }
  Real get_p_cov() const { return p_cov; }

  // Uniform tracker API
  Real getFrequencyHz() const {
    return f_est_hz;
  }

  Real getRawFrequencyHz() const {
    return f_est_hz;
  }

  Real getConfidence() const {
    return confidence;
  }

  bool isLocked() const {
    return locked;
  }

  bool hasCoarseEstimate() const {
    return false;
  }

  Real getCoarseFrequencyHz() const {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    throw std::logic_error("KalmANFFreqTracker: coarse frequency estimate is not implemented");
#else
    std::abort();
#endif
  }

  // Optional diagnostics
  Real getSignalRms() const {
    return signal_rms;
  }

  Real getErrorRms() const {
    return err_rms;
  }

  Real getLastError() const {
    return last_error;
  }

  Real getSmoothedFrequencyHz() const {
    return f_smooth_hz;
  }

private:
  static constexpr Real clamp01(Real x) {
    return (x < Real(0)) ? Real(0) : (x > Real(1) ? Real(1) : x);
  }

  static Real onePoleAlpha(Real dt, Real tau) {
    if (!(std::isfinite(dt) && dt > Real(0))) return Real(0);
    if (!(std::isfinite(tau) && tau > Real(0))) return Real(1);
    const Real a = dt / (tau + dt);
    return clamp01(a);
  }

  void resetConfidenceState_() {
    signal_power = Real(0);
    signal_rms   = Real(0);
    err_power    = Real(0);
    err_rms      = Real(0);
    last_error   = Real(0);
    confidence   = Real(0);
    locked       = false;
    f_smooth_hz  = (std::isfinite(f_est_hz) && f_est_hz > Real(0)) ? f_est_hz : Real(0);
  }

  void updateConfidenceAndLock_(Real y, Real e, Real f_now, Real dt) {
    // Signal level: combine raw input and resonator state a bit so confidence
    // does not depend only on raw y amplitude.
    const Real sig_inst =
        std::fma(y, y, Real(0.5) * res.s_prev1 * res.s_prev1);

    const Real a_sig = onePoleAlpha(dt, rms_tau_s);
    signal_power += a_sig * (sig_inst - signal_power);
    if (!(std::isfinite(signal_power) && signal_power >= Real(0))) {
      signal_power = Real(0);
    }
    signal_rms = std::sqrt(signal_power);

    // Notch/residual error RMS: lower relative error should mean better lock.
    const Real a_err = onePoleAlpha(dt, err_tau_s);
    err_power += a_err * (e * e - err_power);
    if (!(std::isfinite(err_power) && err_power >= Real(0))) {
      err_power = Real(0);
    }
    err_rms = std::sqrt(err_power);

    // Smoothed frequency used only for stability/confidence shaping.
    if (std::isfinite(f_now) && f_now > Real(0)) {
      if (!(std::isfinite(f_smooth_hz) && f_smooth_hz > Real(0))) {
        f_smooth_hz = f_now;
      } else {
        const Real a_f = onePoleAlpha(dt, freq_smooth_tau_s);
        f_smooth_hz += a_f * (f_now - f_smooth_hz);
      }
    }

    const bool freq_valid =
        std::isfinite(f_now) &&
        f_now >= f_lock_min_hz &&
        f_now <= f_lock_max_hz;

    Real conf_target = Real(0);

    if (freq_valid) {
      const Real amp_ratio =
          clamp01(signal_rms / std::max(lock_rms_min, Real(1e-9)));

      // Relative error score: smaller residual compared with signal level
      // should increase confidence.
      const Real rel_err =
          err_rms / std::max(signal_rms, Real(1e-9));
      const Real err_score =
          Real(1) - std::min(Real(1), rel_err / rel_err_lock_soft);

      // Frequency stability score: if instantaneous and smoothed estimates stay
      // close, confidence should rise.
      const Real denom =
          std::max(Real(0.03), Real(0.25) * std::max(f_smooth_hz, f_lock_min_hz));
      const Real stab =
          Real(1) - std::min(Real(1), std::abs(f_now - f_smooth_hz) / denom);

      conf_target = amp_ratio * (Real(0.20) + Real(0.45) * err_score + Real(0.35) * stab);
    }

    const Real a_c = onePoleAlpha(dt, confidence_tau_s);
    confidence += a_c * (conf_target - confidence);
    confidence = clamp01(confidence);

    locked = freq_valid &&
             (signal_rms >= lock_rms_min) &&
             (confidence >= lock_conf_threshold);
  }
};

#endif // KALMANF_FREQ_TRACKER_H
