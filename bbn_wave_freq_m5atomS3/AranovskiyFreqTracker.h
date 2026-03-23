#ifndef ARANOVSKIY_FREQ_TRACKER_H
#define ARANOVSKIY_FREQ_TRACKER_H

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstdlib>

/*
  Copyright 2024-2026, Mikhail Grushinskiy

  Aranovskiy frequency estimator (C++ version)

  Reference:
  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy
  "The New Algorithm of Sinusoidal Signal Frequency Estimation",
  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing, July 2013

  This is a nonlinear adaptive observer that tracks the frequency and phase of a sinusoidal signal.

  Added API:
    - getConfidence()
    - isLocked()
    - getRawFrequencyHz()
    - hasCoarseEstimate()
    - getCoarseFrequencyHz()

  Notes:
    - Confidence/lock are synthesized heuristics, because the original Aranovskiy
      observer does not natively provide them.
    - Default lock thresholds assume input is roughly normalized to "g" units,
      as in your TrackerPolicy where a/g_std is passed in.
*/

template <typename Real = double>
class AranovskiyFreqTracker {
public:
  // Parameters
  Real a     = Real(1);    // Input filter gain
  Real b     = Real(1);    // Input filter gain
  Real k     = Real(8);    // Adaptive gain

  // State
  Real y         = Real(0);      // Last measurement
  Real x1        = Real(0);      // Internal filtered state
  Real theta     = Real(-0.09);  // Estimator variable
  Real sigma     = Real(-0.09);  // Estimator variable
  Real x1_dot    = Real(0);
  Real sigma_dot = Real(0);
  Real omega     = Real(3.0);    // Estimated angular frequency (rad/s), internal scaled
  Real f         = Real(0);      // Estimated frequency (Hz), internal scaled
  Real phase     = Real(0);      // Estimated phase (radians)

  // Confidence / lock tuning
  Real rms_tau_s           = Real(8.0);
  Real confidence_tau_s    = Real(6.0);
  Real freq_smooth_tau_s   = Real(3.0);
  Real lock_rms_min        = Real(0.012);
  Real lock_conf_threshold = Real(0.65);
  Real f_lock_min_hz       = Real(0.03);
  Real f_lock_max_hz       = Real(2.0);

  // Confidence / lock state
  Real signal_power = Real(0);
  Real signal_rms   = Real(0);
  Real f_smooth_hz  = Real(0);
  Real confidence   = Real(0);
  bool locked       = false;

  // Internal time scaling for better low-frequency stability
  static constexpr Real TIME_SCALE = Real(20);

  // Constructor
  AranovskiyFreqTracker(Real omega_up = Real(0.5) * Real(2) * Real(M_PI),
                        Real gain     = Real(8),
                        Real x1_0     = Real(0),
                        Real theta_0  = Real(-0.09),
                        Real sigma_0  = Real(-0.09))
  {
    setParams(omega_up, gain);
    setState(x1_0, theta_0, sigma_0);
  }

  // Set filter parameters
  void setParams(Real omega_up, Real gain) {
    a = TIME_SCALE * omega_up;
    b = TIME_SCALE * omega_up;
    k = TIME_SCALE * gain;
  }

  // Optional tuning for synthesized lock / confidence
  void setConfidenceParams(Real rms_tau_s_in,
                           Real confidence_tau_s_in,
                           Real freq_smooth_tau_s_in,
                           Real lock_rms_min_in,
                           Real lock_conf_threshold_in,
                           Real f_lock_min_hz_in = Real(0.03),
                           Real f_lock_max_hz_in = Real(2.0))
  {
    rms_tau_s           = std::max(rms_tau_s_in, Real(1e-6));
    confidence_tau_s    = std::max(confidence_tau_s_in, Real(1e-6));
    freq_smooth_tau_s   = std::max(freq_smooth_tau_s_in, Real(1e-6));
    lock_rms_min        = std::max(lock_rms_min_in, Real(0));
    lock_conf_threshold = clamp01(lock_conf_threshold_in);
    f_lock_min_hz       = std::max(f_lock_min_hz_in, Real(1e-6));
    f_lock_max_hz       = std::max(f_lock_max_hz_in, f_lock_min_hz + Real(1e-6));
  }

  // Set initial state
  void setState(Real x1_init, Real theta_init, Real sigma_init) {
    x1    = x1_init;
    theta = TIME_SCALE * TIME_SCALE * theta_init;
    sigma = TIME_SCALE * TIME_SCALE * sigma_init;
    y     = Real(0);

    omega = std::sqrt(std::max(Real(1e-12), std::abs(theta)));
    f     = omega / (Real(2) * Real(M_PI));
    phase = Real(0);

    resetConfidenceState_();
  }

  // Reset synthesized confidence/lock only
  void resetConfidenceState() {
    resetConfidenceState_();
  }

  // Update filter with new measurement and time step
  void update(Real y_meas, Real dt) {
    if (!std::isfinite(y_meas)) {
      return;
    }
    if (!(std::isfinite(dt) && dt > Real(0))) {
      return;
    }

    const Real delta_t = dt / TIME_SCALE;
    y = y_meas;

    // First-order low-pass filter
    x1_dot = -a * x1 + b * y;

    // Signal energy check
    const Real signal_energy = std::fma(x1, x1, y * y) + Real(1e-12);
    const Real gain_scaling  = signal_energy / (signal_energy + Real(1e-6));

    // Nonlinear adaptation law
    const Real x1_sq = x1 * x1;
    const Real tmp   = std::fma(b, y, a * x1);
    const Real phi   = std::fma(x1_dot, tmp, x1_sq * theta);

    const Real update_term = -k * phi * gain_scaling;

    sigma_dot = clamp_value(update_term, Real(-1e7), Real(1e7));

    // Update theta and omega
    theta = std::fma(k * b * x1, y, sigma);

    omega = std::sqrt(std::max(Real(1e-7), std::abs(theta)));
    f     = omega / (Real(2) * Real(M_PI));

    // State integration
    x1    += x1_dot    * delta_t;
    sigma += sigma_dot * delta_t;

    // Phase estimation
    phase = std::atan2(x1, y);

    updateConfidenceAndLock_(dt);
  }

  // Main estimate
  Real getFrequencyHz() const {
    return f / TIME_SCALE;
  }

  // Alias for uniform API
  Real getRawFrequencyHz() const {
    return getFrequencyHz();
  }

  // Get current angular frequency estimate (rad/s, internal scaled)
  Real getOmega() const {
    return omega;
  }

  // Get current oscillator phase estimate (radians)
  Real getPhase() const {
    return phase;
  }

  // Synthesized confidence in [0, 1]
  Real getConfidence() const {
    return confidence;
  }

  // Synthesized lock flag
  bool isLocked() const {
    return locked;
  }

  // Optional diagnostics
  Real getSignalRms() const {
    return signal_rms;
  }

  Real getSmoothedFrequencyHz() const {
    return f_smooth_hz;
  }

  // Uniform API with trackers that may support coarse estimate
  bool hasCoarseEstimate() const {
    return false;
  }

  Real getCoarseFrequencyHz() const {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    throw std::logic_error("AranovskiyFreqTracker: coarse frequency estimate is not implemented");
#else
    std::abort();
#endif
  }

private:
  static constexpr Real clamp_value(Real val, Real low, Real high) {
    return (val < low) ? low : (val > high ? high : val);
  }

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
    f_smooth_hz  = getFrequencyHz();
    if (!(std::isfinite(f_smooth_hz) && f_smooth_hz > Real(0))) {
      f_smooth_hz = Real(0);
    }
    confidence = Real(0);
    locked     = false;
  }

  void updateConfidenceAndLock_(Real dt) {
    const Real f_hz = getFrequencyHz();

    // Track signal RMS from observer state + measurement
    const Real inst_power = std::fma(x1, x1, y * y);
    const Real alpha_pow  = onePoleAlpha(dt, rms_tau_s);
    signal_power += alpha_pow * (inst_power - signal_power);

    if (!(std::isfinite(signal_power) && signal_power >= Real(0))) {
      signal_power = Real(0);
    }
    signal_rms = std::sqrt(signal_power);

    // Smooth frequency only for stability/confidence logic
    if (std::isfinite(f_hz) && f_hz > Real(0)) {
      if (!(std::isfinite(f_smooth_hz) && f_smooth_hz > Real(0))) {
        f_smooth_hz = f_hz;
      } else {
        const Real alpha_f = onePoleAlpha(dt, freq_smooth_tau_s);
        f_smooth_hz += alpha_f * (f_hz - f_smooth_hz);
      }
    }

    const bool freq_valid =
        std::isfinite(f_hz) &&
        f_hz >= f_lock_min_hz &&
        f_hz <= f_lock_max_hz;

    Real conf_target = Real(0);

    if (freq_valid) {
      const Real amp_ratio =
          clamp01(signal_rms / std::max(lock_rms_min, Real(1e-9)));

      const Real denom = std::max(Real(0.03),
                                  Real(0.25) * std::max(f_smooth_hz, f_lock_min_hz));
      const Real stab =
          Real(1) - std::min(Real(1), std::abs(f_hz - f_smooth_hz) / denom);

      conf_target = amp_ratio * (Real(0.35) + Real(0.65) * stab);
    }

    const Real alpha_c = onePoleAlpha(dt, confidence_tau_s);
    confidence += alpha_c * (conf_target - confidence);
    confidence = clamp01(confidence);

    locked = freq_valid &&
             (signal_rms >= lock_rms_min) &&
             (confidence >= lock_conf_threshold);
  }
};

#endif // ARANOVSKIY_FREQ_TRACKER_H
