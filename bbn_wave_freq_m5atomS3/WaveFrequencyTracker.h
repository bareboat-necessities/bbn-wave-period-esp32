#ifndef WAVE_FREQUENCY_TRACKER_H
#define WAVE_FREQUENCY_TRACKER_H

#include <cmath>
#include <cstdint>
#include <limits>

/*
  WaveFrequencyTracker.h
  ----------------------

  Standalone frequency tracker for dominant wave frequency from an acceleration-like signal.

  Design goals:
    - No dependency on old estimator cores
    - Numerically stable with variable dt
    - Control-theory style tracker (PLL/FLL-like), not Kalman
    - Robust to low-frequency drift and moderate broadband noise
    - Explicit frequency band limits
    - Lock/confidence logic
    - Optional coarse acquisition aid

  Recommended input:
    - World-frame vertical inertial acceleration (gravity removed)
    - Units can be m/s^2, but any consistent acceleration-like units are fine

  Core idea:
    1) Remove very slow drift with a low cutoff HP stage
    2) Suppress high-frequency junk with a low-pass stage
    3) Demodulate against an internal oscillator
    4) Low-pass the I/Q baseband terms
    5) Use atan2(Q, I) as phase error
    6) Run a bounded PI loop on oscillator frequency

  Notes:
    - Acceleration and displacement share the same oscillation frequency.
      So this estimates wave displacement frequency directly from acceleration,
      without integrating to displacement.
    - This tracks ONE dominant narrowband component. If the sea is strongly
      bimodal/multimodal, any single-frequency tracker can hop or average.
    - For best results, combine this with a slower outer spectral estimate.

  Usage:
    WaveFrequencyTracker<float> trk;
    trk.reset(0.12f); // optional initial guess in Hz
    ...
    trk.update(a_world_z, dt);
    float f_hz = trk.getFrequencyHz();

  Tuning:
    - f_min_hz / f_max_hz:
        physical search band
    - pre_hp_hz:
        should be below the slowest wave you care about
    - pre_lp_hz:
        should be above the fastest wave you care about
    - loop_bandwidth_hz:
        lower = smoother / slower, higher = faster / noisier
    - demod_lp_hz:
        how quickly I/Q baseband follows phase mismatch
    - lock_rms_min:
        minimum useful signal level in your input units

  Default band:
    0.04 .. 0.60 Hz  (about 25 s .. 1.67 s period)
*/

template <typename Real = float>
class WaveFrequencyTracker {
public:
  struct Config {
    // Search band for dominant wave frequency.
    Real f_min_hz = Real(0.04);
    Real f_max_hz = Real(0.60);

    // Initial / nominal frequency used at reset and for recentering when unlocked.
    Real f_init_hz = Real(0.12);

    // Front-end preprocessing:
    //   HP removes tilt leakage / drift / quasi-DC contamination
    //   LP suppresses higher-frequency noise and non-wave content
    Real pre_hp_hz = Real(0.03);
    Real pre_lp_hz = Real(0.80);

    // Baseband LP for demodulated I/Q terms.
    // Should usually be smaller than the wave band, but not too tiny.
    Real demod_lp_hz = Real(0.08);

    // PLL natural bandwidth and damping.
    // Lower bandwidth -> less jitter, more lag.
    // Higher bandwidth -> faster tracking, more noise sensitivity.
    Real loop_bandwidth_hz = Real(0.03);
    Real loop_damping      = Real(0.90);

    // Output frequency smoothing.
    Real output_smooth_tau_s = Real(1.5);

    // Power / confidence smoothing.
    Real power_tau_s      = Real(6.0);
    Real confidence_tau_s = Real(3.0);

    // Minimum RMS of the prefiltered signal required for reliable tracking.
    // Units are the same as the input units.
    Real lock_rms_min = Real(0.005);

    // Coherence thresholds.
    // coherence = 1 means "very sinusoidal / strongly dominant single tone"
    Real lock_coherence_min   = Real(0.45);
    Real unlock_coherence_min = Real(0.25);

    // Frequency slew limit for stability.
    Real max_dfdt_hz_per_s = Real(0.08);

    // When confidence is poor, slowly drift frequency back toward f_init_hz.
    Real recenter_tau_s = Real(20.0);

    // Internal substep size.
    // Larger dt is split into smaller stable substeps.
    Real max_internal_step_s = Real(0.05);

    // Optional coarse acquisition assist from thresholded period detection.
    bool enable_coarse_assist = true;
    Real coarse_hysteresis_frac = Real(0.30); // threshold = frac * RMS
    Real coarse_smooth_tau_s    = Real(6.0);
    Real coarse_pull_tau_s      = Real(4.0);
    Real coarse_timeout_s       = Real(20.0);
  };

  explicit WaveFrequencyTracker(const Config& cfg = Config()) {
    configure(cfg);
    reset(cfg_.f_init_hz);
  }

  void configure(const Config& cfg) {
    cfg_ = sanitizeConfig(cfg);

    // Clamp current state into the new valid range if already initialized.
    const Real wmin = hzToRad(cfg_.f_min_hz);
    const Real wmax = hzToRad(cfg_.f_max_hz);
    if (initialized_) {
      omega_rad_s_   = clamp(omega_rad_s_, wmin, wmax);
      raw_freq_hz_   = radToHz(omega_rad_s_);
      smooth_freq_hz_ = clamp(smooth_freq_hz_, cfg_.f_min_hz, cfg_.f_max_hz);
      nominal_omega_rad_s_ = hzToRad(cfg_.f_init_hz);
    }
  }

  void reset(Real f_init_hz = Real(-1)) {
    if (!isFinite(f_init_hz) || f_init_hz <= Real(0)) {
      f_init_hz = cfg_.f_init_hz;
    }

    f_init_hz = clamp(f_init_hz, cfg_.f_min_hz, cfg_.f_max_hz);

    nominal_omega_rad_s_ = hzToRad(cfg_.f_init_hz);
    omega_rad_s_         = hzToRad(f_init_hz);
    raw_freq_hz_         = f_init_hz;
    smooth_freq_hz_      = f_init_hz;

    phase_rad_ = Real(0);
    phase_error_rad_ = Real(0);

    // Front-end states
    dc_lp_ = Real(0);
    bp_    = Real(0);

    // Demod states
    i_lp_ = Real(0);
    q_lp_ = Real(0);

    // Stats / confidence
    power_lp_   = Real(0);
    rms_        = Real(0);
    amplitude_  = Real(0);
    coherence_  = Real(0);
    confidence_ = Real(0);
    locked_     = false;

    // Coarse assist states
    time_s_                 = Real(0);
    coarse_freq_hz_         = f_init_hz;
    coarse_valid_           = false;
    coarse_age_s_           = std::numeric_limits<Real>::infinity();
    coarse_last_event_t_s_  = Real(-1);
    coarse_seen_negative_   = false;

    initialized_ = true;
  }

  // Update with one new sample.
  // x is typically vertical inertial acceleration (world Z, gravity removed).
  void update(Real x, Real dt_s) {
    if (!initialized_) {
      reset(cfg_.f_init_hz);
    }

    if (!isFinite(x) || !isFinite(dt_s) || dt_s <= Real(0)) {
      return;
    }

    // Split long dt into smaller internal steps for stability.
    const Real max_step = maxValue(cfg_.max_internal_step_s, Real(1e-4));
    int n_steps = int(std::ceil(dt_s / max_step));
    if (n_steps < 1) n_steps = 1;
    const Real sub_dt = dt_s / Real(n_steps);

    for (int i = 0; i < n_steps; ++i) {
      stepInternal(x, sub_dt);
    }

    // Coarse assist is based on the final prefiltered sample after the update.
    updateCoarseAssist(bp_, dt_s);
  }

  // Main tracked frequency (smoothed)
  Real getFrequencyHz() const {
    return smooth_freq_hz_;
  }

  // Raw loop frequency before output smoothing
  Real getRawFrequencyHz() const {
    return raw_freq_hz_;
  }

  Real getAngularFrequencyRadPerSec() const {
    return omega_rad_s_;
  }

  Real getPeriodSec() const {
    const Real f = getFrequencyHz();
    return (f > Real(0)) ? (Real(1) / f) : std::numeric_limits<Real>::infinity();
  }

  Real getPhaseRad() const {
    return phase_rad_;
  }

  Real getPhaseErrorRad() const {
    return phase_error_rad_;
  }

  Real getBandpassedSignal() const {
    return bp_;
  }

  Real getAmplitudeEstimate() const {
    return amplitude_;
  }

  Real getRmsEstimate() const {
    return rms_;
  }

  Real getCoherence() const {
    return coherence_;
  }

  Real getConfidence() const {
    return confidence_;
  }

  bool isLocked() const {
    return locked_;
  }

  bool hasCoarseEstimate() const {
    return coarse_valid_;
  }

  Real getCoarseFrequencyHz() const {
    return coarse_freq_hz_;
  }

  const Config& config() const {
    return cfg_;
  }

private:
  Config cfg_{};

  // Constants
  static constexpr Real kPi    = Real(3.1415926535897932384626433832795);
  static constexpr Real kTwoPi = Real(6.2831853071795864769252867665590);

  // Initialization
  bool initialized_ = false;

  // Oscillator / PLL states
  Real omega_rad_s_ = Real(0);
  Real nominal_omega_rad_s_ = Real(0);
  Real phase_rad_  = Real(0);
  Real phase_error_rad_ = Real(0);

  // Front-end states
  Real dc_lp_ = Real(0); // low-pass state used to create HP: hp = x - dc_lp
  Real bp_    = Real(0); // low-passed HP signal (broad bandpassed output)

  // Demod / baseband states
  Real i_lp_ = Real(0);
  Real q_lp_ = Real(0);

  // Statistics
  Real power_lp_   = Real(0);
  Real rms_        = Real(0);
  Real amplitude_  = Real(0);
  Real coherence_  = Real(0);
  Real confidence_ = Real(0);

  // Outputs
  Real raw_freq_hz_    = Real(0);
  Real smooth_freq_hz_ = Real(0);

  // Lock state
  bool locked_ = false;

  // Coarse acquisition aid
  Real time_s_                = Real(0);
  Real coarse_freq_hz_        = Real(0);
  bool coarse_valid_          = false;
  Real coarse_age_s_          = Real(0);
  Real coarse_last_event_t_s_ = Real(-1);
  bool coarse_seen_negative_  = false;

private:
  void stepInternal(Real x, Real dt) {
    const Real eps = Real(1e-12);

    // ---------- 1) Front-end broad bandpass ----------
    //
    // HP formed as x - LP(x)
    // Then LP the HP result to limit high-frequency junk.
    //
    // All one-pole filters use exact exponential smoothing:
    //   state = alpha * state + (1-alpha) * input
    // which is stable for any dt > 0.
    const Real a_hp = alphaFromCutoffHz(cfg_.pre_hp_hz, dt);
    const Real a_lp = alphaFromCutoffHz(cfg_.pre_lp_hz, dt);

    dc_lp_ = mixExp(dc_lp_, x, a_hp);
    const Real hp = x - dc_lp_;

    bp_ = mixExp(bp_, hp, a_lp);

    // ---------- 2) Update energy / RMS ----------
    const Real a_pow = alphaFromTau(cfg_.power_tau_s, dt);
    power_lp_ = mixExp(power_lp_, bp_ * bp_, a_pow);
    power_lp_ = maxValue(power_lp_, Real(0));
    rms_ = std::sqrt(power_lp_ + eps);

    // ---------- 3) Demodulate with internal oscillator ----------
    //
    // If bp_ ~ A cos(phi + delta), then after LP:
    //   I ~ A/2 cos(delta)
    //   Q ~ A/2 sin(delta)
    // so phase error is atan2(Q, I).
    const Real c = std::cos(phase_rad_);
    const Real s = std::sin(phase_rad_);

    const Real i_meas = bp_ * c;
    const Real q_meas = bp_ * (-s);

    const Real a_demod = alphaFromCutoffHz(cfg_.demod_lp_hz, dt);
    i_lp_ = mixExp(i_lp_, i_meas, a_demod);
    q_lp_ = mixExp(q_lp_, q_meas, a_demod);

    // Tone amplitude estimate.
    // Because synchronous demod halves sinusoid amplitude, use factor 2.
    amplitude_ = Real(2) * safeHypot(i_lp_, q_lp_);

    // Coherence estimate:
    //   pure single tone -> amplitude / (sqrt(2)*RMS) ~ 1
    const Real denom = maxValue(std::sqrt(Real(2)) * rms_, eps);
    coherence_ = clamp(amplitude_ / denom, Real(0), Real(1));

    // ---------- 4) Lock / confidence logic ----------
    const Real rms_score = clamp(rms_ / maxValue(cfg_.lock_rms_min, eps), Real(0), Real(1));
    const Real conf_target = clamp(rms_score * coherence_, Real(0), Real(1));

    const Real a_conf = alphaFromTau(cfg_.confidence_tau_s, dt);
    confidence_ = mixExp(confidence_, conf_target, a_conf);

    // Hysteretic lock decision
    const bool enough_rms_lock   = (rms_ >= cfg_.lock_rms_min);
    const bool enough_rms_unlock = (rms_ >= cfg_.lock_rms_min * Real(0.70));

    if (!locked_) {
      if (enough_rms_lock && coherence_ >= cfg_.lock_coherence_min) {
        locked_ = true;
      }
    } else {
      if (!enough_rms_unlock || coherence_ < cfg_.unlock_coherence_min) {
        locked_ = false;
      }
    }

    // ---------- 5) Phase detector ----------
    phase_error_rad_ = std::atan2(q_lp_, i_lp_);

    // Clamp very large errors to keep loop tame during acquisition / disturbances.
    const Real max_phase_err = Real(1.2); // ~69 degrees
    phase_error_rad_ = clamp(phase_error_rad_, -max_phase_err, max_phase_err);

    // ---------- 6) PI loop on oscillator frequency ----------
    //
    // Type-II PLL style:
    //   phi_dot   = omega + Kp * e
    //   omega_dot = Ki * e
    //
    // Continuous-time gains for loop natural frequency wn:
    //   Kp = 2*zeta*wn
    //   Ki = wn^2
    //
    // Here the phase detector and NCO gains are taken as ~1.
    const Real wn = kTwoPi * cfg_.loop_bandwidth_hz;
    const Real kp = Real(2) * cfg_.loop_damping * wn;
    const Real ki = wn * wn;

    // Adaptation gain:
    // - full authority when confidence is high
    // - much weaker when confidence is poor
    Real adapt_gain = confidence_;
    if (!locked_) {
      adapt_gain *= Real(0.50);
    }

    // Integrate omega with rate limiting and band limits.
    Real omega_candidate = omega_rad_s_ + (ki * adapt_gain * phase_error_rad_) * dt;

    // Optional coarse assist:
    // when confidence is poor or not locked, gently pull toward coarse estimate.
    if (cfg_.enable_coarse_assist && coarse_valid_ && coarse_age_s_ <= cfg_.coarse_timeout_s) {
      const Real assist_alpha = Real(1) - alphaFromTau(cfg_.coarse_pull_tau_s, dt);
      const Real assist_gain  = (locked_ ? Real(0.15) : Real(1.0)) * (Real(1) - confidence_);
      const Real coarse_omega = hzToRad(coarse_freq_hz_);
      omega_candidate += assist_gain * assist_alpha * (coarse_omega - omega_candidate);
    }

    // If confidence is very poor, slowly recenter toward nominal frequency.
    if (confidence_ < Real(0.10)) {
      const Real recenter_alpha = Real(1) - alphaFromTau(cfg_.recenter_tau_s, dt);
      omega_candidate += recenter_alpha * (nominal_omega_rad_s_ - omega_candidate);
    }

    // Slew-rate limit frequency for robustness.
    const Real max_domega = hzToRad(cfg_.max_dfdt_hz_per_s) * dt;
    omega_candidate = clamp(omega_candidate,
                            omega_rad_s_ - max_domega,
                            omega_rad_s_ + max_domega);

    // Hard band constraints.
    const Real wmin = hzToRad(cfg_.f_min_hz);
    const Real wmax = hzToRad(cfg_.f_max_hz);
    omega_rad_s_ = clamp(omega_candidate, wmin, wmax);

    // Advance phase with proportional correction term.
    phase_rad_ += (omega_rad_s_ + kp * adapt_gain * phase_error_rad_) * dt;
    phase_rad_ = wrapPi(phase_rad_);

    // ---------- 7) Output frequency ----------
    raw_freq_hz_ = radToHz(omega_rad_s_);

    const Real a_out = alphaFromTau(cfg_.output_smooth_tau_s, dt);
    smooth_freq_hz_ = mixExp(smooth_freq_hz_, raw_freq_hz_, a_out);
    smooth_freq_hz_ = clamp(smooth_freq_hz_, cfg_.f_min_hz, cfg_.f_max_hz);

    time_s_ += dt;
  }

  void updateCoarseAssist(Real y, Real dt) {
    coarse_age_s_ += dt;

    if (!cfg_.enable_coarse_assist || !isFinite(y) || !isFinite(rms_)) {
      return;
    }

    // Hysteretic threshold tied to current RMS.
    const Real thr = maxValue(cfg_.coarse_hysteresis_frac * rms_, Real(1e-9));

    // Arm on a sufficiently negative excursion.
    if (y <= -thr) {
      coarse_seen_negative_ = true;
    }

    // When later crossing high enough positive level, record one event.
    // Consecutive such events are approximately one period apart.
    if (coarse_seen_negative_ && y >= +thr) {
      if (coarse_last_event_t_s_ >= Real(0)) {
        const Real period_s = time_s_ - coarse_last_event_t_s_;
        if (period_s > Real(0)) {
          const Real f_meas = Real(1) / period_s;
          if (f_meas >= cfg_.f_min_hz && f_meas <= cfg_.f_max_hz) {
            const Real a_coarse = alphaFromTau(cfg_.coarse_smooth_tau_s, dt);
            if (!coarse_valid_) {
              coarse_freq_hz_ = f_meas;
            } else {
              coarse_freq_hz_ = mixExp(coarse_freq_hz_, f_meas, a_coarse);
            }
            coarse_valid_ = true;
            coarse_age_s_ = Real(0);
          }
        }
      }

      coarse_last_event_t_s_ = time_s_;
      coarse_seen_negative_  = false;
    }

    // Time out stale coarse estimate.
    if (coarse_age_s_ > cfg_.coarse_timeout_s) {
      coarse_valid_ = false;
    }
  }

  static Config sanitizeConfig(Config c) {
    // Basic band sanity
    c.f_min_hz = maxValue(c.f_min_hz, Real(1e-4));
    c.f_max_hz = maxValue(c.f_max_hz, c.f_min_hz + Real(1e-4));
    c.f_init_hz = clamp(c.f_init_hz, c.f_min_hz, c.f_max_hz);

    // Preprocessing sanity
    c.pre_hp_hz = clamp(c.pre_hp_hz, Real(0), c.f_max_hz);
    c.pre_lp_hz = maxValue(c.pre_lp_hz, c.pre_hp_hz + Real(1e-4));

    // Reasonable default if user accidentally sets LP below band top
    if (c.pre_lp_hz < c.f_max_hz) {
      c.pre_lp_hz = c.f_max_hz * Real(1.25);
    }

    c.demod_lp_hz         = maxValue(c.demod_lp_hz, Real(1e-4));
    c.loop_bandwidth_hz   = maxValue(c.loop_bandwidth_hz, Real(1e-4));
    c.loop_damping        = maxValue(c.loop_damping, Real(0.05));
    c.output_smooth_tau_s = maxValue(c.output_smooth_tau_s, Real(1e-4));
    c.power_tau_s         = maxValue(c.power_tau_s, Real(1e-4));
    c.confidence_tau_s    = maxValue(c.confidence_tau_s, Real(1e-4));
    c.lock_rms_min        = maxValue(c.lock_rms_min, Real(1e-9));
    c.lock_coherence_min  = clamp(c.lock_coherence_min, Real(0), Real(1));
    c.unlock_coherence_min = clamp(c.unlock_coherence_min, Real(0), c.lock_coherence_min);
    c.max_dfdt_hz_per_s   = maxValue(c.max_dfdt_hz_per_s, Real(1e-5));
    c.recenter_tau_s      = maxValue(c.recenter_tau_s, Real(1e-3));
    c.max_internal_step_s = maxValue(c.max_internal_step_s, Real(1e-4));

    c.coarse_hysteresis_frac = clamp(c.coarse_hysteresis_frac, Real(0.01), Real(0.95));
    c.coarse_smooth_tau_s    = maxValue(c.coarse_smooth_tau_s, Real(1e-3));
    c.coarse_pull_tau_s      = maxValue(c.coarse_pull_tau_s, Real(1e-3));
    c.coarse_timeout_s       = maxValue(c.coarse_timeout_s, Real(1e-3));

    return c;
  }

  static Real alphaFromCutoffHz(Real cutoff_hz, Real dt) {
    // Exact one-pole discrete alpha from analog cutoff:
    // alpha = exp(-2*pi*fc*dt)
    if (!(cutoff_hz > Real(0)) || !(dt > Real(0))) {
      return Real(0);
    }
    const Real x = -kTwoPi * cutoff_hz * dt;
    return safeExp(x);
  }

  static Real alphaFromTau(Real tau_s, Real dt) {
    // Exact one-pole discrete alpha from time constant tau:
    // alpha = exp(-dt/tau)
    if (!(tau_s > Real(0)) || !(dt > Real(0))) {
      return Real(0);
    }
    const Real x = -dt / tau_s;
    return safeExp(x);
  }

  static Real mixExp(Real state, Real input, Real alpha) {
    // state <- alpha*state + (1-alpha)*input
    return alpha * state + (Real(1) - alpha) * input;
  }

  static Real wrapPi(Real x) {
    while (x >  kPi) x -= kTwoPi;
    while (x < -kPi) x += kTwoPi;
    return x;
  }

  static Real hzToRad(Real f_hz) {
    return kTwoPi * f_hz;
  }

  static Real radToHz(Real w_rad_s) {
    return w_rad_s / kTwoPi;
  }

  static Real clamp(Real x, Real lo, Real hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
  }

  static Real maxValue(Real a, Real b) {
    return (a > b) ? a : b;
  }

  static bool isFinite(Real x) {
    return std::isfinite(static_cast<double>(x)) != 0;
  }

  static Real safeExp(Real x) {
    // Prevent underflow/overflow nonsense in extreme cases.
    if (x < Real(-80)) return Real(0);
    if (x > Real(0))   return Real(1); // should not happen for our use, but keeps it bounded
    return std::exp(x);
  }

  static Real safeHypot(Real a, Real b) {
    // Stable sqrt(a^2 + b^2) without overflow/underflow drama.
    a = std::abs(a);
    b = std::abs(b);
    if (a < b) {
      const Real t = a;
      a = b;
      b = t;
    }
    if (a <= Real(0)) return Real(0);
    const Real r = b / a;
    return a * std::sqrt(Real(1) + r * r);
  }
};

#endif // WAVE_FREQUENCY_TRACKER_H
