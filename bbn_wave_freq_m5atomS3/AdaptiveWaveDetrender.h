#pragma once

#include <math.h>
#include <stdint.h>

class AdaptiveWaveDetrender {
public:
  struct Config {
    // Dominant wave-frequency search band [Hz].
    // Covers long swell through short chop by default.
    float init_wave_freq_hz = 0.12f;   // startup guess (~8.3 s period)
    float min_wave_freq_hz  = 0.02f;   // 50 s period
    float max_wave_freq_hz  = 1.20f;   // 0.83 s period

    // Slow baseline cutoff:
    //   f_baseline = clamp(baseline_cutoff_fraction * f_wave,
    //                      min_baseline_cutoff_hz,
    //                      max_baseline_cutoff_hz)
    //
    // Larger fraction => stronger drift removal, more risk of eating the wave.
    float baseline_cutoff_fraction = 0.25f;
    float min_baseline_cutoff_hz   = 0.003f;
    float max_baseline_cutoff_hz   = 0.25f;

    // Smoothing time constant for learned frequency [s].
    float freq_smooth_tau_s = 12.0f;

    // Internal slope/velocity-proxy processing.
    // Slope is computed as dx/dt and used only for frequency learning.
    float slope_lpf_tau_s = 0.20f;
    float slope_rms_tau_s = 8.0f;

    // Schmitt threshold on filtered slope:
    //   threshold = clamp(threshold_rms_fraction * slope_rms,
    //                     min_slope_threshold_abs,
    //                     max_slope_threshold_abs)
    //
    // Units are input-units/second:
    //   meters input  -> m/s threshold
    //   centimeters   -> cm/s threshold
    float threshold_rms_fraction  = 0.30f;
    float min_slope_threshold_abs = 0.001f;
    float max_slope_threshold_abs = 1.0e9f;

    // Wait this long before internal frequency learning starts [s].
    float startup_hold_s = 2.0f;

    // Frequency validity timeout, measured in cycles of current wave frequency.
    float freq_timeout_cycles = 3.0f;

    // Optional cleanup on residual wave channel.
    // This does NOT affect baseline subtraction.
    // If disabled, wave_clean == wave_raw.
    bool  enable_wave_cleanup = false;
    float cleanup_cutoff_fraction = 3.0f;  // cutoff = cleanup_cutoff_fraction * f_wave
    float min_cleanup_cutoff_hz   = 0.05f;
    float max_cleanup_cutoff_hz   = 3.0f;

    // dt guards
    float min_dt_s = 1.0e-4f;
    float max_dt_s = 2.0f;

    // Optional hard clamp on outputs. <= 0 disables.
    float output_abs_limit = 0.0f;
  };

  struct Output {
    float input = 0.0f;                // latest input sample
    float baseline_slow = 0.0f;        // the actual smooth baseline being subtracted
    float wave_raw = 0.0f;             // input - baseline_slow
    float wave_clean = 0.0f;           // optional cleaned version of wave_raw
    float wave_freq_hz = 0.0f;         // learned dominant wave frequency
    float wave_period_s = 0.0f;        // 1 / wave_freq_hz
    float baseline_cutoff_hz = 0.0f;   // current adaptive baseline cutoff
    float baseline_tau_s = 0.0f;       // equivalent time constant of baseline filter
    float cleanup_cutoff_hz = 0.0f;    // current cleanup cutoff (0 if disabled)
    float slope_rms = 0.0f;            // RMS of filtered slope proxy
    float slope_threshold = 0.0f;      // current Schmitt threshold
    bool  freq_valid = false;          // learned frequency currently fresh
    int8_t schmitt_state = 0;          // -1, 0, +1
  };

  AdaptiveWaveDetrender() {
    sanitizeConfig_(cfg_);
    reset();
  }

  explicit AdaptiveWaveDetrender(const Config& cfg) : cfg_(cfg) {
    sanitizeConfig_(cfg_);
    reset();
  }

  void setConfig(const Config& cfg) {
    cfg_ = cfg;
    sanitizeConfig_(cfg_);
    f_used_hz_ = clampf_(f_used_hz_, cfg_.min_wave_freq_hz, cfg_.max_wave_freq_hz);
    last_output_ = buildOutput_();
  }

  const Config& config() const { return cfg_; }

  void reset() {
    initialized_ = false;
    time_s_ = 0.0f;

    current_input_ = 0.0f;
    x_prev_ = 0.0f;

    baseline_slow_ = 0.0f;
    wave_clean_state_ = 0.0f;

    slope_filt_ = 0.0f;
    slope_prev_ = 0.0f;
    slope_rms2_ = 0.0f;

    f_used_hz_ = cfg_.init_wave_freq_hz;

    schmitt_state_ = 0;
    have_last_pos_cross_ = false;
    have_last_neg_cross_ = false;
    last_pos_cross_t_ = 0.0f;
    last_neg_cross_t_ = 0.0f;

    valid_period_count_ = 0;
    last_valid_freq_t_ = -1.0e30f;

    last_wave_raw_ = 0.0f;
    last_wave_clean_ = 0.0f;

    last_output_ = buildOutput_();
  }

  void reset(float x0) {
    sanitizeConfig_(cfg_);

    initialized_ = true;
    time_s_ = 0.0f;

    current_input_ = x0;
    x_prev_ = x0;

    baseline_slow_ = x0;
    wave_clean_state_ = 0.0f;

    slope_filt_ = 0.0f;
    slope_prev_ = 0.0f;
    slope_rms2_ = 0.0f;

    f_used_hz_ = cfg_.init_wave_freq_hz;

    schmitt_state_ = 0;
    have_last_pos_cross_ = false;
    have_last_neg_cross_ = false;
    last_pos_cross_t_ = 0.0f;
    last_neg_cross_t_ = 0.0f;

    valid_period_count_ = 0;
    last_valid_freq_t_ = -1.0e30f;

    last_wave_raw_ = 0.0f;
    last_wave_clean_ = 0.0f;

    last_output_ = buildOutput_();
  }

  // Main update: learn frequency internally from the input signal.
  Output update(float x, float dt_s) {
    return updateImpl_(x, dt_s, 0.0f, false);
  }

  // Optional overload: blend in an external dominant frequency estimate [Hz].
  // The baseline cutoff will then use the blended frequency estimate.
  Output update(float x, float dt_s, float external_wave_freq_hz, bool external_valid) {
    return updateImpl_(x, dt_s, external_wave_freq_hz, external_valid);
  }

  float currentBaselineSlow() const { return baseline_slow_; }
  float currentWaveRaw() const { return last_wave_raw_; }
  float currentWaveClean() const { return last_wave_clean_; }
  float currentWaveFreqHz() const { return f_used_hz_; }
  float currentWavePeriodS() const { return 1.0f / maxf_(f_used_hz_, 1.0e-6f); }
  float currentBaselineCutoffHz() const { return currentBaselineCutoffHz_(); }
  bool frequencyValid() const { return isFrequencyValid_(); }
  const Output& lastOutput() const { return last_output_; }

private:
  static constexpr float kPi_ = 3.14159265358979323846f;

  Config cfg_;

  bool initialized_ = false;
  float time_s_ = 0.0f;

  float current_input_ = 0.0f;
  float x_prev_ = 0.0f;

  float baseline_slow_ = 0.0f;   // actual smooth baseline that is subtracted
  float wave_clean_state_ = 0.0f;

  float slope_filt_ = 0.0f;
  float slope_prev_ = 0.0f;
  float slope_rms2_ = 0.0f;

  float f_used_hz_ = 0.12f;

  int8_t schmitt_state_ = 0;
  bool have_last_pos_cross_ = false;
  bool have_last_neg_cross_ = false;
  float last_pos_cross_t_ = 0.0f;
  float last_neg_cross_t_ = 0.0f;

  int valid_period_count_ = 0;
  float last_valid_freq_t_ = -1.0e30f;

  float last_wave_raw_ = 0.0f;
  float last_wave_clean_ = 0.0f;

  Output last_output_;

private:
  static bool isFinite_(float x) {
    return isfinite(x) != 0;
  }

  static float clampf_(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
  }

  static float maxf_(float a, float b) {
    return (a > b) ? a : b;
  }

  static float minf_(float a, float b) {
    return (a < b) ? a : b;
  }

  static float cutoffToTau_(float cutoff_hz) {
    cutoff_hz = maxf_(cutoff_hz, 1.0e-6f);
    return 1.0f / (2.0f * kPi_ * cutoff_hz);
  }

  static float expAlphaFromTau_(float dt_s, float tau_s) {
    tau_s = maxf_(tau_s, 1.0e-6f);
    return expf(-dt_s / tau_s);
  }

  static float interpCrossTime_(float t_prev, float t_curr,
                                float y_prev, float y_curr,
                                float level) {
    const float dy = y_curr - y_prev;
    if (fabsf(dy) < 1.0e-12f) {
      return t_curr;
    }
    float frac = (level - y_prev) / dy;
    frac = clampf_(frac, 0.0f, 1.0f);
    return t_prev + frac * (t_curr - t_prev);
  }

  void sanitizeConfig_(Config& c) {
    if (!isFinite_(c.init_wave_freq_hz) || c.init_wave_freq_hz <= 0.0f) c.init_wave_freq_hz = 0.12f;
    if (!isFinite_(c.min_wave_freq_hz)  || c.min_wave_freq_hz  <= 0.0f) c.min_wave_freq_hz  = 0.02f;
    if (!isFinite_(c.max_wave_freq_hz)  || c.max_wave_freq_hz  <= c.min_wave_freq_hz) {
      c.max_wave_freq_hz = maxf_(1.20f, 2.0f * c.min_wave_freq_hz);
    }
    c.init_wave_freq_hz = clampf_(c.init_wave_freq_hz, c.min_wave_freq_hz, c.max_wave_freq_hz);

    if (!isFinite_(c.baseline_cutoff_fraction) || c.baseline_cutoff_fraction <= 0.0f) {
      c.baseline_cutoff_fraction = 0.25f;
    }
    if (!isFinite_(c.min_baseline_cutoff_hz) || c.min_baseline_cutoff_hz <= 0.0f) {
      c.min_baseline_cutoff_hz = 0.003f;
    }
    if (!isFinite_(c.max_baseline_cutoff_hz) || c.max_baseline_cutoff_hz < c.min_baseline_cutoff_hz) {
      c.max_baseline_cutoff_hz = maxf_(0.25f, c.min_baseline_cutoff_hz);
    }

    if (!isFinite_(c.freq_smooth_tau_s) || c.freq_smooth_tau_s <= 0.0f) c.freq_smooth_tau_s = 12.0f;
    if (!isFinite_(c.slope_lpf_tau_s) || c.slope_lpf_tau_s <= 0.0f) c.slope_lpf_tau_s = 0.20f;
    if (!isFinite_(c.slope_rms_tau_s) || c.slope_rms_tau_s <= 0.0f) c.slope_rms_tau_s = 8.0f;

    if (!isFinite_(c.threshold_rms_fraction) || c.threshold_rms_fraction <= 0.0f) {
      c.threshold_rms_fraction = 0.30f;
    }
    if (!isFinite_(c.min_slope_threshold_abs) || c.min_slope_threshold_abs < 0.0f) {
      c.min_slope_threshold_abs = 0.001f;
    }
    if (!isFinite_(c.max_slope_threshold_abs) || c.max_slope_threshold_abs < c.min_slope_threshold_abs) {
      c.max_slope_threshold_abs = c.min_slope_threshold_abs;
    }

    if (!isFinite_(c.startup_hold_s) || c.startup_hold_s < 0.0f) c.startup_hold_s = 2.0f;
    if (!isFinite_(c.freq_timeout_cycles) || c.freq_timeout_cycles <= 0.0f) c.freq_timeout_cycles = 3.0f;

    if (!isFinite_(c.cleanup_cutoff_fraction) || c.cleanup_cutoff_fraction <= 0.0f) {
      c.cleanup_cutoff_fraction = 3.0f;
    }
    if (!isFinite_(c.min_cleanup_cutoff_hz) || c.min_cleanup_cutoff_hz <= 0.0f) {
      c.min_cleanup_cutoff_hz = 0.05f;
    }
    if (!isFinite_(c.max_cleanup_cutoff_hz) || c.max_cleanup_cutoff_hz < c.min_cleanup_cutoff_hz) {
      c.max_cleanup_cutoff_hz = maxf_(3.0f, c.min_cleanup_cutoff_hz);
    }

    if (!isFinite_(c.min_dt_s) || c.min_dt_s <= 0.0f) c.min_dt_s = 1.0e-4f;
    if (!isFinite_(c.max_dt_s) || c.max_dt_s < c.min_dt_s) c.max_dt_s = maxf_(2.0f, c.min_dt_s);

    if (!isFinite_(c.output_abs_limit)) c.output_abs_limit = 0.0f;
  }

  float sanitizeDt_(float dt_s) const {
    if (!isFinite_(dt_s)) return cfg_.min_dt_s;
    return clampf_(dt_s, cfg_.min_dt_s, cfg_.max_dt_s);
  }

  float currentBaselineCutoffHz_() const {
    float fc = cfg_.baseline_cutoff_fraction * f_used_hz_;
    fc = maxf_(fc, cfg_.min_baseline_cutoff_hz);
    fc = minf_(fc, cfg_.max_baseline_cutoff_hz);
    return fc;
  }

  float currentCleanupCutoffHz_() const {
    if (!cfg_.enable_wave_cleanup) return 0.0f;
    float fc = cfg_.cleanup_cutoff_fraction * f_used_hz_;
    fc = maxf_(fc, cfg_.min_cleanup_cutoff_hz);
    fc = minf_(fc, cfg_.max_cleanup_cutoff_hz);
    return fc;
  }

  bool isFrequencyValid_() const {
    if (valid_period_count_ < 2) return false;
    const float f = maxf_(f_used_hz_, 1.0e-4f);
    const float timeout_s = cfg_.freq_timeout_cycles / f;
    return (time_s_ - last_valid_freq_t_) <= timeout_s;
  }

  void acceptFrequencyMeasurement_(float f_meas_hz, float meas_interval_s) {
    if (!isFinite_(f_meas_hz) || f_meas_hz <= 0.0f) return;

    f_meas_hz = clampf_(f_meas_hz, cfg_.min_wave_freq_hz, cfg_.max_wave_freq_hz);

    const float dt_f = maxf_(meas_interval_s, cfg_.min_dt_s);
    const float a_f = expAlphaFromTau_(dt_f, cfg_.freq_smooth_tau_s);
    f_used_hz_ = a_f * f_used_hz_ + (1.0f - a_f) * f_meas_hz;

    last_valid_freq_t_ = time_s_;
    if (valid_period_count_ < 1000000) ++valid_period_count_;
  }

  void blendExternalFrequency_(float f_ext_hz, float dt_s) {
    if (!isFinite_(f_ext_hz) || f_ext_hz <= 0.0f) return;

    f_ext_hz = clampf_(f_ext_hz, cfg_.min_wave_freq_hz, cfg_.max_wave_freq_hz);
    const float a_f = expAlphaFromTau_(dt_s, cfg_.freq_smooth_tau_s);
    f_used_hz_ = a_f * f_used_hz_ + (1.0f - a_f) * f_ext_hz;

    last_valid_freq_t_ = time_s_;
    if (valid_period_count_ < 1000000) ++valid_period_count_;
  }

  void updateFrequencyFromSlope_(float y_prev, float y_curr, float thr, float dt_s) {
    if (time_s_ < cfg_.startup_hold_s) {
      return;
    }

    const float t_prev = time_s_ - dt_s;
    const float t_curr = time_s_;

    const float min_period_s = 1.0f / cfg_.max_wave_freq_hz;
    const float max_period_s = 1.0f / cfg_.min_wave_freq_hz;

    // Same-sign Schmitt crossing at +threshold
    if (schmitt_state_ != +1 && y_prev < +thr && y_curr >= +thr) {
      const float t_cross = interpCrossTime_(t_prev, t_curr, y_prev, y_curr, +thr);

      if (have_last_pos_cross_) {
        const float T = t_cross - last_pos_cross_t_;
        if (T >= min_period_s && T <= max_period_s) {
          acceptFrequencyMeasurement_(1.0f / T, T);
        }
      }

      last_pos_cross_t_ = t_cross;
      have_last_pos_cross_ = true;
      schmitt_state_ = +1;
      return;
    }

    // Same-sign Schmitt crossing at -threshold
    if (schmitt_state_ != -1 && y_prev > -thr && y_curr <= -thr) {
      const float t_cross = interpCrossTime_(t_prev, t_curr, y_prev, y_curr, -thr);

      if (have_last_neg_cross_) {
        const float T = t_cross - last_neg_cross_t_;
        if (T >= min_period_s && T <= max_period_s) {
          acceptFrequencyMeasurement_(1.0f / T, T);
        }
      }

      last_neg_cross_t_ = t_cross;
      have_last_neg_cross_ = true;
      schmitt_state_ = -1;
      return;
    }
  }

  Output updateImpl_(float x, float dt_s, float external_wave_freq_hz, bool external_valid) {
    if (!isFinite_(x)) {
      return last_output_;
    }

    dt_s = sanitizeDt_(dt_s);

    if (!initialized_) {
      reset(x);
      return last_output_;
    }

    time_s_ += dt_s;
    current_input_ = x;

    // Learn dominant wave frequency from slope/velocity proxy.
    const float slope_raw = (x - x_prev_) / dt_s;

    const float a_slope = expAlphaFromTau_(dt_s, cfg_.slope_lpf_tau_s);
    slope_filt_ = a_slope * slope_filt_ + (1.0f - a_slope) * slope_raw;

    const float a_rms = expAlphaFromTau_(dt_s, cfg_.slope_rms_tau_s);
    slope_rms2_ = a_rms * slope_rms2_ + (1.0f - a_rms) * (slope_filt_ * slope_filt_);
    if (slope_rms2_ < 0.0f) slope_rms2_ = 0.0f;

    const float slope_rms = sqrtf(slope_rms2_);
    float slope_thr = cfg_.threshold_rms_fraction * slope_rms;
    slope_thr = maxf_(slope_thr, cfg_.min_slope_threshold_abs);
    slope_thr = minf_(slope_thr, cfg_.max_slope_threshold_abs);

    updateFrequencyFromSlope_(slope_prev_, slope_filt_, slope_thr, dt_s);

    if (external_valid) {
      blendExternalFrequency_(external_wave_freq_hz, dt_s);
    }

    // Update the actual slow baseline that is subtracted.
    const float fc_base = currentBaselineCutoffHz_();
    const float a_base = expf(-2.0f * kPi_ * fc_base * dt_s);

    baseline_slow_ = a_base * baseline_slow_ + (1.0f - a_base) * x;

    float wave_raw = x - baseline_slow_;

    // Cleanup on the residual ONLY. Does not affect baseline_slow_.
    float wave_clean = wave_raw;

    if (cfg_.enable_wave_cleanup) {
      const float fc_clean = currentCleanupCutoffHz_();
      const float a_clean = expf(-2.0f * kPi_ * fc_clean * dt_s);
      wave_clean_state_ = a_clean * wave_clean_state_ + (1.0f - a_clean) * wave_raw;
      wave_clean = wave_clean_state_;
    } else {
      wave_clean_state_ = wave_raw;
    }

    if (cfg_.output_abs_limit > 0.0f) {
      wave_raw = clampf_(wave_raw, -cfg_.output_abs_limit, cfg_.output_abs_limit);
      wave_clean = clampf_(wave_clean, -cfg_.output_abs_limit, cfg_.output_abs_limit);
    }

    x_prev_ = x;
    slope_prev_ = slope_filt_;

    last_wave_raw_ = wave_raw;
    last_wave_clean_ = wave_clean;
    last_output_ = buildOutput_();
    return last_output_;
  }

  Output buildOutput_() const {
    Output out;
    out.input = current_input_;
    out.baseline_slow = baseline_slow_;
    out.wave_raw = last_wave_raw_;
    out.wave_clean = last_wave_clean_;
    out.wave_freq_hz = f_used_hz_;
    out.wave_period_s = 1.0f / maxf_(f_used_hz_, 1.0e-6f);
    out.baseline_cutoff_hz = currentBaselineCutoffHz_();
    out.baseline_tau_s = cutoffToTau_(out.baseline_cutoff_hz);
    out.cleanup_cutoff_hz = currentCleanupCutoffHz_();
    out.slope_rms = sqrtf(maxf_(0.0f, slope_rms2_));

    float thr = cfg_.threshold_rms_fraction * out.slope_rms;
    thr = maxf_(thr, cfg_.min_slope_threshold_abs);
    thr = minf_(thr, cfg_.max_slope_threshold_abs);
    out.slope_threshold = thr;

    out.freq_valid = isFrequencyValid_();
    out.schmitt_state = schmitt_state_;
    return out;
  }
};
