#pragma once

/*

  Copyright 2026, Mikhail Grushinskiy
    
  WaveSpectrumEstimator2
  
  Adaptation-focused wave spectrum estimator for Kalman3D_Wave_2 tuning.

  Design goals:
    - Fast enough for online adaptation (defaults: fs_raw=240 Hz, decim=15, block=128, hop=32)
    - Estimates displacement PSD S_eta(f) from vertical inertial acceleration a_z (m/s^2)
    - Provides direct helpers to split spectrum energy into K oscillator modes
    - Provides direct helper to convert mode displacement variance -> q_k (process intensity)

  Signal chain (raw sample rate):
      a_z(raw) -> HP2 -> HP2 -> LP2 -> decimate -> sliding block
      block -> detrend -> Hann -> Goertzel -> S_aa(f)
      deconvolve filters -> regularized 1/omega^4 -> S_eta(f)

  Units:
    - Input acceleration: m/s^2
    - Output displacement PSD S_eta: m^2/Hz
    - Mode variance: m^2
    - q_k from mode variance: m^2/s^5

*/

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

template<int Nfreq = 24, int Nblock = 128>
class EIGEN_ALIGN_MAX WaveSpectrumEstimator2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Vec = Eigen::Matrix<double, Nfreq, 1>;

  static constexpr double kG  = 9.80665;
  static constexpr double kPi = 3.1415926535897932384626433832795;

  struct Config {
    double fs_raw_hz      = 240.0; // IMU sample rate
    int    decim_factor   = 15;    // => fs = 16 Hz
    int    hop_decimated  = 32;    // => update every 2 s with Nblock=128
    bool   hann_enabled   = true;

    // Filter / inversion knobs
    double hp_f0_hz       = 0.025; // 4th-order HP (2x biquad) for bias suppression
    double reg_f0_hz      = 0.05;  // low-f regularization for 1/omega^4 inversion

    // PSD smoothing
    bool   psd_ema_enable = true;
    double psd_ema_alpha  = 0.18;  // 0.1..0.3 good range

    // Analysis band
    double f_min_hz       = 0.05;
    double f_transition_hz= 0.12;  // lower ~40% log-spaced, upper linear
    double f_max_hz       = 1.20;
  };

  explicit WaveSpectrumEstimator2(const Config& cfg = Config()) {
    configure(cfg);
  }

  void configure(const Config& cfg) {
    cfg_ = cfg;

    if (!(cfg_.fs_raw_hz > 1.0)) cfg_.fs_raw_hz = 240.0;
    if (cfg_.decim_factor < 1)   cfg_.decim_factor = 15;
    if (cfg_.hop_decimated < 1)  cfg_.hop_decimated = std::max(1, Nblock / 4);

    fs_raw_ = cfg_.fs_raw_hz;
    fs_     = fs_raw_ / double(cfg_.decim_factor);

    buildFrequencyGrid_();
    precomputeGoertzel_();
    buildWindow_();

    designHighpassBiquad_(hp1_, cfg_.hp_f0_hz, fs_raw_);
    designHighpassBiquad_(hp2_, cfg_.hp_f0_hz, fs_raw_);

    // anti-alias LP close to post-decimation Nyquist
    const double lp_cut = 0.45 * (fs_raw_ / (2.0 * cfg_.decim_factor));
    designLowpassBiquad_(lp_, lp_cut, fs_raw_);

    reset();
  }

  void reset() {
    buffer_.fill(0.0);
    write_index_ = 0;
    decim_counter_ = 0;
    decimated_samples_total_ = 0;
    decimated_since_spec_ = 0;
    warm_ = false;

    hp1_.reset(); hp2_.reset(); lp_.reset();

    last_psd_eta_.setZero();
    psd_ema_.setZero();
    have_psd_ema_ = false;
  }

  // Feed one raw vertical inertial acceleration sample [m/s^2].
  // Returns true when a NEW spectrum is available.
  bool processSample(double a_vert_inertial) {
    // Bias suppression (4th-order HP via 2 cascaded HP biquads)
    double x = hp1_.process(a_vert_inertial);
    x = hp2_.process(x);

    // Anti-alias LP before decimation
    x = lp_.process(x);

    // Decimate
    decim_counter_++;
    if (decim_counter_ < cfg_.decim_factor) return false;
    decim_counter_ = 0;

    // Push decimated sample into circular buffer
    buffer_[write_index_] = x;
    write_index_ = (write_index_ + 1) % Nblock;

    decimated_samples_total_++;
    decimated_since_spec_++;

    if (decimated_samples_total_ >= Nblock) warm_ = true;
    if (!warm_) return false;

    // Sliding-block update every hop_decimated samples
    if (decimated_since_spec_ >= cfg_.hop_decimated) {
      decimated_since_spec_ = 0;
      computeSpectrum_();
      return true;
    }

    return false;
  }

  bool ready() const { return warm_; }

  // Effective decimated sample rate
  double sampleRateDecimatedHz() const { return fs_; }
  double blockDurationSec() const { return (fs_ > 0.0) ? (double(Nblock) / fs_) : 0.0; }
  double updatePeriodSec() const { return (fs_ > 0.0) ? (double(cfg_.hop_decimated) / fs_) : 0.0; }

  // Spectrum outputs
  Vec getDisplacementSpectrum() const { return last_psd_eta_; }          // S_eta [m^2/Hz]
  std::array<double, Nfreq> getFrequencies() const { return freqs_; }    // [Hz]
  std::array<double, Nfreq> getBinWidths() const { return df_; }         // [Hz]

  // Significant wave height estimate from S_eta
  double computeHs() const {
    double m0 = 0.0;
    for (int i = 0; i < Nfreq; ++i) {
      m0 += std::max(0.0, last_psd_eta_[i]) * std::max(0.0, df_[i]);
    }
    return 4.0 * std::sqrt(std::max(0.0, m0));
  }

  // Peak frequency of displacement spectrum using log-parabolic interpolation
  double estimatePeakFrequencyHz() const {
    int k = 0;
    double vmax = -1.0;
    for (int i = 0; i < Nfreq; ++i) {
      if (last_psd_eta_[i] > vmax) { vmax = last_psd_eta_[i]; k = i; }
    }

    if (k <= 0 || k >= (Nfreq - 1)) return freqs_[k];

    const double f0 = freqs_[k - 1], f1 = freqs_[k], f2 = freqs_[k + 1];
    if (!(f0 > 0.0 && f1 > 0.0 && f2 > 0.0)) return f1;

    const double x0 = std::log(f0), x1 = std::log(f1), x2 = std::log(f2);
    const double y0 = safeLog_(last_psd_eta_[k - 1]);
    const double y1 = safeLog_(last_psd_eta_[k]);
    const double y2 = safeLog_(last_psd_eta_[k + 1]);

    const double d01 = x0 - x1;
    const double d02 = x0 - x2;
    const double d12 = x1 - x2;

    if (std::fabs(d01 * d02 * d12) < 1e-18) return f1;

    const double L0a = 1.0 / (d01 * d02);
    const double L1a = 1.0 / ((x1 - x0) * (x1 - x2));
    const double L2a = 1.0 / ((x2 - x0) * (x2 - x1));

    const double a = y0 * L0a + y1 * L1a + y2 * L2a;
    const double b = y0 * (-(x1 + x2) * L0a)
                   + y1 * (-(x0 + x2) * L1a)
                   + y2 * (-(x0 + x1) * L2a);

    if (!(a < 0.0)) return f1;

    const double xpk = -b / (2.0 * a);
    const double fpk = std::exp(xpk);

    if (!(fpk > std::min(f0, f2) && fpk < std::max(f0, f2))) return f1;
    return fpk;
  }

  /*
    Split displacement spectrum energy into K overlapping mode bands.

    Outputs:
      - mode_f_hz[k]    : mode center frequencies [Hz]
      - mode_var_m2[k]  : displacement variance assigned to each mode [m^2]

    This is the direct quantity needed for:
      q_k = 4*zeta_k*omega_k^3 * mode_var_m2[k]
  */
  template<int K>
  void estimateModeDisplacementVariance(std::array<float, K>& mode_f_hz,
                                        std::array<double, K>& mode_var_m2,
                                        float min_mode_hz,
                                        float max_mode_hz) const
  {
    mode_f_hz.fill(0.0f);
    mode_var_m2.fill(0.0);

    if (!warm_) return;

    // Total displacement variance and log-frequency moments
    double m0 = 0.0;
    double mu_logf = 0.0;

    for (int i = 0; i < Nfreq; ++i) {
      const double fi = freqs_[i];
      const double Ei = std::max(0.0, last_psd_eta_[i]) * std::max(0.0, df_[i]); // [m^2]
      if (!(fi > 0.0) || !std::isfinite(Ei)) continue;
      m0 += Ei;
      mu_logf += Ei * std::log(fi);
    }

    if (!(m0 > 1e-12)) return;
    mu_logf /= m0;

    double var_logf = 0.0;
    for (int i = 0; i < Nfreq; ++i) {
      const double fi = freqs_[i];
      const double Ei = std::max(0.0, last_psd_eta_[i]) * std::max(0.0, df_[i]);
      if (!(fi > 0.0) || !std::isfinite(Ei)) continue;
      const double d = std::log(fi) - mu_logf;
      var_logf += Ei * d * d;
    }
    var_logf /= m0;

    const double sig_logf = std::sqrt(std::max(0.0, var_logf));
    const double fp_est = estimatePeakFrequencyHz();
    const double center_f = (std::isfinite(fp_est) && fp_est > 0.0) ? fp_est : std::exp(mu_logf);

    const double minf = std::max(1e-3, double(min_mode_hz));
    const double maxf = std::max(minf + 1e-3, double(max_mode_hz));

    // Adaptive spacing / width on log-frequency axis
    const double dlog = std::clamp(0.85 * sig_logf, 0.18, 0.55);
    const double sigma_band = std::max(0.22, 0.70 * dlog);

    const double x_center = std::log(std::clamp(center_f, minf, maxf));
    const double mid = 0.5 * double(K - 1);

    // Mode centers
    for (int k = 0; k < K; ++k) {
      const double xk = x_center + (double(k) - mid) * dlog;
      const double fk = std::exp(xk);
      mode_f_hz[k] = float(std::clamp(fk, minf, maxf));
    }

    // Soft bin -> mode energy split (Gaussian weights on log-f)
    for (int i = 0; i < Nfreq; ++i) {
      const double fi = freqs_[i];
      if (!(fi > 0.0)) continue;

      const double Ei = std::max(0.0, last_psd_eta_[i]) * std::max(0.0, df_[i]); // [m^2]
      if (!(Ei > 0.0) || !std::isfinite(Ei)) continue;

      const double xi = std::log(fi);

      std::array<double, K> w{};
      double wsum = 0.0;

      for (int k = 0; k < K; ++k) {
        const double xk = std::log(std::max(1e-6f, mode_f_hz[k]));
        const double u = (xi - xk) / sigma_band;
        const double wk = std::exp(-0.5 * u * u);
        w[k] = wk;
        wsum += wk;
      }

      if (wsum <= 1e-18) continue;
      const double inv = 1.0 / wsum;
      for (int k = 0; k < K; ++k) {
        mode_var_m2[k] += Ei * (w[k] * inv);
      }
    }
  }

  /*
    Convenience helper: directly estimate per-mode q_k (vertical process intensity)
    using the stationary variance relation:

      var_p = q / (4*zeta*omega^3)   =>   q = 4*zeta*omega^3*var_p

    Inputs:
      - zeta_mode[k] : damping ratios for each mode
      - q_gain       : global gain fudge (1.0 nominal)
      - q_min/q_max  : clamps to avoid spikes
  */
  template<int K>
  void estimateModeQz(std::array<float, K>& mode_f_hz,
                      std::array<float, K>& qz_mode,
                      const std::array<float, K>& zeta_mode,
                      float min_mode_hz,
                      float max_mode_hz,
                      float q_gain = 1.0f,
                      float q_min = 1e-8f,
                      float q_max = 50.0f) const
  {
    std::array<double, K> var_mode{};
    estimateModeDisplacementVariance<K>(mode_f_hz, var_mode, min_mode_hz, max_mode_hz);

    for (int k = 0; k < K; ++k) {
      const double fk = std::max(1e-6f, mode_f_hz[k]);
      const double zt = std::max(1e-5f, zeta_mode[k]);
      const double w  = 2.0 * kPi * fk;

      double q = 4.0 * zt * w * w * w * std::max(0.0, var_mode[k]); // [m^2/s^5]
      if (!std::isfinite(q)) q = 0.0;
      q *= std::max(0.0f, q_gain);
      q = std::clamp(q, double(q_min), double(q_max));

      qz_mode[k] = static_cast<float>(q);
    }
  }

  // Runtime knobs
  void setRegularizationF0Hz(double f0_hz) {
    cfg_.reg_f0_hz = std::max(1e-6, f0_hz);
  }

  void setHighpassF0Hz(double f0_hz) {
    cfg_.hp_f0_hz = std::max(0.0, f0_hz);
    designHighpassBiquad_(hp1_, cfg_.hp_f0_hz, fs_raw_);
    designHighpassBiquad_(hp2_, cfg_.hp_f0_hz, fs_raw_);
  }

  void setPsdEma(bool enable, double alpha = 0.18) {
    cfg_.psd_ema_enable = enable;
    cfg_.psd_ema_alpha  = std::clamp(alpha, 0.01, 1.0);
  }

private:
  // -------- Small biquad (TDF-II transposed) --------
  struct Biquad {
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;
    double z1 = 0.0, z2 = 0.0;

    inline double process(double x) {
      const double y = b0 * x + z1;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      return y;
    }

    inline void reset() { z1 = 0.0; z2 = 0.0; }
  };

  static constexpr double kBiquadQ_ = 0.7071067811865476;

  // -------- Internal helpers --------
  inline double safeLog_(double v) const {
    return std::log(std::max(v, 1e-18));
  }

  void buildFrequencyGrid_() {
    // Hybrid grid: low-frequency log spacing + upper linear spacing
    const double fmin = std::max(0.01, cfg_.f_min_hz);
    const double ftr  = std::max(fmin * 1.1, cfg_.f_transition_hz);
    const double fmax = std::max(ftr + 0.05, cfg_.f_max_hz);

    int n_log = int(std::round(double(Nfreq) * 0.40));
    n_log = std::clamp(n_log, 2, Nfreq - 2);
    const int n_lin = Nfreq - n_log;

    // edges (Nfreq+1)
    for (int i = 0; i <= n_log; ++i) {
      const double t = double(i) / double(n_log);
      f_edges_[i] = fmin * std::pow(ftr / fmin, t);
    }
    for (int i = 1; i <= n_lin; ++i) {
      const double t = double(i) / double(n_lin);
      f_edges_[n_log + i] = ftr + t * (fmax - ftr);
    }

    // centers + widths
    for (int i = 0; i < Nfreq; ++i) {
      if (i < n_log) {
        freqs_[i] = std::sqrt(f_edges_[i] * f_edges_[i + 1]); // geometric center
      } else {
        freqs_[i] = 0.5 * (f_edges_[i] + f_edges_[i + 1]);     // arithmetic center
      }
      df_[i] = std::max(1e-9, f_edges_[i + 1] - f_edges_[i]);
    }
  }

  void precomputeGoertzel_() {
    for (int i = 0; i < Nfreq; ++i) {
      const double omega = 2.0 * kPi * freqs_[i] / fs_; // rad/sample at decimated rate
      goertzel_coeff_[i] = 2.0 * std::cos(omega);
    }
  }

  void buildWindow_() {
    double sumsq = 0.0;
    for (int n = 0; n < Nblock; ++n) {
      double w = 1.0;
      if (cfg_.hann_enabled) {
        w = 0.5 * (1.0 - std::cos(2.0 * kPi * double(n) / double(Nblock - 1)));
      }
      window_[n] = w;
      sumsq += w * w;
    }
    window_sum_sq_ = std::max(1e-12, sumsq);
  }

  void designLowpassBiquad_(Biquad& bq, double f_cut_hz, double Fs) {
    const double fmax = 0.49 * Fs;
    const double fc = std::clamp(f_cut_hz, 1e-6, fmax);
    const double K  = std::tan(kPi * (fc / Fs));
    const double norm = 1.0 / (1.0 + K / kBiquadQ_ + K * K);

    bq.b0 = K * K * norm;
    bq.b1 = 2.0 * bq.b0;
    bq.b2 = bq.b0;
    bq.a1 = 2.0 * (K * K - 1.0) * norm;
    bq.a2 = (1.0 - K / kBiquadQ_ + K * K) * norm;
    bq.reset();
  }

  void designHighpassBiquad_(Biquad& bq, double f_cut_hz, double Fs) {
    if (!(f_cut_hz > 0.0)) {
      // bypass
      bq.b0 = 1.0; bq.b1 = 0.0; bq.b2 = 0.0;
      bq.a1 = 0.0; bq.a2 = 0.0;
      bq.reset();
      return;
    }

    const double fmax = 0.49 * Fs;
    const double fc = std::clamp(f_cut_hz, 1e-6, fmax);
    const double K  = std::tan(kPi * (fc / Fs));
    const double norm = 1.0 / (1.0 + K / kBiquadQ_ + K * K);

    bq.b0 = 1.0 * norm;
    bq.b1 = -2.0 * norm;
    bq.b2 = 1.0 * norm;
    bq.a1 = 2.0 * (K * K - 1.0) * norm;
    bq.a2 = (1.0 - K / kBiquadQ_ + K * K) * norm;
    bq.reset();
  }

  inline double biquadMag2_(const Biquad& bq, double Omega_raw) const {
    const double c1 = std::cos(Omega_raw), s1 = std::sin(Omega_raw);
    const double c2 = std::cos(2.0 * Omega_raw), s2 = std::sin(2.0 * Omega_raw);

    const double num_re = bq.b0 + bq.b1 * c1 + bq.b2 * c2;
    const double num_im = -(bq.b1 * s1 + bq.b2 * s2);

    const double den_re = 1.0 + bq.a1 * c1 + bq.a2 * c2;
    const double den_im = -(bq.a1 * s1 + bq.a2 * s2);

    const double num2 = num_re * num_re + num_im * num_im;
    const double den2 = den_re * den_re + den_im * den_im;

    return num2 / std::max(den2, 1e-16);
  }

void computeSpectrum_() {
  // Read the latest Nblock samples from circular buffer (oldest -> newest)
  const int N = Nblock;
  const int start_idx = write_index_; // write_index_ points to oldest slot after wrap

  // Linear detrend statistics
  double sumx = 0.0, sumn = 0.0, sumn2 = 0.0, sumnx = 0.0;
  int idx = start_idx;
  for (int n = 0; n < N; ++n) {
    const double x = buffer_[idx];
    sumx  += x;
    sumn  += double(n);
    sumn2 += double(n) * double(n);
    sumnx += double(n) * x;
    idx = (idx + 1) % Nblock;
  }

  const double denom_lin = double(N) * sumn2 - sumn * sumn;
  const double a_lin = (std::fabs(denom_lin) > 1e-18)
                     ? ((double(N) * sumnx - sumn * sumx) / denom_lin)
                     : 0.0;
  const double b_lin = (sumx - a_lin * sumn) / double(N);

  // One-sided PSD scale [units^2/Hz] for periodogram with window
  const double base_scale = 2.0 / (fs_ * window_sum_sq_);

  // Low-frequency regularization knee also depends on block length
  // (keeps inversion from exploding below what the block can resolve)
  const double Tblk = double(N) / fs_;
  const double f_blk = 1.0 / std::max(1e-6, 6.0 * Tblk); // conservative
  const double f_knee = std::max(cfg_.reg_f0_hz, f_blk);

  // Constant regularization knee (IMPORTANT: do NOT tie to 0.6*f)
  const double wr = 2.0 * kPi * f_knee;
  const double wr4 = wr * wr * wr * wr;

  for (int i = 0; i < Nfreq; ++i) {
    const double f = freqs_[i];

    // ---- Goertzel on detrended+windowed block -> measured acceleration PSD S_aa_meas
    double s1 = 0.0, s2 = 0.0;
    int j = start_idx;
    for (int n = 0; n < N; ++n) {
      const double x = buffer_[j];
      const double xd = x - (a_lin * double(n) + b_lin);
      const double xw = xd * window_[n];

      const double sn = xw + goertzel_coeff_[i] * s1 - s2;
      s2 = s1;
      s1 = sn;

      j = (j + 1) % Nblock;
    }

    const double power = s1 * s1 + s2 * s2 - s1 * s2 * goertzel_coeff_[i];
    const double S_aa_meas = std::max(0.0, power * base_scale); // [m^2/s^4]/Hz

    // ---- Deconvolve HP/LP magnitude response at raw rate (guarded, but not over-suppressive)
    const double Om_raw = 2.0 * kPi * f / fs_raw_;
    const double H2 = biquadMag2_(hp1_, Om_raw)
                    * biquadMag2_(hp2_, Om_raw)
                    * biquadMag2_(lp_,  Om_raw);

    // Floor on |H|^2 to cap deconvolution gain instead of additive epsilon.
    // Example: H2_floor=1e-4 -> max deconv gain = 1e4 (40 dB)
    const double H2_floor = 1e-4;
    const double H2_eff = std::max(H2, H2_floor);

    double S_aa_true = S_aa_meas / H2_eff;
    if (!std::isfinite(S_aa_true) || S_aa_true < 0.0) S_aa_true = 0.0;

    // ---- Displacement conversion with stable low-f regularization
    // Use: S_eta = S_aa / (w^4 + wr^4)
    // This behaves like 1/w^4 well above knee, but avoids crushing the whole band.
    const double w = 2.0 * kPi * f;
    const double w2 = w * w;
    const double w4 = w2 * w2;

    const double denom_disp = std::max(1e-24, w4 + wr4);
    double S_eta = S_aa_true / denom_disp;

    if (!std::isfinite(S_eta) || S_eta < 0.0) S_eta = 0.0;

    // Optional PSD EMA (stabilizes adaptation)
    if (cfg_.psd_ema_enable) {
      if (!have_psd_ema_) psd_ema_[i] = S_eta;
      else                psd_ema_[i] = (1.0 - cfg_.psd_ema_alpha) * psd_ema_[i]
                                      + cfg_.psd_ema_alpha * S_eta;
      last_psd_eta_[i] = psd_ema_[i];
    } else {
      last_psd_eta_[i] = S_eta;
    }
  }

  have_psd_ema_ = true;

  // Tiny 3-tap smoothing on log-f axis (reduces jitter in mode split)
  smoothLogFreq3Tap_();
}

  void smoothLogFreq3Tap_() {
    if (Nfreq < 3) return;

    Vec in = last_psd_eta_;
    Vec out = in;

    for (int i = 0; i < Nfreq; ++i) {
      const double Sm1 = (i > 0)        ? in[i - 1] : in[i];
      const double S0  = in[i];
      const double Sp1 = (i < Nfreq - 1)? in[i + 1] : in[i];

      // Mild constant-Q-ish smoothing
      const double wL = 0.22;
      const double wC = 0.56;
      const double wR = 0.22;

      out[i] = wL * Sm1 + wC * S0 + wR * Sp1;
    }

    last_psd_eta_ = out;
  }

private:
  // Config / rates
  Config cfg_{};
  double fs_raw_ = 240.0;
  double fs_     = 16.0;

  // Filters
  Biquad hp1_{}, hp2_{}, lp_{};

  // Frequency grid
  std::array<double, Nfreq>   freqs_{};
  std::array<double, Nfreq>   df_{};
  std::array<double, Nfreq+1> f_edges_{};
  std::array<double, Nfreq>   goertzel_coeff_{};

  // Window
  std::array<double, Nblock> window_{};
  double window_sum_sq_ = 1.0;

  // Circular block buffer (decimated samples)
  std::array<double, Nblock> buffer_{};
  int write_index_ = 0;

  // Runtime counters
  int  decim_counter_ = 0;
  int  decimated_samples_total_ = 0;
  int  decimated_since_spec_ = 0;
  bool warm_ = false;

  // Output PSD
  Vec last_psd_eta_{};
  Vec psd_ema_{};
  bool have_psd_ema_ = false;
};
