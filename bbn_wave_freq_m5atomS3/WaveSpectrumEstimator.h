#pragma once

#include <ArduinoEigenDense.h>
#include <array>
#include <vector>
#include <cmath>
#include <limits>

#ifdef SPECTRUM_TEST
#include <iostream>
#include <cassert>
#endif

/*
  WaveSpectrumEstimator

  This class estimates the ocean wave spectrum from acceleration measurements.
  It implements a decimated, sliding-window Goertzel algorithm with optional
  Hann windowing and a low-pass biquad filter to reduce high-frequency noise.

  Features:
    - Computes the displacement spectrum from vertical acceleration.
    - Provides estimates of significant wave height (Hs) and peak wave frequency (Fp).
    - Supports Pierson-Moskowitz spectrum fitting to estimate spectral parameters.
    - Handles arbitrary frequency grid sizes (Nfreq) and block lengths (Nblock).
    - Embedded-friendly: uses fixed-size arrays and Eigen matrices.

  Typical workflow:
    1. Create an instance with desired parameters (sample rate, decimation, window shift, etc.).
    2. Call processSample() for each acceleration sample.
    3. When processSample() returns true, spectrum is ready:
        - getDisplacementSpectrum()
        - computeHs()
        - estimateFp()
        - fitPiersonMoskowitz()

  Copyright 2025, Mikhail Grushinskiy
*/
template<int Nfreq = 32, int Nblock = 1024>
class WaveSpectrumEstimator {
  public:
    static constexpr double g = 9.80665;
    using Vec = Eigen::Matrix<double, Nfreq, 1>;

    struct PMFitResult {
      double alpha;
      double fp;
      double cost;
    };

    WaveSpectrumEstimator(double fs_raw_ = 240.0,
                          int decimFactor_ = 5,
                          int shift_samples_ = 64,
                          bool hannEnabled_ = true)
      : fs_raw(fs_raw_), decimFactor(decimFactor_), shift(shift_samples_), hannEnabled(hannEnabled_)
    {
      // Validate shift
      if (shift > Nblock) shift = Nblock;

      // Compute decimated sample rate
      fs = fs_raw / decimFactor;

      // Initialize default frequency grid
      buildFrequencyGrid();

      // Precompute Goertzel coefficients
      for (int i = 0; i < Nfreq; i++) {
        double k = 0.5 + (Nblock * freqs_[i]) / fs;   // mid-bin placement
        double omega = 2.0 * M_PI * k / Nblock;
        coeffs_[i] = 2.0 * std::cos(omega);

        cos1_[i] = std::cos(omega);
        sin1_[i] = std::sin(omega);
        cosN_[i] = std::cos(omega * Nblock);
        sinN_[i] = std::sin(omega * Nblock);
      }

      // Initialize window
      for (int n = 0; n < Nblock; n++) {
        window_[n] = hannEnabled
                     ? 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1)))
                     : 1.0;
      }
      windowGain = hannEnabled ? std::sqrt(3.0 / 8.0) : 1.0;  // RMS normalization

      // Clear accumulators and buffers
      reset();

      // Design low-pass biquad
      double cutoffHz = 0.45 * (fs_raw_ / (2.0 * decimFactor)); // 45% of post-decimation Nyquist
      designLowpassBiquad(cutoffHz, fs_raw_);
    }

    void reset() {
      buffer_.fill(0.0);
      s1_.setZero();
      s2_.setZero();
      s1_old_.setZero();
      s2_old_.setZero();
      writeIndex = 0;
      decimCounter = 0;
      samplesSinceLast = 0;
      z1 = z2 = 0.0;
      filledSamples = 0;
      isWarm = false;
    }

    void resetSpectrum() {
      s1_.setZero();
      s2_.setZero();
      s1_old_.setZero();
      s2_old_.setZero();
    }

bool processSample(double x_raw) {
    // Apply low-pass biquad filter
    double y = b0 * x_raw + z1;
    z1 = b1 * x_raw + a1 * y + z2;
    z2 = b2 * x_raw + a2 * y;

    // Decimation
    if (++decimCounter < decimFactor) return false;
    decimCounter = 0;

    // Get oldest sample before overwrite
    double oldSample = buffer_[readIndex];

    // Store new sample at writeIndex
    buffer_[writeIndex] = y;

    // Apply windowing
    double newWinSample = y * window_[writeIndex];
    double oldWinSample = oldSample * window_[readIndex];

    // Update Goertzel accumulators
    for (int i = 0; i < Nfreq; i++) {
        // Add new contribution
        double s = newWinSample + coeffs_[i] * s1_[i] - s2_[i];
        s2_[i] = s1_[i];
        s1_[i] = s;

        // Subtract oldest contribution
        double so = oldWinSample + coeffs_[i] * s1_old_[i] - s2_old_[i];
        s2_old_[i] = s1_old_[i];
        s1_old_[i] = so;
    }

    // Advance indices
    writeIndex = (writeIndex + 1) % Nblock;
    readIndex  = (readIndex  + 1) % Nblock;

    // Warm-up check
    if (filledSamples < Nblock) {
        filledSamples++;
        if (filledSamples == Nblock) isWarm = true;
    }

    // Shift / ready flag
    if (++samplesSinceLast >= shift && isWarm) {
        samplesSinceLast = 0;
        return true;
    }
    return false;
}

Vec getDisplacementSpectrum() const {
  Vec S;
  const double invNorm = 1.0 / (double(Nblock) * double(Nblock) * windowGain * windowGain);

  for (int i = 0; i < Nfreq; i++) {
    // Goertzel complex outputs: Y = s1 - s2 * e^{-jω}
    double re_cur = s1_[i] - s2_[i] * cos1_[i];
    double im_cur =            s2_[i] * sin1_[i];

    double re_old = s1_old_[i] - s2_old_[i] * cos1_[i];
    double im_old =              s2_old_[i] * sin1_[i];

    // Align OLD by +ωN (because old stream accrued an extra e^{-jωN})
    double re_old_al = re_old * cosN_[i] - im_old * sinN_[i];
    double im_old_al = re_old * sinN_[i] + im_old * cosN_[i];

    // Window DFT = current cumulative − aligned old cumulative
    double re_win = re_cur - re_old_al;
    double im_win = im_cur - im_old_al;

    double mag2 = (re_win * re_win + im_win * im_win) * invNorm;
    if (!std::isfinite(mag2) || mag2 < 0.0) mag2 = 0.0;

    // Accel -> displacement
    double f = freqs_[i];
    double denom = std::pow(2.0 * M_PI * f, 4);
    double Si = (denom > 0.0) ? (mag2 / denom) : 0.0;

    S[i] = (std::isfinite(Si) && Si > 0.0) ? Si : 0.0;
  }
  return S;
}

    double computeHs() const {
      Vec S = getDisplacementSpectrum();
      double m0 = 0.0;

      for (int i = 0; i < Nfreq - 1; i++) {
        double df = freqs_[i + 1] - freqs_[i]; // non-uniform spacing
        m0 += 0.5 * (S[i] + S[i + 1]) * df;    // trapezoidal rule
      }
      return 4.0 * std::sqrt(std::max(m0, 0.0));
    }

    double estimateFp() const {
      if (Nfreq == 0) return 0.0;  // Handle empty frequency grid

      Vec S = getDisplacementSpectrum();

      // Find maximum
      int idx = 0;
      double maxVal = 0;
      for (int i = 0; i < Nfreq; i++) {
        if (S[i] > maxVal) {
          maxVal = S[i];
          idx = i;
        }
      }

      // Parabolic interpolation (log scale)
      if (idx > 0 && idx < Nfreq - 1) {
        double y0 = safeLog(S[idx - 1]);
        double y1 = safeLog(S[idx]);
        double y2 = safeLog(S[idx + 1]);

        double denominator = (y0 - 2 * y1 + y2);
        if (std::abs(denominator) < 1e-12) return freqs_[idx];
        double p = 0.5 * (y0 - y2) / denominator;

        double df_left = freqs_[idx] - freqs_[idx - 1];
        double df_right = freqs_[idx + 1] - freqs_[idx];
        double df_avg = 0.5 * (df_left + df_right);

        return freqs_[idx] + p * df_avg;
      }
      else if (idx == 0 && Nfreq > 1) {
        // One-sided forward interpolation
        double y1 = safeLog(S[0]);
        double y2 = safeLog(S[1]);
        double p = (y2 - y1) / (y2 + 1e-12); // simple slope approximation
        double df = freqs_[1] - freqs_[0];
        return freqs_[0] + p * df;
      }
      else if (idx == Nfreq - 1 && Nfreq > 1) {
        // One-sided backward interpolation
        double y0 = safeLog(S[Nfreq - 2]);
        double y1 = safeLog(S[Nfreq - 1]);
        double p = (y1 - y0) / (y1 + 1e-12); // simple slope approximation
        double df = freqs_[Nfreq - 1] - freqs_[Nfreq - 2];
        return freqs_[Nfreq - 1] + p * df;
      }

      return freqs_[idx];  // fallback
    }

    PMFitResult fitPiersonMoskowitz() const {
      auto S_obs = getDisplacementSpectrum();
      for (int i = 0; i < Nfreq; i++) if (S_obs[i] <= 0) S_obs[i] = 1e-12;

      auto cost_fn = [&](double a, double fp) {
        double omega_p = 2 * M_PI * fp;
        double cost = 0.0;
        constexpr double beta = 0.74;

        for (int i = 0; i < Nfreq; i++) {
          double f = freqs_[i];
          double model = a * g * g * std::pow(2 * M_PI * f, -5.0)
                         * std::exp(-beta * std::pow(omega_p / (2 * M_PI * f), 4.0));
          if (model <= 0) model = 1e-12;
          double d = safeLog(S_obs[i]) - safeLog(model);
          cost += d * d;
        }
        return cost;
      };

      // Build a search grid for fp using hybrid log-linear spacing
      constexpr int N_fp_search = 32;
      constexpr double fp_min = 0.05;
      constexpr double fp_transition = 0.1;
      constexpr double fp_max = 1.0;

      std::array<double, N_fp_search> fp_grid;
      int n_log = static_cast<int>(N_fp_search * 0.4);
      int n_lin = N_fp_search - n_log;

      // Log portion
      for (int i = 0; i < n_log; i++) {
        double t = static_cast<double>(i) / (n_log - 1);
        fp_grid[i] = fp_min * std::pow(fp_transition / fp_min, t);
      }

      // Linear portion
      for (int i = 0; i < n_lin; i++) {
        double t = static_cast<double>(i) / (n_lin - 1);
        fp_grid[n_log + i] = fp_transition + t * (fp_max - fp_transition);
      }

      // Coarse search
      double bestA = 1e-5, bestFp = fp_grid[0], bestC = std::numeric_limits<double>::infinity();
      for (int ia = 0; ia < 8; ia++) {
        double a = 1e-5 + ia * (1.0 - 1e-5) / 7;
        for (int ifp = 0; ifp < N_fp_search; ifp++) {
          double fp = fp_grid[ifp];
          double c = cost_fn(a, fp);
          if (c < bestC) {
            bestC = c;
            bestA = a;
            bestFp = fp;
          }
        }
      }

      // Local refinement
      double alpha = bestA, fp = bestFp;
      double stepA = 0.1, stepFp = 0.1;
      for (int iter = 0; iter < 40; iter++) {
        bool improved = false;
        double c;
        c = cost_fn(alpha + stepA, fp); if (c < bestC) {
          bestC = c;
          alpha += stepA;
          improved = true;
        }
        c = cost_fn(alpha - stepA, fp); if (c < bestC) {
          bestC = c;
          alpha -= stepA;
          improved = true;
        }
        c = cost_fn(alpha, fp + stepFp); if (c < bestC) {
          bestC = c;
          fp += stepFp;
          improved = true;
        }
        c = cost_fn(alpha, fp - stepFp); if (c < bestC) {
          bestC = c;
          fp -= stepFp;
          improved = true;
        }
        if (!improved) {
          stepA *= 0.5;
          stepFp *= 0.5;
          if (stepA < 1e-12 && stepFp < 1e-12) break;
        }
      }

      PMFitResult res; res.alpha = alpha; res.fp = fp; res.cost = bestC;
      return res;
    }

    bool ready() const {
      return isWarm;
    }

  private:

inline double safeLog(double v) const {
    return std::log(std::max(v, 1e-18));  // avoid log(0)
}

// f_cut in Hz, Fs in Hz
void designLowpassBiquad(double f_cut, double Fs) {
    double Fc = f_cut / Fs;          // normalized cutoff (0..0.5)
    double K  = std::tan(M_PI * Fc);
    double norm = 1.0 / (1.0 + K / Q + K * K);

    // Feedforward
    b0 = K * K * norm;
    b1 = 2.0 * b0;
    b2 = b0;

    // Feedback
    a1 = 2.0 * (K * K - 1.0) * norm;
    a2 = (1.0 - K / Q + K * K) * norm;
}

    void buildFrequencyGrid() {
      constexpr double f_min = 0.03;  // Hz, lowest wave frequency
      constexpr double f_transition = 0.1; // Hz, where log → linear transition
      constexpr double f_max = 1.0;   // Hz, highest frequency of interest

      // Number of bins in log and linear portions
      int n_log = static_cast<int>(Nfreq * 0.4);   // ~40% of bins for log spacing
      int n_lin = Nfreq - n_log;

      // Logarithmic portion (f_min to f_transition)
      for (int i = 0; i < n_log; i++) {
        double t = static_cast<double>(i) / (n_log - 1);
        freqs_[i] = f_min * std::pow(f_transition / f_min, t);
      }

      // Linear portion (f_transition to f_max)
      for (int i = 0; i < n_lin; i++) {
        double t = static_cast<double>(i) / (n_lin - 1);
        freqs_[n_log + i] = f_transition + t * (f_max - f_transition);
      }
    }

    double fs_raw, fs;
    int decimFactor, shift;
    bool hannEnabled;

    std::array<double, Nfreq> freqs_;
    std::array<double, Nfreq> coeffs_;
    std::array<double, Nblock> buffer_;
    std::array<double, Nblock> window_;
    double windowGain;

    // filter state
    double b0, b1, b2, a1, a2;
    double z1 = 0, z2 = 0;

    // Goertzel accumulators
    Eigen::Matrix<double, Nfreq, 1> s1_, s2_, s1_old_, s2_old_;

    std::array<double, Nfreq> cos1_, sin1_, cosN_, sinN_;
    double Q = std::sqrt(0.5); // Butterworth, ~0.707

    // counters
    int writeIndex = 0;
    int decimCounter = 0;
    int samplesSinceLast = 0;

    // warm-up
    int filledSamples = 0;
    bool isWarm = false;
};

#ifdef SPECTRUM_TEST
void WaveSpectrumEstimator_test() {
    constexpr int Nfreq = 32;
    constexpr int Nblock = 256;

    double fs = 240.0;           // sample rate
    WaveSpectrumEstimator<Nfreq, Nblock> estimator(fs, 2, 64, true);

    // Generate a test sine wave at 0.2 Hz (simulating vertical acceleration)
    double f_test = 0.2;         // Hz
    double A_test = 1.0;         // amplitude
    int N_samples = 2000;

    int ready_count = 0;
    for (int n = 0; n < N_samples; n++) {
        double t = n / fs;
        double acc = A_test * std::sin(2.0 * M_PI * f_test * t);
        if (estimator.processSample(acc)) {
            ready_count++;

            auto S = estimator.getDisplacementSpectrum();
            double Hs = estimator.computeHs();
            double Fp = estimator.estimateFp();
            auto pm = estimator.fitPiersonMoskowitz();

            std::cerr << "Spectrum ready: Hs = " << Hs 
                      << ", Fp = " << Fp 
                      << ", PM fit: alpha = " << pm.alpha 
                      << ", fp = " << pm.fp 
                      << ", cost = " << pm.cost << "\n";

            // Basic checks
            assert(Hs > 0);                       // Hs should be positive
            assert(Fp > 0);                       // Fp should be positive
            assert(pm.alpha > 0);                 // PM fit alpha positive
            assert(pm.fp > 0);                    // PM fit fp positive
        }
    }

    // Ensure the estimator returned ready at least once
    assert(ready_count > 0);
}
#endif
