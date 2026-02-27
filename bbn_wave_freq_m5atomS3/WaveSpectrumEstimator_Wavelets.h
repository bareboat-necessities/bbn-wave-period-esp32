#pragma once

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm> // for std::clamp

#ifdef SPECTRUM_TEST
#include <iostream>
#include <cassert>
#endif

/*

    Copyright 2025, Mikhail Grushinskiy

    WaveSpectrumEstimator  (Wavelet-based)

    Same interface as the original Goertzel estimator, but the block spectrum is computed
    using a complex Morlet wavelet filter bank (CWT-style) evaluated on the same fixed
    log-spaced frequency grid.

    Signal chain (per raw sample):
        raw a_z  →  HP (2nd)  →  HP (2nd)   →  LP (2nd)   →  ↓ (decimate by D)  →  block buffer

    Wavelet spectrum estimation (once per full block at decimated fs):
        - Linear detrend the decimated block.
        - Apply Hann window (optional).
        - For each grid bin f_i:
            * Convolve the block with a complex Morlet wavelet centered at f_i.
            * Estimate coefficient power E{|c_i|^2} over valid samples.
            * Convert to one-sided acceleration PSD via:
                  S_aa(f_i) ≈ Var_out / G_i
              where G_i is the one-sided gain integral:
                  G_i = ∫_0^{fs/2} (|H(f)|^2 + |H(-f)|^2)/2  df
              computed numerically (DFT sampling) once in the constructor per bin.

        - Deconvolve ONLY the raw-rate HP chain magnitude at f_i (same as original).
        - Convert to displacement PSD with Tikhonov-regularized 1/ω^4:
              S_eta(f) = S_aa_true(f) / ( (ω^2 + λ^2)^2 )

    Notes:
      - This produces a constant-Q-ish spectral estimate that tends to behave better on
        nonstationary / short blocks than a single long FFT/Goertzel, while keeping your
        exact same downstream interface and PM fitting, Hs, Fp helpers.

*/

template<int Nfreq = 32, int Nblock = 256>
class EIGEN_ALIGN_MAX WaveSpectrumEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr double g = 9.80665;
    using Vec = Eigen::Matrix<double, Nfreq, 1>;

    struct PMFitResult { double alpha, fp, cost; };

    // Small reusable biquad in TDF-II transposed form
    // y[n] = b0*x[n] + z1;  z1 = b1*x[n] - a1*y[n] + z2;  z2 = b2*x[n] - a2*y[n]
    struct Biquad {
        double b0 = 0, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
        double z1 = 0, z2 = 0;

        inline double process(double x) {
            double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
        inline void reset() { z1 = z2 = 0; }
    };

    WaveSpectrumEstimator(double fs_raw_ = 240.0,
                          int decimFactor_ = 30,
                          bool hannEnabled_ = true)
        : fs_raw(fs_raw_), decimFactor(decimFactor_), hannEnabled(hannEnabled_)
    {
        fs = fs_raw / decimFactor;

        buildFrequencyGrid();

        // Hann window (or rectangular) and energy sumsq
        double sumsq = 0.0;
        for (int n = 0; n < Nblock; n++) {
            if (hannEnabled) {
                window_[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1)));
            } else {
                window_[n] = 1.0;
            }
            sumsq += window_[n] * window_[n];
        }
        window_sum_sq = sumsq;

        reset();

        // Design raw-rate biquads (same as original)
        const double fny_dec = fs_raw_ / (2.0 * decimFactor_);
        const double lp_cutoffHz = 0.32 * fny_dec;   // stronger anti-alias guard than 0.45
        designLowpassBiquad(lp1_, lp_cutoffHz, fs_raw_);
        designLowpassBiquad(lp2_, lp_cutoffHz, fs_raw_);
        designHighpassBiquad(hp1_, hp_f0_hz, fs_raw_);
        designHighpassBiquad(hp2_, hp_f0_hz, fs_raw_);

        // Build wavelet bank and per-bin one-sided gain integrals
        buildWaveletBank_();
    }

    void reset() {
        buffer_.fill(0.0);
        writeIndex = 0;
        decimCounter = 0;
        filledSamples = 0;
        isWarm = false;
        lastSpectrum_.setZero();
        hp1_.reset(); hp2_.reset(); lp1_.reset(); lp2_.reset();

        have_ema = false;
        psd_ema_.setZero();
    }

    // Feed ONE raw acceleration sample at fs_raw. Returns true when a new block spectrum was computed.
    bool processSample(double x_raw) {
        // 4th-order high-pass
        double x_hp = hp2_.process(hp1_.process(x_raw));

        // low-pass (anti-aliasing before decimation)
        double y = lp2_.process(lp1_.process(x_hp));

        // Decimate by integer factor
        if (++decimCounter < decimFactor) return false;
        decimCounter = 0;

        // Push to circular buffer (post-decimation)
        buffer_[writeIndex] = y;
        writeIndex = (writeIndex + 1) % Nblock;
        filledSamples++;

        if (filledSamples >= Nblock) isWarm = true;

        // Compute spectrum once per full block
        if ((filledSamples % Nblock) == 0) {
            computeSpectrum();
            return true;
        }
        return false;
    }

    Vec getDisplacementSpectrum() const { return lastSpectrum_; }
    std::array<double, Nfreq> getFrequencies() const { return freqs_; }
    bool ready() const { return isWarm; }

    double computeHs() const {
        double m0 = 0.0;
        for (int i = 0; i < Nfreq; i++) m0 += lastSpectrum_[i] * df_[i];
        return 4.0 * std::sqrt(std::max(m0, 0.0));
    }

    double estimateFp() const {
        int k = 0; double vmax = 0.0;
        for (int i = 0; i < Nfreq; ++i) {
            if (lastSpectrum_[i] > vmax) { vmax = lastSpectrum_[i]; k = i; }
        }
        if (k == 0 || k == Nfreq - 1) return freqs_[k];

        const double f0 = freqs_[k-1], f1 = freqs_[k], f2 = freqs_[k+1];
        if (f0 <= 0 || f1 <= 0 || f2 <= 0) return f1;

        const double x0 = std::log(f0), x1 = std::log(f1), x2 = std::log(f2);
        const double y0 = safeLog(lastSpectrum_[k-1]);
        const double y1 = safeLog(lastSpectrum_[k]);
        const double y2 = safeLog(lastSpectrum_[k+1]);

        const double dx01 = x0 - x1, dx02 = x0 - x2, dx12 = x1 - x2;
        const double denom = (dx01 * dx02 * dx12);
        if (std::abs(denom) < 1e-18) return f1;

        const double L0a = 1.0 / (dx01 * dx02);
        const double L1a = 1.0 / ((x1 - x0) * (x1 - x2));
        const double L2a = 1.0 / ((x2 - x0) * (x2 - x1));
        const double a = y0 * L0a + y1 * L1a + y2 * L2a;

        const double b =
            y0 * (-(x1 + x2) * L0a) +
            y1 * (-(x0 + x2) * L1a) +
            y2 * (-(x0 + x1) * L2a);

        if (a >= 0.0) return f1;

        const double x_peak = -b / (2.0 * a);
        const double f_peak = std::exp(x_peak);

        if (f_peak <= std::min(f0,f2) || f_peak >= std::max(f0,f2)) return f1;
        return f_peak;
    }

    PMFitResult fitPiersonMoskowitz() const {
        Vec S_obs = lastSpectrum_;
        for (int i = 0; i < Nfreq; i++) if (S_obs[i] <= 0) S_obs[i] = 1e-12;

        auto cost_fn = [&](double a, double fp) {
            double omega_p = 2.0 * M_PI * fp;
            double cost = 0.0;
            constexpr double beta = 0.74;
            for (int i = 0; i < Nfreq - 1; i++) {
                double df = df_[i];
                double f = freqs_[i];
                double model = a * g * g * std::pow(2.0 * M_PI * f, -5.0)
                               * std::exp(-beta * std::pow(omega_p / (2.0 * M_PI * f), 4.0));
                if (model <= 0) model = 1e-12;
                double d = safeLog(S_obs[i]) - safeLog(model);
                cost += df * d * d;
            }
            return cost;
        };

        constexpr int N_fp_search = 32;
        constexpr double fp_min = 0.05, fp_transition = 0.1, fp_max = 1.0;
        std::array<double, N_fp_search> fp_grid;
        int n_log = static_cast<int>(N_fp_search * 0.4);
        int n_lin = N_fp_search - n_log;
        for (int i = 0; i < n_log; i++) {
            double t = double(i) / (n_log - 1);
            fp_grid[i] = fp_min * std::pow(fp_transition / fp_min, t);
        }
        for (int i = 0; i < n_lin; i++) {
            double t = double(i) / (n_lin - 1);
            fp_grid[n_log + i] = fp_transition + t * (fp_max - fp_transition);
        }

        double bestA = 1e-5, bestFp = fp_grid[0], bestC = std::numeric_limits<double>::infinity();
        for (int ia = 0; ia < 8; ia++) {
            double a = 1e-5 + ia * (1.0 - 1e-5) / 7.0;
            for (int ifp = 0; ifp < N_fp_search; ifp++) {
                double fp = fp_grid[ifp];
                double c = cost_fn(a, fp);
                if (c < bestC) { bestC = c; bestA = a; bestFp = fp; }
            }
        }

        double alpha = bestA, fp = bestFp;
        double stepA = 0.1, stepFp = 0.1;
        for (int iter = 0; iter < 40; iter++) {
            bool improved = false;
            double c;
            c = cost_fn(alpha + stepA, fp); if (c < bestC) { bestC = c; alpha += stepA; improved = true; }
            c = cost_fn(alpha - stepA, fp); if (c < bestC) { bestC = c; alpha -= stepA; improved = true; }
            c = cost_fn(alpha, fp + stepFp); if (c < bestC) { bestC = c; fp += stepFp; improved = true; }
            c = cost_fn(alpha, fp - stepFp); if (c < bestC) { bestC = c; fp -= stepFp; improved = true; }
            if (!improved) { stepA *= 0.5; stepFp *= 0.5; if (stepA < 1e-12 && stepFp < 1e-12) break; }
        }
        return {alpha, fp, bestC};
    }

    void set_regularization_f0(double f0_hz) {
        reg_f0_hz = std::max(1e-6, f0_hz);
    }

    void set_highpass_f0(double f0_hz) {
        hp_f0_hz = std::max(0.0, f0_hz);
        designHighpassBiquad(hp1_, hp_f0_hz, fs_raw);
        designHighpassBiquad(hp2_, hp_f0_hz, fs_raw);
    }

private:
    inline double safeLog(double v) const { return std::log(std::max(v, 1e-18)); }

    inline double alpha_for_f(double f) const {
        const double fmin = freqs_[0];
        const double fmax = freqs_[Nfreq-1];
        double t = (f - fmin) / std::max(1e-12, (fmax - fmin));
        t = std::clamp(t, 0.0, 1.0);
        return ema_alpha_low + (ema_alpha_high - ema_alpha_low) * t;
    }

    void smooth_logfreq_3tap() {
        if (Nfreq < 3) return;
        Eigen::Matrix<double, Nfreq, 1> S = lastSpectrum_;
        Eigen::Matrix<double, Nfreq, 1> Sout = S;

        auto wpair = [&](int i) {
            const double eps = 1e-12;
            double x_im1 = (i>0) ? std::log(std::max(freqs_[i-1], eps)) : std::log(std::max(freqs_[i], eps));
            double x_i   = std::log(std::max(freqs_[i],   eps));
            double x_ip1 = (i<Nfreq-1)? std::log(std::max(freqs_[i+1], eps)) : std::log(std::max(freqs_[i], eps));
            double dL = std::max(0.0, x_i - x_im1);
            double dR = std::max(0.0, x_ip1 - x_i);
            const double k = 0.35;
            const double minC = 0.40;
            double wL = k * dL;
            double wR = k * dR;
            double wC = std::max(minC, 1.0 - (wL + wR));
            double W  = wL + wC + wR;
            return std::array<double,3>{ wL/W, wC/W, wR/W };
        };

        for (int i = 0; i < Nfreq; ++i) {
            auto w = wpair(i);
            double Sm1 = (i>0)? S[i-1] : S[i];
            double Sp1 = (i<Nfreq-1)? S[i+1] : S[i];
            Sout[i] = w[0]*Sm1 + w[1]*S[i] + w[2]*Sp1;
        }
        lastSpectrum_ = Sout;
    }

    void buildFrequencyGrid() {
        constexpr double f_min = 0.04, f_max = 1.2;

        for (int i = 0; i <= Nfreq; ++i) {
            double t = double(i) / double(Nfreq);
            f_edges_[i] = f_min * std::pow(f_max / f_min, t);
        }

        for (int i = 0; i < Nfreq; ++i) {
            freqs_[i] = std::sqrt(f_edges_[i] * f_edges_[i + 1]);
            df_[i]    = f_edges_[i + 1] - f_edges_[i];
        }
    }

    void designLowpassBiquad(Biquad &bq, double f_cut, double Fs) {
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        bq.b0 = K * K * norm;  bq.b1 = 2.0 * bq.b0;  bq.b2 = bq.b0;
        bq.a1 = 2.0 * (K * K - 1.0) * norm;  bq.a2 = (1.0 - K / Q + K * K) * norm;
    }

    void designHighpassBiquad(Biquad &bq, double f_cut, double Fs) {
        if (f_cut <= 0.0) {
            bq.b0 = 1.0; bq.b1 = 0.0; bq.b2 = 0.0; bq.a1 = 0.0; bq.a2 = 0.0;
            bq.reset();
            return;
        }
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        bq.b0 = 1.0 * norm;  bq.b1 = -2.0 * norm;  bq.b2 = 1.0 * norm;
        bq.a1 = 2.0 * (K * K - 1.0) * norm;  bq.a2 = (1.0 - K / Q + K * K) * norm;
        bq.reset();
    }

    inline double biquad_mag2_raw(const Biquad& bq, double Omega_raw) const {
        const double c1 = std::cos(Omega_raw), s1 = std::sin(Omega_raw);
        const double c2 = std::cos(2 * Omega_raw), s2 = std::sin(2 * Omega_raw);

        const double num_re = bq.b0 + bq.b1 * c1 + bq.b2 * c2;
        const double num_im = -(bq.b1 * s1 + bq.b2 * s2);

        const double den_re = 1.0 + bq.a1 * c1 + bq.a2 * c2;
        const double den_im = -(bq.a1 * s1 + bq.a2 * s2);

        const double num2 = num_re * num_re + num_im * num_im;
        const double den2 = den_re * den_re + den_im * den_im;
        return num2 / std::max(den2, 1e-16);
    }

    // ---------------- Wavelet bank ----------------

    // Choose a power-of-two NFFT for gain integral sampling (compile-time)
    static constexpr int WAVELET_NFFT =
        (Nblock <= 256) ? 1024 :
        (Nblock <= 512) ? 2048 : 4096;

    // Morlet shaping knobs
    static constexpr double WAVELET_CYCLES_TARGET = 6.0;   // target cycles under Gaussian
    static constexpr double WAVELET_SUPPORT_SIGMA = 3.0;   // truncate at ±support*sigma_t
    static constexpr int    WAVELET_MIN_HALF      = 8;     // min half-length (taps)
    static constexpr double WAVELET_MEAN_CORR_EPS = 1e-6;

    // Store taps per bin packed into [i*Nblock + 0..L_i-1], rest unused.
    std::array<float,  Nfreq * Nblock> wave_re_{}; // real taps
    std::array<float,  Nfreq * Nblock> wave_im_{}; // imag taps
    std::array<int,    Nfreq>          wave_half_{}; // half length
    std::array<double, Nfreq>          wave_gain_onesided_hz_{}; // gain integral for one-sided PSD

    inline int tapIndex_(int i, int n) const { return i * Nblock + n; }

    void buildWaveletBank_() {
        // Zero arrays
        wave_re_.fill(0.0f);
        wave_im_.fill(0.0f);
        wave_half_.fill(0);
        wave_gain_onesided_hz_.fill(1.0);

        // Maximum sigma_t allowed by block (so we always have some valid region)
        const double halfMax = double((Nblock - 1) / 2);
        const double sigma_t_max = (fs > 0.0) ? (halfMax / (WAVELET_SUPPORT_SIGMA * fs)) : 0.0;

        // Temp DFT buffers for gain computation (stack OK, WAVELET_NFFT is modest)
        std::array<double, WAVELET_NFFT> Hre{};
        std::array<double, WAVELET_NFFT> Him{};

        for (int i = 0; i < Nfreq; ++i) {
            const double f0 = freqs_[i];
            if (!(f0 > 0.0) || !(fs > 0.0)) {
                wave_gain_onesided_hz_[i] = 1.0;
                wave_half_[i] = WAVELET_MIN_HALF;
                continue;
            }

            // Target sigma_t, then clamp to what fits in the block
            double sigma_t = WAVELET_CYCLES_TARGET / (2.0 * M_PI * f0);
            sigma_t = std::min(sigma_t, std::max(1e-6, sigma_t_max));

            int half = int(std::ceil(WAVELET_SUPPORT_SIGMA * sigma_t * fs));
            half = std::clamp(half, WAVELET_MIN_HALF, int(halfMax));
            sigma_t = double(half) / (WAVELET_SUPPORT_SIGMA * fs);

            const int L = 2 * half + 1;
            wave_half_[i] = half;

            // Mean correction term (classic Morlet admissibility fix)
            const double w0 = 2.0 * M_PI * f0;
            const double C0 = std::exp(-0.5 * (w0 * sigma_t) * (w0 * sigma_t)); // may underflow -> ok

            // Build taps centered at zero (index n-half)
            for (int n = 0; n < L; ++n) {
                const int k = n - half;
                const double t = double(k) / fs;
                const double u = t / std::max(1e-9, sigma_t);
                const double ga = std::exp(-0.5 * u * u);
                const double phi = 2.0 * M_PI * f0 * t;
                double re = ga * std::cos(phi);
                double im = ga * std::sin(phi);
                // subtract small DC component (pure real)
                re -= (C0 * ga);

                wave_re_[tapIndex_(i, n)] = (float)re;
                wave_im_[tapIndex_(i, n)] = (float)im;
            }

            // Normalize taps to unit magnitude response at +f0 (magnitude only).
            // Compute H(f0) = Σ h[k] e^{-j 2π f0 (k/fs)} using centered k.
            double H0r = 0.0, H0i = 0.0;
            for (int n = 0; n < L; ++n) {
                const int k = n - half;
                const double t = double(k) / fs;
                const double phi = 2.0 * M_PI * f0 * t;
                const double c = std::cos(phi);
                const double s = std::sin(phi);
                const double re = (double)wave_re_[tapIndex_(i, n)];
                const double im = (double)wave_im_[tapIndex_(i, n)];
                // (re + j im) * (c - j s)
                H0r += re * c + im * s;
                H0i += im * c - re * s;
            }
            const double H0mag = std::sqrt(H0r*H0r + H0i*H0i);
            const double scale = (H0mag > 1e-12) ? (1.0 / H0mag) : 1.0;

            for (int n = 0; n < L; ++n) {
                wave_re_[tapIndex_(i, n)] = (float)((double)wave_re_[tapIndex_(i, n)] * scale);
                wave_im_[tapIndex_(i, n)] = (float)((double)wave_im_[tapIndex_(i, n)] * scale);
            }

            // Compute one-sided gain integral G_i for one-sided PSD mapping:
            //   G_i = ∫_0^{fs/2} (|H(f)|^2 + |H(-f)|^2)/2 df
            // using DFT samples of the FIR (zero-padded to WAVELET_NFFT).
            Hre.fill(0.0);
            Him.fill(0.0);

            const double df = fs / double(WAVELET_NFFT);

            for (int k = 0; k < WAVELET_NFFT; ++k) {
                // Compute H[k] = Σ_{n=0..L-1} h[n] e^{-j 2π k n / NFFT}
                const double w = 2.0 * M_PI * double(k) / double(WAVELET_NFFT);
                const double cw = std::cos(w);
                const double sw = std::sin(w);

                double cr = 1.0, ci = 0.0; // cos(w n), sin(w n)
                double sr = 0.0, si = 0.0;

                for (int n = 0; n < L; ++n) {
                    const double re = (double)wave_re_[tapIndex_(i, n)];
                    const double im = (double)wave_im_[tapIndex_(i, n)];

                    // e^{-j w n} = cr - j ci
                    // (re + j im) * (cr - j ci) => (re*cr + im*ci) + j(im*cr - re*ci)
                    sr += re * cr + im * ci;
                    si += im * cr - re * ci;

                    // Rotate (cr,ci) by w: (cr,ci) <- (cr*cosw - ci*sinw, cr*sinw + ci*cosw)
                    const double crn = cr * cw - ci * sw;
                    const double cin = cr * sw + ci * cw;
                    cr = crn; ci = cin;
                }
                Hre[k] = sr;
                Him[k] = si;
            }

            double gain = 0.0;
            for (int k = 0; k <= WAVELET_NFFT/2; ++k) {
                const int kneg = (k == 0 || k == WAVELET_NFFT/2) ? k : (WAVELET_NFFT - k);
                const double mag2_pos = Hre[k]*Hre[k] + Him[k]*Him[k];
                const double mag2_neg = Hre[kneg]*Hre[kneg] + Him[kneg]*Him[kneg];
                gain += 0.5 * (mag2_pos + mag2_neg) * df;
            }

            // Floor to avoid divide-by-zero if something pathological happens
            wave_gain_onesided_hz_[i] = std::max(gain, 1e-12);
        }
    }

    // ---------------- Main spectrum computation ----------------

    void computeSpectrum() {
        const int N = Nblock;

        // Extract block in time order
        std::array<double, Nblock> x{};
        const int startIdx = writeIndex; // writeIndex points to "next" spot => oldest is at writeIndex
        int idx = startIdx;
        for (int n = 0; n < N; ++n) {
            x[n] = buffer_[idx];
            idx = (idx + 1) % Nblock;
        }

        // Linear detrend
        double sumx = 0.0, sumn = 0.0, sumn2 = 0.0, sumnx = 0.0;
        for (int n = 0; n < N; ++n) {
            sumx  += x[n];
            sumn  += n;
            sumn2 += double(n) * double(n);
            sumnx += double(n) * x[n];
        }
        const double denom = double(N) * sumn2 - sumn * sumn;
        const double a_lin = (denom != 0.0) ? (double(N) * sumnx - sumn * sumx) / denom : 0.0;
        const double b_lin = (sumx - a_lin * sumn) / double(N);

        // Apply window and detrend
        std::array<double, Nblock> xw{};
        for (int n = 0; n < N; ++n) {
            const double xdet = x[n] - (a_lin * n + b_lin);
            xw[n] = xdet * window_[n];
        }

        // Window RMS^2 correction
        const double w_rms2 = (N > 0) ? (window_sum_sq / double(N)) : 1.0;
        const double inv_w_rms2 = 1.0 / std::max(w_rms2, 1e-12);

        // Block-based knee (same idea as original)
        const double Tblk = (fs > 0.0) ? (double(N) / fs) : 0.0;
        const double f_blk = (Tblk > 0.0) ? (1.0 / (6.0 * Tblk)) : 0.0;
        const double f_knee = std::max(reg_f0_hz, f_blk);
        const double lam    = 2.0 * M_PI * f_knee;

        for (int i = 0; i < Nfreq; ++i) {
            const double f = freqs_[i];
            const int half = wave_half_[i];
            const int L = 2 * half + 1;

            // Convolve with complex wavelet (valid region only)
            double pwr = 0.0;
            int M = 0;
            for (int n = half; n < N - half; ++n) {
                double yr = 0.0, yi = 0.0;
                // taps index: 0..L-1 corresponds to k=-half..+half
                for (int tn = 0; tn < L; ++tn) {
                    const int k = tn - half;
                    const double xv = xw[n - k];
                    const double re = (double)wave_re_[tapIndex_(i, tn)];
                    const double im = (double)wave_im_[tapIndex_(i, tn)];
                    yr += re * xv;
                    yi += im * xv;
                }
                pwr += (yr * yr + yi * yi);
                ++M;
            }
            const double var_out = (M > 0) ? (pwr / double(M)) : 0.0;

            // Convert to one-sided acceleration PSD (per Hz)
            double S_aa_meas = (var_out * inv_w_rms2) / std::max(wave_gain_onesided_hz_[i], 1e-12);

            // Deconvolve ONLY the HP chain at raw Fs (narrowband approximation at f)
            const double Omega_raw = 2.0 * M_PI * f / fs_raw;
            const double H2_hp =
                biquad_mag2_raw(hp1_, Omega_raw) *
                biquad_mag2_raw(hp2_, Omega_raw);

            constexpr double H2_floor = 0.05; // ~ -13 dB floor on |H|^2
            double S_aa_true = S_aa_meas / std::max(H2_hp, H2_floor);

            // Tikhonov regularized 1/ω^4 -> displacement PSD
            const double w = 2.0 * M_PI * f;
            const double den = (w * w + lam * lam);
            double S_eta = (den > 0.0) ? (S_aa_true / (den * den)) : 0.0;

            if (!std::isfinite(S_eta) || S_eta < 0.0) S_eta = 0.0;

            // EMA smoothing (kept)
            if (use_psd_ema) {
                double a = alpha_for_f(f);
                if (!have_ema) psd_ema_[i] = S_eta;
                else           psd_ema_[i] = (1.0 - a) * psd_ema_[i] + a * S_eta;
                lastSpectrum_[i] = psd_ema_[i];
            } else {
                lastSpectrum_[i] = S_eta;
            }
        }

        have_ema = true;
        smooth_logfreq_3tap();
    }

    // Members / State

    // Rates and decimation
    double fs_raw = 0.0, fs = 0.0;
    int decimFactor = 1;
    bool hannEnabled = true;

    // Regularization and HP corner (Hz)
    double reg_f0_hz = 0.008;
    double hp_f0_hz  = 0.025;

    // Frequency grid and bin widths
    std::array<double, Nfreq>   freqs_{};
    std::array<double, Nfreq+1> f_edges_{};
    std::array<double, Nfreq>   df_{};

    // Block buffer (post-decimation) and window
    std::array<double, Nblock> buffer_{};
    std::array<double, Nblock> window_{};
    double window_sum_sq = 1.0;

    // IIR filters at raw Fs
    Biquad hp1_, hp2_, lp1_, lp2_;

    // Output spectrum
    Eigen::Matrix<double, Nfreq, 1> lastSpectrum_;

    // Smoothing controls
    bool  use_psd_ema = true;
    double ema_alpha_low  = 0.20;
    double ema_alpha_high = 0.06;
    bool  have_ema = false;
    Eigen::Matrix<double, Nfreq, 1> psd_ema_;

    // Circular buffer indices/state
    int writeIndex = 0;
    int decimCounter = 0;
    int filledSamples = 0;
    bool isWarm = false;

    // Filter Q (per biquad)
    static constexpr double Q = 0.707;
};

#ifdef SPECTRUM_TEST
// Minimal offline test with a single-tone acceleration.
// For wavelets the exact Hs number depends on gain calibration; we keep this as a
// finite / sanity check rather than a strict numeric expectation.
void WaveSpectrumEstimator_test() {
    constexpr int Nfreq = 32;
    constexpr int Nblock = 256;

    double fs = 240.0;
    WaveSpectrumEstimator<Nfreq, Nblock> estimator(fs, 2, true);

    double f_test = 0.2;
    double A_test = 1.0;
    int N_samples = 6000;

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

            assert(std::isfinite(Hs) && Hs >= 0.0);
            assert(std::isfinite(Fp) && Fp > 0.0);
            assert(std::isfinite(pm.alpha) && pm.alpha > 0.0);
            assert(std::isfinite(pm.fp) && pm.fp > 0.0);
            (void)S;
        }
    }
    assert(ready_count > 0);
}
#endif
