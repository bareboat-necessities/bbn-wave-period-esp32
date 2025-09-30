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

#ifdef SPECTRUM_TEST
#include <iostream>
#include <cassert>
#endif

/*

    Copyright 2025, Mikhail Grushinskiy

    WaveSpectrumEstimator

    This class estimates the ocean wave spectrum from acceleration measurements.
    It implements a decimated, sliding-window Goertzel algorithm with optional
    Hann windowing and cascaded biquad filtering at the raw sample rate.

    Signal chain (per raw sample):
        raw a_z  →  HP (2nd)  →  HP (2nd)   →  LP (2nd)   →  ↓ (decimate by D)  →  block buffer
                          \_____ 4th-order HP cascade _____/

    Why HP cascade?
        A single 2nd-order HP is often not enough to suppress accelerometer bias / very-low-f drift.
        Cascading two identical 2nd-order sections gives a 4th-order Butterworth-like HP with a
        much steeper rolloff below ~0.05–0.07 Hz, which reduces inflated power in the lowest bins.

    Spectrum estimation:
        - Linear detrend the decimated block.
        - Apply Hann window.
        - Goertzel per requested frequency bin → acceleration PSD S_aa(f).
        - Convert to displacement PSD via regularized inversion:
            S_eta(f) = S_aa(f) / (ω^2 + λ^2)^2   with  λ = 2π·f_reg
          (Tikhonov regularization stabilizes the 1/ω^4 inversion at low f.)

    Outputs:
        - getDisplacementSpectrum() : S_eta(f) over the user-defined grid
        - computeHs()               : 4√m0 with m0 = ∫ S_eta(f) df (trapezoidal)
        - estimateFp()              : log-parabolic peak interpolation
        - fitPiersonMoskowitz()     : log-least-squares fit for (α, f_p)

    Embedded-friendly:
        - Fixed-size Eigen vector for spectrum, std::array buffers, no heap allocations
        - Biquad sections in TDF-II transposed form for numerical stability

*/

template<int Nfreq = 32, int Nblock = 2048>
class EIGEN_ALIGN_MAX WaveSpectrumEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr double g = 9.80665;
    using Vec = Eigen::Matrix<double, Nfreq, 1>;

    struct PMFitResult { double alpha, fp, cost; };

    // ---------------------------------------------------------------------
    // Small reusable biquad in TDF-II transposed form (stable, low state)
    // y[n] = b0*x[n] + z1;  z1 = b1*x[n] - a1*y[n] + z2;  z2 = b2*x[n] - a2*y[n]
    // ---------------------------------------------------------------------
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
                          int decimFactor_ = 60,
                          bool hannEnabled_ = true)
        : fs_raw(fs_raw_), decimFactor(decimFactor_), hannEnabled(hannEnabled_)
    {
        // Effective (post-decimation) sample rate for spectral block
        fs = fs_raw / decimFactor;

        // ---------------- Build analysis frequency grid -----------------
        buildFrequencyGrid();

        // ---------------- Precompute Goertzel coefficients ---------------
        // Each frequency bin is represented by a single Goertzel recursion
        // with rad/sample omega_rs defined at the decimated rate.
        for (int i = 0; i < Nfreq; i++) {
            double omega_rs = 2.0 * M_PI * freqs_[i] / fs; // rad/sample
            coeffs_[i] = 2.0 * std::cos(omega_rs);
            cos1_[i]   = std::cos(omega_rs);
            sin1_[i]   = std::sin(omega_rs);
        }

        // ---------------- Hann window (or rectangular) -------------------
        // Precompute window shape and its total energy sum of squares (U).
        double sumsq = 0.0;
        for (int n = 0; n < Nblock; n++) {
            if (hannEnabled) {
                window_[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1)));
            } else {
                window_[n] = 1.0;
            }
            sumsq += window_[n] * window_[n];
        }
        window_sum_sq = sumsq; // used in PSD scaling

        // ---------------- Reset runtime state ----------------------------
        reset();

        // ---------------- Design raw-rate biquads ------------------------
        // LP near pre-decimation Nyquist to suppress imaging/aliasing.
        // Two identical HP sections cascaded → 4th-order high-pass.
        const double lp_cutoffHz = 0.45 * (fs_raw_ / (2.0 * decimFactor)); // conservative guard
        designLowpassBiquad(lp_, lp_cutoffHz, fs_raw_);
        designHighpassBiquad(hp1_, hp_f0_hz, fs_raw_);
        designHighpassBiquad(hp2_, hp_f0_hz, fs_raw_);
    }

    // Reset states and circular buffer; does not change design parameters
    void reset() {
        buffer_.fill(0.0);
        writeIndex = 0;
        decimCounter = 0;
        filledSamples = 0;
        isWarm = false;
        lastSpectrum_.setZero();
        hp1_.reset(); hp2_.reset(); lp_.reset();
    }

    // ---------------------------------------------------------------------
    // Feed ONE raw acceleration sample (units: m/s^2) at fs_raw.
    // Returns true exactly when a new block spectrum has been computed.
    // ---------------------------------------------------------------------
    bool processSample(double x_raw) {
        // 4th-order high-pass (cascade two identical 2nd-order sections)
        double x_hp = hp2_.process(hp1_.process(x_raw));

        // 2nd-order low-pass (anti-aliasing before decimation)
        double y = lp_.process(x_hp);

        // Decimate by integer factor (keep every D-th sample)
        if (++decimCounter < decimFactor) return false;
        decimCounter = 0;

        // Push to circular buffer at the block (post-decimation) rate
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

    // Latest displacement spectrum S_eta(f) on the fixed grid
    Vec getDisplacementSpectrum() const { return lastSpectrum_; }

    // Access original frequency grid
    std::array<double, Nfreq> getFrequencies() const { return freqs_; }

    // True if at least one full block has been accumulated
    bool ready() const { return isWarm; }

    // Significant wave height from m0 (trapezoidal over the discrete grid)
    // ---------------------------------------------------------------------
    // Significant wave height Hs = 4 * sqrt(m0)
    // m0 = ∫ S_eta(f) df approximated by trapezoidal rule
    // Works for both linear and log-spaced bins (uses Δf[i]).
    // ---------------------------------------------------------------------
    double computeHs() const {
        double m0 = 0.0;

        // trapezoidal integration over uneven frequency grid
        for (int i = 0; i < Nfreq; i++) {
            m0 += lastSpectrum_[i] * df_[i];
        }

        return 4.0 * std::sqrt(std::max(m0, 0.0));
    }

    // Peak frequency using proper log-parabolic interpolation
    double estimateFp() const {
        int k = 0; double vmax = 0.0;
        for (int i = 0; i < Nfreq; ++i) {
            if (lastSpectrum_[i] > vmax) { vmax = lastSpectrum_[i]; k = i; }
        }
        if (k == 0 || k == Nfreq - 1) return freqs_[k];

        // Use log-frequency for true log-parabolic interpolation
        const double f0 = freqs_[k-1], f1 = freqs_[k], f2 = freqs_[k+1];
        if (f0 <= 0 || f1 <= 0 || f2 <= 0) return f1;

        const double x0 = std::log(f0), x1 = std::log(f1), x2 = std::log(f2);
        const double y0 = safeLog(lastSpectrum_[k-1]);
        const double y1 = safeLog(lastSpectrum_[k]);
        const double y2 = safeLog(lastSpectrum_[k+1]);

        // Quadratic fit y = a x^2 + b x + c
        const double dx01 = x0 - x1, dx02 = x0 - x2, dx12 = x1 - x2;
        const double denom = (dx01 * dx02 * dx12);
        if (std::abs(denom) < 1e-18) return f1;

        // Lagrange-based coefficients
        const double L0a = 1.0 / (dx01 * dx02);
        const double L1a = 1.0 / ((x1 - x0) * (x1 - x2));
        const double L2a = 1.0 / ((x2 - x0) * (x2 - x1));
        const double a = y0 * L0a + y1 * L1a + y2 * L2a;

        const double b =
            y0 * (-(x1 + x2) * L0a) +
            y1 * (-(x0 + x2) * L1a) +
            y2 * (-(x0 + x1) * L2a);

        if (a >= 0.0) return f1; // not a concave peak

        const double x_peak = -b / (2.0 * a);
        const double f_peak = std::exp(x_peak);

        if (f_peak <= std::min(f0,f2) || f_peak >= std::max(f0,f2)) return f1;
        return f_peak;
    }

    // Simple PM log-LS fit (α, f_p) with frequency-weighted cost
    PMFitResult fitPiersonMoskowitz() const {
        Vec S_obs = lastSpectrum_;
        for (int i = 0; i < Nfreq; i++) if (S_obs[i] <= 0) S_obs[i] = 1e-12;

        auto cost_fn = [&](double a, double fp) {
            double omega_p = 2.0 * M_PI * fp;
            double cost = 0.0;
            constexpr double beta = 0.74;
            for (int i = 0; i < Nfreq - 1; i++) {
                // Bin width for integration weight
                double df = df_[i];
                double f = freqs_[i];
                double model = a * g * g * std::pow(2.0 * M_PI * f, -5.0)
                               * std::exp(-beta * std::pow(omega_p / (2.0 * M_PI * f), 4.0));
                if (model <= 0) model = 1e-12;
                double d = safeLog(S_obs[i]) - safeLog(model);
                cost += df * d * d;   // weighted by frequency spacing
            }
            return cost;
        };

        // Coarse grid + coordinate descent refinement
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

    // Runtime knobs
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

    void buildFrequencyGrid() {
        constexpr double f_min = 0.04, f_transition = 0.1, f_max = 0.75;
        int n_log = int(Nfreq * 0.4), n_lin = Nfreq - n_log;

        // --- log edges ---
        for (int i = 0; i <= n_log; i++) {
            double t = double(i) / double(n_log);
            f_edges_[i] = f_min * std::pow(f_transition / f_min, t);
        }
        // --- linear edges ---
        for (int i = 1; i <= n_lin; i++) {
            double t = double(i) / double(n_lin);
            f_edges_[n_log + i] = f_transition + t * (f_max - f_transition);
        }

        // --- centers and widths ---
        for (int i = 0; i < Nfreq; i++) {
            if (i < n_log) {
                // geometric mean for log bins
                freqs_[i] = std::sqrt(f_edges_[i] * f_edges_[i+1]);
            } else {
                // arithmetic mean for linear bins
                freqs_[i] = 0.5 * (f_edges_[i] + f_edges_[i+1]);
            }
            df_[i] = f_edges_[i+1] - f_edges_[i];
        }
    }

    // ---------------------------------------------------------------------
    // Bilinear transform (TDF-II transposed) Butterworth-like designs
    // Q≈0.707 per biquad. For HP cascade, two identical sections are used.
    // ---------------------------------------------------------------------
    void designLowpassBiquad(Biquad &bq, double f_cut, double Fs) {
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        bq.b0 = K * K * norm;  bq.b1 = 2.0 * bq.b0;  bq.b2 = bq.b0;
        bq.a1 = 2.0 * (K * K - 1.0) * norm;  bq.a2 = (1.0 - K / Q + K * K) * norm;
    }

    void designHighpassBiquad(Biquad &bq, double f_cut, double Fs) {
        if (f_cut <= 0.0) { // bypass if disabled
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

    // ---------------------------------------------------------------------
    // Compute the block spectrum (called once per filled block at fs)
    // Periodogram is per-Hz (one-sided). HP deconvolution uses
    // frequency-dependent Tikhonov regularization and adaptive η-from-a
    // mapping, so low-frequency bins do not collapse into a flat plateau.
    // ---------------------------------------------------------------------
    void computeSpectrum() {
        const int blockSize = std::min(filledSamples, Nblock);
        const int startIdx  = (writeIndex + Nblock - blockSize) % Nblock;

        // ---- Mean-only detrend ----
        double sumx = 0.0;
        {
            int idx = startIdx;
            for (int n = 0; n < blockSize; ++n) {
                sumx += buffer_[idx];
                idx = (idx + 1) % Nblock;
            }
        }
        const double mean = sumx / double(blockSize);

        // ---- Per-Hz periodogram scaling (one-sided) ----
        const double U = window_sum_sq;                         
        const double base_scale = (U > 0.0 && fs > 0.0) ? (2.0 / (fs * U)) : 0.0;

        // ---- Regularization knee for η-from-a mapping ----
        const double f_reg = std::max(reg_f0_hz, 1e-6);         
        const double wr    = 2.0 * M_PI * f_reg;                

        for (int i = 0; i < Nfreq; i++) {
            const double f = freqs_[i]; // Hz

            // ---- Goertzel at f ----
            double s1 = 0.0, s2 = 0.0;
            int idx = startIdx;
            for (int n = 0; n < blockSize; n++) {
                const double xw = (buffer_[idx] - mean) * window_[n];
                const double s_new = xw + coeffs_[i] * s1 - s2;
                s2 = s1;
                s1 = s_new;
                idx = (idx + 1) % Nblock;
            }

            // |DTFT|^2 at f
            const double power = s1 * s1 + s2 * s2 - s1 * s2 * coeffs_[i];

            // Per-Hz acceleration PSD (one-sided)
            double S_aa_meas = power * base_scale;

            // ---- Deconvolve front-end filters with frequency-dependent regularization ----
            const double Omega_raw = 2.0 * M_PI * f / fs_raw;
            const double H2 =
                biquad_mag2_raw(hp1_, Omega_raw) *
                biquad_mag2_raw(hp2_, Omega_raw) *
                biquad_mag2_raw(lp_ , Omega_raw);

            const double eps = (H2 + 1e-12) * (H2 + 1e-12);

            const double S_aa_true = (H2 > 0.0)
                ? (S_aa_meas * H2 / (H2 * H2 + eps))
                : 0.0;

            // ---- Map acceleration PSD → displacement PSD (adaptive knee) ----
            const double w = 2.0 * M_PI * f;
            const double w_eff = std::sqrt(w * w + wr * wr); // adaptive regularization
            double S_eta = (w_eff > 0.0) ? (S_aa_true / (w_eff * w_eff * w_eff * w_eff)) : 0.0;

            if (!std::isfinite(S_eta) || S_eta < 0.0) S_eta = 0.0;
            lastSpectrum_[i] = S_eta;                             
        }
    }

    // ---------------------------------------------------------------------
    // Magnitude-squared frequency response of a biquad at raw Fs
    //   bq: filter coefficients (TDF-II transposed form)
    //   Ω : normalized rad/sample frequency at fs_raw
    // ---------------------------------------------------------------------
    inline double biquad_mag2_raw(const Biquad& bq, double Omega_raw) const {
        const double c1 = std::cos(Omega_raw), s1 = std::sin(Omega_raw);
        const double c2 = std::cos(2 * Omega_raw), s2 = std::sin(2 * Omega_raw);

        const double num_re = bq.b0 + bq.b1 * c1 + bq.b2 * c2;
        const double num_im = -(bq.b1 * s1 + bq.b2 * s2);

        const double den_re = 1.0 + bq.a1 * c1 + bq.a2 * c2;
        const double den_im = -(bq.a1 * s1 + bq.a2 * s2);

        const double num2 = num_re * num_re + num_im * num_im;
        const double den2 = den_re * den_re + den_im * den_im;

        // Clamp instead of zeroing → prevents log-scale dropouts at low f
        return num2 / std::max(den2, 1e-16);
    }

    // ------------------------- Members / State ----------------------------

    // Rates and decimation
    double fs_raw = 0.0, fs = 0.0;
    int decimFactor = 1;
    bool hannEnabled = true;

    // Regularization and HP corner (Hz)
    double reg_f0_hz = 0.015;  // set to lowest physically meaningful wave frequency
    double hp_f0_hz  = 0.015;   // bias-removal corner for 4th-order HP cascade

    // Spectral grid and Goertzel tables
    std::array<double, Nfreq> freqs_{};
    std::array<double, Nfreq> coeffs_{};
    std::array<double, Nfreq> cos1_{}, sin1_{};

    // Frequency grid edges and bin widths
    std::array<double, Nfreq+1> f_edges_{};
    std::array<double, Nfreq>   df_{};

    // Block buffer (post-decimation) and window
    std::array<double, Nblock> buffer_{};
    std::array<double, Nblock> window_{};
    double window_sum_sq = 1.0;

    // IIR filters at raw Fs
    Biquad hp1_, hp2_, lp_;

    // Output spectrum
    Eigen::Matrix<double, Nfreq, 1> lastSpectrum_;

    // Circular buffer indices/state
    int writeIndex = 0;
    int decimCounter = 0;
    int filledSamples = 0;
    bool isWarm = false;

    // Filter Q (per biquad)
    static constexpr double Q = 0.707;
};

#ifdef SPECTRUM_TEST
// ---------------------------------------------------------
// Minimal offline test with a single-tone acceleration.
// Verifies pipeline runs and returns finite, reasonable stats.
// ---------------------------------------------------------
void WaveSpectrumEstimator_test() {
    constexpr int Nfreq = 32;
    constexpr int Nblock = 256;

    double fs = 240.0;
    WaveSpectrumEstimator<Nfreq, Nblock> estimator(fs, 2, true);

    double f_test = 0.2;
    double A_test = 1.0;
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

            assert(Hs > 0);
            assert(Hs < 3);
            assert(Fp > 0);
            assert(pm.alpha > 0);
            assert(pm.fp > 0);
            (void)S; // silence unused warning in this simple smoke test
        }
    }
    assert(ready_count > 0);
}
#endif
