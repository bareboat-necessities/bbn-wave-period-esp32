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

    struct PMFitResult { double alpha, fp, cost; };

    WaveSpectrumEstimator(double fs_raw_ = 240.0,
                          int decimFactor_ = 5,
                          bool hannEnabled_ = true)
        : fs_raw(fs_raw_), decimFactor(decimFactor_), hannEnabled(hannEnabled_)
    {
        fs = fs_raw / decimFactor;
        buildFrequencyGrid();

        // Precompute Goertzel coefficients (rad/sample)
        for (int i = 0; i < Nfreq; i++) {
            double omega_rs = 2.0 * M_PI * freqs_[i] / fs; // rad/sample
            coeffs_[i] = 2.0 * std::cos(omega_rs);
            cos1_[i]  = std::cos(omega_rs);
            sin1_[i]  = std::sin(omega_rs);
        }

        // Hann window and its squared-sum (full-block)
        double sumsq = 0.0;
        for (int n = 0; n < Nblock; n++) {
            window_[n] = hannEnabled ? 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1))) : 1.0;
            sumsq += window_[n] * window_[n];
        }
        window_sum_sq = sumsq;

        reset();

        // Low-pass biquad (applied at raw Fs)
        double cutoffHz = 0.45 * (fs_raw_ / (2.0 * decimFactor));
        designLowpassBiquad(cutoffHz, fs_raw_);
    }

    void reset() {
        buffer_.fill(0.0);
        writeIndex = 0; decimCounter = 0;
        filledSamples = 0; z1 = z2 = 0.0;
        isWarm = false;
        lastSpectrum_.setZero();
    }

    // feed raw acceleration sample (Hz = fs_raw)
    // returns true when a block-spectrum has been computed
bool processSample(double x_raw) {
    // ---- Fixed low-pass biquad (TDF-II transposed form) ----
    double y = b0 * x_raw + z1;
    z1 = b1 * x_raw - a1 * y + z2;   // <-- minus a1
    z2 = b2 * x_raw - a2 * y;        // <-- minus a2

    // --- Decimation ---
    if (++decimCounter < decimFactor)
        return false;                // skip until next keep-sample
    decimCounter = 0;

    // --- Circular buffer insert ---
    buffer_[writeIndex] = y;
    writeIndex = (writeIndex + 1) % Nblock;
    filledSamples++;

    // Warm-up flag
    if (filledSamples >= Nblock)
        isWarm = true;

    // Trigger spectrum computation once per full block
    if (filledSamples > 0 && (filledSamples % Nblock) == 0) {
        computeSpectrum();           // (your DC-removed version)
        return true;                 // new spectrum ready
    }
    return false;                    // no spectrum yet
}

    Vec getDisplacementSpectrum() const { return lastSpectrum_; }

    double computeHs() const {
        double m0 = 0.0;
        for (int i = 0; i < Nfreq - 1; i++) {
            double df = freqs_[i + 1] - freqs_[i];
            m0 += 0.5 * (lastSpectrum_[i] + lastSpectrum_[i + 1]) * df;
        }
        return 4.0 * std::sqrt(std::max(m0, 0.0));
    }

    double estimateFp() const {
        int idx = 0; double maxVal = 0;
        for (int i = 0; i < Nfreq; i++) {
            if (lastSpectrum_[i] > maxVal) { maxVal = lastSpectrum_[i]; idx = i; }
        }

        if (idx > 0 && idx < Nfreq - 1) {
            double y0 = safeLog(lastSpectrum_[idx - 1]);
            double y1 = safeLog(lastSpectrum_[idx]);
            double y2 = safeLog(lastSpectrum_[idx + 1]);
            double denom = y0 - 2.0 * y1 + y2;
            if (std::abs(denom) < 1e-12) return freqs_[idx];
            double p = 0.5 * (y0 - y2) / denom;
            double df_avg = 0.5 * ((freqs_[idx] - freqs_[idx - 1]) + (freqs_[idx + 1] - freqs_[idx]));
            return freqs_[idx] + p * df_avg;
        }
        return freqs_[idx];
    }

    PMFitResult fitPiersonMoskowitz() const {
        Vec S_obs = lastSpectrum_;
        for (int i = 0; i < Nfreq; i++) if (S_obs[i] <= 0) S_obs[i] = 1e-12;

        auto cost_fn = [&](double a, double fp) {
            double omega_p = 2.0 * M_PI * fp;
            double cost = 0.0;
            constexpr double beta = 0.74;
            for (int i = 0; i < Nfreq; i++) {
                double f = freqs_[i];
                double model = a * g * g * std::pow(2.0 * M_PI * f, -5.0)
                               * std::exp(-beta * std::pow(omega_p / (2.0 * M_PI * f), 4.0));
                if (model <= 0) model = 1e-12;
                double d = safeLog(S_obs[i]) - safeLog(model);
                cost += d * d;
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

    bool ready() const { return isWarm; }

private:
    inline double safeLog(double v) const { return std::log(std::max(v, 1e-18)); }

    void buildFrequencyGrid() {
        constexpr double f_min = 0.04, f_transition = 0.1, f_max = 1.0;
        int n_log = int(Nfreq * 0.4), n_lin = Nfreq - n_log;
        for (int i = 0; i < n_log; i++) {
            double t = double(i) / (n_log - 1);
            freqs_[i] = f_min * std::pow(f_transition / f_min, t);
        }
        for (int i = 0; i < n_lin; i++) {
            double t = double(i) / (n_lin - 1);
            freqs_[n_log + i] = f_transition + t * (f_max - f_transition);
        }
    }

    void designLowpassBiquad(double f_cut, double Fs) {
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        b0 = K * K * norm; b1 = 2.0 * b0; b2 = b0;
        a1 = 2.0 * (K * K - 1.0) * norm; a2 = (1.0 - K / Q + K * K) * norm;
    }

void computeSpectrum() {
    // number of valid decimated samples available (use up to Nblock)
    const int blockSize = std::min(filledSamples, Nblock);

    // start index = oldest sample in circular buffer
    const int startIdx = (writeIndex + Nblock - blockSize) % Nblock;

    // --- NEW: compute (unwindowed) block mean to remove DC ---
    double xmean = 0.0;
    {
        int idx = startIdx;
        for (int n = 0; n < blockSize; n++) {
            xmean += buffer_[idx];
            idx = (idx + 1) % Nblock;
        }
        xmean /= double(blockSize);
    }

    // window energy for the actual block size
    double U = 0.0;
    for (int n = 0; n < blockSize; n++) U += window_[n] * window_[n];

    // single-sided PSD scale for acceleration: 2 / (fs * sum(w^2))
    const double scale_factor = (U > 0.0) ? (2.0 / (fs * U)) : 0.0;

    for (int i = 0; i < Nfreq; i++) {
        double s1 = 0.0, s2 = 0.0;

        int idx = startIdx;
        for (int n = 0; n < blockSize; n++) {
            // --- NEW: subtract mean before windowing ---
            const double xw = (buffer_[idx] - xmean) * window_[n];
            const double s_new = xw + coeffs_[i] * s1 - s2;
            s2 = s1; s1 = s_new;
            idx = (idx + 1) % Nblock;
        }

        // standard Goertzel recombination (equivalent to s1^2 + s2^2 - 2cos*w*s1*s2)
        const double real = s1 - s2 * cos1_[i];
        const double imag = s2 * sin1_[i];

        // PSD of acceleration at this frequency ( (m/s^2)^2 / Hz )
        const double S_aa = (real*real + imag*imag) * scale_factor;

        // acceleration -> displacement PSD: divide by ω^4
        const double omega = 2.0 * M_PI * freqs_[i];     // rad/s
        const double omega_safe = std::max(omega, 2.0 * M_PI * 0.03); // same floor as before
        double S_eta = S_aa / (omega_safe * omega_safe * omega_safe * omega_safe);

        if (!std::isfinite(S_eta) || S_eta < 0.0) S_eta = 0.0;
        lastSpectrum_[i] = S_eta; // [m^2/Hz]
    }
}

    double fs_raw, fs;
    int decimFactor;
    bool hannEnabled;

    std::array<double, Nfreq> freqs_;
    std::array<double, Nfreq> coeffs_;
    std::array<double, Nblock> buffer_;
    std::array<double, Nblock> window_;
    double window_sum_sq = 1.0;

    double b0, b1, b2, a1, a2;
    double z1 = 0, z2 = 0;

    std::array<double, Nfreq> cos1_, sin1_;
    Eigen::Matrix<double, Nfreq, 1> lastSpectrum_;

    int writeIndex = 0;
    int decimCounter = 0;
    int filledSamples = 0;
    bool isWarm = false;

    static constexpr double Q = 0.707;
};

#ifdef SPECTRUM_TEST
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
            assert(Fp > 0);
            assert(pm.alpha > 0);
            assert(pm.fp > 0);
        }
    }

    assert(ready_count > 0);
}
#endif
