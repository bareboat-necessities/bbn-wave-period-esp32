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
#include <algorithm> // std::clamp, std::max

#ifdef SPECTRUM_TEST
#include <iostream>
#include <cassert>
#endif

/*
    WaveSpectrumEstimator — Uniform-bin version

    - Frequency grid is uniformly spaced between f_min..f_max (Nfreq bins → Nfreq+1 edges).
    - Goertzel evaluates exactly at the bin centers used for reporting (consistency).
    - Hs uses midpoint-rule integration: m0 = Σ S_eta(f_i) * Δf_i; Hs = 4√m0.
    - EMA and 3-tap smoothing operate in the ENERGY domain (E_i = S_eta * Δf_i),
      then convert back to density per Hz to avoid inflating wide bins.
    - Raw-rate cascade: HP (2nd) + HP (2nd) + LP (2nd) → decimate → window → Goertzel.
*/

template<int Nfreq = 32, int Nblock = 256>
class EIGEN_ALIGN_MAX WaveSpectrumEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr double g = 9.80665;
    using Vec = Eigen::Matrix<double, Nfreq, 1>;

    struct PMFitResult { double alpha, fp, cost; };

    struct Biquad {
        double b0=0, b1=0, b2=0, a1=0, a2=0;
        double z1=0, z2=0;
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

        buildFrequencyGridAndGoertzel();

        double sumsq = 0.0;
        for (int n = 0; n < Nblock; n++) {
            window_[n] = hannEnabled ? 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (Nblock - 1))) : 1.0;
            sumsq += window_[n] * window_[n];
        }
        window_sum_sq = sumsq;

        reset();

        const double lp_cut = 0.45 * (fs_raw_ / (2.0 * decimFactor)); // guard anti-alias
        designLowpassBiquad(lp_, lp_cut, fs_raw_);
        designHighpassBiquad(hp1_, hp_f0_hz, fs_raw_);
        designHighpassBiquad(hp2_, hp_f0_hz, fs_raw_);
    }

    void reset() {
        buffer_.fill(0.0);
        writeIndex = 0;
        decimCounter = 0;
        filledSamples = 0;
        isWarm = false;

        lastSpectrum_.setZero();
        hp1_.reset(); hp2_.reset(); lp_.reset();

        have_ema = false;
        psd_ema_.setZero();
    }

    // Feed one raw acceleration sample; returns true exactly when a new spectrum is ready.
    bool processSample(double x_raw) {
        // raw-rate HP cascade + LP
        double x_hp = hp2_.process(hp1_.process(x_raw));
        double y    = lp_.process(x_hp);

        // decimate
        if (++decimCounter < decimFactor) return false;
        decimCounter = 0;

        buffer_[writeIndex] = y;
        writeIndex = (writeIndex + 1) % Nblock;
        filledSamples++;

        if (filledSamples >= Nblock) isWarm = true;

        if ((filledSamples % Nblock) == 0) {
            computeSpectrum();
            return true;
        }
        return false;
    }

    Vec getDisplacementSpectrum() const { return lastSpectrum_; }
    std::array<double, Nfreq> getFrequencies() const { return freqs_; }
    bool ready() const { return isWarm; }

    // Hs = 4 * sqrt( ∑ S_eta(f_i) Δf_i ), midpoint rule (values are at bin centers)
    double computeHs() const {
        double m0 = 0.0;
        for (int i = 0; i < Nfreq; i++) m0 += lastSpectrum_[i] * df_[i];
        return 4.0 * std::sqrt(std::max(m0, 0.0));
    }

    // Log-parabolic interpolation around the max bin in log-f / log-S space
    double estimateFp() const {
        int k = 0;
        double vmax = 0.0;
        for (int i = 0; i < Nfreq; ++i) if (lastSpectrum_[i] > vmax) { vmax = lastSpectrum_[i]; k = i; }
        if (k == 0 || k == Nfreq - 1) return freqs_[k];

        const double f0 = freqs_[k-1], f1 = freqs_[k], f2 = freqs_[k+1];
        if (f0 <= 0 || f1 <= 0 || f2 <= 0) return f1;

        const double x0 = std::log(f0), x1 = std::log(f1), x2 = std::log(f2);
        const double y0 = safeLog(lastSpectrum_[k-1]);
        const double y1 = safeLog(lastSpectrum_[k]);
        const double y2 = safeLog(lastSpectrum_[k+1]);

        const double dx01 = x0 - x1, dx02 = x0 - x2, dx12 = x1 - x2;
        const double denom = dx01 * dx02 * dx12;
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
        if (f_peak <= std::min(f0, f2) || f_peak >= std::max(f0, f2)) return f1;
        return f_peak;
    }

    // PM log-LS fit with Δf-weighted cost
    PMFitResult fitPiersonMoskowitz() const {
        Vec S_obs = lastSpectrum_;
        for (int i = 0; i < Nfreq; i++) if (S_obs[i] <= 0) S_obs[i] = 1e-12;

        auto cost_fn = [&](double a, double fp) {
            double omega_p = 2.0 * M_PI * fp;
            double cost = 0.0;
            constexpr double beta = 0.74;
            for (int i = 0; i < Nfreq - 1; i++) {
                double f = freqs_[i], df = df_[i];
                double model = a * g * g * std::pow(2.0 * M_PI * f, -5.0)
                               * std::exp(-beta * std::pow(omega_p / (2.0 * M_PI * f), 4.0));
                if (model <= 0) model = 1e-12;
                double d = safeLog(S_obs[i]) - safeLog(model);
                cost += df * d * d;
            }
            return cost;
        };

        // Uniform grid over fp for robustness (uniform bins case)
        constexpr int N_fp_search = 32;
        constexpr double fp_min = 0.05, fp_max = 1.0;
        std::array<double, N_fp_search> fp_grid;
        for (int i = 0; i < N_fp_search; i++) {
            double t = double(i) / double(N_fp_search - 1);
            fp_grid[i] = fp_min + t * (fp_max - fp_min);
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
            bool improved = false; double c;
            c = cost_fn(alpha + stepA, fp); if (c < bestC) { bestC = c; alpha += stepA; improved = true; }
            c = cost_fn(alpha - stepA, fp); if (c < bestC) { bestC = c; alpha -= stepA; improved = true; }
            c = cost_fn(alpha, fp + stepFp); if (c < bestC) { bestC = c; fp += stepFp; improved = true; }
            c = cost_fn(alpha, fp - stepFp); if (c < bestC) { bestC = c; fp -= stepFp; improved = true; }
            if (!improved) { stepA *= 0.5; stepFp *= 0.5; if (stepA < 1e-12 && stepFp < 1e-12) break; }
        }
        return {alpha, fp, bestC};
    }

    void set_regularization_f0(double f0_hz) { reg_f0_hz = std::max(1e-6, f0_hz); }
    void set_highpass_f0(double f0_hz) {
        hp_f0_hz = std::max(0.0, f0_hz);
        designHighpassBiquad(hp1_, hp_f0_hz, fs_raw);
        designHighpassBiquad(hp2_, hp_f0_hz, fs_raw);
    }

private:
    inline double safeLog(double v) const { return std::log(std::max(v, 1e-18)); }

    inline double alpha_for_f(double f) const {
        const double fmin = freqs_[0];
        const double fmax = freqs_[Nfreq - 1];
        double t = (f - fmin) / std::max(1e-12, (fmax - fmin));
        t = std::clamp(t, 0.0, 1.0);
        return ema_alpha_low + (ema_alpha_high - ema_alpha_low) * t;
    }

    // Energy-domain 3-tap smoothing (uniform bins → symmetric fixed weights ok)
    void smooth_logfreq_3tap() {
        if (Nfreq < 3) return;
        Eigen::Matrix<double, Nfreq, 1> S = lastSpectrum_; // density
        Eigen::Matrix<double, Nfreq, 1> E;                 // energy per bin
        for (int i = 0; i < Nfreq; ++i) E[i] = S[i] * df_[i];

        constexpr double wL = 0.25, wC = 0.5, wR = 0.25;
        Eigen::Matrix<double, Nfreq, 1> Eout = E;
        for (int i = 0; i < Nfreq; ++i) {
            double Em1 = (i > 0) ? E[i-1] : E[i];
            double Ep1 = (i < Nfreq - 1) ? E[i+1] : E[i];
            Eout[i] = wL * Em1 + wC * E[i] + wR * Ep1;
        }
        for (int i = 0; i < Nfreq; ++i) lastSpectrum_[i] = (df_[i] > 0.0) ? (Eout[i] / df_[i]) : 0.0;
    }

    // Uniform frequency grid + Goertzel coefficients at decimated fs
    void buildFrequencyGridAndGoertzel() {
        constexpr double f_min = 0.04;
        constexpr double f_max = 1.2;

        for (int i = 0; i <= Nfreq; i++) {
            double t = double(i) / double(Nfreq);
            f_edges_[i] = f_min + t * (f_max - f_min);
        }
        for (int i = 0; i < Nfreq; i++) {
            freqs_[i] = 0.5 * (f_edges_[i] + f_edges_[i+1]);     // center
            df_[i]    = f_edges_[i+1] - f_edges_[i];             // width
            double omega_rs = 2.0 * M_PI * freqs_[i] / fs;       // rad/sample @ fs
            coeffs_[i] = 2.0 * std::cos(omega_rs);
        }
    }

    // Biquad designs (Butterworth-like, TDF-II transposed)
    void designLowpassBiquad(Biquad &bq, double f_cut, double Fs) {
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        bq.b0 = K * K * norm;  bq.b1 = 2.0 * bq.b0;  bq.b2 = bq.b0;
        bq.a1 = 2.0 * (K * K - 1.0) * norm;  bq.a2 = (1.0 - K / Q + K * K) * norm;
        bq.reset();
    }
    void designHighpassBiquad(Biquad &bq, double f_cut, double Fs) {
        if (f_cut <= 0.0) { bq.b0 = 1.0; bq.b1 = bq.b2 = bq.a1 = bq.a2 = 0.0; bq.reset(); return; }
        double Fc = f_cut / Fs;
        double K = std::tan(M_PI * Fc);
        double norm = 1.0 / (1.0 + K / Q + K * K);
        bq.b0 = 1.0 * norm;  bq.b1 = -2.0 * norm;  bq.b2 = 1.0 * norm;
        bq.a1 = 2.0 * (K * K - 1.0) * norm;  bq.a2 = (1.0 - K / Q + K * K) * norm;
        bq.reset();
    }

    // |H(e^{jΩ})|^2 at raw sample rate
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

    // Compute one spectrum at the decimated rate
    void computeSpectrum() {
        const int N = std::min(filledSamples, Nblock);
        const int startIdx = (writeIndex + Nblock - N) % Nblock;

        // linear detrend over block
        double sumx = 0.0, sumn = 0.0, sumn2 = 0.0, sumnx = 0.0;
        {
            int idx = startIdx;
            for (int n = 0; n < N; ++n) {
                const double xn = buffer_[idx];
                sumx  += xn;
                sumn  += n;
                sumn2 += double(n) * double(n);
                sumnx += double(n) * xn;
                idx = (idx + 1) % Nblock;
            }
        }
        const double denom = double(N) * sumn2 - sumn * sumn;
        const double a_lin = (denom != 0.0) ? (double(N) * sumnx - sumn * sumx) / denom : 0.0;
        const double b_lin = (sumx - a_lin * sumn) / double(N);

        // periodogram scale for one-sided PSD
        const double U = window_sum_sq;
        const double base_scale = (U > 0.0 && fs > 0.0) ? (2.0 / (fs * U)) : 0.0;

        // regularization floor from block duration
        const double Tblk = (fs > 0.0) ? (double(N) / fs) : 0.0;
        const double f_blk = (Tblk > 0.0) ? (1.0 / (6.0 * Tblk)) : 0.0;

        for (int i = 0; i < Nfreq; i++) {
            const double f = freqs_[i];

            // Goertzel at frequency f (rad/sample = 2π f / fs)
            double s1 = 0.0, s2 = 0.0;
            int idx = startIdx;
            for (int n = 0; n < N; n++) {
                const double xn   = buffer_[idx];
                const double xdet = xn - (a_lin * n + b_lin);
                const double xw   = xdet * window_[n];
                const double s_new = xw + coeffs_[i] * s1 - s2;
                s2 = s1;
                s1 = s_new;
                idx = (idx + 1) % Nblock;
            }
            const double power = s1 * s1 + s2 * s2 - s1 * s2 * coeffs_[i];
            double S_aa_meas = power * base_scale;

            // filter deconvolution at RAW rate
            const double Omega_raw = 2.0 * M_PI * f / fs_raw;
            const double H2 =
                biquad_mag2_raw(hp1_, Omega_raw) *
                biquad_mag2_raw(hp2_, Omega_raw) *
                biquad_mag2_raw(lp_ , Omega_raw);

            const double epsilon_H = 0.02 + 0.5 * (0.05 / (f + 1e-6));
            const double S_aa_true = S_aa_meas / (H2 + epsilon_H);

            // displacement PSD via regularized 1/ω^4
            const double f_knee = std::max({reg_f0_hz, f_blk});
            const double wr = 2.0 * M_PI * std::max(f_knee, 0.6 * f);
            const double w = 2.0 * M_PI * f;
            const double w_eff2 = w * w + wr * wr;
            double S_eta = (w_eff2 > 0.0) ? (S_aa_true / (w_eff2 * w_eff2)) : 0.0;

            if (!std::isfinite(S_eta) || S_eta < 0.0) S_eta = 0.0;

            // gentle guard very near DC (optional)
            const double f_guard = 0.055;
            if (f < f_guard) {
                double r = f / f_guard;
                S_eta *= r * r;
            }

            // EMA in ENERGY domain, then convert back to density
            double E_eta = S_eta * df_[i];
            if (use_psd_ema) {
                double a = alpha_for_f(f);
                if (!have_ema) psd_ema_[i] = E_eta;
                else           psd_ema_[i] = (1.0 - a) * psd_ema_[i] + a * E_eta;
                lastSpectrum_[i] = (df_[i] > 0.0) ? (psd_ema_[i] / df_[i]) : 0.0;
            } else {
                lastSpectrum_[i] = S_eta;
            }
        }

        have_ema = true;
        smooth_logfreq_3tap();
    }

    // Members / state
    double fs_raw = 0.0, fs = 0.0;
    int    decimFactor = 1;
    bool   hannEnabled = true;

    // regularization and HP corner (Hz)
    double reg_f0_hz = 0.008;
    double hp_f0_hz  = 0.025;

    // frequency grid
    std::array<double, Nfreq>   freqs_{};
    std::array<double, Nfreq>   coeffs_{};
    std::array<double, Nfreq>   df_{};
    std::array<double, Nfreq+1> f_edges_{};

    // block buffers
    std::array<double, Nblock> buffer_{};
    std::array<double, Nblock> window_{};
    double window_sum_sq = 1.0;

    // filters (raw Fs)
    Biquad hp1_, hp2_, lp_;

    // outputs
    Eigen::Matrix<double, Nfreq, 1> lastSpectrum_;

    // smoothing / EMA
    bool  use_psd_ema = true;
    double ema_alpha_low  = 0.20;
    double ema_alpha_high = 0.06;
    bool  have_ema = false;
    Eigen::Matrix<double, Nfreq, 1> psd_ema_;

    // circular buffer
    int  writeIndex = 0;
    int  decimCounter = 0;
    int  filledSamples = 0;
    bool isWarm = false;

    static constexpr double Q = 0.707;
};

#ifdef SPECTRUM_TEST
inline void WaveSpectrumEstimator_test() {
    constexpr int Nfreq = 32;
    constexpr int Nblock = 256;

    double fs_raw = 240.0;
    int decim = 2;      // fs = 120 Hz
    WaveSpectrumEstimator<Nfreq, Nblock> est(fs_raw, decim, true);

    double f_test = 0.2;
    double A_test = 1.0;
    int N_samples = 2000;

    int ready_count = 0;
    for (int n = 0; n < N_samples; n++) {
        double t = n / fs_raw;
        double acc = A_test * std::sin(2.0 * M_PI * f_test * t);
        if (est.processSample(acc)) {
            ready_count++;
            auto S = est.getDisplacementSpectrum();
            double Hs = est.computeHs();
            double Fp = est.estimateFp();
            auto pm = est.fitPiersonMoskowitz();
            std::cerr << "Hs=" << Hs << " Fp=" << Fp
                      << " PM(alpha=" << pm.alpha << ", fp=" << pm.fp
                      << ", cost=" << pm.cost << ")\n";
            assert(Hs > 0);
            assert(Fp > 0);
            (void)S;
        }
    }
    assert(ready_count > 0);
}
#endif
