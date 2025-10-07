#pragma once

#include <cmath>
#include <limits>
#include <algorithm>

/*
   Copyright 2025, Mikhail Grushinskiy

   SeaStateRegularity : no-tracker, dt-aware, MCU-optimized spectral estimator

   Input:
       a_z(t)  - vertical acceleration [m/s^2]
       dt      - time step [s]

   Output (getters):
       getRegularity()               - fused regularity score [0–1]
       getRegularitySpectral()       - bandwidth-based regularity
       getRegularityPhase()          - phase-coherence regularity
       getNarrowness()               - normalized spectral width ν
       getDisplacementFrequencyHz()  - dominant displacement frequency
       getDisplacementPeriodSec()    - corresponding period
       getAccelerationVariance()     - broadband variance ⟨a²⟩
       getAccelerationSigma()        - RMS acceleration σ_a = √⟨a²⟩
       getWaveHeightEnvelopeEst()    - blended significant wave height Hs

   Physical basis:
       a = −ω² η    ⇒   S_eta(ω) = S_accel(ω) / ω⁴
       M0 = ∫ S_eta dω
       M1 = ∫ ω S_eta dω
       M2 = ∫ ω² S_eta dω
       ν  = sqrt(M2/M0 − (M1/M0)²) / (M1/M0)
       Hs_rand = 4√M0      (Rayleigh/random sea)
       Hs_mono = 2√(2M0)   (deterministic sine)
       Hs_blend = R_phase·Hs_mono + (1−R_phase)·Hs_rand

   Implementation notes:
       * Fixed log-spaced ω grid (ratio ≈1.06)
       * Each bin maintains its own oscillator and 1-pole LPF
       * Jensen correction removes bias in M1/M0 ratio
       * Handles small timing jitter: recomputes sin/cos & α only if |dt−dt_nom|>tol
       * Pure single-precision; all math deterministic and portable
*/

struct DebiasedEMA {
    float value = 0.0f, weight = 0.0f;
    void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float a) {
        value  = (1 - a) * value + a * x;
        weight = (1 - a) * weight + a;
    }
    inline float get() const { return (weight > 1e-12f) ? value / weight : 0.0f; }
    inline bool  isReady() const { return weight > 1e-6f; }
};

template<int MAX_K = 20>  // 41 bins (2*20+1). Use 25 for 51-bin version.
class SeaStateRegularity {
public:
    static constexpr int   NBINS       = 2 * MAX_K + 1;
    static constexpr float PI_         = 3.14159265358979323846f;
    static constexpr float TWO_PI_     = 2.0f * PI_;
    static constexpr float EPS         = 1e-12f;
    static constexpr float STEP_LOG    = 0.06f;
    static constexpr float F_MIN_HZ    = 0.01f;
    static constexpr float F_MAX_HZ    = 3.00f;
    static constexpr float MIN_FC_HZ   = 0.02f;
    static constexpr float K_EFF_MIX   = 2.0f;
    static constexpr float BETA_SPEC   = 1.0f;

    explicit SeaStateRegularity(float sample_rate_hz = 240.0f) {
        fs_nom_ = sample_rate_hz;
        dt_nom_ = 1.0f / std::max(1e-6f, sample_rate_hz);
        buildGrid();
        reset();
    }

    void reset() {
        for (int i = 0; i < NBINS; ++i) {
            c_[i] = 1.0f; s_[i] = 0.0f;
            zr_[i] = zi_[i] = 0.0f;
            Seta_last_[i] = 0.0f;
        }
        M0_.reset(); M1_.reset(); M2_.reset(); A0_.reset();
        Q00_.reset(); Q10_.reset();
        coh_r_.reset(); coh_i_.reset(); R_out_.reset();
        R_spec_ = R_phase_ = nu_ = 0.0f;
        omega_bar_naive_ = omega_bar_corr_ = 0.0f;
        omega_peak_ = omega_peak_smooth_ = w_disp_ = 0.0f;
    }

    inline void update(float dt, float accel_z) {
        if (!(dt > 0.0f) || !std::isfinite(accel_z)) return;
        bool recompute = std::fabs(dt - dt_nom_) > tol_dt_;

        float S0 = 0, S1 = 0, S2 = 0, Avar = 0;
        float sumWr = 0, sumWi = 0, W = 0;

        for (int i = 0; i < NBINS; ++i) {
            float cd = cd_[i], sd = sd_[i];
            if (recompute) {
                float dphi = w_[i] * dt;
                cd = std::cos(dphi);
                sd = std::sin(dphi);
            }
            float c0 = c_[i], s0 = s_[i];
            float c1 = c0 * cd - s0 * sd;
            float s1 = c0 * sd + s0 * cd;
            c_[i] = c1; s_[i] = s1;

            float y_r =  accel_z * c1;
            float y_i = -accel_z * s1;

            float a = alpha_k_[i];
            if (recompute) {
                const float fc_hz = fc_k_[i];
                a = 1.0f - std::exp(-dt * TWO_PI_ * fc_hz);
            }
            float zr = zr_[i] + a * (y_r - zr_[i]);
            float zi = zi_[i] + a * (y_i - zi_[i]);
            zr_[i] = zr; zi_[i] = zi;

            float mag2 = zr * zr + zi * zi;
            float Seta = KEFF_over_ENBW_[i] * mag2 * inv_w4_[i];
            Seta_last_[i] = Seta;

            float mass = Seta * d_omega_[i];
            S0 += mass;
            S1 += mass * w_[i];
            S2 += mass * w2_[i];
            Avar += Seta * w4domega_[i];

            float m = std::sqrt(mag2);
            if (m > EPS) {
                float ur = zr / m, ui = zi / m;
                sumWr += mass * ur;
                sumWi += mass * ui;
                W += mass;
            }
        }

        updateMomentsAndRegularity(S0, S1, S2, Avar, sumWr, sumWi, W, dt);
    }

    inline float getRegularity() const             { return R_out_.get(); }
    inline float getRegularitySpectral() const     { return R_spec_; }
    inline float getRegularityPhase() const        { return R_phase_; }
    inline float getNarrowness() const             { return nu_; }

    inline float getDisplacementFrequencyHz() const {
        return (w_disp_ > EPS) ? w_disp_ / TWO_PI_ : 0.0f;
    }
    inline float getDisplacementPeriodSec() const {
        return (w_disp_ > EPS) ? (TWO_PI_ / w_disp_) : 0.0f;
    }

    inline float getAccelerationVariance() const { return A0_.get(); }
    inline float getAccelerationSigma() const {
        float v = A0_.get();
        return (v > 0.0f) ? std::sqrt(v) : 0.0f;
    }

    inline float getWaveHeightEnvelopeEst() const {
        float m0 = M0_.get();
        if (!(m0 > 0)) return 0;
        float Hs_rand = 4.0f * std::sqrt(m0);
        float Hs_mono = 2.0f * std::sqrt(2.0f * m0);
        float R = std::clamp(R_phase_, 0.0f, 1.0f);
        float Hs = R * Hs_mono + (1 - R) * Hs_rand;
        return (std::isfinite(Hs) && Hs > 0) ? Hs : 0;
    }

private:
    float fs_nom_ = 240.0f, dt_nom_ = 1.0f / 240.0f, tol_dt_ = 0.0008f;
    float tau_mom_ = 180.0f, tau_coh_ = 60.0f, tau_out_ = 45.0f;

    float w_[NBINS]{}, w2_[NBINS]{}, inv_w4_[NBINS]{};
    float d_omega_[NBINS]{}, w2domega_[NBINS]{}, w4domega_[NBINS]{};
    float cd_[NBINS]{}, sd_[NBINS]{}, alpha_k_[NBINS]{}, fc_k_[NBINS]{}, KEFF_over_ENBW_[NBINS]{};

    float c_[NBINS]{}, s_[NBINS]{}, zr_[NBINS]{}, zi_[NBINS]{}, Seta_last_[NBINS]{};

    DebiasedEMA M0_, M1_, M2_, A0_, Q00_, Q10_, coh_r_, coh_i_, R_out_;
    float R_spec_ = 0, R_phase_ = 0, nu_ = 0;
    float omega_bar_naive_ = 0, omega_bar_corr_ = 0;
    float omega_peak_ = 0, omega_peak_smooth_ = 0, w_disp_ = 0;

    int i_peak_ = 0;

    void buildGrid() {
        const float w_min = TWO_PI_ * F_MIN_HZ;
        const float w_max = TWO_PI_ * F_MAX_HZ;

        // Exact log grid over [w_min, w_max]
        const float r    = std::pow(w_max / w_min, 1.0f / float(NBINS - 1));
        const float hlog = std::log(r);

        for (int i = 0; i < NBINS; ++i) {
            w_[i]   = w_min * std::pow(r, float(i));
            w2_[i]  = w_[i] * w_[i];
            inv_w4_[i] = 1.0f / std::max(w2_[i] * w2_[i], EPS);
        }

        // Midpoint-based Δω for a log grid: dω ≈ w * (√r − 1/√r)
        const float wfac = 2.0f * std::sinh(0.5f * hlog);      // = √r − 1/√r
        for (int i = 0; i < NBINS; ++i) {
            const float domega = std::max(EPS, w_[i] * wfac);
            d_omega_[i]  = domega;
            w2domega_[i] = w2_[i] * domega;
            w4domega_[i] = w2_[i] * w2_[i] * domega;
        }

        // LPF cutoff: narrower than bin spacing (good isolation, high R_phase)
        // Δf_bin ≈ (r − 1) * f
        constexpr float C_FC = 0.35f;   // 0.3–0.5 works; 0.35 is a solid start
        for (int i = 0; i < NBINS; ++i) {
            const float f_i_hz  = w_[i] / TWO_PI_;
            const float df_bin  = (r - 1.0f) * f_i_hz;               // Hz
            const float fc_hz   = std::max(MIN_FC_HZ, C_FC * df_bin);
            fc_k_[i]   = fc_hz;
            alpha_k_[i]= 1.0f - std::exp(-dt_nom_ * TWO_PI_ * fc_hz);

            // ENBW in rad/s, single-sided: ENBW_ω = (π/2)*ω_c = π² * fc
            const float ENBW_omega = PI_ * PI_ * fc_hz;
            KEFF_over_ENBW_[i] = K_EFF_MIX / std::max(ENBW_omega, EPS);

            const float dphi = w_[i] * dt_nom_;
            cd_[i] = std::cos(dphi);
            sd_[i] = std::sin(dphi);
        }
    }

    void pickPeak() {
        int i_max = 0; float s_max = Seta_last_[0];
        for (int i = 1; i < NBINS; ++i)
            if (Seta_last_[i] > s_max) { s_max = Seta_last_[i]; i_max = i; }

        // Log-parabolic refine around the local max
        float wpk = w_[i_max];
        if (i_max > 0 && i_max < NBINS - 1) {
            float hlog = std::log(w_[i_max + 1] / w_[i_max]); // == log(r)
            float yL = Seta_last_[i_max - 1];
            float y0 = Seta_last_[i_max];
            float yR = Seta_last_[i_max + 1];
            float denom = std::max(EPS, (yL - 2 * y0 + yR));
            float delta = 0.5f * (yL - yR) / denom;
            delta = std::clamp(delta, -1.0f, 1.0f);
            float xstar = std::log(w_[i_max]) + delta * hlog;
            wpk = std::exp(xstar);
        }
        i_peak_ = i_max;                                    // <— remember it
        omega_peak_ = wpk;

        const float alpha_mom = 1.0f - std::exp(-dt_nom_ / tau_mom_);
        omega_peak_smooth_ = (omega_peak_smooth_ <= 0.0f)
                             ? omega_peak_
                             : omega_peak_smooth_ + alpha_mom * (omega_peak_ - omega_peak_smooth_);
    }

    void updateMomentsAndRegularity(float S0, float S1, float S2, float Avar,
                                    float sumWr, float sumWi, float W, float dt) {
        float alpha_mom  = 1.0f - std::exp(-dt / tau_mom_);
        float alpha_coh  = 1.0f - std::exp(-dt / tau_coh_);
        float alpha_out  = 1.0f - std::exp(-dt / tau_out_);
        float alpha_disp = 1.0f - std::exp(-dt / 7.0f);

        M0_.update(S0, alpha_mom);
        M1_.update(S1, alpha_mom);
        M2_.update(S2, alpha_mom);
        Q00_.update(S0 * S0, alpha_mom);
        Q10_.update(S0 * S1, alpha_mom);
        A0_.update(Avar, alpha_mom);

        // detect dominant peak first
        pickPeak();

        // Phase coherence around the dominant peak only
        int i0 = std::max(0, i_peak_ - 2);
        int i1 = std::min(NBINS - 1, i_peak_ + 2);

        float sum_r = 0.0f, sum_i = 0.0f, Wloc = 0.0f;
        for (int i = i0; i <= i1; ++i) {
            const float zr = zr_[i], zi = zi_[i];
            const float m  = std::sqrt(zr * zr + zi * zi);
            if (m > EPS) {
                const float ur = zr / m, ui = zi / m;
                const float mass = Seta_last_[i] * d_omega_[i];
                sum_r += mass * ur;
                sum_i += mass * ui;
                Wloc  += mass;
            }
        }
        if (Wloc > EPS) {
            const float urW = sum_r / Wloc;
            const float uiW = sum_i / Wloc;
            coh_r_.update(urW, alpha_coh);
            coh_i_.update(uiW, alpha_coh);
            R_phase_ = std::clamp(std::hypot(coh_r_.get(), coh_i_.get()), 0.0f, 1.0f);
        }

        computeRegularity(alpha_out, alpha_disp);
    }

    void computeRegularity(float alpha_out, float alpha_disp) {
        if (!M0_.isReady()) {
            R_out_.update(R_phase_, alpha_out);
            R_spec_ = R_phase_; nu_ = 0; omega_bar_naive_ = omega_bar_corr_ = 0;
            return;
        }

        float m0 = M0_.get(), m1 = M1_.get(), m2 = M2_.get();
        if (!(m0 > EPS)) {
            R_out_.update(0, alpha_out);
            R_spec_ = 0; nu_ = 0; omega_bar_naive_ = omega_bar_corr_ = 0;
            return;
        }

        omega_bar_naive_ = m1 / m0;
        float omega2_bar = m2 / m0;
        float mu2 = std::max(0.0f, omega2_bar - omega_bar_naive_ * omega_bar_naive_);
        float varM0 = std::max(0.0f, Q00_.get() - m0 * m0);
        float cov10 = Q10_.get() - m1 * m0;
        float invM0_2 = 1.0f / std::max(m0 * m0, EPS);
        omega_bar_corr_ = omega_bar_naive_ + (omega_bar_naive_ * varM0 - cov10) * invM0_2;

        nu_ = (omega_bar_corr_ > EPS) ? (std::sqrt(mu2) / omega_bar_corr_) : 0;
        if (!(std::isfinite(nu_) && nu_ >= 0)) nu_ = 0;
        if (R_phase_ > 0.9f && nu_ < 0.2f) {
            float w = std::clamp((R_phase_ - 0.9f) / 0.1f, 0.0f, 1.0f);
            nu_ *= (1.0f - w);
        }

        R_spec_ = std::clamp(std::exp(-BETA_SPEC * nu_), 0.0f, 1.0f);
        float R_comb = std::clamp(
            0.5f * (R_phase_ + R_spec_) + 0.5f * std::fabs(R_phase_ - R_spec_), 0.0f, 1.0f);
        R_out_.update(R_comb, alpha_out);

        if (omega_peak_smooth_ > 0.0f)
            w_disp_ = (w_disp_ <= 0) ? omega_peak_smooth_
                                     : w_disp_ + alpha_disp * (omega_peak_smooth_ - w_disp_);
    }
};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

constexpr float SAMPLE_FREQ_HZ   = 240.0f;
constexpr float DT               = 1.0f / SAMPLE_FREQ_HZ;
constexpr float SIM_DURATION_SEC = 60.0f;
constexpr float SINE_AMPLITUDE   = 1.0f;
constexpr float SINE_FREQ_HZ     = 0.3f;

struct SineWave {
    float amplitude;
    float omega;
    float phi;
    SineWave(float A, float f_hz)
        : amplitude(A), omega(2.0f * float(M_PI) * f_hz), phi(0.0f) {}
    std::pair<float, float> step(float dt) {
        phi += omega * dt;
        if (phi > 2.0f * float(M_PI)) phi -= 2.0f * float(M_PI);
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

inline void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity<> reg;
    float R_spec = 0.0f, R_phase = 0.0f, Hs_est = 0.0f, nu = 0.0f;
    float f_disp_corr = 0.0f, Tp = 0.0f;

    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        float a = za.second;
        reg.update(DT, a);
        R_spec      = reg.getRegularitySpectral();
        R_phase     = reg.getRegularityPhase();
        Hs_est      = reg.getWaveHeightEnvelopeEst();
        nu          = reg.getNarrowness();
        f_disp_corr = reg.getDisplacementFrequencyHz();
        Tp          = reg.getDisplacementPeriodSec();
    }

    const float Hs_expected = 4.0f * SINE_AMPLITUDE;

    if (!(R_spec > 0.90f))
        throw std::runtime_error("Sine: R_spec did not converge near 1.");
    if (!(R_phase > 0.80f))
        throw std::runtime_error("Sine: R_phase did not converge near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.25f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate not within tolerance.");
    if (!(nu < 0.05f))
        throw std::runtime_error("Sine: Narrowness should be close to 0 for a pure tone.");

    std::cerr << "[PASS] Sine wave test passed. "
              << "Hs_est=" << Hs_est
              << " (expected ~" << Hs_expected << "), Narrowness=" << nu
              << ", f_disp_corr=" << f_disp_corr << " Hz"
              << ", Tp=" << Tp << " s\n";
}
#endif
