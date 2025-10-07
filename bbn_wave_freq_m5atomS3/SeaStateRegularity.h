#pragma once

#include <cmath>
#include <limits>
#include <algorithm>

/*
   SeaStateRegularity : dt-aware, MCU-optimized Bayesian spectral estimator
   -----------------------------------------------------------
   Physical model per bin k (acceleration → displacement):

       a = −ω² η
       Demodulate with cos(ω_k t), −sin(ω_k t):
           y_r ≈ −ω_k² x_r + v
           y_i ≈ −ω_k² x_i + v
           x_{r,i}[n+1] = ρ_k x_{r,i}[n] + w

       ρ_k = exp(−2π f_c,k dt)    — AR(1) process pole
       Q_k = (1 − ρ_k²) σ_x²      — stationary process variance

   Posterior displacement-spectrum power (unbiased):
       Sη(ω_k) = 0.5 (μ_r² + μ_i² + P_rr + P_ii) / ENBW_k
       ENBW_k  = π² f_c,k     (rad/s equivalent noise bandwidth)

   Continuous spectral moments (discrete approximation):
       M₀ = Σ Sη(ω_k) Δω_k
       M₁ = Σ ω_k Sη(ω_k) Δω_k
       M₂ = Σ ω_k² Sη(ω_k) Δω_k
       ν  = √(M₂/M₀ − (M₁/M₀)²) / (M₁/M₀)

   Oceanographic height formulas:
       Hs_rand = 4√M₀            (random sea)
       Hs_mono = 2√(2M₀)         (monochromatic)
       Hs_blend = R_phase·Hs_mono + (1−R_phase)·Hs_rand
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

template<int MAX_K = 20>  // 41 bins by default (2*20+1)
class SeaStateRegularity {
public:
    static constexpr int   NBINS       = 2 * MAX_K + 1;
    static constexpr float PI_         = 3.14159265358979323846f;
    static constexpr float TWO_PI_     = 2.0f * PI_;
    static constexpr float EPS         = 1e-12f;
    static constexpr float F_MIN_HZ    = 0.01f;
    static constexpr float F_MAX_HZ    = 3.00f;
    static constexpr float MIN_FC_HZ   = 0.02f;
    static constexpr float BETA_SPEC   = 1.0f;

    // Noise/tuning
    static constexpr float G_STD       = 9.80665f;
    static constexpr float ACC_NOISE_G = 0.02f;    // IMU noise ≈ 0.02 g RMS
    static constexpr float SIGMA_X0    = 0.10f;    // prior disp envelope [m]
    static constexpr float FC_FRAC     = 0.35f;    // linewidth = 0.35 * Δf_bin

    explicit SeaStateRegularity(float sample_rate_hz = 240.0f) {
        fs_nom_ = sample_rate_hz;
        dt_nom_ = 1.0f / std::max(1e-6f, sample_rate_hz);
        buildGrid();
        reset();
    }

    void reset() {
        for (int i = 0; i < NBINS; ++i) {
            c_[i] = 1.0f; s_[i] = 0.0f;
            mu_r_[i] = mu_i_[i] = 0.0f;
            P_rr_[i] = P_ii_[i] = 4.0f * SIGMA_X0 * SIGMA_X0; // gentle start
            Seta_last_[i] = 0.0f;
            Epow_pk_[i] = 0.0f;
        }
        // Bias KF init
        b_mu_ = 0.0f;
        b_P_  = (0.10f * G_STD) * (0.10f * G_STD);  // prior var ~ (0.10 g)^2
        a2_ema_.reset();

        R_out_.reset();
        R_spec_ = R_phase_ = nu_ = 0.0f;
        omega_peak_ = omega_peak_smooth_ = w_disp_ = 0.0f;
        A0_inst_ = 0.0f;
    }

    inline void update(float dt, float accel_z) {
        if (!(dt > 0.0f) || !std::isfinite(accel_z)) return;
        const bool recompute = std::fabs(dt - dt_nom_) > tol_dt_;
        if (recompute) dt_nom_ = dt;  // keep oscillators consistent if dt wanders

        // --- Bias KF on raw a: a = b + noise (random walk) ---
        b_P_ += b_Q_;                                 // predict
        float S_b   = b_P_ + b_R_;
        float K_b   = b_P_ / std::max(S_b, 1e-20f);   // gain
        float innov_b = accel_z - b_mu_;
        b_mu_ += K_b * innov_b;
        b_P_   = (1.0f - K_b) * b_P_;

        // High-passed acceleration for demod
        const float a_hp = accel_z - b_mu_;

        // Demod measurement variance: track residual variance of a_hp
        a2_ema_.update(a_hp * a_hp, 1.0f - std::exp(-dt / 2.0f));  // ~2 s time-const
        const float Var_hp  = std::max(a2_ema_.get(), b_P_ + b_R_);
        const float R_demod = 0.5f * Var_hp;  // cos/sin halve variance

        // --- advance oscillators + per-bin scalar KFs ---
        for (int i = 0; i < NBINS; ++i) {
            // advance local oscillator
            float cd = cd_[i], sd = sd_[i];
            if (recompute) {
                const float dphi = w_[i] * dt;
                cd = std::cos(dphi);
                sd = std::sin(dphi);
                cd_[i] = cd; sd_[i] = sd;
            }
            const float c0 = c_[i], s0 = s_[i];
            const float c1 = c0 * cd - s0 * sd;
            const float s1 = c0 * sd + s0 * cd;
            c_[i] = c1; s_[i] = s1;

            // demodulated measurements (Kalman does the smoothing)
            const float y_r =  a_hp * c1;
            const float y_i = -a_hp * s1;

            // per-component AR(1) prediction
            const float rho = rho_k_[i];
            mu_r_[i] *= rho;  mu_i_[i] *= rho;
            P_rr_[i] = rho*rho*P_rr_[i] + Qk_[i];
            P_ii_[i] = rho*rho*P_ii_[i] + Qk_[i];

            // measurement update: y ≈ H * x + v, H = -ω^2
            const float H = -w2_[i];

            // real (Joseph-stable)
            {
                const float S = H*H*P_rr_[i] + R_demod;
                const float K = (P_rr_[i] * H) / std::max(S, 1e-20f);
                const float innov = y_r - H * mu_r_[i];
                mu_r_[i] += K * innov;
                P_rr_[i] = (1.0f - K*H) * P_rr_[i] * (1.0f - K*H) + K*R_demod*K;
                P_rr_[i] = std::max(P_rr_[i], 0.0f);
            }
            // imag (Joseph-stable)
            {
                const float S = H*H*P_ii_[i] + R_demod;
                const float K = (P_ii_[i] * H) / std::max(S, 1e-20f);
                const float innov = y_i - H * mu_i_[i];
                mu_i_[i] += K * innov;
                P_ii_[i] = (1.0f - K*H) * P_ii_[i] * (1.0f - K*H) + K*R_demod*K;
                P_ii_[i] = std::max(P_ii_[i], 0.0f);
            }

            // posterior power (ENBW-compensated displacement spectrum)
            const float mu2 = mu_r_[i]*mu_r_[i] + mu_i_[i]*mu_i_[i];
            const float trP = P_rr_[i] + P_ii_[i];
            const float Seta = 0.5f * (mu2 + trP) / enbw_rad_[i]; // PSD at ω_k

            Epow_pk_[i]   = 0.5f * mu2;  // for peak picking ONLY (stable)
            Seta_last_[i] = Seta;        // for moments (unbiased expectation)
        }

        // --- spectral moments from posterior ---
        float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f, Avar = 0.0f;
        for (int i = 0; i < NBINS; ++i) {
            const float mass = Seta_last_[i] * d_omega_[i];
            M0 += mass;
            M1 += mass * w_[i];
            M2 += mass * w2_[i];
            Avar += mass * w2_[i] * w2_[i];  // ⟨a²⟩ = ∫ ω⁴ Sη dω (here integrated via mass*ω⁴)
        }

        // --- peak on μ²-only (refined)
        int ipk = 0; float smax = Epow_pk_[0];
        for (int i = 1; i < NBINS; ++i) if (Epow_pk_[i] > smax) { smax = Epow_pk_[i]; ipk = i; }
        float wpk = w_[ipk];
        if (ipk > 0 && ipk < NBINS - 1) {
            const float hlog = std::log(w_[ipk + 1] / w_[ipk]);
            const float yL = Epow_pk_[ipk - 1], y0 = Epow_pk_[ipk], yR = Epow_pk_[ipk + 1];
            const float denom = std::max(EPS, (yL - 2.0f * y0 + yR));
            float delta = 0.5f * (yL - yR) / denom;
            delta = std::clamp(delta, -1.0f, 1.0f);
            wpk = std::exp(std::log(w_[ipk]) + delta * hlog);
        }
        omega_peak_ = wpk;

        // --- peak smoothing (no centroid mixing; quicker response)
        const float alpha_pk = 1.0f - std::exp(-dt / 2.0f);  // ~2 s
        omega_peak_smooth_ = (omega_peak_smooth_ <= 0.0f)
                           ? wpk
                           : omega_peak_smooth_ + alpha_pk * (wpk - omega_peak_smooth_);
        if (omega_peak_smooth_ > 0.0f) w_disp_ = omega_peak_smooth_;

        // --- phase coherence around peak using posterior means ---
        {
            const int i0 = std::max(0, ipk - 2);
            const int i1 = std::min(NBINS - 1, ipk + 2);
            float Ur = 0.0f, Ui = 0.0f, Wm = 0.0f;
            for (int i = i0; i <= i1; ++i) {
                const float mr = mu_r_[i], mi = mu_i_[i];
                const float amp = std::sqrt(mr*mr + mi*mi);
                if (amp > 1e-12f) {
                    const float ur = mr / amp, ui = mi / amp;
                    const float mass = Seta_last_[i] * d_omega_[i];
                    Ur += mass * ur; Ui += mass * ui; Wm += mass;
                }
            }
            if (Wm > 0.0f) R_phase_ = std::clamp(std::hypot(Ur / Wm, Ui / Wm), 0.0f, 1.0f);
        }

        // --- compute ν, R_spec, and fused regularity from current posterior ---
        if (M0 > EPS) {
            const float omega_bar  = M1 / M0;
            const float omega2_bar = M2 / M0;
            const float var_omega  = std::max(0.0f, omega2_bar - omega_bar * omega_bar);
            nu_ = (omega_bar > EPS) ? (std::sqrt(var_omega) / omega_bar) : 0.0f;
            if (!(std::isfinite(nu_) && nu_ >= 0.0f)) nu_ = 0.0f;

            R_spec_ = std::clamp(std::exp(-BETA_SPEC * nu_), 0.0f, 1.0f);

            const float R_comb   = std::clamp(0.5f * (R_phase_ + R_spec_)
                                            + 0.5f * std::fabs(R_phase_ - R_spec_), 0.0f, 1.0f);
            const float alpha_out = 1.0f - std::exp(-dt / tau_out_);
            R_out_.update(R_comb, alpha_out);
        }

        A0_inst_ = Avar;  // instantaneous ⟨a²⟩ implied by posterior spectrum
    }

    // --- getters (unchanged API) ---
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

    inline float getAccelerationVariance() const { return A0_inst_; }
    inline float getAccelerationSigma() const {
        return (A0_inst_ > 0.0f) ? std::sqrt(A0_inst_) : 0.0f;
    }

    inline float getWaveHeightEnvelopeEst() const {
        float M0 = 0.0f;
        for (int i = 0; i < NBINS; ++i) M0 += Seta_last_[i] * d_omega_[i];
        if (!(M0 > 0.0f)) return 0.0f;
        const float Hs_rand = 4.0f * std::sqrt(M0);
        const float Hs_mono = 2.0f * std::sqrt(2.0f * M0);
        const float R = std::clamp(R_phase_, 0.0f, 1.0f);
        const float Hs = R * Hs_mono + (1.0f - R) * Hs_rand;
        return (std::isfinite(Hs) && Hs > 0.0f) ? Hs : 0.0f;
    }

private:
    // timing
    float fs_nom_ = 240.0f, dt_nom_ = 1.0f / 240.0f, tol_dt_ = 0.0005f;

    // smoothing constants (UI/output only)
    float tau_out_ = 45.0f;

    // frequency grid
    float w_[NBINS]{}, w2_[NBINS]{};
    float d_omega_[NBINS]{};
    float cd_[NBINS]{}, sd_[NBINS]{};
    float c_[NBINS]{}, s_[NBINS]{};

    // per-bin KF parameters
    float rho_k_[NBINS]{}, Qk_[NBINS]{};
    float fc_[NBINS]{}, enbw_rad_[NBINS]{};

    // per-bin KF state (independent scalars for Re/Im)
    float mu_r_[NBINS]{}, mu_i_[NBINS]{};
    float P_rr_[NBINS]{}, P_ii_[NBINS]{};

    // posterior power (moments) and μ²-only (peak)
    float Seta_last_[NBINS]{};
    float Epow_pk_[NBINS]{};

    // bias (DC/tilt) scalar KF on raw acceleration: a = b + noise
    float b_mu_ = 0.0f;
    float b_P_  = (0.10f * G_STD) * (0.10f * G_STD);            // prior variance
    float b_Q_  = (0.002f * G_STD) * (0.002f * G_STD);          // process var per sample (slow drift)
    float b_R_  = (ACC_NOISE_G * G_STD) * (ACC_NOISE_G * G_STD);// measurement variance

    // trackers
    DebiasedEMA a2_ema_;
    DebiasedEMA R_out_;

    // outputs
    float R_spec_ = 0.0f, R_phase_ = 0.0f, nu_ = 0.0f;
    float omega_peak_ = 0.0f, omega_peak_smooth_ = 0.0f, w_disp_ = 0.0f;
    float A0_inst_ = 0.0f;

    void buildGrid() {
        // exact log grid
        const float w_min = TWO_PI_ * F_MIN_HZ;
        const float w_max = TWO_PI_ * F_MAX_HZ;
        const float r     = std::pow(w_max / w_min, 1.0f / float(NBINS - 1));
        const float hlog  = std::log(r);

        for (int i = 0; i < NBINS; ++i) {
            w_[i]  = w_min * std::pow(r, float(i));
            w2_[i] = w_[i] * w_[i];
        }
        const float wfac = 2.0f * std::sinh(0.5f * hlog); // ≈ √r − 1/√r
        for (int i = 0; i < NBINS; ++i)
            d_omega_[i] = std::max(EPS, w_[i] * wfac);

        // initialize oscillators for nominal dt
        for (int i = 0; i < NBINS; ++i) {
            const float dphi = w_[i] * dt_nom_;
            cd_[i] = std::cos(dphi);
            sd_[i] = std::sin(dphi);
            c_[i] = 1.0f; s_[i] = 0.0f;
        }

        // per-bin dynamic + process noise (stationary variance σ_x^2)
        for (int i = 0; i < NBINS; ++i) {
            const float f_i_hz = w_[i] / TWO_PI_;
            const float df_bin = (r - 1.0f) * f_i_hz;
            const float fc     = std::max(MIN_FC_HZ, FC_FRAC * df_bin);
            const float rho    = std::exp(-2.0f * PI_ * fc * dt_nom_);
            rho_k_[i] = rho;
            const float sigma_x2 = SIGMA_X0 * SIGMA_X0;
            Qk_[i] = (1.0f - rho*rho) * sigma_x2;  // AR(1) stationary

            fc_[i]       = fc;
            enbw_rad_[i] = std::max(EPS, float(M_PI) * float(M_PI) * fc); // π²·fc (rad/s)
        }
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
    float amplitude; float omega; float phi;
    SineWave(float A, float f_hz) : amplitude(A), omega(2.0f * float(M_PI) * f_hz), phi(0.0f) {}
    std::pair<float, float> step(float dt) {
        phi += omega * dt; if (phi > 2.0f * float(M_PI)) phi -= 2.0f * float(M_PI);
        float z = amplitude * std::sin(phi);
        float a = -amplitude * omega * omega * std::sin(phi);
        return {z, a};
    }
};

inline void SeaState_sine_wave_test() {
    SineWave wave(SINE_AMPLITUDE, SINE_FREQ_HZ);
    SeaStateRegularity<> reg;
    float R_spec=0, R_phase=0, Hs_est=0, nu=0, f_disp=0, Tp=0;

    for (int i = 0; i < int(SIM_DURATION_SEC / DT); i++) {
        auto za = wave.step(DT);
        reg.update(DT, za.second);
        R_spec  = reg.getRegularitySpectral();
        R_phase = reg.getRegularityPhase();
        Hs_est  = reg.getWaveHeightEnvelopeEst();
        nu      = reg.getNarrowness();
        f_disp  = reg.getDisplacementFrequencyHz();
        Tp      = reg.getDisplacementPeriodSec();
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
              << ", Narrowness=" << nu
              << ", f_disp=" << f_disp << " Hz"
              << ", Tp=" << Tp << " s\n";
}
#endif
