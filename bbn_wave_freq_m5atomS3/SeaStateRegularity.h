#pragma once

#include <cmath>
#include <limits>
#include <algorithm>

/*
   SeaStateRegularity : dt-aware, MCU-optimized Bayesian spectral estimator
   -----------------------------------------------------------
   Model per bin k (acceleration → displacement demod, normalized):

       a = −ω² η
       Demod with cos(ω_k t), +sin(ω_k t):
         y_r = LPF[a·cos(ω_k t)]
         y_i = LPF[a·sin(ω_k t)]

       Normalize measurement with g_k = 2/ω_k² so each bin has H=+1:
         ỹ_r = g_k · y_r ≈ (−1)·x_r + ṽ,   ỹ_i = g_k · y_i ≈ (+1)·x_i + ṽ
         Var[ṽ] = g_k² · R_demod

       State (per bin, real/imag independently):
         x_{r,i}[n+1] = ρ_k x_{r,i}[n] + w,   with   ρ_k = exp(−2π f_c,k dt)
         Q_k = (1 − ρ_k²) σ_x²

   Posterior displacement PSD:
       Sη(ω_k) = 0.5 (μ_r² + μ_i² + P_rr + P_ii) / ENBW_k
       ENBW_k (rad/s) = π² f_c,k

   Moments (discrete):
       M₀ = Σ Sη Δω
       M₁ = Σ ω Sη Δω
       M₂ = Σ ω² Sη Δω
       ν  = √(M₂/M₀ − (M₁/M₀)²) / (M₁/M₀)

   Heights:
       Hs_rand = 4√M₀
       Hs_mono = 2√(2M₀)
       Hs_blend = max(R_phase, R_spec)·Hs_mono + (1−max)·Hs_rand
*/

enum class PoleMap { EXPONENTIAL, TUSTIN };

static inline float rho_from_fc(float fc, float dt, PoleMap map) {
    fc = std::max(0.0f, fc);
    if (map == PoleMap::EXPONENTIAL)
        return std::exp(-2.0f * float(M_PI) * fc * dt);
    const float alpha = float(M_PI) * fc * dt;
    const float denom = 1.0f + alpha;
    return (denom > 1e-20f) ? std::max(0.0f, (1.0f - alpha) / denom) : 0.0f;
}

static inline float enbw_from_rho(float rho, float dt) {
    const float num = 1.0f - rho;
    const float den = 1.0f + rho;
    const float ratio = (den > 1e-20f) ? (num / den) : 0.0f;
    return float(M_PI) * ratio / dt;     // exact one-sided ENBW in rad/s
}

struct DebiasedEMA {
    float value = 0.0f, weight = 0.0f;
    void reset() { value = 0.0f; weight = 0.0f; }
    inline void update(float x, float a) {
        value  = (1.0f - a) * value + a * x;
        weight = (1.0f - a) * weight + a;
    }
    inline void decay(float a) {
        value  = (1.0f - a) * value;
        weight = (1.0f - a) * weight;
    }
    inline float get() const { return (weight > 1e-12f) ? value / weight : 0.0f; }
    inline bool  isReady() const { return weight > 1e-6f; }
};

template<int MAX_K = 15>
class SeaStateRegularity {
public:

    // Grid
    static constexpr int   NBINS       = 2 * MAX_K + 1;
    static constexpr float PI_         = 3.14159265358979323846f;
    static constexpr float TWO_PI_     = 2.0f * PI_;
    static constexpr float EPS         = 1e-12f;
    static constexpr float F_MIN_HZ    = 0.01f;
    static constexpr float F_MAX_HZ    = 3.00f;
    static constexpr float BETA_SPEC   = 1.0f;

    // Noise/tuning
    static constexpr float G_STD       = 9.80665f;
    static constexpr float ACC_NOISE_G = 0.02f;     // sensor white noise [g_rms]
    static constexpr float SIGMA_X0    = 0.10f;     // prior displacement envelope scale

    // fc: fractional linewidth and floors
    static constexpr float MIN_FC_HZ   = 0.0015f;
    static constexpr float MAX_FC_HZ   = 0.30f;
    static constexpr float FC_FRAC     = 0.20f;     // fraction of bin spacing
    static constexpr float FC_REL      = 0.15f;     // relative floor: fc ≥ FC_REL * f

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
            P_rr_[i] = P_ii_[i] = 4.0f * SIGMA_X0 * SIGMA_X0;
            Seta_last_[i] = 0.0f;
            Epow_pk_[i] = 0.0f;
            y_r_lp_[i] = 0.0f;
            y_i_lp_[i] = 0.0f;
        }
        // Bias KF init (raw accel)
        b_mu_ = 0.0f;
        b_P_  = (0.10f * G_STD) * (0.10f * G_STD);
        a2_ema_.reset();

        R_out_.reset();
        R_spec_ = R_phase_ = nu_ = 0.0f;
        omega_peak_ = omega_peak_smooth_ = w_disp_ = 0.0f;
        accel_var_ = 0.0f;
    }

inline void update(float dt, float accel_z) {
    if (!(dt > 0.0f) || !std::isfinite(accel_z)) return;
    const bool recompute = std::fabs(dt - dt_nom_) > tol_dt_;
    if (recompute) dt_nom_ = dt;

    // If dt changed: update oscillators and per-bin filter dynamics
    if (recompute) {
        // update oscillator increments
        for (int i = 0; i < NBINS; ++i) {
            const float dphi = w_[i] * dt_nom_;
            cd_[i] = std::cos(dphi);
            sd_[i] = std::sin(dphi);
        }

        // recompute per-bin filter dynamics and ENBW (exact ρ-based mapping)
        const float r = std::pow(w_[NBINS - 1] / w_[0], 1.0f / float(NBINS - 1));
        for (int i = 0; i < NBINS; ++i) {
            const float f_i_hz = w_[i] / TWO_PI_;
            const float df_bin = (r - 1.0f) * f_i_hz;
            const float fc_raw = std::max({ MIN_FC_HZ, FC_FRAC * df_bin, FC_REL * f_i_hz });
            const float fc     = std::min(fc_raw, MAX_FC_HZ);

            const float rho = rho_from_fc(fc, dt_nom_, pole_map_);
            rho_k_[i] = rho;
            Qk_[i]    = (1.0f - rho * rho) * (SIGMA_X0 * SIGMA_X0);

            fc_[i]       = fc;
            enbw_rad_[i] = std::max(EPS, enbw_from_rho(rho, dt_nom_));  // exact one-sided ENBW (rad/s)
        }
    }

    // === Bias KF on raw acceleration: a = b + noise ===
    b_P_ += b_Q_ * dt;
    const float S_b = b_P_ + b_R_;
    const float K_b = b_P_ / std::max(S_b, 1e-20f);
    const float innov_b = accel_z - b_mu_;
    b_mu_ += K_b * innov_b;
    b_P_   = (1.0f - K_b) * b_P_;

    // High-pass acceleration (bias removed)
    const float a_hp = accel_z - b_mu_;

    // Measurement noise for demod: sensor noise only (no signal bleed)
    a2_ema_.update(a_hp * a_hp, 1.0f - std::exp(-dt / 2.0f)); // diagnostics
    const float R_demod = 0.5f * b_R_; // per I/Q LPF branch

    // === Per-bin Kalman demodulation chain ===
    for (int i = 0; i < NBINS; ++i) {
        // advance local oscillator
        const float c0 = c_[i], s0 = s_[i];
        const float c1 = c0 * cd_[i] - s0 * sd_[i];
        const float s1 = c0 * sd_[i] + s0 * cd_[i];
        c_[i] = c1; s_[i] = s1;

        // demodulated measurements (pre-normalization)
        const float y_r = a_hp * c1;
        const float y_i = a_hp * s1;

        // AR(1) prediction
        const float rho = rho_k_[i];
        mu_r_[i] *= rho;  mu_i_[i] *= rho;
        P_rr_[i] = rho * rho * P_rr_[i] + Qk_[i];
        P_ii_[i] = rho * rho * P_ii_[i] + Qk_[i];

        // LPF the demodulated measurements with the bin pole (α = 1 − ρ)
        const float alpha_lp = 1.0f - rho;
        y_r_lp_[i] += alpha_lp * (y_r - y_r_lp_[i]);
        y_i_lp_[i] += alpha_lp * (y_i - y_i_lp_[i]);

        // Normalization: g = 2 / ω²  →  H = +1 (sign absorbed in innovation)
        const float w2 = w2_[i];
        // Include pole attenuation correction
const float g = (w2 > EPS) ? (2.0f / (w2 * (1.0f - rho))) : 0.0f;

// Effective measurement noise after LPF: (1−ρ)/(1+ρ)
const float R_demod_eff = 0.5f * b_R_ * ((1.0f - rho) / (1.0f + rho));
const float Rm = (g * g) * R_demod_eff;

// --- Real channel ---
{
    const float y_tilde = g * y_r_lp_[i];
    const float innov   = -(y_tilde) - mu_r_[i];
    const float S = P_rr_[i] + Rm;
    const float K = (S > 1e-20f) ? (P_rr_[i] / S) : 0.0f;
    mu_r_[i] += K * innov;
    const float I_K = 1.0f - K;
    P_rr_[i] = I_K * P_rr_[i] * I_K + K * Rm * K;
    if (P_rr_[i] < 0.0f) P_rr_[i] = 0.0f;
}

// --- Imag channel ---  (make it symmetrical with the real channel)
{
    const float y_tilde = g * y_i_lp_[i];
    const float innov   = -(y_tilde) - mu_i_[i];  // <-- was: y_tilde - mu_i_[i]
    const float S = P_ii_[i] + Rm;
    const float K = (S > 1e-20f) ? (P_ii_[i] / S) : 0.0f;
    mu_i_[i] += K * innov;
    const float I_K = 1.0f - K;
    P_ii_[i] = I_K * P_ii_[i] * I_K + K * Rm * K;
    if (P_ii_[i] < 0.0f) P_ii_[i] = 0.0f;
}
       
        // Posterior displacement PSD (ENBW-compensated)
        const float mu2  = mu_r_[i] * mu_r_[i] + mu_i_[i] * mu_i_[i];
        const float trP  = P_rr_[i] + P_ii_[i];
        const float Seta = 0.5f * (mu2) / (enbw_rad_[i]);
       
        // Peak metric on ω·Sη prevents low-ω bias
        Epow_pk_[i]   = w_[i] * Seta;
        Seta_last_[i] = Seta;
    }

    // === Spectral moments and regularity ===
    float M0 = 0.0f, M1 = 0.0f, M2 = 0.0f, Avar = 0.0f;
    for (int i = 0; i < NBINS; ++i) {
        const float mass = Seta_last_[i] * d_omega_[i];
        M0 += mass;
        M1 += mass * w_[i];
        M2 += mass * w2_[i];
        Avar += mass * w2_[i] * w2_[i];
    }

    // peak frequency (log-parabolic refinement)
    int ipk = 0; float smax = Epow_pk_[0];
    for (int i = 1; i < NBINS; ++i)
        if (Epow_pk_[i] > smax) { smax = Epow_pk_[i]; ipk = i; }

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

    // smooth frequency (τ≈3s)
    const float alpha_pk = 1.0f - std::exp(-dt / 3.0f);
    omega_peak_smooth_ = (omega_peak_smooth_ <= 0.0f)
        ? wpk
        : omega_peak_smooth_ + alpha_pk * (wpk - omega_peak_smooth_);
    if (omega_peak_smooth_ > 0.0f)
        w_disp_ = omega_peak_smooth_;

    // === Phase coherence around peak (±3 ENBWs) ===
    {
        const float span = 3.0f * enbw_rad_[ipk];
        int i0 = ipk, i1 = ipk;
        while (i0 > 0 && std::fabs(w_[i0 - 1] - wpk) <= span) --i0;
        while (i1 < NBINS - 1 && std::fabs(w_[i1 + 1] - wpk) <= span) ++i1;

        float Ur = 0.0f, Ui = 0.0f, Wm = 0.0f;
        for (int i = i0; i <= i1; ++i) {
            const float mr = mu_r_[i], mi = mu_i_[i];
            const float amp = std::sqrt(mr * mr + mi * mi);
            if (amp > 1e-12f) {
                const float ur = mr / amp, ui = mi / amp;
                const float mass = Seta_last_[i] * d_omega_[i];
                Ur += mass * ur; Ui += mass * ui; Wm += mass;
            }
        }
        if (Wm > 0.0f)
            R_phase_ = std::clamp(std::hypot(Ur / Wm, Ui / Wm), 0.0f, 1.0f);
    }

    // === Spectral narrowness and regularity fusion ===
    if (M0 > EPS) {
        const float omega_bar  = M1 / M0;
        const float omega2_bar = M2 / M0;
        const float var_omega  = std::max(0.0f, omega2_bar - omega_bar * omega_bar);
        nu_ = (omega_bar > EPS) ? (std::sqrt(var_omega) / omega_bar) : 0.0f;
        if (!(std::isfinite(nu_) && nu_ >= 0.0f)) nu_ = 0.0f;

        R_spec_ = std::clamp(std::exp(-BETA_SPEC * nu_), 0.0f, 1.0f);

        const float R_comb = std::max(R_phase_, R_spec_);
        const float alpha_out = 1.0f - std::exp(-dt / tau_out_);
        R_out_.update(R_comb, alpha_out);
    }

    accel_var_ = Avar;  // instantaneous ⟨a²⟩ from posterior spectrum
}

    // getters
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

    inline float getAccelerationVariance() const { return accel_var_; }
    inline float getAccelerationSigma() const {
        return (accel_var_ > 0.0f) ? std::sqrt(accel_var_) : 0.0f;
    }

    inline float getWaveHeightEnvelopeEst() const {
        float M0 = 0.0f;
        for (int i = 0; i < NBINS; ++i) M0 += Seta_last_[i] * d_omega_[i];
        if (!(M0 > 0.0f)) return 0.0f;
        const float R = std::clamp(std::max(R_phase_, R_spec_), 0.0f, 1.0f);
        const float Hs_rand = 4.0f * std::sqrt(M0);
        const float Hs_mono = 2.0f * std::sqrt(2.0f * M0);
        const float Hs = R * Hs_mono + (1.0f - R) * Hs_rand;
        return (std::isfinite(Hs) && Hs > 0.0f) ? Hs : 0.0f;
    }

private:

    PoleMap pole_map_ = PoleMap::EXPONENTIAL;   // or TUSTIN if you use bilinear

    // LPF state for demodulated I/Q (matched to bin pole ρ)
    float y_r_lp_[NBINS]{}, y_i_lp_[NBINS]{};

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

    // posterior power (moments) and peak metric
    float Seta_last_[NBINS]{};
    float Epow_pk_[NBINS]{};

    // bias (DC/tilt) scalar KF on raw acceleration: a = b + noise
    float b_mu_ = 0.0f;
    float b_P_  = (0.10f * G_STD) * (0.10f * G_STD);
    float b_Q_  = (0.002f * G_STD) * (0.002f * G_STD); // per second; integrated with + b_Q_*dt
    float b_R_  = (ACC_NOISE_G * G_STD) * (ACC_NOISE_G * G_STD); // measurement variance

    // trackers
    DebiasedEMA a2_ema_;
    DebiasedEMA R_out_;

    // outputs
    float R_spec_ = 0.0f, R_phase_ = 0.0f, nu_ = 0.0f;
    float omega_peak_ = 0.0f, omega_peak_smooth_ = 0.0f, w_disp_ = 0.0f;
    float accel_var_ = 0.0f;

void buildGrid() {
    // exact log grid
    const float w_min = TWO_PI_ * F_MIN_HZ;
    const float w_max = TWO_PI_ * F_MAX_HZ;
    const float r     = std::pow(w_max / w_min, 1.0f / float(NBINS - 1));
    const float hlog  = std::log(r);

    // frequency grid and ω²
    for (int i = 0; i < NBINS; ++i) {
        w_[i]  = w_min * std::pow(r, float(i));
        w2_[i] = w_[i] * w_[i];
    }

    // Δω from log grid spacing (accurate second-order)
    const float wfac = 2.0f * std::sinh(0.5f * hlog);
    for (int i = 0; i < NBINS; ++i)
        d_omega_[i] = std::max(EPS, w_[i] * wfac);

    // initialize oscillators for nominal dt
    for (int i = 0; i < NBINS; ++i) {
        const float dphi = w_[i] * dt_nom_;
        cd_[i] = std::cos(dphi);
        sd_[i] = std::sin(dphi);
        c_[i]  = 1.0f; s_[i] = 0.0f;
    }

    // per-bin dynamics and ENBW (using exact rho-based mapping)
    for (int i = 0; i < NBINS; ++i) {
        const float f_i_hz = w_[i] / TWO_PI_;
        const float r      = std::pow(w_[NBINS-1] / w_[0], 1.0f / float(NBINS - 1));
        const float df_bin = (r - 1.0f) * f_i_hz;

        // Wider relative floor for faster convergence and less spectral smearing
        const float fc_raw = std::max({ MIN_FC_HZ, 0.3f * df_bin, 0.15f * f_i_hz });
        const float fc     = std::min(fc_raw, MAX_FC_HZ);

        const float rho = rho_from_fc(fc, dt_nom_, pole_map_);
        rho_k_[i] = rho;
        Qk_[i]    = (1.0f - rho * rho) * (SIGMA_X0 * SIGMA_X0);

        fc_[i]       = fc;
        enbw_rad_[i] = std::max(EPS, enbw_from_rho(rho, dt_nom_));   // exact one-sided ENBW (rad/s)
    }
}

};

#ifdef SEA_STATE_TEST
#include <iostream>
#include <stdexcept>

// Simple self-check: 60 s of a 0.3 Hz sine with amplitude 1 m.
// Expect: R_spec>0.9, R_phase>0.8, nu<0.05, Hs≈4 m (±20%), Tp≈3.33 s (±20%).
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
        float z = amplitude * std::cos(phi);
        float a = -amplitude * omega * omega * std::cos(phi);
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
    const float Tp_expected = 1.0f / SINE_FREQ_HZ;

    if (!(R_spec > 0.90f))  throw std::runtime_error("Sine: R_spec not near 1.");
    if (!(R_phase > 0.80f)) throw std::runtime_error("Sine: R_phase not near 1.");
    if (!(std::fabs(Hs_est - Hs_expected) < 0.20f * Hs_expected))
        throw std::runtime_error("Sine: Hs estimate exceeds 20% error.");
    if (!(std::fabs(Tp - Tp_expected) < 0.20f * Tp_expected))
        throw std::runtime_error("Sine: Tp estimate exceeds 20% error.");
    if (!(nu < 0.05f))      throw std::runtime_error("Sine: ν should be near 0.");

    std::cerr << "[PASS] Sine wave test. "
              << "Hs_est=" << Hs_est
              << ", ν=" << nu
              << ", f_disp=" << f_disp << " Hz"
              << ", Tp=" << Tp << " s\n";
}
#endif
