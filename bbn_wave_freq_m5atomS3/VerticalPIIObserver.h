#pragma once

/*
  Header-only, embedded-friendly vertical-motion observer.

  Purpose
  
  Estimate vertical displacement / velocity from an already gravity-compensated,
  world-frame vertical inertial acceleration input.

  Core non-oscillatory model
      a_f_dot = (a_meas - a_f) / tau_a
      S_dot   = p
      p_dot   = v
      v_dot   = a_f - b_hat - kv*v - kp*p - ks*S

  Optional compile-time very-slow bias-trend channel
      d_dot     = (p - d) / tau_d
      b_hat_dot = kb*d - lambda_b*b_hat

  PII core
  The base PII core is parameterized by a repeated real pole:
      (s + r)^3 = s^3 + 3r s^2 + 3r^2 s + r^3

  so:
      kv = 3r
      kp = 3r^2
      ks = r^3

  Adaptation hooks
  This class includes OPTIONAL runtime hooks to schedule parameters from:
    - displacement-frequency estimate [Hz]
    - acceleration sigma / RMS estimate [same units as acceleration input]

  The intended usage is:
    1) your external tracker estimates:
         f_disp_hz   = dominant / representative displacement frequency
         sigma_accel = std dev / RMS of vertical acceleration
         confidence  = [0..1] quality of those estimates (use 1 if unavailable)
    2) you call:
         observer.update_adaptation(f_disp_hz, sigma_accel, confidence, dt_track);
       whenever those estimates update (for example 2..10 Hz or even slower).
    3) on every IMU sample, call:
         observer.update(a_meas, dt_imu);

  Important notes on adaptation
  - Adaptation is intentionally conservative and all-real-pole.
  - Displacement frequency is treated as a scheduling HINT, not ground truth.
  - Acceleration sigma moderates aggressiveness:
      high sigma -> less aggressive restoring and slower bias learning
      low  sigma -> more aggressive restoring allowed
  - If confidence is low or inputs are invalid, the scheduler simply holds
    its current parameters.

  Embedded-friendliness
  - No dynamic allocation.
  - No exceptions.
  - No Eigen / STL containers.
  - Uses only simple scalar math.
  - Requires C++17 for if constexpr and std::clamp.

  Sign convention
  - Input acceleration must already be the vertical inertial acceleration in
    your chosen world-frame sign convention.
  - Output displacement / velocity follow the same sign convention.
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace marine_obs {

template<bool Enabled, typename T>
struct VerticalBiasState;

template<typename T>
struct VerticalBiasState<true, T> {
    T d = T(0);   // slow displacement trend
    T b = T(0);   // estimated accel bias
};

template<typename T>
struct VerticalBiasState<false, T> {
    // empty
};

template<typename T = float, bool WithBias = true>
class VerticalPIIObserver {
    static_assert(std::is_floating_point<T>::value,
                  "VerticalPIIObserver<T>: T must be a floating-point type.");

public:
    struct Config {
        // Base (reference) observer parameters
        T r      = T(0.16);  // repeated real pole rate for PII core [1/s]
        T tau_a  = T(0.60);  // acceleration LPF time constant [s]

        // Optional bias-trend channel
        T tau_d    = T(40.0);   // slow trend extractor time constant [s]
        T kb       = T(1e-4);   // bias adaptation gain
        T lambda_b = T(1e-2);   // bias leak [1/s]
        T bias_limit = T(0.25); // |b_hat| clamp [m/s^2], <=0 disables clamp

        // State clamps for safety; <=0 disables clamp
        T a_f_limit = T(50.0);
        T v_limit   = T(100.0);
        T p_limit   = T(1000.0);
        T S_limit   = T(10000.0);
        T d_limit   = T(1000.0); // only used when WithBias=true
    };

    struct AdaptConfig {
        bool enabled = false;

        // If confidence < min_confidence, update_adaptation() holds parameters.
        T min_confidence = T(0.25);

        // Valid input ranges for scheduling signals.
        T f_disp_min_hz = T(0.02);
        T f_disp_max_hz = T(1.50);

        T sigma_a_min = T(1e-4);
        T sigma_a_max = T(50.0);

        // Reference values for normalized scheduling.
        // These should reflect a "typical" sea state for which Config::r and
        // Config::tau_a are already reasonable.
        T f_disp_ref_hz = T(0.17);
        T sigma_a_ref   = T(0.30);

        // Smoothing of tracker inputs and scheduled parameters.
        T input_smooth_tau = T(5.0);   // [s] smoothing for f_disp and sigma_a
        T param_smooth_tau = T(10.0);  // [s] smoothing for scheduled params

        // Frequency and sigma influence on the PII pole rate:
        //
        //   r_cmd = r_base
        //           * (f_filt / f_ref) ^ r_freq_exp
        //           * (sigma_ref / sigma_filt) ^ r_sigma_exp
        //
        // Recommended interpretation:
        //   r_freq_exp  > 0 : faster displacement waves allow a somewhat faster PII core
        //   r_sigma_exp > 0 : higher accel sigma reduces restoring aggressiveness
        T r_freq_exp  = T(0.50);
        T r_sigma_exp = T(0.50);

        // Frequency and sigma influence on acceleration LPF time constant:
        //
        //   tau_a_cmd = tau_a_base
        //               * (f_filt / f_ref) ^ tau_a_freq_exp
        //               * (sigma_ref / sigma_filt) ^ tau_a_sigma_exp
        //
        // With defaults:
        //   faster displacement waves -> smaller tau_a (faster LPF)
        //   larger sigma              -> larger tau_a (more smoothing)
        T tau_a_freq_exp  = T(-0.75);
        T tau_a_sigma_exp = T(-0.50);

        // Optional scheduling of bias-trend channel.
        // Conservative defaults:
        //   - tau_d becomes larger when sigma is high
        //   - kb becomes smaller when sigma is high
        T tau_d_freq_exp  = T(0.0);
        T tau_d_sigma_exp = T(-0.50);

        T kb_freq_exp     = T(0.0);
        T kb_sigma_exp    = T(1.00);

        // Hard bounds on scheduled parameters.
        T r_min     = T(0.05);
        T r_max     = T(0.30);

        T tau_a_min = T(0.10);
        T tau_a_max = T(2.00);

        T tau_d_min = T(5.0);
        T tau_d_max = T(120.0);

        T kb_min    = T(0.0);
        T kb_max    = T(1e-2);
    };

    struct Snapshot {
        // States
        T a_f = T(0);
        T v   = T(0);
        T p   = T(0);
        T S   = T(0);
        T d   = T(0);
        T b   = T(0);

        // Base parameters
        T base_r      = T(0);
        T base_tau_a  = T(0);
        T base_tau_d  = T(0);
        T base_kb     = T(0);
        T lambda_b    = T(0);

        // Current scheduled / active parameters
        T r      = T(0);
        T kv     = T(0);
        T kp     = T(0);
        T ks     = T(0);
        T tau_a  = T(0);
        T tau_d  = T(0);
        T kb     = T(0);

        // Adaptation state
        bool adaptation_enabled = false;
        T f_disp_filt_hz = T(0);
        T sigma_a_filt   = T(0);
        T last_confidence = T(0);
    };

public:
    explicit VerticalPIIObserver(const Config& cfg = Config(),
                                 const AdaptConfig& adapt_cfg = AdaptConfig()) {
        configure(cfg);
        configure_adaptation(adapt_cfg);
        reset();
    }

    // Configuration
    void configure(const Config& cfg) {
        cfg_ = cfg;

        base_r_     = std::max(finite_or_default_(cfg.r, T(0.16)), eps_());
        base_tau_a_ = std::max(finite_or_default_(cfg.tau_a, T(0.60)), eps_());

        if constexpr (WithBias) {
            base_tau_d_ = std::max(finite_or_default_(cfg.tau_d, T(40.0)), eps_());
            base_kb_    = std::max(finite_or_zero_(cfg.kb), T(0));
            lambda_b_   = std::max(finite_or_zero_(cfg.lambda_b), T(0));
        } else {
            base_tau_d_ = T(0);
            base_kb_    = T(0);
            lambda_b_   = T(0);
        }

        // When base config changes, reset active params to the new base values.
        active_r_     = base_r_;
        active_tau_a_ = base_tau_a_;
        if constexpr (WithBias) {
            active_tau_d_ = base_tau_d_;
            active_kb_    = base_kb_;
        }

        update_gains_from_active_r_();
        clamp_active_params_();
        clamp_states_();
    }

    void configure_adaptation(const AdaptConfig& cfg) {
        adapt_cfg_ = cfg;
        reset_adaptation_state();
    }

    void set_adaptation_enabled(bool on) {
        adapt_cfg_.enabled = on;
        if (!on) {
            // Return active params to base values immediately.
            active_r_     = base_r_;
            active_tau_a_ = base_tau_a_;
            if constexpr (WithBias) {
                active_tau_d_ = base_tau_d_;
                active_kb_    = base_kb_;
            }
            update_gains_from_active_r_();
            clamp_active_params_();
        }
    }

    bool adaptation_enabled() const {
        return adapt_cfg_.enabled;
    }

    void reset_adaptation_state() {
        f_disp_filt_hz_  = adapt_cfg_.f_disp_ref_hz;
        sigma_a_filt_    = adapt_cfg_.sigma_a_ref;
        last_confidence_ = T(0);

        active_r_     = base_r_;
        active_tau_a_ = base_tau_a_;
        if constexpr (WithBias) {
            active_tau_d_ = base_tau_d_;
            active_kb_    = base_kb_;
        }

        update_gains_from_active_r_();
        clamp_active_params_();
    }

    // State reset
    void reset(T p0 = T(0),
               T v0 = T(0),
               T a_f0 = T(0),
               T S0 = T(0),
               T d0 = T(0),
               T b0 = T(0))
    {
        p_   = finite_or_zero_(p0);
        v_   = finite_or_zero_(v0);
        a_f_ = finite_or_zero_(a_f0);
        S_   = finite_or_zero_(S0);

        if constexpr (WithBias) {
            bias_.d = finite_or_zero_(d0);
            bias_.b = finite_or_zero_(b0);
        }

        clamp_states_();
    }

    // Base parameter setters
    // These define the REFERENCE values for adaptation, and the active values
    // immediately if adaptation is disabled.
    void set_motion_pole_rate(T r) {
        base_r_ = std::max(finite_or_default_(r, base_r_), eps_());
        if (!adapt_cfg_.enabled) {
            active_r_ = base_r_;
            update_gains_from_active_r_();
            clamp_active_params_();
        }
    }

    // Direct manual override of ACTIVE gains.
    // This also disables adaptation because the repeated-pole relation is broken.
    void set_active_pii_gains(T kv, T kp, T ks) {
        set_adaptation_enabled(false);
        kv_ = std::max(finite_or_zero_(kv), T(0));
        kp_ = std::max(finite_or_zero_(kp), T(0));
        ks_ = std::max(finite_or_zero_(ks), T(0));
    }

    void set_accel_filter_tau(T tau_a) {
        base_tau_a_ = std::max(finite_or_default_(tau_a, base_tau_a_), eps_());
        if (!adapt_cfg_.enabled) {
            active_tau_a_ = base_tau_a_;
            clamp_active_params_();
        }
    }

    void set_bias_channel(T tau_d, T kb, T lambda_b, T bias_limit = T(0.25)) {
        if constexpr (WithBias) {
            base_tau_d_ = std::max(finite_or_default_(tau_d, base_tau_d_), eps_());
            base_kb_    = std::max(finite_or_zero_(kb), T(0));
            lambda_b_   = std::max(finite_or_zero_(lambda_b), T(0));
            cfg_.bias_limit = finite_or_zero_(bias_limit);

            if (!adapt_cfg_.enabled) {
                active_tau_d_ = base_tau_d_;
                active_kb_    = base_kb_;
                clamp_active_params_();
            }
        } else {
            (void)tau_d; (void)kb; (void)lambda_b; (void)bias_limit;
        }
    }

    void set_safety_limits(T a_f_limit, T v_limit, T p_limit, T S_limit, T d_limit = T(1000)) {
        cfg_.a_f_limit = finite_or_zero_(a_f_limit);
        cfg_.v_limit   = finite_or_zero_(v_limit);
        cfg_.p_limit   = finite_or_zero_(p_limit);
        cfg_.S_limit   = finite_or_zero_(S_limit);
        cfg_.d_limit   = finite_or_zero_(d_limit);
        clamp_states_();
    }

    // Adaptation hook
    //
    // Call this whenever your external trackers update.
    //
    // Inputs:
    //   f_disp_hz  : representative displacement frequency estimate [Hz]
    //   sigma_a    : std dev / RMS of vertical acceleration
    //   confidence : quality in [0..1], or pass 1 if unavailable
    //   dt         : elapsed time since previous adaptation update [s]
    //
    // Behavior:
    //   - smooths f_disp_hz and sigma_a
    //   - computes scheduled target parameters
    //   - slowly moves active params toward those targets
    //   - if invalid or low-confidence, holds active params
    //
    // You do NOT need to call this at IMU rate.
    // Typical tracker/update rates like 2..10 Hz are fine.
    void update_adaptation(T f_disp_hz, T sigma_a, T confidence, T dt) {
        if (!adapt_cfg_.enabled) return;
        if (!(std::isfinite(dt) && dt > T(0))) return;

        confidence = clamp01_(finite_or_zero_(confidence));
        last_confidence_ = confidence;
        if (confidence < adapt_cfg_.min_confidence) {
            return; // hold current params
        }

        const bool f_ok =
            std::isfinite(f_disp_hz) &&
            f_disp_hz >= adapt_cfg_.f_disp_min_hz &&
            f_disp_hz <= adapt_cfg_.f_disp_max_hz;

        const bool s_ok =
            std::isfinite(sigma_a) &&
            sigma_a >= adapt_cfg_.sigma_a_min &&
            sigma_a <= adapt_cfg_.sigma_a_max;

        if (!(f_ok && s_ok)) {
            return; // hold current params
        }

        // 1) Smooth tracker inputs
        const T alpha_in = one_pole_alpha_(dt, adapt_cfg_.input_smooth_tau);
        f_disp_filt_hz_ += alpha_in * (f_disp_hz - f_disp_filt_hz_);
        sigma_a_filt_   += alpha_in * (sigma_a   - sigma_a_filt_);

        // 2) Normalize
        const T f_ref = std::max(adapt_cfg_.f_disp_ref_hz, eps_());
        const T s_ref = std::max(adapt_cfg_.sigma_a_ref,   eps_());

        const T f_ratio_raw = std::max(f_disp_filt_hz_, eps_()) / f_ref;
        const T s_ratio_raw = s_ref / std::max(sigma_a_filt_, eps_());

        // Mild clamping of ratios before exponentiation.
        const T f_ratio = std::clamp(f_ratio_raw, T(0.25), T(4.0));
        const T s_ratio = std::clamp(s_ratio_raw, T(0.25), T(4.0));

        // 3) Build conservative scheduled targets
        T r_cmd = base_r_
            * safe_pow_(f_ratio, adapt_cfg_.r_freq_exp)
            * safe_pow_(s_ratio, adapt_cfg_.r_sigma_exp);
        r_cmd = std::clamp(r_cmd, adapt_cfg_.r_min, adapt_cfg_.r_max);

        T tau_a_cmd = base_tau_a_
            * safe_pow_(f_ratio, adapt_cfg_.tau_a_freq_exp)
            * safe_pow_(s_ratio, adapt_cfg_.tau_a_sigma_exp);
        tau_a_cmd = std::clamp(tau_a_cmd, adapt_cfg_.tau_a_min, adapt_cfg_.tau_a_max);

        T tau_d_cmd = active_tau_d_;
        T kb_cmd    = active_kb_;

        if constexpr (WithBias) {
            tau_d_cmd = base_tau_d_
                * safe_pow_(f_ratio, adapt_cfg_.tau_d_freq_exp)
                * safe_pow_(s_ratio, adapt_cfg_.tau_d_sigma_exp);
            tau_d_cmd = std::clamp(tau_d_cmd, adapt_cfg_.tau_d_min, adapt_cfg_.tau_d_max);

            kb_cmd = base_kb_
                * safe_pow_(f_ratio, adapt_cfg_.kb_freq_exp)
                * safe_pow_(s_ratio, adapt_cfg_.kb_sigma_exp);
            kb_cmd = std::clamp(kb_cmd, adapt_cfg_.kb_min, adapt_cfg_.kb_max);
        }

        // 4) Smooth active params toward targets
        const T alpha_p = one_pole_alpha_(dt, adapt_cfg_.param_smooth_tau);

        active_r_     += alpha_p * (r_cmd     - active_r_);
        active_tau_a_ += alpha_p * (tau_a_cmd - active_tau_a_);

        if constexpr (WithBias) {
            active_tau_d_ += alpha_p * (tau_d_cmd - active_tau_d_);
            active_kb_    += alpha_p * (kb_cmd    - active_kb_);
        }

        update_gains_from_active_r_();
        clamp_active_params_();
    }

    // Per-sample observer update
    // Returns displacement estimate after update.
    T update(T a_meas, T dt) {
        if (!(std::isfinite(dt) && dt > T(0))) {
            return p_;
        }

        if (!std::isfinite(a_meas)) {
            a_meas = a_f_;
        }

        // 1) Input accel LPF
        {
            const T alpha_a = one_pole_alpha_(dt, active_tau_a_);
            a_f_ += alpha_a * (a_meas - a_f_);
            a_f_ = clamp_symmetric_(a_f_, cfg_.a_f_limit);
        }

        // 2) Optional very-slow bias-trend channel
        T b_use = T(0);
        if constexpr (WithBias) {
            const T alpha_d = one_pole_alpha_(dt, active_tau_d_);

            bias_.d += alpha_d * (p_ - bias_.d);
            bias_.d = clamp_symmetric_(bias_.d, cfg_.d_limit);

            bias_.b += dt * (active_kb_ * bias_.d - lambda_b_ * bias_.b);
            bias_.b = clamp_symmetric_(bias_.b, cfg_.bias_limit);

            b_use = bias_.b;
        }

        // 3) PII core
        const T vdot = a_f_ - b_use - kv_ * v_ - kp_ * p_ - ks_ * S_;

        // Semi-implicit update
        v_ += dt * vdot;
        p_ += dt * v_;
        S_ += dt * p_;

        clamp_states_();
        return p_;
    }

    // Accessors
    // States
    T accel_filtered() const      { return a_f_; }
    T velocity() const            { return v_;   }
    T displacement() const        { return p_;   }
    T integral_displacement() const { return S_; }
    T slow_trend() const {
        if constexpr (WithBias) return bias_.d;
        return T(0);
    }
    T bias_estimate() const {
        if constexpr (WithBias) return bias_.b;
        return T(0);
    }

    // Base params
    T base_motion_pole_rate() const { return base_r_; }
    T base_accel_filter_tau() const { return base_tau_a_; }
    T base_bias_trend_tau() const {
        if constexpr (WithBias) return base_tau_d_;
        return T(0);
    }
    T base_bias_gain() const {
        if constexpr (WithBias) return base_kb_;
        return T(0);
    }
    T bias_leak() const {
        if constexpr (WithBias) return lambda_b_;
        return T(0);
    }

    // Active / scheduled params
    T motion_pole_rate() const { return active_r_; }
    T kv() const { return kv_; }
    T kp() const { return kp_; }
    T ks() const { return ks_; }

    T accel_filter_tau() const { return active_tau_a_; }
    T bias_trend_tau() const {
        if constexpr (WithBias) return active_tau_d_;
        return T(0);
    }
    T bias_gain() const {
        if constexpr (WithBias) return active_kb_;
        return T(0);
    }

    // Adaptation state
    T filtered_displacement_frequency_hz() const { return f_disp_filt_hz_; }
    T filtered_accel_sigma() const               { return sigma_a_filt_;   }
    T last_confidence() const                    { return last_confidence_; }

    Snapshot snapshot() const {
        Snapshot s;
        s.a_f = a_f_;
        s.v   = v_;
        s.p   = p_;
        s.S   = S_;

        s.base_r     = base_r_;
        s.base_tau_a = base_tau_a_;
        s.base_tau_d = base_tau_d_;
        s.base_kb    = base_kb_;
        s.lambda_b   = lambda_b_;

        s.r     = active_r_;
        s.kv    = kv_;
        s.kp    = kp_;
        s.ks    = ks_;
        s.tau_a = active_tau_a_;
        s.tau_d = active_tau_d_;
        s.kb    = active_kb_;

        s.adaptation_enabled = adapt_cfg_.enabled;
        s.f_disp_filt_hz     = f_disp_filt_hz_;
        s.sigma_a_filt       = sigma_a_filt_;
        s.last_confidence    = last_confidence_;

        if constexpr (WithBias) {
            s.d = bias_.d;
            s.b = bias_.b;
        }
        return s;
    }

private:
    // Helpers

    static constexpr T eps_() {
        return T(1e-9);
    }

    static T finite_or_zero_(T x) {
        return std::isfinite(x) ? x : T(0);
    }

    static T finite_or_default_(T x, T def) {
        return std::isfinite(x) ? x : def;
    }

    static T clamp01_(T x) {
        return std::clamp(x, T(0), T(1));
    }

    static T clamp_symmetric_(T x, T limit_abs) {
        if (!(std::isfinite(limit_abs) && limit_abs > T(0))) {
            return x;
        }
        return std::clamp(x, -limit_abs, limit_abs);
    }

    static T one_pole_alpha_(T dt, T tau) {
        if (!(std::isfinite(dt) && dt > T(0))) return T(0);
        if (!(std::isfinite(tau) && tau > T(0))) return T(1);
        const T a = dt / (tau + dt); // no exp(); very embedded-friendly
        return std::clamp(a, T(0), T(1));
    }

    static T safe_pow_(T x, T e) {
        x = std::max(x, eps_());
        return std::pow(x, e);
    }

    void update_gains_from_active_r_() {
        const T r = std::max(active_r_, eps_());
        kv_ = T(3) * r;
        kp_ = T(3) * r * r;
        ks_ = r * r * r;
    }

    void clamp_active_params_() {
        active_r_     = std::max(active_r_, eps_());
        active_tau_a_ = std::max(active_tau_a_, eps_());

        if constexpr (WithBias) {
            active_tau_d_ = std::max(active_tau_d_, eps_());
            active_kb_    = std::max(active_kb_, T(0));
        }
    }

    void clamp_states_() {
        a_f_ = clamp_symmetric_(a_f_, cfg_.a_f_limit);
        v_   = clamp_symmetric_(v_,   cfg_.v_limit);
        p_   = clamp_symmetric_(p_,   cfg_.p_limit);
        S_   = clamp_symmetric_(S_,   cfg_.S_limit);

        if constexpr (WithBias) {
            bias_.d = clamp_symmetric_(bias_.d, cfg_.d_limit);
            bias_.b = clamp_symmetric_(bias_.b, cfg_.bias_limit);
        }
    }

private:
    // User config
    Config cfg_{};
    AdaptConfig adapt_cfg_{};

    // Base / reference params
    T base_r_     = T(0.16);
    T base_tau_a_ = T(0.60);
    T base_tau_d_ = T(40.0);
    T base_kb_    = T(1e-4);
    T lambda_b_   = T(1e-2);

    // Active / scheduled params
    T active_r_     = T(0.16);
    T active_tau_a_ = T(0.60);
    T active_tau_d_ = T(40.0);
    T active_kb_    = T(1e-4);

    // PII gains derived from active_r_
    T kv_ = T(0.48);
    T kp_ = T(0.0768);
    T ks_ = T(0.004096);

    // States
    T a_f_ = T(0);
    T v_   = T(0);
    T p_   = T(0);
    T S_   = T(0);
    VerticalBiasState<WithBias, T> bias_{};

    // Adaptation state
    T f_disp_filt_hz_  = T(0.17);
    T sigma_a_filt_    = T(0.30);
    T last_confidence_ = T(0);
};

} // namespace marine_obs


/*
USAGE EXAMPLE

#include "VerticalPIIObserver.h"

// Bias-enabled observer
marine_obs::VerticalPIIObserver<float, true>::Config cfg;
marine_obs::VerticalPIIObserver<float, true>::AdaptConfig acfg;

void setup_observer() {
    cfg.r      = 0.16f;
    cfg.tau_a  = 0.60f;
    cfg.tau_d  = 40.0f;
    cfg.kb     = 1e-4f;
    cfg.lambda_b = 1e-2f;

    acfg.enabled = true;

    // "Typical" operating point for which cfg.r and cfg.tau_a already feel good
    acfg.f_disp_ref_hz = 0.17f;
    acfg.sigma_a_ref   = 0.30f;

    // Conservative scheduling
    acfg.r_freq_exp    = 0.50f;
    acfg.r_sigma_exp   = 0.50f;
    acfg.tau_a_freq_exp  = -0.75f;
    acfg.tau_a_sigma_exp = -0.50f;

    // Slower bias learning when sigma gets high
    acfg.tau_d_sigma_exp = -0.50f;
    acfg.kb_sigma_exp    =  1.00f;

    // Bounds
    acfg.r_min = 0.05f;
    acfg.r_max = 0.30f;

    marine_obs::VerticalPIIObserver<float, true> obs(cfg, acfg);
}

Typical runtime pattern:

// (1) Whenever your external trackers update, for example at 5 Hz:
obs.update_adaptation(
    displacement_frequency_hz,   // from your displacement-frequency tracker
    accel_sigma,                 // std dev / RMS of vertical acceleration
    tracker_confidence,          // [0..1], or 1.0 if unavailable
    dt_tracker                   // elapsed time since previous tracker update
);

// (2) Every IMU sample:
float z = obs.update(vertical_world_accel, dt_imu);

// Read outputs:
float v = obs.velocity();
float p = obs.displacement();
float b = obs.bias_estimate();

Notes:
- If you do not want adaptation, set acfg.enabled = false and never call
  update_adaptation().
- If you do not want bias estimation at all, instantiate:
    marine_obs::VerticalPIIObserver<float, false>
*/
