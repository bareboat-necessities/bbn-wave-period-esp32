#pragma once

/*
  VerticalPIIObserver.h
  Header-only, embedded-friendly vertical-motion observer.

  Purpose
  -------
  Estimate vertical displacement/velocity from an already gravity-compensated
  vertical world-frame acceleration input.

  Core non-oscillatory model
  --------------------------
      a_f_dot = (a_meas - a_f) / tau_a
      S_dot   = p
      p_dot   = v
      v_dot   = a_f - b_hat - kv*v - kp*p - ks*S

  Optional very-slow bias-trend channel (compile-time)
  ----------------------------------------------------
      d_dot     = (p - d) / tau_d
      b_hat_dot = kb*d - lambda_b*b_hat

  The PII core is parameterized by a repeated real pole:
      (s + r)^3 = s^3 + 3r s^2 + 3r^2 s + r^3

  so:
      kv = 3r
      kp = 3r^2
      ks = r^3

  Notes
  -----
  - Input acceleration must already be "vertical inertial acceleration" in your
    chosen sign convention.
  - Output displacement/velocity follow the same sign convention.
  - No dynamic allocation.
  - Requires C++17 for if constexpr.
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
        // PII repeated real pole (s + r)^3
        T r = T(0.16);

        // Input accel LPF time constant [s]
        T tau_a = T(0.6);

        // Optional slow bias-trend channel
        T tau_d    = T(40.0);   // slow trend extractor time constant [s]
        T kb       = T(1e-4);   // bias adaptation gain
        T lambda_b = T(1e-2);   // bias leak [1/s]
        T bias_limit = T(0.25); // |b_hat| clamp [m/s^2], <=0 disables clamp

        // State clamps for safety; <=0 disables the clamp
        T a_f_limit = T(50.0);
        T v_limit   = T(100.0);
        T p_limit   = T(1000.0);
        T S_limit   = T(10000.0);
    };

    struct Snapshot {
        T a_f = T(0);
        T v   = T(0);
        T p   = T(0);
        T S   = T(0);
        T d   = T(0);
        T b   = T(0);

        T r   = T(0);
        T kv  = T(0);
        T kp  = T(0);
        T ks  = T(0);
    };

    explicit VerticalPIIObserver(const Config& cfg = Config()) {
        configure(cfg);
        reset();
    }

    void configure(const Config& cfg) {
        cfg_ = cfg;
        set_motion_pole_rate(cfg.r);
        set_accel_filter_tau(cfg.tau_a);
        if constexpr (WithBias) {
            set_bias_channel(cfg.tau_d, cfg.kb, cfg.lambda_b, cfg.bias_limit);
        }
    }

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

    // Repeated real pole (s + r)^3
    void set_motion_pole_rate(T r) {
        r_  = std::max(finite_or_default_(r, T(0.16)), eps_());
        kv_ = T(3) * r_;
        kp_ = T(3) * r_ * r_;
        ks_ = r_ * r_ * r_;
    }

    // Direct override if you want custom all-real gains.
    void set_pii_gains(T kv, T kp, T ks) {
        kv_ = std::max(finite_or_zero_(kv), T(0));
        kp_ = std::max(finite_or_zero_(kp), T(0));
        ks_ = std::max(finite_or_zero_(ks), T(0));
    }

    void set_accel_filter_tau(T tau_a) {
        tau_a_ = std::max(finite_or_default_(tau_a, T(0.6)), eps_());
    }

    void set_safety_limits(T a_f_limit, T v_limit, T p_limit, T S_limit) {
        cfg_.a_f_limit = finite_or_zero_(a_f_limit);
        cfg_.v_limit   = finite_or_zero_(v_limit);
        cfg_.p_limit   = finite_or_zero_(p_limit);
        cfg_.S_limit   = finite_or_zero_(S_limit);
        clamp_states_();
    }

    void set_bias_channel(T tau_d, T kb, T lambda_b, T bias_limit = T(0.25)) {
        cfg_.tau_d      = std::max(finite_or_default_(tau_d, T(40)), eps_());
        cfg_.kb         = std::max(finite_or_zero_(kb), T(0));
        cfg_.lambda_b   = std::max(finite_or_zero_(lambda_b), T(0));
        cfg_.bias_limit = finite_or_zero_(bias_limit);
    }

    // One sample update.
    // Returns current displacement estimate after update.
    T update(T a_meas, T dt) {
        if (!(std::isfinite(dt) && dt > T(0))) {
            return p_;
        }

        // Sanitize input
        if (!std::isfinite(a_meas)) {
            a_meas = a_f_;
        }

        // 1) Input accel LPF: alpha = dt / (tau + dt)
        {
            const T alpha_a = one_pole_alpha_(dt, tau_a_);
            a_f_ += alpha_a * (a_meas - a_f_);
            a_f_ = clamp_symmetric_(a_f_, cfg_.a_f_limit);
        }

        // 2) Very-slow bias-trend channel (optional)
        T b_use = T(0);
        if constexpr (WithBias) {
            const T alpha_d = one_pole_alpha_(dt, cfg_.tau_d);

            // Slow displacement trend extractor
            bias_.d += alpha_d * (p_ - bias_.d);

            // Slow, leaky bias adaptation
            bias_.b += dt * (cfg_.kb * bias_.d - cfg_.lambda_b * bias_.b);
            bias_.b = clamp_symmetric_(bias_.b, cfg_.bias_limit);

            b_use = bias_.b;
        }

        // 3) Non-oscillatory PII core
        const T vdot = a_f_ - b_use - kv_ * v_ - kp_ * p_ - ks_ * S_;

        // Semi-implicit update
        v_ += dt * vdot;
        p_ += dt * v_;
        S_ += dt * p_;

        clamp_states_();
        return p_;
    }

    // Accessors
    T accel_filtered() const { return a_f_; }
    T velocity() const       { return v_;   }
    T displacement() const   { return p_;   }
    T integral_displacement() const { return S_; }

    T motion_pole_rate() const { return r_; }
    T kv() const { return kv_; }
    T kp() const { return kp_; }
    T ks() const { return ks_; }

    T accel_filter_tau() const { return tau_a_; }

    T slow_trend() const {
        if constexpr (WithBias) return bias_.d;
        return T(0);
    }

    T bias_estimate() const {
        if constexpr (WithBias) return bias_.b;
        return T(0);
    }

    Snapshot snapshot() const {
        Snapshot s;
        s.a_f = a_f_;
        s.v   = v_;
        s.p   = p_;
        s.S   = S_;
        s.r   = r_;
        s.kv  = kv_;
        s.kp  = kp_;
        s.ks  = ks_;
        if constexpr (WithBias) {
            s.d = bias_.d;
            s.b = bias_.b;
        }
        return s;
    }

private:
    static constexpr T eps_() {
        return T(1e-9);
    }

    static T finite_or_zero_(T x) {
        return std::isfinite(x) ? x : T(0);
    }

    static T finite_or_default_(T x, T def) {
        return std::isfinite(x) ? x : def;
    }

    static T one_pole_alpha_(T dt, T tau) {
        if (!(std::isfinite(dt) && dt > T(0))) return T(0);
        if (!(std::isfinite(tau) && tau > T(0))) return T(1);
        const T a = dt / (tau + dt); // embed-friendly, no exp()
        return std::clamp(a, T(0), T(1));
    }

    static T clamp_symmetric_(T x, T limit_abs) {
        if (!(std::isfinite(limit_abs) && limit_abs > T(0))) {
            return x;
        }
        return std::clamp(x, -limit_abs, limit_abs);
    }

    void clamp_states_() {
        a_f_ = clamp_symmetric_(a_f_, cfg_.a_f_limit);
        v_   = clamp_symmetric_(v_,   cfg_.v_limit);
        p_   = clamp_symmetric_(p_,   cfg_.p_limit);
        S_   = clamp_symmetric_(S_,   cfg_.S_limit);

        if constexpr (WithBias) {
            bias_.b = clamp_symmetric_(bias_.b, cfg_.bias_limit);
        }
    }

private:
    Config cfg_{};

    // PII core params
    T r_    = T(0.16);
    T kv_   = T(0.48);
    T kp_   = T(0.0768);
    T ks_   = T(0.004096);

    // Input accel LPF
    T tau_a_ = T(0.6);

    // States
    T a_f_ = T(0);
    T v_   = T(0);
    T p_   = T(0);
    T S_   = T(0);

    VerticalBiasState<WithBias, T> bias_{};
};

} // namespace marine_obs
