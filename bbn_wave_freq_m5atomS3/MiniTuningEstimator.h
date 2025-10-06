#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * Copyright 2025, Mikhail Grushinskiy
 *
 * MiniTuningEstimator
 * --------------------
 *
 * Purpose:
 *   Tracks RMS acceleration σₐ and the smoothed acceleration-domain peak frequency ωₚₑₐₖ
 *   from streaming acceleration data. Derived quantities include correlation time τ,
 *   dominant period Tₚ, and a heuristic pseudo-measurement noise coefficient R_S
 *   used for stabilizing displacement integral drift correction in Kalman filters.
 *
 * Inputs per sample:
 *   dt_s [s]          : timestep
 *   accel_z [m/s²]    : vertical acceleration
 *   omega_inst [rad/s]: instantaneous angular frequency of the ACCELERATION signal
 *
 * Outputs:
 *   σₐ [m/s²]         : RMS acceleration (bias-corrected)
 *   τ  [s]            : correlation time ≈ c_tau / ωₚₑₐₖ
 *   Tₚ [s]            : acceleration-domain period = 2π / ωₚₑₐₖ
 *   R_S [m²·s²]       : heuristic pseudo-measurement noise scale for displacement integral drift
 *
 * Notes:
 *   • ωₚₑₐₖ refers to the *acceleration* spectral peak, not displacement.
 */

class MiniTuningEstimator {
public:
    explicit MiniTuningEstimator(float tau_mom_sec        = 120.0f,   // averaging window for accel variance
                                 float tau_peak_smooth_sec = 12.0f,   // smoothing window for ωₚₑₐₖ
                                 float c_tau               = 1.0f,    // τ = c_tau / ωₚₑₐₖ
                                 float R_S_base            = 1.9f,    // baseline pseudo-noise (m·s)²
                                 float T_p_base            = 2.6f)    // reference period [s]
        : c_tau_(c_tau),
          R_S_base_(R_S_base),
          T_p_base_(T_p_base)
    {
        tau_mom_         = std::max(1e-3f, tau_mom_sec);
        tau_peak_smooth_ = std::max(1e-3f, tau_peak_smooth_sec);
        reset();
    }

    // Reset internal state
    void reset() noexcept {
        M2_ = 0.0f;
        weight_mom_ = 0.0f;
        has_moments_ = false;
        omega_peak_smooth_ = 0.0f;
        has_peak_ = false;
    }

    // Update with one sample
    void update(float dt_s, float accel_z, float omega_inst) noexcept {
        if (!(dt_s > 0.0f)) return;

        // Clamp and validate ω_inst
        if (!(omega_inst > 0.0f))
            omega_inst = OMEGA_MIN;
        omega_inst = std::clamp(omega_inst, OMEGA_MIN, OMEGA_MAX);

        // Accel variance (σₐ², bias-corrected)
        const float a2 = accel_z * accel_z;
        const float a_m = alpha(dt_s, tau_mom_);
        const float b_m = 1.0f - a_m;
        if (!has_moments_) {
            M2_ = a2;
            weight_mom_ = 1.0f;
            has_moments_ = true;
        } else {
            M2_ = b_m * M2_ + a_m * a2;
            weight_mom_ = b_m * weight_mom_ + a_m;
        }

        // Peak frequency tracker (smoothed accel ω)
        const float a_pk = alpha(dt_s, tau_peak_smooth_);
        if (!has_peak_) {
            omega_peak_smooth_ = omega_inst;
            has_peak_ = true;
        } else {
            // Robust smoothing: slower update when ω_inst drops sharply
            const float rel = std::clamp(
                omega_inst / std::max(omega_peak_smooth_, OMEGA_MIN), 0.25f, 4.0f);
            const float adj_alpha = a_pk * std::pow(rel, 0.3f);
            omega_peak_smooth_ =
                (1.0f - adj_alpha) * omega_peak_smooth_ + adj_alpha * omega_inst;
        }
    }

    // Outputs
    [[nodiscard]] float getSigmaA() const noexcept {
        return (weight_mom_ > EPS && M2_ > EPS)
                   ? std::sqrt(M2_ / weight_mom_)
                   : 0.0f;
    }

    [[nodiscard]] float getOmegaPeak() const noexcept {
        return has_peak_
                   ? std::clamp(omega_peak_smooth_, OMEGA_MIN, OMEGA_MAX)
                   : 0.0f;
    }

    [[nodiscard]] float getTau() const noexcept {
        const float wpk = getOmegaPeak();
        return (std::isfinite(wpk) && wpk > EPS) ? (c_tau_ * static_cast<float>(M_PI) / wpk) : 0.0f;
    }

    [[nodiscard]] float getPeriodPeak() const noexcept {
        const float w = getOmegaPeak();
        return (w > EPS) ? (2.0f * static_cast<float>(M_PI) / w) : 0.0f;
    }

    // Heuristic pseudo-measurement noise scaling law
    //
    // Interpretation:
    //   • R_S provides a dimensionally consistent scaling term for the
    //     displacement pseudo-measurement covariance used to regularize
    //     integral drift in the Kalman update.
    [[nodiscard]] float getR_S() const noexcept {
        return R_S_law(getPeriodPeak());
    }

    // R_S(Tₚ) = R_S_base_ * (Tₚ / Tₚ_base_)^(1/3)
    [[nodiscard]] float R_S_law(float T_p) const noexcept {
        if (!(T_p > 1e-6f))
            return 0.0f;
        return R_S_base_ * std::pow(T_p / T_p_base_, 1.0f / 3.0f);
    }

private:
    // Accel variance state
    float M2_ = 0.0f;
    float weight_mom_ = 0.0f;
    bool  has_moments_ = false;

    // Peak frequency (acceleration domain)
    float omega_peak_smooth_ = 0.0f;
    bool  has_peak_          = false;

    // Time constants
    float tau_mom_ = 120.0f;
    float tau_peak_smooth_ = 12.0f;
    float c_tau_ = 1.0f;

    // R_S-law baseline parameters
    float R_S_base_ = 1.9f;
    float T_p_base_ = 2.6f;

    // Numeric limits
    static constexpr float OMEGA_MIN = 0.2f;   // rad/s (accel freq lower bound)
    static constexpr float OMEGA_MAX = 8.0f;   // rad/s (accel freq upper bound)
    static constexpr float EPS       = 1e-12f;

    // Helper
    static constexpr float alpha(float dt, float tau) noexcept {
        return 1.0f - std::exp(-dt / std::max(tau, 1e-6f));
    }
};

