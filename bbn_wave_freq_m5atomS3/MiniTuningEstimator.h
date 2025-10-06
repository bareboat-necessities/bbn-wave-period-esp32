#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * MiniTuningEstimator (robust version)
 *
 * Inputs per sample:
 *   dt_s [s]          : timestep
 *   accel_z [m/s²]    : vertical acceleration
 *   omega_inst [rad/s]: instantaneous angular frequency of the ACCELERATION signal
 *
 * Outputs:
 *   σₐ [m/s²]         : RMS acceleration (bias-corrected)
 *   τ  [s]            : correlation time ≈ c_tau / ω_peak
 *
 * Notes:
 *   • ω_peak refers to the *acceleration* spectral peak, not displacement.
 *   • If you want displacement-domain frequency, use getDisplacementOmegaPeak().
 */

class MiniTuningEstimator {
public:
    explicit MiniTuningEstimator(float tau_mom_sec        = 120.0f,  // averaging for accel variance
                                 float tau_peak_smooth_sec = 12.0f,   // smoothing for accel ω_peak
                                 float c_tau               = 1.0f)    // τ = c_tau / ω_peak
        : c_tau_(c_tau)
    {
        tau_mom_         = std::max(1e-3f, tau_mom_sec);
        tau_peak_smooth_ = std::max(1e-3f, tau_peak_smooth_sec);
        reset();
    }

    void reset() noexcept {
        M2_ = 0.0f;
        weight_mom_ = 0.0f;
        has_moments_ = false;
        omega_peak_smooth_ = 0.0f;
        has_peak_ = false;
    }

    // === Update with one sample ===
    void update(float dt_s, float accel_z, float omega_inst) noexcept {
        if (!(dt_s > 0.0f)) return;

        // --- Clamp and validate ω_inst ---
        if (!(omega_inst > 0.0f))
            omega_inst = OMEGA_MIN;
        omega_inst = std::clamp(omega_inst, OMEGA_MIN, OMEGA_MAX);

        // --- Accel variance (σa², bias-corrected) ---
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

        // --- Peak frequency tracker (smoothed accel ω) ---
        const float a_pk = alpha(dt_s, tau_peak_smooth_);
        if (!has_peak_) {
            omega_peak_smooth_ = omega_inst;
            has_peak_ = true;
        } else {
            // Robust smoothing: slower update when ω_inst drops abnormally
            const float rel = std::clamp(
                omega_inst / std::max(omega_peak_smooth_, OMEGA_MIN), 0.25f, 4.0f);
            const float adj_alpha = a_pk * std::pow(rel, 0.3f);
            omega_peak_smooth_ =
                (1.0f - adj_alpha) * omega_peak_smooth_ + adj_alpha * omega_inst;
        }
    }

    // === Outputs ===
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
        return (std::isfinite(wpk) && wpk > EPS) ? (c_tau_ / wpk) : 0.0f;
    }

    // Optional helpers for readability / integration
    [[nodiscard]] float getPeriodPeak() const noexcept {
        const float w = getOmegaPeak();
        return (w > EPS) ? (2.0f * static_cast<float>(M_PI) / w) : 0.0f;
    }

    [[nodiscard]] float getDisplacementOmegaPeak() const noexcept {
        // Empirical mapping accel-domain → displacement-domain frequency
        return 0.6f * getOmegaPeak();
    }

private:
    // --- Accel variance ---
    float M2_ = 0.0f;
    float weight_mom_ = 0.0f;
    bool  has_moments_ = false;

    // --- Peak frequency (acceleration domain) ---
    float omega_peak_smooth_ = 0.0f;
    bool  has_peak_          = false;

    // --- Time constants ---
    float tau_mom_ = 120.0f;
    float tau_peak_smooth_ = 12.0f;
    float c_tau_ = 1.0f;

    // --- Numeric limits ---
    static constexpr float OMEGA_MIN = 0.2f;   // rad/s (accel freq lower bound)
    static constexpr float OMEGA_MAX = 8.0f;   // rad/s (accel freq upper bound)
    static constexpr float EPS       = 1e-12f;

    // --- Utility ---
    static constexpr float alpha(float dt, float tau) noexcept {
        return 1.0f - std::exp(-dt / std::max(tau, 1e-6f));
    }
};
