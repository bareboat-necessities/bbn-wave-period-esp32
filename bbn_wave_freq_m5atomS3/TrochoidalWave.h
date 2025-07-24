#pragma once

#include <cmath>
#include <stdexcept>

/**
 * TrochoidalWave<Real> – First-order deep water wave model (linear theory).
 *
 * Models vertical particle motion in ocean waves using a trochoidal approximation.
 * Templated by `Real`, e.g., float, double, or autodiff scalar types.
 */
template <typename Real = float>
class TrochoidalWave {
public:
    static constexpr Real g = static_cast<Real>(9.80665); // gravity [m/s²]

    /// Construct wave with amplitude [m], frequency [Hz], and phase [rad]
    TrochoidalWave(Real amplitude, Real frequency, Real phase = Real(0))
        : A(amplitude), f(frequency), φ(phase)
    {
        if (A < Real(0))      throw std::domain_error("Amplitude must be non-negative");
        if (f <= Real(0))     throw std::domain_error("Frequency must be positive");
    }

    /// Vertical displacement η(t) = −A·cos(ωt + φ)
    Real displacement(Real t) const {
        return -A * std::cos(ω() * t + φ);
    }

    /// Vertical velocity dη/dt = A·ω·sin(ωt + φ)
    Real vertical_speed(Real t) const {
        return A * ω() * std::sin(ω() * t + φ);
    }

    /// Vertical acceleration d²η/dt² = −A·ω²·cos(ωt + φ)
    Real vertical_acceleration(Real t) const {
        return -A * ω_squared() * std::cos(ω() * t + φ);
    }

    /// Get amplitude [m]
    Real amplitude() const { return A; }

    /// Get frequency [Hz]
    Real frequency() const { return f; }

    /// Get phase [rad]
    Real phase() const { return φ; }

    /// Angular frequency ω = 2πf
    Real ω() const { return static_cast<Real>(2.0 * M_PI) * f; }

    /// Angular frequency squared
    Real ω_squared() const {
        Real w = ω();
        return w * w;
    }

    // ── Static wave relationships ──────────────────────────────

    /// λ = (2π·g) / ω²
    static Real wavelength_from_omega(Real omega) {
        if (omega <= Real(0)) throw std::domain_error("Omega must be positive");
        return static_cast<Real>(2.0 * M_PI) * g / (omega * omega);
    }

    /// k = 2π / λ
    static Real wavenumber(Real wavelength) {
        if (wavelength <= Real(0)) throw std::domain_error("Wavelength must be positive");
        return static_cast<Real>(2.0 * M_PI) / wavelength;
    }

    /// Wave speed c = sqrt(g / k)
    static Real wave_speed(Real k) {
        if (k <= Real(0)) throw std::domain_error("Wavenumber must be positive");
        return std::sqrt(g / k);
    }

    /// λ = g·T² / (2π)
    static Real wavelength_from_period(Real T) {
        if (T <= Real(0)) throw std::domain_error("Period must be positive");
        return g * T * T / static_cast<Real>(2.0 * M_PI);
    }

    /// T = 2π·sqrt(|disp / accel|)
    static Real period_from_displacement_accel(Real disp, Real accel) {
        if (accel == Real(0)) throw std::domain_error("Acceleration cannot be zero");
        return static_cast<Real>(2.0 * M_PI) * std::sqrt(std::abs(disp / accel));
    }

    /// f = (1 / 2π)·sqrt(|accel / disp|)
    static Real frequency_from_displacement_accel(Real disp, Real accel) {
        if (disp == Real(0)) throw std::domain_error("Displacement cannot be zero");
        return std::sqrt(std::abs(accel / disp)) / static_cast<Real>(2.0 * M_PI);
    }

private:
    Real A;   // amplitude [m]
    Real f;   // frequency [Hz]
    Real φ;   // phase [rad]
};

