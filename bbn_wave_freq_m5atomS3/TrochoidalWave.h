#pragma once

#include <cmath>
#include <stdexcept>

const float g_std = 9.80665; // standard gravity acceleration m/s2

/**
 * TrochoidalWave – Gerstner (trochoidal) deep‐water wave model.
 *
 * Particle trajectories are circular at the surface and decay exponentially with depth.
 * Provides displacement, velocity, acceleration (both horizontal & vertical),
 * plus bulk wave properties and energy density.
 */
template<typename Real = float>
class TrochoidalWave {
public:
    /**
     * @param amplitude   Wave amplitude [m]
     * @param period      Wave period T [s]
     * @param depth       Reference depth z₀ [m] (default = 0 → surface)
     * @param phase       Phase offset φ [rad]
     * @param gravity     Gravity acceleration g [m/s²] (default = 9.80665)
     */
    TrochoidalWave(Real amplitude,
                   Real period,
                   Real depth   = Real(0),
                   Real phase   = Real(0),
                   Real gravity = Real(g_std))
      : A(amplitude)
      , T(period)
      , h(depth)
      , φ(phase)
      , g(gravity)
      , ω(static_cast<Real>(2 * M_PI) / T)
      , k(ω * ω / g)
      , λ(static_cast<Real>(2 * M_PI) / k)
      , c(ω / k)
    {
        if (A < Real(0))       throw std::domain_error("Amplitude must be ≥ 0");
        if (T <= Real(0))      throw std::domain_error("Period must be > 0");
        if (g <= Real(0))      throw std::domain_error("Gravity must be > 0");
    }

    // ── Static utility methods ─────────────────────────────────────────
    
    /// Compute wavelength (λ) from angular frequency (ω)
    static Real wavelengthFromAngularFrequency(Real omega, Real gravity = Real(g_std)) {
        return (2 * M_PI * gravity) / (omega * omega);
    }

    /// Compute wavenumber (k) from wavelength (λ)
    static Real wavenumberFromWavelength(Real wavelength) {
        return (2 * M_PI) / wavelength;
    }

    /// Compute wave speed (c) from wavenumber (k)
    static Real waveSpeedFromWavenumber(Real k, Real gravity = Real(g_std)) {
        return std::sqrt(gravity / k);
    }

    /// Compute wavelength (λ) from wave period (T)
    static Real wavelengthFromPeriod(Real period, Real gravity = Real(g_std)) {
        return gravity * period * period / (2 * M_PI);
    }

    /// Compute wave period (T) from vertical displacement and vertical acceleration
    static Real periodFromDisplacementAcceleration(Real displacement, Real accel) {
        return 2.0 * M_PI * std::sqrt(std::abs(displacement / accel));
    }

    /// Compute wave frequency (f) from vertical displacement and vertical acceleration
    static Real frequencyFromDisplacementAcceleration(Real displacement, Real accel) {
        return std::sqrt(std::abs(accel / displacement)) / (2.0 * M_PI);
    }

    // ── Surface kinematics ─────────────────────────────────────────────

    /// η(t) = −A·cos(ωt + φ)
    Real surfaceElevation(Real t) const {
        return -A * std::cos(ω * t + φ);
    }

    /// ∂η/∂t =  A·ω·sin(ωt + φ)
    Real surfaceVerticalSpeed(Real t) const {
        return  A * ω * std::sin(ω * t + φ);
    }

    /// ∂²η/∂t² = −A·ω²·cos(ωt + φ)
    Real surfaceVerticalAcceleration(Real t) const {
        return -A * ω * ω * std::cos(ω * t + φ);
    }

    // ── Particle kinematics (at reference position x₀, z₀) ─────────────────

    /// x(t) = x₀ − A·e^{k z₀}·sin(θ)
    Real horizontalPosition(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return x0 - A * R * std::sin(θ);
    }

    /// z(t) = z₀ − A·e^{k z₀}·cos(θ)
    Real verticalPosition(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return z0 - A * R * std::cos(θ);
    }

    /// u = dx/dt =  A·ω·e^{k z₀}·cos(θ)
    Real horizontalVelocity(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return  A * ω * R * std::cos(θ);
    }

    /// w = dz/dt =  A·ω·e^{k z₀}·sin(θ)
    Real verticalVelocity(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return  A * ω * R * std::sin(θ);
    }

    /// du/dt = −A·ω²·e^{k z₀}·sin(θ)
    Real horizontalAcceleration(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return -A * ω * ω * R * std::sin(θ);
    }

    /// dw/dt = −A·ω²·e^{k z₀}·cos(θ)
    Real verticalAcceleration(Real x0, Real z0, Real t) const {
        Real θ = k * x0 - ω * t + φ;
        Real R = std::exp(k * z0);
        return -A * ω * ω * R * std::cos(θ);
    }

    // ── Bulk wave properties ────────────────────────────────────────────

    Real amplitude() const        { return A; }
    Real period() const           { return T; }
    Real phase() const            { return φ; }
    Real gravity() const          { return g; }
    Real angularFrequency() const { return ω; }
    Real frequency() const        { return ω / Real(2 * M_PI); }
    Real wavenumber() const       { return k; }
    Real wavelength() const       { return λ; }
    Real waveSpeed() const        { return c; }

    /**
     * Mean wave energy density per unit horizontal area [J/m²]:
     *    E = ½ ρ g A²
     */
    Real energyDensity(Real rho = Real(1025.0)) const {
        return Real(0.5) * rho * g * A * A;
    }

private:
    Real A;   // amplitude [m]
    Real T;   // period [s]
    Real φ;   // phase [rad]
    Real g;   // gravity [m/s²]

    Real ω;   // angular frequency ω = 2π / T
    Real k;   // wavenumber k = ω² / g
    Real λ;   // wavelength λ = 2π / k
    Real c;   // phase speed    c = ω / k
};
