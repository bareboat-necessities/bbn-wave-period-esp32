#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy

   AI-assisted translation of https://github.com/TormodLandet/raschii/blob/master/raschii/fenton.py

*/

#include <ArduinoEigenDense.h>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <limits>
#include <algorithm>

#ifdef FENTON_TEST
#include <iostream>
#include <fstream>
#endif

template <typename T>
constexpr const T& clamp_value(const T& val, const T& low, const T& high) {
    return (val < low) ? low : (val > high) ? high : val;
}

template <typename T>
T sinh_by_cosh(T a, T b) {
    if (a == 0) return 0;
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }
    return std::sinh(a) / std::cosh(b);
}

template <typename T>
T cosh_by_cosh(T a, T b) {
    if (a == 0) return 1.0 / std::cosh(b);
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }
    return std::cosh(a) / std::cosh(b);
}

template <int N>
class FentonFFT {
public:
    using Real = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;

    // Inverse DCT-I (irfft-style): reconstruct cosine coefficients E from eta
    static Vector compute_inverse_cosine_transform(const Vector& eta) {
        Vector E;
        for (int j = 0; j <= N; ++j) {
            Real sum = 0;
            for (int m = 0; m <= N; ++m) {
                Real weight = (m == 0 || m == N) ? 0.5f : 1.0f;
                sum += weight * eta(m) * std::cos(j * m * M_PI / N);
            }
            E(j) = 2.0f * sum / N;
        }
        return E;
    }

    // Forward DCT-I: reconstruct eta values at collocation points from cosine coeffs
    static Vector compute_forward_cosine_transform(const Vector& E) {
        Vector eta;
        for (int m = 0; m <= N; ++m) {
            Real sum = 0;
            for (int j = 0; j <= N; ++j) {
                Real weight = (j == 0 || j == N) ? 0.5f : 1.0f;
                sum += weight * E(j) * std::cos(j * m * M_PI / N);
            }
            eta(m) = sum;
        }
        return eta;
    }
};


/**
 * FentonWave - Implements John Fenton's nonlinear wave theory for surface water waves
 * 
 * PURPOSE:
 * Computes accurate nonlinear periodic wave solutions (Stokes waves) in finite water depth
 * using Fourier approximation methods. Provides wave properties and kinematics.
 * 
 * KEY FEATURES:
 * - Higher-order nonlinear wave solutions (beyond linear/Airy wave theory)
 * - Arbitrary depth (shallow to deep water)
 * - Spectral representation with N Fourier modes
 * - Includes wave-current interaction via Q and R parameters
 * 
 * MATHEMATICAL MODEL:
 * 
 * 1. Stream Function Representation:
 *    ψ(x,z) = B₀*z + Σ Bⱼ*sinh(jk(z+d))/cosh(jkd) * cos(jkx)
 *    where Bⱼ are Fourier coefficients, k is wavenumber, d is depth
 * 
 * 2. Surface Elevation:
 *    η(x) = Σ Eⱼ*cos(jkx)  (via DCT-I transform of collocation points)
 * 
 * 3. Wave Parameters:
 *    - c = B₀ = wave phase velocity
 *    - k = 2π/L = wavenumber
 *    - Q = volume flux
 *    - R = Bernoulli constant
 * 
 * 4. Kinematic Fields:
 *    - Horizontal velocity: u = ∂ψ/∂z
 *    - Vertical velocity: w = -∂ψ/∂x
 *    - Surface elevation: η(x,t)
 *    - Derivatives (slopes, curvature, time derivatives)
 * 
 * NUMERICAL METHODS:
 * - Discrete Cosine Transform (DCT-I) for η <-> E conversion
 * - Newton-Raphson iteration to solve nonlinear equations
 * - Automatic step control in wave height progression
 * - Special handling of hyperbolic function numerical stability
 * 
 * USAGE:
 * 1. Initialize with wave parameters (height, depth, length)
 * 2. Access wave properties (c, k, T, etc.)
 * 3. Query kinematics at any (x,z,t) point
 * 
 * Reference: Fenton (1988) "The Numerical Solution of Steady Water Wave Problems"
 */
template <int N = 4>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;

public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega, Q, R;
    VectorF eta, x, E, B;

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        compute();
    }

/**
 * Calculates the stream function ψ(x,z,t) at a given point and time
 * 
 * Represents the mathematical function whose contours are flow streamlines:
 * ψ(x,z,t) = B₀*(z + d) + Σ[Bⱼ*sinh(jk(z+d))/cosh(jkd) * cos(jk(x-ct))] for j=1..N
 * 
 * Where:
 * - B₀ is the mean flow (wave phase speed c)
 * - Bⱼ are the Fourier coefficients of the stream function
 * - k is the wavenumber (2π/wavelength)
 * - d is the water depth
 * - c is the wave phase speed
 * 
 * Physical significance:
 * - Difference in ψ between two points equals the volume flow rate between them
 * - Contours of constant ψ represent fluid particle paths
 * - Used to derive velocity components (u = ∂ψ/∂z, w = -∂ψ/∂x)
 * 
 * @param x_val Horizontal position (m)
 * @param z_val Vertical position (m, 0=surface, negative below)
 * @param t Time (s)
 * @return Stream function value (m²/s)
 */
    Real stream_function(Real x_val, Real z_val, Real t = 0) const {
        Real psi = B(0) * (z_val + depth);  // Mean flow component
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * std::sinh(kj * (z_val + depth)) / denom;
            psi += term * std::cos(kj * (x_val - c * t));
        }
        return psi;
    }

/**
 * Calculates the horizontal velocity component u(x,z,t) at a given point and time
 * 
 * Represents the horizontal water particle velocity:
 * u(x,z,t) = ∂ψ/∂z = B₀ + Σ[Bⱼ*jk*cosh(jk(z+d))/cosh(jkd) * cos(jk(x-ct))] for j=1..N
 * 
 * Where:
 * - B₀ is the mean flow (wave phase speed c)
 * - The sum represents oscillatory wave-induced velocities
 * - Hyperbolic cosine term produces depth attenuation
 * 
 * Physical characteristics:
 * - Maximum at wave crest/trough, minimum at nodes
 * - Decays with depth (faster for higher frequencies)
 * - Phase varies with position in wave
 * 
 * @param x_val Horizontal position (m)
 * @param z_val Vertical position (m, 0=surface, negative below)
 * @param t Time (s)
 * @return Horizontal velocity (m/s, positive in wave direction)
 */
    Real horizontal_velocity(Real x_val, Real z_val, Real t = 0) const {
        Real u = B(0);  // Mean flow
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * kj / denom;
            u += term * std::cosh(kj * (z_val + depth)) * std::cos(kj * (x_val - c * t));
        }
        return u;
    }

/**
 * Calculates the vertical velocity component w(x,z,t) at a given point and time
 * 
 * Represents the vertical water particle velocity:
 * w(x,z,t) = -∂ψ/∂x = Σ[Bⱼ*jk*sinh(jk(z+d))/cosh(jkd) * sin(jk(x-ct))] for j=1..N
 * 
 * Where:
 * - The sum represents oscillatory wave-induced velocities
 * - Hyperbolic sine term produces depth attenuation
 * - 90° phase shift from horizontal velocity
 * 
 * Physical characteristics:
 * - Maximum at wave nodes, zero at crests/troughs
 * - Decays with depth (faster for higher frequencies)
 * - Leads horizontal velocity by 90° in progressive waves
 * 
 * @param x_val Horizontal position (m)
 * @param z_val Vertical position (m, 0=surface, negative below)
 * @param t Time (s)
 * @return Vertical velocity (m/s, positive upward)
 */
    Real vertical_velocity(Real x_val, Real z_val, Real t = 0) const {
        Real w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * kj / denom;
            w += term * std::sinh(kj * (z_val + depth)) * std::sin(kj * (x_val - c * t));
        }
        return w;
    }

/**
 * Calculates the dynamic pressure p(x,z,t) at a given point and time
 * 
 * Represents the fluid pressure accounting for:
 * - Hydrostatic pressure (ρg(z-η))
 * - Dynamic pressure from fluid motion (½ρ(u²+w²))
 * - Wave-induced pressure variations
 * 
 * Derived from Bernoulli's equation in the wave frame:
 * p/ρ = R - ½(u²+w²) - g(z-η) + c*u
 * 
 * Where:
 * - R is the Bernoulli constant
 * - c*u accounts for transformation to moving frame
 * - η is the surface elevation
 * 
 * Physical significance:
 * - Pressure lags behind surface elevation
 * - Important for wave loading on structures
 * - Used in buoyancy calculations
 * 
 * @param x_val Horizontal position (m)
 * @param z_val Vertical position (m, 0=surface, negative below)
 * @param t Time (s)
 * @param rho Water density (kg/m³, default 1025 for seawater)
 * @return Pressure (Pa)
 */
    Real pressure(Real x_val, Real z_val, Real t = 0, Real rho = 1025.0f) const {
        Real u = horizontal_velocity(x_val, z_val, t);
        Real w = vertical_velocity(x_val, z_val, t);
        Real eta = surface_elevation(x_val, t);
        
        // Bernoulli equation: p/ρ + ½(u²+w²) + g(z-η) + ∂φ/∂t = R
        // For steady flow in wave frame: ∂φ/∂t = -c*u
        return rho * (R - 0.5*(u*u + w*w) - rho*g*(z_val - eta) + rho*c*u);
    }

/**
 * Calculates the surface elevation (wave height) at position x_val and time t
 * 
 * Represents the actual wave shape at any point in space and time:
 * η(x,t) = E₀ + Σ[Eⱼ * cos(j*k*(x - c*t))] for j=1..N
 * Where:
 * - E₀ is the mean water level (0th coefficient)
 * - Eⱼ are the Fourier coefficients (wave amplitude components)
 * - j*k is the wavenumber for each harmonic
 * - c is the wave phase speed
 */
    Real surface_elevation(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            sum += E(j) * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

/**
 * Calculates the spatial derivative of surface elevation (wave slope)
 * 
 * Represents how steep the wave is at a given point:
 * dη/dx = -k * Σ[j*Eⱼ * sin(j*k*(x - c*t))] for j=1..N
 * 
 * Used for:
 * - Determining breaking wave conditions
 * - Calculating kinematic surface quantities
 */
    Real surface_slope(Real x_val, Real t = 0) const {
        Real d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= E(j) * j * k * std::sin(j * k * (x_val - c * t));
        }
        return d_eta;
    }

/**
 * Calculates the temporal derivative of surface elevation (vertical velocity at surface)
 * 
 * Represents how fast the water surface is moving up/down at a fixed point:
 * dη/dt = -c * dη/dx (using chain rule from slope calculation)
 * 
 * Used for:
 * - Wave energy calculations
 * - Surface boundary conditions
 */
    Real surface_time_derivative(Real x_val, Real t = 0) const {
        return -c * surface_slope(x_val, t);
    }

/**
 * Calculates the second temporal derivative of surface elevation (vertical acceleration at surface)
 * 
 * Represents the acceleration of the water surface:
 * d²η/dt² = -Σ[(j*ω)² * Eⱼ * cos(j*k*(x - c*t))] for j=1..N
 * Where ω = c*k is the angular frequency
 * 
 * Used for:
 * - Dynamic pressure calculations
 * - Wave impact studies
 */
    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real omega_j = j * omega;
            sum -= E(j) * omega_j * omega_j * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

/**
 * Calculates the mixed space-time derivative of surface elevation
 * 
 * Represents the rate of change of slope over time:
 * d²η/dxdt = Σ[j²*k*ω * Eⱼ * sin(j*k*(x - c*t))] for j=1..N
 * 
 * Used for:
 * - Nonlinear wave studies
 * - Wave-current interaction models
 */
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real term = j * k * j * omega;
            sum += E(j) * term * std::sin(j * k * (x_val - c * t));
        }
        return sum;
    }

/**
 * Calculates the second spatial derivative of surface elevation (curvature)
 * 
 * Represents how sharply curved the wave surface is:
 * d²η/dx² = -k² * Σ[j²*Eⱼ * cos(j*k*(x - c*t))] for j=1..N
 * 
 * Used for:
 * - Surface tension effects
 * - Wave instability analysis
 */
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real coeff = -j * k * j * k;
            sum += E(j) * coeff * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

    Real get_c() const { return c; }
    Real get_k() const { return k; }
    Real get_T() const { return T; }
    Real get_omega() const { return omega; }
    Real get_length() const { return length; }
    Real get_height() const { return height; }
    const VectorF& get_eta() const { return eta; }

private:
    void compute() {
        if (depth < 0) depth = 25.0f * length;
        Real H = height / depth;
        Real lam = length / depth;
        k = 2 * M_PI / lam;
        Real D = 1.0f;
        Real kc = k;
        Real c0 = std::sqrt(std::tanh(kc) / kc);

        VectorF x_nd;
        for (int m = 0; m <= N; ++m)
            x_nd(m) = lam * m / (2.0f * N);

        B.setZero();
        B(0) = c0;
        B(1) = -H / (4.0f * c0 * k);

        VectorF eta_nd;
        for (int m = 0; m <= N; ++m)
            eta_nd(m) = 1.0f + H / 2.0f * std::cos(k * x_nd(m));

        Q = c0;
        R = 1.0f + 0.5f * c0 * c0;

        for (Real Hi : wave_height_steps(H, D, lam)) {
            optimize(B, Q, R, eta_nd, Hi, k, D);
        }

        Real sqrt_gd = std::sqrt(g * depth);
        B(0) *= sqrt_gd;
        for (int j = 1; j <= N; ++j)
            B(j) *= std::sqrt(g * std::pow(depth, 3));
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;

        for (int i = 0; i <= N; ++i) {
            x(i) = x_nd(i) * depth;
            eta(i) = (eta_nd(i) - 1.0f) * depth;
        }

        k = k / depth;
        c = B(0);
        T = length / c;
        omega = c * k;

        compute_elevation_coefficients();
    }

    void compute_elevation_coefficients() {
        E = FentonFFT<N>::compute_inverse_cosine_transform(eta);
    }

    std::vector<Real> wave_height_steps(Real H, Real D, Real lam) {
        Real Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
        int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
        std::vector<Real> steps(num);
        for (int i = 0; i < num; ++i)
            steps[i] = H * (i + 1) / num;
        return steps;
    }

    void optimize(VectorF& B, Real& Q, Real& R,
                 VectorF& eta, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> coeffs;
        coeffs.template segment<N + 1>(0) = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;

        Real error = std::numeric_limits<Real>::max();
        for (int iter = 0; iter < 500 && error > 1e-8f; ++iter) {
            Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k, D);
            error = f.cwiseAbs().maxCoeff();
            
            Real eta_max = coeffs.template segment<N + 1>(N + 1).maxCoeff();
            Real eta_min = coeffs.template segment<N + 1>(N + 1).minCoeff();
            if (eta_max > 2.0f || eta_min < 0.1f || !std::isfinite(error)) {
                throw std::runtime_error("Optimization failed");
            }

            if (error < 1e-8f) break;

            Eigen::Matrix<Real, NU, NU> J = compute_jacobian(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;
        }

        B = coeffs.template segment<N + 1>(0);
        eta = coeffs.template segment<N + 1>(N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
    }

    Eigen::Matrix<Real, StateDim, 1>
    compute_residual(const Eigen::Matrix<Real, StateDim, 1>& coeffs, Real H, Real k, Real D) {
        Eigen::Matrix<Real, StateDim, 1> f;
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real Q = coeffs(2 * N + 2);
        Real R = coeffs(2 * N + 3);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real x_m = M_PI * m / N;
            Real eta_m = eta(m);
            
            Real um = -B0;
            Real vm = 0;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                Real S1 = sinh_by_cosh(kj * eta_m, kj * D);
                Real C1 = cosh_by_cosh(kj * eta_m, kj * D);
                Real S2 = std::sin(j * x_m);
                Real C2 = std::cos(j * x_m);
                um += kj * B(j) * C1 * C2;
                vm += kj * B(j) * S1 * S2;
            }

            f(m) = -B0 * eta_m;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                Real S1 = sinh_by_cosh(kj * eta_m, kj * D);
                Real C2 = std::cos(j * x_m);
                f(m) += B(j) * S1 * C2;
            }
            f(m) += Q;

            f(N + 1 + m) = 0.5f * (um * um + vm * vm) + eta_m - R;
        }

        f(2 * N + 2) = (eta.sum() - 0.5f * (eta(0) + eta(N))) / N - 1.0f;
        f(2 * N + 3) = eta.maxCoeff() - eta.minCoeff() - H;

        return f;
    }

    Eigen::Matrix<Real, StateDim, StateDim>
    compute_jacobian(const Eigen::Matrix<Real, StateDim, 1>& coeffs, Real H, Real k, Real D) {
        Eigen::Matrix<Real, StateDim, StateDim> J = Eigen::Matrix<Real, StateDim, StateDim>::Zero();
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real eta_m = eta(m);
            Real x_m = M_PI * m / N;
            Real um = -B0;
            Real vm = 0;
            
            Eigen::Matrix<Real, N, 1> SC, SS, CC, CS;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                SC(j-1) = sinh_by_cosh(kj * eta_m, kj * D) * std::cos(j * x_m);
                SS(j-1) = sinh_by_cosh(kj * eta_m, kj * D) * std::sin(j * x_m);
                CC(j-1) = cosh_by_cosh(kj * eta_m, kj * D) * std::cos(j * x_m);
                CS(j-1) = cosh_by_cosh(kj * eta_m, kj * D) * std::sin(j * x_m);
                
                um += kj * B(j) * CC(j-1);
                vm += kj * B(j) * SS(j-1);
            }

            J(m, 0) = -eta_m;
            for (int j = 1; j <= N; ++j) {
                J(m, j) = SC(j-1);
            }
            J(m, N + 1 + m) = -B0;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                J(m, N + 1 + m) += B(j) * kj * CC(j-1);
            }
            J(m, 2 * N + 2) = 1;

            J(N + 1 + m, 0) = -um;
            for (int j = 1; j <= N; ++j) {
                J(N + 1 + m, j) = k * j * (um * CC(j-1) + vm * SS(j-1));
            }
            J(N + 1 + m, N + 1 + m) = 1;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                J(N + 1 + m, N + 1 + m) += um * B(j) * kj * kj * SC(j-1);
                J(N + 1 + m, N + 1 + m) += vm * B(j) * kj * kj * CS(j-1);
            }
            J(N + 1 + m, 2 * N + 3) = -1;
        }

        for (int j = 0; j <= N; ++j) {
            J(2 * N + 2, N + 1 + j) = (j == 0 || j == N) ? 0.5f/N : 1.0f/N;
        }

        J(2 * N + 3, N + 1) = 1;
        J(2 * N + 3, 2 * N + 1) = -1;

        return J;
    }
};

/**
 * WaveSurfaceTracker - Simulates a floating object moving on a nonlinear wave surface
 * 
 * PHYSICS MODEL:
 * 
 * 1. Wave Surface Definition:
 *    z = η(x,t) = wave surface elevation at position x and time t
 * 
 * 2. Object Constraints:
 *    - Perfectly follows surface vertically (z = η(x,t))
 *    - Moves horizontally according to forces
 * 
 * 3. Key Derivatives:
 *    - η_x = ∂η/∂x = wave slope
 *    - η_t = ∂η/∂t = wave vertical velocity
 *    - η_xx = ∂²η/∂x² = wave curvature
 * 
 * 4. Object Kinematics:
 *    Vertical position: z(t) = η(x(t),t)
 *    Vertical velocity: dz/dt = ∂η/∂t + (∂η/∂x)(dx/dt) = η_t + η_x*vx
 *    Vertical acceleration: d²z/dt² = (dz/dt_{t+Δt} - dz/dt_t)/Δt (finite difference)
 * 
 * 5. Horizontal Dynamics:
 *    Forces:
 *      - Wave force: F_wave = -m*g*η_x (simplified buoyancy slope effect)
 *      - Drag force: F_drag = -c*vx (linear damping)
 *    Acceleration: d²x/dt² = (F_wave + F_drag)/m = -g*η_x - (c/m)*vx
 * 
 * 6. Numerical Integration:
 *    - RK4 method solves dx/dt = vx and dvx/dt = acceleration
 *    - Time step Δt must be small enough to capture wave dynamics
 * 
 * Note: This is a simplified model that assumes:
 * - Small object size compared to wavelength
 * - No added mass effects
 * - Perfect vertical surface following
 * - Linear drag model
 */
template<int N = 4>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;

    float t = 0.0f;
    float dt = 0.005f;

    // Object state
    float x = 0.0f;     // Horizontal position (m)
    float vx = 0.0f;    // Horizontal velocity (m/s)

    float mass = 1.0f;  // Mass of floating object (kg)

    // Wave and physics parameters
    float drag_coeff = 0.1f;  // Simple horizontal drag coefficient

    // Periodicity wrap helper
    float wrap_periodic(float val, float period) const {
        while (val < 0.0f) val += period;
        while (val >= period) val -= period;
        return val;
    }

    // Horizontal acceleration from wave slope and drag
    float compute_horizontal_acceleration(float x_pos, float vx_curr, float time) const {
        // Wave surface slope (∂η/∂x)
        float eta_x = wave.surface_slope(x_pos, time);

        // Simple driving force proportional to slope (restoring force)
        float force_wave = -9.81f * eta_x;  // gravity times slope (can be tuned)

        // Simple linear drag opposing velocity
        float force_drag = -drag_coeff * vx_curr;

        // Newton's second law
        return (force_wave + force_drag) / mass;
    }

    // RK4 integration for horizontal motion
    void rk4_step(float& x_curr, float& vx_curr, float t_curr, float dt_step) {
        auto accel = [this](float x_in, float vx_in, float t_in) {
            return compute_horizontal_acceleration(x_in, vx_in, t_in);
        };

        float k1_v = accel(x_curr, vx_curr, t_curr);
        float k1_x = vx_curr;

        float k2_v = accel(x_curr + 0.5f * dt_step * k1_x, vx_curr + 0.5f * dt_step * k1_v, t_curr + 0.5f * dt_step);
        float k2_x = vx_curr + 0.5f * dt_step * k1_v;

        float k3_v = accel(x_curr + 0.5f * dt_step * k2_x, vx_curr + 0.5f * dt_step * k2_v, t_curr + 0.5f * dt_step);
        float k3_x = vx_curr + 0.5f * dt_step * k2_v;

        float k4_v = accel(x_curr + dt_step * k3_x, vx_curr + dt_step * k3_v, t_curr + dt_step);
        float k4_x = vx_curr + dt_step * k3_v;

        x_curr += dt_step * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f;
        vx_curr += dt_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0f;

        // Periodicity wrap
        x_curr = wrap_periodic(x_curr, wave.get_length());
    }

public:
    WaveSurfaceTracker(float height, float depth, float length, float mass_kg, float drag_coeff_)
        : wave(height, depth, length), mass(mass_kg), drag_coeff(drag_coeff_) {}

    /**
     * @brief Track the floating object on the wave surface over time.
     * 
     * @param duration Total simulation time (s)
     * @param timestep Time step for integration (s)
     * @param callback Function called every step with:
     *        void callback(t, vertical_displacement, vertical_velocity, vertical_acceleration, x, vx);
     */
    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float, float)> callback)
    {
        dt = std::clamp(timestep, 1e-5f, 0.1f);

        t = 0.0f;
        x = 0.0f;
        vx = 0.0f;

        // Initialize vertical velocity and acceleration to zero
        float prev_z_dot = 0.0f;

        while (t <= duration) {
            // Compute current vertical displacement on wave surface
            float z = wave.surface_elevation(x, t);

            // Compute vertical velocity by chain rule:
            // dz/dt = ∂η/∂t + ∂η/∂x * dx/dt
            float eta_t = wave.surface_time_derivative(x, t);
            float eta_x = wave.surface_slope(x, t);
            float z_dot = eta_t + eta_x * vx;

            // Compute vertical acceleration by finite difference of vertical velocity
            float z_ddot = (z_dot - prev_z_dot) / dt;

            // Call user callback with current state
            callback(t, z, z_dot, z_ddot, x, vx);

            prev_z_dot = z_dot;

            // Integrate horizontal position and velocity with RK4
            rk4_step(x, vx, t, dt);

            t += dt;
        }
    }
};


#ifdef FENTON_TEST
template class FentonWave<4>;
template class WaveSurfaceTracker<4>;

void FentonWave_test_1() {
    const float height = 2.0f;
    const float depth = 10.0f;
    const float length = 50.0f;

    FentonWave<4> wave(height, depth, length);

    std::ofstream out("wave_data.csv");
    out << "x,elevation\n";
    for (float x = 0; x <= length; x += 0.05f) {
        float eta = wave.surface_elevation(x, 0);
        out << x << "," << eta << "\n";
    }
    std::cerr << "Expected wave length: " << length << "\n";
    std::cerr << "Computed wave length: " << 2 * M_PI / wave.get_k() << "\n";
}

void FentonWave_test_2() {
    // Wave parameters
    const float height = 2.0f;   // Wave height (m)
    const float depth = 10.0f;   // Water depth (m)
    const float length = 50.0f;  // Wavelength (m)
    const float mass = 100.0f;     // Mass (kg)
    const float drag = 0.1f;     // Linear drag coeff opposing velocity
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.005f;      // Time step (s)

    // Create a 4th-order Fenton wave and a surface tracker
    WaveSurfaceTracker<4> tracker(height, depth, length, mass, drag);

    // Output file
    std::ofstream out("wave_tracker_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position, float horizontal_speed) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track floating object (using callback)
    tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif

