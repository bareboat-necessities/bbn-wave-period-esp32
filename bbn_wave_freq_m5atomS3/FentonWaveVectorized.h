#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy
   AI-assisted translation of https://github.com/bareboat-necessities/bbn-wave-period-esp32/blob/main/bbn_wave_freq_m5atomS3/FentonWave.h
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <cmath>
#include <stdexcept>
#include <functional>
#include <limits>
#include <vector>
#include <algorithm>

#ifdef FENTON_TEST
#include <iostream>
#include <fstream>
#endif

template <typename T>
constexpr T clamp_value(const T& val, const T& low, const T& high) {
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
template <unsigned int N, typename Real = float>
class FentonFFT {
  public:
    using Vector = Eigen::Matrix<Real, N + 1, 1>;
    using Matrix = Eigen::Matrix<Real, N + 1, N + 1>;

    static const Matrix& cosine_matrix() {
      static const Matrix M = []() {
        Matrix m;
        for (int j = 0; j <= N; ++j)
          for (int i = 0; i <= N; ++i)
            m(j, i) = std::cos(j * i * Real(M_PI) / N);
        return m;
      }(); return M;
    }

    static const Vector& weights() {
      static Vector w = []() {
        Vector v = Vector::Ones();
        v(0) = v(N) = Real(0.5);
        return v;
      }(); return w;
    }

    static Vector compute_inverse_cosine_transform(const Vector& eta) {
      return (Real(2) / N) * (cosine_matrix() * (eta.array() * weights().array()).matrix());
    }

    static Vector compute_forward_cosine_transform(const Vector& E) {
      return cosine_matrix().transpose() * (E.array() * weights().array()).matrix();
    }
};

template <unsigned int N = 4, typename Real = float>
class FentonWave {
  private:
    static constexpr int StateDim = 2 * (N + 1) + 2;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using VectorN = Eigen::Matrix<Real, N, 1>;
    using MatrixNxP = Eigen::Matrix<Real, N, N + 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;
    using RealArray = Eigen::Array<Real, N + 1, 1>;
    using PhaseArray = Eigen::Array<Real, N + 1, 1>;
    using VelocityTerms = Eigen::Array<Real, N, 1>;

  public:
    struct WaveInitParams {
       Real height;
       Real depth;
       Real length;
       Real initial_x;
    };

    Real height, depth, length, g, relax;
    Real k, c, T, omega, Q, R;
    VectorF eta, x, E, B;
    VectorN kj_cache, j_cache;

    FentonWave(Real height, Real depth, Real length, Real g = Real(9.81), Real relax = Real(0.5))
      : height(height), depth(depth), length(length), g(g), relax(relax) {
      for (int j = 1; j <= N; ++j) {
        kj_cache(j - 1) = j * (2 * Real(M_PI) / length);
        j_cache(j - 1) = j;
      }
      compute();
    }

    // Returns the surface elevation η(x, t)
    Real surface_elevation(Real x_val, Real t = 0) const {
      const PhaseArray phases = compute_phases(x_val, t);
      return (E.array() * phases.cos()).sum();
    }

    // Returns the surface slope ∂η/∂x at (x, t)
    Real surface_slope(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return -(E.array() * j * k * phases.sin()).sum();
    }

    // Returns the surface vertical velocity ∂η/∂t at (x, t)
    Real surface_time_derivative(Real x_val, Real t = 0) const {
      return -c * surface_slope(x_val, t);
    }

    // Returns ∂²η/∂t² at (x, t) — vertical acceleration of surface
    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      const RealArray omega_j = j * omega;
      return -(E.array() * omega_j.square() * phases.cos()).sum();
    }

    // Returns ∂²η/∂x∂t at (x, t) — space-time mixed derivative
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return (E.array() * j.square() * k * omega * phases.sin()).sum();
    }

    // Returns ∂²η/∂x² at (x, t) — spatial curvature of surface
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return -(E.array() * j.square() * k * k * phases.cos()).sum();
    }

    // Returns stream function ψ(x, z, t)
    Real stream_function(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, false);
      return B(0) * (z_val + depth) + (terms * (j_cache.array() * phase).cos()).sum();
    }

    // Returns horizontal velocity u(x, z, t)
    Real horizontal_velocity(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, true);
      return B(0) + (terms * (j_cache.array() * phase).cos()).sum();
    }

    // Returns vertical velocity w(x, z, t)
    Real vertical_velocity(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, false);
      return (terms * (j_cache.array() * phase).sin()).sum();
    }

    // Returns dynamic pressure at (x, z, t) for fluid density rho (default: seawater)
    Real pressure(Real x_val, Real z_val, Real t = 0, Real rho = Real(1025)) const {
      const Real u = horizontal_velocity(x_val, z_val, t);
      const Real w = vertical_velocity(x_val, z_val, t);
      const Real eta = surface_elevation(x_val, t);

      const Real kinetic_energy = Real(0.5) * (u * u + w * w);
      const Real potential_energy = g * (z_val - eta);
      const Real flow_work = c * u;

      return rho * (R - kinetic_energy - potential_energy + flow_work);
    }

    // Computes average kinetic energy density per unit area over one wave period
    Real mean_kinetic_energy_density(int samples = 100) const {
      // f(x,z) = 0.5*(u²+w²)
      return integrate2D(
        [&](Real x, Real z) {
          Real u = horizontal_velocity(x, z);
          Real w = vertical_velocity  (x, z);
          return Real(0.5) * (u*u + w*w);
        },
        samples,      // x-samples
        10            // z-samples (you can also expose as parameter)
      );
    }

    // Computes average potential energy density per unit area
    Real mean_potential_energy_density(int samples = 100) const {
      Real dx = length / samples;
      Real PE_total = Real(0);
      for (int i = 0; i <= samples; ++i) {
        Real x_val = i * dx;
        Real z = surface_elevation(x_val);           // η(x)
        Real PE = g * z;                             // Potential Energy
        PE_total += (i == 0 || i == samples) ? Real(0.5) * PE : PE;
      }
      return dx * PE_total;
    }

    // Returns total (kinetic + potential) energy density per unit area
    Real total_energy_density(int samples = 100) const {
      return mean_kinetic_energy_density(samples) + mean_potential_energy_density(samples);
    }

    // Returns energy flux per unit width (W/m)
    Real energy_flux(int samples = 100) const {
      return c * total_energy_density(samples);
    }

    // Returns mean Eulerian current (ū) across depth and wavelength
    Real mean_eulerian_current(int samples = 100) const {
      // f(x,z) = u  (normalized by depth*length outside)
      // integrate2D already divides by length; multiply by depth here to undo its /length
      Real integral = integrate2D(
        [&](Real x, Real z) { return horizontal_velocity(x, z); },
        samples, 10
      );
      return integral / depth;
    }

    // Returns mean Stokes drift on the surface
    Real mean_stokes_drift(int samples = 100) const {
      Real dx = length / samples;
      Real total = Real(0);
      for (int i = 0; i <= samples; ++i) {
        Real x_val = i * dx;
        Real eta_val = surface_elevation(x_val);
        Real u = horizontal_velocity(x_val, eta_val);
        Real weight = (i == 0 || i == samples) ? Real(0.5) : Real(1);
        total += u * weight;
      }
      return dx * total / length;
    }

    // Returns total horizontal momentum (impulse) over one wavelength
    Real wave_impulse(int samples = 100) const {
      // f(x,z) = u
      return integrate2D(
        [&](Real x, Real z) { return horizontal_velocity(x, z); },
        samples, 10
      ) * length;  // undo internal /length
    }

    // Returns horizontal momentum flux ⟨u²⟩
    Real momentum_flux(int samples = 100) const {
      // f(x,z) = u²
      return integrate2D(
        [&](Real x, Real z) { Real u = horizontal_velocity(x, z); return u*u; },
        samples, 10
      );
    }

    // Returns radiation stress component Sxx = ρ⟨u²⟩
    Real radiation_stress_xx(int samples = 100) const {
      Real rho = Real(1025);
      Real flux = momentum_flux(samples);
      return rho * flux;
    }

    // Computes wavelength from ω, depth, gravity via dispersion relation
    static Real compute_wavelength(Real omega, Real depth, Real g = Real(9.81), Real tol = Real(1e-10), int max_iter = 50) {
      Real k = omega * omega / g; // Initial guess (deep water)
      for (int i = 0; i < max_iter; ++i) {
        Real f = g * k * std::tanh(k * depth) - omega * omega;
        Real df = g * std::tanh(k * depth) + g * k * depth * (Real(1) - std::pow(std::tanh(k * depth), 2));
        Real k_next = k - f / df;
        if (std::abs(k_next - k) < tol) break;
        k = k_next;
      }
      return Real(2) * Real(M_PI) / k;
    }

    // Infers Fenton wave parameters from amplitude, depth, frequency, and phase
    static WaveInitParams infer_fenton_parameters_from_amplitude(
      Real amplitude, Real depth, Real omega, Real phase_radians, Real g = Real(9.81)) {
  
      if (amplitude <= 0 || depth <= 0 || omega <= 0)
        throw std::invalid_argument("Amplitude, depth, and omega must be positive");

      Real height = Real(2) * amplitude;  // wave height = crest-to-trough
      Real length = compute_wavelength(omega, depth, g);
      Real initial_x = std::fmod(phase_radians / (Real(2) * Real(M_PI)) * length, length);
      if (initial_x < Real(0)) initial_x += length; // wrap to [0, length)

      return { height, depth, length, initial_x };
    }

    // Returns phase speed c
    Real get_c() const {
      return c;
    }

    // Returns wave number k
    Real get_k() const {
      return k;
    }

    // Returns wave period T
    Real get_T() const {
      return T;
    }

    // Returns angular frequency ω
    Real get_omega() const {
      return omega;
    }

    // Returns wavelength λ
    Real get_length() const {
      return length;
    }

    // Returns wave height H
    Real get_height() const {
      return height;
    }

    // Returns elevation profile η(x) (vector of sampled values)
    const VectorF& get_eta() const {
      return eta;
    }

  private:
    void compute() {
      if (depth < 0) depth = Real(25.0) * length;
      Real H = height / depth;
      Real lam = length / depth;
      k = 2 * Real(M_PI) / lam;
      Real D = Real(1);
      Real c0 = std::sqrt(std::tanh(k) / k);

      VectorF x_nd = VectorF::LinSpaced(N + 1, 0, lam / Real(2));
      B.setZero();
      B(0) = c0;
      B(1) = -H / (Real(4) * c0 * k);

      VectorF eta_nd = (VectorF::Ones().array() + (H / Real(2)) * (k * x_nd.array()).cos()).eval();
      Q = c0;
      R = Real(1) + Real(0.5) * c0 * c0;

      for (Real Hi : wave_height_steps(H, D, lam)) {
        optimize(B, Q, R, eta_nd, Hi, k, D);
      }

      const Real sqrt_gd = std::sqrt(g * depth);
      const Real gd = g * depth;
      const Real sqrt_gd3 = sqrt_gd * depth;
      B(0) *= sqrt_gd;
      B.tail(N) *= sqrt_gd3;
      Q *= sqrt_gd3;
      R *= gd;      

      x = x_nd * depth;
      eta = (eta_nd.array() - Real(1)) * depth;
      k /= depth;
      c = B(0);
      T = length / c;
      omega = c * k;

      compute_elevation_coefficients();
    }

    RealArray create_harmonic_indices() const {
      return RealArray::LinSpaced(N + 1, 0, N);
    }

    PhaseArray compute_phases(Real x_val, Real t) const {
      return create_harmonic_indices() * k * (x_val - c * t);
    }

    VelocityTerms compute_velocity_terms(Real z_val, Real phase, bool for_horizontal) const {
      const auto kj = kj_cache.array();
      const auto denom = (kj * depth).cosh();
      VelocityTerms terms = kj * B.tail(N).array() / denom;
      if (for_horizontal) {
        return terms * (kj * (z_val + depth)).cosh();
      } else {
        return terms * (kj * (z_val + depth)).sinh();
      }
    }

    void compute_elevation_coefficients() {
      E = FentonFFT<N>::compute_inverse_cosine_transform(eta);
    }

    std::vector<Real> wave_height_steps(Real H, Real D, Real lam) {
      const Real BREAKING_WAVE_CONST = Real(0.142);
      Real Hb = BREAKING_WAVE_CONST * std::tanh(2 * Real(M_PI) * D / lam) * lam;
      int num = (H > Real(0.75) * Hb) ? 10 : (H > Real(0.65) * Hb) ? 5 : 3;
      Eigen::Array<Real, Eigen::Dynamic, 1> steps = Eigen::Array<Real, Eigen::Dynamic, 1>::LinSpaced(num, 1, num) * H / num;
      return std::vector<Real>(steps.data(), steps.data() + steps.size());
    }

    void optimize(VectorF& B, Real& Q, Real& R, VectorF& eta, Real H, Real k, Real D) {
      constexpr int NU = 2 * (N + 1) + 2;
      Eigen::Matrix<Real, NU, 1> coeffs;
      coeffs.template segment<N + 1>(0) = B;
      coeffs.template segment<N + 1>(N + 1) = eta;
      coeffs(2 * N + 2) = Q;
      coeffs(2 * N + 3) = R;

      Real error = std::numeric_limits<Real>::max();
      for (int iter = 0; iter < 500 && error > Real(1e-8); ++iter) {
        Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k, D);
        error = f.cwiseAbs().maxCoeff();

        Real eta_max = coeffs.template segment<N + 1>(N + 1).maxCoeff();
        Real eta_min = coeffs.template segment<N + 1>(N + 1).minCoeff();
        if (eta_max > Real(2) || eta_min < Real(0.1) || !std::isfinite(error)) {
          throw std::runtime_error("Optimization failed");
        }
        if (error < Real(1e-8)) break;

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
      Eigen::Matrix<Real, StateDim, 1> f = Eigen::Matrix<Real, StateDim, 1>::Zero();
      auto B = coeffs.template segment<N + 1>(0);
      auto eta = coeffs.template segment<N + 1>(N + 1);
      Real Q = coeffs(2 * N + 2);
      Real R = coeffs(2 * N + 3);
      Real B0 = B(0);

      Eigen::Array<Real, N, 1> j = Eigen::Array<Real, N, 1>::LinSpaced(N, 1, N);
      Eigen::Array<Real, N, 1> kj = j * k;

      for (int m = 0; m <= N; ++m) {
        Real x_m = Real(M_PI) * m / N;
        Real eta_m = eta(m);

        Eigen::Array<Real, N, 1> kj_eta = kj * eta_m;
        Eigen::Array<Real, N, 1> kj_D   = kj * D;

        Eigen::Array<Real, N, 1> sin_jx = (j * x_m).sin();
        Eigen::Array<Real, N, 1> cos_jx = (j * x_m).cos();

        Eigen::Array<Real, N, 1> S1 = kj_eta.binaryExpr(kj_D, [](Real a, Real b) {
          return sinh_by_cosh(a, b);
        });
        Eigen::Array<Real, N, 1> C1 = kj_eta.binaryExpr(kj_D, [](Real a, Real b) {
          return cosh_by_cosh(a, b);
        });

        Eigen::Array<Real, N, 1> SC = S1 * cos_jx;
        Eigen::Array<Real, N, 1> CC = C1 * cos_jx;
        Eigen::Array<Real, N, 1> SS = S1 * sin_jx;

        Real um = -B0 + (kj * B.tail(N).array() * CC).sum();
        Real vm = (kj * B.tail(N).array() * SS).sum();

        // First N+1 rows: f[m]
        f(m) = -B0 * eta_m + (B.tail(N).array() * SC).sum() + Q;

        // Next N+1 rows: f[N+1+m]
        f(N + 1 + m) = Real(0.5) * (um * um + vm * vm) + eta_m - R;
      }

      // Mean elevation constraint
      f(2 * N + 2) = (eta.sum() - Real(0.5) * (eta(0) + eta(N))) / N - Real(1);

      // Wave height constraint
      f(2 * N + 3) = eta.maxCoeff() - eta.minCoeff() - H;

      return f;
    }

    Eigen::Matrix<Real, StateDim, StateDim>
    compute_jacobian(const Eigen::Matrix<Real, StateDim, 1>& coeffs, Real H, Real k, Real D) {
      Eigen::Matrix<Real, StateDim, StateDim> J = Eigen::Matrix<Real, StateDim, StateDim>::Zero();
      auto B   = coeffs.template segment<N + 1>(0);
      auto eta = coeffs.template segment<N + 1>(N + 1);
      Real B0  = B(0);

      const Eigen::Array<Real, N, 1> j_arr = j_cache;
      const Eigen::Array<Real, N, 1> kj = j_arr * k;
      const Eigen::Array<Real, N, 1> kj2 = kj.square();
      const Eigen::Array<Real, N, 1> Bj = B.tail(N).array();

      for (int m = 0; m <= N; ++m) {
        Real eta_m = eta(m);
        Real x_m = Real(M_PI) * m / N;
        Eigen::Array<Real, N, 1> S1, C1, S2, C2;
        Eigen::Array<Real, N, 1> SC, SS, CC, CS;

        for (int j = 0; j < N; ++j) {
          Real kj_eta = kj(j) * eta_m;
          Real kj_D   = kj(j) * D;
          S1(j) = sinh_by_cosh(kj_eta, kj_D);
          C1(j) = cosh_by_cosh(kj_eta, kj_D);
          Real jx = j_arr(j) * x_m;
          S2(j) = std::sin(jx);
          C2(j) = std::cos(jx);
        }

        SC = S1 * C2;
        SS = S1 * S2;
        CC = C1 * C2;
        CS = C1 * S2;

        Real um = -B0 + (kj * Bj * CC).sum();
        Real vm =       (kj * Bj * SS).sum();

        // First N+1 rows
        J(m, 0) = -eta_m;
        for (int j = 0; j < N; ++j) {
          J(m, j + 1) = SC(j);
        }
        J(m, N + 1 + m) = -B0 + (Bj * kj * CC).sum();
        J(m, 2 * N + 2) = Real(1);

        // Next N+1 rows
        J(N + 1 + m, 0) = -um;
        for (int j = 0; j < N; ++j) {
          J(N + 1 + m, j + 1) = k * j_arr(j) * (um * CC(j) + vm * SS(j));
        }

        const Real d_eta_m = Real(1) + (um * (Bj * kj2 * SC).matrix()).sum() + (vm * (Bj * kj2 * CS).matrix()).sum();

        J(N + 1 + m, N + 1 + m) = d_eta_m;
        J(N + 1 + m, 2 * N + 3) = Real(-1);
      }

      // Mean elevation constraint (row 2N+2)
      for (int j = 0; j <= N; ++j)
        J(2 * N + 2, N + 1 + j) = (j == 0 || j == N) ? Real(0.5) / N : Real(1) / N;

      // Wave height constraint (row 2N+3)
      int max_idx = 0, min_idx = 0;
      Real max_val = eta(0), min_val = eta(0);
      for (int j = 1; j <= N; ++j) {
        if (eta(j) > max_val) {
          max_val = eta(j);
          max_idx = j;
        }
        if (eta(j) < min_val) {
          min_val = eta(j);
          min_idx = j;
        }
      }
      J(2 * N + 3, N + 1 + max_idx) = Real(1);
      J(2 * N + 3, N + 1 + min_idx) = Real(-1);
      return J;
    }

    // Generic 2D trapezoidal integration over x∈[0,length], z∈[-depth,η(x)]
    template<typename Func>
    Real integrate2D(Func f, int x_samples = 100, int z_samples = 10) const {
      const Real dx = length / x_samples;
      const int Nx = x_samples + 1;
      const int Nz = z_samples + 1;

      // 1) Precompute x grid and surface elevations η(x)
      std::vector<Real> x_vals(Nx), eta_vals(Nx);
      for (int i = 0; i < Nx; ++i) {
        x_vals[i] = i * dx;
        eta_vals[i] = surface_elevation(x_vals[i]);   // only once per x
      }

      // 2) Precompute z‐weights (0.5 at ends, 1.0 interior)
      std::vector<Real> wz(Nz, Real(1));
      wz[0] = wz[Nz-1] = Real(0.5);

      Real total = Real(0);
      for (int i = 0; i < Nx; ++i) {
        Real x_i = x_vals[i];
        Real eta  = eta_vals[i];
        Real dz   = (eta + depth) / z_samples;

        // integrate in z at this x
        Real sum_z = Real(0);
        for (int zi = 0; zi < Nz; ++zi) {
          Real z = -depth + zi * dz;
          sum_z   += f(x_i, z) * wz[zi];
        }
        sum_z *= dz;

        // trapezoid weight in x
        Real wx = (i==0 || i==Nx-1) ? Real(0.5) : Real(1);
        total += sum_z * wx;
      }

      // normalize by dx/length
      return total * dx / length;
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
template<unsigned int N = 4, typename Real = float>
class WaveSurfaceTracker {
  private:
    FentonWave<N> wave;

    Real t = Real(0);
    Real dt = Real(0.005);

    // Object state
    Real x = Real(0);     // Horizontal position (m)
    Real vx = Real(0);    // Horizontal velocity (m/s)

    Real mass = Real(1);  // Mass of floating object (kg)

    // Wave and physics parameters
    Real drag_coeff = Real(0.1);  // Simple horizontal drag coefficient

    // Periodicity wrap helper
    Real wrap_periodic(Real val, Real period) const {
      val = std::fmod(val, period);
      if (val < Real(0)) val += period;
      return val;
    }

    // Horizontal acceleration from wave slope and drag
    Real compute_horizontal_acceleration(Real x_pos, Real vx_curr, Real time) const {
      // Wave surface slope (∂η/∂x)
      Real eta_x = wave.surface_slope(x_pos, time);

      // Simple driving force proportional to slope (restoring force)
      Real force_wave = -Real(9.81) * eta_x;  // gravity times slope (can be tuned)

      // Simple linear drag opposing velocity
      Real force_drag = -drag_coeff * vx_curr;

      // Newton's second law
      return (force_wave + force_drag) / mass;
    }

    // RK4 integration for horizontal motion
    void rk4_step(Real& x_curr, Real& vx_curr, Real t_curr, Real dt_step) {
      auto accel = [&](Real x_in, Real vx_in, Real t_in) {
        return compute_horizontal_acceleration(x_in, vx_in, t_in);
      };

      Real k1_v = accel(x_curr, vx_curr, t_curr);
      Real k1_x = vx_curr;

      Real k2_v = accel(x_curr + Real(0.5) * dt_step * k1_x, vx_curr + Real(0.5) * dt_step * k1_v, t_curr + Real(0.5) * dt_step);
      Real k2_x = k1_x + Real(0.5) * dt_step * k1_v;

      Real k3_v = accel(x_curr + Real(0.5) * dt_step * k2_x, vx_curr + Real(0.5) * dt_step * k2_v, t_curr + Real(0.5) * dt_step);
      Real k3_x = k1_x + Real(0.5) * dt_step * k2_v;

      Real k4_v = accel(x_curr + dt_step * k3_x, vx_curr + dt_step * k3_v, t_curr + dt_step);
      Real k4_x = k1_x + dt_step * k3_v;

      x_curr += dt_step * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / Real(6);
      vx_curr += dt_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / Real(6);

      x_curr = wrap_periodic(x_curr, wave.get_length());
    }

  public:
    WaveSurfaceTracker(Real height, Real depth, Real length, Real x0, Real mass_kg, Real drag_coeff_)
      : wave(height, depth, length), mass(mass_kg), drag_coeff(drag_coeff_)
    {
      x = x0;
      vx = Real(0);
    }

    void track_floating_object(
      Real duration,
      Real timestep,
      std::function<void(Real, Real, Real, Real, Real, Real, Real)> callback)
    {
      dt = clamp_value(timestep, Real(1e-5), Real(0.1));

      t = Real(0);

      // Initialize vertical velocity
      Real prev_z_dot = wave.surface_time_derivative(x, 0) + wave.surface_slope(x, 0) * vx;

      while (t <= duration) {
        // Compute current vertical displacement on wave surface
        Real z = wave.surface_elevation(x, t);

        // Compute vertical velocity by chain rule:
        // dz/dt = ∂η/∂t + ∂η/∂x * dx/dt
        Real eta_t = wave.surface_time_derivative(x, t);
        Real eta_x = wave.surface_slope(x, t);
        Real z_dot = eta_t + eta_x * vx;

        // Compute vertical acceleration by finite difference of vertical velocity
        Real z_ddot = (z_dot - prev_z_dot) / dt;

        // Call user callback with current state
        if (t > dt) {
          callback(t, timestep, z, z_dot, z_ddot, x, vx);
        }

        prev_z_dot = z_dot;

        // Integrate horizontal position and velocity with RK4
        rk4_step(x, vx, t, dt);

        t += dt;
      }
    }

    FentonWave<N> get_wave() {
      return wave;
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
  const float init_x = 10.0f;  // Initial x (m)
  const float mass = 5.0f;     // Mass (kg)
  const float drag = 0.1f;     // Linear drag coeff opposing velocity

  // Simulation parameters
  const float duration = 30.0f; // Simulation duration (s)
  const float dt = 0.005f;      // Time step (s)

  // Create a 4th-order Fenton wave and a surface tracker
  WaveSurfaceTracker<4> tracker(height, depth, length, init_x, mass, drag);

  // Output file
  std::ofstream out("wave_tracker_data.csv");
  out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

  // Define the kinematics callback (writes data to file)
  auto kinematics_callback = [&out](
      float time, float dt, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position, float horizontal_speed) {
    out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
  };

  // Track floating object (using callback)
  tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif
