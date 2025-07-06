#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy
   AI-assisted translation of https://github.com/bareboat-necessities/bbn-wave-period-esp32/blob/main/bbn_wave_freq_m5atomS3/FentonWave.h
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
    using Real   = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;
    using Matrix = Eigen::Matrix<Real, N + 1, N + 1>;

    static const Matrix& cosine_matrix() {
      static const Matrix M = []() {
        Matrix m;
        for (int j = 0; j <= N; ++j)
          for (int i = 0; i <= N; ++i)
            m(j, i) = std::cos(j * i * M_PI / N);
        return m;
      }(); return M;
    }

    static const Vector& weights() {
      static Vector w = []() {
        Vector v = Vector::Ones();
        v(0) = v(N) = 0.5f;
        return v;
      }(); return w;
    }

    static Vector compute_inverse_cosine_transform(const Vector& eta) {
      return (2.0f / N) * (cosine_matrix() * (eta.array() * weights().array()).matrix());
    }

    static Vector compute_forward_cosine_transform(const Vector& E) {
      return cosine_matrix().transpose() * (E.array() * weights().array()).matrix();
    }
};

struct WaveInitParams {
  float height;
  float depth;
  float length;
  float initial_x;
};

template <int N = 4>
class FentonWave {
  private:
    static constexpr int StateDim = 2 * (N + 1) + 2;
    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using VectorN = Eigen::Matrix<Real, N, 1>;
    using MatrixNxP = Eigen::Matrix<Real, N, N + 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;
    using RealArray = Eigen::Array<Real, N + 1, 1>;
    using PhaseArray = Eigen::Array<Real, N + 1, 1>;
    using VelocityTerms = Eigen::Array<Real, N, 1>;

  public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega, Q, R;
    VectorF eta, x, E, B;
    VectorN kj_cache, j_cache;

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
      : height(height), depth(depth), length(length), g(g), relax(relax) {
      for (int j = 1; j <= N; ++j) {
        kj_cache(j - 1) = j * (2 * M_PI / length);
        j_cache(j - 1) = j;
      }
      compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
      const PhaseArray phases = compute_phases(x_val, t);
      return (E.array() * phases.cos()).sum();
    }

    Real surface_slope(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return -(E.array() * j * k * phases.sin()).sum();
    }

    Real surface_time_derivative(Real x_val, Real t = 0) const {
      return -c * surface_slope(x_val, t);
    }

    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      const RealArray omega_j = j * omega;
      return -(E.array() * omega_j.square() * phases.cos()).sum();
    }

    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return (E.array() * j.square() * k * omega * phases.sin()).sum();
    }

    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
      const RealArray j = create_harmonic_indices();
      const PhaseArray phases = compute_phases(x_val, t);
      return -(E.array() * j.square() * k * k * phases.cos()).sum();
    }

    Real stream_function(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, false);
      return B(0) * (z_val + depth) + (terms * (j_cache.array() * phase).cos()).sum();
    }

    Real horizontal_velocity(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, true);
      return B(0) + (terms * (j_cache.array() * phase).cos()).sum();
    }

    Real vertical_velocity(Real x_val, Real z_val, Real t = 0) const {
      const Real phase = k * (x_val - c * t);
      const VelocityTerms terms = compute_velocity_terms(z_val, phase, false);
      return (terms * (j_cache.array() * phase).sin()).sum();
    }

    Real pressure(Real x_val, Real z_val, Real t = 0, Real rho = 1025.0f) const {
      const Real u = horizontal_velocity(x_val, z_val, t);
      const Real w = vertical_velocity(x_val, z_val, t);
      const Real eta = surface_elevation(x_val, t);

      // Bernoulli equation components
      const Real kinetic_energy = 0.5f * (u * u + w * w);
      const Real potential_energy = g * (z_val - eta);
      const Real flow_work = c * u;

      return rho * (R - kinetic_energy - potential_energy + flow_work);
    }

    Real mean_kinetic_energy_density(int samples = 100) const {
      Real dx = length / samples;
      Real KE_total = 0.0f;
      for (int i = 0; i <= samples; ++i) {
        Real x_val = i * dx;
        Real eta_val = surface_elevation(x_val);
        
        // Integrate kinetic energy from bottom to surface
        for (int zi = 0; zi <= 10; ++zi) {
           Real z_val = -depth + zi * (eta_val + depth) / 10.0f;
           Real u = horizontal_velocity(x_val, z_val);
           Real w = vertical_velocity(x_val, z_val);
           Real KE_density = 0.5f * (u * u + w * w);
           Real weight = (zi == 0 || zi == 10) ? 0.5f : 1.0f;
           KE_total += weight * KE_density * (eta_val + depth) / 10.0f;
        }
      }
      return dx * KE_total / length;  // Average over wavelength
    }

    Real mean_potential_energy_density(int samples = 100) const {
      Real dx = length / samples;
      Real PE_total = 0.0f;
      for (int i = 0; i <= samples; ++i) {
        Real x_val = i * dx;
        Real z = surface_elevation(x_val);           // η(x)
        Real PE = g * z;                             // Potential Energy
        PE_total += (i == 0 || i == samples) ? 0.5f * PE : PE;
      }
      return dx * PE_total;  // Trapezoidal rule
    }

    Real total_energy_density(int samples = 100) const {
      return mean_kinetic_energy_density(samples) + mean_potential_energy_density(samples);
    }

    Real energy_flux(int samples = 100) const {
      return c * total_energy_density(samples);
    }

    static float compute_wavelength(float omega, float depth, float g = 9.81f, float tol = 1e-6f, int max_iter = 50) {
      float k = omega * omega / g; // Initial guess (deep water)
      for (int i = 0; i < max_iter; ++i) {
        float f = g * k * std::tanh(k * depth) - omega * omega;
        float df = g * std::tanh(k * depth) + g * k * depth * (1.0f - std::pow(std::tanh(k * depth), 2));
        float k_next = k - f / df;
        if (std::abs(k_next - k) < tol) break;
        k = k_next;
      }
      return 2.0f * M_PI / k;
    }

    static WaveInitParams infer_fenton_parameters_from_amplitude(
      float amplitude, float depth, float omega, float phase_radians, float g = 9.81f) {
  
      if (amplitude <= 0 || depth <= 0 || omega <= 0)
        throw std::invalid_argument("Amplitude, depth, and omega must be positive");

      float height = 2.0f * amplitude;  // wave height = crest-to-trough
      float length = compute_wavelength(omega, depth, g);
      float initial_x = std::fmod(phase_radians / (2.0f * M_PI) * length, length);
      if (initial_x < 0.0f) initial_x += length; // wrap to [0, length)

      return { height, depth, length, initial_x };
    }

    Real get_c() const {
      return c;
    }
    Real get_k() const {
      return k;
    }
    Real get_T() const {
      return T;
    }
    Real get_omega() const {
      return omega;
    }
    Real get_length() const {
      return length;
    }
    Real get_height() const {
      return height;
    }
    const VectorF& get_eta() const {
      return eta;
    }

  private:
    void compute() {
      if (depth < 0) depth = 25.0f * length;
      Real H = height / depth;
      Real lam = length / depth;
      k = 2 * M_PI / lam;
      Real D = 1.0f;
      Real c0 = std::sqrt(std::tanh(k) / k);

      VectorF x_nd = VectorF::LinSpaced(N + 1, 0, lam / 2.0f);
      B.setZero();
      B(0) = c0;
      B(1) = -H / (4.0f * c0 * k);

      VectorF eta_nd = (VectorF::Ones().array() + (H / 2.0f) * (k * x_nd.array()).cos()).eval();
      Q = c0;
      R = 1.0f + 0.5f * c0 * c0;

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
      eta = (eta_nd.array() - 1.0f) * depth;
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
      Real Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
      int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
      Eigen::Array<Real, Eigen::Dynamic, 1> steps = Eigen::Array<Real, Eigen::Dynamic, 1>::LinSpaced(num, 1, num) * H / num;
      return std::vector<Real>(steps.data(), steps.data() + steps.size());
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
      Eigen::Matrix<Real, StateDim, 1> f = Eigen::Matrix<Real, StateDim, 1>::Zero();
      auto B = coeffs.template segment<N + 1>(0);
      auto eta = coeffs.template segment<N + 1>(N + 1);
      Real Q = coeffs(2 * N + 2);
      Real R = coeffs(2 * N + 3);
      Real B0 = B(0);

      Eigen::Array<Real, N, 1> j = Eigen::Array<Real, N, 1>::LinSpaced(N, 1, N);
      Eigen::Array<Real, N, 1> kj = j * k;

      for (int m = 0; m <= N; ++m) {
        Real x_m = M_PI * m / N;
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
        f(N + 1 + m) = 0.5f * (um * um + vm * vm) + eta_m - R;
      }

      // Mean elevation constraint
      f(2 * N + 2) = (eta.sum() - 0.5f * (eta(0) + eta(N))) / N - 1.0f;

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
        Real x_m = M_PI * m / N;

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
        J(m, 2 * N + 2) = 1.0f;

        // Next N+1 rows
        J(N + 1 + m, 0) = -um;
        for (int j = 0; j < N; ++j) {
          J(N + 1 + m, j + 1) = k * j_arr(j) * (um * CC(j) + vm * SS(j));
        }

        const Real d_eta_m = 1.0f + (um * (Bj * kj2 * SC).matrix()).sum() + (vm * (Bj * kj2 * CS).matrix()).sum();

        J(N + 1 + m, N + 1 + m) = d_eta_m;
        J(N + 1 + m, 2 * N + 3) = -1.0f;
      }

      // Mean elevation constraint (row 2N+2)
      for (int j = 0; j <= N; ++j)
        J(2 * N + 2, N + 1 + j) = (j == 0 || j == N) ? 0.5f / N : 1.0f / N;

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
      J(2 * N + 3, N + 1 + max_idx) = 1.0f;
      J(2 * N + 3, N + 1 + min_idx) = -1.0f;
      return J;
    }
};

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
      auto accel = [&](float x_in, float vx_in, float t_in) {
        return compute_horizontal_acceleration(x_in, vx_in, t_in);
      };

      float k1_v = accel(x_curr, vx_curr, t_curr);
      float k1_x = vx_curr;

      float k2_v = accel(x_curr + 0.5f * dt_step * k1_x, vx_curr + 0.5f * dt_step * k1_v, t_curr + 0.5f * dt_step);
      float k2_x = k1_x + 0.5f * dt_step * k1_v;

      float k3_v = accel(x_curr + 0.5f * dt_step * k2_x, vx_curr + 0.5f * dt_step * k2_v, t_curr + 0.5f * dt_step);
      float k3_x = k1_x + 0.5f * dt_step * k2_v;

      float k4_v = accel(x_curr + dt_step * k3_x, vx_curr + dt_step * k3_v, t_curr + dt_step);
      float k4_x = k1_x + dt_step * k3_v;

      x_curr += dt_step * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f;
      vx_curr += dt_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0f;

      // Periodicity wrap
      x_curr = wrap_periodic(x_curr, wave.get_length());
    }

  public:
    WaveSurfaceTracker(float height, float depth, float length, float x0, float mass_kg, float drag_coeff_)
      : wave(height, depth, length), mass(mass_kg), drag_coeff(drag_coeff_)
    {
      x = x0;
      vx = 0.0f;
    }

    void track_floating_object(
      float duration,
      float timestep,
      std::function<void(float, float, float, float, float, float, float)> callback)
    {
      dt = clamp_value(timestep, 1e-5f, 0.1f);

      t = 0.0f;

      // Initialize vertical velocity
      float prev_z_dot = wave.surface_time_derivative(x, 0) + wave.surface_slope(x, 0) * vx;

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
        if (t > dt) {
          callback(t, timestep, z, z_dot, z_ddot, x, vx);
        }

        prev_z_dot = z_dot;

        // Integrate horizontal position and velocity with RK4
        rk4_step(x, vx, t, dt);

        t += dt;
      }
    }

    FentonWave& get_wave() {
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
