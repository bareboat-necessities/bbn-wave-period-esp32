#pragma once

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

// ===================================================
// Minimal Real FFT implementation compatible with Eigen
// ===================================================
template <typename Scalar, int N>
class FentonFFT {
public:
    using Real = Scalar;
    using Complex = std::complex<Real>;
    using RealVec = Eigen::Matrix<Real, N, 1>;
    using ComplexVec = Eigen::Matrix<Complex, N, 1>;

    void forward(const RealVec& in, ComplexVec& out) {
        for (int k = 0; k < N; ++k) {
            Complex sum(0, 0);
            for (int n = 0; n < N; ++n) {
                Real angle = -2 * M_PI * k * n / N;
                sum += in(n) * Complex(std::cos(angle), std::sin(angle));
            }
            out(k) = sum;
        }
    }

    void inverse(const ComplexVec& in, RealVec& out) {
        for (int n = 0; n < N; ++n) {
            Complex sum(0, 0);
            for (int k = 0; k < N; ++k) {
                Real angle = 2 * M_PI * k * n / N;
                sum += in(k) * Complex(std::cos(angle), std::sin(angle));
            }
            out(n) = sum.real() / N;
        }
    }

    void inverse_cosine(const RealVec& coeffs, RealVec& signal) {
        const int M = N - 1;
        for (int i = 0; i < N; ++i) {
            Real sum = 0;
            for (int j = 0; j <= M; ++j) {
                Real phi = M_PI * j * i / M;
                Real w = (j == 0 || j == M) ? 0.5f : 1.0f;
                sum += coeffs(j) * std::cos(phi) * w;
            }
            signal(i) = 2.0f * sum / M;
        }
    }
};

// ===================================================
// FentonWave class
// ===================================================
template <int N>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;

public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega;
    VectorF eta, x, E, B;

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            sum += E(j) * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

    Real surface_slope(Real x, Real t = 0) const {
        Real d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= E(j) * j * k * std::sin(j * k * (x - c * t));
        }
        return d_eta;
    }

    Real surface_time_derivative(Real x, Real t = 0) const {
        return -c * surface_slope(x, t);
    }

    Real vertical_velocity(Real x, Real z, Real t = 0) const {
        Real w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real arg = kj * (x - c * t);
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * kj / denom;
            w += term * std::sin(arg) * std::sinh(kj * (z + depth));
        }
        return w;
    }

    // Getters
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

        Real Q = c0, R = 1.0f + 0.5f * c0 * c0;

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
            eta(i) = eta_nd(i) * depth;
        }

        k = k / depth;
        c = B(0);
        T = length / c;
        omega = c * k;

        // Use irfft-like logic to compute E coefficients
        FentonFFT<Real, N + 1> fft;
        fft.inverse_cosine(eta, E);
    }

    std::array<Real, 10> wave_height_steps(Real H, Real D, Real lam) {
        Real Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
        int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
        std::array<Real, 10> steps{};
        for (int i = 0; i < num; ++i)
            steps[i] = H * (i + 1) / num;
        return steps;
    }

    void optimize(VectorF& B, Real& Q, Real& R,
                  VectorF& eta, Real H, Real k, Real D)
    {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> coeffs;
        coeffs.template segment<N + 1>(0) = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;

        for (int iter = 0; iter < 100; ++iter) {
            Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, NU> J = compute_jacobian(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;

            if (f.cwiseAbs().maxCoeff() < 1e-8f) break;
        }

        B = coeffs.template segment<N + 1>(0);
        eta = coeffs.template segment<N + 1>(N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
    }

    Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>
    compute_residual(const Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>& coeffs, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> f;
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real Q = coeffs(2 * N + 2);
        Real R = coeffs(2 * N + 3);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real x_m = M_PI * m / N;
            Real eta_m = eta(m);
            Real um = -B0, vm = 0, S1, C1, S2, C2;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real cosh_ratio = std::cosh(aj) / std::cosh(bj);
                S1 = sinh_ratio;
                C1 = cosh_ratio;
                S2 = std::sin(j * x_m);
                C2 = std::cos(j * x_m);
                um += k * j * B(j) * C1 * C2;
                vm += k * j * B(j) * S1 * S2;
            }
            f(m) = -B0 * eta_m;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real C2 = std::cos(j * M_PI * m / N);
                f(m) += B(j) * sinh_ratio * C2;
            }
            f(m) += Q;
            f(N + 1 + m) = 0.5f * (um * um + vm * vm) + eta_m - R;
        }

        // mean(eta) = 1
        f(2 * N + 2) = eta.sum() / N - 1.0f;

        // eta(0) - eta(N) = H
        f(2 * N + 3) = eta(0) - eta(N) - H;

        return f;
    }

    Eigen::Matrix<Real, 2 * (N + 1) + 2, 2 * (N + 1) + 2>
    compute_jacobian(const Eigen::Matrix<Real, 2 * (N + 1) + 2, 1>& coeffs, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, NU> J = Eigen::Matrix<Real, NU, NU>::Zero();
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real eta_m = eta(m);
            Real x_m = M_PI * m / N;
            Real um = -B0, vm = 0;

            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real cosh_ratio = std::cosh(aj) / std::cosh(bj);
                Real s2 = std::sin(j * x_m);
                Real c2 = std::cos(j * x_m);
                um += k * j * B(j) * cosh_ratio * c2;
                vm += k * j * B(j) * sinh_ratio * s2;
            }

            J(m, 0) = -eta_m;
            for (int j = 1; j <= N; ++j) {
                Real aj = j * k * eta_m;
                Real bj = j * k * D;
                Real sinh_ratio = std::sinh(aj) / std::cosh(bj);
                Real c2 = std::cos(j * M_PI * m / N);
                J(m, j) = sinh_ratio * c2;
            }
            J(m, N + 1 + m) = -B0;
            J(m, 2 * N + 2) = 1;

            J(N + 1 + m, N + 1 + m) = 1.0f;
            J(N + 1 + m, 0) = -um;
        }

        for (int j = 0; j <= N; ++j)
            J(2 * N + 2, N + 1 + j) = 1.0f / N;

        J(2 * N + 3, N + 1) = 1;
        J(2 * N + 3, 2 * N + 1) = -1;

        return J;
    }
};


/**
 * @brief Class for tracking vertical kinematics of a floating object on a nonlinear wave surface.
 */
template<int N = 4>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;

    float t = 0.0f;
    float x = 0.0f;
    float dt = 0.005f;

    float prev_z = 0.0f;
    float prev_dzdt = 0.0f;

    static constexpr float slope_eps = 1e-6f;  // Prevent division by zero

    float mean_eta = 0.0f;

    // Robust periodic wrapping for horizontal position x
    float wrap_periodic(float val, float period) const {
        while (val < 0.0f) val += period;
        while (val >= period) val -= period;
        return val;
    }

    // Compute mean elevation offset by sampling wave surface at t=0
    void compute_mean_elevation(int samples = 100) {
        float sum = 0.0f;
        float L = wave.get_length();
        for (int i = 0; i < samples; ++i) {
            float xi = L * i / static_cast<float>(samples - 1);
            sum += wave.surface_elevation(xi, 0.0f);
        }
        mean_eta = sum / static_cast<float>(samples);
    }

    // Compute horizontal speed dx/dt using kinematic constraint
    float compute_horizontal_speed(float x_pos, float time) const {
        float eta      = wave.surface_elevation(x_pos, time) - mean_eta;
        float eta_dot  = wave.surface_time_derivative(x_pos, time);
        float eta_x    = wave.surface_slope(x_pos, time);
        // Pass physical elevation (add mean_eta back) for vertical velocity
        float w        = wave.vertical_velocity(x_pos, eta + mean_eta, time);

        // Clamp slope to avoid division by zero or extreme values
        if (std::abs(eta_x) < slope_eps)
            eta_x = (eta_x >= 0.0f) ? slope_eps : -slope_eps;

        return (w - eta_dot) / eta_x;
    }

    // 4th-order Runge-Kutta integration for horizontal position
    float rk4_integrate_x(float x_curr, float t_curr, float dt_step) const {
        float k1 = compute_horizontal_speed(x_curr, t_curr);
        float k2 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k1, t_curr + 0.5f * dt_step);
        float k3 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k2, t_curr + 0.5f * dt_step);
        float k4 = compute_horizontal_speed(x_curr + dt_step * k3, t_curr + dt_step);
        return x_curr + (dt_step / 6.0f) * (k1 + 2.0f*k2 + 2.0f*k3 + k4);
    }

public:
    WaveSurfaceTracker(float height, float depth, float length)
        : wave(height, depth, length) {
        compute_mean_elevation();
    }

    /**
     * @brief Track vertical motion of a floating object on the wave surface.
     * 
     * @param duration  Total simulation time (seconds)
     * @param timestep  Integration time step (seconds)
     * @param callback  Function to call at each time step:
     *                  void callback(t, z, dzdt, ddzdt2, x);
     */
    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float)> callback)
    {
        const float wave_T = wave.get_T();
        const float wave_L = wave.get_length();

        dt = std::clamp(timestep, 1e-5f, 0.2f * wave_T / 20.0f);

        t = 0.0f;
        x = 0.0f;

        // Initialize previous vertical position and vertical velocity properly
        prev_z = wave.surface_elevation(x, t) - mean_eta;
        prev_dzdt = wave.vertical_velocity(x, prev_z + mean_eta, t);

        while (t <= duration) {
            // RK4 step for horizontal position
            float x_next = rk4_integrate_x(x, t, dt);

            // Periodicity wrap
            x_next = wrap_periodic(x_next, wave_L);

            t += dt;
            x = x_next;

            // Surface elevation at new x and t (zero-centered)
            float z = wave.surface_elevation(x, t) - mean_eta;

            // Vertical velocity and acceleration by finite difference
            float dzdt = (z - prev_z) / dt;
            float ddzdt2 = (dzdt - prev_dzdt) / dt;

            callback(t, z, dzdt, ddzdt2, x);

            prev_z = z;
            prev_dzdt = dzdt;
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
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.05f;         // Time step (s)

    // Create a 4th-order Fenton wave and a surface tracker
    WaveSurfaceTracker<4> tracker(height, depth, length);

    // Output file
    std::ofstream out("wave_tracker_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track floating object (using callback)
    tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif


