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

template <int N>
class FentonWave {
private:

    static constexpr int StateDim = 2 * (N + 1) + 2;

    using VectorF = Eigen::Matrix<float, N + 1, 1>;
    using BigVector = Eigen::Matrix<float, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<float, StateDim, StateDim>;

public:
    float height, depth, length, g, relax;
    float k, c, T, omega;
    Eigen::Matrix<float, N + 1, 1> eta, x;
    Eigen::Matrix<float, N + 1, 1> E;

    FentonWave(float height, float depth, float length, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax)
    {
        compute();
    }

    float surface_elevation(float x_val, float t = 0.0f) const {
        float sum = 0.0f;
        for (int j = 0; j <= N; ++j) {
            float arg = j * k * (x_val - c * t);
            sum += E(j) * std::cos(arg);
        }
        return 2.0f * sum / N;
    }

    Eigen::Vector2f velocity(float x, float z, float t = 0) const {
        if (depth < 0)
            throw std::runtime_error("Cannot compute velocity for infinite depth");

        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        float phase = k * (x - c * t);

        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float denom = std::cosh(kj * depth);
            float term = B[j] * kj / denom;

            vel[0] += term * std::cos(j * phase) * std::cosh(kj * z);
            vel[1] += term * std::sin(j * phase) * std::sinh(kj * z);
        }

        return vel;
    }

    float surface_slope(float x, float t = 0) const {
        float phase = k * (x - c * t);
        float d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= 2.0f * E[j] * j * k * std::sin(j * phase) / N;
        }
        return d_eta;
    }

    float surface_time_derivative(float x, float t = 0) const {
        return -c * surface_slope(x, t);
    }

    float vertical_velocity(float x, float z, float t = 0) const {
        float w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float arg = kj * (x - c * t);
            float denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<float>::epsilon()) continue;
            float term = B[j] * kj / denom;
            w += term * std::sin(arg) * std::sinh(kj * (z + depth));
        }
        return w;
    }

    // Getters
    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }
    float get_length() const { return length; }
    float get_height() const { return height; }
    const VectorF& get_eta() const { return eta; }

private:
    void compute() {
        if (depth < 0) depth = 25.0f * length;
        float H = height / depth;
        float lam = length / depth;
        k = 2 * M_PI / lam;
        float D = 1.0f;
        float kc = k;
        float c0 = std::sqrt(std::tanh(kc) / kc);

        Eigen::Matrix<float, N + 1, 1> x_nd;
        for (int m = 0; m <= N; ++m)
            x_nd(m) = lam * m / (2.0f * N);

        // Initial guess
        Eigen::Matrix<float, N + 1, 1> B = Eigen::Matrix<float, N + 1, 1>::Zero();
        B(0) = c0;
        B(1) = -H / (4.0f * c0 * k);

        Eigen::Matrix<float, N + 1, 1> eta_nd;
        for (int m = 0; m <= N; ++m)
            eta_nd(m) = 1.0f + H / 2.0f * std::cos(k * x_nd(m));

        float Q = c0, R = 1.0f + 0.5f * c0 * c0;

        // Stepping
        for (float Hi : wave_height_steps(H, D, lam)) {
            optimize(B, Q, R, eta_nd, Hi, k, D);
        }

        // Scale to physical
        float sqrt_gd = std::sqrt(g * depth);
        B(0) *= sqrt_gd;
        for (int j = 1; j <= N; ++j)
            B(j) *= std::sqrt(g * std::pow(depth, 3));
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;

        // Store results
        for (int i = 0; i <= N; ++i) {
            x(i) = x_nd(i) * depth;
            eta(i) = eta_nd(i) * depth;
        }
        k = k / depth;
        c = B(0);
        T = length / c;
        omega = c * k;

        // Compute E for elevation interpolation
        for (int j = 0; j <= N; ++j) {
            float sum = 0.0f;
            for (int i = 0; i <= N; ++i) {
                float phi = M_PI * j * i / N;
                float w = (i == 0 || i == N) ? 0.5f : 1.0f;
                sum += eta(i) * std::cos(phi) * w;
            }
            E(j) = sum / N;
        }
    }

    std::vector<float> wave_height_steps(float H, float D, float lam) {
        float Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
        int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
        std::vector<float> steps;
        for (int i = 1; i <= num; ++i)
            steps.push_back(H * i / num);
        return steps;
    }

    void optimize(Eigen::Matrix<float, N + 1, 1>& B, float& Q, float& R,
                  Eigen::Matrix<float, N + 1, 1>& eta, float H, float k, float D)
    {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<float, NU, 1> coeffs;
        coeffs.segment(0, N + 1) = B;
        coeffs.segment(N + 1, N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;

        for (int iter = 0; iter < 100; ++iter) {
            Eigen::Matrix<float, NU, 1> f = compute_residual(coeffs, H, k, D);
            Eigen::Matrix<float, NU, NU> J = compute_jacobian(coeffs, H, k, D);
            Eigen::Matrix<float, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;

            if (f.cwiseAbs().maxCoeff() < 1e-8f) break;
        }

        B = coeffs.segment(0, N + 1);
        eta = coeffs.segment(N + 1, N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
    }

    Eigen::Matrix<float, 2 * (N + 1) + 2, 1>
    compute_residual(const Eigen::Matrix<float, 2 * (N + 1) + 2, 1>& coeffs, float H, float k, float D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<float, NU, 1> f;
        auto B = coeffs.segment(0, N + 1);
        auto eta = coeffs.segment(N + 1, N + 1);
        float Q = coeffs(2 * N + 2);
        float R = coeffs(2 * N + 3);
        float B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            float x_m = M_PI * m / N;
            float eta_m = eta(m);
            float um = -B0, vm = 0, S1, C1, S2, C2;
            for (int j = 1; j <= N; ++j) {
                float aj = j * k * eta_m;
                float bj = j * k * D;
                float sinh_ratio = std::sinh(aj) / std::cosh(bj);
                float cosh_ratio = std::cosh(aj) / std::cosh(bj);
                S1 = sinh_ratio;
                C1 = cosh_ratio;
                S2 = std::sin(j * x_m);
                C2 = std::cos(j * x_m);
                um += k * j * B(j) * C1 * C2;
                vm += k * j * B(j) * S1 * S2;
            }
            f(m) = -B0 * eta_m;
            for (int j = 1; j <= N; ++j) {
                float aj = j * k * eta_m;
                float bj = j * k * D;
                float sinh_ratio = std::sinh(aj) / std::cosh(bj);
                float cosh_ratio = std::cosh(aj) / std::cosh(bj);
                float C2 = std::cos(j * M_PI * m / N);
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

    Eigen::Matrix<float, 2 * (N + 1) + 2, 2 * (N + 1) + 2>
    compute_jacobian(const Eigen::Matrix<float, 2 * (N + 1) + 2, 1>& coeffs, float H, float k, float D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<float, NU, NU> J = Eigen::Matrix<float, NU, NU>::Zero();
        auto B = coeffs.segment(0, N + 1);
        auto eta = coeffs.segment(N + 1, N + 1);
        float B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            float eta_m = eta(m);
            float x_m = M_PI * m / N;
            float um = -B0, vm = 0;

            Eigen::Array<float, N, 1> S1, C1, S2, C2;
            for (int j = 1; j <= N; ++j) {
                float aj = j * k * eta_m;
                float bj = j * k * D;
                float sinh_ratio = std::sinh(aj) / std::cosh(bj);
                float cosh_ratio = std::cosh(aj) / std::cosh(bj);
                float s2 = std::sin(j * x_m);
                float c2 = std::cos(j * x_m);
                um += k * j * B(j) * cosh_ratio * c2;
                vm += k * j * B(j) * sinh_ratio * s2;
            }

            J(m, 0) = -eta_m;
            for (int j = 1; j <= N; ++j) {
                float aj = j * k * eta_m;
                float bj = j * k * D;
                float sinh_ratio = std::sinh(aj) / std::cosh(bj);
                float c2 = std::cos(j * M_PI * m / N);
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


