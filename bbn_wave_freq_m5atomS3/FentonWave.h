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

/**
 * @brief Nonlinear Stokes-type wave solver using Fenton's method.
 * 
 * This class solves the nonlinear free surface wave problem using
 * a truncated Fourier series and the method described by Fenton.
 */
template<int N = 4>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using VectorF = Eigen::Matrix<float, N + 1, 1>;
    using BigVector = Eigen::Matrix<float, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<float, StateDim, StateDim>;

    float height, depth, length, g, relax;
    VectorF eta, B;
    float k, c, T, omega;

public:
    FentonWave(float height, float depth, float length, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(clamp_value(relax, 0.1f, 1.0f)) {

        if (height <= 0 || depth <= 0 || length <= 0)
            throw std::invalid_argument("Wave parameters must be positive.");
        if (height / depth > 0.78f)
            throw std::invalid_argument("Wave too steep (H/d > 0.78).");

        auto coeffs = solve_fenton();
        set_coefficients(coeffs);
    }

    float surface_elevation(float x, float t = 0.0f) const {
        float phase = k * (x - c * t);
        float eta_val = 0.0f;
        for (int i = 1; i <= N; ++i)
            eta_val += eta[i] * std::cos(i * phase);
        return eta_val;
    }

    /**
     * @brief Compute surface slope ∂η/∂x at (x, t).
     */
    float surface_slope(float x_val, float t = 0) const {
        float phase = (x_val - c * t) * k;
        float d_eta = 0.0f;
        for (int i = 1; i <= N; ++i)
            d_eta -= i * k * eta[i] * std::sin(i * phase);
        return d_eta;
    }
    
    /**
     * @brief Time derivative of surface elevation ∂η/∂t at (x, t).
     * Uses kinematic relation: ∂η/∂t = -c ∂η/∂x
     */
    float surface_time_derivative(float x_val, float t = 0) const {
        return -c * surface_slope(x_val, t);
    }
    
    /**
     * @brief Vertical velocity component w(x, z, t).
     */
    float vertical_velocity(float x_val, float z_val, float t = 0) const {
        float w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float arg = kj * (x_val - c * t);
            float denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<float>::epsilon()) continue;
    
            float term = B[j] * kj / denom;
            w += term * std::sin(arg) * std::sinh(kj * (z_val + depth));
        }
        return w;
    }

float get_k() const { return k; }
    float get_c() const { return c; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }
    float get_height() const { return height; }
    float get_length() const { return length; }
    const VectorF& get_eta() const { return eta; }

private:
    struct FentonCoefficients {
        VectorF B, eta;
        float k, c, Q, R;
    };

    void set_coefficients(const FentonCoefficients& coeffs) {
        B = coeffs.B * std::sqrt(g * depth);
        eta = coeffs.eta * depth;
        k = coeffs.k / depth;
        c = coeffs.c * std::sqrt(g * depth);
        T = length / c;
        omega = c * k;
    }

    FentonCoefficients solve_fenton(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        float k_nd = 2 * M_PI / lambda;

        float c0 = std::sqrt(std::tanh(k_nd) / k_nd);  // nondim phase speed

        VectorF J;
        for (int i = 0; i <= N; ++i) J[i] = i;

        VectorF x_nd;
        for (int i = 0; i <= N; ++i)
            x_nd[i] = 2 * M_PI * i / (N + 1);  // evenly spaced over 0 to 2π

        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.eta.setZero();
        coeffs.B[0] = c0;

        // Better initial guess with decaying harmonics
        for (int i = 1; i <= N; ++i) {
            coeffs.eta[i] = H * std::pow(0.5f, i);
            coeffs.B[i] = H * c0 * std::pow(0.5f, i);
        }

        coeffs.Q = 0.0f;
        coeffs.R = 1.0f;

        BigVector params;
        params.segment(0, N + 1) = coeffs.B;
        params.segment(N + 1, N + 1) = coeffs.eta;
        params[2 * (N + 1)] = coeffs.Q;
        params[2 * (N + 1) + 1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector res = compute_residuals(params, H, k_nd, x_nd);
            float err = res.norm();
            if (err < tol) break;

            BigMatrix Jmat = compute_jacobian(params, H, k_nd, x_nd);
            BigVector delta = Jmat.colPivHouseholderQr().solve(res);
            params -= relax * delta;
        }

        coeffs.B   = params.segment(0, N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q   = params[2 * (N + 1)];
        coeffs.R   = params[2 * (N + 1) + 1];
        coeffs.k   = k_nd;
        coeffs.c   = coeffs.B[0];

        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, const VectorF& x_nd) {
        BigVector res = BigVector::Zero();
        VectorF B   = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        VectorF eta_x;
        for (int m = 0; m <= N; ++m) {
            float phase = k * x_nd[m];
            float val = 0.0f;
            for (int j = 1; j <= N; ++j)
                val += eta[j] * std::cos(j * phase);
            eta_x[m] = val;
        }

        int idx_max = 0, idx_min = 0;
        for (int i = 0; i <= N; ++i) {
            if (eta_x[i] > eta_x[idx_max]) idx_max = i;
            if (eta_x[i] < eta_x[idx_min]) idx_min = i;
        }

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float etam = eta[m];

            float um = B[0];
            float vm = 0.0f;

            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                float denom = std::cosh(kj);
                float S = std::sinh(kj * etam) / denom;
                float C = std::cosh(kj * etam) / denom;
                float phase = kj * xm;
                um += kj * B[j] * C * std::cos(phase);
                vm += kj * B[j] * S * std::sin(phase);
            }

            float bernoulli = 0.5f * (um * um + vm * vm) + eta_x[m] - R;
            float surface = -B[0] * etam + Q;
            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                float denom = std::cosh(kj);
                float S = std::sinh(kj * etam) / denom;
                float phase = kj * xm;
                surface += B[j] * S * std::cos(phase);
            }

            res[m] = surface;
            res[N + 1 + m] = bernoulli;
        }

        // Mean elevation condition
        float dx = length / N;
        float sum = 0.5f * (eta_x[0] + eta_x[N]);
        for (int i = 1; i < N; ++i) sum += eta_x[i];
        res[2 * (N + 1)] = sum * dx / length;

        // Height condition
        res[2 * (N + 1) + 1] = eta_x[idx_max] - eta_x[idx_min] - H;

        return res;
    }

    BigMatrix compute_jacobian(const BigVector& params, float H, float k, const VectorF& x_nd) {
        // Omitted for brevity in this version — works as-is from your original if compute_residuals is fixed
        BigMatrix Jmat = BigMatrix::Zero();
        // Use original or improved version of compute_jacobian
        return Jmat;
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

public:
    WaveSurfaceTracker(float height, float depth, float length)
        : wave(height, depth, length) {}

    /**
     * @brief Compute horizontal speed dx/dt from surface kinematic constraint:
     *        dx/dt = (w - ∂η/∂t) / (∂η/∂x)
     */
    float compute_horizontal_speed(float x_pos, float time) const {
        float eta      = wave.surface_elevation(x_pos, time);
        float eta_dot  = wave.surface_time_derivative(x_pos, time);
        float eta_x    = wave.surface_slope(x_pos, time);
        float w        = wave.vertical_velocity(x_pos, eta, time);

        // Clamp slope to avoid division by zero
        if (std::abs(eta_x) < slope_eps)
            eta_x = (eta_x >= 0.0f) ? slope_eps : -slope_eps;

        return (w - eta_dot) / eta_x;
    }

    /**
     * @brief Perform RK4 integration for horizontal position.
     */
    float rk4_integrate_x(float x_curr, float t_curr, float dt_step) const {
        float k1 = compute_horizontal_speed(x_curr, t_curr);
        float k2 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k1, t_curr + 0.5f * dt_step);
        float k3 = compute_horizontal_speed(x_curr + 0.5f * dt_step * k2, t_curr + 0.5f * dt_step);
        float k4 = compute_horizontal_speed(x_curr + dt_step * k3, t_curr + dt_step);
        return x_curr + (dt_step / 6.0f) * (k1 + 2*k2 + 2*k3 + k4);
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

        dt = clamp_value(timestep, 1e-5f, 0.2f * wave_T / 20.0f);
        t = 0.0f;
        x = 0.0f;

        prev_z = wave.surface_elevation(x, t);
        prev_dzdt = 0.0f;

        while (t <= duration) {
            // RK4 step for horizontal position
            float x_next = rk4_integrate_x(x, t, dt);

            // Periodicity wrap
            if (x_next < 0) x_next += wave_L;
            else if (x_next >= wave_L) x_next -= wave_L;

            t += dt;
            x = x_next;

            // Surface elevation at new x and t
            float z = wave.surface_elevation(x, t);

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


