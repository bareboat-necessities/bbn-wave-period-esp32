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
    VectorF eta, x, B;
    float k, c, T, omega;

public:
    // Constructor with physical parameters (height, depth, length)
    FentonWave(float height, float depth, float length, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(clamp_value(relax, 0.1f, 1.0f)) {
        if (depth <= 0 || length <= 0 || height <= 0)
            throw std::invalid_argument("Physical parameters must be positive");
        if (height / depth > 0.78f)
            throw std::invalid_argument("Wave too steep (H/d > 0.78)");

        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    // Surface elevation η(x,t) in meters
    float surface_elevation(float x_val, float t = 0) const {
        float phase = (x_val - c * t) * k;
        float eta_val = eta[0];
        for (int i = 1; i <= N; ++i)
            eta_val += eta[i] * std::cos(i * phase);
        return eta_val;
    }

    // Velocity vector (u,w) at position (x,z) and time t
    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        vel[0] = B[0];
        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float arg = kj * (x_val - c * t);
            float denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<float>::epsilon()) continue;
            float term = B[j] * kj / denom;
            vel[0] += term * std::cos(arg) * std::cosh(kj * (z_val + depth));
            vel[1] += term * std::sin(arg) * std::sinh(kj * (z_val + depth));
        }
        return vel;
    }

    // Surface slope ∂η/∂x at position x and time t
    float surface_slope(float x_val, float t = 0) const {
        float phase = (x_val - c * t) * k;
        float d_eta = 0.0f;
        for (int i = 1; i <= N; ++i)
            d_eta -= i * k * eta[i] * std::sin(i * phase);
        return d_eta;
    }

    // Surface time derivative ∂η/∂t at position x and time t
    float surface_time_derivative(float x_val, float t = 0) const {
        return -c * surface_slope(x_val, t);
    }

    // Vertical velocity w(x,z,t) at position (x,z) and time t
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

    // Getters for wave parameters
    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }
    float get_length() const { return length; }
    float get_height() const { return height; }
    const VectorF& get_eta() const { return eta; }

private:
    struct FentonCoefficients {
        VectorF x, eta, B;
        float k, c, Q, R;
    };

    void set_coefficients(const FentonCoefficients& coeffs) {
        eta = coeffs.eta * depth;
        x = coeffs.x * depth;
        k = coeffs.k;
        c = coeffs.c * std::sqrt(g * depth);
        B = coeffs.B * std::sqrt(g * depth);
        T = length / c;
        omega = c * k;
    }

    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        float k_nd = 2 * M_PI / lambda;

        // Initial guess for wave speed c (nondim)
        float c_guess = std::sqrt(std::tanh(k_nd) / k_nd);

        // Collocation points
        Eigen::Matrix<float, N + 1, 1> J;
        for (int i = 0; i <= N; ++i) J[i] = i;
        VectorF x_nd;
        for (int i = 0; i <= N; ++i) x_nd[i] = M_PI * (2.0f * i + 1.0f) / (N + 1);

        // Initial coefficients guess
        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.B[0] = c_guess;
        coeffs.B[1] = H * coeffs.B[0] / (2 * std::tanh(k_nd));
        coeffs.eta.setZero();
        coeffs.eta[1] = H / 2.0f;
        coeffs.Q = 0.0f;
        coeffs.R = 1.0f + 0.5f * H * H / (4.0f * std::tanh(k_nd));
        coeffs.x = x_nd;

        BigVector params;
        params.segment(0, N + 1) = coeffs.B;
        params.segment(N + 1, N + 1) = coeffs.eta;
        params[2 * (N + 1)] = coeffs.Q;
        params[2 * (N + 1) + 1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector res = compute_residuals(params, H, k_nd, x_nd);
            if (res.norm() < tol) break;
            BigMatrix Jmat = compute_jacobian(params, H, k_nd, x_nd);
            Eigen::ColPivHouseholderQR<BigMatrix> solver(Jmat);
            BigVector delta = solver.solve(res);
            params -= relax * delta;

            // Clamp parameters to avoid NaNs
            for (int i = 0; i < params.size(); ++i) {
                if (std::isnan(params[i]) || std::isinf(params[i])) {
                    throw std::runtime_error("NaN or Inf detected in parameters during iteration");
                }
            }
        }

        coeffs.B = params.segment(0, N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q = params[2 * (N + 1)];
        coeffs.R = params[2 * (N + 1) + 1];
        coeffs.k = k_nd / depth;
        coeffs.c = coeffs.B[0];

        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, const VectorF& x_nd) {
        BigVector res = BigVector::Zero();

        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        // Calculate eta_spatial = η(x_nd) at collocation points
        Eigen::Matrix<float, N + 1, 1> eta_spatial;
        for (int m = 0; m <= N; ++m) {
            float phase = k * x_nd[m];
            float val = eta[0];
            for (int j = 1; j <= N; ++j)
                val += eta[j] * std::cos(j * phase);
            eta_spatial[m] = val;
        }

        // Precompute denominator: cosh(kj * depth)
        Eigen::Matrix<float, N + 1, 1> denom;
        denom[0] = 1.0f;
        for (int j = 1; j <= N; ++j)
            denom[j] = std::cosh(k * j * depth);

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float eta_m = eta_spatial[m];

            float u_m = B[0];
            float v_m = 0.0f;
            for (int j = 1; j <= N; ++j) {
                float kj = k * j;
                float cos_nx = std::cos(j * k * xm);
                float sin_nx = std::sin(j * k * xm);
                float cosh_kj_eta = std::cosh(kj * eta_m);
                float sinh_kj_eta = std::sinh(kj * eta_m);
                float denom_j = denom[j];

                u_m += kj * B[j] * cosh_kj_eta / denom_j * cos_nx;
                v_m += kj * B[j] * sinh_kj_eta / denom_j * sin_nx;
            }

            // Bernoulli residual (momentum)
            res[m] = -B[0] * eta_m + Q + 0.5f * (u_m * u_m + v_m * v_m);
            // Kinematic residual (free surface condition)
            res[N + 1 + m] = eta_m - R + (u_m * v_m);
        }

        // Mean elevation zero residual
        float mean_eta = eta_spatial.sum() / (N + 1);
        res[2 * (N + 1)] = mean_eta;

        // Wave height residual
        float eta_max = eta_spatial[0];
        float eta_min = eta_spatial[0];
        for (int m = 1; m <= N; ++m) {
            if (eta_spatial[m] > eta_max) eta_max = eta_spatial[m];
            if (eta_spatial[m] < eta_min) eta_min = eta_spatial[m];
        }
        res[2 * (N + 1) + 1] = eta_max - eta_min - H;

        return res;
    }

    BigMatrix compute_jacobian(const BigVector& params, float H, float k, const VectorF& x_nd) {
        BigMatrix J = BigMatrix::Zero();

        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);

        // Calculate eta_spatial = η(x_nd) at collocation points
        Eigen::Matrix<float, N + 1, 1> eta_spatial;
        for (int m = 0; m <= N; ++m) {
            float phase = k * x_nd[m];
            float val = eta[0];
            for (int j = 1; j <= N; ++j)
                val += eta[j] * std::cos(j * phase);
            eta_spatial[m] = val;
        }

        // Precompute denominator: cosh(kj * depth)
        Eigen::Matrix<float, N + 1, 1> denom;
        denom[0] = 1.0f;
        for (int j = 1; j <= N; ++j)
            denom[j] = std::cosh(k * j * depth);

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float eta_m = eta_spatial[m];

            float u_m = B[0];
            float v_m = 0.0f;

            Eigen::Matrix<float, N + 1, 1> du_dB = Eigen::Matrix<float, N + 1, 1>::Zero();
            Eigen::Matrix<float, N + 1, 1> dv_dB = Eigen::Matrix<float, N + 1, 1>::Zero();

            Eigen::Matrix<float, N + 1, 1> du_deta = Eigen::Matrix<float, N + 1, 1>::Zero();
            Eigen::Matrix<float, N + 1, 1> dv_deta = Eigen::Matrix<float, N + 1, 1>::Zero();

            du_dB[0] = 1.0f;
            for (int j = 1; j <= N; ++j) {
                float kj = k * j;
                float cos_nx = std::cos(j * k * xm);
                float sin_nx = std::sin(j * k * xm);
                float denom_j = denom[j];
                float cosh_kj_eta = std::cosh(kj * eta_m);
                float sinh_kj_eta = std::sinh(kj * eta_m);

                u_m += kj * B[j] * cosh_kj_eta / denom_j * cos_nx;
                v_m += kj * B[j] * sinh_kj_eta / denom_j * sin_nx;

                du_dB[j] = kj * cosh_kj_eta / denom_j * cos_nx;
                dv_dB[j] = kj * sinh_kj_eta / denom_j * sin_nx;

                for (int n = 1; n <= N; ++n) {
                    float d_eta_n = std::cos(n * k * xm);
                    du_deta[n] += kj * B[j] * kj * sinh_kj_eta / denom_j * cos_nx * d_eta_n;
                    dv_deta[n] += kj * B[j] * kj * cosh_kj_eta / denom_j * sin_nx * d_eta_n;
                }
            }

            // Jacobian for Bernoulli residuals:
            for (int i = 0; i <= N; ++i) {
                float dres_dB = (i == 0 ? -eta_m : 0.0f) + u_m * du_dB[i] + v_m * dv_dB[i];
                J(m, i) = dres_dB;
            }
            for (int i = 1; i <= N; ++i) {
                float dres_deta = -B[0] * std::cos(i * k * xm) + u_m * du_deta[i] + v_m * dv_deta[i];
                J(m, N + 1 + i) = dres_deta;
            }
            J(m, 2 * (N + 1)) = 1.0f;     // d/dQ
            J(m, 2 * (N + 1) + 1) = 0.0f; // d/dR

            // Jacobian for kinematic residuals:
            for (int i = 0; i <= N; ++i) {
                J(N + 1 + m, i) = v_m * du_dB[i] + u_m * dv_dB[i];
            }
            for (int i = 1; i <= N; ++i) {
                float d_eta_i = std::cos(i * k * xm);
                J(N + 1 + m, N + 1 + i) = d_eta_i + v_m * du_deta[i] + u_m * dv_deta[i];
            }
            J(N + 1 + m, 2 * (N + 1)) = 0.0f;
            J(N + 1 + m, 2 * (N + 1) + 1) = -1.0f;
        }

        // Mean elevation residual derivatives:
        for (int j = 1; j <= N; ++j) {
            float sum_cos = 0.0f;
            for (int m = 0; m <= N; ++m) {
                sum_cos += std::cos(j * k * x_nd[m]);
            }
            J(2 * (N + 1), N + 1 + j) = sum_cos / (N + 1);
        }
        for (int i = 0; i <= N; ++i) J(2 * (N + 1), i) = 0.0f;
        J(2 * (N + 1), 2 * (N + 1)) = 0.0f;
        J(2 * (N + 1), 2 * (N + 1) + 1) = 0.0f;

        // Wave height residual derivatives:
        // Approximate with max and min points gradients.
        // For better accuracy, this should be a smooth approximation,
        // but here we use a simplified approach for demonstration.
        J(2 * (N + 1) + 1, 0) = 0.0f;
        for (int j = 1; j <= N; ++j) {
            J(2 * (N + 1) + 1, N + 1 + j) = 0.0f; // Needs improvement for exact sensitivity
        }
        J(2 * (N + 1) + 1, 2 * (N + 1)) = 0.0f;
        J(2 * (N + 1) + 1, 2 * (N + 1) + 1) = 0.0f;

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


