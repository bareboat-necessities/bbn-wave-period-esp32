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
template<int N = 3>
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
    FentonWave(float height, float depth, float length,
               float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length),
          g(g), relax(clamp_value(relax, 0.1f, 1.0f)) {

        if (depth <= 0 || length <= 0 || height <= 0)
            throw std::invalid_argument("Physical parameters must be positive");
        if (height / depth > 0.78f)
            throw std::invalid_argument("Wave too steep (H/d > 0.78)");

        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    /**
     * @brief Get surface elevation η(x, t) in meters.
     */
    float surface_elevation(float x_val, float t = 0) const {
        float x_nd = (x_val - c * t) / depth;  // nondimensional x
        float eta_val = 0.0f;
        for (int i = 0; i <= N; ++i)
            eta_val += eta[i] * std::cos(i * k * x_nd);
        return eta_val;
    }

    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        vel[0] = B[0];  // mean horizontal velocity

        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float arg = kj * (x_val - c * t) / depth;
            float denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<float>::epsilon()) continue;

            float term = B[j] * kj / denom;
            vel[0] += term * std::cos(arg) * std::cosh(kj * (z_val + depth)); // u
            vel[1] += term * std::sin(arg) * std::sinh(kj * (z_val + depth)); // w
        }
        return vel;
    }

    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }
    const VectorF& get_eta() const { return eta; }

private:
    struct FentonCoefficients {
        VectorF x, eta, B;
        float k, c, Q, R;
    };

    void set_coefficients(const FentonCoefficients& coeffs) {
        eta = coeffs.eta * depth;
        x = coeffs.x * depth;
        k = coeffs.k / depth;
        c = coeffs.c * std::sqrt(g * depth);
        B = coeffs.B * std::sqrt(g * depth);
        T = length / c;
        omega = c * k;
    }

    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        float k_nd = 2 * M_PI / lambda;

        float c_guess = std::sqrt(g * depth * std::tanh(k_nd * 1.0f) / k_nd);

        VectorF grid;
        grid.setLinSpaced(N + 1, 0, N);
        VectorF x_nd = (grid.array() * lambda / N).matrix(); // nondim x ∈ [0, λ]

        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.B[0] = c_guess / std::sqrt(g * depth); // nondim
        if (N >= 1)
            coeffs.B[1] = H * coeffs.B[0] / (2 * std::tanh(k_nd));

        coeffs.eta = (H / 2) * (k_nd * x_nd.array()).cos().matrix(); // nondim
        coeffs.Q = 0;
        coeffs.R = 1 + 0.5f * H * H / (4 * std::tanh(k_nd));

        BigVector params;
        params.segment(0, N + 1) = coeffs.B;
        params.segment(N + 1, N + 1) = coeffs.eta;
        params[2 * (N + 1)] = coeffs.Q;
        params[2 * (N + 1) + 1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector f = compute_residuals(params, H, k_nd, x_nd);
            if (f.norm() < tol)
                break;

            BigMatrix J = compute_jacobian(params, H, k_nd, x_nd);
            BigVector delta = J.colPivHouseholderQr().solve(f);
            params -= relax * delta;
        }

        coeffs.B = params.segment(0, N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q = params[2 * (N + 1)];
        coeffs.R = params[2 * (N + 1) + 1];
        coeffs.x = x_nd;
        coeffs.k = k_nd;
        coeffs.c = coeffs.B[0];
        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, const VectorF& x_nd) {
        BigVector res = BigVector::Zero();
        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        VectorF J;
        J.setLinSpaced(N + 1, 0, N);

        int idx_max = 0, idx_min = 0;
        for (int j = 1; j <= N; ++j) {
            if (eta[j] > eta[idx_max]) idx_max = j;
            if (eta[j] < eta[idx_min]) idx_min = j;
        }

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float etam = eta[m];

            VectorF Jk_eta = J * k * etam;
            VectorF denom = (J * k).array().cosh();

            VectorF S1 = Jk_eta.array().sinh() / denom.array();
            VectorF C1 = Jk_eta.array().cosh() / denom.array();
            VectorF S2 = (J * k * xm).array().sin();
            VectorF C2 = (J * k * xm).array().cos();

            float um = B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }

            res[m] = -B[0] * etam + Q + (B.tail(N).array() * S1.tail(N).array() * C2.tail(N).array()).sum();
            res[N + 1 + m] = 0.5f * (um * um + vm * vm) + etam - R;
        }

        res[2 * (N + 1)] = trapezoid_integration(eta) / length;
        res[2 * (N + 1) + 1] = eta[idx_max] - eta[idx_min] - H;

        return res;
    }

    BigMatrix compute_jacobian(const BigVector& params, float H, float k, const VectorF& x_nd) {
        BigMatrix Jmat = BigMatrix::Zero();

        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        VectorF J;
        J.setLinSpaced(N + 1, 0, N);

        int idx_max = 0, idx_min = 0;
        for (int j = 1; j <= N; ++j) {
            if (eta[j] > eta[idx_max]) idx_max = j;
            if (eta[j] < eta[idx_min]) idx_min = j;
        }

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float etam = eta[m];

            VectorF Jk_eta = J * k * etam;
            VectorF denom = (J * k).array().cosh();

            VectorF S1 = Jk_eta.array().sinh() / denom.array();
            VectorF C1 = Jk_eta.array().cosh() / denom.array();
            VectorF S2 = (J * k * xm).array().sin();
            VectorF C2 = (J * k * xm).array().cos();

            VectorF dS1_deta = (J * k).array() * C1.array() / denom.array();
            VectorF dC1_deta = (J * k).array() * S1.array() / denom.array();

            float um = B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }

            Jmat(m, 0) = -etam;
            for (int j = 1; j <= N; ++j)
                Jmat(m, j) = S1[j] * C2[j];

            Jmat(m, N + 1 + m) = (B.tail(N).array() * dS1_deta.tail(N).array() * C2.tail(N).array()).sum();
            Jmat(m, 2 * (N + 1)) = 1.0f;

            Jmat(N + 1 + m, 0) = um;
            for (int j = 1; j <= N; ++j) {
                float d_um = k * C1[j] * C2[j] * J[j];
                float d_vm = k * S1[j] * S2[j] * J[j];
                Jmat(N + 1 + m, j) = um * d_um + vm * d_vm;
            }

            float d_um_deta = 0, d_vm_deta = 0;
            for (int j = 1; j <= N; ++j) {
                d_um_deta += B[j] * dC1_deta[j] * C2[j] * J[j];
                d_vm_deta += B[j] * dS1_deta[j] * S2[j] * J[j];
            }
            Jmat(N + 1 + m, N + 1 + m) = k * (um * d_um_deta + vm * d_vm_deta) + 1.0f;
            Jmat(N + 1 + m, 2 * (N + 1) + 1) = -1.0f;
        }

        float dx = length / N;
        for (int j = 0; j <= N; ++j) {
            float w = (j == 0 || j == N) ? 0.5f : 1.0f;
            Jmat(2 * (N + 1), N + 1 + j) = w * dx / length;
        }

        Jmat(2 * (N + 1) + 1, N + 1 + idx_max) = 1.0f;
        Jmat(2 * (N + 1) + 1, N + 1 + idx_min) = -1.0f;

        return Jmat;
    }

    float trapezoid_integration(const VectorF& y) const {
        float dx = length / N;
        float sum = 0.5f * (y[0] + y[N]);
        for (int i = 1; i < N; ++i) sum += y[i];
        return sum * dx;
    }
};

/**
 * @brief Tracks vertical surface kinematics at the crest using Lagrangian motion.
 *
 * This class uses the FentonWave model to simulate the vertical displacement,
 * velocity, and acceleration of a water parcel following the surface motion
 * over time, assuming it starts at the wave crest and follows the wave.
 */
template<int N = 3>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;        // Underlying nonlinear wave model
    float phase_velocity;      // Wave celerity c
    float wave_number;         // Wave number k

    // History for finite difference velocity/acceleration estimates
    float eta_prev2 = 0, eta_prev = 0, eta_current = 0;
    bool has_prev = false, has_prev2 = false;

public:
    WaveSurfaceTracker(float height, float depth, float length)
        : wave(height, depth, length),
          phase_velocity(wave.get_c()),
          wave_number(wave.get_k()) {}

    /**
     * @brief Simulates the Lagrangian kinematics of a surface particle.
     *
     * @param duration       Total time of simulation (seconds)
     * @param timestep       Sampling timestep (seconds)
     * @param callback       Function called on each timestep with:
     *                       (time, elevation, vertical_velocity, vertical_acceleration, horizontal_position)
     */
    void track_lagrangian_kinematics(
        float duration,
        float timestep,
        std::function<void(
            float time,
            float elevation,
            float vertical_velocity,
            float vertical_acceleration,
            float horizontal_position)> callback) 
    {
        if (timestep <= 0) throw std::invalid_argument("Timestep must be positive");
        if (duration <= 0) throw std::invalid_argument("Duration must be positive");

        // Reset history state
        eta_prev2 = eta_prev = eta_current = 0;
        has_prev = has_prev2 = false;

        for (float time = 0; time <= duration; time += timestep) {
            // Assume particle follows wave crest: x(t) = c * t
            float x = phase_velocity * time;

            // Evaluate surface elevation at this position and time
            float elevation = wave.surface_elevation(x, time);

            // Shift history buffers
            eta_prev2 = eta_prev;
            eta_prev = eta_current;
            eta_current = elevation;

            has_prev2 = has_prev;
            has_prev = true;

            // Estimate velocity and acceleration
            float vertical_velocity = 0;
            float vertical_acceleration = 0;

            if (has_prev2) {
                // Central difference (second-order accurate)
                vertical_velocity = (eta_current - eta_prev2) / (2 * timestep);
                vertical_acceleration = (eta_current - 2 * eta_prev + eta_prev2) / (timestep * timestep);
            } else if (has_prev) {
                // Forward difference (first-order fallback)
                vertical_velocity = (eta_current - eta_prev) / timestep;
                vertical_acceleration = (eta_current - eta_prev) / timestep;
            }

            // Emit current state
            callback(time, elevation, vertical_velocity, vertical_acceleration, x);
        }
    }
};

#ifdef FENTON_TEST
template class FentonWave<3>;
template class WaveSurfaceTracker<3>;

void FentonWave_test() {
    // Wave parameters
    const float height = 2.0f;   // Wave height (m)
    const float depth = 10.0f;   // Water depth (m)
    const float length = 50.0f;  // Wavelength (m)
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.1f;       // Time step (s)

    // Create a 3rd-order Fenton wave and a surface tracker
    WaveSurfaceTracker<3> tracker(height, depth, length);

    // Output file
    std::ofstream out("wave_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track Lagrangian kinematics (using callback)
    tracker.track_lagrangian_kinematics(duration, dt, kinematics_callback);

    std::cout << "Wave data saved to wave_data.csv\n";
}
#endif


