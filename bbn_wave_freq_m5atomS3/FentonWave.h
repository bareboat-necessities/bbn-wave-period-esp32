#pragma once

#include <ArduinoEigenDense.h>
#include <cmath>
#include <stdexcept>
#include <functional>

template<int N = 3>
class FentonWave {
private:
    static constexpr float PI = 3.1415926f;
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using VectorF = Eigen::Matrix<float, N + 1, 1>;                // Vector of size N+1
    using BigVector = Eigen::Matrix<float, StateDim, 1>;          // Newton vector
    using BigMatrix = Eigen::Matrix<float, StateDim, StateDim>;   // Jacobian matrix

    // Wave parameters
    float height, depth, length, g, relax;

    // Wave state variables
    VectorF eta, x, B, E;
    float k, c, cs, T, omega;

public:
    // Constructor: initializes and solves the Fenton equations
    FentonWave(float height, float depth, float length,
               float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length),
          g(g), relax(relax) {
        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    // Compute surface elevation η(x, t)
    float surface_elevation(float x_val, float t = 0) const {
        VectorF J;
        J.setLinSpaced(N + 1, 0, N);
        return (2.0f / N) * (E.array() * (J.array() * k * (x_val - c * t)).cos()).sum();
    }

    // Compute horizontal and vertical particle velocities (u, w)
    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        for (int j = 1; j <= N; ++j) {
            float Jk = j * k;
            float arg = Jk * (x_val - c * t);
            float denom = std::cosh(Jk * depth);
            if (denom < 1e-6f) denom = 1e-6f;
            float term = B[j] * Jk / denom;
            vel[0] += term * std::cos(arg) * std::cosh(Jk * z_val); // u
            vel[1] += term * std::sin(arg) * std::sinh(Jk * z_val); // w
        }
        return vel;
    }

    // Getters for wave parameters
    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }

private:
    // Struct to hold solution of Fenton equations
    struct FentonCoefficients {
        VectorF x, eta, B;
        float k, c, Q, R;
    };

    // Set solution coefficients and compute Fourier spectrum of η
    void set_coefficients(const FentonCoefficients& coeffs) {
        eta = coeffs.eta;
        x = coeffs.x;
        k = coeffs.k;
        c = coeffs.c;
        cs = c - coeffs.Q;
        T = length / c;
        omega = c * k;
        B = coeffs.B;

        // Compute Fourier coefficients E_j = ∫ η(x) cos(jπx/λ) dx
        for (int j = 0; j <= N; ++j) {
            VectorF cosine;
            for (int i = 0; i <= N; ++i)
                cosine[i] = std::cos(j * PI * x[i] / length);
            E[j] = trapezoid_integration(eta.array() * cosine.array());
        }
    }

    // Solve the Fenton equations using Newton-Raphson iteration
    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        k = 2 * PI / lambda;
        float D = 1.0f;

        float c_guess = std::sqrt(g * depth * std::tanh(k * D) / k);

        VectorF grid;
        grid.setLinSpaced(N + 1, 0, N);
        VectorF x_nd = (grid.array() * lambda / N).matrix();
        VectorF x_phys = x_nd * depth;

        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.B[0] = c_guess;
        if (N >= 1) coeffs.B[1] = -H / (2 * k * std::cosh(k * D));
        coeffs.eta = (H / 2) * (k * x_nd.array()).cos().matrix();
        coeffs.Q = 0;
        coeffs.R = 1 + 0.5f * c_guess * c_guess;

        BigVector params;
        params.segment(0, N + 1) = coeffs.B;
        params.segment(N + 1, N + 1) = coeffs.eta;
        params[2 * (N + 1)] = coeffs.Q;
        params[2 * (N + 1) + 1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector f = compute_residuals(params, H, k, D, x_nd);
            if (!std::isfinite(f.norm()))
                throw std::runtime_error("Residual diverged: NaN or Inf");
            if (f.norm() < tol) break;

            BigMatrix J = compute_jacobian(params, H, k, D, x_nd);
            BigVector delta = J.colPivHouseholderQr().solve(f);
            params -= relax * delta;
        }

        coeffs.B = params.segment(0, N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q = params[2 * (N + 1)];
        coeffs.R = params[2 * (N + 1) + 1];
        coeffs.x = x_phys;
        scale_to_physical(coeffs);
        return coeffs;
    }

    // Compute the residual vector for the Fenton equations
    BigVector compute_residuals(const BigVector& params, float H, float k, float D, const VectorF& x_nd) {
        BigVector res;
        res.setZero();

        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        VectorF J;
        J.setLinSpaced(N + 1, 0, N);

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float etam = eta[m];

            VectorF Jk_eta = J * k * etam;
            VectorF Jk_D = J * k * D;
            VectorF denom = Jk_D.array().cosh().max(1e-6f);

            VectorF S1 = Jk_eta.array().sinh() / denom.array();
            VectorF C1 = Jk_eta.array().cosh() / denom.array();
            VectorF S2 = (J * k * xm).array().sin();
            VectorF C2 = (J * k * xm).array().cos();

            float um = -B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }

            // Kinematic boundary condition
            res[m] = -B[0] * etam + Q;
            for (int j = 1; j <= N; ++j)
                res[m] += B[j] * S1[j] * C2[j];

            // Dynamic boundary condition
            res[N + 1 + m] = 0.5f * (um * um + vm * vm) + etam - R;
        }

        // Constraint: mean surface elevation is zero
        res[2 * (N + 1)] = trapezoid_integration(eta) / length;

        // Constraint: total wave height equals input H
        float eta_max = eta.maxCoeff();
        float eta_min = eta.minCoeff();
        res[2 * (N + 1) + 1] = eta_max - eta_min - H;

        return res;
    }

    // Compute the Jacobian matrix of the residual vector
    BigMatrix compute_jacobian(const BigVector& params, float H, float k, float D, const VectorF& x_nd) {
        BigMatrix Jmat;
        Jmat.setZero();

        VectorF B = params.segment(0, N + 1);
        VectorF eta = params.segment(N + 1, N + 1);

        VectorF J;
        J.setLinSpaced(N + 1, 0, N);

        for (int m = 0; m <= N; ++m) {
            float xm = x_nd[m];
            float etam = eta[m];

            VectorF Jk_eta = J * k * etam;
            VectorF Jk_D = J * k * D;
            VectorF denom = Jk_D.array().cosh().max(1e-6f);

            VectorF S1 = Jk_eta.array().sinh() / denom.array();
            VectorF C1 = Jk_eta.array().cosh() / denom.array();
            VectorF S2 = (J * k * xm).array().sin();
            VectorF C2 = (J * k * xm).array().cos();

            VectorF dS1_deta = (J * k).array() * C1.array() / denom.array();
            VectorF dC1_deta = (J * k).array() * S1.array() / denom.array();

            float um = -B[0];
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

            Jmat(N + 1 + m, 0) = -um;
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
            Jmat(N + 1 + m, N + 1 + m) = um * k * d_um_deta + vm * k * d_vm_deta + 1.0f;
            Jmat(N + 1 + m, 2 * (N + 1) + 1) = -1.0f;
        }

        float dx = length / N;
        for (int j = 0; j <= N; ++j) {
            float w = (j == 0 || j == N) ? 0.5f : 1.0f;
            Jmat(2 * (N + 1), N + 1 + j) = w * dx / length;
        }

        int max_idx = 0, min_idx = 0;
        eta.maxCoeff(&max_idx);
        eta.minCoeff(&min_idx);
        Jmat(2 * (N + 1) + 1, N + 1 + max_idx) = 1.0f;
        Jmat(2 * (N + 1) + 1, N + 1 + min_idx) = -1.0f;

        return Jmat;
    }

    // Scale solution from nondimensional to physical units
    void scale_to_physical(FentonCoefficients& coeffs) {
        coeffs.B[0] *= std::sqrt(g * depth);
        coeffs.B.tail(N) *= std::sqrt(g * depth * depth * depth);
        coeffs.eta *= depth;
        coeffs.x *= depth;
        coeffs.k /= depth;
        coeffs.c = coeffs.B[0];
        coeffs.Q *= std::sqrt(g * depth * depth * depth);
        coeffs.R *= g * depth;
    }

    // Trapezoidal integration over uniformly spaced grid
    float trapezoid_integration(const Eigen::Array<float, N + 1, 1>& y) const {
        float dx = length / N;
        float sum = 0.5f * (y[0] + y[N]);
        for (int i = 1; i < N; ++i) sum += y[i];
        return sum * dx;
    }
};

// Tracks the Lagrangian vertical kinematics of the free surface
template<int N = 3>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;
    float phase_velocity;
    float wave_number;

    float eta_prev2 = 0, eta_prev = 0, eta_current = 0;
    bool has_prev = false, has_prev2 = false;

public:
    WaveSurfaceTracker(float height, float depth, float length)
        : wave(height, depth, length),
          phase_velocity(wave.get_c()),
          wave_number(wave.get_k()) {}

    void track_lagrangian_kinematics(
        float duration,
        float timestep,
        std::function<void(
            float time,
            float elevation,
            float vertical_velocity,
            float vertical_acceleration,
            float horizontal_position)> callback) {

        eta_prev2 = eta_prev = eta_current = 0;
        has_prev = has_prev2 = false;

        for (float time = 0; time <= duration; time += timestep) {
            float x = phase_velocity * time;
            float elevation = wave.surface_elevation(x, time);

            eta_prev2 = eta_prev;
            eta_prev = eta_current;
            eta_current = elevation;

            has_prev2 = has_prev;
            has_prev = true;

            float vertical_velocity = 0, vertical_acceleration = 0;
            if (has_prev2) {
                vertical_velocity = (eta_current - eta_prev2) / (2 * timestep);
                vertical_acceleration = (eta_current - 2 * eta_prev + eta_prev2) / (timestep * timestep);
            } else if (has_prev) {
                vertical_velocity = (eta_current - eta_prev) / timestep;
            }

            callback(time, elevation, vertical_velocity, vertical_acceleration, x);
        }
    }
};
