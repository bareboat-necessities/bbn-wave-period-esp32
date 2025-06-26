#pragma once

#include <ArduinoEigenDense.h>

#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <functional>

template<int N = 3>
class FentonWave {
private:
    // Type aliases for Eigen matrices/vectors
    using VectorF = Eigen::Matrix<float, N+1, 1>;   // Vector of size N+1
    using VectorJ = Eigen::Matrix<float, N, 1>;     // Vector of size N
    using MatrixF = Eigen::Matrix<float, N+1, N+1>; // Matrix of size (N+1)x(N+1)
    using BigVector = Eigen::VectorXf;              // Vector for Newton solver
    using BigMatrix = Eigen::MatrixXf;              // Jacobian for Newton solver

    // Wave parameters
    float height, depth, length, g, relax, eta_eps;

    // Wave state variables
    VectorF eta, x, B, E;
    float k, c, cs, T, omega;

public:
    // Constructor: initializes and solves the Fenton equations
    FentonWave(float height, float depth, float length,
               float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length),
          g(g), relax(relax), eta_eps(height / 1e5f) {
        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    // Compute surface elevation η(x, t)
    float surface_elevation(float x_val, float t = 0) const {
        VectorF J = VectorF::LinSpaced(N+1, 0, N);
        return (2.0f / N) * (E.array() * (J.array() * k * (x_val - c * t)).cos()).sum();
    }

    // Compute horizontal and vertical particle velocities (u, w)
    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        VectorJ J = VectorJ::LinSpaced(N, 1, N);
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        for (int i = 0; i < N; ++i) {
            float Jk = J[i] * k;
            float arg = Jk * (x_val - c * t);
            float term = B[i + 1] * Jk / std::cosh(Jk * depth);
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

    // Save coefficients and compute Fourier series E
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
        E.resize(N + 1);
        for (int j = 0; j <= N; ++j) {
            E[j] = trapezoid_integration(
                eta.array() * (j * M_PI * x.array() / length).cos()
            );
        }
    }

    // Main Newton-Raphson solver for Fenton wave equations
    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        k = 2 * M_PI / lambda;
        float c_guess = std::sqrt(g * depth * std::tanh(k) / k);
        float D = 1.0f;

        VectorF J_vec = VectorF::LinSpaced(N + 1, 0, N);
        VectorF M_vec = VectorF::LinSpaced(N + 1, 0, N);
        VectorF x = (M_vec.array() * lambda / (2 * N)).matrix(); // Uniform spatial grid

        // Initial guess using Stokes 1st order theory
        FentonCoefficients coeffs;
        coeffs.B = VectorF::Zero();
        coeffs.B[0] = c_guess;
        if (N >= 1) coeffs.B[1] = -H / (2 * k * std::cosh(k * depth));
        coeffs.eta = (H / 2) * (k * x.array()).cos().matrix();
        coeffs.Q = 0;
        coeffs.R = 1 + 0.5f * c_guess * c_guess;

        // Pack into Newton solver vector
        BigVector params(2 * (N + 1) + 2);
        params << coeffs.B, coeffs.eta, coeffs.Q, coeffs.R;

        // Newton-Raphson loop
        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector f = compute_residuals(params, H, k, D, J_vec, M_vec);
            if (f.norm() < tol) break;
            BigMatrix J_mat = compute_jacobian(params, H, k, D, J_vec, M_vec);
            params -= relax * J_mat.colPivHouseholderQr().solve(f);
        }

        // Unpack solution
        coeffs.B = params.head(N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q = params[2 * (N + 1)];
        coeffs.R = params[2 * (N + 1) + 1];
        coeffs.x = x;
        scale_to_physical(coeffs);
        return coeffs;
    }

    // Evaluate residuals of Fenton equations and constraints
    BigVector compute_residuals(const BigVector& params, float H, float k, float D,
                                const VectorF& J_vec, const VectorF& M_vec) {
        BigVector res = BigVector::Zero(2 * (N + 1) + 2);
        VectorF B = params.head(N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        for (int m = 0; m <= N; ++m) {
            // Precompute terms for stream function and Bernoulli
            VectorF Jk_eta = J_vec * k * eta[m];
            VectorF Jk_D = J_vec * k * D;
            VectorF S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorF S2 = (J_vec * m * M_PI / N).array().sin();
            VectorF C2 = (J_vec * m * M_PI / N).array().cos();

            float um = -B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J_vec[j];
                vm += k * B[j] * S1[j] * S2[j] * J_vec[j];
            }

            res[m] = -B[0] * eta[m] + Q;
            for (int j = 1; j <= N; ++j) {
                res[m] += B[j] * S1[j] * C2[j];
            }

            res[N + 1 + m] = 0.5f * (um * um + vm * vm) + eta[m] - R;
        }

        // Mean level constraint
        res[2 * (N + 1)] = trapezoid_integration(eta) / N - 1.0f;

        // Wave height constraint (using global max and min)
        float eta_max = eta.maxCoeff();
        float eta_min = eta.minCoeff();
        res[2 * (N + 1) + 1] = eta_max - eta_min - H;

        return res;
    }

    // Compute Jacobian matrix for Newton solver
    BigMatrix compute_jacobian(const BigVector& params, float H, float k, float D,
                               const VectorF& J, const VectorF& M) {
        const int total_size = 2 * (N + 1) + 2;
        BigMatrix jac = BigMatrix::Zero(total_size, total_size);

        VectorF B = params.head(N + 1);
        VectorF eta = params.segment(N + 1, N + 1);
        float Q = params[2 * (N + 1)];
        float R = params[2 * (N + 1) + 1];

        for (int m = 0; m <= N; ++m) {
            VectorF Jk_eta = J * k * eta[m];
            VectorF Jk_D = J * k * D;
            VectorF S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorF S2 = (J * m * M_PI / N).array().sin();
            VectorF C2 = (J * m * M_PI / N).array().cos();

            VectorF dS1_deta = (J * k).array() * C1.array();
            VectorF dC1_deta = (J * k).array() * S1.array();

            float um = -B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }

            jac(m, 0) = -eta[m];
            for (int j = 1; j <= N; ++j) {
                jac(m, j) = S1[j] * C2[j];
            }

            float dS1_sum = 0;
            for (int j = 1; j <= N; ++j) {
                dS1_sum += B[j] * dS1_deta[j] * C2[j];
            }
            jac(m, N + 1 + m) = dS1_sum;
            jac(m, 2 * (N + 1)) = 1.0f;

            jac(N + 1 + m, 0) = -um;
            for (int j = 1; j <= N; ++j) {
                float term1 = um * k * C1[j] * C2[j] * J[j];
                float term2 = vm * k * S1[j] * S2[j] * J[j];
                jac(N + 1 + m, j) = term1 + term2;
            }

            float sum1 = 0, sum2 = 0;
            for (int j = 1; j <= N; ++j) {
                sum1 += B[j] * dC1_deta[j] * C2[j] * J[j];
                sum2 += B[j] * dS1_deta[j] * S2[j] * J[j];
            }
            jac(N + 1 + m, N + 1 + m) = um * k * sum1 + vm * k * sum2 + 1.0f;
            jac(N + 1 + m, 2 * (N + 1) + 1) = -1.0f;
        }

        // Mean elevation constraint: ∫η = 0
        for (int j = 0; j <= N; ++j) {
            float weight = (j == 0 || j == N) ? 0.5f : 1.0f;
            jac(2 * (N + 1), N + 1 + j) = weight / N;
        }

        // Wave height constraint: d/dη[max - min]
        int max_idx = 0, min_idx = 0;
        eta.maxCoeff(&max_idx);
        eta.minCoeff(&min_idx);
        jac(2 * (N + 1) + 1, N + 1 + max_idx) = 1.0f;
        jac(2 * (N + 1) + 1, N + 1 + min_idx) = -1.0f;

        return jac;
    }

    // Scale from non-dimensional to dimensional units
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

    // Trapezoidal integration over fixed grid
    float trapezoid_integration(const VectorF& y) const {
        float dx = length / N;
        float sum = 0.5f * (y[0] + y[N]);
        for (int i = 1; i < N; ++i) sum += y[i];
        return sum * dx;
    }
};

template<int N = 3>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;
    float phase_velocity;
    float wave_number;
    
    float eta_prev2 = 0;
    float eta_prev = 0;
    float eta_current = 0;
    bool has_prev = false;
    bool has_prev2 = false;

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
            float horizontal_position)> kinematics_callback) {
        
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

            float vertical_velocity = 0;
            float vertical_acceleration = 0;
            
            if (has_prev2) {
                vertical_velocity = (eta_current - eta_prev2) / (2*timestep);
                vertical_acceleration = (eta_current - 2*eta_prev + eta_prev2) / (timestep*timestep);
            } else if (has_prev) {
                vertical_velocity = (eta_current - eta_prev) / timestep;
            }

            kinematics_callback(time, elevation, vertical_velocity, 
                              vertical_acceleration, x);
        }
    }
};


