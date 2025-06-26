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

template<int N>
class FentonWave {
private:
    // Type aliases for Eigen matrices/vectors
    using VectorF = Eigen::Matrix<float, N+1, 1>;  // Vector of size N+1
    using VectorJ = Eigen::Matrix<float, N, 1>;    // Vector of size N
    using MatrixF = Eigen::Matrix<float, N+1, N+1>; // Matrix of size (N+1)x(N+1)
    using BigVector = Eigen::Matrix<float, 2*(N+1)+2, 1>; // For Newton solver
    using BigMatrix = Eigen::Matrix<float, 2*(N+1)+2, 2*(N+1)+2>; // Jacobian

    // Wave parameters
    float height, depth, length, g, relax, eta_eps;
    
    // Wave state variables
    VectorF eta, x, B, E;
    float k, c, cs, T, omega;

public:
    FentonWave(float height, float depth, float length, 
              float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), 
          g(g), relax(relax), eta_eps(height / 1e5f) {
        static_assert(N >= 1, "Wave order N must be at least 1");
        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    // Wave kinematics functions
    float surface_elevation(float x_val, float t = 0) const {
        VectorF J = VectorF::LinSpaced(N+1, 0, N);
        return (2.0f/N) * (E.array() * (J.array() * k * (x_val - c * t)).cos()).sum();
    }

    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        VectorJ J = VectorJ::LinSpaced(N, 1, N);
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        
        for (int i = 0; i < N; ++i) {
            float Jk = J[i] * k;
            float arg = Jk * (x_val - c * t);
            float term = B[i+1] * Jk / std::cosh(Jk * depth);
            vel[0] += term * std::cos(arg) * std::cosh(Jk * z_val);
            vel[1] += term * std::sin(arg) * std::sinh(Jk * z_val);
        }
        return vel;
    }

    // Getters for wave parameters
    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }

private:
    struct FentonCoefficients {
        VectorF x, eta, B;
        float k, c, Q, R;
    };

    void set_coefficients(const FentonCoefficients& coeffs) {
        eta = coeffs.eta;
        x = coeffs.x;
        k = coeffs.k;
        c = coeffs.c;
        cs = c - coeffs.Q;
        T = length / c;
        omega = c * k;
        B = coeffs.B;

        // Compute Fourier coefficients E
        E.resize(N+1);
        VectorF J = VectorF::LinSpaced(N+1, 0, N);
        for (int j = 0; j <= N; ++j) {
            E[j] = trapezoid_integration(
                (eta.array() * (J[j] * J.array() * M_PI / N).cos()).matrix()
            );
        }
    }
    
    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        // Non-dimensionalization
        float H = height / depth;
        float lambda = length / depth;
        k = 2 * M_PI / lambda;
        float c_guess = std::sqrt(g * depth * std::tanh(k) / k);  // Added g factor
        float D = 1.0f; // Non-dimensional depth
    
        // Coordinate setup
        VectorF J = VectorF::LinSpaced(N+1, 0, N);
        VectorF M = VectorF::LinSpaced(N+1, 0, N);
        VectorF x = (M * lambda / (2 * N)).array();  // Ensure array operation
    
        // Initial guess (Stokes 1st order solution)
        FentonCoefficients coeffs;
        coeffs.B = VectorF::Zero();
        coeffs.B[0] = c_guess;
        if (N >= 1) coeffs.B[1] = -H / (2 * k);  // Fixed coefficient
    
        coeffs.eta = VectorF::Ones();  // Initialize to 1
        coeffs.eta.array() += (H/2) * (k * x.array()).cos();  // Array operation
    
        coeffs.Q = 0;  // Initial guess for Q
        coeffs.R = 1 + 0.5f * c_guess * c_guess;
    
        // Newton-Raphson solver
        BigVector params(2*(N+1)+2);
        params << coeffs.B, coeffs.eta, coeffs.Q, coeffs.R;
    
        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector f = compute_residuals(params, H, k, D, J, M);
            if (f.norm() < tol) break;
    
            BigMatrix J = compute_jacobian(params, H, k, D, J, M);
            params -= relax * J.fullPivLu().solve(f);
        }
    
        // Extract solution
        coeffs.B = params.head(N+1);
        coeffs.eta = params.segment(N+1, N+1);
        coeffs.Q = params[2*(N+1)];
        coeffs.R = params[2*(N+1)+1];
    
        // Scale back to physical units
        scale_to_physical(coeffs);
        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, float D, 
                               const VectorF& J, const VectorF& M) {
        BigVector res;
        res.setZero();
    
        VectorF B = params.template head<N+1>();
        VectorF eta = params.template segment<N+1>(N+1);
        float Q = params[2*(N+1)];
        float R = params[2*(N+1)+1];
    
        // Residual equations (Fenton's equations 14a-b)
        for (int m = 0; m <= N; ++m) {
            // Precompute trigonometric terms - use only first N elements for B terms
            VectorF Jk_eta = J * k * eta[m];
            VectorF Jk_D = J * k * D;
            VectorF S1 = Jk_eta.array().sinh().array() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh().array() / Jk_D.array().cosh();
            VectorF S2 = (J * m * M_PI / N).array().sin();
            VectorF C2 = (J * m * M_PI / N).array().cos();
    
            // Velocity components - note B[0] is handled separately
            float um = -B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }
    
            // Stream function residual (Eq. 14a)
            res[m] = -B[0] * eta[m] + Q;
            for (int j = 1; j <= N; ++j) {
                res[m] += B[j] * S1[j] * C2[j];
            }
    
            // Bernoulli residual (Eq. 14b)
            res[N+1+m] = 0.5f * (um*um + vm*vm) + eta[m] - R;
        }
    
        // Constraints
        res[2*(N+1)] = trapezoid_integration(eta)/N - 1.0f;  // Mean water level
        res[2*(N+1)+1] = eta[0] - eta[N] - H;               // Wave height
    
        return res;
    }
    
    BigMatrix compute_jacobian(const BigVector& params, float H, float k, float D,
                              const VectorF& J, const VectorF& M) {
        BigMatrix jac;
        jac.setZero();
    
        VectorF B = params.template head<N+1>();
        VectorF eta = params.template segment<N+1>(N+1);
        float Q = params[2*(N+1)];
        float R = params[2*(N+1)+1];
    
        for (int m = 0; m <= N; ++m) {
            // Precompute trigonometric terms
            VectorF Jk_eta = J * k * eta[m];
            VectorF Jk_D = J * k * D;
            VectorF S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorF S2 = (J * m * M_PI / N).array().sin();
            VectorF C2 = (J * m * M_PI / N).array().cos();
    
            // Derivatives of S1 and C1
            VectorF dS1_deta = (J * k).array() * C1.array();
            VectorF dC1_deta = (J * k).array() * S1.array();
    
            // Velocity components
            float um = -B[0];
            float vm = 0;
            for (int j = 1; j <= N; ++j) {
                um += k * B[j] * C1[j] * C2[j] * J[j];
                vm += k * B[j] * S1[j] * S2[j] * J[j];
            }
    
            // --- df1/dB --- (Stream function derivatives)
            jac(m, 0) = -eta[m];  // df1/dB0
            
            // df1/dBj (j=1..N)
            for (int j = 1; j <= N; ++j) {
                jac(m, j) = S1[j] * C2[j];
            }
    
            // df1/deta_m
            float dS1_sum = 0;
            for (int j = 1; j <= N; ++j) {
                dS1_sum += B[j] * dS1_deta[j] * C2[j];
            }
            jac(m, N+1+m) = dS1_sum;
    
            // df1/dQ
            jac(m, 2*(N+1)) = 1.0f;
    
            // --- df2/dB --- (Bernoulli derivatives)
            jac(N+1+m, 0) = -um;  // df2/dB0
            
            // df2/dBj (j=1..N)
            for (int j = 1; j <= N; ++j) {
                float term1 = um * k * C1[j] * C2[j] * J[j];
                float term2 = vm * k * S1[j] * S2[j] * J[j];
                jac(N+1+m, j) = term1 + term2;
            }
    
            // df2/deta_m
            float sum1 = 0, sum2 = 0;
            for (int j = 1; j <= N; ++j) {
                sum1 += B[j] * dC1_deta[j] * C2[j] * J[j];
                sum2 += B[j] * dS1_deta[j] * S2[j] * J[j];
            }
            jac(N+1+m, N+1+m) = um * k * sum1 + vm * k * sum2 + 1.0f;
    
            // df2/dR
            jac(N+1+m, 2*(N+1)+1) = -1.0f;
        }
    
        // Constraints derivatives
        // df3/deta (mean water level)
        for (int j = 0; j <= N; ++j) {
            jac(2*(N+1), N+1+j) = (j == 0 || j == N) ? 0.5f/N : 1.0f/N;
        }
    
        // df4/deta (wave height)
        jac(2*(N+1)+1, N+1) = 1.0f;    // eta[0]
        jac(2*(N+1)+1, 2*N+1) = -1.0f; // eta[N]
    
        return jac;
    }

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

    float trapezoid_integration(const VectorF& y) const {
        float sum = 0.5f * (y[0] + y[N]);
        for (int i = 1; i < N; ++i) sum += y[i];
        return sum;
    }
};

template<int N>
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

void FentonWave_test() {
    // Wave parameters
    const float height = 2.0f;   // Wave height (m)
    const float depth = 10.0f;   // Water depth (m)
    const float length = 50.0f;  // Wavelength (m)
    
    // Simulation parameters
    const float duration = 20.0f; // Simulation duration (s)
    const float dt = 0.1f;       // Time step (s)
    
    // Create 3rd-order wave model
    FentonWave<3> wave(height, depth, length);
    
    // Output file
    std::ofstream out("wave_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²)\n";
    
    // Previous values for finite differences
    float prev_d = 0, prev_prev_d = 0;
    
    for (float t = 0; t <= duration; t += dt) {
        // Current surface elevation at x=0 (floating object position)
        float d = wave.surface_elevation(0, t);
        
        // Calculate velocity and acceleration
        float v = 0, a = 0;
        if (t >= 2*dt) {  // Wait until we have enough history
            v = (d - prev_prev_d) / (2*dt);          // Central difference
            a = (d - 2*prev_d + prev_prev_d) / (dt*dt); // 2nd derivative
        } else if (t >= dt) {
            v = (d - prev_d) / dt;  // Forward difference
        }
        
        // Write to file
        out << t << "," << d << "," << v << "," << a << "\n";
        
        // Update previous values
        prev_prev_d = prev_d;
        prev_d = d;
    }
    
    std::cout << "Wave data saved to wave_data.csv\n";
    std::cout << "Wave period: " << wave.get_T() << "s\n";
    std::cout << "Phase speed: " << wave.get_c() << "m/s\n";
}      
