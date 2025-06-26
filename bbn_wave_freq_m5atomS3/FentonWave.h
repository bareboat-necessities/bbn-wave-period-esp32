#pragma once

/*
  AI assisted translation into C++ of https://github.com/TormodLandet/raschii/blob/master/raschii/fenton.py
*/

#include <ArduinoEigenDense.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <functional>

using namespace Eigen;

class FentonWave {
private:

    using VectorXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::ColMajor, 6, 1>;
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 6, 6>;

    float height;
    float depth;
    float length;
    int order;
    float g;
    float relax;
    float eta_eps;
    
    // Wave parameters
    VectorXf eta;
    VectorXf x;
    float k;
    float c;
    float cs;
    float T;
    float omega;
    VectorXf E;
    VectorXf B;
    
public:
    FentonWave(float height, float depth, float length, int N, float g = 9.81, float relax = 0.5)
        : height(height), depth(depth), length(length), order(N), g(g), relax(relax),
          eta_eps(height / 1e5)  // Initialize in member initializer list
    {
        // Compute coefficients
        auto data = fenton_coefficients(height, depth, length, N, g, relax);
        set_data(data);
    }

    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }

    void set_data(const std::map<std::string, VectorXf>& data) {
        eta = data.at("eta");
        x = data.at("x");
        k = data.at("k")(0);  // Use () instead of [] for Eigen vector access
        c = data.at("c")(0);
        cs = c - data.at("Q")(0);
        T = length / c;
        omega = c * k;
        B = data.at("B");
        
        // Compute cosine series coefficients for elevation
        int N = eta.size() - 1;
        E.resize(N + 1);
        VectorXf J = VectorXf::LinSpaced(N + 1, 0, N);
        
        for (int j = 0; j <= N; j++) {
            E(j) = trapezoid_integration(eta.array() * (J(j) * J * M_PI / N).array().cos()).matrix();
        }
    }
    
    float stream_function(float x_val, float z_val, float t = 0, const std::string& frame = "b") {
        int N = eta.size() - 1;
        VectorXf J = VectorXf::LinSpaced(N, 1, N);
        
        float x2 = x_val - c * t;
        float psi = 0;
        
        for (int i = 0; i < N; i++) {
            float Jk = J(i) * k;
            psi += B(i+1) * std::sinh(Jk * z_val) / std::cosh(Jk * depth) * std::cos(Jk * x2);
        }
        
        if (frame == "b") {
            return B(0) * z_val + psi;
        } else if (frame == "c") {
            return psi;
        }
        return 0;
    }
    
    float surface_elevation(float x_val, float t = 0) {
        int N = E.size() - 1;
        VectorXf J = VectorXf::LinSpaced(N + 1, 0, N);
        
        float sum = 0;
        for (int j = 0; j <= N; j++) {
            sum += E(j) * std::cos(J(j) * k * (x_val - c * t));
        }
        return 2 * sum / N;
    }
    
    float surface_slope(float x_val, float t = 0) {
        int N = E.size() - 1;
        VectorXf J = VectorXf::LinSpaced(N + 1, 0, N);
        
        float sum = 0;
        for (int j = 0; j <= N; j++) {
            sum += E(j) * J(j) * k * std::sin(J(j) * k * (x_val - c * t));
        }
        return -2 * sum / N;
    }
    
    Vector2f velocity(float x_val, float z_val, float t = 0, bool all_points_wet = false) {
        int N = eta.size() - 1;
        VectorXf J = VectorXf::LinSpaced(N, 1, N);
        
        Vector2f vel = Vector2f::Zero();
        
        for (int i = 0; i < N; i++) {
            float Jk = J(i) * k;
            float term = B(i+1) * Jk / std::cosh(Jk * depth);
            
            vel(0) += term * std::cos(Jk * (x_val - c * t)) * std::cosh(Jk * z_val);
            vel(1) += term * std::sin(Jk * (x_val - c * t)) * std::sinh(Jk * z_val);
        }
        
        // Air blending would go here if implemented
        return vel;
    }
    
private:
    struct FentonCoefficients {
        VectorXf B;
        VectorXf eta;
        float Q;
        float R;
    };
    
    float trapezoid_integration(const VectorXf& y) {
        int n = y.size();
        float sum = 0.5 * (y(0) + y(n-1));
        for (int i = 1; i < n-1; i++) {
            sum += y(i);
        }
        return sum;
    }
    
    FentonCoefficients initial_guess(float H, int N, float c, float k, const VectorXf& x) {
        FentonCoefficients guess;
        guess.B = VectorXf::Zero(N + 1);
        guess.B(0) = c;
        guess.B(1) = -H / (4 * c * k);
        guess.eta = VectorXf::Ones(x.size()) + (H / 2) * (k * x.array()).cos().matrix();
        guess.Q = c;
        guess.R = 1 + 0.5 * c * c;
        return guess;
    }
    
    std::map<std::string, VectorXf> fenton_coefficients(
        float height, float depth, float length, int N, float g, 
        float relax, int maxiter = 500, float tolerance = 1e-8) {
        
        // Non-dimensionalized input
        float H = height / depth;
        float lam = length / depth;
        float k = 2 * M_PI / lam;
        float c = std::sqrt(std::tanh(k) / k);
        float D = 1;
        int N_unknowns = 2 * (N + 1) + 2;
        
        VectorXf J = VectorXf::LinSpaced(N, 1, N);
        VectorXf M = VectorXf::LinSpaced(N + 1, 0, N);
        VectorXf x = M * lam / (2 * N);
        
        // Initial guess - using struct instead of tuple
        FentonCoefficients guess = initial_guess(H, N, c, k, x);
        VectorXf B = guess.B;
        float Q = guess.Q;
        float R = guess.R;
        VectorXf eta = guess.eta;
        
        // Optimization
        VectorXf coeffs(N_unknowns);
        coeffs.head(N + 1) = B;
        coeffs.segment(N + 1, N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;
        
        VectorXf f = func(coeffs, H, k, D, J, M);
        
        for (int it = 0; it < maxiter; it++) {
            MatrixXf jac = fprime(coeffs, H, k, D, J, M);
            VectorXf delta = jac.fullPivLu().solve(-f);
            coeffs += delta * relax;
            f = func(coeffs, H, k, D, J, M);
            
            float error = f.array().abs().maxCoeff();
            float eta_max = coeffs.segment(N + 1, N + 1).maxCoeff();
            float eta_min = coeffs.segment(N + 1, N + 1).minCoeff();
            
            if (error < tolerance) {
                break;
            }
        }
        
        // Scale back to physical space
        B = coeffs.head(N + 1);
        eta = coeffs.segment(N + 1, N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
        
        B(0) *= std::sqrt(g * depth);
        B.tail(N) *= std::sqrt(g * depth * depth * depth);
        
        std::map<std::string, VectorXf> result;
        result["x"] = x * depth;
        result["eta"] = eta * depth;
        result["B"] = B;
        result["Q"] = VectorXf::Constant(1, Q * std::sqrt(g * depth * depth * depth));
        result["R"] = VectorXf::Constant(1, R * g * depth);
        result["k"] = VectorXf::Constant(1, k / depth);
        result["c"] = VectorXf::Constant(1, B(0));
        
        return result;
    }
    
    VectorXf func(const VectorXf& coeffs, float H, float k, float D, const VectorXf& J, const VectorXf& M) {
        int N_unknowns = coeffs.size();
        int N = J.size();
        
        float B0 = coeffs(0);
        VectorXf B = coeffs.segment(1, N);
        VectorXf eta = coeffs.segment(N + 1, N + 1);
        float Q = coeffs(2 * N + 2);
        float R = coeffs(2 * N + 3);
        
        VectorXf f = VectorXf::Zero(N_unknowns);
        
        for (int m = 0; m <= N; m++) {
            VectorXf Jk_eta = (J * k * eta(m)).eval();
            VectorXf Jk_D = (J * k * D).eval();
            VectorXf S1 = Jk_eta.array().sinh() / Jk_D.array().cosh().matrix();
            VectorXf C1 = Jk_eta.array().cosh() / Jk_D.array().cosh().matrix();
            VectorXf S2 = (J * m * M_PI / N).array().sin();
            VectorXf C2 = (J * m * M_PI / N).array().cos();
            
            float um = -B0 + k * (B.array() * C1.array() * C2.array() * J.array()).sum();
            float vm = k * (B.array() * S1.array() * S2.array() * J.array()).sum();
            
            f(m) = -B0 * eta(m) + (B.array() * S1.array() * C2.array()).sum() + Q;
            f(N + 1 + m) = (um * um + vm * vm) / 2 + eta(m) - R;
        }
        
        f(2 * N + 2) = trapezoid_integration(eta) / N - 1;
        f(2 * N + 3) = eta(0) - eta(N) - H;
        
        return f;
    }
    
    MatrixXf fprime(const VectorXf& coeffs, float H, float k, float D, const VectorXf& J, const VectorXf& M) {
        int N_unknowns = coeffs.size();
        int N = J.size();
        
        MatrixXf jac = MatrixXf::Zero(N_unknowns, N_unknowns);
        float B0 = coeffs(0);
        VectorXf B = coeffs.segment(1, N);
        VectorXf eta = coeffs.segment(N + 1, N + 1);
        
        for (int m = 0; m <= N; m++) {
            VectorXf Jk_eta = J * k * eta(m);
            VectorXf Jk_D = J * k * D;
            VectorXf S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorXf C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorXf S2 = (J * m * M_PI / N).array().sin();
            VectorXf C2 = (J * m * M_PI / N).array().cos();
            
            VectorXf SC = S1.array() * C2.array();
            VectorXf SS = S1.array() * S2.array();
            VectorXf CC = C1.array() * C2.array();
            VectorXf CS = C1.array() * S2.array();
            
            float um = -B0 + k * (B.array() * CC.array() * J.array()).sum();
            float vm = k * (B.array() * SS.array() * J.array()).sum();
            
            jac(m, N + 1 + m) = -B0 + (B.array() * k * J.array() * C1.array() * C2.array()).sum();
            jac(m, 0) = -eta(m);
            for (int j = 1; j <= N; j++) {
                jac(m, j) = SC(j-1);
            }
            jac(m, 2 * N + 2) = 1;
            
            jac(N + 1 + m, N + 1 + m) = 1 + (um * k * k * (B.array() * J.array().square() * SC.array()).sum() + 
                                             vm * k * k * (B.array() * J.array().square() * CS.array()).sum());
            jac(N + 1 + m, 2 * N + 3) = -1;
            jac(N + 1 + m, 0) = -um;
            for (int j = 1; j <= N; j++) {
                jac(N + 1 + m, j) = k * um * J(j-1) * CC(j-1) + k * vm * J(j-1) * SS(j-1);
            }
        }
        
        for (int j = 0; j <= N; j++) {
            jac(2 * N + 2, N + 1 + j) = (j == 0 || j == N) ? 1.0/(2*N) : 1.0/N;
        }
        
        jac(2 * N + 3, N + 1) = 1;
        jac(2 * N + 3, 2 * N + 1) = -1;
        
        return jac;
    }
};

class WaveSurfaceTracker {
private:
    FentonWave wave;
    float phase_velocity;  // More descriptive than 'c'
    float wave_number;     // More descriptive than 'k'
    
    // Finite difference stencil
    float eta_prev2 = 0;   // η at t-2Δt
    float eta_prev = 0;    // η at t-Δt
    float eta_current = 0; // η at t
    bool has_prev = false;
    bool has_prev2 = false;

public:
    WaveSurfaceTracker(float height, float depth, float length, int order)
        : wave(height, depth, length, order), 
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
        
        // Reset tracking state
        eta_prev2 = eta_prev = eta_current = 0;
        has_prev = has_prev2 = false;

        for (float time = 0; time <= duration; time += timestep) {
            float x = phase_velocity * time;
            float elevation = wave.surface_elevation(x, time);
            
            // Update stencil
            eta_prev2 = eta_prev;
            eta_prev = eta_current;
            eta_current = elevation;
            
            // Update state tracking
            has_prev2 = has_prev;
            has_prev = true;

            // Calculate derivatives
            float vertical_velocity = 0;
            float vertical_acceleration = 0;
            
            if (has_prev2) {  // Central differences
                vertical_velocity = (eta_current - eta_prev2) / (2*timestep);
                vertical_acceleration = (eta_current - 2*eta_prev + eta_prev2) / (timestep*timestep);
            } else if (has_prev) {  // Forward difference
                vertical_velocity = (eta_current - eta_prev) / timestep;
            }

            kinematics_callback(time, elevation, vertical_velocity, 
                              vertical_acceleration, x);
        }
    }
};

int FentonWave_test() {
    try {
        // Wave parameters
        const float wave_height = 2.0;
        const float water_depth = 10.0;
        const float wavelength = 50.0;
        const int approximation_order = 3;

        // Simulation parameters
        const float simulation_duration = 20.0;
        const float timestep = 0.1;

        WaveSurfaceTracker tracker(wave_height, water_depth, 
                                 wavelength, approximation_order);

        // CSV output handler
        auto csv_handler = [](float time, float elevation, 
                            float velocity, float acceleration, float x) {
            static std::ofstream outfile("wave_kinematics.csv");
            static bool header_written = false;
            
            if (!header_written) {
                outfile << "Time(s),Elevation(m),Velocity(m/s),Acceleration(m/s²),X(m)\n";
                header_written = true;
            }
            
            outfile << time << "," << elevation << "," 
                   << velocity << "," << acceleration << "," << x << "\n";
        };

        tracker.track_lagrangian_kinematics(simulation_duration, timestep, csv_handler);

        std::cout << "Lagrangian wave kinematics tracking complete.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Simulation error: " << e.what() << "\n";
        return 1;
    }
}

