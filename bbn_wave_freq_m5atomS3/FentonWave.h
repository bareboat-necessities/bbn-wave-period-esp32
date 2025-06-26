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
    // Fixed-size Eigen types parameterized by N
    using VectorF = Eigen::Matrix<float, N+1, 1>;  // N+1 elements (0 to N)
    using MatrixF = Eigen::Matrix<float, N+1, N+1>;
    using VectorJ = Eigen::Matrix<float, N, 1>;    // N elements (1 to N)

    float height;
    float depth;
    float length;
    float g;
    float relax;
    float eta_eps;
    
    // Wave parameters
    VectorF eta;
    VectorF x;
    float k;
    float c;
    float cs;
    float T;
    float omega;
    VectorF E;
    VectorF B;
    
public:
    FentonWave(float height, float depth, float length, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax),
          eta_eps(height / 1e5f) {
        auto data = fenton_coefficients(height, depth, length, g, relax);
        set_data(data);
    }

    float get_c() const { return c; }
    float get_k() const { return k; }
    float get_T() const { return T; }
    float get_omega() const { return omega; }

    void set_data(const std::map<std::string, VectorF>& data) {
        eta = data.at("eta");
        x = data.at("x");
        k = data.at("k")(0);
        c = data.at("c")(0);
        cs = c - data.at("Q")(0);
        T = length / c;
        omega = c * k;
        B = data.at("B");
        
        E.resize(N + 1);
        VectorF J = VectorF::LinSpaced(N + 1, 0, N);
        
        for (int j = 0; j <= N; j++) {
            E(j) = trapezoid_integration((eta.array() * (J(j) * J * M_PI / N).array().cos()).matrix());
        }
    }
    
    float stream_function(float x_val, float z_val, float t = 0, const std::string& frame = "b") const {
        VectorJ J = VectorJ::LinSpaced(N, 1, N);
        
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
    
    float surface_elevation(float x_val, float t = 0) const {
        VectorF J = VectorF::LinSpaced(N + 1, 0, N);
        
        float sum = 0;
        for (int j = 0; j <= N; j++) {
            float theta = J(j) * k * (x_val - c * t);
            sum += E(j) * std::cos(theta);
        }
        return 2 * sum / N;
    }
    
    float surface_slope(float x_val, float t = 0) const {
        VectorF J = VectorF::LinSpaced(N + 1, 0, N);
        
        float sum = 0;
        for (int j = 0; j <= N; j++) {
            sum += E(j) * J(j) * k * std::sin(J(j) * k * (x_val - c * t));
        }
        return -2 * sum / N;
    }
    
    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0, bool all_points_wet = false) const {
        VectorJ J = VectorJ::LinSpaced(N, 1, N);
        
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        
        for (int i = 0; i < N; i++) {
            float Jk = J(i) * k;
            float arg = Jk * (x_val - c * t);
            float term = B(i+1) * Jk / std::cosh(Jk * depth);
            
            vel(0) += term * std::cos(arg) * std::cosh(Jk * z_val);
            vel(1) += term * std::sin(arg) * std::sinh(Jk * z_val);
        }
        return vel;
    }
    
private:
    struct FentonCoefficients {
        VectorF B;
        VectorF eta;
        float Q;
        float R;
    };
    
    float trapezoid_integration(const VectorF& y) const {
        float sum = 0.5f * (y(0) + y(N));
        for (int i = 1; i < N; i++) {
            sum += y(i);
        }
        return sum;
    }
    
    FentonCoefficients initial_guess(float H, float c, float k, const VectorF& x) const {
        FentonCoefficients guess;
        guess.B = VectorF::Zero();
        guess.B(0) = c;
        guess.B(1) = -H / (4 * c * k);
        
        guess.eta = VectorF::Zero();
        for (int i = 0; i <= N; i++) {
            guess.eta(i) = 1 + (H/2) * std::cos(k * x(i));
        }
        
        guess.Q = c;
        guess.R = 1 + 0.5f * c * c;
        return guess;
    }
    
    std::map<std::string, VectorF> fenton_coefficients(
        float height, float depth, float length, float g, 
        float relax, int maxiter = 500, float tolerance = 1e-8) {
        
        // Non-dimensionalized input
        float H = height / depth;
        float lam = length / depth;
        float k = 2 * M_PI / lam;
        float c = std::sqrt(std::tanh(k) / k);
        float D = 1;
        constexpr int N_unknowns = 2 * (N + 1) + 2;
        
        VectorJ J = VectorJ::LinSpaced(N, 1, N);
        VectorF M = VectorF::LinSpaced(N + 1, 0, N);
        VectorF x = M * lam / (2 * N);
        
        // Initial guess
        FentonCoefficients guess = initial_guess(H, c, k, x);
        VectorF B = guess.B;
        float Q = guess.Q;
        float R = guess.R;
        VectorF eta = guess.eta;
        
        // Optimization
        Eigen::Matrix<float, N_unknowns, 1> coeffs;
        coeffs.template head<N + 1>() = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;
        
        Eigen::Matrix<float, N_unknowns, 1> f = func(coeffs, H, k, D, J, M);
        
        for (int it = 0; it < maxiter; it++) {
            Eigen::Matrix<float, N_unknowns, N_unknowns> jac = fprime(coeffs, H, k, D, J, M);
            Eigen::Matrix<float, N_unknowns, 1> delta = jac.fullPivLu().solve(-f);
            coeffs += delta * relax;
            f = func(coeffs, H, k, D, J, M);
            
            float error = f.array().abs().maxCoeff();
            float eta_max = coeffs.template segment<N + 1>(N + 1).maxCoeff();
            float eta_min = coeffs.template segment<N + 1>(N + 1).minCoeff();
            
            if (error < tolerance) {
                break;
            }
        }
        
        // Scale back to physical space
        B = coeffs.template head<N + 1>();
        eta = coeffs.template segment<N + 1>(N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
        
        B(0) *= std::sqrt(g * depth);
        B.template tail<N>() *= std::sqrt(g * depth * depth * depth);
        
        std::map<std::string, VectorF> result;
        result["x"] = x * depth;
        result["eta"] = eta * depth;
        result["B"] = B;
        result["Q"] = VectorF::Constant(1, Q * std::sqrt(g * depth * depth * depth));
        result["R"] = VectorF::Constant(1, R * g * depth);
        result["k"] = VectorF::Constant(1, k / depth);
        result["c"] = VectorF::Constant(1, B(0));
        
        return result;
    }
    
    Eigen::Matrix<float, 2*(N+1)+2, 1> func(
        const Eigen::Matrix<float, 2*(N+1)+2, 1>& coeffs, 
        float H, float k, float D, const VectorJ& J, const VectorF& M) const {
        
        Eigen::Matrix<float, 2*(N+1)+2, 1> f = Eigen::Matrix<float, 2*(N+1)+2, 1>::Zero();
        
        float B0 = coeffs(0);
        VectorF B = coeffs.template segment<N>(1);
        VectorF eta = coeffs.template segment<N + 1>(N + 1);
        float Q = coeffs(2 * N + 2);
        float R = coeffs(2 * N + 3);
        
        for (int m = 0; m <= N; m++) {
            VectorF Jk_eta = J * k * eta(m);
            VectorF Jk_D = J * k * D;
            VectorF S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorF S2 = (J * m * M_PI / N).array().sin();
            VectorF C2 = (J * m * M_PI / N).array().cos();
            
            float um = -B0 + k * (B.array() * C1.array() * C2.array() * J.array()).sum();
            float vm = k * (B.array() * S1.array() * S2.array() * J.array()).sum();
            
            f(m) = -B0 * eta(m) + (B.array() * S1.array() * C2.array()).sum() + Q;
            f(N + 1 + m) = (um * um + vm * vm) / 2 + eta(m) - R;
        }
        
        f(2 * N + 2) = trapezoid_integration(eta) / N - 1;
        f(2 * N + 3) = eta(0) - eta(N) - H;
        
        return f;
    }
    
    Eigen::Matrix<float, 2*(N+1)+2, 2*(N+1)+2> fprime(
        const Eigen::Matrix<float, 2*(N+1)+2, 1>& coeffs, 
        float H, float k, float D, const VectorJ& J, const VectorF& M) const {
        
        Eigen::Matrix<float, 2*(N+1)+2, 2*(N+1)+2> jac = 
            Eigen::Matrix<float, 2*(N+1)+2, 2*(N+1)+2>::Zero();
        
        float B0 = coeffs(0);
        VectorF B = coeffs.template segment<N>(1);
        VectorF eta = coeffs.template segment<N + 1>(N + 1);
        
        for (int m = 0; m <= N; m++) {
            VectorF Jk_eta = J * k * eta(m);
            VectorF Jk_D = J * k * D;
            VectorF S1 = Jk_eta.array().sinh() / Jk_D.array().cosh();
            VectorF C1 = Jk_eta.array().cosh() / Jk_D.array().cosh();
            VectorF S2 = (J * m * M_PI / N).array().sin();
            VectorF C2 = (J * m * M_PI / N).array().cos();
            
            VectorF SC = S1.array() * C2.array();
            VectorF SS = S1.array() * S2.array();
            VectorF CC = C1.array() * C2.array();
            VectorF CS = C1.array() * S2.array();
            
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

int FentonWave_test() {
    try {
        constexpr int WaveOrder = 5;  // Compile-time wave order
        const float wave_height = 2.0;
        const float water_depth = 10.0;
        const float wavelength = 50.0;

        const float simulation_duration = 20.0;
        const float timestep = 0.1;

        WaveSurfaceTracker<WaveOrder> tracker(wave_height, water_depth, wavelength);

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
