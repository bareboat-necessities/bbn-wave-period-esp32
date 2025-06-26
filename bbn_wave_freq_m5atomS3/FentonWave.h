#pragma once

// Eigen configuration must come before any Eigen includes
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MAX_ALIGN_BYTES 0

#include <ArduinoEigenDense.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <functional>

class FentonWave {
private:
    // Simplified Eigen types
    using VectorXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

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
    FentonWave(float height, float depth, float length, int N, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), order(N), g(g), relax(relax),
          eta_eps(height / 1e5f) {
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
        k = data.at("k")(0);
        c = data.at("c")(0);
        cs = c - data.at("Q")(0);
        T = length / c;
        omega = c * k;
        B = data.at("B");
        
        int N = eta.size() - 1;
        E.resize(N + 1);
        VectorXf J = VectorXf::LinSpaced(N + 1, 0, N);
        
        for (int j = 0; j <= N; j++) {
            E(j) = trapezoid_integration((eta.array() * (J(j) * J * M_PI / N).array().cos()).matrix());
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
            // Explicit conversion to scalar operation
            float theta = J(j) * k * (x_val - c * t);
            sum += E(j) * std::cos(theta);
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
    
    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0, bool all_points_wet = false) {
        int N = eta.size() - 1;
        VectorF J = VectorF::LinSpaced(N, 1, N);
        
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        
        for (int i = 0; i < N; i++) {
            // Breaking down operations into scalar steps
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
        VectorXf B;
        VectorXf eta;
        float Q;
        float R;
    };
    
    float trapezoid_integration(const VectorF& y) {
        // Convert array operations to explicit scalar form
        int n = y.size();
        if (n == 0) return 0;
        
        float sum = 0.5f * (y(0) + y(n-1));
        for (int i = 1; i < n-1; i++) {
            sum += y(i);
        }
        return sum;
    }
    
    FentonCoefficients initial_guess(float H, int N, float c, float k, const VectorF& x) {
        FentonCoefficients guess;
        guess.B = VectorF::Zero(N + 1);
        guess.B(0) = c;
        guess.B(1) = -H / (4 * c * k);
        
        // Explicit element-wise cos operation
        guess.eta = VectorF::Zero(x.size());
        for (int i = 0; i < x.size(); i++) {
            guess.eta(i) = 1 + (H/2) * std::cos(k * x(i));
        }
        
        guess.Q = c;
        guess.R = 1 + 0.5f * c * c;
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
