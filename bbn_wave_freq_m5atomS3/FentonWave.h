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

template<int N = 5>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using VectorF = Eigen::Matrix<float, N + 1, 1>;
    using BigVector = Eigen::Matrix<float, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<float, StateDim, StateDim>;

    float height, depth, length, g, relax;
    VectorF eta, B, E;
    float k, c, T, omega, Q, R;

public:
    FentonWave(float height, float depth, float length, float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(std::clamp(relax, 0.1f, 1.0f)) {
        
        if (length <= 0 || height <= 0)
            throw std::invalid_argument("Physical parameters must be positive");
        if (depth > 0 && height / depth > 0.78f)
            throw std::invalid_argument("Wave too steep (H/d > 0.78)");

        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    float surface_elevation(float x, float t = 0) const {
        float phase = k * (x - c * t);
        float eta_val = 0.0f;
        for (int j = 0; j <= N; ++j) {
            eta_val += 2.0f * E[j] * std::cos(j * phase) / N;
        }
        return eta_val - (depth < 0 ? 25.0f * length : depth);
    }

    Eigen::Vector2f velocity(float x, float z, float t = 0) const {
        if (depth < 0) throw std::runtime_error("Cannot compute velocity for infinite depth");
        
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
        VectorF B;
        VectorF eta;
        float Q, R;
        float k, c;
        float error;
        int niter;
    };

    void set_coefficients(const FentonCoefficients& coeffs) {
        float scale = (depth > 0) ? depth : 25.0f * length;
        
        // Scale to physical units
        B = coeffs.B * std::sqrt(g * scale);
        for (int i = 1; i <= N; ++i) {
            B[i] *= scale;  // Additional √(D²) factor for B[1..N]
        }
        
        eta = coeffs.eta * scale;
        Q = coeffs.Q * std::sqrt(g * scale * scale * scale);
        R = coeffs.R * g * scale;
        k = coeffs.k / scale;
        c = B[0];  // Already scaled
        T = length / c;
        omega = c * k;

        // Compute cosine series coefficients
        E.resize(N + 1);
        for (int j = 0; j <= N; ++j) {
            float sum = 0.5f * (eta[0] * std::cos(0) + eta[N] * std::cos(j * PI));
            for (int i = 1; i < N; ++i) {
                sum += eta[i] * std::cos(j * i * PI / N);
            }
            E[j] = sum / N;
        }
    }

    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        const float D = (depth > 0) ? depth : 25.0f * length;
        const float H = height / D;
        const float lambda = length / D;
        const float k_nd = 2.0f * PI / lambda;
        const float c_guess = std::sqrt(std::tanh(k_nd)/k_nd);

        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.eta.setZero();
        
        // Initial guess (Stokes first order)
        coeffs.B[0] = c_guess;
        coeffs.B[1] = -H / (4.0f * c_guess * k_nd);
        
        for (int i = 0; i <= N; ++i) {
            coeffs.eta[i] = 1.0f + H/2.0f * std::cos(k_nd * i * lambda/(2*N));
        }
        
        coeffs.Q = c_guess;
        coeffs.R = 1.0f + 0.5f * c_guess * c_guess;
        coeffs.k = k_nd;
        coeffs.c = c_guess;

        BigVector params;
        params.segment(0, N+1) = coeffs.B;
        params.segment(N+1, N+1) = coeffs.eta;
        params[2*(N+1)] = coeffs.Q;
        params[2*(N+1)+1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector res = compute_residuals(params, H, k_nd, D);
            if (res.norm() < tol) {
                coeffs.error = res.norm();
                coeffs.niter = iter + 1;
                break;
            }

            BigMatrix J = compute_jacobian(params, H, k_nd, D);
            Eigen::ColPivHouseholderQR<BigMatrix> solver(J);
            BigVector delta = solver.solve(-res);
            params += relax * delta;

            if (!params.allFinite()) {
                throw std::runtime_error("NaN/Inf detected during iteration");
            }
        }

        // Store final parameters
        coeffs.B = params.segment(0, N+1);
        coeffs.eta = params.segment(N+1, N+1);
        coeffs.Q = params[2*(N+1)];
        coeffs.R = params[2*(N+1)+1];
        
        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, float D) {
        VectorF B = params.segment(0, N+1);
        VectorF eta = params.segment(N+1, N+1);
        float Q = params[2*(N+1)];
        float R = params[2*(N+1)+1];
        
        BigVector res = BigVector::Zero();
        
        // Full phase sweep for accurate wave height calculation
        constexpr int M = 200;
        Eigen::VectorXf eta_full(M);
        Eigen::VectorXf x_full = Eigen::VectorXf::LinSpaced(M, 0, 2*PI*(1-1.0f/M));
        
        for (int m = 0; m < M; ++m) {
            float eta_val = eta[0];
            for (int j = 1; j <= N; ++j) {
                eta_val += eta[j] * std::cos(j * x_full[m]);
            }
            eta_full[m] = eta_val;
        }
        
        // Collocation point residuals
        for (int m = 0; m <= N; ++m) {
            float xm = m * PI / N;
            float S1 = 0, C1 = 0;
            
            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                float denom = std::cosh(kj * D);
                S1 += B[j] * std::sinh(kj*eta[m])/denom * std::cos(j*xm);
                C1 += B[j] * std::cosh(kj*eta[m])/denom * std::cos(j*xm);
            }
            
            float um = -B[0] + k * C1;
            float vm = k * S1;
            
            res[m] = -B[0]*eta[m] + S1 + Q;
            res[N+1+m] = 0.5f*(um*um + vm*vm) + eta[m] - R;
        }
        
        // Global constraints
        res[2*(N+1)] = eta_full.mean() - 1.0f;         // Mean elevation
        res[2*(N+1)+1] = eta_full.maxCoeff() - eta_full.minCoeff() - H; // Wave height
        
        return res;
    }

    BigMatrix compute_jacobian(const BigVector& params, float H, float k, float D) {
        BigMatrix J = BigMatrix::Zero();
        VectorF B = params.segment(0, N+1);
        VectorF eta = params.segment(N+1, N+1);
        
        for (int m = 0; m <= N; ++m) {
            float xm = m * PI / N;
            float S1 = 0, C1 = 0, dS1_deta = 0, dC1_deta = 0;
            
            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                float denom = std::cosh(kj * D);
                float cos_jxm = std::cos(j * xm);
                float sin_jxm = std::sin(j * xm);
                
                S1 += B[j] * std::sinh(kj*eta[m])/denom * cos_jxm;
                C1 += B[j] * std::cosh(kj*eta[m])/denom * cos_jxm;
                
                dS1_deta += B[j] * kj * std::cosh(kj*eta[m])/denom * cos_jxm;
                dC1_deta += B[j] * kj * std::sinh(kj*eta[m])/denom * cos_jxm;
            }
            
            float um = -B[0] + k * C1;
            float vm = k * S1;
            
            // Bernoulli residual derivatives
            J(m, 0) = -eta[m];  // dF/dB0
            
            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                J(m, j) = std::sinh(kj*eta[m])/std::cosh(kj*D) * std::cos(j*xm); // dF/dBj
                J(m, N+1+j) = -B[0] + B[j] * kj * std::cosh(kj*eta[m])/std::cosh(kj*D) * std::cos(j*xm); // dF/dηj
            }
            J(m, 2*(N+1)) = 1.0f;  // dF/dQ
            
            // Dynamic BC derivatives
            J(N+1+m, 0) = -um;  // dG/dB0
            
            for (int j = 1; j <= N; ++j) {
                float kj = j * k;
                float denom = std::cosh(kj * D);
                J(N+1+m, j) = k * (um * std::cosh(kj*eta[m])/denom * std::cos(j*xm) +
                                  vm * std::sinh(kj*eta[m])/denom * std::sin(j*xm));
                
                J(N+1+m, N+1+j) = 1.0f + 
                    um * k * B[j] * kj * std::sinh(kj*eta[m])/denom * std::cos(j*xm) +
                    vm * k * B[j] * kj * std::cosh(kj*eta[m])/denom * std::sin(j*xm);
            }
            J(N+1+m, 2*(N+1)+1) = -1.0f;  // dG/dR
        }
        
        // Mean elevation derivatives (trapezoidal rule)
        J(2*(N+1), N+1) = 0.5f/N;
        J(2*(N+1), 2*(N+1)-1) = 0.5f/N;
        for (int j = 1; j < N; ++j) {
            J(2*(N+1), N+1+j) = 1.0f/N;
        }
        
        // Wave height derivatives
        J(2*(N+1)+1, N+1) = 1.0f;    // dH/dη0
        J(2*(N+1)+1, 2*(N+1)-1) = -1.0f; // dH/dηN
        
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


