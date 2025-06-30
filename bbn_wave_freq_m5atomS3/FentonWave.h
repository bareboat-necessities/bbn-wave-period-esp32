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

template <typename T>
T sinh_by_cosh(T a, T b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        throw std::runtime_error("sinh_by_cosh received non-finite input");
    }

    constexpr T MAX_EXP_ARG = 80.0f;  // Adjustable, avoid exp overflow

    // Clamp to avoid overflow
    a = clamp_value(a, -MAX_EXP_ARG, MAX_EXP_ARG);
    b = clamp_value(b, -MAX_EXP_ARG, MAX_EXP_ARG);

    if (a == 0) return 0;
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }

    T result = std::sinh(a) / std::cosh(b);
    if (!std::isfinite(result)) {
        throw std::runtime_error("sinh_by_cosh produced non-finite result");
    }
    return result;
}

template <typename T>
T cosh_by_cosh(T a, T b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        throw std::runtime_error("cosh_by_cosh received non-finite input");
    }

    constexpr T MAX_EXP_ARG = 80.0f;

    a = clamp_value(a, -MAX_EXP_ARG, MAX_EXP_ARG);
    b = clamp_value(b, -MAX_EXP_ARG, MAX_EXP_ARG);

    if (a == 0) return 1.0 / std::cosh(b);
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }

    T result = std::cosh(a) / std::cosh(b);
    if (!std::isfinite(result)) {
        throw std::runtime_error("cosh_by_cosh produced non-finite result");
    }
    return result;
}

template <int N>
class FentonFFT {
public:
    using Real = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;

    // Compute DCT-I coefficients (equivalent to irfft for real-even signals)
    static Vector compute_inverse_cosine_transform(const Vector& eta) {
        Vector E;
        for (int j = 0; j <= N; ++j) {
            Real sum = 0.0f;
            for (int m = 0; m <= N; ++m) {
                Real w = (m == 0 || m == N) ? 0.5f : 1.0f;
                sum += w * eta(m) * std::cos(M_PI * j * m / N);
            }
            E(j) = 2.0f * sum / N;
        }
        return E;
    }

    // Forward DCT-I to reconstruct η from cosine coefficients
    static Vector compute_forward_cosine_transform(const Vector& E) {
        Vector eta;
        for (int m = 0; m <= N; ++m) {
            Real sum = 0.0f;
            for (int j = 0; j <= N; ++j) {
                Real w = (j == 0 || j == N) ? 0.5f : 1.0f;
                sum += w * E(j) * std::cos(M_PI * j * m / N);
            }
            eta(m) = sum;
        }
        return eta;
    }
};

template <int N = 4>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;

public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega, Q, R;
    VectorF eta, x, E, B;
    VectorF x_nd;  // nondimensional collocation points

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            sum += E(j) * std::cos(j * 2.0f * M_PI * (x_val - c * t) / length);
        }
        return sum;
    }

    Real surface_slope(Real x_val, Real t = 0) const {
        Real d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= E(j) * j * 2.0f * M_PI * std::sin(j * 2.0f * M_PI * (x_val - c * t) / length);
        }
        return d_eta;
    }

    Real surface_time_derivative(Real x_val, Real t = 0) const {
        return -c * surface_slope(x_val, t);
    }

    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real omega_j = j * omega;
            sum -= E(j) * omega_j * omega_j * std::cos(j * 2.0f * M_PI * (x_val - c * t) / length);
        }
        return sum;
    }
    
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real term = j * 2.0f * M_PI / length * j * omega;
            sum += E(j) * term * std::sin(j * 2.0f * M_PI * (x_val - c * t) / length);
        }
        return sum;
    }
    
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real coeff = -std::pow(j * 2.0f * M_PI / length, 2);
            sum += E(j) * coeff * std::cos(j * 2.0f * M_PI * (x_val - c * t) / length);
        }
        return sum;
    }

    Real vertical_velocity(Real x_val, Real z, Real t = 0) const {
        Real w = 0.0f;
        for (int j = 1; j <= N; ++j) {
            Real kj = j * k;
            Real arg = kj * (x_val - c * t);
            Real denom = std::cosh(kj * depth);
            if (denom < std::numeric_limits<Real>::epsilon()) continue;
            Real term = B(j) * kj / denom;
            w += term * std::sin(arg) * std::sinh(kj * (z + depth));
        }
        return w;
    }

    Real get_c() const { return c; }
    Real get_k() const { return k; }
    Real get_T() const { return T; }
    Real get_omega() const { return omega; }
    Real get_length() const { return length; }
    Real get_height() const { return height; }
    const VectorF& get_eta() const { return eta; }

private:
    void compute() {
        // Step 1: Setup nondimensional parameters
        if (depth < 0) depth = 25.0f * length;
    
        Real H = height / depth;
        Real lam = length / depth;          // nondimensional wavelength
        Real k_nd = 2.0f * M_PI / lam;      // nondimensional wavenumber
    
        k = k_nd; // Use nondimensional k until after optimization
    
        Real D = 1.0f;                      // nondimensional depth
        Real kc = k_nd;
        Real c0 = std::sqrt(std::tanh(kc) / kc); // linear wave phase speed
    
        // Step 2: Setup nondimensional x positions (collocation points)
        for (int m = 0; m <= N; ++m)
            x_nd(m) = lam * m / N;  // nondimensional collocation x
    
        // Step 3: Initialize wave coefficients
        B.setZero();
        B(0) = c0;
        Q = c0;
        R = 1.0f + 0.5f * c0 * c0;
    
        VectorF eta_nd;
    
        // Step 4: Ramp-up steps from 0 to target wave height
        for (Real Hi : wave_height_steps(H, D, lam)) {
            for (int m = 0; m <= N; ++m) {
                eta_nd(m) = Hi / 2.0f * std::cos(k_nd * x_nd(m));
                if (!std::isfinite(eta_nd(m))) {
                    throw std::runtime_error("Non-finite value in eta_nd before optimization");
                }
            }
            optimize(B, Q, R, eta_nd, Hi, k_nd, D);
        }
    
        // Step 5: Convert optimized results to dimensional units
        Real sqrt_gd = std::sqrt(g * depth);
    
        // Rescale velocity potential coefficients B
        B(0) *= sqrt_gd;
        for (int j = 1; j <= N; ++j)
            B(j) *= std::sqrt(g * std::pow(depth, 3));
    
        // Rescale constants Q and R
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;
    
        // Rescale eta and x to dimensional
        for (int i = 0; i <= N; ++i) {
            x(i) = x_nd(i) * depth;
            eta(i) = eta_nd(i) * depth;
        }
    
        // Step 6: Rescale wavenumber to dimensional
        k = k_nd / depth;
        c = B(0);                  // phase speed in m/s
        T = length / c;            // period
        omega = c * k;             // angular frequency
    
        // Step 7: Compute FFT
        compute_elevation_coefficients();  // E = irfft(eta)
    }

    void compute_elevation_coefficients() {
        E = FentonFFT<N>::compute_inverse_cosine_transform(eta);
    }

    std::vector<Real> wave_height_steps(Real H, Real D, Real lam) {
        Real Hb = 0.142f * std::tanh(2 * M_PI * D / lam) * lam;
        int num = (H > 0.75f * Hb) ? 10 : (H > 0.65f * Hb) ? 5 : 3;
        std::vector<Real> steps(num);
        for (int i = 0; i < num; ++i)
            steps[i] = H * (i + 1) / num;
        return steps;
    }

    void optimize(VectorF& B, Real& Q, Real& R,
                  VectorF& eta, Real H, Real k_nd, Real D) {
        constexpr int NU = 2 * (N + 1) + 2; // total number of unknowns
        Eigen::Matrix<Real, NU, 1> coeffs;
    
        // Initialize: [B0 ... BN, eta0 ... etaN, Q, R]
        coeffs.template segment<N + 1>(0) = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;
    
        const int max_iter = 500;
        const Real tol = 1e-12f;
    
        Real error = std::numeric_limits<Real>::max();
    
        for (int iter = 0; iter < max_iter && error > tol; ++iter) {
            Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k_nd, D);
            if (!f.allFinite()) {
                throw std::runtime_error("Residual vector contains non-finite values");
            }
            error = f.cwiseAbs().maxCoeff();
    
            if (!std::isfinite(error)) {
                throw std::runtime_error("Non-finite residual during optimization");
            }
            if (error < tol) break;
    
            Eigen::Matrix<Real, NU, NU> J = compute_jacobian(coeffs, H, k_nd, D);
            if (!J.allFinite()) {
                throw std::runtime_error("Jacobian contains non-finite values");
            }
            
            Eigen::Matrix<Real, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;
        }
    
        // Unpack solution
        B   = coeffs.template segment<N + 1>(0);
        eta = coeffs.template segment<N + 1>(N + 1);
        Q   = coeffs(2 * N + 2);
        R   = coeffs(2 * N + 3);
    }

    Eigen::Matrix<Real, StateDim, 1>
    compute_residual(const Eigen::Matrix<Real, StateDim, 1>& coeffs, Real H, Real k, Real D) {
        Eigen::Matrix<Real, StateDim, 1> f;
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real Q = coeffs(2 * N + 2);
        Real R = coeffs(2 * N + 3);
        Real B0 = B(0);
    
        for (int m = 0; m <= N; ++m) {
            Real x_m = x_nd(m);
            Real eta_m = eta(m);
            
            Real um = -B0;
            Real vm = 0;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                Real S1 = sinh_by_cosh(kj * eta_m, kj * D);
                Real C1 = cosh_by_cosh(kj * eta_m, kj * D);                
                if (!std::isfinite(S1) || !std::isfinite(C1)) {
                    throw std::runtime_error("Non-finite S1 or C1 in residual: j=" + std::to_string(j) +
                                             ", eta_m=" + std::to_string(eta_m) +
                                             ", kj*eta_m=" + std::to_string(kj * eta_m));
                }

                Real S2 = std::sin(j * x_m);
                Real C2 = std::cos(j * x_m);
                if (!std::isfinite(S2) || !std::isfinite(C2)) {
                    throw std::runtime_error("Non-finite S2 or C2 in residual: j=" + std::to_string(j) +
                                             ", eta_m=" + std::to_string(eta_m) +
                                             ", kj*eta_m=" + std::to_string(kj * eta_m));
                }

                um += kj * B(j) * C1 * C2;
                vm += kj * B(j) * S1 * S2;
            }
    
            f(m) = -B0 * eta_m;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                Real S1 = sinh_by_cosh(kj * eta_m, kj * D);
                Real C2 = std::cos(j * x_m);
                f(m) += B(j) * S1 * C2;
            }
            f(m) += Q;
    
            f(N + 1 + m) = 0.5f * (um * um + vm * vm) + eta_m - R;
        }
    
        // Match Python: enforce mean elevation = 0
        f(2 * N + 2) = eta.mean();
        f(2 * N + 3) = eta(0) - eta(N) - H;
    
        return f;
    }

    Eigen::Matrix<Real, StateDim, StateDim>
    compute_jacobian(const Eigen::Matrix<Real, StateDim, 1>& coeffs, Real H, Real k, Real D) {
        Eigen::Matrix<Real, StateDim, StateDim> J = Eigen::Matrix<Real, StateDim, StateDim>::Zero();
        auto B = coeffs.template segment<N + 1>(0);
        auto eta = coeffs.template segment<N + 1>(N + 1);
        Real B0 = B(0);

        for (int m = 0; m <= N; ++m) {
            Real eta_m = eta(m);
            Real x_m = x_nd(m);
            Real um = -B0;
            Real vm = 0;
            
            Eigen::Matrix<Real, N, 1> SC, SS, CC, CS;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                SC(j-1) = sinh_by_cosh(kj * eta_m, kj * D) * std::cos(j * x_m);
                SS(j-1) = sinh_by_cosh(kj * eta_m, kj * D) * std::sin(j * x_m);
                CC(j-1) = cosh_by_cosh(kj * eta_m, kj * D) * std::cos(j * x_m);
                CS(j-1) = cosh_by_cosh(kj * eta_m, kj * D) * std::sin(j * x_m);
                
                um += kj * B(j) * CC(j-1);
                vm += kj * B(j) * SS(j-1);
            }

            J(m, 0) = -eta_m;
            for (int j = 1; j <= N; ++j) {
                J(m, j) = SC(j-1);
            }
            J(m, N + 1 + m) = -B0;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                J(m, N + 1 + m) += B(j) * kj * CC(j-1);
            }
            J(m, 2 * N + 2) = 1;

            J(N + 1 + m, 0) = -um;
            for (int j = 1; j <= N; ++j) {
                J(N + 1 + m, j) = k * j * (um * CC(j-1) + vm * SS(j-1));
            }
            J(N + 1 + m, N + 1 + m) = 1;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                J(N + 1 + m, N + 1 + m) += um * B(j) * kj * kj * SC(j-1);
                J(N + 1 + m, N + 1 + m) += vm * B(j) * kj * kj * CS(j-1);
            }
            J(N + 1 + m, 2 * N + 3) = -1;
        }

        for (int j = 0; j <= N; ++j) {
            J(2 * N + 2, N + 1 + j) = (j == 0 || j == N) ? 0.5f/N : 1.0f/N;
        }

        J(2 * N + 3, N + 1) = 1;
        J(2 * N + 3, 2 * N + 1) = -1;

        return J;
    }
};


template<int N = 4>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;

    float t = 0.0f;
    float dt = 0.005f;

    // Object state
    float x = 0.0f;     // Horizontal position (m)
    float vx = 0.0f;    // Horizontal velocity (m/s)

    float mass = 1.0f;  // Mass of floating object (kg)

    // Wave and physics parameters
    float drag_coeff = 0.1f;  // Simple horizontal drag coefficient

    // Periodicity wrap helper
    float wrap_periodic(float val, float period) const {
        while (val < 0.0f) val += period;
        while (val >= period) val -= period;
        return val;
    }

    // Horizontal acceleration from wave slope and drag
    float compute_horizontal_acceleration(float x_pos, float vx_curr, float time) const {
        // Wave surface slope (∂η/∂x)
        float eta_x = wave.surface_slope(x_pos, time);

        // Simple driving force proportional to slope (restoring force)
        float force_wave = -9.81f * eta_x;  // gravity times slope (can be tuned)

        // Simple linear drag opposing velocity
        float force_drag = -drag_coeff * vx_curr;

        // Newton's second law
        return (force_wave + force_drag) / mass;
    }

    // RK4 integration for horizontal motion
    void rk4_step(float& x_curr, float& vx_curr, float t_curr, float dt_step) {
        auto accel = [this](float x_in, float vx_in, float t_in) {
            return compute_horizontal_acceleration(x_in, vx_in, t_in);
        };

        float k1_v = accel(x_curr, vx_curr, t_curr);
        float k1_x = vx_curr;

        float k2_v = accel(x_curr + 0.5f * dt_step * k1_x, vx_curr + 0.5f * dt_step * k1_v, t_curr + 0.5f * dt_step);
        float k2_x = vx_curr + 0.5f * dt_step * k1_v;

        float k3_v = accel(x_curr + 0.5f * dt_step * k2_x, vx_curr + 0.5f * dt_step * k2_v, t_curr + 0.5f * dt_step);
        float k3_x = vx_curr + 0.5f * dt_step * k2_v;

        float k4_v = accel(x_curr + dt_step * k3_x, vx_curr + dt_step * k3_v, t_curr + dt_step);
        float k4_x = vx_curr + dt_step * k3_v;

        x_curr += dt_step * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0f;
        vx_curr += dt_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0f;

        // Periodicity wrap
        x_curr = wrap_periodic(x_curr, wave.get_length());
    }

public:
    WaveSurfaceTracker(float height, float depth, float length, float mass_kg = 1.0f, float drag_coeff_ = 0.1f)
        : wave(height, depth, length), mass(mass_kg), drag_coeff(drag_coeff_) {}

    /**
     * @brief Track the floating object on the wave surface over time.
     * 
     * @param duration Total simulation time (s)
     * @param timestep Time step for integration (s)
     * @param callback Function called every step with:
     *        void callback(t, vertical_displacement, vertical_velocity, vertical_acceleration, x, vx);
     */
    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float, float)> callback)
    {
        dt = std::clamp(timestep, 1e-5f, 0.1f);

        t = 0.0f;
        x = 0.0f;
        vx = 0.0f;

        // Initialize vertical velocity and acceleration to zero
        float prev_z_dot = 0.0f;

        while (t <= duration) {
            // Compute current vertical displacement on wave surface
            float z = wave.surface_elevation(x, t);

            // Compute vertical velocity by chain rule:
            // dz/dt = ∂η/∂t + ∂η/∂x * dx/dt
            float eta_t = wave.surface_time_derivative(x, t);
            float eta_x = wave.surface_slope(x, t);
            float z_dot = eta_t + eta_x * vx;

            // Compute vertical acceleration by finite difference of vertical velocity
            float z_ddot = (z_dot - prev_z_dot) / dt;

            // Call user callback with current state
            callback(t, z, z_dot, z_ddot, x, vx);

            prev_z_dot = z_dot;

            // Integrate horizontal position and velocity with RK4
            rk4_step(x, vx, t, dt);

            t += dt;
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
    const float dt = 0.005f;         // Time step (s)

    // Create a 4th-order Fenton wave and a surface tracker
    WaveSurfaceTracker<4> tracker(height, depth, length);

    // Output file
    std::ofstream out("wave_tracker_data.csv");
    out << "Time(s),Displacement(m),Velocity(m/s),Acceleration(m/s²),X_Position(m)\n";

    // Define the kinematics callback (writes data to file)
    auto kinematics_callback = [&out](
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position, float horizontal_speed) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track floating object (using callback)
    tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif


