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
    if (a == 0) return 0;
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }
    return std::sinh(a) / std::cosh(b);
}

template <typename T>
T cosh_by_cosh(T a, T b) {
    if (a == 0) return 1.0 / std::cosh(b);
    T f = b / a;
    if ((a > 30 && 0.5 < f && f < 1.5) || (a > 200 && 0.1 < f && f < 1.9)) {
        return std::exp(a * (1 - f));
    }
    return std::cosh(a) / std::cosh(b);
}

template <int N>
class FentonFFT {
public:
    using Real = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;

    // Inverse DCT-I (irfft-style): reconstruct cosine coefficients E from eta
    static Vector compute_inverse_cosine_transform(const Vector& eta) {
        Vector E;
        for (int j = 0; j <= N; ++j) {
            Real sum = 0;
            for (int m = 0; m <= N; ++m) {
                Real weight = (m == 0 || m == N) ? 0.5f : 1.0f;
                sum += weight * eta(m) * std::cos(j * m * M_PI / N);
            }
            E(j) = 2.0f * sum / N;
        }
        return E;
    }

    // Forward DCT-I: reconstruct eta values at collocation points from cosine coeffs
    static Vector compute_forward_cosine_transform(const Vector& E) {
        Vector eta;
        for (int m = 0; m <= N; ++m) {
            Real sum = 0;
            for (int j = 0; j <= N; ++j) {
                Real weight = (j == 0 || j == N) ? 0.5f : 1.0f;
                sum += weight * E(j) * std::cos(j * m * M_PI / N);
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

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            sum += E(j) * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }

    Real surface_slope(Real x_val, Real t = 0) const {
        Real d_eta = 0.0f;
        for (int j = 0; j <= N; ++j) {
            d_eta -= E(j) * j * k * std::sin(j * k * (x_val - c * t));
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
            sum -= E(j) * omega_j * omega_j * std::cos(j * k * (x_val - c * t));
        }
        return sum;
    }
    
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real term = j * k * j * omega;
            sum += E(j) * term * std::sin(j * k * (x_val - c * t));
        }
        return sum;
    }
    
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
        Real sum = 0;
        for (int j = 0; j <= N; ++j) {
            Real coeff = -j * k * j * k;
            sum += E(j) * coeff * std::cos(j * k * (x_val - c * t));
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
        if (depth < 0) depth = 25.0f * length;
        Real H = height / depth;
        Real lam = length / depth;
        k = 2 * M_PI / lam;
        Real D = 1.0f;
        Real kc = k;
        Real c0 = std::sqrt(std::tanh(kc) / kc);

        VectorF x_nd;
        for (int m = 0; m <= N; ++m)
            x_nd(m) = lam * m / (2.0f * N);

        B.setZero();
        B(0) = c0;
        B(1) = -H / (4.0f * c0 * k);

        VectorF eta_nd;
        for (int m = 0; m <= N; ++m)
            eta_nd(m) = 1.0f + H / 2.0f * std::cos(k * x_nd(m));

        Q = c0;
        R = 1.0f + 0.5f * c0 * c0;

        for (Real Hi : wave_height_steps(H, D, lam)) {
            optimize(B, Q, R, eta_nd, Hi, k, D);
        }

        Real sqrt_gd = std::sqrt(g * depth);
        B(0) *= sqrt_gd;
        for (int j = 1; j <= N; ++j)
            B(j) *= std::sqrt(g * std::pow(depth, 3));
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;

        for (int i = 0; i <= N; ++i) {
            x(i) = x_nd(i) * depth;
            eta(i) = eta_nd(i) * depth;
        }

        k = k / depth;
        c = B(0);
        T = length / c;
        omega = c * k;

        compute_elevation_coefficients();
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
                 VectorF& eta, Real H, Real k, Real D) {
        constexpr int NU = 2 * (N + 1) + 2;
        Eigen::Matrix<Real, NU, 1> coeffs;
        coeffs.template segment<N + 1>(0) = B;
        coeffs.template segment<N + 1>(N + 1) = eta;
        coeffs(2 * N + 2) = Q;
        coeffs(2 * N + 3) = R;

        Real error = std::numeric_limits<Real>::max();
        for (int iter = 0; iter < 500 && error > 1e-8f; ++iter) {
            Eigen::Matrix<Real, NU, 1> f = compute_residual(coeffs, H, k, D);
            error = f.cwiseAbs().maxCoeff();
            
            Real eta_max = coeffs.template segment<N + 1>(N + 1).maxCoeff();
            Real eta_min = coeffs.template segment<N + 1>(N + 1).minCoeff();
            if (eta_max > 2.0f || eta_min < 0.1f || !std::isfinite(error)) {
                throw std::runtime_error("Optimization failed");
            }

            if (error < 1e-8f) break;

            Eigen::Matrix<Real, NU, NU> J = compute_jacobian(coeffs, H, k, D);
            Eigen::Matrix<Real, NU, 1> delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;
        }

        B = coeffs.template segment<N + 1>(0);
        eta = coeffs.template segment<N + 1>(N + 1);
        Q = coeffs(2 * N + 2);
        R = coeffs(2 * N + 3);
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
            Real x_m = M_PI * m / N;
            Real eta_m = eta(m);
            
            Real um = -B0;
            Real vm = 0;
            for (int j = 1; j <= N; ++j) {
                Real kj = j * k;
                Real S1 = sinh_by_cosh(kj * eta_m, kj * D);
                Real C1 = cosh_by_cosh(kj * eta_m, kj * D);
                Real S2 = std::sin(j * x_m);
                Real C2 = std::cos(j * x_m);
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

        f(2 * N + 2) = (eta.sum() - 0.5f * (eta(0) + eta(N))) / N - 1.0f;
        f(2 * N + 3) = eta.maxCoeff() - eta.minCoeff() - H;

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
            Real x_m = M_PI * m / N;
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
    float x = 0.0f;
    float v = 0.0f; // horizontal velocity
    float dt = 0.005f;

    float mean_eta = 0.0f;

    static constexpr float slope_eps = 1e-6f;

    float mass = 1.0f;     // mass of floating object
    float gravity = 9.81f; // gravity constant

    // Wrap x into one wavelength
    float wrap_periodic(float val, float period) const {
        while (val < 0.0f) val += period;
        while (val >= period) val -= period;
        return val;
    }

    void compute_mean_elevation(int samples = 100) {
        float sum = 0.0f;
        float L = wave.get_length();
        for (int i = 0; i < samples; ++i) {
            float xi = L * i / (samples - 1);
            sum += wave.surface_elevation(xi, 0.0f);
        }
        mean_eta = sum / samples;
    }

    // Compute dx/dt = (w - eta_t) / eta_x (used only for initial guess or fallback)
    float compute_horizontal_speed(float x_pos, float time) const {
        float eta = wave.surface_elevation(x_pos, time) - mean_eta;
        float eta_dot = wave.surface_time_derivative(x_pos, time);
        float eta_x = wave.surface_slope(x_pos, time);
        float w = wave.vertical_velocity(x_pos, eta + mean_eta, time);

        if (std::abs(eta_x) < slope_eps)
            eta_x = (eta_x >= 0.0f) ? slope_eps : -slope_eps;

        return (w - eta_dot) / eta_x;
    }

    // Compute horizontal wave slope force: F = -mg * slope
    float compute_horizontal_force(float x_pos, float time) const {
        float slope = wave.surface_slope(x_pos, time);
        return -mass * gravity * slope;
    }

public:
    WaveSurfaceTracker(float height, float depth, float length, float object_mass = 1.0f)
        : wave(height, depth, length), mass(object_mass)
    {
        compute_mean_elevation();
    }

    /**
     * @brief Track floating object on wave with mass and inertia.
     * @param duration Duration (s)
     * @param timestep Timestep (s)
     * @param callback void(float t, float z, float dzdt, float ddzdt2, float x)
     */
    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float)> callback)
    {
        const float wave_T = wave.get_T();
        const float wave_L = wave.get_length();

        dt = std::clamp(timestep, 1e-5f, 0.2f * wave_T / 20.0f);

        // Initial conditions
        t = 0.0f;
        x = 0.0f;
        v = compute_horizontal_speed(x, t); // good initial guess

        float prev_z = wave.surface_elevation(x, t) - mean_eta;
        float prev_dzdt = wave.vertical_velocity(x, prev_z + mean_eta, t);

        while (t <= duration) {
            // Compute horizontal force and update horizontal acceleration
            float F = compute_horizontal_force(x, t);
            float a = F / mass;

            // Integrate horizontal motion (Euler or RK2)
            v += a * dt;
            x += v * dt;
            x = wrap_periodic(x, wave_L);
            t += dt;

            // Surface elevation = vertical position
            float z = wave.surface_elevation(x, t) - mean_eta;
            float dzdt = (z - prev_z) / dt;
            float ddzdt2 = (dzdt - prev_dzdt) / dt;

            callback(t, z, dzdt, ddzdt2, x);

            prev_z = z;
            prev_dzdt = dzdt;
        }
    }

    // Optional helper for computing restoring coefficient
    float compute_alpha() const {
        return gravity / mean_eta;
    }

    // Accessors
    float get_time() const { return t; }
    float get_x() const { return x; }
    float get_dt() const { return dt; }
    float get_mean_eta() const { return mean_eta; }

    const FentonWave<N>& get_wave() const { return wave; }
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
        float time, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position) {
        out << time << "," << elevation << "," << vertical_velocity << "," << vertical_acceleration << "," << horizontal_position << "\n";
    };

    // Track floating object (using callback)
    tracker.track_floating_object(duration, dt, kinematics_callback);
}

#endif


