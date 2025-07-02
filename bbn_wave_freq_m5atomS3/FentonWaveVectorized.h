#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy
   AI-assisted translation of https://github.com/TormodLandet/raschii/blob/master/raschii/fenton.py
*/

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
    using Real   = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;
    using Matrix = Eigen::Matrix<Real, N + 1, N + 1>;

    static const Matrix& cosine_matrix() {
        static const Matrix M = [](){
            Matrix m;
            for(int j = 0; j <= N; ++j)
                for(int i = 0; i <= N; ++i)
                    m(j,i) = std::cos(j * i * M_PI / N);
            return m;
        }(); return M;
    }
    static const Vector& weights() {
        static Vector w = [](){
            Vector v = Vector::Ones();
            v(0) = v(N) = 0.5f;
            return v;
        }(); return w;
    }

    static Vector compute_inverse_cosine_transform(const Vector& eta) {
        return (2.0f / N) * (cosine_matrix() * (eta.array() * weights().array()).matrix());
    }

    static Vector compute_forward_cosine_transform(const Vector& E) {
        return cosine_matrix().transpose() * (E.array() * weights().array()).matrix();
    }
};

template <int N = 4>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;
    using Real = float;
    using VectorF = Eigen::Matrix<Real, N + 1, 1>;
    using VectorN = Eigen::Matrix<Real, N, 1>;
    using MatrixNxP = Eigen::Matrix<Real, N, N + 1>;
    using BigVector = Eigen::Matrix<Real, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<Real, StateDim, StateDim>;

public:
    Real height, depth, length, g, relax;
    Real k, c, T, omega, Q, R;
    VectorF eta, x, E, B;
    VectorN kj_cache, j_cache;

    FentonWave(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        for (int j = 1; j <= N; ++j) {
            kj_cache(j-1) = j * (2 * M_PI / length);
            j_cache(j-1) = j;
        }
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return E(0) + (E.tail(N).array() * (j_cache.array() * phase_base).cos()).sum();
    }

    Real surface_slope(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -k * (E.tail(N).array() * j_cache.array() * (j_cache.array() * phase_base).sin()).sum();
    }

    Real surface_time_derivative(Real x_val, Real t = 0) const {
        return -c * surface_slope(x_val, t);
    }

    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -(E.tail(N).array() * (j_cache.array() * omega).square() * (j_cache.array() * phase_base).cos()).sum();
    }
    
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return (E.tail(N).array() * j_cache.array().square() * (k * omega) * 
               (j_cache.array() * phase_base).sin()).sum();
    }
    
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -(E.tail(N).array() * j_cache.array().square() * k * k * 
                (j_cache.array() * phase_base).cos()).sum();
    }

    Real vertical_velocity(Real x_val, Real z, Real t = 0) const {
        const VectorN arg = kj_cache.array() * (x_val - c * t);
        const VectorN denom = (kj_cache * depth).array().cosh();
        const VectorN sinh_z = (kj_cache.array() * (z + depth)).sinh();
        return (B.tail(N).array() * kj_cache.array() * arg.array().sin() * sinh_z.array() / denom).sum();
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

        VectorF x_nd = VectorF::LinSpaced(N+1, 0, lam/2.0f);
        B.setZero();
        B(0) = c0;
        B(1) = -H / (4.0f * c0 * k);

        VectorF eta_nd = (VectorF::Ones() + (H/2.0f) * (k * x_nd.array()).cos()).eval();
        Q = c0;
        R = 1.0f + 0.5f * c0 * c0;

        for (Real Hi : wave_height_steps(H, D, lam)) {
            optimize(B, Q, R, eta_nd, Hi, k, D);
        }

        Real sqrt_gd = std::sqrt(g * depth);
        B(0) *= sqrt_gd;
        B.tail(N) *= std::sqrt(g * std::pow(depth, 3));
        Q *= std::sqrt(g * std::pow(depth, 3));
        R *= g * depth;

        x = x_nd * depth;
        eta = (eta_nd.array() - 1.0f) * depth;
        k /= depth;
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
        Eigen::Array<Real, Eigen::Dynamic, 1> steps = 
            Eigen::Array<Real, Eigen::Dynamic, 1>::LinSpaced(num, 1, num) * H / num;
        return std::vector<Real>(steps.data(), steps.data() + steps.size());
    }

    void optimize(VectorF& B, Real& Q, Real& R, VectorF& eta, Real H, Real k, Real D) {
        BigVector coeffs;
        coeffs << B, eta, Q, R;

        Real error = std::numeric_limits<Real>::max();
        for (int iter = 0; iter < 500 && error > 1e-8f; ++iter) {
            BigVector f = compute_residual(coeffs, H, k, D);
            error = f.cwiseAbs().maxCoeff();
            
            if (eta.maxCoeff() > 2.0f || eta.minCoeff() < 0.1f || !std::isfinite(error)) {
                throw std::runtime_error("Optimization failed");
            }

            if (error < 1e-8f) break;

            BigMatrix J = compute_jacobian(coeffs, H, k, D);
            BigVector delta = J.fullPivLu().solve(-f);
            coeffs += relax * delta;
        }

        B = coeffs.template segment<N+1>(0);
        eta = coeffs.template segment<N+1>(N+1);
        Q = coeffs(2*N+2);
        R = coeffs(2*N+3);
    }

    BigVector compute_residual(const BigVector& coeffs, Real H, Real k, Real D) {
        BigVector residual = BigVector::Zero();
        const VectorF B = coeffs.template segment<N+1>(0);
        const VectorF eta = coeffs.template segment<N+1>(N+1);
        const Real Q = coeffs(2*N+2);
        const Real R = coeffs(2*N+3);
        const Real B0 = B(0);

        const VectorF x_m = VectorF::LinSpaced(N+1, 0, N) * (M_PI / N);
        const VectorN kj = VectorN::LinSpaced(N, 1, N) * k;

        const MatrixNxP trig_terms = (kj * x_m.transpose()).array();
        const MatrixNxP cos_terms = trig_terms.cos();
        const MatrixNxP sin_terms = trig_terms.sin();

        const MatrixNxP eta_kj = (kj * eta.transpose()).array();
        MatrixNxP S1, C1;
        for (int j = 0; j < N; ++j) {
            Real kj_val = kj(j);
            S1.row(j) = eta.transpose().unaryExpr([&](Real eta_val) {
                return sinh_by_cosh(kj_val * eta_val, kj_val * D);
            });
            C1.row(j) = eta.transpose().unaryExpr([&](Real eta_val) {
                return cosh_by_cosh(kj_val * eta_val, kj_val * D);
            });
        }
        const VectorF um = VectorF::Constant(-B0) + 
                         (B.tail(N).transpose() * (kj.asDiagonal() * C1.array() * cos_terms.array())).transpose();
        const VectorF vm = (B.tail(N).transpose() * (kj.asDiagonal() * S1.array() * sin_terms.array())).transpose();

        residual.head(N+1) = VectorF::Constant(-B0).cwiseProduct(eta) + 
                           (B.tail(N).transpose() * (S1.array() * cos_terms.array())).transpose() + 
                           VectorF::Constant(Q);

        residual.segment(N+1, N+1) = 0.5 * (um.array().square() + vm.array().square()) + 
                                   eta.array() - R;

        residual(2*N+2) = eta.mean() * (N+1)/N - 1.0f;
        residual(2*N+3) = eta.maxCoeff() - eta.minCoeff() - H;

        return residual;
    }

    BigMatrix compute_jacobian(const BigVector& coeffs, Real H, Real k, Real D) {
        BigMatrix J = BigMatrix::Zero();
        const VectorF B = coeffs.template segment<N+1>(0);
        const VectorF eta = coeffs.template segment<N+1>(N+1);
        const Real B0 = B(0);

        const VectorF x_m = VectorF::LinSpaced(N+1, 0, N) * (M_PI / N);
        const VectorN kj = VectorN::LinSpaced(N, 1, N) * k;
        const VectorN kj_sq = kj.array().square();

        const MatrixNxP trig_args = kj * x_m.transpose();
        const MatrixNxP cos_terms = trig_args.array().cos();
        const MatrixNxP sin_terms = trig_args.array().sin();

        const MatrixNxP eta_kj = kj * eta.transpose();
        MatrixNxP S1, C1;
        for (int j = 0; j < N; ++j) {
            Real kj_val = kj(j);
            S1.row(j) = eta.transpose().unaryExpr([&](Real eta_val) {
                return sinh_by_cosh(kj_val * eta_val, kj_val * D);
            });
            C1.row(j) = eta.transpose().unaryExpr([&](Real eta_val) {
                return cosh_by_cosh(kj_val * eta_val, kj_val * D);
            });
        }

        const MatrixNxP SC = S1 * cos_terms.array();
        const MatrixNxP SS = S1 * sin_terms.array();
        const MatrixNxP CC = C1 * cos_terms.array();
        const MatrixNxP CS = C1 * sin_terms.array();

        const VectorF um = VectorF::Constant(-B0) + 
                         (B.tail(N).transpose() * (kj.asDiagonal() * CC)).transpose();
        const VectorF vm = (B.tail(N).transpose() * (kj.asDiagonal() * SS)).transpose();

        J.block(0, 0, N+1, 1) = -eta;
        J.block(0, 1, N+1, N) = SC.transpose();
        
        VectorF diag_terms = VectorF::Constant(-B0) + 
            (B.tail(N).transpose() * (kj_sq.asDiagonal() * CC)).transpose()
            .cwiseQuotient(eta.transpose().replicate(N,1))
            .cwiseProduct(SC);
        
        J.block(N+1, N+1, N+1, N+1).diagonal() = diag_terms;
        J.block(N+1, 0, N+1, 1) = -um;
        
        for (int j = 1; j <= N; ++j) {
            J.block(N+1, j, N+1, 1) = k * j * 
                (um * CC.row(j-1).transpose().array() + 
                 vm * SS.row(j-1).transpose().array());
        }
        
        J.block(N+1, 2*N+3, N+1, 1) = VectorF::Constant(-1);
        
        for (int m = 0; m <= N; ++m) {
            J(2*N+2, N+1+m) = (m == 0 || m == N) ? 0.5f/N : 1.0f/N;
        }

        int max_idx, min_idx;
        eta.maxCoeff(&max_idx);
        eta.minCoeff(&min_idx);
        J(2*N+3, N+1+max_idx) = 1;
        J(2*N+3, N+1+min_idx) = -1;

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
    WaveSurfaceTracker(float height, float depth, float length, float x0, float mass_kg, float drag_coeff_)
        : wave(height, depth, length), mass(mass_kg), drag_coeff(drag_coeff_) 
    {
        x = x0;
        vx = 0.0f;
    }

    void track_floating_object(
        float duration,
        float timestep,
        std::function<void(float, float, float, float, float, float)> callback)
    {
        dt = clamp_value(timestep, 1e-5f, 0.1f);

        t = 0.0f;

        // Initialize vertical velocity
        float prev_z_dot = wave.surface_time_derivative(x, 0) + wave.surface_slope(x, 0) * vx;

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
            if (t > dt) {
                callback(t, z, z_dot, z_ddot, x, vx);
            }

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
    const float init_x = 10.0f;  // Initial x (m)
    const float mass = 5.0f;     // Mass (kg)
    const float drag = 0.1f;     // Linear drag coeff opposing velocity
    
    // Simulation parameters
    const float duration = 30.0f; // Simulation duration (s)
    const float dt = 0.005f;      // Time step (s)

    // Create a 4th-order Fenton wave and a surface tracker
    WaveSurfaceTracker<4> tracker(height, depth, length, init_x, mass, drag);

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
