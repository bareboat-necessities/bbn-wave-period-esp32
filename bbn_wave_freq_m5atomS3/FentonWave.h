#pragma once

#include <ArduinoEigenDense.h>
#include <cmath>
#include <stdexcept>
#include <functional>

template<int N = 3>
class FentonWave {
private:
    static constexpr int StateDim = 2 * (N + 1) + 2;

    using VectorF = Eigen::Matrix<float, N + 1, 1>;
    using BigVector = Eigen::Matrix<float, StateDim, 1>;
    using BigMatrix = Eigen::Matrix<float, StateDim, StateDim>;

    float height, depth, length, g, relax;
    VectorF eta, x, B, E;
    float k, c, cs, T, omega;

public:
    FentonWave(float height, float depth, float length,
               float g = 9.81f, float relax = 0.5f)
        : height(height), depth(depth), length(length),
          g(g), relax(relax) {
        auto coeffs = solve_fenton_equations();
        set_coefficients(coeffs);
    }

    float surface_elevation(float x_val, float t = 0) const {
        VectorF J;
        J.setLinSpaced(N + 1, 0, N);
        float arg_scale = M_PI / length;
        return (2.0f / N) * (E.array() * (J.array() * arg_scale * (x_val - c * t)).cos()).sum();
    }

    Eigen::Vector2f velocity(float x_val, float z_val, float t = 0) const {
        Eigen::Vector2f vel = Eigen::Vector2f::Zero();
        for (int j = 1; j <= N; ++j) {
            float kj = j * k;
            float arg = kj * (x_val - c * t);
            float denom = std::cosh(kj * depth);
            if (denom < 1e-6f) denom = 1e-6f;
            float term = B[j] * kj / denom;
            vel[0] += term * std::cos(arg) * std::cosh(kj * z_val); // u
            vel[1] += term * std::sin(arg) * std::sinh(kj * z_val); // w
        }
        return vel;
    }

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

        for (int j = 0; j <= N; ++j) {
            VectorF cosine;
            for (int i = 0; i <= N; ++i)
                cosine[i] = std::cos(j * M_PI * x[i] / length);
            E[j] = trapezoid_integration(eta.array() * cosine.array());
        }
    }

    FentonCoefficients solve_fenton_equations(int maxiter = 100, float tol = 1e-6f) {
        float H = height / depth;
        float lambda = length / depth;
        float k_nd = 2 * M_PI / lambda;
        float D = 1.0f;

        float c_guess = std::sqrt(g * depth * std::tanh(k_nd * D) / k_nd);

        VectorF grid;
        grid.setLinSpaced(N + 1, 0, N);
        VectorF x_nd = (grid.array() * lambda / N).matrix();
        VectorF x_phys = x_nd * depth;

        FentonCoefficients coeffs;
        coeffs.B.setZero();
        coeffs.B[0] = c_guess;
        if (N >= 1)
            coeffs.B[1] = -H / (2 * k_nd * std::cosh(k_nd * D));
        coeffs.eta = (H / 2) * (2 * M_PI * x_nd.array() / lambda).cos().matrix();
        coeffs.Q = 0;
        coeffs.R = 1 + 0.5f * c_guess * c_guess;

        BigVector params;
        params.segment(0, N + 1) = coeffs.B;
        params.segment(N + 1, N + 1) = coeffs.eta;
        params[2 * (N + 1)] = coeffs.Q;
        params[2 * (N + 1) + 1] = coeffs.R;

        for (int iter = 0; iter < maxiter; ++iter) {
            BigVector f = compute_residuals(params, H, k_nd, D, x_nd);
            if (!std::isfinite(f.norm()))
                throw std::runtime_error("Residual diverged: NaN or Inf");
            if (f.norm() < tol) break;

            BigMatrix J = compute_jacobian(params, H, k_nd, D, x_nd);
            BigVector delta = J.colPivHouseholderQr().solve(f);
            params -= relax * delta;
        }

        coeffs.B = params.segment(0, N + 1);
        coeffs.eta = params.segment(N + 1, N + 1);
        coeffs.Q = params[2 * (N + 1)];
        coeffs.R = params[2 * (N + 1) + 1];
        coeffs.x = x_phys;
        coeffs.k = k_nd;

        scale_to_physical(coeffs);
        return coeffs;
    }

    BigVector compute_residuals(const BigVector& params, float H, float k, float D, const VectorF& x_nd) {
        BigVector res = BigVector::Zero();

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

            res[m] = -B[0] * etam + Q;
            for (int j = 1; j <= N; ++j)
                res[m] += B[j] * S1[j] * C2[j];

            res[N + 1 + m] = 0.5f * (um * um + vm * vm) + etam - R;
        }

        // Mean elevation constraint
        res[2 * (N + 1)] = trapezoid_integration(eta) / length;

        // Wave height constraint using crest and trough
        res[2 * (N + 1) + 1] = eta[0] - eta[N / 2] - H;

        return res;
    }

    BigMatrix compute_jacobian(const BigVector& params, float H, float k, float D, const VectorF& x_nd) {
        BigMatrix Jmat = BigMatrix::Zero();

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

        Jmat(2 * (N + 1) + 1, N + 1 + 0) = 1.0f;
        Jmat(2 * (N + 1) + 1, N + 1 + N / 2) = -1.0f;

        return Jmat;
    }

    void scale_to_physical(FentonCoefficients& coeffs) {
        coeffs.B[0] *= std::sqrt(g * depth);
        coeffs.B.tail(N) *= std::sqrt(g * std::pow(depth, 3));
        coeffs.eta *= depth;
        coeffs.x *= depth;
        coeffs.k /= depth;
        coeffs.c = coeffs.B[0];
        coeffs.Q *= std::sqrt(g * std::pow(depth, 3));
        coeffs.R *= g * depth;
    }

    float trapezoid_integration(const Eigen::Array<float, N + 1, 1>& y) const {
        float dx = length / N;
        float sum = 0.5f * (y[0] + y[N]);
        for (int i = 1; i < N; ++i) sum += y[i];
        return sum * dx;
    }
};

// Tracks vertical surface motion at the crest
template<int N = 3>
class WaveSurfaceTracker {
private:
    FentonWave<N> wave;
    float phase_velocity, wave_number;
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
