#pragma once

/*
   Copyright 2025, Mikhail Grushinskiy

   AI-assisted translation (with eigen vecrorization)
   of https://github.com/bareboat-necessities/bbn-wave-period-esp32/blob/main/bbn_wave_freq_m5atomS3/FentonWaveVectorizedVectorized.h

*/

#pragma once

#include <ArduinoEigenDense.h>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <limits>
#include <algorithm>

template <int N>
class FentonFFTVectorized {
public:
    using Real   = float;
    using Vector = Eigen::Matrix<Real, N + 1, 1>;
    using Matrix = Eigen::Matrix<Real, N + 1, N + 1>;

    static const Matrix& cosine_matrix() {
        static const Matrix M = [](){
            Matrix m;
            for(int j = 0; j <= N; ++j)
                for(int i = 0; i <= N; ++i)
                    m(j,i) = std::cos(j * i * PI / N);
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
class FentonWaveVectorizedVectorized {
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

    FentonWaveVectorizedVectorized(Real height, Real depth, Real length, Real g = 9.81f, Real relax = 0.5f)
        : height(height), depth(depth), length(length), g(g), relax(relax) {
        for (int j = 1; j <= N; ++j) {
            kj_cache(j-1) = j * (2 * M_PI / length);
            j_cache(j-1) = j;
        }
        compute();
    }

    Real surface_elevation(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return E(0) + (E.tail(N).array() * (j_cache * phase_base).cos()).sum();
    }

    Real surface_slope(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -k * (E.tail(N).array() * j_cache.array() * (j_cache * phase_base).sin()).sum();
    }

    Real surface_time_derivative(Real x_val, Real t = 0) const {
        return -c * surface_slope(x_val, t);
    }

    Real surface_second_time_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -(E.tail(N).array() * (j_cache * omega).square() * (j_cache * phase_base).cos()).sum();
    }
    
    Real surface_space_time_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return (E.tail(N).array() * j_cache.array().square() * (k * omega) * 
               (j_cache * phase_base).sin()).sum();
    }
    
    Real surface_second_space_derivative(Real x_val, Real t = 0) const {
        const Real phase_base = k * (x_val - c * t);
        return -(E.tail(N).array() * j_cache.array().square() * k * k * 
                (j_cache * phase_base).cos()).sum();
    }

    Real vertical_velocity(Real x_val, Real z, Real t = 0) const {
        const VectorN arg = kj_cache.array() * (x_val - c * t);
        const VectorN denom = (kj_cache * depth).array().cosh();
        const VectorN sinh_z = (kj_cache.array() * (z + depth)).sinh();
        return (B.tail(N).array() * kj_cache.array() * arg.sin() * sinh_z / denom).sum();
    }

    // ... (getters remain the same) ...

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
        E = FentonFFTVectorized<N>::compute_inverse_cosine_transform(eta);
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
        const MatrixNxP S1 = eta_kj.unaryExpr([&](Real val) { 
            return sinh_by_cosh(val, kj(0) * D); 
        });
        const MatrixNxP C1 = eta_kj.unaryExpr([&](Real val) { 
            return cosh_by_cosh(val, kj(0) * D); 
        });

        const VectorF um = VectorF::Constant(-B0) + 
                         (B.tail(N).transpose() * (kj.asDiagonal() * C1 * cos_terms)).transpose();
        const VectorF vm = (B.tail(N).transpose() * (kj.asDiagonal() * S1 * sin_terms)).transpose();

        residual.head(N+1) = VectorF::Constant(-B0).cwiseProduct(eta) + 
                           (B.tail(N).transpose() * (S1 * cos_terms)).transpose() + 
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
        const MatrixNxP S1 = eta_kj.array().unaryExpr([&](Real val) {
            return sinh_by_cosh(val, kj(0) * D);
        });
        const MatrixNxP C1 = eta_kj.array().unaryExpr([&](Real val) {
            return cosh_by_cosh(val, kj(0) * D);
        });

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


template <int N>
class WaveSurfaceTrackerVectorized {
private:
    FentonWaveVectorized<N>& wave;

public:
    float t=0, dt=0.005f, x=0, vx=0, mass=1.0f, drag=0.1f;

    WaveSurfaceTrackerVectorized(FentonWaveVectorized<N>& w): wave(w) {}

    void track(float duration, float timestep,
               std::function<void(float,float,float,float,float,float)> cb) 
    {
        dt = std::clamp(timestep, 1e-5f, 0.1f);
        t = 0; x = 0; vx = 0;
        float prev_zdot = 0;

        while(t <= duration) {
            float z = wave.surface_elevation(x, t);
            float eta_x = wave.surface_slope(x, t);
            float zdot = wave.surface_time_derivative(x, t) + eta_x * vx;
            float zddot = (zdot - prev_zdot) / dt;
            cb(t, z, zdot, zddot, x, vx);
            prev_zdot = zdot;

            auto acc = [&](float xp, float vp) {
                return (-9.81f * wave.surface_slope(xp, t) - drag * vp) / mass;
            };

            float k1_v = acc(x, vx), k1_x = vx;
            float k2_v = acc(x + 0.5f*dt*k1_x, vx + 0.5f*dt*k1_v);
            float k2_x = vx + 0.5f*dt*k1_v;
            float k3_v = acc(x + 0.5f*dt*k2_x, vx + 0.5f*dt*k2_v);
            float k3_x = vx + 0.5f*dt*k2_v;
            float k4_v = acc(x + dt*k3_x, vx + dt*k3_v);
            float k4_x = vx + dt*k3_v;

            x += dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6;
            vx += dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6;

            if(x < 0) x += wave.length;
            else if(x >= wave.length) x -= wave.length;

            t += dt;
        }
    }
};

