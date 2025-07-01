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
public:
    using Real       = float;
    using VecF       = Eigen::Matrix<Real, N + 1, 1>;
    using VecN       = Eigen::Matrix<Real, N, 1>;
    using MatNxP     = Eigen::Matrix<Real, N, N + 1>;
    static constexpr int StateDim = 2 * (N + 1) + 2;
    using BigVec     = Eigen::Matrix<Real, StateDim, 1>;
    using BigMat     = Eigen::Matrix<Real, StateDim, StateDim>;

    Real height, depth, length, g = 9.81f, relax = 0.5f;
    Real k, c, T, omega, Q, R;
    VecF eta_nd, x_nd, eta, x, E, B;
    VecN kj, jidx;
    MatNxP cos_ph, sin_ph;

    FentonWaveVectorizedVectorized(Real h, Real d, Real L, Real gravity=9.81f, Real r=0.5f)
      : height(h), depth(d), length(L), g(gravity), relax(r)
    {
        Real invL = 2 * PI / L;
        for(int j=1; j<=N; ++j) {
            kj(j-1)   = j * invL;
            jidx(j-1) = j;
        }
        precompute_phase();
        compute();
    }

    // --- Surface evaluation ---
    Real surface_elevation(Real x0, Real t=0) const {
        Real p = k * (x0 - c*t);
        return E(0) + (E.tail(N).array() * (jidx * p).cos()).sum();
    }
    Real surface_slope(Real x0, Real t=0) const {
        Real p = k * (x0 - c*t);
        return -k * (E.tail(N).array() * jidx.array() * (jidx * p).sin()).sum();
    }
    Real surface_time_derivative(Real x0, Real t=0) const {
        return -c * surface_slope(x0, t);
    }
    Real surface_second_time_derivative(Real x0, Real t=0) const {
        Real p = k * (x0 - c*t);
        return -(E.tail(N).array() * (jidx * omega).square() * (jidx * p).cos()).sum();
    }
    Real vertical_velocity(Real x0, Real z, Real t=0) const {
        VecN arg   = kj * (x0 - c*t);
        VecN denom = (kj*depth).array().cosh();
        VecN sinhz = (kj*(z+depth)).array().sinh();
        return (B.tail(N).array() * kj.array() * arg.array().sin()
                * sinhz.array() / denom.array()).sum();
    }

    // ... (getters remain the same) ...

private:
    void precompute_phase() {
        x_nd = VecF::LinSpaced(N+1, 0, PI);
        MatNxP KJ = kj.replicate(1, N+1);
        MatNxP XM = x_nd.transpose().replicate(N, 1);
        cos_ph = (KJ.cwiseProduct(XM)).array().cos();
        sin_ph = (KJ.cwiseProduct(XM)).array().sin();
    }

    void compute() {
        if(depth < 0) depth = 25.0f * length;
        Real H = height / depth;
        Real lam = length / depth;
        k = 2 * PI / lam;
        Real D = 1, kc = k, c0 = std::sqrt(std::tanh(kc)/kc);

        x_nd = VecF::LinSpaced(N+1, 0, lam/2);
        B.setZero(); B(0)=c0; B(1)=-H/(4*c0*k);
        eta_nd = VecF::Ones() + (H/2)*(k*x_nd).array().cos().matrix();
        Q = c0; R = 1 + 0.5f*c0*c0;

        for(Real Hi : wave_height_steps(H, D, lam))
            optimize(B, Q, R, eta_nd, H, D);

        Real sg = std::sqrt(g * depth);
        B(0) *= sg;
        B.tail(N) *= std::sqrt(g * depth * depth * depth);
        Q *= std::sqrt(g * depth * depth * depth);
        R *= g * depth;

        x   = x_nd * depth;
        eta = (eta_nd.array() - 1).matrix() * depth;
        k /= depth; c = B(0); T = length / c; omega = c * k;
        E = FentonFFTVectorized<N>::compute_inverse_cosine_transform(eta);
    }

    std::vector<Real> wave_height_steps(Real H, Real, Real lam) {
        Real Hb = 0.142f * std::tanh(2*PI*depth/lam) * lam;
        int count = (H > 0.75f*Hb) ? 10 : (H > 0.65f*Hb) ? 5 : 3;
        std::vector<Real> out(count);
        for(int i=0; i<count; ++i)
            out[i] = H * (i + 1) / count;
        return out;
    }

    static Real sinh_by_cosh(Real a, Real b) {
        if(a == 0) return 0;
        Real f = b / a;
        if((a > 30 && f > 0.5f && f < 1.5f) || (a > 200 && f > 0.1f && f < 1.9f))
            return std::exp(a * (1 - f));
        return std::sinh(a) / std::cosh(b);
    }

    static Real cosh_by_cosh(Real a, Real b) {
        if(a == 0) return 1.f / std::cosh(b);
        Real f = b / a;
        if((a > 30 && f > 0.5f && f < 1.5f) || (a > 200 && f > 0.1f && f < 1.9f))
            return std::exp(a * (1 - f));
        return std::cosh(a) / std::cosh(b);
    }

    void optimize(VectorF& B, Real& Q, Real& R, VectorF& eta, Real H, Real k, Real D) {
        BigVec X; X << Bv, etav, Qv, Rv;
        Real err = std::numeric_limits<Real>::infinity();

        for(int it=0; it<500 && err>1e-8f; ++it) {
            BigVec f = compute_residual(X, H, D);
            err = f.template head<2*N+2>().cwiseAbs().maxCoeff();
            if(!std::isfinite(err) || etav.maxCoeff()>2 || etav.minCoeff()<0.1f)
                throw std::runtime_error("Optimization failed");
            if(err < 1e-8f) break;
            BigMat J = compute_jacobian(X, H, D);
            X += relax * J.fullPivLu().solve(-f);
        }

        Bv   = X.template segment<N+1>(0);
        etav = X.template segment<N+1>(N+1);
        Qv   = X(2*N+2);
        Rv   = X(2*N+3);
    }

    BigVector compute_residual(const BigVector& coeffs, Real H, Real k, Real D) {
        BigVec residual = BigVec::Zero();
        VecF Bv = X.template segment<N+1>(0),
             etav = X.template segment<N+1>(N+1);
        Real Qv = X(2*N+2), Rv = X(2*N+3), B0 = Bv(0);

        VecF um = VecF::Constant(-B0)
          + (Bv.tail(N).transpose() *
             ((kj.asDiagonal() * etav.transpose().replicate(N,1)).cwiseProduct(cos_ph))).transpose();

        VecF vm = (Bv.tail(N).transpose() *
                   ((kj.asDiagonal() * etav.transpose().replicate(N,1)).cwiseProduct(sin_ph))).transpose();

        residual.head(N+1) = VecF::Constant(-B0).cwiseProduct(etav)
                     + (Bv.tail(N).transpose() *
                        (etav.transpose().replicate(N,1).cwiseProduct(cos_ph))).transpose()
                     + VecF::Constant(Qv);

        residual.segment(N+1, N+1) = 0.5f*(um.array().square() + vm.array().square()).matrix()
                              + etav - Rv;

        residual(2*N+2) = etav.mean()*(N+1)/N - 1.0f;
        residual(2*N+3) = etav.maxCoeff() - etav.minCoeff() - H;
        return r;
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
               std::function<void(float,float,float,float,float,float)> callback) 
    {
        dt = std::clamp(timestep, 1e-5f, 0.1f);
        t = 0; x = 0; vx = 0;
        float prev_zdot = 0;

        while(t <= duration) {
            float z = wave.surface_elevation(x, t);
            float eta_x = wave.surface_slope(x, t);
            float zdot = wave.surface_time_derivative(x, t) + eta_x * vx;
            float zddot = (zdot - prev_zdot) / dt;
            callback(t, z, zdot, zddot, x, vx);
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

