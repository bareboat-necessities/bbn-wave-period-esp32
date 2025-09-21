#pragma once

/*
  Purely analytic closed-form INS EKF variant

  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  Enhancements & extensions:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file defines Kalman_INS<T,with_gyro_bias,with_accel_bias>, an extended
  error-state Kalman filter that adds linear navigation states:
     v (3)      : velocity in world frame
     p (3)      : displacement/position in world frame
     S (3)      : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction
     a_w (3)    : latent acceleration (world frame)
     j_w (3)    : latent acceleration derivative (Matérn-3/2 process)
     b_a (3)    : optional accelerometer bias

  - Attitude error + optional gyro bias block discretized analytically (θ–bias).
  - Linear subsystem [v p S a j] discretized analytically (Matérn-3/2).
  - No Van Loan, no Padé, no Eigen unsupported headers.
*/

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

#include <limits>
#include <stdexcept>
#include <cmath>

using Eigen::Matrix;

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman_INS {

    // Base error-state: attitude error (3) + optional gyro bias (3)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended states: v(3), p(3), S(3), a(3), j(3) [+ b_a(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 18 : 15;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets
    static constexpr int OFF_V  = BASE_N + 0;
    static constexpr int OFF_P  = BASE_N + 3;
    static constexpr int OFF_S  = BASE_N + 6;
    static constexpr int OFF_AW = BASE_N + 9;
    static constexpr int OFF_JW = BASE_N + 12;
    static constexpr int OFF_BA = with_accel_bias ? (BASE_N + 15) : -1;

    // Meas dim (acc + mag)
    static constexpr int M = 6;

    using Vector3    = Matrix<T,3,1>;
    using Vector6    = Matrix<T,6,1>;
    using Matrix3    = Matrix<T,3,3>;
    using MatrixM    = Matrix<T,M,M>;
    using MatrixNX   = Matrix<T,NX,NX>;
    using MatrixBaseN= Matrix<T,BASE_N,BASE_N>;

    static constexpr T half         = T(0.5);
    static constexpr T STD_GRAVITY  = T(9.80665);
    static constexpr T tempC_ref    = T(35.0);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kalman_INS(const Vector3& sigma_a, const Vector3& sigma_g, const Vector3& sigma_m,
               T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-11), T R_S_noise = T(5e-2),
               T gravity_magnitude = T(STD_GRAVITY))
      : gravity_magnitude_(gravity_magnitude),
        Rmag(sigma_m.array().square().matrix().asDiagonal()),
        R(MatrixM::Zero()),
        Racc(sigma_a.array().square().matrix().asDiagonal()),
        R_S(Matrix3::Identity() * R_S_noise),
        Qbase(initialize_Q(sigma_g, b0))
    {
        qref.setIdentity();

        Pbase.setIdentity();
        // initial base covariance
        Pbase.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;
        if constexpr (with_gyro_bias) {
            Pbase.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;
        }

        xext.setZero();
        Pext.setZero();

        // Copy base block into full covariance
        Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

        // Seed linear-state uncertainties
        const T sigma_v0 = T(1.0);
        const T sigma_p0 = T(20.0);
        const T sigma_S0 = T(50.0);
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0*sigma_v0);
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0*sigma_p0);
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0*sigma_S0);

        // Seed latent a & j
        Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat_;
        Pext.template block<3,3>(OFF_JW, OFF_JW) = Matrix3::Identity() * T(0.05);

        if constexpr (with_accel_bias) {
            Pext.template block<3,3>(OFF_BA, OFF_BA) = Matrix3::Identity() * (sigma_bacc0_*sigma_bacc0_);
        }

        // Measurement covariance (acc | mag)
        R.setZero();
        R.template topLeftCorner<3,3>()     = Racc;
        R.template bottomRightCorner<3,3>() = Rmag;
    }

    // ===== Initialization =====
    void initialize_from_acc_mag(const Vector3& acc_body, const Vector3& mag_body) {
        T anorm = acc_body.norm();
        if (anorm < T(1e-8)) throw std::runtime_error("acc vector too small");
        Vector3 acc_n = acc_body / anorm;

        // World axes in BODY coords
        Vector3 z_world = -acc_n; // world +Z (down) in body
        Vector3 mag_h   = mag_body - (mag_body.dot(z_world))*z_world;
        if (mag_h.norm() < T(1e-8)) throw std::runtime_error("mag parallel to gravity");
        mag_h.normalize();
        Vector3 x_world = mag_h;                       // world +X (north) in body
        Vector3 y_world = z_world.cross(x_world).normalized();

        Matrix3 Rwb;
        Rwb.col(0)=x_world; Rwb.col(1)=y_world; Rwb.col(2)=z_world;
        qref = Eigen::Quaternion<T>(Rwb);
        qref.normalize();

        v2ref = R_bw() * mag_body.normalized(); // store world-field unit
    }

    void initialize_from_acc(const Vector3& acc) {
        T anorm = acc.norm();
        if (anorm < T(1e-8)) throw std::runtime_error("acc vector too small");
        qref = quaternion_from_acc(acc / anorm);
        qref.normalize();
    }

    static Eigen::Quaternion<T> quaternion_from_acc(const Vector3& acc) {
        Vector3 an = acc.normalized();
        Vector3 zb = Vector3::UnitZ();
        Vector3 target = -an;

        T c = zb.dot(target);
        Vector3 axis = zb.cross(target);
        T n = axis.norm();

        if (n < T(1e-8)) {
            if (c > 0) return Eigen::Quaternion<T>::Identity();
            return Eigen::Quaternion<T>(0,1,0,0); // 180° about X
        }
        axis /= n;
        c = std::max(T(-1), std::min(T(1), c));
        T angle = std::acos(c);
        Eigen::AngleAxis<T> aa(angle, axis);
        Eigen::Quaternion<T> q(aa);
        q.normalize();
        return q;
    }

    // ===== Propagation =====
    void time_update(const Vector3& gyr, T Ts) {
        // attitude mean propagation
        Vector3 gyro_bias = Vector3::Zero();
        if constexpr (with_gyro_bias) gyro_bias = xext.template segment<3>(3);

        last_gyr_bias_corrected = gyr - gyro_bias;

        T ang = last_gyr_bias_corrected.norm() * Ts;
        Eigen::Quaternion<T> dq;
        if (ang > T(1e-9)) {
            Vector3 axis = last_gyr_bias_corrected.normalized();
            dq = Eigen::AngleAxis<T>(ang, axis);
        } else {
            dq.setIdentity();
        }
        qref = qref * dq;
        qref.normalize();

        // Full F and Q (discrete) one step (purely analytic)
        MatrixNX F_a_ext; MatrixNX Q_a_ext;
        assembleExtendedFandQ(Ts, F_a_ext, Q_a_ext);

        // Propagate linear block mean: [v p S a j] (15 states)
        using Mat15 = Eigen::Matrix<T,15,15>;
        Eigen::Matrix<T,15,1> x_lin_prev;
        x_lin_prev.template segment<3>(0)   = xext.template segment<3>(OFF_V);
        x_lin_prev.template segment<3>(3)   = xext.template segment<3>(OFF_P);
        x_lin_prev.template segment<3>(6)   = xext.template segment<3>(OFF_S);
        x_lin_prev.template segment<3>(9)   = xext.template segment<3>(OFF_AW);
        x_lin_prev.template segment<3>(12)  = xext.template segment<3>(OFF_JW);

        const Mat15 Phi_lin = F_a_ext.template block<15,15>(OFF_V, OFF_V);
        Eigen::Matrix<T,15,1> x_lin_next = Phi_lin * x_lin_prev;

        xext.template segment<3>(OFF_V)  = x_lin_next.template segment<3>(0);
        xext.template segment<3>(OFF_P)  = x_lin_next.template segment<3>(3);
        xext.template segment<3>(OFF_S)  = x_lin_next.template segment<3>(6);
        xext.template segment<3>(OFF_AW) = x_lin_next.template segment<3>(9);
        xext.template segment<3>(OFF_JW) = x_lin_next.template segment<3>(12);

        // Covariance propagation
        Pext = F_a_ext * Pext * F_a_ext.transpose() + Q_a_ext;
        Pext = T(0.5) * (Pext + Pext.transpose());

        // Mirror base covariance
        Pbase = Pext.topLeftCorner(BASE_N, BASE_N);

        // Optional drift control
        applyIntegralZeroPseudoMeas();
    }

    // ===== Measurements =====
    void measurement_update(const Vector3& acc, const Vector3& mag, T tempC = tempC_ref) {
        Vector3 v1hat = accelerometer_measurement_func(tempC);
        Vector3 v2hat = magnetometer_measurement_func();

        Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
        // accel rows
        Cext.template block<3,3>(0,0)       = -skew_symmetric_matrix(v1hat); // d f_b / d (att err)
        Cext.template block<3,3>(0,OFF_AW)  = R_wb();                        // d f_b / d a_w
        if constexpr (with_accel_bias) {
            Cext.template block<3,3>(0,OFF_BA) = Matrix3::Identity();        // d f_b / d b_acc
        }
        // mag rows
        Cext.template block<3,3>(3,0)       = -skew_symmetric_matrix(v2hat); // d m_b / d (att err)

        Vector6 yhat; yhat << v1hat, v2hat;
        Vector6 y;    y    << acc, mag;
        Vector6 inno = y - yhat;

        MatrixM S_mat = Cext * Pext * Cext.transpose() + R;
        Matrix<T, NX, M> PCt = Pext * Cext.transpose();

        Eigen::LDLT<MatrixM> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            S_mat += MatrixM::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R.norm());
            ldlt.compute(S_mat);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T, NX, M> K = PCt * ldlt.solve(MatrixM::Identity());

        xext.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext = (I - K*Cext) * Pext * (I - K*Cext).transpose() + K * R * K.transpose();
        Pext = T(0.5) * (Pext + Pext.transpose());

        applyQuaternionCorrectionFromErrorState();
        xext.template head<3>().setZero();
        Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
    }

    void measurement_update_acc_only(const Vector3& acc_meas, T tempC = tempC_ref) {
        const Vector3 v1hat = accelerometer_measurement_func(tempC);

        Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
        Cext.template block<3,3>(0,0)      = -skew_symmetric_matrix(v1hat);
        Cext.template block<3,3>(0,OFF_AW) = R_wb();
        if constexpr (with_accel_bias) {
            Cext.template block<3,3>(0,OFF_BA) = Matrix3::Identity();
        }

        Vector3 inno = acc_meas - v1hat;

        Matrix3 S_mat = Cext * Pext * Cext.transpose() + Racc;
        Matrix<T, NX, 3> PCt = Pext * Cext.transpose();

        Eigen::LDLT<Matrix3> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Racc.norm());
            ldlt.compute(S_mat);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T, NX, 3> K = PCt * ldlt.solve(Matrix3::Identity());
        xext.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext = (I - K*Cext) * Pext * (I - K*Cext).transpose() + K * Racc * K.transpose();
        Pext = T(0.5) * (Pext + Pext.transpose());

        applyQuaternionCorrectionFromErrorState();
        xext.template head<3>().setZero();
    }

    void measurement_update_mag_only(const Vector3& mag) {
        const Vector3 v2hat = magnetometer_measurement_func();

        Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
        Cext.template block<3,3>(0,0) = -skew_symmetric_matrix(v2hat);

        Vector3 inno = mag - v2hat;

        Matrix3 S_mat = Cext * Pext * Cext.transpose() + Rmag;
        Matrix<T, NX, 3> PCt = Pext * Cext.transpose();

        Eigen::LDLT<Matrix3> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Rmag.norm());
            ldlt.compute(S_mat);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T, NX, 3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext = (I - K*Cext) * Pext * (I - K*Cext).transpose() + K * Rmag * K.transpose();
        Pext = T(0.5) * (Pext + Pext.transpose());

        applyQuaternionCorrectionFromErrorState();
        xext.template head<3>().setZero();
    }

    // ===== Pseudo-measurement on S =====
    void applyIntegralZeroPseudoMeas() {
        Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
        H.template block<3,3>(0, OFF_S) = Matrix3::Identity();

        Vector3 inno = - H * xext;

        Matrix3 S_mat = H * Pext * H.transpose() + R_S;
        Matrix<T, NX, 3> PHt = Pext * H.transpose();

        Eigen::LDLT<Matrix3> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_S.norm());
            ldlt.compute(S_mat);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T, NX, 3> K = PHt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext = (I - K*H) * Pext * (I - K*H).transpose() + K * R_S * K.transpose();
        Pext = T(0.5) * (Pext + Pext.transpose());

        applyQuaternionCorrectionFromErrorState();
        xext.template head<3>().setZero();
        Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
    }

    // ===== Accessors & config =====
    Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    const MatrixBaseN& covariance_base() const { return Pbase; }
    const MatrixNX&    covariance_full() const { return Pext;  }

    Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) return xext.template segment<3>(3);
        return Vector3::Zero();
    }
    Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) return xext.template segment<3>(OFF_BA);
        return Vector3::Zero();
    }
    Vector3 get_velocity() const { return xext.template segment<3>(OFF_V); }
    Vector3 get_position() const { return xext.template segment<3>(OFF_P); }
    Vector3 get_integral_displacement() const { return xext.template segment<3>(OFF_S); }
    Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }
    Vector3 get_world_jerk()  const { return xext.template segment<3>(OFF_JW); }

    void set_mag_world_ref(const Vector3& B_world) { v2ref = B_world.normalized(); }
    void set_Racc(const Vector3& sigma_acc) {
        Racc = sigma_acc.array().square().matrix().asDiagonal();
        R.template topLeftCorner<3,3>() = Racc;
    }
    void set_Rmag(const Vector3& sigma_mag) {
        Matrix3 Rmag_new = sigma_mag.array().square().matrix().asDiagonal();
        Rmag = Rmag_new;
        R.template bottomRightCorner<3,3>() = Rmag_new;
    }
    void set_RS_noise(const Vector3& sigma_S) {
        R_S = sigma_S.array().square().matrix().asDiagonal();
    }

    // Matérn-3/2 latent tuning
    void set_aw_time_constant(T tau_seconds) { tau_lat_ = std::max(T(1e-3), tau_seconds); }
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat_ = std_aw.array().square().matrix().asDiagonal();
    }

    // Accel bias (optional)
    void set_initial_acc_bias(const Vector3& b0) {
        if constexpr (with_accel_bias) xext.template segment<3>(OFF_BA) = b0;
    }
    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) sigma_bacc0_ = std::max(T(0), s);
    }
    void set_Q_bacc_rw(const Vector3& rw_std_per_sqrt_s) {
        if constexpr (with_accel_bias)
            Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
    }
    void set_accel_bias_temp_coeff(const Vector3& ka_per_degC) { k_a_ = ka_per_degC; }
    void set_initial_jerk_std(T s) {
        Pext.template block<3,3>(OFF_JW, OFF_JW) = Matrix3::Identity() * (s*s);
    }
    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0*sigma_v0);
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0*sigma_p0);
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0*sigma_S0);
    }

  private:
    // ===== Internals =====
    const T gravity_magnitude_;

    // Orientation & refs
    Eigen::Quaternion<T> qref;         // world→body
    Vector3 v2ref = Vector3::UnitX();  // magnetic field in world frame (unit)

    // Covariances
    MatrixBaseN Pbase;
    MatrixNX    Pext;

    // State (error-state form)
    Matrix<T, NX, 1> xext;   // [att_err(3), (gyro bias 3), v(3), p(3), S(3), a(3), j(3), (b_acc 3)]

    // last corrected gyro
    Vector3 last_gyr_bias_corrected{};

    // Accel bias model (optional)
    T       sigma_bacc0_ = T(0.1);
    Matrix3 Q_bacc_      = Matrix3::Identity() * T(1e-8);
    Vector3 k_a_         = Vector3::Constant(T(0.003)); // m/s² per °C

    // Measurement covariances
    Matrix3 Rmag;
    MatrixM R;
    Matrix3 Racc;
    Matrix3 R_S;

    // Base process for attitude/bias
    MatrixBaseN Qbase;

    // Latent acceleration Matérn-3/2 params
    T       tau_lat_       = T(1.5);                              // correlation time [s]
    Matrix3 Sigma_aw_stat_ = Matrix3::Identity() * T(0.25*0.25);  // Var[a] diag [(m/s²)²]

    // Helpers
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }              // world→body
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }  // body→world

    static MatrixBaseN initialize_Q(const Vector3& sigma_g, T b0) {
        MatrixBaseN Q; Q.setZero();
        if constexpr (with_gyro_bias) {
            Q.template topLeftCorner<3,3>()  = sigma_g.array().square().matrix().asDiagonal(); // gyro noise density
            Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * b0;                    // gyro bias RW power
        } else {
            Q = sigma_g.array().square().matrix().asDiagonal();
        }
        return Q;
    }

    Matrix3 skew_symmetric_matrix(const Vector3& v) const {
        Matrix3 M;
        M << 0, -v(2), v(1),
             v(2), 0, -v(0),
            -v(1), v(0), 0;
        return M;
    }

    Vector3 accelerometer_measurement_func(T tempC) const {
        const Vector3 g_world(0,0,+gravity_magnitude_);
        const Vector3 aw = xext.template segment<3>(OFF_AW);

        Vector3 fb = R_wb() * (aw - g_world);
        if constexpr (with_accel_bias) {
            Vector3 ba0 = xext.template segment<3>(OFF_BA);
            Vector3 ba  = ba0 + k_a_ * (tempC - tempC_ref);
            fb += ba;
        }
        return fb;
    }

    Vector3 magnetometer_measurement_func() const {
        return R_wb() * v2ref;
    }

    void applyQuaternionCorrectionFromErrorState() {
        Eigen::Quaternion<T> corr(T(1), half*xext(0), half*xext(1), half*xext(2));
        corr.normalize();
        qref = qref * corr;
        qref.normalize();
    }

    // ===== Build F and Q (discrete) — analytic only =====
    void assembleExtendedFandQ(T Ts, MatrixNX& F_a_ext, MatrixNX& Q_a_ext) {
        F_a_ext.setIdentity();
        Q_a_ext.setZero();

        // --- attitude error transition exp(-[w]_x h), analytic Rodrigues ---
        Matrix3 I3 = Matrix3::Identity();
        Vector3 w = last_gyr_bias_corrected;
        T omega = w.norm();
        T h = Ts;

        if (omega < T(1e-12)) {
            // series up to h^2 (stable for tiny rates)
            Matrix3 Wx = skew_symmetric_matrix(w);
            F_a_ext.template block<3,3>(0,0) = I3 - Wx*h + (Wx*Wx)*(h*h/T(2));
        } else {
            Vector3 u = w / omega;
            Matrix3 K = skew_symmetric_matrix(u);
            T th = omega * h;
            T s = std::sin(th), c = std::cos(th);
            F_a_ext.template block<3,3>(0,0) = I3 - (s)*K + (T(1)-c)*(K*K);
        }

        // attitude error depends on gyro bias: θ_{k+1} ≈ θ_k - ∫ exp(-[w] τ) dτ * b
        if constexpr (with_gyro_bias) {
            Matrix3 Phi_tb;
            // J = ∫_0^h exp(-[w] τ) dτ; Phi_tb = -J
            if (omega < T(1e-12)) {
                Matrix3 Wx = skew_symmetric_matrix(w);
                Matrix3 Wx2 = Wx*Wx;
                Matrix3 J = I3*h - Wx*(h*h/T(2)) + Wx2*(h*h*h/T(6));
                Phi_tb = -J;
            } else {
                Vector3 u = w / omega;
                Matrix3 K = skew_symmetric_matrix(u);
                T th = omega*h;
                T c = std::cos(th), s = std::sin(th);

                Matrix3 J = I3*h
                          + K * ((c - T(1)) / omega)
                          + (K*K) * (h - s/omega);
                Phi_tb = -J;
            }
            F_a_ext.template block<3,3>(0,3) = Phi_tb;
            F_a_ext.template block<3,3>(3,3) = I3; // bias is RW: b_{k+1} = b_k + noise
        }

        // --- process noise for base block (analytic, small-Δt-consistent) ---
        // Closed form for Q_d of the coupled θ–b system is lengthy; we use a robust
        // first-order accurate mapping that matches continuous spectra:
        //   Qθθ ≈ Sg*h, Qbb ≈ Sbg*h, Qθb ≈ 0
        if constexpr (with_gyro_bias) {
            Q_a_ext.template block<3,3>(0,0) = Qbase.template topLeftCorner<3,3>() * h;
            Q_a_ext.template block<3,3>(3,3) = Qbase.template block<3,3>(3,3) * h;
        } else {
            Q_a_ext.template block<3,3>(0,0) = Qbase * h;
        }

        // --- linear subsystem [v p S a j] (15 states across 3 axes), Matérn-3/2 ---
        using Mat15 = Eigen::Matrix<T,15,15>;
        Mat15 Phi_lin, Qd_lin;
        assembleLinearBlock15x15(std::max(T(1e-6), tau_lat_), h, Sigma_aw_stat_, Phi_lin, Qd_lin);

        F_a_ext.block(OFF_V, OFF_V, 15,15) = Phi_lin;
        Q_a_ext.block(OFF_V, OFF_V, 15,15) = Qd_lin;

        // --- accelerometer bias random walk (if enabled) ---
        if constexpr (with_accel_bias) {
            // state transition already I at construction; add noise power
            Q_a_ext.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * h;
        }

        // numerical hygiene
        Q_a_ext = T(0.5) * (Q_a_ext + Q_a_ext.transpose());
    }

    // ===== Analytic per-axis Φ and Qd for [v p S a j] with Matérn-3/2 =====
    static void PhiAxis5x1_analytic(T tau, T h, Eigen::Matrix<T,5,5>& Phi) {
        using std::exp;
        const T inv_tau = T(1) / std::max(T(1e-12), tau);
        const T alpha   = exp(-h * inv_tau);

        // Short-time integrals
        const T C0 = h;
        const T C1 = T(0.5) * h*h;

        // Exp-weighted primitives over [0,h] (appear in chained integrals)
        const T A0 = tau * (T(1) - alpha);
        const T A1 = tau*tau * (T(1) - alpha) - tau * h * alpha;
        const T A2 = T(2)*tau*tau*tau * (T(1) - alpha) - tau * h * (h + T(2)*tau) * alpha;

        // Helper coeffs for couplings from a/j into v/p/S
        const T phi_va = tau * (T(1) - alpha) + h * alpha;                  // v ← a
        const T phi_vj = tau * (h - tau * (T(1) - alpha));                  // v ← j
        const T phi_pa = C0 * phi_va - A1;                                  // p ← a
        const T phi_pj = C0 * phi_vj - A0 * tau;                            // p ← j
        const T phi_Sa = C1 * phi_va - C0 * A1 + A2;                        // S ← a
        const T phi_Sj = C1 * phi_vj - C0 * A0 * tau + A1 * tau;            // S ← j

        Phi.setZero();
        // Kinematics (v,p,S)
        Phi(0,0) = T(1);
        Phi(1,0) = C0;    Phi(1,1) = T(1);
        Phi(2,0) = C1;    Phi(2,1) = C0;    Phi(2,2) = T(1);

        // Latent [a j] exact exponential (2x2 underdamped OU of order 2)
        Phi(3,3) = alpha * (T(1) + h * inv_tau);
        Phi(3,4) = tau * (T(1) - alpha);
        Phi(4,3) = -(h * alpha) * (inv_tau * inv_tau);
        Phi(4,4) = alpha * (T(1) - h * inv_tau);

        // Couplings
        Phi(0,3) = phi_va;  Phi(0,4) = phi_vj;
        Phi(1,3) = phi_pa;  Phi(1,4) = phi_pj;
        Phi(2,3) = phi_Sa;  Phi(2,4) = phi_Sj;
    }

    static void QdAxis5x1_analytic(T tau, T h, T sigma2_a, Eigen::Matrix<T,5,5>& Qd) {
        using std::exp;
        const T inv_tau = T(1) / std::max(T(1e-12), tau);

        // Basic polynomial integrals
        const T C0 = h;
        const T C1 = T(0.5)*h*h;
        const T C2 = (h*h*h)/T(3);
        const T C3 = (h*h*h*h)/T(4);
        const T C4 = (h*h*h*h*h)/T(5);

        const T alpha  = exp(-h*inv_tau);
        const T A0 = tau*(T(1)-alpha);
        const T A1 = tau*tau*(T(1)-alpha) - tau*h*alpha;
        const T A2 = T(2)*tau*tau*tau*(T(1)-alpha) - tau*h*(h + T(2)*tau)*alpha;
        const T A3 = T(6)*tau*tau*tau*tau*(T(1)-alpha)
                   - tau*h*(h*h + T(3)*h*tau + T(6)*tau*tau)*alpha;

        const T alpha2 = exp(-T(2)*h*inv_tau);
        const T B0 = (tau/T(2)) * (T(1)-alpha2);
        const T B1 = (tau*tau/T(4)) * (T(1) - alpha2*(T(1) + T(2)*h*inv_tau));
        const T B2 = (tau*tau*tau/T(4)) * (T(1) - alpha2*(T(1) + T(2)*h*inv_tau + T(2)*h*h*inv_tau*inv_tau));

        // White-noise power to make Var[a] = sigma2_a at stationarity for Matérn-3/2: qc = 4*sigma2/τ^3
        const T qc = (T(4)*sigma2_a) * inv_tau*inv_tau*inv_tau;

        Eigen::Matrix<T,5,5> K; K.setZero();

        // --- a-j 2x2 sub-block ---
        const T K_jj = B0;
        const T K_aj = A0 - B0;
        const T K_aa = C0 - T(2)*A0 + B0;

        // --- couplings with v ---
        const T K_vj = tau * (A0 - B0);
        const T K_va = tau * (C0 - T(2)*A0 + B0);
        const T K_vv = tau*tau * (C0 - T(2)*A0 + B0);

        // --- couplings with p ---
        const T K_pj = tau * (A1 - B1);
        const T K_pa = tau * (C1 - T(2)*A1 + B1);
        const T K_pv = tau*tau * (C1 - T(2)*A1 + B1);
        const T K_pp = tau*tau * (C2 - T(2)*A2 + B2);

        // --- couplings with S ---
        const T K_Sj = tau * (A2 - B2);
        const T K_Sa = tau * (C2 - T(2)*A2 + B2);
        const T K_Sv = tau*tau * (C2 - T(2)*A2 + B2);

        // A compact closed form for K_Sp and K_SS is lengthy; below are
        // algebraically reduced expressions consistent with the hierarchy:
        const T K_Sp =
            tau*tau * ( C3
                      - T(2)*A3
                      + (tau*tau*tau*T(3)/T(8)) * (
                          (T(8)/(T(3)*tau*tau*tau)) *
                          ((tau*tau*tau/T(3)) - alpha2 * ((tau*tau*tau/T(3)) + tau*tau*h + tau*h*h + (T(2)/T(3))*h*h*h*inv_tau))
                        )
                      );

        const T K_SS =
            (T(0.25))*tau*tau*C4
          - tau*tau*tau*C3
          + T(2)*tau*tau*tau*tau*C2
          - T(2)*tau*tau*tau*tau*tau*C1
          + tau*tau*tau*tau*tau*tau*C0
          - tau*tau*tau*tau*A2 + T(2)*tau*tau*tau*tau*tau*A1 - T(2)*tau*tau*tau*tau*tau*tau*A0
          + tau*tau*tau*tau*tau*tau*B0;

        // Upper triangle placement
        K(0,0)=K_vv;  K(0,1)=K_pv;  K(0,2)=K_Sv;  K(0,3)=K_va;  K(0,4)=K_vj;
        K(1,1)=K_pp;  K(1,2)=K_Sp;  K(1,3)=K_pa;  K(1,4)=K_pj;
        K(2,2)=K_SS;  K(2,3)=K_Sa;  K(2,4)=K_Sj;
        K(3,3)=K_aa;  K(3,4)=K_aj;
        K(4,4)=K_jj;

        // mirror symmetry
        for (int i=0;i<5;++i)
            for (int j=i+1;j<5;++j) K(j,i)=K(i,j);

        Qd = qc * K;
    }

    static void assembleLinearBlock15x15(
        T tau, T Ts, const Matrix3& Sigma_aw_stat,
        Eigen::Matrix<T,15,15>& Phi_lin, Eigen::Matrix<T,15,15>& Qd_lin)
    {
        Phi_lin.setZero();
        Qd_lin.setZero();

        for (int axis=0; axis<3; ++axis) {
            T sigma2 = Sigma_aw_stat(axis,axis);

            Eigen::Matrix<T,5,5> Phi_ax, Qd_ax;
            PhiAxis5x1_analytic(tau, Ts, Phi_ax);
            QdAxis5x1_analytic (tau, Ts, sigma2, Qd_ax);

            // order in 15x15: interleave axis across each of 5 stacked states
            int idx[5] = {0,3,6,9,12};
            for (int i=0;i<5;++i)
                for (int j=0;j<5;++j) {
                    Phi_lin(idx[i]+axis, idx[j]+axis) = Phi_ax(i,j);
                    Qd_lin (idx[i]+axis, idx[j]+axis) = Qd_ax (i,j);
                }
        }
    }
};

