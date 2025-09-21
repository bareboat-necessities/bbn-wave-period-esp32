#pragma once
/*
  Copyright 2025, Mikhail Grushinskiy
  All-analytic MEKF + Matérn-3/2 INS extension (header-only)

  - No Van Loan, no Padé, no matrix .exp(): Φ and Qd are derived together.
  - Latent acceleration follows Matérn-3/2 (critically damped 2nd order GM):
        ȧ = j
        j̇ = -(2/τ) j - (1/τ²) a + w,   with  E[w wᵀ] = q_c δ(t-t'),  q_c = 4 σ_a² / τ³
  - Linear kinematics (per axis):
        v̇ = a ,  ṗ = v ,  Ṡ = p
  - The 5x5 axis block is x=[v p S a j]ᵀ. Φ(h)=exp(Ah) is closed-form,
    and Qd(h)=∫₀ʰ Φ(s) G q_c Gᵀ Φ(s)ᵀ ds is built from analytic primitives.

  Public API matches the previous "Kalman_INS" sketch you shared.
*/

#include <limits>
#include <stdexcept>
#include <cmath>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

using Eigen::Matrix;

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman_INS {

    // Base: attitude error (3) + optional gyro bias (3)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended: v(3), p(3), S(3), a(3), j(3) [+ optional b_a(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 18 : 15;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets
    static constexpr int OFF_V  = BASE_N + 0;
    static constexpr int OFF_P  = BASE_N + 3;
    static constexpr int OFF_S  = BASE_N + 6;
    static constexpr int OFF_A  = BASE_N + 9;    // a
    static constexpr int OFF_J  = BASE_N + 12;   // j
    static constexpr int OFF_BA = with_accel_bias ? (BASE_N + 15) : -1;

    // Measurements: accel + mag
    static constexpr int M = 6;

    using Vector3    = Matrix<T,3,1>;
    using Vector4    = Matrix<T,4,1>;
    using Vector6    = Matrix<T,6,1>;
    using Matrix3    = Matrix<T,3,3>;
    using Matrix5    = Matrix<T,5,5>;
    using Matrix15   = Matrix<T,15,15>;
    using MatrixM    = Matrix<T,M,M>;
    using MatrixNX   = Matrix<T,NX,NX>;
    using MatrixBase = Matrix<T,BASE_N,BASE_N>;

    static constexpr T half = T(0.5);
    static constexpr T STD_GRAV = T(9.80665);
    static constexpr T TEMP_REF = T(35.0);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kalman_INS(const Vector3& sigma_a, const Vector3& sigma_g, const Vector3& sigma_m,
               T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-11), T R_S_noise = T(5e-2),
               T gravity_magnitude = T(STD_GRAV))
      : gravity_magnitude_(gravity_magnitude),
        Racc_(sigma_a.array().square().matrix().asDiagonal()),
        Rmag_(sigma_m.array().square().matrix().asDiagonal()),
        Qbase_(initialize_Q_(sigma_g, b0))
    {
        qref_.setIdentity();
        Pbase_.setIdentity();

        Pbase_.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;
        if constexpr (with_gyro_bias) {
            Pbase_.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;
        }

        xext_.setZero();
        Pext_.setZero();
        Pext_.topLeftCorner(BASE_N, BASE_N) = Pbase_;

        // Seed linear states
        set_initial_linear_uncertainty(T(1.0), T(20.0), T(50.0));

        // Latent acceleration stationary variance seed
        Pext_.template block<3,3>(OFF_A, OFF_A) = Sigma_a_stat_;
        Pext_.template block<3,3>(OFF_J, OFF_J) = Matrix3::Identity() * T(0.05);

        if constexpr (with_accel_bias) {
            Pext_.template block<3,3>(OFF_BA, OFF_BA) =
                Matrix3::Identity() * (sigma_bacc0_ * sigma_bacc0_);
        }

        R_.setZero();
        R_.template topLeftCorner<3,3>() = Racc_;
        R_.template bottomRightCorner<3,3>() = Rmag_;

        R_S_ = Matrix3::Identity() * R_S_noise;
    }

    // ========== Initialization ==========
    void initialize_from_acc_mag(const Vector3& acc_body, const Vector3& mag_body) {
        T an = acc_body.norm();
        if (an < T(1e-8)) throw std::runtime_error("acc vector too small");
        Vector3 acc_n = acc_body / an;

        Vector3 z_world = -acc_n; // +Z down
        Vector3 mag_h   = mag_body - (mag_body.dot(z_world))*z_world;
        if (mag_h.norm() < T(1e-8)) throw std::runtime_error("mag parallel to gravity");
        mag_h.normalize();
        Vector3 x_world = mag_h;                       // +X north
        Vector3 y_world = z_world.cross(x_world).normalized();

        Matrix3 Rwb;
        Rwb.col(0)=x_world; Rwb.col(1)=y_world; Rwb.col(2)=z_world;
        qref_ = Eigen::Quaternion<T>(Rwb);
        qref_.normalize();

        v2ref_ = R_bw_() * mag_body.normalized(); // store world field unit
    }

    void initialize_from_acc(const Vector3& acc) {
        T an = acc.norm();
        if (an < T(1e-8)) throw std::runtime_error("acc vector too small");
        qref_ = quaternion_from_acc(acc / an);
        qref_.normalize();
    }

    static Eigen::Quaternion<T> quaternion_from_acc(const Vector3& acc) {
        Vector3 an = acc.normalized();
        Vector3 zb = Vector3::UnitZ();
        Vector3 tgt = -an;
        T c = zb.dot(tgt);
        Vector3 axis = zb.cross(tgt);
        T n = axis.norm();
        if (n < T(1e-8)) {
            if (c > 0) return Eigen::Quaternion<T>::Identity();
            return Eigen::Quaternion<T>(0,1,0,0);
        }
        axis /= n;
        c = std::max(T(-1), std::min(T(1), c));
        T ang = std::acos(c);
        Eigen::AngleAxis<T> aa(ang, axis);
        Eigen::Quaternion<T> q(aa);
        q.normalize();
        return q;
    }

    // ========== Config ==========
    void set_mag_world_ref(const Vector3& B_world) { v2ref_ = B_world.normalized(); }

    void set_Racc(const Vector3& sigma_acc) {
        Racc_ = sigma_acc.array().square().matrix().asDiagonal();
        R_.template topLeftCorner<3,3>() = Racc_;
    }

    void set_Rmag(const Vector3& sigma_mag) {
        Matrix3 Rmag_new = sigma_mag.array().square().matrix().asDiagonal();
        Rmag_ = Rmag_new;
        R_.template bottomRightCorner<3,3>() = Rmag_new;
    }

    void set_RS_noise(const Vector3& sigma_S) {
        R_S_ = sigma_S.array().square().matrix().asDiagonal();
    }

    void set_aw_time_constant(T tau_seconds) { tau_lat_ = std::max(T(1e-3), tau_seconds); }

    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_a_stat_ = std_aw.array().square().matrix().asDiagonal();   // Var[a] per axis
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext_.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0*sigma_v0);
        Pext_.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0*sigma_p0);
        Pext_.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0*sigma_S0);
    }

    void set_initial_acc_bias(const Vector3& b0) {
        if constexpr (with_accel_bias) xext_.template segment<3>(OFF_BA) = b0;
    }
    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) sigma_bacc0_ = std::max(T(0), s);
    }
    void set_Q_bacc_rw(const Vector3& rw_std_per_sqrt_s) {
        if constexpr (with_accel_bias) Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
    }
    void set_accel_bias_temp_coeff(const Vector3& ka_per_degC) { k_a_ = ka_per_degC; }

    void set_initial_jerk_std(T s) {
        Pext_.template block<3,3>(OFF_J, OFF_J) = Matrix3::Identity() * (s*s);
    }

    // ========== Accessors ==========
    Eigen::Quaternion<T> quaternion() const { return qref_.conjugate(); }
    const MatrixBase& covariance_base() const { return Pbase_; }
    const MatrixNX&   covariance_full() const { return Pext_;  }

    Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) return xext_.template segment<3>(3);
        else                           return Vector3::Zero();
    }
    Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) return xext_.template segment<3>(OFF_BA);
        else                            return Vector3::Zero();
    }
    Vector3 get_velocity() const { return xext_.template segment<3>(OFF_V); }
    Vector3 get_position() const { return xext_.template segment<3>(OFF_P); }
    Vector3 get_integral_displacement() const { return xext_.template segment<3>(OFF_S); }
    Vector3 get_world_accel() const { return xext_.template segment<3>(OFF_A); }
    Vector3 get_world_jerk()  const { return xext_.template segment<3>(OFF_J); }

    // ========== Cycle ==========
    void time_update(const Vector3& gyr, T Ts) {
        // attitude mean propagation (right-multiply)
        Vector3 gb = Vector3::Zero();
        if constexpr (with_gyro_bias) gb = xext_.template segment<3>(3);
        last_gyr_bias_corrected_ = gyr - gb;

        T ang = last_gyr_bias_corrected_.norm() * Ts;
        Eigen::Quaternion<T> dq;
        if (ang > T(1e-9)) {
            Vector3 axis = last_gyr_bias_corrected_.normalized();
            dq = Eigen::AngleAxis<T>(ang, axis);
        } else dq.setIdentity();
        qref_ = qref_ * dq;
        qref_.normalize();

        // Build all-analytic discrete Φ and Qd
        MatrixNX F, Qd;
        assembleExtendedFandQ_(Ts, F, Qd);

        // Mean propagation for linear block [v p S a j] per axis (3× of 5 states)
        Eigen::Matrix<T,15,1> xl;
        xl.template segment<3>(0)  = xext_.template segment<3>(OFF_V);
        xl.template segment<3>(3)  = xext_.template segment<3>(OFF_P);
        xl.template segment<3>(6)  = xext_.template segment<3>(OFF_S);
        xl.template segment<3>(9)  = xext_.template segment<3>(OFF_A);
        xl.template segment<3>(12) = xext_.template segment<3>(OFF_J);

        Eigen::Matrix<T,15,1> xl_next = F.template block<15,15>(OFF_V, OFF_V) * xl;

        xext_.template segment<3>(OFF_V) = xl_next.template segment<3>(0);
        xext_.template segment<3>(OFF_P) = xl_next.template segment<3>(3);
        xext_.template segment<3>(OFF_S) = xl_next.template segment<3>(6);
        xext_.template segment<3>(OFF_A) = xl_next.template segment<3>(9);
        xext_.template segment<3>(OFF_J) = xl_next.template segment<3>(12);

        // Covariance propagation
        Pext_ = F * Pext_ * F.transpose() + Qd;
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        // Mirror base
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);

        // Optional S drift control
        applyIntegralZeroPseudoMeas();
    }

    void measurement_update(const Vector3& acc, const Vector3& mag, T tempC = TEMP_REF) {
        Vector3 v1hat = accelerometer_measurement_func_(tempC);
        Vector3 v2hat = magnetometer_measurement_func_();

        Matrix<T,M,NX> C = Matrix<T,M,NX>::Zero();
        // accel rows
        C.template block<3,3>(0,0)  = -skew_(v1hat);
        C.template block<3,3>(0,OFF_A) = R_wb_();
        if constexpr (with_accel_bias) C.template block<3,3>(0,OFF_BA) = Matrix3::Identity();
        // mag rows
        C.template block<3,3>(3,0)  = -skew_(v2hat);

        Vector6 yhat; yhat << v1hat, v2hat;
        Vector6 y;    y    << acc, mag;
        Vector6 inno = y - yhat;

        MatrixM S = C * Pext_ * C.transpose() + R_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<MatrixM> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += MatrixM::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,M> K = PCt * ldlt.solve(MatrixM::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * R_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);
    }

    void measurement_update_acc_only(const Vector3& acc_meas, T tempC = TEMP_REF) {
        const Vector3 v1hat = accelerometer_measurement_func_(tempC);

        Matrix<T,3,NX> C = Matrix<T,3,NX>::Zero();
        C.template block<3,3>(0,0)    = -skew_(v1hat);
        C.template block<3,3>(0,OFF_A)= R_wb_();
        if constexpr (with_accel_bias) C.template block<3,3>(0,OFF_BA) = Matrix3::Identity();

        Vector3 inno = acc_meas - v1hat;

        Matrix3 S = C * Pext_ * C.transpose() + Racc_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Racc_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * Racc_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
    }

    void measurement_update_mag_only(const Vector3& mag) {
        const Vector3 v2hat = magnetometer_measurement_func_();

        Matrix<T,3,NX> C = Matrix<T,3,NX>::Zero();
        C.template block<3,3>(0,0) = -skew_(v2hat);

        Vector3 inno = mag - v2hat;

        Matrix3 S = C * Pext_ * C.transpose() + Rmag_;
        auto PCt  = Pext_ * C.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Rmag_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*C) * Pext_ * (I - K*C).transpose() + K * Rmag_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
    }

    // ========== Pseudo-measurement on S ==========
    void applyIntegralZeroPseudoMeas() {
        Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
        H.template block<3,3>(0, OFF_S) = Matrix3::Identity();

        Vector3 inno = - H * xext_;

        Matrix3 S = H * Pext_ * H.transpose() + R_S_;
        auto PHt  = Pext_ * H.transpose();

        Eigen::LDLT<Matrix3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            S += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_S_.norm());
            ldlt.compute(S);
            if (ldlt.info() != Eigen::Success) return;
        }
        Matrix<T,NX,3> K = PHt * ldlt.solve(Matrix3::Identity());

        xext_.noalias() += K * inno;

        MatrixNX I = MatrixNX::Identity();
        Pext_ = (I - K*H) * Pext_ * (I - K*H).transpose() + K * R_S_ * K.transpose();
        Pext_ = T(0.5) * (Pext_ + Pext_.transpose());

        applyQuaternionCorrection_();
        xext_.template head<3>().setZero();
        Pbase_ = Pext_.topLeftCorner(BASE_N, BASE_N);
    }

  private:
    // ===== Members =====
    const T gravity_magnitude_;

    Eigen::Quaternion<T> qref_;      // world→body
    Vector3 v2ref_ = Vector3::UnitX();

    MatrixBase Pbase_;
    MatrixNX   Pext_;
    Matrix<T,NX,1> xext_;

    Vector3 last_gyr_bias_corrected_{};

    // accel-bias (optional)
    T       sigma_bacc0_ = T(0.1);
    Matrix3 Q_bacc_      = Matrix3::Identity() * T(1e-8);
    Vector3 k_a_         = Vector3::Constant(T(0.003)); // m/s² per °C

    // measurements
    Matrix3  Racc_;
    Matrix3  Rmag_;
    MatrixM  R_;
    Matrix3  R_S_;

    // base process
    MatrixBase Qbase_;

    // latent a Matérn-3/2 params
    T       tau_lat_       = T(1.5);                             // [s]
    Matrix3 Sigma_a_stat_  = Matrix3::Identity() * T(0.25*0.25); // Var[a]

    // ===== Utils =====
    Matrix3 R_wb_() const { return qref_.toRotationMatrix(); }              // world→body
    Matrix3 R_bw_() const { return qref_.toRotationMatrix().transpose(); }  // body→world

    static MatrixBase initialize_Q_(const Vector3& sigma_g, T b0) {
        MatrixBase Q; Q.setZero();
        if constexpr (with_gyro_bias) {
            Q.template topLeftCorner<3,3>()    = sigma_g.array().square().matrix().asDiagonal();
            Q.template bottomRightCorner<3,3>()= Matrix3::Identity() * b0;
        } else {
            Q = sigma_g.array().square().matrix().asDiagonal();
        }
        return Q;
    }

    Matrix3 skew_(const Vector3& v) const {
        Matrix3 M;
        M << 0, -v(2), v(1),
             v(2), 0, -v(0),
            -v(1), v(0), 0;
        return M;
    }

    Vector3 accelerometer_measurement_func_(T tempC) const {
        const Vector3 g_world(0,0,+gravity_magnitude_);
        const Vector3 a = xext_.template segment<3>(OFF_A);
        Vector3 fb = R_wb_() * (a - g_world);
        if constexpr (with_accel_bias) {
            Vector3 ba0 = xext_.template segment<3>(OFF_BA);
            Vector3 ba  = ba0 + k_a_ * (tempC - TEMP_REF);
            fb += ba;
        }
        return fb;
    }

    Vector3 magnetometer_measurement_func_() const {
        return R_wb_() * v2ref_;
    }

    void applyQuaternionCorrection_() {
        Eigen::Quaternion<T> corr(T(1),
                                  half * xext_(0),
                                  half * xext_(1),
                                  half * xext_(2));
        corr.normalize();
        qref_ = qref_ * corr;
        qref_.normalize();
    }

    // ===== Analytic primitives =====
    // ∫₀ʰ s^m e^{-λ s} ds, with λ ≥ 0
    static T integ_pow_exp_(int m, T lambda, T h) {
        if (lambda <= T(0)) {
            // ∫ s^m ds
            return std::pow(h, T(m+1)) / T(m+1);
        }
        // m! / λ^{m+1} * [ 1 - e^{-λh} ∑_{k=0}^m (λh)^k/k! ]
        T fact = T(1);
        for (int i=2;i<=m;i++) fact *= T(i);
        T lam_pow = std::pow(lambda, T(m+1));
        // truncated series S_m(x) = sum_{k=0}^m x^k/k!
        T x = lambda * h;
        T Sm = T(1), term = T(1);
        for (int k=1;k<=m;k++) {
            term *= x / T(k);
            Sm += term;
        }
        return (fact/lam_pow) * (T(1) - std::exp(-x)*Sm);
    }

    // Build axis Φ(h) and the "column response" g(s)=Φ(s)G for j-noise;
    // then Qd = q_c ∫ g(s) g(s)ᵀ ds via termwise analytic integration.
    static void axis_Phi_and_Qd_(T h, T tau, T sigma2_a,
                                 Matrix5& Phi, Matrix5& Qd)
    {
        using Term = struct { T c; int n; T lam; }; // c * s^n * exp(-lam s), lam ∈ {0, 1/τ}

        auto alpha = std::exp(-h / tau);

        // ---- exact latent 2×2 exponential (a,j) with critical damping ----
        // a(h) = e^{-h/τ}[(1 + h/τ)a0 + h j0]
        // j(h) = e^{-h/τ}[      - (h/τ²)a0 + (1 - h/τ) j0]
        Phi.setZero();
        // kinematics base
        Phi(0,0)=T(1);               // v<-v
        Phi(1,0)=h;                  // p<-v
        Phi(1,1)=T(1);               // p<-p
        Phi(2,0)=T(0.5)*h*h;         // S<-v
        Phi(2,1)=h;                  // S<-p
        Phi(2,2)=T(1);               // S<-S

        // helper En(h): E0=1-α; E1=1-α(1+h/τ); E2=1-α(1+h/τ+(h²)/(2τ²)); E3=...
        auto inv_tau = T(1)/tau;
        auto E0 = T(1) - alpha;
        auto E1 = T(1) - alpha*(T(1) + h*inv_tau);
        auto E2 = T(1) - alpha*(T(1) + h*inv_tau + (h*h)*inv_tau*inv_tau/T(2));
        auto E3 = T(1) - alpha*(T(1) + h*inv_tau + (h*h)*inv_tau*inv_tau/T(2)
                              + (h*h*h)*inv_tau*inv_tau*inv_tau/T(6));

        // Couplings (derived from integrals of a(s), j(s))
        // v <- a0, j0
        T I_a = tau*(E0 + E1);             // ∫ (1 + s/τ) e^{-s/τ} ds
        T I_j = tau*tau*E1;                // ∫ s e^{-s/τ} ds
        Phi(0,3) = I_a;                    // v<-a
        Phi(0,4) = I_j;                    // v<-j

        // p <- a0, j0
        T J_a = h*I_a - tau*tau*(E1 + T(2)*E2);
        T J_j = h*(tau*tau*E1) - T(2)*tau*tau*tau*E2;
        Phi(1,3) = J_a;
        Phi(1,4) = J_j;

        // S <- a0, j0
        T Sa  = T(0.5)*( h*h*tau*(E0+E1)
                       - T(2)*h*tau*tau*(E1+T(2)*E2)
                       + T(2)*tau*tau*tau*(E2+T(3)*E3) );
        T Sj  = T(0.5)*( h*h*(tau*tau*E1)
                       - T(4)*h*(tau*tau*tau*E2)
                       + T(6)*tau*tau*tau*tau*E3 );
        Phi(2,3) = Sa;
        Phi(2,4) = Sj;

        // latent 2×2
        Phi(3,3) = alpha * (T(1) + h*inv_tau);  // a<-a
        Phi(3,4) = alpha * h;                   // a<-j
        Phi(4,3) = alpha * (-h*inv_tau*inv_tau);// j<-a
        Phi(4,4) = alpha * (T(1) - h*inv_tau);  // j<-j

        // ---- build g(s) terms for each state (response to initial j0=1) ----
        // We express each g_i(s) as sum of Term {c, n, λ} with λ ∈ {0, 1/τ}
        auto push = [](auto& vec, T c, int n, T lam){ vec.emplace_back(Term{c,n,lam}); };

        // g_v(s) = τ² - τ² e^{-s/τ} - τ s e^{-s/τ}
        std::array<std::vector<Term>,5> g;
        g[0].reserve(3); g[1].reserve(4); g[2].reserve(4); g[3].reserve(2); g[4].reserve(2);

        push(g[0],  tau*tau, 0, T(0));
        push(g[0], -tau*tau, 0, inv_tau);
        push(g[0], -tau,     1, inv_tau);

        // g_p(s) = τ² s - 2τ³ + e^{-s/τ}(2τ³ + τ² s)
        push(g[1],  tau*tau, 1, T(0));
        push(g[1], -T(2)*tau*tau*tau, 0, T(0));
        push(g[1],  T(2)*tau*tau*tau, 0, inv_tau);
        push(g[1],  tau*tau, 1, inv_tau);

        // g_S(s) = 0.5 τ² s² - 2 τ³ s + 3 τ⁴ + e^{-s/τ}( -3 τ⁴ - τ³ s )
        push(g[2],  T(0.5)*tau*tau, 2, T(0));
        push(g[2], -T(2)*tau*tau*tau, 1, T(0));
        push(g[2],  T(3)*tau*tau*tau*tau, 0, T(0));
        push(g[2], -T(3)*tau*tau*tau*tau, 0, inv_tau);
        push(g[2], -tau*tau*tau, 1, inv_tau);

        // g_a(s) = s e^{-s/τ}
        push(g[3], T(1), 1, inv_tau);

        // g_j(s) = e^{-s/τ}(1 - s/τ)
        push(g[4], T(1), 0, inv_tau);
        push(g[4], -inv_tau, 1, inv_tau);

        // ---- assemble Qd = q_c ∫ g gᵀ ds ----
        const T qc = T(4) * sigma2_a / (tau*tau*tau);
        Qd.setZero();

        for (int i=0;i<5;++i) {
            for (int j=i;j<5;++j) {
                T acc = T(0);
                for (const auto& ti : g[i]) {
                    for (const auto& tj : g[j]) {
                        T lam = ti.lam + tj.lam;   // ∈ {0, 1/τ, 2/τ}
                        int n  = ti.n  + tj.n;
                        acc += ti.c * tj.c * integ_pow_exp_(n, lam, h);
                    }
                }
                T val = qc * acc;
                Qd(i,j) = val;
                if (i!=j) Qd(j,i) = val;
            }
        }
    }

    // Build full NX×NX Φ and Qd
    void assembleExtendedFandQ_(T Ts, MatrixNX& F, MatrixNX& Qd) {
        F.setIdentity();
        Qd.setZero();

        // ---- Attitude-error small-angle transition (exact Rodrigues on constant ω) ----
        Matrix3 I3 = Matrix3::Identity();
        const Vector3& w = last_gyr_bias_corrected_;
        T wn = w.norm();
        if (wn < T(1e-9)) {
            // 2nd-order accurate for tiny angles
            Matrix3 Wx = skew_(w);
            F.template block<3,3>(0,0) = I3 - Wx*Ts + (Wx*Wx)*(Ts*Ts*T(0.5));
        } else {
            Vector3 u = w / wn;
            Matrix3 Ux = skew_(u);
            T th = wn * Ts;
            T s = std::sin(th), c = std::cos(th);
            F.template block<3,3>(0,0) = I3 - s*Ux + (T(1)-c)*(Ux*Ux);
        }
        if constexpr (with_gyro_bias) {
            F.template block<3,3>(0,3) = -I3 * Ts; // θ_k+1 ≈ θ_k - Ts b_g  (bias in error-state)
        }
        // Base process (RW for bias; gyro meas mapped in Qbase_)
        Qd.topLeftCorner(BASE_N, BASE_N) = Qbase_ * Ts;

        // ---- 3 axes of [v p S a j] each ----
        for (int axis=0; axis<3; ++axis) {
            Matrix5 Phi_ax, Q_ax;
            const T tau = std::max(T(1e-6), tau_lat_);
            const T sigma2 = Sigma_a_stat_(axis,axis);
            axis_Phi_and_Qd_(Ts, tau, sigma2, Phi_ax, Q_ax);

            int idx[5]={0,3,6,9,12};
            for (int r=0;r<5;++r)
                for (int c=0;c<5;++c) {
                    F(OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Phi_ax(r,c);
                    Qd(OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Q_ax(r,c);
                }
        }

        // ---- accelerometer bias RW (optional) ----
        if constexpr (with_accel_bias) {
            Qd.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
        }
    }
};

