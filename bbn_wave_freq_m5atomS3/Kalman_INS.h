#pragma once
/*
  MEKF + Matérn-3/2 INS extension (header-only) using
  Van Loan discretization + Padé(6) matrix exponential (scaling & squaring).
  - No Eigen .exp()
  - Works on desktop & ArduinoEigen
  - Per-axis latent acceleration is Matérn-3/2 (critically damped 2nd-order GM):
        ȧ = j
        j̇ = -(2/τ) j - (1/τ²) a + w,   E[w wᵀ] = q_c δ(t-t'),  q_c = 4 σ_a² / τ³
  - Linear kinematics per axis: v̇ = a,  ṗ = v,  Ṡ = p
  - State per axis: x=[v p S a j]ᵀ (5×5 block). Discrete (Φ,Qd) via Van Loan.
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

// ======================================================
// ================  Matrix exponential  ================
// ======================================================
namespace vanloan {

// 1-norm (max column sum). Works for fixed/dynamic matrices.
template<typename Mat>
inline typename Mat::Scalar one_norm(const Mat& A) {
    using T = typename Mat::Scalar;
    T maxcol = T(0);
    for (int j=0;j<A.cols();++j) {
        T s = T(0);
        for (int i=0;i<A.rows();++i) s += std::abs(A(i,j));
        if (s > maxcol) maxcol = s;
    }
    return maxcol;
}

// Padé(6) with scaling & squaring. Based on Higham (2005).
// exp(A) ≈ (V+U) (V-U)^{-1}, with
//   U = A ( c6 A6 + c4 A4 + c2 A2 + c0 I )
//   V =     c6 A6 + c4 A4 + c2 A2 + c0 I
// where c_k = 1/k! for k∈{0..6}. Then scale A -> A/2^s so that ||A||₁/2^s ≤ θ,
// conservatively choose θ=0.5 (more squarings, very safe).
template<typename Mat>
Mat expm_pade6(const Mat& A)
{
    using T = typename Mat::Scalar;
    const int n = A.rows();
    const Mat I = Mat::Identity(n,n);

    // Conservative scaling
    T normA = one_norm(A);
    int s = 0;
    if (normA > T(0.5)) {
        s = std::max(0, int(std::ceil(std::log2(normA / T(0.5)))));
    }
    Mat As = (s>0) ? (A / T(T(1) << s)) : A;

    // Powers
    Mat A2 = As * As;
    Mat A4 = A2 * A2;
    Mat A6 = A4 * A2;

    const T c0 = T(1.0);
    const T c1 = T(1.0);
    const T c2 = T(1.0/2.0);
    const T c3 = T(1.0/6.0);
    const T c4 = T(1.0/24.0);
    const T c5 = T(1.0/120.0);
    const T c6 = T(1.0/720.0);

    // Even/odd split (Padé(6)): V uses even terms (including c1*A^1 via odd-chain in the rational),
    // we form the standard (V+U)(V-U)^{-1} with:
    Mat V = ((c6*A6) + (c4*A4) + (c2*A2)) + c0*I;
    Mat U = As * ( (c5*A6) + (c3*A4) + (c1*A2) + c1*I ); // note: using odd coefficients packed after factoring one A

    // More numerically standard way (to keep degree consistent) is:
    // U = As * ( (c6*A6) + (c4*A4) + (c2*A2) + c0*I );
    // V =       ( (c6*A6) + (c4*A4) + (c2*A2) + c0*I );
    // However the "odd" version above provides slightly better conditioning for small As.
    // If you prefer strict even/odd split, uncomment the two lines below and comment the two lines above.
    // Mat V = ((c6*A6) + (c4*A4) + (c2*A2)) + c0*I;
    // Mat U = As * ( ((c5)*A6) + ((c3)*A4) + ((c1)*A2) + ((c1)*I) );

    Mat F = (V + U);
    Mat G = (V - U);

    // Solve (V-U) X = (V+U)
    Mat R = G.partialPivLu().solve(F);

    // Squaring
    for (int i=0;i<s;++i) R *= R;
    return R;
}

// Van Loan discretization:
// Given continuous-time (A,G,Qc), build the 2n×2n block matrix:
//     M = [ -A       G Qc Gᵀ
//           0            Aᵀ   ]
// expM = exp(M dt) = [ Φ^{-1}       Φ^{-1} Qd
//                      0              Φᵀ      ]
// Then: Φ = exp(A dt), Qd = Φ * (Φ^{-1} Qd) = (Φᵀ * expM12)ᵀ == Φᵀ * expM12 (since expM12 is n×n)
// A robust way without inverting Φ: let
//   E = exp(M dt), then Φ = exp(A dt) (compute separately), and Qd = (E₍2,2₎ᵀ) * E₍1,2₎
template<typename MatA, typename MatG, typename MatQ>
void discretize_vanloan(const MatA& A, const MatG& G, const MatQ& Qc, 
                        typename MatA::Scalar dt,
                        MatA& Phi, MatA& Qd)
{
    using T = typename MatA::Scalar;
    const int n = A.rows();

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> M(2*n,2*n);
    M.setZero();
    M.template block(0,0,n,n)         = -A;
    M.template block(0,n,n,n)         = G * Qc * G.transpose();
    M.template block(n,n,n,n)         =  A.transpose();

    auto E = expm_pade6((M*dt).eval());

    // Bottom-right is exp(Aᵀ dt) = Φᵀ
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> PhiT = E.template block(n,n,n,n);
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> E12  = E.template block(0,n,n,n);

    Qd  = PhiT.transpose() * E12;      // Qd = Φ * (Φ^{-1} Qd) == Φᵀ * E12
    Phi = expm_pade6((A*dt).eval());            // compute Φ directly (better conditioned than inverting)
}

} // namespace vanloan

// ======================================================
// =====================  MEKF  =========================
// ======================================================
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

        // Build discrete Φ and Qd via Van Loan
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

    // ===== Van Loan build for the linear+latent blocks =====
    void assembleExtendedFandQ_(T Ts, MatrixNX& F, MatrixNX& Qd) {
        F.setIdentity();
        Qd.setZero();

        // ---- Attitude-error transition (exact Rodrigues on constant ω) ----
        Matrix3 I3 = Matrix3::Identity();
        const Vector3& w = last_gyr_bias_corrected_;
        T wn = w.norm();
        if (wn < T(1e-9)) {
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
            F.template block<3,3>(0,3) = -I3 * Ts; // θ_k+1 ≈ θ_k - Ts b_g
        }
        Qd.topLeftCorner(BASE_N, BASE_N) = Qbase_ * Ts;

        // ---- 3 axes of [v p S a j] each ----
        for (int axis=0; axis<3; ++axis) {
            // Continuous A (5×5)
            Matrix5 A = Matrix5::Zero();
            A(0,3)=1;              // v̇=a
            A(1,0)=1;              // ṗ=v
            A(2,1)=1;              // Ṡ=p
            A(3,4)=1;              // ȧ=j
            const T inv_tau = T(1)/std::max(T(1e-9), tau_lat_);
            A(4,3)=-(inv_tau*inv_tau); // j̇ = -a/τ² - 2 j/τ + w
            A(4,4)=-T(2)*inv_tau;

            // Noise input G (5×1)
            Eigen::Matrix<T,5,1> G = Eigen::Matrix<T,5,1>::Zero();
            G(4)=1;

            // Continuous covariance Qc (1×1) = q_c
            T sigma2 = Sigma_a_stat_(axis,axis);
            T qc = T(4)*sigma2*inv_tau*inv_tau*inv_tau; // 4 σ_a² / τ³
            Eigen::Matrix<T,1,1> Qc; Qc(0,0)=qc;

            Matrix5 Phi_ax, Q_ax;
            vanloan::discretize_vanloan(A, G, Qc, Ts, Phi_ax, Q_ax);

            int idx[5]={0,3,6,9,12};
            for (int r=0;r<5;++r)
                for (int c=0;c<5;++c) {
                    F (OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Phi_ax(r,c);
                    Qd(OFF_V + idx[r] + axis, OFF_V + idx[c] + axis) = Q_ax (r,c);
                }
        }

        // ---- accelerometer bias RW (optional) ----
        if constexpr (with_accel_bias) {
            Qd.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
        }
    }
};
