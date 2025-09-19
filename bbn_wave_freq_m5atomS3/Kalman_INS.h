#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  Enhancements:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file defines Kalman_INS<T,with_gyro_bias,with_accel_bias>, an extended
  error-state Kalman filter that adds linear navigation states:
     v (3)      : velocity in world frame
     p (3)      : displacement/position in world frame
     S (3)      : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction
     a_w (3)    : latent acceleration (world frame)
     ȧ_w (3)    : latent acceleration derivative (Matérn(3/2) model)
     b_a (3)    : optional accelerometer bias

  - The quaternion MEKF logic (time_update, measurement_update, quaternion correction)
    is preserved where possible.
  - The extended linear states are driven by a latent 2nd-order Gauss–Markov (Matérn-3/2) process
    instead of a simple OU.
  - A full extended covariance (Pext) and transition Jacobian are constructed;
    the top-left corner contains the original MEKF's P/Q blocks (attitude error + optional gyro bias).
  - Accelerometer and magnetometer inputs must be given in aerospace/NED (x north, y east, z down).
*/

#include <limits>
#include <stdexcept>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
  #include <unsupported/Eigen/MatrixFunctions>  // enables .exp() on matrices (desktop)
#else
  #include <ArduinoEigenDense.h>
#endif

using Eigen::Matrix;
using Eigen::Map;

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman_INS {

    // Base block: attitude error (3) + optional gyro bias (3)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended added states:
    //   v(3), p(3), S(3), a_w(3), j_w(3)  [+ optional b_acc(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 18 : 15;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets into the full state vector
    static constexpr int OFF_V   = BASE_N + 0;
    static constexpr int OFF_P   = BASE_N + 3;
    static constexpr int OFF_S   = BASE_N + 6;
    static constexpr int OFF_AW  = BASE_N + 9;   // a
    static constexpr int OFF_JW  = BASE_N + 12;  // j = \dot a
    static constexpr int OFF_BA  = with_accel_bias ? (BASE_N + 15) : -1; // accelerometer bias (optional)

    // Measurement dimension (acc + mag)
    static constexpr int M = 6;

    using Vector3   = Matrix<T, 3, 1>;
    using Vector4   = Matrix<T, 4, 1>;
    using Vector6   = Matrix<T, 6, 1>;
    using Matrix3   = Matrix<T, 3, 3>;
    using Matrix4   = Matrix<T, 4, 4>;
    using Matrix5   = Matrix<T, 5, 5>;
    using MatrixM   = Matrix<T, M, M>;
    using MatrixNX  = Matrix<T, NX, NX>;
    using MatrixBaseN = Matrix<T, BASE_N, BASE_N>;

    static constexpr T half = T(0.5);
    static constexpr T STD_GRAVITY = T(9.80665);   // m/s²
    static constexpr T tempC_ref   = T(35.0);      // °C reference for accel-bias linear drift

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor (keeps prior signature)
    Kalman_INS(const Vector3& sigma_a, const Vector3& sigma_g, const Vector3& sigma_m,
               T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(5e-2),
               T gravity_magnitude = T(STD_GRAVITY));

    // Initialization
    void initialize_from_acc_mag(const Vector3& acc, const Vector3& mag);
    void initialize_from_acc(const Vector3& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(const Vector3& acc);

    // Config
    void set_mag_world_ref(const Vector3& B_world) { v2ref = B_world.normalized(); }

    void set_Racc(const Vector3& sigma_acc) {
        Racc = sigma_acc.array().square().matrix().asDiagonal();
        R.template topLeftCorner<3,3>() = Racc;
    }
    void set_Rmag(const Vector3& sigma_mag) {
        Matrix3 Rmag_new = sigma_mag.array().square().matrix().asDiagonal();
        const_cast<Matrix3&>(Rmag) = Rmag_new;
        R.template bottomRightCorner<3,3>() = Rmag_new;
    }
    void set_RS_noise(const Vector3& sigma_S) {
        R_S = sigma_S.array().square().matrix().asDiagonal();
    }

    // Matérn-3/2 latent: correlation time τ; stationary std of a_w per axis
    void set_aw_time_constant(T tau_seconds) { tau_lat_ = std::max(T(1e-3), tau_seconds); }
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat_ = std_aw.array().square().matrix().asDiagonal();   // Var[a] per axis
    }

    // Optional accel bias controls (if compiled with_accel_bias=true)
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

    // Initial covariance for jerk states
    void set_initial_jerk_std(T s) {
        Pext.template block<3,3>(OFF_JW, OFF_JW) = Matrix3::Identity() * (s*s);
    }
  
    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0 * sigma_v0);
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0 * sigma_p0);
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0 * sigma_S0);
    }

    // Cycle
    void time_update(const Vector3& gyr, T Ts);
    void measurement_update(const Vector3& acc, const Vector3& mag, T tempC = tempC_ref);
    void measurement_update_acc_only(const Vector3& acc, T tempC = tempC_ref);
    void measurement_update_mag_only(const Vector3& mag);

    // Pseudo-measurement on S
    void applyIntegralZeroPseudoMeas();

    // Accessors
    Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    const MatrixBaseN& covariance_base() const { return Pbase; }
    const MatrixNX&    covariance_full() const { return Pext;  }

    Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) return xext.template segment<3>(3);
        else                           return Vector3::Zero();
    }
    Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) return xext.template segment<3>(OFF_BA);
        else                            return Vector3::Zero();
    }
    Vector3 get_velocity() const { return xext.template segment<3>(OFF_V); }
    Vector3 get_position() const { return xext.template segment<3>(OFF_P); }
    Vector3 get_integral_displacement() const { return xext.template segment<3>(OFF_S); }
    Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }
    Vector3 get_world_jerk()  const { return xext.template segment<3>(OFF_JW); }

  private:
    // Constants
    const T gravity_magnitude_;

    // Orientation & references
    Eigen::Quaternion<T> qref;         // world→body
    Vector3 v2ref = Vector3::UnitX();  // magnetic field in world frame (unit)

    // Covariances
    MatrixBaseN Pbase;
    MatrixNX    Pext;

    // State
    Matrix<T, NX, 1> xext;   // [att_err(3), (gyro bias 3), v(3), p(3), S(3), a(3), j(3), (b_acc 3)]

    // Last corrected gyro
    Vector3 last_gyr_bias_corrected{};

    // Accel bias model (optional)
    T       sigma_bacc0_ = T(0.1);
    Matrix3 Q_bacc_      = Matrix3::Identity() * T(1e-8);
    Vector3 k_a_         = Vector3::Constant(T(0.003)); // m/s² per °C

    // Measurement covariances
    const Matrix3 Rmag;
    MatrixM       R;
    Matrix3       Racc;
    Matrix3       R_S;

    // Base process for attitude/bias
    MatrixBaseN Qbase;

    // Latent acceleration Matérn-3/2 params
    T       tau_lat_      = T(1.5);                              // correlation time [s]
    Matrix3 Sigma_aw_stat_ = Matrix3::Identity() * T(0.24*0.24); // Var[a] (diag) [(m/s²)²]

    // Helpers
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }              // world→body
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }  // body→world

    static MatrixBaseN initialize_Q(const Vector3& sigma_g, T b0);

    Matrix3 skew_symmetric_matrix(const Vector3& v) const {
        Matrix3 M;
        M << 0, -v(2), v(1),
             v(2), 0, -v(0),
            -v(1), v(0), 0;
        return M;
    }

    Vector3 accelerometer_measurement_func(T tempC) const;
    Vector3 magnetometer_measurement_func() const;

    void applyQuaternionCorrectionFromErrorState() {
        Eigen::Quaternion<T> corr(T(1), half * xext(0), half * xext(1), half * xext(2));
        corr.normalize();
        qref = qref * corr;
        qref.normalize();
    }

    // Build F and Q (discrete) for the full extended system in one step
    void assembleExtendedFandQ(T Ts, MatrixNX& F_a_ext, MatrixNX& Q_a_ext);

    // Van Loan utilities
#ifdef EIGEN_NON_ARDUINO
    static void vanLoanDiscretization_15x3(const Eigen::Matrix<T,15,15>& A,
                                           const Eigen::Matrix<T,15,3>&  G,
                                           const Eigen::Matrix<T,3,3>&   Sigma_c,
                                           T Ts,
                                           Eigen::Matrix<T,15,15>& Phi,
                                           Eigen::Matrix<T,15,15>& Qd)
    {
        Eigen::Matrix<T,30,30> M; M.setZero();
        M.block(0,0,15,15)    = -A * Ts;
        M.block(0,15,15,15)   =  G * Sigma_c * G.transpose() * Ts;
        M.block(15,15,15,15)  =  A.transpose() * Ts;

        Eigen::Matrix<T,30,30> expM = M.exp();
        auto PhiT = expM.block(15,15,15,15);
        auto Qblk = expM.block(0,15,15,15);
        Phi = PhiT.transpose();
        Qd  = Phi * Qblk;
    }
#else
    template<typename Mat>
    static Mat expm_pade6(const Mat& A) {
        using S = typename Mat::Scalar;
        const int n = A.rows();
        const S theta = S(3);
        const int max_squarings = 8;

        const S c0 = S(1);
        const S c1 = S(1)/S(2);
        const S c2 = S(1)/S(10);
        const S c3 = S(1)/S(120);
        const S c4 = S(1)/S(1680);
        const S c5 = S(1)/S(30240);
        const S c6 = S(1)/S(665280);

        S normA = A.cwiseAbs().colwise().sum().maxCoeff();
        int s = 0;
        if (normA > theta) {
            s = std::min(max_squarings, int(std::ceil(std::log2(normA/theta))));
        }
        Mat As = A / S(1 << s);

        Mat I  = Mat::Identity(n,n);
        Mat A2 = As * As;
        Mat A4 = A2 * A2;
        Mat A6 = A4 * A2;

        Mat U = As * (c1*I + c3*A2 + c5*A4);
        Mat V = c0*I + c2*A2 + c4*A4 + c6*A6;

        Mat P = V + U;
        Mat Q = V - U;

        Mat R = Q.fullPivLu().solve(P);
        for (int i=0;i<s;++i) R = R*R;
        return R;
    }

    static void vanLoanAxis5x1(T tau, T sigma2, T Ts,
                               Matrix5& Phi, Matrix5& Qd)
    {
        // Axis order: [v p S a j]
        Matrix5 A; A.setZero();
        A(0,3) = T(1);           // v̇ = a
        A(1,0) = T(1);           // ṗ = v
        A(2,1) = T(1);           // Ṡ = p
        A(3,4) = T(1);           // ȧ = j
        A(4,3) = -T(1)/(tau*tau);// j̇ = -(1/τ²) a
        A(4,4) = -T(2)/tau;      //      - (2/τ) j

        Eigen::Matrix<T,5,1> G; G.setZero();
        G(4,0) = T(1);           // noise drives j

        // White-noise power to make Var[a] = sigma2 at stationarity: qc = 4*sigma2/τ^3
        T Sigma_c = T(4) * sigma2 / (tau*tau*tau);

        Eigen::Matrix<T,10,10> M; M.setZero();
        M.block(0,0,5,5)   = -A * Ts;
        M.block(0,5,5,5)   =  (G * (Sigma_c) * G.transpose()) * Ts;
        M.block(5,5,5,5)   =  A.transpose() * Ts;

        auto expM = expm_pade6(M);
        Matrix5 PhiT = expM.block(5,5,5,5);
        Matrix5 Qblk = expM.block(0,5,5,5);

        Phi = PhiT.transpose();
        Qd  = Phi * Qblk;
    }
#endif
};

// ===== Implementation =====

template <typename T, bool with_gyro_bias, bool with_accel_bias>
Kalman_INS<T, with_gyro_bias, with_accel_bias>::Kalman_INS(
    const Vector3& sigma_a,
    const Vector3& sigma_g,
    const Vector3& sigma_m,
    T Pq0, T Pb0, T b0, T R_S_noise, T gravity_magnitude)
  : gravity_magnitude_(gravity_magnitude),
    Racc(sigma_a.array().square().matrix().asDiagonal()),
    Rmag(sigma_m.array().square().matrix().asDiagonal()),
    Qbase(initialize_Q(sigma_g, b0))
{
    qref.setIdentity();
    Pbase.setIdentity();

    // Base covariance init
    Pbase.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;
    if constexpr (with_gyro_bias) {
        Pbase.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;
    }

    xext.setZero();
    Pext.setZero();
    Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

    // Seed linear states
    const T sigma_v0 = T(1.0);
    const T sigma_p0 = T(20.0);
    const T sigma_S0 = T(50.0);
    set_initial_linear_uncertainty(sigma_v0, sigma_p0, sigma_S0);

    // Seed latent a (and jerk) covariance to stationary Var[a] for a; modest for jerk
    Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat_;
    Pext.template block<3,3>(OFF_JW, OFF_JW) = (Matrix3::Identity()* (T(0.05)));

    if constexpr (with_accel_bias) {
        Pext.template block<3,3>(OFF_BA, OFF_BA) = Matrix3::Identity() * (sigma_bacc0_ * sigma_bacc0_);
    }

    R.setZero();
    R.template topLeftCorner<3,3>()  = Racc;
    R.template bottomRightCorner<3,3>() = Rmag;

    R_S = Matrix3::Identity() * R_S_noise;
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
typename Kalman_INS<T, with_gyro_bias, with_accel_bias>::MatrixBaseN
Kalman_INS<T, with_gyro_bias, with_accel_bias>::initialize_Q(const Vector3& sigma_g, T b0)
{
    MatrixBaseN Q; Q.setZero();
    if constexpr (with_gyro_bias) {
        Q.template topLeftCorner<3,3>()  = sigma_g.array().square().matrix().asDiagonal(); // gyro meas noise mapped
        Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * b0;                    // gyro bias RW power
    } else {
        Q = sigma_g.array().square().matrix().asDiagonal();
    }
    return Q;
}

// ===== Initialization =====

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::initialize_from_acc_mag(
    const Vector3& acc_body, const Vector3& mag_body)
{
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

    v2ref = R_bw() * mag_body.normalized(); // store world field unit
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
Eigen::Quaternion<T>
Kalman_INS<T, with_gyro_bias, with_accel_bias>::quaternion_from_acc(const Vector3& acc)
{
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::initialize_from_acc(const Vector3& acc)
{
    T anorm = acc.norm();
    if (anorm < T(1e-8)) throw std::runtime_error("acc vector too small");
    qref = quaternion_from_acc(acc / anorm);
    qref.normalize();
}

// ===== Propagation =====

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::time_update(const Vector3& gyr, T Ts)
{
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

    // Full F and Q (discrete) for one step
    MatrixNX F_a_ext; MatrixNX Q_a_ext;
    assembleExtendedFandQ(Ts, F_a_ext, Q_a_ext);

    // Mean propagation for linear block can be done by state-transition multiply
    // linear block is [v p S a j] (15 states for 3 axes) starting at OFF_V
    Eigen::Matrix<T,15,1> x_lin_prev;
    x_lin_prev.template segment<3>(0)   = xext.template segment<3>(OFF_V);
    x_lin_prev.template segment<3>(3)   = xext.template segment<3>(OFF_P);
    x_lin_prev.template segment<3>(6)   = xext.template segment<3>(OFF_S);
    x_lin_prev.template segment<3>(9)   = xext.template segment<3>(OFF_AW);
    x_lin_prev.template segment<3>(12)  = xext.template segment<3>(OFF_JW);

    const auto Phi_lin = F_a_ext.template block<15,15>(OFF_V, OFF_V);
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
typename Kalman_INS<T, with_gyro_bias, with_accel_bias>::Vector3
Kalman_INS<T, with_gyro_bias, with_accel_bias>::accelerometer_measurement_func(T tempC) const
{
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
typename Kalman_INS<T, with_gyro_bias, with_accel_bias>::Vector3
Kalman_INS<T, with_gyro_bias, with_accel_bias>::magnetometer_measurement_func() const
{
    return R_wb() * v2ref;
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::measurement_update(
    const Vector3& acc, const Vector3& mag, T tempC)
{
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::measurement_update_acc_only(
    const Vector3& acc_meas, T tempC)
{
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::measurement_update_mag_only(const Vector3& mag)
{
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::applyIntegralZeroPseudoMeas()
{
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

// ===== Build F and Q (discrete) =====

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman_INS<T, with_gyro_bias, with_accel_bias>::assembleExtendedFandQ(
    T Ts, MatrixNX& F_a_ext, MatrixNX& Q_a_ext)
{
    F_a_ext.setIdentity();
    Q_a_ext.setZero();

    // Attitude error (+ optional gyro bias) transition (Rodrigues small-angle)
    Matrix3 I3 = Matrix3::Identity();
    Vector3 w = last_gyr_bias_corrected;
    T omega = w.norm();
    T theta = omega * Ts;

    if (theta < T(1e-8)) {
        Matrix3 Wx = skew_symmetric_matrix(w);
        F_a_ext.template block<3,3>(0,0) = I3 - Wx*Ts + (Wx*Wx)*(Ts*Ts/2);
    } else {
        Matrix3 W = skew_symmetric_matrix(w / omega);
        T s = std::sin(theta), c = std::cos(theta);
        F_a_ext.template block<3,3>(0,0) = I3 - s*W + (T(1)-c)*(W*W);
    }
    if constexpr (with_gyro_bias) {
        F_a_ext.template block<3,3>(0,3) = -I3 * Ts;
    }

    // Process noise for base
    Q_a_ext.topLeftCorner(BASE_N, BASE_N) = Qbase * Ts;

    // Linear subsystem (15 states: v,p,S,a,j), 3 axes (independent)
#ifdef EIGEN_NON_ARDUINO
    using Mat15   = Eigen::Matrix<T,15,15>;
    using Mat15x3 = Eigen::Matrix<T,15,3>;

    Mat15 A; A.setZero();
    // v̇ = a
    A.block(0, 9, 3,3) = I3;
    // ṗ = v
    A.block(3, 0, 3,3) = I3;
    // Ṡ = p
    A.block(6, 3, 3,3) = I3;
    // ȧ = j
    A.block(9,12, 3,3) = I3;
    // j̇ = -(1/τ²) a - (2/τ) j
    const T tau = std::max(T(1e-6), tau_lat_);
    A.block(12, 9, 3,3) = -(T(1)/(tau*tau)) * I3;
    A.block(12,12, 3,3) = -(T(2)/tau) * I3;

    Mat15x3 G; G.setZero();
    G.block(12,0,3,3) = I3; // noise drives jerk j

    // White-noise power to make Var[a] = Sigma_aw_stat_
    Matrix3 Sigma_c = (T(4)/(tau*tau*tau)) * Sigma_aw_stat_;

    Mat15 Phi_lin, Qd_lin;
    vanLoanDiscretization_15x3(A, G, Sigma_c, Ts, Phi_lin, Qd_lin);

    F_a_ext.block(OFF_V, OFF_V, 15,15) = Phi_lin;
    Q_a_ext.block(OFF_V, OFF_V, 15,15) = Qd_lin;

#else
    // Embedded: do per-axis 5x5 Van Loan and assemble
    using Mat15 = Eigen::Matrix<T,15,15>;
    Mat15 Phi_lin; Phi_lin.setZero();
    Mat15 Qd_lin;  Qd_lin.setZero();

    const T tau = std::max(T(1e-6), tau_lat_);
    for (int axis=0; axis<3; ++axis) {
        T sigma2 = Sigma_aw_stat_(axis,axis);

        Matrix5 Phi_ax, Qd_ax;
        vanLoanAxis5x1(tau, sigma2, Ts, Phi_ax, Qd_ax);

        // axis layout indices inside the 15x15 block: [v p S a j] with stride 3
        int idx[5] = {0,3,6,9,12};
        for (int i=0;i<5;++i)
            for (int j=0;j<5;++j) {
                Phi_lin(idx[i]+axis, idx[j]+axis) = Phi_ax(i,j);
                Qd_lin (idx[i]+axis, idx[j]+axis) = Qd_ax (i,j);
            }
    }

    F_a_ext.block(OFF_V, OFF_V, 15,15) = Phi_lin;
    Q_a_ext.block(OFF_V, OFF_V, 15,15) = Qd_lin;
#endif

    if constexpr (with_accel_bias) {
        Q_a_ext.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
    }
}
