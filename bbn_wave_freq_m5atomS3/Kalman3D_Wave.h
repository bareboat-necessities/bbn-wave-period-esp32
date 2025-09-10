#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  Enhancements:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file merges your original Kalman3D_Wave<T,with_bias> with an extended
  full-matrix Kalman that adds linear navigation states:
     v (3)  : velocity in world frame
     p (3)  : displacement/position in world frame
     S (3)  : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction

  - The original quaternion MEKF logic (time_update, measurement_update, partial updates, quaternion correction)
    is preserved *verbatim* where possible.
  - The extended linear states are integrated with Taylor series (second order for p, first for v,
    and third-order contribution used in S).
  - A full extended covariance (Pext) and transition Jacobian Fext are constructed; the top-left corner
    contains the original MEKF's P/Q blocks (attitude error + optional gyro bias).
  - Accelerometer is expected in IMU/body frame input to time_update(gyr, acc, Ts).
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <limits>

using Eigen::Matrix;
using Eigen::Map;

template <typename T = float, bool with_bias = true>
class EIGEN_ALIGN_MAX Kalman3D_Wave {
    static constexpr T gravity_magnitude = T(9.81);

    // Original base state dimension (attitude-error (3) [+ gyro-bias (3) if with_bias])
    static constexpr int BASE_N = with_bias ? 6 : 3;
    // Extended added states: v(3), p(3), S(3)
    static constexpr int EXT_ADD = 12;
    // New full state dimension
    static constexpr int NX = BASE_N + EXT_ADD;

    static constexpr int OFF_V  = BASE_N + 0;
    static constexpr int OFF_P  = BASE_N + 3;
    static constexpr int OFF_S  = BASE_N + 6;
    static constexpr int OFF_AW = BASE_N + 9;

    // Measurement dimension
    static constexpr int M = 6;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, 4, 1> Vector4;
    typedef Matrix<T, 6, 1> Vector6;
    typedef Matrix<T, BASE_N, BASE_N> MatrixBaseN;
    typedef Matrix<T, NX, NX> MatrixNX;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, 4, 4> Matrix4;
    typedef Matrix<T, M, M> MatrixM;

    static constexpr T half = T(1) / T(2);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor signatures preserved, additional defaults for linear process noise
    Kalman3D_Wave(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(5e0));

    // Initialization / measurement API 
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    void time_update(Vector3 const& gyr, Vector3 const& acc, T Ts);  // new: uses acc to drive v/p/S

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_mag_only(Vector3 const& mag);

    // Extended-only API:
    // Apply zero pseudo-measurement on S (integral drift correction)
    void applyIntegralZeroPseudoMeas();

    // Accessors
    Eigen::Quaternion<T> quaternion() const { return qref; }
    MatrixBaseN const& covariance_base() const { return Pbase; } // top-left original block
    MatrixNX const& covariance_full() const { return Pext; }     // full extended covariance

    Vector3 gyroscope_bias() const {
        if constexpr (with_bias) {
            return xext.template segment<3>(3);
        } else {
            return Vector3::Zero();
        }
    }

    Vector3 get_velocity() const {
        // velocity state at offset BASE_N
        return xext.template segment<3>(BASE_N);
    }

    Vector3 get_position() const {
        // position state at offset BASE_N+3
        return xext.template segment<3>(BASE_N + 3);
    }

    Vector3 get_integral_displacement() const {
       // integral of displacement state at offset BASE_N+6
       return xext.template segment<3>(BASE_N + 6);
    }

    Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }

    // Tuning setters
    void setExtendedQ(MatrixNX const& Qext_in) { Qext = Qext_in; }
    void set_aw_time_constant(T tau_seconds) { tau_aw = std::max(T(1e-3), tau_seconds); }
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat = std_aw.array().square().matrix().asDiagonal();
    }

  private:
    // Original MEKF internals (kept nomenclature)
    Eigen::Quaternion<T> qref;
    Vector3 v1ref;
    Vector3 v2ref;

    // Original base error-state (first BASE_N elements) — now stored inside xext (top portion)
    // But we keep a mirror of original P for compatibility
    MatrixBaseN Pbase;

    // Extended full state xext and Pext (NX x NX)
    Matrix<T, NX, 1> xext; // [ att_err(3), (gyro bias 3 optional), v(3), p(3), S(3) ]
    MatrixNX Pext;

    // Original quaternion transition matrix
    Matrix4 F;

    // Last gyro 
    Vector3 last_gyr_bias_corrected{};

    // Original constant matrices (kept)
    const Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase; // original Q for attitude & bias

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 R_S;  // Triple integration measurement noise

    MatrixNX Qext; // Extended process noise / Q
    Matrix3 Q_Racc_noise; // Process noise for rules using acceleration

    // --- World-acceleration OU process a_w dynamics parameters ---
    T tau_aw = T(1.5);            // correlation time [s], tune 1–5 s for sea states
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(0.8*0.8); // stationary variance diag [ (m/s^2)^2 ]

    // convenience getters
    Matrix3 Rt_from_quat() const { return R_from_quat().transpose(); }
  
    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    void set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func() const;
    Vector3 magnetometer_measurement_func() const;

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Extended helpers
    void assembleExtendedFandQ(const Vector3& acc_body, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);
    Matrix3 R_from_quat() const { return qref.toRotationMatrix(); }

    // Quaternion & small-angle helpers (kept)
    Vector4 quatMultiply(const Vector4& a, const Vector4& b) const;
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)
    void normalizeQuat();
    void vanLoanDiscretization_12x3(const Eigen::Matrix<T,12,12>& A,
                                const Eigen::Matrix<T,12,3>&  G,
                                const Eigen::Matrix<T,3,3>&   Sigma_c,
                                T Ts,
                                Eigen::Matrix<T,12,12>& Phi,
                                Eigen::Matrix<T,12,12>& Qd) const;
};

// Implementation

template <typename T, bool with_bias>
Kalman3D_Wave<T, with_bias>::Kalman3D_Wave(
    Vector3 const& sigma_a,
    Vector3 const& sigma_g,
    Vector3 const& sigma_m,
    T Pq0, T Pb0, T b0, T R_S_noise)
  : Qbase(initialize_Q(sigma_g, b0)),
    Racc(sigma_a.array().square().matrix().asDiagonal()),
    Rmag(sigma_m.array().square().matrix().asDiagonal()),
    // initialize Q_Racc_noise from sigma_a as well
    Q_Racc_noise(sigma_a.array().square().matrix().asDiagonal())
{
  // quaternion init
  qref.setIdentity();

  R_S = Matrix3::Identity() * R_S_noise;

  // initialize base / extended states
  Pbase.setZero();
  Pbase.setIdentity(); // default small initial cov unless user overwrites
  
  // initialize base covariance
  Pbase.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;   // attitude error covariance
  if constexpr (with_bias) {
      Pbase.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;    // bias covariance
  }

  // Extended state
  xext.setZero();
  Pext.setZero();
  
  // Place original base P into top-left of Pext
  Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

  // Seed covariance for a_w (world acceleration)
  Pext.block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;

  // Initialize Qext: top-left is original Qbase; rest zeros until we compute process noise by template
  Qext.setZero();
  Qext.topLeftCorner(BASE_N, BASE_N) = Qbase;

  R.setZero();
  R.template topLeftCorner<3,3>()  = Racc;     // accelerometer measurement noise
  R.template bottomRightCorner<3,3>() = Rmag;  // magnetometer measurement noise
}

template<typename T, bool with_bias>
typename Kalman3D_Wave<T, with_bias>::MatrixBaseN
Kalman3D_Wave<T, with_bias>::initialize_Q(typename Kalman3D_Wave<T, with_bias>::Vector3 sigma_g, T b0) {
  MatrixBaseN Q; Q.setZero();
  if constexpr (with_bias) {
    Q.template topLeftCorner<3,3>() = sigma_g.array().square().matrix().asDiagonal(); // gyro RW
    Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * b0;                   // bias RW
  } else {
    Q = sigma_g.array().square().matrix().asDiagonal();
  }
  return Q;
}

//  initialization helpers
template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag) {
  T const anorm = acc.norm();
  v1ref << 0, 0, -anorm;

  Vector3 const acc_normalized = acc / anorm;
  Vector3 const mag_normalized = mag.normalized();

  Vector3 const Rz = -acc_normalized;
  Vector3 const Ry = Rz.cross(mag_normalized).normalized();
  Vector3 const Rx = Ry.cross(Rz).normalized();

  Matrix3 const Rm = (Matrix3() << Rx, Ry, Rz).finished();
  qref = Eigen::Quaternion<T>(Rm.transpose());
  qref.normalize();

  v2ref = qref * mag;
}

template<typename T, bool with_bias>
Eigen::Quaternion<T> Kalman3D_Wave<T, with_bias>::quaternion_from_acc(Vector3 const& acc) {
  T qx, qy, qz, qw;
  if (acc[2] >= 0) {
    qx = std::sqrt((1 + acc[2]) / 2);
    qw = acc[1] / (2 * qx);
    qy = 0;
    qz = -acc[0] / (2 * qx);
  }
  else {
    qw = std::sqrt((1 - acc[2]) / 2);
    qx = acc[1] / (2 * qw);
    qy = -acc[0] / (2 * qw);
    qz = 0;
  }
  Eigen::Quaternion<T> qref_local = Eigen::Quaternion<T>(qw, -qx, -qy, -qz);
  qref_local.normalize();
  return qref_local;
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::initialize_from_acc(Vector3 const& acc) {
  T const anorm = acc.norm();
  v1ref << 0, 0, -anorm;
  qref = quaternion_from_acc(acc);
  qref.normalize();
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::time_update(Vector3 const& gyr,
                                              Vector3 const& /*acc_body_unused*/,
                                              T Ts)
{
    // ---- Attitude mean propagation ----
    Vector3 gyro_bias = with_bias ? xext.template segment<3>(3) : Vector3::Zero();
    last_gyr_bias_corrected = gyr - gyro_bias;
    set_transition_matrix(last_gyr_bias_corrected, Ts);   // fills F (4x4)
    qref.coeffs() = F * qref.coeffs();
    qref.normalize();

    // ---- Build exact discrete transition & process Q ----
    MatrixNX F_a_ext; MatrixNX Q_a_ext;
    assembleExtendedFandQ(Vector3::Zero(), Ts, F_a_ext, Q_a_ext);

    // ---- Mean propagation for linear subsystem [v,p,S,a_w] ----
    Eigen::Matrix<T,12,1> x_lin_prev;
    x_lin_prev.template segment<3>(0)  = xext.template segment<3>(OFF_V);
    x_lin_prev.template segment<3>(3)  = xext.template segment<3>(OFF_P);
    x_lin_prev.template segment<3>(6)  = xext.template segment<3>(OFF_S);
    x_lin_prev.template segment<3>(9)  = xext.template segment<3>(OFF_AW);

    const auto Phi_lin = F_a_ext.template block<12,12>(OFF_V, OFF_V);
    Eigen::Matrix<T,12,1> x_lin_next = Phi_lin * x_lin_prev;

    // write back mean
    xext.template segment<3>(OFF_V)  = x_lin_next.template segment<3>(0);
    xext.template segment<3>(OFF_P)  = x_lin_next.template segment<3>(3);
    xext.template segment<3>(OFF_S)  = x_lin_next.template segment<3>(6);
    xext.template segment<3>(OFF_AW) = x_lin_next.template segment<3>(9);

    // ---- Covariance propagation ----
    Pext = F_a_ext * Pext * F_a_ext.transpose() + Q_a_ext;
    Pext = T(0.5) * (Pext + Pext.transpose()); // enforce symmetry

    // Mirror base covariance
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);

    // Optional drift correction on S
    applyIntegralZeroPseudoMeas();
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update(Vector3 const& acc, Vector3 const& mag)
{
    // Predicted measurements
    Vector3 v1hat = accelerometer_measurement_func(); // depends on a_w now
    Vector3 v2hat = magnetometer_measurement_func();

    Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
    // accel rows 0..2
    Cext.template block<3,3>(0,0)        = skew_symmetric_matrix(v1hat); // d f_b / d attitude
    Cext.template block<3,3>(0,OFF_AW)   = Rt_from_quat();               // d f_b / d a_w
    // mag rows 3..5 (unchanged)
    Cext.template block<3,3>(3,0)        = skew_symmetric_matrix(v2hat);

    Vector6 yhat; yhat << v1hat, v2hat;
    Vector6 y;    y << acc, mag;
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
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * R * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());

    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update_partial(
    const Eigen::Ref<const Vector3>& meas,
    const Eigen::Ref<const Vector3>& vhat,
    const Eigen::Ref<const Matrix3>& Rm)
{
    // Cext: (3 x NX)
    Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
    Cext.template block<3,3>(0,0) = skew_symmetric_matrix(vhat);
    
    // Innovation
    Vector3 inno = meas - vhat;
    
    // S = C P C^T + Rm  (3x3)
    Matrix3 S_mat = Cext * Pext * Cext.transpose() + Rm;
    
    // PCt = P C^T  (NX x 3)
    Matrix<T, NX, 3> PCt = Pext * Cext.transpose();
    
    // Factor S (SPD) and solve
    Eigen::LDLT<Matrix3> ldlt(S_mat);
    if (ldlt.info() != Eigen::Success) {
        S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R.norm());
        ldlt.compute(S_mat);
        if (ldlt.info() != Eigen::Success) return;
    }
    Matrix<T, NX, 3> K = PCt * ldlt.solve(Matrix3::Identity());
    
    // State update
    xext.noalias() += K * inno;
    
    // Joseph covariance update
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * Rm * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());
    
    // Quaternion correction + zero small-angle
    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update_acc_only(Vector3 const& acc_meas) {
    const Vector3 v1hat = accelerometer_measurement_func();

    // Cext: (3 x NX)
    Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
    // d f_b / d (attitude error)
    Cext.template block<3,3>(0,0) = skew_symmetric_matrix(v1hat);
    // d f_b / d a_w
    Cext.template block<3,3>(0, OFF_AW) = Rt_from_quat();

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
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * Racc * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());

    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update_mag_only(Vector3 const& mag) {
    Vector3 const v2hat = magnetometer_measurement_func();
    measurement_update_partial(mag, v2hat, Rmag);
}

// specific force prediction: f_b = R^T (a_w - g)
template<typename T, bool with_bias>
Matrix<T,3,1> Kalman3D_Wave<T, with_bias>::accelerometer_measurement_func() const {
    const Vector3 g_world(0,0,gravity_magnitude);
    const Vector3 aw = xext.template segment<3>(OFF_AW);
    return Rt_from_quat() * (aw - g_world);
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> Kalman3D_Wave<T, with_bias>::magnetometer_measurement_func() const {
    return qref.inverse() * v2ref;
}

// utility functions
template<typename T, bool with_bias>
Matrix<T, 3, 3> Kalman3D_Wave<T, with_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
  Matrix3 M;
  M << 0, -vec(2), vec(1),
       vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return M;
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts) {
  const Vector3 delta_theta = gyr * Ts;
  T un = delta_theta.norm();
  if (un == T(0)) un = std::numeric_limits<T>::min();

  Matrix4 Omega; Omega.setZero();
  Omega.template topLeftCorner<3,3>() = -skew_symmetric_matrix(delta_theta);
  Omega.template topRightCorner<3,1>() =  delta_theta;
  Omega.template bottomLeftCorner<1,3>() = -delta_theta.transpose();
  // Omega(3,3) already zero

  F = std::cos(half * un) * Matrix4::Identity() + std::sin(half * un) / un * Omega;
}

// quaternion multiplication helper (vector form used rarely in this file)
template<typename T, bool with_bias>
typename Kalman3D_Wave<T, with_bias>::Vector4 Kalman3D_Wave<T, with_bias>::quatMultiply(const Vector4& a, const Vector4& b) const {
  Vector4 r;
  Eigen::Matrix<T,3,1> av = a.template head<3>(), bv = b.template head<3>();
  T aw = a(3), bw = b(3);
  r.template head<3>() = aw * bv + bw * av + av.cross(bv);
  r(3) = aw * bw - av.dot(bv);
  return r;
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::applyQuaternionCorrectionFromErrorState() {
  // xext(0..2) contains the small-angle error — same as original code; create corr quaternion and apply
  Eigen::Quaternion<T> corr(T(1), half * xext(0), half * xext(1), half * xext(2));
  corr.normalize();
  qref = qref * corr;
  qref.normalize();
}

// normalize quaternion
template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::normalizeQuat() {
  qref.normalize();
}

//  Extended pseudo-measurement: zero S 
template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::applyIntegralZeroPseudoMeas() {
    // Build measurement matrix H (picks S block)
    Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
    H.template block<3,3>(0, BASE_N + 6) = Matrix3::Identity();

    // Innovation (desired S = 0)
    Vector3 z = Vector3::Zero();
    Vector3 inno = z - H * xext;

    // Innovation covariance
    Matrix3 S_mat = H * Pext * H.transpose() + R_S;
    
    // PHt (NX x 3)
    Matrix<T, NX, 3> PHt = Pext * H.transpose();
    
    // Factor S and compute K = PHt * S^{-1}
    Eigen::LDLT<Matrix3> ldlt(S_mat);
    if (ldlt.info() != Eigen::Success) {
        S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R_S.norm());  // small jitter; epsilon may be too tiny in practice
        ldlt.compute(S_mat);
        if (ldlt.info() != Eigen::Success) return;
    }
    Matrix<T, NX, 3> K = PHt * ldlt.solve(Matrix3::Identity());
    
    // Update
    xext.noalias() += K * inno;
    
    // Joseph form + symmetrize
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * H) * Pext * (I - K * H).transpose() + K * R_S * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());
    
    // Quaternion correction + zero small-angle + mirror base
    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::vanLoanDiscretization_12x3(
    const Eigen::Matrix<T,12,12>& A,
    const Eigen::Matrix<T,12,3>&  G,
    const Eigen::Matrix<T,3,3>&   Sigma_c,
    T Ts,
    Eigen::Matrix<T,12,12>& Phi,
    Eigen::Matrix<T,12,12>& Qd) const
{
    // Build Van-Loan block matrix (24x24)
    Eigen::Matrix<T,24,24> M; M.setZero();
    M.block(0,0,12,12)   = -A * Ts;
    M.block(0,12,12,12)  =  G * Sigma_c * G.transpose() * Ts;
    M.block(12,12,12,12) =  A.transpose() * Ts;

    // Matrix exponential
    Eigen::Matrix<T,24,24> expM = M.exp();

    // Extract blocks
    Eigen::Matrix<T,12,12> PhiT = expM.block(12,12,12,12);
    Eigen::Matrix<T,12,12> Qblk = expM.block(0,12,12,12);

    // Van-Loan identities
    Phi = PhiT.transpose();      // exp(A Ts)
    Qd  = Phi * Qblk;            // exact discrete process noise
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::assembleExtendedFandQ(
    const Vector3& /*acc_body_unused*/, T Ts,
    Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext)
{
    F_a_ext.setIdentity();
    Q_a_ext.setZero();

    // === Attitude error (+ optional bias) discrete transition ===
    Matrix3 Atheta = Matrix3::Identity() - skew_symmetric_matrix(last_gyr_bias_corrected) * Ts;
    F_a_ext.template block<3,3>(0,0) = Atheta;
    if constexpr (with_bias) {
        F_a_ext.template block<3,3>(0,3) = -Matrix3::Identity() * Ts;
    }

    // Process noise for attitude/bias
    Q_a_ext.topLeftCorner(BASE_N, BASE_N) = Qbase;

    // === Linear subsystem [v, p, S, a_w] exact Van-Loan ===
    using Mat12   = Eigen::Matrix<T,12,12>;
    using Mat12x3 = Eigen::Matrix<T,12,3>;

    Mat12 A; A.setZero();
    // v̇ = a_w
    A.template block<3,3>(0,9) = Matrix3::Identity();
    // ṗ = v
    A.template block<3,3>(3,0) = Matrix3::Identity();
    // Ṡ = p
    A.template block<3,3>(6,3) = Matrix3::Identity();
    // ȧ_w = -(1/τ) a_w
    A.template block<3,3>(9,9) = -(T(1)/std::max(T(1e-6), tau_aw)) * Matrix3::Identity();

    Mat12x3 G; G.setZero();
    G.template block<3,3>(9,0) = Matrix3::Identity(); // noise drives a_w

    Mat12 Phi_lin, Qd_lin;
    vanLoanDiscretization_12x3(A, G, Sigma_aw_stat, Ts, Phi_lin, Qd_lin);

    // Insert into extended transition + process covariance
    F_a_ext.template block<12,12>(OFF_V, OFF_V) = Phi_lin;
    Q_a_ext.template block<12,12>(OFF_V, OFF_V) = Qd_lin;
}

