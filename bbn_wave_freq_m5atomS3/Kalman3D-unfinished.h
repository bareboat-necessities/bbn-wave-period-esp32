#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  This file merges your original QuaternionMEKF<T,with_bias> with an extended
  full-matrix Kalman that adds linear navigation states:
     v (3)  : velocity in world frame
     p (3)  : displacement/position in world frame
     S (3)  : integral of displacement (∫ p dt) — with zero pseudo-measurement drift correction

  - The original quaternion MEKF logic (time_update, measurement_update, partial updates, quaternion correction)
    is preserved *verbatim* where possible.
  - The extended linear states are integrated with Taylor series (second order for p, first for v,
    and third-order contribution used in S).
  - A full extended covariance (Pext) and transition Jacobian Fext are constructed; the top-left corner
    contains the original MEKF's P/Q blocks (attitude error + optional gyro bias).
  - Accelerometer is expected in IMU/body frame input to time_update(gyr, acc, Ts).
  - No gravity removal is performed automatically (user should subtract gravity if needed).
*/

#include <ArduinoEigenDense.h>
#include <limits>

using Eigen::Matrix;
using Eigen::Map;

template <typename T = float, bool with_bias = true>
class QuaternionMEKF {
    // Original base state dimension (attitude-error (3) [+ gyro-bias (3) if with_bias])
    static constexpr int BASE_N = with_bias ? 6 : 3;
    // Extended added states: v(3), p(3), S(3)
    static constexpr int EXT_ADD = 9;
    // New full state dimension
    static constexpr int NX = BASE_N + EXT_ADD;

    // Measurement dimension (unchanged)
    static const int M = 6;

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
    // Constructor signatures preserved, additional defaults for linear process noise
    QuaternionMEKF(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(5e0));

    // Initialization / measurement API preserved
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Time update: preserved old signature (no acc) plus new overload (gyr + acc)
    void time_update(Vector3 const& gyr, T Ts);                      // original behavior (acc=0)
    void time_update(Vector3 const& gyr, Vector3 const& acc, T Ts);  // new: uses acc to drive v/p/S

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_mag_only(Vector3 const& mag);

    // Extended-only API:
    // Apply zero pseudo-measurement on S (integral drift correction)
    void applyIntegralZeroPseudoMeas();

    // Accessors (quaternion preserved)
    Vector4 const& quaternion() const { return qref.coeffs(); }
    MatrixBaseN const& covariance_base() const { return Pbase; } // top-left original block
    MatrixNX const& covariance_full() const { return Pext; }     // full extended covariance
    Vector3 gyroscope_bias() const {
        if constexpr (with_bias) {
            return xext.template segment<3>(3);
        } else {
            return Vector3::Zero();
        }
    }

    // Tuning setters
    void setLinearProcessNoise(Matrix3 const& Racc_in) { Racc = Racc_in; computeLinearProcessNoiseTemplate(); }
    void setExtendedQ(MatrixNX const& Qext_in) { Qext = Qext_in; }

  private:
    // Original MEKF internals (kept nomenclature)
    Eigen::Quaternion<T> qref;
    Vector3 v1ref;
    Vector3 v2ref;

    // Original base error-state (first BASE_N elements) — now stored inside xext (top portion)
    // But we keep a mirror of original P for compatibility
    Matrix<T, BASE_N, 1> xbase;
    MatrixBaseN Pbase;

    // Extended full state xext and Pext (NX x NX)
    Matrix<T, NX, 1> xext; // [ att_err(3), (bias 3 optional), v(3), p(3), S(3) ]
    MatrixNX Pext;

    // Original quaternion transition matrix
    Matrix4 F;

    // Original constant matrices (kept)
    const Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase; // original Q for attitude & bias

    // Extended process noise / Q
    MatrixNX Qext;

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3

    Matrix3 Q_Racc_noise; // Process noise for rules using acceleration

    Matrix3 R_S; // triple integration measurement noise

    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    void set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func() const;
    Vector3 magnetometer_measurement_func() const;

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Extended helpers
    void computeLinearProcessNoiseTemplate(); // computes blocks of Qext from Q_Racc_noise and Ts template (Ts supplied in time_update)
    void assembleExtendedFandQ(const Vector3& acc_body, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);
    Matrix3 R_from_quat() const { return qref.toRotationMatrix(); }

    // Quaternion & small-angle helpers (kept)
    Matrix4 smallOmegaMatrix(const Eigen::Ref<const Vector3>& delta_theta) const;
    Vector4 quatMultiply(const Vector4& a, const Vector4& b) const;
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)
    void normalizeQuat();
};

// Implementation

template <typename T, bool with_bias>
QuaternionMEKF<T, with_bias>::QuaternionMEKF(
    Vector3 const& sigma_a,
    Vector3 const& sigma_g,
    Vector3 const& sigma_m,
    T Pq0, T Pb0, T b0, T R_S_noise)
  : Qbase(initialize_Q(sigma_g, b0)),
    Racc(sigma_a.array().square().matrix().asDiagonal()),
    Rmag(sigma_m.array().square().matrix().asDiagonal()),
    R((Vector6() << sigma_a, sigma_m).finished().array().square().matrix().asDiagonal()),
    // initialize Q_Racc_noise from sigma_a as well
    Q_Racc_noise(sigma_a.array().square().matrix().asDiagonal())
{
  // quaternion init
  qref.setIdentity();

  R_S = Matrix3::Identity() * R_S_noise;

  // initialize base / extended states
  xbase.setZero();
  Pbase.setZero();
  Pbase.setIdentity(); // default small initial cov unless user overwrites

  // Extended state
  xext.setZero();
  Pext.setZero();
  // Place original base P into top-left of Pext
  for (int i = 0; i < BASE_N; ++i) for (int j = 0; j < BASE_N; ++j) Pext(i,j) = Pbase(i,j);

  // Initialize Qext: top-left is original Qbase; rest zeros until we compute process noise by template
  Qext.setZero();
  for (int i = 0; i < BASE_N; ++i) for (int j = 0; j < BASE_N; ++j) Qext(i,j) = Qbase(i,j);

  // default extra linear noise: small values
  // computeLinearProcessNoiseTemplate(); // called in time_update when Ts is known
}

template<typename T, bool with_bias>
Matrix<T, BASE_N, BASE_N> QuaternionMEKF<T, with_bias>::initialize_Q(Vector3 sigma_g, T b0) {
  if constexpr (with_bias) {
    return (Matrix<T, BASE_N, BASE_N>() << sigma_g.array().square().matrix(), Matrix3::Zero(),
             Matrix3::Zero(), Matrix3::Identity() * b0).finished();
  } else {
    return sigma_g.array().square().matrix().asDiagonal();
  }
}

//  initialization helpers
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag) {
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
Eigen::Quaternion<T> QuaternionMEKF<T, with_bias>::quaternion_from_acc(Vector3 const& acc) {
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
void QuaternionMEKF<T, with_bias>::initialize_from_acc(Vector3 const& acc) {
  T const anorm = acc.norm();
  v1ref << 0, 0, -anorm;
  qref = quaternion_from_acc(acc);
  qref.normalize();
}

// original signature preserved: no accel input
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(Vector3 const& gyr, T Ts) {
  // call new overload with zero acceleration vector for backward compatibility
  Vector3 acc_zero = Vector3::Zero();
  time_update(gyr, acc_zero, Ts);
}

template <typename T, bool with_bias>
Eigen::Quaternion<T> QuaternionMEKF<T, with_bias>::get_quaternion() const {
  return qref;
}

template <typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector3
QuaternionMEKF<T, with_bias>::get_bias() const {
  if constexpr (with_bias) {
    return xext.template segment<3>(3);
  } else {
    return Vector3::Zero();
  }
}

template <typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector3
QuaternionMEKF<T, with_bias>::get_velocity() const {
  // velocity state at offset BASE_N
  return xext.template segment<3>(BASE_N);
}

template <typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector3
QuaternionMEKF<T, with_bias>::get_position() const {
  // position state at offset BASE_N+3
  return xext.template segment<3>(BASE_N + 3);
}

template <typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector3
QuaternionMEKF<T, with_bias>::get_integral_acceleration() const {
  // integral of acceleration state at offset BASE_N+6
  return xext.template segment<3>(BASE_N + 6);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(Vector3 const& gyr, Vector3 const& acc_body, T Ts) {
    // 1) Build quaternion transition matrix
    if constexpr (with_bias) {
        set_transition_matrix(gyr - xext.template segment<3>(3), Ts);
    } else {
        set_transition_matrix(gyr, Ts);
    }

    // 2) Update quaternion
    qref.coeffs() = F * qref.coeffs();
    qref.normalize();

    // 3) World-frame linear acceleration
    Matrix3 Rw = R_from_quat();
    Vector3 g_world{0,0,9.81};
    Vector3 a_w = Rw * acc_body - g_world;  // remove gravity

    // 4) Extract current linear states
    auto v = xext.template segment<3>(BASE_N);
    auto p = xext.template segment<3>(BASE_N + 3);
    auto S = xext.template segment<3>(BASE_N + 6);

    // 5) Taylor-series propagation
    xext.template segment<3>(BASE_N)     = v + a_w * Ts;
    xext.template segment<3>(BASE_N + 3) = p + v * Ts + 0.5 * a_w * Ts*Ts;
    xext.template segment<3>(BASE_N + 6) = S + p * Ts + 0.5 * v * Ts*Ts + (Ts*Ts*Ts / T(6.0)) * a_w;

    // 6) Assemble extended Jacobian and Q
    MatrixNX F_a_ext, Q_a_ext;
    assembleExtendedFandQ(acc_body, Ts, F_a_ext, Q_a_ext);

    // 7) Covariance propagation using Joseph form
    Pext = F_a_ext * Pext * F_a_ext.transpose() + Q_a_ext;

    // 8) Mirror base covariance
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

// measurement update
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(
    Vector3 const& acc,
    Vector3 const& mag)
{
    // --- Predicted measurements ---
    Vector3 v1hat = accelerometer_measurement_func();
    Vector3 v2hat = magnetometer_measurement_func();

    // --- Build extended measurement Jacobian (6 x NX) ---
    Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
    Cext.block<3,3>(0,0) = skew_symmetric_matrix(v1hat); // accelerometer
    Cext.block<3,3>(3,0) = skew_symmetric_matrix(v2hat); // magnetometer

    // --- Measurement vector & innovation ---
    Vector6 yhat; yhat << v1hat, v2hat;
    Vector6 y;    y << acc, mag;
    Vector6 inno = y - yhat;

    // --- Innovation covariance ---
    MatrixM S_mat = Cext * Pext * Cext.transpose() + R;

    // --- Kalman gain via solve ---
    Eigen::FullPivLU<MatrixM> lu(S_mat);
    if (!lu.isInvertible()) return;
    Matrix<T, NX, M> K = lu.solve(Pext * Cext.transpose());

    // --- Update state ---
    xext += K * inno;

    // --- Joseph-form covariance update ---
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * R * K.transpose();

    // --- Quaternion correction ---
    applyQuaternionCorrectionFromErrorState();

    // --- Clear small-angle error entries ---
    xext.head<3>().setZero();
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_partial(
    const Eigen::Ref<const Vector3>& meas,
    const Eigen::Ref<const Vector3>& vhat,
    const Eigen::Ref<const Matrix3>& Rm)
{
  Matrix3 const C1 = skew_symmetric_matrix(vhat);
  // Build Cext (3 x NX)
  Matrix<T, 3, NX> Cext;
  Cext.setZero();
  Cext.template block<3,3>(0,0) = C1; // attitude part
  // rest zeros (including bias, v,p,S)

  Vector3 const inno = meas - vhat;
  Matrix3 const S3 = Cext * Pext * Cext.transpose() + Rm;

  Eigen::FullPivLU<Matrix3> lu(S3);
  if (lu.isInvertible()) {
    Matrix<T, NX, 3> Kext = Pext * Cext.transpose() * lu.inverse();
    xext += Kext * inno;

    Matrix<T, NX, NX> Iext = Matrix<T, NX, NX>::Identity();
    Matrix<T, NX, NX> temp = Iext - Kext * Cext;
    Pext = temp * Pext * temp.transpose() + Kext * Rm * Kext.transpose();

    // apply quaternion correction same as before
    applyQuaternionCorrectionFromErrorState();
    for (int i=0;i<3;++i) xext(i) = T(0);
  }
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(Vector3 const& acc) {
  Vector3 const v1hat = accelerometer_measurement_func();
  measurement_update_partial(acc, v1hat, Racc);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(Vector3 const& mag) {
  Vector3 const v2hat = magnetometer_measurement_func();
  measurement_update_partial(mag, v2hat, Rmag);
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::accelerometer_measurement_func() const {
  return qref.inverse() * v1ref;
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::magnetometer_measurement_func() const {
  return qref.inverse() * v2ref;
}

// utility functions
template<typename T, bool with_bias>
Matrix<T, 3, 3> QuaternionMEKF<T, with_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
  Matrix3 M;
  M << 0, -vec(2), vec(1),
       vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return M;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::set_transition_matrix(Eigen::Ref<const Vector3> const& gyr, T Ts) {
  Vector3 const delta_theta = gyr * Ts;
  T un = delta_theta.norm();
  if (un == 0) {
    un = std::numeric_limits<T>::min();
  }
  Matrix4 const Omega = (Matrix4() << -skew_symmetric_matrix(delta_theta), delta_theta,
                                      -delta_theta.transpose(),            0          ).finished();
  F = std::cos(half * un) * Matrix4::Identity() + std::sin(half * un) / un * Omega;
}

// quaternion multiplication helper (vector form used rarely in this file)
template<typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector4 QuaternionMEKF<T, with_bias>::quatMultiply(const Vector4& a, const Vector4& b) const {
  Vector4 r;
  Eigen::Matrix<T,3,1> av = a.template head<3>(), bv = b.template head<3>();
  T aw = a(3), bw = b(3);
  r.template head<3>() = aw * bv + bw * av + av.cross(bv);
  r(3) = aw * bw - av.dot(bv);
  return r;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::applyQuaternionCorrectionFromErrorState() {
  // xext(0..2) contains the small-angle error — same as original code; create corr quaternion and apply
  Eigen::Quaternion<T> corr(T(1), half * xext(0), half * xext(1), half * xext(2));
  corr.normalize();
  qref = qref * corr;
  qref.normalize();
}

// normalize quaternion
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::normalizeQuat() {
  qref.normalize();
}

//  Extended pseudo-measurement: zero S 
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::applyIntegralZeroPseudoMeas() {
    // --- Build measurement matrix H (picks S block) ---
    Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
    H.block<3,3>(0, BASE_N + 6) = Matrix3::Identity();

    // --- Innovation (desired S = 0) ---
    Vector3 z = Vector3::Zero();
    Vector3 inno = z - H * xext;

    // --- Innovation covariance ---
    Matrix3 S_mat = H * Pext * H.transpose() + R_S;

    // --- Solve K * S_mat = Pext * H^T efficiently ---
    Eigen::FullPivLU<Matrix3> lu(S_mat);
    if (!lu.isInvertible()) return;

    Matrix<T, NX, 3> K = lu.solve(Pext * H.transpose());

    // --- Update extended state ---
    xext += K * inno;

    // --- Covariance Joseph form ---
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * H) * Pext * (I - K * H).transpose() + K * R_S * K.transpose();

    // --- Apply quaternion correction if attitude error changed ---
    applyQuaternionCorrectionFromErrorState();

    // --- Clear small-angle entries (first 3) ---
    xext.head<3>().setZero();

    // --- Update base covariance ---
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::assembleExtendedFandQ(
    const Vector3& acc_body,
    T Ts,
    Matrix<T, NX, NX>& F_a_ext,
    MatrixNX& Q_a_ext)
{
    F_a_ext.setIdentity();
    Q_a_ext = Qext; // start with template

    // --- Top-left: original MEKF F ---
    if constexpr (with_bias) {
        F_a_ext.topLeftCorner(3,3) = F.topLeftCorner(3,3);
        F_a_ext.block<3,3>(0,3) = Matrix3::Identity() * (-Ts); // attitude->bias
    } else {
        F_a_ext.topLeftCorner(3,3) = F.topLeftCorner(3,3);
    }

    // --- Gravity-free acceleration ---
    Matrix3 Rw = R_from_quat();
    Vector3 g_world{0,0,9.81};
    Vector3 a_w = Rw * acc_body - g_world; // remove gravity
    Matrix3 skew_aw = skew_symmetric_matrix(a_w);

    // --- Attitude → linear Jacobians ---
    F_a_ext.block<3,3>(BASE_N, 0)     = -Ts * (Rw * skew_aw);         // v
    F_a_ext.block<3,3>(BASE_N+3,0)    = -0.5*Ts*Ts*(Rw*skew_aw);     // p
    F_a_ext.block<3,3>(BASE_N+6,0)    = -(Ts*Ts*Ts/6.0)*(Rw*skew_aw); // S

    // --- Linear dependencies ---
    F_a_ext.block<3,3>(BASE_N+3, BASE_N) = Matrix3::Identity() * Ts;        // v -> p
    F_a_ext.block<3,3>(BASE_N+6, BASE_N) = Matrix3::Identity() * (0.5*Ts*Ts); // v -> S
    F_a_ext.block<3,3>(BASE_N+6, BASE_N+3) = Matrix3::Identity() * Ts;      // p -> S

    // --- Process noise ---
    Matrix<T,9,3> G;
    G << Ts*Rw, 0.5*Ts*Ts*Rw, (Ts*Ts*Ts/6.0)*Rw;  // block-wise concatenation
    Q_a_ext.block(BASE_N, BASE_N, 9,9) = G * Q_Racc_noise * G.transpose();
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::computeLinearProcessNoiseTemplate() {
    // Precompute the template for linear-state process noise (v,p,S) using Racc
    // G_template contains only rotation matrices, without Ts scaling
    // So for time_update, Qlin = G(Ts) * Racc * G(Ts)^T

    // Just store identity template; actual scaling by Ts^1/2, Ts^2/2 etc. is done in assembleExtendedFandQ
    // Essentially, we store Racc here for convenience
    Q_Racc_noise = Racc;

    // Optional: could zero out bottom-right of Qext to be safe
    for (int i = BASE_N; i < NX; ++i)
        for (int j = BASE_N; j < NX; ++j)
            Qext(i,j) = 0;
}
