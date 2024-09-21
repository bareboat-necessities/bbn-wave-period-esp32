#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf

  C++ implementation of the Quaternion Multiplicative Extended Kalman Filter (Q-MEKF), with support 
  for accelerometer and magnetometer measurements.

  Based on the following papers:

    Lefferts, Ern J., F. Landis Markley, and Malcolm D. Shuster. "Kalman filtering for spacecraft attitude estimation." 
    Journal of Guidance, control, and Dynamics 5.5 (1982): 417-429.
    
    Markley, F. Landis. "Attitude error representations for Kalman filtering." 
    Journal of guidance, control, and dynamics 26.2 (2003): 311-317.


    https://www.researchgate.net/profile/Landis-Markley/publication/245432681_Attitude_Error_Representations_for_Kalman_Filtering/links/557c5d4108aeea18b776a727/Attitude-Error-Representations-for-Kalman-Filtering.pdf

  Dependencies: Eigen3

 */

#include <ArduinoEigenDense.h>

using Eigen::Matrix;
using Eigen::Map;

template <typename T = float, bool with_bias = true>
class QuaternionMEKF {
    // State dimension
    static constexpr int N = with_bias ? 6 : 3;
    // Measurement dimension
    static const int M = 6;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, 4, 1> Vector4;
    typedef Matrix<T, 6, 1> Vector6;
    typedef Matrix<T, N, N> MatrixN;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, 4, 4> Matrix4;
    typedef Matrix<T, M, M> MatrixM;
    static constexpr T half = T(1) / T(2);

  public:
    QuaternionMEKF(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m, T Pq0 = 1e-6, T Pb0 = 1e-1);
    constexpr QuaternionMEKF(T const sigma_a[3], T const sigma_g[3], T const sigma_m[3], T Pq0 = 1e-6, T Pb0 = 1e-1);
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc_mag(T const acc[3], T const mag[3]);
    void initialize_from_acc(Vector3 const& acc);
    void initialize_from_acc(T const acc[3]);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);
    void time_update(Vector3 const& gyr, T Ts);
    void time_update(T const gyr[3], T Ts);
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update(T const acc[3], T const mag[3]);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_acc_only(T const acc[3]);
    void measurement_update_mag_only(Vector3 const& mag);
    void measurement_update_mag_only(T const mag[3]);
    Vector4 const& quaternion() const;
    MatrixN const& covariance() const;
    Vector3 gyroscope_bias() const;

  private:
    Eigen::Quaternion<T> qref;

    Vector3 v1ref;
    Vector3 v2ref;

    // State
    Matrix<T, N, 1> x;
    // State covariance
    MatrixN P;

    // Quaternion update matrix
    Matrix4 F;

    // Constant matrices
    const Matrix3 Racc, Rmag;
    const MatrixM R;
    const MatrixN Q;

    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    void set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func() const;
    Vector3 magnetometer_measurement_func() const;

    static constexpr MatrixN initialize_Q(Vector3 sigma_g);
};

template <typename T, bool with_bias>
QuaternionMEKF<T, with_bias>::QuaternionMEKF(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m, T Pq0, T Pb0)
  : Q(initialize_Q(sigma_g)),
  Racc(sigma_a.array().square().matrix().asDiagonal()),
  Rmag(sigma_m.array().square().matrix().asDiagonal()),
  R((Vector6() << sigma_a, sigma_m).finished().array().square().matrix().asDiagonal()) {

  qref.setIdentity();
  x.setZero();

  if constexpr (with_bias) {
    P << Pq0*Matrix3::Identity(), Matrix3::Zero(),
         Matrix3::Zero(), Pb0*Matrix3::Identity();
  }
  else {
    P = Pq0 * Matrix3::Identity();
  }
}

template<typename T, bool with_bias>
constexpr QuaternionMEKF<T, with_bias>::QuaternionMEKF(T const sigma_a[3], T const sigma_g[3], T const sigma_m[3], T Pq0, T Pb0) :
  QuaternionMEKF(Map<Matrix<T, 3, 1>>(sigma_a), Map<Matrix<T, 3, 1>>(sigma_g), Map<Matrix<T, 3, 1>>(sigma_m), Pq0, Pb0) {
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag) {
  T const anorm = acc.norm();
  v1ref << 0, 0, -anorm;

  Vector3 const acc_normalized = acc / anorm;
  Vector3 const mag_normalized = mag.normalized();

  Vector3 const Rz = -acc_normalized;
  Vector3 const Ry = Rz.cross(mag_normalized).normalized();
  Vector3 const Rx = Ry.cross(Rz).normalized();

  // Construct the rotation matrix
  Matrix3 const R = (Matrix3() << Rx, Ry, Rz).finished();

  // Eigen can convert it to a quaternion
  qref = R.transpose();

  // Reference magnetic field vector
  v2ref = qref * mag;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(T const acc[3], T const mag[3]) {
  initialize_from_acc_mag(Map<Matrix<T, 3, 1>>(acc), Map<Matrix<T, 3, 1>>(mag));
}

template<typename T, bool with_bias>
Eigen::Quaternion<T> QuaternionMEKF<T, with_bias>::quaternion_from_acc(Vector3 const& acc) {
  // This finds inverse of qref
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
  // Invert the quaternion
  Eigen::Quaternion<T> qref = Eigen::Quaternion<T>(qw, -qx, -qy, -qz);
  return qref;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc(Vector3 const& acc) {
  T const anorm = acc.norm();
  v1ref << 0, 0, -anorm;
  qref = quat_from_acc(acc);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc(T const acc[3]) {
  initialize_from_acc(Map<Matrix<T, 3, 1>>(acc));
}

template <typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(Vector3 const& gyr, T Ts) {
  if constexpr (with_bias) {
    set_transition_matrix(gyr - x.tail(3), Ts);
  }
  else {
    set_transition_matrix(gyr, Ts);
  }

  // Quaternionf.coeffs() get the components in [x,y,z,w] order
  qref = F * qref.coeffs();
  qref.normalize();

  MatrixN F_a;
  // Slice 3x3 block from F
  if constexpr (with_bias) {
    F_a << F.block(0, 0, 3, 3), (-Matrix3::Identity()*Ts),
           Matrix3::Zero(), Matrix3::Identity();
  }
  else {
    F_a = F.block(0, 0, 3, 3);
  }
  P = F_a * P * F_a.transpose() + Q;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(T const gyr[3], T Ts) {
  time_update(Map<Matrix<T, 3, 1>>(gyr), Ts);
}

template <typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(Vector3 const& acc, Vector3 const& mag) {
  // Predicted measurements
  Vector3 const v1hat = accelerometer_measurement_func();
  Vector3 const v2hat = magnetometer_measurement_func();

  Matrix3 const C1 = skew_symmetric_matrix(v1hat);
  Matrix3 const C2 = skew_symmetric_matrix(v2hat);

  Matrix<T, M, N> C;
  if constexpr (with_bias) {
    C << C1, Matrix<T, 3, 3>::Zero(),
    C2, Matrix<T, 3, 3>::Zero();
  }
  else {
    C << C1,
    C2;
  }

  Vector6 const yhat = (Vector6() << v1hat,
                                     v2hat).finished();
  Vector6 const y = (Vector6() << acc,
                                  mag).finished();
  Vector6 const inno = y - yhat;
  MatrixM const s = C * P * C.transpose() + R;

  // K = P * C.T *(s)^-1
  // K * s = P*C.T

  // This is the form
  // x * A = b
  // Which can be solved with the code below
  Eigen::FullPivLU<MatrixM> lu(s);
  if (lu.isInvertible()) {
    Matrix<T, N, M> const K = P * C.transpose() * lu.inverse();

    x += K * inno;

    // Joseph form of covariance measurement update
    MatrixN const temp = MatrixN::Identity() - K * C;
    P = temp * P * temp.transpose() + K * R * K.transpose();
    // Apply correction to qref
    Eigen::Quaternion<T> corr(1, half * x(0), half * x(1), half * x(2));
    corr.normalize();
    qref = qref * corr;

    // We only want to reset the quaternion part of the state
    x(0) = 0;
    x(1) = 0;
    x(2) = 0;
  }
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(T const acc[3], T const mag[3]) {
  measurement_update(Map<Matrix<T, 3, 1>>(acc), Map<Matrix<T, 3, 1>>(mag));
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_partial(Eigen::Ref<Vector3 const> const& meas, Eigen::Ref<Vector3 const> const& vhat, Eigen::Ref<Matrix3 const>const& Rm) {
  Matrix3 const C1 = skew_symmetric_matrix(vhat);

  Matrix<T, 3, N> C;
  if constexpr (with_bias) {
    C << C1, Matrix<T, 3, 3>::Zero();
  }
  else {
    C = C1;
  }
  Vector3 const inno = meas - vhat;
  Matrix3 const s = C * P * C.transpose() + Rm;

  // K = P * C.T *(s)^-1
  // K * s = P*C.T

  // This is the form
  // x * A = b
  // Which can be solved with the code below
  Eigen::FullPivLU<Matrix3> lu(s);
  if (lu.isInvertible()) {
    Matrix<T, N, 3> const K = P * C.transpose() * lu.inverse();

    x += K * inno;

    // Joseph form of covariance measurement update
    MatrixN const temp = MatrixN::Identity() - K * C;
    P = temp * P * temp.transpose() + K * Racc * K.transpose();
    // Apply correction to qref
    Eigen::Quaternion<T> corr(1, half * x(0), half * x(1), half * x(2));
    corr.normalize();
    qref = qref * corr;

    // We only want to reset the quaternion part of the state
    x(0) = 0;
    x(1) = 0;
    x(2) = 0;
  }
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(Vector3 const& acc) {
  Vector3 const v1hat = accelerometer_measurement_func();
  measurement_update_partial(acc, v1hat, Racc);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(T const acc[3]) {
  measurement_update_acc_only(Map<Matrix<T, 3, 1>>(acc));
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(Vector3 const& mag) {
  Vector3 const v2hat = magnetometer_measurement_func();
  measurement_update_partial(mag, v2hat, Rmag);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(T const mag[3]) {
  measurement_update_mag_only(Map<Matrix<T, 3, 1>>(mag));
}

template<typename T, bool with_bias>
Matrix<T, 4, 1> const& QuaternionMEKF<T, with_bias>::quaternion() const {
  return qref.coeffs();
}

template<typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::MatrixN const& QuaternionMEKF<T, with_bias>::covariance() const {
  return P;
}

template<typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::Vector3 QuaternionMEKF<T, with_bias>::gyroscope_bias() const {
  return x.tail(3);
}

template <typename T, bool with_bias>
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

template <typename T, bool with_bias>
Matrix<T, 3, 3> QuaternionMEKF<T, with_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
  Matrix3 M;
  M << 0, -vec(2), vec(1),
       vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return M;
}

template <typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::accelerometer_measurement_func() const {
  return qref.inverse() * v1ref;
}

template <typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::magnetometer_measurement_func() const {
  return qref.inverse() * v2ref;
}

template<typename T, bool with_bias>
constexpr typename QuaternionMEKF<T, with_bias>::MatrixN QuaternionMEKF<T, with_bias>::initialize_Q(Vector3 sigma_g) {
  if constexpr (with_bias) {
    return (Vector6() << sigma_g.array().square().matrix(), 1e-12, 1e-12, 1e-12).finished().asDiagonal();
  }
  else {
    return sigma_g.array().square().matrix().asDiagonal();
  }
}

typedef Matrix<float, 3, 1> Vector3f;
typedef Matrix<float, 4, 1> Vector4f;
    
typedef struct QMEKF_vars {
  Vector3f sigma_a = {20.78e-3, 20.78e-3, 20.78e-3};
  Vector3f sigma_g = {0.2020*M_PI/180, 0.2020*M_PI/180, 0.2020*M_PI/180};
  Vector3f sigma_m = {3.2e-3, 3.2e-3, 4.1e-3};
  QuaternionMEKF<float, true>* mekf;
} Kalman_QMEKF_vars;

void QMEKF_init(Kalman_QMEKF_vars* qmekf, Vector3f sigma_a, Vector3f sigma_g, Vector3f sigma_m) {
  static QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m);
  qmekf->mekf = &mekf;
}

