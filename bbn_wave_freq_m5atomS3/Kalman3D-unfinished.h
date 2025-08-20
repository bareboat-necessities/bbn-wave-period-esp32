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
    // ------------------- Constructor -------------------
    QuaternionMEKF(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12)) :
      Qbase(initialize_Q(sigma_g, b0)),
      Racc(sigma_a.array().square().matrix().asDiagonal()),
      Rmag(sigma_m.array().square().matrix().asDiagonal()),
      R((Vector6() << sigma_a, sigma_m).finished().array().square().matrix().asDiagonal()),
      Racc_noise(sigma_a.array().square().matrix().asDiagonal())
    {
      // quaternion init
      qref.setIdentity();

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
    }

    constexpr QuaternionMEKF(T const sigma_a[3], T const sigma_g[3], T const sigma_m[3],
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12)) :
      QuaternionMEKF(Vector3(sigma_a), Vector3(sigma_g), Vector3(sigma_m), Pq0, Pb0, b0) {}

    // ------------------- Initialization -------------------
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag) {
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

    void initialize_from_acc_mag(T const acc[3], T const mag[3]) {
      initialize_from_acc_mag(Map<Vector3>(acc), Map<Vector3>(mag));
    }

    void initialize_from_acc(Vector3 const& acc) {
      T const anorm = acc.norm();
      v1ref << 0, 0, -anorm;
      qref = quaternion_from_acc(acc);
      qref.normalize();
    }

    void initialize_from_acc(T const acc[3]) {
      initialize_from_acc(Map<Vector3>(acc));
    }

    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc) {
      T qx, qy, qz, qw;
      if (acc[2] >= 0) {
        qx = std::sqrt((1 + acc[2]) / 2);
        qw = acc[1] / (2 * qx);
        qy = 0;
        qz = -acc[0] / (2 * qx);
      } else {
        qw = std::sqrt((1 - acc[2]) / 2);
        qx = acc[1] / (2 * qw);
        qy = -acc[0] / (2 * qw);
        qz = 0;
      }
      Eigen::Quaternion<T> qref_local = Eigen::Quaternion<T>(qw, -qx, -qy, -qz);
      qref_local.normalize();
      return qref_local;
    }

    // ------------------- Time update -------------------
    void time_update(Vector3 const& gyr, T Ts) {
      Vector3 acc_zero = Vector3::Zero();
      time_update(gyr, acc_zero, Ts);
    }

    void time_update(Vector3 const& gyr, Vector3 const& acc_body, T Ts) {
      if constexpr (with_bias) {
        Vector3 bias = xext.template segment<3>(3);
        set_transition_matrix(gyr - bias, Ts);
      } else {
        set_transition_matrix(gyr, Ts);
      }

      qref = F * qref.coeffs();
      qref.normalize();

      Matrix3 Rw = R_from_quat();
      Vector3 a_w = Rw * acc_body;

      Vector3 v = xext.template segment<3>(BASE_N + 0);
      Vector3 p = xext.template segment<3>(BASE_N + 3);
      Vector3 S = xext.template segment<3>(BASE_N + 6);

      Vector3 v_next = v + a_w * Ts;
      Vector3 p_next = p + v * Ts + 0.5 * a_w * Ts * Ts + (Ts*Ts*Ts / T(6.0)) * a_w;
      Vector3 S_next = S + p * Ts + 0.5 * v * Ts * Ts + (Ts*Ts*Ts / T(6.0)) * a_w;

      xext.template segment<3>(BASE_N + 0) = v_next;
      xext.template segment<3>(BASE_N + 3) = p_next;
      xext.template segment<3>(BASE_N + 6) = S_next;

      Matrix<T, NX, NX> F_a_ext = Matrix<T, NX, NX>::Identity();
      if constexpr (with_bias) {
        Matrix3 F33 = F.block(0,0,3,3);
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(r,c) = F33(r,c);
        for (int r=0;r<3;++r) F_a_ext(r, 3 + r) = -T(Ts);
      } else {
        Matrix3 F33 = F.block(0,0,3,3);
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(r,c) = F33(r,c);
      }

      Matrix3 skew_ab = skew_symmetric_matrix(acc_body);
      Matrix3 J_att_to_v = -Ts * (Rw * skew_ab);
      for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + r, c) = J_att_to_v(r,c);

      Matrix3 J_att_to_p = -0.5 * Ts * Ts * (Rw * skew_ab);
      for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + 3 + r, c) = J_att_to_p(r,c);

      Matrix3 J_att_to_S = -(Ts*Ts*Ts / T(6.0)) * (Rw * skew_ab);
      for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + 6 + r, c) = J_att_to_S(r,c);

      for (int r=0;r<3;++r) F_a_ext(BASE_N + 3 + r, BASE_N + r) = Ts;
      T halfTs2 = T(0.5) * Ts * Ts;
      for (int r=0;r<3;++r) F_a_ext(BASE_N + 6 + r, BASE_N + r) = halfTs2;
      for (int r=0;r<3;++r) F_a_ext(BASE_N + 6 + r, BASE_N + 3 + r) = Ts;

      Matrix<T,9,3> G;
      Matrix3 Rw_local = Rw;
      Matrix3 g1 = Ts * Rw_local;
      Matrix3 g2 = 0.5 * Ts * Ts * Rw_local;
      Matrix3 g3 = (Ts*Ts*Ts / T(6.0)) * Rw_local;
      for (int r=0;r<3;++r) for (int c=0;c<3;++c) { G(r,c) = g1(r,c); G(r+3,c) = g2(r,c); G(r+6,c) = g3(r,c); }

      Matrix<T,9,9> Qlin = G * Racc_noise * G.transpose();
      for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j) if (i >= BASE_N && j >= BASE_N) Qext(i,j) = 0;
      for (int r = 0; r < 9; ++r) for (int c=0;c<9;++c) Qext(BASE_N + r, BASE_N + c) = Qlin(r,c);

      Pext = F_a_ext * Pext * F_a_ext.transpose() + Qext;
      for (int i=0;i<BASE_N;++i) for (int j=0;j<BASE_N;++j) Pbase(i,j) = Pext(i,j);
    }

    void time_update(T const gyr[3], T Ts) {
      time_update(Map<Vector3>(gyr), Ts);
    }

    void time_update(T const gyr[3], T const acc[3], T Ts) {
      time_update(Map<Vector3>(gyr), Map<Vector3>(acc), Ts);
    }

    // ------------------- Measurement update -------------------
    void measurement_update(Vector3 const& acc, Vector3 const& mag) {
      Vector3 const v1hat = accelerometer_measurement_func();
      Vector3 const v2hat = magnetometer_measurement_func();

      Matrix3 const C1 = skew_symmetric_matrix(v1hat);
      Matrix3 const C2 = skew_symmetric_matrix(v2hat);

      Matrix<T, M, NX> Cext;
      Cext.setZero();
      if constexpr (with_bias) {
        Cext.template block<3,3>(0,0) = C1;
        Cext.template block<3,3>(3,0) = C2;
      } else {
        Cext.template block<3,3>(0,0) = C1;
        Cext.template block<3,3>(3,0) = C2;
      }

      Vector6 const yhat = (Vector6() << v1hat, v2hat).finished();
      Vector6 const y = (Vector6() << acc, mag).finished();
      Vector6 const inno = y - yhat;

      MatrixM const Smat = Cext * Pext * Cext.transpose() + R;
      Eigen::FullPivLU<MatrixM> lu(Smat);
      if (lu.isInvertible()) {
        Matrix<T, NX, M> const Kext = Pext * Cext.transpose() * lu.inverse();
        xext += Kext * inno;

        Matrix<T, NX, NX> Iext = Matrix<T, NX, NX>::Identity();
        Matrix<T, NX, NX> temp = Iext - Kext * Cext;
        Pext = temp * Pext * temp.transpose() + Kext * R * Kext.transpose();

        applyQuaternionCorrectionFromErrorState();
        for (int i = 0; i < 3; ++i) xext(i) = T(0);
      }
    }

    void measurement_update(T const acc[3], T const mag[3]) {
      measurement_update(Map<Vector3>(acc), Map<Vector3>(mag));
    }

    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas,
                                    const Eigen::Ref<const Vector3>& vhat,
                                    const Eigen::Ref<const Matrix3>& Rm)
    {
      Matrix3 const C1 = skew_symmetric_matrix(vhat);
      Matrix<T, 3, NX> Cext;
      Cext.setZero();
      Cext.template block<3,3>(0,0) = C1;

      Vector3 const inno = meas - vhat;
      Matrix3 const S3 = Cext * Pext * Cext.transpose() + Rm;

      Eigen::FullPivLU<Matrix3> lu(S3);
      if (lu.isInvertible()) {
        Matrix<T, NX, 3> Kext = Pext * Cext.transpose() * lu.inverse();
        xext += Kext * inno;

        Matrix<T, NX, NX> Iext = Matrix<T, NX, NX>::Identity();
        Matrix<T, NX, NX> temp = Iext - Kext * Cext;
        Pext = temp * Pext * temp.transpose() + Kext * Rm * Kext.transpose();

        applyQuaternionCorrectionFromErrorState();
        for (int i=0;i<3;++i) xext(i) = T(0);
      }
    }

    void measurement_update_acc_only(Vector3 const& acc) {
      Vector3 const v1hat = accelerometer_measurement_func();
      measurement_update_partial(acc, v1hat, Racc);
    }

    void measurement_update_acc_only(T const acc[3]) {
      measurement_update_acc_only(Map<Vector3>(acc));
    }

    void measurement_update_mag_only(Vector3 const& mag) {
      Vector3 const v2hat = magnetometer_measurement_func();
      measurement_update_partial(mag, v2hat, Rmag);
    }

    void measurement_update_mag_only(T const mag[3]) {
      measurement_update_mag_only(Map<Vector3>(mag));
    }

    // ------------------- Extended pseudo-measurement -------------------
    void applyIntegralZeroPseudoMeas() {
      Matrix<T, 3, NX> H;
      H.setZero();
      H.template block<3,3>(0, BASE_N + 6) = Matrix3::Identity();

      Vector3 z = Vector3::Zero();
      Matrix3 S = H * Pext * H.transpose() + Racc;
      Eigen::FullPivLU<Matrix3> lu(S);
      if (!lu.isInvertible()) return;
      Matrix<T,NX,3> K = Pext * H.transpose() * lu.inverse();
      xext = xext + K * (z - H * xext);

      Matrix<T,NX,NX> Iext = Matrix<T,NX,NX>::Identity();
      Matrix<T,NX,NX> temp = Iext - K * H;
      Pext = temp * Pext * temp.transpose() + K * Racc * K.transpose();

      applyQuaternionCorrectionFromErrorState();
      for (int i=0;i<3;++i) xext(i) = T(0);
      for (int i=0;i<BASE_N;++i) for (int j=0;j<BASE_N;++j) Pbase(i,j) = Pext(i,j);
    }

    // ------------------- Accessors -------------------
    Vector4 const& quaternion() const { return qref.coeffs(); }
    MatrixBaseN const& covariance_base() const { return Pbase; }
    MatrixNX const& covariance_full() const { return Pext; }
    Vector3 gyroscope_bias() const { if constexpr (with_bias) return xext.template segment<3>(3); else return Vector3::Zero(); }

    void setLinearProcessNoise(Matrix3 const& Racc_in) { Racc = Racc_in; computeLinearProcessNoiseTemplate(); }
    void setExtendedQ(MatrixNX const& Qext_in) { Qext = Qext_in; }

  private:
    // ------------------- Internals -------------------
    Eigen::Quaternion<T> qref;
    Vector3 v1ref;
    Vector3 v2ref;

    Matrix<T, BASE_N, 1> xbase;
    MatrixBaseN Pbase;

    Matrix<T, NX, 1> xext;
    MatrixNX Pext;

    Matrix4 F;

    const Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase;
    MatrixNX Qext;
    Matrix3 Racc;
    Matrix3 Racc_noise;

    void set_transition_matrix(Eigen::Ref<const Vector3> const& gyr, T Ts) {
      Vector3 delta_theta = gyr * Ts;
      T un = delta_theta.norm();
      if (un == 0) un = std::numeric_limits<T>::min();
      Matrix4 Omega = (Matrix4() << -skew_symmetric_matrix(delta_theta), delta_theta,
                                     -delta_theta.transpose(), 0).finished();
      F = std::cos(half*un) * Matrix4::Identity() + std::sin(half*un)/un * Omega;
    }

    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
      Matrix3 M;
      M << 0,-vec(2),vec(1), vec(2),0,-vec(0), -vec(1),vec(0),0;
      return M;
    }

    Vector3 accelerometer_measurement_func() const { return qref.inverse()*v1ref; }
    Vector3 magnetometer_measurement_func() const { return qref.inverse()*v2ref; }
    Matrix3 R_from_quat() const { return qref.toRotationMatrix(); }

    void applyQuaternionCorrectionFromErrorState() {
      Eigen::Quaternion<T> corr(T(1), half*xext(0), half*xext(1), half*xext(2));
      corr.normalize();
      qref = qref * corr;
      qref.normalize();
    }

    void normalizeQuat() { qref.normalize(); }

    static constexpr MatrixBaseN initialize_Q(Vector3 sigma_g, T b0) {
      if constexpr(with_bias){
        return (Matrix<T,BASE_N,BASE_N>() << sigma_g.array().square().matrix(),
                Matrix3::Zero(),
                Matrix3::Zero(),
                Matrix3::Identity()*b0).finished();
      } else {
        return sigma_g.array().square().matrix().asDiagonal();
      }
    }

    void computeLinearProcessNoiseTemplate() {}
};
