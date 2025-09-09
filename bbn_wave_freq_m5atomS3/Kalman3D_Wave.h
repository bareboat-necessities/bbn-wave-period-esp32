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
    static constexpr int EXT_ADD = 9;
    // New full state dimension
    static constexpr int NX = BASE_N + EXT_ADD;

    // Measurement dimension (unchanged)
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

    // Initialization / measurement API preserved
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Time update: preserved old signature (no acc) plus new overload (gyr + acc)
    void time_update(Vector3 const& gyr, Vector3 const& acc, T Ts);  // new: uses acc to drive v/p/S

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_mag_only(Vector3 const& 
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
    Vector4 quatMultiply(const Vector4& a, const Vector4& b) const;
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)
    void normalizeQuat();
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

  // Initialize Qext: top-left is original Qbase; rest zeros until we compute process noise by template
  Qext.setZero();
  Qext.topLeftCorner(BASE_N, BASE_N) = Qbase;

  R.setZero();
  R.template topLeftCorner<3,3>()  = Racc;     // accelerometer measurement noise
  R.template bottomRightCorner<3,3>() = Rmag;  // magnetometer measurement noise

  // default extra linear noise: small values
  // computeLinearProcessNoiseTemplate(); // called in time_update when Ts is known
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
  v1ref << 0, 0, -gravity_magnitude;

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
  v1ref << 0, 0, -gravity_magnitude;
  qref = quaternion_from_acc(acc);
  qref.normalize();
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::time_update(const Vector3& gyr_m,
                                              const Vector3& acc_m_b,
                                              T Ts)
{
    // --- 0) Bias-corrected gyro (attitude) ---
    Vector3 bg = Vector3::Zero();
    if constexpr (with_bias) bg = xext.template segment<3>(3);
    const Vector3 omega_b = gyr_m - bg;                // [rad/s] body frame
    const Eigen::Quaternion<T> qk = qref;              // pre-update attitude
    const Matrix3 Rk = qk.toRotationMatrix();          // body->world

    // Previous linear states
    const Vector3 v0 = xext.template segment<3>(BASE_N);
    const Vector3 p0 = xext.template segment<3>(BASE_N + 3);
    const Vector3 S0 = xext.template segment<3>(BASE_N + 6);

    // --- 1) Midpoint attitude for force rotation ---
    auto quat_from_small = [&](const Vector3& dtheta)->Eigen::Quaternion<T> {
        const T th = dtheta.norm();
        if (th <= std::numeric_limits<T>::epsilon()) return Eigen::Quaternion<T>::Identity();
        const T half_th = T(0.5) * th;
        const Vector3 axis = dtheta / th;
        return Eigen::Quaternion<T>(std::cos(half_th),
                                    axis.x()*std::sin(half_th),
                                    axis.y()*std::sin(half_th),
                                    axis.z()*std::sin(half_th));
    };
    const Eigen::Quaternion<T> dq_half = quat_from_small(omega_b * (T(0.5) * Ts));
    const Matrix3 Rmid = (qk * dq_half).toRotationMatrix();

    // --- 2) Strapdown mechanization with midpoint force rotation ---
    const Vector3 gW(0, 0, -gravity_magnitude);
    const Vector3 aW = Rmid * acc_m_b + gW;            // world acceleration at midpoint

    const Vector3 v1 = v0 + aW * Ts;
    const Vector3 p1 = p0 + v0 * Ts + T(0.5) * aW * Ts * Ts;
    const Vector3 S1 = S0 + p0 * Ts + T(0.5) * v0 * Ts * Ts + (Ts*Ts*Ts / T(6)) * aW;

    xext.template segment<3>(BASE_N)       = v1;
    xext.template segment<3>(BASE_N + 3)   = p1;
    xext.template segment<3>(BASE_N + 6)   = S1;

    // --- 3) Attitude update with full-step gyro increment ---
    const Eigen::Quaternion<T> dq_full = quat_from_small(omega_b * Ts);
    qref = (qk * dq_full).normalized();

    // --- 4) Error-state transition matrix F_ext ---
    MatrixNX F_ext = MatrixNX::Identity();
    auto skew = [&](const Vector3& v)->Matrix3 {
        Matrix3 M; M <<    0, -v.z(),  v.y(),
                         v.z(),     0, -v.x(),
                        -v.y(),  v.x(),     0;
        return M;
    };

    F_ext.template block<3,3>(0,0) = Matrix3::Identity() - skew(omega_b) * Ts;
    if constexpr (with_bias) {
        F_ext.template block<3,3>(0,3) = -Matrix3::Identity() * Ts;
    }

    // Attitude → linear coupling via force rotation at midpoint
    const Matrix3 Rmid_skew_ab = Rmid * skew(acc_m_b);
    F_ext.template block<3,3>(BASE_N,   0) = -Ts              * Rmid_skew_ab;
    F_ext.template block<3,3>(BASE_N+3, 0) = -T(0.5)*Ts*Ts    * Rmid_skew_ab;
    F_ext.template block<3,3>(BASE_N+6, 0) = -(Ts*Ts*Ts/T(6)) * Rmid_skew_ab;

    // Linear chain: v→p, (v,p)→S
    F_ext.template block<3,3>(BASE_N+3, BASE_N)   = Matrix3::Identity() * Ts;
    F_ext.template block<3,3>(BASE_N+6, BASE_N)   = Matrix3::Identity() * (T(0.5)*Ts*Ts);
    F_ext.template block<3,3>(BASE_N+6, BASE_N+3) = Matrix3::Identity() * Ts;

    // --- 5) Closed-form discrete Q for [v,p,S] ---
    const Matrix3 Qa_w = Rmid * Q_Racc_noise * Rmid.transpose();

    const T dt  = Ts, dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt2*dt2, dt5 = dt2*dt3;

    const Matrix3 Qvv = Qa_w * dt;
    const Matrix3 Qvp = Qa_w * (dt2 * T(0.5));
    const Matrix3 Qpp = Qa_w * (dt3 / T(3));
    const Matrix3 QvS = Qa_w * (dt3 / T(6));
    const Matrix3 QpS = Qa_w * (dt4 / T(8));
    const Matrix3 QSS = Qa_w * (dt5 / T(20));

    Matrix<T,9,9> Qlin; Qlin.setZero();
    auto blk = [&](int r,int c)->Eigen::Block<Matrix<T,9,9>,3,3>{ return Qlin.template block<3,3>(r,c); };
    blk(0,0)=Qvv; blk(0,3)=Qvp; blk(0,6)=QvS;
    blk(3,0)=Qvp; blk(3,3)=Qpp; blk(3,6)=QpS;
    blk(6,0)=QvS; blk(6,3)=QpS; blk(6,6)=QSS;

    MatrixNX Qk = Qext;
    Qk.template block(BASE_N, BASE_N, 9, 9) = Qlin;

    // --- 6) Covariance update (Joseph form) ---
    MatrixNX I = MatrixNX::Identity();
    Pext = F_ext * Pext * F_ext.transpose() + Qk;
    Pext = T(0.5) * (Pext + Pext.transpose());   // enforce symmetry
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);

    // --- 7) Bookkeeping for optional pseudo-meas ---
    last_gyr_bias_corrected = omega_b;
    last_Ts = Ts;

    applyIntegralZeroPseudoMeas();
}

// measurement update
template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update(
    Vector3 const& acc,
    Vector3 const& mag)
{
    // Predicted measurements
    Vector3 v1hat = accelerometer_measurement_func();
    Vector3 v2hat = magnetometer_measurement_func();
    
    // Cext: (6 x NX)
    Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
    Cext.template block<3,3>(0,0) = skew_symmetric_matrix(v1hat);
    Cext.template block<3,3>(3,0) = skew_symmetric_matrix(v2hat);
    
    // Innovation
    Vector6 yhat; yhat << v1hat, v2hat;
    Vector6 y;    y << acc, mag;
    Vector6 inno = y - yhat;
    
    // S = C P C^T + R  (6x6)
    MatrixM S_mat = Cext * Pext * Cext.transpose() + R;
    
    // PCt = P C^T  (NX x 6)
    Matrix<T, NX, M> PCt = Pext * Cext.transpose();
    
    // Factor S (SPD), solve K = PCt * S^{-1}
    Eigen::LDLT<MatrixM> ldlt(S_mat);
    if (ldlt.info() != Eigen::Success) {
        // small jitter if needed
        S_mat += MatrixM::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R.norm());
        ldlt.compute(S_mat);
        if (ldlt.info() != Eigen::Success) return;
    }
    Matrix<T, NX, M> K = PCt * ldlt.solve(MatrixM::Identity());
    
    // State update
    xext.noalias() += K * inno;
    
    // Joseph covariance update (keeps symmetry/PSD)
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * R * K.transpose();
    // optional: enforce exact symmetry numerically
    Pext = T(0.5) * (Pext + Pext.transpose());
    
    // Quaternion correction + zero small-angle
    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
    
    // Mirror base covariance
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
void Kalman3D_Wave<T, with_bias>::measurement_update_acc_only(Vector3 const& acc) {
    Vector3 const v1hat = accelerometer_measurement_func();
    measurement_update_partial(acc, v1hat, Racc);
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::measurement_update_mag_only(Vector3 const& mag) {
    Vector3 const v2hat = magnetometer_measurement_func();
    measurement_update_partial(mag, v2hat, Rmag);
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> Kalman3D_Wave<T, with_bias>::accelerometer_measurement_func() const {
    return qref.inverse() * v1ref;
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
        S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * R.norm());  // small jitter; epsilon may be too tiny in practice
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
void Kalman3D_Wave<T, with_bias>::assembleExtendedFandQ(
    const Vector3& acc_body,   // measured specific force in body (f_b ≈ a^W in body + gravity removed)
    T Ts,
    Matrix<T, NX, NX>& F_a_ext,   // OUT: discrete-time error-state transition
    MatrixNX&          Q_a_ext)   // OUT: discrete-time process covariance
{
    // === 0) Setup ===
    F_a_ext.setIdentity();
    Q_a_ext = Qext;  // start from your template; we'll overwrite the (v,p,S) 9x9 block

    // Access pre-update attitude (qref is the *current* nominal at call time).
    // We use the gyroscope bias-corrected rate captured in last_gyr_bias_corrected (set by caller).
    const Eigen::Quaternion<T> qk = qref;
    const Matrix3 Rk = qk.toRotationMatrix();

    // Small helper
    auto skew = [&](const Vector3& v)->Matrix3 {
        Matrix3 M; M <<    0, -v.z(),  v.y(),
                         v.z(),     0, -v.x(),
                        -v.y(),  v.x(),     0;
        return M;
    };

    // === 1) Midpoint attitude (for force rotation & Jacobians) ===
    // R_mid = R_k * Exp(0.5 * (ω_m - b_g) * Ts)
    const Vector3 omega_b = last_gyr_bias_corrected;  // already gyro-bias corrected by caller
    const T th = (omega_b * (T(0.5) * Ts)).norm();
    Eigen::Quaternion<T> dq_half;
    if (th <= std::numeric_limits<T>::epsilon()) {
        dq_half = Eigen::Quaternion<T>::Identity();
    } else {
        const T half_th = th;
        const Vector3 axis = (omega_b * (T(0.5) * Ts)) / th;
        dq_half = Eigen::Quaternion<T>(std::cos(half_th),
                                       axis.x()*std::sin(half_th),
                                       axis.y()*std::sin(half_th),
                                       axis.z()*std::sin(half_th));
    }
    const Matrix3 Rmid = (qk * dq_half).toRotationMatrix();

    // === 2) Discrete-time error-state transition F_ext ===
    // Attitude error sub-block: δθ_{k+1} ≈ (I - [ω_b]× Δt) δθ_k - I Δt δb_g
    F_a_ext.template block<3,3>(0,0) = Matrix3::Identity() - skew(omega_b) * Ts;
    if constexpr (with_bias) {
        F_a_ext.template block<3,3>(0,3) = -Matrix3::Identity() * Ts;   // θ depends on b_g
    }

    // Attitude → linear coupling via midpoint rotation:
    // a^W = R_mid * f_b + g  ⇒ δa^W ≈ - R_mid [f_b]× δθ
    const Matrix3 Rmid_skew_ab = Rmid * skew(acc_body);
    F_a_ext.template block<3,3>(BASE_N,   0) = -Ts                  * Rmid_skew_ab;         // into v
    F_a_ext.template block<3,3>(BASE_N+3, 0) = -T(0.5)*Ts*Ts        * Rmid_skew_ab;         // into p
    F_a_ext.template block<3,3>(BASE_N+6, 0) = -(Ts*Ts*Ts/T(6))     * Rmid_skew_ab;         // into S

    // Linear chain (v → p, (v,p) → S)
    F_a_ext.template block<3,3>(BASE_N+3, BASE_N)     = Matrix3::Identity() * Ts;           // v -> p
    F_a_ext.template block<3,3>(BASE_N+6, BASE_N)     = Matrix3::Identity() * (T(0.5)*Ts*Ts); // v -> S
    F_a_ext.template block<3,3>(BASE_N+6, BASE_N+3)   = Matrix3::Identity() * Ts;           // p -> S

    // === 3) Closed-form discrete Q_k for [v,p,S] under white accel noise ===
    // Rotate body-frame accel PSD into world with R_mid
    const Matrix3 Qa_w = Rmid * Q_Racc_noise * Rmid.transpose();

    const T dt  = Ts;
    const T dt2 = dt*dt;
    const T dt3 = dt2*dt;
    const T dt4 = dt2*dt2;
    const T dt5 = dt2*dt3;

    // Classic double integrator blocks (v,p)
    const Matrix3 Qvv = Qa_w * dt;                  // ∫_0^dt 1 dτ
    const Matrix3 Qvp = Qa_w * (dt2 * T(0.5));      // ∫_0^dt τ dτ = dt^2/2
    const Matrix3 Qpp = Qa_w * (dt3 / T(3));        // ∫_0^dt τ^2 dτ = dt^3/3

    // Extend to S = ∫ p dt (third integrator)
    // Coefficients come from integrating the polynomial basis once more.
    const Matrix3 QvS = Qa_w * (dt3 / T(6));        // cross(v,S)
    const Matrix3 QpS = Qa_w * (dt4 / T(8));        // cross(p,S)
    const Matrix3 QSS = Qa_w * (dt5 / T(20));       // var(S)

    Matrix<T,9,9> Qlin; Qlin.setZero();
    auto blk = [&](int r,int c)->Eigen::Block<Matrix<T,9,9>,3,3>{ return Qlin.template block<3,3>(r,c); };
    // Order inside the 9×9 block: [v, p, S]
    blk(0,0) = Qvv;  blk(0,3) = Qvp;  blk(0,6) = QvS;
    blk(3,0) = Qvp;  blk(3,3) = Qpp;  blk(3,6) = QpS;
    blk(6,0) = QvS;  blk(6,3) = QpS;  blk(6,6) = QSS;

    // Drop into the extended covariance
    Q_a_ext.template block(BASE_N, BASE_N, 9, 9) = Qlin;
}

template<typename T, bool with_bias>
void Kalman3D_Wave<T, with_bias>::computeLinearProcessNoiseTemplate() {
    // Precompute the template for linear-state process noise (v,p,S) using Racc
    // G_template contains only rotation matrices, without Ts scaling
    // So for time_update, Qlin = G(Ts) * Racc * G(Ts)^T

    // Just store identity template; actual scaling by Ts^1/2, Ts^2/2 etc. is done in assembleExtendedFandQ
    // Essentially, we store Racc here for convenience
    Q_Racc_noise = Racc;

    // zero out bottom-right of Qext to be safe
    Qext.template block(BASE_N, BASE_N, NX-BASE_N, NX-BASE_N).setZero();
}
