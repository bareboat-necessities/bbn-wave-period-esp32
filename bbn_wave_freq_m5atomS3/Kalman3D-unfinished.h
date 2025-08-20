
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
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12));
    constexpr QuaternionMEKF(T const sigma_a[3], T const sigma_g[3], T const sigma_m[3],
                   T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12));

    // Initialization / measurement API preserved
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc_mag(T const acc[3], T const mag[3]);
    void initialize_from_acc(Vector3 const& acc);
    void initialize_from_acc(T const acc[3]);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Time update: preserved old signature (no acc) plus new overload (gyr + acc)
    void time_update(Vector3 const& gyr, T Ts);                      // original behavior (acc=0)
    void time_update(Vector3 const& gyr, Vector3 const& acc, T Ts);  // new: uses acc to drive v/p/S
    void time_update(T const gyr[3], T Ts);
    void time_update(T const gyr[3], T const acc[3], T Ts);

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update(T const acc[3], T const mag[3]);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_acc_only(T const acc[3]);
    void measurement_update_mag_only(Vector3 const& mag);
    void measurement_update_mag_only(T const mag[3]);

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

    // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 Racc; // already declared const above; but we need to mutate: we'll shadow it; to avoid const conflicts we keep a member Racc_noise
    Matrix3 Racc_noise;

    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    void set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func() const;
    Vector3 magnetometer_measurement_func() const;

    static constexpr MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Extended helpers
    void computeLinearProcessNoiseTemplate(); // computes blocks of Qext from Racc_noise and Ts template (Ts supplied in time_update)
    void assembleExtendedFandQ(const Vector3& acc_body, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);
    Matrix3 R_from_quat() const { return qref.toRotationMatrix(); }

    // Quaternion & small-angle helpers (kept)
    Matrix4 smallOmegaMatrix(const Eigen::Ref<const Vector3>& delta_theta) const;
    Vector4 quatMultiply(const Vector4& a, const Vector4& b) const;
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)
    void normalizeQuat();
};

// ----------------------------- Implementation --------------------------------

template <typename T, bool with_bias>
QuaternionMEKF<T, with_bias>::QuaternionMEKF(
    Vector3 const& sigma_a,
    Vector3 const& sigma_g,
    Vector3 const& sigma_m,
    T Pq0, T Pb0, T b0)
  : Qbase(initialize_Q(sigma_g, b0)),
    Racc(sigma_a.array().square().matrix().asDiagonal()),
    Rmag(sigma_m.array().square().matrix().asDiagonal()),
    R((Vector6() << sigma_a, sigma_m).finished().array().square().matrix().asDiagonal()),
    // initialize Racc_noise from sigma_a as well
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

  // default extra linear noise: small values
  // computeLinearProcessNoiseTemplate(); // called in time_update when Ts is known
}

template<typename T, bool with_bias>
constexpr Matrix<T, BASE_N, BASE_N> QuaternionMEKF<T, with_bias>::initialize_Q(Vector3 sigma_g, T b0) {
  if constexpr (with_bias) {
    return (Matrix<T, BASE_N, BASE_N>() << sigma_g.array().square().matrix(), Matrix3::Zero(),
             Matrix3::Zero(), Matrix3::Identity() * b0).finished();
  } else {
    return sigma_g.array().square().matrix().asDiagonal();
  }
}

// --- initialization helpers preserved ---
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
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(T const acc[3], T const mag[3]) {
  initialize_from_acc_mag(Map<Matrix<T, 3, 1>>(acc), Map<Matrix<T, 3, 1>>(mag));
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

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc(T const acc[3]) {
  initialize_from_acc(Map<Matrix<T, 3, 1>>(acc));
}

// ----------------- core time update (overloads) -----------------

// original signature preserved: no accel input
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(Vector3 const& gyr, T Ts) {
  // call new overload with zero acceleration vector for backward compatibility
  Vector3 acc_zero = Vector3::Zero();
  time_update(gyr, acc_zero, Ts);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(T const gyr[3], T Ts) {
  time_update(Map<Matrix<T, 3, 1>>(gyr), Ts);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(T const gyr[3], T const acc[3], T Ts) {
  time_update(Map<Matrix<T, 3, 1>>(gyr), Map<Matrix<T,3,1>>(acc), Ts);
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
  // 1) Build original quaternion transition matrix F (4x4) based on gyro (as original)
  if constexpr (with_bias) {
    // bias lives in xext: xext[3..5] is gyro bias
    Vector3 bias = xext.template segment<3>(3);
    set_transition_matrix(gyr - bias, Ts);
  } else {
    set_transition_matrix(gyr, Ts);
  }

  // update quaternion as original
  qref = F * qref.coeffs();
  qref.normalize();

  // 2) Compute world-frame acceleration using current quaternion (rotate body->world)
  Matrix3 Rw = R_from_quat();
  Vector3 a_w = Rw * acc_body; // note: gravity is not removed here (apply externally if desired)

  // 3) Extended-state discrete-time kinematics (Taylor-series)
  //    xext ordering:
  //    [ 0..(BASE_N-1) ] : original att-error (3) [+ bias 3 if with_bias]
  //    [ BASE_N .. BASE_N+2 ] : v (3)
  //    [ BASE_N+3 .. BASE_N+5 ] : p (3)
  //    [ BASE_N+6 .. BASE_N+8 ] : S (3)

  // Extract current linear states:
  Vector3 v = xext.template segment<3>(BASE_N + 0);
  Vector3 p = xext.template segment<3>(BASE_N + 3);
  Vector3 S = xext.template segment<3>(BASE_N + 6);

  // Taylor-series propagation:
  // velocity: v_{k+1} = v_k + a_w * Ts  [+ optional 0.5*Ts^2*a term if desired — kept minimal here]
  Vector3 v_next = v + a_w * Ts;

  // position: p_{k+1} = p_k + v_k*Ts + 0.5*a_w*Ts^2 + (1/6)*a_w*Ts^3  (we keep dt^3/6 correction)
  Vector3 p_next = p + v * Ts + 0.5 * a_w * Ts * Ts + (Ts*Ts*Ts / T(6.0)) * a_w;

  // integral S: S_{k+1} = S_k + p_k*Ts + 0.5*v_k*Ts^2 + (1/6)*a_w*Ts^3
  Vector3 S_next = S + p * Ts + 0.5 * v * Ts * Ts + (Ts*Ts*Ts / T(6.0)) * a_w;

  // Write back linear-state predictions into xext
  xext.template segment<3>(BASE_N + 0) = v_next;
  xext.template segment<3>(BASE_N + 3) = p_next;
  xext.template segment<3>(BASE_N + 6) = S_next;

  // 4) Extended covariance propagation: build linearized F_a_ext and Qext for this Ts
  Matrix<T, NX, NX> F_a_ext = Matrix<T, NX, NX>::Identity();
  // Top-left (BASE_N x BASE_N) -- original F_a computed from F (original code)
  // Original code built F_a as either F.block(0,0,3,3) or a 6x6 for bias; mimic that:
  if constexpr (with_bias) {
    // original F_a (6x6) in your code:
    // F_a = [ F.block(0,0,3,3)   -I*Ts
    //         0                I     ]
    Matrix3 F33 = F.block(0,0,3,3);
    // fill top-left 3x3
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(r,c) = F33(r,c);
    // fill top-right block (attitude->bias) with -I*Ts in original
    for (int r=0;r<3;++r) F_a_ext(r, 3 + r) = -T(Ts);
    // bias->bias identity already there
  } else {
    Matrix3 F33 = F.block(0,0,3,3);
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(r,c) = F33(r,c);
  }

  // Now fill linear-state Jacobians:
  // v depends on a_w; linearize v wrt attitude error (phi): δv ≈ -Ts * R * skew(acc_body) * δphi
  Matrix3 skew_ab = skew_symmetric_matrix(acc_body);
  Matrix3 J_att_to_v = -Ts * (Rw * skew_ab);

  // place into F_a_ext: rows for v (BASE_N..BASE_N+2), cols for att-error (0..2)
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + r, c) = J_att_to_v(r,c);

  // p depends on v and a_w: derivative w.r.t att-error = -0.5*Ts^2 * R * skew(acc_body)  (from 0.5 * a_w Ts^2) plus contribution from v term (v depends on att too via previous)
  Matrix3 J_att_to_p = -0.5 * Ts * Ts * (Rw * skew_ab);
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + 3 + r, c) = J_att_to_p(r,c);

  // S depends similarly: derivative = - (Ts^3/6) * R * skew(acc_body)
  Matrix3 J_att_to_S = -(Ts*Ts*Ts / T(6.0)) * (Rw * skew_ab);
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) F_a_ext(BASE_N + 6 + r, c) = J_att_to_S(r,c);

  // Fill v->p mapping: p <- p + v*Ts  => F(p,v) = Ts
  for (int r=0;r<3;++r) F_a_ext(BASE_N + 3 + r, BASE_N + r) = Ts;

  // Fill v->S mapping: S <- S + 0.5 * v * Ts^2  => F(S,v) = 0.5 * Ts^2
  T halfTs2 = T(0.5) * Ts * Ts;
  for (int r=0;r<3;++r) F_a_ext(BASE_N + 6 + r, BASE_N + r) = halfTs2;

  // Fill p->S mapping: S <- S + p * Ts => F(S,p) = Ts
  for (int r=0;r<3;++r) F_a_ext(BASE_N + 6 + r, BASE_N + 3 + r) = Ts;

  // Note: we also keep identity on the diagonal for v,p,S (already set)

  // 5) Build Qext for this Ts
  // Top-left BASE_N x BASE_N remains Qbase (we preserved it in Qext initially)
  // Now compute process noise contribution from accelerometer noise for v,p,S
  // Use discrete integration mapping G = [Ts*Rw; 0.5 Ts^2 * Rw; (1/6) Ts^3 * Rw]
  Matrix<T, 9, 3> G;
  Matrix3 Rw_local = Rw;
  Matrix3 g1 = Ts * Rw_local;
  Matrix3 g2 = (T(0.5) * Ts * Ts) * Rw_local;
  Matrix3 g3 = (Ts*Ts*Ts / T(6.0)) * Rw_local;
  // fill G as block rows
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) { G(r, c) = g1(r,c); G(r+3,c) = g2(r,c); G(r+6,c) = g3(r,c); }

  // Noise covariance for acc in body frame = Racc_noise (3x3)
  Matrix<T,9,9> Qlin = G * Racc_noise * G.transpose();

  // Place Qlin into Qext bottom-right block (for v,p,S)
  // First ensure Qext top-left contains Qbase (already set in ctor)
  // Zero the linear blocks for safety
  for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j) if (i >= BASE_N && j >= BASE_N) Qext(i,j) = 0;

  for (int r = 0; r < 9; ++r) for (int c=0;c<9;++c) Qext(BASE_N + r, BASE_N + c) = Qlin(r,c);

  // 6) Propagate covariance: Pext = F_a_ext * Pext * F_a_ext^T + Qext
  Pext = F_a_ext * Pext * F_a_ext.transpose() + Qext;

  // 7) Also update Pbase mirror for convenience (extract top-left BASE_N)
  for (int i=0;i<BASE_N;++i) for (int j=0;j<BASE_N;++j) Pbase(i,j) = Pext(i,j);

  // Done time_update
}

// ---------------- measurement update (unchanged semantics but applied to extended P)
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(Vector3 const& acc, Vector3 const& mag) {
  // Predicted measurements (using quaternion)
  Vector3 const v1hat = accelerometer_measurement_func();
  Vector3 const v2hat = magnetometer_measurement_func();

  Matrix3 const C1 = skew_symmetric_matrix(v1hat);
  Matrix3 const C2 = skew_symmetric_matrix(v2hat);

  // Build C ext (M x NX) by expanding original C with zeros for extended states
  Matrix<T, M, NX> Cext;
  Cext.setZero();
  if constexpr (with_bias) {
    // original had: C = [C1, 0; C2, 0] where columns are [3 attitude, 3 bias]
    // place these into left columns of Cext
    // top row block
    Cext.template block<3,3>(0,0) = C1;
    // bottom row block
    Cext.template block<3,3>(3,0) = C2;
    // bias columns at 3..5 are zeros as original
  } else {
    Cext.template block<3,3>(0,0) = C1;
    Cext.template block<3,3>(3,0) = C2;
  }

  // Build measurement vectors and innovation
  Vector6 const yhat = (Vector6() << v1hat, v2hat).finished();
  Vector6 const y = (Vector6() << acc, mag).finished();
  Vector6 const inno = y - yhat;

  // Innovation covariance S = Cext * Pext * Cext^T + R
  MatrixM const Smat = Cext * Pext * Cext.transpose() + R;

  // Solve for K: K * Smat = Pext * Cext^T  => K = Pext * Cext^T * Smat^-1
  Eigen::FullPivLU<MatrixM> lu(Smat);
  if (lu.isInvertible()) {
    Matrix<T, NX, M> const Kext = Pext * Cext.transpose() * lu.inverse();

    // state update (extended)
    xext += Kext * inno;

    // Joseph form covariance update on extended covariance
    Matrix<T, NX, NX> Iext = Matrix<T, NX, NX>::Identity();
    Matrix<T, NX, NX> temp = Iext - Kext * Cext;
    Pext = temp * Pext * temp.transpose() + Kext * R * Kext.transpose();

    // Apply quaternion correction from xext(0..2) exactly as original:
    applyQuaternionCorrectionFromErrorState();

    // Clear attitude error entries (first 3) as original did:
    for (int i = 0; i < 3; ++i) xext(i) = T(0);
  }
}

// Overloads calling the above
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(T const acc[3], T const mag[3]) {
  measurement_update(Map<Matrix<T,3,1>>(acc), Map<Matrix<T,3,1>>(mag));
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
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(T const acc[3]) {
  measurement_update_acc_only(Map<Matrix<T,3,1>>(acc));
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(Vector3 const& mag) {
  Vector3 const v2hat = magnetometer_measurement_func();
  measurement_update_partial(mag, v2hat, Rmag);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(T const mag[3]) {
  measurement_update_mag_only(Map<Matrix<T,3,1>>(mag));
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::accelerometer_measurement_func() const {
  return qref.inverse() * v1ref;
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::magnetometer_measurement_func() const {
  return qref.inverse() * v2ref;
}

// ---------------- utility functions ----------------
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

// ---------------- Extended pseudo-measurement: zero S ----------------
template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::applyIntegralZeroPseudoMeas() {
  // H picks S block (3 rows, NX columns)
  Matrix<T, 3, NX> H;
  H.setZero();
  H.template block<3,3>(0, BASE_N + 6) = Matrix3::Identity();

  Vector3 z = Vector3::Zero();
  Matrix3 S = H * Pext * H.transpose() + Racc; // reuse Racc as pseudo cov or user can provide R_S separately
  Eigen::FullPivLU<Matrix3> lu(S);
  if (!lu.isInvertible()) return;
  Matrix<T,NX,3> K = Pext * H.transpose() * lu.inverse();
  xext = xext + K * (z - H * xext);

  // Covariance Joseph form
  Matrix<T,NX,NX> Iext = Matrix<T,NX,NX>::Identity();
  Matrix<T,NX,NX> temp = Iext - K * H;
  Pext = temp * Pext * temp.transpose() + K * Racc * K.transpose();

  // Apply quaternion correction if attitude error component was modified
  applyQuaternionCorrectionFromErrorState();
  // Clear small-angle entries as done elsewhere
  for (int i=0;i<3;++i) xext(i) = T(0);

  // update Pbase with top-left block
  for (int i=0;i<BASE_N;++i) for (int j=0;j<BASE_N;++j) Pbase(i,j) = Pext(i,j);
}


