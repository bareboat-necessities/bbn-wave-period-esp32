#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  Enhancements:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file merges your original Kalman3D_Wave<T,with_gyro_bias> with an extended
  full-matrix Kalman that adds linear navigation states:
     v (3)   : velocity in world frame
     p (3)   : displacement/position in world frame
     S (3)   : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction
     a_w (3) : estimate of acceleration (specific force)

  - The quaternion MEKF logic (time_update, measurement_update, partial updates, quaternion correction)
    is preserved where possible.
  - The extended linear states are driven by a latent OU world-acceleration a_w
    (accelerometer input is used only in the measurement update).
  - A full extended covariance (Pext) and transition Jacobian Fext are constructed; the top-left corner
    contains the original MEKF's P/Q blocks (attitude error + optional gyro bias).
  - Accelerometer and magnetometer inputs must be given in aerospace/NED (x north, y east, z down)
*/

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif

#include <limits>

using Eigen::Matrix;
using Eigen::Map;

#ifdef EIGEN_NON_ARDUINO
  #include <unsupported/Eigen/MatrixFunctions>  // enables .exp() on matrices
#endif

template <typename T = float, bool with_gyro_bias = true>
class EIGEN_ALIGN_MAX Kalman3D_Wave {

    // Original base state dimension (attitude-error (3) [+ gyro-bias (3) if with_gyro_bias])
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;
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
    static constexpr T STD_GRAVITY = T(9.80665);  // standard gravity acceleration m/s²

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor signatures preserved, additional defaults for linear process noise
    Kalman3D_Wave(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
                  T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(1e+5),
                  T gravity_magnitude = T(STD_GRAVITY));

    // Initialization / measurement API 
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Set the world-frame magnetic reference vector used by the mag measurement model.
    // Pass NED units. If you want yaw-only, pass the horizontal field (z = 0).
    void set_mag_world_ref(const Vector3& B_world) {
        v2ref = B_world;
    }

    void time_update(Vector3 const& gyr, T Ts); 

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag);
    void measurement_update_acc_only(Vector3 const& acc);
    void measurement_update_mag_only(Vector3 const& mag);

    // Extended-only API:
    // Apply zero pseudo-measurement on S (integral drift correction)
    void applyIntegralZeroPseudoMeas();

    // Accessors
    Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    MatrixBaseN const& covariance_base() const { return Pbase; } // top-left original block
    MatrixNX const& covariance_full() const { return Pext; }     // full extended covariance

    Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) {
            return xext.template segment<3>(3);
        } else {
            return Vector3::Zero();
        }
    }

    // Velocity in world (NED)
    Vector3 get_velocity() const {
        // velocity state at offset BASE_N
        return xext.template segment<3>(BASE_N);
    }

    // Position in world (NED)
    Vector3 get_position() const {
        // position state at offset BASE_N+3
        return xext.template segment<3>(BASE_N + 3);
    }

    // Integral displacement in world (NED)
    Vector3 get_integral_displacement() const {
       // integral of displacement state at offset BASE_N+6
       return xext.template segment<3>(BASE_N + 6);
    }

    // Latent OU world-acceleration a_w (world, NED)
    Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }

    // Tuning setters
    void set_aw_time_constant(T tau_seconds) { tau_aw = std::max(T(1e-3), tau_seconds); }
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat = std_aw.array().square().matrix().asDiagonal();
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0 * sigma_v0);   // v (3)
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0 * sigma_p0);   // p (3)
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0 * sigma_S0);   // S (3)
    }
  
    static Eigen::Matrix<T,3,1> ned_field_from_decl_incl(T D_rad, T I_rad, T B = T(1)) {
        const T cI = std::cos(I_rad), sI = std::sin(I_rad);
        const T cD = std::cos(D_rad), sD = std::sin(D_rad);
        return (Eigen::Matrix<T,3,1>() <<
            B * cI * cD,   // N
            B * cI * sD,   // E
            B * sI         // D (down positive)
        ).finished();
    }

  private:
    const T gravity_magnitude_ = T(STD_GRAVITY);
  
    // Original MEKF internals (kept nomenclature)
    Eigen::Quaternion<T> qref;
    Vector3 v2ref = Vector3::UnitX();

    // Original base error-state (first BASE_N elements) — now stored inside xext (top portion)
    // But we keep a mirror of original P for compatibility
    MatrixBaseN Pbase;

    // Extended full state xext and Pext (NX x NX)
    Matrix<T, NX, 1> xext; // [ att_err(3), (gyro bias 3 optional), v(3), p(3), S(3) ]
    MatrixNX Pext;

    // Last gyro 
    Vector3 last_gyr_bias_corrected{};

    // Original constant matrices (kept)
    const Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase; // original Q for attitude & bias

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 R_S;  // Triple integration measurement noise

    // World-acceleration OU process a_w dynamics parameters
    T tau_aw = T(0.5);            // correlation time [s], tune 1–5 s for sea states
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(0.5*0.5); // stationary variance diag [ (m/s^2)^2 ]

    // convenience getters
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }               // world→body
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }   // body→world
  
    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func() const;
    Vector3 magnetometer_measurement_func() const;

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Extended helpers
    void assembleExtendedFandQ(const Vector3& acc_body, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);

    // Quaternion & small-angle helpers (kept)
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

template <typename T, bool with_gyro_bias>
Kalman3D_Wave<T, with_gyro_bias>::Kalman3D_Wave(
    Vector3 const& sigma_a,
    Vector3 const& sigma_g,
    Vector3 const& sigma_m,
    T Pq0, T Pb0, T b0, T R_S_noise, T gravity_magnitude)
  : Qbase(initialize_Q(sigma_g, b0)),
    gravity_magnitude_(gravity_magnitude),
    Racc(sigma_a.array().square().matrix().asDiagonal()),
    Rmag(sigma_m.array().square().matrix().asDiagonal())
{
  // quaternion init
  qref.setIdentity();

  R_S = Matrix3::Identity() * R_S_noise;

  // initialize base / extended states
  Pbase.setZero();
  Pbase.setIdentity(); // default small initial cov unless user overwrites
  
  // initialize base covariance
  Pbase.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;   // attitude error covariance
  if constexpr (with_gyro_bias) {
      Pbase.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;    // bias covariance
  }

  // Extended state
  xext.setZero();
  Pext.setZero();
  
  // Place original base P into top-left of Pext
  Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

  // Seed covariance for a_w (world acceleration)
  Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;

  const T sigma_v0 = T(0.5);   // m/s
  const T sigma_p0 = T(2.0);   // m
  const T sigma_S0 = T(8.0);   // m·s
  set_initial_linear_uncertainty(sigma_v0, sigma_p0, sigma_S0);

  R.setZero();
  R.template topLeftCorner<3,3>()  = Racc;     // accelerometer measurement noise
  R.template bottomRightCorner<3,3>() = Rmag;  // magnetometer measurement noise
}

template<typename T, bool with_gyro_bias>
typename Kalman3D_Wave<T, with_gyro_bias>::MatrixBaseN
Kalman3D_Wave<T, with_gyro_bias>::initialize_Q(typename Kalman3D_Wave<T, with_gyro_bias>::Vector3 sigma_g, T b0) {
  MatrixBaseN Q; Q.setZero();
  if constexpr (with_gyro_bias) {
    Q.template topLeftCorner<3,3>() = sigma_g.array().square().matrix().asDiagonal(); // gyro RW
    Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * b0;                   // bias RW
  } else {
    Q = sigma_g.array().square().matrix().asDiagonal();
  }
  return Q;
}

// initialization helpers
    
// Initialization from accelerometer + magnetometer
// Inputs:
//   acc_body  — accelerometer specific force in body frame (NED)
//   mag_body  — magnetometer measurement in body frame (NED)
template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::initialize_from_acc_mag(
    Vector3 const& acc_body,
    Vector3 const& mag_body)
{
    // Normalize accelerometer
    T anorm = acc_body.norm();
    if (anorm < T(1e-8)) {
        throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
    }
    Vector3 acc_n = acc_body / anorm;

    // Build WORLD axes expressed in BODY coords
    Vector3 z_world = -acc_n;                         // world Z (down) in body coord
    Vector3 mag_h   = mag_body - (mag_body.dot(z_world)) * z_world;
    if (mag_h.norm() < 1e-8) {
        throw std::runtime_error("Magnetometer vector parallel to gravity — cannot initialize yaw");
    }
    mag_h.normalize();
    Vector3 x_world = mag_h;                          // world X (north) in body coords
    Vector3 y_world = z_world.cross(x_world).normalized();

    // R_wb: world→body rotation (columns = world axes in body coords)
    Matrix3 R_wb;
    R_wb.col(0) = x_world;
    R_wb.col(1) = y_world;
    R_wb.col(2) = z_world;

    // Store quaternion as world→body
    qref = Eigen::Quaternion<T>(R_wb);
    qref.normalize();

    // Store reference magnetic vector in world frame
    v2ref = R_bw() * mag_body.normalized();  // body to world
}

template<typename T, bool with_gyro_bias>
Eigen::Quaternion<T>
Kalman3D_Wave<T, with_gyro_bias>::quaternion_from_acc(Vector3 const& acc)
{
    // Raw accelerometer is specific force; at rest in NED: acc ≈ (0,0,-g) in body
    // We want body +Z (down) to align with world +Z (down), i.e. align zb to -acc.
    Vector3 an = acc.normalized();
    Vector3 zb = Vector3::UnitZ();

    // Rotate zb to -an
    Vector3 target = -an;
    T cos_theta = zb.dot(target);
    Vector3 axis = zb.cross(target);
    T norm_axis = axis.norm();

    if (norm_axis < T(1e-8)) {
        // Almost parallel or anti-parallel
        if (cos_theta > 0) {
            return Eigen::Quaternion<T>::Identity();        // zb ≈ target
        } else {
            return Eigen::Quaternion<T>(0, 1, 0, 0);        // 180° about X
        }
    }

    axis /= norm_axis;
    // clamp to avoid NaNs
    cos_theta = std::max(T(-1), std::min(T(1), cos_theta));
    T angle = std::acos(cos_theta);

    Eigen::AngleAxis<T> aa(angle, axis);
    Eigen::Quaternion<T> q(aa);
    q.normalize();
    return q;
}

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::initialize_from_acc(Vector3 const& acc)
{
    T anorm = acc.norm();
    if (anorm < T(1e-8)) {
       throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
    }
    Vector3 acc_n = acc / anorm;

    // Use accelerometer to align z axis, yaw remains arbitrary
    qref = quaternion_from_acc(acc_n);
    qref.normalize();
}

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::time_update(Vector3 const& gyr, T Ts)
{
    // Attitude mean propagation
    Vector3 gyro_bias;
    if constexpr (with_gyro_bias) {
        gyro_bias = xext.template segment<3>(3);
    } else {
        gyro_bias = Vector3::Zero();
    }
  
    last_gyr_bias_corrected = gyr - gyro_bias;

    // Build delta quaternion from gyro increment
    T ang = last_gyr_bias_corrected.norm() * Ts;
    Eigen::Quaternion<T> dq;
    if (ang > T(1e-9)) {
        Vector3 axis = last_gyr_bias_corrected.normalized();
        dq = Eigen::AngleAxis<T>(ang, axis);     // +ang
    } else {
        dq.setIdentity();
    }

    // Propagate: right-multiply (matches correction side and F/Jacobians signs )
    qref = qref * dq;
    qref.normalize();
  
    // Build exact discrete transition & process Q
    MatrixNX F_a_ext; MatrixNX Q_a_ext;
    assembleExtendedFandQ(Vector3::Zero(), Ts, F_a_ext, Q_a_ext);

    // Mean propagation for linear subsystem [v,p,S,a_w]
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

    // Covariance propagation
    Pext = F_a_ext * Pext * F_a_ext.transpose() + Q_a_ext;
    Pext = T(0.5) * (Pext + Pext.transpose()); // enforce symmetry

    // Mirror base covariance
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);

    // Optional drift correction on S
    applyIntegralZeroPseudoMeas();
}

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::measurement_update(Vector3 const& acc, Vector3 const& mag)
{
    // Predicted measurements
    Vector3 v1hat = accelerometer_measurement_func(); // depends on a_w 
    Vector3 v2hat = magnetometer_measurement_func();

    Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
    // accel rows 0..2
    Cext.template block<3,3>(0,0)        = -skew_symmetric_matrix(v1hat); // d f_b / d attitude
    Cext.template block<3,3>(0,OFF_AW)   = R_wb();               // d f_b / d a_w
    // mag rows 3..5 (unchanged)
    Cext.template block<3,3>(3,0)        = -skew_symmetric_matrix(v2hat);

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

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::measurement_update_partial(
    const Eigen::Ref<const Vector3>& meas,
    const Eigen::Ref<const Vector3>& vhat,
    const Eigen::Ref<const Matrix3>& Rm)
{
    // Cext: (3 x NX)
    Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
    Cext.template block<3,3>(0,0) = -skew_symmetric_matrix(vhat);
    
    // Innovation
    Vector3 inno = meas - vhat;
    
    // S = C P C^T + Rm  (3x3)
    Matrix3 S_mat = Cext * Pext * Cext.transpose() + Rm;
    
    // PCt = P C^T  (NX x 3)
    Matrix<T, NX, 3> PCt = Pext * Cext.transpose();
    
    // Factor S (SPD) and solve
    Eigen::LDLT<Matrix3> ldlt(S_mat);
    if (ldlt.info() != Eigen::Success) {
        S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Rm.norm());
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

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::measurement_update_acc_only(Vector3 const& acc_meas) {
    const Vector3 v1hat = accelerometer_measurement_func();

    // Cext: (3 x NX)
    Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
    // d f_b / d (attitude error)
    Cext.template block<3,3>(0,0) = -skew_symmetric_matrix(v1hat);
    // d f_b / d a_w
    Cext.template block<3,3>(0, OFF_AW) = R_wb();

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

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::measurement_update_mag_only(Vector3 const& mag) {
    Vector3 const v2hat = magnetometer_measurement_func();
    measurement_update_partial(mag, v2hat, Rmag);
}

// specific force prediction: f_b = R_wb (a_w - g)
template<typename T, bool with_gyro_bias>
Matrix<T,3,1> Kalman3D_Wave<T, with_gyro_bias>::accelerometer_measurement_func() const {
    const Vector3 g_world(0, 0, +gravity_magnitude_);  // NED g_world
    const Vector3 aw = xext.template segment<3>(OFF_AW);
    return R_wb() * (aw - g_world);
}

template<typename T, bool with_gyro_bias>
Matrix<T, 3, 1> Kalman3D_Wave<T, with_gyro_bias>::magnetometer_measurement_func() const {
    return R_wb() * v2ref;
}

// utility functions
template<typename T, bool with_gyro_bias>
Matrix<T, 3, 3> Kalman3D_Wave<T, with_gyro_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
  Matrix3 M;
  M << 0, -vec(2), vec(1),
       vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return M;
}

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::applyQuaternionCorrectionFromErrorState() {
  // xext(0..2) contains the small-angle error — same as original code; create corr quaternion and apply
  Eigen::Quaternion<T> corr(T(1), half * xext(0), half * xext(1), half * xext(2));
  corr.normalize();
  qref = qref * corr;
  qref.normalize();
}

// normalize quaternion
template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::normalizeQuat() {
  qref.normalize();
}

//  Extended pseudo-measurement: zero S 
template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::applyIntegralZeroPseudoMeas() {
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

// Exact Van-Loan using Eigen's MatrixFunctions (desktop)
#ifdef EIGEN_NON_ARDUINO
template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::vanLoanDiscretization_12x3(
    const Eigen::Matrix<T,12,12>& A,
    const Eigen::Matrix<T,12,3>&  G,
    const Eigen::Matrix<T,3,3>&   Sigma_c,
    T Ts,
    Eigen::Matrix<T,12,12>& Phi,
    Eigen::Matrix<T,12,12>& Qd) const
{
    Eigen::Matrix<T,24,24> M; M.setZero();
    M.block(0,0,12,12)   = -A * Ts;
    M.block(0,12,12,12)  =  G * Sigma_c * G.transpose() * Ts;
    M.block(12,12,12,12) =  A.transpose() * Ts;

    // Requires <unsupported/Eigen/MatrixFunctions>
    Eigen::Matrix<T,24,24> expM = M.exp();

    // Van-Loan unpack
    const auto PhiT = expM.block(12,12,12,12);
    const auto Qblk = expM.block(0,12,12,12);

    Phi = PhiT.transpose();       // exp(A Ts)
    Qd  = Phi * Qblk;             // exact discrete process noise
}
#else
// Padé(6) matrix exponential with scaling & squaring
// Works well for small fixed-size matrices (e.g. 8x8)
template<typename Mat>
Mat expm_pade6(const Mat& A) {
    using T = typename Mat::Scalar;
    const int n = A.rows();
    assert(n == A.cols());

    const T theta = 3.0;
    const int max_squarings = 8;

    const T c0 = 1.0;
    const T c2 = 1.0/2.0;
    const T c4 = 1.0/24.0;
    const T c6 = 1.0/720.0;

    // 1-norm estimate
    T normA = A.cwiseAbs().colwise().sum().maxCoeff();
    int s = 0;
    if (normA > theta) {
        s = std::min(max_squarings, static_cast<int>(std::ceil(std::log2(normA/theta))));
    }
    Mat As = A / T(1<<s);

    Mat A2 = As * As;
    Mat A4 = A2 * A2;
    Mat A6 = A4 * A2;

    Mat U = As * (c2*Mat::Identity(n,n) + c6*A2);
    Mat V = c0*Mat::Identity(n,n) + c4*A2 + (1.0/5040.0)*A6;

    Mat P = V + U;
    Mat Q = V - U;

    Mat R = Q.lu().solve(P);

    for (int i=0;i<s;i++) {
        R = R * R;
    }
    return R;
}
template<typename T>
static void vanLoanAxis4x1(
    T tau, T sigma2, T Ts,
    Eigen::Matrix<T,4,4>& Phi, Eigen::Matrix<T,4,4>& Qd)
{
    using Mat4 = Eigen::Matrix<T,4,4>;
    using Mat8 = Eigen::Matrix<T,8,8>;

    Mat4 A; A.setZero();
    A(0,3) = 1;          // v̇ = a
    A(1,0) = 1;          // ṗ = v
    A(2,1) = 1;          // Ṡ = p
    A(3,3) = -1/tau;     // ȧ = -a/τ

    Eigen::Matrix<T,4,1> G; G.setZero();
    G(3,0) = 1;

    const T Sigma_c = (2/tau) * sigma2;

    Mat8 M; M.setZero();
    M.block(0,0,4,4)   = -A * Ts;
    M.block(0,4,4,4)   = G * (Sigma_c * G.transpose()) * Ts;
    M.block(4,4,4,4)   = A.transpose() * Ts;

    Mat8 expM = expm_pade6(M);

    Mat4 PhiT = expM.block(4,4,4,4);
    Mat4 Qblk = expM.block(0,4,4,4);

    Phi = PhiT.transpose();
    Qd  = Phi * Qblk;
}
template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::vanLoanDiscretization_12x3(
    const Eigen::Matrix<T,12,12>& /*A*/,
    const Eigen::Matrix<T,12,3>&  /*G*/,
    const Eigen::Matrix<T,3,3>&   /*Sigma_c_unused*/,
    T Ts,
    Eigen::Matrix<T,12,12>& Phi,
    Eigen::Matrix<T,12,12>& Qd) const
{
    using Mat12 = Eigen::Matrix<T,12,12>;

    Mat12 Phi_lin; Phi_lin.setZero();
    Mat12 Qd_lin;  Qd_lin.setZero();

    for (int axis=0; axis<3; ++axis) {
        T tau    = std::max(T(1e-6), tau_aw);
        T sigma2 = Sigma_aw_stat(axis,axis);

        Eigen::Matrix<T,4,4> Phi_axis, Qd_axis;
        vanLoanAxis4x1(tau, sigma2, Ts, Phi_axis, Qd_axis);

        int idx[4] = {0,3,6,9}; // v,p,S,a offsets
        for (int i=0;i<4;i++)
            for (int j=0;j<4;j++) {
                Phi_lin(idx[i]+axis, idx[j]+axis) = Phi_axis(i,j);
                Qd_lin (idx[i]+axis, idx[j]+axis) = Qd_axis (i,j);
            }
    }

    Phi = Phi_lin;
    Qd  = Qd_lin;
}
#endif

template<typename T, bool with_gyro_bias>
void Kalman3D_Wave<T, with_gyro_bias>::assembleExtendedFandQ(
    const Vector3& /*acc_body_unused*/, T Ts,
    Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext)
{
    F_a_ext.setIdentity();
    Q_a_ext.setZero();

    // Attitude error (+ optional bias) discrete transition
    Matrix3 I = Matrix3::Identity();
    Vector3 w = last_gyr_bias_corrected;   // bias-corrected gyro [rad/s]
    T omega = w.norm();
    T theta = omega * Ts;

    if (theta < T(1e-6)) {
        Matrix3 Wx = skew_symmetric_matrix(w);
        F_a_ext.template block<3,3>(0,0) = I - Wx*Ts + (Wx*Wx)*(Ts*Ts/2);
    } else {
        Matrix3 W = skew_symmetric_matrix(w / omega);
        T s = std::sin(theta);
        T c = std::cos(theta);
        F_a_ext.template block<3,3>(0,0) = I - s * W + (1 - c) * (W * W);
    }

    if constexpr (with_gyro_bias) {
        F_a_ext.template block<3,3>(0,3) = -Matrix3::Identity() * Ts;
    }

    // Process noise for attitude/bias
    Q_a_ext.topLeftCorner(BASE_N, BASE_N) = Qbase * Ts;
  
    // Linear subsystem [v, p, S, a_w]
    using Mat12   = Eigen::Matrix<T,12,12>;
    using Mat12x3 = Eigen::Matrix<T,12,3>;

    Mat12 A; A.setZero();
    A.template block<3,3>(0,9) = Matrix3::Identity();                       // v̇ = a_w
    A.template block<3,3>(3,0) = Matrix3::Identity();                       // ṗ = v
    A.template block<3,3>(6,3) = Matrix3::Identity();                       // Ṡ = p
    A.template block<3,3>(9,9) = -(T(1)/std::max(T(1e-6), tau_aw)) * Matrix3::Identity(); // ȧ_w

    Mat12x3 G; G.setZero();
    G.template block<3,3>(9,0) = Matrix3::Identity(); // noise drives a_w

    Mat12 Phi_lin, Qd_lin;

    // Desktop and embedded handle this differently
#ifdef EIGEN_NON_ARDUINO
    const Matrix3 Sigma_c = (T(2)/std::max(T(1e-6), tau_aw)) * Sigma_aw_stat;
    vanLoanDiscretization_12x3(A, G, Sigma_c, Ts, Phi_lin, Qd_lin);
#else
    vanLoanDiscretization_12x3(A, G, Matrix3::Zero(), Ts, Phi_lin, Qd_lin);
#endif

    F_a_ext.template block<12,12>(OFF_V, OFF_V) = Phi_lin;
    Q_a_ext.template block<12,12>(OFF_V, OFF_V) = Qd_lin;
}
