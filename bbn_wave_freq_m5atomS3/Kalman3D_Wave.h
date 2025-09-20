#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  Enhancements:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file merges your original Kalman3D_Wave<T,with_gyro_bias, with_accel_bias> with an extended
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
#include <stdexcept>

using Eigen::Matrix;
using Eigen::Map;

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman3D_Wave {

    // Original base (att_err + optional gyro bias)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended added states: v(3), p(3), S(3), a_w(3) [+ optional b_acc(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 15 : 12;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets (always defined)
    static constexpr int OFF_V   = BASE_N + 0;
    static constexpr int OFF_P   = BASE_N + 3;
    static constexpr int OFF_S   = BASE_N + 6;
    static constexpr int OFF_AW  = BASE_N + 9;
    static constexpr int OFF_BA  = with_accel_bias ? (BASE_N + 12) : -1; // -1 = not present

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
    static constexpr T tempC_ref = T(35.0); // Reference temperature for temperature related accel bias drift °C

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor signatures preserved, additional defaults for linear process noise
    Kalman3D_Wave(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
                  T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(3e-2),
                  T gravity_magnitude = T(STD_GRAVITY));

    // Initialization / measurement API
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Set the world-frame magnetic reference vector used by the mag measurement model.
    // Pass NED units. If you want yaw-only, pass the horizontal field (z = 0).
    void set_mag_world_ref(const Vector3& B_world) {
        v2ref = B_world.normalized();
    }

    void time_update(Vector3 const& gyr, T Ts);

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update(Vector3 const& acc, Vector3 const& mag, T tempC = tempC_ref);
    void measurement_update_acc_only(Vector3 const& acc, T tempC = tempC_ref);
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

    Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) {
            return xext.template segment<3>(OFF_BA);
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

    // OU stationary std [m/s²] for a_w (per axis)
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat = std_aw.array().square().matrix().asDiagonal();
    }

    // Covariances for ∫p dt pseudo-measurement
    void set_RS_noise(const Vector3& sigma_S) {
        R_S = sigma_S.array().square().matrix().asDiagonal();
    }

    // Accelerometer measurement noise (std in m/s² per axis)
    void set_Racc(const Vector3& sigma_acc) {
        Racc = sigma_acc.array().square().matrix().asDiagonal();
        R.template topLeftCorner<3,3>() = Racc;
    }

    // Magnetometer measurement noise (std per axis, μT or unitless)
    void set_Rmag(const Vector3& sigma_mag) {
        Matrix3 Rmag_new = sigma_mag.array().square().matrix().asDiagonal();
        const_cast<Matrix3&>(Rmag) = Rmag_new;
        R.template bottomRightCorner<3,3>() = Rmag_new;
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0 * sigma_v0);   // v (3)
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0 * sigma_p0);   // p (3)
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0 * sigma_S0);   // S (3)
    }

    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) sigma_bacc0_ = std::max(T(0), s);
    }

    void set_Q_bacc_rw(const Vector3& rw_std_per_sqrt_s) {
        if constexpr (with_accel_bias)
            Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
    }

    void set_initial_acc_bias(const Vector3& b0) {
        if constexpr (with_accel_bias)
            xext.template segment<3>(OFF_BA) = b0;
    }

    // Set accelerometer bias temperature coefficient k_a  [m/s^2 per °C] per axis.
    // Model: b_a(tempC) = b_a0 + k_a * (tempC - 30)
    void set_accel_bias_temp_coeff(const Vector3& ka_per_degC) { k_a_ = ka_per_degC; }

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

    T sigma_bacc0_ = T(0.1); // initial accel bias std
    Matrix3 Q_bacc_ = Matrix3::Identity() * T(1e-8);

    // Accelerometer bias temperature coefficient (per-axis), units: m/s^2 per °C.
    // Default here reflects BMI270 typical accel drift (~0.003 m/s^2/°C).
    Vector3 k_a_ = Vector3::Constant(T(0.003));

    // Original constant matrices (kept)
    const Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase; // original Q for attitude & bias

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 R_S;  // Triple integration measurement noise

    // World-acceleration OU process a_w dynamics parameters
    T tau_aw = T(1.2);            // correlation time [s], tune 1–5 s for sea states
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(0.2*0.2); // stationary variance diag [ (m/s^2)^2 ]

    // convenience getters
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }               // world→body
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }   // body→world

    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func(T tempC) const;
    Vector3 magnetometer_measurement_func() const;

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Extended helpers
    void assembleExtendedFandQ(const Vector3& acc_body, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);

    // Quaternion & small-angle helpers (kept)
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)
    void normalizeQuat();

    static void PhiAxis4x1_analytic(T tau, T h, Eigen::Matrix<T,4,4>& Phi_axis);
    static void QdAxis4x1_analytic(T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis);
};

// Implementation

template <typename T, bool with_gyro_bias, bool with_accel_bias>
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::Kalman3D_Wave(
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

  if constexpr (with_accel_bias) {
      Pext.template block<3,3>(OFF_BA, OFF_BA) = Matrix3::Identity() * sigma_bacc0_ * sigma_bacc0_;
  }

  const T sigma_v0 = T(1.0);    // m/s
  const T sigma_p0 = T(20.0);   // m
  const T sigma_S0 = T(50.0);   // m·s
  set_initial_linear_uncertainty(sigma_v0, sigma_p0, sigma_S0);

  R.setZero();
  R.template topLeftCorner<3,3>()  = Racc;     // accelerometer measurement noise
  R.template bottomRightCorner<3,3>() = Rmag;  // magnetometer measurement noise
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::MatrixBaseN
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::initialize_Q(typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::Vector3 sigma_g, T b0) {
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
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::initialize_from_acc_mag(
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
Eigen::Quaternion<T>
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::quaternion_from_acc(Vector3 const& acc)
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::initialize_from_acc(Vector3 const& acc)
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::time_update(Vector3 const& gyr, T Ts)
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update(Vector3 const& acc, Vector3 const& mag, T tempC)
{
    // Predicted measurements
    Vector3 v1hat = accelerometer_measurement_func(tempC); // depends on a_w
    Vector3 v2hat = magnetometer_measurement_func();

    Matrix<T, M, NX> Cext = Matrix<T, M, NX>::Zero();
    // accel rows 0..2
    Cext.template block<3,3>(0,0)        = -skew_symmetric_matrix(v1hat); // d f_b / d attitude
    Cext.template block<3,3>(0,OFF_AW)   = R_wb(); // d f_b / d a_w
    if constexpr (with_accel_bias) {
        Cext.template block<3,3>(0,OFF_BA) = Matrix3::Identity(); // d f_b / d b_acc
    }
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_partial(
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_acc_only(Vector3 const& acc_meas, T tempC) {
    const Vector3 v1hat = accelerometer_measurement_func(tempC);

    // Cext: (3 x NX)
    Matrix<T, 3, NX> Cext = Matrix<T, 3, NX>::Zero();
    // d f_b / d (attitude error)
    Cext.template block<3,3>(0,0) = -skew_symmetric_matrix(v1hat);
    // d f_b / d a_w
    Cext.template block<3,3>(0, OFF_AW) = R_wb();
    if constexpr (with_accel_bias) {
        Cext.template block<3,3>(0,OFF_BA) = Matrix3::Identity(); // d f_b / d b_acc
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
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * Racc * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());

    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_mag_only(Vector3 const& mag) {
    Vector3 const v2hat = magnetometer_measurement_func();
    measurement_update_partial(mag, v2hat, Rmag);
}

// specific force prediction: f_b = R_wb (a_w - g) + b_a(temp)
// with temp correction: b_a(temp) = b_a0 + k_a * (tempC - tempC_ref) 
template<typename T, bool with_gyro_bias, bool with_accel_bias>
Matrix<T,3,1>
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::accelerometer_measurement_func(T tempC) const {
    const Vector3 g_world(0,0,+gravity_magnitude_);
    const Vector3 aw = xext.template segment<3>(OFF_AW);

    Vector3 fb = R_wb() * (aw - g_world);

    if constexpr (with_accel_bias) {
        Vector3 ba0 = xext.template segment<3>(OFF_BA);
        Vector3 ba  = ba0 + k_a_ * (tempC - tempC_ref); // temperature related drift
        fb += ba;
    }
    return fb;
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
Matrix<T, 3, 1> Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::magnetometer_measurement_func() const {
    return R_wb() * v2ref;
}

// utility functions
template<typename T, bool with_gyro_bias, bool with_accel_bias>
Matrix<T, 3, 3> Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
  Matrix3 M;
  M << 0, -vec(2), vec(1),
       vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return M;
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::applyQuaternionCorrectionFromErrorState() {
  // xext(0..2) contains the small-angle error — same as original code; create corr quaternion and apply
  Eigen::Quaternion<T> corr(T(1), half * xext(0), half * xext(1), half * xext(2));
  corr.normalize();
  qref = qref * corr;
  qref.normalize();
}

// normalize quaternion
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::normalizeQuat() {
  qref.normalize();
}

//  Extended pseudo-measurement: zero S
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::applyIntegralZeroPseudoMeas() {
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::assembleExtendedFandQ(
    const Vector3& /*acc_body_unused*/, T Ts,
    Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext)
{
    F_a_ext.setIdentity();
    Q_a_ext.setZero();

    // Attitude error (+ optional bias)
    Matrix3 I = Matrix3::Identity();
    Vector3 w = last_gyr_bias_corrected;
    T omega = w.norm();
    T theta = omega * Ts;

    if (theta < T(1e-8)) {
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

    // Linear subsystem [v,p,S,a_w]
    using Mat12 = Eigen::Matrix<T,12,12>;
    Mat12 Phi_lin; Phi_lin.setZero();
    Mat12 Qd_lin;  Qd_lin.setZero();

        for (int axis = 0; axis < 3; ++axis) {
            T tau    = std::max(T(1e-6), tau_aw);
            T sigma2 = Sigma_aw_stat(axis,axis);

            Eigen::Matrix<T,4,4> Phi_axis, Qd_axis;
            PhiAxis4x1_analytic(tau, Ts, Phi_axis);
            QdAxis4x1_analytic(tau, Ts, sigma2, Qd_axis);

            static const int idx[4] = {0,3,6,9}; // v,p,S,a offsets
            for (int i=0;i<4;i++)
                for (int j=0;j<4;j++) {
                    Phi_lin(idx[i]+axis, idx[j]+axis) = Phi_axis(i,j);
                    Qd_lin (idx[i]+axis, idx[j]+axis) = Qd_axis(i,j);
                }
        }

    F_a_ext.template block<12,12>(OFF_V, OFF_V) = Phi_lin;
    Q_a_ext.template block<12,12>(OFF_V, OFF_V) = Qd_lin;

    if constexpr (with_accel_bias) {
        Q_a_ext.template block<3,3>(OFF_BA, OFF_BA) = Q_bacc_ * Ts;
    }
}

// Discrete-time state transition Φ for the [v, p, S, a] axis subsystem.
//
// Definitions (integrals of exp(-ξ/τ)):
//   α    = e^{-h/τ}
//
//   φ_va = ∫₀ʰ e^{-ξ/τ} dξ
//        = τ (1 - e^{-h/τ})
//
//   φ_pa = ∫₀ʰ ξ e^{-ξ/τ} dξ
//        = τh - τ² (1 - e^{-h/τ})
//
//   φ_Sa = ∫₀ʰ ξ² e^{-ξ/τ} dξ / 2
//        = 0.5 τ h² - τ² h + τ³ (1 - e^{-h/τ})
//
// Series expansions (for h/τ ≪ 1):
//   φ_va ≈ h - h²/(2τ) + h³/(6τ²) - h⁴/(24τ³)
//   φ_pa ≈ 0.5 h² - (1/6)(h³/τ) + (1/24)(h⁴/τ²)
//   φ_Sa ≈ (1/6) h³ - (1/24)(h⁴/τ) + (1/120)(h⁵/τ²)
//
// State recursion (per axis):
//   v_{k+1} = v_k + φ_va a_k
//   p_{k+1} = p_k + h v_k + φ_pa a_k
//   S_{k+1} = S_k + h p_k + 0.5 h² v_k + φ_Sa a_k
//   a_{k+1} = α a_k
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::PhiAxis4x1_analytic(
    T tau, T h, Eigen::Matrix<T,4,4>& Phi_axis)
{
    using std::exp;
    const T x = h / std::max(T(1e-12), tau);

    T alpha, phi_va, phi_pa, phi_Sa;

    if (x < T(1e-3)) {
        // --- Series expansions around x = 0 (safe for float) ---
        // α = e^{-x} ≈ 1 - x + x^2/2 - x^3/6 + x^4/24
        alpha = T(1) - x + x*x/T(2) - x*x*x/T(6) + x*x*x*x/T(24);

        // φ_va = ∫₀ʰ e^{-ξ/τ} dξ = τ(1 - α)
        //      ≈ h - h²/(2τ) + h³/(6τ²) - h⁴/(24τ³)
        phi_va = h - (h*h)/(T(2)*tau) + (h*h*h)/(T(6)*tau*tau) - (h*h*h*h)/(T(24)*tau*tau*tau);

        // φ_pa = ∫₀ʰ ξ e^{-ξ/τ} dξ = τh - τ²(1 - α)
        //      ≈ 0.5 h² - (1/6)(h³/τ) + (1/24)(h⁴/τ²)
        phi_pa = (h*h)/T(2) - (h*h*h)/(T(6)*tau) + (h*h*h*h)/(T(24)*tau*tau);

        // φ_Sa = ∫₀ʰ ξ² e^{-ξ/τ} dξ / 2 = 0.5 τ h² - τ² h + τ³(1 - α)
        //      ≈ (1/6) h³ - (1/24)(h⁴/τ) + (1/120)(h⁵/τ²)
        phi_Sa = (h*h*h)/T(6) - (h*h*h*h)/(T(24)*tau) + (h*h*h*h*h)/(T(120)*tau*tau);
    } else {
        // --- Exact closed forms ---
        alpha = exp(-x);

        // φ_va = τ(1 - e^{-h/τ})
        phi_va = tau * (T(1) - alpha);

        // φ_pa = τh - τ²(1 - e^{-h/τ})
        phi_pa = tau*h - tau*tau*(T(1) - alpha);

        // φ_Sa = 0.5 τ h² - τ² h + τ³(1 - e^{-h/τ})
        phi_Sa = (T(0.5)*tau*h*h) - (tau*tau*h) + (tau*tau*tau)*(T(1) - alpha);
    }

    // Assemble discrete-time state transition matrix for [v, p, S, a]
    Phi_axis.setZero();

    // v_{k+1} = v_k + φ_va a_k
    Phi_axis(0,0) = T(1);     // v -> v
    Phi_axis(0,3) = phi_va;   // a -> v

    // p_{k+1} = p_k + h v_k + φ_pa a_k
    Phi_axis(1,0) = h;        // v -> p
    Phi_axis(1,1) = T(1);     // p -> p
    Phi_axis(1,3) = phi_pa;   // a -> p

    // S_{k+1} = S_k + h p_k + 0.5 h² v_k + φ_Sa a_k
    Phi_axis(2,0) = T(0.5)*h*h;   // v -> S
    Phi_axis(2,1) = h;            // p -> S
    Phi_axis(2,2) = T(1);         // S -> S
    Phi_axis(2,3) = phi_Sa;       // a -> S

    // a_{k+1} = α a_k
    Phi_axis(3,3) = alpha;        // a -> a
}

// Primitives with exp(-ξ/τ) and exp(-2ξ/τ):
//   A0 = ∫₀ʰ e^{-ξ/τ} dξ = τ(1 - e^{-h/τ})
//   A1 = ∫₀ʰ ξ e^{-ξ/τ} dξ = τ²(1 - e^{-h/τ}) - τh e^{-h/τ}
//   A2 = ∫₀ʰ ξ² e^{-ξ/τ} dξ = 2τ³(1 - e^{-h/τ}) - τh(h+2τ)e^{-h/τ}
//   B0 = ∫₀ʰ e^{-2ξ/τ} dξ = (τ/2)(1 - e^{-2h/τ})
//
// Series expansions (for h/τ ≪ 1):
//   A0 ≈ h - h²/(2τ) + h³/(6τ²) - h⁴/(24τ³)
//   A1 ≈ 0.5 h² - (1/3)(h³/τ) + (1/8)(h⁴/τ²)
//   A2 ≈ (1/3) h³ - (1/4)(h⁴/τ) + (1/10)(h⁵/τ²)
//   B0 ≈ h - h²/τ + (2/3)(h³/τ²) - (1/3)(h⁴/τ³)
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::QdAxis4x1_analytic(
    T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis) 
{
    using std::exp;

    const T x   = h / std::max(T(1e-12), tau);
    const T q_c = (T(2) / tau) * sigma2;  // continuous-time noise intensity

    // === Primitives with exp(-ξ/τ) and exp(-2ξ/τ) ===
    // A0 = ∫₀ʰ e^{-ξ/τ} dξ
    //    = τ(1 - e^{-h/τ})
    //
    // A1 = ∫₀ʰ ξ e^{-ξ/τ} dξ
    //    = τ²(1 - e^{-h/τ}) - τh e^{-h/τ}
    //
    // A2 = ∫₀ʰ ξ² e^{-ξ/τ} dξ
    //    = 2τ³(1 - e^{-h/τ}) - τh(h+2τ)e^{-h/τ}
    //
    // B0 = ∫₀ʰ e^{-2ξ/τ} dξ
    //    = (τ/2)(1 - e^{-2h/τ})
    //
    // Series expansions for h/τ ≪ 1:
    //   A0 ≈ h - h²/(2τ) + h³/(6τ²) - h⁴/(24τ³)
    //   A1 ≈ 0.5 h² - (1/3)(h³/τ) + (1/8)(h⁴/τ²)
    //   A2 ≈ (1/3) h³ - (1/4)(h⁴/τ) + (1/10)(h⁵/τ²)
    //   B0 ≈ h - h²/τ + (2/3)(h³/τ²) - (1/3)(h⁴/τ³)

    T A0, A1, A2, B0, alpha, alpha2;

    if (x < T(1e-3)) {
        // --- Series expansions (safe for float) ---
        alpha  = T(1) - x + x*x/T(2) - x*x*x/T(6) + x*x*x*x/T(24);
        alpha2 = T(1) - T(2)*x + T(2)*x*x - (T(4)/T(3))*x*x*x + (T(2)/T(3))*x*x*x*x;

        A0 = h - (h*h)/(T(2)*tau) + (h*h*h)/(T(6)*tau*tau) - (h*h*h*h)/(T(24)*tau*tau*tau);
        A1 = (h*h)/T(2) - (h*h*h)/(T(3)*tau) + (h*h*h*h)/(T(8)*tau*tau);
        A2 = (h*h*h)/T(3) - (h*h*h*h)/(T(4)*tau) + (h*h*h*h*h)/(T(10)*tau*tau);
        B0 = h - (h*h)/tau + (T(2)/T(3))*(h*h*h)/(tau*tau) - (h*h*h*h)/(T(3)*tau*tau*tau);
    } else {
        // --- Exact closed forms ---
        alpha   = exp(-x);
        alpha2  = exp(-T(2)*x);

        A0 = tau * (T(1) - alpha);
        A1 = tau*tau * (T(1) - alpha) - tau * h * alpha;
        A2 = T(2)*tau*tau*tau * (T(1) - alpha) - tau * h * (h + T(2)*tau) * alpha;
        B0 = (tau / T(2)) * (T(1) - alpha2);
    }

    // Short-hands for polynomial integrals ∫₀ʰ ξⁿ dξ
    const T C0 = h;
    const T C1 = T(0.5) * h*h;
    const T C2 = (h*h*h) / T(3);
    const T C3 = (h*h*h*h) / T(4);
    const T C4 = (h*h*h*h*h) / T(5);

    const T T1 = tau;
    const T T2 = tau*tau;
    const T T3 = T2*tau;
    const T T4 = T3*tau;
    const T T5 = T4*tau;
    const T T6 = T5*tau;

    // === Build K = ∫₀ʰ k(ξ)k(ξ)^T dξ, where ===
    //   k_v(ξ) = τ(1 - e^{-ξ/τ})
    //   k_p(ξ) = τξ - τ²(1 - e^{-ξ/τ})
    //   k_S(ξ) = 0.5 τ ξ² - τ² ξ + τ³(1 - e^{-ξ/τ})
    //   k_a(ξ) = e^{-ξ/τ}

    // K_aa = ∫ k_a² = ∫ e^{-2ξ/τ} dξ = B0
    const T K_aa = B0;

    // K_va = ∫ k_v k_a = τ ∫ (e^{-ξ/τ} - e^{-2ξ/τ}) = τ(A0 - B0)
    const T K_va = T1 * (A0 - B0);

    // K_vv = ∫ [τ(1 - e^{-ξ/τ})]² = τ² (C0 - 2A0 + B0)
    const T K_vv = T2 * (C0 - T(2)*A0 + B0);

    // K_pa = ∫ k_p k_a = τ A1 - τ² A0 + τ² B0
    const T K_pa = T1*A1 - T2*A0 + T2*B0;

    // K_pv = ∫ k_p k_v
    //      = τ² ∫ ξ(1 - e^{-ξ/τ}) - τ³ ∫ (1 - e^{-ξ/τ}) + τ³ ∫ e^{-ξ/τ}(1 - e^{-ξ/τ})
    //      = τ² (C1 - A1) - τ³ (C0 - A0) + τ³(A0 - B0)
    const T K_pv = T2*(C1 - A1) - T3*(C0 - A0) + T3*(A0 - B0);

    // K_pp = ∫ (τ ξ - τ² + τ² e^{-ξ/τ})²
    //      = τ² C2 - 2τ³ C1 + 2τ³ A1 + τ⁴ C0 - 2τ⁴ A0 + τ⁴ B0
    const T K_pp = T2*C2 - T(2)*T3*C1 + T(2)*T3*A1 + T4*C0 - T(2)*T4*A0 + T4*B0;

    // K_Sa = ∫ k_S k_a = 0.5 τ A2 - τ² A1 + τ³ A0 - τ³ B0
    const T K_Sa = T(0.5)*T1*A2 - T2*A1 + T3*A0 - T3*B0;

    // K_Sv = ∫ k_S k_v
    //      = 0.5 τ² (C2 - A2) - τ³ (C1 - A1) + τ⁴ (C0 - A0) - τ⁴ (A0 - B0)
    const T K_Sv = T(0.5)*T2*(C2 - A2) - T3*(C1 - A1) + T4*(C0 - A0) - T4*(A0 - B0);

    // K_Sp = ∫ k_S k_p
    //      = 0.5 τ² C3 - 1.5 τ³ C2 + 2 τ⁴ C1 - τ⁵ C0
    //      + 0.5 τ³ A2 - 2 τ⁴ A1 + 2 τ⁵ A0 - τ⁵ B0
    const T K_Sp = T(0.5)*T2*C3 - T(1.5)*T3*C2 + T(2)*T4*C1 - T5*C0
                 + T(0.5)*T3*A2 - T(2)*T4*A1 + T(2)*T5*A0 - T5*B0;

    // K_SS = ∫ k_S²
    // Poly²:   0.25 τ² C4 - τ³ C3 + 2 τ⁴ C2 - 2 τ⁵ C1 + τ⁶ C0
    // Cross:  -τ⁴ A2 + 2 τ⁵ A1 - 2 τ⁶ A0
    // α²:     +τ⁶ B0
    const T K_SS = T(0.25)*T2*C4 - T3*C3 + T(2)*T4*C2 - T(2)*T5*C1 + T6*C0
                 - T4*A2 + T(2)*T5*A1 - T(2)*T6*A0
                 + T6*B0;

    // === Assemble symmetric 4x4 discrete covariance ===
    Eigen::Matrix<T,4,4> K;
    K.setZero();
    K(0,0) = K_vv; K(0,1) = K_pv; K(0,2) = K_Sv; K(0,3) = K_va;
    K(1,0) = K_pv; K(1,1) = K_pp; K(1,2) = K_Sp; K(1,3) = K_pa;
    K(2,0) = K_Sv; K(2,1) = K_Sp; K(2,2) = K_SS; K(2,3) = K_Sa;
    K(3,0) = K_va; K(3,1) = K_pa; K(3,2) = K_Sa; K(3,3) = K_aa;

    Qd_axis = q_c * K;
}
