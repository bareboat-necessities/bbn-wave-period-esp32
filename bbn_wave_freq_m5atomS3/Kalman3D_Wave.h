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
                  T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(27.0),
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
        Rmag = Rmag_new;
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
    Matrix3 Rmag;
    MatrixM R;
    MatrixBaseN Qbase; // original Q for attitude & bias

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 R_S;  // Triple integration measurement noise

    // World-acceleration OU process a_w dynamics parameters
    T tau_aw = T(2.3);            // correlation time [s], tune 1–5 s for sea states
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(2.4*2.4); // stationary variance diag [ (m/s^2)^2 ]

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
    void vanLoanDiscretization_12x3(const Eigen::Matrix<T,12,12>& A,
                                    const Eigen::Matrix<T,12,3>&  G,
                                    const Eigen::Matrix<T,3,3>&   Sigma_c,
                                    T Ts,
                                    Eigen::Matrix<T,12,12>& Phi,
                                    Eigen::Matrix<T,12,12>& Qd) const;
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

    // Accel magnitude sanity check
    T g_meas = acc_meas.norm();
    if (std::abs(g_meas - gravity_magnitude_) > T(2.0 * gravity_magnitude_)) {
        return; // reject this update
    }

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

    // Remove yaw component from small-angle correction coming from accel
    {
        const Vector3 gb = (R_wb() * Vector3(0,0,1)).normalized(); // gravity dir in body
        Vector3 dth = xext.template head<3>();
        // Project out yaw (component along gb)
        xext.template head<3>() = dth - gb * dth.dot(gb);
    }
    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_mag_only(
    Vector3 const& mag_meas_body)
{
    const Vector3 v2hat = magnetometer_measurement_func();

    // Norm/gating on raw 3D vectors
    T n_meas = mag_meas_body.norm();
    T n_pred = v2hat.norm();
    if (n_meas < T(1e-6) || n_pred < T(1e-6)) return;

    Vector3 meas_n = mag_meas_body / n_meas;
    Vector3 pred_n = v2hat       / n_pred;
    T dotp = meas_n.dot(pred_n);

    // thresholds
    const T DOT_DANGEROUS = T(0.2);  // |dot| < 0.2 (~>78°) → ill-conditioned
    const T YAW_CLAMP     = T(0.105); // ~6° max yaw correction per update

    if (std::abs(dotp) >= DOT_DANGEROUS) {
        // SAFE → full 3D update (with hemisphere disambiguation to avoid 180° flips)
        const Vector3 meas_fixed = (dotp >= T(0)) ? mag_meas_body : -mag_meas_body;
        measurement_update_partial(meas_fixed, v2hat, Rmag);
        return;
    }

    // DANGEROUS → yaw-only (horizontal) update with clamp
    // body gravity dir (down) from current attitude
    const Vector3 gb = (R_wb() * Vector3(0,0,1)).normalized();
    const Matrix3 Hb = Matrix3::Identity() - gb * gb.transpose(); // project to horizontal plane

    Vector3 m_h = Hb * mag_meas_body;
    Vector3 v_h = Hb * v2hat;

    T nm = m_h.norm(), nv = v_h.norm();
    if (nm < T(1e-6) || nv < T(1e-6)) return; // no horizontal info

    // hemisphere disambiguation on horizontal components
    Vector3 m_hn = m_h / nm, v_hn = v_h / nv;
    if (m_hn.dot(v_hn) < T(0)) m_h = -m_h;

    // innovation & projected Jacobian: r = H(m - v), C = H * (-skew(v))
    Vector3 r = Hb * (mag_meas_body - v2hat);

    Matrix<T,3,NX> Cext = Matrix<T,3,NX>::Zero();
    Cext.template block<3,3>(0,0) = Hb * (-skew_symmetric_matrix(v2hat));

    Matrix3 Rproj = Hb * Rmag * Hb.transpose();

    // Kalman gain
    Matrix3 S_mat = Cext * Pext * Cext.transpose() + Rproj;
    Matrix<T, NX, 3> PCt = Pext * Cext.transpose();
    Eigen::LDLT<Matrix3> ldlt(S_mat);
    if (ldlt.info() != Eigen::Success) {
        S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * Rproj.norm());
        ldlt.compute(S_mat);
        if (ldlt.info() != Eigen::Success) return;
    }
    Matrix<T, NX, 3> K = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Limit the attitude error to yaw only and clamp magnitude
    Vector3 dth = xext.template head<3>();
    T dpsi = dth.dot(gb);
    T dpsi_clamped = std::max(-YAW_CLAMP, std::min(YAW_CLAMP, dpsi));
    xext.template head<3>() = dth - gb * dpsi + gb * dpsi_clamped;

    // Joseph covariance update
    MatrixNX I = MatrixNX::Identity();
    Pext = (I - K * Cext) * Pext * (I - K * Cext).transpose() + K * Rproj * K.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());

    // Apply correction
    applyQuaternionCorrectionFromErrorState();
    xext.template head<3>().setZero();
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
    using std::cos;
    using std::sin;

    // 1) Build correction quaternion from the 3×1 attitude error in the error state
    Eigen::Matrix<T,3,1> dtheta = xext.template head<3>();
    T a = dtheta.norm();

    Eigen::Quaternion<T> corr;
    if (a < T(1e-8)) {
        // first-order for numerical stability
        corr.w()   = T(1);
        corr.vec() = T(0.5) * dtheta;
    } else {
        T half_a = T(0.5) * a;
        T s = sin(half_a) / a;
        corr.w()   = cos(half_a);
        corr.vec() = s * dtheta;
    }
    corr.normalize();

    // 2) Apply to nominal quaternion (right-multiply, consistent with your propagation)
    qref = qref * corr;
    qref.normalize();

    // 3) MEKF reset: propagate covariance through the change of linearization point
    //    G is identity except the 3×3 attitude-error block:
    //    G_att ≈ I - 0.5 [dθ×]
    Eigen::Matrix<T, NX, NX> G = Eigen::Matrix<T, NX, NX>::Identity();
    const Eigen::Matrix<T,3,3> S = skew_symmetric_matrix(dtheta);
    G.template block<3,3>(0,0) = Eigen::Matrix<T,3,3>::Identity() - T(0.5) * S;

    Pext = G * Pext * G.transpose();
    Pext = T(0.5) * (Pext + Pext.transpose());   // enforce symmetry

    // 4) Zero the attitude error in the error-state and mirror Pbase
    xext.template head<3>().setZero();
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

// normalize quaternion
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::normalizeQuat() {
  qref.normalize();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::applyIntegralZeroPseudoMeas() {
    // H picks S block only
    Matrix<T,3,NX> H = Matrix<T,3,NX>::Zero();
    H.template block<3,3>(0, OFF_S) = Matrix3::Identity();

    // Innovation (target S = 0)
    Vector3 inno = - xext.template segment<3>(OFF_S);

    // Innovation covariance and PHt
    Matrix3 S_mat = H * Pext * H.transpose() + R_S;
    Matrix<T, NX, 3> PHt = Pext * H.transpose();

    // Solve for K
// Solve for K
Eigen::LDLT<Matrix3> ldlt(S_mat);
if (ldlt.info() != Eigen::Success) {
    S_mat += Matrix3::Identity() * std::max(std::numeric_limits<T>::epsilon(), T(1e-7) * R_S.norm());
    ldlt.compute(S_mat);
    if (ldlt.info() != Eigen::Success) return;
}

// == block attitude (and optionally gyro bias) BEFORE computing K ==
Matrix<T, NX, 3> PHt = Pext * H.transpose();
PHt.template block<3,3>(0,0).setZero();        // no attitude update from S
if constexpr (with_gyro_bias) {
    PHt.template block<3,3>(3,0).setZero();    // no gyro-bias update from S
}
// If you also want S not to steer a_w, uncomment:
// PHt.template block<3,3>(OFF_AW,0).setZero();

Matrix<T, NX, 3> K = PHt * ldlt.solve(Matrix3::Identity());

// State update
xext.noalias() += K * inno;

// Joseph covariance update
MatrixNX I = MatrixNX::Identity();
Pext = (I - K * H) * Pext * (I - K * H).transpose() + K * R_S * K.transpose();
Pext = T(0.5) * (Pext + Pext.transpose());

// === hard decorrelate S with attitude so it won’t leak back ===
Pext.template block<3,3>(OFF_S, 0).setZero();
Pext.template block<3,3>(0,      OFF_S).setZero();
if constexpr (with_gyro_bias) {
    Pext.template block<3,3>(OFF_S, 3).setZero();
    Pext.template block<3,3>(3,      OFF_S).setZero();
}

// Mirror base covariance
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

    // === Linear subsystem [v,p,S,a_w] ===
    using Mat12 = Eigen::Matrix<T,12,12>;
    Mat12 Phi_lin; Phi_lin.setZero();
    Mat12 Qd_lin;  Qd_lin.setZero();

    for (int axis = 0; axis < 3; ++axis) {
        T tau    = std::max(T(1e-6), tau_aw);
        T sigma2 = Sigma_aw_stat(axis,axis);

        Eigen::Matrix<T,4,4> Phi_axis, Qd_axis;
        PhiAxis4x1_analytic(tau, Ts, Phi_axis);
        QdAxis4x1_analytic(tau, Ts, sigma2, Qd_axis);

        int idx[4] = {0,3,6,9}; // v,p,S,a offsets
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

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::PhiAxis4x1_analytic(
    T tau, T h, Eigen::Matrix<T,4,4>& Phi_axis)
{
    using std::exp;
    const T inv_tau = T(1) / std::max(T(1e-12), tau);
    const T alpha   = exp(-h * inv_tau);

    // Coefficients (exact)
    const T phi_va = tau * (T(1) - alpha);
    const T phi_pa = tau * h - tau * tau * (T(1) - alpha);
    const T phi_Sa = (T(0.5)*tau*h*h) - (tau*tau*h) + (tau*tau*tau)*(T(1) - alpha);

    Phi_axis.setZero();
    // v_{k+1} = v_k + phi_va * a_k
    Phi_axis(0,0) = T(1);     // v -> v
    Phi_axis(0,3) = phi_va;   // a -> v

    // p_{k+1} = p_k + h v_k + phi_pa * a_k
    Phi_axis(1,0) = h;        // v -> p
    Phi_axis(1,1) = T(1);     // p -> p
    Phi_axis(1,3) = phi_pa;   // a -> p

    // S_{k+1} = S_k + h p_k + 0.5 h^2 v_k + phi_Sa * a_k
    Phi_axis(2,0) = T(0.5)*h*h;   // v -> S
    Phi_axis(2,1) = h;            // p -> S
    Phi_axis(2,2) = T(1);         // S -> S
    Phi_axis(2,3) = phi_Sa;       // a -> S

    // a_{k+1} = alpha * a_k
    Phi_axis(3,3) = alpha;    // a -> a
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::QdAxis4x1_analytic(
    T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis)
{
    // States order: [v, p, S, a]
    using std::exp;
    const T Tinv    = T(1) / std::max(T(1e-12), tau);
    const T alpha   = exp(-h * Tinv);
    const T alpha2  = exp(-T(2) * h * Tinv);
    const T q_c     = (T(2) / tau) * sigma2;  // continuous-time noise intensity

    // Short-hands for simple polynomial integrals over [0,h]
    const T C0 = h;
    const T C1 = T(0.5) * h*h;
    const T C2 = (h*h*h) / T(3);
    const T C3 = (h*h*h*h) / T(4);
    const T C4 = (h*h*h*h*h) / T(5);

    // Primitives with exp(-ξ/τ) over [0,h]
    const T A0 = tau * (T(1) - alpha);
    const T A1 = tau*tau * (T(1) - alpha) - tau * h * alpha;
    const T A2 = T(2)*tau*tau*tau * (T(1) - alpha) - tau * h * (h + T(2)*tau) * alpha;
    // We’ll also need ∫ ξ^3 e^{-ξ/τ} dξ:
    // ∫ ξ^3 e^{-ξ/τ} dξ = e^{-ξ/τ} * [ -τ ξ^3 - 3 τ^2 ξ^2 - 6 τ^3 ξ - 6 τ^4 ] |_0^h
    const T A3 = ( -tau * h*h*h - T(3)*tau*tau * h*h - T(6)*tau*tau*tau * h - T(6)*tau*tau*tau*tau ) * alpha
               + T(6)*tau*tau*tau*tau;

    // Primitives with exp(-2ξ/τ) over [0,h]
    // General: ∫ e^{-λ ξ} dξ = (1 - e^{-λ h})/λ;  λ = 2/τ
    const T B0 = (tau / T(2)) * (T(1) - alpha2);

    // Build k(ξ) components for the driven column (a-column of Φ(ξ)):
    // k_v = τ (1 - e^{-ξ/τ}); k_p = τ ξ - τ^2 (1 - e^{-ξ/τ});
    // k_S = 0.5 τ ξ^2 - τ^2 ξ + τ^3 (1 - e^{-ξ/τ}); k_a = e^{-ξ/τ}
    // Qd = q_c * ∫ k(ξ) k(ξ)^T dξ  (all entries below are the integral K; multiply by q_c at the end)

    const T T1 = tau;
    const T T2 = tau*tau;
    const T T3 = T2*tau;
    const T T4 = T3*tau;
    const T T5 = T4*tau;
    const T T6 = T5*tau;

    // Useful combos
    // For (1 - e^{-ξ/τ}): integrals reduce to C• - A•
    const T I1mA0 = C0 - A0;  // ∫ (1 - α) dξ
    const T Ix1mA1 = C1 - A1; // ∫ ξ (1 - α) dξ
    const T Ix21mA2 = C2 - A2;// ∫ ξ^2 (1 - α) dξ

    // K_aa
    const T K_aa = B0;

    // K_va = ∫ k_v * k_a = τ ∫ (α - α^2) = τ (A0 - B0)
    const T K_va = T1 * (A0 - B0);

    // K_vv = ∫ [τ(1 - α)]^2 = τ^2 ∫ (1 - 2α + α^2) = τ^2 (C0 - 2A0 + B0)
    const T K_vv = T2 * (C0 - T(2)*A0 + B0);

    // K_pa = ∫ k_p * k_a = τ A1 - τ^2 A0 + τ^2 B0
    const T K_pa = T1*A1 - T2*A0 + T2*B0;

    // K_pv = ∫ k_p * k_v = τ^2 ∫ ξ(1-α) - τ^3 ∫ (1-α) + τ^3 ∫ α(1-α)
    const T K_pv = T2*Ix1mA1 - T3*I1mA0 + T3*(A0 - B0);

    // K_pp = ∫ (T ξ - T^2 + T^2 α)^2
    // = T^2 C2 - 2 T^3 C1 + 2 T^3 A1 + T^4 C0 - 2 T^4 A0 + T^4 B0
    const T K_pp = T2*C2 - T(2)*T3*C1 + T(2)*T3*A1 + T4*C0 - T(2)*T4*A0 + T4*B0;

    // K_Sa = 0.5 T A2 - T^2 A1 + T^3 A0 - T^3 B0
    const T K_Sa = T(0.5)*T1*A2 - T2*A1 + T3*A0 - T3*B0;

    // K_Sv = 0.5 T^2 (C2 - A2) - T^3 (C1 - A1) + T^4 (C0 - A0) - T^4 (A0 - B0)
    const T K_Sv = T(0.5)*T2*Ix21mA2 - T3*Ix1mA1 + T4*I1mA0 - T4*(A0 - B0);

    // K_Sp = 0.5 T^2 C3 - 1.5 T^3 C2 + 2 T^4 C1 - T^5 C0
    //       + 0.5 T^3 A2 - 2 T^4 A1 + 2 T^5 A0 - T^5 B0
    const T K_Sp = T(0.5)*T2*C3 - T(1.5)*T3*C2 + T(2)*T4*C1 - T5*C0
                 + T(0.5)*T3*A2 - T(2)*T4*A1 + T(2)*T5*A0 - T5*B0;

    // K_SS = (poly square) + (cross with α) + (α^2 term)
    // Poly^2: 0.25 T^2 C4 - T^3 C3 + 2 T^4 C2 - 2 T^5 C1 + T^6 C0
    // Cross:  - T^4 A2 + 2 T^5 A1 - 2 T^6 A0
    // Alpha^2: + T^6 B0
    const T K_SS = T(0.25)*T2*C4 - T3*C3 + T(2)*T4*C2 - T(2)*T5*C1 + T6*C0
                 - T4*A2 + T(2)*T5*A1 - T(2)*T6*A0
                 + T6*B0;

    // Assemble symmetric 4x4 K, then scale by q_c
    Eigen::Matrix<T,4,4> K;
    K.setZero();
    K(0,0) = K_vv; K(0,1) = K_pv; K(0,2) = K_Sv; K(0,3) = K_va;
    K(1,0) = K_pv; K(1,1) = K_pp; K(1,2) = K_Sp; K(1,3) = K_pa;
    K(2,0) = K_Sv; K(2,1) = K_Sp; K(2,2) = K_SS; K(2,3) = K_Sa;
    K(3,0) = K_va; K(3,1) = K_pa; K(3,2) = K_Sa; K(3,3) = K_aa;

    Qd_axis = q_c * K;
}
