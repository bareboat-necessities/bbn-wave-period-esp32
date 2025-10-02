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
#include <cmath>
#include <stdexcept>

using Eigen::Matrix;

template<typename T>
inline T safe_inv_tau(T tau) {
    // Prevent division by ~0 while preserving sign
    return T(1) / ((tau >= T(1e-8)) ? tau : T(1e-8));
}

template<typename T>
struct OUPrims {
    T x;        // h/tau
    T alpha;    // e^{-x}
    T em1;      // expm1(-x) = e^{-x} - 1  (negative)
    T alpha2;   // e^{-2x}
    T em1_2;    // expm1(-2x) = e^{-2x} - 1 (negative)
};

template<typename T>
inline OUPrims<T> make_prims(T h, T tau) {
    const T inv_tau = safe_inv_tau(tau);
    const T x = h * inv_tau;
    const T alpha  = std::exp(-x);
    const T em1    = std::expm1(-x);    // high-accuracy for small x
    const T alpha2 = std::exp(-T(2)*x);
    const T em1_2  = std::expm1(-T(2)*x);
    return {x, alpha, em1, alpha2, em1_2};
}

// Full exponential-map correction (Rodrigues in quaternion form).
// Accurate for both small and large |δθ|.
// Right-multiply convention: q_new = qref ⊗ δq(δθ), where δθ = vector increment.
// Uses Maclaurin expansion with FMA for small |δθ| to avoid 0/0 and cancellation.
// Series preserves 2nd-order accuracy in propagation and correction.
template<typename T>
inline Eigen::Quaternion<T> quat_from_delta_theta(const Eigen::Matrix<T,3,1>& dtheta) {
    const T theta = dtheta.norm();
    const T half_theta = T(0.5) * theta;

    T w, k; // scalar part, vector scale = sin(|δθ|/2)/|δθ|
    if (theta < T(1e-2)) {
        // Maclaurin expansion (FMA-friendly)
        const T t2 = theta * theta;
        const T t4 = t2 * t2;

        // w = cos(theta/2) ≈ 1 - θ²/8 + θ⁴/384
        w = T(1);
        w = std::fma(-t2, T(1)/T(8), w);
        w = std::fma( t4, T(1)/T(384), w);

        // k = sin(theta/2)/θ ≈ 1/2 - θ²/48 + θ⁴/3840
        k = T(0.5);
        k = std::fma(-t2, T(1)/T(48), k);
        k = std::fma( t4, T(1)/T(3840), k);
    } else {
        w = std::cos(half_theta);
        k = std::sin(half_theta) / theta;
    }

    const Eigen::Matrix<T,3,1> v = k * dtheta;
    Eigen::Quaternion<T> q(w, v.x(), v.y(), v.z());

    // (Unit by construction, but normalize for safety)
    q.normalize();
    return q;
}

// Safe expansions for OU discrete coefficients.
// Provides phi_pa, phi_Sa, A1, A2 with series fallback for small x = h/tau.
template<typename T>
struct OUDiscreteCoeffs {
    T phi_pa; // coefficient for position vs. accel
    T phi_Sa; // coefficient for S vs. accel
    T A1;     // helper for Qd
    T A2;     // helper for Qd
};

template<typename T>
inline OUDiscreteCoeffs<T> safe_phi_A_coeffs(T h, T tau) {
    OUDiscreteCoeffs<T> c;
    const T inv_tau = safe_inv_tau(tau);
    const T x = h * inv_tau;
    const T tau2 = tau*tau;
    const T tau3 = tau2*tau;

    if (x < T(1e-2)) {
        // Maclaurin expansions
        const T x2 = x*x;
        const T x3 = x2*x;
        const T x4 = x3*x;
        const T x5 = x4*x;

        // phi_pa ≈ τ² (x²/2 - x³/6 + x⁴/24)
        c.phi_pa = tau2 * (T(0.5)*x2 - T(1.0/6.0)*x3 + T(1.0/24.0)*x4);

        // phi_Sa ≈ τ³ (x³/6 - x⁴/24 + x⁵/120)
        c.phi_Sa = tau3 * (T(1.0/6.0)*x3 - T(1.0/24.0)*x4 + T(1.0/120.0)*x5);

        // A1 ≈ τ² (x²/2 - x³/3 + x⁴/8)
        c.A1 = tau2 * (T(0.5)*x2 - T(1.0/3.0)*x3 + T(1.0/8.0)*x4);

        // A2 ≈ τ³ ( -x³/3 + x⁴/4 - x⁵/10 )
        c.A2 = tau3 * (-(T(1.0/3.0))*x3 + T(1.0/4.0)*x4 - T(1.0/10.0)*x5);

    } else {
        // General closed-form branch
        const T alpha  = std::exp(-x);
        const T em1    = std::expm1(-x);    
        // reuse for stability
        const T phi_pa = tau2 * (x + em1);
        const T phi_Sa = tau3 * (T(0.5)*x*x - x - em1);

        c.phi_pa = phi_pa;
        c.phi_Sa = phi_Sa;

        // A1, A2 in terms of primitives
        c.A1 = tau2 * (-em1 - x*alpha);
        c.A2 = tau3 * (-T(2)*em1 - alpha*(x*(x+T(2))));
    }
    return c;
}

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
                  T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(1.5),
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

    int pseudo_update_counter_ = 0;   // counts time_update calls
    static constexpr int PSEUDO_UPDATE_PERIOD = 3; // every N-th update

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

    // Reusable helpers for all 3×3 measurement updates
    
    // Factor S with LDLT and a diagonal safety boost if needed.
    // The `noise_scale` argument should be something like R.norm() or S.norm() from the branch.
    EIGEN_STRONG_INLINE bool safe_ldlt3_(Matrix3& S, Eigen::LDLT<Matrix3>& ldlt, T noise_scale) const {
        ldlt.compute(S);
        if (ldlt.info() == Eigen::Success) return true;
    
        // Minimum positive bump
        const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * std::max(noise_scale, T(1)));
        S.diagonal().array() += bump;
    
        ldlt.compute(S);
        return (ldlt.info() == Eigen::Success);
    }
    
    // Joseph covariance update: P ← P - KCP - (KCP)ᵀ + K S Kᵀ, enforce symmetry.
    EIGEN_STRONG_INLINE void joseph_update3_(const Eigen::Matrix<T,NX,3>& K, const Matrix3& S, const Eigen::Matrix<T,NX,3>& PCt) {
        const Eigen::Matrix<T,3,NX> CP  = PCt.transpose();
        const Eigen::Matrix<T,NX,NX> KCP = K * CP;
        const Eigen::Matrix<T,NX,NX> KSKt= K * S * K.transpose();
    
        Pext.noalias() -= KCP;
        Pext.noalias() -= KCP.transpose();
        Pext.noalias() += KSKt;
    
        // Symmetrize for numerical hygiene
        Pext = T(0.5) * (Pext + Pext.transpose());
    }
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
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::time_update(
    Vector3 const& gyr, T Ts)
{
    // Attitude mean propagation
    Vector3 gyro_bias;
    if constexpr (with_gyro_bias) {
        gyro_bias = xext.template segment<3>(3);
    } else {
        gyro_bias = Vector3::Zero();
    }
    last_gyr_bias_corrected = gyr - gyro_bias;

    // Δθ = ω Ts → quaternion increment (right-multiplicative)
    Eigen::Quaternion<T> dq = quat_from_delta_theta((last_gyr_bias_corrected * Ts).eval());
    qref = qref * dq;
    qref.normalize();

    // Build only the blocks we actually need: F_AA, Q_AA, F_LL, Q_LL

    // Attitude error (+ optional gyro bias) block
    Matrix3 I = Matrix3::Identity();
    const Vector3 w = last_gyr_bias_corrected;
    const T omega = w.norm();
    const T theta = omega * Ts;

    // F_AA : BASE_N x BASE_N
    MatrixBaseN F_AA; F_AA.setIdentity();
    if (theta < T(1e-5)) {
        Matrix3 Wx = skew_symmetric_matrix(w);
        F_AA.template topLeftCorner<3,3>() = I - Wx*Ts + (Wx*Wx)*(Ts*Ts/T(2));
    } else {
        Matrix3 W = skew_symmetric_matrix(w / (omega + std::numeric_limits<T>::epsilon()));
        const T s = std::sin(theta), c = std::cos(theta);
        F_AA.template topLeftCorner<3,3>() = I - s*W + (T(1)-c)*(W*W);
    }
    if constexpr (with_gyro_bias) {
        F_AA.template block<3,3>(0,3) = -Matrix3::Identity() * Ts;
    }

    MatrixBaseN Q_AA = Qbase * Ts;

    // Linear subsystem [v,p,S,a_w] block: 12x12
    using Mat12 = Eigen::Matrix<T,12,12>;
    Mat12 F_LL; F_LL.setZero();
    Mat12 Q_LL; Q_LL.setZero();

    for (int axis = 0; axis < 3; ++axis) {
        const T tau    = std::max(T(1e-6), tau_aw);
        const T sigma2 = Sigma_aw_stat(axis, axis);

        Eigen::Matrix<T,4,4> Phi_axis, Qd_axis;
        PhiAxis4x1_analytic(tau, Ts, Phi_axis);
        QdAxis4x1_analytic (tau, Ts, sigma2, Qd_axis);

        const int idx[4] = {0,3,6,9}; // v,p,S,a offsets per axis in the 12x12
        for (int i = 0; i < 4; ++i)  {
            for (int j = 0; j < 4; ++j) {
                F_LL(idx[i] + axis, idx[j] + axis) = Phi_axis(i,j);
                Q_LL(idx[i] + axis, idx[j] + axis) = Qd_axis(i,j);
            }
        }
    }

    // Mean propagation for [v,p,S,a_w] using F_LL
    Eigen::Matrix<T,12,1> x_lin_prev;
    x_lin_prev.template segment<3>(0)  = xext.template segment<3>(OFF_V);
    x_lin_prev.template segment<3>(3)  = xext.template segment<3>(OFF_P);
    x_lin_prev.template segment<3>(6)  = xext.template segment<3>(OFF_S);
    x_lin_prev.template segment<3>(9)  = xext.template segment<3>(OFF_AW);

    Eigen::Matrix<T,12,1> x_lin_next = F_LL * x_lin_prev;

    xext.template segment<3>(OFF_V)  = x_lin_next.template segment<3>(0);
    xext.template segment<3>(OFF_P)  = x_lin_next.template segment<3>(3);
    xext.template segment<3>(OFF_S)  = x_lin_next.template segment<3>(6);
    xext.template segment<3>(OFF_AW) = x_lin_next.template segment<3>(9);

    // Covariance propagation: exact block form of P ← FPFᵀ + Q
    // Keeps ALL cross terms. No NX×NX temporaries are formed.
    {
        constexpr int NA = BASE_N;
        constexpr int NL = 12;

        // Extract current blocks
        Eigen::Matrix<T,NA,NA> P_AA = Pext.template block<NA,NA>(0,0);
        Eigen::Matrix<T,NL,NL> P_LL = Pext.template block<NL,NL>(OFF_V,OFF_V);
        Eigen::Matrix<T,NA,NL> P_AL = Pext.template block<NA,NL>(0,OFF_V);

        // AA: P_AA = F_AA P_AA F_AAᵀ + Q_AA
        Eigen::Matrix<T,NA,NA> tmpAA;
        tmpAA.noalias() = F_AA * P_AA;
        P_AA.noalias()  = tmpAA * F_AA.transpose();
        P_AA.noalias() += Q_AA;

        // LL: P_LL = F_LL P_LL F_LLᵀ + Q_LL
        Eigen::Matrix<T,NL,NL> tmpLL;
        tmpLL.noalias() = F_LL * P_LL;
        P_LL.noalias()  = tmpLL * F_LL.transpose();
        P_LL.noalias() += Q_LL;

        // AL: P_AL = F_AA P_AL F_LLᵀ
        Eigen::Matrix<T,NA,NL> tmpAL;
        tmpAL.noalias() = F_AA * P_AL;
        P_AL.noalias()  = tmpAL * F_LL.transpose();

        // Write back AA, LL, AL (and symmetric counterpart)
        Pext.template block<NA,NA>(0,0)         = P_AA;
        Pext.template block<NL,NL>(OFF_V,OFF_V) = P_LL;
        Pext.template block<NA,NL>(0,OFF_V)     = P_AL;
        Pext.template block<NL,NA>(OFF_V,0)     = P_AL.transpose();

        if constexpr (with_accel_bias) {
            constexpr int NB = 3;

            // Bias block (random walk): P_BB ← P_BB + Q_bacc_*Ts
            auto P_BB = Pext.template block<NB,NB>(OFF_BA,OFF_BA);
            P_BB.noalias() += Q_bacc_ * Ts;
            Pext.template block<NB,NB>(OFF_BA,OFF_BA) = P_BB;

            // Cross terms with bias: because F_BB = I
            auto P_AB = Pext.template block<NA,NB>(0,OFF_BA);
            auto P_LB = Pext.template block<NL,NB>(OFF_V,OFF_BA);

            // P_AB ← F_AA P_AB ;  P_LB ← F_LL P_LB
            Eigen::Matrix<T,NA,NB> tmpAB; tmpAB.noalias() = F_AA * P_AB;
            Eigen::Matrix<T,NL,NB> tmpLB; tmpLB.noalias() = F_LL * P_LB;
            Pext.template block<NA,NB>(0,OFF_BA) = tmpAB;
            Pext.template block<NL,NB>(OFF_V,OFF_BA) = tmpLB;

            // Symmetric counterparts
            Pext.template block<NB,NA>(OFF_BA,0)       = tmpAB.transpose();
            Pext.template block<NB,NL>(OFF_BA,OFF_V)   = tmpLB.transpose();
        }

        // Final hygiene (keep symmetry)
        Pext = T(0.5) * (Pext + Pext.transpose());
    }

    // Mirror base covariance for legacy accessor
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);

    // Drift correction on S (unchanged)
    if (++pseudo_update_counter_ >= PSEUDO_UPDATE_PERIOD) {
        applyIntegralZeroPseudoMeas();
        pseudo_update_counter_ = 0;
    }
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_partial(
    const Eigen::Ref<const Vector3>& meas,
    const Eigen::Ref<const Vector3>& vhat,
    const Eigen::Ref<const Matrix3>& Rm)
{
    // Residual r = (measured - predicted)
    const Vector3 r = meas - vhat;

    // Jacobian wrt attitude error (right-multiplicative convention)
    // J_att = -skew(vhat)
    const Matrix3 J_att = -skew_symmetric_matrix(vhat);

    // Innovation covariance S = C P Cᵀ + Rm (3×3), with C touching only attitude
    Matrix3 S_mat = Rm;
    {
        constexpr int OFF_TH = 0;
        const Matrix3 P_th_th = Pext.template block<3,3>(OFF_TH, OFF_TH);
        S_mat.noalias() += J_att * P_th_th * J_att.transpose();
    }

    // PCᵀ = P Cᵀ (NX×3) — only attitude columns contribute
    Eigen::Matrix<T, NX, 3> PCt = Pext.template block<NX,3>(0,0) * J_att.transpose();

    // Gain
    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, Rm.norm())) return;
    const Eigen::Matrix<T, NX, 3> K = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update (Joseph form)
    joseph_update3_(K, S_mat, PCt);

    // Apply small-angle correction to quaternion and zero the attitude error in xext
    applyQuaternionCorrectionFromErrorState();

    // Mirror base block for legacy accessor
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_acc_only(
    Vector3 const& acc_meas, T tempC)
{
    // Gate accel magnitude
    const T g_meas = acc_meas.norm();
    if (std::abs(g_meas - gravity_magnitude_) > T(2.0) * gravity_magnitude_) return;

    // Predicted specific force
    const Vector3 v1hat = accelerometer_measurement_func(tempC);

    // Jacobians wrt the only columns C touches
    const Matrix3 J_att = -skew_symmetric_matrix(v1hat); // d f / d θ
    const Matrix3 J_aw  =  R_wb();                       // d f / d a_w
    Matrix3       J_ba;                                  // d f / d b_a
    if constexpr (with_accel_bias) J_ba.setIdentity();

    const Vector3 r = acc_meas - v1hat;

    // Innovation covariance S = C P Cᵀ + Racc (3×3)
    Matrix3 S_mat = Racc;
    {
        constexpr int OFF_TH = 0;
        const int off_aw = OFF_AW;
        const int off_ba = OFF_BA;

        const Matrix3 P_th_th = Pext.template block<3,3>(OFF_TH, OFF_TH);
        const Matrix3 P_th_aw = Pext.template block<3,3>(OFF_TH, off_aw);
        const Matrix3 P_aw_aw = Pext.template block<3,3>(off_aw,  off_aw);

        S_mat.noalias() += J_att * P_th_th * J_att.transpose();
        S_mat.noalias() += J_att * P_th_aw * J_aw.transpose();
        S_mat.noalias() += J_aw  * P_th_aw.transpose() * J_att.transpose();
        S_mat.noalias() += J_aw  * P_aw_aw * J_aw.transpose();

        if constexpr (with_accel_bias) {
            const Matrix3 P_th_ba = Pext.template block<3,3>(OFF_TH, off_ba);
            const Matrix3 P_aw_ba = Pext.template block<3,3>(off_aw,  off_ba);
            const Matrix3 P_ba_ba = Pext.template block<3,3>(off_ba,  off_ba);

            S_mat.noalias() += J_att * P_th_ba * J_ba.transpose();
            S_mat.noalias() += J_aw  * P_aw_ba * J_ba.transpose();

            S_mat.noalias() += J_ba  * P_th_ba.transpose() * J_att.transpose();
            S_mat.noalias() += J_ba  * P_aw_ba.transpose() * J_aw.transpose();

            S_mat.noalias() += J_ba  * P_ba_ba * J_ba.transpose();
        }
    }

    // PCᵀ = P Cᵀ (NX×3)
    Eigen::Matrix<T,NX,3> PCt; PCt.setZero();
    {
        constexpr int OFF_TH = 0;
        const auto P_all_th = Pext.template block<NX,3>(0, OFF_TH);
        const auto P_all_aw = Pext.template block<NX,3>(0, OFF_AW);
        PCt.noalias() += P_all_th * J_att.transpose();
        PCt.noalias() += P_all_aw * J_aw.transpose();

        if constexpr (with_accel_bias) {
            const auto P_all_ba = Pext.template block<NX,3>(0, OFF_BA);
            PCt.noalias() += P_all_ba * J_ba.transpose(); // J_ba = I
        }
    }

    // Gain
    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, Racc.norm())) return;
    const Eigen::Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update
    joseph_update3_(K, S_mat, PCt);

    // Apply quaternion correction
    applyQuaternionCorrectionFromErrorState();

    // Mirror base block
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_mag_only(
    const Vector3& mag_meas_body)
{
    // Predicted magnetic field in body frame
    const Vector3 v2hat = magnetometer_measurement_func();

    // Gating on norms
    const T n_meas = mag_meas_body.norm();
    const T n_pred = v2hat.norm();
    if (n_meas < T(1e-6) || n_pred < T(1e-6)) return;

    // Unit vectors for dot product check
    Vector3 meas_n = mag_meas_body / n_meas;
    Vector3 pred_n = v2hat       / n_pred;
    T dotp = meas_n.dot(pred_n);

    const T DOT_DANGEROUS = T(0.2);   // if |dot| < 0.2 (~>78°) → yaw-only branch
    const T YAW_CLAMP     = T(0.105); // ~6° clamp per update

    if (std::abs(dotp) >= DOT_DANGEROUS) {
        // BRANCH 1: SAFE FULL 3D UPDATE
        const Vector3 meas_fixed = (dotp >= T(0)) ? mag_meas_body : -mag_meas_body;
        const Vector3 r = meas_fixed - v2hat;

        const Matrix3 J_att = -skew_symmetric_matrix(v2hat);

        Matrix3 S_mat = Rmag;
        const Matrix3 P_th_th = Pext.template block<3,3>(0,0);
        S_mat.noalias() += J_att * P_th_th * J_att.transpose();

        Eigen::Matrix<T,NX,3> PCt = Pext.template block<NX,3>(0,0) * J_att.transpose();

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, Rmag.norm())) return;
        const Eigen::Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * r;

        joseph_update3_(K, S_mat, PCt);
    } else {
        // BRANCH 2: YAW-ONLY PROJECTED UPDATE
        const Vector3 gb = (R_wb() * Vector3(0,0,1)).normalized();
        const Matrix3 Hb = Matrix3::Identity() - gb * gb.transpose();

        Vector3 m_h = Hb * mag_meas_body;
        Vector3 v_h = Hb * v2hat;

        T nm = m_h.norm(), nv = v_h.norm();
        if (nm < T(1e-6) || nv < T(1e-6)) return;

        Vector3 m_hn = m_h / nm, v_hn = v_h / nv;
        if (m_hn.dot(v_hn) < T(0)) m_h = -m_h;

        const Vector3 r = Hb * (mag_meas_body - v2hat);
        const Matrix3 J_att_proj = Hb * (-skew_symmetric_matrix(v2hat));

        Matrix3 S_mat = Hb * Rmag * Hb.transpose();
        const Matrix3 P_th_th = Pext.template block<3,3>(0,0);
        S_mat.noalias() += J_att_proj * P_th_th * J_att_proj.transpose();

        Eigen::Matrix<T,NX,3> PCt = Pext.template block<NX,3>(0,0) * J_att_proj.transpose();

        Eigen::LDLT<Matrix3> ldlt;
        if (!safe_ldlt3_(S_mat, ldlt, S_mat.norm())) return;
        const Eigen::Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

        xext.noalias() += K * r;

        // Limit correction to yaw only
        Vector3 dth = xext.template head<3>();
        const T dpsi = dth.dot(gb);
        const T dpsi_clamped = std::max(-YAW_CLAMP, std::min(YAW_CLAMP, dpsi));
        xext.template head<3>() = (dth - gb * dpsi + gb * dpsi_clamped).eval();

        joseph_update3_(K, S_mat, PCt);
    }

    // Apply quaternion correction
    applyQuaternionCorrectionFromErrorState();

    // Mirror base block
    Pbase = Pext.topLeftCorner(BASE_N, BASE_N);
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
    Eigen::Quaternion<T> corr = quat_from_delta_theta((xext.template segment<3>(0)).eval());
    qref = qref * corr;
    qref.normalize();

    // Clear error-state attitude correction after applying
    xext.template head<3>().setZero();
}

// normalize quaternion
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::normalizeQuat() {
  qref.normalize();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::applyIntegralZeroPseudoMeas()
{
    constexpr int off_S = OFF_S;   // offset of S block (3 states)

    // Innovation: target S = 0
    const Vector3 r = -xext.template segment<3>(off_S);

    // Innovation covariance S = P_SS + R_S
    Matrix3 S_mat = Pext.template block<3,3>(off_S, off_S) + R_S;

    // Cross covariance PCᵀ = P(:,S) (NX×3)
    Eigen::Matrix<T,NX,3> PCt = Pext.template block<NX,3>(0, off_S);

    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, R_S.norm())) return;
    const Eigen::Matrix<T,NX,3> K = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update
    joseph_update3_(K, S_mat, PCt);

    // Apply quaternion correction (attitude may get nudged via cross-covariances)
    applyQuaternionCorrectionFromErrorState();

    // Mirror base block
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

    if (theta < T(1e-5)) {
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
    const auto P = make_prims<T>(h, tau);
    // Stable re-expressions (let x = h/tau, em1 = expm1(-x)):
    // 1 - alpha = -em1
    // phi_va = tau*(1 - alpha)               = -tau*em1
    // phi_pa = tau*h - tau^2*(1 - alpha)     = tau^2*(x + em1)
    // phi_Sa = 0.5*tau*h^2 - tau^2*h + tau^3*(1 - alpha)
    //        = tau^3*(0.5*x^2 - x - em1)
    const T tau2 = tau * tau;
    const T tau3 = tau2 * tau;

    const T phi_va = -tau * P.em1;
    auto coeffs = safe_phi_A_coeffs<T>(h, tau);
    const T phi_pa = coeffs.phi_pa;
    const T phi_Sa = coeffs.phi_Sa;

    Phi_axis.setZero();
    // v_{k+1}
    Phi_axis(0,0) = T(1);
    Phi_axis(0,3) = phi_va;

    // p_{k+1}
    Phi_axis(1,0) = h;
    Phi_axis(1,1) = T(1);
    Phi_axis(1,3) = phi_pa;

    // S_{k+1}
    Phi_axis(2,0) = T(0.5)*h*h;
    Phi_axis(2,1) = h;
    Phi_axis(2,2) = T(1);
    Phi_axis(2,3) = phi_Sa;

    // a_{k+1}
    Phi_axis(3,3) = P.alpha;
}

// Discrete OU covariance for [v, p, S, a] axis subsystem.
// Inputs:
//   tau     = OU correlation time constant [s]
//   h       = step size (sampling interval Ts) [s]
//   sigma2  = stationary variance of a [ (m/s^2)^2 ]
// Outputs:
//   Qd_axis = (4x4) discrete covariance contribution for [v, p, S, a]
//
// Strategy:
//   - For general h/tau, use your original expm1-based analytic formulas.
//   - For small x = h/tau < 1e-3, switch to Maclaurin series expansions to
//     avoid catastrophic cancellation when subtracting nearly-equal terms.
//
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::QdAxis4x1_analytic(
    T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis)
{
    const T x = h / tau;                       // dimensionless step ratio
    const T q_c  = (T(2) / tau) * sigma2;      // continuous-time OU intensity

    Eigen::Matrix<T,4,4> K;
    K.setZero();

    if (x < T(1e-2)) {
        // Small-x branch: use series expansions to avoid cancellation
        // Derived from Maclaurin expansions of exp(-x), expm1(-x), expm1(-2x).
        // Keep up to O(h^4/τ^3) or higher depending on the entry.
        const T h2 = h*h;
        const T h3 = h2*h;
        const T h4 = h3*h;
        const T h5 = h4*h;
        const T h6 = h5*h;
        const T h7 = h6*h;
        const T h8 = h7*h;

        // Variance terms
        const T K_vv = h3 / T(3) - h4 / (T(4)*tau);
        const T K_pp = h5 / T(20) - h6 / (T(30)*tau);
        const T K_SS = h7 / T(840) - h8 / (T(960)*tau);

        // Cross-covariance terms
        const T K_aa = h - h2/tau + (T(2)/3) * h3/(tau*tau) - h4/(T(3)*tau*tau*tau);
        const T K_va = h2 / T(2) - h3/(T(3)*tau) + h4/(T(8)*tau*tau);
        const T K_pa = h3 / T(6) - h4/(T(8)*tau);
        const T K_pv = h4 / T(12) - h5/(T(15)*tau);
        const T K_Sa = h4 / T(24) - h5/(T(30)*tau);
        const T K_Sv = h5 / T(60) - h6/(T(72)*tau);
        const T K_Sp = h6 / T(360) - h7/(T(420)*tau);

        // Fill symmetric K
        K(0,0) = K_vv; K(0,1) = K_pv; K(0,2) = K_Sv; K(0,3) = K_va;
        K(1,0) = K_pv; K(1,1) = K_pp; K(1,2) = K_Sp; K(1,3) = K_pa;
        K(2,0) = K_Sv; K(2,1) = K_Sp; K(2,2) = K_SS; K(2,3) = K_Sa;
        K(3,0) = K_va; K(3,1) = K_pa; K(3,2) = K_Sa; K(3,3) = K_aa;
    } else {
        // General-x branch: stable expm1-based closed forms
        const auto P = make_prims<T>(h, tau);

        const T tau2 = tau*tau;
        const T tau3 = tau2*tau;
        const T tau4 = tau3*tau;
        const T tau5 = tau4*tau;
        const T tau6 = tau5*tau;

        // A0..A3 primitives (safe with expm1 for small/moderate x)
        const T A0 = -tau * P.em1;                                   // τ(1-α)
        auto coeffs = safe_phi_A_coeffs<T>(h, tau);
        const T A1 = coeffs.A1;
        const T A2 = coeffs.A2;
        const T A3 = tau4 * (T(6) - P.alpha * (T(6) + T(6)*P.x
                      + T(3)*P.x*P.x + P.x*P.x*P.x));

        // B0 primitive (uses expm1(-2x))
        const T B0 = -(tau/T(2)) * P.em1_2;                          // (τ/2)(1-α²)

        // Poly integrals
        const T C0 = h;
        const T C1 = T(0.5) * h*h;
        const T C2 = h*h*h / T(3);
        const T C3 = h*h*h*h / T(4);
        const T C4 = h*h*h*h*h / T(5);

        // Convenience combos
        const T I1mA0   = C0 - A0;
        const T Ix1mA1  = C1 - A1;
        const T Ix21mA2 = C2 - A2;

        // Build K (same as your original code)
        const T K_aa = B0;
        const T K_va = tau * (A0 - B0);
        const T K_vv = tau2 * (C0 - T(2)*A0 + B0);
        const T K_pa = tau*A1 - tau2*A0 + tau2*B0;
        const T K_pv = tau2*Ix1mA1 - tau3*I1mA0 + tau3*(A0 - B0);
        const T K_pp = tau2*C2 - T(2)*tau3*C1 + T(2)*tau3*A1
                     + tau4*C0 - T(2)*tau4*A0 + tau4*B0;
        const T K_Sa = T(0.5)*tau*A2 - tau2*A1 + tau3*A0 - tau3*B0;
        const T K_Sv = T(0.5)*tau2*Ix21mA2 - tau3*Ix1mA1
                     + tau4*I1mA0 - tau4*(A0-B0);
        const T K_Sp = T(0.5)*tau2*C3 - T(1.5)*tau3*C2 + T(2)*tau4*C1
                     - tau5*C0 + T(0.5)*tau3*A2 - T(2)*tau4*A1
                     + T(2)*tau5*A0 - tau5*B0;
        const T K_SS = T(0.25)*tau2*C4 - tau3*C3 + T(2)*tau4*C2
                     - T(2)*tau5*C1 + tau6*C0 - tau4*A2
                     + T(2)*tau5*A1 - T(2)*tau6*A0 + tau6*B0;

        K(0,0) = K_vv; K(0,1) = K_pv; K(0,2) = K_Sv; K(0,3) = K_va;
        K(1,0) = K_pv; K(1,1) = K_pp; K(1,2) = K_Sp; K(1,3) = K_pa;
        K(2,0) = K_Sv; K(2,1) = K_Sp; K(2,2) = K_SS; K(2,3) = K_Sa;
        K(3,0) = K_va; K(3,1) = K_pa; K(3,2) = K_Sa; K(3,3) = K_aa;
    }

    // Final scaling with continuous-time noise intensity
    Qd_axis = q_c * K;
}
