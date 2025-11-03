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
#include <Eigen/Eigenvalues>
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

        // A2 = τ³ (x³/3 − x⁴/4 + x⁵/10 + … )
        c.A2 = tau3 * (T(1.0/3.0)*x3 - T(1.0/4.0)*x4 + T(1.0/10.0)*x5);

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
        c.A2 = tau3 * (-T(2)*em1 + alpha*(x*(x+T(2))));
    }
    return c;
}

// Helper: project a symmetric NxN to PSD
template<typename T, int N>
static inline void project_psd(Eigen::Matrix<T,N,N>& S, T eps = T(1e-12)) {
    // Ensure symmetry first
    S = T(0.5) * (S + S.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N>> es(S);
    if (es.info() != Eigen::Success) {
        // Fallback: add small jitter on the diagonal and re-symmetrize
        S.diagonal().array() += eps;
        S = T(0.5) * (S + S.transpose());
        return;
    }
    Eigen::Matrix<T,N,1> lam = es.eigenvalues();
    for (int i = 0; i < N; ++i) {
        if (!(lam(i) > T(0))) lam(i) = eps; // clamp negatives/NaNs to small +ve
    }
    S = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
    // Re-symmetrize to clean float noise
    S = T(0.5) * (S + S.transpose());
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
        v2ref = B_world;    // keep µT magnitude
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
    MatrixBaseN const& covariance_base() const { return Pext.topLeftCorner(BASE_N, BASE_N); } // top-left original block
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

    // Set OU stationary covariance for world-acceleration a_w,
    // using per-axis standard deviations and an optional cross-axis correlation.
    //
    // Arguments:
    //   std_aw     : per-axis standard deviations [m/s²] for (x, y, z)
    //   rho_corr   : dimensionless correlation coefficient between horizontal
    //                and vertical accelerations (default -0.9 for NED frames)
    //
    // Notes:
    //   • Negative rho_corr is appropriate for NED (z down), because when the
    //     surface moves upward (negative az), horizontal acceleration is forward.
    //   • Positive rho_corr fits ENU (z up).
    //   • The resulting covariance is projected to SPD for numerical stability.
    //   • Also reseeds Pext(OFF_AW, OFF_AW) to keep filter covariance consistent.
    void set_aw_stationary_corr_std(const Vector3& std_aw, T rho_corr = T(-0.65)) {
      // Clamp correlation for numerical safety
      rho_corr = std::max(T(-0.999), std::min(rho_corr, T(0.999))); 

      const T sx = std::max(T(1e-9), std_aw.x());
      const T sy = std::max(T(1e-9), std_aw.y());
      const T sz = std::max(T(1e-9), std_aw.z());

      // Construct correlated covariance (symmetric)
      Matrix3 S;
      S <<
          sx*sx,  T(0),          rho_corr*sx*sz,
          T(0),   sy*sy,         rho_corr*sy*sz,
          rho_corr*sx*sz, rho_corr*sy*sz, sz*sz;

      // Symmetrize & project to SPD for robustness
      S = T(0.5) * (S + S.transpose());
      Eigen::SelfAdjointEigenSolver<Matrix3> es(S);
      if (es.info() == Eigen::Success) {
        auto lam = es.eigenvalues();
        for (int i = 0; i < 3; ++i) {
          if (!(lam(i) > T(0))) lam(i) = T(1e-12);
        }
        Sigma_aw_stat =
            (es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose()).eval();
      } else {
        Sigma_aw_stat = S.diagonal().asDiagonal();  // safe fallback
      }

      // Reseed Pext a_w block with new stationary covariance
      if (Pext.size() > 0) {
        if (!has_cross_cov_a_xy) {
          Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;
        } else {
          // Otherwise, softly merge (blend) to avoid discontinuity
          Pext.template block<3,3>(OFF_AW, OFF_AW) =
              0.8f * Pext.template block<3,3>(OFF_AW, OFF_AW)
            + 0.2f * Sigma_aw_stat;
        }
        // keep global symmetry
        Pext = T(0.5) * (Pext + Pext.transpose());
      }
      has_cross_cov_a_xy = true;
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

    // Toggle exact/structured Qd for the attitude+gyro-bias block (option 3).
    void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }
        
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

    // Original constant matrices
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

    bool has_cross_cov_a_xy = false;

bool use_exact_att_bias_Qd_ = true;

// Closed-form helpers for rotation & integrals (constant ω over [0, t])

// Rodrigues rotation and the integral B(t) = -∫_0^t exp(-[ω]× τ) dτ
EIGEN_STRONG_INLINE void rot_and_B_from_wt_(const Vector3& w, T t, Matrix3& R, Matrix3& B) const {
    const T wnorm = w.norm();
    const Matrix3 W = skew_symmetric_matrix(w);

    if (wnorm < T(1e-7)) {
        // Series (stable as ω→0)
        const T t2 = t*t, t3 = t2*t;
        R = Matrix3::Identity() - W * t + T(0.5) * (W*W) * t2;
        // B = -( t I - 1/2 W t^2 + 1/6 W^2 t^3 )
        B = -( Matrix3::Identity()*t - T(0.5)*W*t2 + (W*W)*(t3/T(6)) );
        return;
    }

    const T theta = wnorm * t;
    const T s = std::sin(theta), c = std::cos(theta);
    const Matrix3 K = W / (wnorm + std::numeric_limits<T>::epsilon()); // [u]×

    // exp(-[ω]× t) = I - sinθ K + (1 - cosθ) K^2
    R = Matrix3::Identity() - s*K + (T(1)-c)*(K*K);

    // B(t) = - ∫_0^t R(τ) dτ = -[ t I - (1 - cosθ)/ω^2 W + (t - sinθ/ω)/ω^2 W^2 ]
    const T invw  = T(1) / wnorm;
    const T invw2 = invw * invw;

    const Matrix3 term1 = Matrix3::Identity() * t;
    const Matrix3 term2 = ((T(1)-c) * invw2) * W;
    const Matrix3 term3 = ((t - s*invw) * invw2) * (W*W);
    B = -( term1 - term2 + term3 );
}

// ∫_0^T B(s) ds  (closed form; used for Q_{θb})
EIGEN_STRONG_INLINE void integral_B_ds_(const Vector3& w, T Tstep, Matrix3& IB) const {
    const T wnorm = w.norm();
    const Matrix3 W = skew_symmetric_matrix(w);

    if (wnorm < T(1e-7)) {
        // ∫ B ≈ -[ 1/2 T^2 I - 1/6 W T^3 + 1/24 W^2 T^4 ]
        const T T2 = Tstep*Tstep, T3 = T2*Tstep, T4 = T3*Tstep;
        IB = -( Matrix3::Identity()*(T(0.5)*T2)
              - W*(T(1.0/6.0)*T3)
              + (W*W)*(T(1.0/24.0)*T4) );
        return;
    }

    const T theta = wnorm * Tstep;
    const T s = std::sin(theta), c = std::cos(theta);
    const T invw  = T(1) / wnorm;
    const T invw2 = invw * invw;
    const T invw4 = invw2 * invw2;

    // IB = ∫_0^T B(s) ds = -[ 1/2 T^2 I - ((T - sinθ/ω)/ω^2) W + ((1/2 T^2) + (cosθ - 1)/ω^2)/ω^2 W^2 ]
    const Matrix3 termI = Matrix3::Identity() * (T(0.5) * Tstep*Tstep);
    const Matrix3 termW = ((Tstep - s*invw) * invw2) * W;
    const Matrix3 termW2 = ( (T(0.5)*Tstep*Tstep) + ((c - T(1)) * invw2) ) * invw2 * (W*W);

    IB = -( termI - termW + termW2 );
}

// Simpson’s rule for ∫_0^T R(s) Q R(s)^T ds (fast, excellent for anisotropic Q)
EIGEN_STRONG_INLINE Matrix3 simpson_R_Q_RT_(const Vector3& w, T Tstep, const Matrix3& Q) const {
    Matrix3 R0, Btmp, Rm, R1;
    rot_and_B_from_wt_(w, T(0),   R0, Btmp);
    rot_and_B_from_wt_(w, T(0.5)*Tstep, Rm, Btmp);
    rot_and_B_from_wt_(w, Tstep, R1, Btmp);

    const Matrix3 f0 = R0 * Q * R0.transpose(); // = Q
    const Matrix3 f1 = Rm * Q * Rm.transpose();
    const Matrix3 f2 = R1 * Q * R1.transpose();
    return (Tstep / T(6)) * (f0 + T(4)*f1 + f2);
}

// Simpson’s rule for ∫_0^T B(s) Q B(s)^T ds
EIGEN_STRONG_INLINE Matrix3 simpson_B_Q_BT_(const Vector3& w, T Tstep, const Matrix3& Q) const {
    Matrix3 Rtmp, B0, Bm, B1;
    rot_and_B_from_wt_(w, T(0),   Rtmp, B0);           // B(0) = 0
    rot_and_B_from_wt_(w, T(0.5)*Tstep, Rtmp, Bm);
    rot_and_B_from_wt_(w, Tstep, Rtmp, B1);

    const Matrix3 g0 = B0 * Q * B0.transpose(); // = 0
    const Matrix3 g1 = Bm * Q * Bm.transpose();
    const Matrix3 g2 = B1 * Q * B1.transpose();
    return (Tstep / T(6)) * (g0 + T(4)*g1 + g2);
}

EIGEN_STRONG_INLINE bool is_isotropic3_(const Matrix3& S, T tol = T(1e-9)) const {
    const T a = S(0,0), b = S(1,1), c = S(2,2);

    // Sum of absolute values of off-diagonal entries (1-norm of off-diagonal part)
    Matrix3 Off = S;
    Off.diagonal().setZero();
    const T off = Off.cwiseAbs().sum();

    const T mean = (a + b + c) / T(3);
    return (std::abs(a-mean) + std::abs(b-mean) + std::abs(c-mean) + off)
           <= tol * (T(1) + std::abs(mean));
}
        
    // convenience getters
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }               // world→body
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }   // body→world

    // Helpers and original methods kept
    void measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const;
    Vector3 accelerometer_measurement_func(T tempC) const;
    Vector3 magnetometer_measurement_func() const;

    static MatrixBaseN initialize_Q(Vector3 sigma_g, T b0);

    // Quaternion & small-angle helpers (kept)
    void applyQuaternionCorrectionFromErrorState(); // apply correction to qref using xext(0..2)

    static void PhiAxis4x1_analytic(T tau, T h, Eigen::Matrix<T,4,4>& Phi_axis);
    static void QdAxis4x1_analytic(T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis);

    // Reusable helpers for all 3×3 measurement updates
    
    // Factor S with LDLT and a diagonal safety boost if needed.
    // The `noise_scale` argument should be something like R.norm() or S.norm() from the branch.
    EIGEN_STRONG_INLINE bool safe_ldlt3_(Matrix3& S, Eigen::LDLT<Matrix3>& ldlt, T noise_scale) const {
        ldlt.compute(S);
        if (ldlt.info() == Eigen::Success) return true;
    
        // Minimum positive bump
        const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * (noise_scale + T(1)));
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
  MatrixBaseN Pbase;
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
    v2ref = R_bw() * mag_body;  // body to world, keep µT
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
    // Attitude propagation
    Vector3 gyro_bias;
    if constexpr (with_gyro_bias) {
        gyro_bias = xext.template segment<3>(3);
    } else {
        gyro_bias = Vector3::Zero();
    }
    last_gyr_bias_corrected = gyr - gyro_bias;

    // Δθ = ω·Ts → right-multiplicative quaternion increment
    Eigen::Quaternion<T> dq = quat_from_delta_theta((last_gyr_bias_corrected * Ts).eval());
    qref = qref * dq;
    qref.normalize();

    // Attitude block F_AA, Q_AA
    Matrix3 I = Matrix3::Identity();
    const Vector3 w = last_gyr_bias_corrected;
    const T omega = w.norm();
    const T theta = omega * Ts;

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

MatrixBaseN Q_AA; Q_AA.setZero();

if (!use_exact_att_bias_Qd_) {
    // Original fast path
    Q_AA = Qbase * Ts;
} else {
    // Structured/closed-form path for [δθ, b_g] with constant ω over the step
    const Matrix3 Qg  = Qbase.template topLeftCorner<3,3>();
    Matrix3 Qbg; Qbg.setZero();
    if constexpr (with_gyro_bias) {
        Qbg = Qbase.template bottomRightCorner<3,3>();
    }

    // ∫ R(s) Qg R(s)^T ds  (exact if isotropic; Simpson for anisotropic)
    Matrix3 I_R;
    if (is_isotropic3_(Qg)) {
        I_R = Matrix3::Identity() * (Qg(0,0) * Ts);     // exact
    } else {
        I_R = simpson_R_Q_RT_(w, Ts, Qg);               // very accurate, still cheap
    }

    // ∫ B(s) Qbg B(s)^T ds  (Simpson)
    Matrix3 I_BB = Matrix3::Zero();
    if constexpr (with_gyro_bias) {
        I_BB = simpson_B_Q_BT_(w, Ts, Qbg);
    }

    // Q_{θθ} = I_R + I_BB
    const Matrix3 Qtt = I_R + I_BB;

    // Q_{bb} = Qbg * Ts
    Matrix3 Qbb = Matrix3::Zero();
    if constexpr (with_gyro_bias) {
        Qbb = Qbg * Ts;
    }

    // Q_{θb} = (∫ B(s) ds) * Qbg    (closed form)
    Matrix3 Qtb = Matrix3::Zero();
    if constexpr (with_gyro_bias) {
        Matrix3 IB;
        integral_B_ds_(w, Ts, IB);
        Qtb = IB * Qbg;
    }

    // Pack into BASE_N × BASE_N
    Q_AA.template topLeftCorner<3,3>() = Qtt;
    if constexpr (with_gyro_bias) {
        Q_AA.template topRightCorner<3,3>()    = Qtb;
        Q_AA.template bottomLeftCorner<3,3>()  = Qtb.transpose();
        Q_AA.template bottomRightCorner<3,3>() = Qbb;
    }

    // Hygiene
    Q_AA = T(0.5) * (Q_AA + Q_AA.transpose());
    if constexpr (with_gyro_bias) {
        project_psd<T,6>(Q_AA, T(1e-16));
    } else {
        project_psd<T,3>(Q_AA, T(1e-16));
    }
}

    // Linear subsystem [v,p,S,a_w] (12×12)
    using Mat12 = Eigen::Matrix<T,12,12>;
    Mat12 F_LL; F_LL.setZero();
    Mat12 Q_LL; Q_LL.setZero();

    // Build F_LL per axis
    for (int axis = 0; axis < 3; ++axis) {
        const T tau = std::max(T(1e-6), tau_aw);

        Eigen::Matrix<T,4,4> Phi_axis;
        PhiAxis4x1_analytic(tau, Ts, Phi_axis);

        const int idx[4] = {0,3,6,9}; // [v,p,S,a] offsets
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                F_LL(idx[i] + axis, idx[j] + axis) = Phi_axis(i,j);
    }

    // Build Q_LL
    if (has_cross_cov_a_xy) {


    // Correlated vector-OU with shared tau: Q_LL = (Σ ⊗ Qaxis_unit)
    // written directly in group-first order: [v(3), p(3), S(3), a(3)].
    Eigen::Matrix<T,4,4> Qaxis_unit;
    QdAxis4x1_analytic(tau_aw, Ts, T(1), Qaxis_unit);
    Qaxis_unit = T(0.5) * (Qaxis_unit + Qaxis_unit.transpose()); // hygiene

    // SPD, symmetric copy of Σ_aw (3x3)
    const Matrix3 Sig = T(0.5) * (Sigma_aw_stat + Sigma_aw_stat.transpose());

    Q_LL.setZero();
    // group offsets in your interleaved state order
    const int goff[4] = {0, 3, 6, 9}; // v, p, S, a

    // Blockwise assembly: each 3x3 block is Sig scaled by the 4x4 scalar
    for (int g = 0; g < 4; ++g) {
        for (int h = 0; h < 4; ++h) {
            Q_LL.template block<3,3>(goff[g], goff[h]).noalias()
                = Sig * Qaxis_unit(g,h);
        }
    }

    // Symmetry + PSD cleanup
    Q_LL = T(0.5) * (Q_LL + Q_LL.transpose());
    project_psd<T,12>(Q_LL, T(1e-16));



    } else {
        // Independent axes (no cross-correlation) — per-axis Qd on the diagonal
        const int idx[4] = {0,3,6,9};
        for (int axis = 0; axis < 3; ++axis) {
            const T tau    = std::max(T(1e-6), tau_aw);
            const T sigma2 = Sigma_aw_stat(axis, axis);
            Eigen::Matrix<T,4,4> Qd_axis;
            QdAxis4x1_analytic(tau, Ts, sigma2, Qd_axis);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    Q_LL(idx[i] + axis, idx[j] + axis) = Qd_axis(i,j);
        }
        Q_LL = T(0.5) * (Q_LL + Q_LL.transpose());
    }

    // Mean propagation for [v,p,S,a_w]
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

    // Covariance propagation (blockwise)
    constexpr int NA = BASE_N;
    constexpr int NL = 12;

    Eigen::Matrix<T,NA,NA> P_AA = Pext.template block<NA,NA>(0,0);
    Eigen::Matrix<T,NL,NL> P_LL = Pext.template block<NL,NL>(OFF_V,OFF_V);
    Eigen::Matrix<T,NA,NL> P_AL = Pext.template block<NA,NL>(0,OFF_V);

    // AA
    Eigen::Matrix<T,NA,NA> tmpAA = F_AA * P_AA;
    P_AA = tmpAA * F_AA.transpose() + Q_AA;

    // LL
    Eigen::Matrix<T,NL,NL> tmpLL = F_LL * P_LL;
    P_LL = tmpLL * F_LL.transpose() + Q_LL;

    // AL
    Eigen::Matrix<T,NA,NL> tmpAL = F_AA * P_AL;
    P_AL = tmpAL * F_LL.transpose();

    // Write back
    Pext.template block<NA,NA>(0,0)         = P_AA;
    Pext.template block<NL,NL>(OFF_V,OFF_V) = P_LL;
    Pext.template block<NA,NL>(0,OFF_V)     = P_AL;
    Pext.template block<NL,NA>(OFF_V,0)     = P_AL.transpose();

    // Optional accel bias RW and cross terms (F_BB = I)
    if constexpr (with_accel_bias) {
        constexpr int NB = 3;

        auto P_BB = Pext.template block<NB,NB>(OFF_BA,OFF_BA);
        P_BB.noalias() += Q_bacc_ * Ts; // ensure Q_bacc_ units match this convention
        Pext.template block<NB,NB>(OFF_BA,OFF_BA) = P_BB;

        Eigen::Matrix<T,NA,NB> tmpAB = F_AA * Pext.template block<NA,NB>(0,OFF_BA);
        Eigen::Matrix<T,NL,NB> tmpLB = F_LL * Pext.template block<NL,NB>(OFF_V,OFF_BA);
        Pext.template block<NA,NB>(0,OFF_BA) = tmpAB;
        Pext.template block<NL,NB>(OFF_V,OFF_BA) = tmpLB;
        Pext.template block<NB,NA>(OFF_BA,0) = tmpAB.transpose();
        Pext.template block<NB,NL>(OFF_BA,OFF_V) = tmpLB.transpose();
    }

    // Symmetry hygiene
    Pext = T(0.5) * (Pext + Pext.transpose());

    // Integral pseudo-measurement drift correction
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
}

template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::measurement_update_acc_only(
    Vector3 const& acc_meas, T tempC)
{
    // Gate accel magnitude
    const T g_meas = acc_meas.norm();
    if (std::abs(g_meas - gravity_magnitude_) > T(2.0) * gravity_magnitude_) return;

    // Physical accelerometer measurement model
    // f_b = R_wb * (a_w - g) + b_a + noise
    const Vector3 f_pred = accelerometer_measurement_func(tempC);
    const Vector3 f_meas = acc_meas;
    const Vector3 r = f_meas - f_pred; // innovation in true units (m/s²)

    // Proper Jacobians from linearization of f_b
    const Matrix3 J_att = -skew_symmetric_matrix(f_pred); // ∂f/∂θ = -[f_b]_×
    const Matrix3 J_aw  =  R_wb();                         // ∂f/∂a_w = R_wb
    Matrix3 J_ba;
    if constexpr (with_accel_bias) J_ba.setIdentity();      // ∂f/∂b_a = I

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

    // Apply quaternion correction
    applyQuaternionCorrectionFromErrorState();
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
    Phi_axis(3,3) = std::max(T(1e-7), std::min(P.alpha, T(1)));
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
//   - For small x = h/tau < 3e-3, switch to Maclaurin series expansions to
//     avoid catastrophic cancellation when subtracting nearly-equal terms.
//
template<typename T, bool with_gyro_bias, bool with_accel_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias>::QdAxis4x1_analytic(
    T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis)
{
    const T inv_tau = T(1) / std::max(tau, T(1e-7));
    const T x = h * inv_tau;
    const T q_c = (T(2) * sigma2) * inv_tau;

    Eigen::Matrix<T,4,4> K; K.setZero();

    if (x < T(3e-3)) {
        // Small-x Maclaurin with FMAs to reduce rounding
        const T h2=h*h, h3=h2*h, h4=h3*h, h5=h4*h, h6=h5*h, h7=h6*h, h8=h7*h;

        const T K_vv = std::fma(-h4/(T(4)*tau), T(1), h3/T(3));
        const T K_pp = std::fma(-h6/(T(30)*tau), T(1), h5/T(20));
        const T K_SS = std::fma(-h8/(T(960)*tau), T(1), h7/T(840));

        const T K_aa = std::fma(std::fma(std::fma(-h4, T(1)/(T(3)*tau*tau*tau), T(2)*h3/(T(3)*tau*tau)), -T(1), h),
                                 T(1), -h2/tau);
        const T K_va = std::fma(std::fma(h4, T(1)/(T(8)*tau*tau), -h3/(T(3)*tau)), T(1), h2/T(2));
        const T K_pa = std::fma(-h4/(T(8)*tau), T(1), h3/T(6));
        const T K_pv = std::fma(-h5/(T(15)*tau), T(1), h4/T(12));
        const T K_Sa = std::fma(-h5/(T(30)*tau), T(1), h4/T(24));
        const T K_Sv = std::fma(-h6/(T(72)*tau), T(1), h5/T(60));
        const T K_Sp = std::fma(-h7/(T(420)*tau), T(1), h6/T(360));

        K(0,0)=K_vv; K(0,1)=K_pv; K(0,2)=K_Sv; K(0,3)=K_va;
        K(1,0)=K_pv; K(1,1)=K_pp; K(1,2)=K_Sp; K(1,3)=K_pa;
        K(2,0)=K_Sv; K(2,1)=K_Sp; K(2,2)=K_SS; K(2,3)=K_Sa;
        K(3,0)=K_va; K(3,1)=K_pa; K(3,2)=K_Sa; K(3,3)=K_aa;
    } else {
        // General-x branch with FMA-safe combos
        const auto P = make_prims<T>(h, tau); // {x, alpha, em1, alpha2, em1_2}
        const T tau2 = tau*tau, tau3 = tau2*tau, tau4=tau3*tau, tau5=tau4*tau, tau6=tau5*tau;

        const T A0 = -tau * P.em1;  // τ(1-α)
        const auto coeffs = safe_phi_A_coeffs<T>(h, tau); // FMA-safe A1/A2/phi_pa/phi_Sa
        const T A1 = coeffs.A1;
        const T A2 = coeffs.A2;
        const T B0 = -(tau * T(0.5)) * P.em1_2;

        const T C0=h;
        const T C1=T(0.5)*h*h;
        const T C2=(h*h*h)/T(3);
        const T C3=(h*h*h*h)/T(4);
        const T C4=(h*h*h*h*h)/T(5);

        // FMA-protected differences
        const T I1mA0   = std::fma(-T(1), A0, C0);
        const T Ix1mA1  = std::fma(-T(1), A1, C1);
        const T Ix21mA2 = std::fma(-T(1), A2, C2);

        const T K_aa = B0;

        const T K_va = tau * std::fma(T(1), A0, -B0);

        const T K_vv = tau2 * std::fma(T(1),
                            std::fma(T(1), C0, std::fma(-T(2), A0, B0)),
                            T(0));

        const T K_pa = std::fma(tau, A1, std::fma(-tau2, A0, tau2*B0));

        const T K_pv = std::fma(tau2, Ix1mA1,
                        std::fma(-tau3, I1mA0, tau3*(A0 - B0)));

        const T K_pp = std::fma(tau2, C2,
                        std::fma(-T(2)*tau3, C1,
                        std::fma(T(2)*tau3, A1,
                        std::fma(tau4, C0,
                        std::fma(-T(2)*tau4, A0, tau4*B0)))));

        const T K_Sa = std::fma(T(0.5)*tau, A2,
                        std::fma(-tau2, A1,
                        std::fma(tau3, A0, -tau3*B0)));

        const T K_Sv = std::fma(T(0.5)*tau2, Ix21mA2,
                        std::fma(-tau3, Ix1mA1,
                        std::fma(tau4, I1mA0, -tau4*(A0 - B0))));

        const T K_Sp = std::fma(T(0.5)*tau2, C3,
                        std::fma(-T(1.5)*tau3, C2,
                        std::fma(T(2)*tau4, C1,
                        std::fma(-tau5, C0,
                        std::fma(T(0.5)*tau3, A2,
                        std::fma(-T(2)*tau4, A1,
                        std::fma(T(2)*tau5, A0, -tau5*B0)))))));

        const T K_SS = std::fma(T(0.25)*tau2, C4,
                        std::fma(-tau3, C3,
                        std::fma(T(2)*tau4, C2,
                        std::fma(-T(2)*tau5, C1,
                        std::fma(tau6, C0,
                        std::fma(-tau4, A2,
                        std::fma(T(2)*tau5, A1,
                        std::fma(-T(2)*tau6, A0, tau6*B0))))))));

        K(0,0)=K_vv; K(0,1)=K_pv; K(0,2)=K_Sv; K(0,3)=K_va;
        K(1,0)=K_pv; K(1,1)=K_pp; K(1,2)=K_Sp; K(1,3)=K_pa;
        K(2,0)=K_Sv; K(2,1)=K_Sp; K(2,2)=K_SS; K(2,3)=K_Sa;
        K(3,0)=K_va; K(3,1)=K_pa; K(3,2)=K_Sa; K(3,3)=K_aa;
    }

    // Build final Qd, symmetrize, scrub, then project *Qd* to PSD
    Qd_axis = (q_c * (T(0.5) * (K + K.transpose()))).eval();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T& v = Qd_axis(i,j);
            if (!std::isfinite(v)) {
                // Preserve SPD: put a tiny positive value on the diagonal only
                v = (i == j) ? T(1e-18) : T(0);
            }
        }
    }

    project_psd<T,4>(Qd_axis, T(1e-16));

    for (int i=0; i<4; ++i) if (!(Qd_axis(i,i) > T(0))) Qd_axis(i,i) = T(1e-17);
    Qd_axis = T(0.5) * (Qd_axis + Qd_axis.transpose());
}
