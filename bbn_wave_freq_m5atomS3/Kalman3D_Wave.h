#pragma once

/*
  Copyright (c) 2025 Mikhail Grushinskiy

  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  Full-matrix Kalman that with linear navigation states:
     v (3)   : velocity in world frame
     p (3)   : displacement/position in world frame
     S (3)   : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction
     a_w (3) : latent world-frame inertial acceleration (NED)     

  - The quaternion MEKF logic (time_update, measurement_update, partial updates, quaternion correction)
    is preserved where possible.
  - The extended linear states are driven by a latent OU world-acceleration a_w
    (accelerometer input is used only in the measurement update).
  - A full extended covariance (Pext) and transition Jacobian Fext are constructed; the top-left corner
    contains the MEKF's P/Q blocks (attitude error + optional gyro bias).
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
#include <algorithm>
#include <stdexcept>

using Eigen::Matrix;

template<typename T>
inline T safe_inv_tau(T tau) {
    // Prevent division by ~0 while preserving sign
    return T(1) / ((std::abs(tau) >= T(1e-8)) ? tau : std::copysign(T(1e-8), tau));
}

template<typename T>
struct OUPrims {
    T x;        // h/tau
    T alpha;    // e^{-x}
    T em1;      // expm1(-x) = e^{-x} - 1  (negative)
};

template<typename T>
inline OUPrims<T> make_prims(T h, T tau) {
    const T inv_tau = safe_inv_tau(tau);
    const T x = h * inv_tau;
    const T alpha  = std::exp(-x);
    const T em1    = std::expm1(-x);    // high-accuracy for small x
    return {x, alpha, em1};
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
// Provides phi_pa, phi_Sa with series fallback for small x = h/tau.
template<typename T>
struct OUDiscreteCoeffs {
    T phi_pa; // coefficient for position vs. accel
    T phi_Sa; // coefficient for S vs. accel
};

template<typename T>
inline OUDiscreteCoeffs<T> safe_phi_A_coeffs(T h, T tau) {
    OUDiscreteCoeffs<T> c;
    const T inv_tau = safe_inv_tau(tau);
    const T x = h * inv_tau;
    const T tau2 = tau*tau;
    const T tau3 = tau2*tau;

    if (std::abs(x) < T(1e-2)) {
        // Maclaurin expansions
        const T x2 = x*x;
        const T x3 = x2*x;
        const T x4 = x3*x;
        const T x5 = x4*x;

        // phi_pa ≈ τ² (x²/2 - x³/6 + x⁴/24)
        c.phi_pa = tau2 * (T(0.5)*x2 - T(1.0/6.0)*x3 + T(1.0/24.0)*x4);

        // phi_Sa ≈ τ³ (x³/6 - x⁴/24 + x⁵/120)
        c.phi_Sa = tau3 * (T(1.0/6.0)*x3 - T(1.0/24.0)*x4 + T(1.0/120.0)*x5);  
    } else {
        // General closed-form branch
        const T alpha  = std::exp(-x);
        const T em1    = std::expm1(-x);    
        // reuse for stability
        const T phi_pa = tau2 * (x + em1);
        const T phi_Sa = tau3 * (T(0.5)*x*x - x - em1);

        c.phi_pa = phi_pa;
        c.phi_Sa = phi_Sa;
    }
    return c;
}

// Helper: project a symmetric NxN to PSD
template<typename T, int N>
static inline void project_psd(Eigen::Matrix<T,N,N>& S, T eps = T(1e-12)) {
    // Always symmetrize first (we assume S is "almost" symmetric)
    S = T(0.5) * (S + S.transpose());
    // Scrub NaNs / infinities, keep symmetry
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!std::isfinite(S(i,j))) {
                S(i,j) = (i == j) ? eps : T(0);
            }
        }
    }
    if constexpr (N <= 4) {
        // Small matrices: use exact eigen projection (stack cost is tiny here).
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N>> es(S);
        if (es.info() != Eigen::Success) {
            // Fallback: light diagonal bump & re-symmetrize
            S.diagonal().array() += eps;
            S = T(0.5) * (S + S.transpose());
            return;
        }
        Eigen::Matrix<T,N,1> lam = es.eigenvalues();
        for (int i = 0; i < N; ++i) {
            if (!(lam(i) > T(0))) lam(i) = eps;  // clamp negatives / NaNs
        }
        S = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
        S = T(0.5) * (S + S.transpose()); // clean float noise
    } else {
        // Larger matrices (6×6, 12×12, ...):
        // enforce *strict diagonal dominance* row-by-row.
        // For a symmetric matrix, strictly diagonally dominant with positive
        // diagonal ⇒ SPD. This is cheap, O(N²), and uses almost no stack.

        for (int i = 0; i < N; ++i) {
            T row_sum = T(0);
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                row_sum += std::abs(S(i,j));
            }
            const T min_diag = row_sum + eps;
            if (!(S(i,i) > min_diag)) {
                S(i,i) = min_diag;
            }
        }
        // Final clean symmetrization
        S = T(0.5) * (S + S.transpose());
    }
}

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true, bool with_mag_bias = true>
class Kalman3D_Wave {

    // Base (att_err + optional gyro bias)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended added states: v(3), p(3), S(3), a_w(3) [+ b_acc(3)] [+ b_mag(3)]
    static constexpr int EXT_ADD = 12
        + (with_accel_bias ? 3 : 0)
        + (with_mag_bias   ? 3 : 0);
    
    static constexpr int NX = BASE_N + EXT_ADD;
    
    // Offsets (always defined)
    static constexpr int OFF_V   = BASE_N + 0;
    static constexpr int OFF_P   = BASE_N + 3;
    static constexpr int OFF_S   = BASE_N + 6;
    static constexpr int OFF_AW  = BASE_N + 9;
    
    static constexpr int OFF_BA  = with_accel_bias ? (BASE_N + 12) : -1;
    
    // mag bias comes after accel bias if present, otherwise after a_w
    static constexpr int OFF_BM  = with_mag_bias
        ? (BASE_N + 12 + (with_accel_bias ? 3 : 0))
        : -1;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, BASE_N, BASE_N> MatrixBaseN;
    typedef Matrix<T, NX, NX> MatrixNX;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, 4, 4> Matrix4;

    // Fixed-size helpers for internal scratch
    typedef Matrix<T, 12, 12> Matrix12;
    typedef Matrix<T, 12,  1> Vector12;
    typedef Matrix<T, BASE_N, 12> MatrixBaseN12;
    typedef Matrix<T, NX,  3> MatrixNX3;

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
    // Pass NED units. For yaw-only, pass the horizontal field (z = 0).
    void set_mag_world_ref(const Vector3& B_world) {
        v2ref = B_world;    // keep µT magnitude
    }

    [[nodiscard]] Vector3 get_mag_bias() const {
        if constexpr (with_mag_bias) return xext.template segment<3>(OFF_BM);
        else return Vector3::Zero();
    }
    
    void set_initial_mag_bias_std(T s_uT) {
        if constexpr (with_mag_bias) {
            sigma_bmag0_ = std::max(T(0), s_uT);
            Pext.template block<3,3>(OFF_BM, OFF_BM) =
                Matrix3::Identity() * sigma_bmag0_ * sigma_bmag0_;
        }
    }
    
    void set_initial_mag_bias(const Vector3& b_uT) {
        if constexpr (with_mag_bias) xext.template segment<3>(OFF_BM) = b_uT;
    }
    
    void set_Q_bmag_rw(const Vector3& rw_std_uT_per_sqrt_s) {
        if constexpr (with_mag_bias)
            Q_bmag_ = rw_std_uT_per_sqrt_s.array().square().matrix().asDiagonal();
    }

    void time_update(Vector3 const& gyr, T Ts);

    // Measurement updates preserved (operate on extended state internally)
    void measurement_update_acc_only(Vector3 const& acc, T tempC = tempC_ref);
    void measurement_update_mag_only(Vector3 const& mag);

    // Extended-only API:
    // Apply zero pseudo-measurement on S (integral drift correction)
    void applyIntegralZeroPseudoMeas();

    // 3D pseudo-measurement on position p (world, NED):
    //   p_meas ≈ p (meters), with per-axis std sigma_meas.
    // This does a full 3x3 Joseph update on the position block and its cross-covariances.
    void measurement_update_position_pseudo(const Vector3& p_meas, const Vector3& sigma_meas);

    // Accessors
    [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    [[nodiscard]] MatrixBaseN covariance_base() const { return Pext.topLeftCorner(BASE_N, BASE_N); } // top-left block
    [[nodiscard]] MatrixNX covariance_full() const { return Pext; }     // full extended covariance

    // Boat attitude in the *physical* heeled frame B → W (NED).
    [[nodiscard]] Eigen::Quaternion<T> quaternion_boat() const {
        // Internal quaternion() = B'→W (un-heeled frame to world)
        const Eigen::Quaternion<T> q_WBprime = quaternion();

        // q_B'B = rotation that maps physical body B to un-heeled frame B'
        // Roll of -wind_heel about X: B'←B ("un-heel").
        const T half = -wind_heel_rad_ * T(0.5);
        const T c = std::cos(half);
        const T s = std::sin(half);
        const Eigen::Quaternion<T> q_BprimeB(c, s, 0, 0); // (w, x, y, z)

        // Composition: B→W = (B'→W) ∘ (B→B') = q_WB' * q_B'B
        return q_WBprime * q_BprimeB;
    }

    [[nodiscard]] Vector3 gyroscope_bias() const {
        if constexpr (with_gyro_bias) {
            return xext.template segment<3>(3);
        } else {
            return Vector3::Zero();
        }
    }

    [[nodiscard]] Vector3 get_acc_bias() const {
        if constexpr (with_accel_bias) {
            return xext.template segment<3>(OFF_BA);
        } else {
            return Vector3::Zero();
        }
    }

    // Enable/disable propagation & drift-correction of linear states [v,p,S,a_w].
    // When disabled, the filter behaves like a pure attitude/bias MEKF:
    //  • [v,p,S,a_w] are frozen (no process, no S pseudo-measurements),
    //  • attitude & biases still propagate and accept accel/mag updates.
  void set_linear_block_enabled(bool on) {
      if (linear_block_enabled_ && !on) {
          // Just disabled: decouple base (A) from linear (L)
          zero_AL_cross_cov_once_();

          // Optional: also decouple accel/mag biases from the linear block
          // if you consider those part of the "A" subsystem when linear is off.
          if constexpr (with_accel_bias) {
              Pext.template block<12,3>(OFF_V, OFF_BA).setZero();
              Pext.template block<3,12>(OFF_BA, OFF_V).setZero();
          }
          if constexpr (with_mag_bias) {
              Pext.template block<12,3>(OFF_V, OFF_BM).setZero();
              Pext.template block<3,12>(OFF_BM, OFF_V).setZero();
          }

          // keep cadence sane
          pseudo_update_counter_ = 0;
      }

      linear_block_enabled_ = on;
  }
    bool linear_block_enabled() const      { return linear_block_enabled_; }

    // Velocity in world (NED)
    [[nodiscard]] Vector3 get_velocity() const {
        // velocity state at offset BASE_N
        return xext.template segment<3>(BASE_N);
    }

    // Position in world (NED)
    [[nodiscard]] Vector3 get_position() const {
        // position state at offset BASE_N+3
        return xext.template segment<3>(BASE_N + 3);
    }

    // Integral displacement in world (NED)
    [[nodiscard]] Vector3 get_integral_displacement() const {
        // integral of displacement state at offset BASE_N+6
        return xext.template segment<3>(BASE_N + 6);
    }

    // Latent OU world-acceleration a_w (world, NED)
    [[nodiscard]] Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }

    // Tuning setters
    void set_aw_time_constant(T tau_seconds) { tau_aw = std::max(T(1e-3), tau_seconds); }

    // OU stationary std [m/s²] for a_w (per axis)
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat = std_aw.array().square().matrix().asDiagonal();
        has_cross_cov_a_xy = false;

        // keep P consistent with the new stationary prior
        Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;
        symmetrize_Pext_();
    }

    // Accept a full 3×3 SPD stationary covariance for a_w.
    void set_aw_stationary_cov_full(const Matrix3& Sigma);

    // Set OU stationary covariance for world-acceleration a_w,
    // using per-axis standard deviations and an optional cross-axis correlation.
    //
    // Arguments:
    //   std_aw     : per-axis standard deviations [m/s²] for (x, y, z)
    //   rho_corr   : dimensionless correlation coefficient between horizontal
    //                and vertical accelerations
    //
    // Notes:
    //   • Negative rho_corr is appropriate for NED (z down), because when the
    //     surface moves upward (negative az), horizontal acceleration is forward.
    //   • Positive rho_corr fits ENU (z up).
    //   • The resulting covariance is projected to SPD for numerical stability.
    //   • Also reseeds Pext(OFF_AW, OFF_AW) to keep filter covariance consistent.
    void set_aw_stationary_corr_std(const Vector3& std_aw, T rho_xz_corr = T(-0.65), T rho_yz_corr = T(-0.65)) {
        // Clamp correlation for numerical safety
        rho_xz_corr = std::max(T(-0.999), std::min(rho_xz_corr, T(0.999))); 
        rho_yz_corr = std::max(T(-0.999), std::min(rho_yz_corr, T(0.999))); 
  
        const T sx = std::max(T(1e-9), std_aw.x());
        const T sy = std::max(T(1e-9), std_aw.y());
        const T sz = std::max(T(1e-9), std_aw.z());
  
        // Construct correlated covariance (symmetric)
        Matrix3 S;
        S <<
            sx*sx,  T(0),             rho_xz_corr*sx*sz,
            T(0),   sy*sy,            rho_yz_corr*sy*sz,
            rho_xz_corr*sx*sz, rho_yz_corr*sy*sz, sz*sz;
  
        // Symmetrize & project to SPD for robustness
        S = T(0.5) * (S + S.transpose());
        project_psd<T,3>(S, T(1e-12));
        Sigma_aw_stat = S;

        // Reseed Pext a_w block with new stationary covariance
        if (!has_cross_cov_a_xy) {
            Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;
        } else {
            // Otherwise, softly merge (blend) to avoid discontinuity
            Pext.template block<3,3>(OFF_AW, OFF_AW) =
                T(0.8) * Pext.template block<3,3>(OFF_AW, OFF_AW)
              + T(0.2) * Sigma_aw_stat;
        }
        // keep global symmetry
        symmetrize_Pext_();
        has_cross_cov_a_xy = true;
    }
  
    // Covariances for ∫p dt pseudo-measurement
    void set_RS_noise(const Vector3& sigma_S) {
        R_S = sigma_S.array().square().matrix().asDiagonal();
        R_S = T(0.5) * (R_S + R_S.transpose());
    }

    void set_RS_noise_matrix(const Matrix3& R) {
        Matrix3 S = T(0.5) * (R + R.transpose());         // symmetrize
        project_psd<T,3>(S, T(1e-8));
        R_S = S;
    }
        
    // Accelerometer measurement noise (std in m/s² per axis)
    void set_Racc(const Vector3& sigma_acc) {
        Racc = sigma_acc.array().square().matrix().asDiagonal();
    }

    // Magnetometer measurement noise (std per axis, μT or unitless)
    void set_Rmag(const Vector3& sigma_mag) {
        Rmag = sigma_mag.array().square().matrix().asDiagonal();
    }

    void set_initial_linear_uncertainty(T sigma_v0, T sigma_p0, T sigma_S0) {
        Pext.template block<3,3>(OFF_V, OFF_V) = Matrix3::Identity() * (sigma_v0 * sigma_v0);   // v (3)
        Pext.template block<3,3>(OFF_P, OFF_P) = Matrix3::Identity() * (sigma_p0 * sigma_p0);   // p (3)
        Pext.template block<3,3>(OFF_S, OFF_S) = Matrix3::Identity() * (sigma_S0 * sigma_S0);   // S (3)
    }

    void set_initial_acc_bias_std(T s) {
        if constexpr (with_accel_bias) {
            sigma_bacc0_ = std::max(T(0), s);
            Pext.template block<3,3>(OFF_BA, OFF_BA) = Matrix3::Identity() * sigma_bacc0_ * sigma_bacc0_;
        }
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
    // Model: b_a(tempC) = b_a0 + k_a * (tempC - tempC_ref)
    void set_accel_bias_temp_coeff(const Vector3& ka_per_degC) { k_a_ = ka_per_degC; }

    // Toggle exact/structured Qd for the attitude+gyro-bias block.
    void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }

    // Initialize full extended state from "truth"
    void initialize_from_truth(const Vector3 &p_ned, const Vector3 &v_ned,
                               const Eigen::Quaternion<T> &q_bw, const Vector3 &a_w_ned);
              
    // IMU lever-arm API
    // r_b: IMU position w.r.t. CoG in the *physical* BODY frame B [m].
    // Internally we de-heel this into B' each step when applying lever-arm kinematics.
    void set_imu_lever_arm_body(const Vector3& r_b) {
        r_imu_wrt_cog_body_phys_ = r_b;
        use_imu_lever_arm_ = (r_b.squaredNorm() > T(0));
    }
    void clear_imu_lever_arm() {
        r_imu_wrt_cog_body_phys_.setZero();
        use_imu_lever_arm_ = false;
    }

    void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

    // Set / update steady wind heel (roll about BODY X, rad).
    // Call this periodically (e.g. when your wind model changes) *before*
    // calling time_update/measurement_update_* for the next step.
    void update_wind_heel(T heel_rad) {
        wind_heel_rad_ = heel_rad;
        update_unheel_trig_();
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

    // MEKF internals
    Eigen::Quaternion<T> qref;
    Vector3 v2ref = Vector3::UnitX();

    // Extended full state xext and Pext (NX x NX)
    Matrix<T, NX, 1> xext; // [ δθ(3), (gyro bias 3 optional), v(3), p(3), S(3), a_w(3), (accel bias 3 optional) ]          
    MatrixNX Pext;

    // Last gyro
    Vector3 last_gyr_bias_corrected{};

    T sigma_bacc0_ = T(0.1); // initial accel bias std
    Matrix3 Q_bacc_ = Matrix3::Identity() * T(1e-8);

    // Accelerometer bias temperature coefficient (per-axis), units: m/s^2 per °C.
    // Default here reflects BMI270 typical accel drift (~0.003 m/s^2/°C).
    Vector3 k_a_ = Vector3::Constant(T(0.003));

    T sigma_bmag0_ = T(10.0);                 // µT (start loose so it can learn)
    Matrix3 Q_bmag_ = Matrix3::Identity() * T(1e-6); // (µT^2)/s  (tune)
              
    // Constant matrices
    Matrix3 Rmag;
    MatrixBaseN Qbase; // Q for attitude & bias

    Matrix3 Racc; // Accelerometer noise (diagonal) stored as Matrix3
    Matrix3 R_S;  // Triple integration measurement noise

    // World-acceleration OU process a_w dynamics parameters
    T tau_aw = T(2.3);            // correlation time [s], tune 1–5 s for sea states
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(2.4*2.4); // stationary variance diag [ (m/s^2)^2 ]

    int pseudo_update_counter_ = 0;   // counts time_update calls
    static constexpr int PSEUDO_UPDATE_PERIOD = 3; // every N-th update

    bool linear_block_enabled_ = true;
              
    bool has_cross_cov_a_xy = false;
    bool use_exact_att_bias_Qd_ = true;

    // IMU lever-arm (off-CoG) support
    bool   use_imu_lever_arm_       = false;
    // Lever arm in *physical* BODY frame B (what you measure on the boat).
    Vector3 r_imu_wrt_cog_body_phys_ = Vector3::Zero();

    // Cached kinematics in the virtual un-heeled frame B'
    Vector3 prev_omega_b_ = Vector3::Zero(); // ω^{B'}
    Vector3 alpha_b_      = Vector3::Zero(); // α^{B'}
    bool    have_prev_omega_ = false;

    // Optional smoothing for alpha (0 = off)
    T alpha_smooth_tau_ = T(0.05); // seconds

    // Scratch buffers to avoid large stack allocation in time/measurement updates
    MatrixBaseN  F_AA_scratch_;
    MatrixBaseN  Q_AA_scratch_;

    Matrix12     F_LL_scratch_;
    Matrix12     Q_LL_scratch_;
    Matrix12     tmpLL_scratch_;

    Vector12     x_lin_prev_scratch_;
    Vector12     x_lin_next_scratch_;

    MatrixBaseN  tmpAA_scratch_;
    MatrixBaseN12 tmpAL_scratch_;

    MatrixNX3    PCt_scratch_;
    MatrixNX3    K_scratch_;
    Matrix3      S_scratch_;
              
    EIGEN_STRONG_INLINE void symmetrize_Pext_() {
        for (int i = 0; i < NX; ++i) {
            for (int j = i + 1; j < NX; ++j) {
                const T v = T(0.5) * (Pext(i,j) + Pext(j,i));
                Pext(i,j) = v;
                Pext(j,i) = v;
            }
        }
    }
              
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
        const T invw = T(1) / wnorm;
        const Matrix3 K = W * invw; // [u]×
    
        // exp(-[ω]× t) = I - sinθ K + (1 - cosθ) K^2
        R = Matrix3::Identity() - s*K + (T(1)-c)*(K*K);
    
        // B(t) = - ∫_0^t R(τ) dτ = -[ t I - (1 - cosθ)/ω^2 W + (t - sinθ/ω)/ω^2 W^2 ]
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
    Matrix3 R_wb() const { return qref.toRotationMatrix(); }               // world→body'
    Matrix3 R_bw() const { return qref.toRotationMatrix().transpose(); }   // body'→world

    // Helpers
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
    
    // Joseph covariance update: P ← P - KCP - (KCP)ᵀ + K S Kᵀ.
    // Stack-light version: no NX×NX temporaries, only scalar loops.
    // Assumes Pext is symmetric on entry.
    EIGEN_STRONG_INLINE void joseph_update3_(const Eigen::Matrix<T,NX,3>& K,
                                             const Matrix3& S,
                                             const Eigen::Matrix<T,NX,3>& PCt)
    {
        // We use that:
        //  • PCt = P Cᵀ  (N×3)
        //  • CP ≈ (PCt)ᵀ since P ≈ Pᵀ
        //
        // So:
        //   KCP(i,j) = Σ_l K(i,l) * CP(l,j) ≈ Σ_l K(i,l) * PCt(j,l)
        //   (KCP)ᵀ(j,i) = KCP(j,i)
        //
        // And K S Kᵀ is symmetric because S is symmetric.
    
        for (int i = 0; i < NX; ++i) {
            for (int j = i; j < NX; ++j) {
    
                // KCP_ij = (K C P)(i,j)
                T KCP_ij = T(0);
                T KCP_ji = T(0);
                for (int l = 0; l < 3; ++l) {
                    const T Ki_l = K(i,l);
                    const T Kj_l = K(j,l);
                    const T Pj_l = PCt(j,l); // CP(l,j) ≈ PCt(j,l)
                    const T Pi_l = PCt(i,l); // CP(l,i) ≈ PCt(i,l)
    
                    KCP_ij += Ki_l * Pj_l;
                    if (j != i) {
                        KCP_ji += Kj_l * Pi_l;
                    }
                }
                if (j == i) {
                    KCP_ji = KCP_ij;
                }
    
                // K S Kᵀ (i,j)
                T KSK_ij = T(0);
                for (int a = 0; a < 3; ++a) {
                    const T Kia = K(i,a);
                    for (int b = 0; b < 3; ++b) {
                        const T Kjb = K(j,b);
                        KSK_ij += Kia * S(a,b) * Kjb;
                    }
                }
    
                const T delta = - (KCP_ij + KCP_ji) + KSK_ij;
    
                // Apply symmetric update
                Pext(i,j) += delta;
                if (j != i) {
                    Pext(j,i) = Pext(i,j);
                }
            }
        }
        // Final symmetry clean-up (cheap but good hygiene)
        symmetrize_Pext_();
    }

  void zero_AL_cross_cov_once_() {
      constexpr int NA = BASE_N;
      constexpr int NL = 12; // [v,p,S,a_w]
      Pext.template block<NA,NL>(0, OFF_V).setZero();
      Pext.template block<NL,NA>(OFF_V, 0).setZero();
  }
              
    // Steady wind heel model (roll about BODY X)
    // wind_heel_rad_ : current steady heel in radians (hull frame)
    // Internally we work in a virtual "un-heeled" body frame B'
    // with rotation R_x(-wind_heel_rad_). We cache cos/sin for speed.
    T wind_heel_rad_  = T(0);
    T cos_unheel_x_   = T(1);  // cos(-wind_heel)
    T sin_unheel_x_   = T(0);  // sin(-wind_heel)

    EIGEN_STRONG_INLINE void update_unheel_trig_() {
        if (std::abs(wind_heel_rad_) < T(1e-9)) {
            cos_unheel_x_ = T(1);
            sin_unheel_x_ = T(0);
        } else {
            const T angle = -wind_heel_rad_;
            cos_unheel_x_ = std::cos(angle);
            sin_unheel_x_ = std::sin(angle);
        }
    }

    // Rotate a BODY-frame vector into the virtual un-heeled frame B'
    EIGEN_STRONG_INLINE Vector3 deheel_vector_(const Vector3& v_body) const {
        // R_x(-heel) * v
        if (std::abs(wind_heel_rad_) < T(1e-9)) {
            return v_body;
        }
        Vector3 v;
        v.x() = v_body.x();
        v.y() = cos_unheel_x_ * v_body.y() - sin_unheel_x_ * v_body.z();
        v.z() = sin_unheel_x_ * v_body.y() + cos_unheel_x_ * v_body.z();
        return v;
    }             
};

// Implementation

template <typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::Kalman3D_Wave(
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
    Pbase.setIdentity(); // default small initial cov unless user overwrites

    // initialize base covariance
    Pbase.template topLeftCorner<3,3>() = Matrix3::Identity() * Pq0;   // attitude error covariance
    if constexpr (with_gyro_bias) {
        Pbase.template block<3,3>(3,3) = Matrix3::Identity() * Pb0;    // bias covariance
    }

    // Extended state
    xext.setZero();
    Pext.setZero();

    // Place base P into top-left of Pext
    Pext.topLeftCorner(BASE_N, BASE_N) = Pbase;

    // Seed covariance for a_w (world acceleration)
    Pext.template block<3,3>(OFF_AW, OFF_AW) = Sigma_aw_stat;

    if constexpr (with_accel_bias) {
        Pext.template block<3,3>(OFF_BA, OFF_BA) = Matrix3::Identity() * sigma_bacc0_ * sigma_bacc0_;
    }

if constexpr (with_mag_bias) {
    Pext.template block<3,3>(OFF_BM, OFF_BM) =
        Matrix3::Identity() * sigma_bmag0_ * sigma_bmag0_;
    xext.template segment<3>(OFF_BM).setZero();
}
              
    const T sigma_v0 = T(1.0);    // m/s
    const T sigma_p0 = T(20.0);   // m
    const T sigma_S0 = T(50.0);   // m·s
    set_initial_linear_uncertainty(sigma_v0, sigma_p0, sigma_S0);
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::MatrixBaseN
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::initialize_Q(
              typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::Vector3 sigma_g, T b0) {
    MatrixBaseN Q; Q.setZero();
    if constexpr (with_gyro_bias) {
        Q.template topLeftCorner<3,3>() = sigma_g.array().square().matrix().asDiagonal(); // gyro RW
        Q.template bottomRightCorner<3,3>() = Matrix3::Identity() * b0;                   // bias RW
    } else {
        Q = sigma_g.array().square().matrix().asDiagonal();
    }
    return Q;
}

template <typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::set_aw_stationary_cov_full(const Matrix3& Sigma_in)
{
    // Symmetrize + very light SPD projection
    Matrix3 S = T(0.5) * (Sigma_in + Sigma_in.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix3> es(S);
    if (es.info() == Eigen::Success) {
        auto d = es.eigenvalues().cwiseMax(T(1e-12));
        Sigma_aw_stat = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
    } else {
        // Fallback: keep only diagonal, clamp to tiny+
        Matrix3 D = S.diagonal().cwiseMax(T(1e-12)).asDiagonal();
        Sigma_aw_stat = D;
    }

    // Reseed/merge Pext a_w block
    Pext.template block<3,3>(OFF_AW, OFF_AW) =
          T(0.2) * Pext.template block<3,3>(OFF_AW, OFF_AW)
        + T(0.8) * Sigma_aw_stat;
    symmetrize_Pext_();
    has_cross_cov_a_xy = true;
}

// initialization helpers

// Initialization from accelerometer + magnetometer
// Inputs:
//   acc_body  — accelerometer specific force in body frame (NED)
//   mag_body  — magnetometer measurement in body frame (NED)
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::initialize_from_acc_mag(
    Vector3 const& acc_body,
    Vector3 const& mag_body)
{
    const Vector3 acc = deheel_vector_(acc_body);
    const Vector3 mag = deheel_vector_(mag_body);

    // use acc & mag as if they are BODY-frame
    // (now interpreted as B').
  
    // Normalize accelerometer
    T anorm = acc.norm();
    if (anorm < T(1e-8)) {
        throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
    }
    Vector3 acc_n = acc / anorm;

    // Build WORLD axes expressed in BODY coords
    Vector3 z_world = -acc_n;                         // world Z (down) in body coord
    Vector3 mag_h   = mag - (mag.dot(z_world)) * z_world;
    if (mag_h.norm() < T(1e-8)) {
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
    v2ref = R_bw() * mag;  // body to world, µT
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
Eigen::Quaternion<T>
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::quaternion_from_acc(Vector3 const& acc)
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

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::initialize_from_acc(Vector3 const& acc_body)
{
    const Vector3 acc = deheel_vector_(acc_body);
  
    T anorm = acc.norm();
    if (anorm < T(1e-8)) {
       throw std::runtime_error("Invalid accelerometer vector: norm too small for initialization");
    }
    Vector3 acc_n = acc / anorm;

    // Use accelerometer to align z axis, yaw remains arbitrary
    qref = quaternion_from_acc(acc_n);
    qref.normalize();
}

template <typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::initialize_from_truth(
    const Vector3 &p_ned, const Vector3 &v_ned,
    const Eigen::Quaternion<T> &q_bw, const Vector3 &a_w_ned)
{
    // Reset entire error state to 0
    xext.setZero();

    // Load linear states directly from truth
    xext.template segment<3>(OFF_V)  = v_ned;    // velocity (world NED)
    xext.template segment<3>(OFF_P)  = p_ned;    // position (world NED)
    xext.template segment<3>(OFF_S).setZero();   // integral of p, start at 0
    xext.template segment<3>(OFF_AW) = a_w_ned;  // world acceleration (OU state)

    // Bias states = 0
    if constexpr (with_gyro_bias) {
        xext.template segment<3>(3).setZero();       // gyro bias block
    }
    if constexpr (with_accel_bias) {
        xext.template segment<3>(OFF_BA).setZero();  // accel bias block
    }

    // q_bw is BODY→WORLD (NED). Internally we store WORLD→BODY'.
    qref = q_bw.conjugate();
    qref.normalize();

    // Reset covariance
    Pext.setZero();
    const T p_0 = T(1e-5);
    Pext.diagonal().array() = p_0;
}    
              
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::time_update(
    Vector3 const& gyr_body, T Ts)
{
    // De-heel gyro into virtual frame B' using current wind_heel_rad_
    const Vector3 gyr = deheel_vector_(gyr_body);   // ω^{B'}

    // Attitude propagation
    Vector3 gyro_bias;
    if constexpr (with_gyro_bias) {
        gyro_bias = xext.template segment<3>(3);
    } else {
        gyro_bias = Vector3::Zero();
    }
    last_gyr_bias_corrected = gyr - gyro_bias;

    // IMU lever-arm: estimate angular acceleration α^{B'} (from bias-corrected ω^{B'})
    const Vector3 omega_b = last_gyr_bias_corrected;
    if (have_prev_omega_ && Ts > T(0)) {
        const Vector3 alpha_raw = (omega_b - prev_omega_b_) / Ts;
        if (alpha_smooth_tau_ > T(0)) {
            const T a = T(1) - std::exp(-Ts / alpha_smooth_tau_);
            alpha_b_ = (T(1) - a) * alpha_b_ + a * alpha_raw;
        } else {
            alpha_b_ = alpha_raw;
        }
    } else {
        alpha_b_.setZero();
        have_prev_omega_ = true;
    }
    prev_omega_b_ = omega_b;           

    // Δθ = ω·Ts → right-multiplicative quaternion increment
    Eigen::Quaternion<T> dq = quat_from_delta_theta((last_gyr_bias_corrected * Ts).eval());
    qref = qref * dq;
    qref.normalize();

    // Attitude block F_AA, Q_AA
    Matrix3 I = Matrix3::Identity();
    const Vector3 w = last_gyr_bias_corrected;
    const T omega = w.norm();
    const T theta = omega * Ts;

    MatrixBaseN& F_AA = F_AA_scratch_;
    F_AA.setIdentity();

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

    MatrixBaseN& Q_AA = Q_AA_scratch_;
    Q_AA.setZero();
    
    if (!use_exact_att_bias_Qd_) {
        // fast path
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
            project_psd<T,6>(Q_AA, T(1e-12));
        } else {
            project_psd<T,3>(Q_AA, T(1e-12));
        }
    }

// Always propagate attitude(+gyro-bias) covariance
{
    constexpr int NA = BASE_N;
    MatrixBaseN& tmpAA = tmpAA_scratch_;

    tmpAA.noalias() = F_AA * Pext.template block<NA,NA>(0,0);
    Pext.template block<NA,NA>(0,0).noalias() = tmpAA * F_AA.transpose();
    Pext.template block<NA,NA>(0,0).noalias() += Q_AA;
}
              
    // Linear subsystem [v,p,S,a_w] (12×12)
    Matrix12& F_LL = F_LL_scratch_; F_LL.setZero();
    Matrix12& Q_LL = Q_LL_scratch_; Q_LL.setZero();

    if (linear_block_enabled_) {
          
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
            // group offsets in interleaved state order
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
            project_psd<T,12>(Q_LL, T(1e-12));
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
        Vector12& x_lin_prev = x_lin_prev_scratch_;
        x_lin_prev.template segment<3>(0)  = xext.template segment<3>(OFF_V);
        x_lin_prev.template segment<3>(3)  = xext.template segment<3>(OFF_P);
        x_lin_prev.template segment<3>(6)  = xext.template segment<3>(OFF_S);
        x_lin_prev.template segment<3>(9)  = xext.template segment<3>(OFF_AW);
    
        Vector12& x_lin_next = x_lin_next_scratch_;
        x_lin_next.noalias() = F_LL * x_lin_prev;
        xext.template segment<3>(OFF_V)  = x_lin_next.template segment<3>(0);
        xext.template segment<3>(OFF_P)  = x_lin_next.template segment<3>(3);
        xext.template segment<3>(OFF_S)  = x_lin_next.template segment<3>(6);
        xext.template segment<3>(OFF_AW) = x_lin_next.template segment<3>(9);
    
        // Covariance propagation (blockwise)
        constexpr int NA = BASE_N;
        constexpr int NL = 12;
                
        // LL block
        Matrix12& tmpLL = tmpLL_scratch_;
        
        // tmpLL = F_LL * P_LL_old
        tmpLL.noalias() = F_LL * Pext.template block<NL,NL>(OFF_V,OFF_V);
        
        // P_LL_new = tmpLL * F_LLᵀ + Q_LL
        Pext.template block<NL,NL>(OFF_V,OFF_V).noalias() = tmpLL * F_LL.transpose();
        Pext.template block<NL,NL>(OFF_V,OFF_V).noalias() += Q_LL;
        
        // AL block (cross-covariance)
        MatrixBaseN12& tmpAL = tmpAL_scratch_;
        
        // tmpAL = F_AA * P_AL_old
        tmpAL.noalias() = F_AA * Pext.template block<NA,NL>(0,OFF_V);
        
        // P_AL_new = tmpAL * F_LLᵀ
        Pext.template block<NA,NL>(0,OFF_V).noalias() = tmpAL * F_LL.transpose();
        
        // Keep symmetry: P_LA = P_ALᵀ
        Pext.template block<NL,NA>(OFF_V,0) = Pext.template block<NA,NL>(0,OFF_V).transpose();
    }
          
    // Optional accel bias RW and cross terms (F_BB = I)
    if constexpr (with_accel_bias) {
        constexpr int NB = 3;
        auto P_BB = Pext.template block<NB,NB>(OFF_BA,OFF_BA);
        P_BB.noalias() += Q_bacc_ * Ts;
        Pext.template block<NB,NB>(OFF_BA,OFF_BA) = P_BB;

        constexpr int NA = BASE_N;
        Eigen::Matrix<T,NA,NB> tmpAB = F_AA * Pext.template block<NA,NB>(0,OFF_BA);
        Pext.template block<NA,NB>(0,OFF_BA) = tmpAB;
        Pext.template block<NB,NA>(OFF_BA,0) = tmpAB.transpose();

        if (linear_block_enabled_) {
            constexpr int NL = 12;
            Eigen::Matrix<T,NL,NB> tmpLB = F_LL * Pext.template block<NL,NB>(OFF_V,OFF_BA);
            Pext.template block<NL,NB>(OFF_V,OFF_BA) = tmpLB;
            Pext.template block<NB,NL>(OFF_BA,OFF_V) = tmpLB.transpose();
        }
    }

    // Optional mag bias RW and cross terms (F_BM = I)
    if constexpr (with_mag_bias) {
        constexpr int NB = 3;
    
        // P_BM_BM += Q_bmag * Ts
        auto P_BM = Pext.template block<NB,NB>(OFF_BM, OFF_BM);
        P_BM.noalias() += Q_bmag_ * Ts;
        Pext.template block<NB,NB>(OFF_BM, OFF_BM) = P_BM;
    
        // Cross-covariances: AB, LB propagate with F_AA / F_LL
        constexpr int NA = BASE_N;
        Eigen::Matrix<T,NA,NB> tmpAM = F_AA * Pext.template block<NA,NB>(0, OFF_BM);
        Pext.template block<NA,NB>(0, OFF_BM) = tmpAM;
        Pext.template block<NB,NA>(OFF_BM, 0) = tmpAM.transpose();
    
        if (linear_block_enabled_) {
            constexpr int NL = 12;
            Eigen::Matrix<T,NL,NB> tmpLM = F_LL * Pext.template block<NL,NB>(OFF_V, OFF_BM);
            Pext.template block<NL,NB>(OFF_V, OFF_BM) = tmpLM;
            Pext.template block<NB,NL>(OFF_BM, OFF_V) = tmpLM.transpose();
        }
    
        // If accel-bias exists too, keep BA<->BM symmetry (F=I for both)
        if constexpr (with_accel_bias) {
            // nothing to do: both are constant states; cross-cov stays as-is
            // (your symmetrize_Pext_() will keep it clean)
        }
    }          
          
    // Symmetry hygiene
    symmetrize_Pext_();

    // Integral pseudo-measurement drift correction (only if linear block is live)
    if (linear_block_enabled_) {
        if (++pseudo_update_counter_ >= PSEUDO_UPDATE_PERIOD) {
            applyIntegralZeroPseudoMeas();
            pseudo_update_counter_ = 0;
        }
    } else {
        // avoid weird cadence when re-enabling
        pseudo_update_counter_ = 0;
    }
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::measurement_update_acc_only(
    Vector3 const& acc_meas_body, T tempC)
{  
    // De-heel measured accel into B'
    const Vector3 acc_meas = deheel_vector_(acc_meas_body);
          
    // Physical accelerometer measurement model
    // f_b' = R_wb (a_w - g) + a_lever^{B'} + b_a(temp) + noise
    const Vector3 f_pred = accelerometer_measurement_func(tempC);
    const Vector3 f_meas = acc_meas;
    const Vector3 r = f_meas - f_pred; // innovation in true units (m/s²)

    // Residual gate: only reject clearly insane outliers.
    // r is in m/s², gravity_magnitude_ is ~9.80665.
    const T sigma_r = std::sqrt(Racc.trace() / T(3));
    if (!std::isfinite(sigma_r) || sigma_r <= T(0)) {
        // If noise is bogus, skip this update entirely.
        return;
    }
    
    // Use a physically-based threshold: e.g. 2 g of residual.
    // That’s huge; normal sea-state mismatch should be well below this.
    const T g = gravity_magnitude_;
    const T thresh = T(2.0) * g;   // ~19.6 m/s²
    
    if (r.norm() > thresh) {
        // Only bail out on completely inconsistent samples
        return;
    }
              
    // Proper Jacobians from linearization at CoG-only part (lever-arm is attitude-independent)
    const Vector3 g_world(0,0,+gravity_magnitude_);
    const Vector3 aw = xext.template segment<3>(OFF_AW);
    const Vector3 f_cog_b = R_wb() * (aw - g_world);
    const Matrix3 J_att = -skew_symmetric_matrix(f_cog_b); // ∂f/∂θ = -[f_cog_b]_×
    const Matrix3 J_aw  =  R_wb();                         // ∂f/∂a_w = R_wb
              
    // Innovation covariance S = C P Cᵀ + Racc (3×3)
    Matrix3& S_mat = S_scratch_;
    S_mat = Racc;
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

            S_mat.noalias() += J_att * P_th_ba; // J_ba = ∂f/∂b_a = I
            S_mat.noalias() += J_aw  * P_aw_ba;

            S_mat.noalias() += P_th_ba.transpose() * J_att.transpose();
            S_mat.noalias() += P_aw_ba.transpose() * J_aw.transpose();
            S_mat.noalias() += P_ba_ba;
        }
    }

    // PCᵀ = P Cᵀ (NX×3)
    MatrixNX3& PCt = PCt_scratch_; PCt.setZero();
    {
        constexpr int OFF_TH = 0;
        const auto P_all_th = Pext.template block<NX,3>(0, OFF_TH);
        const auto P_all_aw = Pext.template block<NX,3>(0, OFF_AW);
        PCt.noalias() += P_all_th * J_att.transpose();
        PCt.noalias() += P_all_aw * J_aw.transpose();

        if constexpr (with_accel_bias) {
            const auto P_all_ba = Pext.template block<NX,3>(0, OFF_BA);
            PCt.noalias() += P_all_ba; // J_ba = ∂f/∂b_a = I
        }
    }

    // Gain
    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, Racc.norm())) return;
    MatrixNX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update
    joseph_update3_(K, S_mat, PCt);

    // Apply quaternion correction
    applyQuaternionCorrectionFromErrorState();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::measurement_update_mag_only(
    const Vector3& mag_meas_body)
{
    // De-heel magnetometer into B'
    const Vector3 mag_meas = deheel_vector_(mag_meas_body);
          
    // Predicted magnetic field in body frame (now includes bias if enabled)
    const Vector3 zhat = magnetometer_measurement_func();
    
    // Innovation
    const Vector3 r = mag_meas - zhat;
    
    // Jacobians
    const Vector3 v2hat_no_bias = R_wb() * v2ref;               // for attitude jacobian
    const Matrix3 J_att = -skew_symmetric_matrix(v2hat_no_bias);
    
    // Innovation covariance S = C P Cᵀ + R
    Matrix3& S_mat = S_scratch_;
    S_mat = Rmag;
    
    const Matrix3 P_th_th = Pext.template block<3,3>(0,0);
    S_mat.noalias() += J_att * P_th_th * J_att.transpose();
    
    if constexpr (with_mag_bias) {
        const Matrix3 P_th_bm = Pext.template block<3,3>(0, OFF_BM);
        const Matrix3 P_bm_bm = Pext.template block<3,3>(OFF_BM, OFF_BM);
    
        S_mat.noalias() += J_att * P_th_bm;                      // J_bm = I
        S_mat.noalias() += P_th_bm.transpose() * J_att.transpose();
        S_mat.noalias() += P_bm_bm;
    }
    
    // PCᵀ = P Cᵀ
    MatrixNX3& PCt = PCt_scratch_;
    PCt.setZero();
    PCt.noalias() += Pext.template block<NX,3>(0,0) * J_att.transpose();
    if constexpr (with_mag_bias) {
        PCt.noalias() += Pext.template block<NX,3>(0, OFF_BM);  // J_bm = I
    }
    
    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, Rmag.norm())) return;
    
    MatrixNX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Matrix3::Identity());
    
    // State + covariance update
    xext.noalias() += K * r;
    joseph_update3_(K, S_mat, PCt);
    applyQuaternionCorrectionFromErrorState();          
}

// specific force prediction (BODY'):
//   f_b' = R_wb (a_w − g) + α^{B'} × r_imu^{B'} + ω^{B'} × (ω^{B'} × r_imu^{B'}) + b_a(temp)
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::Vector3
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::accelerometer_measurement_func(T tempC) const {
    const Vector3 g_world(0,0,+gravity_magnitude_);
    const Vector3 aw = xext.template segment<3>(OFF_AW);

    // CoG specific force in B'
    const Vector3 f_cog_b = R_wb() * (aw - g_world);
    Vector3 fb = f_cog_b;

    // Optional IMU lever-arm correction:
    // Use ω^{B'}, α^{B'} and r_imu expressed in B' (via de-heel).
    if (use_imu_lever_arm_) {
        const Vector3& omega_bprime = last_gyr_bias_corrected; // ω^{B'}
        const Vector3& alpha_bprime = alpha_b_;                // α^{B'}
        const Vector3  r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);

        fb.noalias() += alpha_bprime.cross(r_imu_bprime)
                     +  omega_bprime.cross(omega_bprime.cross(r_imu_bprime));
    }

    if constexpr (with_accel_bias) {
        const Vector3 ba0 = xext.template segment<3>(OFF_BA);
        const Vector3 ba  = ba0 + k_a_ * (tempC - tempC_ref);
        fb += ba;
    }
    return fb;             
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
typename Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::Vector3
Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::magnetometer_measurement_func() const {
    Vector3 pred = R_wb() * v2ref;
    if constexpr (with_mag_bias) {
        pred += xext.template segment<3>(OFF_BM); // b_m in BODY' (µT)
    }
    return pred;
}

// utility functions
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
Matrix<T, 3, 3> Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec) const {
    Matrix3 M;
    M << 0, -vec(2), vec(1),
         vec(2), 0, -vec(0),
        -vec(1), vec(0), 0;
    return M;
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::applyQuaternionCorrectionFromErrorState() {
    Eigen::Quaternion<T> corr = quat_from_delta_theta((xext.template segment<3>(0)).eval());
    qref = qref * corr;
    qref.normalize();

    // Clear error-state attitude correction after applying
    xext.template head<3>().setZero();
}

template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::applyIntegralZeroPseudoMeas()
{
    constexpr int off_S = OFF_S;   // offset of S block (3 states)

    // Innovation: target S = 0
    const Vector3 r = -xext.template segment<3>(off_S);

    // Innovation covariance S = P_SS + R_S
    Matrix3& S_mat = S_scratch_;
    S_mat = Pext.template block<3,3>(off_S, off_S) + R_S;

    // Cross covariance PCᵀ = P(:,S) (NX×3)
    MatrixNX3& PCt = PCt_scratch_;
    PCt.noalias() = Pext.template block<NX,3>(0, off_S);

    Eigen::LDLT<Matrix3> ldlt;
    if (!safe_ldlt3_(S_mat, ldlt, R_S.norm())) return;
    MatrixNX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update
    joseph_update3_(K, S_mat, PCt);

    // Apply quaternion correction (attitude may get nudged via cross-covariances)
    applyQuaternionCorrectionFromErrorState();
}
              
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::measurement_update_position_pseudo(
    const Vector3& p_meas,
    const Vector3& sigma_meas)
{
    if (!linear_block_enabled_) {
        return;
    }
    
    constexpr int off_P = OFF_P; // position block

    // Predicted position (world, NED)
    const Vector3 p_pred = xext.template segment<3>(off_P);
    Vector3 r = p_meas - p_pred;          // innovation (meters)

    if (!r.allFinite()) {
        return;
    }

    // Innovation covariance S = H P Hᵀ + R, with H selecting the p-block.
    // Here H is [0 ... I_3 ... 0], so:
    //
    //   S = P_pp + R
    //
    Matrix3& S_mat = S_scratch_;
    S_mat = Pext.template block<3,3>(off_P, off_P);

    Matrix3 R_meas = Matrix3::Zero();
    const T sx = std::max(T(0), sigma_meas.x());
    const T sy = std::max(T(0), sigma_meas.y());
    const T sz = std::max(T(0), sigma_meas.z());
    R_meas(0,0) = sx * sx;
    R_meas(1,1) = sy * sy;
    R_meas(2,2) = sz * sz;
    S_mat.noalias() += R_meas;

    // Cross-covariance PCᵀ = P(:,p) (N×3)
    MatrixNX3& PCt = PCt_scratch_;
    PCt.noalias() = Pext.template block<NX,3>(0, off_P);

    // Gain K = PCᵀ S⁻¹
    Eigen::LDLT<Matrix3> ldlt;
    const T noise_scale = R_meas.norm();
    if (!safe_ldlt3_(S_mat, ldlt, noise_scale)) {
        return;
    }

    MatrixNX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Matrix3::Identity());

    // State update
    xext.noalias() += K * r;

    // Covariance update (Joseph form, 3D)
    joseph_update3_(K, S_mat, PCt);

    // Attitude may have been nudged via cross-covariance → apply correction
    applyQuaternionCorrectionFromErrorState();
}              
              
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::PhiAxis4x1_analytic(
    T tau, T h, Eigen::Matrix<T,4,4>& Phi_axis)
{
    const auto P = make_prims<T>(h, tau);
    // Stable re-expressions (let x = h/tau, em1 = expm1(-x)):
    // 1 - alpha = -em1
    // phi_va_c = tau*(1 - alpha)               = -tau*em1
    // phi_pa_c = tau*h - tau^2*(1 - alpha)     = tau^2*(x + em1)
    // phi_Sa_c = 0.5*tau*h^2 - tau^2*h + tau^3*(1 - alpha)
    //        = tau^3*(0.5*x^2 - x - em1)

    const T phi_va_c = -tau * P.em1;
    auto coeffs = safe_phi_A_coeffs<T>(h, tau);
    const T phi_pa_c = coeffs.phi_pa;
    const T phi_Sa_c = coeffs.phi_Sa;

    Phi_axis.setZero();
    // v_{k+1}
    Phi_axis(0,0) = T(1);
    Phi_axis(0,3) = phi_va_c;

    // p_{k+1}
    Phi_axis(1,0) = h;
    Phi_axis(1,1) = T(1);
    Phi_axis(1,3) = phi_pa_c;

    // S_{k+1}
    Phi_axis(2,0) = T(0.5)*h*h;
    Phi_axis(2,1) = h;
    Phi_axis(2,2) = T(1);
    Phi_axis(2,3) = phi_Sa_c;

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
template<typename T, bool with_gyro_bias, bool with_accel_bias, bool with_mag_bias>
void Kalman3D_Wave<T, with_gyro_bias, with_accel_bias, with_mag_bias>::QdAxis4x1_analytic(
    T tau, T h, T sigma2, Eigen::Matrix<T,4,4>& Qd_axis)
{
    // Guard very small tau
    const T tau_eff = std::max(tau, T(1e-7));
    const T inv_tau = T(1) / tau_eff;
    const T x       = h * inv_tau;      // x = h/τ

    // Small-x series branch (|h/τ| ≪ 1)
    // Derived from Qd = ∫_0^h Φ(t) G q_c Gᵀ dt with
    //   q_c = 2 σ² / τ,  a is OU with corr. time τ, stat. var σ².
    //
    // Series in h up to O(h^5) (equivalently O(x^5)):
    //
    //   Q_vv ≈ σ² (  2 h^3/(3 τ)  −  h^4/(2 τ²)   +  7 h^5/(30 τ^3)     )
    //   Q_vp ≈ σ² (    h^4/(4 τ)  −  h^5/(6 τ²)                         )
    //   Q_vS ≈ σ² (    h^5/(15 τ)                                       )
    //   Q_va ≈ σ² (    h^2/τ  −  h^3/τ² + 7 h^4/(12 τ^3) − h^5/(4 τ^4)  )
    //   Q_pp ≈ σ² (    h^5/(10 τ)                                       )
    //   Q_pa ≈ σ² (    h^3/(3 τ) − h^4/(3 τ²) + 11 h^5/(60 τ^3)         )
    //   Q_Sa ≈ σ² (    h^4/(12 τ) − h^5/(12 τ²)                         )
    //   Q_aa ≈ σ² ( 2 h/τ − 2 h^2/τ² + 4 h^3/(3 τ^3)
    //                         − 2 h^4/(3 τ^4) + 4 h^5/(15 τ^5) )
    //         
    // and Q is symmetric.  Terms like Q_pS, Q_SS are O(h^6..h^7) and
    // safely negligible in the x≲1e−2 regime.

    if (std::abs(x) < T(1e-2)) {
        const T inv_tau2 = inv_tau * inv_tau;
        const T inv_tau3 = inv_tau2 * inv_tau;
        const T inv_tau4 = inv_tau3 * inv_tau;
        const T inv_tau5 = inv_tau4 * inv_tau;

        const T h2 = h * h;
        const T h3 = h2 * h;
        const T h4 = h3 * h;
        const T h5 = h4 * h;

        const T s = sigma2;

        Qd_axis.setZero();

        // Row 0: v
        Qd_axis(0,0) = s * ( (T(2) / T(3)) * h3 * inv_tau
                           - T(1) / T(2)   * h4 * inv_tau2
                           + T(7) / T(30)  * h5 * inv_tau3 );
        Qd_axis(0,1) = s * (  T(1) / T(4)  * h4 * inv_tau
                           - T(1) / T(6)   * h5 * inv_tau2 );
        Qd_axis(0,2) = s * (  T(1) / T(15) * h5 * inv_tau );
        Qd_axis(0,3) = s * (  h2 * inv_tau
                           -  h3 * inv_tau2
                           + (T(7) / T(12)) * h4 * inv_tau3
                           -  T(1) / T(4)   * h5 * inv_tau4 );

        // Row 1: p
        Qd_axis(1,0) = Qd_axis(0,1);
        Qd_axis(1,1) = s * (  T(1) / T(10) * h5 * inv_tau );
        Qd_axis(1,2) = T(0); // O(h^6) and smaller, negligible here
        Qd_axis(1,3) = s * (  T(1) / T(3)  * h3 * inv_tau
                           -  T(1) / T(3)  * h4 * inv_tau2
                           + T(11) / T(60) * h5 * inv_tau3 );

        // Row 2: S
        Qd_axis(2,0) = Qd_axis(0,2);
        Qd_axis(2,1) = Qd_axis(1,2); // = 0
        Qd_axis(2,2) = T(0);         // leading term is O(h^7)
        Qd_axis(2,3) = s * (  T(1) / T(12) * h4 * inv_tau
                           -  T(1) / T(12) * h5 * inv_tau2 );

        // Row 3: a
        Qd_axis(3,0) = Qd_axis(0,3);
        Qd_axis(3,1) = Qd_axis(1,3);
        Qd_axis(3,2) = Qd_axis(2,3);
        Qd_axis(3,3) = s * (  T(2)        * h  * inv_tau
                           -  T(2)        * h2 * inv_tau2
                           + (T(4) / T(3))  * h3 * inv_tau3
                           - (T(2) / T(3))  * h4 * inv_tau4
                           + (T(4) / T(15)) * h5 * inv_tau5 );

        // Scrub NaNs / infs and enforce PSD
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                T &v = Qd_axis(i,j);
                if (!std::isfinite(v)) {
                    v = (i == j) ? T(1e-18) : T(0);
                }
            }
        }
        project_psd<T,4>(Qd_axis, T(1e-12));
        Qd_axis = T(0.5) * (Qd_axis + Qd_axis.transpose());
        return;
    }

    // General closed-form branch
    // Scalar OU for [v, p, S, a]:
    //   dv = a dt
    //   dp = v dt
    //   dS = p dt
    //   da = -(1/τ) a dt + sqrt(q_c) dW,  q_c = 2 σ² / τ
    //
    // Exact discrete covariance:
    //   Qd(h) = ∫_0^h Φ(t) G q_c Gᵀ Φ(t)ᵀ dt
    // with closed-form entries in terms of x = h/τ and α = e^{-x}.
    const T x_full  = x;
    const T alpha   = std::exp(-x_full);
    const T alpha2  = alpha * alpha;

    const T q_c     = (T(2) * sigma2) * inv_tau;

    const T tau2 = tau_eff * tau_eff; const T tau3 = tau2 * tau_eff;
    const T tau4 = tau3 * tau_eff;    const T tau5 = tau4 * tau_eff;
    const T tau6 = tau5 * tau_eff;    const T tau7 = tau6 * tau_eff;

    const T x2 = x_full * x_full;  const T x3 = x2 * x_full;
    const T x4 = x3 * x_full;      const T x5 = x4 * x_full;

    const T K00 = tau3 * (-alpha2 + T(4)*alpha + T(2)*x_full - T(3)) / T(2);
    const T K01 = tau4 * ( alpha2 + T(2)*alpha*(x_full - T(1)) + x2 - T(2)*x_full + T(1)) / T(2);
    const T K02 = tau5 * (-T(3)*alpha2 + T(3)*alpha*(x2 + T(4)) + x3 - T(3)*x2 + T(6)*x_full - T(9)) / T(6);
    const T K03 = tau2 * ( alpha2 - T(2)*alpha + T(1)) / T(2);

    const T K11 = tau5 * (-alpha2/T(2) - T(2)*alpha*x_full + x3/T(3) - x2 + x_full + T(1)/T(2));
    const T K12 = tau6 * ( alpha2/T(2)
                         + alpha * (-x2 + T(2)*x_full - T(2)) / T(2)
                         + x4/T(8) - x3/T(2) + x2 - x_full + T(1)/T(2));
    const T K13 = tau3 * (-alpha2 - T(2)*alpha*x_full + T(1)) / T(2);

    const T K22 = tau7 * (-alpha2/T(2)
                         + alpha * x2 + T(2)*alpha
                         + x5/T(20) - x4/T(4) + T(2)*x3/T(3)
                         - x2 + x_full - T(3)/T(2));
    const T K23 = tau4 * ( alpha2 - alpha * (x2 + T(2)) + T(1)) / T(2);
    const T K33 = tau_eff * (T(1) - alpha2) / T(2);

    Qd_axis.setZero();
    Qd_axis(0,0) = q_c * K00;
    Qd_axis(0,1) = q_c * K01;
    Qd_axis(0,2) = q_c * K02;
    Qd_axis(0,3) = q_c * K03;

    Qd_axis(1,0) = Qd_axis(0,1);
    Qd_axis(1,1) = q_c * K11;
    Qd_axis(1,2) = q_c * K12;
    Qd_axis(1,3) = q_c * K13;

    Qd_axis(2,0) = Qd_axis(0,2);
    Qd_axis(2,1) = Qd_axis(1,2);
    Qd_axis(2,2) = q_c * K22;
    Qd_axis(2,3) = q_c * K23;

    Qd_axis(3,0) = Qd_axis(0,3);
    Qd_axis(3,1) = Qd_axis(1,3);
    Qd_axis(3,2) = Qd_axis(2,3);
    Qd_axis(3,3) = q_c * K33;

    // Scrub NaNs / infs and enforce PSD
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T &v = Qd_axis(i,j);
            if (!std::isfinite(v)) {
                v = (i == j) ? T(1e-18) : T(0);
            }
        }
    }
    project_psd<T,4>(Qd_axis, T(1e-12));
    Qd_axis = T(0.5) * (Qd_axis + Qd_axis.transpose());
}
