#pragma once

/*
  Based on: https://github.com/thomaspasser/q-mekf
  MIT License, Copyright (c) 2023 Thomas Passer

  q-mekf (merged extension)

  Enhancements:
  Copyright (c) 2025 Mikhail Grushinskiy

  This file defines Kalman_INS<T,with_gyro_bias,with_accel_bias>, an extended
  error-state Kalman filter that adds linear navigation states:
     v (3)      : velocity in world frame
     p (3)      : displacement/position in world frame
     S (3)      : integral of displacement (∫ p dt) — with zero pseudo-measurement for drift correction
     a_w (3)    : latent acceleration (world frame)
     ȧ_w (3)   : latent acceleration derivative (Matérn(3/2) model)
     b_a (3)    : optional accelerometer bias

  - The quaternion MEKF logic (time_update, measurement_update, quaternion correction)
    is preserved where possible.
  - The extended linear states are driven by a latent 2nd-order Gauss–Markov (Matérn-3/2) process
    instead of a simple OU.
  - A full extended covariance (Pext) and transition Jacobian are constructed;
    the top-left corner contains the original MEKF's P/Q blocks (attitude error + optional gyro bias).
  - Accelerometer and magnetometer inputs must be given in aerospace/NED (x north, y east, z down).
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

#ifdef EIGEN_NON_ARDUINO
  #include <unsupported/Eigen/MatrixFunctions>  // enables .exp() on matrices
#endif

template <typename T = float, bool with_gyro_bias = true, bool with_accel_bias = true>
class EIGEN_ALIGN_MAX Kalman_INS {

    // Original base (att_err + optional gyro bias)
    static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

    // Extended added states: v(3), p(3), S(3), a_w(3), ȧ_w(3) [+ optional b_acc(3)]
    static constexpr int EXT_ADD = with_accel_bias ? 18 : 15;
    static constexpr int NX      = BASE_N + EXT_ADD;

    // Offsets
    static constexpr int OFF_V    = BASE_N + 0;
    static constexpr int OFF_P    = BASE_N + 3;
    static constexpr int OFF_S    = BASE_N + 6;
    static constexpr int OFF_AW   = BASE_N + 9;
    static constexpr int OFF_AW_D = BASE_N + 12;
    static constexpr int OFF_BA   = with_accel_bias ? (BASE_N + 15) : -1;

    // Measurement dimension
    static constexpr int M = 6;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, 6, 1> Vector6;
    typedef Matrix<T, BASE_N, BASE_N> MatrixBaseN;
    typedef Matrix<T, NX, NX> MatrixNX;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, M, M> MatrixM;

    static constexpr T half = T(1) / T(2);
    static constexpr T STD_GRAVITY = T(9.80665);

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Kalman_INS(Vector3 const& sigma_a, Vector3 const& sigma_g, Vector3 const& sigma_m,
               T Pq0 = T(1e-6), T Pb0 = T(1e-1), T b0 = T(1e-12), T R_S_noise = T(5e-2),
               T gravity_magnitude = T(STD_GRAVITY));

    // Initialization
    void initialize_from_acc_mag(Vector3 const& acc, Vector3 const& mag);
    void initialize_from_acc(Vector3 const& acc);
    static Eigen::Quaternion<T> quaternion_from_acc(Vector3 const& acc);

    // Measurement updates
    void measurement_update(Vector3 const& acc, Vector3 const& mag, T tempC = T(35.0));
    void measurement_update_acc_only(Vector3 const& acc, T tempC = T(35.0));
    void measurement_update_mag_only(Vector3 const& mag);

    // Time update
    void time_update(Vector3 const& gyr, T Ts);

    // Pseudo-measurement
    void applyIntegralZeroPseudoMeas();

    // Accessors
    Eigen::Quaternion<T> quaternion() const { return qref.conjugate(); }
    MatrixBaseN const& covariance_base() const { return Pbase; }
    MatrixNX const& covariance_full() const { return Pext; }

    Vector3 get_velocity() const { return xext.template segment<3>(OFF_V); }
    Vector3 get_position() const { return xext.template segment<3>(OFF_P); }
    Vector3 get_integral_displacement() const { return xext.template segment<3>(OFF_S); }
    Vector3 get_world_accel() const { return xext.template segment<3>(OFF_AW); }
    Vector3 get_world_accel_dot() const { return xext.template segment<3>(OFF_AW_D); }

    // Tuning setters
    void set_aw_time_constant(T tau_seconds) { tau_aw = std::max(T(1e-3), tau_seconds); }
    void set_aw_stationary_std(const Vector3& std_aw) {
        Sigma_aw_stat = std_aw.array().square().matrix().asDiagonal();
    }

  private:
    const T gravity_magnitude_;

    Eigen::Quaternion<T> qref;
    Vector3 v2ref = Vector3::UnitX();

    MatrixBaseN Pbase;
    Matrix<T, NX, 1> xext;
    MatrixNX Pext;

    Vector3 last_gyr_bias_corrected{};

    // Noise/bias
    MatrixBaseN Qbase;
    Matrix3 Racc;
    Matrix3 Rmag;
    MatrixM R;
    Matrix3 R_S;

    T tau_aw = T(1.1);
    Matrix3 Sigma_aw_stat = Matrix3::Identity() * T(0.2*0.2);

    // Helpers
    void assembleExtendedFandQ(const Vector3&, T Ts, Matrix<T, NX, NX>& F_a_ext, MatrixNX& Q_a_ext);
    void vanLoanDiscretization_15x3(const Eigen::Matrix<T,15,15>& A,
                                    const Eigen::Matrix<T,15,3>& G,
                                    const Eigen::Matrix<T,3,3>& Sigma_c,
                                    T Ts,
                                    Eigen::Matrix<T,15,15>& Phi,
                                    Eigen::Matrix<T,15,15>& Qd) const;

    Matrix3 skew_symmetric_matrix(const Eigen::Ref<const Vector3>& v) const;
    void applyQuaternionCorrectionFromErrorState();
    void normalizeQuat();
};

