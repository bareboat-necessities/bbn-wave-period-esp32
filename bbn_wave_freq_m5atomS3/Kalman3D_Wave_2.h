#pragma once
/*
  Copyright (c) 2026  Mikhail Grushinskiy

  Kalman3D_Wave_2  (Broadband oscillator wave model + qMEKF attitude)

  - qref_ stores WORLD -> BODY' (virtual un-heeled frame B')
  - quaternion() returns BODY' -> WORLD (qref_.conjugate())
  - quaternion_boat() returns physical BODY -> WORLD by re-applying heel
  - Right-multiply quaternion updates: qref_ = qref_ ⊗ δq
  - Error-state attitude δθ is applied then cleared
  - Structured base Qd path (Simpson + B(t) integrals)

  Wave model (K modes), per mode k and per axis:
      p' = v
      v' = -ω_k^2 p - 2ζ_k ω_k v + ξ(t)
  ξ is white acceleration-noise driving v' (units m/s^3), intensity q_k (units m^2/s^5).

  Wave acceleration in WORLD:
      a_w = Σ_k ( -ω_k^2 p_k - 2ζ_k ω_k v_k )

  State:
    [ δθ(3),
      (b_g 3 optional),
      (p1(3), v1(3), ..., pK(3), vK(3)),
      (b_a 3 optional) ]

  Template parameters:
    T, KMODES, with_gyro_bias, with_accel_bias, with_mag
*/

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
  #include <Eigen/Eigenvalues>
#else
  #include <ArduinoEigenDense.h>
#endif

#include <cmath>
#include <limits>
#include <algorithm>

#ifndef M_PI
  #define M_PI 3.14159265358979323846264338327950288
#endif

// Small helpers

template<typename T>
static inline Eigen::Matrix<T,3,3> skew3(const Eigen::Matrix<T,3,1>& a) {
  Eigen::Matrix<T,3,3> S;
  S << T(0),   -a.z(),  a.y(),
       a.z(),  T(0),   -a.x(),
      -a.y(),  a.x(),   T(0);
  return S;
}

// Full exponential-map correction (Rodrigues in quaternion form).
template<typename T>
inline Eigen::Quaternion<T> quat_from_delta_theta(const Eigen::Matrix<T,3,1>& dtheta) {
  const T theta = dtheta.norm();
  const T half_theta = T(0.5) * theta;

  T w, k; // scalar part, vector scale
  if (theta < T(1e-2)) {
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
  q.normalize();
  return q;
}

// Helper: project symmetric NxN to PSD
template<typename T, int N>
static inline void project_psd(Eigen::Matrix<T,N,N>& S, T eps = T(1e-12)) {
  S = T(0.5) * (S + S.transpose());
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (!std::isfinite(S(i,j))) S(i,j) = (i==j) ? eps : T(0);
    }
  }
  if constexpr (N <= 4) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N>> es(S);
    if (es.info() != Eigen::Success) {
      S.diagonal().array() += eps;
      S = T(0.5) * (S + S.transpose());
      return;
    }
    Eigen::Matrix<T,N,1> lam = es.eigenvalues();
    for (int i=0;i<N;++i) if (!(lam(i) > T(0))) lam(i) = eps;
    S = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
  } else {
    Eigen::LDLT<Eigen::Matrix<T,N,N>> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      T min_lb = std::numeric_limits<T>::infinity();
      for (int i=0;i<N;++i) {
        T row_sum = T(0);
        for (int j=0;j<N;++j) if (j!=i) row_sum += std::abs(S(i,j));
        const T lb = S(i,i) - row_sum;
        if (lb < min_lb) min_lb = lb;
      }
      if (!(min_lb > eps)) S.diagonal().array() += (eps - min_lb);
      ldlt.compute(S);
      if (ldlt.info() != Eigen::Success) S.diagonal().array() += (T(10)*eps);
    }
  }
  S = T(0.5) * (S + S.transpose());
}

// Base (att+bias) exact-ish Qd helpers

template<typename T>
static inline bool is_isotropic3_(const Eigen::Matrix<T,3,3>& S, T tol = T(1e-9)) {
  const T a=S(0,0), b=S(1,1), c=S(2,2);
  Eigen::Matrix<T,3,3> Off = S; Off.diagonal().setZero();
  const T off = Off.cwiseAbs().sum();
  const T mean = (a+b+c)/T(3);
  return (std::abs(a-mean)+std::abs(b-mean)+std::abs(c-mean)+off) <= tol*(T(1)+std::abs(mean));
}

// Rodrigues rotation and B(t) = -∫_0^t exp(-[ω]× τ) dτ
template<typename T>
static inline void rot_and_B_from_wt_(const Eigen::Matrix<T,3,1>& w, T t,
                                     Eigen::Matrix<T,3,3>& R,
                                     Eigen::Matrix<T,3,3>& B)
{
  const T wnorm = w.norm();
  const Eigen::Matrix<T,3,3> W = skew3<T>(w);

  if (wnorm < T(1e-7)) {
    const T t2=t*t, t3=t2*t;
    R = Eigen::Matrix<T,3,3>::Identity() - W*t + T(0.5)*(W*W)*t2;
    B = -( Eigen::Matrix<T,3,3>::Identity()*t - T(0.5)*W*t2 + (W*W)*(t3/T(6)) );
    return;
  }

  const T theta = wnorm * t;
  const T s = std::sin(theta), c = std::cos(theta);
  const T invw = T(1)/wnorm;
  const Eigen::Matrix<T,3,3> K = W * invw;

  R = Eigen::Matrix<T,3,3>::Identity() - s*K + (T(1)-c)*(K*K);

  const T invw2 = invw*invw;
  const Eigen::Matrix<T,3,3> term1 = Eigen::Matrix<T,3,3>::Identity()*t;
  const Eigen::Matrix<T,3,3> term2 = ((T(1)-c)*invw2) * W;
  const Eigen::Matrix<T,3,3> term3 = ((t - s*invw)*invw2) * (W*W);
  B = -( term1 - term2 + term3 );
}

template<typename T>
static inline void integral_B_ds_(const Eigen::Matrix<T,3,1>& w, T Tstep,
                                  Eigen::Matrix<T,3,3>& IB)
{
  const T wnorm = w.norm();
  const Eigen::Matrix<T,3,3> W = skew3<T>(w);

  if (wnorm < T(1e-7)) {
    const T T2=Tstep*Tstep, T3=T2*Tstep, T4=T3*Tstep;
    IB = -( Eigen::Matrix<T,3,3>::Identity()*(T(0.5)*T2)
          - W*(T(1.0/6.0)*T3)
          + (W*W)*(T(1.0/24.0)*T4) );
    return;
  }

  const T theta = wnorm * Tstep;
  const T s = std::sin(theta), c = std::cos(theta);
  const T invw = T(1)/wnorm;
  const T invw2 = invw*invw;

  const Eigen::Matrix<T,3,3> termI  = Eigen::Matrix<T,3,3>::Identity()*(T(0.5)*Tstep*Tstep);
  const Eigen::Matrix<T,3,3> termW  = ((Tstep - s*invw)*invw2) * W;
  const Eigen::Matrix<T,3,3> termW2 = ( (T(0.5)*Tstep*Tstep) + ((c-T(1))*invw2) ) * invw2 * (W*W);

  IB = -( termI - termW + termW2 );
}

template<typename T>
static inline Eigen::Matrix<T,3,3> simpson_R_Q_RT_(const Eigen::Matrix<T,3,1>& w, T Tstep,
                                                   const Eigen::Matrix<T,3,3>& Q)
{
  Eigen::Matrix<T,3,3> R0,Btmp,Rm,R1;
  rot_and_B_from_wt_(w, T(0), R0, Btmp);
  rot_and_B_from_wt_(w, T(0.5)*Tstep, Rm, Btmp);
  rot_and_B_from_wt_(w, Tstep, R1, Btmp);

  const Eigen::Matrix<T,3,3> f0 = R0 * Q * R0.transpose();
  const Eigen::Matrix<T,3,3> f1 = Rm * Q * Rm.transpose();
  const Eigen::Matrix<T,3,3> f2 = R1 * Q * R1.transpose();
  return (Tstep/T(6)) * (f0 + T(4)*f1 + f2);
}

template<typename T>
static inline Eigen::Matrix<T,3,3> simpson_B_Q_BT_(const Eigen::Matrix<T,3,1>& w, T Tstep,
                                                   const Eigen::Matrix<T,3,3>& Q)
{
  Eigen::Matrix<T,3,3> Rtmp,B0,Bm,B1;
  rot_and_B_from_wt_(w, T(0), Rtmp, B0);
  rot_and_B_from_wt_(w, T(0.5)*Tstep, Rtmp, Bm);
  rot_and_B_from_wt_(w, Tstep, Rtmp, B1);

  const Eigen::Matrix<T,3,3> g0 = B0 * Q * B0.transpose();
  const Eigen::Matrix<T,3,3> g1 = Bm * Q * Bm.transpose();
  const Eigen::Matrix<T,3,3> g2 = B1 * Q * B1.transpose();
  return (Tstep/T(6)) * (g0 + T(4)*g1 + g2);
}

// d/dω of (ω×(ω×r)) = (ω·r) I + ω rᵀ - 2 r ωᵀ
template<typename T>
static inline Eigen::Matrix<T,3,3> d_omega_x_omega_x_r_domega_(const Eigen::Matrix<T,3,1>& w,
                                                               const Eigen::Matrix<T,3,1>& r)
{
  const T s = w.dot(r);
  return Eigen::Matrix<T,3,3>::Identity()*s + (w * r.transpose()) - T(2) * (r * w.transpose());
}

// The filter

template <typename T = float, int KMODES = 3,
          bool with_gyro_bias = true, bool with_accel_bias = true, bool with_mag = true>
class Kalman3D_Wave_2 {

  static_assert(KMODES >= 1, "KMODES must be >= 1");

  // Base (att_err + optional gyro bias)
  static constexpr int BASE_N = with_gyro_bias ? 6 : 3;

  // Wave block: per mode k: p_k(3), v_k(3) => 6 each
  static constexpr int WAVE_N = 6 * KMODES;

  // Optional accel bias
  static constexpr int BA_N   = with_accel_bias ? 3 : 0;

  static constexpr int NX     = BASE_N + WAVE_N + BA_N;

  static constexpr T STD_GRAVITY = T(9.80665);
  static constexpr T tempC_ref   = T(35.0);

  // Offsets
  static constexpr int OFF_DTH  = 0;
  static constexpr int OFF_BG   = with_gyro_bias ? 3 : -1;
  static constexpr int OFF_WAVE = BASE_N;
  static constexpr int OFF_BA   = with_accel_bias ? (BASE_N + WAVE_N) : -1;

  static constexpr int OFF_Pk(int k) { return OFF_WAVE + 6*k + 0; } // 3
  static constexpr int OFF_Vk(int k) { return OFF_WAVE + 6*k + 3; } // 3

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Vec3   = Eigen::Matrix<T,3,1>;
  using Mat3   = Eigen::Matrix<T,3,3>;
  using VecX   = Eigen::Matrix<T,NX,1>;
  using MatX   = Eigen::Matrix<T,NX,NX>;
  using MatX3  = Eigen::Matrix<T,NX,3>;

  struct MeasDiag3 {
    Vec3 r = Vec3::Zero();
    Mat3 S = Mat3::Zero();
    T nis = std::numeric_limits<T>::quiet_NaN();
    bool accepted = false;
  };

  Kalman3D_Wave_2(const Vec3& sigma_a_meas,
                  const Vec3& sigma_g_rw,
                  const Vec3& sigma_m_meas,
                  T Pq0 = T(5e-4),
                  T Pb0 = T(1e-6),
                  T b0  = T(1e-11),
                  T gravity_magnitude = T(STD_GRAVITY))
  : gravity_magnitude_(gravity_magnitude)
  {
    qref_.setIdentity();
    x_.setZero();
    P_.setZero();

    // Defensive init: scratch / temporaries
    F_AA_.setIdentity();
    Q_AA_.setZero();
    tmp_AA_.setZero();
    tmp_Ak_.setZero();

    Phi6_.setIdentity();
    Qd6_.setZero();
    tmp6_.setZero();

    for (int ax = 0; ax < 3; ++ax) {
      Phi2_[ax].setIdentity();
      Qd2_[ax].setZero();
    }

    S_scratch_.setZero();
    PCt_scratch_.setZero();
    K_scratch_.setZero();

    // Measurement noise
    Racc_ = sigma_a_meas.array().square().matrix().asDiagonal();
    Rmag_ = sigma_m_meas.array().square().matrix().asDiagonal();

    // Base process noise:
    Qg_  = sigma_g_rw.array().square().matrix().asDiagonal();
    if constexpr (with_gyro_bias) Qbg_ = Mat3::Identity() * b0;

    // Seed base covariance
    P_.template block<3,3>(OFF_DTH, OFF_DTH) = Mat3::Identity() * Pq0;
    if constexpr (with_gyro_bias) P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * Pb0;

    // Seed accel bias
    if constexpr (with_accel_bias) {
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * (sigma_bacc0_*sigma_bacc0_);
    }

    // Seed wave covariance
    const T sigma_p0 = T(20.0);
    const T sigma_v0 = T(1.0);
    for (int k=0;k<KMODES;++k) {
      P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
      P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
    }

    // Mag reference default
    B_world_ref_ = Vec3::UnitX();

    // Default broadband wave params (user should retune)
    set_broadband_params(T(0.12), T(1.0));

    // Defaults
    use_exact_att_bias_Qd_ = true;

    // Warmup ON by default
    set_warmup_mode(true);

    // Ensure trig consistent
    update_unheel_trig_();
  }

  // Accessors

  [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return qref_.conjugate(); } // BODY'->WORLD

  [[nodiscard]] Eigen::Quaternion<T> quaternion_boat() const {
    const Eigen::Quaternion<T> q_WBprime = quaternion();
    const T half = -wind_heel_rad_ * T(0.5);
    const T c = std::cos(half);
    const T s = std::sin(half);
    const Eigen::Quaternion<T> q_BprimeB(c, s, 0, 0);
    return q_WBprime * q_BprimeB;
  }

  [[nodiscard]] Vec3 gyroscope_bias() const {
    if constexpr (with_gyro_bias) return x_.template segment<3>(OFF_BG);
    else return Vec3::Zero();
  }

  [[nodiscard]] Vec3 get_acc_bias() const {
    if constexpr (with_accel_bias) return x_.template segment<3>(OFF_BA);
    else return Vec3::Zero();
  }

  [[nodiscard]] Vec3 get_position() const {
    Vec3 p=Vec3::Zero();
    for (int k=0;k<KMODES;++k) p += x_.template segment<3>(OFF_Pk(k));
    return p;
  }

  [[nodiscard]] Vec3 get_velocity() const {
    Vec3 v=Vec3::Zero();
    for (int k=0;k<KMODES;++k) v += x_.template segment<3>(OFF_Vk(k));
    return v;
  }

  [[nodiscard]] Vec3 get_world_accel() const { return wave_world_accel_(); }

  const MeasDiag3& lastAccDiag() const noexcept { return last_acc_; }
  const MeasDiag3& lastMagDiag() const noexcept { return last_mag_; }

  // Config

  void set_mag_world_ref(const Vec3& B_world) {
    if constexpr (with_mag) B_world_ref_ = B_world;
    else (void)B_world;
  }

  void set_accel_bias_temp_coeff(const Vec3& ka_per_degC) { k_a_ = ka_per_degC; }

  void set_Q_bacc_rw(const Vec3& rw_std_per_sqrt_s) {
    if constexpr (with_accel_bias) Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
  }

  void set_initial_acc_bias_std(T s) {
    if constexpr (with_accel_bias) {
      sigma_bacc0_ = std::max(T(0), s);
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * (sigma_bacc0_*sigma_bacc0_);
    }
  }

  void set_initial_acc_bias(const Vec3& b0) {
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = b0;
  }

  // Set accelerometer measurement noise (BODY' frame), as per-axis std-dev.
  void set_Racc(const Vec3& sigma_a_meas) {
    const Vec3 s = sigma_a_meas.array().max(T(0.05)).matrix();
    Racc_ = s.array().square().matrix().asDiagonal();
    for (int i=0;i<3;++i) Racc_(i,i) = std::max(Racc_(i,i), T(1e-8));
  }

  // Set full covariance directly (will be symmetrized + PSD-projected).
  void set_Racc(const Mat3& Racc) {
    Racc_ = T(0.5) * (Racc + Racc.transpose());
    for (int i=0; i<3; ++i) Racc_(i,i) = std::max(Racc_(i,i), T(5e-2));
    project_psd<T,3>(Racc_, T(1e-8));
    Racc_ = T(0.5) * (Racc_ + Racc_.transpose());
  }

  // Stationarity gating for warmup gyro-bias learning
  void set_warmup_stationary_thresholds(T gyro_rad_s, T acc_motion_m_s2) {
    warmup_gyro_stationary_thr_ = std::max(T(0), gyro_rad_s);
    warmup_acc_stationary_thr_  = std::max(T(0), acc_motion_m_s2);
  }

  // Toggle structured Qd for base block
  void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }

  // marginalization for disabled wave block
  void set_disabled_wave_accel_cov_world(const Mat3& Sigma_aw_world) {
    Sigma_aw_disabled_world_ = T(0.5) * (Sigma_aw_world + Sigma_aw_world.transpose());
    for (int i=0;i<3;++i) Sigma_aw_disabled_world_(i,i) = std::max(Sigma_aw_disabled_world_(i,i), T(0));
    project_psd<T,3>(Sigma_aw_disabled_world_, T(1e-18));
    Sigma_aw_disabled_world_ = T(0.5) * (Sigma_aw_disabled_world_ + Sigma_aw_disabled_world_.transpose());
  }

  void set_disabled_wave_accel_std_world(const Vec3& sigma_aw_world) {
    Mat3 S = Mat3::Zero();
    S(0,0) = std::max(T(0), sigma_aw_world.x()) * std::max(T(0), sigma_aw_world.x());
    S(1,1) = std::max(T(0), sigma_aw_world.y()) * std::max(T(0), sigma_aw_world.y());
    S(2,2) = std::max(T(0), sigma_aw_world.z()) * std::max(T(0), sigma_aw_world.z());
    set_disabled_wave_accel_cov_world(S);
  }

  // Staged init controls

  void set_wave_block_enabled(bool on) {
    if (wave_block_enabled_ && !on) {
      P_.template block<BASE_N,WAVE_N>(0, OFF_WAVE).setZero();
      P_.template block<WAVE_N,BASE_N>(OFF_WAVE, 0).setZero();
      if constexpr (with_accel_bias) {
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      }

      P_.template block<WAVE_N,WAVE_N>(OFF_WAVE, OFF_WAVE).setZero();
      x_.template segment<WAVE_N>(OFF_WAVE).setZero();
    }
    wave_block_enabled_ = on;
    symmetrize_P_();
    clamp_P_diag_(T(1e-15));
    symmetrize_P_();
  }

  bool wave_block_enabled() const { return wave_block_enabled_; }

  void set_acc_bias_updates_enabled(bool en) {
    if (acc_bias_updates_enabled_ == en) return;
    if constexpr (with_accel_bias) {
      if (!en) {
        P_.template block<3,BASE_N>(OFF_BA,0).setZero();
        P_.template block<BASE_N,3>(0,OFF_BA).setZero();
        P_.template block<3,WAVE_N>(OFF_BA,OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE,OFF_BA).setZero();
      } else {
        auto Pba = P_.template block<3,3>(OFF_BA,OFF_BA);
        const T target = sigma_bacc0_*sigma_bacc0_;
        for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), target);
        P_.template block<3,3>(OFF_BA,OFF_BA) = Pba;
      }
    }
    acc_bias_updates_enabled_ = en;
  }

  void set_warmup_mode(bool on) {
    warmup_mode_ = on;

    set_wave_block_enabled(!on);
    set_acc_bias_updates_enabled(!on);

    if (on) {
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity()*T(1e-12);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity()*T(1e-12);
      }
      clear_imu_lever_arm();
      pseudo_motion_dist_ = T(0);
      pseudo_motion_time_ = T(0);
      vel_detect_.setZero();

      bg_accum_.setZero();
      bg_accum_count_ = 0;
    } else {
      for (int k=0;k<KMODES;++k) {
        const T varp = std::max(T(1e-12), init_var_p_[k]) * T(50);
        const T varv = std::max(T(1e-12), init_var_v_[k]) * T(10);
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * varp;
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * varv;
      }

      x_.template segment<WAVE_N>(OFF_WAVE).setZero();

      P_.template block<BASE_N,WAVE_N>(0, OFF_WAVE).setZero();
      P_.template block<WAVE_N,BASE_N>(OFF_WAVE, 0).setZero();
      if constexpr (with_accel_bias) {
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      }
    }

    symmetrize_P_();
    clamp_P_diag_(T(1e-15));
    symmetrize_P_();
  }

  bool warmup_mode() const { return warmup_mode_; }

  void set_motion_exit_thresholds(T min_distance_m, T min_time_sec) {
    exit_min_distance_ = min_distance_m;
    exit_min_time_     = min_time_sec;
  }

  void update_initialization(const Vec3& acc_body, const Vec3& gyr_body, const Vec3& mag_body, T dt) {
    if (!warmup_mode_) return;
    (void)mag_body;

    const Vec3 acc = deheel_vector_(acc_body);

    const Vec3 g_world(0,0,+gravity_magnitude_);

    const Vec3 f_world    = R_bw() * acc;
    const Vec3 acc_motion = f_world + g_world;

    vel_detect_ += acc_motion * dt;
    pseudo_motion_dist_ += vel_detect_.norm() * dt;
    pseudo_motion_time_ += dt;

    if (pseudo_motion_time_ > exit_min_time_ || pseudo_motion_dist_ > exit_min_distance_) {
      set_warmup_mode(false);
    }

    if constexpr (with_gyro_bias) {
      const Vec3 gyr = deheel_vector_(gyr_body);

      const bool stationary =
        (gyr.norm() <= warmup_gyro_stationary_thr_) &&
        (acc_motion.norm() <= warmup_acc_stationary_thr_);

      if (stationary) {
        const T alpha = T(0.01);
        if (bg_accum_count_ == 0) bg_accum_ = gyr;
        else bg_accum_ = (T(1)-alpha)*bg_accum_ + alpha*gyr;

        bg_accum_count_ = std::min(bg_accum_count_ + 1, 500);

        if (bg_accum_count_ >= warmup_bias_min_samples_) {
          x_.template segment<3>(OFF_BG) = bg_accum_;
          P_.template block<3,3>(OFF_BG,OFF_BG) = Mat3::Identity()*T(1e-8);
        }
      } else {
        bg_accum_count_ = std::max(0, bg_accum_count_ - 2);
      }
    }
  }

  // IMU lever arm

  void set_imu_lever_arm_body(const Vec3& r_b_phys) {
    r_imu_wrt_cog_body_phys_ = r_b_phys;
    use_imu_lever_arm_ = (r_b_phys.squaredNorm() > T(0));
  }

  void clear_imu_lever_arm() {
    r_imu_wrt_cog_body_phys_.setZero();
    use_imu_lever_arm_ = false;
  }

  void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

  // Heel update

  void update_wind_heel(T heel_rad) {
    const T old = wind_heel_rad_;
    if (heel_rad == old) return;
    retarget_bodyprime_frame_(heel_rad - old);
    wind_heel_rad_ = heel_rad;
    update_unheel_trig_();
  }

  // Initialization from acc/mag

  void initialize_from_acc_mag(const Vec3& acc_body, const Vec3& mag_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 mag = deheel_vector_(mag_body);

    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    const Vec3 acc_n = acc / an;

    const Vec3 z_world = -acc_n;
    Vec3 mag_h = mag - (mag.dot(z_world))*z_world;
    if (!(mag_h.norm() > T(1e-8))) return;
    mag_h.normalize();

    const Vec3 x_world = mag_h;
    const Vec3 y_world = z_world.cross(x_world).normalized();

    Mat3 R_wb;
    R_wb.col(0) = x_world;
    R_wb.col(1) = y_world;
    R_wb.col(2) = z_world;

    qref_ = Eigen::Quaternion<T>(R_wb);
    qref_.normalize();

    if constexpr (with_mag) {
      B_world_ref_ = R_bw() * mag;
    }

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  void initialize_from_acc(const Vec3& acc_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    const Vec3 acc_n = acc / an;
    const Vec3 z_world_b = -acc_n;
    const Vec3 e3_world  = Vec3(T(0), T(0), T(1));

    qref_ = Eigen::Quaternion<T>::FromTwoVectors(e3_world, z_world_b);
    qref_.normalize();

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  // Wave tuning

  void set_broadband_params(T f0_hz, T Hs_m, T zeta_mid = T(0.08), T horiz_scale = T(0.35)) {
    const T w0 = T(2)*T(M_PI)*std::max(T(1e-4), f0_hz);

    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1));
      const T lo = std::log(T(0.6));
      const T hi = std::log(T(1.7));
      omega_[k] = std::exp(lo + (hi-lo)*u) * w0;

      const T scale = (k==0 || k==KMODES-1) ? T(1.25) : T(1.0);
      zeta_[k] = std::max(T(0.01), zeta_mid * scale);
    }

    T wsum = T(0);
    for (int k=0;k<KMODES;++k) {
      const T c = (KMODES==1) ? T(0) : (T(k) - T(0.5)*(KMODES-1));
      const T wk = std::exp(-(c*c) / (T(0.7)*T(0.7)));
      weights_[k] = wk;
      wsum += wk;
    }
    for (int k=0;k<KMODES;++k) weights_[k] /= std::max(T(1e-9), wsum);

    const T sigma_total = std::max(T(0), Hs_m) / T(4);
    const T var_total = sigma_total * sigma_total;

    for (int k=0;k<KMODES;++k) {
      const T var_k = weights_[k] * var_total;
      const T om = std::max(T(1e-4), omega_[k]);
      const T ze = std::max(T(1e-3), zeta_[k]);
      const T qk = T(4) * ze * om*om*om * var_k;

      last_f0_hz_ = f0_hz;
      last_Hs_m_  = Hs_m;

      init_var_p_[k] = std::max(T(1e-12), var_k);
      init_var_v_[k] = std::max(T(1e-12), qk / (T(4) * ze * om));

      q_axis_[k].x() = horiz_scale * qk;
      q_axis_[k].y() = horiz_scale * qk;
      q_axis_[k].z() = qk;
    }

    Vec3 var_aw = Vec3::Zero();
    for (int ax=0; ax<3; ++ax) {
      T v = T(0);
      for (int k=0;k<KMODES;++k) {
        const T om = std::max(T(1e-4), omega_[k]);
        const T ze = std::max(T(1e-3), zeta_[k]);
        const T q  = std::max(T(0), q_axis_[k](ax));
        v += (om * q) * (T(1)/(T(4)*ze) + ze);
      }
      var_aw(ax) = std::max(T(0), v);
    }
    Sigma_aw_disabled_world_ = Mat3::Zero();
    Sigma_aw_disabled_world_(0,0) = var_aw.x();
    Sigma_aw_disabled_world_(1,1) = var_aw.y();
    Sigma_aw_disabled_world_(2,2) = var_aw.z();
  }

  // Time update

  void time_update(const Vec3& gyr_body, T Ts) {
    if (!(Ts > T(0)) || !std::isfinite(Ts)) return;
    Ts = std::min(Ts, T(0.1));

    last_dt_ = Ts;

    const Vec3 gyr = deheel_vector_(gyr_body);

    Vec3 bg = Vec3::Zero();
    if constexpr (with_gyro_bias) bg = x_.template segment<3>(OFF_BG);
    last_gyr_bias_corrected_ = gyr - bg;

    const Vec3 omega_b = last_gyr_bias_corrected_;
    if (have_prev_omega_ && Ts > T(0)) {
      const Vec3 alpha_raw = (omega_b - prev_omega_b_) / Ts;
      if (alpha_smooth_tau_ > T(0)) {
        const T a = T(1) - std::exp(-Ts / alpha_smooth_tau_);
        alpha_b_ = (T(1)-a)*alpha_b_ + a*alpha_raw;
      } else {
        alpha_b_ = alpha_raw;
      }
    } else {
      alpha_b_.setZero();
      have_prev_omega_ = true;
    }
    prev_omega_b_ = omega_b;

    qref_ = qref_ * quat_from_delta_theta<T>((omega_b * Ts).eval());
    qref_.normalize();

    F_AA_.setIdentity();
    const Vec3 w = omega_b;
    const T omega = w.norm();
    const T theta = omega * Ts;

    const Mat3 I = Mat3::Identity();
    if (theta < T(1e-5)) {
      const Mat3 Wx = skew3<T>(w);
      F_AA_.template block<3,3>(0,0) = I - Wx*Ts + (Wx*Wx)*(Ts*Ts/T(2));
    } else {
      const Mat3 Wn = skew3<T>(w / (omega + std::numeric_limits<T>::epsilon()));
      const T s = std::sin(theta), c = std::cos(theta);
      F_AA_.template block<3,3>(0,0) = I - s*Wn + (T(1)-c)*(Wn*Wn);
    }
    if constexpr (with_gyro_bias) {
      F_AA_.template block<3,3>(0,3) = -Mat3::Identity() * Ts;
    }

    Q_AA_.setZero();
    if (!use_exact_att_bias_Qd_) {
      Q_AA_.template block<3,3>(0,0) = Qg_ * Ts;
      if constexpr (with_gyro_bias) Q_AA_.template block<3,3>(3,3) = Qbg_ * Ts;
    } else {
      const Mat3 Qg = Qg_;
      Mat3 Qbg = Mat3::Zero();
      if constexpr (with_gyro_bias) Qbg = Qbg_;

      Mat3 I_R;
      if (is_isotropic3_<T>(Qg)) I_R = Mat3::Identity() * (Qg(0,0)*Ts);
      else I_R = simpson_R_Q_RT_<T>(w, Ts, Qg);

      Mat3 I_BB = Mat3::Zero();
      if constexpr (with_gyro_bias) I_BB = simpson_B_Q_BT_<T>(w, Ts, Qbg);

      const Mat3 Qtt = I_R + I_BB;

      Mat3 Qbb = Mat3::Zero();
      if constexpr (with_gyro_bias) Qbb = Qbg * Ts;

      Mat3 Qtb = Mat3::Zero();
      if constexpr (with_gyro_bias) {
        Mat3 IB;
        integral_B_ds_<T>(w, Ts, IB);
        Qtb = IB * Qbg;
      }

      Q_AA_.setZero();
      Q_AA_.template block<3,3>(0,0) = Qtt;
      if constexpr (with_gyro_bias) {
        Q_AA_.template block<3,3>(0,3) = Qtb;
        Q_AA_.template block<3,3>(3,0) = Qtb.transpose();
        Q_AA_.template block<3,3>(3,3) = Qbb;
        Q_AA_ = T(0.5)*(Q_AA_ + Q_AA_.transpose());
        project_psd<T,6>(Q_AA_, T(1e-12));
      } else {
        Q_AA_ = T(0.5)*(Q_AA_ + Q_AA_.transpose());
        project_psd<T,3>(Q_AA_, T(1e-12));
      }
    }

    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      tmp_AA_.noalias() = F_AA_ * Paa;
      Paa.noalias() = tmp_AA_ * F_AA_.transpose();
      Paa.noalias() += Q_AA_;
    }

    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        for (int ax=0; ax<3; ++ax) {
          const T om_k = std::max(T(1e-4), omega_[k]);
          const T ze_k = std::min(std::max(T(1e-4), zeta_[k]), T(5));
          discretize_osc_axis_(Ts, om_k, ze_k, q_axis_[k](ax), Phi2_[ax], Qd2_[ax]);
        }
        Phi6_.setZero();
        Qd6_.setZero();
        for (int ax=0; ax<3; ++ax) {
          Phi6_.template block<2,2>(2*ax,2*ax) = Phi2_[ax];
          Qd6_ .template block<2,2>(2*ax,2*ax) = Qd2_[ax];
          Qd6_ *= T(10);
        }

        Vec3 p = x_.template segment<3>(OFF_Pk(k));
        Vec3 v = x_.template segment<3>(OFF_Vk(k));
        for (int ax=0; ax<3; ++ax) {
          Eigen::Matrix<T,2,1> xv; xv << p(ax), v(ax);
          xv = Phi2_[ax] * xv;
          p(ax) = xv(0);
          v(ax) = xv(1);
        }
        x_.template segment<3>(OFF_Pk(k)) = p;
        x_.template segment<3>(OFF_Vk(k)) = v;

        const int offk = OFF_Pk(k);

        {
          auto Pkk = P_.template block<6,6>(offk, offk);
          tmp6_.noalias() = Phi6_ * Pkk;
          Pkk.noalias() = tmp6_ * Phi6_.transpose();
          Pkk.noalias() += Qd6_;
        }

        {
          auto P_Ak = P_.template block<BASE_N,6>(0, offk);
          tmp_Ak_.noalias() = F_AA_ * P_Ak;
          P_Ak.noalias() = tmp_Ak_ * Phi6_.transpose();
          P_.template block<6,BASE_N>(offk,0) = P_Ak.transpose().eval();
        }

        if constexpr (with_accel_bias) {
          auto P_BAk = P_.template block<3,6>(OFF_BA, offk);
          Eigen::Matrix<T,3,6> tmp_BAk;
          tmp_BAk.noalias() = P_BAk * Phi6_.transpose();
          P_BAk = tmp_BAk;
          P_.template block<6,3>(offk,OFF_BA) = P_BAk.transpose().eval();
        }
      }
    }

    if constexpr (with_accel_bias) {
      auto Pba = P_.template block<3,3>(OFF_BA,OFF_BA);
      if (acc_bias_updates_enabled_) Pba.noalias() += Q_bacc_ * Ts;
      for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), T(1e-12));
      P_.template block<3,3>(OFF_BA,OFF_BA) = Pba;

      auto P_Aba = P_.template block<BASE_N,3>(0,OFF_BA);
      Eigen::Matrix<T,BASE_N,3> tmp_Aba;
      tmp_Aba.noalias() = F_AA_ * P_Aba;
      P_Aba = tmp_Aba;
      P_.template block<3,BASE_N>(OFF_BA,0) = P_Aba.transpose();
    }

    symmetrize_P_();
    clamp_P_diag_(T(1e-15));
    symmetrize_P_();
  }

  // Measurement update: accelerometer (Joseph)
  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref) {
    last_acc_ = MeasDiag3{};
    last_acc_.accepted = false;

    const Vec3 z = deheel_vector_(acc_meas_body);
    if (!z.allFinite()) return;

    // Lever arm term (mean only)
    Vec3 lever = Vec3::Zero();
    if (use_imu_lever_arm_) {
      const Vec3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
      lever.noalias() += alpha_b_.cross(r_imu_bprime)
                      +  last_gyr_bias_corrected_.cross(last_gyr_bias_corrected_.cross(r_imu_bprime));
    }

    // Accel-bias handling (mean)
    Vec3 ba_term = Vec3::Zero();
    if constexpr (with_accel_bias) {
      const Vec3 temp_term = k_a_ * (tempC - tempC_ref);
      if (acc_bias_updates_enabled_) {
        const Vec3 ba0 = x_.template segment<3>(OFF_BA);
        ba_term = ba0 + temp_term;     // Mode A
      } else {
        ba_term = temp_term;           // Mode B
      }
    }

    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 aw = wave_world_accel_();
    const Mat3 Rwb = R_wb();

    // Predicted specific force (BODY')
    const Vec3 f_cog_b = Rwb * (aw - g_world);
    const Vec3 f_pred  = f_cog_b + lever + ba_term;

    const Vec3 r = z - f_pred;
    last_acc_.r = r;

    // ------------------------------------------------------------
    // Jacobians: measurement Jacobians J = ∂h/∂x, with r = z - h
    // ------------------------------------------------------------

    // Attitude Jacobian (BODY' residual):
    // δ( Rwb * u ) = -[Rwb*u]x δθ  =>  J_att = ∂f_pred/∂δθ = -[f_cog_b]x
    const Mat3 J_att = -skew3<T>(f_cog_b);

    // Accel bias Jacobian (Mode A only): f_pred has +b_a => J_ba = +I
    const Mat3 J_ba = Mat3::Identity();

    // Wave Jacobian (only if wave block enabled)
    Eigen::Matrix<T,3,WAVE_N> Jw;
    Jw.setZero();
    if (wave_block_enabled_) {
      for (int k=0; k<KMODES; ++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];

        // aw includes -(om^2) p_k -(2 ζ om) v_k  (WORLD)
        // f_pred = Rwb*(aw-g) => ∂f/∂p_k = Rwb * (-(om^2)I) = -(om^2)Rwb
        // similarly for v_k: ∂f/∂v_k = -(2 ζ om)Rwb
        const Mat3 Jp = -(om*om) * Rwb;
        const Mat3 Jv = -(T(2)*ze*om) * Rwb;

        Jw.template block<3,3>(0, 6*k + 0) = Jp;
        Jw.template block<3,3>(0, 6*k + 3) = Jv;
      }
    }

    // ------------------------------------------------------------
    // Innovation covariance S = R + J P J^T (+ special-mode terms)
    // ------------------------------------------------------------

    Mat3& S = S_scratch_;
    S = Racc_;

    if (!wave_block_enabled_) {
      S.noalias() += Rwb * Sigma_aw_disabled_world_ * Rwb.transpose();
    }

    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH, OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    if constexpr (with_accel_bias) {
      const Mat3 Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
      if (acc_bias_updates_enabled_) {
        const Mat3 Ptba = P_.template block<3,3>(OFF_DTH, OFF_BA);
        const Mat3 X = J_att * Ptba * J_ba.transpose();
        S.noalias() += X + X.transpose();
        S.noalias() += J_ba * Pba * J_ba.transpose(); // == Pba
      } else {
        // Mode B: treat BA uncertainty as nuisance noise only
        S.noalias() += Pba;
      }
    }

    if (wave_block_enabled_) {
      const auto Pww = P_.template block<WAVE_N, WAVE_N>(OFF_WAVE, OFF_WAVE);
      S.noalias() += Jw * Pww * Jw.transpose();

      const auto Ptw = P_.template block<3, WAVE_N>(OFF_DTH, OFF_WAVE);
      const Mat3 AW = J_att * Ptw * Jw.transpose();
      S.noalias() += AW + AW.transpose();

      if constexpr (with_accel_bias) {
        if (acc_bias_updates_enabled_) {
          const auto Pbaw = P_.template block<3, WAVE_N>(OFF_BA, OFF_WAVE);
          const Mat3 BW = J_ba * Pbaw * Jw.transpose();
          S.noalias() += BW + BW.transpose();
        }
      }

      // BG terms intentionally omitted (lever-arm mean uses cached omega, so J_bg forced 0 by design)
    }

    S = T(0.5) * (S + S.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));

    Eigen::LDLT<Mat3> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6)*(S.norm()+T(1)));
      S.diagonal().array() += bump;
      ldlt.compute(S);
      if (ldlt.info() != Eigen::Success) return;
    }

    last_acc_.S = S;
    {
      const Vec3 sol = ldlt.solve(r);
      const T nis = r.dot(sol);
      last_acc_.nis = std::isfinite(nis) ? nis : std::numeric_limits<T>::quiet_NaN();
    }

    const T nis_gate = T(50.0);
    if (!(last_acc_.nis < nis_gate)) return;

    // ------------------------------------------------------------
    // PCt = P J^T  (use SAME J blocks as in S)
    // ------------------------------------------------------------

    MatX3& PCt = PCt_scratch_;
    PCt.setZero();

    PCt.noalias() += P_.template block<NX,3>(0, OFF_DTH) * J_att.transpose();

    if constexpr (with_accel_bias) {
      if (acc_bias_updates_enabled_) {
        PCt.noalias() += P_.template block<NX,3>(0, OFF_BA) * J_ba.transpose();
      }
    }

    if (wave_block_enabled_) {
      PCt.noalias() += P_.template block<NX, WAVE_N>(0, OFF_WAVE) * Jw.transpose();
    } else {
      freeze_wave_rows_(PCt);
    }

    // ------------------------------------------------------------
    // Gain + state/cov update
    // ------------------------------------------------------------

    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());
    if (!(K.allFinite() && r.allFinite())) return;

    if (!wave_block_enabled_) freeze_wave_rows_(K);

    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) {
        K.template block<3,3>(OFF_BA,0).setZero();
      }
    }

    x_.noalias() += K * r;
    if (!x_.allFinite()) return;

    joseph_update3_(K, S, PCt);
    applyQuaternionCorrectionFromErrorState_();

    last_acc_.accepted = true;
  }

  // Measurement update: magnetometer (robust tilt-compensated direction-only)
  void measurement_update_mag_only(const Vec3& mag_meas_body) {
    last_mag_ = MeasDiag3{};
    last_mag_.accepted = false;

    if constexpr (!with_mag) {
      (void)mag_meas_body;
      return;
    }

    const Vec3 mag_meas = deheel_vector_(mag_meas_body);
    if (!mag_meas.allFinite()) return;

    const T n = mag_meas.norm();
    if (!(n > T(1e-6))) return;

    const Mat3 R = R_wb();

    // Predicted field in BODY'
    const Vec3 b_pred = R * B_world_ref_;

    // Predicted down direction in BODY' (world is NED, +Z down)
    Vec3 d = R * Vec3(T(0), T(0), T(1));
    const T dn = d.norm();
    if (!(dn > T(1e-9))) return;
    d /= dn;

    const Mat3 I = Mat3::Identity();
    const Mat3 P_h = I - d * d.transpose();

    Vec3 m_proj = P_h * mag_meas;
    Vec3 u      = P_h * b_pred;      // u = horizontal component of predicted

    const T mnorm = m_proj.norm();
    const T un    = u.norm();
    if (!(mnorm > T(1e-6) && un > T(1e-6))) return;

    Vec3 z_dir = m_proj / mnorm;   // measurement (direction only)
    Vec3 h_dir = u / un;           // prediction  (direction only)

    // 180° ambiguity resolution: choose branch closer to measurement
    bool flipped = false;
    if (z_dir.dot(h_dir) < T(0)) {
      h_dir = -h_dir;
      flipped = true;
    }

    const Vec3 r = z_dir - h_dir;
    last_mag_.r = r;

    // finite-difference ∂h_dir/∂δθ
    // Matches your correction convention: qref_ = qref_ * exp(δθ)
    // and keeps the same 180° branch as the nominal (post-flip) h_dir.
    Mat3 J_att;
    {
      const Vec3 h_nom = h_dir;               // already includes the flip choice
      const T eps = T(1e-4);                  // rad (tune 1e-5..1e-3 if desired)
    
      auto hdir_from_q = [&](const Eigen::Quaternion<T>& q)->Vec3 {
        const Mat3 Rq = q.toRotationMatrix();
    
        // Predicted field and down in BODY'
        Vec3 b = Rq * B_world_ref_;
        Vec3 dd = Rq * Vec3(T(0), T(0), T(1));
        const T dn2 = dd.squaredNorm();
        if (!(dn2 > T(1e-12))) return h_nom;
        dd /= std::sqrt(dn2);
    
        const Mat3 Ph = Mat3::Identity() - dd * dd.transpose();
    
        Vec3 u2 = Ph * b;
        const T un2 = u2.norm();
        if (!(un2 > T(1e-8))) return h_nom;
    
        Vec3 h2 = u2 / un2;
    
        // Keep the same branch as nominal to avoid sign flips in the Jacobian
        if (h2.dot(h_nom) < T(0)) h2 = -h2;
    
        return h2;
      };
    
      for (int i = 0; i < 3; ++i) {
        Vec3 dth = Vec3::Zero();
        dth(i) = eps;
    
        Eigen::Quaternion<T> qp = qref_ * quat_from_delta_theta<T>( dth);
        Eigen::Quaternion<T> qm = qref_ * quat_from_delta_theta<T>(-dth);
        qp.normalize();
        qm.normalize();
    
        const Vec3 hp = hdir_from_q(qp);
        const Vec3 hm = hdir_from_q(qm);
    
        J_att.col(i) = (hp - hm) / (T(2) * eps);
      }
    }

    // Direction-domain noise scaling
    Mat3& S = S_scratch_;
    const T var_raw_mean = (Rmag_(0,0) + Rmag_(1,1) + Rmag_(2,2)) / T(3);
    const T denom = std::max(T(1e-12), mnorm*mnorm);
    const T var_dir = std::max(T(1e-18), var_raw_mean / denom);
    S = I * var_dir;

    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH,OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    S = T(0.5)*(S + S.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));

    Eigen::LDLT<Mat3> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6)*(S.norm()+T(1)));
      S.diagonal().array() += bump;
      ldlt.compute(S);
      if (ldlt.info() != Eigen::Success) return;
    }

    last_mag_.S = S;
    {
      const Vec3 sol = ldlt.solve(r);
      const T nis = r.dot(sol);
      last_mag_.nis = std::isfinite(nis) ? nis : std::numeric_limits<T>::quiet_NaN();
    }

    MatX3& PCt = PCt_scratch_;
    PCt.setZero();
    PCt.noalias() += P_.template block<NX,3>(0,OFF_DTH) * J_att.transpose();

    if (!wave_block_enabled_) freeze_wave_rows_(PCt);

    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) PCt.template block<3,3>(OFF_BA,0).setZero();
    }

    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());
    if (!K.allFinite() || !r.allFinite()) return;

    if (!wave_block_enabled_) freeze_wave_rows_(K);
    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) K.template block<3,3>(OFF_BA,0).setZero();
    }

    x_.noalias() += K * r;
    if (!x_.allFinite()) return;

    joseph_update3_(K, S, PCt);
    applyQuaternionCorrectionFromErrorState_();

    last_mag_.accepted = true;
  }

  // Pseudo-measurement on total WORLD-frame wave displacement:
  void measurement_update_position_pseudo(const Vec3& p_meas_world,
                                          const Mat3& Rpos_world)
  {
    if (!wave_block_enabled_) return;

    const Vec3 p_pred = get_position();
    const Vec3 r = p_meas_world - p_pred;
    if (!r.allFinite()) return;

    Mat3& S = S_scratch_;
    S = T(0.5) * (Rpos_world + Rpos_world.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));
    project_psd<T,3>(S, T(1e-18));
    S = T(0.5) * (S + S.transpose());

    for (int k=0;k<KMODES;++k) {
      const int opk = OFF_Pk(k);
      for (int m=0;m<KMODES;++m) {
        const int opm = OFF_Pk(m);
        S.noalias() += P_.template block<3,3>(opk, opm);
      }
    }

    S = T(0.5) * (S + S.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));

    Eigen::LDLT<Mat3> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6)*(S.norm()+T(1)));
      S.diagonal().array() += bump;
      ldlt.compute(S);
      if (ldlt.info() != Eigen::Success) return;
    }

    // PCt = P H^T, where H sums p_k (WORLD) blocks with +I
    MatX3& PCt = PCt_scratch_;
    PCt.setZero();
    for (int k=0;k<KMODES;++k) {
      const int opk = OFF_Pk(k);
      PCt.noalias() += P_.template block<NX,3>(0, opk);
    }

    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());
    if (!K.allFinite()) return;

    // Only update wave states
    K.template block<BASE_N,3>(0,0).setZero();
    if constexpr (with_accel_bias) K.template block<3,3>(OFF_BA,0).setZero();

    x_.noalias() += K * r;
    if (!x_.allFinite()) return;

    joseph_update3_(K, S, PCt);
  }

  void measurement_update_position_pseudo(const Vec3& sigma_p_world,
                                         const Vec3& p_meas_world = Vec3::Zero())
  {
    Mat3 R = Mat3::Zero();
    const Vec3 s = sigma_p_world.array().max(T(0)).matrix();
    R(0,0) = std::max(T(1e-12), s.x()*s.x());
    R(1,1) = std::max(T(1e-12), s.y()*s.y());
    R(2,2) = std::max(T(1e-12), s.z()*s.z());
    measurement_update_position_pseudo(p_meas_world, R);
  }

private:
  // Constants / state
  const T gravity_magnitude_ = T(STD_GRAVITY);

  Eigen::Quaternion<T> qref_;        // WORLD -> BODY'
  Vec3 B_world_ref_ = Vec3::UnitX(); // mag ref in WORLD (if enabled)

  VecX x_;
  MatX P_;

  // Base process noises
  Mat3 Qg_  = Mat3::Identity()*T(1e-6);
  Mat3 Qbg_ = Mat3::Identity()*T(1e-11);

  // Acc bias
  T sigma_bacc0_ = T(0.004);
  Mat3 Q_bacc_ = Mat3::Identity()*T(1e-6);
  Vec3 k_a_ = Vec3::Constant(T(0.002));

  // Measurement noise
  Mat3 Racc_ = Mat3::Identity()*T(0.04);
  Mat3 Rmag_ = Mat3::Identity()*T(1.0);

  // Wave params
  T omega_[KMODES]{};
  T zeta_[KMODES]{};
  T weights_[KMODES]{};
  Vec3 q_axis_[KMODES]{};

  bool wave_block_enabled_ = false;
  bool warmup_mode_ = true;
  bool acc_bias_updates_enabled_ = true;
  bool use_exact_att_bias_Qd_ = true;

  Mat3 Sigma_aw_disabled_world_ = Mat3::Identity() * T(0.0);

  // Warmup exit detection
  T exit_min_distance_ = T(50.0);
  T exit_min_time_     = T(8.0);
  Vec3 vel_detect_ = Vec3::Zero();
  T pseudo_motion_dist_ = T(0);
  T pseudo_motion_time_ = T(0);

  // Optional gyro bias learning during warmup (gated)
  Vec3 bg_accum_ = Vec3::Zero();
  int  bg_accum_count_ = 0;
  int  warmup_bias_min_samples_ = 150;
  T    warmup_gyro_stationary_thr_ = T(0.03);
  T    warmup_acc_stationary_thr_  = T(0.35);

  // Wave init priors derived from Hs/params (per-mode steady-state)
  T last_f0_hz_ = T(0.12);
  T last_Hs_m_  = T(1.0);
  T init_var_p_[KMODES]{};
  T init_var_v_[KMODES]{};

  // Lever arm caches
  bool use_imu_lever_arm_ = false;
  Vec3 r_imu_wrt_cog_body_phys_ = Vec3::Zero();

  Vec3 prev_omega_b_ = Vec3::Zero();
  Vec3 alpha_b_      = Vec3::Zero();
  bool have_prev_omega_ = false;

  Vec3 last_gyr_bias_corrected_ = Vec3::Zero();
  T last_dt_ = T(1.0/240);
  T alpha_smooth_tau_ = T(0.05);

  // Heel / de-heel
  T wind_heel_rad_ = T(0);
  T cos_unheel_x_  = T(1);
  T sin_unheel_x_  = T(0);

  // Diags
  MeasDiag3 last_acc_;
  MeasDiag3 last_mag_;

  // Scratch
  Eigen::Matrix<T,BASE_N,BASE_N> F_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> Q_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> tmp_AA_;
  Eigen::Matrix<T,BASE_N,6>      tmp_Ak_;

  Eigen::Matrix<T,6,6> Phi6_;
  Eigen::Matrix<T,6,6> Qd6_;
  Eigen::Matrix<T,6,6> tmp6_;
  Eigen::Matrix<T,2,2> Phi2_[3];
  Eigen::Matrix<T,2,2> Qd2_[3];

  Mat3   S_scratch_;
  MatX3  PCt_scratch_;
  MatX3  K_scratch_;

private:
  // Frame helpers
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); }

  void symmetrize_P_() {
    for (int i=0;i<NX;++i) {
      for (int j=i+1;j<NX;++j) {
        const T v = T(0.5)*(P_(i,j) + P_(j,i));
        P_(i,j) = v;
        P_(j,i) = v;
      }
    }
  }

  void clamp_P_diag_(T min_diag) {
    for (int i=0;i<NX;++i) {
      if (!std::isfinite(P_(i,i)) || P_(i,i) < min_diag) P_(i,i) = min_diag;
    }
  }

  void applyQuaternionCorrectionFromErrorState_() {
    Vec3 dth = x_.template segment<3>(OFF_DTH);

    const T max_dth = T(0.25);
    const T n2 = dth.squaredNorm();
    if (n2 > max_dth * max_dth) {
      const T n = std::sqrt(n2);
      dth *= (max_dth / n);
      x_.template segment<3>(OFF_DTH) = dth;
    }

    const Eigen::Quaternion<T> corr = quat_from_delta_theta<T>(dth);
    qref_ = qref_ * corr;
    qref_.normalize();

    x_.template segment<3>(OFF_DTH).setZero();
  }

  Vec3 wave_world_accel_() const {
    if (!wave_block_enabled_) return Vec3::Zero();
    Vec3 aw = Vec3::Zero();
    for (int k=0;k<KMODES;++k) {
      const T om = omega_[k];
      const T ze = zeta_[k];
      const Vec3 p = x_.template segment<3>(OFF_Pk(k));
      const Vec3 v = x_.template segment<3>(OFF_Vk(k));
      aw.noalias() += -(om*om)*p - (T(2)*ze*om)*v;
    }
    return aw;
  }

  template<typename Derived>
  EIGEN_STRONG_INLINE void freeze_wave_rows_(Eigen::MatrixBase<Derived>& M) const {
    M.template block<WAVE_N,3>(OFF_WAVE,0).setZero();
  }

  EIGEN_STRONG_INLINE void joseph_update3_(const MatX3& K, const Mat3& S, const MatX3& PCt) {
    for (int i=0;i<NX;++i) {
      for (int j=i;j<NX;++j) {
        T KHP_ij = T(0), KHP_ji = T(0);
        for (int l=0;l<3;++l) {
          const T Ki_l = K(i,l);
          const T Kj_l = K(j,l);
          const T Pj_l = PCt(j,l);
          const T Pi_l = PCt(i,l);
          KHP_ij += Ki_l * Pj_l;
          if (j != i) KHP_ji += Kj_l * Pi_l;
        }
        if (j==i) KHP_ji = KHP_ij;

        T KSK_ij = T(0);
        for (int a=0;a<3;++a) {
          const T Kia = K(i,a);
          for (int b=0;b<3;++b) {
            KSK_ij += Kia * S(a,b) * K(j,b);
          }
        }
        const T delta = -(KHP_ij + KHP_ji) + KSK_ij;
        P_(i,j) += delta;
        if (j!=i) P_(j,i) = P_(i,j);
      }
    }
    symmetrize_P_();
  }

  // Heel / de-heel

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

  EIGEN_STRONG_INLINE Vec3 deheel_vector_(const Vec3& v_body) const {
    if (std::abs(wind_heel_rad_) < T(1e-9)) return v_body;
    Vec3 v;
    v.x() = v_body.x();
    v.y() = cos_unheel_x_ * v_body.y() - sin_unheel_x_ * v_body.z();
    v.z() = sin_unheel_x_ * v_body.y() + cos_unheel_x_ * v_body.z();
    return v;
  }

  EIGEN_STRONG_INLINE Mat3 Rx_(T angle_rad) const {
    const T c = std::cos(angle_rad);
    const T s = std::sin(angle_rad);
    Mat3 R;
    R << T(1), T(0), T(0),
         T(0),    c,   -s,
         T(0),    s,    c;
    return R;
  }

  void retarget_bodyprime_frame_(T delta_heel_rad) {
    if (std::abs(delta_heel_rad) < T(1e-12)) return;

    const Mat3 R = Rx_(-delta_heel_rad);

    const Eigen::Quaternion<T> qR(R);
    qref_ = qR * qref_;
    qref_.normalize();

    x_.template segment<3>(OFF_DTH) = R * x_.template segment<3>(OFF_DTH);
    if constexpr (with_gyro_bias)  x_.template segment<3>(OFF_BG) = R * x_.template segment<3>(OFF_BG);
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = R * x_.template segment<3>(OFF_BA);

    last_gyr_bias_corrected_ = R * last_gyr_bias_corrected_;
    prev_omega_b_            = R * prev_omega_b_;
    alpha_b_                 = R * alpha_b_;

    MatX Tm = MatX::Identity();
    Tm.template block<3,3>(OFF_DTH,OFF_DTH) = R;
    if constexpr (with_gyro_bias)  Tm.template block<3,3>(OFF_BG,OFF_BG) = R;
    if constexpr (with_accel_bias) Tm.template block<3,3>(OFF_BA,OFF_BA) = R;
    P_ = (Tm * P_ * Tm.transpose()).eval();
    symmetrize_P_();
    clamp_P_diag_(T(1e-15));
    symmetrize_P_();
  }

  // Oscillator discretization

  static inline void phi_osc_2x2_(T t, T w, T z, Eigen::Matrix<T,2,2>& Phi) {
    const T om = std::max(T(1e-7), w);
    const T ze = std::max(T(0), z);
    const T eps = T(1e-7);

    const T a = ze * om;

    if (std::abs(ze - T(1)) < T(1e-3)) {
      const T e = std::exp(-a*t);
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * t;
      Phi(1,0) = e * (-(om*om)*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }

    if (ze < T(1)) {
      const T wd = om * std::sqrt(std::max(T(0), T(1) - ze*ze));
      if (wd < eps) {
        const T e = std::exp(-a*t);
        Phi(0,0) = e * (T(1) + a*t);
        Phi(0,1) = e * t;
        Phi(1,0) = e * (-(om*om)*t);
        Phi(1,1) = e * (T(1) - a*t);
        return;
      }

      const T e = std::exp(-a*t);
      const T c = std::cos(wd*t);
      const T s = std::sin(wd*t);

      const T inv_wd = T(1)/wd;
      const T a_over_wd = a*inv_wd;

      Phi(0,0) = e * (c + a_over_wd*s);
      Phi(0,1) = e * (inv_wd*s);
      Phi(1,0) = e * (-(om*om)*inv_wd*s);
      Phi(1,1) = e * (c - a_over_wd*s);
      return;
    }

    const T srt = std::sqrt(std::max(T(0), ze*ze - T(1)));
    const T r1  = -om*(ze - srt);
    const T r2  = -om*(ze + srt);

    const T denom = (r2 - r1);
    if (std::abs(denom) < eps) {
      const T e = std::exp(-a*t);
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * t;
      Phi(1,0) = e * (-(om*om)*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }

    const T e1 = std::exp(r1*t);
    const T dr = (r2 - r1) * t;
    const T e21 = std::exp(dr);
    const T e2  = e1 * e21;

    const T invd = T(1)/denom;
    const T e21m1 = std::expm1(dr);
    const T e2_minus_e1 = e1 * e21m1;

    Phi(0,0) = (r2*e1 - r1*e2) * invd;
    Phi(0,1) = (e2_minus_e1) * invd;
    Phi(1,0) = (r1*r2) * (-e2_minus_e1) * invd;
    Phi(1,1) = (r2*e2 - r1*e1) * invd;
  }

  static inline void discretize_osc_axis_(T dt, T w, T z, T q,
                                         Eigen::Matrix<T,2,2>& Phi,
                                         Eigen::Matrix<T,2,2>& Qd)
  {
    phi_osc_2x2_(dt, w, z, Phi);

    const T dt_abs = std::abs(dt);
    if (dt_abs < T(5e-4)) {
      const T dt2 = dt * dt;
      const T dt3 = dt2 * dt;
      Qd.setZero();
      Qd(0,0) = q * (dt3 / T(3));
      Qd(0,1) = q * (dt2 / T(2));
      Qd(1,0) = Qd(0,1);
      Qd(1,1) = q * dt_abs;
      return;
    }

    q = std::max(T(0), q);

    auto uuT = [&](T t)->Eigen::Matrix<T,2,2> {
      Eigen::Matrix<T,2,2> Pt;
      phi_osc_2x2_(t, w, z, Pt);
      const T u0 = Pt(0,1);
      const T u1 = Pt(1,1);
      Eigen::Matrix<T,2,2> M;
      M(0,0) = u0*u0;
      M(0,1) = u0*u1;
      M(1,0) = u1*u0;
      M(1,1) = u1*u1;
      return M;
    };

    const Eigen::Matrix<T,2,2> f0 = uuT(T(0));
    const Eigen::Matrix<T,2,2> fm = uuT(T(0.5)*dt);
    const Eigen::Matrix<T,2,2> f1 = uuT(dt);

    Qd = (dt / T(6)) * (f0 + T(4)*fm + f1) * q;

    Qd = T(0.5) * (Qd + Qd.transpose());
    for (int i=0;i<2;++i) Qd(i,i) = std::max(Qd(i,i), T(0));
    project_psd<T,2>(Qd, T(1e-18));
    Qd = T(0.5) * (Qd + Qd.transpose());
  }
};
