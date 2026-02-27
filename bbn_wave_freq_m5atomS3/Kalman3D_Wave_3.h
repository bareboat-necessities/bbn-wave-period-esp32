#pragma once
/*
  Copyright (c) 2026  Mikhail Grushinskiy

  Kalman3D_Wave_3

  Same state + measurement interface as Kalman3D_Wave_2, BUT:
    - Spectrum adaptation is treated as COMMANDS/OBSERVATIONS, not truth.
    - Wave frequency + amplitude are modeled with RANDOM-WALK priors (log-domain).
    - External updates (fp/fc/mode centers/qz) are fused robustly (no confidence input required).
    - q-axis process intensities are DERIVED from latent displacement variances:
          var_p = q / (4*zeta*omega^3)  =>  q = 4*zeta*omega^3*var_p
    - No state/covariance rescale is performed on spectrum-driven updates.

  Core wave state (p,v) and MEKF attitude updates remain unchanged.
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
#include <array>

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

  T w, k;
  if (theta < T(1e-2)) {
    const T t2 = theta * theta;
    const T t4 = t2 * t2;

    w = T(1);
    w = std::fma(-t2, T(1)/T(8), w);
    w = std::fma( t4, T(1)/T(384), w);

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

  if constexpr (N <= 16) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,N,N>> es(S);
    if (es.info() != Eigen::Success) {
      S.diagonal().array() += eps;
      S = T(0.5) * (S + S.transpose());
      return;
    }
    Eigen::Matrix<T,N,1> lam = es.eigenvalues();
    for (int i=0;i<N;++i) if (!(lam(i) > T(0))) lam(i) = eps;
    S = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
    S = T(0.5) * (S + S.transpose());
    return;
  }

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
class Kalman3D_Wave_3 {

  static_assert(KMODES >= 1, "KMODES must be >= 1");

  static constexpr int BASE_N = with_gyro_bias ? 6 : 3;
  static constexpr int WAVE_N = 6 * KMODES;
  static constexpr int BA_N   = with_accel_bias ? 3 : 0;
  static constexpr int NX     = BASE_N + WAVE_N + BA_N;

  // Per-axis independent block dimension:
  // [ δθ_axis(1), (bg_axis 1 optional), (p_k_axis,v_k_axis)*K, (ba_axis 1 optional) ]
  static constexpr int AX_N = 1 + (with_gyro_bias ? 1 : 0) + (2 * KMODES) + (with_accel_bias ? 1 : 0);

  static constexpr T STD_GRAVITY = T(9.80665);
  static constexpr T tempC_ref   = T(35.0);

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

  Kalman3D_Wave_3(const Vec3& sigma_a_meas,
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

    // Measurement noise (DIAGONAL)
    Racc_ = sigma_a_meas.array().square().matrix().asDiagonal();
    Rmag_ = sigma_m_meas.array().square().matrix().asDiagonal();

    // Base process noise (DIAGONAL)
    Qg_  = sigma_g_rw.array().square().matrix().asDiagonal();
    if constexpr (with_gyro_bias) Qbg_ = Mat3::Identity() * b0;

    // Seed base covariance (DIAGONAL)
    P_.template block<3,3>(OFF_DTH, OFF_DTH) = Mat3::Identity() * Pq0;
    if constexpr (with_gyro_bias) P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * Pb0;

    // Seed accel bias (DIAGONAL)
    if constexpr (with_accel_bias) {
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * (sigma_bacc0_*sigma_bacc0_);
    }

    // Seed wave covariance (DIAGONAL per axis)
    const T sigma_p0 = T(20.0);
    const T sigma_v0 = T(1.0);
    for (int k=0;k<KMODES;++k) {
      P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
      P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
    }

    // Mag reference default
    B_world_ref_ = Vec3::UnitX();

    // Default broadband wave params (also initializes latent RW param states)
    set_broadband_params(T(0.12), T(1.0));

    use_exact_att_bias_Qd_ = true;

    // Warmup ON by default
    set_warmup_mode(true);

    update_unheel_trig_();
    enforce_axis_independence_P_();
  }

  // Accessors / diags

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

  const MeasDiag3& lastAccDiag()     const noexcept { return last_acc_; }
  const MeasDiag3& lastMagDiag()     const noexcept { return last_mag_; }
  const MeasDiag3& lastUprightDiag() const noexcept { return last_upright_; }

  // -----------------------------
  // Config
  // -----------------------------

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
      enforce_axis_independence_P_();
    }
  }

  void set_initial_acc_bias(const Vec3& b0) {
    if constexpr (with_accel_bias) {
      x_.template segment<3>(OFF_BA) = b0;
      enforce_axis_independence_P_();
    }
  }

  // Set accelerometer measurement noise (BODY' frame), as per-axis std-dev.
  void set_Racc(const Vec3& sigma_a_meas) {
    const Vec3 s = sigma_a_meas.array().max(T(0.05)).matrix();
    Racc_ = s.array().square().matrix().asDiagonal();
    for (int i=0;i<3;++i) Racc_(i,i) = std::max(Racc_(i,i), T(1e-8));
  }

  // Set full covariance directly (DIAGONAL ONLY; cross-axis terms discarded).
  void set_Racc(const Mat3& Racc) {
    const T min_var = T(0.05) * T(0.05);
    Racc_.setZero();
    Racc_(0,0) = std::max(min_var, Racc(0,0));
    Racc_(1,1) = std::max(min_var, Racc(1,1));
    Racc_(2,2) = std::max(min_var, Racc(2,2));
    for (int i=0;i<3;++i) Racc_(i,i) = std::max(Racc_(i,i), T(1e-8));
  }

  void set_warmup_stationary_thresholds(T gyro_rad_s, T acc_motion_m_s2) {
    warmup_gyro_stationary_thr_ = std::max(T(0), gyro_rad_s);
    warmup_acc_stationary_thr_  = std::max(T(0), acc_motion_m_s2);
  }

  void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }

  // marginalization for disabled wave block (DIAGONAL ONLY; cross-axis terms discarded)
  void set_disabled_wave_accel_cov_world(const Mat3& Sigma_aw_world) {
    Sigma_aw_disabled_world_.setZero();
    Sigma_aw_disabled_world_(0,0) = std::max(T(0), Sigma_aw_world(0,0));
    Sigma_aw_disabled_world_(1,1) = std::max(T(0), Sigma_aw_world(1,1));
    Sigma_aw_disabled_world_(2,2) = std::max(T(0), Sigma_aw_world(2,2));
  }

  void set_disabled_wave_accel_std_world(const Vec3& sigma_aw_world) {
    Mat3 S = Mat3::Zero();
    S(0,0) = std::max(T(0), sigma_aw_world.x()) * std::max(T(0), sigma_aw_world.x());
    S(1,1) = std::max(T(0), sigma_aw_world.y()) * std::max(T(0), sigma_aw_world.y());
    S(2,2) = std::max(T(0), sigma_aw_world.z()) * std::max(T(0), sigma_aw_world.z());
    set_disabled_wave_accel_cov_world(S);
  }

  // Upright (righting) pseudo-measurement controls
  void set_upright_restoring_enabled(bool en) { upright_enabled_ = en; }
  void set_upright_restoring_only_when_acc_rejected(bool en) { upright_only_when_acc_rejected_ = en; }
  void set_upright_restoring_sigma_deg(T sigma_deg) { upright_sigma_deg_ = std::max(T(0), sigma_deg); }
  void set_upright_restoring_nis_gate(T nis_gate)   { upright_nis_gate_   = std::max(T(0), nis_gate); }

  void set_upright_restoring_params(T sigma_deg, T nis_gate, bool only_when_acc_rejected = true) {
    upright_sigma_deg_ = std::max(T(0), sigma_deg);
    upright_nis_gate_  = std::max(T(0), nis_gate);
    upright_only_when_acc_rejected_ = only_when_acc_rejected;
    upright_enabled_ = true;
  }

  // Extra tuning knobs (RMS-focused)
  void set_wave_Q_scale(T s) { wave_Q_scale_ = std::clamp(s, T(0), T(50)); }
  void set_accel_bias_update_scale(T s) { ba_update_scale_ = std::clamp(s, T(0), T(1)); }
  void set_accel_bias_abs_max(T m) { ba_abs_max_ = std::max(T(0), m); }

  // Spectral-command model knobs (NEW in _3)
  // These are in LOG-domain.
  void set_spectrum_command_filter(T rw_sigma_log_omega_per_sqrt_s,
                                   T meas_sigma_log_omega,
                                   T rw_sigma_log_var_per_sqrt_s,
                                   T meas_sigma_log_var)
  {
    Q_log_omega_per_s_ = std::max(T(0), rw_sigma_log_omega_per_sqrt_s * rw_sigma_log_omega_per_sqrt_s);
    R_log_omega_cmd_   = std::max(T(1e-12), meas_sigma_log_omega * meas_sigma_log_omega);
    Q_log_var_per_s_   = std::max(T(0), rw_sigma_log_var_per_sqrt_s * rw_sigma_log_var_per_sqrt_s);
    R_log_var_cmd_     = std::max(T(1e-12), meas_sigma_log_var * meas_sigma_log_var);
  }

  void set_spectrum_fp_fc_command_noise(T meas_sigma_log_f0) {
    R_log_f0_cmd_ = std::max(T(1e-12), meas_sigma_log_f0 * meas_sigma_log_f0);
  }

  // Horizontal/vertical mapping from qk derived from vertical var_p:
  void set_wave_horiz_q_ratio(T r) { horiz_q_ratio_ = std::clamp(r, T(0), T(1)); }
  void set_wave_vert_q_scale(T s)  { vert_q_scale_  = std::max(T(0), s); }

  // -----------------------------
  // Staged init controls
  // -----------------------------

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
    enforce_axis_independence_P_();
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
    enforce_axis_independence_P_();
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
        const T varp = std::max(T(1e-12), init_var_p_[k]) * T(10);
        const T varv = std::max(T(1e-12), init_var_v_[k]) * T(2);
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
    enforce_axis_independence_P_();
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
          enforce_axis_independence_P_();
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
    enforce_axis_independence_P_();
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
    enforce_axis_independence_P_();
  }

  // -----------------------------
  // Wave tuning / spectral commands (NEW model)
  // -----------------------------

  static constexpr int kWaveModes = KMODES;

  void get_wave_mode_freqs_hz(std::array<float, KMODES>& out_f_hz) const {
    for (int k = 0; k < KMODES; ++k) {
      const float omega = float(omega_[k]);
      out_f_hz[k] = omega / (2.0f * float(M_PI));
    }
  }

  void get_wave_mode_zetas(std::array<float, KMODES>& out_zeta) const {
    for (int k = 0; k < KMODES; ++k) out_zeta[k] = float(zeta_[k]);
  }

  void get_wave_mode_qz(std::array<float, KMODES>& out_qz) const {
    for (int k = 0; k < KMODES; ++k) out_qz[k] = float(std::max(T(0), q_axis_[k].z()));
  }

  // COMMAND update from fp/fc + per-mode centers + per-mode qz.
  // - fp/fc may be NaN if not available.
  // - f_hz/qz arrays must be provided; entries may be NaN/<=0 to skip that mode measurement.
  // - horiz_ratio controls XY q as horiz_ratio * qz (clamped 0..1).
  //
  // IMPORTANT: This is NOT a hard set. It updates latent RW priors and rebuilds omega_/q_axis_.
  void apply_spectrum_command(float fp_hz,
                              float fc_hz,
                              const std::array<float, KMODES>& f_hz,
                              const std::array<float, KMODES>& qz,
                              float horiz_ratio,
                              float q_floor = 1e-8f)
  {
    const T hr = std::clamp(T(horiz_ratio), T(0), T(1));
    horiz_q_ratio_ = hr;

    // 1) Update global f0 RW from fp/fc (self-confidence via disagreement)
    const auto valid_f = [&](float f)->bool {
      return std::isfinite(f) && f > 0.02f && f < 5.0f;
    };

    if (valid_f(fp_hz) || valid_f(fc_hz)) {
      T R0 = R_log_f0_cmd_;
      if (valid_f(fp_hz) && valid_f(fc_hz)) {
        const T lratio = std::abs(std::log(T(fp_hz) / T(fc_hz)));
        R0 += (lratio*lratio) * T(1.0); // disagreement inflation (tunable)
        const T fmean = T(0.5) * (T(fp_hz) + T(fc_hz));
        const T y = std::log(std::max(T(1e-6), T(2)*T(M_PI)*fmean));
        robust_kf1d_update_(log_f0_, P_log_f0_, y, R0);
      } else {
        const T f = valid_f(fp_hz) ? T(fp_hz) : T(fc_hz);
        const T y = std::log(std::max(T(1e-6), T(2)*T(M_PI)*f));
        robust_kf1d_update_(log_f0_, P_log_f0_, y, R0);
      }
    }

    // 2) Per-mode frequency + amplitude (var_p) commands.
    // Convert qz command to var_p command using current/commanded omega and current zeta:
    //   var_p_cmd = qz / (4*zeta*omega^3)
    for (int k=0;k<KMODES;++k) {
      // Coupling prior: keep mode centers near broadband spacing around f0 (weakly)
      {
        const T prior = log_f0_ + log_kappa_[k];
        robust_kf1d_update_(log_omega_[k], P_log_omega_[k], prior, R_log_omega_couple_);
      }

      // Mode center command
      if (std::isfinite(f_hz[k]) && f_hz[k] > 0.02f && f_hz[k] < 5.0f) {
        const T y = std::log(std::max(T(1e-6), T(2)*T(M_PI)*T(f_hz[k])));
        robust_kf1d_update_(log_omega_[k], P_log_omega_[k], y, R_log_omega_cmd_);
      }

      // Use the *updated* omega estimate to interpret qz -> var.
      const T omega_est = std::exp(log_omega_[k]);
      const T ze = std::max(T(1e-5), zeta_[k]);

      if (std::isfinite(qz[k]) && qz[k] > 0.0f) {
        const T qcmd = std::max(T(q_floor), T(qz[k]));
        const T denom = std::max(T(1e-12), T(4) * ze * omega_est * omega_est * omega_est);
        const T var_cmd = std::max(T(1e-18), qcmd / denom);
        const T y = std::log(var_cmd);
        robust_kf1d_update_(log_var_p_[k], P_log_var_p_[k], y, R_log_var_cmd_);
      }
    }

    // 3) Materialize omega_/q_axis_ from latent states
    refresh_wave_params_from_latents_();
  }

  // Backwards-compatible call site (SeaStateFusionFilter2 etc):
  // This now behaves as a SPECTRAL COMMAND update (RW), not a hard set.
  void set_wave_mode_freqs_and_qz(const std::array<float, KMODES>& f_hz,
                                  const std::array<float, KMODES>& qz,
                                  float horiz_ratio,
                                  float q_floor = 1e-8f)
  {
    apply_spectrum_command(std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN(),
                           f_hz, qz, horiz_ratio, q_floor);
  }

  // Damping ratios are still explicit; after changing zeta we rebuild q from var_p.
  // (No state/cov rescale in _3.)
  void set_wave_mode_zetas(const std::array<float, KMODES>& zetas) {
    for (int k=0;k<KMODES;++k) {
      T z = T(zetas[k]);
      if (!std::isfinite(z)) z = zeta_[k];
      zeta_[k] = std::clamp(z, T(0.005), T(0.20));
    }
    refresh_wave_params_from_latents_();
  }

  // Hard broadband init (still allowed). Also resets latent RW states to match.
  void set_broadband_params(T f0_hz, T Hs_m, T zeta_mid = T(0.08), T horiz_scale = T(0.35)) {
    const T w0 = T(2)*T(M_PI)*std::max(T(1e-4), f0_hz);

    // mode omega spacing and zetas
    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1));
      const T lo = std::log(T(0.6));
      const T hi = std::log(T(1.7));
      omega_[k] = std::exp(lo + (hi-lo)*u) * w0;

      const T scale = (k==0 || k==KMODES-1) ? T(1.25) : T(1.0);
      zeta_[k] = std::max(T(0.01), zeta_mid * scale);
    }

    // weights for variance split
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

    // initialize latent RW states to match this broadband model
    log_f0_ = std::log(std::max(T(1e-6), w0));
    P_log_f0_ = T(0.30)*T(0.30);

    for (int k=0;k<KMODES;++k) {
      const T var_k = std::max(T(1e-18), weights_[k] * var_total);
      log_var_p_[k] = std::log(var_k);
      P_log_var_p_[k] = T(0.80); // loose

      log_omega_[k] = std::log(std::max(T(1e-6), omega_[k]));
      P_log_omega_[k] = T(0.20)*T(0.20);

      log_kappa_[k] = log_omega_[k] - log_f0_;
    }

    // Set mapping defaults
    horiz_q_ratio_ = std::clamp(T(horiz_scale), T(0), T(1));
    vert_q_scale_ = T(1.0);

    last_f0_hz_ = f0_hz;
    last_Hs_m_  = Hs_m;

    refresh_wave_params_from_latents_();
  }

  // -----------------------------
  // Time update
  // -----------------------------

  void time_update(const Vec3& gyr_body, T Ts) {
    if (!(Ts > T(0)) || !std::isfinite(Ts)) return;
    Ts = std::min(Ts, T(0.1));

    last_dt_ = Ts;

    // RW prior propagation for latent spectral parameters
    propagate_spectral_priors_(Ts);

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

    // Base transition
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

    // Base Q
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

    // Base covariance propagate
    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      tmp_AA_.noalias() = F_AA_ * Paa;
      Paa.noalias() = tmp_AA_ * F_AA_.transpose();
      Paa.noalias() += Q_AA_;
    }

    // Wave propagate
    if (wave_block_enabled_) {
      Eigen::Matrix<T,6,6> Phi6_modes[KMODES];

      for (int k=0;k<KMODES;++k) {
        for (int ax=0; ax<3; ++ax) {
          const T om_k = std::max(T(1e-4), omega_[k]);
          const T ze_k = std::min(std::max(T(1e-4), zeta_[k]), T(5));
          discretize_osc_axis_(Ts, om_k, ze_k, q_axis_[k](ax), Phi2_[ax], Qd2_[ax]);
        }

        Phi6_.setZero();
        Qd6_.setZero();
        for (int ax=0; ax<3; ++ax) {
          const T a = Phi2_[ax](0,0);
          const T b = Phi2_[ax](0,1);
          const T c = Phi2_[ax](1,0);
          const T d = Phi2_[ax](1,1);

          Phi6_(ax,   ax  ) = a;
          Phi6_(ax,   3+ax) = b;
          Phi6_(3+ax, ax  ) = c;
          Phi6_(3+ax, 3+ax) = d;

          const T q00 = Qd2_[ax](0,0);
          const T q01 = Qd2_[ax](0,1);
          const T q11 = Qd2_[ax](1,1);

          Qd6_(ax,   ax  ) = q00;
          Qd6_(ax,   3+ax) = q01;
          Qd6_(3+ax, ax  ) = q01;
          Qd6_(3+ax, 3+ax) = q11;
        }

        Qd6_ *= wave_Q_scale_;
        Phi6_modes[k] = Phi6_;

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

      // Cross-mode covariances
      for (int k = 0; k < KMODES; ++k) {
        const int offk = OFF_Pk(k);
        for (int m = k + 1; m < KMODES; ++m) {
          const int offm = OFF_Pk(m);

          Eigen::Matrix<T,6,6> Pkm = P_.template block<6,6>(offk, offm);
          Pkm = (Phi6_modes[k] * Pkm * Phi6_modes[m].transpose()).eval();

          P_.template block<6,6>(offk, offm) = Pkm;
          P_.template block<6,6>(offm, offk) = Pkm.transpose();
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
    enforce_axis_independence_P_();
  }

  // -----------------------------
  // Measurement updates (unchanged vs your _2 baseline)
  // -----------------------------
  // NOTE: To keep this reply within reason, I am not duplicating your entire
  // measurement_update_upright_only / acc / mag / position_pseudo bodies here.
  // Paste them from your Kalman3D_Wave_2 exactly as-is into this class.
  //
  // The ONLY changes required inside those bodies are: none.
  // (They already use omega_/zeta_/q_axis_ and will automatically benefit.)

  // ---- BEGIN: copy/paste from your Kalman3D_Wave_2 ----
  void measurement_update_upright_only(T sigma_deg_override = T(-1), T nis_gate_override = T(-1));
  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref);
  void measurement_update_mag_only(const Vec3& mag_meas_body);
  void measurement_update_position_pseudo(const Vec3& p_meas_world, const Mat3& Rpos_world);
  void measurement_update_position_pseudo(const Vec3& sigma_p_world, const Vec3& p_meas_world = Vec3::Zero());
  // ---- END: copy/paste ----

private:
  // Constants / state
  const T gravity_magnitude_ = T(STD_GRAVITY);

  Eigen::Quaternion<T> qref_;        // WORLD -> BODY'
  Vec3 B_world_ref_ = Vec3::UnitX(); // mag ref in WORLD (if enabled)

  VecX x_;
  MatX P_;

  // Base process noises (DIAGONAL)
  Mat3 Qg_  = Mat3::Identity()*T(1e-6);
  Mat3 Qbg_ = Mat3::Identity()*T(1e-11);

  // Acc bias (DIAGONAL)
  T sigma_bacc0_ = T(0.004);
  Mat3 Q_bacc_ = Mat3::Identity()*T(1e-6);
  Vec3 k_a_ = Vec3::Constant(T(0.002));

  // Measurement noise (DIAGONAL)
  Mat3 Racc_ = Mat3::Identity()*T(0.04);
  Mat3 Rmag_ = Mat3::Identity()*T(1.0);

  // Wave params (materialized plant params)
  T omega_[KMODES]{};
  T zeta_[KMODES]{};
  T weights_[KMODES]{};
  Vec3 q_axis_[KMODES]{};

  bool wave_block_enabled_ = false;
  bool warmup_mode_ = true;
  bool acc_bias_updates_enabled_ = true;
  bool use_exact_att_bias_Qd_ = true;

  Mat3 Sigma_aw_disabled_world_ = Mat3::Identity() * T(0.0); // DIAGONAL

  // Upright (righting) pseudo-measurement params
  bool upright_enabled_ = true;
  bool upright_only_when_acc_rejected_ = true;
  T    upright_sigma_deg_ = T(8.0);
  T    upright_nis_gate_  = T(50.0);

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

  // Wave init priors derived from Hs/params
  T last_f0_hz_ = T(0.12);
  T last_Hs_m_  = T(1.0);
  T init_var_p_[KMODES]{};
  T init_var_v_[KMODES]{};

  // RMS tuning defaults
  T wave_Q_scale_   = T(1.5);
  T ba_update_scale_= T(0.02);
  T ba_abs_max_     = T(0.08);

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
  MeasDiag3 last_upright_;

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

  // -----------------------------
  // NEW: latent spectral parameter RW model (log-domain)
  // -----------------------------
  // global center: log_f0 = log(omega0)
  T log_f0_ = std::log(T(2)*T(M_PI)*T(0.12));
  T P_log_f0_ = T(0.30)*T(0.30);

  // per-mode frequency: log_omega[k]
  T log_omega_[KMODES]{};
  T P_log_omega_[KMODES]{};

  // per-mode displacement variance: log_var_p[k] (vertical)
  T log_var_p_[KMODES]{};
  T P_log_var_p_[KMODES]{};

  // fixed broadband spacing anchors: log_kappa[k] = log_omega[k] - log_f0
  T log_kappa_[KMODES]{};

  // RW process noise spectral densities (variance per second)
  T Q_log_f0_per_s_    = T(0.05)*T(0.05);
  T Q_log_omega_per_s_ = T(0.06)*T(0.06);
  T Q_log_var_per_s_   = T(0.25)*T(0.25);

  // Measurement noise (log-domain)
  T R_log_f0_cmd_      = T(0.18)*T(0.18);
  T R_log_omega_cmd_   = T(0.20)*T(0.20);
  T R_log_omega_couple_= T(0.35)*T(0.35); // weak coupling to broadband layout
  T R_log_var_cmd_     = T(0.80)*T(0.80);

  // mapping vertical -> XY
  T horiz_q_ratio_ = T(0.35);
  T vert_q_scale_  = T(1.0);

private:
  // Frame helpers
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); }

  // Robust 1D KF update in log-domain (no confidence input required)
  static inline bool robust_kf1d_update_(T& x, T& P, T y, T R0,
                                        T nis_soft = T(9), T nis_hard = T(25))
  {
    if (!std::isfinite(y) || !std::isfinite(x) || !(P > T(0)) || !(R0 > T(0))) return false;
    const T r  = y - x;
    const T S0 = P + R0;
    if (!(S0 > T(0))) return false;
    const T nis0 = (r*r) / S0;
    if (!(nis0 < nis_hard)) return false;

    T R = R0;
    if (nis0 > nis_soft) {
      const T R_needed = (r*r)/nis_soft - P;
      if (R_needed > R) R = R_needed;
    }

    const T S = P + R;
    const T K = P / std::max(T(1e-12), S);
    x = x + K * (y - x);
    P = std::max(T(1e-12), (T(1) - K) * P);
    return true;
  }

  void propagate_spectral_priors_(T dt) {
    // global
    P_log_f0_ = std::max(T(1e-12), P_log_f0_ + Q_log_f0_per_s_ * dt);
    // per-mode
    for (int k=0;k<KMODES;++k) {
      P_log_omega_[k] = std::max(T(1e-12), P_log_omega_[k] + Q_log_omega_per_s_ * dt);
      P_log_var_p_[k] = std::max(T(1e-12), P_log_var_p_[k] + Q_log_var_per_s_ * dt);
    }
  }

  void refresh_wave_params_from_latents_() {
    // materialize omega
    for (int k=0;k<KMODES;++k) {
      const T om = std::exp(std::clamp(log_omega_[k], T(-8), T(8)));
      omega_[k] = std::clamp(om, T(2)*T(M_PI)*T(0.02), T(2)*T(M_PI)*T(5.0));
    }

    // materialize q from var_p + current omega,zeta
    T m0 = T(0);
    for (int k=0;k<KMODES;++k) {
      const T ze = std::max(T(1e-5), zeta_[k]);
      const T om = std::max(T(1e-4), omega_[k]);
      const T var_p = std::max(T(1e-18), std::exp(std::clamp(log_var_p_[k], T(-50), T(50))));

      const T qk = T(4) * ze * om*om*om * var_p;          // [m^2/s^5]
      const T qz = vert_q_scale_ * qk;
      const T qxy = horiz_q_ratio_ * qz;

      q_axis_[k] = Vec3(qxy, qxy, qz);

      init_var_p_[k] = var_p;
      init_var_v_[k] = std::max(T(1e-12), qk / (T(4)*ze*om));

      m0 += var_p;
    }

    last_Hs_m_ = T(4) * std::sqrt(std::max(T(0), m0));
    last_f0_hz_ = std::exp(log_f0_) / (T(2)*T(M_PI));

    on_wave_mode_params_changed_();
  }

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

  // Build index list for axis block (X/Y/Z) in the current state ordering
  void axis_indices_(int ax, std::array<int,AX_N>& idx) const {
    int t = 0;
    idx[t++] = OFF_DTH + ax;
    if constexpr (with_gyro_bias) idx[t++] = OFF_BG + ax;
    for (int k=0;k<KMODES;++k) {
      idx[t++] = OFF_Pk(k) + ax;
      idx[t++] = OFF_Vk(k) + ax;
    }
    if constexpr (with_accel_bias) idx[t++] = OFF_BA + ax;
    (void)t;
  }

  // Zero cross-axis covariances and PSD-project each axis block.
  void enforce_axis_independence_P_() {
    std::array<int,AX_N> idx[3];
    for (int ax=0; ax<3; ++ax) axis_indices_(ax, idx[ax]);

    for (int a=0; a<3; ++a) {
      for (int b=a+1; b<3; ++b) {
        for (int i=0; i<AX_N; ++i) {
          const int ii = idx[a][i];
          for (int j=0; j<AX_N; ++j) {
            const int jj = idx[b][j];
            P_(ii,jj) = T(0);
            P_(jj,ii) = T(0);
          }
        }
      }
    }

    for (int a=0; a<3; ++a) {
      Eigen::Matrix<T,AX_N,AX_N> Pa;
      for (int i=0; i<AX_N; ++i) {
        const int ii = idx[a][i];
        for (int j=0; j<AX_N; ++j) {
          const int jj = idx[a][j];
          Pa(i,j) = P_(ii,jj);
        }
      }

      Pa = T(0.5) * (Pa + Pa.transpose());
      for (int i=0;i<AX_N;++i) Pa(i,i) = std::max(Pa(i,i), T(1e-15));
      project_psd<T,AX_N>(Pa, T(1e-15));
      Pa = T(0.5) * (Pa + Pa.transpose());

      for (int i=0; i<AX_N; ++i) {
        const int ii = idx[a][i];
        for (int j=0; j<AX_N; ++j) {
          const int jj = idx[a][j];
          P_(ii,jj) = Pa(i,j);
        }
      }
    }

    symmetrize_P_();
    clamp_P_diag_(T(1e-15));
    symmetrize_P_();
  }

  EIGEN_STRONG_INLINE void clamp_accel_bias_mean_() {
    if constexpr (with_accel_bias) {
      if (ba_abs_max_ <= T(0)) {
        x_.template segment<3>(OFF_BA).setZero();
        return;
      }
      auto ba = x_.template segment<3>(OFF_BA);
      ba.x() = std::clamp(ba.x(), -ba_abs_max_, ba_abs_max_);
      ba.y() = std::clamp(ba.y(), -ba_abs_max_, ba_abs_max_);
      ba.z() = std::clamp(ba.z(), -ba_abs_max_, ba_abs_max_);
      x_.template segment<3>(OFF_BA) = ba;
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

  // Keep disabled-wave nuisance covariance consistent with latest mode params.
  inline void on_wave_mode_params_changed_() {
    Vec3 var_aw = Vec3::Zero();
    for (int ax = 0; ax < 3; ++ax) {
      T v = T(0);
      for (int k = 0; k < KMODES; ++k) {
        const T om = std::max(T(1e-4), omega_[k]);
        const T ze = std::max(T(1e-3), zeta_[k]);
        const T q  = std::max(T(0), q_axis_[k](ax));
        v += (om * q) * (T(1) / (T(4) * ze) + ze);
      }
      var_aw(ax) = std::max(T(0), v);
    }

    Sigma_aw_disabled_world_.setZero();
    Sigma_aw_disabled_world_(0,0) = var_aw.x();
    Sigma_aw_disabled_world_(1,1) = var_aw.y();
    Sigma_aw_disabled_world_(2,2) = var_aw.z();
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
    clamp_accel_bias_mean_();

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
    enforce_axis_independence_P_();
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
