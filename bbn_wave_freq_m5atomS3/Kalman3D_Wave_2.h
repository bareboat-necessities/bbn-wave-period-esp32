#pragma once
/*
  Kalman3D_Wave_2  (Broadband oscillator wave model + qMEKF attitude)
  ================================================================
  CONSISTENT with your OU Kalman3D_Wave conventions:

  - qref_ stores WORLD -> BODY' (virtual un-heeled frame B')
  - quaternion() returns BODY' -> WORLD (qref_.conjugate())
  - quaternion_boat() returns physical BODY -> WORLD by re-applying heel
  - Right-multiply quaternion updates: qref_ = qref_ ⊗ δq
  - Error-state attitude δθ is applied then cleared
  - Structured base Qd path (Simpson + B(t) integrals) copied from OU

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

  Notes vs OU:
    - OU had [v,p,S,a_w] + pseudo-meas on S to correct drift.
      Oscillators are mean-reverting in p/v, reducing integration drift naturally.

  Template parameters:
    T, KMODES, with_gyro_bias, with_accel_bias, with_mag

  FIXES APPLIED (bugs):
    1) Propagate full wave-wave cross-covariances P(k,j) in time_update (k!=j).
    2) Accel update now accounts for full Pww (including cross-mode) via Jw*Pww*Jwᵀ
       and uses P(:,wave)*Jwᵀ in PCt (no per-mode diagonal-only shortcut).
    3) Correct lever-arm gyro-bias Jacobian sign: d/d(bg) = - d/d(ω).
    4) Avoid M_PI portability issues (use local kPi constant).
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

// ---------------------- Small helpers (consistent with OU) ----------------------

template<typename T>
static inline Eigen::Matrix<T,3,3> skew3(const Eigen::Matrix<T,3,1>& a) {
  Eigen::Matrix<T,3,3> S;
  S << T(0),   -a.z(),  a.y(),
       a.z(),  T(0),   -a.x(),
      -a.y(),  a.x(),   T(0);
  return S;
}

// Full exponential-map correction (Rodrigues in quaternion form). Same as OU.
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

// Helper: project symmetric NxN to PSD (same spirit as OU)
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

// ---------------------- Base (att+bias) exact-ish Qd helpers (from OU) ----------------------

template<typename T>
static inline bool is_isotropic3_(const Eigen::Matrix<T,3,3>& S, T tol = T(1e-9)) {
  const T a=S(0,0), b=S(1,1), c=S(2,2);
  Eigen::Matrix<T,3,3> Off = S; Off.diagonal().setZero();
  const T off = Off.cwiseAbs().sum();
  const T mean = (a+b+c)/T(3);
  return (std::abs(a-mean)+std::abs(b-mean)+std::abs(c-mean)+off) <= tol*(T(1)+std::abs(mean));
}

// Rodrigues rotation and B(t) = -∫_0^t exp(-[ω]× τ) dτ  (consistent with OU)
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

// d/dω of (ω×(ω×r)) = (ω·r) I + ω rᵀ - 2 r ωᵀ  (from OU)
template<typename T>
static inline Eigen::Matrix<T,3,3> d_omega_x_omega_x_r_domega_(const Eigen::Matrix<T,3,1>& w,
                                                               const Eigen::Matrix<T,3,1>& r)
{
  const T s = w.dot(r);
  return Eigen::Matrix<T,3,3>::Identity()*s + (w * r.transpose()) - T(2) * (r * w.transpose());
}

// ---------------------- The filter ----------------------

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
  static constexpr T kPi         = T(3.1415926535897932384626433832795);

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

    // Measurement noise
    Racc_ = sigma_a_meas.array().square().matrix().asDiagonal();
    Rmag_ = sigma_m_meas.array().square().matrix().asDiagonal();

    // Base process noise (same meaning as OU):
    Qg_  = sigma_g_rw.array().square().matrix().asDiagonal();
    if constexpr (with_gyro_bias) Qbg_ = Mat3::Identity() * b0;

    // Seed base covariance
    P_.template block<3,3>(OFF_DTH, OFF_DTH) = Mat3::Identity() * Pq0;
    if constexpr (with_gyro_bias) P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * Pb0;

    // Seed accel bias
    if constexpr (with_accel_bias) {
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * (sigma_bacc0_*sigma_bacc0_);
    }

    // Seed wave covariance (reasonable; warmup will override)
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

    // Defaults consistent with OU toggles
    use_exact_att_bias_Qd_ = true;

    // Warmup ON by default (staged init behavior like OU)
    set_warmup_mode(true);
  }

  // ---------------- Accessors (same convention as OU) ----------------

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

  [[nodiscard]] Vec3 wave_position_world() const {
    Vec3 p=Vec3::Zero();
    for (int k=0;k<KMODES;++k) p += x_.template segment<3>(OFF_Pk(k));
    return p;
  }
  [[nodiscard]] Vec3 wave_velocity_world() const {
    Vec3 v=Vec3::Zero();
    for (int k=0;k<KMODES;++k) v += x_.template segment<3>(OFF_Vk(k));
    return v;
  }
  [[nodiscard]] Vec3 wave_accel_world() const { return wave_world_accel_(); }

  const MeasDiag3& lastAccDiag() const noexcept { return last_acc_; }
  const MeasDiag3& lastMagDiag() const noexcept { return last_mag_; }

  // ---------------- Config (matching OU toggles) ----------------

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

  // Toggle structured Qd for base block (same meaning as OU)
  void set_exact_att_bias_Qd(bool on) { use_exact_att_bias_Qd_ = on; }

  // ---------------- Staged init controls (OU-equivalent concept) ----------------

  void set_wave_block_enabled(bool on) {
    if (wave_block_enabled_ && !on) {
      // Decouple BASE<->WAVE and BA<->WAVE like OU does for linear block
      P_.template block<BASE_N,WAVE_N>(0, OFF_WAVE).setZero();
      P_.template block<WAVE_N,BASE_N>(OFF_WAVE, 0).setZero();
      if constexpr (with_accel_bias) {
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      }
      // Optional: if you want wave fully inert when disabled, also clear wave-wave off-diagonals.
      // P_.template block<WAVE_N,WAVE_N>(OFF_WAVE, OFF_WAVE).setZero();
    }
    wave_block_enabled_ = on;
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

    // OU warmup: disable linear block + freeze BA updates
    set_wave_block_enabled(!on);
    set_acc_bias_updates_enabled(!on);

    if (on) {
      // Freeze wave cov to tiny to prevent cross-driving through residuals
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity()*T(1e-12);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity()*T(1e-12);
      }
      // Safest: lever arm off during warmup (OU did this)
      clear_imu_lever_arm();
      pseudo_motion_dist_ = T(0);
      pseudo_motion_time_ = T(0);
      vel_detect_.setZero();
    } else {
      // Restore reasonable covariances
      const T sigma_p0 = T(20.0);
      const T sigma_v0 = T(1.0);
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity()*(sigma_p0*sigma_p0);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity()*(sigma_v0*sigma_v0);
      }
    }
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

    // crude motion detect in WORLD using current attitude (BODY'->WORLD)
    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 acc_world = R_bw() * acc;
    const Vec3 acc_motion = acc_world - g_world;

    vel_detect_ += acc_motion * dt;
    pseudo_motion_dist_ += vel_detect_.norm() * dt;
    pseudo_motion_time_ += dt;

    if (pseudo_motion_dist_ > exit_min_distance_ && pseudo_motion_time_ > exit_min_time_) {
      set_warmup_mode(false);
    }

    if constexpr (with_gyro_bias) {
      // gentle low-pass of bias estimate during warmup
      const Vec3 gyr = deheel_vector_(gyr_body);
      const T alpha = T(0.01);
      if (bg_accum_count_ == 0) bg_accum_ = gyr;
      else bg_accum_ = (T(1)-alpha)*bg_accum_ + alpha*gyr;
      bg_accum_count_++;
      if (bg_accum_count_ > 100) {
        x_.template segment<3>(OFF_BG) = bg_accum_;
        P_.template block<3,3>(OFF_BG,OFF_BG) = Mat3::Identity()*T(1e-8);
        bg_accum_count_ = 101;
      }
    }
  }

  // ---------------- IMU lever arm (same as OU) ----------------

  void set_imu_lever_arm_body(const Vec3& r_b_phys) {
    r_imu_wrt_cog_body_phys_ = r_b_phys;
    use_imu_lever_arm_ = (r_b_phys.squaredNorm() > T(0));
  }
  void clear_imu_lever_arm() {
    r_imu_wrt_cog_body_phys_.setZero();
    use_imu_lever_arm_ = false;
  }
  void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

  // ---------------- Heel update (same retarget as OU) ----------------

  void update_wind_heel(T heel_rad) {
    const T old = wind_heel_rad_;
    if (heel_rad == old) return;
    retarget_bodyprime_frame_(heel_rad - old);
    wind_heel_rad_ = heel_rad;
    update_unheel_trig_();
  }

  // ---------------- Initialization from acc/mag (OU-style construction) ----------------

  void initialize_from_acc_mag(const Vec3& acc_body, const Vec3& mag_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 mag = deheel_vector_(mag_body);

    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    const Vec3 acc_n = acc / an;

    // WORLD axes expressed in BODY coords (OU-style)
    const Vec3 z_world = -acc_n; // world +Z(down) expressed in body
    Vec3 mag_h = mag - (mag.dot(z_world))*z_world;
    if (!(mag_h.norm() > T(1e-8))) return;
    mag_h.normalize();

    const Vec3 x_world = mag_h;                      // world X (north) in body
    const Vec3 y_world = z_world.cross(x_world).normalized();

    Mat3 R_wb;
    R_wb.col(0) = x_world;
    R_wb.col(1) = y_world;
    R_wb.col(2) = z_world;

    qref_ = Eigen::Quaternion<T>(R_wb); // WORLD->BODY'
    qref_.normalize();

    if constexpr (with_mag) {
      B_world_ref_ = R_bw() * mag; // µT in world
    }

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  void initialize_from_acc(const Vec3& acc_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    const Vec3 anorm = acc / an;
    const Vec3 zb = Vec3::UnitZ();
    const Vec3 target = -anorm;

    T c = std::max(T(-1), std::min(T(1), zb.dot(target)));
    Vec3 axis = zb.cross(target);
    T sn = axis.norm();

    if (sn < T(1e-8)) {
      qref_.setIdentity();
    } else {
      axis /= sn;
      const T ang = std::acos(c);
      qref_ = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(ang, axis));
      qref_.normalize();
    }
    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  // ---------------- Wave tuning ----------------

  void set_broadband_params(T f0_hz, T Hs_m, T zeta_mid = T(0.08), T horiz_scale = T(0.35)) {
    const T w0 = T(2) * kPi * std::max(T(1e-4), f0_hz);

    // log-spaced ω around w0
    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1)); // 0..1
      const T lo = std::log(T(0.6));
      const T hi = std::log(T(1.7));
      omega_[k] = std::exp(lo + (hi-lo)*u) * w0;

      const T scale = (k==0 || k==KMODES-1) ? T(1.25) : T(1.0);
      zeta_[k] = std::max(T(0.01), zeta_mid * scale);
    }

    // weights (soft bell)
    T wsum = T(0);
    for (int k=0;k<KMODES;++k) {
      const T c = (KMODES==1) ? T(0) : (T(k) - T(0.5)*(KMODES-1));
      const T wk = std::exp(-(c*c) / (T(0.7)*T(0.7)));
      weights_[k] = wk;
      wsum += wk;
    }
    for (int k=0;k<KMODES;++k) weights_[k] /= std::max(T(1e-9), wsum);

    // total displacement variance ~ (Hs/4)^2
    const T sigma_total = std::max(T(0), Hs_m) / T(4);
    const T var_total = sigma_total * sigma_total;

    // Driving noise intensity q such that stationary var(p)=var_k approximately:
    // var(p) ≈ q / (4 ζ ω^3)  -> q ≈ 4 ζ ω^3 var(p)
    for (int k=0;k<KMODES;++k) {
      const T var_k = weights_[k] * var_total;
      const T om = std::max(T(1e-4), omega_[k]);
      const T ze = std::max(T(1e-3), zeta_[k]);
      const T qk = T(4) * ze * om*om*om * var_k;

      q_axis_[k].x() = horiz_scale * qk;
      q_axis_[k].y() = horiz_scale * qk;
      q_axis_[k].z() = qk;
    }
  }

  // ---------------- Time update ----------------

  void time_update(const Vec3& gyr_body, T Ts) {
    last_dt_ = Ts;

    // De-heel gyro into B' (OU consistent)
    const Vec3 gyr = deheel_vector_(gyr_body);

    // Bias-corrected ω
    Vec3 bg = Vec3::Zero();
    if constexpr (with_gyro_bias) bg = x_.template segment<3>(OFF_BG);
    last_gyr_bias_corrected_ = gyr - bg;

    // α estimate for lever arm
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

    // Quaternion propagation: right-multiply (OU)
    qref_ = qref_ * quat_from_delta_theta<T>((omega_b * Ts).eval());
    qref_.normalize();

    // -------- Base covariance propagation (match OU style) --------
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

    // P_AA = F P F^T + Q
    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      tmp_AA_.noalias() = F_AA_ * Paa;
      Paa.noalias() = tmp_AA_ * F_AA_.transpose();
      Paa.noalias() += Q_AA_;
    }

    // -------- Wave block propagation --------
    if (wave_block_enabled_) {

      // cache per-mode Phi/Qd (needed for cross-mode covariance propagation)
      Eigen::Matrix<T,6,6> Phi6_k[KMODES];
      Eigen::Matrix<T,6,6> Qd6_k [KMODES];

      for (int k=0;k<KMODES;++k) {
        // Build per-axis Phi2 and Qd2, then assemble Phi6/Qd6
        for (int ax=0; ax<3; ++ax) {
          discretize_osc_axis_(Ts, omega_[k], zeta_[k], q_axis_[k](ax), Phi2_[ax], Qd2_[ax]);
        }
        Phi6_.setZero();
        Qd6_.setZero();
        for (int ax=0; ax<3; ++ax) {
          Phi6_.template block<2,2>(2*ax,2*ax) = Phi2_[ax];
          Qd6_ .template block<2,2>(2*ax,2*ax) = Qd2_[ax];
        }

        Phi6_k[k] = Phi6_;
        Qd6_k[k]  = Qd6_;

        // Mean
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

        // P_kk
        {
          auto Pkk = P_.template block<6,6>(offk, offk);
          tmp6_.noalias() = Phi6_ * Pkk;
          Pkk.noalias() = tmp6_ * Phi6_.transpose();
          Pkk.noalias() += Qd6_; // noise only on diagonals
        }

        // Cross with base: P_Ak = F_AA * P_Ak * Phi6^T
        {
          auto P_Ak = P_.template block<BASE_N,6>(0, offk);
          tmp_Ak_.noalias() = F_AA_ * P_Ak;
          P_Ak.noalias() = tmp_Ak_ * Phi6_.transpose();
          P_.template block<6,BASE_N>(offk,0) = P_Ak.transpose();
        }

        // Cross with accel bias (F=I): P_BAk = P_BAk * Phi6^T
        if constexpr (with_accel_bias) {
          auto P_BAk = P_.template block<3,6>(OFF_BA, offk);
          P_BAk.noalias() = P_BAk * Phi6_.transpose();
          P_.template block<6,3>(offk,OFF_BA) = P_BAk.transpose();
        }
      }

      // NEW: propagate full wave-wave cross-covariances (k!=j)
      for (int k = 0; k < KMODES; ++k) {
        const int ok = OFF_Pk(k);
        for (int j = k + 1; j < KMODES; ++j) {
          const int oj = OFF_Pk(j);

          auto Pkj = P_.template block<6,6>(ok, oj);
          tmp6_.noalias() = Phi6_k[k] * Pkj;
          Pkj.noalias()   = tmp6_ * Phi6_k[j].transpose();
          P_.template block<6,6>(oj, ok) = Pkj.transpose();
        }
      }
    }

    // -------- Accel bias RW (OU-consistent) --------
    if constexpr (with_accel_bias) {
      auto Pba = P_.template block<3,3>(OFF_BA,OFF_BA);
      if (acc_bias_updates_enabled_) Pba.noalias() += Q_bacc_ * Ts;
      for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), T(1e-12));
      P_.template block<3,3>(OFF_BA,OFF_BA) = Pba;

      // Cross BA with base: P_Aba = F_AA * P_Aba
      auto P_Aba = P_.template block<BASE_N,3>(0,OFF_BA);
      P_Aba.noalias() = F_AA_ * P_Aba;
      P_.template block<3,BASE_N>(OFF_BA,0) = P_Aba.transpose();
    }

    symmetrize_P_();
  }

  // ---------------- Measurement update: accelerometer (OU-style Joseph) ----------------

  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref) {
    last_acc_ = MeasDiag3{};
    last_acc_.accepted = false;

    const bool use_ba = (with_accel_bias && acc_bias_updates_enabled_);

    const Vec3 acc_meas = deheel_vector_(acc_meas_body);

    // Lever arm term
    Vec3 lever = Vec3::Zero();
    if (use_imu_lever_arm_) {
      const Vec3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
      lever.noalias() += alpha_b_.cross(r_imu_bprime)
                      +  last_gyr_bias_corrected_.cross(last_gyr_bias_corrected_.cross(r_imu_bprime));
    }

    // J_bg (copied meaning from OU) --- FIXED SIGN
    Mat3 J_bg = Mat3::Zero();
    if constexpr (with_gyro_bias) {
      if (use_imu_lever_arm_) {
        const Vec3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
        const Vec3& w = last_gyr_bias_corrected_; // ω = gyr - b_g

        T k_alpha = T(0);
        if (have_prev_omega_ && last_dt_ > T(0)) {
          if (alpha_smooth_tau_ > T(0)) {
            const T a = T(1) - std::exp(-last_dt_ / alpha_smooth_tau_);
            k_alpha = a / last_dt_;
          } else {
            k_alpha = T(1) / last_dt_;
          }
        }
        const Mat3 J_alpha_part = k_alpha * skew3<T>(r_imu_bprime);
        const Mat3 J_omega_part = d_omega_x_omega_x_r_domega_<T>(w, r_imu_bprime);

        // ω = gyr - bg => d/d(bg) = - d/d(ω)
        J_bg = -J_alpha_part - J_omega_part;
      }
    }

    // accel bias term (mean always includes BA, even if frozen)
    Vec3 ba_term = Vec3::Zero();
    if constexpr (with_accel_bias) {
      const Vec3 ba0 = x_.template segment<3>(OFF_BA);
      ba_term = ba0 + k_a_ * (tempC - tempC_ref);
    }

    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 aw = wave_world_accel_(); // zero if wave block disabled

    const Vec3 f_pred = R_wb() * (aw - g_world) + lever + ba_term;
    const Vec3 r = acc_meas - f_pred;

    last_acc_.r = r;

    // Jacobian wrt attitude
    const Vec3 f_cog_b = R_wb() * (aw - g_world);
    const Mat3 J_att = -skew3<T>(f_cog_b);

    // Innovation covariance S (3x3)
    Mat3& S = S_scratch_;
    S = Racc_;

    // θ
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH,OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    // BA (J_ba = I): always marginalized through S
    if constexpr (with_accel_bias) {
      const Mat3 Pba  = P_.template block<3,3>(OFF_BA,OFF_BA);
      const Mat3 Ptba = P_.template block<3,3>(OFF_DTH,OFF_BA);
      S.noalias() += J_att * Ptba;
      S.noalias() += Ptba.transpose() * J_att.transpose();
      S.noalias() += Pba;
    }

    // Gyro bias via lever arm (if enabled)
    if constexpr (with_gyro_bias) {
      if (use_imu_lever_arm_) {
        const Mat3 Pth_bg = P_.template block<3,3>(OFF_DTH, OFF_BG);
        const Mat3 Pbg_bg = P_.template block<3,3>(OFF_BG, OFF_BG);
        S.noalias() += J_att * Pth_bg * J_bg.transpose();
        S.noalias() += J_bg * Pth_bg.transpose() * J_att.transpose();
        S.noalias() += J_bg * Pbg_bg * J_bg.transpose();

        if constexpr (with_accel_bias) {
          const Mat3 Pbg_ba = P_.template block<3,3>(OFF_BG, OFF_BA);
          S.noalias() += J_bg * Pbg_ba;
          S.noalias() += Pbg_ba.transpose() * J_bg.transpose();
        }
      }
    }

    // Wave states contributions (FULL Pww, includes cross-mode)  --- FIXED
    Eigen::Matrix<T,3,WAVE_N> Jw;
    Jw.setZero();
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];

        const Mat3 Jp = R_wb() * (-(om*om) * Mat3::Identity());
        const Mat3 Jv = R_wb() * (-(T(2)*ze*om) * Mat3::Identity());

        Jw.template block<3,3>(0, 6*k + 0) = Jp;
        Jw.template block<3,3>(0, 6*k + 3) = Jv;
      }

      const auto Pww = P_.template block<WAVE_N,WAVE_N>(OFF_WAVE, OFF_WAVE);
      S.noalias() += Jw * Pww * Jw.transpose();
    }

    S = T(0.5)*(S + S.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));

    // LDLT
    Eigen::LDLT<Mat3> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6)*(Racc_.norm()+T(1)));
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

    // PCt = P C^T (NXx3)
    MatX3& PCt = PCt_scratch_;
    PCt.setZero();

    PCt.noalias() += P_.template block<NX,3>(0,OFF_DTH) * J_att.transpose();

    // BA column (J_ba = I) but allow update only if use_ba
    if constexpr (with_accel_bias) {
      PCt.noalias() += P_.template block<NX,3>(0,OFF_BA);
      if (!use_ba) PCt.template block<3,3>(OFF_BA,0).setZero();
    }

    // BG via lever Jacobian
    if constexpr (with_gyro_bias) {
      if (use_imu_lever_arm_) {
        PCt.noalias() += P_.template block<NX,3>(0,OFF_BG) * J_bg.transpose();
      }
    }

    // Wave contribution to PCt: FULL P(:,wave)*Jwᵀ  --- FIXED
    if (wave_block_enabled_) {
      PCt.noalias() += P_.template block<NX, WAVE_N>(0, OFF_WAVE) * Jw.transpose();
    } else {
      freeze_wave_rows_(PCt);
    }

    // Gain K = PCt S^-1
    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());

    if (!wave_block_enabled_) freeze_wave_rows_(K);
    if constexpr (with_accel_bias) if (!use_ba) K.template block<3,3>(OFF_BA,0).setZero();

    // State update
    x_.noalias() += K * r;

    // Joseph
    joseph_update3_(K, S, PCt);

    applyQuaternionCorrectionFromErrorState_();

    last_acc_.accepted = true;
  }

  // ---------------- Measurement update: magnetometer (OU-style) ----------------

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

    Vec3 v2hat = R_wb() * B_world_ref_;
    if (v2hat.dot(mag_meas) < T(0)) v2hat = -v2hat;

    const Vec3 r = mag_meas - v2hat;
    last_mag_.r = r;

    const Mat3 J_att = -skew3<T>(v2hat);

    Mat3& S = S_scratch_;
    S = Rmag_;
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH,OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();
    S = T(0.5)*(S + S.transpose());
    for (int i=0;i<3;++i) S(i,i) = std::max(S(i,i), T(1e-12));

    Eigen::LDLT<Mat3> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success) {
      const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6)*(Rmag_.norm()+T(1)));
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

    if (!wave_block_enabled_) freeze_wave_rows_(K);
    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) K.template block<3,3>(OFF_BA,0).setZero();
    }

    x_.noalias() += K * r;
    joseph_update3_(K, S, PCt);

    applyQuaternionCorrectionFromErrorState_();

    last_mag_.accepted = true;
  }

private:
  // ---------------- Constants / state ----------------
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

  bool wave_block_enabled_ = true;
  bool acc_bias_updates_enabled_ = true;
  bool warmup_mode_ = true;
  bool use_exact_att_bias_Qd_ = true;

  // Warmup exit detection
  T exit_min_distance_ = T(10.0);
  T exit_min_time_     = T(5.0);
  Vec3 vel_detect_ = Vec3::Zero();
  T pseudo_motion_dist_ = T(0);
  T pseudo_motion_time_ = T(0);

  // Optional gyro bias learning during warmup
  Vec3 bg_accum_ = Vec3::Zero();
  int  bg_accum_count_ = 0;

  // Lever arm caches
  bool use_imu_lever_arm_ = false;
  Vec3 r_imu_wrt_cog_body_phys_ = Vec3::Zero();

  Vec3 prev_omega_b_ = Vec3::Zero();
  Vec3 alpha_b_      = Vec3::Zero();
  bool have_prev_omega_ = false;

  Vec3 last_gyr_bias_corrected_ = Vec3::Zero();
  T last_dt_ = T(1.0/240);
  T alpha_smooth_tau_ = T(0.05);

  // Heel / de-heel (same as OU)
  T wind_heel_rad_ = T(0);
  T cos_unheel_x_  = T(1);
  T sin_unheel_x_  = T(0);

  // Diags
  MeasDiag3 last_acc_;
  MeasDiag3 last_mag_;

  // Scratch (avoid heap/stack churn)
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
  // ---------------- Frame helpers ----------------
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }              // WORLD->BODY'
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); }  // BODY'->WORLD

  void symmetrize_P_() {
    for (int i=0;i<NX;++i) {
      for (int j=i+1;j<NX;++j) {
        const T v = T(0.5)*(P_(i,j) + P_(j,i));
        P_(i,j) = v;
        P_(j,i) = v;
      }
    }
  }

  void applyQuaternionCorrectionFromErrorState_() {
    const Vec3 dth = x_.template segment<3>(OFF_DTH);
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

  // Freeze wave rows in an NXx3 matrix (OU "freeze linear rows" analog)
  template<typename Derived>
  EIGEN_STRONG_INLINE void freeze_wave_rows_(Eigen::MatrixBase<Derived>& M) const {
    M.template block<WAVE_N,3>(OFF_WAVE,0).setZero();
  }

  // Joseph covariance update (stack-light) exactly like OU version
  EIGEN_STRONG_INLINE void joseph_update3_(const MatX3& K, const Mat3& S, const MatX3& PCt) {
    for (int i=0;i<NX;++i) {
      for (int j=i;j<NX;++j) {
        T KCP_ij = T(0), KCP_ji = T(0);
        for (int l=0;l<3;++l) {
          const T Ki_l = K(i,l);
          const T Kj_l = K(j,l);
          const T Pj_l = PCt(j,l);
          const T Pi_l = PCt(i,l);
          KCP_ij += Ki_l * Pj_l;
          if (j != i) KCP_ji += Kj_l * Pi_l;
        }
        if (j==i) KCP_ji = KCP_ij;

        T KSK_ij = T(0);
        for (int a=0;a<3;++a) {
          const T Kia = K(i,a);
          for (int b=0;b<3;++b) {
            KSK_ij += Kia * S(a,b) * K(j,b);
          }
        }
        const T delta = -(KCP_ij + KCP_ji) + KSK_ij;
        P_(i,j) += delta;
        if (j!=i) P_(j,i) = P_(i,j);
      }
    }
    symmetrize_P_();
  }

  // ---------------- Heel / de-heel (copied from OU style) ----------------

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

    // R = Rx(-Δheel): maps old B' coords -> new B' coords.
    const Mat3 R = Rx_(-delta_heel_rad);

    // Rotate attitude reference: qref is WORLD->B'. Left-multiply by frame change.
    const Eigen::Quaternion<T> qR(R);
    qref_ = qR * qref_;
    qref_.normalize();

    // Rotate any vector states that live in B' coordinates:
    // δθ lives in B', and biases are stored in B' coordinates as well.
    x_.template segment<3>(OFF_DTH) = R * x_.template segment<3>(OFF_DTH);
    if constexpr (with_gyro_bias)  x_.template segment<3>(OFF_BG) = R * x_.template segment<3>(OFF_BG);
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = R * x_.template segment<3>(OFF_BA);

    // Rotate cached B' kinematics
    last_gyr_bias_corrected_ = R * last_gyr_bias_corrected_;
    prev_omega_b_            = R * prev_omega_b_;
    alpha_b_                 = R * alpha_b_;

    // Similarity-transform covariance for the rotated blocks (cheap; NX is small)
    MatX Tm = MatX::Identity();
    Tm.template block<3,3>(OFF_DTH,OFF_DTH) = R;
    if constexpr (with_gyro_bias)  Tm.template block<3,3>(OFF_BG,OFF_BG) = R;
    if constexpr (with_accel_bias) Tm.template block<3,3>(OFF_BA,OFF_BA) = R;
    P_ = Tm * P_ * Tm.transpose();
    symmetrize_P_();
  }

  // ---------------- Oscillator discretization with cancellation guards ----------------
  // State per axis: [p; v], dynamics:
  //   p' = v
  //   v' = -w^2 p - 2 z w v + ξ(t),  with E[ξξ]=q δ(t)
  //
  // Compute:
  //   Phi = exp(A dt)
  //   Qd  = ∫_0^dt Φ(t) G q G^T Φ(t)^T dt, with G=[0;1]
  // Using Simpson rule with stable Phi evaluation.

  static inline void phi_osc_2x2_(T t, T w, T z, Eigen::Matrix<T,2,2>& Phi) {
    const T om = std::max(T(1e-7), w);
    const T ze = std::max(T(0), z);
    const T eps = T(1e-7);

    // Near-critical (series-like fallback)
    if (std::abs(ze - T(1)) < T(1e-3)) {
      const T e = std::exp(-om*t);
      Phi(0,0) = e * (T(1) + om*t);
      Phi(0,1) = e * t;
      Phi(1,0) = e * (-om*om*t);
      Phi(1,1) = e * (T(1) - om*t);
      return;
    }

    if (ze < T(1)) {
      const T wd = om * std::sqrt(std::max(T(0), T(1) - ze*ze));
      if (wd < eps) {
        const T e = std::exp(-om*t);
        Phi(0,0) = e * (T(1) + om*t);
        Phi(0,1) = e * t;
        Phi(1,0) = e * (-om*om*t);
        Phi(1,1) = e * (T(1) - om*t);
        return;
      }
      const T a = ze * om;
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

    // Overdamped
    const T srt = std::sqrt(std::max(T(0), ze*ze - T(1)));
    const T r1 = -om*(ze - srt);
    const T r2 = -om*(ze + srt);

    const T e1 = std::exp(r1*t);
    const T e2 = std::exp(r2*t);

    const T denom = (r2 - r1);
    if (std::abs(denom) < eps) {
      const T e = std::exp(-om*t);
      Phi(0,0) = e * (T(1) + om*t);
      Phi(0,1) = e * t;
      Phi(1,0) = e * (-om*om*t);
      Phi(1,1) = e * (T(1) - om*t);
      return;
    }
    const T invd = T(1)/denom;

    Phi(0,0) = (r2*e1 - r1*e2) * invd;
    Phi(0,1) = (e2 - e1) * invd;
    Phi(1,0) = (r1*r2) * (e1 - e2) * invd;
    Phi(1,1) = (r2*e2 - r1*e1) * invd;
  }

  static inline void discretize_osc_axis_(T dt, T w, T z, T q,
                                         Eigen::Matrix<T,2,2>& Phi,
                                         Eigen::Matrix<T,2,2>& Qd)
  {
    phi_osc_2x2_(dt, w, z, Phi);

    // Qd = ∫ Φ(t) G q G^T Φ(t)^T dt, G=[0;1]
    // Let u(t) = Φ(t) G = [Φ01(t); Φ11(t)], then integrand = q * u u^T.
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

    // Hygiene
    Qd = T(0.5) * (Qd + Qd.transpose());
    for (int i=0;i<2;++i) Qd(i,i) = std::max(Qd(i,i), T(0));
    project_psd<T,2>(Qd, T(1e-18));
    Qd = T(0.5) * (Qd + Qd.transpose());
  }
};
