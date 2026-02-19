#pragma once
/*
  Kalman3D_Wave_2  (template KMODES + with_mag)
  --------------------------------------------
  DROP-IN header. Same conventions as your old OU-based Kalman3D_Wave:

  Frames / conventions (MATCH OU):
  - World frame is NED, +Z down.
  - qref_ stores WORLD -> BODY' quaternion (virtual un-heeled body frame B').
  - quaternion() returns BODY' -> WORLD, i.e. qref_.conjugate().
  - quaternion_boat() returns physical BODY -> WORLD by re-applying heel about +X (roll).

  Accel specific-force model (BODY'):
      f_b' = R_wb'( a_w - g_world ) + lever(ω,α,r_imu) + b_a(temp)
    where g_world = (0,0,+g) in NED.

  Wave model (broadband, per mode k=1..KMODES, per axis):
      p' = v
      v' = -ω_k^2 p - 2ζ_k ω_k v + ξ(t)      (ξ: white accel noise PSD q_k)
    Wave acceleration:
      a_w = Σ_k ( -ω_k^2 p_k - 2ζ_k ω_k v_k )

  State (error-state MEKF):
    [ dtheta(3),
      b_g(3) optional,
      p1(3), v1(3), ..., pK(3), vK(3),
      b_a(3) optional ]

  Numerical stability (OU-parity):
  - Stable oscillator discretization with small-argument series + expm1 guards
  - Discrete Q via robust Simpson; small-dt series fallback
  - Stable 3x3 LDLT with adaptive diagonal bump
  - True Joseph form update WITHOUT solving with P (OU-style, low memory)
  - Covariance symmetrization + adaptive regularization on failure
  - Warm-up / staged initialization learning (gyro bias + mag ref + motion exit)
*/

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

#include <cmath>
#include <limits>
#include <algorithm>

template<typename T>
static inline Eigen::Matrix<T,3,3> skew3(const Eigen::Matrix<T,3,1>& a) {
  Eigen::Matrix<T,3,3> S;
  S << T(0),   -a.z(),  a.y(),
       a.z(),  T(0),   -a.x(),
      -a.y(),  a.x(),   T(0);
  return S;
}

template<typename T>
static inline Eigen::Quaternion<T> quat_from_delta_theta(const Eigen::Matrix<T,3,1>& dtheta) {
  const T theta = dtheta.norm();
  const T half  = T(0.5) * theta;

  T w, k;
  // series for small angles to avoid catastrophic cancellation
  if (theta < T(1e-3)) {
    const T t2 = theta*theta;
    const T t4 = t2*t2;
    w = T(1) - t2*(T(1)/T(8)) + t4*(T(1)/T(384));
    k = T(0.5) - t2*(T(1)/T(48)) + t4*(T(1)/T(3840));
  } else {
    w = std::cos(half);
    k = std::sin(half) / theta;
  }

  const auto v = k * dtheta;
  Eigen::Quaternion<T> q(w, v.x(), v.y(), v.z());
  q.normalize();
  return q;
}

template<typename T=float, int KMODES=3, bool with_gyro_bias=true, bool with_accel_bias=true, bool with_mag=true>
class Kalman3D_Wave_2 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vec3 = Eigen::Matrix<T,3,1>;
  using Mat3 = Eigen::Matrix<T,3,3>;

  static_assert(KMODES >= 1, "KMODES must be >= 1");

  // Base (attitude error + optional gyro bias)
  static constexpr int BASE_N = with_gyro_bias ? 6 : 3;
  // Wave block: (p_k(3), v_k(3)) * K
  static constexpr int WAVE_N = 6 * KMODES;
  // Accel bias
  static constexpr int BA_N   = with_accel_bias ? 3 : 0;
  static constexpr int NX     = BASE_N + WAVE_N + BA_N;

  using VecX  = Eigen::Matrix<T,NX,1>;
  using MatX  = Eigen::Matrix<T,NX,NX>;
  using MatX3 = Eigen::Matrix<T,NX,3>;

  // Offsets (MATCH OU layout)
  static constexpr int OFF_DTH  = 0;                          // δθ (always)
  static constexpr int OFF_BG   = with_gyro_bias ? 3 : -1;     // gyro bias if enabled
  static constexpr int OFF_WAVE = BASE_N;                     // wave start
  static constexpr int OFF_BA   = with_accel_bias ? (BASE_N + WAVE_N) : -1;

  static constexpr int OFF_Pk(int k) { return OFF_WAVE + 6*k + 0; } // 3
  static constexpr int OFF_Vk(int k) { return OFF_WAVE + 6*k + 3; } // 3

  struct MeasDiag3 {
    Vec3 r = Vec3::Zero();
    Mat3 S = Mat3::Zero();
    T   nis = std::numeric_limits<T>::quiet_NaN();
    bool accepted = false;
  };

  Kalman3D_Wave_2(const Vec3& sigma_a_meas,
                  const Vec3& sigma_g_rw,
                  const Vec3& sigma_m_meas,
                  T Pq0 = T(5e-4),
                  T Pb0 = T(1e-6),
                  T b0  = T(1e-11),
                  T gravity_magnitude = T(9.80665))
  : gravity_magnitude_(gravity_magnitude)
  {
    qref_.setIdentity();
    x_.setZero();
    P_.setZero();

    // Measurement noise
    Racc_ = sigma_a_meas.array().square().matrix().asDiagonal();
    Rmag_ = sigma_m_meas.array().square().matrix().asDiagonal();

    // Base process noise rates
    Qg_  = sigma_g_rw.array().square().matrix().asDiagonal(); // drives δθ
    if constexpr (with_gyro_bias) {
      Qbg_ = Mat3::Identity() * b0;                           // bias RW rate
    }

    // Seed base covariance
    P_.template block<3,3>(OFF_DTH, OFF_DTH) = Mat3::Identity() * Pq0;
    if constexpr (with_gyro_bias) {
      P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * Pb0;
    }

    // Seed accel bias
    if constexpr (with_accel_bias) {
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * (sigma_bacc0_*sigma_bacc0_);
    }

    // Seed wave covariances (normal running values)
    const T sigma_p0 = T(20.0);
    const T sigma_v0 = T(1.0);
    for (int k=0;k<KMODES;++k) {
      P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
      P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
    }

    // Default mag reference if mag enabled
    B_world_ref_ = Vec3::UnitX();

    set_broadband_params(T(0.12), T(1.0));
    update_unheel_trig_();

    // Warm-up ON by default (OU behavior)
    set_warmup_mode(true);
  }

  // ===== Conventions / accessors (MATCH OU) =====
  [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return qref_.conjugate(); } // BODY'->WORLD

  [[nodiscard]] Eigen::Quaternion<T> quaternion_boat() const {
    const Eigen::Quaternion<T> q_WBprime = quaternion();
    const T half = -wind_heel_rad_ * T(0.5);
    const T c = std::cos(half);
    const T s = std::sin(half);
    const Eigen::Quaternion<T> q_BprimeB(c, s, 0, 0); // roll about +X
    return q_WBprime * q_BprimeB;                     // BODY->WORLD
  }

  [[nodiscard]] Vec3 gyroscope_bias() const {
    if constexpr (with_gyro_bias) return x_.template segment<3>(OFF_BG);
    else return Vec3::Zero();
  }

  [[nodiscard]] Vec3 accel_bias() const {
    if constexpr (with_accel_bias) return x_.template segment<3>(OFF_BA);
    else return Vec3::Zero();
  }

  // Total wave states in world (NED)
  [[nodiscard]] Vec3 wave_position_world() const {
    Vec3 p = Vec3::Zero();
    if (!wave_block_enabled_) return p;
    for (int k=0;k<KMODES;++k) p += x_.template segment<3>(OFF_Pk(k));
    return p;
  }
  [[nodiscard]] Vec3 wave_velocity_world() const {
    Vec3 v = Vec3::Zero();
    if (!wave_block_enabled_) return v;
    for (int k=0;k<KMODES;++k) v += x_.template segment<3>(OFF_Vk(k));
    return v;
  }
  [[nodiscard]] Vec3 wave_accel_world() const { return wave_world_accel_(); }

  const MeasDiag3& lastAccDiag() const noexcept { return last_acc_; }
  const MeasDiag3& lastMagDiag() const noexcept { return last_mag_; }

  // ===== Mag ref =====
  void set_mag_world_ref(const Vec3& B_world) {
    if constexpr (with_mag) B_world_ref_ = B_world;
    else (void)B_world;
  }

  // ===== Acc bias temp model / RW =====
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

  // ===== Warm-up control (OU-style staged init) =====
  void set_warmup_mode(bool on) {
    if (warmup_mode_ == on) return;
    warmup_mode_ = on;

    if (on) {
      // Enter warm-up:
      // - wave disabled (states fixed near zero)
      // - accel bias updates enabled (learn)
      set_wave_block_enabled(false);
      set_acc_bias_updates_enabled(true);

      const T tiny = type_floor_diag_(T(1e-12));
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * tiny;
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * tiny;
        x_.template segment<3>(OFF_Pk(k)).setZero();
        x_.template segment<3>(OFF_Vk(k)).setZero();
      }

      gyro_bias_accum_.setZero();
      gyro_bias_accum_count_ = 0;
      mag_ref_accum_.setZero();
      mag_ref_accum_count_ = 0;

      motion_distance_ = T(0);
      motion_time_ = T(0);
      init_vel_world_.setZero();
    } else {
      // Exit warm-up:
      const T sigma_p0 = T(20.0);
      const T sigma_v0 = T(1.0);
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
      }
      set_wave_block_enabled(true);
    }

    regularize_covariance_(/*strong=*/true);
    symmetrize_P_();
  }

  bool warmup_mode() const { return warmup_mode_; }

  void set_wave_block_enabled(bool on) {
    if (wave_block_enabled_ && !on) {
      // zero cross-cov to keep block isolated in warm-up
      P_.template block<BASE_N,WAVE_N>(0, OFF_WAVE).setZero();
      P_.template block<WAVE_N,BASE_N>(OFF_WAVE, 0).setZero();
      if constexpr (with_accel_bias) {
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      }
    }
    wave_block_enabled_ = on;
  }
  bool wave_block_enabled() const { return wave_block_enabled_; }

  void set_acc_bias_updates_enabled(bool en) {
    if (acc_bias_updates_enabled_ == en) return;
    if constexpr (with_accel_bias) {
      if (!en) {
        // isolate BA (OU style)
        P_.template block<3,BASE_N>(OFF_BA, 0).setZero();
        P_.template block<BASE_N,3>(0, OFF_BA).setZero();
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      } else {
        // ensure BA diagonal not collapsed
        auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
        const T target = sigma_bacc0_*sigma_bacc0_;
        for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), target);
        P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;
      }
    }
    acc_bias_updates_enabled_ = en;
  }

  // ===== Motion thresholds for auto-exit from warm-up (OU behavior) =====
  void set_motion_exit_thresholds(T min_distance_m, T min_time_sec) {
    exit_min_distance_ = std::max(T(0), min_distance_m);
    exit_min_time_     = std::max(T(0), min_time_sec);
  }

  // ===== Call periodically during warm-up =====
  void update_initialization(const Vec3& acc_body, const Vec3& gyr_body,
                             const Vec3& mag_body, T dt) {
    if (!warmup_mode_) return;
    if (!(dt > T(0)) || !std::isfinite((double)dt)) return;

    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 gyr = deheel_vector_(gyr_body);
    const Vec3 mag = deheel_vector_(mag_body);

    // 1) Learn gyro bias (OU-style gentle EMA)
    if constexpr (with_gyro_bias) {
      if (gyro_bias_accum_count_ == 0) gyro_bias_accum_ = gyr;
      else {
        const T alpha = T(0.01);
        gyro_bias_accum_ = (T(1)-alpha) * gyro_bias_accum_ + alpha * gyr;
      }
      gyro_bias_accum_count_++;
      if (gyro_bias_accum_count_ > 100) {
        x_.template segment<3>(OFF_BG) = gyro_bias_accum_;
        P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * type_floor_diag_(T(1e-8));
        gyro_bias_accum_count_ = 101;
      }
    }

    // 2) Learn magnetic reference (OU-style, only near 1g)
    if constexpr (with_mag) {
      const T an = acc.norm();
      if (std::isfinite((double)an) && std::abs(an - gravity_magnitude_) < T(0.5)) {
        const Vec3 z_body = -acc / std::max(T(1e-9), an);
        Vec3 mag_h = mag - (mag.dot(z_body)) * z_body;
        const T mh = mag_h.norm();
        if (mh > T(1e-6)) {
          mag_h /= mh;
          const Vec3 mag_world = R_bw() * mag_h;
          mag_ref_accum_ += mag_world;
          mag_ref_accum_count_++;

          if (mag_ref_accum_count_ > 10) {
            Vec3 new_ref = mag_ref_accum_ / T(mag_ref_accum_count_);
            const T nr = new_ref.norm();
            if (nr > T(1e-9)) new_ref /= nr;

            const T alpha = T(0.01);
            B_world_ref_ = (T(1)-alpha) * B_world_ref_ + alpha * new_ref;
            const T br = B_world_ref_.norm();
            if (br > T(1e-9)) B_world_ref_ /= br;

            if (mag_ref_accum_count_ > 100) {
              mag_ref_accum_.setZero();
              mag_ref_accum_count_ = 0;
            }
          }
        }
      }
    }

    // 3) crude motion integration for exit decision (OU-style)
    const Vec3 acc_world = R_bw() * acc;
    const Vec3 acc_motion = acc_world - Vec3(0,0,gravity_magnitude_);
    init_vel_world_ += acc_motion * dt;
    motion_distance_ += init_vel_world_.norm() * dt;
    motion_time_ += dt;

    // 4) auto-exit
    if (motion_distance_ > exit_min_distance_ && motion_time_ > exit_min_time_) {
      set_warmup_mode(false);
    }
  }

  // ===== IMU lever arm (MATCH OU: computed in BODY' after de-heel) =====
  void set_imu_lever_arm_body(const Vec3& r_b_phys) {
    r_imu_wrt_cog_body_phys_ = r_b_phys;
    use_imu_lever_arm_ = (r_b_phys.squaredNorm() > T(0));
  }
  void clear_imu_lever_arm() {
    r_imu_wrt_cog_body_phys_.setZero();
    use_imu_lever_arm_ = false;
  }
  void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

  // ===== Heel update (MATCH OU retarget logic) =====
  void update_wind_heel(T heel_rad) {
    const T old = wind_heel_rad_;
    if (heel_rad == old) return;
    retarget_bodyprime_frame_(heel_rad - old);
    wind_heel_rad_ = heel_rad;
    update_unheel_trig_();
  }

  // ===== Initialization from acc/mag (same "tilt-compass" idea as OU) =====
  void initialize_from_acc_mag(const Vec3& acc_body, const Vec3& mag_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 mag = deheel_vector_(mag_body);

    const T an = acc.norm();
    if (!(an > T(1e-8)) || !std::isfinite((double)an)) return;
    const Vec3 acc_n = acc / an;

    // In NED, z_world is "down". Acc measures specific force; at rest, acc ≈ -g*z_world in BODY'
    // Thus body down direction is -acc_n in body coordinates. We treat z_world expressed in body as (-acc_n).
    const Vec3 z_body = -acc_n;

    Vec3 mag_h = mag - (mag.dot(z_body))*z_body;
    const T mh = mag_h.norm();
    if (!(mh > T(1e-8)) || !std::isfinite((double)mh)) return;
    mag_h /= mh;

    // Build body axes from (mag_h, z_body)
    const Vec3 x_body = mag_h;                    // "magnetic north-ish" in body horizontal
    Vec3 y_body = z_body.cross(x_body);
    const T yn = y_body.norm();
    if (!(yn > T(1e-8))) return;
    y_body /= yn;
    Vec3 x_body_ortho = y_body.cross(z_body);     // re-orthogonalize

    Mat3 R_wb;
    // R_wb maps world basis to body basis; columns are world axes expressed in body.
    // We choose world-X aligned with x_body_ortho, world-Y with y_body, world-Z with z_body.
    R_wb.col(0) = x_body_ortho;
    R_wb.col(1) = y_body;
    R_wb.col(2) = z_body;

    qref_ = Eigen::Quaternion<T>(R_wb); // WORLD->BODY'
    qref_.normalize();

    if constexpr (with_mag) {
      // Store the current world mag direction as reference
      const Vec3 mw = R_bw() * mag;
      const T mn = mw.norm();
      if (mn > T(1e-9)) B_world_ref_ = mw / mn;
    }

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  void initialize_from_acc(const Vec3& acc_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const T an = acc.norm();
    if (!(an > T(1e-8)) || !std::isfinite((double)an)) return;

    const Vec3 anorm = acc / an;
    const Vec3 zb = Vec3::UnitZ();
    const Vec3 target = -anorm; // down-body direction

    T c = std::max(T(-1), std::min(T(1), zb.dot(target)));
    Vec3 axis = zb.cross(target);
    T sn = axis.norm();

    if (sn < T(1e-8)) {
      qref_.setIdentity();
    } else {
      axis /= sn;
      T ang = std::acos(c);
      qref_ = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(ang, axis));
      qref_.normalize();
    }
    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true);
  }

  // ===== Wave tuning (broadband) =====
  void set_broadband_params(T f0_hz, T Hs_m, T zeta_mid = T(0.08), T horiz_scale = T(0.35)) {
    const T f0 = std::max(T(1e-4), f0_hz);
    const T w0 = T(2)*T(M_PI)*f0;

    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1)); // 0..1
      const T lo = std::log(T(0.6));
      const T hi = std::log(T(1.7));
      omega_[k] = std::exp(lo + (hi-lo)*u) * w0;

      const T scale = (k==0 || k==KMODES-1) ? T(1.25) : T(1.0);
      zeta_[k] = std::max(T(0.01), zeta_mid * scale);
    }

    // weights (Gaussian-ish over modes)
    T wsum = T(0);
    for (int k=0;k<KMODES;++k) {
      const T c = (KMODES==1) ? T(0) : (T(k) - T(0.5)*(KMODES-1));
      const T wk = std::exp(-(c*c) / (T(0.7)*T(0.7)));
      weights_[k] = wk;
      wsum += wk;
    }
    for (int k=0;k<KMODES;++k) weights_[k] /= std::max(T(1e-12), wsum);

    // map Hs -> total variance: sigma ~= Hs/4
    const T sigma_total = std::max(T(0), Hs_m) / T(4);
    const T var_total   = sigma_total * sigma_total;

    for (int k=0;k<KMODES;++k) {
      const T var_k = weights_[k] * var_total;
      const T om = std::max(T(1e-6), omega_[k]);
      const T ze = std::max(T(1e-6), zeta_[k]);

      // For oscillator driven by white accel noise on v', stationary position variance approx:
      // Var(p) ≈ q / (4 ζ ω^3)  -> q ≈ 4 ζ ω^3 Var(p)
      const T qk = T(4) * ze * om*om*om * var_k;

      q_axis_[k].x() = horiz_scale * qk;
      q_axis_[k].y() = horiz_scale * qk;
      q_axis_[k].z() = qk;
    }
  }

  // ===== Time update (propagation) =====
  void time_update(const Vec3& gyr_body, T Ts) {
    last_dt_ = Ts;
    if (!(Ts > T(0)) || !std::isfinite((double)Ts)) return;

    const Vec3 gyr = deheel_vector_(gyr_body);

    Vec3 bg = Vec3::Zero();
    if constexpr (with_gyro_bias) bg = x_.template segment<3>(OFF_BG);
    last_gyr_bias_corrected_ = gyr - bg;

    const Vec3 omega_b = last_gyr_bias_corrected_;

    // alpha estimate (OU-style smoothing)
    if (have_prev_omega_) {
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

    // propagate quaternion (WORLD->BODY')
    qref_ = qref_ * quat_from_delta_theta<T>((omega_b * Ts).eval());
    qref_.normalize();

    // ===== Base covariance propagate (MEKF linearized) =====
    const Mat3 Fth = Mat3::Identity() - skew3<T>(omega_b) * Ts;

    F_AA_.setIdentity();
    F_AA_.template block<3,3>(0,0) = Fth;
    if constexpr (with_gyro_bias) {
      F_AA_.template block<3,3>(0,3) = -Mat3::Identity() * Ts;
    }

    Q_AA_.setZero();
    Q_AA_.template block<3,3>(0,0) = Qg_ * Ts;
    if constexpr (with_gyro_bias) {
      Q_AA_.template block<3,3>(3,3) = Qbg_ * Ts;
    }

    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      tmp_AA_.noalias() = F_AA_ * Paa;
      Paa.noalias() = tmp_AA_ * F_AA_.transpose();
      Paa.noalias() += Q_AA_;
    }

    // ===== Wave block propagate =====
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        // Build per-axis Phi2 and Qd2 with safety guards
        for (int ax=0; ax<3; ++ax) {
          discretize_osc_axis_(Ts, omega_[k], zeta_[k], q_axis_[k](ax), Phi2_[ax], Qd2_[ax]);
        }

        // mean propagate (6 states packed by axis)
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

        // assemble Phi6 / Qd6 (block-diag over axes)
        Phi6_.setZero();
        Qd6_.setZero();
        for (int ax=0; ax<3; ++ax) {
          Phi6_.template block<2,2>(2*ax,2*ax) = Phi2_[ax];
          Qd6_ .template block<2,2>(2*ax,2*ax) = Qd2_[ax];
        }

        const int offk = OFF_Pk(k);

        // Pkk
        {
          auto Pkk = P_.template block<6,6>(offk, offk);
          tmp6_.noalias() = Phi6_ * Pkk;
          Pkk.noalias() = tmp6_ * Phi6_.transpose();
          Pkk.noalias() += Qd6_;
        }

        // Cross with base: P_Ak = F * P_Ak * Phi^T
        {
          auto P_Ak = P_.template block<BASE_N,6>(0, offk);
          tmp_Ak_.noalias() = F_AA_ * P_Ak;               // BASE_N x 6
          P_Ak.noalias() = tmp_Ak_ * Phi6_.transpose();
          P_.template block<6,BASE_N>(offk, 0) = P_Ak.transpose();
        }

        // Cross with BA: P_BAk = P_BAk * Phi^T   (BA is RW, independent)
        if constexpr (with_accel_bias) {
          auto P_BAk = P_.template block<3,6>(OFF_BA, offk);
          P_BAk.noalias() = P_BAk * Phi6_.transpose();
          P_.template block<6,3>(offk, OFF_BA) = P_BAk.transpose();
        }
      }
    }

    // ===== Accel bias RW =====
    if constexpr (with_accel_bias) {
      auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
      if (acc_bias_updates_enabled_) Pba.noalias() += Q_bacc_ * Ts;

      // ensure diagonal sensible for float/double
      const T mind = type_floor_diag_(T(0));
      for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), mind);
      P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;

      // Cross BA with base: P_Aba = F * P_Aba
      auto P_Aba = P_.template block<BASE_N,3>(0, OFF_BA);
      P_Aba.noalias() = F_AA_ * P_Aba;
      P_.template block<3,BASE_N>(OFF_BA, 0) = P_Aba.transpose();
    }

    symmetrize_P_();
    regularize_covariance_(/*strong=*/false);
    symmetrize_P_();
  }

  // ===== Measurement update: accelerometer (OU-style Joseph, low memory) =====
  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref_) {
    last_acc_ = MeasDiag3{};
    last_acc_.accepted = false;

    const Vec3 acc_meas = deheel_vector_(acc_meas_body);
    if (!acc_meas.allFinite()) return;

    // lever arm (BODY') : alpha x r + omega x (omega x r)
    Vec3 lever = Vec3::Zero();
    if (use_imu_lever_arm_) {
      const Vec3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
      lever.noalias() += alpha_b_.cross(r_imu_bprime)
                      +  last_gyr_bias_corrected_.cross(last_gyr_bias_corrected_.cross(r_imu_bprime));
    }

    // accel bias + temp (BODY')
    Vec3 ba_term = Vec3::Zero();
    if constexpr (with_accel_bias) {
      ba_term = x_.template segment<3>(OFF_BA) + k_a_ * (tempC - tempC_ref_);
    }

    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 aw = wave_world_accel_();

    // predicted specific force in BODY'
    const Vec3 f_pred = R_wb() * (aw - g_world) + lever + ba_term;

    const Vec3 r = acc_meas - f_pred;
    last_acc_.r = r;

    // Build C (measurement Jacobian) implicitly via PCt = P*C^T and S = C P C^T + R
    // Attitude part: d(f) ≈ -[f_cog_b]x dtheta
    const Vec3 f_cog_b = R_wb() * (aw - g_world);
    const Mat3 J_att = -skew3<T>(f_cog_b);

    // S = R + J_att Ptt J_att^T + wave + bias + cross terms (all assembled explicitly)
    Mat3 S = Racc_;
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH, OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    // accel bias contribution (J_ba = I in BODY')
    if constexpr (with_accel_bias) {
      const Mat3 Pba  = P_.template block<3,3>(OFF_BA, OFF_BA);
      const Mat3 Ptba = P_.template block<3,3>(OFF_DTH, OFF_BA);
      if (acc_bias_updates_enabled_) {
        S.noalias() += Pba;
        S.noalias() += J_att * Ptba;
        S.noalias() += Ptba.transpose() * J_att.transpose();
      }
    }

    // wave contributions
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];

        const Mat3 Jp = R_wb() * (-(om*om) * Mat3::Identity());
        const Mat3 Jv = R_wb() * (-(T(2)*ze*om) * Mat3::Identity());

        const int op = OFF_Pk(k);
        const int ov = OFF_Vk(k);

        const Mat3 Ppp = P_.template block<3,3>(op, op);
        const Mat3 Pvv = P_.template block<3,3>(ov, ov);
        const Mat3 Ppv = P_.template block<3,3>(op, ov);

        const Mat3 Ptp = P_.template block<3,3>(OFF_DTH, op);
        const Mat3 Ptv = P_.template block<3,3>(OFF_DTH, ov);

        S.noalias() += Jp * Ppp * Jp.transpose();
        S.noalias() += Jv * Pvv * Jv.transpose();
        S.noalias() += Jp * Ppv * Jv.transpose();
        S.noalias() += Jv * Ppv.transpose() * Jp.transpose();

        S.noalias() += J_att * Ptp * Jp.transpose();
        S.noalias() += Jp * Ptp.transpose() * J_att.transpose();
        S.noalias() += J_att * Ptv * Jv.transpose();
        S.noalias() += Jv * Ptv.transpose() * J_att.transpose();

        if constexpr (with_accel_bias) {
          if (acc_bias_updates_enabled_) {
            const Mat3 Pbap = P_.template block<3,3>(OFF_BA, op);
            const Mat3 Pbav = P_.template block<3,3>(OFF_BA, ov);
            S.noalias() += Pbap * Jp.transpose();
            S.noalias() += Jp * Pbap.transpose();
            S.noalias() += Pbav * Jv.transpose();
            S.noalias() += Jv * Pbav.transpose();
          }
        }
      }
    }

    // Safe LDLT (OU-style)
    Eigen::LDLT<Mat3> ldlt;
    if (!safe_ldlt3_(S, ldlt)) return;

    last_acc_.S = S;
    last_acc_.nis = nis3_from_ldlt_(r, ldlt);

    // PCt = P*C^T (NXx3)
    PCt_acc_.setZero();
    PCt_acc_.noalias() += P_.template block<NX,3>(0, OFF_DTH) * J_att.transpose();

    if constexpr (with_accel_bias) {
      if (acc_bias_updates_enabled_) {
        PCt_acc_.noalias() += P_.template block<NX,3>(0, OFF_BA); // J_ba = I
      }
    }

    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];
        const Mat3 Jp = R_wb() * (-(om*om) * Mat3::Identity());
        const Mat3 Jv = R_wb() * (-(T(2)*ze*om) * Mat3::Identity());

        PCt_acc_.noalias() += P_.template block<NX,3>(0, OFF_Pk(k)) * Jp.transpose();
        PCt_acc_.noalias() += P_.template block<NX,3>(0, OFF_Vk(k)) * Jv.transpose();
      }
    }

    // K = PCt * S^{-1}
    K_acc_.noalias() = PCt_acc_ * ldlt.solve(Mat3::Identity());

    // State update
    x_.noalias() += K_acc_ * r;

    // True Joseph (OU-style): P = P - K*PCt^T - PCt*K^T + K*S*K^T
    joseph_update3_(P_, K_acc_, PCt_acc_, S);

    symmetrize_P_();
    regularize_covariance_(/*strong=*/false);
    symmetrize_P_();

    applyQuaternionCorrectionFromErrorState_();
    last_acc_.accepted = true;
  }

  // ===== Measurement update: magnetometer (OU-style Joseph) =====
  void measurement_update_mag_only(const Vec3& mag_meas_body) {
    last_mag_ = MeasDiag3{};
    last_mag_.accepted = false;

    if constexpr (!with_mag) { (void)mag_meas_body; return; }

    const Vec3 mag_meas = deheel_vector_(mag_meas_body);
    if (!mag_meas.allFinite()) return;
    const T n = mag_meas.norm();
    if (!(n > T(1e-6))) return;

    // predicted mag in body'
    Vec3 v2hat = R_wb() * B_world_ref_;
    // keep consistent sign
    if (v2hat.dot(mag_meas) < T(0)) v2hat = -v2hat;

    const Vec3 r = mag_meas - v2hat;
    last_mag_.r = r;

    const Mat3 J_att = -skew3<T>(v2hat);

    Mat3 S = Rmag_;
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH, OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    Eigen::LDLT<Mat3> ldlt;
    if (!safe_ldlt3_(S, ldlt)) return;

    last_mag_.S = S;
    last_mag_.nis = nis3_from_ldlt_(r, ldlt);

    PCt_mag_.setZero();
    PCt_mag_.noalias() += P_.template block<NX,3>(0, OFF_DTH) * J_att.transpose();

    K_mag_.noalias() = PCt_mag_ * ldlt.solve(Mat3::Identity());

    x_.noalias() += K_mag_ * r;

    joseph_update3_(P_, K_mag_, PCt_mag_, S);

    symmetrize_P_();
    regularize_covariance_(/*strong=*/false);
    symmetrize_P_();

    applyQuaternionCorrectionFromErrorState_();
    last_mag_.accepted = true;
  }

private:
  // ===== Constants / tuning =====
  const T gravity_magnitude_ = T(9.80665);

  // ===== MEKF internals =====
  Eigen::Quaternion<T> qref_;           // WORLD->BODY'
  Vec3 B_world_ref_ = Vec3::UnitX();    // only if with_mag

  VecX x_;
  MatX P_;

  // Base process noise rates
  Mat3 Qg_  = Mat3::Identity() * T(1e-6);
  Mat3 Qbg_ = Mat3::Identity() * T(1e-11);

  // Acc bias
  T sigma_bacc0_ = T(0.004);
  Mat3 Q_bacc_ = Mat3::Identity() * T(1e-6);
  static constexpr T tempC_ref_ = T(35.0);
  Vec3 k_a_ = Vec3::Constant(T(0.002));

  // Measurement noise
  Mat3 Racc_ = Mat3::Identity() * T(0.04);
  Mat3 Rmag_ = Mat3::Identity() * T(1.0);

  // Wave params
  T   omega_[KMODES]{};
  T   zeta_[KMODES]{};
  T   weights_[KMODES]{};
  Vec3 q_axis_[KMODES]{};

  bool wave_block_enabled_ = true;
  bool acc_bias_updates_enabled_ = true;

  // Warm-up
  bool warmup_mode_ = true;
  Vec3 gyro_bias_accum_ = Vec3::Zero();
  int  gyro_bias_accum_count_ = 0;
  Vec3 mag_ref_accum_ = Vec3::Zero();
  int  mag_ref_accum_count_ = 0;

  // Motion detection
  Vec3 init_vel_world_ = Vec3::Zero();
  T motion_distance_ = T(0);
  T motion_time_ = T(0);
  T exit_min_distance_ = T(10.0);
  T exit_min_time_     = T(5.0);

  // Lever-arm caches (BODY')
  bool use_imu_lever_arm_ = false;
  Vec3 r_imu_wrt_cog_body_phys_ = Vec3::Zero();

  // gyro/alpha caches (BODY')
  Vec3 prev_omega_b_ = Vec3::Zero();
  Vec3 alpha_b_      = Vec3::Zero();
  bool have_prev_omega_ = false;
  Vec3 last_gyr_bias_corrected_ = Vec3::Zero();
  T last_dt_ = T(1.0/240);
  T alpha_smooth_tau_ = T(0.05);

  // Heel
  T wind_heel_rad_ = T(0);
  T cos_unheel_x_  = T(1);
  T sin_unheel_x_  = T(0);

  // Diags
  MeasDiag3 last_acc_;
  MeasDiag3 last_mag_;

  // ===== Temps (OU-parity, low memory) =====
  Eigen::Matrix<T,BASE_N,BASE_N> F_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> Q_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> tmp_AA_;

  Eigen::Matrix<T,6,6> Phi6_;
  Eigen::Matrix<T,6,6> Qd6_;
  Eigen::Matrix<T,6,6> tmp6_;
  Eigen::Matrix<T,2,2> Phi2_[3];
  Eigen::Matrix<T,2,2> Qd2_[3];

  Eigen::Matrix<T,BASE_N,6> tmp_Ak_; // BASE_N x 6 cross-cov temp

  MatX3 PCt_acc_;
  MatX3 K_acc_;

  MatX3 PCt_mag_;
  MatX3 K_mag_;

private:
  // rotation helpers
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }             // world->body'
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); } // body'->world

  // Type-aware diagonal floor (float vs double)
  static inline T type_floor_diag_(T hint) {
    const T eps = std::numeric_limits<T>::epsilon();
    // For float, eps ~ 1e-7; for double ~ 2e-16.
    // Keep a practical minimum that is not subnormal-noise.
    const T base = T(100) * eps;
    const T minp = (std::is_same<T,float>::value) ? T(1e-9) : T(1e-14);
    return std::max(std::max(base, minp), hint);
  }

  // Regularize covariance; strong mode bumps more aggressively
  void regularize_covariance_(bool strong) {
    const T mind = type_floor_diag_(T(0));
    for (int i=0;i<NX;++i) {
      if (!std::isfinite((double)P_(i,i)) || P_(i,i) < mind) P_(i,i) = mind;
    }
    if (strong) {
      // mild diagonal bump scaled by trace to recover from near-indefinite states
      T tr = T(0);
      for (int i=0;i<NX;++i) tr += P_(i,i);
      const T bump = std::max(mind, T(1e-6) * (T(1) + tr / T(NX)));
      for (int i=0;i<NX;++i) P_(i,i) += bump;
    }
  }

  void symmetrize_P_() {
    for (int i=0;i<NX;++i) {
      for (int j=i+1;j<NX;++j) {
        const T v = T(0.5) * (P_(i,j) + P_(j,i));
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

  // ===== OU-style safe LDLT (3x3) =====
  static inline bool safe_ldlt3_(Mat3& S, Eigen::LDLT<Mat3>& ldlt) {
    // symmetrize first
    S = T(0.5) * (S + S.transpose());

    const T eps = std::numeric_limits<T>::epsilon();
    auto try_factor = [&](T bump)->bool {
      Mat3 St = S;
      St.diagonal().array() += bump;
      // floor diag as well
      const T mind = type_floor_diag_(T(0));
      for (int i=0;i<3;++i) St(i,i) = std::max(St(i,i), mind);
      ldlt.compute(St);
      if (ldlt.info() != Eigen::Success) return false;
      // accept and store back
      S = St;
      return true;
    };

    // scale bump by trace
    const T tr = std::max(T(0), S.trace());
    const T base = std::max(type_floor_diag_(T(0)), T(1e2)*eps*(T(1)+tr));
    if (try_factor(T(0))) return true;
    if (try_factor(base)) return true;
    if (try_factor(T(1e3)*base)) return true;
    if (try_factor(T(1e6)*base)) return true;
    return false;
  }

  static inline T nis3_from_ldlt_(const Vec3& r, const Eigen::LDLT<Mat3>& ldlt) {
    Vec3 y = ldlt.solve(r);
    if (!y.allFinite()) return std::numeric_limits<T>::infinity();
    const T v = r.dot(y);
    if (!std::isfinite((double)v)) return std::numeric_limits<T>::infinity();
    return v;
  }

  // ===== OU-style Joseph update without H =====
  // P <- P - K*PCt^T - PCt*K^T + K*S*K^T
  static inline void joseph_update3_(MatX& P, const MatX3& K, const MatX3& PCt, const Mat3& S) {
    // Use low-memory, symmetric-safe form
    // term1 = K*PCt^T  (NX x NX)
    MatX term = K * PCt.transpose();
    // P = P - term - term^T + K*S*K^T
    P.noalias() -= term;
    P.noalias() -= term.transpose();
    P.noalias() += (K * S) * K.transpose();
  }

  // ===== Oscillator discretization (per axis) with cancellation guards =====

  // Helper: sin(x)/x with series
  static inline T sinc_(T x) {
    const T ax = std::abs(x);
    if (ax < T(1e-4)) {
      const T x2 = x*x;
      return T(1) - x2/T(6) + (x2*x2)/T(120);
    }
    return std::sin(x) / x;
  }

  // Helper: expm1 for T
  static inline T expm1_(T x) {
    // std::expm1 exists in <cmath> but some embedded toolchains omit it; provide stable fallback.
    #if defined(__cpp_lib_math_special_functions) || defined(__GNUC__) || defined(_MSC_VER)
      return std::expm1(x);
    #else
      const T ax = std::abs(x);
      if (ax < T(1e-4)) {
        const T x2 = x*x;
        return x + x2/T(2) + x2*x/T(6);
      }
      return std::exp(x) - T(1);
    #endif
  }

  static inline void phi_osc_2x2_(T t, T w, T z, Eigen::Matrix<T,2,2>& Phi) {
    const T om = std::max(T(1e-8), w);
    const T ze = std::max(T(0), z);

    // Handle extremely small dt
    if (std::abs(t) < T(1e-9)) {
      Phi.setIdentity();
      Phi(0,1) = t;
      Phi(1,0) = -(om*om)*t;
      Phi(1,1) = T(1) - T(2)*ze*om*t;
      return;
    }

    // near critical: use critically damped closed form with a = ze*om ~ om
    // We treat ze within ~1e-3 as critical to avoid wd -> 0 cancellation
    if (std::abs(ze - T(1)) < T(1e-3)) {
      const T a = ze * om;
      const T e = std::exp(-a * t);
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * (t);
      Phi(1,0) = e * (-(om*om)*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }

    if (ze < T(1)) {
      const T one_m_zz = std::max(T(0), T(1) - ze*ze);
      const T wd = om * std::sqrt(one_m_zz);
      // If wd is tiny anyway, fall back to critical form (safer)
      if (wd < T(1e-6)) {
        const T a = ze * om;
        const T e = std::exp(-a * t);
        Phi(0,0) = e * (T(1) + a*t);
        Phi(0,1) = e * (t);
        Phi(1,0) = e * (-(om*om)*t);
        Phi(1,1) = e * (T(1) - a*t);
        return;
      }

      const T a = ze * om;
      const T e = std::exp(-a * t);
      const T x = wd * t;

      // use sinc for sin(x)/x stability
      const T s_over_wd = t * sinc_(x);         // sin(wd t)/wd
      const T c = std::cos(x);
      const T s = std::sin(x);
      const T a_over_wd = a / wd;

      Phi(0,0) = e * (c + a_over_wd * s);
      Phi(0,1) = e * (s_over_wd);
      Phi(1,0) = e * (-(om*om) * s_over_wd);
      Phi(1,1) = e * (c - a_over_wd * s);
      return;
    }

    // Overdamped: use expm1 to reduce cancellation when (r2-r1)t small
    const T s = std::sqrt(std::max(T(0), ze*ze - T(1)));
    const T r1 = -om * (ze - s);
    const T r2 = -om * (ze + s);

    const T dr = (r2 - r1);
    if (std::abs(dr) < T(1e-9)) {
      // fallback to critical-like
      const T a = ze * om;
      const T e = std::exp(-a * t);
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * (t);
      Phi(1,0) = e * (-(om*om)*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }

    const T e1 = std::exp(r1 * t);
    const T e2 = std::exp(r2 * t);

    const T invd  = T(1) / dr;

    // stable differences
    const T de = e2 - e1; // OK; if needed, could use expm1 around mid, but dr guard above usually enough

    Phi(0,0) = (r2*e1 - r1*e2) * invd;
    Phi(0,1) = (de) * invd;
    Phi(1,0) = (r1*r2) * (e1 - e2) * invd;
    Phi(1,1) = (r2*e2 - r1*e1) * invd;
  }

  static inline void discretize_osc_axis_(T dt, T w, T z, T q,
                                         Eigen::Matrix<T,2,2>& Phi,
                                         Eigen::Matrix<T,2,2>& Qd)
  {
    // For very small dt (or small w*dt), use series for Qd to avoid underflow in float
    const T om = std::max(T(1e-8), w);
    const T x = om * dt;

    phi_osc_2x2_(dt, w, z, Phi);

    const T small = (std::is_same<T,float>::value) ? T(2e-3) : T(2e-5);
    if (std::abs(dt) < small || std::abs(x) < small) {
      // v' = ... + ξ, with Var(ξ)=q (PSD). Over small dt:
      // Qvv ≈ q dt, Qpv ≈ q dt^2 / 2, Qpp ≈ q dt^3 / 3
      const T d1 = dt;
      const T d2 = dt*dt;
      const T d3 = d2*dt;
      Qd(0,0) = std::max(T(0), q * d3 / T(3));
      Qd(0,1) = q * d2 / T(2);
      Qd(1,0) = Qd(0,1);
      Qd(1,1) = std::max(T(0), q * d1);
      return;
    }

    // Simpson integration on G(t) = Phi(t) * L * q * L^T * Phi(t)^T, with L = [0;1]
    auto G = [&](T t)->Eigen::Matrix<T,2,2> {
      Eigen::Matrix<T,2,2> Pt;
      phi_osc_2x2_(t, w, z, Pt);
      const T u0 = Pt(0,1);
      const T u1 = Pt(1,1);
      Eigen::Matrix<T,2,2> M;
      M(0,0) = q * u0*u0;
      M(0,1) = q * u0*u1;
      M(1,0) = M(0,1);
      M(1,1) = q * u1*u1;
      return M;
    };

    const T h = dt;
    const auto G0 = G(T(0));
    const auto G1 = G(T(0.5)*h);
    const auto G2 = G(h);

    Qd = (h / T(6)) * (G0 + T(4)*G1 + G2);

    // symmetrize and clamp PSD-ish
    Qd = T(0.5) * (Qd + Qd.transpose());
    Qd(0,0) = std::max(Qd(0,0), T(0));
    Qd(1,1) = std::max(Qd(1,1), T(0));
    const T max_off = std::sqrt(std::max(T(0), Qd(0,0) * Qd(1,1)));
    Qd(0,1) = std::max(-max_off, std::min(Qd(0,1), max_off));
    Qd(1,0) = Qd(0,1);
  }

  // ===== Heel helpers (MATCH OU de-heel about +X) =====
  void update_unheel_trig_() {
    if (std::abs(wind_heel_rad_) < T(1e-12)) {
      cos_unheel_x_ = T(1);
      sin_unheel_x_ = T(0);
    } else {
      const T angle = -wind_heel_rad_;
      cos_unheel_x_ = std::cos(angle);
      sin_unheel_x_ = std::sin(angle);
    }
  }

  Vec3 deheel_vector_(const Vec3& v_body) const {
    if (std::abs(wind_heel_rad_) < T(1e-12)) return v_body;
    Vec3 v;
    v.x() = v_body.x();
    v.y() = cos_unheel_x_ * v_body.y() - sin_unheel_x_ * v_body.z();
    v.z() = sin_unheel_x_ * v_body.y() + cos_unheel_x_ * v_body.z();
    return v;
  }

  Mat3 Rx_(T angle_rad) const {
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

    // Rotate the BODY' frame so the physical BODY remains continuous
    const Mat3 R = Rx_(-delta_heel_rad);
    const Eigen::Quaternion<T> qR(R);

    qref_ = qR * qref_;
    qref_.normalize();

    // rotate error state pieces that live in BODY'
    x_.template segment<3>(OFF_DTH) = R * x_.template segment<3>(OFF_DTH);
    if constexpr (with_gyro_bias)  x_.template segment<3>(OFF_BG) = R * x_.template segment<3>(OFF_BG);
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = R * x_.template segment<3>(OFF_BA);

    last_gyr_bias_corrected_ = R * last_gyr_bias_corrected_;
    prev_omega_b_            = R * prev_omega_b_;
    alpha_b_                 = R * alpha_b_;

    // transform covariance with block-diag similarity
    MatX Tm = MatX::Identity();
    Tm.template block<3,3>(OFF_DTH, OFF_DTH) = R;
    if constexpr (with_gyro_bias)  Tm.template block<3,3>(OFF_BG, OFF_BG) = R;
    if constexpr (with_accel_bias) Tm.template block<3,3>(OFF_BA, OFF_BA) = R;

    P_ = Tm * P_ * Tm.transpose();
    symmetrize_P_();
    regularize_covariance_(/*strong=*/true);
    symmetrize_P_();
  }
};
