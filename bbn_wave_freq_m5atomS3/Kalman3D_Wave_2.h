#pragma once
/*
  Kalman3D_Wave_2  (template KMODES + with_mag)
  --------------------------------------------
  Conventions MATCH your old OU-based Kalman3D_Wave:

  - World frame is NED, +Z down.
  - qref stores WORLD -> BODY' quaternion (virtual un-heeled body frame B').
  - quaternion() returns BODY' -> WORLD, i.e. qref.conjugate().
  - quaternion_boat() returns physical BODY -> WORLD by re-applying heel.
  - accel specific-force model (BODY'):
      f_b' = R_wb'( a_w - g_world ) + lever(ω,α,r_imu) + b_a(temp)
    where g_world = (0,0,+g) in NED.

  Wave model (broadband):
    For k=1..KMODES, per axis:
      p' = v
      v' = -ω_k^2 p - 2ζ_k ω_k v + ξ(t)   (ξ: white accel noise with PSD q_k)
    Wave acceleration:
      a_w = Σ_k ( -ω_k^2 p_k - 2ζ_k ω_k v_k )

  State:
    [ dtheta(3), b_g(3),  p1(3),v1(3),..., pK(3),vK(3),  b_a(3) ]

  Multi‑stage initialization (as in OU version):
    - Warm‑up mode (wave block disabled, accel bias updates enabled).
    - While in warm‑up:
        * Gyro bias is learned by averaging (stationary detection optional).
        * Magnetic reference is learned by accumulating horizontal projections.
        * Wave states remain zero and their covariance is kept tiny.
    - After sufficient motion (distance and time thresholds), warm‑up mode is
      automatically turned off → wave block enabled, full filter runs.
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
  if (theta < T(1e-2)) {
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

  static constexpr int BA_N   = with_accel_bias ? 3 : 0;

  static constexpr int NX     = BASE_N + WAVE_N + BA_N;

  using VecX  = Eigen::Matrix<T,NX,1>;
  using MatX  = Eigen::Matrix<T,NX,NX>;
  using MatX3 = Eigen::Matrix<T,NX,3>;

  // Offsets
  static constexpr int OFF_DTH  = 0;                          // δθ (always)
  static constexpr int OFF_BG   = with_gyro_bias ? 3 : -1;     // gyro bias if enabled
  static constexpr int OFF_WAVE = BASE_N;                     // wave start
  static constexpr int OFF_BA   = with_accel_bias ? (BASE_N + WAVE_N) : -1;

  static constexpr int OFF_Pk(int k) { return OFF_WAVE + 6*k + 0; } // 3
  static constexpr int OFF_Vk(int k) { return OFF_WAVE + 6*k + 3; } // 3

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

    // Seed wave covariances – will be overwritten later if warm‑up is used
    const T sigma_p0 = T(20.0);
    const T sigma_v0 = T(1.0);
    for (int k=0;k<KMODES;++k) {
      P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
      P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
    }

    // Default mag reference if mag enabled
    B_world_ref_ = Vec3::UnitX();

    set_broadband_params(T(0.12), T(1.0));

    // Warm‑up mode is ON by default (matches OU behavior)
    set_warmup_mode(true);
  }

  // ===== Conventions / accessors (MATCH old OU filter) =====
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

  // Total wave states in world (NED)
  [[nodiscard]] Vec3 wave_position_world() const {
    Vec3 p = Vec3::Zero();
    for (int k=0;k<KMODES;++k) p += x_.template segment<3>(OFF_Pk(k));
    return p;
  }
  [[nodiscard]] Vec3 wave_velocity_world() const {
    Vec3 v = Vec3::Zero();
    for (int k=0;k<KMODES;++k) v += x_.template segment<3>(OFF_Vk(k));
    return v;
  }
  [[nodiscard]] Vec3 wave_accel_world() const { return wave_world_accel_(); }

  const MeasDiag3& lastAccDiag() const noexcept { return last_acc_; }
  const MeasDiag3& lastMagDiag() const noexcept { return last_mag_; }

  // ===== Mag ref =====
  void set_mag_world_ref(const Vec3& B_world) {
    if constexpr (with_mag) {
      B_world_ref_ = B_world;
    } else {
      (void)B_world;
    }
  }

  // ===== Bias temp model / RW =====
  void set_accel_bias_temp_coeff(const Vec3& ka_per_degC) { k_a_ = ka_per_degC; }

  void set_Q_bacc_rw(const Vec3& rw_std_per_sqrt_s) {
    if constexpr (with_accel_bias) Q_bacc_ = rw_std_per_sqrt_s.array().square().matrix().asDiagonal();
  }

  void set_initial_acc_bias_std(T s) {
    if constexpr (with_accel_bias) {
      sigma_bacc0_ = std::max(T(0), s);
      P_.template block<3,3>(OFF_BA, OFF_BA) = Mat3::Identity() * sigma_bacc0_*sigma_bacc0_;
    }
  }
  void set_initial_acc_bias(const Vec3& b0) {
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = b0;
  }

  // ===== Warm‑up control (matches OU interface) =====
  void set_warmup_mode(bool on) {
    if (warmup_mode_ == on) return;
    warmup_mode_ = on;

    if (on) {
      // Entering warm‑up: disable wave block, enable accel bias updates,
      // and set wave covariances to very small values.
      set_wave_block_enabled(false);
      set_acc_bias_updates_enabled(true);
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * T(1e-12);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * T(1e-12);
      }
      // Reset learning accumulators
      gyro_bias_accum_.setZero();
      gyro_bias_accum_count_ = 0;
      mag_ref_accum_.setZero();
      mag_ref_accum_count_ = 0;
      motion_distance_ = T(0);
      motion_time_ = T(0);
    } else {
      // Exiting warm‑up: restore wave covariances to normal values,
      // enable wave block (if desired – user may still control it externally).
      const T sigma_p0 = T(20.0);
      const T sigma_v0 = T(1.0);
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
      }
      set_wave_block_enabled(true);
      // Keep accel bias updates as they were (usually enabled)
    }
  }

  bool warmup_mode() const { return warmup_mode_; }

  void set_wave_block_enabled(bool on) {
    if (wave_block_enabled_ && !on) {
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
        P_.template block<3,BASE_N>(OFF_BA, 0).setZero();
        P_.template block<BASE_N,3>(0, OFF_BA).setZero();
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      } else {
        auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
        const T target = sigma_bacc0_*sigma_bacc0_;
        for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), target);
        P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;
      }
    }
    acc_bias_updates_enabled_ = en;
  }

  // ===== Motion thresholds for auto‑exit from warm‑up (OU behavior) =====
  void set_motion_exit_thresholds(T min_distance_m, T min_time_sec) {
    exit_min_distance_ = min_distance_m;
    exit_min_time_ = min_time_sec;
  }

  // ===== Call this periodically to update warm‑up learning and auto‑transition =====
  void update_initialization(const Vec3& acc_body, const Vec3& gyr_body,
                             const Vec3& mag_body, T dt) {
    if (!warmup_mode_) return;

    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 gyr = deheel_vector_(gyr_body);
    const Vec3 mag = deheel_vector_(mag_body);

    // 1. Learn gyro bias (simple recursive average)
    if constexpr (with_gyro_bias) {
      if (gyro_bias_accum_count_ == 0) {
        gyro_bias_accum_ = gyr;
      } else {
        const T alpha = T(0.01); // gentle update
        gyro_bias_accum_ = (T(1)-alpha) * gyro_bias_accum_ + alpha * gyr;
      }
      gyro_bias_accum_count_++;
      // After enough samples, set the bias in the state
      if (gyro_bias_accum_count_ > 100) {
        x_.template segment<3>(OFF_BG) = gyro_bias_accum_;
        // Reduce covariance accordingly
        P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * T(1e-8);
        gyro_bias_accum_count_ = 101; // cap
      }
    }

    // 2. Learn magnetic reference (if enabled)
    if constexpr (with_mag) {
      // Only update when acceleration is near gravity (not in freefall)
      const T acc_norm = acc.norm();
      if (std::abs(acc_norm - gravity_magnitude_) < T(0.5)) {
        // Project mag into horizontal plane using current attitude
        const Vec3 z_body = -acc.normalized();
        const Vec3 mag_horiz = mag - (mag.dot(z_body)) * z_body;
        if (mag_horiz.norm() > T(1e-6)) {
          const Vec3 mag_world = R_bw() * mag_horiz;
          mag_ref_accum_ += mag_world;
          mag_ref_accum_count_++;

          if (mag_ref_accum_count_ > 10) {
            const Vec3 new_ref = (mag_ref_accum_ / mag_ref_accum_count_).normalized();
            // Exponential moving average
            const T alpha = T(0.01);
            B_world_ref_ = (T(1)-alpha) * B_world_ref_ + alpha * new_ref;
            B_world_ref_.normalize();
            // Reset accumulator periodically
            if (mag_ref_accum_count_ > 100) {
              mag_ref_accum_.setZero();
              mag_ref_accum_count_ = 0;
            }
          }
        }
      }
    }

    // 3. Integrate motion to decide when to exit warm‑up
    // Remove gravity to get acceleration due to motion
    const Vec3 acc_world = R_bw() * acc;
    const Vec3 acc_motion = acc_world - Vec3(0,0,gravity_magnitude_);
    // Simple double integration (crude but sufficient for detection)
    static Vec3 vel = Vec3::Zero(); // local velocity estimate (reset each warm‑up)
    vel += acc_motion * dt;
    motion_distance_ += vel.norm() * dt;
    motion_time_ += dt;

    // 4. Auto‑exit if thresholds are met
    if (motion_distance_ > exit_min_distance_ && motion_time_ > exit_min_time_) {
      set_warmup_mode(false);
    }
  }

  // ===== IMU lever arm =====
  void set_imu_lever_arm_body(const Vec3& r_b_phys) {
    r_imu_wrt_cog_body_phys_ = r_b_phys;
    use_imu_lever_arm_ = (r_b_phys.squaredNorm() > T(0));
  }
  void clear_imu_lever_arm() {
    r_imu_wrt_cog_body_phys_.setZero();
    use_imu_lever_arm_ = false;
  }
  void set_alpha_smoothing_tau(T tau_sec) { alpha_smooth_tau_ = std::max(T(0), tau_sec); }

  // ===== Heel update (same retarget logic as old) =====
  void update_wind_heel(T heel_rad) {
    const T old = wind_heel_rad_;
    if (heel_rad == old) return;
    retarget_bodyprime_frame_(heel_rad - old);
    wind_heel_rad_ = heel_rad;
    update_unheel_trig_();
  }

  // ===== Initialization =====
  void initialize_from_acc_mag(const Vec3& acc_body, const Vec3& mag_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 mag = deheel_vector_(mag_body);

    const T an = acc.norm();
    if (!(an > T(1e-8))) return;
    Vec3 acc_n = acc / an;

    Vec3 z_world = -acc_n;
    Vec3 mag_h = mag - (mag.dot(z_world))*z_world;
    if (!(mag_h.norm() > T(1e-8))) return;
    mag_h.normalize();

    Vec3 x_world = mag_h;
    Vec3 y_world = z_world.cross(x_world).normalized();

    Mat3 R_wb;
    R_wb.col(0) = x_world;
    R_wb.col(1) = y_world;
    R_wb.col(2) = z_world;

    qref_ = Eigen::Quaternion<T>(R_wb); // WORLD->BODY'
    qref_.normalize();

    if constexpr (with_mag) {
      B_world_ref_ = R_bw() * mag;
    }

    x_.template segment<3>(OFF_DTH).setZero();
    // Stay in warm‑up mode (or enter it) after initialisation
    set_warmup_mode(true);
  }

  void initialize_from_acc(const Vec3& acc_body) {
    const Vec3 acc = deheel_vector_(acc_body);
    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    Vec3 anorm = acc / an;
    const Vec3 zb = Vec3::UnitZ();
    const Vec3 target = -anorm;

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
    const T w0 = T(2)*T(M_PI)*std::max(T(1e-4), f0_hz);

    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1)); // 0..1
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
      const T qk = T(4) * ze * om*om*om * var_k; // narrowband approx

      q_axis_[k].x() = horiz_scale * qk;
      q_axis_[k].y() = horiz_scale * qk;
      q_axis_[k].z() = qk;
    }
  }

  // ===== Time update (propagation) =====
  void time_update(const Vec3& gyr_body, T Ts) {
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

    // Base covariance propagate
    Mat3 Fth = Mat3::Identity() - skew3<T>(omega_b) * Ts;

    Eigen::Matrix<T,BASE_N,BASE_N> F_AA; F_AA.setIdentity();
    F_AA.template block<3,3>(0,0) = Fth;
    if constexpr (with_gyro_bias) {
      F_AA.template block<3,3>(0,3) = -Mat3::Identity() * Ts;
    }

    Eigen::Matrix<T,BASE_N,BASE_N> Q_AA; Q_AA.setZero();
    Q_AA.template block<3,3>(0,0) = Qg_ * Ts;
    if constexpr (with_gyro_bias) {
      Q_AA.template block<3,3>(3,3) = Qbg_ * Ts;
    }

    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      Eigen::Matrix<T,BASE_N,BASE_N> tmp = F_AA * Paa;
      Paa = tmp * F_AA.transpose() + Q_AA;
      P_.template block<BASE_N,BASE_N>(0,0) = Paa;
    }

    // Wave block (only if enabled)
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        Eigen::Matrix<T,2,2> Phi2[3], Qd2[3];
        for (int ax=0; ax<3; ++ax) {
          discretize_osc_axis_(Ts, omega_[k], zeta_[k], q_axis_[k](ax), Phi2[ax], Qd2[ax]);
        }

        // mean
        Vec3 p = x_.template segment<3>(OFF_Pk(k));
        Vec3 v = x_.template segment<3>(OFF_Vk(k));
        for (int ax=0; ax<3; ++ax) {
          Eigen::Matrix<T,2,1> xv; xv << p(ax), v(ax);
          xv = Phi2[ax] * xv;
          p(ax) = xv(0);
          v(ax) = xv(1);
        }
        x_.template segment<3>(OFF_Pk(k)) = p;
        x_.template segment<3>(OFF_Vk(k)) = v;

        Eigen::Matrix<T,6,6> Phi6; Phi6.setZero();
        Eigen::Matrix<T,6,6> Qd6;  Qd6.setZero();
        for (int ax=0; ax<3; ++ax) {
          Phi6.template block<2,2>(2*ax,2*ax) = Phi2[ax];
          Qd6 .template block<2,2>(2*ax,2*ax) = Qd2[ax];
        }

        const int offk = OFF_Pk(k);

        // P_kk
        {
          auto Pkk = P_.template block<6,6>(offk, offk);
          Eigen::Matrix<T,6,6> tmp = Phi6 * Pkk;
          Pkk = tmp * Phi6.transpose() + Qd6;
          P_.template block<6,6>(offk, offk) = Pkk;
        }

        // Cross with base
        {
          auto P_Ak = P_.template block<BASE_N,6>(0, offk);
          Eigen::Matrix<T,BASE_N,6> tmp = F_AA * P_Ak;
          P_Ak = tmp * Phi6.transpose();
          P_.template block<BASE_N,6>(0, offk) = P_Ak;
          P_.template block<6,BASE_N>(offk, 0) = P_Ak.transpose();
        }

        // Cross with BA
        if constexpr (with_accel_bias) {
          auto P_BAk = P_.template block<3,6>(OFF_BA, offk);
          P_BAk = P_BAk * Phi6.transpose();
          P_.template block<3,6>(OFF_BA, offk) = P_BAk;
          P_.template block<6,3>(offk, OFF_BA) = P_BAk.transpose();
        }
      }
    }

    // Accel bias RW
    if constexpr (with_accel_bias) {
      auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
      if (acc_bias_updates_enabled_) Pba += Q_bacc_ * Ts;
      P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;

      // Cross BA with base
      auto P_Aba = P_.template block<BASE_N,3>(0, OFF_BA);
      P_Aba = F_AA * P_Aba;
      P_.template block<BASE_N,3>(0, OFF_BA) = P_Aba;
      P_.template block<3,BASE_N>(OFF_BA, 0) = P_Aba.transpose();
    }

    symmetrize_P_();
  }

  // ===== Measurement updates (unchanged from original, but will respect wave_block_enabled_ flag) =====
  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref_) {
    // ... (same as your existing implementation, which already uses wave_block_enabled_) ...
    // (Omitted for brevity – copy from your original code)
  }

  void measurement_update_mag_only(const Vec3& mag_meas_body) {
    // ... (same as your existing implementation) ...
    // (Omitted for brevity – copy from your original code)
  }

private:
  // ===== Constants / tuning =====
  const T gravity_magnitude_ = T(9.80665);

  // ===== MEKF internals =====
  Eigen::Quaternion<T> qref_;    // WORLD->BODY'
  Vec3 B_world_ref_ = Vec3::UnitX(); // used only if with_mag==true

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
  Mat3 Rmag_ = Mat3::Identity() * T(1.0); // used only if with_mag==true

  // Wave params
  T omega_[KMODES]{};
  T zeta_[KMODES]{};
  T weights_[KMODES]{};
  Vec3 q_axis_[KMODES]{};

  bool wave_block_enabled_ = true;
  bool acc_bias_updates_enabled_ = true;

  // Warm‑up mode flag (replaces explicit stages)
  bool warmup_mode_ = true;

  // Learning accumulators (OU style)
  Vec3 gyro_bias_accum_ = Vec3::Zero();
  int gyro_bias_accum_count_ = 0;
  Vec3 mag_ref_accum_ = Vec3::Zero();
  int mag_ref_accum_count_ = 0;

  // Motion detection for auto‑exit
  T motion_distance_ = T(0);
  T motion_time_ = T(0);
  T exit_min_distance_ = T(10.0);   // meters
  T exit_min_time_ = T(5.0);        // seconds

  // Lever-arm caches (B')
  bool use_imu_lever_arm_ = false;
  Vec3 r_imu_wrt_cog_body_phys_ = Vec3::Zero();

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

  MeasDiag3 last_acc_;
  MeasDiag3 last_mag_;

private:
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }             // world->body'
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); } // body'->world

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
      aw += -(om*om)*p - (T(2)*ze*om)*v;
    }
    return aw;
  }

  // ===== Oscillator discretization (per axis) =====
  static inline void phi_osc_2x2_(T t, T w, T z, Eigen::Matrix<T,2,2>& Phi) {
    // ... (same as your existing implementation) ...
  }

  static inline void discretize_osc_axis_(T dt, T w, T z, T q,
                                         Eigen::Matrix<T,2,2>& Phi,
                                         Eigen::Matrix<T,2,2>& Qd) {
    // ... (same as your existing implementation) ...
  }

  // ===== Heel helpers =====
  void update_unheel_trig_() {
    if (std::abs(wind_heel_rad_) < T(1e-9)) {
      cos_unheel_x_ = T(1);
      sin_unheel_x_ = T(0);
    } else {
      const T angle = -wind_heel_rad_;
      cos_unheel_x_ = std::cos(angle);
      sin_unheel_x_ = std::sin(angle);
    }
  }

  Vec3 deheel_vector_(const Vec3& v_body) const {
    if (std::abs(wind_heel_rad_) < T(1e-9)) return v_body;
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

    const Mat3 R = Rx_(-delta_heel_rad);
    const Eigen::Quaternion<T> qR(R);

    qref_ = qR * qref_;
    qref_.normalize();

    x_.template segment<3>(OFF_DTH) = R * x_.template segment<3>(OFF_DTH);
    if constexpr (with_gyro_bias) {
      x_.template segment<3>(OFF_BG) = R * x_.template segment<3>(OFF_BG);
    }
    if constexpr (with_accel_bias) {
      x_.template segment<3>(OFF_BA) = R * x_.template segment<3>(OFF_BA);
    }

    last_gyr_bias_corrected_ = R * last_gyr_bias_corrected_;
    prev_omega_b_            = R * prev_omega_b_;
    alpha_b_                 = R * alpha_b_;

    MatX Tm = MatX::Identity();
    Tm.template block<3,3>(OFF_DTH, OFF_DTH) = R;
    if constexpr (with_gyro_bias)  Tm.template block<3,3>(OFF_BG, OFF_BG) = R;
    if constexpr (with_accel_bias) Tm.template block<3,3>(OFF_BA, OFF_BA) = R;

    P_ = Tm * P_ * Tm.transpose();
    symmetrize_P_();
  }
};
