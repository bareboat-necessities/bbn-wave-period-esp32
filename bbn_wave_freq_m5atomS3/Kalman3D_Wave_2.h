#pragma once
/*
  Kalman3D_Wave_2  (template KMODES + with_mag)
  --------------------------------------------
  Drop-in, OU-convention-matching broadband wave + qMEKF attitude/bias filter.

  Conventions MATCH your old OU-based Kalman3D_Wave:

  - World frame is NED, +Z down.
  - qref_ stores WORLD -> BODY' quaternion (virtual un-heeled body frame B').
  - quaternion() returns BODY' -> WORLD, i.e. qref_.conjugate().
  - quaternion_boat() returns physical BODY -> WORLD by re-applying heel.
  - Accelerometer specific-force model (BODY'):
        f_b' = R_wb'( a_w - g_world ) + lever(ω,α,r_imu) + b_a(temp)
    where g_world = (0,0,+g) in NED.

  Wave model (broadband, KMODES modes per axis):
    For k = 1..KMODES, per axis:
      p' = v
      v' = -ω_k^2 p - 2ζ_k ω_k v + ξ(t)   (ξ: white accel input, PSD q_k)
    World wave acceleration:
      a_w = Σ_k ( -ω_k^2 p_k - 2ζ_k ω_k v_k )

  State:
    [ dtheta(3),
      (b_g(3) if with_gyro_bias),
      p1(3),v1(3),...,pK(3),vK(3),
      (b_a(3) if with_accel_bias)
    ]

  Warm-up / staged initialization:
    - Warm-up mode ON by default:
        * wave block disabled (no propagation / no accel coupling)
        * gyro bias learned by averaging
        * mag world ref learned by accumulating horizontal projections (if with_mag)
        * accel bias updates enabled (you can disable if you want)
    - Auto-exit warm-up after motion thresholds (distance/time), then wave enabled.

  Numerical safety:
    - small-angle quaternion correction (Maclaurin)
    - stable oscillator Phi with under/critical/over damped branches and fallbacks
    - Simpson integration for Qd with SPD clamping
    - robust LDLT for 3x3 innovations with diagonal bumps
    - Joseph covariance update in stack-light form (no NX×NX temporaries)
    - covariance symmetrization + diagonal floor
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
    // cos(theta/2) ≈ 1 - θ²/8 + θ⁴/384
    w = T(1) - t2*(T(1)/T(8)) + t4*(T(1)/T(384));
    // sin(theta/2)/θ ≈ 1/2 - θ²/48 + θ⁴/3840
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

    // Base process noise rates (continuous-time discretized as *dt)
    Qg_  = sigma_g_rw.array().square().matrix().asDiagonal();  // drives δθ
    if constexpr (with_gyro_bias) {
      Qbg_ = Mat3::Identity() * b0;                            // bias RW rate
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

    // Seed wave covariances (reasonable large; will be reduced in warm-up)
    const T sigma_p0 = T(20.0);
    const T sigma_v0 = T(1.0);
    for (int k=0;k<KMODES;++k) {
      P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
      P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
    }

    // Default mag reference if mag enabled
    B_world_ref_ = Vec3::UnitX();

    // Default broadband params (you will probably override)
    set_broadband_params(T(0.12), T(1.0));

    // Warm-up mode ON by default (matches OU staged behavior)
    set_warmup_mode(true, /*force_reset=*/true);
  }

  // ===== Conventions / accessors (MATCH old OU filter) =====
  [[nodiscard]] Eigen::Quaternion<T> quaternion() const { return qref_.conjugate(); } // BODY'->WORLD

  [[nodiscard]] Eigen::Quaternion<T> quaternion_boat() const {
    const Eigen::Quaternion<T> q_WBprime = quaternion();
    const T half = -wind_heel_rad_ * T(0.5);
    const T c = std::cos(half);
    const T s = std::sin(half);
    const Eigen::Quaternion<T> q_BprimeB(c, s, 0, 0); // B' <- B
    return q_WBprime * q_BprimeB;                     // B -> W
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

  // ===== Warm-up control =====
  void set_warmup_mode(bool on, bool force_reset=false) {
    if (!force_reset && warmup_mode_ == on) return;
    warmup_mode_ = on;

    if (on) {
      // Enter warm-up: disable wave block, enable accel bias updates, and shrink wave covariances.
      set_wave_block_enabled(false);
      set_acc_bias_updates_enabled(true);

      // Keep wave states at 0; keep their covariance tiny so they don't back-drive anything.
      for (int k=0;k<KMODES;++k) {
        x_.template segment<3>(OFF_Pk(k)).setZero();
        x_.template segment<3>(OFF_Vk(k)).setZero();
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * T(1e-12);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * T(1e-12);
        // also wipe cross-covariances with wave
        P_.template block<NX,3>(0, OFF_Pk(k)).setZero();
        P_.template block<NX,3>(0, OFF_Vk(k)).setZero();
        P_.template block<3,NX>(OFF_Pk(k), 0).setZero();
        P_.template block<3,NX>(OFF_Vk(k), 0).setZero();
        // restore their own diagonal after the wipe:
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * T(1e-12);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * T(1e-12);
      }

      gyro_bias_accum_.setZero();
      gyro_bias_accum_count_ = 0;
      mag_ref_accum_.setZero();
      mag_ref_accum_count_ = 0;

      motion_distance_ = T(0);
      motion_time_ = T(0);
      warmup_vel_est_.setZero();
    } else {
      // Exit warm-up: restore wave covariances and enable wave block.
      const T sigma_p0 = T(20.0);
      const T sigma_v0 = T(1.0);
      for (int k=0;k<KMODES;++k) {
        P_.template block<3,3>(OFF_Pk(k), OFF_Pk(k)) = Mat3::Identity() * (sigma_p0*sigma_p0);
        P_.template block<3,3>(OFF_Vk(k), OFF_Vk(k)) = Mat3::Identity() * (sigma_v0*sigma_v0);
      }
      set_wave_block_enabled(true);
    }

    regularize_covariance_();
    symmetrize_P_();
  }

  bool warmup_mode() const { return warmup_mode_; }

  void set_wave_block_enabled(bool on) {
    if (wave_block_enabled_ && !on) {
      // Decouple base<->wave and BA<->wave like OU version when disabling the block.
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
        // Prevent BA from being driven by other updates via cross-cov
        P_.template block<3,BASE_N>(OFF_BA, 0).setZero();
        P_.template block<BASE_N,3>(0, OFF_BA).setZero();
        P_.template block<3,WAVE_N>(OFF_BA, OFF_WAVE).setZero();
        P_.template block<WAVE_N,3>(OFF_WAVE, OFF_BA).setZero();
      } else {
        // inflate diag to at least initial
        auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
        const T target = sigma_bacc0_*sigma_bacc0_;
        for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), target);
        P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;
      }
    }
    acc_bias_updates_enabled_ = en;
  }

  // ===== Motion thresholds for auto-exit from warm-up =====
  void set_motion_exit_thresholds(T min_distance_m, T min_time_sec) {
    exit_min_distance_ = min_distance_m;
    exit_min_time_ = min_time_sec;
  }

  // ===== Call this periodically in warm-up to learn biases/ref and auto-exit =====
  void update_initialization(const Vec3& acc_body, const Vec3& gyr_body,
                             const Vec3& mag_body, T dt)
  {
    if (!warmup_mode_) return;

    const Vec3 acc = deheel_vector_(acc_body);
    const Vec3 gyr = deheel_vector_(gyr_body);
    const Vec3 mag = deheel_vector_(mag_body);

    // 1) Learn gyro bias (EMA; assumes mostly steady during warm-up)
    if constexpr (with_gyro_bias) {
      if (gyro_bias_accum_count_ == 0) {
        gyro_bias_accum_ = gyr;
      } else {
        const T a = T(0.01);
        gyro_bias_accum_ = (T(1)-a)*gyro_bias_accum_ + a*gyr;
      }
      ++gyro_bias_accum_count_;

      if (gyro_bias_accum_count_ > 100) {
        x_.template segment<3>(OFF_BG) = gyro_bias_accum_;
        // reduce its covariance
        P_.template block<3,3>(OFF_BG, OFF_BG) = Mat3::Identity() * T(1e-8);
        gyro_bias_accum_count_ = 101; // cap
      }
    }

    // 2) Learn mag world ref (horizontal projection) if enabled
    if constexpr (with_mag) {
      const T acc_norm = acc.norm();
      if (std::abs(acc_norm - gravity_magnitude_) < T(0.5)) {
        const Vec3 down_b = (-acc / std::max(T(1e-9), acc_norm)); // world-down in BODY'
        Vec3 mag_h = mag - (mag.dot(down_b))*down_b;
        const T mn = mag_h.norm();
        if (mn > T(1e-6)) {
          mag_h /= mn;
          const Vec3 mag_world = R_bw() * mag_h; // BODY'->WORLD
          mag_ref_accum_ += mag_world;
          ++mag_ref_accum_count_;
          if (mag_ref_accum_count_ > 10) {
            Vec3 new_ref = mag_ref_accum_ / std::max(T(1), T(mag_ref_accum_count_));
            const T nn = new_ref.norm();
            if (nn > T(1e-6)) {
              new_ref /= nn;
              const T a = T(0.01);
              B_world_ref_ = (T(1)-a)*B_world_ref_ + a*new_ref;
              const T bn = B_world_ref_.norm();
              if (bn > T(1e-9)) B_world_ref_ /= bn;
            }
            if (mag_ref_accum_count_ > 100) {
              mag_ref_accum_.setZero();
              mag_ref_accum_count_ = 0;
            }
          }
        }
      }
    }

    // 3) crude motion estimate for exit thresholds
    // Convert specific force to world accel estimate (very crude during warm-up):
    // a ≈ R_bw * f + g_world
    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 a_world = R_bw() * acc + g_world;
    warmup_vel_est_ += a_world * dt;
    motion_distance_ += warmup_vel_est_.norm() * dt;
    motion_time_ += dt;

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

  // ===== Heel update (same retarget logic as OU) =====
  void update_wind_heel(T heel_rad) {
    const T old = wind_heel_rad_;
    if (heel_rad == old) return;
    retarget_bodyprime_frame_(heel_rad - old);
    wind_heel_rad_ = heel_rad;
    update_unheel_trig_();
  }

  // ===== Initialization (FRAME-CORRECT for OU conventions) =====
  void initialize_from_acc_mag(const Vec3& acc_body, const Vec3& mag_body) {
    const Vec3 acc = deheel_vector_(acc_body); // BODY'
    const Vec3 mag = deheel_vector_(mag_body); // BODY'

    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    // world-down expressed in BODY' (NED, +Z down)
    Vec3 down_b = (-acc / an);

    // East = down × mag, North = east × down  (all in BODY')
    Vec3 east_b  = down_b.cross(mag);
    const T en = east_b.norm();
    if (!(en > T(1e-8))) return;
    east_b /= en;

    Vec3 north_b = east_b.cross(down_b);
    const T nn = north_b.norm();
    if (!(nn > T(1e-8))) return;
    north_b /= nn;

    // R_wb maps WORLD->BODY': columns are world basis vectors expressed in BODY'
    // world basis: X=north, Y=east, Z=down
    Mat3 R_wb;
    R_wb.col(0) = north_b;
    R_wb.col(1) = east_b;
    R_wb.col(2) = down_b;

    qref_ = Eigen::Quaternion<T>(R_wb);
    qref_.normalize();

    if constexpr (with_mag) {
      B_world_ref_ = R_bw() * mag; // BODY'->WORLD
    }

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true, /*force_reset=*/true);
  }

  void initialize_from_acc(const Vec3& acc_body) {
    const Vec3 acc = deheel_vector_(acc_body); // BODY'
    const T an = acc.norm();
    if (!(an > T(1e-8))) return;

    // down_b = world-down expressed in BODY'
    const Vec3 down_b = (-acc / an);

    // Build any orthonormal completion (yaw arbitrary):
    Vec3 north_b;
    // pick something not parallel to down
    if (std::abs(down_b.x()) < T(0.9)) north_b = Vec3::UnitX().cross(down_b);
    else                               north_b = Vec3::UnitY().cross(down_b);
    const T nn = north_b.norm();
    if (!(nn > T(1e-8))) {
      qref_.setIdentity();
    } else {
      north_b /= nn;
      Vec3 east_b = down_b.cross(north_b);
      const T en = east_b.norm();
      if (en > T(1e-8)) east_b /= en;

      Mat3 R_wb;
      R_wb.col(0) = north_b;
      R_wb.col(1) = east_b;
      R_wb.col(2) = down_b;

      qref_ = Eigen::Quaternion<T>(R_wb);
      qref_.normalize();
    }

    x_.template segment<3>(OFF_DTH).setZero();
    set_warmup_mode(true, /*force_reset=*/true);
  }

  // ===== Wave tuning (broadband) =====
  void set_broadband_params(T f0_hz, T Hs_m, T zeta_mid = T(0.08), T horiz_scale = T(0.35)) {
    const T w0 = T(2)*T(M_PI)*std::max(T(1e-4), f0_hz);

    // log-spaced omegas around w0 (roughly 0.6..1.7x)
    for (int k=0;k<KMODES;++k) {
      const T u = (KMODES==1) ? T(0) : (T(k) / T(KMODES-1)); // 0..1
      const T lo = std::log(T(0.6));
      const T hi = std::log(T(1.7));
      omega_[k] = std::exp(lo + (hi-lo)*u) * w0;

      const T scale = (k==0 || k==KMODES-1) ? T(1.25) : T(1.0);
      zeta_[k] = std::max(T(0.01), zeta_mid * scale);
    }

    // smooth weights
    T wsum = T(0);
    for (int k=0;k<KMODES;++k) {
      const T c = (KMODES==1) ? T(0) : (T(k) - T(0.5)*(KMODES-1));
      const T wk = std::exp(-(c*c) / (T(0.7)*T(0.7)));
      weights_[k] = wk;
      wsum += wk;
    }
    for (int k=0;k<KMODES;++k) weights_[k] /= std::max(T(1e-9), wsum);

    // Hs -> sigma of surface elevation ~ Hs/4
    const T sigma_total = std::max(T(0), Hs_m) / T(4);
    const T var_total = sigma_total * sigma_total;

    // Choose q_k so that stationary var(p_k) ≈ var_k (for each axis independently)
    // For SDE: p'' + 2ζω p' + ω^2 p = ξ, with ξ white accel input PSD q.
    // Known stationary var(p) = q / (4 ζ ω^3). => q = 4 ζ ω^3 var(p)
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

  // ===== Time update (propagation) =====
  void time_update(const Vec3& gyr_body, T Ts) {
    last_dt_ = Ts;

    // De-heel gyro into B'
    const Vec3 gyr = deheel_vector_(gyr_body);

    // Bias-corrected ω in B'
    Vec3 bg = Vec3::Zero();
    if constexpr (with_gyro_bias) bg = x_.template segment<3>(OFF_BG);
    last_gyr_bias_corrected_ = gyr - bg;

    // Lever-arm angular acceleration estimate α^{B'}
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

    // Attitude propagation (right-multiply δq(ω dt))
    qref_ = qref_ * quat_from_delta_theta<T>((omega_b * Ts).eval());
    qref_.normalize();

    // Build base transition Jacobian (small-angle approx; consistent with your OU)
    // Fθθ ≈ I - [ω]x dt
    Mat3 Fth = Mat3::Identity() - skew3<T>(omega_b) * Ts;

    F_AA_.setIdentity();
    F_AA_.template block<3,3>(0,0) = Fth;
    if constexpr (with_gyro_bias) {
      F_AA_.template block<3,3>(0,3) = -Mat3::Identity() * Ts;
    }

    // Discrete Q for base: Qd ≈ Q * dt
    Q_AA_.setZero();
    Q_AA_.template block<3,3>(0,0) = Qg_ * Ts;
    if constexpr (with_gyro_bias) {
      Q_AA_.template block<3,3>(3,3) = Qbg_ * Ts;
    }

    // Propagate base covariance (in-place block reference)
    {
      auto Paa = P_.template block<BASE_N,BASE_N>(0,0);
      tmp_AA_.noalias() = F_AA_ * Paa;
      Paa.noalias() = tmp_AA_ * F_AA_.transpose();
      Paa.noalias() += Q_AA_;
    }

    // Wave block propagation (only if enabled)
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        // per-axis Phi2 and Qd2
        for (int ax=0; ax<3; ++ax) {
          discretize_osc_axis_(Ts, omega_[k], zeta_[k], q_axis_[k](ax), Phi2_[ax], Qd2_[ax]);
        }

        // mean propagation for this mode
        Vec3 p = x_.template segment<3>(OFF_Pk(k));
        Vec3 v = x_.template segment<3>(OFF_Vk(k));
        for (int ax=0; ax<3; ++ax) {
          Eigen::Matrix<T,2,1> xv;
          xv << p(ax), v(ax);
          xv = Phi2_[ax] * xv;
          p(ax) = xv(0);
          v(ax) = xv(1);
        }
        x_.template segment<3>(OFF_Pk(k)) = p;
        x_.template segment<3>(OFF_Vk(k)) = v;

        // assemble Phi6 and Qd6
        Phi6_.setZero();
        Qd6_.setZero();
        for (int ax=0; ax<3; ++ax) {
          Phi6_.template block<2,2>(2*ax,2*ax) = Phi2_[ax];
          Qd6_ .template block<2,2>(2*ax,2*ax) = Qd2_[ax];
        }

        const int offk = OFF_Pk(k);

        // P_kk (6x6) in-place
        {
          auto Pkk = P_.template block<6,6>(offk, offk);
          tmp6_.noalias() = Phi6_ * Pkk;
          Pkk.noalias() = tmp6_ * Phi6_.transpose();
          Pkk.noalias() += Qd6_;
        }

        // Cross with base: P_Ak = F_AA * P_Ak_old * Phi6^T
        {
          auto P_Ak = P_.template block<BASE_N,6>(0, offk);
          tmp_A6_.noalias() = F_AA_ * P_Ak;
          P_Ak.noalias() = tmp_A6_ * Phi6_.transpose();
          P_.template block<6,BASE_N>(offk, 0) = P_Ak.transpose();
        }

        // Cross with BA: P_BAk = P_BAk_old * Phi6^T  (F_BA = I)
        if constexpr (with_accel_bias) {
          auto P_BAk = P_.template block<3,6>(OFF_BA, offk);
          P_BAk.noalias() = P_BAk * Phi6_.transpose();
          P_.template block<6,3>(offk, OFF_BA) = P_BAk.transpose();
        }
      }
    }

    // Accel bias RW (and cross with base; cross with wave handled above)
    if constexpr (with_accel_bias) {
      auto Pba = P_.template block<3,3>(OFF_BA, OFF_BA);
      if (acc_bias_updates_enabled_) Pba.noalias() += Q_bacc_ * Ts;
      for (int i=0;i<3;++i) Pba(i,i) = std::max(Pba(i,i), T(1e-12));
      P_.template block<3,3>(OFF_BA, OFF_BA) = Pba;

      // Cross BA with base: P_Aba = F_AA * P_Aba
      auto P_Aba = P_.template block<BASE_N,3>(0, OFF_BA);
      tmp_A3_.noalias() = F_AA_ * P_Aba;
      P_Aba.noalias() = tmp_A3_;
      P_.template block<3,BASE_N>(OFF_BA, 0) = P_Aba.transpose();
    }

    // Hygiene
    regularize_covariance_();
    symmetrize_P_();
  }

  // ===== Measurement update: accelerometer (Joseph, stack-light) =====
  void measurement_update_acc_only(const Vec3& acc_meas_body, T tempC = tempC_ref_) {
    last_acc_ = MeasDiag3{};
    last_acc_.accepted = false;

    const bool use_ba = (with_accel_bias && acc_bias_updates_enabled_);

    const Vec3 acc_meas = deheel_vector_(acc_meas_body);

    // Lever-arm term in BODY'
    Vec3 lever = Vec3::Zero();
    if (use_imu_lever_arm_) {
      const Vec3 r_imu_bprime = deheel_vector_(r_imu_wrt_cog_body_phys_);
      lever.noalias() += alpha_b_.cross(r_imu_bprime)
                      +  last_gyr_bias_corrected_.cross(last_gyr_bias_corrected_.cross(r_imu_bprime));
    }

    // accel bias term (temp-dependent)
    Vec3 ba_term = Vec3::Zero();
    if constexpr (with_accel_bias) {
      const Vec3 ba0 = x_.template segment<3>(OFF_BA);
      ba_term = ba0 + k_a_ * (tempC - tempC_ref_);
    }

    const Vec3 g_world(0,0,+gravity_magnitude_);
    const Vec3 aw = wave_world_accel_(); // world-frame wave accel

    // Prediction
    const Vec3 f_pred = R_wb() * (aw - g_world) + lever + ba_term;
    const Vec3 r = acc_meas - f_pred;

    last_acc_.r = r;

    // Jacobians
    const Vec3 f_cog_b = R_wb() * (aw - g_world);
    const Mat3 J_att = -skew3<T>(f_cog_b);

    // S = C P C^T + R
    Mat3& S = S_scratch_;
    S = Racc_;

    // attitude contribution
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH, OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    // BA contribution (J_ba = I)
    if constexpr (with_accel_bias) {
      const Mat3 Pba  = P_.template block<3,3>(OFF_BA, OFF_BA);
      const Mat3 Ptba = P_.template block<3,3>(OFF_DTH, OFF_BA);
      // always marginalize BA uncertainty into S (even if frozen) for consistency
      S.noalias() += Pba;
      S.noalias() += J_att * Ptba;
      S.noalias() += Ptba.transpose() * J_att.transpose();
    }

    // Wave contributions (if enabled)
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];

        // aw_k = -ω^2 p_k - 2ζω v_k
        // f = R_wb*(aw - g) -> Jp = R_wb*(-ω^2 I), Jv = R_wb*(-2ζω I)
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
          const Mat3 Pbap = P_.template block<3,3>(OFF_BA, op);
          const Mat3 Pbav = P_.template block<3,3>(OFF_BA, ov);
          S.noalias() += Pbap * Jp.transpose();
          S.noalias() += Jp * Pbap.transpose();
          S.noalias() += Pbav * Jv.transpose();
          S.noalias() += Jv * Pbav.transpose();
        }

        if constexpr (with_gyro_bias) {
          // NOTE: you can add lever-arm J_bg coupling here later if you need it.
          // Your OU version had a full J_bg path only when lever-arm enabled; keep it out unless needed.
        }
      }
    } else {
      // wave disabled: nothing to add (consistent with freezing wave block and keeping its cov tiny)
    }

    // Robust LDLT
    Eigen::LDLT<Mat3> ldlt;
    if (!safe_ldlt3_(S, ldlt, Racc_.norm())) {
      last_acc_.S = S;
      last_acc_.accepted = false;
      return;
    }
    last_acc_.S = S;
    last_acc_.nis = nis3_from_ldlt_(ldlt, r);

    // PCt = P C^T  (NX x 3)
    MatX3& PCt = PCt_scratch_;
    PCt.setZero();

    // from attitude
    PCt.noalias() += P_.template block<NX,3>(0, OFF_DTH) * J_att.transpose();

    // from BA (J_ba = I)
    if constexpr (with_accel_bias) {
      if (use_ba) {
        PCt.noalias() += P_.template block<NX,3>(0, OFF_BA);
      } else {
        // BA frozen: do NOT allow state update to change BA
        // (But we already accounted for BA uncertainty in S above.)
        // Therefore, we do not add P(:,BA) here.
      }
    }

    // from wave
    if (wave_block_enabled_) {
      for (int k=0;k<KMODES;++k) {
        const T om = omega_[k];
        const T ze = zeta_[k];
        const Mat3 Jp = R_wb() * (-(om*om) * Mat3::Identity());
        const Mat3 Jv = R_wb() * (-(T(2)*ze*om) * Mat3::Identity());

        PCt.noalias() += P_.template block<NX,3>(0, OFF_Pk(k)) * Jp.transpose();
        PCt.noalias() += P_.template block<NX,3>(0, OFF_Vk(k)) * Jv.transpose();
      }
    } else {
      freeze_wave_rows_(PCt);
    }

    if constexpr (with_accel_bias) {
      if (!use_ba) freeze_acc_bias_rows_(PCt);
    }

    // K = PCt * S^-1
    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());

    if (!wave_block_enabled_) freeze_wave_rows_(K);
    if constexpr (with_accel_bias) {
      if (!use_ba) freeze_acc_bias_rows_(K);
    }

    // State update
    x_.noalias() += K * r;

    // Joseph covariance update (stack-light)
    joseph_update3_(K, S, PCt);

    // Apply quaternion correction (may be nudged via cross-cov)
    applyQuaternionCorrectionFromErrorState_();

    last_acc_.accepted = true;
  }

  // ===== Measurement update: magnetometer (Joseph, stack-light) =====
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
    const Mat3 Ptt = P_.template block<3,3>(OFF_DTH, OFF_DTH);
    S.noalias() += J_att * Ptt * J_att.transpose();

    Eigen::LDLT<Mat3> ldlt;
    if (!safe_ldlt3_(S, ldlt, Rmag_.norm())) {
      last_mag_.S = S;
      last_mag_.accepted = false;
      return;
    }

    last_mag_.S = S;
    last_mag_.nis = nis3_from_ldlt_(ldlt, r);

    MatX3& PCt = PCt_scratch_;
    PCt.setZero();
    PCt.noalias() += P_.template block<NX,3>(0, OFF_DTH) * J_att.transpose();

    if (!wave_block_enabled_) freeze_wave_rows_(PCt);
    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) freeze_acc_bias_rows_(PCt);
    }

    MatX3& K = K_scratch_;
    K.noalias() = PCt * ldlt.solve(Mat3::Identity());

    if (!wave_block_enabled_) freeze_wave_rows_(K);
    if constexpr (with_accel_bias) {
      if (!acc_bias_updates_enabled_) freeze_acc_bias_rows_(K);
    }

    x_.noalias() += K * r;
    joseph_update3_(K, S, PCt);
    applyQuaternionCorrectionFromErrorState_();

    last_mag_.accepted = true;
  }

private:
  // ===== Constants / tuning =====
  const T gravity_magnitude_ = T(9.80665);

  // ===== MEKF internals =====
  Eigen::Quaternion<T> qref_;                 // WORLD->BODY'
  Vec3 B_world_ref_ = Vec3::UnitX();          // if with_mag

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
  T omega_[KMODES]{};
  T zeta_[KMODES]{};
  T weights_[KMODES]{};
  Vec3 q_axis_[KMODES]{};

  bool wave_block_enabled_ = true;
  bool acc_bias_updates_enabled_ = true;

  // Warm-up
  bool warmup_mode_ = true;
  Vec3 gyro_bias_accum_ = Vec3::Zero();
  int  gyro_bias_accum_count_ = 0;
  Vec3 mag_ref_accum_ = Vec3::Zero();
  int  mag_ref_accum_count_ = 0;

  T   motion_distance_ = T(0);
  T   motion_time_ = T(0);
  T   exit_min_distance_ = T(10.0);
  T   exit_min_time_ = T(5.0);
  Vec3 warmup_vel_est_ = Vec3::Zero();

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

  // ===== Scratch (memory optimizations; no NX×NX temporaries in updates) =====
  Eigen::Matrix<T,BASE_N,BASE_N> F_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> Q_AA_;
  Eigen::Matrix<T,BASE_N,BASE_N> tmp_AA_;

  Eigen::Matrix<T,BASE_N,6> tmp_A6_;
  Eigen::Matrix<T,BASE_N,3> tmp_A3_;

  Eigen::Matrix<T,6,6> Phi6_;
  Eigen::Matrix<T,6,6> Qd6_;
  Eigen::Matrix<T,6,6> tmp6_;
  Eigen::Matrix<T,2,2> Phi2_[3];
  Eigen::Matrix<T,2,2> Qd2_[3];

  Mat3  S_scratch_;
  MatX3 PCt_scratch_;
  MatX3 K_scratch_;

private:
  // ===== Rotations =====
  Mat3 R_wb() const { return qref_.toRotationMatrix(); }             // world->body'
  Mat3 R_bw() const { return qref_.toRotationMatrix().transpose(); } // body'->world

  // ===== Numerical helpers =====
  void regularize_covariance_() {
    const T min_diag = T(1e-12);
    for (int i = 0; i < NX; ++i) {
      if (!std::isfinite(P_(i,i)) || P_(i,i) < min_diag) P_(i,i) = min_diag;
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

  // LDLT safety for 3×3
  bool safe_ldlt3_(Mat3& S, Eigen::LDLT<Mat3>& ldlt, T noise_scale) const {
    S = T(0.5) * (S + S.transpose());
    ldlt.compute(S);
    if (ldlt.info() == Eigen::Success) return true;

    const T bump = std::max(std::numeric_limits<T>::epsilon(), T(1e-6) * (noise_scale + T(1)));
    S.diagonal().array() += bump;

    ldlt.compute(S);
    return (ldlt.info() == Eigen::Success);
  }

  T nis3_from_ldlt_(const Eigen::LDLT<Mat3>& ldlt, const Vec3& r) const {
    Vec3 x = ldlt.solve(r);
    if (!x.allFinite()) return std::numeric_limits<T>::quiet_NaN();
    const T v = r.dot(x);
    return std::isfinite(v) ? v : std::numeric_limits<T>::quiet_NaN();
  }

  // Stack-light Joseph covariance update using only K, S, PCt = P*C^T
  void joseph_update3_(const MatX3& K, const Mat3& S, const MatX3& PCt) {
    // P <- P - K C P - (K C P)^T + K S K^T
    // with CP ≈ (PCt)^T (since P symmetric), so KCP(i,j) ≈ sum_l K(i,l) * PCt(j,l)
    for (int i = 0; i < NX; ++i) {
      for (int j = i; j < NX; ++j) {
        T KCP_ij = T(0);
        T KCP_ji = T(0);
        for (int l = 0; l < 3; ++l) {
          const T Ki = K(i,l);
          const T Kj = K(j,l);
          const T Pj = PCt(j,l);
          const T Pi = PCt(i,l);
          KCP_ij += Ki * Pj;
          if (j != i) KCP_ji += Kj * Pi;
        }
        if (j == i) KCP_ji = KCP_ij;

        T KSK_ij = T(0);
        for (int a = 0; a < 3; ++a) {
          const T Kia = K(i,a);
          for (int b = 0; b < 3; ++b) {
            const T Kjb = K(j,b);
            KSK_ij += Kia * S(a,b) * Kjb;
          }
        }

        const T delta = -(KCP_ij + KCP_ji) + KSK_ij;
        P_(i,j) += delta;
        if (j != i) P_(j,i) = P_(i,j);
      }
    }
    symmetrize_P_();
    regularize_covariance_();
  }

  void applyQuaternionCorrectionFromErrorState_() {
    const Vec3 dth = x_.template segment<3>(OFF_DTH);
    const Eigen::Quaternion<T> corr = quat_from_delta_theta<T>(dth);
    qref_ = qref_ * corr;
    qref_.normalize();
    x_.template segment<3>(OFF_DTH).setZero();
  }

  // ===== Freeze helpers (like OU) =====
  void freeze_wave_rows_(MatX3& M) const {
    M.template block<WAVE_N,3>(OFF_WAVE, 0).setZero();
  }
  void freeze_acc_bias_rows_(MatX3& M) const {
    if constexpr (with_accel_bias) {
      M.template block<3,3>(OFF_BA, 0).setZero();
    }
  }

  // ===== Wave accel in world (sum over modes) =====
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
    const T om = std::max(T(1e-7), w);
    const T ze = std::max(T(0), z);
    const T eps = T(1e-7);

    // State: [p; v], dynamics: p' = v, v' = -ω^2 p - 2ζω v
    // This Phi is exact for each damping regime (with guarded fallbacks).

    // Near-critical handling
    if (std::abs(ze - T(1)) < T(1e-3)) {
      const T a = om;
      const T e = std::exp(-a * t);
      // Critically damped closed-form:
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * (t);
      Phi(1,0) = e * (-a*a*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }

    if (ze < T(1)) {
      const T wd = om * std::sqrt(std::max(T(0), T(1) - ze*ze));
      if (wd < eps) {
        // fallback to critical approx
        const T a = om;
        const T e = std::exp(-a * t);
        Phi(0,0) = e * (T(1) + a*t);
        Phi(0,1) = e * t;
        Phi(1,0) = e * (-a*a*t);
        Phi(1,1) = e * (T(1) - a*t);
        return;
      }
      const T a  = ze * om;
      const T e  = std::exp(-a * t);
      const T c  = std::cos(wd * t);
      const T s  = std::sin(wd * t);

      const T inv_wd = T(1) / wd;
      const T a_over_wd = a * inv_wd;

      Phi(0,0) = e * (c + a_over_wd * s);
      Phi(0,1) = e * (inv_wd * s);
      Phi(1,0) = e * (-(om*om) * inv_wd * s);
      Phi(1,1) = e * (c - a_over_wd * s);
      return;
    }

    // Overdamped
    const T s = std::sqrt(std::max(T(0), ze*ze - T(1)));
    const T r1 = -om * (ze - s);
    const T r2 = -om * (ze + s);

    const T e1 = std::exp(r1 * t);
    const T e2 = std::exp(r2 * t);

    const T denom = (r2 - r1);
    if (std::abs(denom) < eps) {
      // fallback
      const T a = om * ze;
      const T e = std::exp(-a * t);
      Phi(0,0) = e * (T(1) + a*t);
      Phi(0,1) = e * t;
      Phi(1,0) = e * (-(om*om)*t);
      Phi(1,1) = e * (T(1) - a*t);
      return;
    }
    const T invd  = T(1) / denom;

    Phi(0,0) = (r2*e1 - r1*e2) * invd;
    Phi(0,1) = (e2 - e1) * invd;
    Phi(1,0) = (r1*r2) * (e1 - e2) * invd;
    Phi(1,1) = (r2*e2 - r1*e1) * invd;
  }

  // Qd = ∫_0^dt Φ(t) B q B^T Φ(t)^T dt, where input enters v equation: B=[0;1]
  // Use Simpson (stable and accurate; same spirit as your OU Simpson integrals).
  static inline void discretize_osc_axis_(T dt, T w, T z, T q,
                                         Eigen::Matrix<T,2,2>& Phi,
                                         Eigen::Matrix<T,2,2>& Qd)
  {
    phi_osc_2x2_(dt, w, z, Phi);

    auto M_of_t = [&](T t)->Eigen::Matrix<T,2,2> {
      Eigen::Matrix<T,2,2> Pt;
      phi_osc_2x2_(t, w, z, Pt);
      // Pt * B = column 1 of Pt (since B=[0;1])
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
    const auto G0 = M_of_t(T(0));
    const auto G1 = M_of_t(T(0.5)*h);
    const auto G2 = M_of_t(h);

    Qd = (h / T(6)) * (G0 + T(4)*G1 + G2);
    Qd = T(0.5) * (Qd + Qd.transpose());

    // PSD hygiene (2x2)
    if (!std::isfinite(Qd(0,0))) Qd(0,0) = T(0);
    if (!std::isfinite(Qd(1,1))) Qd(1,1) = T(0);
    Qd(0,0) = std::max(Qd(0,0), T(0));
    Qd(1,1) = std::max(Qd(1,1), T(0));
    const T max_off = std::sqrt(std::max(T(0), Qd(0,0) * Qd(1,1)));
    if (!std::isfinite(Qd(0,1))) Qd(0,1) = T(0);
    Qd(0,1) = std::max(-max_off, std::min(Qd(0,1), max_off));
    Qd(1,0) = Qd(0,1);
  }

  // ===== Heel helpers (MATCH OU) =====
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

    // Rotate the attitude reference: qref is WORLD->B'
    qref_ = qR * qref_;
    qref_.normalize();

    // Rotate vector-states stored in B' (δθ, b_g, b_a)
    x_.template segment<3>(OFF_DTH) = R * x_.template segment<3>(OFF_DTH);
    if constexpr (with_gyro_bias)  x_.template segment<3>(OFF_BG) = R * x_.template segment<3>(OFF_BG);
    if constexpr (with_accel_bias) x_.template segment<3>(OFF_BA) = R * x_.template segment<3>(OFF_BA);

    // Rotate cached kinematics
    last_gyr_bias_corrected_ = R * last_gyr_bias_corrected_;
    prev_omega_b_            = R * prev_omega_b_;
    alpha_b_                 = R * alpha_b_;

    // Similarity-transform covariance for rotated blocks
    MatX Tm = MatX::Identity();
    Tm.template block<3,3>(OFF_DTH, OFF_DTH) = R;
    if constexpr (with_gyro_bias)  Tm.template block<3,3>(OFF_BG, OFF_BG) = R;
    if constexpr (with_accel_bias) Tm.template block<3,3>(OFF_BA, OFF_BA) = R;

    P_ = Tm * P_ * Tm.transpose();
    regularize_covariance_();
    symmetrize_P_();
  }
};
