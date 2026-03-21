#pragma once

/*

  Wrapper around:
    - Mahony_AHRS
    - AdaptiveVerticalPII

  Purpose

  Uses Mahony AHRS to estimate tilt, rotates body-frame accelerometer specific
  force into the Mahony world frame, extracts vertical inertial acceleration
  (Z-up), and feeds that into AdaptiveVerticalPII.

  IMPORTANT FRAME NOTE

  The supplied Mahony implementation is NOT NED.
  It is effectively a Z-up auxiliary/world frame.

  Evidence from the code:
    - identity quaternion expects accelerometer ~ (0, 0, +g)
    - estimated gravity direction at identity is +Z
    - therefore "world up" is +Z

  So this wrapper computes:
      a_world_up = (R_bw * a_body).z - g

  where:
      a_body     = raw accelerometer specific force in body frame [m/s^2]
      R_bw       = body -> world(Z-up) rotation from Mahony quaternion
      a_world_up = inertial/world vertical acceleration, UP positive [m/s^2]

  That is the exact quantity expected by the vertical observer.

  IMPORTANT EULER NOTE

  Mahony internally stores quaternion as world -> body.
  For user-facing roll/pitch/yaw diagnostics we must first conjugate it to
  body -> world, then compute Euler angles from that body -> world quaternion.

  This fixes the sign errors you were seeing from reading Euler directly from
  the raw world -> body quaternion.

  Usage

      using Heave = marine_obs::AdaptiveVerticalPIIMahony<float, true>;
      Heave::Config cfg;
      Heave filt(cfg);

      // each IMU sample
      float z = filt.updateIMU(gx, gy, gz, ax, ay, az, dt);

      // preferred adaptation source: displacement frequency tracker
      filt.updateAdaptationFromDisplacementFrequency(f_disp_hz, dt_track, confidence);

  Inputs

  gx,gy,gz : gyro [rad/s]
  ax,ay,az : accelerometer specific force [m/s^2]
  dt       : sample period [s]

  Notes
  - Mahony state is templated on T and defaults to float.
  - The adaptive vertical observer remains templated on T.
  - This wrapper does not remap axes. Feed data in the SAME axis convention
    that your Mahony implementation expects.
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "Mahony_AHRS.h"
#include "AdaptiveVerticalPII.h"

namespace marine_obs {

template<typename T = float,
         bool WithBias = true,
         typename AccelFreqTrackerT = WaveFrequencyTracker<T>>
class AdaptiveVerticalPIIMahony {
    static_assert(std::is_floating_point<T>::value,
                  "AdaptiveVerticalPIIMahony<T>: T must be a floating-point type.");

public:
    using Core = AdaptiveVerticalPII<T, WithBias, AccelFreqTrackerT>;
    using CoreConfig = typename Core::Config;
    using CoreSnapshot = typename Core::Snapshot;

    struct QuaternionT {
        T w = T(1);
        T x = T(0);
        T y = T(0);
        T z = T(0);
    };

    struct Config {
        // Underlying adaptive vertical observer
        CoreConfig core{};

        // Mahony gains
        T mahony_twoKp = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
        T mahony_twoKi = static_cast<T>(Mahony_AHRS<T>::twoKiDef);

        // Gravity magnitude used to convert specific force -> inertial accel
        T gravity_mps2 = static_cast<T>(9.80665);

        // If true, updateIMU() will call magless Mahony update only.
        // If you want magnetometer support, call updateIMUMag().
        bool use_mag = false;

        // Optional slow sea-state scheduling of Mahony gains.
        bool adapt_mahony_gains = false;

        T mahony_twoKp_calm  = static_cast<T>(0.90);
        T mahony_twoKp_rough = static_cast<T>(0.18);

        T mahony_twoKi_calm  = static_cast<T>(0.020);
        T mahony_twoKi_rough = static_cast<T>(0.000);

        // Reference values for simple sea-state scheduling.
        T mahony_sigma_ref    = static_cast<T>(0.18);
        T mahony_freq_ref_hz  = static_cast<T>(0.12);
        T mahony_norm_err_ref = static_cast<T>(0.08);

        // Gain smoothing time constant.
        T mahony_gain_smooth_tau_s = static_cast<T>(2.0);

        // Minimum accelerometer trust multiplier.
        T mahony_acc_trust_min = static_cast<T>(0.05);
    };

    struct Snapshot {
        CoreSnapshot core{};

        QuaternionT q_world_to_body{};
        QuaternionT q_body_to_world{};

        T roll_deg  = T(0);
        T pitch_deg = T(0);
        T yaw_deg   = T(0);

        T ax_world = T(0);
        T ay_world = T(0);
        T az_world = T(0);

        T vertical_world_accel_up = T(0);

        T mahony_twoKp = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
        T mahony_twoKi = static_cast<T>(Mahony_AHRS<T>::twoKiDef);
        T gravity_mps2 = static_cast<T>(9.80665);

        T mahony_acc_trust = T(1);
        T mahony_sigma_sched = T(0);
        T mahony_freq_sched_hz = T(0);
    };

public:
    explicit AdaptiveVerticalPIIMahony(const Config& cfg = Config())
        : core_(cfg.core) {
        configure(cfg);
        reset();
    }

    void configure(const Config& cfg) {
        cfg_ = sanitizeConfig_(cfg);

        core_.configure(cfg_.core);

        mahony_AHRS_init(&mahony_, cfg_.mahony_twoKp, cfg_.mahony_twoKi);

        gravity_mps2_ = cfg_.gravity_mps2;
        last_roll_deg_ = last_pitch_deg_ = last_yaw_deg_ = T(0);
        ax_world_ = ay_world_ = az_world_ = T(0);
        vertical_world_accel_up_ = T(0);

        last_acc_trust_ = T(1);
        last_sigma_sched_ = T(0);
        last_freq_sched_hz_ = T(0);
    }

    void reset(T p0 = T(0),
               T v0 = T(0),
               T a_f0 = T(0),
               T S0 = T(0),
               T d0 = T(0),
               T b0 = T(0))
    {
        // Reset Mahony state
        mahony_ = Mahony_AHRS<T>{};
        mahony_AHRS_init(&mahony_, cfg_.mahony_twoKp, cfg_.mahony_twoKi);

        last_roll_deg_ = T(0);
        last_pitch_deg_ = T(0);
        last_yaw_deg_ = T(0);

        ax_world_ = ay_world_ = az_world_ = T(0);
        vertical_world_accel_up_ = T(0);

        last_acc_trust_ = T(1);
        last_sigma_sched_ = T(0);
        last_freq_sched_hz_ = T(0);

        core_.reset(p0, v0, a_f0, S0, d0, b0);
    }

    void setMahonyGains(T twoKp, T twoKi) {
        if (std::isfinite(twoKp)) mahony_.twoKp = twoKp;
        if (std::isfinite(twoKi)) mahony_.twoKi = twoKi;
        cfg_.mahony_twoKp = mahony_.twoKp;
        cfg_.mahony_twoKi = mahony_.twoKi;
    }

    // IMU-only update (no magnetometer)
    // Returns displacement estimate after update.
    T updateIMU(T gx_rad_s, T gy_rad_s, T gz_rad_s,
                T ax_mps2, T ay_mps2, T az_mps2,
                T dt_s)
    {
        if (!(std::isfinite(dt_s) && dt_s > T(0))) {
            return core_.displacement();
        }

        // Mahony API requires Euler outputs, but we intentionally ignore the raw
        // Euler it computes from q_world->body. We recompute user-facing Euler
        // from q_body->world below.
        T pitch_unused = T(0);
        T roll_unused  = T(0);
        T yaw_unused   = T(0);

        mahony_AHRS_update(&mahony_,
                           gx_rad_s, gy_rad_s, gz_rad_s,
                           ax_mps2, ay_mps2, az_mps2,
                           &pitch_unused, &roll_unused, &yaw_unused,
                           dt_s);

        computeWorldAccelAndVertical_(ax_mps2, ay_mps2, az_mps2);
        updateEulerFromBodyToWorld_();

        const T z = core_.update(vertical_world_accel_up_, dt_s);

        if (cfg_.adapt_mahony_gains) {
            updateMahonyGains_(ax_mps2, ay_mps2, az_mps2, dt_s);
        }

        return z;
    }

    // IMU + magnetometer update
    // Returns displacement estimate after update.
    T updateIMUMag(T gx_rad_s, T gy_rad_s, T gz_rad_s,
                   T ax_mps2, T ay_mps2, T az_mps2,
                   T mx, T my, T mz,
                   T dt_s)
    {
        if (!(std::isfinite(dt_s) && dt_s > T(0))) {
            return core_.displacement();
        }

        T pitch_unused = T(0);
        T roll_unused  = T(0);
        T yaw_unused   = T(0);

        mahony_AHRS_update_mag(&mahony_,
                               gx_rad_s, gy_rad_s, gz_rad_s,
                               ax_mps2, ay_mps2, az_mps2,
                               mx, my, mz,
                               &pitch_unused, &roll_unused, &yaw_unused,
                               dt_s);

        computeWorldAccelAndVertical_(ax_mps2, ay_mps2, az_mps2);
        updateEulerFromBodyToWorld_();

        const T z = core_.update(vertical_world_accel_up_, dt_s);

        if (cfg_.adapt_mahony_gains) {
            updateMahonyGains_(ax_mps2, ay_mps2, az_mps2, dt_s);
        }

        return z;
    }

    // Preferred adaptation hook:
    // use externally estimated DISPLACEMENT frequency.
    void updateAdaptationFromDisplacementFrequency(T f_disp_hz,
                                                   T dt_est,
                                                   T confidence = std::numeric_limits<T>::quiet_NaN())
    {
        core_.updateAdaptationFromDisplacementFrequency(f_disp_hz, dt_est, confidence);
    }

    // Full external adaptation hook
    void updateAdaptationExternal(T f_disp_hz,
                                  T sigma_a,
                                  T dt_est,
                                  T confidence = std::numeric_limits<T>::quiet_NaN())
    {
        core_.updateAdaptationExternal(f_disp_hz, sigma_a, dt_est, confidence);
    }

    // Optional fallback: use internal acceleration-frequency tracker
    void updateAdaptationFromAccelFrequencyProxy(T dt_est) {
        core_.updateAdaptationFromAccelFrequencyProxy(dt_est);
    }

    void setAutoScheduleFromAccelFreq(bool on) {
        core_.setAutoScheduleFromAccelFreq(on);
    }

    void setAutoSchedulePeriod(T period_s) {
        core_.setAutoSchedulePeriod(period_s);
    }

    // Accessors
    Core& core() { return core_; }
    const Core& core() const { return core_; }

    Mahony_AHRS<T>& mahonyState() { return mahony_; }
    const Mahony_AHRS<T>& mahonyState() const { return mahony_; }

    QuaternionT quaternionWorldToBody() const {
        QuaternionT q;
        q.w = static_cast<T>(mahony_.q0);
        q.x = static_cast<T>(mahony_.q1);
        q.y = static_cast<T>(mahony_.q2);
        q.z = static_cast<T>(mahony_.q3);
        return q;
    }

    QuaternionT quaternionBodyToWorld() const {
        QuaternionT q = quaternionWorldToBody();
        q.x = -q.x;
        q.y = -q.y;
        q.z = -q.z;
        return q;
    }

    T rollDeg() const  { return last_roll_deg_;  }
    T pitchDeg() const { return last_pitch_deg_; }
    T yawDeg() const   { return last_yaw_deg_;   }

    T worldAccelX() const { return ax_world_; }
    T worldAccelY() const { return ay_world_; }
    T worldAccelZUpSpecificForce() const { return az_world_; }

    // This is the quantity fed into AdaptiveVerticalPII:
    // vertical inertial/world acceleration, UP positive
    T verticalWorldAccelUp() const { return vertical_world_accel_up_; }

    T displacement() const { return core_.displacement(); }
    T velocity() const { return core_.velocity(); }
    T accelFiltered() const { return core_.accelFiltered(); }
    T accelSigma() const { return core_.accelSigma(); }
    T accelFrequencyHz() const { return core_.accelFrequencyHz(); }

    Snapshot snapshot() const {
        Snapshot s;
        s.core = core_.snapshot();
        s.q_world_to_body = quaternionWorldToBody();
        s.q_body_to_world = quaternionBodyToWorld();
        s.roll_deg = last_roll_deg_;
        s.pitch_deg = last_pitch_deg_;
        s.yaw_deg = last_yaw_deg_;
        s.ax_world = ax_world_;
        s.ay_world = ay_world_;
        s.az_world = az_world_;
        s.vertical_world_accel_up = vertical_world_accel_up_;
        s.mahony_twoKp = mahony_.twoKp;
        s.mahony_twoKi = mahony_.twoKi;
        s.gravity_mps2 = gravity_mps2_;
        s.mahony_acc_trust = last_acc_trust_;
        s.mahony_sigma_sched = last_sigma_sched_;
        s.mahony_freq_sched_hz = last_freq_sched_hz_;
        return s;
    }

private:
    static constexpr T kPi_() {
        return static_cast<T>(3.1415926535897932384626433832795L);
    }

    static constexpr T kRadToDeg_() {
        return static_cast<T>(57.295779513082320876798154814105L);
    }

    static T clamp01_(T x) {
        return std::clamp(x, T(0), T(1));
    }

    static T wrapDeg_(T x) {
        while (x > T(180)) x -= T(360);
        while (x < T(-180)) x += T(360);
        return x;
    }

    static T onePoleAlpha_(T dt, T tau) {
        if (!(std::isfinite(dt) && dt > T(0))) return T(0);
        if (!(std::isfinite(tau) && tau > T(0))) return T(1);
        const T a = dt / (tau + dt);
        return std::clamp(a, T(0), T(1));
    }

    static Config sanitizeConfig_(Config cfg) {
        if (!(std::isfinite(cfg.gravity_mps2) && cfg.gravity_mps2 > T(0))) {
            cfg.gravity_mps2 = static_cast<T>(9.80665);
        }
        if (!std::isfinite(cfg.mahony_twoKp)) cfg.mahony_twoKp = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
        if (!std::isfinite(cfg.mahony_twoKi)) cfg.mahony_twoKi = static_cast<T>(Mahony_AHRS<T>::twoKiDef);

        cfg.mahony_twoKp_calm  = std::max(cfg.mahony_twoKp_calm,  T(0));
        cfg.mahony_twoKp_rough = std::max(cfg.mahony_twoKp_rough, T(0));
        cfg.mahony_twoKi_calm  = std::max(cfg.mahony_twoKi_calm,  T(0));
        cfg.mahony_twoKi_rough = std::max(cfg.mahony_twoKi_rough, T(0));

        cfg.mahony_sigma_ref = std::max(cfg.mahony_sigma_ref, T(1e-6));
        cfg.mahony_freq_ref_hz = std::max(cfg.mahony_freq_ref_hz, T(1e-6));
        cfg.mahony_norm_err_ref = std::max(cfg.mahony_norm_err_ref, T(1e-6));
        cfg.mahony_gain_smooth_tau_s = std::max(cfg.mahony_gain_smooth_tau_s, T(1e-3));
        cfg.mahony_acc_trust_min = clamp01_(cfg.mahony_acc_trust_min);

        return cfg;
    }

    // Rotate a body-frame vector into Mahony's world frame (Z-up).
    //
    // Mahony internal quaternion q = world -> body
    // So body -> world is q_conj.
    //
    // Uses the efficient quaternion-vector rotation formula:
    //   v' = v + w*t + cross(qv, t),  t = 2 * cross(qv, v)
    // for q = body->world.
    static void rotateBodyToWorldZUp_(const Mahony_AHRS<T>& m,
                                      T vx, T vy, T vz,
                                      T& ox, T& oy, T& oz)
    {
        // body->world quaternion = conjugate(world->body)
        const T qw = static_cast<T>(m.q0);
        const T qx = static_cast<T>(-m.q1);
        const T qy = static_cast<T>(-m.q2);
        const T qz = static_cast<T>(-m.q3);

        const T tx = T(2) * (qy * vz - qz * vy);
        const T ty = T(2) * (qz * vx - qx * vz);
        const T tz = T(2) * (qx * vy - qy * vx);

        ox = vx + qw * tx + (qy * tz - qz * ty);
        oy = vy + qw * ty + (qz * tx - qx * tz);
        oz = vz + qw * tz + (qx * ty - qy * tx);
    }

    void computeWorldAccelAndVertical_(T ax_body, T ay_body, T az_body) {
        rotateBodyToWorldZUp_(mahony_, ax_body, ay_body, az_body,
                              ax_world_, ay_world_, az_world_);

        // specific force -> inertial acceleration in Z-up world
        // stationary level: az_world_ ~ +g, so vertical_world_accel_up_ ~ 0
        vertical_world_accel_up_ = az_world_ - gravity_mps2_;

        if (!std::isfinite(vertical_world_accel_up_)) {
            vertical_world_accel_up_ = T(0);
        }
    }

    void updateEulerFromBodyToWorld_() {
        const QuaternionT q = quaternionBodyToWorld();

        const T qw = q.w;
        const T qx = q.x;
        const T qy = q.y;
        const T qz = q.z;

        const T sinp = std::clamp(T(2) * (qw * qy - qz * qx), T(-1), T(1));

        const T roll_rad  = std::atan2(T(2) * (qw * qx + qy * qz),
                                       T(1) - T(2) * (qx * qx + qy * qy));
        const T pitch_rad = std::asin(sinp);
        const T yaw_rad   = std::atan2(T(2) * (qw * qz + qx * qy),
                                       T(1) - T(2) * (qy * qy + qz * qz));

        last_roll_deg_  = roll_rad  * kRadToDeg_();
        last_pitch_deg_ = pitch_rad * kRadToDeg_();
        last_yaw_deg_   = wrapDeg_(yaw_rad * kRadToDeg_());
    }

    void updateMahonyGains_(T ax_body, T ay_body, T az_body, T dt_s) {
        const T acc_norm = std::sqrt(ax_body * ax_body + ay_body * ay_body + az_body * az_body);
        const T norm_err = std::abs(acc_norm - gravity_mps2_) / std::max(gravity_mps2_, T(1e-6));

        last_acc_trust_ = T(1) - clamp01_(norm_err / cfg_.mahony_norm_err_ref);
        last_acc_trust_ = std::max(last_acc_trust_, cfg_.mahony_acc_trust_min);

        last_sigma_sched_ = std::max(core_.accelSigma(), T(1e-6));
        last_freq_sched_hz_ = std::max(core_.accelFrequencyHz(), T(1e-6));

        const T sigma_ratio = std::clamp(last_sigma_sched_ / cfg_.mahony_sigma_ref, T(0.25), T(4.0));
        const T freq_ratio  = std::clamp(last_freq_sched_hz_ / cfg_.mahony_freq_ref_hz, T(0.25), T(4.0));

        const T sigma_rough = clamp01_((sigma_ratio - T(1)) / T(1.5));
        const T freq_rough  = clamp01_((freq_ratio  - T(1)) / T(1.5));
        const T trust_rough = T(1) - last_acc_trust_;

        const T roughness = clamp01_(T(0.55) * sigma_rough +
                                     T(0.25) * freq_rough +
                                     T(0.20) * trust_rough);

        const T kp_target = cfg_.mahony_twoKp_calm +
                            roughness * (cfg_.mahony_twoKp_rough - cfg_.mahony_twoKp_calm);
        const T ki_target = cfg_.mahony_twoKi_calm +
                            roughness * (cfg_.mahony_twoKi_rough - cfg_.mahony_twoKi_calm);

        const T alpha = onePoleAlpha_(dt_s, cfg_.mahony_gain_smooth_tau_s);

        mahony_.twoKp += alpha * (kp_target - mahony_.twoKp);
        mahony_.twoKi += alpha * (ki_target - mahony_.twoKi);

        if (!std::isfinite(mahony_.twoKp) || mahony_.twoKp < T(0)) mahony_.twoKp = cfg_.mahony_twoKp;
        if (!std::isfinite(mahony_.twoKi) || mahony_.twoKi < T(0)) mahony_.twoKi = cfg_.mahony_twoKi;
    }

private:
    Config cfg_{};

    Core core_;
    Mahony_AHRS<T> mahony_{};

    T gravity_mps2_ = static_cast<T>(9.80665);

    T last_roll_deg_  = T(0);
    T last_pitch_deg_ = T(0);
    T last_yaw_deg_   = T(0);

    T ax_world_ = T(0);
    T ay_world_ = T(0);
    T az_world_ = T(0);

    T vertical_world_accel_up_ = T(0);

    T last_acc_trust_ = T(1);
    T last_sigma_sched_ = T(0);
    T last_freq_sched_hz_ = T(0);
};

} // namespace marine_obs
