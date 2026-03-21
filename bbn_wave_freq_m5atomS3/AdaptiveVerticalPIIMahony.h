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
  - Mahony state is float-only because your Mahony implementation is float-only.
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
        float mahony_twoKp = twoKpDef;
        float mahony_twoKi = twoKiDef;

        // Gravity magnitude used to convert specific force -> inertial accel
        float gravity_mps2 = 9.80665f;

        // If true, updateIMU() will call magless Mahony update only.
        // If you want magnetometer support, call updateIMUMag().
        bool use_mag = false;
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

        float mahony_twoKp = twoKpDef;
        float mahony_twoKi = twoKiDef;
        float gravity_mps2 = 9.80665f;
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
    }

    void reset(T p0 = T(0),
               T v0 = T(0),
               T a_f0 = T(0),
               T S0 = T(0),
               T d0 = T(0),
               T b0 = T(0))
    {
        // Reset Mahony state
        mahony_ = Mahony_AHRS_Vars{};
        mahony_AHRS_init(&mahony_, cfg_.mahony_twoKp, cfg_.mahony_twoKi);

        last_roll_deg_ = T(0);
        last_pitch_deg_ = T(0);
        last_yaw_deg_ = T(0);

        ax_world_ = ay_world_ = az_world_ = T(0);
        vertical_world_accel_up_ = T(0);

        core_.reset(p0, v0, a_f0, S0, d0, b0);
    }

    void setMahonyGains(float twoKp, float twoKi) {
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

        float pitch_deg = 0.0f;
        float roll_deg  = 0.0f;
        float yaw_deg   = 0.0f;

        mahony_AHRS_update(&mahony_,
                           static_cast<float>(gx_rad_s),
                           static_cast<float>(gy_rad_s),
                           static_cast<float>(gz_rad_s),
                           static_cast<float>(ax_mps2),
                           static_cast<float>(ay_mps2),
                           static_cast<float>(az_mps2),
                           &pitch_deg, &roll_deg, &yaw_deg,
                           static_cast<float>(dt_s));

        last_pitch_deg_ = static_cast<T>(pitch_deg);
        last_roll_deg_  = static_cast<T>(roll_deg);
        last_yaw_deg_   = static_cast<T>(yaw_deg);

        computeWorldAccelAndVertical_(ax_mps2, ay_mps2, az_mps2);

        return core_.update(vertical_world_accel_up_, dt_s);
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

        float pitch_deg = 0.0f;
        float roll_deg  = 0.0f;
        float yaw_deg   = 0.0f;

        mahony_AHRS_update_mag(&mahony_,
                               static_cast<float>(gx_rad_s),
                               static_cast<float>(gy_rad_s),
                               static_cast<float>(gz_rad_s),
                               static_cast<float>(ax_mps2),
                               static_cast<float>(ay_mps2),
                               static_cast<float>(az_mps2),
                               static_cast<float>(mx),
                               static_cast<float>(my),
                               static_cast<float>(mz),
                               &pitch_deg, &roll_deg, &yaw_deg,
                               static_cast<float>(dt_s));

        last_pitch_deg_ = static_cast<T>(pitch_deg);
        last_roll_deg_  = static_cast<T>(roll_deg);
        last_yaw_deg_   = static_cast<T>(yaw_deg);

        computeWorldAccelAndVertical_(ax_mps2, ay_mps2, az_mps2);

        return core_.update(vertical_world_accel_up_, dt_s);
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

    Mahony_AHRS_Vars& mahonyState() { return mahony_; }
    const Mahony_AHRS_Vars& mahonyState() const { return mahony_; }

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
        return s;
    }

private:
    static Config sanitizeConfig_(Config cfg) {
        if (!(std::isfinite(cfg.gravity_mps2) && cfg.gravity_mps2 > 0.0f)) {
            cfg.gravity_mps2 = 9.80665f;
        }
        if (!std::isfinite(cfg.mahony_twoKp)) cfg.mahony_twoKp = twoKpDef;
        if (!std::isfinite(cfg.mahony_twoKi)) cfg.mahony_twoKi = twoKiDef;
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
    static void rotateBodyToWorldZUp_(const Mahony_AHRS_Vars& m,
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
        vertical_world_accel_up_ = az_world_ - static_cast<T>(gravity_mps2_);

        if (!std::isfinite(vertical_world_accel_up_)) {
            vertical_world_accel_up_ = T(0);
        }
    }

private:
    Config cfg_{};

    Core core_;
    Mahony_AHRS_Vars mahony_{};

    float gravity_mps2_ = 9.80665f;

    T last_roll_deg_  = T(0);
    T last_pitch_deg_ = T(0);
    T last_yaw_deg_   = T(0);

    T ax_world_ = T(0);
    T ay_world_ = T(0);
    T az_world_ = T(0);

    T vertical_world_accel_up_ = T(0);
};

} // namespace marine_obs


/*
EXAMPLE

#include "AdaptiveVerticalPIIMahony.h"

using Heave = marine_obs::AdaptiveVerticalPIIMahony<float, true>;

Heave::Config cfg;

// core observer
cfg.core.observer.r = 0.16f;
cfg.core.observer.tau_a = 0.60f;
cfg.core.observer.tau_d = 40.0f;
cfg.core.observer.kb = 1e-4f;
cfg.core.observer.lambda_b = 1e-2f;

// adaptation
cfg.core.adaptation.enabled = true;
cfg.core.adaptation.f_disp_ref_hz = 0.17f;
cfg.core.adaptation.sigma_a_ref = 0.30f;

// optional accel-frequency fallback
cfg.core.auto_schedule_from_accel_freq = false;

// Mahony
cfg.mahony_twoKp = twoKpDef;
cfg.mahony_twoKi = twoKiDef;
cfg.gravity_mps2 = 9.80665f;

Heave filt(cfg);

// Every IMU sample:
float z = filt.updateIMU(gx_rad_s, gy_rad_s, gz_rad_s,
                         ax_mps2, ay_mps2, az_mps2,
                         dt_sec);

// Preferred adaptation source:
filt.updateAdaptationFromDisplacementFrequency(f_disp_hz, dt_track, confidence);

// Diagnostics:
float a_up = filt.verticalWorldAccelUp();
float roll = filt.rollDeg();
float pitch = filt.pitchDeg();

*/
