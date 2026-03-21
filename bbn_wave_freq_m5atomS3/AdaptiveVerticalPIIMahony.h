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

        // Optional sea-state scheduling of Mahony gains.
        //
        // Calm seas / trustworthy accel  -> higher gains
        // Rough seas / poor accel trust  -> lower gains
        bool adapt_mahony_gains = false;

        T mahony_twoKp_calm  = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
        T mahony_twoKp_rough = static_cast<T>(Mahony_AHRS<T>::twoKpDef);

        T mahony_twoKi_calm  = static_cast<T>(Mahony_AHRS<T>::twoKiDef);
        T mahony_twoKi_rough = static_cast<T>(Mahony_AHRS<T>::twoKiDef);

        // Reference sea-state for Mahony scheduling.
        T mahony_sigma_ref      = static_cast<T>(0.18);
        T mahony_freq_ref_hz    = static_cast<T>(0.12);
        T mahony_norm_err_ref   = static_cast<T>(0.08);

        // Gain smoothing time constant.
        T mahony_gain_smooth_tau_s = static_cast<T>(2.0);

        // Minimum retained accel trust when accel norm is distorted.
        // 0 -> can fully shut off accel correction
        // 1 -> never attenuate accel correction
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

        mahony_twoKp_active_ = cfg_.mahony_twoKp;
        mahony_twoKi_active_ = cfg_.mahony_twoKi;

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
        mahony_ = Mahony_AHRS<T>{};
        mahony_AHRS_init(&mahony_, cfg_.mahony_twoKp, cfg_.mahony_twoKi);

        mahony_twoKp_active_ = cfg_.mahony_twoKp;
        mahony_twoKi_active_ = cfg_.mahony_twoKi;

        last_roll_deg_ = T(0);
        last_pitch_deg_ = T(0);
        last_yaw_deg_ = T(0);

        ax_world_ = ay_world_ = az_world_ = T(0);
        vertical_world_accel_up_ = T(0);

        core_.reset(p0, v0, a_f0, S0, d0, b0);
    }

    void setMahonyGains(T twoKp, T twoKi) {
        if (std::isfinite(twoKp)) mahony_.twoKp = twoKp;
        if (std::isfinite(twoKi)) mahony_.twoKi = twoKi;
        cfg_.mahony_twoKp = mahony_.twoKp;
        cfg_.mahony_twoKi = mahony_.twoKi;
        mahony_twoKp_active_ = mahony_.twoKp;
        mahony_twoKi_active_ = mahony_.twoKi;
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

        adaptMahonyGains_(ax_mps2, ay_mps2, az_mps2, dt_s);

        T pitch_deg = T(0);
        T roll_deg  = T(0);
        T yaw_deg   = T(0);

        mahony_AHRS_update(&mahony_,
                           gx_rad_s, gy_rad_s, gz_rad_s,
                           ax_mps2, ay_mps2, az_mps2,
                           &pitch_deg, &roll_deg, &yaw_deg,
                           dt_s);

        (void)pitch_deg;
        (void)roll_deg;
        (void)yaw_deg;

        updateEulerFromBodyToWorld_();
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

        adaptMahonyGains_(ax_mps2, ay_mps2, az_mps2, dt_s);

        T pitch_deg = T(0);
        T roll_deg  = T(0);
        T yaw_deg   = T(0);

        mahony_AHRS_update_mag(&mahony_,
                               gx_rad_s, gy_rad_s, gz_rad_s,
                               ax_mps2, ay_mps2, az_mps2,
                               mx, my, mz,
                               &pitch_deg, &roll_deg, &yaw_deg,
                               dt_s);

        (void)pitch_deg;
        (void)roll_deg;
        (void)yaw_deg;

        updateEulerFromBodyToWorld_();
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
        return s;
    }

private:
    static Config sanitizeConfig_(Config cfg) {
        if (!(std::isfinite(cfg.gravity_mps2) && cfg.gravity_mps2 > T(0))) {
            cfg.gravity_mps2 = static_cast<T>(9.80665);
        }

        if (!std::isfinite(cfg.mahony_twoKp)) {
            cfg.mahony_twoKp = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
        }
        if (!std::isfinite(cfg.mahony_twoKi)) {
            cfg.mahony_twoKi = static_cast<T>(Mahony_AHRS<T>::twoKiDef);
        }

        if (!std::isfinite(cfg.mahony_twoKp_calm) || cfg.mahony_twoKp_calm < T(0)) {
            cfg.mahony_twoKp_calm = cfg.mahony_twoKp;
        }
        if (!std::isfinite(cfg.mahony_twoKp_rough) || cfg.mahony_twoKp_rough < T(0)) {
            cfg.mahony_twoKp_rough = cfg.mahony_twoKp;
        }

        if (!std::isfinite(cfg.mahony_twoKi_calm) || cfg.mahony_twoKi_calm < T(0)) {
            cfg.mahony_twoKi_calm = cfg.mahony_twoKi;
        }
        if (!std::isfinite(cfg.mahony_twoKi_rough) || cfg.mahony_twoKi_rough < T(0)) {
            cfg.mahony_twoKi_rough = cfg.mahony_twoKi;
        }

        if (!(std::isfinite(cfg.mahony_sigma_ref) && cfg.mahony_sigma_ref > T(0))) {
            cfg.mahony_sigma_ref = static_cast<T>(0.18);
        }
        if (!(std::isfinite(cfg.mahony_freq_ref_hz) && cfg.mahony_freq_ref_hz > T(0))) {
            cfg.mahony_freq_ref_hz = static_cast<T>(0.12);
        }
        if (!(std::isfinite(cfg.mahony_norm_err_ref) && cfg.mahony_norm_err_ref > T(0))) {
            cfg.mahony_norm_err_ref = static_cast<T>(0.08);
        }
        if (!(std::isfinite(cfg.mahony_gain_smooth_tau_s) && cfg.mahony_gain_smooth_tau_s > T(0))) {
            cfg.mahony_gain_smooth_tau_s = static_cast<T>(2.0);
        }

        cfg.mahony_acc_trust_min = clamp01_(cfg.mahony_acc_trust_min);
        return cfg;
    }

    static T clamp01_(T x) {
        return std::clamp(x, T(0), T(1));
    }

    static T finiteOr_(T x, T def) {
        return std::isfinite(x) ? x : def;
    }

    static T onePoleAlpha_(T dt, T tau) {
        if (!(std::isfinite(dt) && dt > T(0))) return T(0);
        if (!(std::isfinite(tau) && tau > T(0))) return T(1);
        const T a = dt / (tau + dt);
        return std::clamp(a, T(0), T(1));
    }

    static T lerp_(T a, T b, T t) {
        t = clamp01_(t);
        return a + t * (b - a);
    }

    static T safeNorm3_(T x, T y, T z) {
        return std::sqrt(x * x + y * y + z * z);
    }

    void adaptMahonyGains_(T ax_body, T ay_body, T az_body, T dt_s) {
        if (!cfg_.adapt_mahony_gains) {
            mahony_.twoKp = cfg_.mahony_twoKp;
            mahony_.twoKi = cfg_.mahony_twoKi;
            mahony_twoKp_active_ = mahony_.twoKp;
            mahony_twoKi_active_ = mahony_.twoKi;
            return;
        }

        const auto hs = core_.snapshot();

        T sigma_used = hs.observer.sigma_a_filt;
        if (!(std::isfinite(sigma_used) && sigma_used > T(0))) {
            sigma_used = hs.accel_sigma;
        }
        if (!(std::isfinite(sigma_used) && sigma_used > T(0))) {
            sigma_used = cfg_.mahony_sigma_ref;
        }

        T f_used_hz = hs.accel_freq_sched_hz;
        if (!(std::isfinite(f_used_hz) && f_used_hz > T(0))) {
            f_used_hz = hs.observer.f_disp_filt_hz;
        }
        if (!(std::isfinite(f_used_hz) && f_used_hz > T(0))) {
            f_used_hz = hs.accel_freq_hz;
        }
        if (!(std::isfinite(f_used_hz) && f_used_hz > T(0))) {
            f_used_hz = cfg_.mahony_freq_ref_hz;
        }

        const T sigma_ratio = std::clamp(sigma_used / std::max(cfg_.mahony_sigma_ref, T(1e-6)),
                                         T(0.25), T(4.0));
        const T freq_ratio = std::clamp(f_used_hz / std::max(cfg_.mahony_freq_ref_hz, T(1e-6)),
                                        T(0.25), T(4.0));

        // Roughness estimate:
        //   calm  -> 0
        //   rough -> 1
        const T rough_sigma = clamp01_((sigma_ratio - T(1)) / T(1.0));
        const T rough_freq  = clamp01_((freq_ratio  - T(1)) / T(1.0));
        const T roughness   = clamp01_(T(0.65) * rough_sigma + T(0.35) * rough_freq);

        // Accel-trust from norm consistency.
        const T acc_norm = safeNorm3_(ax_body, ay_body, az_body);
        const T norm_err = std::abs(acc_norm - gravity_mps2_) / std::max(gravity_mps2_, T(1e-6));
        const T norm_trust = T(1) - clamp01_(norm_err / std::max(cfg_.mahony_norm_err_ref, T(1e-6)));
        const T accel_trust = std::max(cfg_.mahony_acc_trust_min, norm_trust);

        const T kp_sea = lerp_(cfg_.mahony_twoKp_calm, cfg_.mahony_twoKp_rough, roughness);
        const T ki_sea = lerp_(cfg_.mahony_twoKi_calm, cfg_.mahony_twoKi_rough, roughness);

        const T kp_cmd = std::max(T(0), kp_sea * accel_trust);
        const T ki_cmd = std::max(T(0), ki_sea * accel_trust * norm_trust);

        const T alpha = onePoleAlpha_(dt_s, cfg_.mahony_gain_smooth_tau_s);
        mahony_twoKp_active_ += alpha * (kp_cmd - mahony_twoKp_active_);
        mahony_twoKi_active_ += alpha * (ki_cmd - mahony_twoKi_active_);

        mahony_.twoKp = std::max(T(0), finiteOr_(mahony_twoKp_active_, cfg_.mahony_twoKp));
        mahony_.twoKi = std::max(T(0), finiteOr_(mahony_twoKi_active_, cfg_.mahony_twoKi));
    }

    void updateEulerFromBodyToWorld_() {
        const T qw = mahony_.q0;
        const T qx = -mahony_.q1;
        const T qy = -mahony_.q2;
        const T qz = -mahony_.q3;

        const T sinr_cosp = T(2) * (qw * qx + qy * qz);
        const T cosr_cosp = T(1) - T(2) * (qx * qx + qy * qy);
        const T roll = std::atan2(sinr_cosp, cosr_cosp);

        T sinp = T(2) * (qw * qy - qz * qx);
        sinp = std::clamp(sinp, T(-1), T(1));
        const T pitch = std::asin(sinp);

        const T siny_cosp = T(2) * (qw * qz + qx * qy);
        const T cosy_cosp = T(1) - T(2) * (qy * qy + qz * qz);
        const T yaw = std::atan2(siny_cosp, cosy_cosp);

        last_roll_deg_  = roll  * static_cast<T>(Mahony_AHRS<T>::kRadToDeg);
        last_pitch_deg_ = pitch * static_cast<T>(Mahony_AHRS<T>::kRadToDeg);
        last_yaw_deg_   = yaw   * static_cast<T>(Mahony_AHRS<T>::kRadToDeg);
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

private:
    Config cfg_{};

    Core core_;
    Mahony_AHRS<T> mahony_{};

    T gravity_mps2_ = static_cast<T>(9.80665);

    T mahony_twoKp_active_ = static_cast<T>(Mahony_AHRS<T>::twoKpDef);
    T mahony_twoKi_active_ = static_cast<T>(Mahony_AHRS<T>::twoKiDef);

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
cfg.core.auto_schedule_from_accel_freq = true;

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
