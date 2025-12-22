#pragma once
#include <cmath>
#include <algorithm>
#include <limits>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

// Robust “only accept good samples” accumulator for initial acc+mag alignment.
// - Acc gating assumes "rest" means |a| ≈ g and |ax|,|ay| small.
// - Z sign is handled robustly: accepts az ≈ +g OR az ≈ -g (common convention mismatch).
// - Mag gating is units-agnostic (EMA stability on norm), BUT we DO NOT normalize
//   the stored mean if you want µT-consistent reference.
// - Output provides BOTH:
//    (1) mag_uT_mean : mean magnetic field vector in body frame in the SAME UNITS as input (e.g. µT)
//    (2) mag_unit_mean : mean unit direction (dimensionless), if you want direction-only usage.
class MagAutoTuner {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    float g = 9.80665f;

    // Gravity-ish gating
    float accel_norm_min = 0.92f;   // as fraction of g
    float accel_norm_max = 1.08f;

    float ax_abs_max = 0.08f;       // as fraction of g (≈0.8 m/s^2)
    float ay_abs_max = 0.08f;

    // Robust rest-Z gating: require | |az| - g | <= az_abs_err_max * g
    float az_abs_err_max = 0.08f;

    // Optional “not rotating” gating (helps a lot)
    bool  use_gyro_gate = true;
    float gyro_norm_max = 8.0f * float(M_PI) / 180.0f; // 8 deg/s

    // Mag gating (units-agnostic): not tiny + stability by EMA on norm
    // For µT inputs, mn is typically ~20..70. For raw counts, depends.
    float mag_norm_min = 1e-6f;
    float mag_ema_ratio_min = 0.65f; // accept if norm within [0.65, 1.55] of EMA
    float mag_ema_ratio_max = 1.55f;
    float mag_norm_ema_alpha = 0.02f;

    // How many accepted samples we require
    int   min_good_samples = 40;    // e.g. 0.4s @ 100 Hz mag
    int   max_total_samples = 400;  // safety cap

    // Optional extra: require some time span of “good” samples
    float min_good_time_sec = 0.45f;
  };

  MagAutoTuner() : cfg_(Config{}) { reset(); }
  explicit MagAutoTuner(const Config& cfg) : cfg_(cfg) { reset(); }

  void reset() {
    t_good_ = 0.0f;
    good_count_ = 0;
    total_count_ = 0;
    acc_sum_.setZero();
    mag_uT_sum_.setZero();
    mag_unit_sum_.setZero();
    mag_norm_ema_ = std::numeric_limits<float>::quiet_NaN();
    ready_ = false;
  }

  // Call only when you actually have a NEW mag sample.
  // acc_body_ned: "acc sample you plan to use for init gating" (can be specific force or accel;
  //               this tuner gates only on rest-ness; it does NOT assume a particular sign for az.)
  // mag_body: raw mag vector (µT recommended if you want µT-consistent output)
  // gyro_body_ned: [rad/s] optional, but recommended
  bool addMagSample(float dt,
                    const Eigen::Vector3f& acc_body_ned,
                    const Eigen::Vector3f& mag_body,
                    const Eigen::Vector3f& gyro_body_ned = Eigen::Vector3f::Zero())
  {
    if (ready_) return true;
    if (!(dt > 0.0f) || !std::isfinite(dt)) return false;
    if (!acc_body_ned.allFinite() || !mag_body.allFinite() || !gyro_body_ned.allFinite()) return false;

    total_count_++;
    if (total_count_ > cfg_.max_total_samples) {
      // Give up gracefully; caller can decide what to do (e.g. keep waiting)
      return false;
    }

    if (!isGoodAccel_(acc_body_ned)) return false;
    if (cfg_.use_gyro_gate && !isGoodGyro_(gyro_body_ned)) return false;

    Eigen::Vector3f mag_u;
    if (!isGoodMag_(mag_body, mag_u)) return false;

    // Accept sample
    acc_sum_      += acc_body_ned;
    mag_uT_sum_   += mag_body; // KEEP RAW UNITS (µT stays µT)
    mag_unit_sum_ += mag_u;    // ALSO keep direction-only mean (dimensionless)

    good_count_++;
    t_good_ += dt;

    if (good_count_ >= cfg_.min_good_samples && t_good_ >= cfg_.min_good_time_sec) {
      // Require direction mean to be well-defined (not near-cancelled)
      Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
      const float mun = mu.norm();
      if (mun > 0.2f) {
        // Require raw mean magnitude not tiny (units preserved)
        Eigen::Vector3f mr = mag_uT_sum_ / float(good_count_);
        const float mrn = mr.norm();
        if (std::isfinite(mrn) && mrn > cfg_.mag_norm_min) {
          ready_ = true;
        }
      }
    }

    return ready_;
  }

  bool  isReady()     const { return ready_; }
  int   goodCount()   const { return good_count_; }
  float goodTimeSec() const { return t_good_; }

  // Outputs:
  //  - acc_mean: averaged accel input (whatever you fed; used for tilt init typically)
  //  - mag_uT_mean: averaged RAW mag vector in body frame, SAME UNITS as input (e.g. µT)
  //  - mag_unit_mean: averaged unit direction (dimensionless), normalized on output
  bool getResult(Eigen::Vector3f& acc_mean,
                 Eigen::Vector3f& mag_uT_mean,
                 Eigen::Vector3f& mag_unit_mean) const
  {
    if (!ready_ || good_count_ <= 0) return false;

    acc_mean    = acc_sum_ / float(good_count_);
    mag_uT_mean = mag_uT_sum_ / float(good_count_);

    Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
    const float mun = mu.norm();
    mag_unit_mean = (mun > 1e-6f) ? (mu / mun) : Eigen::Vector3f(1,0,0);

    return acc_mean.allFinite() && mag_uT_mean.allFinite() && mag_unit_mean.allFinite();
  }

  // Convenience: just µT mean (or “raw units mean”).
  bool getMagMeanRaw(Eigen::Vector3f& mag_uT_mean) const {
    if (!ready_ || good_count_ <= 0) return false;
    mag_uT_mean = mag_uT_sum_ / float(good_count_);
    return mag_uT_mean.allFinite();
  }

  // Convenience: just unit mean.
  bool getMagMeanUnit(Eigen::Vector3f& mag_unit_mean) const {
    if (!ready_ || good_count_ <= 0) return false;
    Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
    const float mun = mu.norm();
    mag_unit_mean = (mun > 1e-6f) ? (mu / mun) : Eigen::Vector3f(1,0,0);
    return mag_unit_mean.allFinite();
  }

private:
  bool isGoodAccel_(const Eigen::Vector3f& a) const {
    const float g = cfg_.g;
    const float an = a.norm();
    if (!(an > cfg_.accel_norm_min * g && an < cfg_.accel_norm_max * g)) return false;

    // At rest: ax≈0, ay≈0 (sign does not matter)
    if (std::fabs(a.x()) > cfg_.ax_abs_max * g) return false;
    if (std::fabs(a.y()) > cfg_.ay_abs_max * g) return false;

    // Robust Z gate: accept az ≈ +g or az ≈ -g
    if (std::fabs(std::fabs(a.z()) - g) > cfg_.az_abs_err_max * g) return false;

    return true;
  }

  bool isGoodGyro_(const Eigen::Vector3f& w) const {
    return (w.norm() < cfg_.gyro_norm_max);
  }

  bool isGoodMag_(const Eigen::Vector3f& m_raw, Eigen::Vector3f& m_unit_out) {
    const float mn = m_raw.norm();
    if (!(mn > cfg_.mag_norm_min) || !std::isfinite(mn)) return false;

    // EMA-based stability gate (units-agnostic)
    if (!std::isfinite(mag_norm_ema_)) {
      mag_norm_ema_ = mn;
    } else {
      mag_norm_ema_ = (1.0f - cfg_.mag_norm_ema_alpha) * mag_norm_ema_
                    + cfg_.mag_norm_ema_alpha * mn;
      const float denom = std::max(1e-9f, mag_norm_ema_);
      const float r = mn / denom;
      if (r < cfg_.mag_ema_ratio_min || r > cfg_.mag_ema_ratio_max) return false;
    }

    m_unit_out = m_raw / mn;
    return m_unit_out.allFinite();
  }

private:
  Config cfg_;

  float t_good_ = 0.0f;
  int   good_count_ = 0;
  int   total_count_ = 0;
  bool  ready_ = false;

  Eigen::Vector3f acc_sum_      = Eigen::Vector3f::Zero();
  Eigen::Vector3f mag_uT_sum_   = Eigen::Vector3f::Zero(); // RAW units accumulator (µT stays µT)
  Eigen::Vector3f mag_unit_sum_ = Eigen::Vector3f::Zero(); // direction-only accumulator

  float mag_norm_ema_ = std::numeric_limits<float>::quiet_NaN();
};
