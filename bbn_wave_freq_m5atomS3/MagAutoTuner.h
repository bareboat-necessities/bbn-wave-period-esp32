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
//
// Key changes vs your earlier version:
//  - DOES NOT require ax≈0, ay≈0 (heel-safe). Instead gates on:
//      • |a| ≈ g
//      • normalized accel direction is STABLE over time (EMA on unit vector)
//  - Optional gyro gate remains (not rotating).
//  - Mag gating remains units-agnostic (EMA stability on norm).
//  - Keeps RAW mag mean in same units as input (e.g. µT) for consistent reference.
//
// Output provides BOTH:
//   (1) mag_raw_mean : mean magnetic field vector in body frame in SAME UNITS as input (e.g. µT)
//   (2) mag_unit_mean: mean unit direction (dimensionless)
class MagAutoTuner {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    float g = 9.80665f;

    // |a| gating (fraction of g)
    float accel_norm_min = 0.92f;
    float accel_norm_max = 1.08f;

    // Heel-safe “direction stability” gate:
    // Keep an EMA of a_unit, require current a_unit to be close to EMA direction.
    // Good starting values: alpha 0.02..0.05, cos_min 0.995..0.999
    float acc_dir_ema_alpha   = 0.03f;
    float acc_dir_cos_min     = 0.997f;   // ~4.4 deg
    float acc_dir_min_valid_n = 1e-6f;

    // Optional “not rotating” gating
    bool  use_gyro_gate = true;
    float gyro_norm_max = 8.0f * float(M_PI) / 180.0f; // 8 deg/s

    // Mag gating (units-agnostic): not tiny + stability by EMA on norm
    float mag_norm_min = 1e-6f;
    float mag_ema_ratio_min = 0.65f;
    float mag_ema_ratio_max = 1.55f;
    float mag_norm_ema_alpha = 0.02f;

    // How many accepted samples we require
    int   min_good_samples  = 40;    // e.g. 0.4s @ 100 Hz mag
    int   max_total_samples = 400;   // safety cap
    float min_good_time_sec = 0.45f;
  };

  MagAutoTuner() : cfg_(Config{}) { reset(); }
  explicit MagAutoTuner(const Config& cfg) : cfg_(cfg) { reset(); }

  void setConfig(const Config& cfg) { cfg_ = cfg; reset(); }

  void reset() {
    t_good_ = 0.0f;
    good_count_ = 0;
    total_count_ = 0;
    acc_sum_.setZero();
    mag_raw_sum_.setZero();
    mag_unit_sum_.setZero();

    mag_norm_ema_ = std::numeric_limits<float>::quiet_NaN();

    acc_dir_ema_.setZero();
    acc_dir_ema_valid_ = false;

    ready_ = false;
  }

  // Call only when you actually have a NEW mag sample.
  // dt: time since previous mag sample (or a reasonable estimate)
  // acc_body_ned: accelerometer sample (specific force or accel; we only use it to detect stable gravity direction)
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
      // Give up gracefully; caller decides fallback.
      return false;
    }

    if (!isGoodAccel_(acc_body_ned)) return false;
    if (cfg_.use_gyro_gate && !isGoodGyro_(gyro_body_ned)) return false;

    Eigen::Vector3f mag_u;
    if (!isGoodMag_(mag_body, mag_u)) return false;

    // Accept sample
    acc_sum_      += acc_body_ned;
    mag_raw_sum_  += mag_body; // KEEP RAW UNITS (µT stays µT)
    mag_unit_sum_ += mag_u;

    good_count_++;
    t_good_ += dt;

    if (good_count_ >= cfg_.min_good_samples && t_good_ >= cfg_.min_good_time_sec) {
      // Direction mean must be well-defined (not near-cancelled)
      Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
      const float mun = mu.norm();
      if (mun > 0.2f) {
        // Raw mean magnitude not tiny
        Eigen::Vector3f mr = mag_raw_sum_ / float(good_count_);
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
  int   totalCount()  const { return total_count_; }

  bool getResult(Eigen::Vector3f& acc_mean,
                 Eigen::Vector3f& mag_raw_mean,
                 Eigen::Vector3f& mag_unit_mean) const
  {
    if (!ready_ || good_count_ <= 0) return false;

    acc_mean     = acc_sum_ / float(good_count_);
    mag_raw_mean = mag_raw_sum_ / float(good_count_);

    Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
    const float mun = mu.norm();
    mag_unit_mean = (mun > 1e-6f) ? (mu / mun) : Eigen::Vector3f(1,0,0);

    return acc_mean.allFinite() && mag_raw_mean.allFinite() && mag_unit_mean.allFinite();
  }

  bool getMagMeanRaw(Eigen::Vector3f& mag_raw_mean) const {
    if (!ready_ || good_count_ <= 0) return false;
    mag_raw_mean = mag_raw_sum_ / float(good_count_);
    return mag_raw_mean.allFinite();
  }

  bool getMagMeanUnit(Eigen::Vector3f& mag_unit_mean) const {
    if (!ready_ || good_count_ <= 0) return false;
    Eigen::Vector3f mu = mag_unit_sum_ / float(good_count_);
    const float mun = mu.norm();
    mag_unit_mean = (mun > 1e-6f) ? (mu / mun) : Eigen::Vector3f(1,0,0);
    return mag_unit_mean.allFinite();
  }

private:
  bool isGoodAccel_(const Eigen::Vector3f& a) {
    const float g = cfg_.g;
    const float an = a.norm();
    if (!(an > cfg_.accel_norm_min * g && an < cfg_.accel_norm_max * g)) return false;

    // Direction stability gate (heel-safe):
    // Use a_unit = a / |a|, compare to EMA direction.
    if (!(an > cfg_.acc_dir_min_valid_n) || !std::isfinite(an)) return false;
    Eigen::Vector3f a_unit = a / an;

    if (!acc_dir_ema_valid_) {
      acc_dir_ema_ = a_unit;
      acc_dir_ema_valid_ = true;
      return true; // first accepted direction
    }

    // Update EMA in a “sign-consistent” way (avoid flip if sensor convention differs)
    // Choose sign that keeps it closest to EMA.
    float dot0 = a_unit.dot(acc_dir_ema_);
    if (dot0 < 0.0f) a_unit = -a_unit;

    // cos(angle) between current and EMA direction
    const float ema_n = std::max(1e-9f, acc_dir_ema_.norm());
    const float cosang = (a_unit.dot(acc_dir_ema_)) / ema_n;

    if (!(cosang >= cfg_.acc_dir_cos_min)) return false;

    // EMA update
    const float aE = std::min(std::max(cfg_.acc_dir_ema_alpha, 0.0f), 1.0f);
    acc_dir_ema_ = (1.0f - aE) * acc_dir_ema_ + aE * a_unit;

    return acc_dir_ema_.allFinite();
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

  Eigen::Vector3f acc_sum_     = Eigen::Vector3f::Zero();
  Eigen::Vector3f mag_raw_sum_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f mag_unit_sum_= Eigen::Vector3f::Zero();

  float mag_norm_ema_ = std::numeric_limits<float>::quiet_NaN();

  // Heel-safe accel direction EMA
  Eigen::Vector3f acc_dir_ema_ = Eigen::Vector3f::Zero();
  bool acc_dir_ema_valid_ = false;
};
