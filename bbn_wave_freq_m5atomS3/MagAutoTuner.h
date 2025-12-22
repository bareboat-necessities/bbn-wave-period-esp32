#pragma once
#include <cmath>
#include <algorithm>

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

// Simple, robust “only accept good samples” accumulator for initial acc+mag alignment.
// Assumes BODY-NED specific force f_b in m/s^2.
// At rest: ax≈0, ay≈0, az≈+g (NED).
// Mag input may be raw; we normalize it to unit per-sample.
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
    float az_err_max = 0.08f;       // |az - g| max as fraction of g

    // Optional “not rotating” gating (helps a lot)
    bool  use_gyro_gate = true;
    float gyro_norm_max = 6.0f * float(M_PI) / 180.0f; // 6 deg/s

    // Mag gating (units-agnostic): just “not tiny / not insane” + stability by EMA
    float mag_norm_min = 1e-6f;
    float mag_ema_ratio_min = 0.65f; // accept if norm within [0.65, 1.55] of EMA
    float mag_ema_ratio_max = 1.55f;
    float mag_norm_ema_alpha = 0.02f;

    // How many accepted samples we require
    int   min_good_samples = 60;    // e.g. 0.6s @ 100 Hz mag
    int   max_total_samples = 400;  // safety cap

    // Optional extra: require some time span of “good” samples
    float min_good_time_sec = 0.3f;
  };

  MagAutoTuner() : cfg_(Config{}) { reset(); }

  explicit MagAutoTuner(const Config& cfg) : cfg_(cfg) { reset(); }

  void reset() {
    t_good_ = 0.0f;
    good_count_ = 0;
    total_count_ = 0;
    acc_sum_.setZero();
    mag_unit_sum_.setZero();
    mag_norm_ema_ = NAN;
    ready_ = false;
  }

  // Call only when you actually have a NEW mag sample (e.g. from your updateMag()).
  // acc_body_ned: specific force [m/s^2]
  // mag_body: raw mag vector (any units ok) – we normalize per-sample
  // gyro_body_ned: [rad/s] (optional, but recommended)
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
    acc_sum_ += acc_body_ned;
    mag_unit_sum_ += mag_u;
    good_count_++;
    t_good_ += dt;

    if (good_count_ >= cfg_.min_good_samples && t_good_ >= cfg_.min_good_time_sec) {
      // Require mag mean to be well-defined
      Eigen::Vector3f m = mag_unit_sum_ / float(good_count_);
      const float mn = m.norm();
      if (mn > 0.2f) { // not near-cancelled
        ready_ = true;
      }
    }
    return ready_;
  }

  bool isReady() const { return ready_; }
  int  goodCount() const { return good_count_; }
  float goodTimeSec() const { return t_good_; }

  // Outputs:
  //  - acc_mean: averaged specific force (for initialize_from_acc_mag)
  //  - mag_unit_mean: averaged *unit* mag in body frame (normalized)
  bool getResult(Eigen::Vector3f& acc_mean, Eigen::Vector3f& mag_unit_mean) const {
    if (!ready_ || good_count_ <= 0) return false;
    acc_mean = acc_sum_ / float(good_count_);
    Eigen::Vector3f m = mag_unit_sum_ / float(good_count_);
    const float mn = m.norm();
    mag_unit_mean = (mn > 1e-6f) ? (m / mn) : Eigen::Vector3f(1,0,0);
    return true;
  }

private:
  bool isGoodAccel_(const Eigen::Vector3f& a) const {
    const float g = cfg_.g;
    const float an = a.norm();
    if (!(an > cfg_.accel_norm_min * g && an < cfg_.accel_norm_max * g)) return false;

    // BODY-NED at rest: ax≈0, ay≈0, az≈+g
    if (std::fabs(a.x()) > cfg_.ax_abs_max * g) return false;
    if (std::fabs(a.y()) > cfg_.ay_abs_max * g) return false;
    if (std::fabs(a.z() - g) > cfg_.az_err_max * g) return false;

    return true;
  }

  bool isGoodGyro_(const Eigen::Vector3f& w) const {
    return (w.norm() < cfg_.gyro_norm_max);
  }

  bool isGoodMag_(const Eigen::Vector3f& m_raw, Eigen::Vector3f& m_unit_out) {
    const float mn = m_raw.norm();
    if (!(mn > cfg_.mag_norm_min)) return false;

    // EMA-based stability gate (units-agnostic)
    if (!std::isfinite(mag_norm_ema_)) {
      mag_norm_ema_ = mn;
    } else {
      mag_norm_ema_ = (1.0f - cfg_.mag_norm_ema_alpha) * mag_norm_ema_
                    + cfg_.mag_norm_ema_alpha * mn;
      const float r = mn / std::max(1e-9f, mag_norm_ema_);
      if (r < cfg_.mag_ema_ratio_min || r > cfg_.mag_ema_ratio_max) return false;
    }

    m_unit_out = m_raw / mn;
    return true;
  }

private:
  Config cfg_;

  float t_good_ = 0.0f;
  int   good_count_ = 0;
  int   total_count_ = 0;
  bool  ready_ = false;

  Eigen::Vector3f acc_sum_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f mag_unit_sum_ = Eigen::Vector3f::Zero();

  float mag_norm_ema_ = NAN;
};
