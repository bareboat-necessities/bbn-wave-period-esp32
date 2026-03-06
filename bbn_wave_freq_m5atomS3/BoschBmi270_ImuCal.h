#pragma once

/*
  BoschBmi270_ImuCal.h  (AtomS3R)

  Production-hardened IMU sample source for calibration + runtime that matches
  the SAME axis convention as AtomS3R_ImuCal.h (BODY NED):

    acc_body = ( ay, ax, -az )    [m/s^2]
    gyr_body = ( gy, gx, -gz )    [rad/s]
    mag_body = ( my, mx, -mz )    [uT]

  - Accel/Gyro: BMI270 FIFO via BoschBmi270Fifo
  - Mag: BMM150 via BMI270 AUX using BoschBmm150Aux manual AUX bridge

  Notes:
  - Reuses atoms3r_ical::ImuSample / Vector3f from AtomS3R_ImuCal.h
    to avoid type duplication and guarantee identical conventions.
  - sample_us is FIFO-time-derived, not read-time-derived.
  - out.mask intentionally reports accel+gyro only, matching AtomS3R_ImuCal.h.
  - Mag validity is communicated by finite out.m versus NaN, plus hasMag().
*/

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "BoschBmi270Fifo.h"
#include "BoschBmm150Aux.h"
#include "AtomS3R_ImuCal.h"

namespace atoms3r_ical {

class BoschBmi270_ImuCal {
public:
  struct Config {
    uint8_t  bmi270_addr = 0x68;
    float    ag_hz       = 100.0f;   // BoschBmi270Fifo will quantize to 100/200 Hz

    bool     enable_mag_aux             = true;
    uint8_t  mag_bmm150_addr            = 0x10;
    float    mag_aux_odr_hz             = 25.0f;    // manual AUX bridge service rate
    uint16_t mag_startup_settle_ms      = 3;
    bool     mag_verify_first_read      = true;

    // Mag is considered stale once older than:
    // max(mag_stale_min_us, mag_stale_factor * poll_period).
    uint32_t mag_stale_min_us           = 75000u;
    uint8_t  mag_stale_factor           = 3u;

    // Best-effort mag recovery after repeated failures.
    bool     enable_mag_recovery        = true;
    uint8_t  mag_recover_after_failures = 6u;
    uint32_t mag_recover_cooldown_us    = 1000000u;

    float    tempC_default              = 25.0f;
    uint32_t i2c_hz                     = 400000u;
  };

  enum class Error : uint8_t {
    NONE = 0,
    NOT_INITIALIZED,
    FIFO_BEGIN_FAILED,
    FIFO_READ_FAILED,
    MAG_BEGIN_FAILED,
    MAG_READ_FAILED,
    MAG_STALE,
    MAG_RECOVER_FAILED,
    END_MAG_FAILED,
    END_AG_FAILED
  };

  BoschBmi270_ImuCal() = default;

  bool ok() const { return ok_; }

  // "hasMag" means currently fresh/usable magnetometer output is available.
  bool hasMag() const { return magHealthy(); }
  bool magConfigured() const { return mag_configured_; }
  bool haveValidMagSample() const { return magHealthy(); }

  bool magHealthy() const {
    return mag_configured_ && have_valid_mag_ && !magCurrentlyStale_();
  }

  Error lastError() const { return last_error_; }

  const char* lastErrorString() const {
    switch (last_error_) {
      case Error::NONE:               return "none";
      case Error::NOT_INITIALIZED:    return "not initialized";
      case Error::FIFO_BEGIN_FAILED:  return "BMI270 FIFO begin failed";
      case Error::FIFO_READ_FAILED:   return "BMI270 FIFO read failed";
      case Error::MAG_BEGIN_FAILED:   return "BMM150 AUX begin failed";
      case Error::MAG_READ_FAILED:    return "BMM150 read failed";
      case Error::MAG_STALE:          return "BMM150 data stale";
      case Error::MAG_RECOVER_FAILED: return "BMM150 recovery failed";
      case Error::END_MAG_FAILED:     return "BMM150 shutdown failed";
      case Error::END_AG_FAILED:      return "BMI270 accel/gyro shutdown failed";
      default:                        return "unknown";
    }
  }

  const Config& config() const { return cfg_; }

  const BoschBmi270Fifo& fifo() const { return fifo_; }
  BoschBmi270Fifo& fifo() { return fifo_; }

  const BoschBmm150Aux& mag() const { return mag_; }
  BoschBmm150Aux& mag() { return mag_; }

  uint32_t readTotal() const                { return read_total_; }
  uint32_t agReadFailuresTotal() const      { return ag_read_fail_total_; }
  uint32_t magPollsTotal() const            { return mag_poll_total_; }
  uint32_t magReadOkTotal() const           { return mag_ok_total_; }
  uint32_t magReadFailuresTotal() const     { return mag_fail_total_; }
  uint32_t magStaleTransitionsTotal() const { return mag_stale_total_; }
  uint32_t magRecoveriesTotal() const       { return mag_recover_total_; }
  uint32_t recoveriesTotal() const          { return recover_total_; }

  uint64_t sampleClockUs64() const { return sample_clock_us_; }
  uint64_t lastMagSampleUs64() const { return last_mag_sample_us_; }

  const Vector3f& lastGoodMag_uT() const { return last_mag_uT_; }
  bool haveLastGoodMag() const { return have_last_good_mag_; }

  bool begin(TwoWire& wire, const Config& cfg = Config()) {
    if (ok_ || mag_configured_) {
      if (!end()) {
        return false;
      }
    }

    wire_ = &wire;
    cfg_  = cfg;

    resetRuntimeState_();
    last_error_ = Error::NONE;

    if (!fifo_.begin(wire, cfg_.bmi270_addr, cfg_.ag_hz, cfg_.i2c_hz)) {
      ok_ = false;
      last_error_ = Error::FIFO_BEGIN_FAILED;
      return false;
    }

    ok_ = true;

#if defined(ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI) && ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI
    if (cfg_.enable_mag_aux) {
      BoschBmm150Aux::Config mcfg;
      mcfg.bmm_addr          = cfg_.mag_bmm150_addr;
      mcfg.aux_odr_hz        = sanitizedMagOdrHz_();
      mcfg.startup_settle_ms = cfg_.mag_startup_settle_ms;
      mcfg.verify_first_read = cfg_.mag_verify_first_read;

      mag_configured_ = mag_.begin(fifo_.rawDev(), mcfg);
      if (!mag_configured_) {
        last_error_ = Error::MAG_BEGIN_FAILED;
      }
    } else {
      mag_configured_ = false;
    }
#else
    mag_configured_ = false;
#endif

    return true;
  }

  bool recover() {
    if (wire_ == nullptr) {
      last_error_ = Error::NOT_INITIALIZED;
      return false;
    }

    const bool ok = begin(*wire_, cfg_);
    if (ok) {
      ++recover_total_;
    }
    return ok;
  }

  bool end() {
    bool all_ok = true;
    Error first_err = Error::NONE;

#if defined(ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI) && ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI
    if (mag_configured_) {
      if (!mag_.end()) {
        all_ok = false;
        first_err = Error::END_MAG_FAILED;
      }
    }
#endif

#if defined(ATOMS3R_HAVE_BOSCH_SENSORAPI) && ATOMS3R_HAVE_BOSCH_SENSORAPI
    if (fifo_.rawDev() != nullptr) {
      const uint8_t sens[2] = { BMI2_ACCEL, BMI2_GYRO };
      if (bmi270_sensor_disable(sens, 2, fifo_.rawDev()) != BMI2_OK) {
        all_ok = false;
        if (first_err == Error::NONE) {
          first_err = Error::END_AG_FAILED;
        }
      }
      (void)bmi2_flush_fifo(fifo_.rawDev());
    }
#endif

    ok_ = false;
    mag_configured_ = false;
    resetRuntimeState_();

    if (!all_ok) {
      last_error_ = first_err;
      return false;
    }

    last_error_ = Error::NONE;
    return true;
  }

  bool read(ImuSample& out) {
    if (!ok_) {
      last_error_ = Error::NOT_INITIALIZED;
      return false;
    }

    BoschAGSample ag;
    if (!fifo_.readOneAG(ag)) {
      ++ag_read_fail_total_;
      last_error_ = Error::FIFO_READ_FAILED;
      return false;
    }

    ++read_total_;

    const uint64_t sample_us64 = advanceSampleClockUs_(ag.dt_s);

    // Sensor frame -> AtomS3R BODY NED
    const Vector3f a_s(ag.ax, ag.ay, ag.az);
    const Vector3f w_s(ag.gx, ag.gy, ag.gz);

    const Vector3f a_b(a_s.y(), a_s.x(), -a_s.z());
    const Vector3f w_b(w_s.y(), w_s.x(), -w_s.z());

    maybePollMag_(sample_us64);
    updateMagFreshness_(sample_us64);

    out.a = a_b;
    out.w = w_b;
    out.m = magHealthy() ? last_mag_uT_ : nanVec_();
    out.tempC = cfg_.tempC_default;
    out.mask = kImuMaskAccelGyro;
    out.sample_us = static_cast<uint32_t>(sample_us64 & 0xFFFFFFFFull);

    // Clear non-fatal mag status once mag becomes healthy again.
    if (magHealthy() &&
        (last_error_ == Error::MAG_READ_FAILED ||
         last_error_ == Error::MAG_STALE ||
         last_error_ == Error::MAG_RECOVER_FAILED)) {
      last_error_ = Error::NONE;
    }

    return true;
  }

private:
  static Vector3f nanVec_() {
    return Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
  }

  static bool finite3_(const Vector3f& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  }

  static uint64_t clampU64_(uint64_t v, uint64_t lo, uint64_t hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
  }

  float sanitizedMagOdrHz_() const {
    const float hz = cfg_.mag_aux_odr_hz;
    if (!(hz > 0.0f) || !std::isfinite(hz)) {
      return 25.0f;
    }
    return hz;
  }

  uint64_t nominalDtUs_() const {
    return (cfg_.ag_hz > 150.0f) ? 5000ull : 10000ull;
  }

  uint64_t magPollUs_() const {
    const float hz = sanitizedMagOdrHz_();
    const float us_f = 1.0e6f / hz;
    if (!(us_f > 0.0f) || !std::isfinite(us_f)) {
      return 40000ull;
    }
    const uint64_t us = static_cast<uint64_t>(us_f + 0.5f);
    return clampU64_(us, 5000ull, 1000000ull);
  }

  uint64_t magStaleAfterUs_() const {
    const uint64_t min_us = (cfg_.mag_stale_min_us > 0u)
                          ? static_cast<uint64_t>(cfg_.mag_stale_min_us)
                          : 75000ull;

    const uint64_t factor = (cfg_.mag_stale_factor > 0u)
                          ? static_cast<uint64_t>(cfg_.mag_stale_factor)
                          : 3ull;

    const uint64_t dyn_us = factor * magPollUs_();
    return (dyn_us > min_us) ? dyn_us : min_us;
  }

  uint64_t advanceSampleClockUs_(float dt_s) {
    if (!have_sample_clock_) {
      have_sample_clock_ = true;
      sample_clock_us_ = 0ull;
      sample_clock_frac_us_ = 0.0;
      return 0ull;
    }

    double dt_us_f = static_cast<double>(dt_s) * 1.0e6;
    if (!(dt_us_f > 0.0) || !std::isfinite(dt_us_f)) {
      dt_us_f = static_cast<double>(nominalDtUs_());
    }

    dt_us_f += sample_clock_frac_us_;

    uint64_t dt_us = static_cast<uint64_t>(dt_us_f);
    sample_clock_frac_us_ = dt_us_f - static_cast<double>(dt_us);

    if (dt_us == 0ull) {
      dt_us = 1ull;
    }

    sample_clock_us_ += dt_us;
    return sample_clock_us_;
  }

  void maybePollMag_(uint64_t sample_us) {
#if defined(ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI) && ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI
    if (!mag_configured_) {
      return;
    }

    if (!have_mag_poll_time_ || (sample_us - last_mag_poll_us_) >= magPollUs_()) {
      have_mag_poll_time_ = true;
      last_mag_poll_us_ = sample_us;
      ++mag_poll_total_;

      Vector3f m_s;
      if (mag_.readMag_uT(m_s) && finite3_(m_s)) {
        // Sensor frame -> AtomS3R BODY NED
        last_mag_uT_ = Vector3f(m_s.y(), m_s.x(), -m_s.z());

        have_last_good_mag_ = true;
        have_valid_mag_ = true;
        mag_marked_stale_ = false;
        last_mag_sample_us_ = sample_us;
        mag_consecutive_failures_ = 0;
        ++mag_ok_total_;
        return;
      }

      ++mag_fail_total_;
      ++mag_consecutive_failures_;
      last_error_ = Error::MAG_READ_FAILED;

      if (cfg_.enable_mag_recovery &&
          mag_consecutive_failures_ >= cfg_.mag_recover_after_failures &&
          (last_mag_recover_attempt_us_ == 0ull ||
           (sample_us - last_mag_recover_attempt_us_) >= static_cast<uint64_t>(cfg_.mag_recover_cooldown_us))) {
        last_mag_recover_attempt_us_ = sample_us;
        if (recoverMag_()) {
          ++mag_recover_total_;
          mag_consecutive_failures_ = 0;
        } else {
          last_error_ = Error::MAG_RECOVER_FAILED;
        }
      }
    }
#else
    (void)sample_us;
#endif
  }

  void updateMagFreshness_(uint64_t sample_us) {
    if (!have_valid_mag_) {
      return;
    }

    if (last_mag_sample_us_ > sample_us) {
      return;
    }

    const uint64_t age_us = sample_us - last_mag_sample_us_;
    if (age_us > magStaleAfterUs_()) {
      have_valid_mag_ = false;
      if (!mag_marked_stale_) {
        mag_marked_stale_ = true;
        ++mag_stale_total_;
      }
      last_error_ = Error::MAG_STALE;
    }
  }

  bool magCurrentlyStale_() const {
    if (!have_valid_mag_ || !have_sample_clock_) {
      return true;
    }
    if (last_mag_sample_us_ > sample_clock_us_) {
      return false;
    }
    return (sample_clock_us_ - last_mag_sample_us_) > magStaleAfterUs_();
  }

  bool recoverMag_() {
#if defined(ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI) && ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI
    if (!ok_ || fifo_.rawDev() == nullptr) {
      return false;
    }

    if (mag_configured_) {
      (void)mag_.end();
    }

    BoschBmm150Aux::Config mcfg;
    mcfg.bmm_addr          = cfg_.mag_bmm150_addr;
    mcfg.aux_odr_hz        = sanitizedMagOdrHz_();
    mcfg.startup_settle_ms = cfg_.mag_startup_settle_ms;
    mcfg.verify_first_read = cfg_.mag_verify_first_read;

    mag_configured_ = mag_.begin(fifo_.rawDev(), mcfg);
    if (!mag_configured_) {
      have_valid_mag_ = false;
      return false;
    }

    return true;
#else
    return false;
#endif
  }

  void resetRuntimeState_() {
    last_mag_uT_ = nanVec_();

    have_last_good_mag_ = false;
    have_valid_mag_ = false;
    mag_marked_stale_ = false;

    have_sample_clock_ = false;
    sample_clock_us_ = 0ull;
    sample_clock_frac_us_ = 0.0;

    have_mag_poll_time_ = false;
    last_mag_poll_us_ = 0ull;
    last_mag_sample_us_ = 0ull;
    last_mag_recover_attempt_us_ = 0ull;

    mag_consecutive_failures_ = 0u;
  }

private:
  TwoWire* wire_ = nullptr;
  Config   cfg_{};

  bool ok_ = false;
  bool mag_configured_ = false;

  Error last_error_ = Error::NONE;

  BoschBmi270Fifo fifo_{};
  BoschBmm150Aux  mag_{};

  Vector3f last_mag_uT_ = nanVec_();

  bool     have_last_good_mag_ = false;
  bool     have_valid_mag_     = false;
  bool     mag_marked_stale_   = false;

  bool     have_sample_clock_   = false;
  uint64_t sample_clock_us_     = 0ull;
  double   sample_clock_frac_us_ = 0.0;

  bool     have_mag_poll_time_         = false;
  uint64_t last_mag_poll_us_           = 0ull;
  uint64_t last_mag_sample_us_         = 0ull;
  uint64_t last_mag_recover_attempt_us_ = 0ull;

  uint8_t  mag_consecutive_failures_ = 0u;

  uint32_t read_total_         = 0u;
  uint32_t ag_read_fail_total_ = 0u;
  uint32_t mag_poll_total_     = 0u;
  uint32_t mag_ok_total_       = 0u;
  uint32_t mag_fail_total_     = 0u;
  uint32_t mag_stale_total_    = 0u;
  uint32_t mag_recover_total_  = 0u;
  uint32_t recover_total_      = 0u;
};

} // namespace atoms3r_ical
