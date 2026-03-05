#pragma once

/*
  BoschBmi270_ImuCal.h  (AtomS3R)

  Production-ready IMU sample source for calibration + runtime that matches the
  SAME axis convention as AtomS3R_ImuCal.h (BODY NED):

    acc_body = ( ay, ax, -az )    [m/s^2]
    gyr_body = ( gy, gx, -gz )    [rad/s]
    mag_body = ( my, mx, -mz )    [uT]

  - Accel/Gyro: BMI270 FIFO via BoschBmi270Fifo (stable, low jitter)
  - Mag: BMM150 via BMI270 AUX using BoschBmm150Aux (AtomS3R-correct path)

  Notes:
  - This header intentionally reuses atoms3r_ical::ImuSample / Vector3f from AtomS3R_ImuCal.h
    to avoid type duplication and guarantee identical conventions across the codebase.
  - sample_us is FIFO-time-derived, not read-time-derived.
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
    float    ag_hz       = 100.0f;

    bool     enable_mag_aux = true;
    float    mag_aux_odr_hz = 25.0f;
    bool     mag_pullups_2k = true;

    float    tempC_default  = 25.0f;

    uint32_t i2c_hz = 400000;
  };

  bool ok() const { return ok_; }
  bool hasMag() const { return mag_ok_; }

  const BoschBmi270Fifo& fifo() const { return fifo_; }
  BoschBmi270Fifo& fifo() { return fifo_; }

  const BoschBmm150Aux& mag() const { return mag_; }
  BoschBmm150Aux& mag() { return mag_; }

  bool begin(TwoWire& wire, const Config& cfg = Config()) {
    ok_ = false;
    mag_ok_ = false;
    cfg_ = cfg;
    wire_ = &wire;

    if (!fifo_.begin(*wire_, cfg_.bmi270_addr, cfg_.ag_hz, cfg_.i2c_hz)) {
      return false;
    }

    if (cfg_.enable_mag_aux) {
      BoschBmm150Aux::Config mcfg;
      mcfg.bmm_addr       = BMM150_DEFAULT_I2C_ADDRESS;
      mcfg.aux_odr_hz     = cfg_.mag_aux_odr_hz;
      mcfg.set_pullups_2k = cfg_.mag_pullups_2k;
      mcfg.preset_mode    = BMM150_PRESETMODE_REGULAR;
      mcfg.normal_mode    = true;

      mag_ok_ = mag_.begin(fifo_.rawDev(), mcfg);
    } else {
      mag_ok_ = false;
    }

    last_mag_uT_.setConstant(std::numeric_limits<float>::quiet_NaN());

    have_sample_clock_    = false;
    sample_clock_us_      = 0;
    sample_clock_frac_us_ = 0.0f;

    last_mag_poll_us_   = 0;
    have_mag_poll_time_ = false;

    ok_ = true;
    return true;
  }

  bool read(ImuSample& out) {
    if (!ok_) return false;

    BoschAGSample ag;
    if (!fifo_.readOneAG(ag)) return false;

    const uint64_t sample_us = advanceSampleClockUs_(ag.dt_s);

    Vector3f a_s(ag.ax, ag.ay, ag.az);
    Vector3f w_s(ag.gx, ag.gy, ag.gz);
    Vector3f a_b(a_s.y(), a_s.x(), -a_s.z());
    Vector3f w_b(w_s.y(), w_s.x(), -w_s.z());

    Vector3f m_b;
    if (mag_ok_) {
      if (!have_mag_poll_time_ || (sample_us - last_mag_poll_us_) >= MAG_POLL_US) {
        last_mag_poll_us_   = sample_us;
        have_mag_poll_time_ = true;

        Vector3f m_s;
        if (mag_.readMag_uT(m_s)) {
          last_mag_uT_ = Vector3f(m_s.y(), m_s.x(), -m_s.z());
        }
      }
      m_b = last_mag_uT_;
    } else {
      m_b.setConstant(std::numeric_limits<float>::quiet_NaN());
    }

    out.a = a_b;
    out.w = w_b;
    out.m = m_b;
    out.tempC = cfg_.tempC_default;
    out.mask = kImuMaskAccelGyro;
    out.sample_us = sample_us;
    return true;
  }

private:
  static constexpr uint64_t MAG_POLL_US = 40000ull;

  TwoWire* wire_ = nullptr;
  Config cfg_{};

  bool ok_ = false;
  bool mag_ok_ = false;

  BoschBmi270Fifo fifo_{};
  BoschBmm150Aux  mag_{};

  Vector3f last_mag_uT_ = Vector3f::Zero();

  bool     have_sample_clock_    = false;
  uint64_t sample_clock_us_      = 0;
  float    sample_clock_frac_us_ = 0.0f;

  uint64_t last_mag_poll_us_   = 0;
  bool     have_mag_poll_time_ = false;

  uint32_t nominalDtUs_() const {
    return (cfg_.ag_hz > 150.0f) ? 5000u : 10000u;
  }

  uint64_t advanceSampleClockUs_(float dt_s) {
    if (!have_sample_clock_) {
      have_sample_clock_    = true;
      sample_clock_us_      = 0;
      sample_clock_frac_us_ = 0.0f;
      return 0ull;
    }

    float dt_us_f = dt_s * 1.0e6f;

    if (!(dt_us_f > 0.0f)) {
      dt_us_f = static_cast<float>(nominalDtUs_());
    }

    dt_us_f += sample_clock_frac_us_;

    const uint64_t dt_us = static_cast<uint64_t>(dt_us_f);
    sample_clock_frac_us_ = dt_us_f - static_cast<float>(dt_us);

    sample_clock_us_ += (dt_us > 0ull) ? dt_us : 1ull;
    return sample_clock_us_;
  }
};

} // namespace atoms3r_ical
