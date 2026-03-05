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
*/

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "BoschBmi270Fifo.h"
#include "BoschBmm150Aux.h"

// Pull in the canonical types + NED convention helpers from your existing plumbing.
// This provides:
//   - atoms3r_ical::Vector3f
//   - atoms3r_ical::ImuSample (with fields a,w,m,tempC,mask,sample_us)
//   - atoms3r_ical::kImuMaskAccelGyro (mask for accel+gyro)
#include "AtomS3R_ImuCal.h"

namespace atoms3r_ical {

class BoschBmi270_ImuCal {
public:
  struct Config {
    uint8_t  bmi270_addr = 0x68;
    float    ag_hz       = 100.0f;

    // Mag over AUX (AtomS3R): recommended ON
    bool     enable_mag_aux = true;
    float    mag_aux_odr_hz = 25.0f;      // 25 Hz is plenty for yaw correction and calibration
    bool     mag_pullups_2k = true;

    // Temp: BMI270 temp read not implemented here; keep a constant for bias(T) models
    float    tempC_default  = 25.0f;

    // I2C clock used by BoschBmi270Fifo::begin
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

    // Accel/Gyro FIFO
    if (!fifo_.begin(*wire_, cfg_.bmi270_addr, cfg_.ag_hz, cfg_.i2c_hz)) {
      return false;
    }

    // Mag over AUX (AtomS3R correct)
    if (cfg_.enable_mag_aux) {
      BoschBmm150Aux::Config mcfg;
      mcfg.bmm_addr       = BMM150_DEFAULT_I2C_ADDRESS; // 0x10
      mcfg.aux_odr_hz     = cfg_.mag_aux_odr_hz;
      mcfg.set_pullups_2k = cfg_.mag_pullups_2k;
      mcfg.preset_mode    = BMM150_PRESETMODE_REGULAR;
      mcfg.normal_mode    = true;

      mag_ok_ = mag_.begin(fifo_.rawDev(), mcfg);
    } else {
      mag_ok_ = false;
    }

    last_mag_uT_.setConstant(std::numeric_limits<float>::quiet_NaN());
    last_mag_ms_ = 0;

    ok_ = true;
    return true;
  }

  // Read one mapped sample (BODY NED convention, same as AtomS3R_ImuCal.h).
  // - Always returns accel+gyro
  // - Mag returns latest aux mag if available, otherwise NaNs
  bool read(ImuSample& out) {
    if (!ok_) return false;

    BoschAGSample ag;
    if (!fifo_.readOneAG(ag)) return false;

    // Bosch FIFO gives sensor-frame accel/gyro already in m/s^2 and rad/s.
    // Apply AtomS3R mapping: (y, x, -z)
    Vector3f a_s(ag.ax, ag.ay, ag.az);
    Vector3f w_s(ag.gx, ag.gy, ag.gz);
    Vector3f a_b(a_s.y(), a_s.x(), -a_s.z());
    Vector3f w_b(w_s.y(), w_s.x(), -w_s.z());

    // Mag
    Vector3f m_b;
    if (mag_ok_) {
      const uint32_t now_ms = millis();
      // cap aux mag polling; you can lower this if you want more frequent updates
      if (last_mag_ms_ == 0 || (uint32_t)(now_ms - last_mag_ms_) >= MAG_POLL_MS) {
        Vector3f m_s;
        if (mag_.readMag_uT(m_s)) {
          // Apply AtomS3R mapping: (y, x, -z)
          last_mag_uT_ = Vector3f(m_s.y(), m_s.x(), -m_s.z());
          last_mag_ms_ = now_ms;
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
    out.mask = kImuMaskAccelGyro;     // accel+gyro are valid in this source
    out.sample_us = micros();
    return true;
  }

private:
  static constexpr uint32_t MAG_POLL_MS = 40; // ~25 Hz

  TwoWire* wire_ = nullptr;
  Config cfg_{};

  bool ok_ = false;
  bool mag_ok_ = false;

  BoschBmi270Fifo fifo_{};
  BoschBmm150Aux  mag_{};

  Vector3f  last_mag_uT_ = Vector3f::Zero();
  uint32_t  last_mag_ms_ = 0;
};

} // namespace atoms3r_ical
