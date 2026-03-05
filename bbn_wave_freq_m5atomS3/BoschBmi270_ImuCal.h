#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoEigenDense.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "BoschBmi270Fifo.h"   // FIFO accel+gyro reader (your production header)

// Use Bosch BMM150 SensorAPI vendored by Arduino_BMI270_BMM150
#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

#include <utilities/BMM150-Sensor-API/bmm150.h>
#include <utilities/BMM150-Sensor-API/bmm150_defs.h>

namespace atoms3r_ical {

using Vector3f = Eigen::Vector3f;

// If you already have ImuSample in AtomS3R_ImuCal.h, you should remove this struct
// and use the existing one. Keep the field names/types identical.
struct ImuSample {
  Vector3f a;      // m/s^2
  Vector3f w;      // rad/s
  Vector3f m;      // uT
  float    tempC;  // degC
  uint32_t sample_us;
};

// NOTE: You MUST replace this with your existing AtomS3R mapping/sign convention.
// For now it is identity mapping.
static inline void mapAxes_(Vector3f& a_mps2, Vector3f& w_rps, Vector3f& m_uT) {
  (void)a_mps2; (void)w_rps; (void)m_uT;
}

// Bosch-backed calibration sample source.
// - Accel/Gyro: BMI270 FIFO via BoschBmi270Fifo
// - Mag: Optional direct BMM150 I2C (NOT present on AtomS3R), otherwise NaN/last-known
class BoschBmi270_ImuCal {
public:
  bool ok() const { return ok_; }
  bool hasMag() const { return have_mag_; } // On AtomS3R this will be false (BMM150 is AUX behind BMI270)

  // bmm150_addr is only used for boards where BMM150 is on main I2C. AtomS3R: ignore.
  bool begin(TwoWire& wire,
             uint8_t bmi270_addr,
             float ag_hz,
             uint8_t bmm150_addr = 0x10)
  {
    ok_ = false;
    wire_ = &wire;

    // Accel/Gyro via FIFO
    if (!fifo_.begin(wire, bmi270_addr, ag_hz)) {
      return false;
    }

    // Optional: try direct BMM150 on main I2C (NOT AtomS3R).
    bmm_addr_ = bmm150_addr;
    have_mag_ = (probeI2c_(bmm_addr_) == 0);

    if (have_mag_) {
      if (!beginBmm150Direct_()) {
        have_mag_ = false;
      }
    }

    last_mag_uT_.setConstant(std::numeric_limits<float>::quiet_NaN());
    last_mag_ms_ = 0;
    last_tempC_  = 25.0f;

    ok_ = true;
    return true;
  }

  // Read ONE sample:
  // - Always returns accel/gyro (from FIFO)
  // - Mag:
  //   - If direct BMM150 present: returns latest (polled at a limited rate)
  //   - Otherwise: returns NaNs (so caller can skip mag stage cleanly)
  bool read(ImuSample& s_out) {
    if (!ok_) return false;

    BoschAGSample ag;
    if (!fifo_.readOneAG(ag)) return false;

    Vector3f a(ag.ax, ag.ay, ag.az);
    Vector3f w(ag.gx, ag.gy, ag.gz);

    // Temp: keep constant unless you implement BMI270 temp read
    const float tempC = last_tempC_;

    // Mag path
    Vector3f m = last_mag_uT_;
    if (have_mag_) {
      const uint32_t now_ms = millis();
      // BMM150 useful ODR is limited; cap poll rate to avoid wasting I2C time
      if (last_mag_ms_ == 0 || (uint32_t)(now_ms - last_mag_ms_) >= MAG_POLL_MS) {
        Vector3f m_new;
        if (readBmm150_(m_new)) {
          last_mag_uT_ = m_new;
          last_mag_ms_ = now_ms;
          m = m_new;
        }
      }
    } else {
      // Ensure NaNs propagate (lets wizard skip mag stage deterministically)
      m.setConstant(std::numeric_limits<float>::quiet_NaN());
    }

    mapAxes_(a, w, m);

    s_out.a = a;
    s_out.w = w;
    s_out.m = m;
    s_out.tempC = tempC;
    s_out.sample_us = micros();
    return true;
  }

private:
  // ---------- Direct BMM150 I2C backend (for boards where BMM150 is visible on main I2C) ----------
  static constexpr uint16_t I2C_CHUNK   = 32;  // safe with default Wire buffers
  static constexpr uint32_t MAG_POLL_MS = 40;  // ~25 Hz cap

  struct BmmIntf {
    TwoWire* wire = nullptr;
    uint8_t  addr = 0x10;
  } bmm_intf_;

  static BMM150_INTF_RET_TYPE bmmRead_(uint8_t reg, uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* i = reinterpret_cast<BmmIntf*>(intf_ptr);
    if (!i || !i->wire || !data) return (BMM150_INTF_RET_TYPE)-1;

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = (uint16_t)std::min<uint32_t>(I2C_CHUNK, len - off);

      i->wire->beginTransmission(i->addr);
      i->wire->write((uint8_t)(reg + off));
      if (i->wire->endTransmission(false) != 0) return (BMM150_INTF_RET_TYPE)-2;

      const uint16_t got = (uint16_t)i->wire->requestFrom((int)i->addr, (int)n);
      if (got != n) return (BMM150_INTF_RET_TYPE)-3;

      for (uint16_t k = 0; k < n; ++k) data[off + k] = (uint8_t)i->wire->read();
      off += n;
    }
    return (BMM150_INTF_RET_TYPE)0;
  }

  static BMM150_INTF_RET_TYPE bmmWrite_(uint8_t reg, const uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* i = reinterpret_cast<BmmIntf*>(intf_ptr);
    if (!i || !i->wire || !data) return (BMM150_INTF_RET_TYPE)-1;

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = (uint16_t)std::min<uint32_t>(I2C_CHUNK, len - off);

      i->wire->beginTransmission(i->addr);
      i->wire->write((uint8_t)(reg + off));
      for (uint16_t k = 0; k < n; ++k) i->wire->write(data[off + k]);
      if (i->wire->endTransmission(true) != 0) return (BMM150_INTF_RET_TYPE)-2;

      off += n;
    }
    return (BMM150_INTF_RET_TYPE)0;
  }

  static void bmmDelayUs_(uint32_t us, void*) { delayMicroseconds(us); }

  bool beginBmm150Direct_() {
    bmm_intf_.wire = wire_;
    bmm_intf_.addr = bmm_addr_;

    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf      = BMM150_I2C_INTF;
    bmm_dev_.intf_ptr  = &bmm_intf_;
    bmm_dev_.read      = &BoschBmi270_ImuCal::bmmRead_;
    bmm_dev_.write     = &BoschBmi270_ImuCal::bmmWrite_;
    bmm_dev_.delay_us  = &BoschBmi270_ImuCal::bmmDelayUs_;

    if (bmm150_init(&bmm_dev_) != 0) return false;

    // Regular preset is a good “wizard” default (balance noise/speed)
    bmm_settings_.preset_mode = BMM150_PRESETMODE_REGULAR;
    (void)bmm150_set_presetmode(&bmm_settings_, &bmm_dev_);

    // Enable settings + continuous mode
    // (Use what presetmode configured; keep it simple and stable.)
    (void)bmm150_set_sensor_settings(
        (uint16_t)(BMM150_SEL_DATA_RATE | BMM150_SEL_XY_REP | BMM150_SEL_Z_REP),
        &bmm_settings_, &bmm_dev_);

    (void)bmm150_set_op_mode(&bmm_settings_, &bmm_dev_);
    return true;
  }

  bool readBmm150_(Vector3f& m_uT_out) {
    bmm150_mag_data md{};
    if (bmm150_read_mag_data(&md, &bmm_dev_) != 0) return false;

    // With BMM150_USE_FLOATING_POINT=1, md fields are float uT.
    m_uT_out = Vector3f((float)md.x, (float)md.y, (float)md.z);

    return std::isfinite(m_uT_out.x()) && std::isfinite(m_uT_out.y()) && std::isfinite(m_uT_out.z());
  }

  int probeI2c_(uint8_t addr) {
    if (!wire_) return -1;
    wire_->beginTransmission(addr);
    return wire_->endTransmission(true);
  }

private:
  TwoWire* wire_ = nullptr;
  bool ok_ = false;

  BoschBmi270Fifo fifo_{};

  // Mag (direct BMM150 only)
  bool have_mag_ = false;
  uint8_t bmm_addr_ = 0x10;

  struct bmm150_dev bmm_dev_{};
  struct bmm150_settings bmm_settings_{};

  Vector3f  last_mag_uT_ = Vector3f::Zero();
  uint32_t  last_mag_ms_ = 0;
  float     last_tempC_  = 25.0f;
};

} // namespace atoms3r_ical
