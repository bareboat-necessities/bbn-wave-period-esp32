#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoEigenDense.h>

#include "BoschBmi270Fifo.h"   // FIFO accel+gyro reader 

// BMM150 SensorAPI (vendor these OR pull from Arduino_BMI270_BMM150)
#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

#include "bmm150.h"
#include "bmm150_defs.h"

namespace atoms3r_ical {

using Vector3f = Eigen::Vector3f;

static inline void mapAxes_(Vector3f& a_mps2, Vector3f& w_rps, Vector3f& m_uT) {
  // TODO: paste your existing AtomS3R mapping/sign convention here.
  (void)a_mps2; (void)w_rps; (void)m_uT;
}

// Minimal ImuSample used by wizard (match your existing struct layout if you already have one)
struct ImuSample {
  Vector3f a;      // m/s^2
  Vector3f w;      // rad/s
  Vector3f m;      // uT
  float    tempC;  // degC
  uint32_t sample_us;
};

class BoschImuCal {
public:
  bool begin(TwoWire& wire, uint8_t bmi270_addr, float ag_hz,
             uint8_t bmm150_addr = 0x10 /* common BMM150 addr */)
  {
    wire_ = &wire;

    if (!fifo_.begin(wire, bmi270_addr, ag_hz)) {
      return false;
    }

    // Try to see if BMM150 is directly visible on the primary I2C bus.
    // If AtomS3R routes BMM150 via BMI270 AUX, this will fail (see note below).
    have_mag_ = (probeI2c_(bmm150_addr) == 0);
    bmm_addr_ = bmm150_addr;

    if (have_mag_) {
      if (!beginBmm150Direct_()) {
        have_mag_ = false; // keep accel/gyro working even if mag init fails
      }
    }

    last_mag_uT_.setZero();
    last_tempC_ = 25.0f;
    last_mag_ms_ = 0;
    return true;
  }

  // Read ONE fused sample for the wizard (accel+gyro always, mag if available).
  bool read(ImuSample& s_out) {
    BoschAGSample ag;
    if (!fifo_.readOneAG(ag)) {          // implement as thin wrapper over readAG(batch,1)
      return false;
    }

    Vector3f a(ag.ax, ag.ay, ag.az);
    Vector3f w(ag.gx, ag.gy, ag.gz);

    // Temp: keep constant unless you implement BMI2_TEMP read
    float tempC = last_tempC_;

    // Mag: poll at a modest rate for wizard (it also downsamples)
    Vector3f m = last_mag_uT_;
    if (have_mag_) {
      const uint32_t now_ms = millis();
      if ((uint32_t)(now_ms - last_mag_ms_) >= 20) { // ~50 Hz max
        Vector3f m_new;
        if (readBmm150_(m_new)) {
          last_mag_uT_ = m_new;
          last_mag_ms_ = now_ms;
          m = m_new;
        }
      }
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
  // BMM150 direct-I2C backend
  struct BmmIntf {
    TwoWire* wire = nullptr;
    uint8_t  addr = 0x10;
  } bmm_intf_;

  static BMM150_INTF_RET_TYPE bmmRead_(uint8_t reg, uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* i = reinterpret_cast<BmmIntf*>(intf_ptr);
    i->wire->beginTransmission(i->addr);
    i->wire->write(reg);
    if (i->wire->endTransmission(false) != 0) return (BMM150_INTF_RET_TYPE)-1;
    const uint32_t got = i->wire->requestFrom((int)i->addr, (int)len);
    if (got != len) return (BMM150_INTF_RET_TYPE)-2;
    for (uint32_t k = 0; k < len; ++k) data[k] = (uint8_t)i->wire->read();
    return (BMM150_INTF_RET_TYPE)0;
  }

  static BMM150_INTF_RET_TYPE bmmWrite_(uint8_t reg, const uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* i = reinterpret_cast<BmmIntf*>(intf_ptr);
    i->wire->beginTransmission(i->addr);
    i->wire->write(reg);
    for (uint32_t k = 0; k < len; ++k) i->wire->write(data[k]);
    return (i->wire->endTransmission(true) == 0) ? (BMM150_INTF_RET_TYPE)0 : (BMM150_INTF_RET_TYPE)-1;
  }

  static void bmmDelayUs_(uint32_t us, void*) { delayMicroseconds(us); }

  bool beginBmm150Direct_() {
    bmm_intf_.wire = wire_;
    bmm_intf_.addr = bmm_addr_;

    memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf = BMM150_I2C_INTF;
    bmm_dev_.intf_ptr = &bmm_intf_;
    bmm_dev_.read = &BoschImuCalSource::bmmRead_;
    bmm_dev_.write = &BoschImuCalSource::bmmWrite_;
    bmm_dev_.delay_us = &BoschImuCalSource::bmmDelayUs_;

    if (bmm150_init(&bmm_dev_) != 0) return false;

    // Reasonable defaults for calibration
    bmm_settings_.preset_mode = BMM150_PRESETMODE_REGULAR;
    (void)bmm150_set_presetmode(&bmm_settings_, &bmm_dev_);

    // Normal mode for continuous reads
    (void)bmm150_set_sensor_settings(BMM150_SEL_DATA_RATE | BMM150_SEL_XY_REP | BMM150_SEL_Z_REP,
                                     &bmm_settings_, &bmm_dev_);
    (void)bmm150_set_op_mode(&bmm_settings_, &bmm_dev_);
    return true;
  }

  bool readBmm150_(Vector3f& m_uT_out) {
    bmm150_mag_data md{};
    if (bmm150_read_mag_data(&md, &bmm_dev_) != 0) return false;

    // With BMM150_USE_FLOATING_POINT, md.x/y/z are floats (uT). :contentReference[oaicite:4]{index=4}
    m_uT_out = Vector3f((float)md.x, (float)md.y, (float)md.z);
    return isfinite(m_uT_out.x()) && isfinite(m_uT_out.y()) && isfinite(m_uT_out.z());
  }

  int probeI2c_(uint8_t addr) {
    wire_->beginTransmission(addr);
    return wire_->endTransmission(true);
  }

private:
  TwoWire* wire_ = nullptr;

  BoschBmi270Fifo fifo_{};

  bool have_mag_ = false;
  uint8_t bmm_addr_ = 0x10;
  struct bmm150_dev bmm_dev_{};
  struct bmm150_settings bmm_settings_{};

  Vector3f last_mag_uT_ = Vector3f::Zero();
  uint32_t last_mag_ms_ = 0;
  float last_tempC_ = 25.0f;
};

} // namespace atoms3r_ical
