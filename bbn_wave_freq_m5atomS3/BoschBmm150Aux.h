// BoschBmm150Aux.h
#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

// Bosch SensorAPI headers (vendored inside Arduino_BMI270_BMM150)
#include <utilities/BMI270-Sensor-API/bmi2.h>
#include <utilities/BMI270-Sensor-API/bmi2_defs.h>
#include <utilities/BMI270-Sensor-API/bmi270.h>

#include <utilities/BMM150-Sensor-API/bmm150.h>
#include <utilities/BMM150-Sensor-API/bmm150_defs.h>

#include <ArduinoEigenDense.h>

namespace atoms3r_ical {

using Vector3f = Eigen::Vector3f;

class BoschBmm150Aux {
public:
  struct Config {
    uint8_t bmm_addr = BMM150_DEFAULT_I2C_ADDRESS; // usually 0x10
    float   aux_odr_hz = 25.0f;                    // 25Hz is plenty for heading/yaw correction
    bool    set_pullups_2k = true;                 // harmless if board already has pullups
    uint8_t preset_mode = BMM150_PRESETMODE_REGULAR;
    bool    normal_mode = true;                    // true => normal/continuous; false => forced (not recommended here)
  };

  BoschBmm150Aux() = default;

  bool ok() const { return ok_; }

  // Initialize BMM150 via BMI270 AUX, using the *existing* BMI270 SensorAPI device.
  // bmi_dev must remain valid for the lifetime of this object.
  bool begin(struct bmi2_dev* bmi_dev, const Config& cfg = Config()) {
    ok_ = false;
    bmi_dev_ = bmi_dev;
    cfg_ = cfg;

    if (!bmi_dev_) return false;

    // AUX is more reliable with advanced power save disabled.
    (void)bmi2_set_adv_power_save(BMI2_DISABLE, bmi_dev_);

    // Optional: configure AUX pullups.
    // (Some Bosch/M5 examples tweak AUX_IF_TRIM; guard because not all builds expose these.)
#if defined(BMI2_AUX_IF_TRIM) && defined(BMI2_ASDA_PUPSEL_2K)
    if (cfg_.set_pullups_2k) {
      uint8_t regdata = BMI2_ASDA_PUPSEL_2K;
      (void)bmi2_set_regs(BMI2_AUX_IF_TRIM, &regdata, 1, bmi_dev_);
    }
#endif

    // Enable AUX sensor block (doesn't touch accel/gyro enables).
    {
      const uint8_t sens = BMI2_AUX;
      (void)bmi2_sensor_enable(&sens, 1, bmi_dev_);
    }

    // Configure AUX in MANUAL mode so BMM150 SensorAPI can talk via bmi2_read_aux_man_mode().
    bmi2_sens_config sc{};
    sc.type = BMI2_AUX;

    int8_t rslt = bmi270_get_sensor_config(&sc, 1, bmi_dev_);
    if (rslt != BMI2_OK) return false;

    // These fields exist in the Bosch BMI270 SensorAPI used by Arduino_BMI270_BMM150.
    sc.cfg.aux.aux_en          = BMI2_ENABLE;
    sc.cfg.aux.manual_en       = BMI2_ENABLE;
    sc.cfg.aux.i2c_device_addr = cfg_.bmm_addr;

    // Critical: allow writes to the AUX device via FCU.
    sc.cfg.aux.fcu_write_en = BMI2_ENABLE;

    // Manual burst read length. BMM150 data block (0x42..0x49) is 8 bytes.
#if defined(BMI2_AUX_READ_LEN_3)
    sc.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_3;
#elif defined(BMI2_AUX_READ_LEN_2)
    sc.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_2;
#elif defined(BMI2_AUX_READ_LEN_1)
    sc.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_1;
#else
    sc.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_0;
#endif

    // Start address for manual reads (BMM150 data block).
#if defined(BMM150_REG_DATA_X_LSB)
    sc.cfg.aux.read_addr = BMM150_REG_DATA_X_LSB;
#elif defined(BMM150_DATA_X_LSB)
    sc.cfg.aux.read_addr = BMM150_DATA_X_LSB;
#else
    sc.cfg.aux.read_addr = 0x42;
#endif

    sc.cfg.aux.odr = auxOdrFromHz_(cfg_.aux_odr_hz);

    rslt = bmi270_set_sensor_config(&sc, 1, bmi_dev_);
    if (rslt != BMI2_OK) return false;

    // Hook BMM150 SensorAPI to BMI270 AUX manual read/write.
    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf     = BMM150_I2C_INTF; // still "I2C" logically; transport is provided by callbacks
    bmm_dev_.intf_ptr = this;
    bmm_dev_.read     = &BoschBmm150Aux::bmm_read_;
    bmm_dev_.write    = &BoschBmm150Aux::bmm_write_;
    bmm_dev_.delay_us = &BoschBmm150Aux::bmm_delay_us_;

    const int8_t br = bmm150_init(&bmm_dev_);
    if (br != BMM150_OK) return false;

    // Configure BMM150 operating mode / preset.
    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));
    bmm_settings_.preset_mode = cfg_.preset_mode;
    (void)bmm150_set_presetmode(&bmm_settings_, &bmm_dev_);

    // Prefer continuous mode for stable “latest sample” reads through AUX.
    // If your SensorAPI lacks these macros, fall back to 0 (common "normal").
    if (cfg_.normal_mode) {
#if defined(BMM150_POWERMODE_NORMAL)
      bmm_settings_.pwr_mode = BMM150_POWERMODE_NORMAL;
#elif defined(BMM150_NORMAL_MODE)
      bmm_settings_.pwr_mode = BMM150_NORMAL_MODE;
#else
      bmm_settings_.pwr_mode = 0x00;
#endif
    } else {
#if defined(BMM150_POWERMODE_FORCED)
      bmm_settings_.pwr_mode = BMM150_POWERMODE_FORCED;
#elif defined(BMM150_FORCED_MODE)
      bmm_settings_.pwr_mode = BMM150_FORCED_MODE;
#else
      bmm_settings_.pwr_mode = 0x01;
#endif
    }

    (void)bmm150_set_op_mode(&bmm_settings_, &bmm_dev_);

    // Optional: make sure we can read chip-id once (helps catch AUX misconfig early).
    uint8_t id = 0;
    if (!readChipId(id)) return false;

    ok_ = true;
    return true;
  }

  // Read compensated magnetometer in micro-tesla.
  bool readMag_uT(Vector3f& m_uT_out) {
    if (!ok_) return false;

    bmm150_mag_data md{};
    const int8_t r = bmm150_read_mag_data(&md, &bmm_dev_);
    if (r != BMM150_OK) return false;

    const float mx = (float)md.x;
    const float my = (float)md.y;
    const float mz = (float)md.z;

    if (!std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(mz)) return false;
    m_uT_out = Vector3f(mx, my, mz);
    return true;
  }

  // Read chip-id via AUX path (optional validation).
  bool readChipId(uint8_t& chip_id_out) {
#if defined(BMM150_REG_CHIP_ID)
    uint8_t id = 0;
    if (bmm_read_(BMM150_REG_CHIP_ID, &id, 1, this) != 0) return false;
    chip_id_out = id;
    return true;
#else
    (void)chip_id_out;
    return false;
#endif
  }

private:
  // Map float Hz to BMI2_AUX_ODR_* macro.
  static uint8_t auxOdrFromHz_(float hz) {
#if defined(BMI2_AUX_ODR_200HZ)
    if (hz >= 150.0f) return BMI2_AUX_ODR_200HZ;
#endif
#if defined(BMI2_AUX_ODR_100HZ)
    if (hz >= 75.0f) return BMI2_AUX_ODR_100HZ;
#endif
#if defined(BMI2_AUX_ODR_50HZ)
    if (hz >= 37.0f) return BMI2_AUX_ODR_50HZ;
#endif
#if defined(BMI2_AUX_ODR_25HZ)
    if (hz >= 18.0f) return BMI2_AUX_ODR_25HZ;
#endif
#if defined(BMI2_AUX_ODR_12_5HZ)
    if (hz >= 9.0f) return BMI2_AUX_ODR_12_5HZ;
#endif
#if defined(BMI2_AUX_ODR_6_25HZ)
    if (hz >= 4.5f) return BMI2_AUX_ODR_6_25HZ;
#endif
#if defined(BMI2_AUX_ODR_3_12HZ)
    if (hz >= 2.2f) return BMI2_AUX_ODR_3_12HZ;
#endif
#if defined(BMI2_AUX_ODR_1_56HZ)
    if (hz >= 1.0f) return BMI2_AUX_ODR_1_56HZ;
#endif
#if defined(BMI2_AUX_ODR_0_78HZ)
    return BMI2_AUX_ODR_0_78HZ;
#else
#if defined(BMI2_AUX_ODR_25HZ)
    return BMI2_AUX_ODR_25HZ;
#else
    return 0;
#endif
#endif
  }

  // BMM150 SensorAPI transport using BMI270 AUX manual mode.
  static BMM150_INTF_RET_TYPE bmm_read_(uint8_t reg, uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_) return (BMM150_INTF_RET_TYPE)-1;

    const int8_t r = bmi2_read_aux_man_mode(reg, data, len, self->bmi_dev_);
    return (BMM150_INTF_RET_TYPE)r;
  }

  static BMM150_INTF_RET_TYPE bmm_write_(uint8_t reg, const uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_) return (BMM150_INTF_RET_TYPE)-1;

    const int8_t r = bmi2_write_aux_man_mode(reg, data, len, self->bmi_dev_);
    return (BMM150_INTF_RET_TYPE)r;
  }

  static void bmm_delay_us_(uint32_t us, void*) { delayMicroseconds(us); }

private:
  bool ok_ = false;

  Config cfg_{};
  struct bmi2_dev* bmi_dev_ = nullptr;

  struct bmm150_dev      bmm_dev_{};
  struct bmm150_settings bmm_settings_{};
};

} // namespace atoms3r_ical
