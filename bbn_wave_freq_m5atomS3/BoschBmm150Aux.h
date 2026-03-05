// BoschBmm150Aux.h
#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstdint>

#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

// Bosch SensorAPI headers (vendored inside Arduino_BMI270_BMM150)
#include <utilities/BMI270-Sensor-API/bmi2.h>
#include <utilities/BMI270-Sensor-API/bmi270.h>
#include <utilities/BMI270-Sensor-API/bmi2_defs.h>

#include <utilities/BMM150-Sensor-API/bmm150.h>
#include <utilities/BMM150-Sensor-API/bmm150_defs.h>

#include <ArduinoEigenDense.h>

namespace atoms3r_ical {

using Vector3f = Eigen::Vector3f;

class BoschBmm150Aux {
public:
  BoschBmm150Aux() = default;

  bool ok() const { return ok_; }

  // Initialize BMM150 via BMI270 AUX, using the *existing* BMI270 SensorAPI device.
  //
  // bmi_dev must remain valid for the lifetime of this object.
  // bmm_addr: almost always BMM150_DEFAULT_I2C_ADDRESS (0x10)
  // aux_odr_hz: AUX ODR setting inside BMI270 (not the IMU FIFO ODR).
  bool begin(struct bmi2_dev* bmi_dev,
             uint8_t bmm_addr = BMM150_DEFAULT_I2C_ADDRESS,
             float aux_odr_hz = 100.0f,
             bool set_pullups_2k = true)
  {
    ok_ = false;
    bmi_dev_ = bmi_dev;
    bmm_addr_ = bmm_addr;

    if (!bmi_dev_) return false;

    // AUX works more reliably with APS disabled (FIFO setups often do this already).
    (void)bmi2_set_adv_power_save(BMI2_DISABLE, bmi_dev_);

    // Optional: configure AUX pullups (seen in vendor examples for BMI270+BMM150).
    // If your board already has pullups, this is usually harmless.
#if defined(BMI2_AUX_IF_TRIM) && defined(BMI2_ASDA_PUPSEL_2K)
    if (set_pullups_2k) {
      uint8_t regdata = BMI2_ASDA_PUPSEL_2K;
      (void)bmi2_set_regs(BMI2_AUX_IF_TRIM, &regdata, 1, bmi_dev_);
    }
#else
    (void)set_pullups_2k;
#endif

    // Enable AUX sensor block (doesn't touch accel/gyro enables).
    {
      const uint8_t sens = BMI2_AUX;
      (void)bmi2_sensor_enable(&sens, 1, bmi_dev_);
    }

    // Configure AUX in MANUAL mode so bmm150_* can talk via bmi2_read_aux_man_mode().
    struct bmi2_sens_config cfg{};
    cfg.type = BMI2_AUX;

    int8_t rslt = bmi270_get_sensor_config(&cfg, 1, bmi_dev_);
    if (rslt != BMI2_OK) return false;

    cfg.cfg.aux.aux_en = BMI2_ENABLE;
    cfg.cfg.aux.manual_en = BMI2_ENABLE;
    cfg.cfg.aux.i2c_device_addr = bmm_addr_;

    // Critical for being able to WRITE BMM150 regs via BMI270 AUX (commonly missed).
    cfg.cfg.aux.fcu_write_en = BMI2_ENABLE;  [oai_citation:1‡Bosch Sensortec Community](https://community.bosch-sensortec.com/mems-sensors-forum-jrmujtaw/post/bmm150-not-updating-in-forced-mode-5cScOBVRV62u0o8?utm_source=chatgpt.com)

    // Manual burst read length: BMM150 mag block (0x42..0x49) is 8 bytes.
#if defined(BMI2_AUX_READ_LEN_3)
    cfg.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_3;
#else
    // Fallback if your build only has _0.._2
    cfg.cfg.aux.man_rd_burst = BMI2_AUX_READ_LEN_2;
#endif

    // Start address for manual reads (BMM150 data block).
#if defined(BMM150_REG_DATA_X_LSB)
    cfg.cfg.aux.read_addr = BMM150_REG_DATA_X_LSB;
#elif defined(BMM150_DATA_X_LSB)
    cfg.cfg.aux.read_addr = BMM150_DATA_X_LSB;
#else
    cfg.cfg.aux.read_addr = 0x42; // datasheet default
#endif

    cfg.cfg.aux.odr = auxOdrFromHz_(aux_odr_hz);

    rslt = bmi270_set_sensor_config(&cfg, 1, bmi_dev_);
    if (rslt != BMI2_OK) return false;

    // Hook BMM150 SensorAPI to BMI270 AUX manual read/write.
    memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf = BMM150_I2C_INTF;
    bmm_dev_.intf_ptr = this;
    bmm_dev_.read = &BoschBmm150Aux::bmm_read_;
    bmm_dev_.write = &BoschBmm150Aux::bmm_write_;
    bmm_dev_.delay_us = &BoschBmm150Aux::bmm_delay_us_;

    int8_t br = bmm150_init(&bmm_dev_);
    if (br != BMM150_OK) return false;

    // Configure BMM150 for continuous reads (good for wizard + runtime).
    memset(&bmm_settings_, 0, sizeof(bmm_settings_));
    bmm_settings_.preset_mode = BMM150_PRESETMODE_REGULAR;
    (void)bmm150_set_presetmode(&bmm_settings_, &bmm_dev_);

#if defined(BMM150_POWERMODE_NORMAL)
    bmm_settings_.pwr_mode = BMM150_POWERMODE_NORMAL;
#elif defined(BMM150_NORMAL_MODE)
    bmm_settings_.pwr_mode = BMM150_NORMAL_MODE;
#else
    bmm_settings_.pwr_mode = 0x00; // normal in many SensorAPI versions
#endif
    (void)bmm150_set_op_mode(&bmm_settings_, &bmm_dev_);

    ok_ = true;
    return true;
  }

  // Read compensated magnetometer in micro-tesla.
  // bmm150_read_mag_data() reads 0x42..0x49 and returns compensated uT.  [oai_citation:2‡GitHub](https://github.com/arduino-libraries/Arduino_BMI270_BMM150/blob/master/src/utilities/BMM150-Sensor-API/bmm150.h)
  bool readMag_uT(Vector3f& m_uT_out) {
    if (!ok_) return false;

    struct bmm150_mag_data md{};
    const int8_t r = bmm150_read_mag_data(&md, &bmm_dev_);
    if (r != BMM150_OK) return false;

    // When BMM150_USE_FLOATING_POINT=1, md.x/y/z are floats in uT.
    const float mx = (float)md.x;
    const float my = (float)md.y;
    const float mz = (float)md.z;

    if (!isfinite(mx) || !isfinite(my) || !isfinite(mz)) return false;
    m_uT_out = Vector3f(mx, my, mz);
    return true;
  }

  // Optional: quick “is it alive?” check
  bool readChipId(uint8_t& chip_id_out) {
    if (!ok_) return false;
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
    // Fallback (many builds at least have 25Hz)
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
    // bmi2_read_aux_man_mode routes through BMI270 AUX I2C master.  [oai_citation:3‡GitHub](https://github.com/m5stack/M5Unit-IMU-Pro-Mini/blob/main/src/BMI270.cpp)
    return (BMM150_INTF_RET_TYPE)bmi2_read_aux_man_mode(reg, data, len, self->bmi_dev_);
  }

  static BMM150_INTF_RET_TYPE bmm_write_(uint8_t reg, const uint8_t* data, uint32_t len, void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_) return (BMM150_INTF_RET_TYPE)-1;
    return (BMM150_INTF_RET_TYPE)bmi2_write_aux_man_mode(reg, data, len, self->bmi_dev_);
  }

  static void bmm_delay_us_(uint32_t us, void*) { delayMicroseconds(us); }

private:
  bool ok_ = false;

  struct bmi2_dev* bmi_dev_ = nullptr;
  uint8_t bmm_addr_ = BMM150_DEFAULT_I2C_ADDRESS;

  struct bmm150_dev bmm_dev_{};
  struct bmm150_settings bmm_settings_{};
};

} // namespace atoms3r_ical
