#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

#include <Arduino_BMI270_BMM150.h>

#if defined(__has_include)
  #if __has_include(<utilities/BMI270-Sensor-API/bmi2.h>) && \
      __has_include(<utilities/BMI270-Sensor-API/bmi270.h>) && \
      __has_include(<utilities/BMM150-Sensor-API/bmm150.h>) && \
      __has_include(<utilities/BMM150-Sensor-API/bmm150_defs.h>)
    #include <utilities/BMI270-Sensor-API/bmi2.h>
    #include <utilities/BMI270-Sensor-API/bmi2_defs.h>
    #include <utilities/BMI270-Sensor-API/bmi270.h>
    #include <utilities/BMM150-Sensor-API/bmm150.h>
    #include <utilities/BMM150-Sensor-API/bmm150_defs.h>
    #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
  #elif __has_include(<utility/BMI270-Sensor-API/bmi2.h>) && \
        __has_include(<utility/BMI270-Sensor-API/bmi270.h>) && \
        __has_include(<utility/BMM150-Sensor-API/bmm150.h>) && \
        __has_include(<utility/BMM150-Sensor-API/bmm150_defs.h>)
    #include <utility/BMI270-Sensor-API/bmi2.h>
    #include <utility/BMI270-Sensor-API/bmi2_defs.h>
    #include <utility/BMI270-Sensor-API/bmi270.h>
    #include <utility/BMM150-Sensor-API/bmm150.h>
    #include <utility/BMM150-Sensor-API/bmm150_defs.h>
    #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
  #else
    #error "Arduino_BMI270_BMM150 is present, but Bosch vendor headers were not found under utility/ or utilities/."
  #endif
#else
  #error "__has_include is required for Bosch vendor header path detection."
#endif

#ifdef EIGEN_NON_ARDUINO
  #include <Eigen/Dense>
#else
  #include <ArduinoEigenDense.h>
#endif

namespace atoms3r_ical {

using Vector3f = Eigen::Vector3f;

#if ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI

class BoschBmm150Aux {
public:
  struct Config {
    uint8_t  bmm_addr = BMM150_DEFAULT_I2C_ADDRESS; // usually 0x10
    float    aux_odr_hz = 25.0f;                    // BMI270 AUX service rate
    uint8_t  preset_mode = BMM150_PRESETMODE_REGULAR;
    uint16_t startup_settle_ms = 200;
    bool     verify_first_read = true;

    // If caller polls faster than the configured mag rate:
    //  - true  => return last good finite sample
    //  - false => wait for next sample period
    bool     return_last_on_fast_poll = false;

    // On read failure, output last good sample if available, else zeros.
    // Function still returns false.
    bool     return_last_good_on_error = true;

    // Board remap:
    //   +1,+2,+3 = +X,+Y,+Z
    //   -1,-2,-3 = -X,-Y,-Z
    int8_t   axis_map[3] = { +1, +2, +3 };
  };

  enum class Error : uint8_t {
    NONE = 0,
    NOT_BUILT,

    NULL_BMI_DEV,
    NOT_INITIALIZED,

    BMI_GET_AUX_CFG_FAILED,
    BMI_ADV_POWER_SAVE_DISABLE_FAILED,
    BMI_AUX_PULLUP_CFG_FAILED,
    BMI_AUX_ENABLE_FAILED,
    BMI_SET_AUX_CFG_FAILED,

    CHIP_ID_READ_FAILED,
    CHIP_ID_MISMATCH,

    BMM_POWER_ON_FAILED,
    BMM_TRIM_READ_FAILED,
    BMM_CONFIG_FAILED,
    BMM_TEST_READ_FAILED,

    AUX_READ_FAILED,
    AUX_WRITE_FAILED,

    RAW_MAG_INVALID,
    COMP_INVALID,
    NONFINITE_MAG,
    ZERO_MAG,

    BMI_RESTORE_AUX_CFG_FAILED,
    BMI_AUX_DISABLE_FAILED
  };

  BoschBmm150Aux() = default;

  bool ok() const { return ok_; }
  bool sessionAttached() const { return session_attached_; }

  Error lastError() const { return last_error_; }
  Error lastEndError() const { return last_end_error_; }

  const char* lastErrorString() const { return errorString_(last_error_); }
  const char* lastEndErrorString() const { return errorString_(last_end_error_); }

  const Config& config() const { return cfg_; }

  uint32_t initFailuresTotal() const            { return init_fail_total_; }
  uint32_t readOkTotal() const                  { return read_ok_total_; }
  uint32_t readFailuresTotal() const            { return read_fail_total_; }
  uint32_t rollbackFailuresTotal() const        { return rollback_fail_total_; }
  uint32_t endFailuresTotal() const             { return end_fail_total_; }
  uint32_t possibleDuplicateReadsTotal() const  { return possible_duplicate_total_; }
  uint32_t lastReadMillis() const               { return last_read_ms_; }

  const Vector3f& lastGoodMag_uT() const { return last_good_uT_; }
  bool haveLastGoodMag() const { return have_last_good_; }

  bool begin(struct bmi2_dev* bmi_dev) {
    return begin(bmi_dev, Config{});
  }

  bool begin(struct bmi2_dev* bmi_dev, const Config& cfg) {
    if (session_attached_) {
      if (!end()) {
        ++init_fail_total_;
        return false;
      }
    }

    clearSessionData_();
    last_error_ = Error::NONE;
    last_end_error_ = Error::NONE;

    cfg_ = cfg;
    sanitizeAxisMap_();

    bmi_dev_ = bmi_dev;
    session_attached_ = (bmi_dev_ != nullptr);

    if (!bmi_dev_) {
      ++init_fail_total_;
      last_error_ = Error::NULL_BMI_DEV;
      clearSessionData_();
      return false;
    }

    saved_aux_cfg_.type = BMI2_AUX;
    if (bmi270_get_sensor_config(&saved_aux_cfg_, 1, bmi_dev_) != BMI2_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMI_GET_AUX_CFG_FAILED;
      clearSessionData_();
      return false;
    }
    saved_aux_cfg_valid_ = true;
    saved_aux_was_enabled_ = (saved_aux_cfg_.cfg.aux.aux_en == BMI2_ENABLE);

    if (bmi2_set_adv_power_save(BMI2_DISABLE, bmi_dev_) != BMI2_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMI_ADV_POWER_SAVE_DISABLE_FAILED;
      bestEffortRollback_();
      return false;
    }
    bmi_dev_->aps_status = BMI2_DISABLE;

    if (!setAuxPullup2k_()) {
      ++init_fail_total_;
      last_error_ = Error::BMI_AUX_PULLUP_CFG_FAILED;
      bestEffortRollback_();
      return false;
    }

    {
      uint8_t sens = BMI2_AUX;
      if (bmi2_sensor_enable(&sens, 1, bmi_dev_) != BMI2_OK) {
        ++init_fail_total_;
        last_error_ = Error::BMI_AUX_ENABLE_FAILED;
        bestEffortRollback_();
        return false;
      }
    }

    bmi2_sens_config sc = saved_aux_cfg_;
    sc.type = BMI2_AUX;
    sc.cfg.aux.odr             = auxOdrFromHz_(cfg_.aux_odr_hz);
    sc.cfg.aux.aux_en          = BMI2_ENABLE;
    sc.cfg.aux.i2c_device_addr = cfg_.bmm_addr;
    sc.cfg.aux.fcu_write_en    = BMI2_ENABLE;
    sc.cfg.aux.man_rd_burst    = manualBurstLen_();
    sc.cfg.aux.read_addr       = kRegDataXLSB;
    sc.cfg.aux.manual_en       = BMI2_ENABLE;

    if (bmi270_set_sensor_config(&sc, 1, bmi_dev_) != BMI2_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMI_SET_AUX_CFG_FAILED;
      bestEffortRollback_();
      return false;
    }

    uint8_t chip_id = 0;
    if (!readChipIdRaw_(chip_id)) {
      ++init_fail_total_;
      last_error_ = Error::CHIP_ID_READ_FAILED;
      bestEffortRollback_();
      return false;
    }
    if (chip_id != kExpectedChipId) {
      ++init_fail_total_;
      last_error_ = Error::CHIP_ID_MISMATCH;
      bestEffortRollback_();
      return false;
    }

    if (!powerOn_()) {
      ++init_fail_total_;
      last_error_ = Error::BMM_POWER_ON_FAILED;
      bestEffortRollback_();
      return false;
    }

    if (!readTrimData_()) {
      ++init_fail_total_;
      last_error_ = Error::BMM_TRIM_READ_FAILED;
      bestEffortRollback_();
      return false;
    }

    if (!configurePresetAndMode_()) {
      ++init_fail_total_;
      last_error_ = Error::BMM_CONFIG_FAILED;
      bestEffortRollback_();
      return false;
    }

    if (cfg_.startup_settle_ms > 0u) {
      delay(cfg_.startup_settle_ms);
    }

    // Discard a couple of initial samples.
    {
      RawSample raw{};
      (void)readRawSample_(raw);
      delay(minReadIntervalMs_());
      (void)readRawSample_(raw);
    }

    if (cfg_.verify_first_read) {
      Vector3f tmp = Vector3f::Zero();
      bool got_valid = false;
      const uint32_t wait_ms = minReadIntervalMs_();

      for (int i = 0; i < 6; ++i) {
        if (i > 0) {
          delay(wait_ms);
        }
        if (readMagInternal_(tmp, false)) {
          got_valid = true;
          break;
        }
      }

      if (!got_valid) {
        ++init_fail_total_;
        last_error_ = Error::BMM_TEST_READ_FAILED;
        bestEffortRollback_();
        return false;
      }
    }

    ok_ = true;
    last_error_ = Error::NONE;
    return true;
  }

  bool end() {
    if (!session_attached_ || !bmi_dev_) {
      clearSessionData_();
      last_error_ = Error::NONE;
      last_end_error_ = Error::NONE;
      return true;
    }

    bool restore_ok = true;
    Error first_restore_error = Error::NONE;

    if (saved_aux_cfg_valid_) {
      if (bmi270_set_sensor_config(&saved_aux_cfg_, 1, bmi_dev_) != BMI2_OK) {
        restore_ok = false;
        first_restore_error = Error::BMI_RESTORE_AUX_CFG_FAILED;
      } else if (!saved_aux_was_enabled_) {
        uint8_t sens = BMI2_AUX;
        if (bmi2_sensor_disable(&sens, 1, bmi_dev_) != BMI2_OK) {
          restore_ok = false;
          first_restore_error = Error::BMI_AUX_DISABLE_FAILED;
        }
      }
    }

    if (!restore_ok) {
      ++end_fail_total_;
      ok_ = false;
      last_error_ = first_restore_error;
      last_end_error_ = first_restore_error;
      return false;
    }

    clearSessionData_();
    last_error_ = Error::NONE;
    last_end_error_ = Error::NONE;
    return true;
  }

  bool readMag_uT(Vector3f& m_uT_out) {
    if (!ok_) {
      safeFailureOutput_(m_uT_out);
      last_error_ = Error::NOT_INITIALIZED;
      ++read_fail_total_;
      return false;
    }
    return readMagInternal_(m_uT_out, true);
  }

  bool readChipId(uint8_t& chip_id_out) {
    if (!bmi_dev_) {
      last_error_ = Error::NOT_INITIALIZED;
      return false;
    }
    return readChipIdRaw_(chip_id_out);
  }

private:
  struct TrimData {
    int8_t   dig_x1   = 0;
    int8_t   dig_y1   = 0;
    int8_t   dig_x2   = 0;
    int8_t   dig_y2   = 0;
    uint16_t dig_z1   = 0;
    int16_t  dig_z2   = 0;
    int16_t  dig_z3   = 0;
    int16_t  dig_z4   = 0;
    uint8_t  dig_xy1  = 0;
    int8_t   dig_xy2  = 0;
    uint16_t dig_xyz1 = 0;
    bool     valid    = false;
  };

  struct RawSample {
    int16_t  x = 0;
    int16_t  y = 0;
    int16_t  z = 0;
    uint16_t rhall = 0;
  };

  static constexpr uint8_t kExpectedChipId = 0x32u;

  static constexpr uint8_t kRegChipId       = 0x40u;
  static constexpr uint8_t kRegDataXLSB     = 0x42u;
  static constexpr uint8_t kRegPowerCtrl    = 0x4Bu;
  static constexpr uint8_t kRegOpCtrl       = 0x4Cu;
  static constexpr uint8_t kRegRepXY        = 0x51u;
  static constexpr uint8_t kRegRepZ         = 0x52u;

  static constexpr uint8_t kRegDigX1        = 0x5Du;
  static constexpr uint8_t kRegDigY1        = 0x5Eu;
  static constexpr uint8_t kRegDigZ4Lsb     = 0x62u;
  static constexpr uint8_t kRegDigZ2Lsb     = 0x68u;
  static constexpr uint8_t kRegDigZ1Lsb     = 0x6Au;
  static constexpr uint8_t kRegDigXYZ1Lsb   = 0x6Cu;
  static constexpr uint8_t kRegDigZ3Lsb     = 0x6Eu;
  static constexpr uint8_t kRegDigXY2       = 0x70u;
  static constexpr uint8_t kRegDigXY1       = 0x71u;
  static constexpr uint8_t kRegDigX2        = 0x64u;
  static constexpr uint8_t kRegDigY2        = 0x65u;

  static constexpr uint8_t kPowerOnBit      = 0x01u;
  static constexpr uint8_t kOpModeNormal    = 0x00u;
  static constexpr uint8_t kOpModeBitsMask  = 0x06u;
  static constexpr uint8_t kOdrBitsMask     = 0x38u;

  static const char* errorString_(Error e) {
    switch (e) {
      case Error::NONE:                              return "none";
      case Error::NOT_BUILT:                         return "Bosch SensorAPI not available in this build";
      case Error::NULL_BMI_DEV:                      return "null bmi2_dev";
      case Error::NOT_INITIALIZED:                   return "not initialized";
      case Error::BMI_GET_AUX_CFG_FAILED:            return "bmi270_get_sensor_config(BMI2_AUX) failed";
      case Error::BMI_ADV_POWER_SAVE_DISABLE_FAILED: return "bmi2_set_adv_power_save(DISABLE) failed";
      case Error::BMI_AUX_PULLUP_CFG_FAILED:         return "failed to configure BMI270 AUX pull-up trim";
      case Error::BMI_AUX_ENABLE_FAILED:             return "bmi2_sensor_enable(BMI2_AUX) failed";
      case Error::BMI_SET_AUX_CFG_FAILED:            return "bmi270_set_sensor_config(BMI2_AUX) failed";
      case Error::CHIP_ID_READ_FAILED:               return "BMM150 chip-id read failed";
      case Error::CHIP_ID_MISMATCH:                  return "BMM150 chip-id mismatch";
      case Error::BMM_POWER_ON_FAILED:               return "BMM150 power-on failed";
      case Error::BMM_TRIM_READ_FAILED:              return "BMM150 trim read failed";
      case Error::BMM_CONFIG_FAILED:                 return "BMM150 config failed";
      case Error::BMM_TEST_READ_FAILED:              return "initial BMM150 read failed";
      case Error::AUX_READ_FAILED:                   return "BMI270 AUX read failed";
      case Error::AUX_WRITE_FAILED:                  return "BMI270 AUX write failed";
      case Error::RAW_MAG_INVALID:                   return "raw BMM150 sample invalid";
      case Error::COMP_INVALID:                      return "BMM150 compensated output invalid";
      case Error::NONFINITE_MAG:                     return "non-finite magnetometer output";
      case Error::ZERO_MAG:                          return "all-zero magnetometer sample";
      case Error::BMI_RESTORE_AUX_CFG_FAILED:        return "failed to restore prior BMI270 AUX config";
      case Error::BMI_AUX_DISABLE_FAILED:            return "failed to disable BMI270 AUX after restore";
      default:                                       return "unknown";
    }
  }

  static uint8_t manualBurstLen_() {
    #if defined(BMI2_AUX_READ_LEN_3)
      return BMI2_AUX_READ_LEN_3;
    #elif defined(BMI2_AUX_READ_LEN_2)
      return BMI2_AUX_READ_LEN_2;
    #elif defined(BMI2_AUX_READ_LEN_1)
      return BMI2_AUX_READ_LEN_1;
    #else
      return BMI2_AUX_READ_LEN_0;
    #endif
  }

  static uint8_t auxOdrFromHz_(float hz) {
    if (!(hz > 0.0f) || !std::isfinite(hz)) {
      hz = 25.0f;
    }

    #if defined(BMI2_AUX_ODR_200HZ)
      if (hz >= 150.0f) return BMI2_AUX_ODR_200HZ;
    #endif
    #if defined(BMI2_AUX_ODR_100HZ)
      if (hz >= 75.0f)  return BMI2_AUX_ODR_100HZ;
    #endif
    #if defined(BMI2_AUX_ODR_50HZ)
      if (hz >= 37.0f)  return BMI2_AUX_ODR_50HZ;
    #endif
    #if defined(BMI2_AUX_ODR_25HZ)
      if (hz >= 18.0f)  return BMI2_AUX_ODR_25HZ;
    #endif
    #if defined(BMI2_AUX_ODR_12_5HZ)
      if (hz >= 9.0f)   return BMI2_AUX_ODR_12_5HZ;
    #endif
    #if defined(BMI2_AUX_ODR_6_25HZ)
      if (hz >= 4.5f)   return BMI2_AUX_ODR_6_25HZ;
    #endif
    #if defined(BMI2_AUX_ODR_3_12HZ)
      if (hz >= 2.2f)   return BMI2_AUX_ODR_3_12HZ;
    #endif
    #if defined(BMI2_AUX_ODR_1_56HZ)
      if (hz >= 1.0f)   return BMI2_AUX_ODR_1_56HZ;
    #endif
    #if defined(BMI2_AUX_ODR_0_78HZ)
      return BMI2_AUX_ODR_0_78HZ;
    #elif defined(BMI2_AUX_ODR_25HZ)
      return BMI2_AUX_ODR_25HZ;
    #elif defined(BMI2_AUX_ODR_100HZ)
      return BMI2_AUX_ODR_100HZ;
    #else
      return 0;
    #endif
  }

  static uint8_t presetRepXY_(uint8_t preset_mode) {
    switch (preset_mode) {
      case BMM150_PRESETMODE_LOWPOWER:      return 0x01u;
      case BMM150_PRESETMODE_REGULAR:       return 0x04u;
      case BMM150_PRESETMODE_ENHANCED:      return 0x07u;
      case BMM150_PRESETMODE_HIGHACCURACY:  return 0x17u;
      default:                              return 0x04u;
    }
  }

  static uint8_t presetRepZ_(uint8_t preset_mode) {
    switch (preset_mode) {
      case BMM150_PRESETMODE_LOWPOWER:      return 0x02u;
      case BMM150_PRESETMODE_REGULAR:       return 0x0Eu;
      case BMM150_PRESETMODE_ENHANCED:      return 0x1Au;
      case BMM150_PRESETMODE_HIGHACCURACY:  return 0x52u;
      default:                              return 0x0Eu;
    }
  }

  static uint8_t presetOdrCode_(uint8_t preset_mode) {
    switch (preset_mode) {
      case BMM150_PRESETMODE_HIGHACCURACY:  return 0x05u; // 20 Hz
      case BMM150_PRESETMODE_LOWPOWER:
      case BMM150_PRESETMODE_REGULAR:
      case BMM150_PRESETMODE_ENHANCED:
      default:                              return 0x00u; // 10 Hz
    }
  }

  static float presetModeHz_(uint8_t preset_mode) {
    switch (preset_mode) {
      case BMM150_PRESETMODE_HIGHACCURACY:  return 20.0f;
      case BMM150_PRESETMODE_LOWPOWER:
      case BMM150_PRESETMODE_REGULAR:
      case BMM150_PRESETMODE_ENHANCED:
      default:                              return 10.0f;
    }
  }

  uint32_t minReadIntervalMs_() const {
    const float hz = (effective_mag_hz_ > 0.0f && std::isfinite(effective_mag_hz_))
                   ? effective_mag_hz_
                   : 10.0f;
    const float ms = 1000.0f / hz;
    const long out = lroundf(ms);
    return out > 0 ? static_cast<uint32_t>(out) : 100u;
  }

  bool setAuxPullup2k_() {
  #if defined(BMI2_AUX_IF_TRIM) && defined(BMI2_ASDA_PUPSEL_2K)
    uint8_t regdata = BMI2_ASDA_PUPSEL_2K;
    return bmi2_set_regs(BMI2_AUX_IF_TRIM, &regdata, 1, bmi_dev_) == BMI2_OK;
  #else
    return true;
  #endif
  }

  static bool finite3_(float x, float y, float z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  static bool allZero3_(float x, float y, float z) {
    return x == 0.0f && y == 0.0f && z == 0.0f;
  }

  static bool allZero3_(const Vector3f& v) {
    return allZero3_(v.x(), v.y(), v.z());
  }

  static float pickAxis_(int8_t code, const Vector3f& v) {
    switch (code) {
      case +1: return  v.x();
      case -1: return -v.x();
      case +2: return  v.y();
      case -2: return -v.y();
      case +3: return  v.z();
      case -3: return -v.z();
      default: return 0.0f;
    }
  }

  void sanitizeAxisMap_() {
    for (int i = 0; i < 3; ++i) {
      const int8_t c = cfg_.axis_map[i];
      const bool ok = (c == +1 || c == -1 || c == +2 || c == -2 || c == +3 || c == -3);
      if (!ok) {
        cfg_.axis_map[0] = +1;
        cfg_.axis_map[1] = +2;
        cfg_.axis_map[2] = +3;
        return;
      }
    }
  }

  Vector3f applyAxisMap_(const Vector3f& raw) const {
    return Vector3f(
      pickAxis_(cfg_.axis_map[0], raw),
      pickAxis_(cfg_.axis_map[1], raw),
      pickAxis_(cfg_.axis_map[2], raw)
    );
  }

  void safeFailureOutput_(Vector3f& out) const {
    if (cfg_.return_last_good_on_error && have_last_good_) {
      out = last_good_uT_;
    } else {
      out = Vector3f::Zero();
    }
  }

  bool auxRead_(uint8_t reg, uint8_t* data, uint32_t len) {
    if (!bmi_dev_ || !data || len == 0u) {
      last_error_ = Error::AUX_READ_FAILED;
      return false;
    }
    const int8_t r = bmi2_read_aux_man_mode(reg, data, len, bmi_dev_);
    if (r != BMI2_OK) {
      last_error_ = Error::AUX_READ_FAILED;
      return false;
    }
    return true;
  }

  bool auxWrite_(uint8_t reg, uint8_t value) {
    if (!bmi_dev_) {
      last_error_ = Error::AUX_WRITE_FAILED;
      return false;
    }
    const int8_t r = bmi2_write_aux_man_mode(reg, &value, 1, bmi_dev_);
    if (r != BMI2_OK) {
      last_error_ = Error::AUX_WRITE_FAILED;
      return false;
    }
    return true;
  }

  bool readChipIdRaw_(uint8_t& chip_id_out) {
    uint8_t id = 0;
    if (!auxRead_(kRegChipId, &id, 1)) {
      return false;
    }
    chip_id_out = id;
    return true;
  }

  bool powerOn_() {
    uint8_t pwr = 0;
    if (!auxRead_(kRegPowerCtrl, &pwr, 1)) {
      return false;
    }
    pwr |= kPowerOnBit;
    if (!auxWrite_(kRegPowerCtrl, pwr)) {
      return false;
    }
    delay(10);
    return true;
  }

  bool configurePresetAndMode_() {
    if (!auxWrite_(kRegRepXY, presetRepXY_(cfg_.preset_mode))) {
      return false;
    }
    if (!auxWrite_(kRegRepZ, presetRepZ_(cfg_.preset_mode))) {
      return false;
    }

    uint8_t op = 0;
    if (!auxRead_(kRegOpCtrl, &op, 1)) {
      return false;
    }

    op &= static_cast<uint8_t>(~(kOpModeBitsMask | kOdrBitsMask));
    op |= static_cast<uint8_t>((presetOdrCode_(cfg_.preset_mode) << 3) & kOdrBitsMask);
    op |= kOpModeNormal;

    if (!auxWrite_(kRegOpCtrl, op)) {
      return false;
    }

    delay(5);
    effective_mag_hz_ = presetModeHz_(cfg_.preset_mode);
    return true;
  }

  static int16_t s16le_(const uint8_t* p) {
    return static_cast<int16_t>(
      static_cast<uint16_t>(p[0]) |
      (static_cast<uint16_t>(p[1]) << 8)
    );
  }

  static uint16_t u16le_(const uint8_t* p) {
    return static_cast<uint16_t>(
      static_cast<uint16_t>(p[0]) |
      (static_cast<uint16_t>(p[1]) << 8)
    );
  }

  bool readTrimData_() {
    uint8_t v = 0;
    uint8_t b[2] = {0, 0};

    if (!auxRead_(kRegDigX1, &v, 1)) return false;
    trim_.dig_x1 = static_cast<int8_t>(v);

    if (!auxRead_(kRegDigY1, &v, 1)) return false;
    trim_.dig_y1 = static_cast<int8_t>(v);

    if (!auxRead_(kRegDigX2, &v, 1)) return false;
    trim_.dig_x2 = static_cast<int8_t>(v);

    if (!auxRead_(kRegDigY2, &v, 1)) return false;
    trim_.dig_y2 = static_cast<int8_t>(v);

    if (!auxRead_(kRegDigZ4Lsb, b, 2)) return false;
    trim_.dig_z4 = s16le_(b);

    if (!auxRead_(kRegDigZ2Lsb, b, 2)) return false;
    trim_.dig_z2 = s16le_(b);

    if (!auxRead_(kRegDigZ1Lsb, b, 2)) return false;
    trim_.dig_z1 = u16le_(b);

    if (!auxRead_(kRegDigXYZ1Lsb, b, 2)) return false;
    trim_.dig_xyz1 = u16le_(b);

    if (!auxRead_(kRegDigZ3Lsb, b, 2)) return false;
    trim_.dig_z3 = s16le_(b);

    if (!auxRead_(kRegDigXY2, &v, 1)) return false;
    trim_.dig_xy2 = static_cast<int8_t>(v);

    if (!auxRead_(kRegDigXY1, &v, 1)) return false;
    trim_.dig_xy1 = v;

    trim_.valid = (trim_.dig_xyz1 != 0u) && (trim_.dig_z1 != 0u);
    return trim_.valid;
  }

  static int16_t signExtend13_(uint16_t v) {
    if (v & 0x1000u) {
      v |= 0xE000u;
    }
    return static_cast<int16_t>(v);
  }

  static int16_t signExtend15_(uint16_t v) {
    if (v & 0x4000u) {
      v |= 0x8000u;
    }
    return static_cast<int16_t>(v);
  }

  bool readRawSample_(RawSample& s) {
    uint8_t buf[8] = {};
    if (!auxRead_(kRegDataXLSB, buf, 8)) {
      return false;
    }

    const uint16_t x_u = static_cast<uint16_t>(
      (static_cast<uint16_t>(buf[1]) << 5) |
      (static_cast<uint16_t>(buf[0]) >> 3)
    );
    const uint16_t y_u = static_cast<uint16_t>(
      (static_cast<uint16_t>(buf[3]) << 5) |
      (static_cast<uint16_t>(buf[2]) >> 3)
    );
    const uint16_t z_u = static_cast<uint16_t>(
      (static_cast<uint16_t>(buf[5]) << 7) |
      (static_cast<uint16_t>(buf[4]) >> 1)
    );
    const uint16_t rh_u = static_cast<uint16_t>(
      (static_cast<uint16_t>(buf[7]) << 6) |
      (static_cast<uint16_t>(buf[6]) >> 2)
    );

    s.x = signExtend13_(x_u);
    s.y = signExtend13_(y_u);
    s.z = signExtend15_(z_u);
    s.rhall = rh_u;

    // Bosch overflow sentinels / invalid hall.
    if (s.rhall == 0u || s.rhall == 0x3FFFu || s.x == -4096 || s.y == -4096 || s.z == -16384) {
      last_error_ = Error::RAW_MAG_INVALID;
      return false;
    }

    // Reject all-zero raw sample too.
    if (s.x == 0 && s.y == 0 && s.z == 0) {
      last_error_ = Error::RAW_MAG_INVALID;
      return false;
    }

    return true;
  }

  bool compensateRawTo_uT_(const RawSample& s, Vector3f& out_uT) {
    if (!trim_.valid || trim_.dig_xyz1 == 0u || trim_.dig_z1 == 0u || s.rhall == 0u) {
      last_error_ = Error::COMP_INVALID;
      return false;
    }

    const float rhall = static_cast<float>(s.rhall);
    const float dig_xyz1 = static_cast<float>(trim_.dig_xyz1);

    // X compensation (Bosch formula, guarded)
    const float x0 = ((dig_xyz1 * 16384.0f) / rhall) - 16384.0f;
    const float x1 = (static_cast<float>(trim_.dig_xy2) * (x0 * x0 / 268435456.0f));
    const float x2 = x0 * (static_cast<float>(trim_.dig_xy1) / 16384.0f);
    const float x3 = static_cast<float>(trim_.dig_x2) + 160.0f;
    const float x = ((((static_cast<float>(s.x) * ((x1 + x2 + 256.0f) * x3)) / 8192.0f) +
                     (static_cast<float>(trim_.dig_x1) * 8.0f)) / 16.0f);

    // Y compensation
    const float y0 = ((dig_xyz1 * 16384.0f) / rhall) - 16384.0f;
    const float y1 = (static_cast<float>(trim_.dig_xy2) * (y0 * y0 / 268435456.0f));
    const float y2 = y0 * (static_cast<float>(trim_.dig_xy1) / 16384.0f);
    const float y3 = static_cast<float>(trim_.dig_y2) + 160.0f;
    const float y = ((((static_cast<float>(s.y) * ((y1 + y2 + 256.0f) * y3)) / 8192.0f) +
                     (static_cast<float>(trim_.dig_y1) * 8.0f)) / 16.0f);

    // Z compensation
    const float z_denom = ((static_cast<float>(trim_.dig_z2)) +
                          ((static_cast<float>(trim_.dig_z1) * rhall) / 32768.0f)) * 4.0f;
    if (z_denom == 0.0f || !std::isfinite(z_denom)) {
      last_error_ = Error::COMP_INVALID;
      return false;
    }

    const float z_num =
      ((static_cast<float>(s.z) - static_cast<float>(trim_.dig_z4)) * 131072.0f) -
      (static_cast<float>(trim_.dig_z3) * (rhall - dig_xyz1));
    const float z = (z_num / z_denom) / 16.0f;

    if (!finite3_(x, y, z)) {
      last_error_ = Error::NONFINITE_MAG;
      return false;
    }

    Vector3f raw_uT(x, y, z);
    raw_uT = applyAxisMap_(raw_uT);

    if (!finite3_(raw_uT.x(), raw_uT.y(), raw_uT.z())) {
      last_error_ = Error::NONFINITE_MAG;
      return false;
    }

    if (allZero3_(raw_uT)) {
      last_error_ = Error::ZERO_MAG;
      return false;
    }

    out_uT = raw_uT;
    return true;
  }

  bool readMagInternal_(Vector3f& m_uT_out, bool count_stats) {
    uint32_t now_ms = millis();
    const uint32_t min_ms = minReadIntervalMs_();

    if (have_last_good_) {
      const uint32_t elapsed = static_cast<uint32_t>(now_ms - last_read_ms_);
      if (elapsed < min_ms) {
        if (cfg_.return_last_on_fast_poll) {
          m_uT_out = last_good_uT_;
          ++possible_duplicate_total_;
          last_error_ = Error::NONE;
          if (count_stats) {
            ++read_ok_total_;
          }
          return true;
        }

        delay(min_ms - elapsed);
        now_ms = millis();
      }
    }

    RawSample raw{};
    Vector3f  out = Vector3f::Zero();

    bool success = false;

    // Try a few times before giving up.
    for (int attempt = 0; attempt < 4; ++attempt) {
      if (attempt == 1) delay(2);
      if (attempt == 2) delay(5);
      if (attempt == 3) delay(min_ms);

      if (!readRawSample_(raw)) {
        continue;
      }
      if (!compensateRawTo_uT_(raw, out)) {
        continue;
      }
      success = true;
      break;
    }

    if (!success) {
      safeFailureOutput_(m_uT_out);
      if (count_stats) {
        ++read_fail_total_;
      }
      return false;
    }

    if (have_last_good_ &&
        !cfg_.return_last_on_fast_poll &&
        out.x() == last_good_uT_.x() &&
        out.y() == last_good_uT_.y() &&
        out.z() == last_good_uT_.z()) {
      ++possible_duplicate_total_;
    }

    m_uT_out = out;
    last_good_uT_ = out;
    have_last_good_ = true;
    last_read_ms_ = now_ms;
    last_error_ = Error::NONE;

    if (count_stats) {
      ++read_ok_total_;
    }

    return true;
  }

  void clearSessionData_() {
    ok_ = false;
    session_attached_ = false;
    cfg_ = Config{};
    bmi_dev_ = nullptr;

    saved_aux_cfg_valid_ = false;
    saved_aux_was_enabled_ = false;
    std::memset(&saved_aux_cfg_, 0, sizeof(saved_aux_cfg_));

    trim_ = TrimData{};

    have_last_good_ = false;
    last_good_uT_ = Vector3f::Zero();
    last_read_ms_ = 0;
    effective_mag_hz_ = 10.0f;
  }

  void bestEffortRollback_() {
    const Error preserved_error = last_error_;
    bool restore_ok = true;

    if (bmi_dev_ && saved_aux_cfg_valid_) {
      if (bmi270_set_sensor_config(&saved_aux_cfg_, 1, bmi_dev_) != BMI2_OK) {
        restore_ok = false;
      } else if (!saved_aux_was_enabled_) {
        uint8_t sens = BMI2_AUX;
        if (bmi2_sensor_disable(&sens, 1, bmi_dev_) != BMI2_OK) {
          restore_ok = false;
        }
      }
    }

    if (!restore_ok) {
      ++rollback_fail_total_;
    }

    clearSessionData_();
    last_error_ = preserved_error;
  }

private:
  bool  ok_ = false;
  bool  session_attached_ = false;
  Error last_error_ = Error::NONE;
  Error last_end_error_ = Error::NONE;

  Config cfg_{};
  struct bmi2_dev* bmi_dev_ = nullptr;

  struct bmi2_sens_config saved_aux_cfg_{};
  bool saved_aux_cfg_valid_ = false;
  bool saved_aux_was_enabled_ = false;

  TrimData trim_{};

  uint32_t init_fail_total_ = 0;
  uint32_t read_ok_total_ = 0;
  uint32_t read_fail_total_ = 0;
  uint32_t rollback_fail_total_ = 0;
  uint32_t end_fail_total_ = 0;
  uint32_t possible_duplicate_total_ = 0;

  bool     have_last_good_ = false;
  Vector3f last_good_uT_ = Vector3f::Zero();
  uint32_t last_read_ms_ = 0;
  float    effective_mag_hz_ = 10.0f;
};

#else

struct bmi2_dev;

class BoschBmm150Aux {
  template <typename> struct always_false : std::false_type {};

public:
  struct Config {
    uint8_t  bmm_addr = 0x10;
    float    aux_odr_hz = 25.0f;
    uint8_t  preset_mode = 0;
    uint16_t startup_settle_ms = 200;
    bool     verify_first_read = true;
    bool     return_last_on_fast_poll = false;
    bool     return_last_good_on_error = true;
    int8_t   axis_map[3] = { +1, +2, +3 };
  };

  enum class Error : uint8_t {
    NONE = 0,
    NOT_BUILT
  };

  BoschBmm150Aux() = default;

  bool ok() const { return false; }
  bool sessionAttached() const { return false; }

  Error lastError() const { return Error::NOT_BUILT; }
  Error lastEndError() const { return Error::NOT_BUILT; }

  const char* lastErrorString() const {
    return "Bosch SensorAPI headers not found in this build";
  }

  const char* lastEndErrorString() const {
    return "Bosch SensorAPI headers not found in this build";
  }

  uint32_t initFailuresTotal() const { return 0; }
  uint32_t readOkTotal() const { return 0; }
  uint32_t readFailuresTotal() const { return 0; }
  uint32_t rollbackFailuresTotal() const { return 0; }
  uint32_t endFailuresTotal() const { return 0; }
  uint32_t possibleDuplicateReadsTotal() const { return 0; }
  uint32_t lastReadMillis() const { return 0; }

  const Vector3f& lastGoodMag_uT() const {
    static const Vector3f kZero = Vector3f::Zero();
    return kZero;
  }

  bool haveLastGoodMag() const { return false; }

  const Config& config() const {
    static const Config kCfg{};
    return kCfg;
  }

  template <typename Dummy = void>
  bool begin(struct bmi2_dev*) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, or exclude BoschBmm150Aux from this build.");
    return false;
  }

  template <typename Dummy = void>
  bool begin(struct bmi2_dev*, const Config&) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, or exclude BoschBmm150Aux from this build.");
    return false;
  }

  bool end() { return false; }

  template <typename Dummy = void>
  bool readMag_uT(Vector3f&) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux unavailable in this build.");
    return false;
  }

  template <typename Dummy = void>
  bool readChipId(uint8_t&) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux unavailable in this build.");
    return false;
  }
};

#endif

} // namespace atoms3r_ical
