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
      __has_include(<utilities/BMM150-Sensor-API/bmm150.h>)
    #include <utilities/BMI270-Sensor-API/bmi2.h>
    #include <utilities/BMI270-Sensor-API/bmi2_defs.h>
    #include <utilities/BMI270-Sensor-API/bmi270.h>
    #include <utilities/BMM150-Sensor-API/bmm150.h>
    #include <utilities/BMM150-Sensor-API/bmm150_defs.h>
    #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
  #elif __has_include(<utility/BMI270-Sensor-API/bmi2.h>) && \
        __has_include(<utility/BMI270-Sensor-API/bmi270.h>) && \
        __has_include(<utility/BMM150-Sensor-API/bmm150.h>)
    #include <utility/BMI270-Sensor-API/bmi2.h>
    #include <utility/BMI270-Sensor-API/bmi2_defs.h>
    #include <utility/BMI270-Sensor-API/bmi270.h>
    #include <utility/BMM150-Sensor-API/bmm150.h>
    #include <utility/BMM150-Sensor-API/bmm150_defs.h>
    #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
  #else
    #error "Arduino_BMI270_BMM150 is present, but BMM150 vendor headers were not found under utility/ or utilities/."
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

    // BMI270 AUX bridge service rate.
    float    aux_odr_hz = 100.0f;

    uint8_t  preset_mode = BMM150_PRESETMODE_REGULAR;

    // Regular/low-power/enhanced are 10 Hz; high-accuracy is 20 Hz.
    // 120 ms guarantees at least one fresh frame for 10 Hz modes.
    uint16_t startup_settle_ms = 120;

    bool     verify_first_read = true;

    // If true and caller polls faster than sensor cadence, return last good sample.
    // If false, readMag_uT() waits until the next expected fresh sample slot.
    bool     return_last_on_fast_poll = false;

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

    BMM_INIT_FAILED,
    BMM_SET_PRESET_FAILED,
    BMM_SET_OPMODE_FAILED,
    BMM_TEST_READ_FAILED,

    MAG_READ_FAILED,
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
    } else {
      resetSessionState_();
    }

    cfg_ = cfg;
    bmi_dev_ = bmi_dev;
    session_attached_ = (bmi_dev_ != nullptr);
    last_error_ = Error::NONE;
    last_end_error_ = Error::NONE;

    std::memset(&saved_aux_cfg_, 0, sizeof(saved_aux_cfg_));
    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));
    saved_aux_cfg_valid_ = false;
    saved_aux_was_enabled_ = false;
    ok_ = false;
    have_last_good_ = false;
    last_good_uT_ = Vector3f::Zero();
    last_read_ms_ = 0;
    effective_mag_hz_ = 10.0f;

    if (!bmi_dev_) {
      ++init_fail_total_;
      last_error_ = Error::NULL_BMI_DEV;
      detachSession_();
      return false;
    }

    saved_aux_cfg_.type = BMI2_AUX;
    if (bmi270_get_sensor_config(&saved_aux_cfg_, 1, bmi_dev_) != BMI2_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMI_GET_AUX_CFG_FAILED;
      detachSession_();
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
      const uint8_t sens = BMI2_AUX;
      if (bmi2_sensor_enable(&sens, 1, bmi_dev_) != BMI2_OK) {
        ++init_fail_total_;
        last_error_ = Error::BMI_AUX_ENABLE_FAILED;
        bestEffortRollback_();
        return false;
      }
    }

    // Preserve existing AUX config and override only what we own.
    bmi2_sens_config sc = saved_aux_cfg_;
    sc.type = BMI2_AUX;
    sc.cfg.aux.odr             = auxOdrFromHz_(cfg_.aux_odr_hz);
    sc.cfg.aux.aux_en          = BMI2_ENABLE;
    sc.cfg.aux.i2c_device_addr = cfg_.bmm_addr;
    sc.cfg.aux.fcu_write_en    = BMI2_ENABLE;
    sc.cfg.aux.man_rd_burst    = manualBurstLen_();
    sc.cfg.aux.read_addr       = dataStartReg_();
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
    if (chip_id != expectedChipId_()) {
      ++init_fail_total_;
      last_error_ = Error::CHIP_ID_MISMATCH;
      bestEffortRollback_();
      return false;
    }

    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf     = BMM150_I2C_INTF;
    bmm_dev_.intf_ptr = this;
    bmm_dev_.read     = &BoschBmm150Aux::bmm_read_;
    bmm_dev_.write    = &BoschBmm150Aux::bmm_write_;
    bmm_dev_.delay_us = &BoschBmm150Aux::bmm_delay_us_;

    if (bmm150_init(&bmm_dev_) != BMM150_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMM_INIT_FAILED;
      bestEffortRollback_();
      return false;
    }

    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));
    bmm_settings_.preset_mode = cfg_.preset_mode;

    if (bmm150_set_presetmode(&bmm_settings_, &bmm_dev_) != BMM150_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMM_SET_PRESET_FAILED;
      bestEffortRollback_();
      return false;
    }

    bmm_settings_.pwr_mode = normalModeValue_();
    if (bmm150_set_op_mode(&bmm_settings_, &bmm_dev_) != BMM150_OK) {
      ++init_fail_total_;
      last_error_ = Error::BMM_SET_OPMODE_FAILED;
      bestEffortRollback_();
      return false;
    }

#if defined(BMM150_SEL_DRDY_PIN_EN)
    bmm_settings_.int_settings.drdy_pin_en = 0x01;
    (void)bmm150_set_sensor_settings(BMM150_SEL_DRDY_PIN_EN, &bmm_settings_, &bmm_dev_);
#endif

    effective_mag_hz_ = presetModeHz_(cfg_.preset_mode);

    if (cfg_.startup_settle_ms > 0u) {
      delay(cfg_.startup_settle_ms);
    }

    if (cfg_.verify_first_read) {
      Vector3f tmp = Vector3f::Zero();
      bool got_valid = false;
      const uint32_t wait_ms = minReadIntervalMs_();

      for (int attempt = 0; attempt < 4; ++attempt) {
        if (readMagInternal_(tmp, false)) {
          got_valid = true;
          break;
        }
        delay(wait_ms);
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
      resetSessionState_();
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
        const uint8_t sens = BMI2_AUX;
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

    detachSession_();
    last_error_ = Error::NONE;
    last_end_error_ = Error::NONE;
    return true;
  }

  bool readMag_uT(Vector3f& m_uT_out) {
    if (!ok_) {
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
      case Error::BMM_INIT_FAILED:                   return "bmm150_init failed";
      case Error::BMM_SET_PRESET_FAILED:             return "bmm150_set_presetmode failed";
      case Error::BMM_SET_OPMODE_FAILED:             return "bmm150_set_op_mode failed";
      case Error::BMM_TEST_READ_FAILED:              return "initial BMM150 read failed";
      case Error::MAG_READ_FAILED:                   return "bmm150_read_mag_data failed";
      case Error::NONFINITE_MAG:                     return "non-finite magnetometer output";
      case Error::ZERO_MAG:                          return "all-zero magnetometer sample";
      case Error::BMI_RESTORE_AUX_CFG_FAILED:        return "failed to restore prior BMI270 AUX config";
      case Error::BMI_AUX_DISABLE_FAILED:            return "failed to disable BMI270 AUX after restore";
      default:                                       return "unknown";
    }
  }

  static constexpr uint8_t expectedChipId_() { return 0x32u; }

  static constexpr uint8_t dataStartReg_() {
    #if defined(BMM150_REG_DATA_X_LSB)
      return BMM150_REG_DATA_X_LSB;
    #elif defined(BMM150_DATA_X_LSB)
      return BMM150_DATA_X_LSB;
    #else
      return 0x42u;
    #endif
  }

  static constexpr uint8_t chipIdReg_() {
    #if defined(BMM150_REG_CHIP_ID)
      return BMM150_REG_CHIP_ID;
    #elif defined(BMM150_CHIP_ID_ADDR)
      return BMM150_CHIP_ID_ADDR;
    #else
      return 0x40u;
    #endif
  }

  static constexpr uint8_t normalModeValue_() {
    #if defined(BMM150_POWERMODE_NORMAL)
      return BMM150_POWERMODE_NORMAL;
    #elif defined(BMM150_NORMAL_MODE)
      return BMM150_NORMAL_MODE;
    #else
      return 0x00u;
    #endif
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
      hz = 100.0f;
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
    #elif defined(BMI2_AUX_ODR_100HZ)
      return BMI2_AUX_ODR_100HZ;
    #else
      return 0;
    #endif
  }

  static float presetModeHz_(uint8_t preset_mode) {
    switch (preset_mode) {
      case BMM150_PRESETMODE_HIGHACCURACY:
        return 20.0f;
      case BMM150_PRESETMODE_LOWPOWER:
      case BMM150_PRESETMODE_REGULAR:
      case BMM150_PRESETMODE_ENHANCED:
      default:
        return 10.0f;
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

  static bool allZero3_(float x, float y, float z) {
    return x == 0.0f && y == 0.0f && z == 0.0f;
  }

  static bool allZero3_(const Vector3f& v) {
    return v.x() == 0.0f && v.y() == 0.0f && v.z() == 0.0f;
  }

  static float pickAxis_(int8_t code, const Vector3f& v) {
    switch (code) {
      case +1: return v.x();
      case -1: return -v.x();
      case +2: return v.y();
      case -2: return -v.y();
      case +3: return v.z();
      case -3: return -v.z();
      default: return 0.0f;
    }
  }

  Vector3f applyAxisMap_(const Vector3f& raw) const {
    return Vector3f(
      pickAxis_(cfg_.axis_map[0], raw),
      pickAxis_(cfg_.axis_map[1], raw),
      pickAxis_(cfg_.axis_map[2], raw)
    );
  }

  static BMM150_INTF_RET_TYPE bmm_read_(uint8_t reg,
                                        uint8_t* data,
                                        uint32_t len,
                                        void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_ || !data || len == 0u) {
      return static_cast<BMM150_INTF_RET_TYPE>(-1);
    }

    return static_cast<BMM150_INTF_RET_TYPE>(
      bmi2_read_aux_man_mode(reg, data, len, self->bmi_dev_));
  }

  static BMM150_INTF_RET_TYPE bmm_write_(uint8_t reg,
                                         const uint8_t* data,
                                         uint32_t len,
                                         void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_ || !data || len == 0u) {
      return static_cast<BMM150_INTF_RET_TYPE>(-1);
    }

    return static_cast<BMM150_INTF_RET_TYPE>(
      bmi2_write_aux_man_mode(reg, data, len, self->bmi_dev_));
  }

  static void bmm_delay_us_(uint32_t us, void*) {
    delayMicroseconds(us);
  }

  static bool finite3_(float x, float y, float z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  bool readChipIdRaw_(uint8_t& chip_id_out) {
    uint8_t id = 0;
    if (bmm_read_(chipIdReg_(), &id, 1, this) != 0) {
      return false;
    }
    chip_id_out = id;
    return true;
  }

  bool readMagInternal_(Vector3f& m_uT_out, bool count_stats) {
    uint32_t now_ms = millis();

    if (have_last_good_) {
      const uint32_t min_ms = minReadIntervalMs_();
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

    bmm150_mag_data md{};
    if (bmm150_read_mag_data(&md, &bmm_dev_) != BMM150_OK) {
      last_error_ = Error::MAG_READ_FAILED;
      if (count_stats) {
        ++read_fail_total_;
      }
      return false;
    }

    const float mx = static_cast<float>(md.x);
    const float my = static_cast<float>(md.y);
    const float mz = static_cast<float>(md.z);

    if (!finite3_(mx, my, mz)) {
      last_error_ = Error::NONFINITE_MAG;
      if (count_stats) {
        ++read_fail_total_;
      }
      return false;
    }

    if (allZero3_(mx, my, mz)) {
      last_error_ = Error::ZERO_MAG;
      if (count_stats) {
        ++read_fail_total_;
      }
      return false;
    }

    const Vector3f raw_uT(mx, my, mz);
    m_uT_out = applyAxisMap_(raw_uT);

    if (allZero3_(m_uT_out)) {
      last_error_ = Error::ZERO_MAG;
      if (count_stats) {
        ++read_fail_total_;
      }
      return false;
    }

    if (have_last_good_) {
      if (m_uT_out.x() == last_good_uT_.x() &&
          m_uT_out.y() == last_good_uT_.y() &&
          m_uT_out.z() == last_good_uT_.z()) {
        ++possible_duplicate_total_;
      }
    }

    last_good_uT_ = m_uT_out;
    have_last_good_ = true;
    last_read_ms_ = now_ms;
    last_error_ = Error::NONE;

    if (count_stats) {
      ++read_ok_total_;
    }

    return true;
  }

  void resetSessionState_() {
    ok_ = false;
    session_attached_ = false;
    last_error_ = Error::NONE;
    last_end_error_ = Error::NONE;

    cfg_ = Config{};
    bmi_dev_ = nullptr;

    saved_aux_cfg_valid_ = false;
    saved_aux_was_enabled_ = false;
    std::memset(&saved_aux_cfg_, 0, sizeof(saved_aux_cfg_));

    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));

    have_last_good_ = false;
    last_good_uT_ = Vector3f::Zero();
    last_read_ms_ = 0;
    effective_mag_hz_ = 10.0f;
  }

  void detachSession_() {
    resetSessionState_();
  }

  void bestEffortRollback_() {
    bool restore_ok = true;

    if (bmi_dev_ && saved_aux_cfg_valid_) {
      if (bmi270_set_sensor_config(&saved_aux_cfg_, 1, bmi_dev_) != BMI2_OK) {
        restore_ok = false;
      } else if (!saved_aux_was_enabled_) {
        const uint8_t sens = BMI2_AUX;
        if (bmi2_sensor_disable(&sens, 1, bmi_dev_) != BMI2_OK) {
          restore_ok = false;
        }
      }
    }

    if (!restore_ok) {
      ++rollback_fail_total_;
      ok_ = false;
      return;
    }

    detachSession_();
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

  struct bmm150_dev      bmm_dev_{};
  struct bmm150_settings bmm_settings_{};

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
    float    aux_odr_hz = 100.0f;
    uint8_t  preset_mode = 0;
    uint16_t startup_settle_ms = 120;
    bool     verify_first_read = true;
    bool     return_last_on_fast_poll = false;
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
      "BoschBmm150Aux: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, "
      "or exclude BoschBmm150Aux from this build.");
    return false;
  }

  template <typename Dummy = void>
  bool begin(struct bmi2_dev*, const Config&) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, "
      "or exclude BoschBmm150Aux from this build.");
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
