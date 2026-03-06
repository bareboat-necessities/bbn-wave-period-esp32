#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#ifndef BMM150_USE_FLOATING_POINT
  #define BMM150_USE_FLOATING_POINT 1
#endif

// Public Arduino library header first, so arduino-cli activates the library
// and exposes its internal Bosch SensorAPI paths.
#if !defined(ATOMS3R_HAVE_ARDUINO_BMI270_BMM150)
  #if defined(__has_include)
    #if __has_include(<Arduino_BMI270_BMM150.h>)
      #include <Arduino_BMI270_BMM150.h>
      #define ATOMS3R_HAVE_ARDUINO_BMI270_BMM150 1
    #else
      #define ATOMS3R_HAVE_ARDUINO_BMI270_BMM150 0
    #endif
  #else
    #define ATOMS3R_HAVE_ARDUINO_BMI270_BMM150 0
  #endif
#endif

#if !defined(ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI)
  #if ATOMS3R_HAVE_ARDUINO_BMI270_BMM150 && defined(__has_include)
    #if __has_include(<utilities/BMI270-Sensor-API/bmi2.h>) && \
        __has_include(<utilities/BMM150-Sensor-API/bmm150.h>)
      #include <utilities/BMI270-Sensor-API/bmi2.h>
      #include <utilities/BMI270-Sensor-API/bmi2_defs.h>
      #include <utilities/BMI270-Sensor-API/bmi270.h>
      #include <utilities/BMM150-Sensor-API/bmm150.h>
      #include <utilities/BMM150-Sensor-API/bmm150_defs.h>
      #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
    #elif __has_include(<utility/BMI270-Sensor-API/bmi2.h>) && \
          __has_include(<utility/BMM150-Sensor-API/bmm150.h>)
      #include <utility/BMI270-Sensor-API/bmi2.h>
      #include <utility/BMI270-Sensor-API/bmi2_defs.h>
      #include <utility/BMI270-Sensor-API/bmi270.h>
      #include <utility/BMM150-Sensor-API/bmm150.h>
      #include <utility/BMM150-Sensor-API/bmm150_defs.h>
      #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 1
    #else
      #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 0
    #endif
  #else
    #define ATOMS3R_HAVE_BOSCH_BMM150_AUX_SENSORAPI 0
  #endif
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
    uint8_t bmm_addr = BMM150_DEFAULT_I2C_ADDRESS; // usually 0x10
    float   aux_odr_hz = 25.0f;                    // 25 Hz is plenty for heading/yaw correction
    bool    set_pullups_2k = true;                 // currently advisory/no-op unless safely supported
    uint8_t preset_mode = BMM150_PRESETMODE_REGULAR;
    bool    normal_mode = true;                    // production path supports continuous/normal mode only
  };

  enum class Error : uint8_t {
    NONE = 0,
    NULL_BMI_DEV,
    BMI_ADV_POWER_SAVE_DISABLE_FAILED,
    BMI_AUX_ENABLE_FAILED,
    BMI_GET_AUX_CFG_FAILED,
    BMI_SET_AUX_CFG_FAILED,
    BMM_INIT_FAILED,
    BMM_SET_PRESET_FAILED,
    BMM_SET_OPMODE_FAILED,
    CHIP_ID_READ_FAILED,
    CHIP_ID_MISMATCH,
    NOT_INITIALIZED,
    MAG_READ_FAILED,
    NONFINITE_MAG,
    UNSUPPORTED_FORCED_MODE
  };

  BoschBmm150Aux() = default;

  bool ok() const { return ok_; }
  Error lastError() const { return last_error_; }

  // Initialize BMM150 via BMI270 AUX, using the already initialized BMI270 device.
  // bmi_dev must remain valid for the lifetime of this object.
  bool begin(struct bmi2_dev* bmi_dev, const Config& cfg = Config()) {
    resetState_();

    bmi_dev_ = bmi_dev;
    cfg_ = cfg;

    if (!bmi_dev_) {
      last_error_ = Error::NULL_BMI_DEV;
      return false;
    }

    // This implementation is intentionally continuous/normal-mode only.
    if (!cfg_.normal_mode) {
      last_error_ = Error::UNSUPPORTED_FORCED_MODE;
      return false;
    }

    int8_t rslt = bmi2_set_adv_power_save(BMI2_DISABLE, bmi_dev_);
    if (rslt != BMI2_OK) {
      last_error_ = Error::BMI_ADV_POWER_SAVE_DISABLE_FAILED;
      return false;
    }

    const uint8_t sens = BMI2_AUX;
    rslt = bmi2_sensor_enable(&sens, 1, bmi_dev_);
    if (rslt != BMI2_OK) {
      last_error_ = Error::BMI_AUX_ENABLE_FAILED;
      return false;
    }

    bmi2_sens_config sc{};
    sc.type = BMI2_AUX;

    rslt = bmi270_get_sensor_config(&sc, 1, bmi_dev_);
    if (rslt != BMI2_OK) {
      last_error_ = Error::BMI_GET_AUX_CFG_FAILED;
      return false;
    }

    sc.cfg.aux.aux_en          = BMI2_ENABLE;
    sc.cfg.aux.manual_en       = BMI2_ENABLE;
    sc.cfg.aux.i2c_device_addr = cfg_.bmm_addr;
    sc.cfg.aux.fcu_write_en    = BMI2_ENABLE;
    sc.cfg.aux.man_rd_burst    = manualBurstLen_();
    sc.cfg.aux.read_addr       = dataStartReg_();
    sc.cfg.aux.odr             = auxOdrFromHz_(cfg_.aux_odr_hz);

    rslt = bmi270_set_sensor_config(&sc, 1, bmi_dev_);
    if (rslt != BMI2_OK) {
      last_error_ = Error::BMI_SET_AUX_CFG_FAILED;
      return false;
    }

    // Do not raw-write AUX_IF_TRIM here. Different library revisions expose
    // different masks/bit layouts, and a full-register write can clobber other fields.
    (void)cfg_.set_pullups_2k;

    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    bmm_dev_.intf     = BMM150_I2C_INTF;
    bmm_dev_.intf_ptr = this;
    bmm_dev_.read     = &BoschBmm150Aux::bmm_read_;
    bmm_dev_.write    = &BoschBmm150Aux::bmm_write_;
    bmm_dev_.delay_us = &BoschBmm150Aux::bmm_delay_us_;

    const int8_t br = bmm150_init(&bmm_dev_);
    if (br != BMM150_OK) {
      last_error_ = Error::BMM_INIT_FAILED;
      return false;
    }

    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));
    bmm_settings_.preset_mode = cfg_.preset_mode;

    const int8_t pr = bmm150_set_presetmode(&bmm_settings_, &bmm_dev_);
    if (pr != BMM150_OK) {
      last_error_ = Error::BMM_SET_PRESET_FAILED;
      return false;
    }

    bmm_settings_.pwr_mode = normalModeValue_();
    const int8_t mr = bmm150_set_op_mode(&bmm_settings_, &bmm_dev_);
    if (mr != BMM150_OK) {
      last_error_ = Error::BMM_SET_OPMODE_FAILED;
      return false;
    }

    // Give the continuous stream a brief moment to settle.
    delay(3);

    uint8_t id = 0;
    if (!readChipId(id)) {
      last_error_ = Error::CHIP_ID_READ_FAILED;
      return false;
    }
    if (id != expectedChipId_()) {
      last_error_ = Error::CHIP_ID_MISMATCH;
      return false;
    }

    ok_ = true;
    last_error_ = Error::NONE;
    return true;
  }

  // Read compensated magnetometer in microtesla.
  bool readMag_uT(Vector3f& m_uT_out) {
    if (!ok_) {
      last_error_ = Error::NOT_INITIALIZED;
      return false;
    }

    bmm150_mag_data md{};
    const int8_t r = bmm150_read_mag_data(&md, &bmm_dev_);
    if (r != BMM150_OK) {
      last_error_ = Error::MAG_READ_FAILED;
      return false;
    }

    const float mx = static_cast<float>(md.x);
    const float my = static_cast<float>(md.y);
    const float mz = static_cast<float>(md.z);

    if (!std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(mz)) {
      last_error_ = Error::NONFINITE_MAG;
      return false;
    }

    m_uT_out = Vector3f(mx, my, mz);
    last_error_ = Error::NONE;
    return true;
  }

  bool readChipId(uint8_t& chip_id_out) {
    uint8_t id = 0;
    if (bmm_read_(chipIdReg_(), &id, 1, this) != 0) {
      return false;
    }
    chip_id_out = id;
    return true;
  }

private:
  static constexpr uint8_t expectedChipId_() { return 0x32u; }

  static constexpr uint8_t chipIdReg_() {
    #if defined(BMM150_REG_CHIP_ID)
      return BMM150_REG_CHIP_ID;
    #elif defined(BMM150_CHIP_ID_ADDR)
      return BMM150_CHIP_ID_ADDR;
    #else
      return 0x40u;
    #endif
  }

  static constexpr uint8_t dataStartReg_() {
    #if defined(BMM150_REG_DATA_X_LSB)
      return BMM150_REG_DATA_X_LSB;
    #elif defined(BMM150_DATA_X_LSB)
      return BMM150_DATA_X_LSB;
    #else
      return 0x42u;
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
    // Need enough for the BMM150 data block.
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
    #elif defined(BMI2_AUX_ODR_25HZ)
      return BMI2_AUX_ODR_25HZ;
    #else
      return 0;
    #endif
  }

  static BMM150_INTF_RET_TYPE bmm_read_(uint8_t reg,
                                        uint8_t* data,
                                        uint32_t len,
                                        void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_ || !data || len == 0u) {
      return static_cast<BMM150_INTF_RET_TYPE>(-1);
    }

    const int8_t r = bmi2_read_aux_man_mode(reg, data, len, self->bmi_dev_);
    return static_cast<BMM150_INTF_RET_TYPE>(r);
  }

  static BMM150_INTF_RET_TYPE bmm_write_(uint8_t reg,
                                         const uint8_t* data,
                                         uint32_t len,
                                         void* intf_ptr) {
    auto* self = reinterpret_cast<BoschBmm150Aux*>(intf_ptr);
    if (!self || !self->bmi_dev_ || !data || len == 0u) {
      return static_cast<BMM150_INTF_RET_TYPE>(-1);
    }

    const int8_t r = bmi2_write_aux_man_mode(reg, data, len, self->bmi_dev_);
    return static_cast<BMM150_INTF_RET_TYPE>(r);
  }

  static void bmm_delay_us_(uint32_t us, void*) {
    delayMicroseconds(us);
  }

  void resetState_() {
    ok_ = false;
    last_error_ = Error::NONE;
    bmi_dev_ = nullptr;
    std::memset(&bmm_dev_, 0, sizeof(bmm_dev_));
    std::memset(&bmm_settings_, 0, sizeof(bmm_settings_));
  }

private:
  bool ok_ = false;
  Error last_error_ = Error::NONE;

  Config cfg_{};
  struct bmi2_dev* bmi_dev_ = nullptr;

  struct bmm150_dev      bmm_dev_{};
  struct bmm150_settings bmm_settings_{};
};

#else

struct bmi2_dev;

class BoschBmm150Aux {
  template <typename> struct always_false : std::false_type {};

public:
  struct Config {
    uint8_t bmm_addr = 0x10;
    float   aux_odr_hz = 25.0f;
    bool    set_pullups_2k = true;
    uint8_t preset_mode = 0;
    bool    normal_mode = true;
  };

  enum class Error : uint8_t {
    NONE = 0
  };

  BoschBmm150Aux() = default;

  bool ok() const { return false; }
  Error lastError() const { return Error::NONE; }

  template <typename Dummy = void>
  bool begin(struct bmi2_dev*, const Config& = Config()) {
    static_assert(always_false<Dummy>::value,
      "BoschBmm150Aux: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, "
      "or exclude BoschBmm150Aux from this build.");
    return false;
  }

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
