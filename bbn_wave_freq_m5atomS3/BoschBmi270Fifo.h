#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// Force Arduino builder to activate the library and add its include path.
#include <Arduino_BMI270_BMM150.h>

// Some library revisions use utilities/, others utility/
#if defined(__has_include)
  #if __has_include(<utilities/BMI270-Sensor-API/bmi2.h>)
    #include <utilities/BMI270-Sensor-API/bmi2.h>
    #include <utilities/BMI270-Sensor-API/bmi2_defs.h>
    #include <utilities/BMI270-Sensor-API/bmi270.h>
    #if __has_include(<utilities/BMI270-Sensor-API/bmi270_maximum_fifo.h>)
      #include <utilities/BMI270-Sensor-API/bmi270_maximum_fifo.h>
      #define ATOMS3R_HAVE_BMI270_MAXIMUM_FIFO_INIT 1
    #else
      #define ATOMS3R_HAVE_BMI270_MAXIMUM_FIFO_INIT 0
    #endif
    #define ATOMS3R_HAVE_BOSCH_SENSORAPI 1
  #elif __has_include(<utility/BMI270-Sensor-API/bmi2.h>)
    #include <utility/BMI270-Sensor-API/bmi2.h>
    #include <utility/BMI270-Sensor-API/bmi2_defs.h>
    #include <utility/BMI270-Sensor-API/bmi270.h>
    #if __has_include(<utility/BMI270-Sensor-API/bmi270_maximum_fifo.h>)
      #include <utility/BMI270-Sensor-API/bmi270_maximum_fifo.h>
      #define ATOMS3R_HAVE_BMI270_MAXIMUM_FIFO_INIT 1
    #else
      #define ATOMS3R_HAVE_BMI270_MAXIMUM_FIFO_INIT 0
    #endif
    #define ATOMS3R_HAVE_BOSCH_SENSORAPI 1
  #else
    #error "Arduino_BMI270_BMM150 is present, but BMI270 vendor headers were not found under utility/ or utilities/."
  #endif
#else
  #error "__has_include is required for Bosch vendor header path detection."
#endif

struct bmi2_dev;

// Timing constants
// BMI270 sensortime tick = 39.0625 us, 24-bit counter wraps.
static constexpr float    BMI270_SENSORTIME_TICK_S = 39.0625e-6f;
static constexpr uint32_t BMI270_SENSORTIME_MASK   = 0x00FFFFFFu;

// Optional compatibility escape hatch.
// Default is production-safe API: mutable device access is only via rawDevUnsafe().
#ifndef ATOMS3R_FIFO_ENABLE_UNSAFE_RAWDEV_COMPAT
  #define ATOMS3R_FIFO_ENABLE_UNSAFE_RAWDEV_COMPAT 0
#endif

// Output sample
struct BoschAGSample {
  float dt_s = 0.0f;
  float ax = 0.0f, ay = 0.0f, az = 0.0f; // m/s^2
  float gx = 0.0f, gy = 0.0f, gz = 0.0f; // rad/s
};

// Driver
class BoschBmi270Fifo {
public:
  enum class Error : uint8_t {
    NONE = 0,
    NOT_BUILT,
    BAD_ARG,
    INIT_FAILED,
    GET_CONFIG_FAILED,
    SET_CONFIG_FAILED,
    ENABLE_SENSOR_FAILED,
    SENSOR_DISABLE_FAILED,
    DISABLE_APS_FAILED,
    ENABLE_APS_FAILED,
    FIFO_CONFIG_FAILED,
    FIFO_DISABLE_FAILED,
    FIFO_WM_FAILED,
    FIFO_FLUSH_FAILED,
    FIFO_LEN_FAILED,
    FIFO_READ_FAILED,
    EXTRACT_FAILED,
    RECOVERY_FAILED,
    NOT_OK
  };

  bool ok() const { return ok_; }

  const bmi2_dev* rawDev() const { return rawDevPtr_(); }

#if ATOMS3R_FIFO_ENABLE_UNSAFE_RAWDEV_COMPAT
  bmi2_dev* rawDev() { return rawDevPtr_(); }
#endif

  bmi2_dev* rawDevUnsafe() { return rawDevPtr_(); }

  uint8_t addr() const { return bmi_addr_; }

  float odrHz() const { return odr_hz_; }
  float nominalDt() const { return nominal_dt_; }

  bool watermarkSetOk() const { return watermark_set_ok_; }

  uint32_t skippedFramesTotal() const    { return skipped_total_; }
  uint32_t unpairedFramesTotal() const   { return unpaired_total_; }
  uint32_t badTimingTotal() const        { return bad_timing_total_; }
  uint32_t fifoReadErrorsTotal() const   { return fifo_read_errors_total_; }
  uint32_t recoveriesTotal() const       { return recoveries_total_; }
  uint32_t recoveryFailuresTotal() const { return recovery_fail_total_; }
  uint32_t shutdownFailuresTotal() const { return shutdown_fail_total_; }
  uint32_t consecutiveReadErrors() const { return consecutive_read_errors_; }

  Error lastError() const { return last_error_; }
  Error lastRecoveryError() const { return last_recovery_error_; }
  Error lastShutdownError() const { return last_shutdown_error_; }

  const char* lastErrorString() const { return errorString_(last_error_); }
  const char* lastRecoveryErrorString() const { return errorString_(last_recovery_error_); }
  const char* lastShutdownErrorString() const { return errorString_(last_shutdown_error_); }

  bool usedMaximumFifoInit() const { return used_maximum_fifo_init_; }
  bool fellBackToPlainInit() const { return fell_back_to_plain_init_; }
  int8_t lastBoschInitResult() const { return last_bosch_init_rslt_; }

  const char* initPathString() const {
    if (used_maximum_fifo_init_) {
      return fell_back_to_plain_init_
          ? "bmi270_maximum_fifo_init -> bmi270_init fallback"
          : "bmi270_maximum_fifo_init";
    }
    return "bmi270_init";
  }

  void resetStatistics()
  {
    skipped_total_           = 0;
    unpaired_total_          = 0;
    bad_timing_total_        = 0;
    fifo_read_errors_total_  = 0;
    recoveries_total_        = 0;
    recovery_fail_total_     = 0;
    shutdown_fail_total_     = 0;
    last_recovery_error_     = Error::NONE;
    last_shutdown_error_     = Error::NONE;
  }

  bool begin(TwoWire& wire,
             uint8_t bmi270_addr = 0x68,
             float odr_hz = 100.0f,
             uint32_t i2c_hz = 400000)
  {
#if !ATOMS3R_HAVE_BOSCH_SENSORAPI
    clearSessionState_();
    wire_ = &wire;
    bmi_addr_ = bmi270_addr;
    requested_odr_hz_ = odr_hz;
    i2c_hz_ = i2c_hz;
    ok_ = false;
    last_error_ = Error::NOT_BUILT;
    return false;
#else
    if ((ok_ || device_initialized_) && !end()) {
      return false;
    }

    clearSessionState_();

    wire_             = &wire;
    bmi_addr_         = bmi270_addr;
    requested_odr_hz_ = odr_hz;
    i2c_hz_           = i2c_hz;
    last_error_       = Error::NONE;

    if (i2c_hz_ > 0u) {
      wire_->setClock(i2c_hz_);
    }

    std::memset(&bmi_, 0, sizeof(bmi_));
    bmi_.intf           = BMI2_I2C_INTF;
    bmi_.read           = &BoschBmi270Fifo::bmi2_i2c_read_;
    bmi_.write          = &BoschBmi270Fifo::bmi2_i2c_write_;
    bmi_.delay_us       = &BoschBmi270Fifo::bmi2_delay_us_;
    bmi_.read_write_len = I2C_CHUNK;
    bmi_.intf_ptr       = this;

    if (!initDeviceForFifo_()) {
      return failBegin_(Error::INIT_FAILED);
    }
    device_initialized_ = true;

    const bool use200 = (odr_hz > 150.0f);

    bmi2_sens_config cfg[2]{};
    cfg[0].type = BMI2_ACCEL;
    cfg[1].type = BMI2_GYRO;

    int8_t rslt = bmi270_get_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::GET_CONFIG_FAILED);
    }

    cfg[0].cfg.acc.odr         = use200 ? BMI2_ACC_ODR_200HZ : BMI2_ACC_ODR_100HZ;
    cfg[0].cfg.acc.range       = BMI2_ACC_RANGE_2G;
    cfg[0].cfg.acc.bwp         = BMI2_ACC_NORMAL_AVG4;
    cfg[0].cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;

    cfg[1].cfg.gyr.odr         = use200 ? BMI2_GYR_ODR_200HZ : BMI2_GYR_ODR_100HZ;
    cfg[1].cfg.gyr.range       = BMI2_GYR_RANGE_2000;
    cfg[1].cfg.gyr.bwp         = BMI2_GYR_NORMAL_MODE;
    cfg[1].cfg.gyr.noise_perf  = BMI2_POWER_OPT_MODE;
    cfg[1].cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;

    rslt = bmi270_set_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::SET_CONFIG_FAILED);
    }

    const uint8_t sens_list[2] = { BMI2_ACCEL, BMI2_GYRO };
    rslt = bmi270_sensor_enable(sens_list, 2, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::ENABLE_SENSOR_FAILED);
    }
    sensors_enabled_ = true;

    rslt = bmi2_set_adv_power_save(BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::DISABLE_APS_FAILED);
    }
    aps_disabled_ = true;

    rslt = bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::FIFO_CONFIG_FAILED);
    }

#ifdef BMI2_FIFO_HEADER_EN
    rslt = bmi2_set_fifo_config(BMI2_FIFO_HEADER_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::FIFO_CONFIG_FAILED);
    }
#endif

#ifdef BMI2_FIFO_TIME_EN
    rslt = bmi2_set_fifo_config(BMI2_FIFO_TIME_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::FIFO_CONFIG_FAILED);
    }
#endif

    rslt = bmi2_set_fifo_config(BMI2_FIFO_ACC_EN | BMI2_FIFO_GYR_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) {
      return failBegin_(Error::FIFO_CONFIG_FAILED);
    }
    fifo_configured_ = true;

    rslt = bmi2_set_fifo_wm(use200 ? 240 : 120, &bmi_);
    if (rslt != BMI2_OK) {
      watermark_set_ok_ = false;
      last_error_ = Error::FIFO_WM_FAILED;
    } else {
      watermark_set_ok_ = true;
      if (last_error_ == Error::FIFO_WM_FAILED) {
        last_error_ = Error::NONE;
      }
    }

    if (!flushFifo_()) {
      return failBegin_(Error::FIFO_FLUSH_FAILED);
    }

    odr_hz_     = use200 ? 200.0f : 100.0f;
    nominal_dt_ = 1.0f / odr_hz_;

    clearReadPipelineState_();

    ok_ = true;
    last_error_ = Error::NONE;
    return true;
#endif
  }

  bool recover()
  {
#if !ATOMS3R_HAVE_BOSCH_SENSORAPI
    ok_ = false;
    last_error_ = Error::NOT_BUILT;
    last_recovery_error_ = Error::NOT_BUILT;
    return false;
#else
    if (in_recovery_) {
      last_error_ = Error::RECOVERY_FAILED;
      last_recovery_error_ = Error::RECOVERY_FAILED;
      ok_ = false;
      ++recovery_fail_total_;
      return false;
    }

    if (wire_ == nullptr) {
      last_error_ = Error::RECOVERY_FAILED;
      last_recovery_error_ = Error::RECOVERY_FAILED;
      ok_ = false;
      ++recovery_fail_total_;
      return false;
    }

    in_recovery_ = true;
    const bool ok = begin(*wire_, bmi_addr_, requested_odr_hz_, i2c_hz_);
    in_recovery_ = false;

    if (ok) {
      ++recoveries_total_;
      last_recovery_error_ = Error::NONE;
      return true;
    }

    ++recovery_fail_total_;
    last_recovery_error_ = last_error_;
    ok_ = false;
    return false;
#endif
  }

  bool end()
  {
#if !ATOMS3R_HAVE_BOSCH_SENSORAPI
    clearSessionState_();
    last_error_ = Error::NONE;
    last_shutdown_error_ = Error::NONE;
    return true;
#else
    Error shutdown_err = Error::NONE;
    const bool ok = teardownDevice_(&shutdown_err, true);

    clearSessionState_();

    if (!ok) {
      last_error_ = shutdown_err;
      return false;
    }

    last_error_ = Error::NONE;
    return true;
#endif
  }

  bool readOneAG(BoschAGSample& out)
  {
    return readAG(&out, 1) == 1;
  }

  int readAG(BoschAGSample* out, int max_out)
  {
#if !ATOMS3R_HAVE_BOSCH_SENSORAPI
    (void)out;
    (void)max_out;
    last_error_ = Error::NOT_BUILT;
    return 0;
#else
    if (!ok_) {
      last_error_ = Error::NOT_OK;
      return 0;
    }
    if (out == nullptr || max_out <= 0) {
      last_error_ = Error::BAD_ARG;
      return 0;
    }

    int produced = 0;

    while (produced < max_out) {
      if (pending_count_ == 0) {
        if (!fillPendingFromFifo_()) {
          if (last_fill_had_error_ && consecutive_read_errors_ >= RECOVERY_THRESHOLD) {
            if (recover()) {
              if (!fillPendingFromFifo_()) {
                break;
              }
            } else {
              break;
            }
          } else {
            break;
          }
        }
      }

      while (produced < max_out && pending_count_ > 0) {
        out[produced++] = pending_[pending_head_];
        pending_head_ = (pending_head_ + 1) % PENDING_CAP;
        --pending_count_;
      }
    }

    return produced;
#endif
  }

private:
  static constexpr uint16_t I2C_CHUNK             = 64;
  static constexpr int      MAX_EXTRACT           = 128;
  static constexpr int      PENDING_CAP           = MAX_EXTRACT;
  static constexpr size_t   FIFO_BUF_CAP          = 2048u + 256u;
  static constexpr uint32_t PAIR_TIME_SLACK_TICKS = 32u;
  static constexpr uint32_t MAX_REASONABLE_DT_US  = 250000;
  static constexpr uint8_t  RECOVERY_THRESHOLD    = 3u;

  static constexpr float kPi = 3.14159265358979323846f;
  static constexpr float kG0 = 9.80665f;

  static const char* errorString_(Error e)
  {
    switch (e) {
      case Error::NONE:                  return "none";
      case Error::NOT_BUILT:             return "Bosch SensorAPI not available in this build";
      case Error::BAD_ARG:               return "bad argument";
      case Error::INIT_FAILED:           return "BMI270 FIFO init failed";
      case Error::GET_CONFIG_FAILED:     return "bmi270_get_sensor_config failed";
      case Error::SET_CONFIG_FAILED:     return "bmi270_set_sensor_config failed";
      case Error::ENABLE_SENSOR_FAILED:  return "bmi270_sensor_enable failed";
      case Error::SENSOR_DISABLE_FAILED: return "bmi270_sensor_disable failed";
      case Error::DISABLE_APS_FAILED:    return "bmi2_set_adv_power_save(DISABLE) failed";
      case Error::ENABLE_APS_FAILED:     return "bmi2_set_adv_power_save(ENABLE) failed";
      case Error::FIFO_CONFIG_FAILED:    return "bmi2_set_fifo_config failed";
      case Error::FIFO_DISABLE_FAILED:   return "bmi2_set_fifo_config disable failed";
      case Error::FIFO_WM_FAILED:        return "bmi2_set_fifo_wm failed";
      case Error::FIFO_FLUSH_FAILED:     return "FIFO flush command failed";
      case Error::FIFO_LEN_FAILED:       return "bmi2_get_fifo_length failed";
      case Error::FIFO_READ_FAILED:      return "bmi2_read_fifo_data failed";
      case Error::EXTRACT_FAILED:        return "bmi2_extract_accel/gyro failed";
      case Error::RECOVERY_FAILED:       return "driver recovery failed";
      case Error::NOT_OK:                return "driver not initialized";
      default:                           return "unknown";
    }
  }

  bool failBegin_(Error primary_err)
  {
#if ATOMS3R_HAVE_BOSCH_SENSORAPI
    last_error_ = primary_err;

    Error cleanup_err = Error::NONE;
    (void)teardownDevice_(&cleanup_err, true);
#endif
    clearSessionState_();
    ok_ = false;
    last_error_ = primary_err;
    return false;
  }

#if ATOMS3R_HAVE_BOSCH_SENSORAPI
  static constexpr uint8_t cmdRegAddr_()
  {
#ifdef BMI2_CMD_REG_ADDR
    return BMI2_CMD_REG_ADDR;
#else
    return 0x7E;
#endif
  }

  static constexpr uint8_t fifoFlushCmd_()
  {
#ifdef BMI2_FIFO_FLUSH_CMD
    return BMI2_FIFO_FLUSH_CMD;
#else
    return 0xB0;
#endif
  }

  bool initDeviceForFifo_()
  {
    used_maximum_fifo_init_  = false;
    fell_back_to_plain_init_ = false;
    last_bosch_init_rslt_    = BMI2_OK;

#if ATOMS3R_HAVE_BMI270_MAXIMUM_FIFO_INIT
    used_maximum_fifo_init_ = true;
    last_bosch_init_rslt_ = bmi270_maximum_fifo_init(&bmi_);
    if (last_bosch_init_rslt_ == BMI2_OK) {
      return true;
    }

    fell_back_to_plain_init_ = true;
    last_bosch_init_rslt_ = bmi270_init(&bmi_);
    return (last_bosch_init_rslt_ == BMI2_OK);
#else
    last_bosch_init_rslt_ = bmi270_init(&bmi_);
    return (last_bosch_init_rslt_ == BMI2_OK);
#endif
  }

  bool flushFifo_()
  {
    uint8_t cmd = fifoFlushCmd_();
    if (bmi2_set_regs(cmdRegAddr_(), &cmd, 1, &bmi_) != BMI2_OK) {
      return false;
    }
    delayMicroseconds(1000);
    return true;
  }

  bool teardownDevice_(Error* first_err_out, bool count_failure)
  {
    Error first = Error::NONE;
    bool all_ok = true;

    if (device_initialized_) {
      if (fifo_configured_) {
        if (bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi_) != BMI2_OK) {
          all_ok = false;
          if (first == Error::NONE) first = Error::FIFO_DISABLE_FAILED;
        } else {
          fifo_configured_ = false;
        }
      }

      if (sensors_enabled_) {
        const uint8_t sens_list[2] = { BMI2_ACCEL, BMI2_GYRO };
        if (bmi270_sensor_disable(sens_list, 2, &bmi_) != BMI2_OK) {
          all_ok = false;
          if (first == Error::NONE) first = Error::SENSOR_DISABLE_FAILED;
        } else {
          sensors_enabled_ = false;
        }
      }

      if (aps_disabled_) {
        if (bmi2_set_adv_power_save(BMI2_ENABLE, &bmi_) != BMI2_OK) {
          all_ok = false;
          if (first == Error::NONE) first = Error::ENABLE_APS_FAILED;
        } else {
          aps_disabled_ = false;
        }
      }
    }

    if (!all_ok && count_failure) {
      ++shutdown_fail_total_;
      last_shutdown_error_ = first;
    } else if (all_ok) {
      last_shutdown_error_ = Error::NONE;
    }

    if (first_err_out != nullptr) {
      *first_err_out = first;
    }
    return all_ok;
  }

  void clearReadPipelineState_()
  {
    pending_head_  = 0;
    pending_count_ = 0;

    have_sens_time_ = false;
    last_sens_time_ = 0;

    consecutive_read_errors_ = 0;
    last_fill_had_error_     = false;

    std::memset(&fifo_, 0, sizeof(fifo_));
    std::memset(accel_, 0, sizeof(accel_));
    std::memset(gyro_,  0, sizeof(gyro_));
    std::memset(pending_, 0, sizeof(pending_));
  }

  void clearSessionState_()
  {
    ok_                  = false;
    watermark_set_ok_    = false;
    last_fill_had_error_ = false;
    in_recovery_         = false;

    device_initialized_ = false;
    sensors_enabled_    = false;
    fifo_configured_    = false;
    aps_disabled_       = false;

    used_maximum_fifo_init_  = false;
    fell_back_to_plain_init_ = false;
    last_bosch_init_rslt_    = BMI2_OK;
    
    odr_hz_     = 100.0f;
    nominal_dt_ = 0.01f;

    std::memset(&bmi_, 0, sizeof(bmi_));

    clearReadPipelineState_();

    wire_ = nullptr;
  }

  bmi2_dev* rawDevPtr_()
  {
    return &bmi_;
  }

  const bmi2_dev* rawDevPtr_() const
  {
    return &bmi_;
  }

  template <typename T>
  static auto sensTimeField_(const T& s, int) -> decltype(s.sens_time, uint32_t())
  {
    return static_cast<uint32_t>(s.sens_time);
  }

  template <typename T>
  static auto sensTimeField_(const T& s, long) -> decltype(s.virt_sens_time, uint32_t())
  {
    return static_cast<uint32_t>(s.virt_sens_time);
  }

  template <typename T>
  static uint32_t sensTimeField_(const T&, ...)
  {
    return 0u;
  }

  static uint32_t sensTime24_(const bmi2_sens_axes_data& s)
  {
    return sensTimeField_(s, 0) & BMI270_SENSORTIME_MASK;
  }

  static bool streamHasUsableTime_(const bmi2_sens_axes_data* s, uint16_t n)
  {
    for (uint16_t i = 0; i < n; ++i) {
      if (sensTime24_(s[i]) != 0u) return true;
    }
    return false;
  }

  static uint32_t absTickDiff24_(uint32_t a, uint32_t b)
  {
    const uint32_t d1 = (a - b) & BMI270_SENSORTIME_MASK;
    const uint32_t d2 = (b - a) & BMI270_SENSORTIME_MASK;
    return (d1 < d2) ? d1 : d2;
  }

  static bool timeBefore24_(uint32_t a, uint32_t b)
  {
    return (a != b) && (((b - a) & BMI270_SENSORTIME_MASK) < 0x00800000u);
  }

  static uint32_t chooseLaterTime24_(uint32_t a, uint32_t b)
  {
    if (a == 0u) return b;
    if (b == 0u) return a;
    return timeBefore24_(a, b) ? b : a;
  }

  void noteReadError_(Error err)
  {
    last_fill_had_error_ = true;
    ++fifo_read_errors_total_;
    ++consecutive_read_errors_;
    last_error_ = err;
  }

  void noteReadSuccess_()
  {
    last_fill_had_error_ = false;
    consecutive_read_errors_ = 0;
    if (last_error_ == Error::FIFO_LEN_FAILED ||
        last_error_ == Error::FIFO_READ_FAILED ||
        last_error_ == Error::EXTRACT_FAILED) {
      last_error_ = Error::NONE;
    }
  }

  float computeDtFromSensTime_(uint32_t st24)
  {
    if (st24 == 0u) {
      ++bad_timing_total_;
      return nominal_dt_;
    }

    if (!have_sens_time_) {
      have_sens_time_ = true;
      last_sens_time_ = st24;
      return nominal_dt_;
    }

    const uint32_t d_ticks = (st24 - last_sens_time_) & BMI270_SENSORTIME_MASK;
    last_sens_time_ = st24;

    float dt_s = static_cast<float>(d_ticks) * BMI270_SENSORTIME_TICK_S;

    if (!(dt_s > 0.0f)) {
      ++bad_timing_total_;
      return nominal_dt_;
    }

    if (dt_s < 0.25f * nominal_dt_) {
      ++bad_timing_total_;
      return nominal_dt_;
    }

    if (dt_s > (static_cast<float>(MAX_REASONABLE_DT_US) * 1.0e-6f)) {
      ++bad_timing_total_;
      dt_s = static_cast<float>(MAX_REASONABLE_DT_US) * 1.0e-6f;
    }

    return dt_s;
  }

  bool pushPairedSample_(const bmi2_sens_axes_data& a,
                         const bmi2_sens_axes_data& g,
                         uint32_t st24)
  {
    if (pending_count_ >= PENDING_CAP) return false;

    const int idx = (pending_head_ + pending_count_) % PENDING_CAP;
    BoschAGSample& out = pending_[idx];

    constexpr float acc_range_g   = 2.0f;
    constexpr float gyr_range_dps = 2000.0f;
    constexpr float dps_to_rps    = kPi / 180.0f;

    out.ax = static_cast<float>(a.x) * (acc_range_g * kG0) / 32768.0f;
    out.ay = static_cast<float>(a.y) * (acc_range_g * kG0) / 32768.0f;
    out.az = static_cast<float>(a.z) * (acc_range_g * kG0) / 32768.0f;

    out.gx = static_cast<float>(g.x) * (gyr_range_dps * dps_to_rps) / 32768.0f;
    out.gy = static_cast<float>(g.y) * (gyr_range_dps * dps_to_rps) / 32768.0f;
    out.gz = static_cast<float>(g.z) * (gyr_range_dps * dps_to_rps) / 32768.0f;

    out.dt_s = computeDtFromSensTime_(st24);

    ++pending_count_;
    return true;
  }

  bool pairByIndex_(uint16_t a_len, uint16_t g_len)
  {
    const uint16_t n = (a_len < g_len) ? a_len : g_len;
    for (uint16_t i = 0; i < n; ++i) {
      const uint32_t ta = sensTime24_(accel_[i]);
      const uint32_t tg = sensTime24_(gyro_[i]);
      const uint32_t st = chooseLaterTime24_(ta, tg);
      if (!pushPairedSample_(accel_[i], gyro_[i], st)) {
        return true;
      }
    }

    if (a_len > n) unpaired_total_ += static_cast<uint32_t>(a_len - n);
    if (g_len > n) unpaired_total_ += static_cast<uint32_t>(g_len - n);

    return n > 0u;
  }

  bool pairLockstep_(uint16_t a_len, uint16_t g_len)
  {
    if (a_len != g_len) return false;

    int compared = 0;
    int good     = 0;

    for (uint16_t i = 0; i < a_len; ++i) {
      const uint32_t ta = sensTime24_(accel_[i]);
      const uint32_t tg = sensTime24_(gyro_[i]);

      if (ta != 0u && tg != 0u) {
        ++compared;
        if (absTickDiff24_(ta, tg) <= PAIR_TIME_SLACK_TICKS) {
          ++good;
        }
      }
    }

    if (compared > 0 && (good * 10 < compared * 9)) {
      return false;
    }

    for (uint16_t i = 0; i < a_len; ++i) {
      const uint32_t ta = sensTime24_(accel_[i]);
      const uint32_t tg = sensTime24_(gyro_[i]);

      if (ta != 0u && tg != 0u && absTickDiff24_(ta, tg) > PAIR_TIME_SLACK_TICKS) {
        unpaired_total_ += 2u;
        ++bad_timing_total_;
        continue;
      }

      const uint32_t st24 = chooseLaterTime24_(ta, tg);
      if (!pushPairedSample_(accel_[i], gyro_[i], st24)) {
        return true;
      }
    }

    return true;
  }

  void pairByMergedTime_(uint16_t a_len, uint16_t g_len)
  {
    uint16_t ia = 0;
    uint16_t ig = 0;

    while (ia < a_len && ig < g_len && pending_count_ < PENDING_CAP) {
      const uint32_t ta = sensTime24_(accel_[ia]);
      const uint32_t tg = sensTime24_(gyro_[ig]);

      if (ta == 0u || tg == 0u) {
        if (ta == 0u) { ++ia; ++unpaired_total_; ++bad_timing_total_; }
        if (tg == 0u) { ++ig; ++unpaired_total_; ++bad_timing_total_; }
        continue;
      }

      const uint32_t d = absTickDiff24_(ta, tg);
      if (d <= PAIR_TIME_SLACK_TICKS) {
        const uint32_t st24 = chooseLaterTime24_(ta, tg);
        (void)pushPairedSample_(accel_[ia], gyro_[ig], st24);
        ++ia;
        ++ig;
        continue;
      }

      if (timeBefore24_(ta, tg)) {
        ++ia;
        ++unpaired_total_;
      } else {
        ++ig;
        ++unpaired_total_;
      }
    }

    unpaired_total_ += static_cast<uint32_t>(a_len - ia);
    unpaired_total_ += static_cast<uint32_t>(g_len - ig);
  }

  bool fillPendingFromFifo_()
  {
    if (!ok_) {
      last_error_ = Error::NOT_OK;
      last_fill_had_error_ = true;
      return false;
    }
    if (pending_count_ > 0) {
      last_fill_had_error_ = false;
      return true;
    }

    uint16_t fifo_len = 0;
    if (bmi2_get_fifo_length(&fifo_len, &bmi_) != BMI2_OK) {
      noteReadError_(Error::FIFO_LEN_FAILED);
      return false;
    }

    if (fifo_len == 0u) {
      last_fill_had_error_ = false;
      return false;
    }

    uint32_t req = static_cast<uint32_t>(fifo_len);
    req += static_cast<uint32_t>(bmi_.dummy_byte);

    if (req > static_cast<uint32_t>(sizeof(fifo_buf_))) {
      req = static_cast<uint32_t>(sizeof(fifo_buf_));
    }

    std::memset(&fifo_, 0, sizeof(fifo_));
    fifo_.data   = fifo_buf_;
    fifo_.length = static_cast<uint16_t>(req);

    if (bmi2_read_fifo_data(&fifo_, &bmi_) != BMI2_OK) {
      noteReadError_(Error::FIFO_READ_FAILED);
      return false;
    }

    skipped_total_ += fifo_.skipped_frame_count;

    std::memset(accel_, 0, sizeof(accel_));
    std::memset(gyro_,  0, sizeof(gyro_));

    uint16_t a_len = MAX_EXTRACT;
    uint16_t g_len = MAX_EXTRACT;

    const int8_t ra = bmi2_extract_accel(accel_, &a_len, &fifo_, &bmi_);
    const int8_t rg = bmi2_extract_gyro (gyro_,  &g_len, &fifo_, &bmi_);

    if (ra != BMI2_OK) {
      a_len = 0;
    }
    if (rg != BMI2_OK) {
      g_len = 0;
    }
    if (ra != BMI2_OK || rg != BMI2_OK) {
      noteReadError_(Error::EXTRACT_FAILED);
    }

    if (a_len == 0u && g_len == 0u) {
      return false;
    }

    const bool a_has_time = streamHasUsableTime_(accel_, a_len);
    const bool g_has_time = streamHasUsableTime_(gyro_,  g_len);

    bool paired_any = false;

    if (!a_has_time || !g_has_time) {
      paired_any = pairByIndex_(a_len, g_len);
    } else {
      if (pairLockstep_(a_len, g_len)) {
        paired_any = (pending_count_ > 0);
      } else {
        pairByMergedTime_(a_len, g_len);
        paired_any = (pending_count_ > 0);
      }
    }

    if (paired_any || !last_fill_had_error_) {
      noteReadSuccess_();
    }

    return pending_count_ > 0;
  }

  static int8_t bmi2_i2c_read_(uint8_t reg_addr, uint8_t* reg_data, uint32_t len, void* intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (self == nullptr || self->wire_ == nullptr || reg_data == nullptr) {
      return static_cast<int8_t>(-1);
    }

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = static_cast<uint16_t>(std::min<uint32_t>(I2C_CHUNK, len - off));

      const uint8_t addr = (reg_addr == BMI2_FIFO_DATA_ADDR)
                         ? reg_addr
                         : static_cast<uint8_t>(reg_addr + off);

      self->wire_->beginTransmission(self->bmi_addr_);
      self->wire_->write(addr);
      if (self->wire_->endTransmission(false) != 0) {
        return static_cast<int8_t>(-1);
      }

      const uint16_t got = static_cast<uint16_t>(
          self->wire_->requestFrom(static_cast<int>(self->bmi_addr_),
                                   static_cast<int>(n)));

      if (got != n) {
        return static_cast<int8_t>(-1);
      }

      for (uint16_t i = 0; i < n; ++i) {
        if (!self->wire_->available()) {
          return static_cast<int8_t>(-1);
        }
        reg_data[off + i] = static_cast<uint8_t>(self->wire_->read());
      }

      off += n;
    }

    return BMI2_OK;
  }

  static int8_t bmi2_i2c_write_(uint8_t reg_addr, const uint8_t* reg_data, uint32_t len, void* intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (self == nullptr || self->wire_ == nullptr || reg_data == nullptr) {
      return static_cast<int8_t>(-1);
    }

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = static_cast<uint16_t>(std::min<uint32_t>(I2C_CHUNK, len - off));

      self->wire_->beginTransmission(self->bmi_addr_);
      self->wire_->write(static_cast<uint8_t>(reg_addr + off));

      for (uint16_t i = 0; i < n; ++i) {
        self->wire_->write(reg_data[off + i]);
      }

      if (self->wire_->endTransmission(true) != 0) {
        return static_cast<int8_t>(-1);
      }

      off += n;
    }

    return BMI2_OK;
  }

  static void bmi2_delay_us_(uint32_t period, void*)
  {
    delayMicroseconds(period);
  }
#endif

private:
  TwoWire* wire_ = nullptr;
  uint8_t  bmi_addr_ = 0x68;
  bool     ok_ = false;
  bool     watermark_set_ok_ = false;
  bool     in_recovery_ = false;
  bool     last_fill_had_error_ = false;

  bool     device_initialized_ = false;
  bool     sensors_enabled_    = false;
  bool     fifo_configured_    = false;
  bool     aps_disabled_       = false;

  float    requested_odr_hz_ = 100.0f;
  uint32_t i2c_hz_ = 400000;

  float odr_hz_     = 100.0f;
  float nominal_dt_ = 0.01f;

  bool     have_sens_time_ = false;
  uint32_t last_sens_time_ = 0;

  uint32_t skipped_total_           = 0;
  uint32_t unpaired_total_          = 0;
  uint32_t bad_timing_total_        = 0;
  uint32_t fifo_read_errors_total_  = 0;
  uint32_t recoveries_total_        = 0;
  uint32_t recovery_fail_total_     = 0;
  uint32_t shutdown_fail_total_     = 0;
  uint32_t consecutive_read_errors_ = 0;

  Error last_error_          = Error::NONE;
  Error last_recovery_error_ = Error::NONE;
  Error last_shutdown_error_ = Error::NONE;

  bool   used_maximum_fifo_init_   = false;
  bool   fell_back_to_plain_init_  = false;
  int8_t last_bosch_init_rslt_     = BMI2_OK;

#if ATOMS3R_HAVE_BOSCH_SENSORAPI
  bmi2_dev        bmi_{};
  bmi2_fifo_frame fifo_{};

  uint8_t             fifo_buf_[FIFO_BUF_CAP] = {0};
  bmi2_sens_axes_data accel_[MAX_EXTRACT]     = {};
  bmi2_sens_axes_data gyro_[MAX_EXTRACT]      = {};
#endif

  BoschAGSample pending_[PENDING_CAP]{};
  int pending_head_  = 0;
  int pending_count_ = 0;
};
