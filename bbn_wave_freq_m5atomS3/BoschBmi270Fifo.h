#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <type_traits>

// Bosch SensorAPI availability detection
// Include the public Arduino library header first so arduino-cli
// resolves/activates the library and adds its src/ include path.
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

#if ATOMS3R_HAVE_ARDUINO_BMI270_BMM150 && defined(__has_include)
  // Try both possible internal folder names (some forks use utility/ vs utilities/)
  #if __has_include(<utilities/BMI270-Sensor-API/bmi2.h>)
    #include <utilities/BMI270-Sensor-API/bmi2.h>
    #include <utilities/BMI270-Sensor-API/bmi2_defs.h>
    #include <utilities/BMI270-Sensor-API/bmi270.h>
    #define ATOMS3R_HAVE_BOSCH_SENSORAPI 1
  #elif __has_include(<utility/BMI270-Sensor-API/bmi2.h>)
    #include <utility/BMI270-Sensor-API/bmi2.h>
    #include <utility/BMI270-Sensor-API/bmi2_defs.h>
    #include <utility/BMI270-Sensor-API/bmi270.h>
    #define ATOMS3R_HAVE_BOSCH_SENSORAPI 1
  #else
    #define ATOMS3R_HAVE_BOSCH_SENSORAPI 0
  #endif
#else
  #define ATOMS3R_HAVE_BOSCH_SENSORAPI 0
#endif

// Forward declaration so stub API can expose rawDev().
struct bmi2_dev;

// BMI270 sensortime tick: 39.0625 us (2^-8 ms), 24-bit wrap
static constexpr float    BMI270_SENSORTIME_TICK_S = 39.0625e-6f;
static constexpr uint32_t BMI270_SENSORTIME_MASK   = 0x00FFFFFFu;

struct BoschAGSample {
  float dt_s = 0.0f;
  float ax = 0.0f, ay = 0.0f, az = 0.0f; // m/s^2
  float gx = 0.0f, gy = 0.0f, gz = 0.0f; // rad/s
};

#if ATOMS3R_HAVE_BOSCH_SENSORAPI

class BoschBmi270Fifo {
public:
  bool ok() const { return ok_; }

  // Access to underlying Bosch device for AUX / extra features (e.g. BMM150 via BMI270 AUX).
  bmi2_dev*       rawDev()       { return &bmi_; }
  const bmi2_dev* rawDev() const { return &bmi_; }
  uint8_t         addr()   const { return bmi_addr_; }

  float odrHz()     const { return odr_hz_; }
  float nominalDt() const { return nominal_dt_; }

  uint32_t skippedFramesTotal() const { return skipped_total_; }
  uint32_t unpairedFramesTotal() const { return unpaired_total_; }
  uint32_t badTimingTotal() const { return bad_timing_total_; }
  uint32_t fifoReadErrorsTotal() const { return fifo_read_errors_total_; }

  // Caller must have already called wire.begin(...).
  // Supported ODRs here are 100 Hz and 200 Hz; the nearest of those two is chosen.
  bool begin(TwoWire& wire,
             uint8_t bmi270_addr = 0x68,
             float odr_hz = 100.0f,
             uint32_t i2c_hz = 400000)
  {
    ok_ = false;
    wire_ = &wire;
    bmi_addr_ = bmi270_addr;

    if (i2c_hz > 0) {
      wire_->setClock(i2c_hz);
    }

    std::memset(&bmi_, 0, sizeof(bmi_));
    bmi_.intf           = BMI2_I2C_INTF;
    bmi_.read           = &BoschBmi270Fifo::bmi2_i2c_read_;
    bmi_.write          = &BoschBmi270Fifo::bmi2_i2c_write_;
    bmi_.delay_us       = &BoschBmi270Fifo::bmi2_delay_us_;
    bmi_.read_write_len = I2C_CHUNK;
    bmi_.intf_ptr       = this;

    int8_t rslt = bmi270_init(&bmi_);
    if (rslt != BMI2_OK) return false;

    // Choose the nearest supported ODR.
    const bool use200 = (odr_hz > 150.0f);

    bmi2_sens_config cfg[2]{};
    cfg[0].type = BMI2_ACCEL;
    cfg[1].type = BMI2_GYRO;

    rslt = bmi270_get_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

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
    if (rslt != BMI2_OK) return false;

    const uint8_t sens_list[2] = { BMI2_ACCEL, BMI2_GYRO };
    rslt = bmi270_sensor_enable(sens_list, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

    // FIFO reads require disabling advanced power save.
    rslt = bmi2_set_adv_power_save(BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // FIFO: disable all first.
    rslt = bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Enable header + sensortime when available.
    #ifdef BMI2_FIFO_HEADER_EN
      rslt = bmi2_set_fifo_config(BMI2_FIFO_HEADER_EN, BMI2_ENABLE, &bmi_);
      if (rslt != BMI2_OK) return false;
    #endif

    #ifdef BMI2_FIFO_TIME_EN
      rslt = bmi2_set_fifo_config(BMI2_FIFO_TIME_EN, BMI2_ENABLE, &bmi_);
      if (rslt != BMI2_OK) return false;
    #endif

    rslt = bmi2_set_fifo_config(BMI2_FIFO_ACC_EN | BMI2_FIFO_GYR_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Conservative watermark, in FIFO bytes.
    (void)bmi2_set_fifo_wm(use200 ? 240 : 120, &bmi_);
    (void)bmi2_flush_fifo(&bmi_);

    odr_hz_     = use200 ? 200.0f : 100.0f;
    nominal_dt_ = 1.0f / odr_hz_;

    clearRuntimeState_();

    ok_ = true;
    return true;
  }

  bool readOneAG(BoschAGSample& out)
  {
    return readAG(&out, 1) == 1;
  }

  // Reads up to max_out already-paired AG samples.
  // Uses internal queueing so FIFO is drained in coherent chunks.
  int readAG(BoschAGSample* out, int max_out)
  {
    if (!ok_ || !out || max_out <= 0) return 0;

    int produced = 0;

    while (produced < max_out) {
      if (pending_count_ == 0) {
        if (!fillPendingFromFifo_()) {
          break;
        }
      }

      while (produced < max_out && pending_count_ > 0) {
        out[produced++] = pending_[pending_head_];
        pending_head_ = (pending_head_ + 1) % PENDING_CAP;
        --pending_count_;
      }
    }

    return produced;
  }

private:
  static constexpr uint16_t I2C_CHUNK   = 64;
  static constexpr int      MAX_EXTRACT = 128;
  static constexpr int      PENDING_CAP = MAX_EXTRACT;

  // Accept small accel/gyro timestamp disagreement when lockstep pairing.
  // 32 ticks ~= 1.25 ms.
  static constexpr uint32_t PAIR_TIME_SLACK_TICKS = 32u;

  // Full current FIFO occupancy fits comfortably here; BMI270 FIFO itself is much smaller.
  static constexpr size_t FIFO_BUF_CAP = 2048u + 256u;

  TwoWire* wire_ = nullptr;
  uint8_t  bmi_addr_ = 0x68;
  bool     ok_ = false;

  bmi2_dev        bmi_{};
  bmi2_fifo_frame fifo_{};

  uint8_t fifo_buf_[FIFO_BUF_CAP] = {0};
  bmi2_sens_axes_data accel_[MAX_EXTRACT] = {};
  bmi2_sens_axes_data gyro_[MAX_EXTRACT]  = {};

  BoschAGSample pending_[PENDING_CAP]{};
  int pending_head_  = 0;
  int pending_count_ = 0;

  float odr_hz_     = 100.0f;
  float nominal_dt_ = 0.01f;

  bool     have_sens_time_ = false;
  uint32_t last_sens_time_ = 0;

  uint32_t skipped_total_          = 0;
  uint32_t unpaired_total_         = 0;
  uint32_t bad_timing_total_       = 0;
  uint32_t fifo_read_errors_total_ = 0;

  void clearRuntimeState_()
  {
    pending_head_  = 0;
    pending_count_ = 0;

    have_sens_time_ = false;
    last_sens_time_ = 0;

    skipped_total_          = 0;
    unpaired_total_         = 0;
    bad_timing_total_       = 0;
    fifo_read_errors_total_ = 0;

    std::memset(&fifo_,  0, sizeof(fifo_));
    std::memset(accel_,  0, sizeof(accel_));
    std::memset(gyro_,   0, sizeof(gyro_));
    std::memset(pending_, 0, sizeof(pending_));
  }

  static uint32_t sensTime24_(const bmi2_sens_axes_data& s)
  {
    return static_cast<uint32_t>(s.sens_time) & BMI270_SENSORTIME_MASK;
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
    // True if a is earlier than b on the 24-bit ring, assuming no huge gap.
    return (a != b) && (((b - a) & BMI270_SENSORTIME_MASK) < 0x00800000u);
  }

  static uint32_t chooseLaterTime24_(uint32_t a, uint32_t b)
  {
    if (a == 0u) return b;
    if (b == 0u) return a;
    return timeBefore24_(a, b) ? b : a;
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

    // Reject only impossible / obviously-corrupt values.
    if (!(dt_s > 0.0f)) {
      ++bad_timing_total_;
      return nominal_dt_;
    }

    // Too small to be physically meaningful for 100/200 Hz AG output.
    if (dt_s < 0.25f * nominal_dt_) {
      ++bad_timing_total_;
      return nominal_dt_;
    }

    // Preserve genuine delayed servicing / FIFO backlog rather than forcing nominal.
    // Only clamp absurd extremes to protect downstream code.
    if (dt_s > 0.25f) {
      ++bad_timing_total_;
      dt_s = 0.25f;
    }

    return dt_s;
  }

  bool pushPairedSample_(const bmi2_sens_axes_data& a,
                         const bmi2_sens_axes_data& g,
                         uint32_t st24)
  {
    if (pending_count_ >= PENDING_CAP) {
      return false;
    }

    const int idx = (pending_head_ + pending_count_) % PENDING_CAP;
    BoschAGSample& out = pending_[idx];

    constexpr float ACC_RANGE_G   = 2.0f;
    constexpr float GYR_RANGE_DPS = 2000.0f;
    constexpr float G0            = 9.80665f;
    const float dps_to_rps = static_cast<float>(M_PI) / 180.0f;

    out.ax = static_cast<float>(a.x) * (ACC_RANGE_G * G0) / 32768.0f;
    out.ay = static_cast<float>(a.y) * (ACC_RANGE_G * G0) / 32768.0f;
    out.az = static_cast<float>(a.z) * (ACC_RANGE_G * G0) / 32768.0f;

    out.gx = static_cast<float>(g.x) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
    out.gy = static_cast<float>(g.y) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
    out.gz = static_cast<float>(g.z) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;

    out.dt_s = computeDtFromSensTime_(st24);

    ++pending_count_;
    return true;
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

    // If timestamps exist and strongly disagree, do not trust index pairing.
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
        // Cannot safely merge missing timestamps here.
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
    if (!ok_) return false;
    if (pending_count_ > 0) return true;

    uint16_t fifo_len = 0;
    if (bmi2_get_fifo_length(&fifo_len, &bmi_) != BMI2_OK) {
      ++fifo_read_errors_total_;
      return false;
    }

    if (fifo_len == 0u) {
      return false;
    }

    uint32_t req = static_cast<uint32_t>(fifo_len) + static_cast<uint32_t>(bmi_.dummy_byte);
    if (req > static_cast<uint32_t>(sizeof(fifo_buf_))) {
      req = static_cast<uint32_t>(sizeof(fifo_buf_));
    }

    std::memset(&fifo_, 0, sizeof(fifo_));
    fifo_.data   = fifo_buf_;
    fifo_.length = static_cast<uint16_t>(req);

    if (bmi2_read_fifo_data(&fifo_, &bmi_) != BMI2_OK) {
      ++fifo_read_errors_total_;
      return false;
    }

    skipped_total_ += fifo_.skipped_frame_count;

    uint16_t a_len = MAX_EXTRACT;
    uint16_t g_len = MAX_EXTRACT;

    (void)bmi2_extract_accel(accel_, &a_len, &fifo_, &bmi_);
    (void)bmi2_extract_gyro (gyro_,  &g_len, &fifo_, &bmi_);

    if (a_len == 0u && g_len == 0u) {
      return false;
    }

    if (!pairLockstep_(a_len, g_len)) {
      pairByMergedTime_(a_len, g_len);
    }

    return pending_count_ > 0;
  }

  static int8_t bmi2_i2c_read_(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_ || !reg_data) return static_cast<int8_t>(-1);

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = static_cast<uint16_t>(std::min<uint32_t>(I2C_CHUNK, len - off));

      // FIFO_DATA address does not increment during burst reads.
      const uint8_t addr = (reg_addr == BMI2_FIFO_DATA_ADDR)
                         ? reg_addr
                         : static_cast<uint8_t>(reg_addr + off);

      self->wire_->beginTransmission(self->bmi_addr_);
      self->wire_->write(addr);
      if (self->wire_->endTransmission(false) != 0) {
        return static_cast<int8_t>(-1);
      }

      const uint16_t got = static_cast<uint16_t>(self->wire_->requestFrom(
          static_cast<int>(self->bmi_addr_), static_cast<int>(n)));

      if (got != n) {
        return static_cast<int8_t>(-1);
      }

      for (uint16_t i = 0; i < n; ++i) {
        reg_data[off + i] = static_cast<uint8_t>(self->wire_->read());
      }

      off += n;
    }

    return BMI2_OK;
  }

  static int8_t bmi2_i2c_write_(uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_ || !reg_data) return static_cast<int8_t>(-1);

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
};

#else  // ATOMS3R_HAVE_BOSCH_SENSORAPI == 0

class BoschBmi270Fifo {
  template <typename> struct always_false : std::false_type {};

public:
  bool ok() const { return false; }

  bmi2_dev*       rawDev()       { return nullptr; }
  const bmi2_dev* rawDev() const { return nullptr; }
  uint8_t         addr()   const { return 0x68; }

  float odrHz()     const { return 0.0f; }
  float nominalDt() const { return 0.0f; }

  uint32_t skippedFramesTotal() const { return 0; }
  uint32_t unpairedFramesTotal() const { return 0; }
  uint32_t badTimingTotal() const { return 0; }
  uint32_t fifoReadErrorsTotal() const { return 0; }

  template <typename Dummy = void>
  bool begin(TwoWire&, uint8_t = 0x68, float = 100.0f, uint32_t = 400000) {
    static_assert(always_false<Dummy>::value,
      "BoschBmi270Fifo: Bosch SensorAPI headers not found. Install Arduino_BMI270_BMM150, "
      "or exclude BoschBmi270Fifo from this sketch.");
    return false;
  }

  template <typename Dummy = void>
  bool readOneAG(BoschAGSample&) {
    static_assert(always_false<Dummy>::value,
      "BoschBmi270Fifo unavailable in this build.");
    return false;
  }

  template <typename Dummy = void>
  int readAG(BoschAGSample*, int) {
    static_assert(always_false<Dummy>::value,
      "BoschBmi270Fifo unavailable in this build.");
    return 0;
  }
};

#endif
