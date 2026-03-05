#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// Bosch vendored SensorAPI headers (from Arduino_BMI270_BMM150 library)
#include <utilities/BMI270-Sensor-API/bmi2.h>
#include <utilities/BMI270-Sensor-API/bmi2_defs.h>
#include <utilities/BMI270-Sensor-API/bmi270.h>

// BMI270 sensortime tick: 39.0625 us (2^-8 ms), 24-bit wrap
static constexpr float    BMI270_SENSORTIME_TICK_S = 39.0625e-6f;
static constexpr uint32_t BMI270_SENSORTIME_MASK   = 0x00FFFFFFu;

struct BoschAGSample {
  float dt_s = 0.0f;
  float ax = 0.0f, ay = 0.0f, az = 0.0f; // m/s^2
  float gx = 0.0f, gy = 0.0f, gz = 0.0f; // rad/s
};

class BoschBmi270Fifo {
public:
  bool ok() const { return ok_; }

  // Access to underlying Bosch device for AUX / extra features (e.g., BMM150 via BMI270 AUX).
  bmi2_dev* rawDev() { return &bmi_; }
  const bmi2_dev* rawDev() const { return &bmi_; }
  uint8_t addr() const { return bmi_addr_; }

  bool begin(TwoWire& wire,
             uint8_t bmi270_addr = 0x68,
             float odr_hz = 100.0f,
             uint32_t i2c_hz = 400000)
  {
    ok_ = false;
    wire_ = &wire;
    bmi_addr_ = bmi270_addr;

    // Ensure I2C is up (safe to call multiple times on ESP32 Arduino)
    wire_->begin();
    wire_->setClock(i2c_hz);

    // Init Bosch device struct
    std::memset(&bmi_, 0, sizeof(bmi_));
    bmi_.intf = BMI2_I2C_INTF;
    bmi_.read = &BoschBmi270Fifo::bmi2_i2c_read_;
    bmi_.write = &BoschBmi270Fifo::bmi2_i2c_write_;
    bmi_.delay_us = &BoschBmi270Fifo::bmi2_delay_us_;
    bmi_.read_write_len = I2C_CHUNK;   // our callback chunks anyway
    bmi_.intf_ptr = this;

    int8_t rslt = bmi270_init(&bmi_);
    if (rslt != BMI2_OK) return false;

    // Configure accel/gyro
    bmi2_sens_config cfg[2];
    cfg[0].type = BMI2_ACCEL;
    cfg[1].type = BMI2_GYRO;

    rslt = bmi270_get_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

    // ODR selection (extend if needed)
    const bool use200 = (odr_hz >= 200.0f);
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

    // FIFO reads require disabling advanced power save
    rslt = bmi2_set_adv_power_save(BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // FIFO: disable all, then enable accel+gyro, header, time
    rslt = bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Enable header mode if available
    #ifdef BMI2_FIFO_HEADER_EN
      (void)bmi2_set_fifo_config(BMI2_FIFO_HEADER_EN, BMI2_ENABLE, &bmi_);
    #endif

    // Enable time frame if available (not required for dt in this implementation,
    // since we use per-sample sens_time, but harmless and useful for debugging).
    #ifdef BMI2_FIFO_TIME_EN
      (void)bmi2_set_fifo_config(BMI2_FIFO_TIME_EN, BMI2_ENABLE, &bmi_);
    #endif

    rslt = bmi2_set_fifo_config(BMI2_FIFO_ACC_EN | BMI2_FIFO_GYR_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Optional: set watermark (bytes)
    (void)bmi2_set_fifo_wm(120, &bmi_);

    // Flush after configuration so old frames don't pollute first dt
    (void)bmi2_flush_fifo(&bmi_);

    odr_hz_      = use200 ? 200.0f : 100.0f;
    nominal_dt_  = 1.0f / odr_hz_;

    skipped_total_   = 0;
    have_sens_time_  = false;
    last_sens_time_  = 0;
    mismatch_ctr_    = 0;

    ok_ = true;
    return true;
  }

  bool readOneAG(BoschAGSample& out) {
    BoschAGSample tmp[1];
    const int n = readAG(tmp, 1);
    if (n <= 0) return false;
    out = tmp[0];
    return true;
  }

  // Read and extract up to max_out accel+gyro samples.
  // dt is computed from FIFO per-sample sens_time (stable even if FIFO not drained to empty).
  int readAG(BoschAGSample* out, int max_out)
  {
    if (!ok_ || !out || max_out <= 0) return 0;

    uint16_t fifo_len = 0;
    if (bmi2_get_fifo_length(&fifo_len, &bmi_) != BMI2_OK) return 0;
    if (fifo_len == 0) return 0;

    // Limit max_out to our internal extract buffers.
    max_out = std::min<int>(max_out, MAX_EXTRACT);

    // Don't read the entire FIFO if caller only wants a few samples; otherwise you
    // drain a lot of data and would be forced to drop it.
    // Rough estimate: header-mode accel frame ~7 bytes, gyro frame ~7 bytes => ~14 bytes per AG pair.
    // Add margin for config/time frames.
    constexpr uint16_t BYTES_PER_AG_EST = 16;
    constexpr uint16_t MARGIN_BYTES     = 96;

    const uint32_t want_bytes_u32 =
        (uint32_t)max_out * (uint32_t)BYTES_PER_AG_EST +
        (uint32_t)MARGIN_BYTES +
        (uint32_t)bmi_.dummy_byte;

    uint16_t req = (uint16_t)std::min<uint32_t>((uint32_t)fifo_len + (uint32_t)bmi_.dummy_byte, want_bytes_u32);
    req = (uint16_t)std::min<uint32_t>(req, (uint32_t)sizeof(fifo_buf_));

    fifo_.data   = fifo_buf_;
    fifo_.length = req;

    if (bmi2_read_fifo_data(&fifo_, &bmi_) != BMI2_OK) return 0;

    uint16_t a_len = (uint16_t)max_out;
    uint16_t g_len = (uint16_t)max_out;

    (void)bmi2_extract_accel(accel_, &a_len, &fifo_, &bmi_);
    (void)bmi2_extract_gyro (gyro_,  &g_len, &fifo_, &bmi_);

    if (a_len != g_len) ++mismatch_ctr_;

    const int n = (int)std::min<uint16_t>(a_len, g_len);
    if (n <= 0) {
      skipped_total_ += fifo_.skipped_frame_count;
      return 0;
    }

    skipped_total_ += fifo_.skipped_frame_count;

    // Convert raw -> physical
    constexpr float ACC_RANGE_G   = 2.0f;
    constexpr float GYR_RANGE_DPS = 2000.0f;
    constexpr float G0            = 9.80665f;
    const float dps_to_rps = (float)M_PI / 180.0f;

    // dt from per-sample sens_time (24-bit wrap)
    // If sens_time is missing/zero, fall back to nominal.
    uint32_t prev_st = have_sens_time_ ? last_sens_time_ : 0;
    bool have_prev   = have_sens_time_;

    for (int i = 0; i < n; ++i) {
      out[i].ax = ((float)accel_[i].x) * (ACC_RANGE_G * G0) / 32768.0f;
      out[i].ay = ((float)accel_[i].y) * (ACC_RANGE_G * G0) / 32768.0f;
      out[i].az = ((float)accel_[i].z) * (ACC_RANGE_G * G0) / 32768.0f;

      out[i].gx = ((float)gyro_[i].x) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
      out[i].gy = ((float)gyro_[i].y) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
      out[i].gz = ((float)gyro_[i].z) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;

      const uint32_t st = ((uint32_t)accel_[i].sens_time) & BMI270_SENSORTIME_MASK;

      float dt_s = nominal_dt_;
      if (st != 0) {
        if (have_prev) {
          const uint32_t d_ticks = (st - prev_st) & BMI270_SENSORTIME_MASK;
          dt_s = (float)d_ticks * BMI270_SENSORTIME_TICK_S;

          // sanity clamp
          if (!(dt_s > 0.0f) || dt_s < 0.0005f) dt_s = 0.0005f;
          if (dt_s > 0.0500f) dt_s = 0.0500f;

          // snap-to-nominal if crazy (protect against a single corrupt frame)
          const float ratio = dt_s / nominal_dt_;
          if (ratio < 0.5f || ratio > 1.5f) dt_s = nominal_dt_;
        } else {
          // first ever sample: use nominal
          have_prev = true;
        }
        prev_st = st;
      }
      out[i].dt_s = dt_s;
    }

    // Save last sens_time for continuity across calls
    const uint32_t st_last = ((uint32_t)accel_[n - 1].sens_time) & BMI270_SENSORTIME_MASK;
    if (st_last != 0) {
      last_sens_time_ = st_last;
      have_sens_time_ = true;
    }

    return n;
  }

  uint32_t skippedFramesTotal() const { return skipped_total_; }
  uint32_t mismatchCountTotal() const { return mismatch_ctr_; }

private:
  // Keep chunks conservative to avoid Wire rx/tx buffer issues on ESP32.
  static constexpr uint16_t I2C_CHUNK   = 64;
  static constexpr int      MAX_EXTRACT = 128;

  TwoWire* wire_ = nullptr;
  uint8_t  bmi_addr_ = 0x68;
  bool     ok_ = false;

  bmi2_dev        bmi_{};
  bmi2_fifo_frame fifo_{};

  // FIFO buffers
  uint8_t fifo_buf_[2048 + 256] = {0};
  bmi2_sens_axes_data accel_[MAX_EXTRACT] = {};
  bmi2_sens_axes_data gyro_[MAX_EXTRACT]  = {};

  float odr_hz_     = 100.0f;
  float nominal_dt_ = 0.01f;

  bool     have_sens_time_ = false;
  uint32_t last_sens_time_ = 0;

  uint32_t skipped_total_  = 0;
  uint32_t mismatch_ctr_   = 0;

  // Bosch BMI2 I2C glue. Must support large FIFO reads even if Wire buffers are small.
  static int8_t bmi2_i2c_read_(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_ || !reg_data) return (int8_t)-1;

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = (uint16_t)std::min<uint32_t>(I2C_CHUNK, len - off);

      // FIFO_DATA (0x26) address does NOT increment during burst reads, must keep it fixed.
      // Other registers should be advanced per chunk.
      const uint8_t addr = (reg_addr == BMI2_FIFO_DATA_ADDR) ? reg_addr : (uint8_t)(reg_addr + off);

      self->wire_->beginTransmission(self->bmi_addr_);
      self->wire_->write(addr);
      if (self->wire_->endTransmission(false) != 0) return (int8_t)-1;

      const uint16_t got = (uint16_t)self->wire_->requestFrom((int)self->bmi_addr_, (int)n);
      if (got != n) return (int8_t)-1;

      for (uint16_t i = 0; i < n; ++i) reg_data[off + i] = (uint8_t)self->wire_->read();
      off += n;
    }
    return BMI2_OK;
  }

  static int8_t bmi2_i2c_write_(uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_ || !reg_data) return (int8_t)-1;

    uint32_t off = 0;
    while (off < len) {
      const uint16_t n = (uint16_t)std::min<uint32_t>(I2C_CHUNK, len - off);

      // Defensive: if anyone ever writes to FIFO_DATA_ADDR, keep addr fixed.
      const uint8_t addr = (reg_addr == BMI2_FIFO_DATA_ADDR) ? reg_addr : (uint8_t)(reg_addr + off);

      self->wire_->beginTransmission(self->bmi_addr_);
      self->wire_->write(addr);
      for (uint16_t i = 0; i < n; ++i) self->wire_->write(reg_data[off + i]);
      if (self->wire_->endTransmission(true) != 0) return (int8_t)-1;

      off += n;
    }
    return BMI2_OK;
  }

  static void bmi2_delay_us_(uint32_t period, void*)
  {
    delayMicroseconds(period);
  }
};
