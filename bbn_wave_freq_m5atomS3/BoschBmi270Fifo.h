#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>

#include "bmi270.h"   // Bosch BMI270 SensorAPI (bmi2 + bmi270)
#include "bmm150.h"   // Optional (see notes below)

// If your BMI270 sensortime resolution differs, change here.
// BMI-family sensors commonly use 39.0625 us ticks (2^(-8) ms).
static constexpr float BMI270_SENSORTIME_TICK_S = 39.0625e-6f;  // 39.0625 µs per tick :contentReference[oaicite:5]{index=5}
static constexpr uint32_t BMI270_SENSORTIME_MASK = 0x00FFFFFFu; // 24-bit counter wrap

struct BoschAGSample {
  float dt_s = 0.0f;
  float ax = 0.0f, ay = 0.0f, az = 0.0f; // m/s^2
  float gx = 0.0f, gy = 0.0f, gz = 0.0f; // rad/s
};

class BoschBmi270Fifo {
public:
  bool begin(TwoWire& wire,
             uint8_t bmi270_addr = 0x68,
             float odr_hz = 100.0f)
  {
    wire_ = &wire;
    bmi_addr_ = bmi270_addr;

    // -------- BMI270 (Bosch vendor API) init --------
    memset(&bmi_, 0, sizeof(bmi_));
    bmi_.intf = BMI2_I2C_INTF;
    bmi_.read = &BoschBmi270Fifo::bmi2_i2c_read_;
    bmi_.write = &BoschBmi270Fifo::bmi2_i2c_write_;
    bmi_.delay_us = &BoschBmi270Fifo::bmi2_delay_us_;
    bmi_.read_write_len = 64;     // safe chunk; some API calls may still read larger
    bmi_.intf_ptr = this;

    int8_t rslt = bmi270_init(&bmi_);
    if (rslt != BMI2_OK) return false;

    // Configure accel/gyro
    bmi2_sens_config cfg[2];
    cfg[0].type = BMI2_ACCEL;
    cfg[1].type = BMI2_GYRO;

    rslt = bmi270_get_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Map requested ODR to the closest macro (extend if you need more)
    cfg[0].cfg.acc.odr = (odr_hz >= 200.0f) ? BMI2_ACC_ODR_200HZ : BMI2_ACC_ODR_100HZ;
    cfg[0].cfg.acc.range = BMI2_ACC_RANGE_2G;
    cfg[0].cfg.acc.bwp = BMI2_ACC_NORMAL_AVG4;
    cfg[0].cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;

    cfg[1].cfg.gyr.odr = (odr_hz >= 200.0f) ? BMI2_GYR_ODR_200HZ : BMI2_GYR_ODR_100HZ;
    cfg[1].cfg.gyr.range = BMI2_GYR_RANGE_2000;
    cfg[1].cfg.gyr.bwp = BMI2_GYR_NORMAL_MODE;
    cfg[1].cfg.gyr.noise_perf = BMI2_POWER_OPT_MODE;
    cfg[1].cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;

    rslt = bmi270_set_sensor_config(cfg, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

    const uint8_t sens_list[2] = { BMI2_ACCEL, BMI2_GYRO };
    rslt = bmi270_sensor_enable(sens_list, 2, &bmi_);
    if (rslt != BMI2_OK) return false;

    // FIFO reads require disabling advanced power save. :contentReference[oaicite:6]{index=6}
    rslt = bmi2_set_adv_power_save(BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // FIFO: start clean then enable accel+gyro.
    rslt = bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    rslt = bmi2_set_fifo_config(BMI2_FIFO_ACC_EN | BMI2_FIFO_GYR_EN, BMI2_ENABLE, &bmi_);
    if (rslt != BMI2_OK) return false;

    // Keep header mode ON (default) because it’s what the Bosch example uses
    // to parse control frames like sensor_time. :contentReference[oaicite:7]{index=7}
    // Ensure FIFO time is enabled if the macro exists in your SensorAPI build:
    #ifdef BMI2_FIFO_TIME_EN
      (void)bmi2_set_fifo_config(BMI2_FIFO_TIME_EN, BMI2_ENABLE, &bmi_);
    #endif

    // Best-effort watermark (optional). FIFO length polling is enough.
    (void)bmi2_set_fifo_wm(120, &bmi_); // bytes; tweak if you want fewer reads

    // Nominal dt (used only for first batch / weird batches)
    odr_hz_ = (odr_hz >= 200.0f) ? 200.0f : 100.0f;
    nominal_dt_ = 1.0f / odr_hz_;

    last_stime_ = 0;
    have_stime_ = false;
    skipped_total_ = 0;
    return true;
  }

  // Read and extract up to max_out samples.
  // Returns number of samples written.
  int readAG(BoschAGSample* out, int max_out)
  {
    if (!out || max_out <= 0) return 0;

    uint16_t fifo_len = 0;
    if (bmi2_get_fifo_length(&fifo_len, &bmi_) != BMI2_OK) return 0;
    if (fifo_len < 16) return 0;

    // Bosch FIFO header-mode example reads extra bytes to ensure sensor_time/control frames are included. :contentReference[oaicite:8]{index=8}
    constexpr uint16_t OVERREAD = 220; // same idea as Bosch example
    uint16_t req = fifo_len + OVERREAD + bmi_.dummy_byte;
    if (req > sizeof(fifo_buf_)) req = sizeof(fifo_buf_);

    fifo_.data = fifo_buf_;
    fifo_.length = req;

    if (bmi2_read_fifo_data(&fifo_, &bmi_) != BMI2_OK) return 0;

    uint16_t a_len = (uint16_t)max_out;
    uint16_t g_len = (uint16_t)max_out;

    (void)bmi2_extract_accel(accel_, &a_len, &fifo_, &bmi_);
    (void)bmi2_extract_gyro (gyro_,  &g_len, &fifo_, &bmi_);

    const int n = (int)std::min(a_len, g_len);
    if (n <= 0) return 0;

    skipped_total_ += fifo_.skipped_frame_count;

    // Convert sensor_time (ticks) -> dt across the *batch*, then per-sample dt.
    float dt_per = nominal_dt_;

    const uint32_t st = ((uint32_t)fifo_.sensor_time) & BMI270_SENSORTIME_MASK;
    if (!have_stime_) {
      have_stime_ = true;
      last_stime_ = st;
      dt_per = nominal_dt_;
    } else {
      const uint32_t d = (st - last_stime_) & BMI270_SENSORTIME_MASK;
      last_stime_ = st;

      const float dt_total = (float)d * BMI270_SENSORTIME_TICK_S;
      dt_per = dt_total / (float)n;

      // sanity clamps (protect the filter if FIFO stalls or bursts oddly)
      if (!(dt_per > 0.0f) || dt_per < 0.0005f) dt_per = 0.0005f;
      if (dt_per > 0.0500f) dt_per = 0.0500f;
    }

    // LSB -> physical units (using configured ranges: ±2g, ±2000 dps)
    // accel: ±2g full-scale maps to ±32768
    // gyro : ±2000 dps full-scale maps to ±32768
    constexpr float ACC_RANGE_G = 2.0f;
    constexpr float GYR_RANGE_DPS = 2000.0f;
    constexpr float G0 = 9.80665f;

    for (int i = 0; i < n; ++i) {
      const int16_t ax = accel_[i].x;
      const int16_t ay = accel_[i].y;
      const int16_t az = accel_[i].z;
      const int16_t gx = gyro_[i].x;
      const int16_t gy = gyro_[i].y;
      const int16_t gz = gyro_[i].z;

      out[i].dt_s = dt_per;

      out[i].ax = ((float)ax) * (ACC_RANGE_G * G0) / 32768.0f;
      out[i].ay = ((float)ay) * (ACC_RANGE_G * G0) / 32768.0f;
      out[i].az = ((float)az) * (ACC_RANGE_G * G0) / 32768.0f;

      const float dps_to_rps = (float)M_PI / 180.0f;
      out[i].gx = ((float)gx) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
      out[i].gy = ((float)gy) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
      out[i].gz = ((float)gz) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
    }

    return n;
  }

  uint32_t skippedFramesTotal() const { return skipped_total_; }

private:
  TwoWire* wire_ = nullptr;
  uint8_t  bmi_addr_ = 0x68;

  bmi2_dev        bmi_{};
  bmi2_fifo_frame fifo_{};

  // FIFO buffers
  uint8_t fifo_buf_[2048 + 256] = {0};
  bmi2_sens_axes_data accel_[64] = {};
  bmi2_sens_axes_data gyro_[64]  = {};

  float odr_hz_ = 100.0f;
  float nominal_dt_ = 0.01f;

  bool have_stime_ = false;
  uint32_t last_stime_ = 0;

  uint32_t skipped_total_ = 0;

  // Bosch BMI2 I2C glue (signature matches SensorAPI style used by Arduino_BMI270_BMM150 too)
  static int8_t bmi2_i2c_read_(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_) return (int8_t)-1;

    self->wire_->beginTransmission(self->bmi_addr_);
    self->wire_->write(reg_addr);
    if (self->wire_->endTransmission(false) != 0) return (int8_t)-1;

    uint32_t got = self->wire_->requestFrom((int)self->bmi_addr_, (int)len);
    if (got != len) {
      for (uint32_t i = 0; i < got && self->wire_->available(); ++i) reg_data[i] = self->wire_->read();
      return (int8_t)-1;
    }
    for (uint32_t i = 0; i < len; ++i) reg_data[i] = (uint8_t)self->wire_->read();
    return BMI2_OK;
  }

  static int8_t bmi2_i2c_write_(uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr)
  {
    auto* self = static_cast<BoschBmi270Fifo*>(intf_ptr);
    if (!self || !self->wire_) return (int8_t)-1;

    self->wire_->beginTransmission(self->bmi_addr_);
    self->wire_->write(reg_addr);
    for (uint32_t i = 0; i < len; ++i) self->wire_->write(reg_data[i]);
    if (self->wire_->endTransmission(true) != 0) return (int8_t)-1;
    return BMI2_OK;
  }

  static void bmi2_delay_us_(uint32_t period, void * /*intf_ptr*/)
  {
    delayMicroseconds(period);
  }
};
