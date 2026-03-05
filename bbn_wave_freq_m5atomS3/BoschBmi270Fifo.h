#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <cmath>
#include <cstdint>

// for use BMM150 vendor API:
#include <Arduino_BMI270_BMM150>

// If BMI270 sensortime resolution differs, change here.
// BMI-family sensors commonly use 39.0625 us ticks (2^(-8) ms).
static constexpr float BMI270_SENSORTIME_TICK_S = 39.0625e-6f;  // 39.0625 µs per tick 
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

    // BMI270 (Bosch vendor API) init
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

    // Map requested ODR to the closest macro
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

    // FIFO reads require disabling advanced power save. 
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

  bool readOneAG(BoschAGSample& out) {
    BoschAGSample tmp[1];
    const int n = readAG(tmp, 1);
    if (n <= 0) return false;
    out = tmp[0];
    return true;
  }

  // Read and extract up to max_out samples.
  // Returns number of samples written.
  int readAG(BoschAGSample* out, int max_out)
  {
    if (!out || max_out <= 0) return 0;
  
    // Ensure FIFO returns sensortime frame (fifo_time_en)
    // "Return sensortime frame after the last valid data frame." :contentReference[oaicite:4]{index=4}
    #ifdef BMI2_FIFO_TIME_EN
      (void)bmi2_set_fifo_config(BMI2_FIFO_TIME_EN, BMI2_ENABLE, &bmi_);
    #endif
  
    const float dt_nom = nominal_dt_;
    const uint32_t st_prev = have_stime_ ? last_stime_ : 0;
  
    int total = 0;
    uint32_t st_last_seen = st_prev;
    bool saw_new_stime = false;
  
    // Drain FIFO to empty so sensortime frame is actually emitted. :contentReference[oaicite:5]{index=5}
    while (total < max_out) {
      uint16_t fifo_len = 0;
      if (bmi2_get_fifo_length(&fifo_len, &bmi_) != BMI2_OK) break;
      if (fifo_len < 16) break; // effectively empty
  
      constexpr uint16_t OVERREAD = 220;
      uint16_t req = fifo_len + OVERREAD + bmi_.dummy_byte;
      if (req > sizeof(fifo_buf_)) req = sizeof(fifo_buf_);
  
      fifo_.data = fifo_buf_;
      fifo_.length = req;
  
      if (bmi2_read_fifo_data(&fifo_, &bmi_) != BMI2_OK) break;
  
      // Extract into temp arrays
      uint16_t a_len = (uint16_t)std::min<int>(64, max_out - total);
      uint16_t g_len = (uint16_t)std::min<int>(64, max_out - total);
  
      (void)bmi2_extract_accel(accel_, &a_len, &fifo_, &bmi_);
      (void)bmi2_extract_gyro (gyro_,  &g_len, &fifo_, &bmi_);
  
      const int n = (int)std::min(a_len, g_len);
      if (n <= 0) {
        // Even if no samples extracted, continue draining to reach FIFO-empty/sensortime.
        skipped_total_ += fifo_.skipped_frame_count;
        continue;
      }
  
      skipped_total_ += fifo_.skipped_frame_count;
  
      // Capture sensortime parsed from FIFO (only valid when FIFO drained to empty at some point)
      const uint32_t st = ((uint32_t)fifo_.sensor_time) & BMI270_SENSORTIME_MASK;
      if (have_stime_) {
        if (st != 0 && st != st_last_seen) { st_last_seen = st; saw_new_stime = true; }
      } else {
        if (st != 0) { st_last_seen = st; saw_new_stime = true; }
      }
  
      // Convert raw -> physical (same as your previous conversion)
      constexpr float ACC_RANGE_G = 2.0f;
      constexpr float GYR_RANGE_DPS = 2000.0f;
      constexpr float G0 = 9.80665f;
      const float dps_to_rps = (float)M_PI / 180.0f;
  
      for (int i = 0; i < n; ++i) {
        out[total + i].dt_s = dt_nom; // temp; we overwrite after we know dt from sensortime
        out[total + i].ax = ((float)accel_[i].x) * (ACC_RANGE_G * G0) / 32768.0f;
        out[total + i].ay = ((float)accel_[i].y) * (ACC_RANGE_G * G0) / 32768.0f;
        out[total + i].az = ((float)accel_[i].z) * (ACC_RANGE_G * G0) / 32768.0f;
  
        out[total + i].gx = ((float)gyro_[i].x) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
        out[total + i].gy = ((float)gyro_[i].y) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
        out[total + i].gz = ((float)gyro_[i].z) * (GYR_RANGE_DPS * dps_to_rps) / 32768.0f;
      }
  
      total += n;
    }
  
    // Compute dt from FIFO sensortime across the whole drained batch.
    // Sensortime resolution is 39.0625us. :contentReference[oaicite:6]{index=6}
    if (total > 0) {
      if (!have_stime_) {
        have_stime_ = saw_new_stime;
        last_stime_ = st_last_seen;
        // keep nominal dt on first batch
        return total;
      }
  
      if (saw_new_stime) {
        const uint32_t d_ticks = (st_last_seen - last_stime_) & BMI270_SENSORTIME_MASK;
        last_stime_ = st_last_seen;
  
        const float dt_total = (float)d_ticks * BMI270_SENSORTIME_TICK_S;
        float dt_per = dt_total / (float)total;
  
        // sanity
        if (!(dt_per > 0.0f) || dt_per < 0.0005f) dt_per = 0.0005f;
        if (dt_per > 0.0500f) dt_per = 0.0500f;
  
        for (int i = 0; i < total; ++i) out[i].dt_s = dt_per;
      } else {
        // No new FIFO sensortime observed (should be rare if we truly drained to empty).
        // Fall back to nominal dt to avoid bursty dt jitter.
        for (int i = 0; i < total; ++i) out[i].dt_s = nominal_dt_;
      }
    }
  
    return total;
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
