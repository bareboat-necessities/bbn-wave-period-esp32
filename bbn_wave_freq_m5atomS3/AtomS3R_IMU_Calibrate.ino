/*
  AtomS3R IMU calibration wizard (Accel + Gyro + Mag) using imu_cal::* calibrators.

  - Wipes M5Unified's built-in IMU offset calibration from NVS:
      M5.Imu.clearOffsetData();   (per M5Unified docs)   [oai_citation:1‡M5Stack Docs](https://docs.m5stack.com/en/arduino/m5unified/imu_class)
  - Stores our own calibration blob to NVS (Preferences, namespace "imu_cal")
  - Provides applyAccel/applyGyro/applyMag helpers for later use

  REQUIREMENTS:
    - M5Unified library
    - ESP32 Arduino core (Preferences)
    - ArduinoEigenDense
    - CalibrateIMU_alt.h

  Units:
    accel: m/s^2
    gyro:  rad/s
    mag:   uT
*/

#include <M5Unified.h>
#include <Arduino.h>
#include <Preferences.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#endif
#include <ArduinoEigenDense.h>

#include "CalibrateIMU_alt.h"   

using Vector3f = Eigen::Matrix<float,3,1>;
using Matrix3f = Eigen::Matrix<float,3,3>;

static constexpr float  g_std = 9.80665f;
static constexpr float  DEG2RAD = 3.14159265358979323846f / 180.0f;

// ----------------------------
// Axis mapping (MATCH YOUR EXISTING CODE PATH)
// Raw M5 data comes in device axes; you mapped to BODY-NED like:
//   acc_body = ( ay, ax, -az ) * g
//   gyr_body = ( gy, gx, -gz ) * deg2rad
//   mag_body = ( my, mx, -mz ) * (1/10)  // per your example
// Keep this consistent for BOTH calibration and later application.
// ----------------------------
static inline Vector3f map_acc_to_body_ned(const m5::imu_3d_t& a_g_units) {
  return Vector3f(
    a_g_units.y * g_std,
    a_g_units.x * g_std,
   -a_g_units.z * g_std
  );
}

static inline Vector3f map_gyr_to_body_ned(const m5::imu_3d_t& w_deg_s) {
  return Vector3f(
    w_deg_s.y * DEG2RAD,
    w_deg_s.x * DEG2RAD,
   -w_deg_s.z * DEG2RAD
  );
}

static inline Vector3f map_mag_to_body_uT(const m5::imu_3d_t& m_raw) {
  // Your reference code used /10.0f (common in M5 examples).
  return Vector3f(
    m_raw.y / 10.0f,
    m_raw.x / 10.0f,
   -m_raw.z / 10.0f
  );
}

// ----------------------------
// NVS blob (our own calibration)
// ----------------------------
static constexpr uint32_t IMU_CAL_MAGIC   = 0x434C554D; // 'MULC' (arbitrary)
static constexpr uint16_t IMU_CAL_VERSION = 1;

static uint32_t crc32_ieee(const uint8_t* data, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= (uint32_t)data[i];
    for (int k = 0; k < 8; ++k) {
      uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

struct CalBlobV1 {
  uint32_t magic = IMU_CAL_MAGIC;
  uint16_t version = IMU_CAL_VERSION;
  uint16_t size_bytes = sizeof(CalBlobV1);

  // accel
  uint8_t  accel_ok = 0;
  float    accel_g = g_std;
  float    accel_S[9];     // row-major 3x3
  float    accel_T0 = 25.0f;
  float    accel_b0[3];
  float    accel_k[3];
  float    accel_rms_mag = 0.0f;

  // gyro
  uint8_t  gyro_ok = 0;
  float    gyro_T0 = 25.0f;
  float    gyro_b0[3];
  float    gyro_k[3];

  // mag
  uint8_t  mag_ok = 0;
  float    mag_A[9];       // row-major 3x3
  float    mag_b[3];
  float    mag_field_uT = 0.0f;
  float    mag_rms = 0.0f;

  // crc at end (computed over everything BEFORE crc)
  uint32_t crc = 0;
};

static Preferences prefs;
static bool have_cal = false;
static CalBlobV1 cal_blob;

// Convert blob -> runtime cal objects
static imu_cal::AccelCalibration<float> g_acc_cal;
static imu_cal::GyroCalibration<float>  g_gyr_cal;
static imu_cal::MagCalibration<float>   g_mag_cal;

static Matrix3f mat_from_rowmajor9(const float a[9]) {
  Matrix3f M;
  M << a[0], a[1], a[2],
       a[3], a[4], a[5],
       a[6], a[7], a[8];
  return M;
}

static void mat_to_rowmajor9(const Matrix3f& M, float a[9]) {
  a[0]=M(0,0); a[1]=M(0,1); a[2]=M(0,2);
  a[3]=M(1,0); a[4]=M(1,1); a[5]=M(1,2);
  a[6]=M(2,0); a[7]=M(2,1); a[8]=M(2,2);
}

static void rebuild_runtime_cals_from_blob() {
  // accel
  g_acc_cal.ok = (cal_blob.accel_ok != 0);
  g_acc_cal.g  = cal_blob.accel_g;
  g_acc_cal.S  = mat_from_rowmajor9(cal_blob.accel_S);
  g_acc_cal.biasT.ok = g_acc_cal.ok;
  g_acc_cal.biasT.T0 = cal_blob.accel_T0;
  g_acc_cal.biasT.b0 = Vector3f(cal_blob.accel_b0[0], cal_blob.accel_b0[1], cal_blob.accel_b0[2]);
  g_acc_cal.biasT.k  = Vector3f(cal_blob.accel_k[0],  cal_blob.accel_k[1],  cal_blob.accel_k[2]);
  g_acc_cal.rms_mag  = cal_blob.accel_rms_mag;

  // gyro
  g_gyr_cal.ok = (cal_blob.gyro_ok != 0);
  g_gyr_cal.S  = Matrix3f::Identity();
  g_gyr_cal.biasT.ok = g_gyr_cal.ok;
  g_gyr_cal.biasT.T0 = cal_blob.gyro_T0;
  g_gyr_cal.biasT.b0 = Vector3f(cal_blob.gyro_b0[0], cal_blob.gyro_b0[1], cal_blob.gyro_b0[2]);
  g_gyr_cal.biasT.k  = Vector3f(cal_blob.gyro_k[0],  cal_blob.gyro_k[1],  cal_blob.gyro_k[2]);

  // mag
  g_mag_cal.ok = (cal_blob.mag_ok != 0);
  g_mag_cal.A  = mat_from_rowmajor9(cal_blob.mag_A);
  g_mag_cal.b  = Vector3f(cal_blob.mag_b[0], cal_blob.mag_b[1], cal_blob.mag_b[2]);
  g_mag_cal.field_uT = cal_blob.mag_field_uT;
  g_mag_cal.rms      = cal_blob.mag_rms;
}

static bool load_cal_from_nvs() {
  prefs.begin("imu_cal", true);
  size_t n = prefs.getBytesLength("blob");
  if (n != sizeof(CalBlobV1)) { prefs.end(); return false; }

  CalBlobV1 tmp;
  size_t got = prefs.getBytes("blob", &tmp, sizeof(tmp));
  prefs.end();
  if (got != sizeof(tmp)) return false;

  if (tmp.magic != IMU_CAL_MAGIC || tmp.version != IMU_CAL_VERSION || tmp.size_bytes != sizeof(CalBlobV1))
    return false;

  uint32_t want = tmp.crc;
  tmp.crc = 0;
  uint32_t have = crc32_ieee((const uint8_t*)&tmp, sizeof(tmp));
  if (have != want) return false;

  cal_blob = tmp;
  rebuild_runtime_cals_from_blob();
  return (g_acc_cal.ok || g_gyr_cal.ok || g_mag_cal.ok);
}

static bool save_cal_to_nvs(const CalBlobV1& blob_in) {
  CalBlobV1 tmp = blob_in;
  tmp.magic = IMU_CAL_MAGIC;
  tmp.version = IMU_CAL_VERSION;
  tmp.size_bytes = sizeof(CalBlobV1);
  tmp.crc = 0;
  tmp.crc = crc32_ieee((const uint8_t*)&tmp, sizeof(tmp));

  prefs.begin("imu_cal", false);
  bool ok = (prefs.putBytes("blob", &tmp, sizeof(tmp)) == sizeof(tmp));
  prefs.end();
  return ok;
}

static void erase_our_cal_from_nvs() {
  prefs.begin("imu_cal", false);
  prefs.remove("blob");
  prefs.end();
}

// ----------------------------
// Apply functions (DROP-INS)
// ----------------------------
static inline Vector3f applyAccel(const Vector3f& a_raw_body, float tempC) {
  if (!g_acc_cal.ok) return a_raw_body;
  return g_acc_cal.apply(a_raw_body, tempC);
}

static inline Vector3f applyGyro(const Vector3f& w_raw_body, float tempC) {
  if (!g_gyr_cal.ok) return w_raw_body;
  return g_gyr_cal.apply(w_raw_body, tempC);
}

static inline Vector3f applyMag(const Vector3f& m_raw_uT_body) {
  if (!g_mag_cal.ok) return m_raw_uT_body;
  return g_mag_cal.apply(m_raw_uT_body);
}

// ----------------------------
// Simple UI helpers
// ----------------------------
static auto& disp = M5.Display;

static void ui_clear(uint16_t bg = TFT_BLACK) {
  disp.fillScreen(bg);
}

static void ui_title(const char* s) {
  disp.setTextColor(TFT_WHITE, TFT_BLACK);
  disp.setCursor(0, 0);
  disp.setTextSize(2);
  disp.println(s);
  disp.setTextSize(1);
  disp.println();
}

static void ui_line(const char* s) {
  disp.setTextColor(TFT_WHITE, TFT_BLACK);
  disp.println(s);
}

static void ui_progress(const char* label, int done, int total) {
  if (total <= 0) total = 1;
  if (done < 0) done = 0;
  if (done > total) done = total;
  int w = disp.width() - 10;
  int x = 5;
  int y = disp.height() - 18;
  int h = 10;
  disp.fillRect(0, y - 14, disp.width(), 32, TFT_BLACK);
  disp.setCursor(0, y - 14);
  disp.setTextColor(TFT_CYAN, TFT_BLACK);
  disp.printf("%s %d/%d\n", label, done, total);
  disp.drawRect(x, y, w, h, TFT_DARKGREY);
  int fillw = (w - 2) * done / total;
  disp.fillRect(x + 1, y + 1, fillw, h - 2, TFT_GREEN);
}

static bool btn_pressed() {
  M5.update();
  return M5.BtnA.wasPressed();
}

// ----------------------------
// Calibration wizard
// ----------------------------
static imu_cal::AccelCalibrator<float, 400, 8> accelCal;
static imu_cal::GyroCalibrator<float, 400, 8>  gyroCal;
static imu_cal::MagCalibrator<float, 400>      magCal;

static bool read_sensors(Vector3f& a_raw_body, Vector3f& w_raw_body, Vector3f& m_raw_uT_body, float& tempC, uint32_t& mask_out) {
  mask_out = M5.Imu.update();
  if (!mask_out) return false;

  auto data = M5.Imu.getImuData();
  tempC = NAN;
  M5.Imu.getTemp(&tempC);

  a_raw_body     = map_acc_to_body_ned(data.accel);
  w_raw_body     = map_gyr_to_body_ned(data.gyro);
  m_raw_uT_body  = map_mag_to_body_uT(data.mag);
  return true;
}

// Collect N accepted accel samples (using AccelCalibrator::addSample gates) with timeout.
static bool collect_accel_samples(const char* prompt, int need, uint32_t timeout_ms) {
  ui_clear();
  ui_title("ACCEL CAL");
  ui_line(prompt);
  ui_line("Hold still...");
  ui_line("BtnA cancels");

  uint32_t t0 = millis();
  int got0 = accelCal.buf.n;
  while ((millis() - t0) < timeout_ms) {
    if (btn_pressed()) return false;

    Vector3f a, w, m;
    float tempC;
    uint32_t mask;
    if (!read_sensors(a, w, m, tempC, mask)) { delay(2); continue; }

    // feed calibrator (it decides if it accepts)
    accelCal.addSample(a, w, tempC);

    int got = accelCal.buf.n - got0;
    ui_progress("Samples", got, need);
    if (got >= need) return true;

    delay(8);
  }
  return false;
}

// Collect gyro samples (stationary) with timeout.
static bool collect_gyro_samples(uint32_t timeout_ms) {
  ui_clear();
  ui_title("GYRO CAL");
  ui_line("Place on desk, still.");
  ui_line("BtnA cancels");

  uint32_t t0 = millis();
  int got0 = gyroCal.buf.n;
  while ((millis() - t0) < timeout_ms) {
    if (btn_pressed()) return false;

    Vector3f a, w, m;
    float tempC;
    uint32_t mask;
    if (!read_sensors(a, w, m, tempC, mask)) { delay(2); continue; }

    gyroCal.addSample(w, a, tempC);

    int got = gyroCal.buf.n - got0;
    ui_progress("Samples", got, 200);
    if (got >= 200) return true;

    delay(8);
  }
  return false;
}

// Collect mag samples while rotating.
static bool collect_mag_samples(uint32_t timeout_ms) {
  ui_clear();
  ui_title("MAG CAL");
  ui_line("Rotate slowly in");
  ui_line("all directions.");
  ui_line("BtnA cancels");

  uint32_t t0 = millis();
  int got0 = magCal.buf.n;
  while ((millis() - t0) < timeout_ms) {
    if (btn_pressed()) return false;

    Vector3f a, w, m;
    float tempC;
    uint32_t mask;
    if (!read_sensors(a, w, m, tempC, mask)) { delay(2); continue; }

    // Only count when mag sensor actually updated (mask includes mag)
    if (mask & (uint32_t)decltype(M5.Imu)::sensor_mask_mag) {
      magCal.addSample(m);
    }

    int got = magCal.buf.n - got0;
    ui_progress("Samples", got, 250);
    if (got >= 250) return true;

    delay(8);
  }
  return false;
}

static void show_fit_results(const imu_cal::AccelCalibration<float>& ac,
                             const imu_cal::GyroCalibration<float>&  gc,
                             const imu_cal::MagCalibration<float>&   mc)
{
  ui_clear();
  ui_title("CAL DONE");

  disp.setTextColor(TFT_WHITE, TFT_BLACK);
  disp.printf("ACC ok:%d  rms|g|:%.4f\n", (int)ac.ok, (double)ac.rms_mag);
  disp.printf("GYR ok:%d\n", (int)gc.ok);
  disp.printf("MAG ok:%d  B:%.1f uT  rms:%.2f\n", (int)mc.ok, (double)mc.field_uT, (double)mc.rms);

  disp.println();
  disp.println("BtnA to exit");
  while (true) {
    M5.update();
    if (M5.BtnA.wasPressed()) break;
    delay(10);
  }
}

static bool run_calibration_wizard() {
  // 1) wipe M5 offsets from NVS + disable M5 internal calibration
  // clearOffsetData() clears IMU offset values in NVS per docs  [oai_citation:2‡M5Stack Docs](https://docs.m5stack.com/en/arduino/m5unified/imu_class)
  M5.Imu.setCalibration(0, 0, 0);
  M5.Imu.clearOffsetData();

  // 2) reset our calibrators
  accelCal.clear();
  gyroCal.clear();
  magCal.clear();

  // 3) ACCEL: 6 faces (just instructions; your robust fit needs coverage)
  // We grab ~60 accepted samples per face => ~360 total (fits within 400 max).
  const struct { const char* msg; } faces[6] = {
    {"Face 1/6: screen UP\n(hold still)"},
    {"Face 2/6: screen DOWN\n(hold still)"},
    {"Face 3/6: USB DOWN\n(hold still)"},
    {"Face 4/6: USB UP\n(hold still)"},
    {"Face 5/6: left side DOWN\n(hold still)"},
    {"Face 6/6: right side DOWN\n(hold still)"},
  };

  for (int i = 0; i < 6; ++i) {
    // small pause so user can reposition
    ui_clear();
    ui_title("ACCEL CAL");
    ui_line(faces[i].msg);
    ui_line("BtnA cancels");
    ui_line("Starting in 2s...");
    uint32_t t0 = millis();
    while (millis() - t0 < 2000) {
      if (btn_pressed()) return false;
      delay(10);
    }

    if (!collect_accel_samples(faces[i].msg, /*need=*/60, /*timeout_ms=*/12000))
      return false;
  }

  imu_cal::AccelCalibration<float> acc_out;
  if (!accelCal.fit(acc_out, /*robust_iters=*/3, /*trim_frac=*/0.15f) || !acc_out.ok) {
    ui_clear();
    ui_title("ACCEL FAIL");
    ui_line("Not enough coverage.");
    ui_line("Try slower moves /");
    ui_line("more distinct faces.");
    delay(2500);
    return false;
  }

  // 4) GYRO
  if (!collect_gyro_samples(/*timeout_ms=*/15000)) return false;

  imu_cal::GyroCalibration<float> gyr_out;
  if (!gyroCal.fit(gyr_out) || !gyr_out.ok) {
    ui_clear();
    ui_title("GYRO FAIL");
    ui_line("Too much motion?");
    delay(2500);
    return false;
  }

  // 5) MAG
  if (!collect_mag_samples(/*timeout_ms=*/25000)) return false;

  imu_cal::MagCalibration<float> mag_out;
  if (!magCal.fit(mag_out, /*robust_iters=*/3, /*trim_frac=*/0.15f) || !mag_out.ok) {
    ui_clear();
    ui_title("MAG FAIL");
    ui_line("Try larger rotations,");
    ui_line("away from metal.");
    delay(2500);
    return false;
  }

  // 6) Save blob
  CalBlobV1 blob;
  // accel
  blob.accel_ok = acc_out.ok ? 1 : 0;
  blob.accel_g = acc_out.g;
  mat_to_rowmajor9(acc_out.S, blob.accel_S);
  blob.accel_T0 = acc_out.biasT.T0;
  blob.accel_b0[0]=acc_out.biasT.b0.x(); blob.accel_b0[1]=acc_out.biasT.b0.y(); blob.accel_b0[2]=acc_out.biasT.b0.z();
  blob.accel_k[0]=acc_out.biasT.k.x();   blob.accel_k[1]=acc_out.biasT.k.y();   blob.accel_k[2]=acc_out.biasT.k.z();
  blob.accel_rms_mag = acc_out.rms_mag;

  // gyro
  blob.gyro_ok = gyr_out.ok ? 1 : 0;
  blob.gyro_T0 = gyr_out.biasT.T0;
  blob.gyro_b0[0]=gyr_out.biasT.b0.x(); blob.gyro_b0[1]=gyr_out.biasT.b0.y(); blob.gyro_b0[2]=gyr_out.biasT.b0.z();
  blob.gyro_k[0]=gyr_out.biasT.k.x();   blob.gyro_k[1]=gyr_out.biasT.k.y();   blob.gyro_k[2]=gyr_out.biasT.k.z();

  // mag
  blob.mag_ok = mag_out.ok ? 1 : 0;
  mat_to_rowmajor9(mag_out.A, blob.mag_A);
  blob.mag_b[0]=mag_out.b.x(); blob.mag_b[1]=mag_out.b.y(); blob.mag_b[2]=mag_out.b.z();
  blob.mag_field_uT = mag_out.field_uT;
  blob.mag_rms = mag_out.rms;

  if (!save_cal_to_nvs(blob)) {
    ui_clear();
    ui_title("SAVE FAIL");
    ui_line("NVS write failed.");
    delay(2500);
    return false;
  }

  // install as active
  cal_blob = blob;
  rebuild_runtime_cals_from_blob();
  have_cal = true;

  show_fit_results(acc_out, gyr_out, mag_out);
  return true;
}

// ----------------------------
// Runtime demo loop
// ----------------------------
static uint32_t last_print_ms = 0;

static void draw_status() {
  ui_clear();
  ui_title("AtomS3R IMU CAL");

  disp.printf("Our cal: %s\n", have_cal ? "LOADED" : "NONE");
  disp.println();
  disp.println("BtnA: run calibration");
  disp.println("Hold BtnA ~2s: erase cal");
  disp.println();
  disp.println("Serial: raw+cal values");
}

static bool btnA_longpress_erase() {
  // crude long-press detector
  static bool was_down = false;
  static uint32_t t_down = 0;
  bool down = M5.BtnA.isPressed();
  if (down && !was_down) { was_down = true; t_down = millis(); }
  if (!down && was_down) { was_down = false; }
  if (down && was_down && (millis() - t_down) > 2000) return true;
  return false;
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);

  // Optional: force a known rotation/orientation for display
  // disp.setRotation(0);

  // Disable M5 internal calibration influence (we apply ours in software)
  M5.Imu.setCalibration(0, 0, 0);

  have_cal = load_cal_from_nvs();
  draw_status();

  if (!M5.Imu.isEnabled()) {
    ui_clear();
    ui_title("IMU FAIL");
    ui_line("IMU not found.");
    while (true) delay(100);
  }
}

void loop() {
  M5.update();

  // Erase our cal + wipe M5 offsets
  if (btnA_longpress_erase()) {
    ui_clear();
    ui_title("ERASING...");
    erase_our_cal_from_nvs();
    M5.Imu.clearOffsetData();   // wipe M5 offsets in NVS  [oai_citation:3‡M5Stack Docs](https://docs.m5stack.com/en/arduino/m5unified/imu_class)
    have_cal = false;
    memset(&cal_blob, 0, sizeof(cal_blob));
    rebuild_runtime_cals_from_blob();
    ui_line("Done.");
    delay(800);
    draw_status();
  }

  // Run wizard
  if (M5.BtnA.wasPressed()) {
    bool ok = run_calibration_wizard();
    (void)ok;
    draw_status();
  }

  // live read/print
  Vector3f a_raw, w_raw, m_raw;
  float tempC;
  uint32_t mask;
  if (read_sensors(a_raw, w_raw, m_raw, tempC, mask)) {
    Vector3f a_cal = applyAccel(a_raw, tempC);
    Vector3f w_cal = applyGyro(w_raw, tempC);
    Vector3f m_cal = applyMag(m_raw);

    uint32_t now = millis();
    if (now - last_print_ms > 200) {
      last_print_ms = now;

      Serial.printf("tempC:%.2f,", (double)tempC);

      Serial.printf("a_raw:%+.4f,%+.4f,%+.4f,",
        (double)a_raw.x(), (double)a_raw.y(), (double)a_raw.z());
      Serial.printf("a_cal:%+.4f,%+.4f,%+.4f,|a|:%.4f,",
        (double)a_cal.x(), (double)a_cal.y(), (double)a_cal.z(), (double)a_cal.norm());

      Serial.printf("w_raw:%+.5f,%+.5f,%+.5f,",
        (double)w_raw.x(), (double)w_raw.y(), (double)w_raw.z());
      Serial.printf("w_cal:%+.5f,%+.5f,%+.5f,",
        (double)w_cal.x(), (double)w_cal.y(), (double)w_cal.z());

      Serial.printf("m_raw:%+.2f,%+.2f,%+.2f,",
        (double)m_raw.x(), (double)m_raw.y(), (double)m_raw.z());
      Serial.printf("m_cal:%+.2f,%+.2f,%+.2f,|m|:%.2f",
        (double)m_cal.x(), (double)m_cal.y(), (double)m_cal.z(), (double)m_cal.norm());

      Serial.println();
    }
  }

  delay(5);
}
