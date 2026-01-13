#pragma once

/*
  Copyright 2026, Mikhail Grushinskiy
  
  AtomS3R IMU calibration wizard UI (Accel + Gyro + Mag) using imu_cal::* calibrators.

  Depends on:
    - AtomS3R_ImuCal.h (blob/store/runtime + mapping/read + clearM5UnifiedImuCalibration())

  Typical usage (boot flow):

    #include <M5Unified.h>
    #include "AtomS3R_ImuCal.h"
    #include "AtomS3R_ImuCalWizard.h"

    atoms3r_ical::ImuCalStoreNvs    store;
    atoms3r_ical::ImuCalBlobV1      blob;
    atoms3r_ical::RuntimeCals       cals;
    atoms3r_ical::M5Ui              ui;
    atoms3r_ical::ImuCalWizard      wiz(ui, store);

    void setup() {
      Serial.begin(115200);
      delay(150);
      Serial.println();

      auto cfg = M5.config();
      M5.begin(cfg);

      // Clear M5Unified cal so it doesn't stack with ours.
      atoms3r_ical::clearM5UnifiedImuCalibration();

      ui.begin();

      if (!M5.Imu.isEnabled()) { Serial.println("IMU missing"); while(true) delay(100); }

      bool have = store.load(blob);
      if (have) {
        Serial.println("[BOOT] Found saved calibration:");
        atoms3r_ical::printBlobSummary(Serial, blob);
        atoms3r_ical::printBlobDetail(Serial, blob);
        cals.rebuildFromBlob(blob);
      } else {
        Serial.println("[BOOT] No calibration -> starting wizard");
        atoms3r_ical::ImuCalBlobV1 out{};
        if (wiz.runAndSave(out)) {
          Serial.println("[BOOT] Wizard saved calibration:");
          atoms3r_ical::printBlobSummary(Serial, out);
          cals.rebuildFromBlob(out);
        } else {
          Serial.println("[BOOT] Wizard failed or not saved");
        }
      }
    }

*/

#include <Arduino.h>
#include <M5Unified.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "AtomS3R_ImuCal.h"   // blob/store/read/mapping/clearM5UnifiedImuCalibration()
#include "CalibrateIMU.h"     // imu_cal::* + FitFail

namespace atoms3r_ical {

// Wizard configuration
struct ImuCalWizardCfg {
  // If text rotated wrong initially, change this to 1 or 3.
  static constexpr uint8_t ROT_READ = 0;
  static constexpr uint32_t KEEP_AWAKE_EVERY_MS = 200;

  // Step pacing
  static constexpr uint32_t PLACE_TIME_MS       = 6500;
  static constexpr uint32_t OK_PAUSE_MS         = 900;
  static constexpr uint32_t ACCEL_TIMEOUT_MS    = 90000;
  static constexpr uint32_t GYRO_TIMEOUT_MS     = 70000;
  static constexpr uint32_t MAG_TIMEOUT_MS      = 220000;
  static constexpr uint32_t STUCK_MS            = 12000;

  // Sample goals (Mag buffer capacity is 400 below)
  static constexpr int ACCEL_NEED_PER_POSE      = 60;
  static constexpr int GYRO_NEED                = 220;

  // MAG: target near-buffer-full, but spread over time
  static constexpr int MAG_NEED                 = 360;   // <= 400
  static constexpr int MAG_MIN_TO_FIT           = 220;

  // Minimums before fitting
  static constexpr int ACCEL_MIN_TO_FIT         = 220;
  static constexpr int GYRO_MIN_TO_FIT          = 120;

  // MAG timing / downsample:
  static constexpr uint32_t MAG_SAMPLE_SPACING_MS = 80;     // accepted max ~12.5 Hz
  static constexpr uint32_t MAG_MIN_TIME_MS       = 45000;  // require >= 45 seconds of motion

  // MAG stale-repeat reject
  static constexpr float MAG_MIN_DELTA_uT         = 0.03f;

  // MAG coverage requirements (ratio-based, unitless)
  static constexpr float MAG_SPAN_MIN_FRAC        = 0.35f;
  static constexpr float MAG_SPAN_MID_FRAC        = 0.55f;

  // Also require centered direction range per axis (0..2). 1.2 means roughly reaching ±0.6.
  static constexpr float MAG_URANGE_TARGET        = 1.05f;

  static constexpr uint32_t FIT_STACK_WORDS     = 16384;
  static constexpr uint32_t FIT_TIMEOUT_MS      = 30000;

  static constexpr uint32_t TAP_WINDOW_MS       = 700;
  static constexpr uint8_t  LCD_BRIGHTNESS      = 200;
  static constexpr uint32_t MENU_TAP_WINDOW_MS  = 650;
};

// Small helpers
static inline int32_t i32_max_(int32_t a, int32_t b) { return (a > b) ? a : b; }

static inline uint8_t rot_add_(uint8_t base, int delta) {
  int r = (int)base + delta;
  r %= 4;
  if (r < 0) r += 4;
  return (uint8_t)r;
}

static inline float clamp01_(float x) {
  if (x < 0.f) return 0.f;
  if (x > 1.f) return 1.f;
  return x;
}

static inline float second_smallest3_(float a, float b, float c) {
  // median of 3 (2nd smallest)
  if (a > b) { float t=a; a=b; b=t; }
  if (b > c) { float t=b; b=c; c=t; }
  if (a > b) { float t=a; a=b; b=t; }
  return b;
}

static inline bool finite3_(const Vector3f& v) {
  return isfinite(v.x()) && isfinite(v.y()) && isfinite(v.z());
}

// "Planarity" coverage measure for MAG capture (determinant of unit-direction covariance).
static inline float unit_dir_cov_det_(const Vector3f* x, int n) {
  if (!x || n < 20) return 0.0f;

  // Mean-center
  Vector3f mu = Vector3f::Zero();
  int n_mu = 0;
  for (int i = 0; i < n; ++i) {
    if (!finite3_(x[i])) continue;
    mu += x[i];
    ++n_mu;
  }
  if (n_mu < 20) return 0.0f;
  mu *= 1.0f / (float)n_mu;

  // C = mean(u u^T), u = (x-mu)/||x-mu||
  Matrix3f C = Matrix3f::Zero();
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (!finite3_(x[i])) continue;
    Vector3f d = x[i] - mu;
    float dn = d.norm();
    if (!(dn > 1e-6f)) continue;
    Vector3f u = d / dn;
    if (!finite3_(u)) continue;
    C.noalias() += u * u.transpose();
    ++m;
  }
  if (m < 20) return 0.0f;

  C *= 1.0f / (float)m;
  float detC = C.determinant();
  if (!isfinite(detC)) return 0.0f;
  return detC;
}

// Input: BtnA ONLY + keep-awake
class Input {
public:
  static void update() {
    M5.update();

    uint32_t now = millis();
    if ((uint32_t)(now - last_keep_awake_ms_) > ImuCalWizardCfg::KEEP_AWAKE_EVERY_MS) {
      last_keep_awake_ms_ = now;
      M5.Display.setBrightness(ImuCalWizardCfg::LCD_BRIGHTNESS);
      M5.Display.wakeup();
    }
    tap_edge_ = M5.BtnA.wasPressed();
  }

  static bool tapPressed() { return tap_edge_; }

private:
  static inline bool tap_edge_ = false;
  static inline uint32_t last_keep_awake_ms_ = 0;
};

// UI helpers
class M5Ui {
public:
  void begin() {
    rot_ = ImuCalWizardCfg::ROT_READ;
    M5.Display.setRotation(rot_);
    M5.Display.setBrightness(ImuCalWizardCfg::LCD_BRIGHTNESS);
    M5.Display.setTextSize(1);
  }

  void setRotation(uint8_t r) {
    rot_ = (uint8_t)(r & 3);
    M5.Display.setRotation(rot_);
  }

  void setReadRotation() { setRotation(ImuCalWizardCfg::ROT_READ); }

  void clear() {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0,0);
  }

  void title(const char* t) {
    clear();
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.setTextSize(2);
    M5.Display.println(t);
    M5.Display.setTextSize(1);
    M5.Display.println();
  }

  void line(const char* s) {
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.println(s);
  }

  void bar01(float t01) {
    t01 = clamp01_(t01);
    int x = 6;
    int w = M5.Display.width() - 12;
    int h = 10;
    int y = M5.Display.height() - 18;
    M5.Display.drawRect(x, y, w, h, TFT_DARKGREY);
    int fillw = (int)((w-2) * t01);
    M5.Display.fillRect(x+1, y+1, fillw, h-2, TFT_GREEN);
  }

  void waitTap(const char* t, const char* l1=nullptr, const char* l2=nullptr) {
    setReadRotation();
    title(t);
    if (l1) line(l1);
    if (l2) line(l2);
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  void showOkAuto(const char* l1=nullptr, const char* l2=nullptr) {
    title("OK");
    if (l1) line(l1);
    if (l2) line(l2);
    uint32_t t0 = millis();
    while (millis() - t0 < ImuCalWizardCfg::OK_PAUSE_MS) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  void fail(const char* where, const char* why) {
    setReadRotation();
    title("FAILED");
    line(where);
    line(why);
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

  bool eraseConfirm() {
    setReadRotation();
    title("ERASE?");
    line("Delete saved cal");
    line("Tap=YES  Wait=NO");
    uint32_t t0 = millis();
    while (millis() - t0 < 4500) {
      Input::update();
      if (Input::tapPressed()) return true;
      delay(10);
    }
    return false;
  }

  enum class MagFailAction : uint8_t { RETRY_MAG=0, REDO_ALL=1, ABORT=2 };

  // NO TIMEOUT 
  MagFailAction magFailMenu(const char* why1, const char* why2=nullptr) {
    setReadRotation();
    title("MAG FAIL");
    if (why1) line(why1);
    if (why2) line(why2);
    line("");
    line("Tap: retry MAG");
    line("Tap x2: redo ALL");
    line("Tap x3: abort");

    uint8_t taps = waitTapGroupNoTimeout_(ImuCalWizardCfg::MENU_TAP_WINDOW_MS);
    if (taps >= 3) return MagFailAction::ABORT;
    if (taps == 2) return MagFailAction::REDO_ALL;
    return MagFailAction::RETRY_MAG;
  }

  void notSavedNotice() {
    setReadRotation();
    title("NOT SAVED");
    line("Calibration not saved");
    line("See Serial log");
    line("");
    line("Tap BtnA");
    while (true) {
      Input::update();
      if (Input::tapPressed()) break;
      delay(10);
    }
  }

private:
  static uint8_t waitTapGroupNoTimeout_(uint32_t window_ms) {
    uint8_t count = 0;
    uint32_t deadline = 0;

    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        count = 1;
        deadline = millis() + window_ms;
        break;
      }
      delay(10);
    }

    while (true) {
      Input::update();
      if (Input::tapPressed()) {
        count++;
        deadline = millis() + window_ms;
      }
      if ((int32_t)(millis() - deadline) > 0) break;
      delay(10);
    }
    return count;
  }

  uint8_t rot_ = ImuCalWizardCfg::ROT_READ;
};

// Wizard
struct Pose {
  const char* short_name;
  const char* instruction;
  uint8_t rot_capture;
};

class ImuCalWizard {
public:
  ImuCalWizard(M5Ui& ui, ImuCalStoreNvs& store) : ui_(ui), store_(store) {}

  // Runs full wizard and saves to NVS. Returns true only if saved successfully.
  // If out_saved is provided, it is filled with the saved blob (readback-validated).
  bool runAndSave(ImuCalBlobV1& out_saved) {
    Serial.println("[WIZ] start");

    for (;;) {
      bool redo_all = false;

      // Prevent "stacking" with any M5Unified offsets
      clearM5UnifiedImuCalibration();

      accelCal_.clear();
      gyroCal_.clear();
      magCal_.clear();
      configureCalibrators_();

      acc_out_ = imu_cal::AccelCalibration<float>{};
      gyr_out_ = imu_cal::GyroCalibration<float>{};
      mag_out_ = imu_cal::MagCalibration<float>{};

      ui_.waitTap("IMU CAL", "Tap to begin");

      const uint8_t R = ImuCalWizardCfg::ROT_READ;
      const Pose poses[6] = {
        {"1/6 SCREEN UP",    "Screen faces up",      R},
        {"2/6 SCREEN DOWN",  "Screen faces table",   R},
        {"3/6 USB UP",       "USB points up",        rot_add_(R, 2)},
        {"4/6 USB DOWN",     "USB points down",      R},
        {"5/6 LEFT DOWN",    "Left edge down",       rot_add_(R, 1)},
        {"6/6 RIGHT DOWN",   "Right edge down",      rot_add_(R, -1)},
      };

      for (int i = 0; i < 6; ++i) {
        if (!captureAccelPose_(poses[i])) return false;
      }

      Serial.printf("[ACC] accepted=%d\n", accelCal_.buf.n);
      if (accelCal_.buf.n < ImuCalWizardCfg::ACCEL_MIN_TO_FIT) {
        ui_.fail("ACCEL", "Too few accepted");
        return false;
      }
      if (!runFitTask_(FitKind::ACCEL, "ACCEL", true)) return false;

      if (!captureGyro_()) return false;
      if (gyroCal_.buf.n < ImuCalWizardCfg::GYRO_MIN_TO_FIT) {
        ui_.fail("GYRO", "Too few accepted");
        return false;
      }
      if (!runFitTask_(FitKind::GYRO, "GYRO", true)) return false;

      // MAG stage with retry loop
      while (true) {
        magCal_.clear();

        const char* cap_why = nullptr;
        if (!captureMag_(cap_why)) {
          auto act = ui_.magFailMenu(cap_why ? cap_why : "Capture failed",
                                     "Flip + roll + pitch");
          if (act == M5Ui::MagFailAction::RETRY_MAG) continue;
          if (act == M5Ui::MagFailAction::REDO_ALL) { redo_all = true; break; }
          return false;
        }

        if (magCal_.buf.n < ImuCalWizardCfg::MAG_MIN_TO_FIT) {
          auto act = ui_.magFailMenu("Too few accepted",
                                     "Rotate longer / slower");
          if (act == M5Ui::MagFailAction::RETRY_MAG) continue;
          if (act == M5Ui::MagFailAction::REDO_ALL) { redo_all = true; break; }
          return false;
        }

        if (!runFitTask_(FitKind::MAG, "MAG", false)) {
          const char* why = imu_cal::fitFailStr(fit_.reason);
          const char* hint = "Try bigger 3D motion";

          if (why && (strstr(why, "NONPOSITIVE") || strstr(why, "NON POSITIVE") || strstr(why, "MODEL_S"))) {
            hint = "Too planar: flip all faces";
          }

          auto act = ui_.magFailMenu(why ? why : "Fit failed", hint);
          if (act == M5Ui::MagFailAction::RETRY_MAG) continue;
          if (act == M5Ui::MagFailAction::REDO_ALL) { redo_all = true; break; }
          return false;
        }

        break; // MAG succeeded
      }

      if (redo_all) {
        Serial.println("[WIZ] redo all requested");
        continue;
      }

      // Build blob from fitted calibrations
      ImuCalBlobV1 blob{};
      fillBlob_(blob);

      ui_.setReadRotation();
      ui_.title("SAVE");
      ui_.line("Writing...");
      bool wrote = store_.save(blob);

      // Readback validation (also ensures CRC correctness)
      ImuCalBlobV1 rb{};
      bool okrb = store_.load(rb);

      Serial.printf("[SAVE] wrote=%d readback=%d\n", (int)wrote, (int)okrb);

      if (!wrote || !okrb) {
        ui_.fail("SAVE", "Write/readback fail");
        return false;
      }

      out_saved = rb;

      ui_.title("DONE");
      M5.Display.printf("A:%d G:%d M:%d\n", (int)rb.accel_ok, (int)rb.gyro_ok, (int)rb.mag_ok);
      ui_.line("Saved OK");
      ui_.line("");
      ui_.line("Tap BtnA");
      while (true) { Input::update(); if (Input::tapPressed()) break; delay(10); }

      Serial.println("[WIZ] done");
      return true;
    }
  }

private:
  // FIT task machinery
  enum class FitKind : uint8_t { ACCEL=0, GYRO=1, MAG=2 };

  struct FitCtx {
    volatile bool done = false;
    volatile bool ok   = false;
    imu_cal::FitFail reason = imu_cal::FitFail::BAD_ARG;
    FitKind kind = FitKind::ACCEL;

    ImuCalWizard* wiz = nullptr;
    TaskHandle_t  task = nullptr;
  } fit_{};

  static void fitTaskAccel_(void* p) {
    FitCtx* ctx = (FitCtx*)p;
    ImuCalWizard* self = ctx->wiz;

    ctx->reason = imu_cal::FitFail::BAD_ARG;
    bool ok = self->accelCal_.fit(self->acc_out_, 3, 0.15f, &ctx->reason);

    Serial.printf("[ACC] fit=%d ok=%d reason=%s\n",
                  (int)ok, (int)self->acc_out_.ok, imu_cal::fitFailStr(ctx->reason));

    ctx->ok = ok && self->acc_out_.ok;
    ctx->done = true;
    vTaskDelete(nullptr);
  }

  static void fitTaskGyro_(void* p) {
    FitCtx* ctx = (FitCtx*)p;
    ImuCalWizard* self = ctx->wiz;

    ctx->reason = imu_cal::FitFail::BAD_ARG;
    bool ok = self->gyroCal_.fit(self->gyr_out_, &ctx->reason);

    Serial.printf("[GYR] fit=%d ok=%d reason=%s\n",
                  (int)ok, (int)self->gyr_out_.ok, imu_cal::fitFailStr(ctx->reason));

    ctx->ok = ok && self->gyr_out_.ok;
    ctx->done = true;
    vTaskDelete(nullptr);
  }

  static void fitTaskMag_(void* p) {
    FitCtx* ctx = (FitCtx*)p;
    ImuCalWizard* self = ctx->wiz;

    // Try a few combinations (helps with MODEL_S_NONPOSITIVE sometimes)
    struct Try { int iters; float trim; float ridge; };
    const Try tries[] = {
      {3, 0.15f, 1e-6f},
      {3, 0.15f, 3e-6f},
      {3, 0.15f, 1e-5f},
      {3, 0.08f, 1e-6f},
      {2, 0.15f, 1e-6f},
    };

    imu_cal::FitFail last_reason = imu_cal::FitFail::BAD_ARG;
    bool any_ok = false;

    for (size_t i = 0; i < sizeof(tries)/sizeof(tries[0]); ++i) {
      imu_cal::FitFail r = imu_cal::FitFail::BAD_ARG;
      bool ok = self->magCal_.fit(self->mag_out_, tries[i].iters, tries[i].trim, tries[i].ridge, &r);
      
      Serial.printf("[MAG] try iters=%d trim=%.3f ridge=%.1e -> fit=%d out.ok=%d reason=%s\n",
                   tries[i].iters, (double)tries[i].trim, (double)tries[i].ridge,
                   (int)ok, (int)self->mag_out_.ok, imu_cal::fitFailStr(r));
      
      last_reason = r;
      if (ok && self->mag_out_.ok) { any_ok = true; break; }
    }

    ctx->reason = last_reason;
    ctx->ok = any_ok;
    ctx->done = true;
    vTaskDelete(nullptr);
  }

  bool runFitTask_(FitKind kind, const char* what, bool show_fail) {
    fit_.done = false;
    fit_.ok = false;
    fit_.reason = imu_cal::FitFail::BAD_ARG;
    fit_.kind = kind;
    fit_.wiz = this;
    fit_.task = nullptr;

    ui_.setReadRotation();
    ui_.title("FIT");
    ui_.line(what);
    ui_.line("Working...");

    TaskFunction_t fn = nullptr;
    switch (kind) {
      case FitKind::ACCEL: fn = &ImuCalWizard::fitTaskAccel_; break;
      case FitKind::GYRO:  fn = &ImuCalWizard::fitTaskGyro_;  break;
      case FitKind::MAG:   fn = &ImuCalWizard::fitTaskMag_;   break;
      default:             fn = &ImuCalWizard::fitTaskGyro_;  break;
    }

    // Pin FIT to core 0 (avoid starving loopTask on core 1)
    BaseType_t rc = xTaskCreatePinnedToCore(
      fn,
      what,
      (uint32_t)ImuCalWizardCfg::FIT_STACK_WORDS,
      &fit_,
      1,
      &fit_.task,
      0
    );

    if (rc != pdPASS || fit_.task == nullptr) {
      if (show_fail) ui_.fail("FIT", "Task create failed");
      return false;
    }

    uint32_t t0 = millis();
    float ph = 0.f;

    while (!fit_.done) {
      Input::update();

      if (millis() - t0 > ImuCalWizardCfg::FIT_TIMEOUT_MS) {
        vTaskDelete(fit_.task);
        fit_.task = nullptr;
        if (show_fail) ui_.fail("FIT", "Timeout");
        return false;
      }

      ph += 0.09f;
      ui_.bar01(0.5f + 0.5f * sinf(ph));
      delay(30);
    }

    fit_.task = nullptr;

    if (!fit_.ok) {
      if (show_fail) ui_.fail(what, imu_cal::fitFailStr(fit_.reason));
      return false;
    }
    return true;
  }

  // Capture steps
  bool readSample_(ImuSample& s) {
    // Uses the mapping/read function from AtomS3R_ImuCal.h
    return readImuMapped(M5.Imu, s);
  }

  void configureCalibrators_() {
    // Use the SAME g that mapping uses (AtomS3R_ImuCal.h)
    const float g = ImuCalCfg::g_std;

    accelCal_.g = g;
    gyroCal_.g  = g;

    // Keep gates reasonable (too tight can bias fits / stall capture)
    accelCal_.accel_mag_tol       = 0.8f;   // m/s^2
    accelCal_.max_gyro_for_static = 0.12f;  // rad/s

    gyroCal_.max_accel_dev = 0.8f;          // m/s^2
    gyroCal_.max_gyro_norm = 0.12f;         // rad/s

    // Allow symmetric cross-axis correction
    using AC = imu_cal::AccelCalibrator<float, 400, 1>;
    accelCal_.accel_S_mode = AC::AccelSMode::PolarSPD;

    // Plausibility gates (tweakable)
    accelCal_.accel_diag_lo = 0.80f;
    accelCal_.accel_diag_hi = 1.25f;
    accelCal_.accel_max_cond = 6.0f;
    accelCal_.accel_max_offdiag_rms = 0.10f; // allow some coupling correction
  }

  bool captureAccelPose_(const Pose& p) {
    ui_.waitTap("ACCEL", p.short_name, "Tap then place");

    ui_.setRotation(p.rot_capture);
    ui_.title("ACCEL");
    ui_.line(p.short_name);
    ui_.line(p.instruction);
    ui_.line("");
    ui_.line("Place now");
    ui_.line("Hold still");

    uint32_t t0 = millis();
    while (millis() - t0 < ImuCalWizardCfg::PLACE_TIME_MS) {
      Input::update();
      ui_.bar01((float)(millis() - t0) / (float)ImuCalWizardCfg::PLACE_TIME_MS);
      delay(30);
    }

    int start_n  = accelCal_.buf.n;
    int target_n = start_n + ImuCalWizardCfg::ACCEL_NEED_PER_POSE;

    ui_.title("ACCEL");
    ui_.line(p.short_name);
    ui_.line("Capturing...");

    uint32_t tcap0 = millis();
    uint32_t last_change = millis();
    int last_n = accelCal_.buf.n;

    while (millis() - tcap0 < ImuCalWizardCfg::ACCEL_TIMEOUT_MS) {
      Input::update();

      ImuSample s;
      if (!readSample_(s)) { delay(2); continue; }

      accelCal_.addSample(s.a, s.w, s.tempC);

      int n = accelCal_.buf.n;
      if (n != last_n) { last_n = n; last_change = millis(); }

      if (millis() - last_change > ImuCalWizardCfg::STUCK_MS) {
        Serial.printf("[ACC] stuck pose='%s' got=%d\n", p.short_name, n - start_n);
        ui_.setReadRotation();
        ui_.fail("ACCEL", "No samples accepted");
        return false;
      }

      ui_.bar01((float)(n - start_n) / (float)ImuCalWizardCfg::ACCEL_NEED_PER_POSE);

      if (n >= target_n) {
        ui_.showOkAuto(p.short_name, "Captured");
        delay(80);
        return true;
      }

      delay(5);
    }

    Serial.printf("[ACC] timeout pose='%s' got=%d\n", p.short_name, accelCal_.buf.n - start_n);
    ui_.setReadRotation();
    ui_.fail("ACCEL", "Timeout");
    return false;
  }

  bool captureGyro_() {
    ui_.waitTap("GYRO", "SCREEN UP", "Tap then place");

    ui_.setReadRotation();
    ui_.title("GYRO");
    ui_.line("SCREEN UP");
    ui_.line("");
    ui_.line("Place on table");
    ui_.line("Do NOT touch");

    uint32_t t0 = millis();
    while (millis() - t0 < ImuCalWizardCfg::PLACE_TIME_MS) {
      Input::update();
      ui_.bar01((float)(millis() - t0) / (float)ImuCalWizardCfg::PLACE_TIME_MS);
      delay(30);
    }

    int start_n  = gyroCal_.buf.n;
    int target_n = start_n + ImuCalWizardCfg::GYRO_NEED;

    ui_.title("GYRO");
    ui_.line("Capturing...");

    uint32_t tcap0 = millis();
    uint32_t last_change = millis();
    int last_n = gyroCal_.buf.n;

    while (millis() - tcap0 < ImuCalWizardCfg::GYRO_TIMEOUT_MS) {
      Input::update();

      ImuSample s;
      if (!readSample_(s)) { delay(2); continue; }

      gyroCal_.addSample(s.w, s.a, s.tempC);

      int n = gyroCal_.buf.n;
      if (n != last_n) { last_n = n; last_change = millis(); }

      if (millis() - last_change > ImuCalWizardCfg::STUCK_MS) {
        Serial.printf("[GYR] stuck got=%d\n", n - start_n);
        ui_.fail("GYRO", "No samples accepted");
        return false;
      }

      ui_.bar01((float)(n - start_n) / (float)ImuCalWizardCfg::GYRO_NEED);

      if (n >= target_n) {
        ui_.showOkAuto("GYRO", "Captured");
        return true;
      }

      delay(5);
    }

    ui_.fail("GYRO", "Timeout");
    return false;
  }

  // MAG capture: downsample + min time + reject stale repeats + 3D coverage tests
  bool captureMag_(const char*& out_why) {
    out_why = nullptr;

    ui_.waitTap("MAG", "Rotate ~45 sec", "Tap to start");

    int start_n  = magCal_.buf.n;
    int target_n = start_n + ImuCalWizardCfg::MAG_NEED;

    ui_.setReadRotation();
    ui_.title("MAG");
    ui_.line("Rotate now");
    ui_.line("Flip all faces");
    ui_.line("Avoid metal");

    uint32_t tcap0 = millis();
    uint32_t last_change = millis();
    int last_n = magCal_.buf.n;

    uint32_t last_add_ms = 0;
    Vector3f last_added = Vector3f(NAN, NAN, NAN);

    // Raw bounds
    Vector3f vmin(+1e9f, +1e9f, +1e9f);
    Vector3f vmax(-1e9f, -1e9f, -1e9f);

    // Direction coverage using a stable-ish center (midpoint of bounds)
    Vector3f umin(+1e9f, +1e9f, +1e9f);
    Vector3f umax(-1e9f, -1e9f, -1e9f);
    Vector3f center = Vector3f::Zero();

    while (millis() - tcap0 < ImuCalWizardCfg::MAG_TIMEOUT_MS) {
      Input::update();

      ImuSample s;
      if (!readSample_(s)) { delay(2); continue; }
      if (!finite3_(s.m)) { delay(2); continue; }

      uint32_t now = millis();

      // downsample accepted samples
      if (last_add_ms == 0 || (uint32_t)(now - last_add_ms) >= ImuCalWizardCfg::MAG_SAMPLE_SPACING_MS) {
        // reject stale repeats (prevents instant fill with identical data)
        bool accept = true;
        if (isfinite(last_added.x())) {
          Vector3f d = s.m - last_added;
          if (d.norm() < ImuCalWizardCfg::MAG_MIN_DELTA_uT) accept = false;
        }

        if (accept) {
          int before = magCal_.buf.n;
          magCal_.addSample(s.m);
          int after = magCal_.buf.n;

          if (after > before) {
            last_add_ms = now;
            last_added = s.m;

            vmin = vmin.cwiseMin(s.m);
            vmax = vmax.cwiseMax(s.m);

            // stable-ish center from bounds midpoint
            center = 0.5f * (vmin + vmax);

            // unit direction about that center
            Vector3f c = s.m - center;
            float cn = c.norm();
            if (cn > 1e-6f) {
              Vector3f u = c / cn;
              umin = umin.cwiseMin(u);
              umax = umax.cwiseMax(u);
            }
          }
        }
      }

      int n = magCal_.buf.n;
      if (n != last_n) { last_n = n; last_change = millis(); }

      if (millis() - last_change > ImuCalWizardCfg::STUCK_MS) {
        Serial.printf("[MAG] stuck n=%d (no accepted samples)\n", n - start_n);
        out_why = "No MAG samples";
        return false;
      }

      uint32_t elapsed = (uint32_t)(millis() - tcap0);

      // progress components
      float pS = (float)(n - start_n) / (float)ImuCalWizardCfg::MAG_NEED;
      float pT = (float)elapsed / (float)ImuCalWizardCfg::MAG_MIN_TIME_MS;

      Vector3f ur = umax - umin;          // 0..2
      float pC = 0.f;
      if (isfinite(ur.x()) && isfinite(ur.y()) && isfinite(ur.z())) {
        float px = clamp01_(ur.x() / ImuCalWizardCfg::MAG_URANGE_TARGET);
        float py = clamp01_(ur.y() / ImuCalWizardCfg::MAG_URANGE_TARGET);
        float pz = clamp01_(ur.z() / ImuCalWizardCfg::MAG_URANGE_TARGET);
        pC = second_smallest3_(px, py, pz); // progress reflects 2 axes covered
      }

      float p = pS;
      if (pT < p) p = pT;
      if (pC < p) p = pC;

      ui_.bar01(p);

      // Only allow finish if we have enough samples, enough time, AND coverage
      if (n >= target_n && elapsed >= ImuCalWizardCfg::MAG_MIN_TIME_MS) {
        Vector3f span = vmax - vmin;

        // sort spans -> smin, smid, smax
        float a = span.x(), b = span.y(), c = span.z();
        float smin = a, smid = b, smax = c;
        if (smin > smid) { float t=smin; smin=smid; smid=t; }
        if (smid > smax) { float t=smid; smid=smax; smax=t; }
        if (smin > smid) { float t=smin; smin=smid; smid=t; }

        float rmin = (smax > 1e-6f) ? (smin / smax) : 0.f;
        float rmid = (smax > 1e-6f) ? (smid / smax) : 0.f;

        Serial.printf("[MAG] n=%d elapsed=%.1fs span=(%.3f,%.3f,%.3f) ratios=(%.2f,%.2f) urange=(%.2f,%.2f,%.2f)\n",
                      n - start_n, (double)(elapsed / 1000.0f),
                      (double)span.x(), (double)span.y(), (double)span.z(),
                      (double)rmin, (double)rmid,
                      (double)ur.x(), (double)ur.y(), (double)ur.z());
        
        // FINAL GATES HERE
        // 1) Span ratio gate
        if (rmin < ImuCalWizardCfg::MAG_SPAN_MIN_FRAC ||
            rmid < ImuCalWizardCfg::MAG_SPAN_MID_FRAC) {
          out_why = "Span ratios too low";
          return false;
        }
        
        // 2) Centered direction range gate: require at least 2 axes to meet target
        int ok_axes = 0;
        ok_axes += (ur.x() >= ImuCalWizardCfg::MAG_URANGE_TARGET) ? 1 : 0;
        ok_axes += (ur.y() >= ImuCalWizardCfg::MAG_URANGE_TARGET) ? 1 : 0;
        ok_axes += (ur.z() >= ImuCalWizardCfg::MAG_URANGE_TARGET) ? 1 : 0;
        if (ok_axes < 2) {
          out_why = "Direction range too small";
          return false;
        }
        
        // Coverage tests:
        if (smax < 1e-3f) { out_why = "MAG not changing"; return false; }
        
        // Planarity test: det(C) small => planar motion
        const float detC = unit_dir_cov_det_(magCal_.buf.v, n);
        Serial.printf("[MAG] detC=%.6f\n", (double)detC);
        
        if (detC < 2.0e-4f) {
          out_why = "Coverage too flat";
          return false;
        }
        
        ui_.showOkAuto("MAG", "Captured");
        return true;
      }

      delay(5);
    }

    out_why = "Timeout";
    return false;
  }

  void fillBlob_(ImuCalBlobV1& blob) {
    memset(&blob, 0, sizeof(blob));
    blob.magic = ImuCalBlobV1::IMU_CAL_MAGIC;
    blob.version = ImuCalBlobV1::IMU_CAL_VERSION;
    blob.size_bytes = sizeof(ImuCalBlobV1);

    blob.accel_ok = acc_out_.ok ? 1 : 0;
    blob.accel_g  = acc_out_.g;
    mat_to_rowmajor9_(acc_out_.S, blob.accel_S);
    blob.accel_T0 = acc_out_.biasT.T0;
    blob.accel_b0[0]=acc_out_.biasT.b0.x(); blob.accel_b0[1]=acc_out_.biasT.b0.y(); blob.accel_b0[2]=acc_out_.biasT.b0.z();
    blob.accel_k[0]=acc_out_.biasT.k.x();   blob.accel_k[1]=acc_out_.biasT.k.y();   blob.accel_k[2]=acc_out_.biasT.k.z();
    blob.accel_rms_mag = acc_out_.rms_mag;

    blob.gyro_ok = gyr_out_.ok ? 1 : 0;
    blob.gyro_T0 = gyr_out_.biasT.T0;
    blob.gyro_b0[0]=gyr_out_.biasT.b0.x(); blob.gyro_b0[1]=gyr_out_.biasT.b0.y(); blob.gyro_b0[2]=gyr_out_.biasT.b0.z();
    blob.gyro_k[0]=gyr_out_.biasT.k.x();   blob.gyro_k[1]=gyr_out_.biasT.k.y();   blob.gyro_k[2]=gyr_out_.biasT.k.z();

    blob.mag_ok = mag_out_.ok ? 1 : 0;
    mat_to_rowmajor9_(mag_out_.A, blob.mag_A);
    blob.mag_b[0]=mag_out_.b.x(); blob.mag_b[1]=mag_out_.b.y(); blob.mag_b[2]=mag_out_.b.z();
    blob.mag_field_uT = mag_out_.field_uT;
    blob.mag_rms      = mag_out_.rms;
  }

private:
  M5Ui& ui_;
  ImuCalStoreNvs& store_;

  // Stored inside wizard => not on stack
public:
  imu_cal::AccelCalibrator<float, 400, 1> accelCal_{};
  imu_cal::GyroCalibrator<float,  400, 8> gyroCal_{};
  imu_cal::MagCalibrator<float,   400>    magCal_{};

  imu_cal::AccelCalibration<float> acc_out_{};
  imu_cal::GyroCalibration<float>  gyr_out_{};
  imu_cal::MagCalibration<float>   mag_out_{};
};

} // namespace atoms3r_ical
