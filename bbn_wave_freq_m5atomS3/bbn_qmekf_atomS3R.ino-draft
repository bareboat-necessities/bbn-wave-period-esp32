/*
  Copyright 2026, Mikhail Grushinskiy

  AtomS3R Tilt-Compensated Compass + IMU Calibration Wizard

  - Uses AtomS3R_ImuCal.h (blob + runtime apply + axis mapping + clearM5UnifiedImuCalibration)
  - Uses AtomS3R_ImuCalWizard.h (UI + wizard capture/fit/save + Input tap logic)
  - Uses your QuaternionMEKF (q-mekf) implementation (KalmanQMEKF.h) for fused roll/pitch/yaw
  - Runs accel+gyro at ~240 Hz, mag is slower and gated by time+delta (stale-repeat rejection)
  - Shows Heading (deg, 0..360), Roll/Pitch (deg) on screen and prints to Serial

  Taps:
    - 1 tap  : run wizard (save new blob)
    - 3 taps : erase blob (and clear M5Unified calibration)

  Notes:
    - Feed calibrated accel (m/s^2) and gyro (rad/s) to MEKF.
    - Feed *unit* magnetometer direction (calibrated, normalized) to MEKF.
*/

#include <Arduino.h>
#include <M5Unified.h>
#include <new>

#include "AtomS3R_ImuCal.h"
#include "AtomS3R_ImuCalWizard.h"
#include "AtomS3R_CompassUI.h"
#include "NmeaCompass.h"

// 1 = graphical compass by default, 0 = text UI by default
#ifndef COMPASS_UI_DEFAULT_GRAPHICS
#define COMPASS_UI_DEFAULT_GRAPHICS 1
#endif

// 0 = keep debug serial
// 1 = emit NMEA0183 (HDM + XDR + ROT) like pypilot
#ifndef COMPASS_SERIAL_NMEA
#define COMPASS_SERIAL_NMEA 1
#endif

// 2-char talker ID
#ifndef COMPASS_NMEA_TALKER
#define COMPASS_NMEA_TALKER "II"
#endif

// q-mekf header that defines QuaternionMEKF
#include "KalmanQMEKF.h"

using namespace atoms3r_ical;
using Vector3f = Eigen::Matrix<float, 3, 1>;

// Sampling / gating tuning
static constexpr float    LOOP_HZ        = 240.0f;
static constexpr uint32_t LOOP_PERIOD_US = (uint32_t)(1000000.0f / LOOP_HZ);

// Mag ODR is slower; also many builds repeat stale values.
// Gate by spacing + delta (stale-repeat rejection).
static constexpr uint32_t MAG_SAMPLE_SPACING_MS = 12;     // ~<=80 Hz mag updates
static constexpr float    MAG_MIN_DELTA_uT      = 0.02f;  // reject repeats (0.02..0.10 typical)

// Display / serial rates
static constexpr uint32_t UI_REFRESH_MS     = 100;   // 10 Hz UI update
static constexpr uint32_t DEBUG_SERIAL_MS   = 200;   // 5 Hz
static constexpr uint32_t NMEA_SERIAL_MS    = 100;   // 10 Hz

// ROT bias removal (stillness-gated gyro bias estimator)
static constexpr float ROT_BIAS_TAU_S          = 20.0f;   // how fast bias adapts when still
static constexpr float ROT_STILL_G_TOL_FRAC    = 0.12f;   // | |a| - g | < 12% g counts as still
static constexpr float ROT_STILL_GYRO_RAD_S    = 0.15f;   // |w| < 0.15 rad/s counts as still

// Helpers
static inline float clampf_(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline float wrap360_(float deg) {
  while (deg < 0.0f) deg += 360.0f;
  while (deg >= 360.0f) deg -= 360.0f;
  return deg;
}

static inline float wrap180_(float deg) {
  while (deg < -180.0f) deg += 360.0f;
  while (deg >= 180.0f) deg -= 360.0f;
  return deg;
}

static inline Vector3f quatRotateBodyToWorld_(float x, float y, float z, float w,
                                             const Vector3f& v_body) {
  // Assumes (x,y,z,w) is a UNIT quaternion rotating body->world
  // v_world = v + w*(2 q×v) + q×(2 q×v)
  Vector3f q(x, y, z);
  Vector3f t = 2.0f * q.cross(v_body);
  return v_body + w * t + q.cross(t);
}


// App
class CompassApp {
public:
  CompassApp() : wizard_(ui_, store_) {}

  void begin() {
    Serial.begin(115200);
    delay(150);
    Serial.println();
    Serial.println("[BOOT] AtomS3R Compass + Cal Wizard");

    auto cfg = M5.config();
    M5.begin(cfg);

    // Critical: clear M5Unified’s own cal/offsets so it can’t stack with ours.
    clearM5UnifiedImuCalibration();
    delay(250);

    ui_.begin();

    // Optional graphical compass UI
    if (use_graphics_) {
      ui_.setReadRotation();     // ensure width/height match ROT_READ
      compass_ui_.begin();
      compass_ui_ready_ = compass_ui_.ok();
      if (!compass_ui_ready_) use_graphics_ = false;
    }

    if (!M5.Imu.isEnabled()) {
      Serial.println("[BOOT] IMU not found / not enabled");
      ui_.fail("IMU", "Not found");
      while (true) delay(100);
    }
    reloadBlobAndRuntime_();

    // If missing, immediately run wizard once (matches your requested flow)
    if (!have_blob_) {
      Serial.println("[BOOT] No saved calibration. Starting wizard...");
      ImuCalBlobV1 saved{};
      if (wizard_.runAndSave(saved)) {
        Serial.println("[BOOT] Wizard saved calibration. Loaded:");
        printBlobSummary(Serial, saved);
        printBlobDetail(Serial, saved);
        blob_ = saved;
        have_blob_ = true;
        runtime_.rebuildFromBlob(blob_);
      } else {
        Serial.println("[BOOT] Wizard did not save calibration. Running with raw values.");
      }
    } else {
      Serial.println("[BOOT] Found saved calibration:");
      printBlobSummary(Serial, blob_);
      printBlobDetail(Serial, blob_);
    }

    resetMEKF_();
    drawHomeStatic_();

    start_us_ = micros();
    next_tick_us_ = micros();
    last_update_us_ = micros();
  }

  void tick() {
    // Pace loop to ~240 Hz (light pacing; wizard UI will override timing when active)
    const uint32_t now_us = micros();
    int32_t wait_us = (int32_t)(next_tick_us_ - now_us);
    if (wait_us > 0) {
      if (wait_us > 200) delayMicroseconds(200);
      else delayMicroseconds((uint32_t)wait_us);
    }
    next_tick_us_ += LOOP_PERIOD_US;

    Input::update();

    // Tap handling (same behavior as your cal app)
    if (Input::tapPressed()) {
      tap_count_++;
      tap_deadline_ms_ = millis() + ImuCalWizardCfg::TAP_WINDOW_MS;
      drawHomePending_();
      Serial.printf("[TAP] count=%d\n", tap_count_);
    }

    if (tap_count_ > 0 && (int32_t)(millis() - tap_deadline_ms_) > 0) {
      if (tap_count_ >= 3) handleErase_();
      else                 handleRunWizard_();
      tap_count_ = 0;
      tap_deadline_ms_ = 0;
      drawHomeStatic_();
    }

    // Read IMU and update filter
    ImuSample s;
    if (readImuMapped(M5.Imu, s)) {
      updateFilter_(s);
    }
    updateUI_();
    streamSerial_();
  }

private:
  // UI mode
  bool use_graphics_ = (COMPASS_UI_DEFAULT_GRAPHICS != 0);
  CompassUI compass_ui_{};
  bool compass_ui_ready_ = false;

  // ROT estimator (deg/min) + stillness-gated gyro bias removal
  bool     rot_inited_    = false;
  float    rot_dpm_filt_  = 0.0f;
  bool     gyro_bias_ok_  = false;
  Vector3f gyro_bias_ema_ = Vector3f::Zero();   // rad/s (BODY)

  // Calibration plumbing
  void reloadBlobAndRuntime_() {
    have_blob_ = store_.load(blob_);
    if (!have_blob_) {
      memset(&blob_, 0, sizeof(blob_));
    }
    runtime_.rebuildFromBlob(blob_);
  }

  void handleErase_() {
    Serial.println("[HOME] triple tap => ERASE");

    if (!ui_.eraseConfirm()) {
      Serial.println("[HOME] erase cancelled");
      return;
    }
    store_.erase();
    clearM5UnifiedImuCalibration();

    reloadBlobAndRuntime_();
    resetMEKF_();

    Serial.println("[HOME] erased blob + cleared M5Unified cal");
  }

  void handleRunWizard_() {
    Serial.println("[HOME] single tap => RUN WIZARD");

    ImuCalBlobV1 saved{};
    bool ok = wizard_.runAndSave(saved);
    if (!ok) {
      ui_.notSavedNotice();
      return;
    }

    Serial.println("[HOME] new calibration saved:");
    printBlobSummary(Serial, saved);
    printBlobDetail(Serial, saved);

    blob_ = saved;
    have_blob_ = true;
    runtime_.rebuildFromBlob(blob_);

    resetMEKF_();
  }

  // MEKF
  void resetMEKF_() {
    // Noise tuning for 240 Hz acc+gyro updates (conservative handheld/boat defaults):
    const float g = ImuCalCfg::g_std;

    Vector3f sigma_a; sigma_a << 0.06f * g, 0.06f * g, 0.06f * g; // m/s^2
    Vector3f sigma_g; sigma_g << 0.0030f, 0.0030f, 0.0030f;       // rad/s
    Vector3f sigma_m; sigma_m << 0.020f, 0.020f, 0.020f;          // unit-vector noise

    const float Pq0 = 0.5f;
    const float Pb0 = 1e-2f;
    const float b0  = 1e-9f;

    if (mekf_) {
      mekf_->~QuaternionMEKF<float, true>();
      mekf_ = nullptr;
    }
    mekf_ = new (mekf_storage_) QuaternionMEKF<float, true>(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0);
    mekf_inited_ = false;

    last_mag_uT_.setZero();
    last_mag_ms_ = 0;

    rot_inited_   = false;
    rot_dpm_filt_ = 0.0f;

    gyro_bias_ok_  = false;
    gyro_bias_ema_.setZero();

    Serial.println("[MEKF] reset");
  }

  void updateFilter_(const ImuSample& s) {
    // Cache raw mapped norms for debug
    a_raw_norm_ = s.a.norm();

    // Calibrate signals (or pass through if missing)
    a_cal_ = runtime_.applyAccel(s.a, s.tempC);
    w_cal_ = runtime_.applyGyro (s.w, s.tempC);
    m_cal_ = runtime_.applyMag  (s.m);

    // dt from micros; clamp sane
    const uint32_t now_us = micros();
    float dt = (now_us - last_update_us_) * 1e-6f;
    last_update_us_ = now_us;
    dt = clampf_(dt, 0.0010f, 0.0200f);

    // Mag validity (magnitude sanity)
    const float mn = m_cal_.norm();
    mag_ok_ = (mn > 5.0f && mn < 200.0f); // broad uT bounds
    mag_norm_uT_ = mn;

    // Normalize mag for MEKF (unit direction)
    Vector3f m_u = Vector3f::Zero();
    if (mag_ok_ && mn > 1e-6f) m_u = m_cal_ / mn;

    // Stale-repeat rejection + ODR gating
    const uint32_t now_ms = millis();
    bool mag_fresh = false;
    if (mag_ok_) {
      const uint32_t dtm = now_ms - last_mag_ms_;
      const float dm = (m_cal_ - last_mag_uT_).norm();
      if (dtm >= MAG_SAMPLE_SPACING_MS && dm >= MAG_MIN_DELTA_uT) {
        mag_fresh = true;
        last_mag_ms_ = now_ms;
        last_mag_uT_ = m_cal_;
      }
    }
    mag_fresh_ = mag_fresh;

    // Initialize once: prefer acc+mag if mag looks OK
    if (!mekf_inited_) {
      // IMPORTANT: initialize should use accel *direction* too
      Vector3f a_init = a_cal_;
      const float an0 = a_init.norm();
      if (an0 > 1e-6f) a_init *= (ImuCalCfg::g_std / an0);

      if (mag_ok_) mekf_->initialize_from_acc_mag(a_init, m_u);
      else         mekf_->initialize_from_acc(a_init);

      mekf_inited_ = true;
    }

    // Core filter updates at ~240Hz
    mekf_->time_update(w_cal_, dt);

    // IMPORTANT: attitude update should use accel *direction*.
    // Your accel calibration S can scale |a| away from g, which biases tilt.
    Vector3f a_att = a_cal_;
    const float an = a_att.norm();
    if (an > 1e-6f) a_att *= (ImuCalCfg::g_std / an);
    mekf_->measurement_update_acc_only(a_att);

    // Mag update at mag ODR (gated)
    if (mag_ok_ && mag_fresh_) {
      mekf_->measurement_update_mag_only(m_u);
    }

    const auto qv = mekf_->quaternion();

    float x = qv(0);
    float y = qv(1);
    float z = qv(2);
    float w = qv(3);

    // Normalize defensively
    const float nn = x*x + y*y + z*z + w*w;
    if (nn > 1e-12f) {
      const float invn = 1.0f / sqrtf(nn);
      x *= invn; y *= invn; z *= invn; w *= invn;
    }

    // ZYX yaw-pitch-roll
    const float siny_cosp = 2.0f * (w*z + x*y);
    const float cosy_cosp = 1.0f - 2.0f * (y*y + z*z);
    const float yaw = atan2f(siny_cosp, cosy_cosp);

    float sinp = 2.0f * (w*y - z*x);
    sinp = clampf_(sinp, -1.0f, 1.0f);
    const float pitch = asinf(sinp);

    const float sinr_cosp = 2.0f * (w*x + y*z);
    const float cosr_cosp = 1.0f - 2.0f * (x*x + y*y);
    const float roll = atan2f(sinr_cosp, cosr_cosp);

    roll_deg_  = roll  * RAD_TO_DEG;
    pitch_deg_ = pitch * RAD_TO_DEG;
    yaw_deg_   = yaw   * RAD_TO_DEG;

    heading_deg_ = wrap360_(yaw_deg_);

    // ROT (deg/min): gyro projected into world-Z, with stillness-gated bias removal
    // Update a slow gyro bias estimate ONLY when still (so it doesn't "learn" real turns).
    const float g = ImuCalCfg::g_std;
    const float a_err = fabsf(a_cal_.norm() - g);
    const bool still = (a_err < ROT_STILL_G_TOL_FRAC * g) && (w_cal_.norm() < ROT_STILL_GYRO_RAD_S);

    if (still) {
      const float alpha_b = 1.0f - expf(-dt / ROT_BIAS_TAU_S);
      if (!gyro_bias_ok_) {
        gyro_bias_ok_  = true;
        gyro_bias_ema_ = w_cal_;
      } else {
        gyro_bias_ema_ += alpha_b * (w_cal_ - gyro_bias_ema_);
      }
    }

    // Bias-correct gyro for ROT
    Vector3f w_use = w_cal_;
    if (gyro_bias_ok_) w_use -= gyro_bias_ema_;

    // Project to world frame using the SAME quaternion convention you already use for yaw/pitch/roll.
    Vector3f w_world = quatRotateBodyToWorld_(x, y, z, w, w_use);

    // In NED, +Z is down. Rate about +Z maps to NMEA ROT sign (if sign is flipped, negate here).
    float rot_dpm_meas = w_world.z() * RAD_TO_DEG * 60.0f;    // rad/s -> deg/min

    // Clamp nonsense spikes (init/glitches)
    rot_dpm_meas = clampf_(rot_dpm_meas, -720.0f, 720.0f);

    // Low-pass ROT
    const float tau_rot = 1.5f;                               // seconds (your previous choice is fine)
    const float alpha_r = 1.0f - expf(-dt / tau_rot);

    if (!rot_inited_) {
      rot_inited_   = true;
      rot_dpm_filt_ = rot_dpm_meas;
    } else {
      rot_dpm_filt_ += alpha_r * (rot_dpm_meas - rot_dpm_filt_);
    }
  }

  // UI
  void drawHomeStatic_() {
    ui_.setReadRotation();
    ui_.title("COMPASS");
    M5.Display.printf("BLOB: %s\n", have_blob_ ? "YES" : "NO");
    M5.Display.printf("A:%d G:%d M:%d\n",
                      (int)runtime_.acc.ok, (int)runtime_.gyr.ok, (int)runtime_.mag.ok);
    ui_.line("Tap: calibrate");
    ui_.line("Tap x3: erase");
    ui_.line("");
  }

  void drawHomePending_() {
    ui_.setReadRotation();
    ui_.title("COMPASS");
    M5.Display.printf("Tap count: %d\n", tap_count_);
    ui_.line("");
    ui_.line("Wait...");
    ui_.line("1 tap=CAL");
    ui_.line("3 taps=ERASE");

    int32_t remain = (int32_t)(tap_deadline_ms_ - millis());
    remain = remain < 0 ? 0 : remain;
    float t01 = 1.0f - (float)remain / (float)ImuCalWizardCfg::TAP_WINDOW_MS;
    ui_.bar01(t01);
  }

  void updateUI_() {
    // If we're in the tap-group decision window, keep the pending screen visible.
    if (tap_count_ > 0) return;

    const uint32_t now_ms = millis();
    if (now_ms - last_ui_ms_ < UI_REFRESH_MS) return;
    last_ui_ms_ = now_ms;

    if (use_graphics_ && compass_ui_ready_) {
      updateUI_graphics_();
    } else {
      updateUI_text_();
    }
  }

  void updateUI_graphics_() {
    // Ensure the display rotation matches what CompassUI sprites were built for.
    ui_.setReadRotation();

    // Simple tilt warning (optional): tweak thresholds as you like
    const bool tiltWarn = (fabsf(roll_deg_) > 35.0f) || (fabsf(pitch_deg_) > 35.0f);

    compass_ui_.draw(heading_deg_, mag_ok_, mag_norm_uT_, tiltWarn);
  }

  void updateUI_text_() {
    ui_.setReadRotation();
    ui_.title("COMPASS");

    M5.Display.printf("HDG: %6.1f deg\n", (double)heading_deg_);
    M5.Display.printf("ROL: %6.1f deg\n", (double)roll_deg_);
    M5.Display.printf("PIT: %6.1f deg\n", (double)pitch_deg_);

    M5.Display.printf("MAG: %s %s\n",
                      mag_ok_ ? "OK " : "BAD",
                      mag_fresh_ ? "NEW" : "OLD");
    M5.Display.printf("|m|: %6.1f uT\n", (double)mag_norm_uT_);
    M5.Display.printf("|aR|:%5.2f |aC|:%5.2f\n",
                      (double)a_raw_norm_, (double)a_cal_.norm());
    ui_.line("");
    M5.Display.printf("A:%d G:%d M:%d  B:%s\n",
                      (int)runtime_.acc.ok, (int)runtime_.gyr.ok, (int)runtime_.mag.ok,
                      have_blob_ ? "YES" : "NO");
  }

  void streamSerial_() {
    const uint32_t now_ms = millis();

#if COMPASS_SERIAL_NMEA
    if (now_ms - last_serial_ms_ < NMEA_SERIAL_MS) return;
#else
    if (now_ms - last_serial_ms_ < DEBUG_SERIAL_MS) return;
#endif

    last_serial_ms_ = now_ms;

#if COMPASS_SERIAL_NMEA
    float rot_dpm = rot_dpm_filt_;
    const bool valid = mekf_inited_;

    nmea_hdm(COMPASS_NMEA_TALKER, heading_deg_);
    nmea_xdr_pitch_roll(COMPASS_NMEA_TALKER, pitch_deg_, roll_deg_);
    nmea_rot(COMPASS_NMEA_TALKER, rot_dpm, valid);
#else
    const float t = (micros() - start_us_) * 1e-6f;

    Serial.printf("t=%.2f ", (double)t);
    Serial.printf("HDG:%6.1f ", (double)heading_deg_);
    Serial.printf("ROLL:%6.1f PITCH:%6.1f ", (double)roll_deg_, (double)pitch_deg_);
    Serial.printf("mag:%s/%s |m|=%.1f ",
                  mag_ok_ ? "OK" : "BAD",
                  mag_fresh_ ? "NEW" : "OLD",
                  (double)mag_norm_uT_);
    Serial.printf("|a_raw|=%.4f |a_cal|=%.4f ",
                  (double)a_raw_norm_, (double)a_cal_.norm());
    Serial.printf("wC:%+.4f,%+.4f,%+.4f ",
                  (double)w_cal_.x(), (double)w_cal_.y(), (double)w_cal_.z());
    Serial.println();
#endif
  }

private:
  // Plumbing
  M5Ui ui_{};
  ImuCalStoreNvs store_{};
  ImuCalWizard wizard_;

  bool have_blob_ = false;
  ImuCalBlobV1 blob_{};
  RuntimeCals runtime_{};

  // Tap state
  int tap_count_ = 0;
  uint32_t tap_deadline_ms_ = 0;

  // Timing
  uint32_t start_us_ = 0;
  uint32_t next_tick_us_ = 0;
  uint32_t last_update_us_ = 0;
  uint32_t last_ui_ms_ = 0;
  uint32_t last_serial_ms_ = 0;

  // MEKF (placement-new)
  alignas(QuaternionMEKF<float, true>) uint8_t mekf_storage_[sizeof(QuaternionMEKF<float, true>)];
  QuaternionMEKF<float, true>* mekf_ = nullptr;
  bool mekf_inited_ = false;

  // Latest calibrated values
  Vector3f a_cal_ = Vector3f::Zero();
  Vector3f w_cal_ = Vector3f::Zero();
  Vector3f m_cal_ = Vector3f::Zero();
  float a_raw_norm_ = 0.0f;

  // Attitude / heading
  float roll_deg_ = 0.0f;
  float pitch_deg_ = 0.0f;
  float yaw_deg_ = 0.0f;
  float heading_deg_ = 0.0f;

  // Mag gating
  bool mag_ok_ = false;
  bool mag_fresh_ = false;
  float mag_norm_uT_ = 0.0f;
  Vector3f last_mag_uT_ = Vector3f::Zero();
  uint32_t last_mag_ms_ = 0;
};

static CompassApp g_app;

void setup() { g_app.begin(); }
void loop()  { g_app.tick(); }
