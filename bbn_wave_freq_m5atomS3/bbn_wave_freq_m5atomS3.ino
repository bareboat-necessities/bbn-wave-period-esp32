/*

  Ocean wave frequency estimator using esp32 (m5atomS3)

  See: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html

  Instead of FFT method for finding main wave frequency we could use Aranovskiy frequency estimator which is a simple on-line filter.

  Ref:

  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy

  The New Algorithm of Sinusoidal Signal Frequency Estimation.

  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing July 3-5, 2013. Caen, France

*/

#include <M5Unified.h>
#include <M5AtomS3.h>
#include <Arduino.h>
#include "AranovskiyFilter.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "Mahony_AHRS.h"

// Strength of the calibration operation;
// 0: disables calibration.
// 1 is weakest and 255 is strongest.
static constexpr const uint8_t calib_value = 64;

// This sample code performs calibration by clicking on a button or screen.
// After 10 seconds of calibration, the results are stored in NVS.
// The saved calibration values are loaded at the next startup.
//
// === How to calibration ===
// ※ Calibration method for Accelerometer
//    Change the direction of the main unit by 90 degrees
//     and hold it still for 2 seconds. Repeat multiple times.
//     It is recommended that as many surfaces as possible be on the bottom.
//
// ※ Calibration method for Gyro
//    Simply place the unit on a quiet desk and hold it still.
//    It is recommended that this be done after the accelerometer calibration.
//
// ※ Calibration method for geomagnetic sensors
//    Rotate the main unit slowly in multiple directions.
//    It is recommended that as many surfaces as possible be oriented to the north.
//
// Values for extremely large attitude changes are ignored.
// During calibration, it is desirable to move the device as gently as possible.

struct rect_t {
  int32_t x;
  int32_t y;
  int32_t w;
  int32_t h;
};

static constexpr const float coefficient_tbl[3] = { 0.5f, (1.0f / 256.0f), (1.0f / 1024.0f) };
static uint8_t calib_countdown = 0;

static auto &dsp = (M5.Display);
static rect_t rect_text_area;

static int prev_xpos[18];

unsigned long now = 0UL, last_refresh = 0UL, last_update = 0UL;
int got_samples = 0;

AranovskiyParams params;
AranovskiyState  state;

double omega_up = 2.5 * (2 * PI);  // upper frequency Hz * 2 * PI
double k_gain = 2.0;

double t_0 = 0.0;
double x1_0 = 0.0;
double theta_0 = - (omega_up * omega_up / 4.0);
double sigma_0 = theta_0;

double delta_t;  // time step sec

KalmanSmootherVars kalman;

int first = 1;

Mahony_AHRS_Vars mahony;

const char* name;

void updateCalibration(uint32_t c, bool clear = false) {
  calib_countdown = c;

  if (c == 0) {
    clear = true;
  }

  if (clear) {
    memset(prev_xpos, 0, sizeof(prev_xpos));
    dsp.fillScreen(TFT_BLACK);
    if (c) {
      // Start calibration.
      M5.Imu.setCalibration(calib_value, calib_value, calib_value);
      // The actual calibration operation is performed each time during M5.Imu.update.
      //
      // There are three arguments, which can be specified in the order of Accelerometer, gyro, and geomagnetic.
      // If you want to calibrate only the Accelerometer, do the following.
      // M5.Imu.setCalibration(100, 0, 0);
      //
      // If you want to calibrate only the gyro, do the following.
      // M5.Imu.setCalibration(0, 100, 0);
      //
      // If you want to calibrate only the geomagnetism, do the following.
      // M5.Imu.setCalibration(0, 0, 100);
    }
    else {
      // Stop calibration. (Continue calibration only for the geomagnetic sensor)
      M5.Imu.setCalibration(0, 0, calib_value);

      // If you want to stop all calibration, write this.
      // M5.Imu.setCalibration(0, 0, 0);

      // save calibration values.
      M5.Imu.saveOffsetToNVS();
    }
  }
  dsp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);

  if (c) {
    dsp.setCursor(2, rect_text_area.h + 1);
    dsp.setTextColor(TFT_WHITE, TFT_BLACK);
    dsp.printf("Countdown:%3d", c);
  }
}

void startCalibration(void) {
  updateCalibration(30, true);
}

void repeatMe() {
  static uint32_t prev_sec = 0;

  auto imu_update = M5.Imu.update();
  if (imu_update) {
    m5::imu_3d_t accel;
    M5.Imu.getAccelData(&accel.x, &accel.y, &accel.z);

    m5::imu_3d_t gyro;
    M5.Imu.getAccelData(&gyro.x, &gyro.y, &gyro.z);
    
    got_samples++;

    now = micros();
    delta_t = ((now - last_update) / 1000000.0);
    last_update = now;

    float pitch, roll, yaw;
    mahony_AHRS_update(&mahony, gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z, &pitch, &roll, &yaw, delta_t);

    double y = (accel.z - 1.0) /* since it includes g */;
    //double y = sin(2 * PI * state.t * 0.25); // dummy test data

    aranovskiy_update(&params, &state, y, delta_t);

    if (first) {
      kalman_smoother_set_initial(&kalman, state.f);
      first = 0;
    }
    double freq_adj = kalman_smoother_update(&kalman, state.f);

    double a = y;  // acceleration in fractions of g
    double period = (state.f > 0 ? 1.0 / state.f : 9999.0);    // or use freq_adj
    double wave_length = trochoid_wave_length(period);
    double heave = - a * wave_length / (2 * PI);

    if (now - last_refresh >= 1000000) {
      dsp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);

      M5.Lcd.setCursor(0, 2);
      M5.Lcd.printf("imu: %s\n", name);
      M5.Lcd.printf("sec: %d\n", now / 200000);
      M5.Lcd.printf("samples: %d\n", got_samples);
      M5.Lcd.printf("period sec: %0.4f\n", (state.f > 0 ? 1.0 / state.f : 9999.0));
      M5.Lcd.printf("period adj: %0.4f\n", (freq_adj > 0 ? 1.0 / freq_adj : 9999.0));
      M5.Lcd.printf("wave len:   %0.4f\n", wave_length);
      M5.Lcd.printf("heave:      %0.4f\n", heave);
      M5.Lcd.printf("%0.3f %0.3f %0.3f\n", accel.x, accel.y, accel.z);
      M5.Lcd.printf("accel abs:  %0.4f\n", sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z));
      M5.Lcd.printf("%0.1f %0.1f %0.1f\n", pitch, roll, yaw);

      last_refresh = now;
      got_samples = 0;
    }
  }
  else {
    AtomS3.update();
    // Calibration is initiated when ascreen is clicked.
    if (AtomS3.BtnA.isPressed()) {
      startCalibration();
    }
  }
  int32_t sec = millis() / 1000;
  if (prev_sec != sec) {
    prev_sec = sec;
    if (calib_countdown) {
      updateCalibration(calib_countdown - 1);
    }
    if ((sec & 7) == 0) {
      // prevent WDT.
      vTaskDelay(1);
    }
  }
}

void setup(void) {
  auto cfg = M5.config();
  AtomS3.begin(cfg);

  auto imu_type = M5.Imu.getType();
  switch (imu_type) {
    case m5::imu_none:        name = "not found";   break;
    case m5::imu_sh200q:      name = "sh200q";      break;
    case m5::imu_mpu6050:     name = "mpu6050";     break;
    case m5::imu_mpu6886:     name = "mpu6886";     break;
    case m5::imu_mpu9250:     name = "mpu9250";     break;
    case m5::imu_bmi270:      name = "bmi270";      break;
    default:                  name = "unknown";     break;
  };
  dsp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);
  M5.Lcd.setCursor(0, 2);
  M5.Lcd.printf("imu: %s\n", name);

  if (imu_type == m5::imu_none) {
    for (;;) {
      delay(1);
    }
  }

  int32_t w = dsp.width();
  int32_t h = dsp.height();
  if (w < h) {
    dsp.setRotation(dsp.getRotation() ^ 1);
    w = dsp.width();
    h = dsp.height();
  }
  int32_t text_area_h = ((h - 8) / 18) * 18;
  rect_text_area = {0, 0, w, text_area_h };

  // Read calibration values from NVS.
  if (!M5.Imu.loadOffsetFromNVS()) {
    startCalibration();
  }

  aranovskiy_default_params(&params, omega_up, k_gain);
  aranovskiy_init_state(&state, t_0, x1_0, theta_0, sigma_0);
  kalman_smoother_init(&kalman, 0.003, 10.0, 100.0);

  last_update = micros();
}

void loop(void) {
  AtomS3.update();
  delay(3);
  repeatMe();
}
