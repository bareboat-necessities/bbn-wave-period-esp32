/*
  Copyright 2024, Mikhail Grushinskiy

  Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32 (m5atomS3)

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
#include "Quaternion.h"
#include "MinMaxLemire.h"
#include "KalmanForWave.h"
#include "KalmanForWaveAlt.h"
#include "NmeaXDR.h"
#include "KalmanQMEKF.h"
#include "WaveFilters.h"

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

bool useMahony = true;

static uint8_t calib_countdown = 0;

static auto &disp = (M5.Display);
static rect_t rect_text_area;

static int prev_xpos[18];

unsigned long now = 0UL, last_refresh = 0UL, start_time = 0UL, last_update = 0UL, last_update_k = 0UL;
unsigned long got_samples = 0;
bool kalm_w_first = true, kalm_w_alt_first = true, kalm_smoother_first = true;

MinMaxLemire min_max_h;
AranovskiyParams arParams;
AranovskiyState arState;
KalmanSmootherVars kalman_freq;
Mahony_AHRS_Vars mahony;
KalmanWaveState waveState;
KalmanWaveAltState waveAltState;
Kalman_QMEKF kalman_mekf;

const char* imu_name;

bool produce_serial_data = true;
bool report_nmea = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

void updateCalibration(uint32_t c, bool clear = false) {
  calib_countdown = c;

  if (c == 0) {
    clear = true;
  }

  if (clear) {
    memset(prev_xpos, 0, sizeof(prev_xpos));
    disp.fillScreen(TFT_BLACK);
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
  disp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);

  if (c) {
    disp.setCursor(2, rect_text_area.h + 1);
    disp.setTextColor(TFT_WHITE, TFT_BLACK);
    disp.printf("Countdown:%3d", c);
  }
}

void startCalibration(void) {
  updateCalibration(30, true);
}

void read_and_processIMU_data() {
  m5::imu_3d_t accel;
  M5.Imu.getAccelData(&accel.x, &accel.y, &accel.z);

  m5::imu_3d_t gyro;
  M5.Imu.getGyroData(&gyro.x, &gyro.y, &gyro.z);

  got_samples++;
  now = micros();

  if ((accel.x * accel.x + accel.y * accel.y + accel.z * accel.z) < ACCEL_MAX_G_SQUARE) {
    // ignore noise (in unbiased way) with unreasonably high Gs

    float delta_t = (now - last_update) / 1000000.0;  // time step sec
    last_update = now;

    float pitch, roll, yaw;
    Quaternion quaternion;

    if (useMahony) {
      mahony_AHRS_update(&mahony, gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD,
                         accel.x, accel.y, accel.z, &pitch, &roll, &yaw, delta_t);
      Quaternion_set(mahony.q0, mahony.q1, mahony.q2, mahony.q3, &quaternion);
    }
    else {
      kalman_mekf.gyr = {gyro.y * DEG_TO_RAD, gyro.x * DEG_TO_RAD, -gyro.z * DEG_TO_RAD};
      kalman_mekf.acc = {accel.y, accel.x, -accel.z}; 
      //kalman_mekf.mag = {magne.x, magne.y, magne.z};

      if (kalm_w_first) {
        //kalman_mekf.mekf->initialize_from_acc_mag(kalman_mekf.acc, kalman_mekf.mag);
        kalman_mekf.mekf->initialize_from_acc(kalman_mekf.acc);
      }

      kalman_mekf.mekf->time_update(kalman_mekf.gyr, delta_t);
      //kalman_mekf.mekf->measurement_update(kalman_mekf.acc, kalman_mekf.mag);
      kalman_mekf.mekf->measurement_update_acc_only(kalman_mekf.acc);
      kalman_mekf.quat = kalman_mekf.mekf->quaternion();
      Quaternion_set(kalman_mekf.quat.w(), kalman_mekf.quat.x(), kalman_mekf.quat.y(), kalman_mekf.quat.z(), &quaternion);
      float euler[3];
      Quaternion_toEulerZYX(&quaternion, euler);
      roll = euler[0] * RAD_TO_DEG;
      pitch = euler[1] * RAD_TO_DEG;
      yaw = euler[2] * RAD_TO_DEG;
    }

    float v[3] = {accel.x, accel.y, accel.z};
    float rotated_a[3];
    Quaternion_rotate(&quaternion, v, rotated_a);

    m5::imu_3d_t accel_rotated;
    accel_rotated.x = rotated_a[0];
    accel_rotated.y = rotated_a[1];
    accel_rotated.z = rotated_a[2];

    float a = (accel_rotated.z - 1.0);  // acceleration in fractions of g
    //float a = - 0.25 * PI * PI * sin(2 * PI * t * 0.25 - 2.0) / g_std; // dummy test data (amplitude of heave = 1m, 4sec - period)
    //float h =  0.25 * PI * PI / (2 * PI * 0.25) / (2 * PI * 0.25) * sin(2 * PI * t * 0.25 - 2.0);

    if (kalm_w_first) {
      kalm_w_first = false;
      float k_hat = - pow(2.0 * PI * FREQ_GUESS, 2);
      waveState.displacement_integral = 0.0f;
      waveState.heave = a * g_std / k_hat;
      waveState.vert_speed = 0.0f;               // ??
      waveState.accel_bias = 0.0f;
      kalman_wave_init_state(&waveState);
    }
    kalman_wave_step(&waveState, a * g_std, delta_t);

    double freq_adj = 0.0;
    if (t > warmup_time_sec(useMahony)) {
      // give some time for other filters to settle first
      aranovskiy_update(&arParams, &arState, waveState.heave / ARANOVSKIY_SCALE, delta_t);
      if (kalm_smoother_first) {
        kalm_smoother_first = false;
        kalman_smoother_set_initial(&kalman_freq, arState.f);
      }
      freq_adj = kalman_smoother_update(&kalman_freq, arState.f);
    }

    if (isnan(freq_adj)) {
      // reset filters
      kalm_w_first = true;
      kalm_w_alt_first = true;
      kalm_smoother_first = true;
      init_filters(&arParams, &arState, &kalman_freq);
      start_time = micros();
      last_update = start_time;
      t = 0.0;
    }
    else if (freq_adj > FREQ_LOWER && freq_adj < FREQ_UPPER) { /* prevent decimal overflows */
      double period = 1.0 / freq_adj;
      uint32_t windowMicros = getWindowMicros(period);
      SampleType sample = { .value = waveState.heave, .timeMicroSec = now };
      min_max_lemire_update(&min_max_h, sample, windowMicros);

      if (fabs(arState.f - freq_adj) < FREQ_COEF * freq_adj) { /* sanity check of convergence for freq */
        float k_hat = - pow(2.0 * PI * freq_adj, 2);
        if (kalm_w_alt_first) {
          kalm_w_alt_first = false;
          waveAltState.displacement_integral = 0.0f;
          waveAltState.heave = waveState.heave;
          waveAltState.vert_speed = waveState.vert_speed;
          waveAltState.vert_accel = k_hat * waveState.heave; //a * g_std;
          waveAltState.accel_bias = 0.0f;
          kalman_wave_alt_init_state(&waveAltState);
        }
        float delta_t_k = last_update_k == 0UL ? delta_t : (now - last_update_k) / 1000000.0;
        kalman_wave_alt_step(&waveAltState, a * g_std, k_hat, delta_t_k);
        last_update_k = now;
      }

      float wave_height = min_max_h.max.value - min_max_h.min.value;
      heave_avg = (min_max_h.max.value + min_max_h.min.value) / 2.0;

      int serial_report_period_micros = 125000;
      if (now - last_refresh >= (produce_serial_data ? serial_report_period_micros : 1000000)) {
        if (produce_serial_data) {
          if (report_nmea) {
            // do not report data for which filters clearly didn't converge
            if (wave_height < 30.0) {
              gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRG1", wave_height);
            }
            if (fabs(waveState.heave) < 15.0) {
              gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRT1", waveState.heave);
            }
            gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DAV1", heave_avg);
            if (fabs(arState.f - freq_adj) < 0.07 * freq_adj) {
              gen_nmea0183_xdr("$BBXDR,F,%.5f,H,FAV1", freq_adj);
              if (fabs(waveAltState.heave) < 15.0) {
                gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRT2", waveAltState.heave);
              }
            }
            if (arState.f > 0.02 && arState.f < 4.0) {
              gen_nmea0183_xdr("$BBXDR,F,%.5f,H,FRT1", arState.f);
            }
            gen_nmea0183_xdr("$BBXDR,F,%.5f,H,SRT1", got_samples / ((now - last_refresh) / 1000000.0) );
            gen_nmea0183_xdr("$BBXDR,N,%.5f,P,ABI1", waveState.accel_bias * 100.0 / g_std);
          }
          else {
            // report for Arduino Serial Plotter
            Serial.printf("heave_cm:%.4f", waveState.heave * 100);
            Serial.printf(",heave_alt:%.4f", waveAltState.heave * 100);
            //Serial.printf(",freq_adj:%.4f", freq_adj * 100);
            //Serial.printf(",freq:%.4f", arState.f * 100);
            //Serial.printf(",h_cm:%.4f", h * 100);
            Serial.printf(",height_cm:%.4f", wave_height * 100);
            Serial.printf(",max_cm:%.4f", min_max_h.max.value * 100);
            Serial.printf(",min_cm:%.4f", min_max_h.min.value * 100);
            Serial.printf(",heave_avg_cm:%.4f", heave_avg * 100);
            //Serial.printf(",period_decisec:%.4f", period * 10);
            //Serial.printf(",accel abs:%0.4f", g_std * sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z));
            //Serial.printf(",accel bias:%0.4f", waveState.accel_bias);

            // for https://github.com/thecountoftuscany/PyTeapot-Quaternion-Euler-cube-rotation
            //Serial.printf("y%0.1fyp%0.1fpr%0.1fr", yaw, pitch, roll);
            Serial.println();
          }
        }
        else {
          disp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);
          M5.Lcd.setCursor(0, 2);
          M5.Lcd.printf("imu: %s\n", imu_name);
          M5.Lcd.printf("sec: %d\n", now / 1000000);
          M5.Lcd.printf("samples: %d\n", got_samples);
          M5.Lcd.printf("period sec: %0.4f\n", period);
          M5.Lcd.printf("wave len:   %0.4f\n", wave_length);
          M5.Lcd.printf("heave:      %0.4f\n", waveState.heave);
          M5.Lcd.printf("wave height:%0.4f\n", wave_height);
          M5.Lcd.printf("range %0.4f %0.4f\n", min_max_h.min.value, min_max_h.max.value);
          M5.Lcd.printf("%0.3f %0.3f %0.3f\n", accel.x, accel.y, accel.z);
          M5.Lcd.printf("accel abs:  %0.4f\n", sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z));
          M5.Lcd.printf("accel vert: %0.4f\n", (accel_rotated.z - 1.0));
          M5.Lcd.printf("%0.1f %0.1f %0.1f\n", pitch, roll, yaw);
        }

        last_refresh = now;
        got_samples = 0;
      }
    }
  }
  t = (now - start_time) / 1000000.0;  // time since start sec
}

void repeatMe() {
  static uint32_t prev_sec = 0;
  auto imu_update = M5.Imu.update();
  if (imu_update && !AtomS3.BtnA.isPressed()) {
    read_and_processIMU_data();
  }
  else {
    // Calibration is initiated when screen is clicked. Screen on atomS3 is a button
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
  Serial.begin(115200);

  auto imu_type = M5.Imu.getType();
  switch (imu_type) {
    case m5::imu_none:        imu_name = "not found";   break;
    case m5::imu_sh200q:      imu_name = "sh200q";      break;
    case m5::imu_mpu6050:     imu_name = "mpu6050";     break;
    case m5::imu_mpu6886:     imu_name = "mpu6886";     break;
    case m5::imu_mpu9250:     imu_name = "mpu9250";     break;
    case m5::imu_bmi270:      imu_name = "bmi270";      break;
    default:                  imu_name = "unknown";     break;
  };
  disp.fillRect(0, 0, rect_text_area.w, rect_text_area.h, TFT_BLACK);
  M5.Lcd.setCursor(0, 2);
  M5.Lcd.printf("imu: %s\n", imu_name);

  if (imu_type == m5::imu_none) {
    for (;;) {
      delay(1);
    }
  }

  int32_t w = disp.width();
  int32_t h = disp.height();
  if (w < h) {
    disp.setRotation(disp.getRotation() ^ 1);
    w = disp.width();
    h = disp.height();
  }
  int32_t text_area_h = ((h - 8) / 18) * 18;
  rect_text_area = {0, 0, w, text_area_h };

  // Read calibration values from NVS.
  if (!M5.Imu.loadOffsetFromNVS()) {
    startCalibration();
  }

  if (useMahony) {
    float twoKp = (2.0f * 1.0f);
    float twoKi = (2.0f * 0.0001f);
    mahony_AHRS_init(&mahony, twoKp, twoKi);
  }
  else {
    static Vector3f sigma_a = {20.78e-2, 20.78e-2, 20.78e-2};
    static Vector3f sigma_g = {0.2020 * M_PI / 180, 0.2020 * M_PI / 180, 0.2020 * M_PI / 180};
    static Vector3f sigma_m = {3.2e-3, 3.2e-3, 3.2e-3};
    float Pq0 = 1e-2;
    float Pb0 = 1e-2;
    float b0 = 1e-12;
    static QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0);
    kalman_mekf.mekf = &mekf;
  }

  init_filters(&arParams, &arState, &kalman_freq);

  start_time = micros();
  last_update = start_time;
}

void loop(void) {
  AtomS3.update();
  delay(3);
  repeatMe();
}
