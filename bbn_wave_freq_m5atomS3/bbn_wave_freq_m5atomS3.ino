/*
  Copyright 2024-2025, Mikhail Grushinskiy

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
#include "M5_Calibr.h"

bool useMahony = true;

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
float freq_good_est = 0.0;

void read_and_processIMU_data() {
  auto data = M5.Imu.getImuData();
  
  m5::imu_3d_t accel;
  accel.x = data.accel.x;
  accel.y = data.accel.y;
  accel.z = data.accel.z;

  m5::imu_3d_t gyro;
  gyro.x = data.gyro.x;
  gyro.y = data.gyro.y;
  gyro.z = data.gyro.z;

  got_samples++;
  now = micros();

  drawCalibrGraph(rect_graph_area, data);
  
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

      if (fabs(arState.f - freq_adj) < FREQ_COEF_TIGHT * freq_adj) {  /* sanity check of convergence for freq */
        freq_good_est = freq_adj;
      }

      // use previous good estimate of frequency
      if (fabs(arState.f - freq_good_est) < FREQ_COEF * freq_good_est) {
        float k_hat = - pow(2.0 * PI * freq_good_est, 2);
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
            if (fabs(arState.f - freq_good_est) < 0.07 * freq_good_est) {
              gen_nmea0183_xdr("$BBXDR,F,%.5f,H,FAV1", freq_good_est);
              if (fabs(waveAltState.heave - waveState.heave) < 0.2 * fabs(waveState.heave)) {
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
            //Serial.printf(",freq_good_est:%.4f", freq_good_est * 100);
            //Serial.printf(",freq_adj:%.4f", freq_adj * 100);
            //Serial.printf(",freq:%.4f", arState.f * 100);
            //Serial.printf(",h_cm:%.4f", h * 100);
            Serial.printf(",height_cm:%.4f", wave_height * 100);
            Serial.printf(",max_cm:%.4f", min_max_h.max.value * 100);
            Serial.printf(",min_cm:%.4f", min_max_h.min.value * 100);
            //Serial.printf(",heave_avg_cm:%.4f", heave_avg * 100);
            //Serial.printf(",period_decisec:%.4f", period * 10);
            //Serial.printf(",accel abs:%0.4f", g_std * sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z));
            //Serial.printf(",accel bias:%0.4f", waveState.accel_bias);

            // for https://github.com/thecountoftuscany/PyTeapot-Quaternion-Euler-cube-rotation
            //Serial.printf("y%0.1fyp%0.1fpr%0.1fr", yaw, pitch, roll);
            Serial.println();
          }
        }

        last_refresh = now;
        got_samples = 0;
      }
    }
  }
  t = (now - start_time) / 1000000.0;  // time since start sec
}

void repeatMe() {
  auto imu_update = M5.Imu.update();
  if (imu_update) {
    read_and_processIMU_data();
  }
  else {
    M5.update();
    bool pressed = M5.BtnA.wasClicked() || M5.Touch.getDetail().wasClicked();
    // Calibration is initiated when screen is clicked. Screen on atomS3 is a button
    if (pressed) {
      startCalibration();
    }
  }
  makeCalibrStep();
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
  //M5.Lcd.setCursor(0, 2);
  //M5.Lcd.printf("imu: %s\n", imu_name);

  if (imu_type == m5::imu_none) {
    for (;;) {
      delay(1);
    }
  }
  
  initCalibrDisplay();

  // Read calibration values from NVS.
  if (!M5.Imu.loadOffsetFromNVS()) {
    //startCalibration();
  }

  if (useMahony) {
    float twoKp = (2.0f * 4.0f);
    float twoKi = (2.0f * 0.0001f);
    mahony_AHRS_init(&mahony, twoKp, twoKi);
  }
  else {
    static Vector3f sigma_a = {6.0e-3, 3.0e-3, 6.0e-3};
    static Vector3f sigma_g = {0.15 * M_PI / 180, 0.4 * M_PI / 180, 0.15 * M_PI / 180};
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
  repeatMe();
  delay(9);
}
