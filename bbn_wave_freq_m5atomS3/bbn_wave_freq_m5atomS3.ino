/*
  Copyright 2024-2025, Mikhail Grushinskiy

  Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32 (m5atomS3)

  See: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html

  Instead of FFT method for finding main wave frequency we could use Zero Crossing, Aranovskiy or KalmANF 
  frequency estimators which is are simple on-line filters.

  Ref:

  1. Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy

  The New Algorithm of Sinusoidal Signal Frequency Estimation.

  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing July 3-5, 2013. Caen, France

  2. R. Ali, T. van Waterschoot

  A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter. KalmANF

  Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023


*/

#include <M5Unified.h>
#include <Arduino.h>
#include "AranovskiyFilter.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmANF.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "Mahony_AHRS.h"
#include "Quaternion.h"
#include "MinMaxLemire.h"
#include "KalmanForWave.h"
#include "KalmanForWaveAlt.h"
#include "TimeAwareSpikeFilter.h"
#include "TimeAwareBandpassFilter.h"
#include "HighPassFilters.h"
#include "FourthOrderLowPass.h"
#include "NmeaXDR.h"
#include "KalmanQMEKF.h"
#include "WaveDirectionEKF.h"
#include "WaveDirection_LTV_KF.h"
#include "WaveFilters.h"
#include "M5_Calibr.h"

bool useMahony = true;
FrequencyTracker useFrequencyTracker = ZeroCrossing;

unsigned long now = 0UL, last_refresh = 0UL, start_time = 0UL, last_update = 0UL;
unsigned long got_samples = 0;
bool kalm_w_first = true, kalm_w_alt_first = true, kalm_smoother_first = true;

// Basic filters
//TimeAwareBandpassFilter bpFilter((FREQ_UPPER + FREQ_LOWER) / 2.0f, FREQ_UPPER - FREQ_LOWER, 0ul);  // Create a bandpass filter for 0.02-4 Hz, Center frequency: 2.01 Hz, Bandwidth: 3.98 Hz
//FourthOrderLowPass lowPassFilter(FREQ_UPPER);
//HighPassFirstOrderFilter highPassFilter(1.0f / FREQ_LOWER /* period in sec */);
//HighPassFirstOrderFilter highPassFilterAlt(1.0f / FREQ_LOWER /* period in sec */);
TimeAwareSpikeFilter spikeFilter(ACCEL_SPIKE_FILTER_SIZE, ACCEL_SPIKE_FILTER_THRESHOLD);

// frequency tracking
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS /* hysteresis (fractions of signal magnitude) */,
    ZERO_CROSSINGS_PERIODS /* periods to run measures on */);
AranovskiyParams arParams;
AranovskiyState arState;
KalmANF kalmANF;
KalmanSmootherVars kalman_freq;

// AHRS
Mahony_AHRS_Vars mahony;
Kalman_QMEKF kalman_mekf;

// Wave
MinMaxLemire min_max_h;
KalmanWaveState waveState;
KalmanWaveAltState waveAltState;

// Wave direction
WaveDirection_LTV_KF wave_dir_kf;
WaveDirectionEKF wave_dir_ekf(1.0f, -1.0f, 0.0f, 0.0f, 0.0f);

const char* imu_name;

bool produce_serial_data = true;
bool report_nmea = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

void initialize_filters() {
  if (useFrequencyTracker == Aranovskiy) {
    init_filters(&arParams, &arState, &kalman_freq);
  } else if (useFrequencyTracker == Kalm_ANF) {
    init_filters_alt(&kalmANF, &kalman_freq);
  } else {
    kalman_smoother_init(&kalman_freq, 0.2f, 2.0f, 100.0f);
    init_wave_filters();
  }
}

void read_and_processIMU_data() {
  auto data = M5.Imu.getImuData();

  now = micros();
  got_samples++;

  m5::imu_3d_t accel;
  accel.x = data.accel.x;
  accel.y = data.accel.y;
  accel.z = data.accel.z;

  m5::imu_3d_t gyro;
  gyro.x = data.gyro.x;
  gyro.y = data.gyro.y;
  gyro.z = data.gyro.z;

  drawCalibrGraph(rect_graph_area, data);

  float delta_t = (now - last_update) / 1000000.0;  // time step sec
  delta_t = clamp(delta_t, 0.001f, 0.1f);
  last_update = now;

  float pitch, roll, yaw;
  Quaternion quaternion;

  if (useMahony) {
    mahony_AHRS_update(&mahony, gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD,
                       accel.x, accel.y, accel.z, &pitch, &roll, &yaw, delta_t);
    Quaternion_set(mahony.q0, mahony.q1, mahony.q2, mahony.q3, &quaternion);
  } else {
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

  float a_noisy = (accel_rotated.z - 1.0);  // acceleration in fractions of g
  //float a_band_passed = lowPassFilter.process(a_noisy, delta_t);
  float a_band_passed = a_noisy; //bpFilter.processWithDelta(a_noisy, delta_t);
  float a_no_spikes = spikeFilter.filterWithDelta(a_band_passed, delta_t);

  a_no_spikes = clamp(a_no_spikes, -ACCEL_CLAMP, ACCEL_CLAMP);
  float a = a_no_spikes;
  
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
  float heave = waveState.heave;
  //float heave = highPassFilter.update(waveState.heave, delta_t);

  double freq = FREQ_GUESS, freq_adj = FREQ_GUESS;
  if (t > warmup_time_sec(useMahony)) {
    // give some time for other filters to settle first
    freq = estimate_freq(useFrequencyTracker, &arParams, &arState, &kalmANF, &freqDetector, a_noisy, a_no_spikes, delta_t);
    if (kalm_smoother_first) {
      kalm_smoother_first = false;
      kalman_smoother_set_initial(&kalman_freq, freq);
    }
    if (!isnan(freq)) {
      freq_adj = kalman_smoother_update(&kalman_freq, freq);
    }
  }
  if (isnan(freq) || isnan(freq_adj)) {
    // reset filters
    kalm_w_first = true;
    kalm_w_alt_first = true;
    kalm_smoother_first = true;
    initialize_filters();
    start_time = micros();
    last_update = start_time;
    t = 0.0;
  } else {
    freq_adj = clamp(freq_adj, (double) FREQ_LOWER, (double) FREQ_UPPER);
    float heaveAlt = waveAltState.heave;

    float k_hat = - pow(2.0 * PI * freq_adj, 2);
    if (kalm_w_alt_first) {
      kalm_w_alt_first = false;
      waveAltState.displacement_integral = 0.0f;
      waveAltState.heave = heave;
      waveAltState.vert_speed = waveState.vert_speed;
      waveAltState.vert_accel = k_hat * heave; //a * g_std;
      waveAltState.accel_bias = 0.0f;
      kalman_wave_alt_init_state(&waveAltState);
    }
    kalman_wave_alt_step(&waveAltState, a * g_std, k_hat, delta_t);
    heaveAlt = waveAltState.heave;
    //heaveAlt = highPassFilterAlt.update(waveAltState.heave, delta_t);

    double period = 1.0 / freq_adj;
    uint32_t windowMicros = getWindowMicros(period);
    SampleType sample = { .value = heaveAlt, .timeMicroSec = now };
    min_max_lemire_update(&min_max_h, sample, windowMicros);

    float wave_height = min_max_h.max.value - min_max_h.min.value;
    heave_avg = (min_max_h.max.value + min_max_h.min.value) / 2.0;

    // Wave direction KF steps
    float omega = freq_adj * 2 * M_PI;
    wave_dir_kf.update(t, omega, accel_rotated.x * g_std, accel_rotated.y * g_std);
    auto wave_dir_state = wave_dir_kf.getState();  // get all values of state vector
    float wave_dir_deg = wave_dir_kf.getAtanAB() * 180 / M_PI;

    wave_dir_ekf.predict(t, omega, dt);
    wave_dir_ekf.update(t, omega, accel_rotated.x * g_std, accel_rotated.y * g_std);
    auto wave_dir_alt_state = wave_dir_ekf.getState();  // get all values of state vector
    float wave_dir_alt_deg = wave_dir_ekf.getAtanAB() * 180 / M_PI;

    int serial_report_period_micros = 125000;
    if (now - last_refresh >= (produce_serial_data ? serial_report_period_micros : 1000000)) {
      if (produce_serial_data) {
        if (report_nmea) {
          // do not report data for which filters clearly didn't converge
          if (wave_height < 30.0) {
            gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRG1", wave_height);
          }
          if (fabs(heave) < 15.0) {
            gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRT1", heaveAlt);
          }
          gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DAV1", heave_avg);
          if (fabs(freq - freq_adj) < FREQ_COEF * freq_adj) {
            gen_nmea0183_xdr("$BBXDR,F,%.5f,H,FAV1", freq_adj);
            gen_nmea0183_xdr("$BBXDR,D,%.5f,M,DRT2", heave);
          }
          if (freq >= FREQ_LOWER && freq <= FREQ_UPPER) {
            gen_nmea0183_xdr("$BBXDR,F,%.5f,H,FRT1", freq);
          }
          gen_nmea0183_xdr("$BBXDR,F,%.5f,H,SRT1", got_samples / ((now - last_refresh) / 1000000.0) );
          gen_nmea0183_xdr("$BBXDR,N,%.5f,P,ABI1", waveAltState.accel_bias * 100.0 / g_std);
        } else {
          // report for Arduino Serial Plotter
          //Serial.printf(",a:%0.4f", g_std * a);
          //Serial.printf(",a_band_passed:%0.4f", g_std * a_band_passed);
          //Serial.printf(",a_noisy:%0.4f", g_std * a_noisy);
          //Serial.printf(",a_no_spikes:%0.4f", g_std * a_no_spikes);
          //Serial.printf(",heave_cm:%.4f", heave * 100);
          //Serial.printf(",heave_alt:%.4f", heaveAlt * 100);
          //Serial.printf(",freq_adj:%.4f", freq_adj * 100);
          //Serial.printf(",freq:%.4f", freq * 100);
          //Serial.printf(",h_cm:%.4f", h * 100);
          //Serial.printf(",height_cm:%.4f", wave_height * 100);
          //Serial.printf(",max_cm:%.4f", min_max_h.max.value * 100);
          //Serial.printf(",min_cm:%.4f", min_max_h.min.value * 100);
          //Serial.printf(",heave_avg_cm:%.4f", heave_avg * 100);
          //Serial.printf(",period_decisec:%.4f", period * 10);
          //Serial.printf(",accel abs:%0.4f", g_std * sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z));
          //Serial.printf(",accel bias:%0.4f", waveState.accel_bias);
          Serial.printf(",wave_dir_deg:%.2f", wave_dir_deg);
          Serial.printf(",wave_dir_alt_deg:%.2f", wave_dir_alt_deg);
          Serial.printf(",φ:%.4f", wave_dir_kf.getPhase() * 180 / M_PI);
          Serial.printf(",φ_alt:%.4f", wave_dir_ekf.getPhase() * 180 / M_PI);

          // for https://github.com/thecountoftuscany/PyTeapot-Quaternion-Euler-cube-rotation
          //Serial.printf("y%0.1fyp%0.1fpr%0.1fr", yaw, pitch, roll);
          Serial.println();
        }
      }
      last_refresh = now;
      got_samples = 0;
    }
  }

  t = (now - start_time) / 1000000.0;  // time since start sec
}

void repeatMe() {
  bool pressed = M5.BtnA.wasPressed();
  // Calibration is initiated when screen is clicked. Screen on atomS3 is a button
  if (pressed) {
    startCalibration();
  } else {
    auto imu_update = M5.Imu.update();
    if (imu_update) {
      read_and_processIMU_data();
    }
  }
  makeCalibrStep();
}

void setup(void) {
  auto cfg = M5.config();
  M5.begin(cfg);
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
    float twoKp = (2.0f * 2.0f);
    float twoKi = (2.0f * 0.0001f);
    mahony_AHRS_init(&mahony, twoKp, twoKi);
  } else {
    static Vector3f sigma_a = {6.0e-3, 3.0e-3, 6.0e-3};
    static Vector3f sigma_g = {0.15 * M_PI / 180, 0.4 * M_PI / 180, 0.15 * M_PI / 180};
    static Vector3f sigma_m = {3.2e-3, 3.2e-3, 3.2e-3};
    float Pq0 = 1e-2;
    float Pb0 = 1e-2;
    float b0 = 1e-12;
    static QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m, Pq0, Pb0, b0);
    kalman_mekf.mekf = &mekf;
  }

  initialize_filters();
  
  Matrix6f wave_dir_Q = Eigen::Matrix<float, 6, 6>::Identity() * 1e-4f;    // Process noise
  Matrix2f wave_dir_R = Eigen::Matrix<float, 2, 2>::Identity() * 0.09f;    // Measurement noise
  Matrix6f wave_dir_P0 = Eigen::Matrix<float, 6, 6>::Identity() * 100.0f;  // Initial covariance
  wave_dir_P0(4, 4) = 1.0f;
  wave_dir_P0(5, 5) = 1.0f;
  wave_dir_kf.init(wave_dir_Q, wave_dir_R, wave_dir_P0);

  wave_dir_ekf.setProcessNoise(1e-6f, 1e-6f, 1e-4f * M_PI * M_PI, 1e-6f, 1e-6f);
  wave_dir_ekf.setMeasurementNoise(0.09f, 0.09f);

  start_time = micros();
  last_update = start_time;
}

void loop(void) {
  unsigned long start = micros();
  M5.update();
  repeatMe();
  long duration = micros() - start;
  long delay_micros = 3700 - duration;
  if (delay_micros > 0) {
    delayMicroseconds(delay_micros);
  }
}
