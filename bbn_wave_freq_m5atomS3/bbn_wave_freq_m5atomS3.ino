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
#include "AngleAveraging.h"
#include "AranovskiyFilter.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "KalmANF.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "Mahony_AHRS.h"
#include "Quaternion.h"
#include "MinMaxLemire.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "TimeAwareSpikeFilter.h"
#include "NmeaXDR.h"
#include "KalmanWaveDirection.h"
#include "KalmanQMEKF.h"
#include "WaveFilters.h"
#include "WaveDirectionDetector.h"
#include "M5_Calibr.h"

bool useMahony = true;
FrequencyTracker useFrequencyTracker = ZeroCrossing;
bool use_kalman_for_wave_dir = true;

unsigned long now = 0UL, last_refresh = 0UL, start_time = 0UL, last_update = 0UL;
unsigned long got_samples = 0;
bool kalm_w_first = true, kalm_w_alt_first = true, kalm_smoother_first = true;

// Basic filters
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
KalmanForWaveBasicState waveState;
KalmanWaveNumStableAltState waveAltState;

#define WRONG_ANGLE_MARKER -360.0f

// Wave direction
WaveDirectionDetector wave_dir_detector(0.002f, 0.005f);
AngleAverager angle_averager(0.004f);
float wave_angle_deg = WRONG_ANGLE_MARKER;
AngleEstimate wave_angle_estimate;

const char* imu_name;

bool produce_serial_data = true;
bool report_nmea = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

float azimuth_deg_180(float a, float b) {
  if (b < 0) {
    a = -a;
    b = -b;
  }
  return atan2(a, b) * 180 / M_PI;
}

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

  float tempC = NAN;
  M5.Imu.getTemp(&tempC);

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
  float a_band_passed = a_noisy; //bpFilter.processWithDelta(a_noisy, delta_t);
  float a_no_spikes = spikeFilter.filterWithDelta(a_band_passed, delta_t);

  a_no_spikes = clamp(a_no_spikes, -ACCEL_CLAMP, ACCEL_CLAMP);
  float a = a_no_spikes;
  
  if (kalm_w_first) {
    kalm_w_first = false;
    float k_hat = - pow(2.0 * PI * FREQ_GUESS, 2);
    waveState.displacement_integral = 0.0f;
    waveState.heave = a * g_std / k_hat;
    waveState.vert_speed = 0.0f;               // waveState.vert_speed = 2.0f * M_PI * FREQ_GUESS * waveState.heave * cosf(known_phase) / sinf(known_phase);
    waveState.accel_bias = 0.0f;
    wave_filter.initState(waveState);
  }
  wave_filter.step(a * g_std, delta_t, waveState);
  float heave = waveState.heave;

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
      wave_alt_filter.initState(waveAltState);
    }
    wave_alt_filter.update(a * g_std, k_hat, delta_t, tempC);
    waveAltState = wave_alt_filter.getState();
    heaveAlt = waveAltState.heave;
    //heaveAlt = highPassFilterAlt.update(waveAltState.heave, delta_t);

    double period = 1.0 / freq_adj;
    uint32_t windowMicros = getWindowMicros(period);
    SampleType sample = { .value = heaveAlt, .timeMicroSec = now };
    min_max_lemire_update(&min_max_h, sample, windowMicros);

    float wave_height = min_max_h.max.value - min_max_h.min.value;
    heave_avg = (min_max_h.max.value + min_max_h.min.value) / 2.0;

    // Wave direction steps
    if (use_kalman_for_wave_dir) {
      wave_dir_kalman.update(accel_rotated.x * g_std, accel_rotated.y * g_std, freq_adj, delta_t);
      float wave_deg = wave_dir_kalman.getDirectionDegrees();
      wave_angle_estimate = angle_averager.average180(wave_deg);
      wave_angle_deg = wave_angle_estimate.angle;
    } else {
      float azimuth = azimuth_deg_180(accel_rotated.x, accel_rotated.y); 
      if (wave_angle_deg != WRONG_ANGLE_MARKER) {
        float accel_magnitude = AngleAverager::magnitude(accel_rotated.x, accel_rotated.y);
        const float IGNORE_LOW_ACCEL =  0.04f;  // m/s^2
        if (accel_magnitude > IGNORE_LOW_ACCEL / g_std) {  // ignore low magnitudes
          wave_angle_estimate = angle_averager.average180(azimuth);
          wave_angle_deg = wave_angle_estimate.angle;
        }
      } else {
        wave_angle_deg = azimuth;
      }
    }
    WaveDirection wave_dir = wave_dir_detector.update(accel_rotated.x * g_std, accel_rotated.y * g_std, a_noisy * g_std, delta_t);

    // other wave parameters (these are not real, they are from observer point of view / apparent)
    // real values would require knowing boat speed, direction and adjustments for Doppler effect
    float ap_wavelength = trochoid_wavelength(2.0 * PI * freq_adj);
    float ap_wave_number = trochoid_wavenumber(ap_wavelength);
    float ap_wave_speed = trochoid_wave_speed(ap_wave_number);

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
          gen_nmea0183_xdr("$BBXDR,G,%.5f,M,AP_WAVE_LENGTH", ap_wavelength);
          gen_nmea0183_xdr("$BBXDR,G,%.5f,,AP_WAVE_PERIOD", period);
          gen_nmea0183_xdr("$BBXDR,A,%.1f,D,AP_WAVE_ANGLE", wave_angle_deg);
          gen_nmea0183_xdr("$BBXDR,G,%1d,,AP_WAVE_DIR", wave_dir);
          gen_nmea0183_xdr("$BBXDR,G,%.5f,,AP_WAVE_SPEED", ap_wave_speed);
        } else {
          // report for Arduino Serial Plotter
          //Serial.printf(",a:%0.4f", g_std * a);
          //Serial.printf(",a_band_passed:%0.4f", g_std * a_band_passed);
          //Serial.printf(",a_noisy:%0.4f", g_std * a_noisy);
          //Serial.printf(",a_no_spikes:%0.4f", g_std * a_no_spikes);
          //Serial.printf(",heave_cm:%.4f", heave * 100);
          Serial.printf(",heave_alt:%.4f", heaveAlt * 100);
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
          //Serial.printf(",ap_wave_speed:%.2f", ap_wave_speed);
          //Serial.printf(",ap_wavelength:%.2f", ap_wavelength);
          //Serial.printf(",ap_wave_dir_est_deg:%.2f", wave_angle_deg);
          //Serial.printf(",ap_wave_dir:%1d", wave_dir);
          //Serial.printf(",ap_wave_dir_P:%.2f", wave_dir_detector.getFilteredP());

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

  start_time = micros();
  last_update = start_time;
}

void loop(void) {
  unsigned long start = micros();
  M5.update();
  repeatMe();
  long duration = micros() - start;
  long delay_micros = 3900 - duration;
  if (delay_micros > 0) {
    delayMicroseconds(delay_micros);
  }
}
