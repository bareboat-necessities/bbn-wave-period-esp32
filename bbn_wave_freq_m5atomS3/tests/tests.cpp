/*
  Copyright 2024, Mikhail Grushinskiy
*/
#include <cmath>

#include "AranovskiyFilter.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "MinMaxLemire.h"
#include "KalmanForWave.h"
#include "KalmanForWaveAlt.h"

MinMaxLemire min_max_h;
AranovskiyParams arParams;
AranovskiyState arState;
KalmanSmootherVars kalman_freq;
KalmanWaveState waveState;
KalmanWaveAltState waveAltState;

bool first = true, kalman_k_first = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

uint32_t now() {
  return t * 1000000;
}

unsigned long last_update_k = 0UL;

void run_fiters(float a, float v, float h, float delta_t) {
  kalman_wave_step(&waveState, a * g_std, delta_t);

  if (t > 10.0 /* sec */) {
    // give some time for other filters to settle first
    aranovskiy_update(&arParams, &arState, waveState.heave, delta_t);
  }

  if (first) {
    kalman_smoother_set_initial(&kalman_freq, arState.f);
    first = false;
  }
  double freq_adj = kalman_smoother_update(&kalman_freq, arState.f);

  //if (1) {
  if (freq_adj > 0.002 && freq_adj < 5.0) { /* prevent decimal overflows */
    double period = 1.0 / freq_adj;
    uint32_t windowMicros = 3 * period * 1000000;
    if (windowMicros <= 10 * 1000000) {
      windowMicros = 10 * 1000000;
    }
    else if (windowMicros >= 60 * 1000000) {
      windowMicros = 60 * 1000000;
    }
    SampleType sample = { .value = waveState.heave, .timeMicroSec = now() };
    min_max_lemire_update(&min_max_h, sample, windowMicros);

    //if (1) {
    if (fabs(arState.f - freq_adj) < 0.3 * freq_adj) { /* sanity check of convergence for freq */
      float k_hat = - pow(2.0 * PI * freq_adj, 2);
      if (kalman_k_first) {
        kalman_k_first = false;
        waveAltState.heave = waveState.heave;
        waveAltState.vert_speed = waveState.vert_speed;
        waveAltState.vert_accel = k_hat * waveState.heave; //a * g_std;
        waveAltState.accel_bias = 0.0f;
        kalman_wave_alt_init_state(&waveAltState);
      }
      float delta_t_k = last_update_k == 0UL ? delta_t : (now() - last_update_k) / 1000000.0;
      kalman_wave_alt_step(&waveAltState, a * g_std, k_hat, delta_t_k);
      last_update_k = now();
    }

    float wave_height = min_max_h.max.value - min_max_h.min.value;
    heave_avg = (min_max_h.max.value + min_max_h.min.value) / 2.0;

    printf("time,%.5f", t);
    printf(",a,%.4f", a * g_std);
    printf(",v,%.4f", v);
    printf(",h,%.4f", h);
    printf(",heave,%.4f", waveState.heave);
    printf(",heave_alt,%.4f", waveAltState.heave);
    printf(",height,%.4f", wave_height);
    printf(",max,%.4f", min_max_h.max.value);
    printf(",min,%.4f", min_max_h.min.value);
    printf(",period,%.4f", period);
    printf(",freq:,%.4f", arState.f);
    printf(",freq_adj,%.4f", freq_adj);
    printf(",heave_avg,%.7f", heave_avg);
    printf(",accel_bias,%.5f", waveState.accel_bias);
    printf("\n");
  }
}

void init_fiters() {

  double omega_init = 0.04 * (2 * PI);  // init frequency Hz * 2 * PI (start converging from omega_init/2)
  double k_gain = 100.0; // Aranovskiy gain. Higher value will give faster convergence, but too high will potentially overflow decimal
  double x1_0 = 0.0;
  double theta_0 = - (omega_init * omega_init / 4.0);
  double sigma_0 = theta_0;
  aranovskiy_default_params(&arParams, omega_init, k_gain);
  aranovskiy_init_state(&arState, x1_0, theta_0, sigma_0);

  {
    double process_noise_covariance = 0.25;
    double measurement_uncertainty = 2.0;
    double estimation_uncertainty = 100.0;
    kalman_smoother_init(&kalman_freq, process_noise_covariance, measurement_uncertainty, estimation_uncertainty);
  }

  kalman_wave_init_defaults();
  kalman_wave_alt_init_defaults();
}

int main(int argc, char *argv[]) {

  float sample_freq = 250.0; // Hz
  float delta_t = 1.0 / sample_freq;
  float test_duration = 2.0 * 60.0;

  init_fiters();

  t = 0.0;
  while (t < test_duration) {

    float displacement_amplitude = 2.0 /* 4m height */, frequency = 1.0 / 8.5 /* 8.5 sec period */, phase_rad = PI / 3.0;
    float a = trochoid_wave_vert_accel(displacement_amplitude, frequency, phase_rad, t);
    float v = trochoid_wave_vert_speed(displacement_amplitude, frequency, phase_rad, t);
    float h = trochoid_wave_displacement(displacement_amplitude, frequency, phase_rad, t);
    
    run_fiters(a / g_std, v, h, delta_t);

    t = t + delta_t;
  }
}
