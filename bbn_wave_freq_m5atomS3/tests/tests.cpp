/*
  Copyright 2024, Mikhail Grushinskiy
*/
#include <cmath>
#include <random>

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "MinMaxLemire.h"
#include "KalmanForWave.h"
#include "KalmanForWaveAlt.h"
#include "WaveFilters.h"
#include "SchmittTriggerFrequencyDetector.h"

enum FrequencyTracker {
    Aranovskiy,
    Kalm_ANF,
    ZeroCrossing
};

MinMaxLemire min_max_h;
AranovskiyParams arParams;
AranovskiyState arState;
KalmanSmootherVars kalman_freq;
KalmanWaveState waveState;
KalmanWaveAltState waveAltState;
KalmANF kalmANF;
SchmittTriggerFrequencyDetector freqDetector(0.002f); // Hysteresis

FrequencyTracker useFrequencyTracker = ZeroCrossing;

bool kalm_w_first = true, kalm_w_alt_first = true, kalm_smoother_first = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

uint32_t now() {
  return t * 1000000;
}

unsigned long last_update_k = 0UL;

void run_filters(float a, float v, float h, float delta_t) {
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

  double freq_adj = 0.0;
  double freq = 0.0;
  if (t > warmup_time_sec(true)) {
    // give some time for other filters to settle first
    if (useFrequencyTracker == Aranovskiy) {
      aranovskiy_update(&arParams, &arState, heave / ARANOVSKIY_SCALE, delta_t);
      freq = arState.f;
    } else if (useFrequencyTracker == Kalm_ANF) {
      float e;
      float f_kalmanANF = kalmANF_process(&kalmANF, heave, delta_t, &e);
      freq = f_kalmanANF;
    } else {
      float signal_a = a;
      float f_byZeroCross = freqDetector.update(signal_a, 2.0, delta_t); 
      freq = f_byZeroCross;
    }
    if (kalm_smoother_first) {
      kalm_smoother_first = false;
      kalman_smoother_set_initial(&kalman_freq, freq);
    }
    freq_adj = kalman_smoother_update(&kalman_freq, freq);
  }

  if (freq_adj > FREQ_LOWER && freq_adj < FREQ_UPPER) { /* prevent decimal overflows */
    double period = 1.0 / freq_adj;
    uint32_t windowMicros = getWindowMicros(period);
    SampleType sample = { .value = heave, .timeMicroSec = now() };
    min_max_lemire_update(&min_max_h, sample, windowMicros);

    if (fabs(freq - freq_adj) < FREQ_COEF * freq_adj) { /* sanity check of convergence for freq */
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
    printf(",heave,%.4f", heave);
    printf(",heave_alt,%.4f", (isnan(waveAltState.heave) || fabs(waveAltState.heave) > 100 ? 0.0f : waveAltState.heave));
    printf(",height,%.4f", wave_height);
    printf(",max,%.4f", min_max_h.max.value);
    printf(",min,%.4f", min_max_h.min.value);
    printf(",period,%.4f", period);
    printf(",freq:,%.4f", freq);
    printf(",freq_adj,%.4f", freq_adj);
    printf(",heave_avg,%.7f", heave_avg);
    printf(",accel_bias,%.5f", waveState.accel_bias);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  float sample_freq = 250.0; // Hz
  float delta_t = 1.0 / sample_freq;
  float test_duration = 5.0 * 60.0;

  if (useFrequencyTracker == Aranovskiy) {
    init_filters(&arParams, &arState, &kalman_freq);
  } else if (useFrequencyTracker == Kalm_ANF) {
    init_filters_alt(&kalmANF, &kalman_freq);
  } else {
    init_smoother(&kalman_freq);
    init_wave_filters();
  }

  //float displacement_amplitude = 0.135 /* 0.27m height */, frequency = 1.0 / 3.0 /* 3.0 sec period */, phase_rad = PI / 3.0;
  float displacement_amplitude = 0.75 /* 1.5m height */, frequency = 1.0 / 5.7 /* 5.7 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 2.0 /* 4m height */, frequency = 1.0 / 8.5 /* 8.5 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 4.25 /* 8.5m height */, frequency = 1.0 / 11.4 /* 11.4 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 7.4 /* 14.8m height */, frequency = 1.0 / 14.3 /* 14.3 sec period */, phase_rad = PI / 3.0;

  const float bias = 0.1;
  const double mean = 0.0;
  const double stddev = 0.05;
  std::default_random_engine generator;
  generator.seed(239);  // seed the engine for deterministic test results
  std::normal_distribution<float> dist(mean, stddev);

  t = 0.0;
  while (t < test_duration) {
    float zero_mean_gauss_noise = dist(generator);
    float a = trochoid_wave_vert_accel(displacement_amplitude, frequency, phase_rad, t) + bias + zero_mean_gauss_noise;
    float v = trochoid_wave_vert_speed(displacement_amplitude, frequency, phase_rad, t);
    float h = trochoid_wave_displacement(displacement_amplitude, frequency, phase_rad, t);

    run_filters(a / g_std, v, h, delta_t);

    t = t + delta_t;
  }
}
