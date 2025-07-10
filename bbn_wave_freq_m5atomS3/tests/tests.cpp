/*
  Copyright 2024-2025, Mikhail Grushinskiy
*/

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_VECTORIZE
#define EIGEN_MPL2_ONLY

#define FENTON_TEST

#include <cmath>
#include <random>

#include "AranovskiyFilter.h"
#include "KalmANF.h"
#include "KalmanSmoother.h"
#include "TrochoidalWave.h"
#include "MinMaxLemire.h"
#include "KalmanForWaveBasic.h"
#include "KalmanWaveNumStableAlt.h"
#include "SchmittTriggerFrequencyDetector.h"
#include "WaveFilters.h"
#include "TimeAwareSpikeFilter.h"
#include "FentonWaveVectorized.h"
#include "WaveSurfaceProfile.h"

MinMaxLemire min_max_h;
AranovskiyParams arParams;
AranovskiyState arState;
KalmanSmootherVars kalman_freq;
KalmanForWaveBasicState waveState;
KalmanWaveNumStableAltState waveAltState;
KalmANF kalmANF;
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);
TimeAwareSpikeFilter spikeFilter(ACCEL_SPIKE_FILTER_SIZE, ACCEL_SPIKE_FILTER_THRESHOLD);
WaveSurfaceProfile<128> waveProfile;

FrequencyTracker useFrequencyTracker = ZeroCrossing;

bool kalm_w_first = true, kalm_w_alt_first = true, kalm_smoother_first = true;

float t = 0.0;
float heave_avg = 0.0;
float wave_length = 0.0;

uint32_t now() {
  return t * 1000000;
}

unsigned long last_update_k = 0UL;

void run_filters(float a_noisy, float v, float h, float delta_t, float ref_freq_4_print) {
  float a_band_passed = a_noisy; //bpFilter.processWithDelta(a_noisy, delta_t);
  float a_no_spikes = spikeFilter.filterWithDelta(a_band_passed, delta_t);

  float a = a_no_spikes;

  if (kalm_w_first) {
    kalm_w_first = false;
    float k_hat = - pow(2.0 * PI * FREQ_GUESS, 2);
    waveState.displacement_integral = 0.0f;
    waveState.heave = a * g_std / k_hat;
    waveState.vert_speed = 0.0f;               // ??
    waveState.accel_bias = 0.0f;
    wave_filter.initState(waveState);
  }
  wave_filter.step(a * g_std, delta_t, waveState);
  float heave = waveState.heave;

  double freq = FREQ_GUESS, freq_adj = FREQ_GUESS;
  float warm_up_time = warmup_time_sec(true);
  if (t > warm_up_time) {
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
  wave_alt_filter.update(a * g_std, k_hat, delta_t);
  waveAltState = wave_alt_filter.getState();
  heaveAlt = waveAltState.heave;

  double period = 1.0 / freq_adj;
  uint32_t windowMicros = getWindowMicros(period);
  SampleType sample = { .value = heaveAlt, .timeMicroSec = now() };
  min_max_lemire_update(&min_max_h, sample, windowMicros);

  float wave_height = min_max_h.max.value - min_max_h.min.value;
  heave_avg = (min_max_h.max.value + min_max_h.min.value) / 2.0;

  waveProfile.updateIfNeeded(heaveAlt, freq_adj, t);
    
  if (t > warm_up_time + 45.0) {
    printf("time,%.5f", t);
    printf(",a,%.4f", a * g_std);
    printf(",v,%.4f", v);
    printf(",h,%.4f", h);
    printf(",heave,%.4f", heave);
    printf(",heave_alt,%.4f", heaveAlt);
    printf(",height,%.4f", wave_height);
    printf(",max,%.4f", min_max_h.max.value);
    printf(",min,%.4f", min_max_h.min.value);
    printf(",period,%.4f", period);
    printf(",freq,%.4f", freq);
    printf(",freq_adj,%.4f", freq_adj);
    printf(",heave_avg,%.7f", heave_avg);
    printf(",accel_bias,%.5f", waveAltState.accel_bias);
    printf(",ref_req,%.5f", ref_freq_4_print);
    printf(",heave_alt_err,%.5f", h - heaveAlt);
    printf(",freq_adj_err,%.5f", ref_freq_4_print - freq_adj);
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
    kalman_smoother_init(&kalman_freq, 0.25f, 2.0f, 100.0f);
    init_wave_filters();
  }

  const float bias = 0.1f;      // m/s^2
  const double mean = 0.0f;     // m/s^2
  const double stddev = 0.08f;  // m/s^2
  std::default_random_engine generator;
  generator.seed(239);  // seed the engine for deterministic test results
  std::normal_distribution<float> dist(mean, stddev);

  //float displacement_amplitude = 0.135 /* 0.27m height */, frequency = 1.0 / 3.0 /* 3.0 sec period */, phase_rad = PI / 3.0;
  float displacement_amplitude = 0.75 /* 1.5m height */, frequency = 1.0 / 5.7 /* 5.7 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 2.0 /* 4m height */, frequency = 1.0 / 8.5 /* 8.5 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 4.25 /* 8.5m height */, frequency = 1.0 / 11.4 /* 11.4 sec period */, phase_rad = PI / 3.0;
  //float displacement_amplitude = 7.4 /* 14.8m height */, frequency = 1.0 / 14.3 /* 14.3 sec period */, phase_rad = PI / 3.0;

  printf("main_amp,%.4f", displacement_amplitude);
  printf(",main_freq,%.4f", frequency);
  printf(",acc_bias,%.7f", bias);
  printf(",acc_noise_std_dev,%.5f", stddev);
  printf("\n");

  t = 0.0;

  bool test_trochoid = false;
  if (test_trochoid) {
    while (t < test_duration) {
      float zero_mean_gauss_noise = dist(generator);
      float a = trochoid_wave_vert_accel(displacement_amplitude, frequency, phase_rad, t) + bias + zero_mean_gauss_noise;
      float v = trochoid_wave_vert_speed(displacement_amplitude, frequency, phase_rad, t);
      float h = trochoid_wave_displacement(displacement_amplitude, frequency, phase_rad, t);

      run_filters(a / g_std, v, h, delta_t, frequency);

      t = t + delta_t;
    }
  } else {
    // Create a 4th-order Fenton wave and a surface tracker
    WaveInitParams wave_params = FentonWave<4>::infer_fenton_parameters_from_amplitude(
      displacement_amplitude, 200.0f, 2 * M_PI * frequency, phase_rad);

    const float mass = 5.0f;     // Mass (kg)
    const float drag = 0.1f;     // Linear drag coeff opposing velocity

    WaveSurfaceTracker<4> tracker(wave_params.height, wave_params.depth, wave_params.length, wave_params.initial_x, mass, drag);

    auto kinematics_callback = [&frequency, &delta_t, &bias, &dist, &generator](
        float time, float dt, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position, float horizontal_speed) {
      float zero_mean_gauss_noise = dist(generator);
      run_filters((vertical_acceleration + bias + zero_mean_gauss_noise) / g_std, vertical_velocity, elevation, dt, frequency);
      t = t + delta_t;
    };

    tracker.track_floating_object(test_duration, delta_t, kinematics_callback);
  }

#ifdef FENTON_TEST
  FentonWave_test_1();
  FentonWave_test_2();
#endif
  
}
