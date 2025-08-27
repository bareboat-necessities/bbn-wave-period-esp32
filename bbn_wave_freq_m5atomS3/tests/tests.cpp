/*
  Copyright 2024-2025, Mikhail Grushinskiy
*/

#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_VECTORIZE
//#define EIGEN_MPL2_ONLY

#define FENTON_TEST
#define JONSWAP_TEST
#define PM_STOKES_TEST
#define KALMAN_WAVE_DIRECTION_TEST
#define SEA_STATE_TEST
#define SPECTRUM_TEST

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
#include "KalmanWaveDirection.h"
#include "WaveFilters.h"
#include "TimeAwareSpikeFilter.h"
#include "FentonWaveVectorized.h"
#include "WaveSurfaceProfile.h"
#include "Jonswap3dStokesWaves.h"
#include "PiersonMoskowitzStokes3D_Waves.h"
#include "SeaStateRegularity.h"
#include "WaveSpectrumEstimator.h"

enum TestType {
  GERSTEN = -1,
  FENTON = 0,
  JONSWAP = 1,
  PM_STOKES = 2
};

MinMaxLemire min_max_h;
AranovskiyFilter<double> arFilter;
KalmanSmootherVars kalman_freq;
KalmanForWaveBasicState waveState;
KalmanWaveNumStableAltState waveAltState;
KalmANF<double> kalmANF;
SchmittTriggerFrequencyDetector freqDetector(ZERO_CROSSINGS_HYSTERESIS, ZERO_CROSSINGS_PERIODS);
TimeAwareSpikeFilter spikeFilter(ACCEL_SPIKE_FILTER_SIZE, ACCEL_SPIKE_FILTER_THRESHOLD);
WaveSurfaceProfile<128> waveProfile;

FrequencyTracker useFrequencyTracker = Aranovskiy;

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
    float k_hat = - pow(2.0 * M_PI * FREQ_GUESS, 2);
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
    freq = estimate_freq(useFrequencyTracker, &arFilter, &kalmANF, &freqDetector, a_noisy, a_no_spikes, delta_t, now());
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

  float k_hat = - pow(2.0 * M_PI * freq /*freq_adj*/, 2);
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
    printf(",heave_predict,%.5f", waveProfile.predictAtPhase(0.0f /* zero phase shift */, t));
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  float sample_freq = 250.0; // Hz
  float delta_t = 1.0 / sample_freq;
  float test_duration = 5.0 * 60.0;

  if (useFrequencyTracker == Aranovskiy) {
    init_filters(&arFilter, &kalman_freq);
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
  const unsigned int seed = 239u;
  generator.seed(seed);  // seed the engine for deterministic test results
  std::normal_distribution<float> dist(mean, stddev);

  TrochoidalWave<float> w1 = TrochoidalWave<float>(0.135, 3.0, M_PI / 3.0);
  TrochoidalWave<float> w2 = TrochoidalWave<float>(0.75, 5.7, M_PI / 3.0);
  TrochoidalWave<float> w3 = TrochoidalWave<float>(2.0, 8.5, M_PI / 3.0);
  TrochoidalWave<float> w4 = TrochoidalWave<float>(4.25, 11.4, M_PI / 3.0);
  TrochoidalWave<float> w5 = TrochoidalWave<float>(7.4, 14.3, M_PI / 3.0);
  TrochoidalWave<float>* w = &w5;

  float frequency = 1.0 / w->period();
  float angularFrequency = w->angularFrequency();
  float amplitude = w->amplitude();
  float phase = w->phase();

  printf("main_amp,%.4f", w->amplitude());
  printf(",main_freq,%.4f", frequency);
  printf(",acc_bias,%.7f", bias);
  printf(",acc_noise_std_dev,%.5f", stddev);
  printf("\n");

  t = 0.0;

  TestType test_type = JONSWAP;
  if (test_type == TestType::GERSTEN) {
    while (t < test_duration) {
      float zero_mean_gauss_noise = dist(generator);
      float a = w->surfaceVerticalAcceleration(t) + bias + zero_mean_gauss_noise;
      float v = w->surfaceVerticalSpeed(t);
      float h = w->surfaceElevation(t);
      run_filters(a / g_std, v, h, delta_t, frequency);
      t = t + delta_t;
    }
  } else if (test_type == TestType::PM_STOKES) {
    PMStokesN3dWaves<256, 5> waveModel(w->amplitude(), w->period(), 30.0 /*dir*/, 0.02, 0.8, g_std, 15.0, seed);
    while (t < test_duration) {
      auto state = waveModel.getLagrangianState(t);
      float zero_mean_gauss_noise = dist(generator);
      float a = state.acceleration.z() + bias + zero_mean_gauss_noise;
      float v = state.velocity.z();
      float h = state.displacement.z();
      run_filters(a / g_std, v, h, delta_t, frequency);
      t += delta_t;
    }
  } else if (test_type == TestType::JONSWAP) {
    auto waveModel = std::make_unique<Jonswap3dStokesWaves<256>>(w->amplitude(), w->period(), 30.0 /*dir*/, 0.02, 0.8, 2.0, g_std, 15.0);
    while (t < test_duration) {
       auto state = waveModel->getLagrangianState(0.0, 0.0, t);
       float zero_mean_gauss_noise = dist(generator);
       float a = state.acceleration.z() + bias + zero_mean_gauss_noise;
       float v = state.velocity.z();
       float h = state.displacement.z();
       run_filters(a / g_std, v, h, delta_t, frequency);
       t = t + delta_t;
    }
  } else {
    // Create a 5th-order Fenton wave and a surface tracker
    FentonWave<5>::WaveInitParams wave_params = FentonWave<5>::infer_fenton_parameters_from_amplitude(
      amplitude, 200.0f, angularFrequency, phase);
    const float mass = 5.0f;     // Mass (kg)
    const float drag = 0.1f;     // Linear drag coeff opposing velocity
    WaveSurfaceTracker<5> tracker(wave_params.height, wave_params.depth, wave_params.length, wave_params.initial_x, mass, drag);
    auto kinematics_callback = [&frequency, &delta_t, &bias, &dist, &generator](
        float time, float dt, float elevation, float vertical_velocity, float vertical_acceleration, float horizontal_position, float horizontal_speed) {
      float zero_mean_gauss_noise = dist(generator);
      run_filters((vertical_acceleration + bias + zero_mean_gauss_noise) / g_std, vertical_velocity, elevation, dt, frequency);
      t = t + delta_t;
    };
    tracker.track_floating_object(test_duration, delta_t, kinematics_callback);
  }

#ifdef PM_STOKES_TEST
  PMStokes_testWavePatterns();
#endif

#ifdef JONSWAP_TEST
  Jonswap_testWavePatterns();
#endif

#ifdef FENTON_TEST
  FentonWave_test_1();
  FentonWave_test_2();
#endif

#ifdef KALMAN_WAVE_DIRECTION_TEST
  KalmanWaveDirection_test_1();
#endif

#ifdef SEA_STATE_TEST
  SeaState_sine_wave_test();
#endif

#ifdef SPECTRUM_TEST
  WaveSpectrumEstimator_test();
#endif
}
