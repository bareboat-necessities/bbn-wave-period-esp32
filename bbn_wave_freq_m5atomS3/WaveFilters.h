#ifndef WaveFilters_h
#define WaveFilters_h

#define FREQ_LOWER 0.04f
#define FREQ_UPPER 2.0f
#define FREQ_GUESS 0.3f   // frequency guess

#define ZERO_CROSSINGS_HYSTERESIS     0.05f
#define ZERO_CROSSINGS_PERIODS        1
#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.16f

#define FREQ_COEF        3.0f

#define ACCEL_CLAMP 0.5f  // fractions of G

#define ACCEL_SPIKE_FILTER_SIZE       5  
#define ACCEL_SPIKE_FILTER_THRESHOLD  1.0f

enum FrequencyTracker {
    Aranovskiy,
    Kalm_ANF,
    ZeroCrossing
};

void init_aranovskiy(AranovskiyParams* ar_param, AranovskiyState* ar_state);
void init_smoother(KalmanSmootherVars* kalman_smoother);
void init_filters(AranovskiyState* ar_param, AranovskiyState* ar_state, KalmanSmootherVars* kalman_smoother);
void init_filters_alt(KalmANF* kalmANF, KalmanSmootherVars* kalman_smoother);
void init_wave_filters();

KalmanForWaveBasic wave_filter;
//KalmanWaveAdaptiveAlt wave_alt_filter;
KalmanWaveNumStableAlt wave_alt_filter;

template <typename T> T clamp(T val, T min, T max) {
  return (val < min) ? min : (val > max) ? max : val;
}

int warmup_time_sec(bool use_mahony);

/*
  From experiments QMEKF somehow introduces bigger bias of vertical acceleration.
  Longer warm up time is needed to engage Aranovskiy filter.
*/
int warmup_time_sec(bool use_mahony) {
  return use_mahony ? 20 : 120;
}

uint32_t getWindowMicros(double period) {
  uint32_t windowMicros = period * 1000000;
  return clamp(windowMicros, (uint32_t) 5 * 1000000, (uint32_t) 30 * 1000000);
}

void init_aranovskiy(AranovskiyParams* ar_param, AranovskiyState* ar_state) {
  /*
    Accelerometer bias creates heave bias and Aranovskiy filter gives
    lower frequency (i. e. higher period).
    Even 2cm bias in heave is too much to affect frequency a lot
  */
  double omega_init = (FREQ_GUESS * 2) * (2 * PI);  // init frequency Hz * 2 * PI (start converging from omega_init/2)
  double k_gain = 8.0; // Aranovskiy gain. Higher value will give faster convergence, but too high will potentially overflow decimal
  double x1_0 = 0.0;
  double theta_0 = - (omega_init * omega_init / 4.0);
  double sigma_0 = theta_0;
  aranovskiy_default_params(ar_param, omega_init, k_gain);
  aranovskiy_init_state(ar_state, x1_0, theta_0, sigma_0);
}

void init_smoother(KalmanSmootherVars* kalman_smoother) {
  double process_noise_covariance = 0.25f;
  double measurement_uncertainty = 2.0f;
  double estimation_uncertainty = 100.0f;
  kalman_smoother_init(kalman_smoother, process_noise_covariance, measurement_uncertainty, estimation_uncertainty);
}

void init_wave_filters() {
  wave_filter.initialize(5.0f, 1e-4f, 1e-2f, 1e-5f);
  wave_filter.initMeasurementNoise(1e-3f);
  wave_alt_filter.initialize(5.0f, 1e-4f, 1e-2f, 5.0f, 1e-5f);
  wave_alt_filter.initMeasurementNoise(1e-3f, 1e-2f);    
}

void init_filters(AranovskiyParams* ar_param, AranovskiyState* ar_state, KalmanSmootherVars* kalman_smoother) {
  init_aranovskiy(ar_param, ar_state);
  init_smoother(kalman_smoother);
  init_wave_filters();
}

void init_filters_alt(KalmANF* kalmANF, KalmanSmootherVars* kalman_smoother) {
  kalmANF_init(kalmANF, 0.985f, 1e-5f, 5e+4f, 1.0f, 0.0f, 0.0f, 1.9999f);
  init_smoother(kalman_smoother);
  init_wave_filters();
}

float estimate_freq(FrequencyTracker tracker, AranovskiyParams* arParams, AranovskiyState* arState, KalmANF* kalmANF,
                    SchmittTriggerFrequencyDetector* freqDetector, float a_noisy, float a_no_spikes, float delta_t) {
  float freq = FREQ_GUESS;
  if (tracker == Aranovskiy) {
    aranovskiy_update(arParams, arState, a_no_spikes, delta_t);
    freq = arState->f;
  } else if (tracker == Kalm_ANF) {
    float e;
    float f_kalmanANF = kalmANF_process(kalmANF, a_noisy, delta_t, &e);
    freq = f_kalmanANF;
  } else {
    float f_byZeroCross = freqDetector->update(a_noisy, ZERO_CROSSINGS_SCALE /* max fractions of g */,
                          ZERO_CROSSINGS_DEBOUNCE_TIME, ZERO_CROSSINGS_STEEPNESS_TIME, delta_t);
    if (f_byZeroCross == SCHMITT_TRIGGER_FREQ_INIT || f_byZeroCross == SCHMITT_TRIGGER_FALLBACK_FREQ) {
      freq = FREQ_GUESS;
    } else {
      freq = f_byZeroCross;
    }
  }
  return clamp(freq, FREQ_LOWER, FREQ_UPPER);
}

#endif
