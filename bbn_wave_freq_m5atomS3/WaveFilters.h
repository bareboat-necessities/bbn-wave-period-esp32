#ifndef WaveFilters_h
#define WaveFilters_h

#define FREQ_LOWER 0.04f
#define FREQ_UPPER 2.0f
#define FREQ_GUESS 0.3f /* frequency guess */

#define ZERO_CROSSINGS_HYSTERESIS     0.05f
#define ZERO_CROSSINGS_PERIODS        1
#define ZERO_CROSSINGS_SCALE          1.0f
#define ZERO_CROSSINGS_DEBOUNCE_TIME  0.12f
#define ZERO_CROSSINGS_STEEPNESS_TIME 0.16f

#define FREQ_COEF        3.0f
#define FREQ_COEF_TIGHT  0.30f
#define ARANOVSKIY_SCALE 10.0f

#define ACCEL_MAX_G_SQUARE 4.84f  // (a/g)^2
#define ACCEL_MAX_G_SQUARE_NO_GRAVITY 1.44f  // (a/g)^2

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
  if (windowMicros <= 5 * 1000000) {
    windowMicros = 5 * 1000000;
  }
  else if (windowMicros >= 30 * 1000000) {
    windowMicros = 30 * 1000000;
  }
  return windowMicros;
}

void init_aranovskiy(AranovskiyParams* ar_param, AranovskiyState* ar_state) {
  /*
    Accelerometer bias creates heave bias and Aranovskiy filter gives
    lower frequency (i. e. higher period).
    Even 2cm bias in heave is too much to affect frequency a lot
  */
  double omega_init = 0.25 * (2 * PI);  // init frequency Hz * 2 * PI (start converging from omega_init/2)
  double k_gain = 200.0; // Aranovskiy gain. Higher value will give faster convergence, but too high will potentially overflow decimal
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
  kalman_wave_init_defaults(1e+1f, 1e-4f, 1e-2f, 1e-5f);
  kalman_wave_alt_init_defaults(1e+1f, 1e-4f, 1e-2f, 2.0f, 1e-5f);
}

void init_filters(AranovskiyParams* ar_param, AranovskiyState* ar_state, KalmanSmootherVars* kalman_smoother) {
  init_aranovskiy(ar_param, ar_state);
  init_smoother(kalman_smoother);
  init_wave_filters();
}

void init_filters_alt(KalmANF* kalmANF, KalmanSmootherVars* kalman_smoother) {
  kalmANF_init(kalmANF, 0.95f, 1e-5f, 1000000.0f, 0.0f, 0.0f, 0.0f, 1.9999f);
  init_smoother(kalman_smoother);
  init_wave_filters();
}

#endif
