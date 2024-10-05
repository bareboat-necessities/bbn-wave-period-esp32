#ifndef WaveFilters_h
#define WaveFilters_h

#define FREQ_LOWER 0.004
#define FREQ_UPPER 4.0
#define FREQ_GUESS 0.3 /* freq guess */

#define FREQ_COEF        1.0
#define FREQ_COEF_TIGHT  0.05
#define ARANOVSKIY_SCALE 10.0

#define ACCEL_MAX_G_SQUARE 16.0

void init_aranovskiy(AranovskiyParams* ar_param, AranovskiyState* ar_state);
void init_smoother(KalmanSmootherVars* kalman_smoother);
void init_filters(AranovskiyState* ar_param, AranovskiyState* ar_state, KalmanSmootherVars* kalman_smoother);
int warmup_time_sec(bool use_mahony);

/*
  From experiments QMEKF somehow introduces bigger bias of vertical acceleration.
  Longer warm up time is needed to engage Aranovskiy filter.
*/
int warmup_time_sec(bool use_mahony) {
  return use_mahony ? 20 : 120;
}

uint32_t getWindowMicros(double period) {
  uint32_t windowMicros = 3 * period * 1000000;
  if (windowMicros <= 10 * 1000000) {
    windowMicros = 10 * 1000000;
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
  double process_noise_covariance = 0.25;
  double measurement_uncertainty = 2.0;
  double estimation_uncertainty = 100.0;
  kalman_smoother_init(kalman_smoother, process_noise_covariance, measurement_uncertainty, estimation_uncertainty);
}

void init_filters(AranovskiyParams* ar_param, AranovskiyState* ar_state, KalmanSmootherVars* kalman_smoother) {
  init_aranovskiy(ar_param, ar_state);
  init_smoother(kalman_smoother);
  kalman_wave_init_defaults(20.0, 0.2, 0.04, 0.0002);
  kalman_wave_alt_init_defaults(20.0, 0.2, 0.04, 1.0e6, 0.0002);
}

#endif
