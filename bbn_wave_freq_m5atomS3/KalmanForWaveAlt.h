#ifndef KalmanForWaveAlt_h
#define KalmanForWaveAlt_h

/*
  Copyright 2024, Mikhail Grushinskiy

  Kalman filter to estimate vertical displacement in wave using accelerometer, 
  correct for accelerometer bias, estimate accelerometer bias. This method
  assumes that displacement follows trochoidal model and the frequency of
  wave is known. Frequency can be estimated using another step with Aranovskiy filter.

  In trochoidal wave model there is simple linear dependency between displacement and 
  acceleration.

  y - displacement (at any time):
  y = - L / (2 *pi) * (a/g),  g - acceleration of free fall constant, a - vertical acceleration

  wave length L: 
  L = g * period^2 / (2 *pi)

  wave period via frequency:
  period = 1 / f

  a = - (2 * pi * f)^2 * y

  let
  k_hat = - (2 * pi * f)^2

  Process model:

  displacement:
  y(k) = y(k-1) + v(k-1)*T + 1/2*a(k-1)*T^2 - 1/2*a_hat*t^2

  velocity:
  v(k) = v(k-1) + a(k-1)*T - a_hat*T

  acceleration (from trochoidal wave model):
  a(k) = k_hat * y(k-1)

  accelerometer bias:
  a_hat(k) = a_hat(k-1)

  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = 0 
  
  State vector:
  
  x = [ y,
        v,
        a,
        a_hat ]

  Input a - vertical acceleration from accelerometer

  Measurement - a (vertical acceleration)

  Observation matrix:
  H = [ 0, 
        0,
        1,
        0 ]  

  F = [[ 1,      T,    1/2*T^2, -1/2*T^2 ],
       [ 0,      1,    T,       -T       ],
       [ k_hat,  0,    0,        0       ],
       [ 0,      0,    0,        1       ]]
         
*/

#include <assert.h>

// create the filter structure
#define KALMAN_NAME wave_alt
#define KALMAN_NUM_STATES 4
#define KALMAN_NUM_INPUTS 1
#include "KalmanFactoryFilter.h"

// create the measurement structure
#define KALMAN_MEASUREMENT_NAME vert_accel
#define KALMAN_NUM_MEASUREMENTS 1
#include "KalmanFactoryMeasurement.h"

// clean up
#include "KalmanFactoryCleanup.h"

typedef struct kalman_wave_alt_state {
  float heave;                 // vertical displacement
  float vert_speed;            // vertical velocity
  float vert_accel;            // vertical acceleration
  float accel_bias;            // accel bias
} KalmanWaveAltState;

matrix_t *kalman_wave_alt_get_state_transition(kalman_t *kf, matrix_data_t k_hat, matrix_data_t delta_t) {
  // transition matrix [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *F = kalman_get_state_transition(kf);

  matrix_set(F, 0, 0, (matrix_data_t)1.0);                                         // 1
  matrix_set(F, 0, 1, (matrix_data_t)delta_t);                                     // T
  matrix_set(F, 0, 2, (matrix_data_t)0.5 * delta_t * delta_t);                     // 0.5 * T^2
  matrix_set(F, 0, 3, (matrix_data_t)-0.5 * delta_t * delta_t);                    // -0.5 * T^2

  matrix_set(F, 1, 0, (matrix_data_t)0.0);                         // 0
  matrix_set(F, 1, 1, (matrix_data_t)1.0);                         // 1
  matrix_set(F, 1, 2, (matrix_data_t)delta_t);                     // T
  matrix_set(F, 1, 3, (matrix_data_t)-delta_t);                    // -T

  matrix_set(F, 2, 0, (matrix_data_t)k_hat);                // k
  matrix_set(F, 2, 1, (matrix_data_t)0.0);              // 0
  matrix_set(F, 2, 2, (matrix_data_t)0.0);              // 0
  matrix_set(F, 2, 3, (matrix_data_t)0.0);              // 0

  matrix_set(F, 3, 0, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 1, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 2, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 3, (matrix_data_t)1.0);              // 1
  return F;
}

void kalman_wave_alt_init_defaults() {

  kalman_t *kf = kalman_filter_wave_alt_init();
  kalman_measurement_t *kfm = kalman_filter_wave_alt_measurement_vert_accel_init();

  // [KALMAN_NUM_STATES * 1]
  matrix_t *x = kalman_get_state_vector(kf);
  x->data[1] = 0.0; // vertical displacement
  x->data[2] = 0.0; // vertical velocity
  x->data[0] = 0.0; // vertical accel
  x->data[3] = 0.0; // accel bias

  // observation matrix [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_STATES]
  matrix_t *H = kalman_get_measurement_transformation(kfm);
  matrix_set(H, 0, 0, (matrix_data_t)0.0);
  matrix_set(H, 0, 1, (matrix_data_t)0.0);
  matrix_set(H, 0, 2, (matrix_data_t)1.0);
  matrix_set(H, 0, 3, (matrix_data_t)0.0);

  // observation covariance [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_MEASUREMENTS]
  matrix_t *R = kalman_get_process_noise(kf);
  matrix_set(R, 0, 0, (matrix_data_t)1.0);

  // initial state covariance [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *P = kalman_get_system_covariance(kf);
  matrix_set_symmetric(P, 0, 0, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 0, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 0, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 1, 1, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 1, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 1, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 2, 2, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 2, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 3, 3, (matrix_data_t)1.0);

  // transition covariance [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *Q = kalman_get_process_noise(kf);
  matrix_data_t variance = (matrix_data_t) 1.0;
  matrix_set_symmetric(Q, 0, 0, (matrix_data_t)variance);
  matrix_set_symmetric(Q, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 1, (matrix_data_t)0.2 * variance);
  matrix_set_symmetric(Q, 1, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 2, 2, (matrix_data_t)0.04 * variance);
  matrix_set_symmetric(Q, 2, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 3, 3, (matrix_data_t)0.008 * variance);
}

matrix_t *kalman_wave_alt_get_transition_offset(kalman_t *kf, matrix_data_t delta_t) {
  // transition offset [KALMAN_NUM_STATES * KALMAN_NUM_INPUTS]
  matrix_t *B = kalman_get_input_transition(kf);
  matrix_set(B, 0, 0, (matrix_data_t)0.0);
  matrix_set(B, 1, 0, (matrix_data_t)0.0);
  matrix_set(B, 2, 0, (matrix_data_t)0.0);
  matrix_set(B, 3, 0, (matrix_data_t)0.0);
  return B;
}

void kalman_wave_alt_step(KalmanWaveAltState* state, float accel, float k_hat, float delta_t) {
  kalman_t *kf = &kalman_filter_wave_alt;
  kalman_measurement_t *kfm = &kalman_filter_wave_alt_measurement_vert_accel;

  matrix_t *x = kalman_get_state_vector(kf);
  matrix_t *z = kalman_get_measurement_vector(kfm);

  matrix_t *F = kalman_wave_alt_get_state_transition(kf, k_hat, delta_t);
  matrix_t *B = kalman_wave_alt_get_transition_offset(kf, delta_t);

  // input vector [KALMAN_NUM_INPUTS * 1]
  matrix_t *u = kalman_get_input_vector(kf);
  matrix_set(u, 0, 0, 0.0);
  
  // prediction.
  kalman_predict(kf);

  // measure ...
  matrix_data_t measurement = accel;
  matrix_set(z, 0, 0, measurement);

  // update
  kalman_correct(kf, kfm);

  state->heave = x->data[0];
  state->vert_speed = x->data[1];
  state->vert_accel = x->data[2];
  state->accel_bias = x->data[3];
}

#endif
