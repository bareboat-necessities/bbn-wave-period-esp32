#ifndef KalmanForWave_h
#define KalmanForWave_h

/*
  Copyright 2024, Mikhail Grushinskiy

  Kalman filter to double integrate vertical acceleration in wave
  into vertical displacement, correct for accelerometer bias,
  estimate accelerometer bias, correct integral for zero average displacement.
  The third integral (responsible for zero average vertical displacement)
  is taken as a measurement of zero.

  Process model:

  velocity:
  v(k) = v(k-1) + a*T - a_hat(k-1)*T

  displacement:
  y(k) = y(k-1) + v(k-1)*T + 1/2*a*T^2 - 1/2*a_hat(k-1)*T^2

  displacement_integral:
  z(k) = z(k-1) + y(k-1)*T + 1/2*v(k-1)*T^2 + 1/6*a*T^3 - 1/6*a_hat(k-1)*T^3

  accelerometer bias:
  a_hat(k) = a_hat(k-1)

  State vector:
  
  x = [ z, 
        y,
        v,
        a_hat ]

  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = a 

  Input a - vertical acceleration from accelerometer

  Measurement - z = 0 (displacement integral)

  Observation matrix:
  H = [ 1, 
        0,
        0,
        0 ]  

  F = [[ 1,  T,  1/2*T^2, -1/6*T^3 ],
       [ 0,  1,  T,       -1/2*T^2 ],
       [ 0,  0,  1,       -T       ],
       [ 0,  0,  0,        1       ]]

  B = [  1/6*T^3,
         1/2*T^2,
         T,
         0       ]

  The issue with this model is that while it is hinted to oscilate around mid sea level,
  there is no hint on for a scale of oscillations. So we depend on selecting variances and
  covariances properly. Another way would be to use this method for finding wave frequency,
  use trochoidal wave model in which apmplitude is derived from acceleration and frequency
  (or from ratio of heave and acceleration) and use heave estimate from trochoidal wave
  model to fuse with accelerometer data. Basically feed something into F matrix on each step
  (or every few steps) from estimate provided by trochoidal wave model.
         
*/

#include <assert.h>

// create the filter structure
#define KALMAN_NAME wave
#define KALMAN_NUM_STATES 4
#define KALMAN_NUM_INPUTS 1
#include "KalmanFactoryFilter.h"

// create the measurement structure
#define KALMAN_MEASUREMENT_NAME displacement_integral
#define KALMAN_NUM_MEASUREMENTS 1
#include "KalmanFactoryMeasurement.h"

// clean up
#include "KalmanFactoryCleanup.h"

typedef struct kalman_wave_state {
  float displacement_integral; // displacement integral
  float heave;                 // vertical displacement
  float vert_speed;            // vertical velocity
  float accel_bias;            // accel bias
} KalmanWaveState;

matrix_t *kalman_wave_get_state_transition(kalman_t *kf, matrix_data_t delta_t) {
  // transition matrix [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *F = kalman_get_state_transition(kf);

  matrix_set(F, 0, 0, (matrix_data_t)1.0);                                         // 1
  matrix_set(F, 0, 1, (matrix_data_t)delta_t);                                     // T
  matrix_set(F, 0, 2, (matrix_data_t)0.5 * delta_t * delta_t);                     // 0.5 * T^2
  matrix_set(F, 0, 3, (matrix_data_t)(-1.0 / 6.0) * delta_t * delta_t * delta_t);  // -(1/6) * T^3

  matrix_set(F, 1, 0, (matrix_data_t)0.0);                         // 0
  matrix_set(F, 1, 1, (matrix_data_t)1.0);                         // 1
  matrix_set(F, 1, 2, (matrix_data_t)delta_t);                     // T
  matrix_set(F, 1, 3, (matrix_data_t)-0.5 * delta_t * delta_t);    // -0.5 * T^2

  matrix_set(F, 2, 0, (matrix_data_t)0.0);              // 0
  matrix_set(F, 2, 1, (matrix_data_t)0.0);              // 0
  matrix_set(F, 2, 2, (matrix_data_t)1.0);              // 1
  matrix_set(F, 2, 3, (matrix_data_t)-delta_t);         // -T

  matrix_set(F, 3, 0, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 1, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 2, (matrix_data_t)0.0);              // 0
  matrix_set(F, 3, 3, (matrix_data_t)1.0);              // 1
  return F;
}

matrix_t *kalman_wave_get_transition_offset(kalman_t *kf, matrix_data_t delta_t) {
  // transition offset [KALMAN_NUM_STATES * KALMAN_NUM_INPUTS]
  matrix_t *B = kalman_get_input_transition(kf);
  matrix_set(B, 0, 0, (matrix_data_t)(1.0 / 6.0) * delta_t * delta_t * delta_t);
  matrix_set(B, 1, 0, (matrix_data_t)0.5 * delta_t * delta_t);
  matrix_set(B, 2, 0, (matrix_data_t)delta_t);
  matrix_set(B, 3, 0, (matrix_data_t)0.0);
  return B;
}

void kalman_wave_init_state(KalmanWaveState* state) {
  kalman_t *kf = &kalman_filter_wave;
  
  // [KALMAN_NUM_STATES * 1]
  matrix_t *x = kalman_get_state_vector(kf);
  x->data[0] = state->displacement_integral; // displacement integral
  x->data[1] = state->heave;                 // vertical displacement
  x->data[2] = state->vert_speed;            // vertical velocity
  x->data[3] = state->accel_bias;            // accel bias  
}

void kalman_wave_init_defaults(float q0, float q1, float q2, float q3) {

  kalman_t *kf = kalman_filter_wave_init();
  kalman_measurement_t *kfm = kalman_filter_wave_measurement_displacement_integral_init();

  // [KALMAN_NUM_STATES * 1]
  matrix_t *x = kalman_get_state_vector(kf);
  x->data[0] = 0.0; // displacement integral
  x->data[1] = 0.0; // vertical displacement
  x->data[2] = 0.0; // vertical velocity
  x->data[3] = 0.0; // accel bias

  // observation matrix [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_STATES]
  matrix_t *H = kalman_get_measurement_transformation(kfm);
  matrix_set(H, 0, 0, (matrix_data_t)1.0);
  matrix_set(H, 0, 1, (matrix_data_t)0.0);
  matrix_set(H, 0, 2, (matrix_data_t)0.0);
  matrix_set(H, 0, 3, (matrix_data_t)0.0);

  // observation covariance [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_MEASUREMENTS]
  matrix_t *R = kalman_get_observation_noise(kfm);
  matrix_set(R, 0, 0, (matrix_data_t)0.01);

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
  matrix_set_symmetric(Q, 0, 0, (matrix_data_t)q0);
  matrix_set_symmetric(Q, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 1, (matrix_data_t)q1);
  matrix_set_symmetric(Q, 1, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 2, 2, (matrix_data_t)q2);
  matrix_set_symmetric(Q, 2, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 3, 3, (matrix_data_t)q3);
}

void kalman_wave_step(KalmanWaveState* state, float accel, float delta_t) {
  kalman_t *kf = &kalman_filter_wave;
  kalman_measurement_t *kfm = &kalman_filter_wave_measurement_displacement_integral;

  matrix_t *x = kalman_get_state_vector(kf);
  matrix_t *z = kalman_get_measurement_vector(kfm);

  matrix_t *F = kalman_wave_get_state_transition(kf, delta_t);
  matrix_t *B = kalman_wave_get_transition_offset(kf, delta_t);

  // input vector [KALMAN_NUM_INPUTS * 1]
  matrix_t *u = kalman_get_input_vector(kf);
  matrix_set(u, 0, 0, (matrix_data_t)accel);

  // prediction.
  kalman_predict(kf);

  // measure ...
  matrix_data_t measurement = 0.0f;
  matrix_set(z, 0, 0, measurement);

  // update
  kalman_correct(kf, kfm);

  state->displacement_integral = x->data[0];
  state->heave = x->data[1];
  state->vert_speed = x->data[2];
  state->accel_bias = x->data[3];
}

#endif
