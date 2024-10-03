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

  displacement_integral:
  z(k) = z(k-1) + y(k-1)*T + 1/2*v(k-1)*T^2 + 1/6*a(k-1)*T^3 - 1/6*a_hat(k-1)*T^3
  
  displacement:
  y(k) = y(k-1) + v(k-1)*T + 1/2*a(k-1)*T^2 - 1/2*a_hat(k-1)*T^2

  velocity:
  v(k) = v(k-1) + a(k-1)*T - a_hat(k-1)*T

  acceleration (from trochoidal wave model):
  a(k) = k_hat*y(k-1) + k_hat*v(k-1)*T + k_hat*1/2*a(k-1)*T^2 - k_hat*1/2*a_hat(k-1)*T^2

  accelerometer bias:
  a_hat(k) = a_hat(k-1)

  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = 0 
  
  State vector:
  
  x = [ z,
        y,
        v,
        a,
        a_hat ]

  Input a - vertical acceleration from accelerometer

  Measurements:
    a (vertical acceleration), z = 0

  Observation matrix:
  H = [[ 1, 0 ],
       [ 0, 0 ],
       [ 0, 0 ],
       [ 0, 1 ],
       [ 0, 0 ]]  

  F = [[ 1,      T,    1/2*T^2,       1/6*T^3,         -1/6*T^3         ],
       [ 0,      1,    T,             1/2*T^2,         -1/2*T^2         ],
       [ 0,      0,    1,             T,               -T               ],
       [ 0,  k_hat,    k_hat*T,       1/2*k_hat*T^2,   -1/2*k_hat*T^2   ],
       [ 0,      0,    0,             0,               1                ]]
         
*/

#include <assert.h>

// create the filter structure
#define KALMAN_NAME wave_alt
#define KALMAN_NUM_STATES 5
#define KALMAN_NUM_INPUTS 0
#include "KalmanFactoryFilter.h"

// create the measurement structure
#define KALMAN_MEASUREMENT_NAME vert_accel
#define KALMAN_NUM_MEASUREMENTS 2
#include "KalmanFactoryMeasurement.h"

// clean up
#include "KalmanFactoryCleanup.h"

typedef struct kalman_wave_alt_state {
  float displacement_integral; // displacement integral
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
  matrix_set(F, 0, 3, (matrix_data_t)(1.0 / 6.0) * delta_t * delta_t * delta_t);   // (1/6) * T^3
  matrix_set(F, 0, 4, (matrix_data_t)(-1.0 / 6.0) * delta_t * delta_t * delta_t);  // -(1/6) * T^3

  matrix_set(F, 1, 0, (matrix_data_t)0.0);                         // 0
  matrix_set(F, 1, 1, (matrix_data_t)1.0);                         // 1
  matrix_set(F, 1, 2, (matrix_data_t)delta_t);                     // T
  matrix_set(F, 1, 3, (matrix_data_t)0.5 * delta_t * delta_t);     // 0.5 * T^2
  matrix_set(F, 1, 4, (matrix_data_t)-0.5 * delta_t * delta_t);    // -0.5 * T^2

  matrix_set(F, 2, 0, (matrix_data_t)0.0);                         // 0
  matrix_set(F, 2, 1, (matrix_data_t)0.0);                         // 0
  matrix_set(F, 2, 2, (matrix_data_t)1.0);                         // 1
  matrix_set(F, 2, 3, (matrix_data_t)delta_t);                     // T
  matrix_set(F, 2, 4, (matrix_data_t)-delta_t);                    // -T

  matrix_set(F, 3, 0, (matrix_data_t)0.0);                                 // 0
  matrix_set(F, 3, 1, (matrix_data_t)k_hat);                               // k_hat
  matrix_set(F, 3, 2, (matrix_data_t)k_hat * delta_t);                     // k_hat * T
  matrix_set(F, 3, 3, (matrix_data_t)0.5 * k_hat * delta_t * delta_t);     // 0.5 * k_hat * T^2
  matrix_set(F, 3, 4, (matrix_data_t)-0.5 * k_hat * delta_t * delta_t);    // -0.5 * k_hat * T^2

  matrix_set(F, 4, 0, (matrix_data_t)0.0);              // 0
  matrix_set(F, 4, 1, (matrix_data_t)0.0);              // 0
  matrix_set(F, 4, 2, (matrix_data_t)0.0);              // 0
  matrix_set(F, 4, 3, (matrix_data_t)0.0);              // 0
  matrix_set(F, 4, 4, (matrix_data_t)1.0);              // 1
  return F;
}

void kalman_wave_alt_init_state(KalmanWaveAltState* state) {
  kalman_t *kf = &kalman_filter_wave_alt;
  
  // [KALMAN_NUM_STATES * 1]
  matrix_t *x = kalman_get_state_vector(kf);
  x->data[0] = state->displacement_integral; // displacement integral
  x->data[1] = state->heave;                 // vertical displacement
  x->data[2] = state->vert_speed;            // vertical velocity
  x->data[3] = state->vert_accel;            // vertical accel
  x->data[4] = state->accel_bias;            // accel bias  
}

void kalman_wave_alt_init_defaults(float q0, float q1, float q2, float q3, float q4) {

  kalman_t *kf = kalman_filter_wave_alt_init();
  kalman_measurement_t *kfm = kalman_filter_wave_alt_measurement_vert_accel_init();

  // [KALMAN_NUM_STATES * 1]
  matrix_t *x = kalman_get_state_vector(kf);
  x->data[1] = 0.0; // displacement integral
  x->data[2] = 0.0; // vertical displacement
  x->data[3] = 0.0; // vertical velocity
  x->data[4] = 0.0; // vertical accel
  x->data[5] = 0.0; // accel bias

  // observation matrix [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_STATES]
  matrix_t *H = kalman_get_measurement_transformation(kfm);
  matrix_set(H, 0, 0, (matrix_data_t)1.0);
  matrix_set(H, 0, 1, (matrix_data_t)0.0);
  matrix_set(H, 0, 2, (matrix_data_t)0.0);
  matrix_set(H, 0, 3, (matrix_data_t)0.0);
  matrix_set(H, 0, 4, (matrix_data_t)0.0);
  matrix_set(H, 1, 0, (matrix_data_t)0.0);
  matrix_set(H, 1, 1, (matrix_data_t)0.0);
  matrix_set(H, 1, 2, (matrix_data_t)0.0);
  matrix_set(H, 1, 3, (matrix_data_t)1.0);
  matrix_set(H, 1, 4, (matrix_data_t)0.0);

  // observation covariance [KALMAN_NUM_MEASUREMENTS * KALMAN_NUM_MEASUREMENTS]
  matrix_t *R = kalman_get_observation_noise(kfm);
  matrix_set_symmetric(R, 0, 0, (matrix_data_t)0.01);
  matrix_set_symmetric(R, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(R, 1, 1, (matrix_data_t)1.0);
  
  // initial state covariance [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *P = kalman_get_system_covariance(kf);
  matrix_set_symmetric(P, 0, 0, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 0, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 0, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 0, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 1, 1, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 1, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 1, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 1, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 2, 2, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 2, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 2, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 3, 3, (matrix_data_t)1.0);
  matrix_set_symmetric(P, 3, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(P, 4, 4, (matrix_data_t)1.0);

  // transition covariance [KALMAN_NUM_STATES * KALMAN_NUM_STATES]
  matrix_t *Q = kalman_get_process_noise(kf);
  matrix_set_symmetric(Q, 0, 0, (matrix_data_t)q0);
  matrix_set_symmetric(Q, 0, 1, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 0, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 1, (matrix_data_t)q1);
  matrix_set_symmetric(Q, 1, 2, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 1, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 2, 2, (matrix_data_t)q2);
  matrix_set_symmetric(Q, 2, 3, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 2, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 3, 3, (matrix_data_t)q3);
  matrix_set_symmetric(Q, 3, 4, (matrix_data_t)0.0);
  matrix_set_symmetric(Q, 4, 4, (matrix_data_t)q4);
}

void kalman_wave_alt_step(KalmanWaveAltState* state, float accel, float k_hat, float delta_t) {
  kalman_t *kf = &kalman_filter_wave_alt;
  kalman_measurement_t *kfm = &kalman_filter_wave_alt_measurement_vert_accel;

  matrix_t *x = kalman_get_state_vector(kf);
  matrix_t *z = kalman_get_measurement_vector(kfm);

  matrix_t *F = kalman_wave_alt_get_state_transition(kf, k_hat, delta_t);
  
  // prediction.
  kalman_predict(kf);

  // measure ... [KALMAN_NUM_MEASUREMENTS, 1]
  matrix_data_t measurement = accel;
  matrix_set(z, 0, 0, 0.0);
  matrix_set(z, 1, 0, measurement);

  // update
  kalman_correct(kf, kfm);

  state->displacement_integral = x->data[0];
  state->heave = x->data[1];
  state->vert_speed = x->data[2];
  state->vert_accel = x->data[3];
  state->accel_bias = x->data[4];
}

#endif
