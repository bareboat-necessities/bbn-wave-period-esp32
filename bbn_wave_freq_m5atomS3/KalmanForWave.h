#ifndef KalmanForWave_h
#define KalmanForWave_h

#include <assert.h>

// create the filter structure
#define KALMAN_NAME wave
#define KALMAN_NUM_STATES 3
#define KALMAN_NUM_INPUTS 0
#include "FalmanFactoryFilter.h"

// create the measurement structure
#define KALMAN_MEASUREMENT_NAME displacement_integral
#define KALMAN_NUM_MEASUREMENTS 1
#include "KalmanFactoryMeasurement.h"

// clean up
#include "KalmanFactoryCleanup.h"

matrix_t *kalman_wave_get_state_transition(kalman_t *kf, matrix_data_t delta_t) {
  matrix_t *F = kalman_get_state_transition(kf);

  matrix_set(F, 0, 0, (matrix_data_t)1.0);                     // 1
  matrix_set(F, 0, 1, (matrix_data_t)delta_t);                 // T
  matrix_set(F, 0, 2, (matrix_data_t)0.5 * delta_t * delta_t); // 0.5 * T^2

  matrix_set(F, 1, 0, (matrix_data_t)0.0);         // 0
  matrix_set(F, 1, 1, (matrix_data_t)1.0);         // 1
  matrix_set(F, 1, 2, (matrix_data_t)delta_t);     // T

  matrix_set(F, 2, 0, (matrix_data_t)0.0);         // 0
  matrix_set(F, 2, 1, (matrix_data_t)0.0);         // 0
  matrix_set(F, 2, 2, (matrix_data_t)1.0);         // 1
  return F;
}

void kalman_wave_init() {

  kalman_t *kf = kalman_filter_wave_init();
  kalman_measurement_t *kfm = kalman_filter_wave_measurement_position_init();

  matrix_t *x = kalman_get_state_vector(kf);
  x->data[0] = 0.0; // displacement integral
  x->data[1] = 0.0; // vertical displacement
  x->data[2] = 0.0; // vertical velocity

  // transition matrix
  matrix_t *F = kalman_wave_get_state_transition(kf, delta_t);

  // observation matrix
  matrix_t *H = kalman_get_measurement_transformation(kfm);
  matrix_set(H, 0, 0, (matrix_data_t)1.0);
  matrix_set(H, 0, 1, (matrix_data_t)0.0);
  matrix_set(H, 0, 2, (matrix_data_t)0.0);

  // observation covariance
  matrix_t *R = kalman_get_process_noise(kfm);
  matrix_set(R, 0, 0, (matrix_data_t)1.0);

  // transition offset
  matrix_t *B = kalman_get_input_transition(kf);
  matrix_set(B, 0, 0, (matrix_data_t)(1.0/6.0) * delta_t * delta_t * delta_t);
  matrix_set(B, 0, 1, (matrix_data_t)0.5 * delta_t * delta_t);
  matrix_set(B, 0, 2, (matrix_data_t)delta_t);
  //kalman_wave_get_transition_offset(kf, delta_t);

  // initial state covariance
  matrix_t *P = kalman_get_system_covariance(kf);
  matrix_set_symmetric(P, 0, 0, (matrix_data_t)1.0);  
  matrix_set_symmetric(P, 0, 1, (matrix_data_t)0.0);  
  matrix_set_symmetric(P, 0, 2, (matrix_data_t)0.0);  
  matrix_set_symmetric(P, 1, 1, (matrix_data_t)1.0);  
  matrix_set_symmetric(P, 1, 2, (matrix_data_t)0.0); 
  matrix_set_symmetric(P, 2, 2, (matrix_data_t)1.0);

  // transition covariance
  matrix_t *Q = kalman_get_process_noise(kf);
  matrix_set_symmetric(Q, 0, 0, (matrix_data_t)1.0);  
  matrix_set_symmetric(Q, 0, 1, (matrix_data_t)0.0);  
  matrix_set_symmetric(Q, 0, 2, (matrix_data_t)0.0);  
  matrix_set_symmetric(Q, 1, 1, (matrix_data_t)0.2);  
  matrix_set_symmetric(Q, 1, 2, (matrix_data_t)0.0); 
  matrix_set_symmetric(Q, 2, 2, (matrix_data_t)0.1);
     
}

{


}

#endif
