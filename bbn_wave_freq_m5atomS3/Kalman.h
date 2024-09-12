#ifndef Kalman_h
#define Kalman_h

/*
   Adopted from https://github.com/cepekLP/kalman-clib
*/

#include <stdint.h>

#include "Cholesky.h"

/*!
   \brief Kalman Filter structure
   \see kalman_measurement_t
*/
typedef struct {
  /*!
     \brief State vector
  */
  matrix_t x;

  /*!
     \brief State-transition matrix
     \see P
  */
  matrix_t F;

  /*!
     \brief System covariance matrix
     \see F
  */
  matrix_t P;

  /*!
     \brief Input vector
  */
  matrix_t u;

  /*!
     \brief Input matrix
     \see Q
  */
  matrix_t B;

  /*!
     \brief Process noise covariance matrix
  */
  matrix_t Q;

  /*!
     \brief Temporary variables.
  */
  struct {
    /*!
       \brief Auxiliary array for matrix multiplication, needs to be MAX(num
       states, num inputs)

       This auxiliary field can also be used as a backing field for the
       predicted x vector, however it MUST NOT be aliased with either
       temporary P or temporary BQ.
    */
    matrix_data_t *aux;

    /*!
       \brief x-sized temporary vector
       \see x
    */
    matrix_t predicted_x;

    /*!
       \brief P-Sized temporary matrix  (number of states x number of
       states)

       The backing field for this temporary MAY be aliased with temporary
       BQ.

       \see P
    */
    matrix_t P;

    /*!
       \brief BxQ-sized temporary matrix (number of states x number of
       inputs)

       The backing field for this temporary MAY be aliased with temporary P.

       \see B
       \see Q
    */
    matrix_t BQ;

  } temporary;

} kalman_t;

/*!
   \brief Kalman Filter measurement structure
   \see kalman_t
*/
typedef struct {
  /*!
     \brief Measurement vector
  */
  matrix_t z;

  /*!
     \brief Measurement transformation matrix
     \see R
  */
  matrix_t H;

  /*!
     \brief Meausurement noise matrix
     \see H
  */
  matrix_t R;

  /*!
     \brief Innovation vector
  */
  matrix_t y;

  /*!
     \brief Residual covariance matrix
     \see F
  */
  matrix_t S;

  /*!
     \brief Kalman gain matrix
  */
  matrix_t K;

  /*!
     \brief Temporary variables.
  */
  struct {
    /*!
       \brief Auxiliary array for matrix multiplication, needs to be MAX(num
       states, num measurements)

       This auxiliary field MUST NOT be aliased with either temporary HP,
       KHP, HPt or S_inverted.
    */
    matrix_data_t *aux;

    /*!
       \brief S-Sized temporary matrix  (number of measurements x number of
       measurements)

       The backing field for this temporary MAY be aliased with temporary
       temp_KHP. The backing field for this temporary MAY be aliased with
       temporary temp_HP (if it is not aliased with temp_PHt). The backing
       field for this temporary MUST NOT be aliased with temporary temp_PHt.
       The backing field for this temporary MUST NOT be aliased with aux.

       \see S
    */
    matrix_t S_inv;

    /*!
       \brief H-Sized temporary matrix  (number of measurements x number of
       states)

       The backing field for this temporary MAY be aliased with temporary
       S_inv. The backing field for this temporary MAY be aliased with
       temporary temp_PHt. The backing field for this temporary MUST NOT be
       aliased with temporary temp_KHP.
    */
    matrix_t HP;

    /*!
       \brief P-Sized temporary matrix  (number of states x number of
       states)

       The backing field for this temporary MAY be aliased with temporary
       S_inv. The backing field for this temporary MAY be aliased with
       temporary temp_PHt. The backing field for this temporary MUST NOT be
       aliased with temporary temp_HP.
    */
    matrix_t KHP;

    /*!
       \brief PxH'-Sized (H'-Sized) temporary matrix  (number of states x
       number of measurements)

       The backing field for this temporary MAY be aliased with temporary
       temp_HP. The backing field for this temporary MAY be aliased with
       temporary temp_KHP. The backing field for this temporary MUST NOT be
       aliased with temporary S_inv.
    */
    matrix_t PHt;

  } temporary;

} kalman_measurement_t;

/*!
   \brief Initializes the Kalman Filter
   \param[in] kf The Kalman Filter structure to initialize
   \param[in] num_states The number of state variables
   \param[in] num_inputs The number of input variables
   \param[in] F The state transition matrix ({\ref num_states} x {\ref
   num_states}) \param[in] x The state vector ({\ref num_states} x \c 1)
   \param[in] B The input transition matrix ({\ref num_states} x {\ref
   num_inputs}) \param[in] u The input vector ({\ref num_inputs} x \c 1)
   \param[in] P The state covariance matrix ({\ref num_states} x {\ref
   num_states}) \param[in] Q The input covariance matrix ({\ref num_inputs} x
   {\ref num_inputs}) \param[in] aux The auxiliary buffer (length {\ref
   num_states} or {\ref num_inputs}, whichever is greater) \param[in] predictedX
   The temporary vector for predicted X ({\ref num_states} x \c 1) \param[in]
   temp_P The temporary matrix for P calculation ({\ref num_states} x {\ref
   num_states}) \param[in] temp_BQ The temporary matrix for BQ calculation
   ({\ref num_states} x {\ref num_inputs})
*/
void kalman_filter_initialize(kalman_t *kf, uint_fast8_t num_states,
                              uint_fast8_t num_inputs, matrix_data_t *F,
                              matrix_data_t *x, matrix_data_t *B,
                              matrix_data_t *u, matrix_data_t *P,
                              matrix_data_t *Q, matrix_data_t *aux,
                              matrix_data_t *predictedX, matrix_data_t *temp_P,
                              matrix_data_t *temp_BQ);

/*!
   \brief Sets the measurement vector
   \param[in] kfm The Kalman Filter measurement structure to initialize
   \param[in] num_states The number of states
   \param[in] num_measurements The number of measurements
   \param[in] H The measurement transformation matrix ({\ref num_measurements} x
   {\ref num_states}) \param[in] z The measurement vector ({\ref
   num_measurements} x \c 1) \param[in] R The process noise / measurement
   uncertainty ({\ref num_measurements} x {\ref num_measurements}) \param[in] y
   The innovation ({\ref num_measurements} x \c 1) \param[in] S The residual
   covariance ({\ref num_measurements} x {\ref num_measurements}) \param[in] K
   The Kalman gain ({\ref num_states} x {\ref num_measurements}) \param[in] aux
   The auxiliary buffer (length {\ref num_states} or {\ref num_measurements},
   whichever is greater) \param[in] S_inv The temporary matrix for the inverted
   residual covariance  ({\ref num_measurements} x {\ref num_measurements})
   \param[in] temp_HP The temporary matrix for HxP ({\ref num_measurements} x
   {\ref num_states}) \param[in] temp_PHt The temporary matrix for PxH' ({\ref
   num_states} x {\ref num_measurements}) \param[in] temp_KHP The temporary
   matrix for KxHxP ({\ref num_states} x {\ref num_states})
*/
void kalman_measurement_initialize(
  kalman_measurement_t *kfm, uint_fast8_t num_states,
  uint_fast8_t num_measurements, matrix_data_t *H, matrix_data_t *z,
  matrix_data_t *R, matrix_data_t *y, matrix_data_t *S, matrix_data_t *K,
  matrix_data_t *aux, matrix_data_t *S_inv, matrix_data_t *temp_HP,
  matrix_data_t *temp_PHt, matrix_data_t *temp_KHP);

/*!
   \brief Initializes the process noise matrix Q
   \param[in] kf The Kalman Filter structure to initialize
   \param[in] size The size of the process noise matrix
   \param[in] dt The time step
   \param[in] variance The variance of the process noise
*/
void kalman_init_process_noise(const matrix_t *Q, float dt, float variance);

/*!
   \brief Performs the time update / prediction step of only the state vector
   \param[in] kf The Kalman Filter structure to predict with.

   \see kalman_predict
   \see kalman_predict_tuned
*/
void kalman_predict_x(register kalman_t *const kf);

/*!
   \brief Performs the time update / prediction step of only the state
   covariance matrix \param[in] kf The Kalman Filter structure to predict with.

   \see kalman_predict
   \see kalman_predict_P_tuned
*/
void kalman_predict_P(register kalman_t *const kf);

/*!
   \brief Performs the time update / prediction step of only the state
   covariance matrix \param[in] kf The Kalman Filter structure to predict with.

   \see kalman_predict_tuned
*/
void kalman_predict_P_tuned(register kalman_t *const kf,
                            matrix_data_t lambda);

/*!
   \brief Performs the time update / prediction step.
   \param[in] kf The Kalman Filter structure to predict with.
   \param[in] lambda Lambda factor (\c 0 < {\ref lambda} <= \c 1) to forcibly
   reduce prediction certainty. Smaller values mean larger uncertainty.

   This call assumes that the input covariance and variables are already set in
   the filter structure.

   \see kalman_predict_x
*/
inline void kalman_predict(kalman_t *kf) {
  /************************************************************************/
  /* Predict next state using system dynamics                             */
  /* x = F*x + B*u + w                                                    */
  /************************************************************************/

  kalman_predict_x(kf);

  /************************************************************************/
  /* Predict next covariance using system dynamics and input              */
  /* P = F*P*F' + Q                                                       */
  /************************************************************************/

  kalman_predict_P(kf);
}

/*!
   \brief Performs the time update / prediction step.
   \param[in] kf The Kalman Filter structure to predict with.
   \param[in] lambda Lambda factor (\c 0 < {\ref lambda} <= \c 1) to forcibly
   reduce prediction certainty. Smaller values mean larger uncertainty.

   This call assumes that the input covariance and variables are already set in
   the filter structure.

   \see kalman_predict_x
   \see kalman_predict_P_tuned
*/
void kalman_predict_tuned(kalman_t *kf, matrix_data_t lambda) {
  /************************************************************************/
  /* Predict next state using system dynamics                             */
  /* x = F*x + B*u + w                                                    */
  /************************************************************************/

  kalman_predict_x(kf);

  /************************************************************************/
  /* Predict next covariance using system dynamics and input              */
  /* P = F*P*F' * 1/lambda^2 + B*Q*B'                                     */
  /************************************************************************/

  kalman_predict_P_tuned(kf, lambda);
}

/*!
   \brief Performs the measurement update step.
   \param[in] kf The Kalman Filter structure to correct.
*/
void kalman_correct(kalman_t *kf, kalman_measurement_t *kfm);

/*!
   \brief Gets a pointer to the state vector x.
   \param[in] kf The Kalman Filter structure
   \return The state vector x.
*/
inline matrix_t *kalman_get_state_vector(kalman_t *kf) {
  return &(kf->x);
}

/*!
   \brief Gets a pointer to the state transition matrix F.
   \param[in] kf The Kalman Filter structure
   \return The state transition matrix F.
*/
inline matrix_t *kalman_get_state_transition(kalman_t *kf) {
  return &(kf->F);
}

/*!
   \brief Gets a pointer to the system covariance matrix P.
   \param[in] kf The Kalman Filter structure
   \return The system covariance matrix.
*/
inline matrix_t *kalman_get_system_covariance(kalman_t *kf) {
  return &(kf->P);
}

/*!
   \brief Gets a pointer to the input vector u.
   \param[in] kf The Kalman Filter structure
   \return The input vector u.
*/
inline matrix_t *kalman_get_input_vector(kalman_t *kf) {
  return &(kf->u);
}

/*!
   \brief Gets a pointer to the input transition matrix B.
   \param[in] kf The Kalman Filter structure
   \return The input transition matrix B.
*/
inline matrix_t *kalman_get_input_transition(kalman_t *kf) {
  return &(kf->B);
}

/*!
   \brief Gets a pointer to the process noise matrix Q.
   \param[in] kf The Kalman Filter structure
   \return The input covariance matrix.
*/
inline matrix_t *kalman_get_process_noise(kalman_t *kf) {
  return &(kf->Q);
}

/*!
   \brief Gets a pointer to the measurement vector z.
   \param[in] kfm The Kalman Filter measurement structure.
   \return The measurement vector z.
*/
inline matrix_t *kalman_get_measurement_vector(kalman_measurement_t *kfm) {
  return &(kfm->z);
}

/*!
   \brief Gets a pointer to the measurement transformation matrix H.
   \param[in] kfm The Kalman Filter measurement structure.
   \return The measurement transformation matrix H.
*/
inline matrix_t *kalman_get_measurement_transformation(
  kalman_measurement_t *kfm) {
  return &(kfm->H);
}

/*!
   \brief Gets a pointer to the process noise matrix R.
   \param[in] kfm The Kalman Filter measurement structure.
   \return The process noise matrix R.
*/
inline matrix_t *kalman_get_observation_noise(kalman_measurement_t *kfm) {
  return &(kfm->R);
}

/*!
   \brief Initializes the Kalman Filter
   \param[in] kf The Kalman Filter structure to initialize
   \param[in] num_states The number of state variables
   \param[in] num_inputs The number of input variables
   \param[in] F The state transition matrix ({\ref num_states} x {\ref
   num_states}) \param[in] x The state vector ({\ref num_states} x \c 1)
   \param[in] B The input transition matrix ({\ref num_states} x {\ref
   num_inputs}) \param[in] u The input vector ({\ref num_inputs} x \c 1)
   \param[in] P The state covariance matrix ({\ref num_states} x {\ref
   num_states}) \param[in] Q The input covariance matrix ({\ref num_inputs} x
   {\ref num_inputs}) \param[in] aux The auxiliary buffer (length {\ref
   num_states} or {\ref num_inputs}, whichever is greater) \param[in] predictedX
   The temporary vector for predicted X ({\ref num_states} x \c 1) \param[in]
   temp_P The temporary matrix for P calculation ({\ref num_states} x {\ref
   num_states}) \param[in] temp_BQ The temporary matrix for BQ calculation
   ({\ref num_states} x {\ref num_inputs})
*/
void kalman_filter_initialize(kalman_t *kf, uint_fast8_t num_states,
                              uint_fast8_t num_inputs, matrix_data_t *F,
                              matrix_data_t *x, matrix_data_t *B,
                              matrix_data_t *u, matrix_data_t *P,
                              matrix_data_t *Q, matrix_data_t *aux,
                              matrix_data_t *predictedX, matrix_data_t *temp_P,
                              matrix_data_t *temp_BQ) {
  matrix_init(&kf->F, num_states, num_states, F);
  matrix_init(&kf->P, num_states, num_states, P);
  matrix_init(&kf->x, num_states, 1, x);

  matrix_init(&kf->B, num_states, num_inputs, B);
  matrix_init(&kf->Q, num_states, num_states, Q);
  matrix_init(&kf->u, num_inputs, 1, u);

  // set auxiliary vector
  kf->temporary.aux = aux;

  // set predicted x vector
  matrix_init(&kf->temporary.predicted_x, num_states, 1, predictedX);

  // set temporary P matrix
  matrix_init(&kf->temporary.P, num_states, num_states, temp_P);

  // set temporary BQ matrix
  matrix_init(&kf->temporary.BQ, num_states, num_inputs, temp_BQ);
}

/*!
   \brief Sets the measurement vector
   \param[in] kfm The Kalman Filter measurement structure to initialize
   \param[in] num_states The number of states
   \param[in] num_measurements The number of measurements
   \param[in] H The measurement transformation matrix ({\ref num_measurements} x
   {\ref num_states}) \param[in] z The measurement vector ({\ref
   num_measurements} x \c 1) \param[in] R The process noise / measurement
   uncertainty ({\ref num_measurements} x {\ref num_measurements}) \param[in] y
   The innovation ({\ref num_measurements} x \c 1) \param[in] S The residual
   covariance ({\ref num_measurements} x {\ref num_measurements}) \param[in] K
   The Kalman gain ({\ref num_states} x {\ref num_measurements}) \param[in] aux
   The auxiliary buffer (length {\ref num_states} or {\ref num_measurements},
   whichever is greater) \param[in] S_inv The temporary matrix for the inverted
   residual covariance  ({\ref num_measurements} x {\ref num_measurements})
   \param[in] temp_HP The temporary matrix for HxP ({\ref num_measurements} x
   {\ref num_states}) \param[in] temp_PHt The temporary matrix for PxH' ({\ref
   num_states} x {\ref num_measurements}) \param[in] temp_KHP The temporary
   matrix for KxHxP ({\ref num_states} x {\ref num_states})
*/
void kalman_measurement_initialize(
  kalman_measurement_t *kfm, uint_fast8_t num_states,
  uint_fast8_t num_measurements, matrix_data_t *H, matrix_data_t *z,
  matrix_data_t *R, matrix_data_t *y, matrix_data_t *S, matrix_data_t *K,
  matrix_data_t *aux, matrix_data_t *S_inv, matrix_data_t *temp_HP,
  matrix_data_t *temp_PHt, matrix_data_t *temp_KHP) {
  matrix_init(&kfm->H, num_measurements, num_states, H);
  matrix_init(&kfm->R, num_measurements, num_measurements, R);
  matrix_init(&kfm->z, num_measurements, 1, z);

  matrix_init(&kfm->K, num_states, num_measurements, K);
  matrix_init(&kfm->S, num_measurements, num_measurements, S);
  matrix_init(&kfm->y, num_measurements, 1, y);

  // set auxiliary vector
  kfm->temporary.aux = aux;

  // set inverted S matrix
  matrix_init(&kfm->temporary.S_inv, num_measurements, num_measurements, S_inv);

  // set temporary HxP matrix
  matrix_init(&kfm->temporary.HP, num_measurements, num_states, temp_HP);

  // set temporary PxH' matrix
  matrix_init(&kfm->temporary.PHt, num_states, num_measurements, temp_PHt);

  // set temporary KxHxP matrix
  matrix_init(&kfm->temporary.KHP, num_states, num_states, temp_KHP);
}

void kalman_init_process_noise(matrix_t *Q, float dt, float variance) {
  if (Q->cols == 2) {
    matrix_set_symmetric(Q, 0, 0, 0.25f * variance * pow(dt, 4));
    matrix_set_symmetric(Q, 0, 1, 0.5f * variance * pow(dt, 3));
    matrix_set_symmetric(Q, 1, 1, variance * pow(dt, 2));
  } else if (Q->cols == 3) {
    matrix_set_symmetric(Q, 0, 0, 0.25f * variance * pow(dt, 4));
    matrix_set_symmetric(Q, 0, 1, 0.5f * variance * pow(dt, 3));
    matrix_set_symmetric(Q, 0, 2, 0.5f * variance * pow(dt, 2));
    matrix_set_symmetric(Q, 1, 1, variance * pow(dt, 2));
    matrix_set_symmetric(Q, 1, 2, variance * dt);
    matrix_set_symmetric(Q, 2, 2, variance);
  } else if (Q->cols == 4) {
    matrix_set_symmetric(Q, 0, 0, variance * pow(dt, 6) / 36.0f);
    matrix_set_symmetric(Q, 0, 1, variance * pow(dt, 5) / 12.0f);
    matrix_set_symmetric(Q, 0, 2, variance * pow(dt, 4) / 6.0f);
    matrix_set_symmetric(Q, 0, 3, variance * pow(dt, 3) / 6.0f);
    matrix_set_symmetric(Q, 1, 1, variance * pow(dt, 4) / 4.0f);
    matrix_set_symmetric(Q, 1, 2, variance * pow(dt, 3) / 2.0f);
    matrix_set_symmetric(Q, 1, 3, variance * pow(dt, 2) / 2.0f);
    matrix_set_symmetric(Q, 2, 2, variance * pow(dt, 2));
    matrix_set_symmetric(Q, 2, 3, variance * dt);
    matrix_set_symmetric(Q, 3, 3, variance);
  } else {
    assert(0);
  }
}

/*!
   \brief Performs the time update / prediction step of only the state vector
   \param[in] kf The Kalman Filter structure to predict with.
*/
void kalman_predict_x(register kalman_t *const kf) {
  // matrices and vectors
  const matrix_t *const F = &kf->F;
  const matrix_t *const B = &kf->B;
  matrix_t *const x = &kf->x;
  matrix_t *const u = &kf->u;
  
  // temporaries
  matrix_t *const xpredicted = &kf->temporary.predicted_x;

  /************************************************************************/
  /* Predict next state using system dynamics                             */
  /* x = F*x + B*u + w                                                    */
  /************************************************************************/

  // x = F*x
  matrix_mult_rowvector(F, x, xpredicted);
  matrix_copy(xpredicted, x);

  // x += B*u
  if (kf->B.rows > 0 && kf->B.cols > 0) {
    matrix_mult_rowvector(B, u, xpredicted);    
    matrix_add_inplace(x, xpredicted);
  }
}

/*!
   \brief Performs the time update / prediction step of only the state
   covariance matrix \param[in] kf The Kalman Filter structure to predict with.
*/
void kalman_predict_P(register kalman_t *const kf) {
  // matrices and vectors
  const matrix_t *const F = &kf->F;
  const matrix_t *const B = &kf->B;
  matrix_t *const P = &kf->P;

  // temporaries
  matrix_data_t *const aux = kf->temporary.aux;
  matrix_t *const P_temp = &kf->temporary.P;
  matrix_t *const BQ_temp = &kf->temporary.BQ;

  /************************************************************************/
  /* Predict next covariance using system dynamics and input              */
  /* P = F*P*F' + Q                                                       */
  /************************************************************************/

  // P = F*P*F'
  matrix_mult(F, P, P_temp, aux);    // temp = F*P
  matrix_mult_transb(P_temp, F, P);  // P = temp*F'

  // P += Q
  matrix_add_inplace(P, &kf->Q);
}

/*!
   \brief Performs the time update / prediction step of only the state
   covariance matrix \param[in] kf The Kalman Filter structure to predict with.
*/
void kalman_predict_P_tuned(register kalman_t *const kf, matrix_data_t lambda) {
  // matrices and vectors
  const matrix_t *const F = &kf->F;
  matrix_t *const P = &kf->P;

  // temporaries
  matrix_data_t *const aux = kf->temporary.aux;
  matrix_t *const P_temp = &kf->temporary.P;

  /************************************************************************/
  /* Predict next covariance using system dynamics and input              */
  /* P = F*P*F' * 1/lambda^2 + Q                                          */
  /************************************************************************/

  // lambda = 1/lambda^2
  lambda = (matrix_data_t)1.0 /
           (lambda * lambda);  // TODO: This should be precalculated, e.g.
  // using kalman_set_lambda(...);

  // P = F*P*F'
  matrix_mult(F, P, P_temp, aux);                 // temp = F*P
  matrix_multscale_transb(P_temp, F, lambda, P);  // P = temp*F' * 1/(lambda^2)

  // P += Q
  matrix_add_inplace(P, &kf->Q);
}

/*!
   \brief Performs the measurement update step.
   \param[in] kf The Kalman Filter structure to correct.
*/
void kalman_correct(kalman_t *kf, kalman_measurement_t *kfm) {
  matrix_t *const P = &kf->P;
  const matrix_t *const H = &kfm->H;
  matrix_t *const K = &kfm->K;
  matrix_t *const S = &kfm->S;
  matrix_t *const y = &kfm->y;
  matrix_t *const x = &kf->x;

  // temporaries
  matrix_data_t *const aux = kfm->temporary.aux;
  matrix_t *const Sinv = &kfm->temporary.S_inv;
  matrix_t *const temp_HP = &kfm->temporary.HP;
  matrix_t *const temp_KHP = &kfm->temporary.KHP;
  matrix_t *const temp_PHt = &kfm->temporary.PHt;

  /************************************************************************/
  /* Calculate innovation and residual covariance                         */
  /* y = z - H*x                                                          */
  /* S = H*P*H' + R                                                       */
  /************************************************************************/

  // y = z - H*x
  matrix_mult_rowvector(H, x, y);
  matrix_sub_inplace_b(&kfm->z, y);

  // S = H*P*H' + R
  matrix_mult(H, P, temp_HP, aux);    // temp = H*P
  matrix_mult_transb(temp_HP, H, S);  // S = temp*H'
  matrix_add_inplace(S, &kfm->R);     // S += R

  /************************************************************************/
  /* Calculate Kalman gain                                                */
  /* K = P*H' * S^-1                                                      */
  /************************************************************************/

  // K = P*H' * S^-1
  cholesky_decompose_lower(S);
  matrix_invert_lower(S, Sinv);  // Sinv = S^-1
  // NOTE that to allow aliasing of Sinv and temp_PHt, a copy must be
  // performed here
  matrix_mult_transb(P, H, temp_PHt);   // temp = P*H'
  matrix_mult(temp_PHt, Sinv, K, aux);  // K = temp*Sinv

  /************************************************************************/
  /* Correct state prediction                                             */
  /* x = x + K*y                                                          */
  /************************************************************************/

  // x = x + K*y
  matrix_multadd_rowvector(K, y, x);

  /************************************************************************/
  /* Correct state covariances                                            */
  /* P = (I-K*H) * P                                                      */
  /*   = P - K*(H*P)                                                      */
  /************************************************************************/

  // P = P - K*(H*P)
  matrix_mult(H, P, temp_HP, aux);         // temp_HP = H*P
  matrix_mult(K, temp_HP, temp_KHP, aux);  // temp_KHP = K*temp_HP
  matrix_sub(P, temp_KHP, P);              // P -= temp_KHP
}

#endif
