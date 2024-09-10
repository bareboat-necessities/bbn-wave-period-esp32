#ifndef KalmanSmoother_h
#define KalmanSmoother_h

/*
   A one dimensional Kalman filter implementation - a single variable smoother filter

   The variables are:

   x for the filtered value,
   q for the process noise,
   r for the measurement (sensor) uncertainty,
   p for the estimation uncertainty,
   k for the Kalman gain.

   The state of the filter is defined by the values of these variables.

   The initial values for p is not very important since it is adjusted
   during the process. It must be just high enough to narrow down.

   q - usually a small number between 0.001 and 1 - how fast your measurement moves.
   Recommended 0.01. Should be tunned to your needs.

   But tweaking the values for the process noise and sensor noise
   is essential to get clear readouts.

   For large noise reduction, you can try to start from:
   (see http://interactive-matter.eu/blog/2009/12/18/filtering-sensor-data-with-a-kalman-filter/ )

   q = 0.002
   r = 30.0
   p = 2000.0 // "large enough to narrow down"

   Example:

   KalmanSmootherVars kalman;
   kalman_smoother_init(&kalman, 0.002, 30.0, 2000.0);
   kalman_smoother_set_initial(&kalman, 1.0);
   while(1) {
     double measure = getSensorValue();
     double measure_adjusted = kalman_smoother_update(&kalman, measure);
   }

   Used also with results of Aranovskiy filter. So use double instead of float to 
   avoid decimal overflows with higher Aranovskiy gain values.
   
*/

typedef struct kalman_smoother_vars {
  /* Kalman filter variables */
  double q; // process noise covariance
  double r; // measurement uncertainty
  double p; // estimation uncertainty
  double k; // kalman gain
  double x; // value
} KalmanSmootherVars;

void kalman_smoother_init(KalmanSmootherVars* s, double process_noise, double sensor_noise, double estimated_error);
void kalman_smoother_set_initial(KalmanSmootherVars* s, double intial_value);
double kalman_smoother_update(KalmanSmootherVars* s, double measurement);

void kalman_smoother_init(KalmanSmootherVars* s, double process_noise, double sensor_noise, double estimated_error) {
  s->q = process_noise;
  s->r = sensor_noise;
  s->p = estimated_error;
}

void kalman_smoother_set_initial(KalmanSmootherVars* s, double initial_value) {
  s->x = initial_value;
}

double kalman_smoother_update(KalmanSmootherVars* s, double measurement) {
  // measurement update
  s->k = s->p / (s->p + s->r);
  double current_estimate = s->x + s->k * (measurement - s->x);
  s->p = (1.0 - s->k) * s->p + fabs(s->x - current_estimate) * s->q;
  s->x = current_estimate;
  return s->x;
}

#endif
