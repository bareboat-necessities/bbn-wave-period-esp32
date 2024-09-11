#ifndef AranovskiyFilter_h
#define AranovskiyFilter_h

/*
  Copyright 2024, Mikhail Grushinskiy

  Aranovskiy frequency estimator which is a simple on-line filter.

  Ref:

  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy

  The New Algorithm of Sinusoidal Signal Frequency Estimation.

  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing July 3-5, 2013. Caen, France

  Usage example:
  
  double omega_up = 1.0 * (2 * PI);  // upper frequency Hz * 2 * PI
  double k_gain = 2.0;
  double t_0 = 0.0;
  double x1_0 = 0.0;
  double theta_0 = - (omega_up * omega_up / 4.0);
  double sigma_0 = theta_0;
  double delta_t;  // time step sec

  AranovskiyParams params;
  AranovskiyState  state;

  aranovskiy_default_params(&params, omega_up, k_gain);
  aranovskiy_init_state(&state, x1_0, theta_0, sigma_0);

  unsigned long now = 0UL, last_update = 0UL;

  double t = t_0;
  last_update = millis();
  while(1) {
    delay(4);
    
    // measure
    double y = getAccelZfrom(imu) - 1.0;  // remove G acceration

    now = millis();
    delta_t = ((now - last_update) / 1000.0);
    last_update = now;

    aranovskiy_update(&params, &state, y, delta_t);

    // state.f contains estimated frequency

    t = t + delta_t;
  }
  
  Use double instead of float to avoid decimal overflows with higher Aranovskiy gain values.

  When Aranovskiy filter is used to estimate frequency of a signal averaged by Kalman filter
  and you look for faster convergence, then Kalman filter will produce a steep function with high gain
  and Aranovskiy filter with high gain will estimate frequency as really high which can cause
  decimal overflows.
  
*/

#define PI 3.1415926535897932384626433832795

typedef struct aranovskiy_params {
  double a = 1.0;
  double b = a;
  double k = 1.0;           // gain
} AranovskiyParams;

typedef struct aranovskiy_state {
  double y;                 // signal measurement
  double x1 = 0.0;
  double theta = -0.25;
  double sigma = -0.25;
  double x1_dot;
  double sigma_dot;
  double omega;             // frequency * 2 * pi
  double f;                 // frequency
} AranovskiyState;

void aranovskiy_default_params(AranovskiyParams* p, double omega_up, double k_gain);
void aranovskiy_init_state(AranovskiyState* s, double x1_0, double theta_0, double sigma_0);
void aranovskiy_update(AranovskiyParams* p, AranovskiyState* s, double y, double delta_t);

void aranovskiy_default_params(AranovskiyParams* p, double omega_up, double k_gain) {
  p->a = omega_up;
  p->b = p->a;
  p->k = k_gain;
}

void aranovskiy_init_state(AranovskiyState* s, double x1_0, double theta_0, double sigma_0) {
  s->x1 = x1_0;
  s->theta = theta_0;
  s->sigma = sigma_0;
}

void aranovskiy_update(AranovskiyParams* p, AranovskiyState* s, double y, double delta_t) {
  s->x1_dot = - p->a * s->x1 + p->b * y;
  s->sigma_dot = - p->k * s->x1 * s->x1 * s->theta - p->k * p->a * s->x1 * s->x1_dot - p->k * p->b * s->x1_dot * y;
  s->theta = s->sigma + p->k * p->b * s->x1 * y;
  s->omega = sqrt(fabs(s->theta));
  s->f = s->omega / (2.0 * PI);
  // step
  s->x1 = s->x1 + s->x1_dot * delta_t;
  s->sigma = s->sigma + s->sigma_dot * delta_t;
}

#endif
