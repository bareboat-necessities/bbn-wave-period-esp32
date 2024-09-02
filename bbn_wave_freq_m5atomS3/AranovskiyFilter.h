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
  
  float omega_up = 1.0 * (2 * PI);  // upper frequency Hz * 2 * PI
  float k_gain = 2.0;
  float t_0 = 0.0;
  float x1_0 = 0.0;
  float theta_0 = - (omega_up * omega_up / 4.0);
  float sigma_0 = theta_0;
  float delta_t;  // time step sec

  AranovskiyParams params;
  AranovskiyState  state;

  aranovskiy_default_params(&params, omega_up, k_gain);
  aranovskiy_init_state(&state, t_0, x1_0, theta_0, sigma_0);

  unsigned long now = 0UL, last_update = 0UL;

  last_update = millis();
  while(1) {
    delay(4);
    
    // measure
    float y = getAccelZfrom(imu) - 1.0;  // remove G acceration

    now = millis();
    delta_t = ((now - last_update) / 1000.0);
    last_update = now;

    aranovskiy_update(&params, &state, y, delta_t);

    // state.f contains estimated frequency
  }

*/

#define PI 3.1415926535897932384626433832795

typedef struct aranovskiy_params {
  float a = 1.0;
  float b = a;
  float k = 1.0;           // gain
} AranovskiyParams;

typedef struct aranovskiy_state {
  float t = 0.0;           // time
  float y;                 // signal measurement
  float x1 = 0.0;
  float theta = -0.25;
  float sigma = -0.25;
  float x1_dot;
  float sigma_dot;
  float omega;             // frequency * 2 * pi
  float f;                 // frequency
} AranovskiyState;

void aranovskiy_default_params(AranovskiyParams* p, float omega_up, float k_gain);
void aranovskiy_init_state(AranovskiyState* s, float t_0, float x1_0, float theta_0, float sigma_0);
void aranovskiy_update(AranovskiyParams* p, AranovskiyState* s, float y, float delta_t);

void aranovskiy_default_params(AranovskiyParams* p, float omega_up, float k_gain) {
  p->a = omega_up;
  p->b = p->a;
  p->k = k_gain;
}

void aranovskiy_init_state(AranovskiyState* s, float t_0, float x1_0, float theta_0, float sigma_0) {
  s->t = t_0;
  s->x1 = x1_0;
  s->theta = theta_0;
  s->sigma = sigma_0;
}

void aranovskiy_update(AranovskiyParams* p, AranovskiyState* s, float y, float delta_t) {
  s->x1_dot = - p->a * s->x1 + p->b * y;
  s->sigma_dot = - p->k * s->x1 * s->x1 * s->theta - p->k * p->a * s->x1 * s->x1_dot - p->k * p->b * s->x1_dot * y;
  s->theta = s->sigma + p->k * p->b * s->x1 * y;
  s->omega = sqrt(fabs(s->theta));
  s->f = s->omega / (2.0 * PI);
  // step
  s->x1 = s->x1 + s->x1_dot * delta_t;
  s->sigma = s->sigma + s->sigma_dot * delta_t;
  s->t = s->t + delta_t;
}

#endif
