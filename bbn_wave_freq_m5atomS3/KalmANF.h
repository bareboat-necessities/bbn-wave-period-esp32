#ifndef KalmANF_h
#define KalmANF_h

/*
   See: https://github.com/randyaliased/KalmANF/

   and

   See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
     Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023

*/
#include <math.h>

typedef struct {
  // Persistent state variables
  float s_prev1;     // s[n-1]
  float s_prev2;     // s[n-2]
  float a_prev;      // a[n-1]
  float p_cov;       // Error covariance

  // Constants (configured at init)
  float rho;         // Pole radius (0 < rho < 1)
  float rho_sq;      // rho^2 (precomputed)
  float q;           // Process noise covariance
  float r;           // Measurement noise covariance
} KalmANF;

// Initialize the filter
void kalmANF_init(KalmANF *f, float rho, float q, float r, float p_cov, float s_prev1, float s_prev2, float a_prev) {
  f->s_prev1 = s_prev1;
  f->s_prev2 = s_prev2;
  f->a_prev = a_prev;
  f->p_cov = p_cov;
  f->rho = rho;
  f->rho_sq = rho * rho;
  f->q = q;
  f->r = r;
}

// Process a single sample, return estimated frequency
float kalmANF_process(KalmANF *f, float y, float delta_t, float *e_out) {
  // 1. Compute intermediate variable s[n]
  float s = y + f->rho * f->s_prev1 * f->a_prev - f->rho_sq * f->s_prev2;

  // 2. Prediction update
  f->p_cov += f->q;

  // 3. Compute Kalman gain
  float K = f->s_prev1 / (f->s_prev1 * f->s_prev1 + f->r / f->p_cov);

  // 4. Compute output e[n]
  float e = s - f->s_prev1 * f->a_prev + f->s_prev2;

  // 5. Update coefficient a[n]
  float a = f->a_prev + K * e;

  // 6. Handle coefficient bounds
  if (a > 2.0 || a < -2.0) {
    //a = 0.0;  // Reset if out of bounds for acos()
    a = a > 2.0 ? 1.99999 : -1.99999;
  }

  // 7. Update error covariance
  f->p_cov = (1.0 - K * f->s_prev1) * f->p_cov;

  // 8. Compute frequency estimate
  float omega_hat = acosf(a / 2.0);
  float f_est = (omega_hat / delta_t) / (2.0 * M_PI);

  // 9. Update state for next iteration
  f->s_prev2 = f->s_prev1;
  f->s_prev1 = s;
  f->a_prev = a;

  // Optional: Return filter output
  if (e_out) *e_out = e;

  return f_est;
}

#endif

