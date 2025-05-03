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
  double s_prev1;     // s[n-1]
  double s_prev2;     // s[n-2]
  double a_prev;      // a[n-1]
  double p_cov;       // Error covariance

  // Constants (configured at init)
  double rho;         // Pole radius (0 < rho < 1)
  double rho_sq;      // rho^2 (precomputed)
  double q;           // Process noise covariance
  double r;           // Measurement noise covariance
} KalmANF;

// Initialize the filter
void kalmANF_init(KalmANF *f, double rho, double q, double r, double p_cov, double s_prev1, double s_prev2, double a_prev) {
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
double kalmANF_process(KalmANF *f, double y, double delta_t, double *e_out) {
  // 1. Compute intermediate variable s[n]
  double s = y + f->rho * f->s_prev1 * f->a_prev - f->rho_sq * f->s_prev2;

  // 2. Prediction update
  f->p_cov += f->q;

  // 3. Compute Kalman gain
  double K = f->s_prev1 / (f->s_prev1 * f->s_prev1 + f->r / f->p_cov);

  // 4. Compute output e[n]
  double e = s - f->s_prev1 * f->a_prev + f->s_prev2;

  // 5. Update coefficient a[n]
  double a = f->a_prev + K * e;

  // 6. Handle coefficient bounds
  if (a > 2.0 || a < -2.0) {
    //a = 0.0;  // Reset if out of bounds for acos()
    a = a > 2.0 ? 1.99999 : -1.99999;
  }

  // 7. Update error covariance
  f->p_cov = (1.0 - K * f->s_prev1) * f->p_cov;

  // 8. Compute frequency estimate
  double omega_hat = acos(a / 2.0);
  double f_est = (omega_hat / delta_t) / (2.0 * M_PI);

  // 9. Update state for next iteration
  f->s_prev2 = f->s_prev1;
  f->s_prev1 = s;
  f->a_prev = a;

  // Optional: Return filter output
  if (e_out) *e_out = e;

  return f_est;
}

#endif

