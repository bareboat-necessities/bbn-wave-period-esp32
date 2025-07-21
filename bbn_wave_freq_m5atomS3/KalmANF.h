#ifndef KALMANF_H
#define KALMANF_H

#include <math.h>
#include <float.h>

/*
   See: https://github.com/randyaliased/KalmANF/

   and

   See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
   Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023
*/

// -------------------- Internal Resonator --------------------

typedef struct {
  float s_prev1;   // s[n-1] — previous resonator output sample
  float s_prev2;   // s[n-2] — two samples ago
  float a;         // a[n] — adaptive filter coefficient = 2*cos(ω)
  float rho;       // Pole radius (0 < rho < 1)
  float rho_sq;    // Precomputed rho^2
} ANFResonator;

// Initialize resonator with filter settings and initial state
static inline void anf_resonator_init(ANFResonator* r, float rho, float a, float s1, float s2) {
  r->rho = rho;
  r->rho_sq = rho * rho;
  r->a = a;
  r->s_prev1 = s1;
  r->s_prev2 = s2;
}

// Compute resonator output s[n] based on input y[n]
static inline float anf_resonator_compute_s(const ANFResonator* r, float y) {
  return y + r->rho * r->s_prev1 * r->a - r->rho_sq * r->s_prev2;
}

// Update internal state for next iteration
static inline void anf_resonator_update_state(ANFResonator* r, float s) {
  r->s_prev2 = r->s_prev1;
  r->s_prev1 = s;
}

// Get current phase estimate in radians
static inline float anf_resonator_get_phase(const ANFResonator* r) {
  return atan2f(r->s_prev1, r->s_prev2);
}

// -------------------- Kalman Filter with Embedded Resonator --------------------

typedef struct {
  ANFResonator res;  // Embedded second-order IIR resonator
  float p_cov;       // Kalman error covariance
  float q;           // Process noise covariance
  float r;           // Measurement noise covariance
} KalmANF;

// Initialize the filter
void kalmANF_init(KalmANF* f, float rho, float q, float r, float p_cov,
                  float s_prev1, float s_prev2, float a_prev) {
  f->p_cov = p_cov;
  f->q = q;
  f->r = r;
  anf_resonator_init(&f->res, rho, a_prev, s_prev1, s_prev2);
}

// Process a single sample, return estimated frequency
float kalmANF_process(KalmANF* f, float y, float delta_t, float* e_out) {
  ANFResonator* r = &f->res;

  // 1. Compute intermediate variable s[n]
  float s = anf_resonator_compute_s(r, y);

  // 2. Prediction update
  f->p_cov += f->q;

  // 3. Compute Kalman gain
  float denom = r->s_prev1 * r->s_prev1 + f->r / (f->p_cov + FLT_EPSILON);
  float K = r->s_prev1 / denom;

  // 4. Compute output e[n]
  float e = s - r->s_prev1 * r->a + r->s_prev2;

  // 5. Update coefficient a[n]
  float a = r->a + K * e;

  // 6. Handle coefficient bounds
  if (a > 2.0f || a < -2.0f) {
    // Prevent domain error in acos()
    a = a > 2.0f ? 1.99999f : -1.99999f;
  }

  // 7. Update error covariance
  f->p_cov = (1.0f - K * r->s_prev1) * f->p_cov;

  // 8. Compute frequency estimate
  float omega_hat = acosf(a / 2.0f); // rad/sample
  float f_est = (omega_hat / delta_t) / (2.0f * M_PI); // Hz

  // 9. Update state for next iteration
  r->a = a;
  anf_resonator_update_state(r, s);

  // Optional: Return filter output
  if (e_out) *e_out = e;

  return f_est;
}

// Get the current phase estimate (in radians)
float kalmANF_get_phase(const KalmANF* f) {
  return anf_resonator_get_phase(&f->res);
}

#endif // KALMANF_H
