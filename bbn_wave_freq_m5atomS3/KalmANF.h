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

class KalmANF {
private:
  // -------------------- Internal Notch Filter Resonator --------------------
  class ANFResonator {
  public:
    float s_prev1 = 0.0f;  // s[n-1] — previous resonator output sample
    float s_prev2 = 0.0f;  // s[n-2] — two samples ago
    float a = 0.0f;        // a[n] — adaptive filter coefficient = 2*cos(ω)
    float rho = 0.0f;      // Pole radius (0 < rho < 1)
    float rho_sq = 0.0f;   // Precomputed rho^2

    void init(float rho_init, float a_init, float s1, float s2) {
      rho = rho_init;
      rho_sq = rho * rho;
      a = a_init;
      s_prev1 = s1;
      s_prev2 = s2;
    }

    float compute_s(float y) const {
      return y + rho * s_prev1 * a - rho_sq * s_prev2;
    }

    void update_state(float s) {
      s_prev2 = s_prev1;
      s_prev1 = s;
    }

    float get_phase() const {
      return std::atan2(s_prev1, s_prev2);
    }
  };

  ANFResonator res;  // Embedded second-order IIR resonator

  // Kalman parameters
  float p_cov = 1.0f;  // Kalman error covariance
  float q = 0.0f;      // Process noise covariance
  float r = 0.0f;      // Measurement noise covariance

public:
  // Initialize the filter
  void init(float rho, float q_, float r_, float p_cov_,
            float s_prev1, float s_prev2, float a_prev) {
    q = q_;
    r = r_;
    p_cov = p_cov_;
    res.init(rho, a_prev, s_prev1, s_prev2);
  }

  // Process a single sample, return estimated frequency in Hz
  float process(float y, float delta_t, float* e_out = nullptr) {
    // 1. Compute intermediate variable s[n]
    float s = res.compute_s(y);

    // 2. Prediction update
    p_cov += q;

    // 3. Compute Kalman gain
    float denom = res.s_prev1 * res.s_prev1 + r / (p_cov + FLT_EPSILON);
    float K = res.s_prev1 / denom;

    // 4. Compute output e[n]
    float e = s - res.s_prev1 * res.a + res.s_prev2;

    // 5. Update coefficient a[n]
    float a = res.a + K * e;

    // 6. Handle coefficient bounds to stay within acos() domain
    if (a > 2.0f || a < -2.0f) {
      a = (a > 2.0f) ? 1.99999f : -1.99999f;
    }

    // 7. Update error covariance
    p_cov = (1.0f - K * res.s_prev1) * p_cov;

    // 8. Compute frequency estimate
    float omega_hat = std::acos(a / 2.0f);         // rad/sample
    float f_est = (omega_hat / delta_t) / (2.0f * static_cast<float>(M_PI));  // Hz

    // 9. Update state
    res.a = a;
    res.update_state(s);

    if (e_out) {
      *e_out = e;
    }

    return f_est;
  }

  // Get the current phase estimate (in radians)
  float get_phase() const {
    return res.get_phase();
  }

  // Optional accessors if needed
  float get_a() const { return res.a; }
  float get_p_cov() const { return p_cov; }
};

#endif // KALMANF_H
