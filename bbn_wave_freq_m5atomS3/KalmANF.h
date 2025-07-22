#ifndef KALMANF_H
#define KALMANF_H

#include <math.h>
#include <limits>

/*
   See: https://github.com/randyaliased/KalmANF/

   and

   See: R. Ali, T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter adaptive notch filter", 
   Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, September 2023
*/

template <typename Real = float>
class KalmANF {
private:
  // Internal Notch Filter Resonator
  class ANFResonator {
  public:
    Real s_prev1 = Real(0);  // s[n-1] — previous resonator output sample
    Real s_prev2 = Real(0);  // s[n-2] — two samples ago
    Real a = Real(0);        // a[n] — adaptive filter coefficient = 2*cos(ω)
    Real rho = Real(0);      // Pole radius (0 < rho < 1)
    Real rho_sq = Real(0);   // Precomputed rho^2

    void init(Real rho_init, Real a_init, Real s1, Real s2) {
      rho = rho_init;
      rho_sq = rho * rho;
      a = a_init;
      s_prev1 = s1;
      s_prev2 = s2;
    }

    Real compute_s(Real y) const {
      return y + rho * s_prev1 * a - rho_sq * s_prev2;
    }

    void update_state(Real s) {
      s_prev2 = s_prev1;
      s_prev1 = s;
    }

    Real get_phase() const {
      return std::atan2(s_prev1, s_prev2);
    }
  };

  ANFResonator res;  // Embedded second-order IIR resonator

  // Kalman parameters
  Real p_cov = 1.0;  // Kalman error covariance
  Real q = 0.0;      // Process noise covariance
  Real r = 0.0;      // Measurement noise covariance

public:
  // Initialize the filter
  void init(Real rho, Real q_, Real r_, Real p_cov_,
            Real s_prev1, Real s_prev2, Real a_prev) {
    q = q_;
    r = r_;
    p_cov = p_cov_;
    res.init(rho, a_prev, s_prev1, s_prev2);
  }

  // Process a single sample, return estimated frequency in Hz
  Real process(Real y, Real delta_t, Real* e_out = nullptr) {
    // Compute intermediate variable s[n]
    Real s = res.compute_s(y);

    // Prediction update
    p_cov += q;

    // Compute Kalman gain
    Real denom = res.s_prev1 * res.s_prev1 + r / (p_cov + std::numeric_limits<Real>::epsilon(););
    Real K = res.s_prev1 / denom;

    // Compute output e[n]
    Real e = s - res.s_prev1 * res.a + res.s_prev2;

    // Update coefficient a[n]
    Real a = res.a + K * e;

    // Handle coefficient bounds to stay within acos() domain
    if (a > 2.0f || a < -2.0f) {
      a = (a > 2.0f) ? 1.99999f : -1.99999f;
    }

    // Update error covariance
    p_cov = (1.0f - K * res.s_prev1) * p_cov;

    // Compute frequency estimate
    Real omega_hat = std::acos(a / 2.0f);         // rad/sample
    Real f_est = (omega_hat / delta_t) / (2.0f * static_cast<Real>(M_PI));  // Hz

    // Update state
    res.a = a;
    res.update_state(s);

    if (e_out) {
      *e_out = e;
    }
    return f_est;
  }

  // Get the current phase estimate (in radians)
  Real get_phase() const {
    return res.get_phase();
  }

  // Optional accessors if needed
  Real get_a() const { return res.a; }
  Real get_p_cov() const { return p_cov; }
};

#endif // KALMANF_H
