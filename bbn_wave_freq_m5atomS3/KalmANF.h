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

template <typename Real = double>
class KalmANF {
private:
  static constexpr float defaultRho = Real(0.99);
  static constexpr float default_a = Real(1.9999);

  // Internal time scaling for better low-frequency stability
  static constexpr Real TIME_SCALE = Real(100); // can be tuned (10x faster internal clock)

  // Internal Notch Filter Resonator
  class ANFResonator {
  public:
    Real s_prev1 = Real(0);  // s[n-1] — previous resonator output sample
    Real s_prev2 = Real(0);  // s[n-2] — two samples ago
    Real a = default_a;      // a[n] — adaptive filter coefficient = 2*cos(ω)
    Real rho = defaultRho;   // Pole radius (0 < rho < 1)
    Real rho_sq = defaultRho * defaultRho;   // Precomputed rho^2

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
  Real p_cov = Real(1);  // Kalman error covariance
  Real q = Real(1e-6f);  // Process noise covariance
  Real r = Real(1e+5);   // Measurement noise covariance

public:
  // Initialize the filter
  void init(Real rho = defaultRho, Real q_ = Real(1e-6), Real r_ = Real(1e+5), Real p_cov_ = Real(1),
            Real s_prev1 = Real(0), Real s_prev2 = Real(0), Real a_ = default_a) {
    q = q_;
    r = r_;
    p_cov = p_cov_;
    res.init(rho, a_, s_prev1, s_prev2);
  }

  // Process a single sample, return estimated frequency in Hz
  Real process(Real y, Real dt, Real* e_out = nullptr) {
    Real delta_t = dt / TIME_SCALE;
    // Compute intermediate variable s[n]
    Real s = res.compute_s(y);

    // Prediction update
    p_cov += q;

    // Compute Kalman gain
    Real signal_power = res.s_prev1 * res.s_prev1;
    Real gain_scaling = signal_power / (signal_power + Real(1e-7)); // Smooth on low (non informative) signal power near zero crossings
    Real denom = signal_power + r / (p_cov + std::numeric_limits<Real>::epsilon());
    Real K = gain_scaling * res.s_prev1 / (denom + Real(1e-12));

    // Compute output e[n]
    Real e = s - res.s_prev1 * res.a + res.s_prev2;

    // Update coefficient a[n]
    Real a = res.a + K * e;

    // Update error covariance
    p_cov = (Real(1) - K * res.s_prev1) * p_cov;
    p_cov = std::max(p_cov, Real(1e-12));
     
    // Handle coefficient bounds to stay within acos() domain
    if (a > Real(2) || a < Real(-2)) {
      a = (a > Real(2)) ? Real(1.99999) : Real(-1.99999);
    }

    // Compute frequency estimate
    Real omega_hat = std::acos(a / Real(2));         // rad/sample
    Real f_est = (omega_hat / delta_t) / (Real(2) * static_cast<Real>(M_PI));  // Hz

    // Update state
    res.a = a;
    res.update_state(s);

    if (e_out) {
      *e_out = e;
    }
    return f_est / TIME_SCALE;
  }

  // Get the current resonator phase estimate (in radians)
  Real get_phase() const {
    return res.get_phase();
  }

  // Optional accessors if needed
  Real get_a() const { return res.a; }
  Real get_p_cov() const { return p_cov; }
};

#endif // KALMANF_H
