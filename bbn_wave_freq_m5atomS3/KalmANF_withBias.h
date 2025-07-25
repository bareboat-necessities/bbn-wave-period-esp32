
#ifndef KALMANF_WITH_BIAS_H
#define KALMANF_WITH_BIAS_H

#include <cmath>
#include <limits>
#include <algorithm> // std::clamp

/**
 * KalmANF – Kalman-augmented Adaptive Notch Filter with Bias Estimation.
 *
 * Implements the frequency tracker from:
 * R. Ali and T. van Waterschoot, "A frequency tracker based on a Kalman filter update of a single parameter
 * adaptive notch filter," DAFx-23.
 *
 * Extended to estimate additive input bias modeled as a random walk.
 */

template <typename Real = float>
class KalmANF {
private:
  class ANFResonator {
  public:
    Real s_prev1 = Real(0);  // s[n-1]
    Real s_prev2 = Real(0);  // s[n-2]
    Real a = Real(0);        // a[n] = 2*cos(omega)
    Real rho = Real(0);      // Pole radius
    Real rho_sq = Real(0);   // Cached rho^2

    void init(Real rho_init, Real a_init, Real s1, Real s2) {
      rho = rho_init;
      rho_sq = rho * rho;
      a = a_init;
      s_prev1 = s1;
      s_prev2 = s2;
    }

    Real compute_s(Real y, Real b) const {
      return (y - b) + rho * s_prev1 * a - rho_sq * s_prev2;
    }

    void update_state(Real s) {
      s_prev2 = s_prev1;
      s_prev1 = s;
    }

    Real get_phase() const {
      return std::atan2(s_prev1, s_prev2);
    }
  };

  ANFResonator res;

  // State variables
  Real a = Real(0);  // Notch filter parameter: 2*cos(omega)
  Real b = Real(0);  // Input signal bias

  // 2x2 error covariance matrix
  Real P11 = Real(1);  // var(a)
  Real P12 = Real(0);  // cov(a, b)
  Real P22 = Real(1);  // var(b)

  // Noise parameters
  Real q_a = Real(0);  // process noise for a
  Real q_b = Real(0);  // process noise for b
  Real r = Real(0);    // measurement noise

public:
  /**
   * Initialize filter and Kalman state
   */
  void init(Real rho, Real q_a_, Real q_b_, Real r_,
            Real P11_, Real P22_,
            Real s_prev1, Real s_prev2,
            Real a_init, Real b_init) {
    q_a = q_a_;
    q_b = q_b_;
    r = r_;
    P11 = P11_;
    P22 = P22_;
    P12 = Real(0);
    a = a_init;
    b = b_init;
    res.init(rho, a_init, s_prev1, s_prev2);
  }

  /**
   * Process a new sample.
   * @param y        input signal
   * @param delta_t  sampling period in seconds
   * @param e_out    optional pointer to store residual error
   * @return         estimated frequency in Hz
   */
  Real process(Real y, Real delta_t, Real* e_out = nullptr) {
    // Compute resonator output with bias removed
    Real s = res.compute_s(y, b);

    // Residual error
    Real e = s - a * res.s_prev1 + res.s_prev2;

    // Measurement Jacobian: h = a * s_prev1 + b
    Real H1 = res.s_prev1; // ∂h/∂a
    Real H2 = Real(1);     // ∂h/∂b

    // Kalman gain
    Real denom = H1 * (H1 * P11 + H2 * P12) + H2 * (H1 * P12 + H2 * P22) + r;
    denom += std::numeric_limits<Real>::epsilon(); // prevent divide-by-zero
    Real K1 = (P11 * H1 + P12 * H2) / denom;
    Real K2 = (P12 * H1 + P22 * H2) / denom;

    // Update estimates
    a += K1 * e;
    b += K2 * e;
    a = std::clamp(a, Real(-1.99999), Real(1.99999));  // stay within acos domain

    // Update covariance
    Real P11_new = P11 - K1 * (H1 * P11 + H2 * P12);
    Real P12_new = P12 - K1 * (H1 * P12 + H2 * P22);
    Real P22_new = P22 - K2 * (H1 * P12 + H2 * P22);
    P11 = P11_new + q_a;
    P12 = P12_new;
    P22 = P22_new + q_b;

    // Update resonator
    res.a = a;
    res.update_state(s);

    if (e_out) {
      *e_out = e;
    }

    // Convert to frequency in Hz
    Real omega_hat = std::acos(a / Real(2));
    Real f_est = (omega_hat / delta_t) / (Real(2) * static_cast<Real>(M_PI));
    return f_est;
  }

  // Accessors
  Real get_phase() const { return res.get_phase(); }
  Real get_bias()  const { return b; }
  Real get_a()     const { return a; }
  Real get_p11()   const { return P11; }
  Real get_p22()   const { return P22; }
  Real get_frequency_rad() const { return std::acos(a / Real(2)); }
};

#endif // KALMANF_WITH_BIAS_H
