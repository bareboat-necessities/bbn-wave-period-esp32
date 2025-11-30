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
  static constexpr Real defaultRho = Real(0.995);
  static constexpr Real default_a  = Real(1.9999);

  // “numerical stability” hook for low frequencies,
  // does not affect adaptation in this implementation.
  static constexpr Real TIME_SCALE = Real(100);

  class ANFResonator {
  public:
    Real s_prev1 = Real(0);
    Real s_prev2 = Real(0);
    Real a       = default_a;
    Real rho     = defaultRho;
    Real rho_sq  = defaultRho * defaultRho;

    void init(Real rho_init, Real a_init, Real s1, Real s2) {
      rho    = rho_init;
      rho_sq = rho * rho;
      a      = a_init;
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

  ANFResonator res;

  // Kalman parameters
  Real p_cov = Real(1);
  Real q     = Real(1e-5);
  Real r     = Real(3.0);

public:
  void init(Real rho      = defaultRho,
            Real q_       = Real(1e-5),
            Real r_       = Real(3.0),
            Real p_cov_   = Real(1),
            Real s_prev1_ = Real(0),
            Real s_prev2_ = Real(0),
            Real a_       = default_a)
  {
    q     = q_;
    r     = r_;
    p_cov = p_cov_;
    res.init(rho, a_, s_prev1_, s_prev2_);
  }

  // y: input sample, in *whatever* units (but tune q,r for that scale)
  // dt: actual sample period (seconds)
  Real process(Real y, Real dt, Real* e_out = nullptr) {
    Real delta_t = dt / TIME_SCALE;  // has no effect on adaptation, only on the freq formula

    // 1. resonator
    Real s = res.compute_s(y);

    // 2. prediction
    p_cov += q;

    // 3. Kalman gain (original form)
    Real signal_power = res.s_prev1 * res.s_prev1;
    Real denom = signal_power + r / (p_cov + std::numeric_limits<Real>::epsilon());
    Real K = res.s_prev1 / (denom + Real(1e-12));

    // 4. error
    Real e = s - res.s_prev1 * res.a + res.s_prev2;

    // 5. update a
    Real a = res.a + K * e;

    // 6. keep a in acos domain
    if (a > Real(2) || a < Real(-2)) {
      a = (a > Real(2)) ? Real(1.99999) : Real(-1.99999);
    }

    // 7. covariance update
    p_cov = (Real(1) - K * res.s_prev1) * p_cov;
    p_cov = std::max(p_cov, Real(1e-12));

    // 8. frequency estimate
    Real omega_hat = std::acos(a / Real(2));  // rad/sample
    Real f_est = (omega_hat / delta_t) / (Real(2) * static_cast<Real>(M_PI)); // Hz

    // 9. state update
    res.a = a;
    res.update_state(s);

    if (e_out) *e_out = e;

    // undo TIME_SCALE for compatibility
    return f_est / TIME_SCALE;
  }

  Real get_phase() const { return res.get_phase(); }
  Real get_a() const     { return res.a; }
  Real get_p_cov() const { return p_cov; }
};

#endif // KALMANF_H
