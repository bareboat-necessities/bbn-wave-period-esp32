#ifndef KALM_ARANOVSKIY_FILTER_H
#define KALM_ARANOVSKIY_FILTER_H

#include <cmath>
#include <algorithm>

/*
  Copyright 2025, Mikhail Grushinskiy

  KalmAranovskiy frequency estimator

  Structure is based on the nonlinear adaptive observer from:
  A. Bobtsov et al., "The New Algorithm of Sinusoidal Signal Frequency Estimation", IFAC 2013

  But coefficient adaptation uses a Kalman-style innovation update:
    σ_dot = K * innovation
  where innovation is a nonlinear function of the model mismatch,
  similar to KalmANF rather than linear Kalman filter.
*/

template <typename Real = double>
class KalmAranovskiy {
public:
  // Parameters
  Real a = Real(1);        // Filter parameter
  Real b = Real(1);        // Filter parameter
  Real K = Real(4);        // Kalman-style adaptation gain (acts like process noise)
  Real theta_min = Real(1e-8); // Lower bound for theta to avoid sqrt issues

  // State
  Real y = Real(0);          // Last input
  Real x1 = Real(0);         // Filtered signal
  Real theta = Real(3.0);    // Estimated theta (omega^2)
  Real sigma = Real(3.0);    // Internal sigma variable
  Real x1_dot = Real(0);     // Derivative of x1
  Real sigma_dot = Real(0);  // Update to sigma
  Real omega = Real(0);      // Estimated omega
  Real f = Real(0);          // Frequency (Hz)
  Real phase = Real(0);      // Estimated phase

  // Constructor
  KalmAranovskiy(Real omega_up = Real(0.5) * 2 * M_PI, Real gain = Real(4),
                 Real x1_0 = Real(0), Real theta_0 = Real(3.0), Real sigma_0 = Real(3.0)) {
    setParams(omega_up, gain);
    setState(x1_0, theta_0, sigma_0);
  }

  void setParams(Real omega_up, Real gain) {
    a = omega_up;
    b = omega_up;
    K = gain;
  }

  void setState(Real x1_init, Real theta_init, Real sigma_init) {
    x1 = x1_init;
    theta = std::max(theta_init, theta_min);
    sigma = sigma_init;
    omega = std::sqrt(theta);
    f = omega / (Real(2) * M_PI);
    phase = Real(0);
  }

  void update(Real y_meas, Real dt) {
    if (!std::isfinite(y_meas) || dt <= Real(0)) return;

    y = y_meas;

    // First-order low-pass filter
    x1_dot = -a * x1 + b * y;
    x1 += x1_dot * dt;

    // Kalman-style nonlinear innovation term
    Real innovation = x1_dot + theta * x1;

    // Nonlinear gain scaling based on signal energy
    Real signal_energy = x1 * x1 + y * y + Real(1e-12);
    Real gain_scaling = signal_energy / (signal_energy + Real(1e-4));

    // Update sigma with scaled innovation
    sigma_dot = -K * x1 * innovation * gain_scaling;
    sigma += sigma_dot * dt;

    // Update theta
    theta = sigma + b * x1 * y;
    theta = std::max(theta, theta_min); // Avoid invalid sqrt

    // Update frequency and phase
    omega = std::sqrt(theta);
    f = omega / (Real(2) * M_PI);
    phase = std::atan2(x1, y);
  }

  Real getFrequencyHz() const { return f; }
  Real getOmega() const { return omega; }
  Real getPhase() const { return phase; }

private:
  Real clamp_value(const Real& val, const Real& low, const Real& high) {
    return (val < low) ? low : (val > high) ? high : val;
  }
};

#endif // KALM_ARANOVSKIY_FILTER_H
