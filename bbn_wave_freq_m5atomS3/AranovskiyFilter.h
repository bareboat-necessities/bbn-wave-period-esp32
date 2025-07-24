#ifndef ARANOVSKIY_FILTER_H
#define ARANOVSKIY_FILTER_H

#include <cmath>
#include <algorithm>

/*
  Copyright 2024-2025, Mikhail Grushinskiy

  Aranovskiy frequency estimator (C++ version)

  Reference:
  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy
  "The New Algorithm of Sinusoidal Signal Frequency Estimation",
  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing, July 2013

  This is a nonlinear adaptive observer that tracks the frequency and phase of a sinusoidal signal.
*/

template <typename Real = double>
class AranovskiyFilter {
public:
  // Parameters
  Real a = 1.0;  // Input filter gain
  Real b = 1.0;  // Input filter gain
  Real k = 1.0;  // Adaptive gain

  // State
  Real y = 0.0;          // Last measurement
  Real x1 = 0.0;         // Internal filtered state
  Real theta = -0.25;    // Estimator variable
  Real sigma = -0.25;    // Estimator variable
  Real x1_dot = 0.0;
  Real sigma_dot = 0.0;
  Real omega = 0.0;      // Estimated angular frequency (rad/s)
  Real f = 0.0;          // Estimated frequency (Hz)
  Real phase = 0.0;      // Estimated phase (radians)

  static constexpr Real PI = 3.14159265358979323846;

  // Constructor
  AranovskiyFilter(Real omega_up = 1.0 * 2 * PI, Real gain = 2.0,
                   Real x1_0 = 0.0, Real theta_0 = -0.25, Real sigma_0 = -0.25)
  {
    setParams(omega_up, gain);
    setState(x1_0, theta_0, sigma_0);
  }

  // Set filter parameters
  void setParams(Real omega_up, Real gain) {
    a = omega_up;
    b = omega_up;
    k = gain;
  }

  // Set initial state
  void setState(Real x1_init, Real theta_init, Real sigma_init) {
    x1 = x1_init;
    theta = theta_init;
    sigma = sigma_init;
    y = 0.0;
    omega = std::sqrt(std::max(1e-10, std::abs(theta)));
    f = omega / (2.0 * PI);
    phase = 0.0;
  }

  // Update filter with new measurement and time step
  void update(Real y_meas, Real delta_t) {
    if (!std::isfinite(y_meas)) return;

    y = y_meas;

    // 1. First-order low-pass filter
    x1_dot = -a * x1 + b * y;

    // 2. Nonlinear adaptation law
    Real update_term = -k * x1 * x1 * theta
                       - k * a * x1 * x1_dot
                       - k * b * x1_dot * y;
    sigma_dot = std::clamp(update_term, -1e7, 1e7);

    // 3. Update theta and omega
    theta = sigma + k * b * x1 * y;
    omega = std::sqrt(std::max(1e-10, std::abs(theta)));
    f = omega / (2.0 * PI);

    // 4. State integration
    x1 += x1_dot * delta_t;
    sigma += sigma_dot * delta_t;

    // 5. Phase estimation
    phase = std::atan2(x1, y);
  }

  // Get current frequency estimate (Hz)
  Real getFrequencyHz() const { return f; }

  // Get current angular frequency estimate (rad/s)
  Real getOmega() const { return omega; }

  // Get current phase estimate (radians)
  Real getPhase() const { return phase; }
};

#endif // ARANOVSKIY_FILTER_H

