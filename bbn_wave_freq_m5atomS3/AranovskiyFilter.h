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
  Real a = Real(1);  // Input filter gain
  Real b = Real(1);  // Input filter gain
  Real k = Real(8);  // Adaptive gain

  // State
  Real y = Real(0);          // Last measurement
  Real x1 = Real(0);         // Internal filtered state
  Real theta = Real(-0.09);  // Estimator variable
  Real sigma = Real(-0.09);  // Estimator variable
  Real x1_dot = Real(0);
  Real sigma_dot = Real(0);
  Real omega = Real(3.0);      // Estimated angular frequency (rad/s)
  Real f = Real(0);          // Estimated frequency (Hz)
  Real phase = Real(0);      // Estimated phase (radians)

  // Constructor
  AranovskiyFilter(Real omega_up = Real(0.5) * 2 * M_PI, Real gain = Real(8),
                   Real x1_0 = Real(0), Real theta_0 = Real(-0.09), Real sigma_0 = Real(-0.09))
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
    y = Real(0);
    omega = std::sqrt(std::max(Real(1e-12), std::abs(theta)));
    f = omega / (Real(2) * M_PI);
    phase = Real(0);
  }

  // Update filter with new measurement and time step
  void update(Real y_meas, Real delta_t) {
      if (!std::isfinite(y_meas)) return;
  
      y = y_meas;
  
      // First-order low-pass filter
      x1_dot = -a * x1 + b * y;
  
      // Signal energy check (x1 ≈ filtered y)
      Real signal_energy = x1 * x1 + y * y + Real(1e-12); // avoid divide-by-zero
      Real gain_scaling = signal_energy / (signal_energy + Real(1e-6)); // softness of update on low energy signal reading
  
      // Nonlinear adaptation law (scaled by gain_scaling)
      Real update_term = (-k * x1 * x1 * theta
                         - k * a * x1 * x1_dot
                         - k * b * x1_dot * y) * gain_scaling;
  
      sigma_dot = clamp_value(update_term, Real(-1e9), Real(1e9));
  
      // Integrate states
      x1 += x1_dot * delta_t;
      sigma += sigma_dot * delta_t;
  
      // Update theta with current sigma
      theta = sigma + k * b * x1 * y;
  
      // Clamp theta to avoid invalid sqrt
      theta = clamp_value(theta, Real(1e-5), Real(100.0));
  
      // Update frequency estimate
      omega = std::sqrt(theta);
      f = omega / (Real(2) * M_PI);
  
      // Phase estimate
      phase = std::atan2(x1, y);
  }
  
  // Get current frequency estimate (Hz)
  Real getFrequencyHz() const { return f; }

  // Get current angular frequency estimate (rad/s)
  Real getOmega() const { return omega; }

  // Get current phase estimate (radians)
  Real getPhase() const { return phase; }

private:
  Real clamp_value(const Real& val, const Real& low, const Real& high) {
    return (val < low) ? low : (val > high) ? high : val;
  }
};

#endif // ARANOVSKIY_FILTER_H

