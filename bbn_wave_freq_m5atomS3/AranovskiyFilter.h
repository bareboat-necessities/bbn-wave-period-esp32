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
  Real a     = Real(1);   // Input filter gain
  Real b     = Real(1);   // Input filter gain
  Real k     = Real(8);   // Adaptive gain

  // State
  Real y         = Real(0);    // Last measurement
  Real x1        = Real(0);    // Internal filtered state
  Real theta     = Real(-0.09);// Estimator variable
  Real sigma     = Real(-0.09);// Estimator variable
  Real x1_dot    = Real(0);
  Real sigma_dot = Real(0);
  Real omega     = Real(3.0);  // Estimated angular frequency (rad/s)
  Real f         = Real(0);    // Estimated frequency (Hz)
  Real phase     = Real(0);    // Estimated phase (radians)

  // Internal time scaling for better low-frequency stability
  static constexpr Real TIME_SCALE = Real(20); // can be tuned (20x faster internal clock)

  // Constructor
  AranovskiyFilter(Real omega_up = Real(0.5) * 2 * M_PI,
                   Real gain     = Real(8),
                   Real x1_0     = Real(0),
                   Real theta_0  = Real(-0.09),
                   Real sigma_0  = Real(-0.09))
  {
    setParams(omega_up, gain);
    setState(x1_0, theta_0, sigma_0);
  }

  // Set filter parameters
  void setParams(Real omega_up, Real gain) {
    a = TIME_SCALE * omega_up;
    b = TIME_SCALE * omega_up;
    k = TIME_SCALE * gain;
  }

  // Set initial state
  void setState(Real x1_init, Real theta_init, Real sigma_init) {
    x1    = x1_init;
    theta = TIME_SCALE * TIME_SCALE * theta_init;
    sigma = TIME_SCALE * TIME_SCALE * sigma_init;
    y     = Real(0);

    omega = std::sqrt(std::max(Real(1e-12), std::abs(theta)));
    f     = omega / (Real(2) * M_PI);
    phase = Real(0);
  }

  // Update filter with new measurement and time step
  void update(Real y_meas, Real dt) {
    const Real delta_t = dt / TIME_SCALE;

    if (!std::isfinite(y_meas)) {
      return;
    }

    y = y_meas;

    // First-order low-pass filter
    x1_dot = -a * x1 + b * y;

    // Signal energy check (x1 ≈ filtered y)
    // signal_energy = x1^2 + y^2 + eps, with FMA for x1^2 + y^2
    const Real signal_energy = std::fma(x1, x1, y * y) + Real(1e-12); // avoid divide-by-zero
    const Real gain_scaling  = signal_energy / (signal_energy + Real(1e-6)); // soften updates on low-energy signals

    // Nonlinear adaptation law (scaled by gain_scaling)
    //
    // Original:
    //   phi = x1^2 * theta + a * x1 * x1_dot + b * x1_dot * y
    // Factorization:
    //   phi = x1^2 * theta + x1_dot * (a * x1 + b * y)
    //
    const Real x1_sq = x1 * x1;
    const Real tmp   = std::fma(b, y, a * x1);              // a * x1 + b * y
    const Real phi   = std::fma(x1_dot, tmp, x1_sq * theta);// x1_sq * theta + x1_dot * tmp

    const Real update_term = -k * phi * gain_scaling;

    sigma_dot = clamp_value(update_term, Real(-1e7), Real(1e7));

    // Update theta and omega
    theta = std::fma(k * b * x1, y, sigma); // theta = sigma + k * b * x1 * y

    omega = std::sqrt(std::max(Real(1e-7), std::abs(theta)));
    f     = omega / (Real(2) * M_PI);

    // State integration (internal scaled time)
    x1    += x1_dot    * delta_t;
    sigma += sigma_dot * delta_t;

    // Phase estimation: phase of (x1 + j y)
    phase = std::atan2(x1, y);
  }

  // Get current frequency estimate (Hz)
  Real getFrequencyHz() const {
    return f / TIME_SCALE;
  }

  // Get current angular frequency estimate (rad/s, *internal* scaled)
  Real getOmega() const {
    return omega;
  }

  // Get current oscillator phase estimate (radians)
  Real getPhase() const {
    return phase;
  }

private:
  static constexpr Real clamp_value(Real val, Real low, Real high) {
    return (val < low) ? low : (val > high ? high : val);
  }
};

#endif // ARANOVSKIY_FILTER_H

