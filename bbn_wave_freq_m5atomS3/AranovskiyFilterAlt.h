#ifndef ARANOVSKIY_FILTER_H
#define ARANOVSKIY_FILTER_H

#include <cmath>
#include <algorithm>

/*
  Copyright 2025, Mikhail Grushinskiy

  Aranovskiy frequency estimator (C++ version)

  Reference:
  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy
  "The New Algorithm of Sinusoidal Signal Frequency Estimation",
  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing, July 2013

  This is a nonlinear adaptive observer that tracks the frequency and phase of a sinusoidal signal.

  Uses Tustin discretization.
  
*/

template <typename Real = double>
class AranovskiyFilter {
public:
  // Parameters
  Real a = Real(1);
  Real b = Real(1);
  Real k = Real(8);

  // States
  Real y = Real(0), y_prev = Real(0);
  Real x1 = Real(0), x1_prev = Real(0);
  Real x1_dot = Real(0);
  Real theta = Real(-0.09);
  Real sigma = Real(-0.09), sigma_prev = Real(-0.09);
  Real sigma_dot = Real(0), sigma_dot_prev = Real(0);
  Real omega = Real(3.0);
  Real f = Real(0);
  Real phase = Real(0);

  static constexpr Real TIME_SCALE = Real(10);

  AranovskiyFilter(Real omega_up = Real(0.5) * 2 * M_PI, Real gain = Real(8),
                   Real x1_0 = Real(0), Real theta_0 = Real(-0.09), Real sigma_0 = Real(-0.09)) {
    setParams(omega_up, gain);
    setState(x1_0, TIME_SCALE * TIME_SCALE * theta_0, TIME_SCALE * TIME_SCALE * sigma_0);
  }

  void setParams(Real omega_up, Real gain) {
    a = TIME_SCALE * omega_up;
    b = TIME_SCALE * omega_up;
    k = TIME_SCALE * gain;
  }

  void setState(Real x1_init, Real theta_init, Real sigma_init) {
    x1 = x1_prev = x1_init;
    theta = theta_init;
    sigma = sigma_prev = sigma_init;
    sigma_dot = sigma_dot_prev = Real(0);
    y = y_prev = Real(0);
    omega = std::sqrt(std::max(Real(1e-12), std::abs(theta)));
    f = omega / (Real(2) * M_PI);
    phase = Real(0);
  }

  void update(Real y_meas, Real dt) {
    if (!std::isfinite(y_meas)) return;

    Real delta_t = dt / TIME_SCALE;

    // Save previous state
    x1_prev = x1;
    y_prev = y;
    sigma_prev = sigma;
    sigma_dot_prev = sigma_dot;

    y = y_meas;

    // === TUSTIN: x1 update ===
    const Real denom = (Real(2) + a * delta_t);
    const Real numer = (Real(2) - a * delta_t);
    x1 = (numer * x1_prev + 2 * b * delta_t * y + 2 * b * delta_t * y_prev) / denom;

    // === TUSTIN-consistent x1_dot ===
    x1_dot = Real(0.5) * (-a * x1 + b * y - a * x1_prev + b * y_prev);

    // === Adaptation ===
    Real signal_energy = x1 * x1 + y * y + Real(1e-12);
    Real gain_scaling = signal_energy / (signal_energy + Real(1e-6));

    Real phi = x1 * x1 * theta + a * x1 * x1_dot + b * x1_dot * y;
    Real update_term = -k * phi * gain_scaling;

    sigma_dot = clamp_value(update_term, Real(-1e12), Real(1e12));

    // === TUSTIN: sigma update ===
    sigma = sigma_prev + (delta_t / Real(2)) * (sigma_dot + sigma_dot_prev);

    // === Frequency update ===
    theta = sigma + k * b * x1 * y;
    omega = std::sqrt(std::max(Real(1e-12), std::abs(theta)));
    f = omega / (Real(2) * M_PI);

    // === Phase ===
    phase = std::atan2(x1, y);
  }

  Real getFrequencyHz() const { return f / TIME_SCALE; }
  Real getOmega() const { return omega; }
  Real getPhase() const { return phase; }

private:
  Real clamp_value(const Real& val, const Real& low, const Real& high) {
    return (val < low) ? low : (val > high) ? high : val;
  }
};

#endif // ARANOVSKIY_FILTER_H
