#ifndef KALM_ARANOVSKIY_FILTER_H
#define KALM_ARANOVSKIY_FILTER_H

#include <cmath>
#include <algorithm>

/*
  KalmAranovskiy Filter

  Hybrid observer using Aranovskiy's internal structure for signal modeling,
  with Kalman-style innovation-based adaptation for frequency estimation.

  Copyright 2024–2025, Mikhail Grushinskiy
*/

template <typename Real = double>
class KalmAranovskiy {
public:
  // Parameters
  Real a = Real(1);      // Low-pass filter gain
  Real b = Real(1);      // Low-pass filter gain
  Real q = Real(1e-4);   // Process noise for sigma
  Real r = Real(1e-2);   // Measurement noise for innovation model

  // State
  Real x1 = Real(0);         // Internal filtered state
  Real sigma = Real(0.01);   // Internal adaptive state
  Real theta = Real(0.01);   // Estimated parameter
  Real omega = Real(1.0);    // Estimated angular frequency (rad/s)
  Real f = Real(omega / (2 * M_PI)); // Estimated frequency (Hz)
  Real phase = Real(0);      // Estimated phase (radians)

  // Kalman variables
  Real P = Real(1.0);        // Covariance of sigma

  // Constructor
  KalmAranovskiy(Real omega_up = Real(1.0), Real q_ = Real(1e-4), Real r_ = Real(1e-2)) {
    setParams(omega_up, q_, r_);
  }

  void setParams(Real omega_up, Real q_, Real r_) {
    a = omega_up;
    b = omega_up;
    q = q_;
    r = r_;
  }

  void setState(Real x1_init, Real sigma_init, Real P_init = Real(1.0)) {
    x1 = x1_init;
    sigma = sigma_init;
    P = P_init;
    theta = std::max(Real(1e-8), sigma);
    omega = std::sqrt(theta);
    f = omega / (2 * M_PI);
  }

  void update(Real y, Real dt) {
    if (!std::isfinite(y)) return;

    // First-order filter
    Real x1_dot = -a * x1 + b * y;
    x1 += x1_dot * dt;

    // Predict sigma (no dynamics, but Q is added)
    P += q * dt;

    // Innovation (pseudo-measurement): we model theta ≈ x1 * y
    Real h = b * x1 * y;               // measurement model: h(sigma)
    Real theta_meas = sigma + h;      // measurement prediction
    Real innovation = theta_meas - theta;  // innovation
    Real S = P + r;                   // innovation covariance
    Real K = P / S;                   // Kalman gain

    // Kalman update for sigma
    sigma += K * innovation;
    P *= (1 - K);

    // Update theta using Aranovskiy correction law
    theta = std::max(Real(1e-8), sigma + h);

    // Frequency estimate
    omega = std::sqrt(theta);
    f = omega / (2 * M_PI);

    // Phase estimate (same as Aranovskiy)
    phase = std::atan2(x1, y);
  }

  Real getFrequencyHz() const { return f; }
  Real getOmega() const { return omega; }
  Real getPhase() const { return phase; }

private:
  Real clamp(const Real& x, const Real& lo, const Real& hi) {
    return std::min(std::max(x, lo), hi);
  }
};

#endif // KALM_ARANOVSKIY_FILTER_H
