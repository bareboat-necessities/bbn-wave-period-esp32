#ifndef KALM_ARANOVSKIY_H
#define KALM_ARANOVSKIY_H

#include <cmath>
#include <algorithm>

template <typename Real = double>
class KalmAranovskiy {
public:
  // Parameters
  Real a = Real(1);     // Filter gain a
  Real b = Real(1);     // Filter gain b

  // Covariance and noise parameters
  Real P = Real(1);     // Covariance of theta
  Real Q = Real(1e-4);  // Process noise variance
  Real R = Real(1e-2);  // Measurement noise variance

  // State
  Real y = Real(0);           // Last measurement
  Real x1 = Real(0);          // Filtered signal
  Real x1_dot = Real(0);      // Derivative
  Real theta = Real(0.1);     // Estimator state (squared frequency)
  Real omega = Real(0.32);    // Angular frequency (rad/s)
  Real f = Real(omega / (2 * M_PI));  // Frequency (Hz)
  Real phase = Real(0);       // Estimated phase (rad)

  // Constructor
  KalmAranovskiy(Real omega_up = Real(0.5) * 2 * M_PI,
                 Real theta0 = Real(0.1),
                 Real P0 = Real(1), Real Q_ = Real(1e-4), Real R_ = Real(1e-2)) {
    setParams(omega_up, Q_, R_);
    setState(theta0, P0);
  }

  void setParams(Real omega_up, Real Q_, Real R_) {
    a = omega_up;
    b = omega_up;
    Q = Q_;
    R = R_;
  }

  void setState(Real theta0, Real P0) {
    theta = theta0;
    P = P0;
    omega = std::sqrt(std::max(Real(1e-8), theta));
    f = omega / (Real(2) * M_PI);
    x1 = x1_dot = y = Real(0);
    phase = Real(0);
  }

  void update(Real y_meas, Real delta_t) {
    if (!std::isfinite(y_meas)) return;

    y = y_meas;

    // First-order filter
    x1_dot = -a * x1 + b * y;
    x1 += x1_dot * delta_t;

    // Nonlinear measurement innovation model (Aranovskiy-style)
    Real innovation = (-x1 * x1 * theta
                      - a * x1 * x1_dot
                      - b * x1_dot * y);

    // Kalman-style gain
    Real S = P + R;            // Innovation covariance
    Real K = P / S;            // Adaptive gain
    P = P - K * P + Q * delta_t;  // Covariance update

    // Theta update using nonlinear innovation term
    theta += K * innovation * delta_t;

    // Clamp theta
    theta = std::max(theta, Real(1e-8));

    // Frequency estimates
    omega = std::sqrt(theta);
    f = omega / (Real(2) * M_PI);

    // Phase estimate
    phase = std::atan2(x1, y);
  }

  Real getFrequencyHz() const { return f; }
  Real getOmega() const { return omega; }
  Real getPhase() const { return phase; }

  // Optional: expose gain and covariance for diagnostics
  Real getGain() const { return P / (P + R); }
  Real getCovariance() const { return P; }
};

#endif // KALM_ARANOVSKIY_H
