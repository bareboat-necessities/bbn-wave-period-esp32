#ifndef KALM_ARANOVSKIY_H
#define KALM_ARANOVSKIY_H

#include <cmath>
#include <algorithm>

template <typename Real = double>
class KalmAranovskiy {
public:
  // Parameters
  Real a = Real(1);  // Filter gain a
  Real b = Real(1);  // Filter gain b

  // Covariance and noise parameters
  Real P = Real(1);     // Covariance of theta
  Real Q = Real(1e-4);  // Process noise variance
  Real R = Real(1e-2);  // Measurement noise variance

  // State
  Real y = Real(0);
  Real x1 = Real(0);
  Real x1_dot = Real(0);
  Real theta = Real(0.1);
  Real omega = std::sqrt(theta);
  Real f = omega / (2 * M_PI);
  Real phase = Real(0);

  // Constructor
  KalmAranovskiy(Real omega_up = Real(0.5) * 2 * M_PI,
                 Real theta0 = Real(0.1),
                 Real P0 = Real(1),
                 Real Q_ = Real(1e-4),
                 Real R_ = Real(1e-2)) {
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
    theta = std::max(theta0, Real(1e-8));
    P = std::max(P0, Real(1e-8));
    omega = std::sqrt(theta);
    f = omega / (2 * M_PI);
    x1 = x1_dot = y = Real(0);
    phase = Real(0);
  }

  void update(Real y_meas, Real delta_t) {
    if (!std::isfinite(y_meas) || delta_t <= Real(0)) return;

    y = y_meas;

    // Filter step (resonator-style)
    x1_dot = -a * x1 + b * y;
    x1 += x1_dot * delta_t;

    // Nonlinear innovation (Aranovskiy-style)
    Real innovation = (-x1 * x1 * theta
                      - a * x1 * x1_dot
                      - b * x1_dot * y);

    // Kalman gain
    Real S = P + R;
    Real K = P / S;

    // Theta update
    theta += K * innovation * delta_t;
    theta = std::max(theta, Real(1e-8));

    // Covariance update (Joseph form for numerical stability)
    Real I_K = Real(1) - K;
    P = I_K * P * I_K + K * R * K + Q * delta_t;
    P = std::max(P, Real(1e-10)); // Keep P positive

    // Frequency and phase
    omega = std::sqrt(theta);
    f = omega / (2 * M_PI);
    phase = std::atan2(x1, y);
  }

  // Accessors
  Real getFrequencyHz() const { return f; }
  Real getOmega() const { return omega; }
  Real getPhase() const { return phase; }
  Real getGain() const { return P / (P + R); }
  Real getCovariance() const { return P; }
};

#endif // KALM_ARANOVSKIY_H
