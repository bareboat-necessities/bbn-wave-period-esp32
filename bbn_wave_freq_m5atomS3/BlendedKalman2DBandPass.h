#ifndef BLENDED_KALMAN_2D_BANDPASS_H
#define BLENDED_KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>
#include <algorithm>
#include <utility>

/**
 * Copyright 2025, Mikhail Grushinskiy
 * 
 * @brief 2D Kalman-adaptive band-pass filter for horizontal wave-like signals.
 *
 * This filter adaptively estimates the dominant frequency and phase of a 2D oscillating signal
 * using a second-order resonator and Kalman-style update for the resonance coefficient.
 * Useful for filtering and analyzing horizontal wave motion (e.g., from IMU acceleration).
 */
class BlendedKalman2DBandPass {
public:
  /**
   * @brief Constructor for the filter.
   * 
   * @param rho Damping factor of the resonator (close to 1.0).
   * @param q Process noise for adaptive frequency update.
   * @param r Measurement noise for adaptive frequency update.
   * @param alpha_blend Smoothing factor for blending new frequency estimate.
   */
  BlendedKalman2DBandPass(float rho = 0.99f, float q = 0.001f, float r = 0.1f, float alpha_blend = 0.95f)
    : rho(rho), q(q), r(r), alpha(alpha_blend)
  {
    rho_sq = rho * rho;
    reset();
  }

  /**
   * @brief Resets the filter's internal state.
   */
  void reset() {
    s_prev1.setZero();
    s_prev2.setZero();
    a_prev = A_CLAMP;  // Safe default for high-frequency rejection
    p_cov = 1.0f;
  }

  /**
   * @brief Sets the initial frequency estimate (in Hz) before filtering begins.
   * 
   * @param freq_hz Initial frequency estimate in Hz.
   * @param delta_t Sampling period in seconds.
   */
  void setFrequencyEstimate(float freq_hz, float delta_t) {
    float omega_dt = 2.0f * M_PI * freq_hz * delta_t;
    float a = 2.0f * std::cos(omega_dt);
    a_prev = std::clamp(a, -A_CLAMP, A_CLAMP);
  }

  /**
   * @brief Processes a new 2D signal sample and updates the internal state.
   * 
   * @param a_x X-component of the signal (e.g., horizontal acceleration).
   * @param a_y Y-component of the signal.
   * @param delta_t Sampling period in seconds.
   * @return Filtered signal magnitude (instantaneous amplitude).
   */
  float process(float a_x, float a_y, float delta_t) {
    Eigen::Vector2f y(a_x, a_y);

    // Apply second-order resonator
    Eigen::Vector2f s = y + rho * a_prev * s_prev1 - rho_sq * s_prev2;

    // Predict covariance for Kalman gain
    p_cov += q;

    float s_norm_sq = s_prev1.squaredNorm();
    float denom = s_norm_sq + r / p_cov;
    float K = (denom > 1e-6f) ? (s_norm_sq / denom) : 0.0f;

    // Residual for a update
    Eigen::Vector2f e = s - a_prev * s_prev1 + s_prev2;

    // Kalman update for resonance coefficient a
    float a_meas = a_prev + K * s_prev1.dot(e);
    a_meas = std::clamp(a_meas, -A_CLAMP, A_CLAMP);

    // Smooth update
    a_prev = alpha * a_meas + (1.0f - alpha) * a_prev;

    // Update covariance
    p_cov = std::max((1.0f - K) * p_cov, 1e-6f);

    // Update resonator state
    s_prev2 = s_prev1;
    s_prev1 = s;

    return s.norm();  // Optionally return filtered signal magnitude
  }

  /**
   * @brief Returns the filtered X-component of the signal.
   * @return Filtered a_x.
   */
  float getFilteredAx() const { return s_prev1.x(); }

  /**
   * @brief Returns the filtered Y-component of the signal.
   * @return Filtered a_y.
   */
  float getFilteredAy() const { return s_prev1.y(); }

  /**
   * @brief Returns the index of the dominant axis.
   * 
   * @return 0 if X is dominant, 1 if Y is dominant.
   */
  int getDominantAxis() const {
    return (std::abs(s_prev1.x()) >= std::abs(s_prev1.y())) ? 0 : 1;
  }

  /**
   * @brief Computes amplitude and phase from the dominant axis.
   * 
   * @param delta_t Sampling period in seconds.
   * @return Pair of {amplitude, phase in radians}.
   */
  std::pair<float, float> getAmplitudePhase(float delta_t) const {
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega_dt = std::acos(std::clamp(a / 2.0f, -1.0f, 1.0f));
    float sin_omega_dt = std::sin(omega_dt);
    float cos_omega_dt = std::cos(omega_dt);

    float amp_x = std::abs(s_prev1.x());
    float amp_y = std::abs(s_prev1.y());

    float s1, s2;
    if (amp_x >= amp_y) {
      s1 = s_prev1.x();
      s2 = s_prev2.x();
    } else {
      s1 = s_prev1.y();
      s2 = s_prev2.y();
    }

    // Avoid divide-by-zero
    if (std::abs(sin_omega_dt) < 1e-6f)
      return {std::abs(s1), 0.0f};

    float q = (s1 - s2 * cos_omega_dt) / sin_omega_dt;
    float amplitude = std::sqrt(s1 * s1 + q * q);
    float phase = std::atan2(q, s1);

    return {amplitude, phase};
  }

  /**
   * @brief Returns the estimated amplitude of the dominant signal.
   * 
   * @param delta_t Sampling period in seconds.
   * @return Amplitude.
   */
  float getAmplitude(float delta_t) const {
    return getAmplitudePhase(delta_t).first;
  }

  /**
   * @brief Returns the estimated phase of the dominant signal in radians.
   * 
   * @param delta_t Sampling period in seconds.
   * @return Phase in radians.
   */
  float getPhase(float delta_t) const {
    return getAmplitudePhase(delta_t).second;
  }

  /**
   * @brief Returns the estimated frequency (Hz) of the signal.
   * 
   * @param delta_t Sampling period in seconds.
   * @return Frequency in Hz.
   */
  float getFrequency(float delta_t) const {
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega = std::acos(std::clamp(a / 2.0f, -1.0f, 1.0f));
    return (omega / delta_t) / (2.0f * M_PI);  // rad/s → Hz
  }

private:
  static constexpr float A_CLAMP = 1.9999f;

  float rho;
  float rho_sq;
  float q;
  float r;
  float alpha;

  float a_prev;     // Previous resonance coefficient (cos(ωΔt) * 2)
  float p_cov;      // Covariance for adaptive update

  Eigen::Vector2f s_prev1;  // Previous resonator state
  Eigen::Vector2f s_prev2;  // Second previous resonator state
};

#endif // BLENDED_KALMAN_2D_BANDPASS_H
