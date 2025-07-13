#ifndef BLENDED_KALMAN_2D_BANDPASS_H
#define BLENDED_KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>
#include <algorithm>
#include <utility>

#ifdef KALMAN_2D_BANDPASS_TEST
#include <iostream>
#include <fstream>
#endif

/**
 * Copyright 2025, Mikhail Grushinskiy
 * 
 * @brief 2D Kalman-adaptive band-pass filter for horizontal wave-like signals.
 *
 * This filter adaptively estimates the dominant frequency (ω), phase (θ), and amplitude (A)
 * of a 2D oscillatory signal using a second-order resonator combined with a Kalman-style
 * update for the resonance coefficient.
 *
 * The signal is assumed to lie in an unknown but fixed plane (e.g., ocean wave horizontal motion).
 *
 *  • Designed for horizontal motion signals (e.g., IMU acceleration in X and Y).
 *  • Handles slowly varying frequency and noisy input.
 *
 * RESONATOR STRUCTURE
 *
 * The core is a second-order band-pass resonator:
 *   s[n] = y[n] + ρ · a · s[n−1] − ρ² · s[n−2]
 *
 * where:
 *   • y[n] — input signal vector [aₓ, aᵧ]
 *   • s[n] — filtered output vector
 *   • ρ — damping factor (near 1.0)
 *   • a ≈ 2 · cos(ω · Δt) — adaptive resonance coefficient
 *
 * ADAPTIVE FREQUENCY TRACKING (Kalman-like update)
 *
 * The coefficient `a` is updated based on the filter output:
 *   1. Prediction:
 *      s[n] = y[n] + ρ · a · s[n−1] − ρ² · s[n−2]
 *   2. Residual:
 *      e = s[n] − a · s[n−1] + s[n−2]
 *   3. Kalman gain:
 *      K = (‖s[n−1]‖²) / (‖s[n−1]‖² + r / p)
 *   4. Update:
 *      a ← a + K · (e · s[n−1])  // dot product
 *   5. Blend (optional smoothing):
 *      a ← α · a_new + (1−α) · a_prev
 *
 * OUTPUT SIGNAL
 *
 * Filtered values:
 *   • getFilteredAx() — s[n].x
 *   • getFilteredAy() — s[n].y
 *
 * FREQUENCY ESTIMATION
 *
 * Angular frequency:
 *   • ωΔt = arccos(a / 2)
 * Frequency in Hz:
 *   • f = ω / (2π) = arccos(a / 2) / (2π · Δt)
 *
 * NOTES
 * • Coefficient `a` is clamped to (−2, 2) to ensure filter stability.
 * • Covariance `p` is adapted and bounded from below to avoid divergence.
 */
class BlendedKalman2DBandPass {
public:
  /**
   * @brief Constructor for the filter.
   * 
   * @param rho            ρ Damping factor for the resonator (close to 1.0).
   * @param q              q Process noise variance for adaptive frequency tracking.
   * @param r              r Measurement noise variance for Kalman update.
   * @param alpha_blend    α Blending factor for smoothing updated frequency (0 = no smoothing, 1 = full).
   */
  BlendedKalman2DBandPass(float rho = 0.99f, float q = 0.001f, float r = 0.1f, float alpha_blend = 0.95f)
    : rho(rho), q(q), r(r), alpha(alpha_blend)
  {
    rho_sq = rho * rho;
    reset();
  }

  /**
   * @brief Resets the filter's internal state.
   *   • Zeroes the resonator state (s[n−1], s[n−2]).
   *   • Initializes adaptive frequency state (a) and covariance (p).
   *   • Sets a to a safe default to suppress high-frequency response initially.
   */
  void reset() {
    s_prev1.setZero();
    s_prev2.setZero();
    a_prev = A_CLAMP;  // Safe default for high-frequency rejection
    p_cov = 1.0f;
    samples_processed = 0;
  }

  /**
   * @brief Sets the initial frequency estimate (in Hz) before filtering begins.
   * 
   * @param freq_hz     f₀ Initial frequency estimate (in Hz).
   * @param delta_t     Δt Sampling interval (seconds).
   *
   * ωΔt = 2π · f₀ · Δt  
   * a = 2 · cos(ωΔt)
   */
  void setFrequencyEstimate(float freq_hz, float delta_t) {
    float omega_dt = 2.0f * M_PI * freq_hz * delta_t;
    float a = 2.0f * std::cos(omega_dt);
    a_prev = std::clamp(a, -A_CLAMP, A_CLAMP);
  }

  /**
   * @brief Processes a new 2D signal sample and updates the internal state.
   * 
   * @param a_x         aₓ X-component of the signal (e.g. IMU horizontal accel).
   * @param a_y         aᵧ Y-component of the signal.
   * @param delta_t     Δt Sampling interval (seconds).
   * @return            Magnitude of the filtered output ‖s[n]‖ (instantaneous amplitude estimate).
   */
  float process(float a_x, float a_y, float freq_est_hz, float delta_t) {
    Eigen::Vector2f y(a_x, a_y);

    if (samples_processed == 0) {
      // Initialize resonator states to match expected oscillation
      setFrequencyEstimate(freq_est_hz, delta_t);
      float omega_dt = 2.0f * M_PI * freq_est_hz * delta_t;
      s_prev1 = Eigen::Vector2f(a_x * std::cos(omega_dt), 
                               a_y * std::cos(omega_dt));
      s_prev2 = Eigen::Vector2f(a_x * std::cos(2*omega_dt), 
                                a_y * std::cos(2*omega_dt));
      samples_processed = 1;
      return s_prev1.norm();
    }

    // Apply second-order resonator
    Eigen::Vector2f s = y + rho * a_prev * s_prev1 - rho_sq * s_prev2;

    // Predict covariance for Kalman gain
    p_cov += q;

    float s_norm_sq = s_prev1.squaredNorm();
    float denom = s_norm_sq + r / p_cov;
    float K = 0.0f;
    if (s_norm_sq > 1e-6f && denom > 1e-6f) {
      K = std::min(s_norm_sq / denom, 0.5f);  // cap the gain
    }

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
   * @return s[n].x (i.e. aₓ band-pass filtered)
   */
  float getFilteredAx() const { return s_prev1.x(); }

  /**
   * @brief Returns the filtered Y-component of the signal.
   * @return s[n].y (i.e. aᵧ band-pass filtered)
   */
  float getFilteredAy() const { return s_prev1.y(); }

  /**
   * @brief Computes amplitude and phase using the direction of motion.
   *
   * Uses dynamic projection of s[n] and s[n-1] onto the estimated direction of oscillation.
   */
  std::pair<float, float> getAmplitudePhase(float delta_t) const {
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega_dt = std::acos(std::clamp(a / 2.0f, -1.0f, 1.0f));
    float sin_omega_dt = std::sin(omega_dt);
    float cos_omega_dt = std::cos(omega_dt);

    if (std::abs(sin_omega_dt) < 1e-6f)
      return {s_prev1.norm(), 0.0f};

    Eigen::Vector2f dir = s_prev1;
    float dir_norm = dir.norm();
    if (dir_norm < 1e-6f)
      return {0.0f, 0.0f};
    dir /= dir_norm;

    float s1 = s_prev1.dot(dir);
    float s2 = s_prev2.dot(dir);

    float q = (s1 - s2 * cos_omega_dt) / sin_omega_dt;
    float amplitude = std::sqrt(s1 * s1 + q * q);
    float phase = std::atan2(q, s1);

    return {amplitude, phase};
  }

  float getAmplitude(float delta_t) const {
    return getAmplitudePhase(delta_t).first;
  }

  float getPhase(float delta_t) const {
    return getAmplitudePhase(delta_t).second;
  }

  float getFrequency(float delta_t) const {
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega = std::acos(std::clamp(a / 2.0f, -1.0f, 1.0f));
    return (omega / delta_t) / (2.0f * M_PI);
  }
  
  /**
   * @brief Returns a confidence estimate (0..1) for frequency tracking stability.
   * 
   * This confidence is derived from the adaptive covariance `p_cov`:
   *   • When `p_cov` is small (stable estimate), confidence approaches 1.0.
   *   • When `p_cov` is large (uncertain estimate), confidence approaches 0.0.
   *
   * Useful for assessing how well the filter is locking onto a dominant frequency.
   *
   * @return Confidence value between 0 (low confidence) and 1 (high confidence).
   */
  float getTrackingConfidence() const {
    return 1.0f / (1.0f + p_cov);
  }

private:
  static constexpr float A_CLAMP = 1.9999f;

  int samples_processed = 0;

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

#ifdef KALMAN_2D_BANDPASS_TEST

void KalmanBandpass_test_signal(float t, float freq, float& ax, float& ay) {
  float amp = 0.2f; //1.0f + 0.5f * std::sin(0.1f * t);  // Slowly varying amplitude
  float phase = 2.0f * PI * freq * t;
  ax = amp * std::cos(phase);
  ay = 1.5f * ax;  
}

void KalmanBandpass_test_1() {
  const float delta_t = 0.005f;  // 200 Hz sample rate
  const float freq = 0.5f;       // Base frequency (Hz)
  const int num_steps = 10000;

  BlendedKalman2DBandPass filter(0.99f, 0.001f, 0.1f, 0.95f);
  filter.setFrequencyEstimate(freq, delta_t);

  std::ofstream out("bandpass.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence\n";

  for (int i = 0; i < num_steps; ++i) {
    float t = i * delta_t;
    float ax, ay;
    KalmanBandpass_test_signal(t, freq, ax, ay);

    float magnitude = filter.process(ax, ay, freq, delta_t);
    float filtered_ax = filter.getFilteredAx();
    float filtered_ay = filter.getFilteredAy();
    float frequency = filter.getFrequency(delta_t);
    float amplitude = filter.getAmplitude(delta_t);
    float phase = filter.getPhase(delta_t);
    float confidence = filter.getTrackingConfidence();

    out << t << "," << ax << "," << ay << "," << filtered_ax << "," << filtered_ay << ","
        << frequency << "," << amplitude << "," << phase << "," << confidence << "\n";
  }
  out.close();
}

#endif

#endif // BLENDED_KALMAN_2D_BANDPASS_H
