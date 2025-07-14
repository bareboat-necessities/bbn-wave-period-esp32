
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
 * @brief 2D Kalman-adaptive band-pass filter with quadrature resonator for accurate amplitude/phase estimation.
 *
 * This version uses two resonators (in-phase and quadrature) to properly track amplitude and phase.
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
  BlendedKalman2DBandPass(float rho = 0.985f, float q = 0.001f, float r = 0.1f, float alpha_blend = 0.95f)
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
    q_prev1.setZero();  // Quadrature component state 1
    q_prev2.setZero();  // Quadrature component state 2
    a_prev = 2.0f * std::cos(2.0f * M_PI * 0.3f * 0.005f);
    p_cov = 1.0f;
    samples_processed = 0;
  }

  /**
   * @brief Sets the initial frequency estimate (in Hz) before filtering begins.
   */
  void setFrequencyEstimate(float freq_hz, float delta_t) {
    float omega_dt = 2.0f * M_PI * freq_hz * delta_t;
    float a = 2.0f * std::cos(omega_dt);
    a_prev = std::clamp(a, -A_CLAMP, A_CLAMP);
  }

  /**
   * @brief Processes a new 2D signal sample and updates the internal state.
   */
  float process(float a_x, float a_y, float freq_est_hz, float delta_t) {
    Eigen::Vector2f y(a_x, a_y);
    float omega_dt = 2.0f * M_PI * freq_est_hz * delta_t;
    float a_est = 2.0f * std::cos(omega_dt);

    if (samples_processed == 0) {
      // Initialize resonator states to match expected oscillation
      setFrequencyEstimate(freq_est_hz, delta_t);
      float amp = y.norm();
      if (amp < 1e-6f) amp = 1e-6f;
      Eigen::Vector2f dir = y.normalized();      
      s_prev1 = dir * amp; 
      q_prev1 = Eigen::Vector2f(-dir.y() , dir.x()) * amp; 
      s_prev2 = dir * (amp * std::cos(2.0f * omega_dt));
      q_prev2 = dir * (amp * std::sin(2.0f * omega_dt));
      plane_dir = s_prev1.normalized();
      plane_perp = Eigen::Vector2f(-plane_dir.y(), plane_dir.x());
      samples_processed = 1;
      return s_prev1.norm();
    }

    // Apply second-order resonator (in-phase component)
    Eigen::Vector2f s = (y + rho * a_prev * s_prev1 - rho_sq * s_prev2);
    
    // Apply second-order resonator (quadrature component)
    Eigen::Vector2f q = (y + rho * a_prev * q_prev1 - rho_sq * q_prev2);
    q = (q - q.dot(s.normalized()) * s.normalilzed()).normalized() * q.norm();

    // Predict covariance for Kalman gain
    p_cov = p_cov + this->q;
    
    // Compute combined Kalman gain using both components
    float denom = (s_prev1.dot(s_prev1) + q_prev1.dot(q_prev1)) * p_cov + r;
    Eigen::Vector2f K_s = (p_cov * s_prev1) / denom;
    Eigen::Vector2f K_q = (p_cov * q_prev1) / denom;

    // Residual for a update (using both components)
    Eigen::Vector2f e_s = (s - a_prev * s_prev1 + s_prev2);
    Eigen::Vector2f e_q = (q - a_prev * q_prev1 + q_prev2);

    // Combined Kalman update for resonance coefficient a
    float a_meas_s = a_prev + K_s.dot(e_s);
    float a_meas_q = a_prev + K_q.dot(e_q);
    float a_meas = (a_meas_s + a_meas_q) * 0.5f;  // Average both estimates
    a_meas = std::clamp(a_meas, -A_CLAMP, A_CLAMP);

    // Smooth update
    a_prev = alpha * a_meas + (1.0f - alpha) * a_est;
    a_prev = std::clamp(a_prev, -A_CLAMP, A_CLAMP);

    // Update covariance (using average of both components)
    p_cov = std::max((1.0f - 0.5f * (K_s.dot(s_prev1) + K_q.dot(q_prev1))) * p_cov, 1e-6f);

    // Update resonator states
    s_prev2 = s_prev1;
    s_prev1 = s;
    q_prev2 = q_prev1;
    q_prev1 = q;

    // Return magnitude of in-phase component (could also return combined magnitude)
    return s.norm();
  }

  /**
   * @brief Returns the filtered X-component of the signal (in-phase component).
   */
  float getFilteredAx() const { return s_prev1.x() * (1.0f - rho_sq); }

  /**
   * @brief Returns the filtered Y-component of the signal (in-phase component).
   */
  float getFilteredAy() const { return s_prev1.y() * (1.0f - rho_sq); }

  /**
   * @brief Returns the quadrature X-component of the signal.
   */
  float getQuadratureAx() const { return q_prev1.x() * (1.0f - rho_sq); }

  /**
   * @brief Returns the quadrature Y-component of the signal.
   */
  float getQuadratureAy() const { return q_prev1.y() * (1.0f - rho_sq); }

  /**
   * @brief Computes amplitude and phase using both in-phase and quadrature components.
   */
  std::pair<float, float> getAmplitudePhase(float delta_t) const {

    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega = std::acos(a / 2.0f);
    float resonator_gain = (1.0f - rho_sq) / std::sqrt(
       (1.0f + rho_sq * rho_sq - 2.0f * rho_sq * std::cos(2.0f * omega)));
    
    Eigen::Vector2f dir = plane_dir;
    float I = s_prev1.dot(dir) * resonator_gain;
    float Q = q_prev1.dot(dir) * resonator_gain;
    float amplitude = std::sqrt(I * I + Q * Q);
    float phase = std::atan2(Q, I);
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
  
  float getTrackingConfidence() const {
    return 1.0f / (1.0f + p_cov);
  }

private:
  static constexpr float A_CLAMP = 1.99999f;

  int samples_processed = 0;

  float rho, rho_sq;
  float q, r, alpha;

  float a_prev;     // Previous resonance coefficient
  float p_cov;      // Covariance for adaptive update

  // In-phase component states
  Eigen::Vector2f s_prev1;
  Eigen::Vector2f s_prev2;

  // Quadrature component states
  Eigen::Vector2f q_prev1;
  Eigen::Vector2f q_prev2;

  Eigen::Vector2f plane_dir, plane_perp;
};

#ifdef KALMAN_2D_BANDPASS_TEST

void KalmanBandpass_test_signal(float t, float freq, float& ax, float& ay) {
  float amp = 0.2f; //1.0f + 0.5f * std::sin(0.1f * t);  // Slowly varying amplitude
  float phase = 2.0f * PI * freq * t;
  Eigen::Vector2f dir(1.0f, 1.5f);
  dir.normalize();
  float signal = amp * std::cos(phase);
  ax = signal * dir.x();
  ay = signal * dir.y();
}

void KalmanBandpass_test_1() {
  const float delta_t = 0.005f;  // 200 Hz sample rate
  const float freq = 0.5f;       // Base frequency (Hz)
  const int num_steps = 10000;

  BlendedKalman2DBandPass filter(0.985f, 0.001f, 0.1f, 1.0f);
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
