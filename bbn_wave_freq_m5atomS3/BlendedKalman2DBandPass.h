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

class BlendedKalman2DBandPass {
public:
  BlendedKalman2DBandPass(float rho = 0.985f,
                          float processNoiseQ = 0.001f,
                          float measNoiseR    = 0.1f,
                          float alpha_blend   = 0.95f)
    : rho(rho),
      rho_sq(rho * rho),
      q(processNoiseQ),
      r(measNoiseR),
      alpha(alpha_blend)
  {
    reset();
  }

  void reset() {
    s_prev1.setZero();
    s_prev2.setZero();
    q_prev1.setZero();
    q_prev2.setZero();
    a_prev = 2.0f * std::cos(2.0f * M_PI * 0.3f * 0.005f);
    p_cov = 1.0f;
    samples_processed = 0;
  }

  void setFrequencyEstimate(float freq_hz, float delta_t) {
    float omega_dt = 2.0f * M_PI * freq_hz * delta_t;
    float a = 2.0f * std::cos(omega_dt);
    a_prev = std::clamp(a, -A_CLAMP, A_CLAMP);
  }

  // Process a new sample (ax, ay) with a frequency estimate;
  // returns the raw resonator in-phase magnitude (not normally used).
  float process(float a_x, float a_y, float freq_est_hz, float delta_t) {
    Eigen::Vector2f y(a_x, a_y);
    float omega_dt = 2.0f * M_PI * freq_est_hz * delta_t;
    float a_est    = 2.0f * std::cos(omega_dt);

    if (samples_processed == 0) {
      // --- Initialize the two resonators on the sampled plane
      float amp = y.norm() * (1.0f - rho_sq);
      amp = std::max(amp, 1e-6f);
      Eigen::Vector2f dir = y.normalized();
      plane_dir  = dir;
      plane_perp = Eigen::Vector2f(-dir.y(), dir.x());

      s_prev1 = dir * (amp * std::cos(omega_dt));
      q_prev1 = plane_perp * (amp * std::sin(omega_dt));
      s_prev2 = dir * (amp * std::cos(2.0f * omega_dt));
      q_prev2 = plane_perp * (amp * std::sin(2.0f * omega_dt));
      samples_processed = 1;
      return s_prev1.norm();
    }

    // --- Resonator updates
    Eigen::Vector2f s  = y + rho * a_prev * s_prev1 - rho_sq * s_prev2;
    Eigen::Vector2f qv = y + rho * a_prev * q_prev1 - rho_sq * q_prev2;

    // --- Kalman covariance predict
    p_cov += q;

    // --- Compute Kalman gain, clamp denominator
    float denom = (s_prev1.squaredNorm() + q_prev1.squaredNorm()) * p_cov + r;
    denom = std::max(denom, 1e-6f);
    Eigen::Vector2f K_s = (p_cov * s_prev1) / denom;
    Eigen::Vector2f K_q = (p_cov * q_prev1) / denom;

    // --- Compute residuals
    Eigen::Vector2f e_s = s  - a_prev * s_prev1 + s_prev2;
    Eigen::Vector2f e_q = qv - a_prev * q_prev1 + q_prev2;

    // --- Update resonance coefficient a_prev
    float a_meas_s = a_prev + K_s.dot(e_s);
    float a_meas_q = a_prev + K_q.dot(e_q);
    float a_meas   = 0.5f * (a_meas_s + a_meas_q);
    a_meas = std::clamp(a_meas, -A_CLAMP, A_CLAMP);

    // --- Blend and clamp
    a_prev = std::clamp(alpha * a_meas + (1.0f - alpha) * a_est, -A_CLAMP, A_CLAMP);

    // --- Covariance update
    float reduction = 0.5f * (K_s.dot(s_prev1) + K_q.dot(q_prev1));
    p_cov = std::max((1.0f - reduction) * p_cov, 1e-6f);

    // --- Shift states
    s_prev2 = s_prev1;   s_prev1 = s;
    q_prev2 = q_prev1;   q_prev1 = qv;

    return s.norm();
  }

  // Full reconstructed 2D signal from I/Q in the known plane.
  Eigen::Vector2f getFilteredSignal(float /*delta_t unused here*/) const {
    // Project onto plane to get in-phase (I) and quadrature (Q)
    float I = s_prev1.dot(plane_dir);
    float Q = q_prev1.dot(plane_perp);
    // Undo resonator damping factor (1 − rho²)
    float gain_inv = 1.0f / (1.0f - rho_sq);
    return (I * plane_dir + Q * plane_perp) * gain_inv;
  }

  float getFilteredAx(float dt) const { return getFilteredSignal(dt).x(); }
  float getFilteredAy(float dt) const { return getFilteredSignal(dt).y(); }

  // Compute analytic amplitude & phase by projecting and undoing damping
  std::pair<float,float> getAmplitudePhase(float /*delta_t unused here*/) const {
    // Clamp before acos
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float c = std::clamp(a * 0.5f, -1.0f, 1.0f);
    float omega = std::acos(c);

    // Project onto plane to get I & Q
    float I = s_prev1.dot(plane_dir);
    float Q = q_prev1.dot(plane_perp);

    // Undo resonator damping
    float amplitude = std::sqrt(I*I + Q*Q) / (1.0f - rho_sq);

    // Phase = angle between I and Q in that plane
    float dot   = s_prev1.dot(q_prev1);
    float cross = s_prev1.x() * q_prev1.y() - s_prev1.y() * q_prev1.x();
    float phase = std::atan2(Q, I); // std::atan2(cross, dot);

    return { amplitude, phase };
  }

  float getAmplitude(float dt) const { return getAmplitudePhase(dt).first; }
  float getPhase(float dt)     const { return getAmplitudePhase(dt).second; }

  // Instantaneous frequency from a_prev
  float getFrequency(float delta_t) const {
    float a = std::clamp(a_prev, -A_CLAMP, A_CLAMP);
    float omega = std::acos(a / 2.0f);
    return (omega / (2.0f * M_PI)) / delta_t; 
  }

  // [0..1] tracking confidence
  float getTrackingConfidence() const {
    return 1.0f / (1.0f + p_cov);
  }

private:
  static constexpr float A_CLAMP = 1.99999f;

  int samples_processed = 0;

  float rho, rho_sq;
  float q, r, alpha;
  float a_prev, p_cov;

  Eigen::Vector2f s_prev1, s_prev2;
  Eigen::Vector2f q_prev1, q_prev2;

  Eigen::Vector2f plane_dir, plane_perp;
};

#ifdef KALMAN_2D_BANDPASS_TEST

// Test signal: 0.2·cos(2π·0.5·t) in-plane [1 : 1.5]
void KalmanBandpass_test_signal(float t, float freq, float& ax, float& ay) {
  constexpr float amp = 0.2f;
  float phase = 2.0f * M_PI * freq * t;
  float s = amp * std::cos(phase);
  ax = s;  ay = s * 1.5f;
}

void KalmanBandpass_test_1() {
  const float dt = 0.005f;
  const float freq = 0.5f;
  const int N = 10000;

  BlendedKalman2DBandPass filt(0.985f, 0.001f, 0.1f, 1.0f);
  filt.setFrequencyEstimate(freq, dt);

  std::ofstream out("bandpass.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence\n";
  for (int i = 0; i < N; ++i) {
    float t = i * dt;
    float ax, ay;
    KalmanBandpass_test_signal(t, freq, ax, ay);

    filt.process(ax, ay, freq, dt);
    float f_ax = filt.getFilteredAx(dt);
    float f_ay = filt.getFilteredAy(dt);
    float f_fr = filt.getFrequency(dt);
    float f_am = filt.getAmplitude(dt);
    float f_ph = filt.getPhase(dt);
    float conf = filt.getTrackingConfidence();

    out << t << "," 
        << ax << "," << ay << ","
        << f_ax << "," << f_ay << ","
        << f_fr << "," << f_am << "," << f_ph << "," << conf << "\n";
  }
  out.close();
}

#endif

#endif // BLENDED_KALMAN_2D_BANDPASS_H
