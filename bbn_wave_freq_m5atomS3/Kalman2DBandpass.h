#ifndef KALMAN_2D_BANDPASS_H
#define KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>

#ifdef KALMAN_2D_BANDPASS_TEST
#include <iostream>
#include <fstream>
#endif

class Kalman2DBandpass {
public:
    Kalman2DBandpass(float initialOmega, float deltaT)
        : omega(initialOmega), phase(0.0f) {
        reset(deltaT);
    }

    void reset(float deltaT) {
        A_est.setZero();
        P = Eigen::Matrix2f::Identity() * 1.0f;
        updatePhase(deltaT);
        confidence = 0.0f;
    }

    void update(float ax, float ay, float currentOmega, float deltaT) {
        if (std::fabs(currentOmega - omega) > 0.01f * omega) {
            omega = currentOmega;
        }

        // Advance phase
        updatePhase(deltaT);

        float c = std::cos(phase);
        Eigen::Matrix2f H = c * Eigen::Matrix2f::Identity();

        // Predict
        Eigen::Vector2f A_pred = A_est;
        Eigen::Matrix2f P_pred = P + Q;

        // Kalman gain
        Eigen::Matrix2f S = H * P_pred * H.transpose() + R;
        Eigen::Matrix2f K = P_pred * H.transpose() * S.ldlt().solve(Eigen::Matrix2f::Identity());

        // Measurement
        Eigen::Vector2f z(ax, ay);

        // Update state
        A_est = A_pred + K * (z - H * A_pred);
        P = (Eigen::Matrix2f::Identity() - K * H) * P_pred;

        confidence = 1.0f / (P.trace() + 1e-6f);
    }

    // Estimated wave propagation direction (unit vector)
    Eigen::Vector2f getDirection() const {
        float norm = A_est.norm();
        return (norm > 1e-6f) ? A_est / norm : Eigen::Vector2f(1.0f, 0.0f);
    }

    // Full amplitude vector A * dir
    Eigen::Vector2f getAmplitudeVector() const {
        return A_est;
    }

    float getAmplitude() const {
        return A_est.norm();
    }

    float getPhase() const {
        return phase;
    }

    float getConfidence() const {
        return confidence;
    }

    void setProcessNoise(float q) {
        Q = Eigen::Matrix2f::Identity() * q;
    }

    void setMeasurementNoise(float r) {
        R = Eigen::Matrix2f::Identity() * r;
    }

private:
    void updatePhase(float deltaT) {
        phase = std::fmod(phase + omega * deltaT, 2.0f * M_PI);
        if (phase < 0.0f) phase += 2.0f * M_PI;
    }

    // State
    Eigen::Vector2f A_est = Eigen::Vector2f::Zero();
    Eigen::Matrix2f P = Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Q = Eigen::Matrix2f::Identity() * 1e-6f;
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.01f;

    float omega;
    float phase;
    float confidence;
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

  Kalman2DBandpass filter(freq, delta_t);
  filter.setMeasurementNoise(0.01f);
  filter.setProcessNoise(1e-6f);

  std::ofstream out("bandpass.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence\n";

  for (int i = 0; i < num_steps; ++i) {
    float t = i * delta_t;
    float ax, ay;
    KalmanBandpass_test_signal(t, freq, ax, ay);

    filter.update(ax, ay, freq, delta_t);
    float filtered_ax = filter.getAmplitudeVector().x();
    float filtered_ay = filter.getAmplitudeVector().y();
    float frequency = freq;
    float amplitude = filter.getAmplitude();
    float phase = filter.getPhase();
    float confidence = filter.getConfidence();

    out << t << "," << ax << "," << ay << "," << filtered_ax << "," << filtered_ay << ","
        << frequency << "," << amplitude << "," << phase << "," << confidence << "\n";
  }
  out.close();
}

#endif

#endif  // KALMAN_2D_BANDPASS_H
