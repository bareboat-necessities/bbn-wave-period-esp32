#ifndef KALMAN_WAVE_DIRECTION_H
#define KALMAN_WAVE_DIRECTION_H

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter for estimating direction of an ocean wave from IMU horizontal x, y accelerations.

 */

#ifdef EIGEN_NON_ARDUINO
#include <Eigen/Dense>
#else
#include <ArduinoEigenDense.h>
#endif
#include <cmath>

#ifdef KALMAN_WAVE_DIRECTION_TEST
#include <iostream>
#include <fstream>
#include <random>
#endif

class KalmanWaveDirection {
public:
    KalmanWaveDirection(float initialOmega, float deltaT)
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
        if (std::fabs(c) < 0.001f) {
            P += Q;
            confidence = 1.0f / (P.trace() + 1e-6f);
            return;
        }
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
    
        // Joseph-form covariance update
        Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        Eigen::Matrix2f KH = K * H;
        P = (I - KH) * P_pred * (I - KH).transpose() + K * R * K.transpose();

        confidence = 1.0f / (P.trace() + 1e-6f);
    }

    // Estimated wave propagation direction (unit vector)
    Eigen::Vector2f getDirection() const {
        float norm = A_est.norm();
        const float AMP_THRESHOLD = 0.08f;
        const float CONFIDENCE_THRESHOLD = 20.0f;
        if (norm > AMP_THRESHOLD && confidence > CONFIDENCE_THRESHOLD) {
          Eigen::Vector2f newDir = A_est / norm;
          if (lastStableDir.dot(newDir) < 0.0f) {
            newDir = -newDir;
          } 
          float alpha = 0.05f;
          lastStableDir = ((1.0f - alpha) * lastStableDir + alpha * newDir).normalized();
        }
        return lastStableDir;
    }

    float getDirectionDegrees() const {
        Eigen::Vector2f dir = getDirection();
        float deg = std::atan2(dir.y(), dir.x()) * (180.0f / M_PI);
        if (deg < 0.0f) deg += 180.0f;
        if (deg >= 180.0f) deg -= 180.0f;
        return deg;
    }

    Eigen::Vector2f getFilteredSignal() const {
        return A_est * std::cos(phase) + Eigen::Vector2f(-A_est.y(), A_est.x()) * std::sin(phase);
    }

    Eigen::Vector2f getOscillationAlongDirection() const {
        return A_est * std::cos(phase);
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
        phase = std::remainder(phase + omega * deltaT, 2.0f * M_PI);
    }

    // State
    Eigen::Vector2f A_est = Eigen::Vector2f::Zero();
    Eigen::Matrix2f P = Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Q = Eigen::Matrix2f::Identity() * 1e-6f;
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.01f;

    float omega;
    float phase;
    float confidence;

    mutable Eigen::Vector2f lastStableDir = Eigen::Vector2f(1.0f, 0.0f);
};

#ifdef KALMAN_WAVE_DIRECTION_TEST

void KalmanWaveDirection_test_signal(float t, float freq, float& ax, float& ay, 
    std::normal_distribution<float>& noise, std::default_random_engine& generator) {
  float amp = 0.8f + 0.4f * std::sin(0.005f * t);  // Slowly varying amplitude
  float phase = 2.0f * M_PI * freq * t;
  Eigen::Vector2f dir(1.0f, 1.5f);
  dir.normalize();
  float signal = amp * std::cos(phase);
  float w1 = noise(generator);
  float w2 = noise(generator);
  ax = signal * dir.x() + w1;
  ay = signal * dir.y() + w2;
}

void KalmanWaveDirection_test_1() {
  const float delta_t = 0.02f;   // 50 Hz sample rate
  const float freq = 0.5f;       // Base frequency (Hz)
  const int num_steps = 2500;

  const double mean = 0.0f;     // m/s^2
  const double stddev = 0.08f;  // m/s^2
  std::default_random_engine generator;
  generator.seed(239);  // seed the engine for deterministic test results
  std::normal_distribution<float> dist(mean, stddev);
  
  KalmanWaveDirection filter(freq, delta_t);
  filter.setMeasurementNoise(0.01f);
  filter.setProcessNoise(1e-6f);

  std::ofstream out("wave_dir.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence,deg\n";

  for (int i = 0; i < num_steps; ++i) {
    float t = i * delta_t;
    float ax, ay;
    KalmanWaveDirection_test_signal(t, freq, ax, ay, dist, generator);

    filter.update(ax, ay, freq, delta_t);
    float filtered_ax = filter.getFilteredSignal().x();
    float filtered_ay = filter.getFilteredSignal().y();
    float frequency = freq;
    float amplitude = filter.getAmplitude();
    float phase = filter.getPhase();
    float confidence = filter.getConfidence();
    float deg = filter.getDirectionDegrees();

    out << t << "," << ax << "," << ay << "," << filtered_ax << "," << filtered_ay << ","
        << frequency << "," << amplitude << "," << phase << "," << confidence << "," << deg << "\n";
  }
  out.close();
}

#endif

#endif  // KALMAN_WAVE_DIRECTION_H
