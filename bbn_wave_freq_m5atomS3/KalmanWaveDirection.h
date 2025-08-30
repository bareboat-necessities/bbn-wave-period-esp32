
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
#include <algorithm> 

#ifdef KALMAN_WAVE_DIRECTION_TEST
#include <iostream>
#include <fstream>
#include <random>
#endif

class KalmanWaveDirection {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KalmanWaveDirection(float initialOmega, float deltaT)
        : omega(initialOmega), phase(0.0f) {
        reset(deltaT);
    }

    void reset(float deltaT) {
        A_est.setZero();
        P = Eigen::Matrix2f::Identity() * 1.0f;
        updatePhase(deltaT);
        confidence = 0.0f;
        lastStableConfidence = 0.0f;
        lastStableCovariance = Eigen::Matrix2f::Identity();
        lastStableDir = Eigen::Vector2f(1.0f, 0.0f);
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
            P = 0.5f * (P + P.transpose());
            // Update confidence conservatively
            confidence = 1.0f / (P.trace() + 1e-6f);
            confidence *= 0.98f; // Decay to avoid stale confidence holding steady
            return;
        }
        Eigen::Matrix2f H = c * Eigen::Matrix2f::Identity();
          
        // Predict
        Eigen::Vector2f A_pred = A_est;
        Eigen::Matrix2f P_pred = P + Q;

        // Regularize
        P_pred = 0.5f * (P_pred + P_pred.transpose());
        P_pred += Eigen::Matrix2f::Identity() * 1e-10f;

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
            lastStableAmplitude = norm;

            // Track last stable confidence and covariance
            lastStableConfidence = confidence;
            lastStableCovariance = P;
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

    // Returns angular uncertainty in degrees at 95% confidence (~2σ)
    // This estimates the maximum deviation of the wave direction, assuming a Gaussian distribution
    // of the amplitude vector's components and projecting the error covariance onto the direction tangent.
    float getDirectionUncertaintyDegrees() const {
        // Compute amplitude (magnitude of the estimated direction vector)
        float amp = lastStableAmplitude;
    
        // If amplitude is too small, direction is meaningless — return full uncertainty
        if (amp < 1e-6f) return 180.0f;
    
        // Unit direction vector (estimated wave direction)
        Eigen::Vector2f dir = A_est / amp;
    
        // Tangent direction (perpendicular to dir) is where angular deviations occur
        Eigen::Vector2f tangent(-dir.y(), dir.x());
    
        // Project covariance matrix onto the tangent vector
        // This gives the variance of noise in the angular direction
        float angular_var = tangent.transpose() * lastStableCovariance * tangent;
    
        // Angular standard deviation in radians (scaled by amplitude)
        float angular_std_rad = std::sqrt(angular_var) / amp;
    
        // Convert to 2σ uncertainty in degrees (95% confidence)
        float angle_rad = 2.0f * angular_std_rad;
        float angle_deg = angle_rad * (180.0f / M_PI);
    
        // Clamp to [0, 180] degrees for safety
        return std::max(0.0f, std::min(angle_deg, 180.0f));
    }

    float getLastStableConfidence() const {
        return lastStableConfidence;
    }

    Eigen::Vector2f getFilteredSignal() const {
        return A_est * std::cos(phase) + Eigen::Vector2f(-A_est.y(), A_est.x()) * std::sin(phase);
    }

    Eigen::Vector2f getOscillationAlongDirection() const {
        return A_est * std::cos(phase);
    }

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
    mutable float lastStableAmplitude = 0.0f;
    mutable float lastStableConfidence = 0.0f;
    mutable Eigen::Matrix2f lastStableCovariance = Eigen::Matrix2f::Identity();
};

#ifdef KALMAN_WAVE_DIRECTION_TEST

void KalmanWaveDirection_test_signal(float t, float freq, float& ax, float& ay, 
    std::normal_distribution<float>& noise, std::default_random_engine& generator) {
  float amp = 0.8f + 0.4f * std::sin(0.005f * t);
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
  const float delta_t = 0.02f;
  const float freq = 0.5f;
  const int num_steps = 2000;

  const double mean = 0.0f;
  const double stddev = 0.08f;
  std::default_random_engine generator;
  generator.seed(239);
  std::normal_distribution<float> dist(mean, stddev);
  
  KalmanWaveDirection filter(freq, delta_t);
  filter.setMeasurementNoise(0.01f);
  filter.setProcessNoise(1e-6f);

  std::ofstream out("wave_dir.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence,deg,uncertaintyDeg\n";

  for (int i = 0; i < num_steps; ++i) {
    float t = i * delta_t;
    float ax, ay;
    KalmanWaveDirection_test_signal(t, freq, ax, ay, dist, generator);

    filter.update(ax, ay, freq, delta_t);
    float filtered_ax = filter.getFilteredSignal().x();
    float filtered_ay = filter.getFilteredSignal().y();
    float amplitude = filter.getAmplitude();
    float phase = filter.getPhase();
    float confidence = filter.getConfidence();
    float deg = filter.getDirectionDegrees();
    float uncertaintyDeg = filter.getDirectionUncertaintyDegrees();

    out << t << "," << ax << "," << ay << "," << filtered_ax << "," << filtered_ay << ","
        << freq << "," << amplitude << "," << phase << "," << confidence << ","
        << deg << "," << uncertaintyDeg << "\n";
  }
  out.close();
}

#endif  // KALMAN_WAVE_DIRECTION_TEST

#endif  // KALMAN_WAVE_DIRECTION_H

