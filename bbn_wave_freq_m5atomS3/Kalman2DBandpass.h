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
        : omega(initialOmega) {
        reset(deltaT);
    }

    void reset(float deltaT) {
        xEst.setZero();
        pEst = Eigen::Matrix2f::Identity() * 1.0f;
        updatePhase(deltaT);
        confidence = 0.0f;
    }

    void update(float ax, float ay, float currentOmega, float deltaT) {
        if (std::fabs(currentOmega - omega) > 0.01f * omega) {
            omega = currentOmega;
        }

        // Advance phase
        updatePhase(deltaT);

        float s = std::cos(phase);
        Eigen::Matrix2f H = s * Eigen::Matrix2f::Identity();

        // Predict
        Eigen::Vector2f xPred = xEst;
        Eigen::Matrix2f pPred = pEst + Q;

        // Kalman Gain
        Eigen::Matrix2f S = H * pPred * H.transpose() + R;
        Eigen::Matrix2f K = pPred * H.transpose() * S.ldlt().solve(Eigen::Matrix2f::Identity());

        // Update
        Eigen::Vector2f z(ax, ay);
        xEst = xPred + K * (z - H * xPred);
        pEst = (Eigen::Matrix2f::Identity() - K * H) * pPred;

        confidence = 1.0f / (pEst(0, 0) + pEst(1, 1));
    }

    Eigen::Vector2f getAmplitudes() const { return xEst; }
    float getAmplitude() const { return xEst.norm(); }
    float getPhase() const { return phase; }
    float getConfidence() const { return confidence; }
    Eigen::Vector2f getDirection() const {
        float norm = xEst.norm();
        return norm > 1e-6f ? xEst / norm : Eigen::Vector2f(1.0f, 0.0f);
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

    float omega;
    float phase = 0.0f;
    float confidence = 0.0f;

    Eigen::Vector2f xEst = Eigen::Vector2f::Zero();
    Eigen::Matrix2f pEst = Eigen::Matrix2f::Identity();
    Eigen::Matrix2f Q = Eigen::Matrix2f::Identity() * 1e-6f;
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.01f;
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
    float filtered_ax = filter.getAmplitudes().x();
    float filtered_ay = filter.getAmplitudes().y();
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
