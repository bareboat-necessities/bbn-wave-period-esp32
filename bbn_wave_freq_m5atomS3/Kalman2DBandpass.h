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
        xEst << 0.0f, 0.0f;
        pEst << 1.0f, 0.0f,
                0.0f, 1.0f;
        updateProcessModel(deltaT);
        confidence = 0.0f;
        phase = 0.0f;
    }

    void update(float ax, float ay, float currentOmega, float deltaT) {
        // Update omega and model if changed significantly
        if (std::fabs(currentOmega - omega) > 0.01f * omega) {
            omega = currentOmega;
            updateProcessModel(deltaT);
        }

        // Advance phase and wrap to [0, 2π)
        phase = std::fmod(phase + omega * deltaT, 2.0f * M_PI);
        if (phase < 0.0f) phase += 2.0f * M_PI;  // ensure positive phase

        float sinPhase = std::sin(phase);
        constexpr float sinThreshold = 0.05f;

        // Predict
        xPred = F * xEst;
        pPred = F * pEst * F.transpose() + Q;

        if (std::fabs(sinPhase) > sinThreshold && (ax * ax + ay * ay) > 1e-4f) {
            Eigen::Vector2f z(ax / sinPhase, ay / sinPhase);

            // Kalman update
            K = pPred * (pPred + R).inverse();
            xEst = xPred + K * (z - xPred);
            pEst = (Eigen::Matrix2f::Identity() - K) * pPred;

            confidence = 1.0f / (pEst(0, 0) + pEst(1, 1));
        } else {
            xEst = xPred;
            pEst = pPred;
            confidence *= 0.95f;
        }
    }

    Eigen::Vector2f getAmplitudes() const { return xEst; }
    float getAmplitude() const { return xEst.norm(); }
    float getPhase() const { return phase; }
    float getConfidence() const { return confidence; }
    Eigen::Vector2f getDirection() const { return xEst.normalized(); }

    void setProcessNoise(float q) {
        Q << q, 0.0f,
             0.0f, q;
    }

    void setMeasurementNoise(float r) {
        R << r, 0.0f,
             0.0f, r;
    }

private:
    void updateProcessModel(float deltaT) {
        float c = std::cos(omega * deltaT);
        float s = std::sin(omega * deltaT);
        F << c, s / omega,
            -omega * s, c;
    }

    // Configuration
    float omega;
    float phase = 0.0f;
    float confidence = 0.0f;

    // Kalman state
    Eigen::Vector2f xEst, xPred;
    Eigen::Matrix2f pEst, pPred, Q = Eigen::Matrix2f::Identity() * 1e-6f;
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.1f;
    Eigen::Matrix2f K, F;
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
  filter.setMeasurementNoise(0.03f);
  filter.setProcessNoise(0.01f);

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
