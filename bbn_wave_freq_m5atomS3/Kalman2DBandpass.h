#ifndef KALMAN_2D_BANDPASS_H
#define KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>
#include <complex>

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
        A = Eigen::Vector2cd::Zero();
        P = Eigen::Matrix2cd::Identity() * 1.0;
        updatePhaseModel(deltaT);
        phase = 0.0f;
        confidence = 0.0f;
    }

    void update(float ax, float ay, float currentOmega, float deltaT) {
        if (std::fabs(currentOmega - omega) > 0.01f * omega) {
            omega = currentOmega;
            updatePhaseModel(deltaT);
        }

        // Update phase
        phase = std::fmod(phase + omega * deltaT, 2.0f * M_PI);
        if (phase < 0.0f) phase += 2.0f * M_PI;

        // Observation as real signal vector
        Eigen::Vector2f a_t(ax, ay);

        // Rotate back using e^{-j φ}
        std::complex<float> phasor = std::exp(std::complex<float>(0.0f, -phase));
        Eigen::Vector2cd z;
        z(0) = std::complex<float>(a_t(0), 0.0f) * phasor;
        z(1) = std::complex<float>(a_t(1), 0.0f) * phasor;

        // Predict (static model)
        Eigen::Vector2cd A_pred = A;
        Eigen::Matrix2cd P_pred = P + Q;

        // Kalman gain
        Eigen::Matrix2cd S = P_pred + R;
        Eigen::Matrix2cd K = P_pred * S.ldlt().solve(Eigen::Matrix2cd::Identity());

        // Update
        A = A_pred + K * (z - A_pred);
        P = (Eigen::Matrix2cd::Identity() - K) * P_pred;

        // Confidence
        confidence = 1.0f / (P.real().trace());
    }

    // Output signal: a(t) = Re{A * e^{j φ(t)}}
    Eigen::Vector2f getFilteredSignal() const {
        std::complex<float> phasor = std::exp(std::complex<float>(0.0f, phase));
        Eigen::Vector2cd signal = A * phasor;
        return Eigen::Vector2f(signal(0).real(), signal(1).real());
    }

    float getAmplitude() const {
        return std::sqrt(std::norm(A(0)) + std::norm(A(1)));
    }

    Eigen::Vector2f getDirection() const {
        Eigen::Vector2f realVec = getAmplitudes();
        float norm = realVec.norm();
        return norm > 1e-6f ? realVec / norm : Eigen::Vector2f(1.0f, 0.0f);
    }

    Eigen::Vector2f getAmplitudes() const {
        return Eigen::Vector2f(A(0).real(), A(1).real());
    }

    float getPhase() const { return phase; }
    float getConfidence() const { return confidence; }

    void setProcessNoise(float q) {
        Q = Eigen::Matrix2cd::Identity() * q;
    }

    void setMeasurementNoise(float r) {
        R = Eigen::Matrix2cd::Identity() * r;
    }

private:
    void updatePhaseModel(float /*deltaT*/) {
        // phase model is handled explicitly
    }

    // State
    Eigen::Vector2cd A;
    Eigen::Matrix2cd P, Q = Eigen::Matrix2cd::Identity() * 1e-6f;
    Eigen::Matrix2cd R = Eigen::Matrix2cd::Identity() * 0.01f;

    float omega;
    float phase = 0.0f;
    float confidence = 0.0f;
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
