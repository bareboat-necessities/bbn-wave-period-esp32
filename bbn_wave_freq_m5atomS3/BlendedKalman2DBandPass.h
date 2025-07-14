#ifndef BLENDED_KALMAN_2D_BANDPASS_H
#define BLENDED_KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>
#include <complex>
#include <algorithm>
#include <utility>

#ifdef KALMAN_2D_BANDPASS_TEST
  #include <iostream>
  #include <fstream>
#endif

class BlendedKalman2DBandPass {
public:
    using cfloat = std::complex<float>;

    struct Output {
        Eigen::Vector2f filtered_xy;
        float amplitude = 0.0f;
        float phase = 0.0f;     // Phase of A_est, in [-π, π]
        float frequency = 0.0f; // Hz, instantaneous from phase drift
        float confidence = 0.0f;
    };

    BlendedKalman2DBandPass(float rho = 0.985f, float q = 1e-5f, float r = 0.01f)
        : rho(rho), rho_sq(rho * rho), q(q), r(r)
    {
        reset();
    }

    void reset() {
        z_prev1 = z_prev2 = cfloat(0.0f, 0.0f);
        A_est = cfloat(1.0f, 0.0f);
        A_prev = A_est;
        p_cov = 1.0f;
        omega = 2.0f * M_PI * 0.3f; // rad/s
        theta = 0.0f;
    }

    void setInitialFrequency(float freq_hz, float delta_t) {
        omega = 2.0f * M_PI * freq_hz;
        theta = 0.0f;
    }

    /// Step 1: Resonator-only
    cfloat stepResonator(const Eigen::Vector2f& input_xy, float delta_t) {
        cfloat input(input_xy.x(), input_xy.y());

        // Resonator: z[n] = x[n] + ρa z[n-1] - ρ² z[n-2]
        cfloat a_complex = std::polar(1.0f, omega * delta_t);
        cfloat z = input + rho * a_complex * z_prev1 - rho_sq * z_prev2;

        z_prev2 = z_prev1;
        z_prev1 = z;

        return z;
    }

    /// Step 2: Kalman + instantaneous frequency estimation
    Output stepKalman(const cfloat& z, float delta_t) {
        theta += omega * delta_t;
        if (theta > M_PI) theta -= 2.0f * M_PI;
        else if (theta < -M_PI) theta += 2.0f * M_PI;

        cfloat h = std::polar(1.0f, theta); // e^(jθ)
        cfloat prediction = A_est * h;
        cfloat innovation = z - prediction;

        float denom = p_cov + r;
        float K = p_cov / denom;

        A_prev = A_est;
        A_est += K * std::conj(h) * innovation;

        p_cov = std::max((1.0f - K) * p_cov + q, 1e-6f);

        // Instantaneous frequency from phase drift
        cfloat ratio = A_est / A_prev;
        float dphi = std::arg(ratio);
        float freq_hz;
        if (std::abs(A_prev) < 1e-4f) {
          freq_hz = omega / (2.0f * M_PI);
        } else {
          freq_hz = dphi / (2.0f * M_PI * delta_t);
        }
      
        // Update internal oscillator
        omega = 2.0f * M_PI * freq_hz;

        // Output
        Output out;
        cfloat filtered = A_est * h;
        out.filtered_xy = Eigen::Vector2f(filtered.real(), filtered.imag());
        out.amplitude = std::abs(A_est);
        out.phase = std::arg(A_est);
        out.frequency = freq_hz;
        out.confidence = 1.0f - (p_cov / (p_cov + r));

        return out;
    }

    /// Optional combined one-step process
    Output process(const Eigen::Vector2f& input_xy, float delta_t) {
        cfloat z = stepResonator(input_xy, delta_t);
        return stepKalman(z, delta_t);
    }

private:
    // Resonator state
    float rho, rho_sq;
    cfloat z_prev1, z_prev2;

    // Kalman state
    cfloat A_est, A_prev;
    float p_cov;
    float q, r;

    // Frequency tracking
    float omega;   // rad/s
    float theta;   // accumulated phase
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

  BlendedKalman2DBandPass filt(0.985f, 0.001f, 0.1f);
  filt.setInitialFrequency(freq, dt);

  std::ofstream out("bandpass.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence\n";
  for (int i = 0; i < N; ++i) {
    float t = i * dt;
    float ax, ay;
    KalmanBandpass_test_signal(t, freq, ax, ay);

    Eigen::Vector2f input_xy(ax, ay);
    BlendedKalman2DBandPass::Output output = filt.process(input_xy, dt);
    float f_ax = output.filtered_xy.x();
    float f_ay = output.filtered_xy.y();
    float f_fr = output.frequency;
    float f_am = output.amplitude;
    float f_ph = output.phase;
    float conf = output.confidence;

    out << t << "," 
        << ax << "," << ay << ","
        << f_ax << "," << f_ay << ","
        << f_fr << "," << f_am << "," << f_ph << "," << conf << "\n";
  }
  out.close();
}

#endif

#endif // BLENDED_KALMAN_2D_BANDPASS_H
