

#ifndef BLENDED_KALMAN_2D_BANDPASS_H
#define BLENDED_KALMAN_2D_BANDPASS_H

#include <ArduinoEigenDense.h>
#include <cmath>
#include <complex>
#include <algorithm>

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
        float phase = 0.0f;     // Phase of filtered output, in [-π, π]
        float frequency = 0.0f; // Hz, instantaneous from frequency parameter
        float confidence = 0.0f;
    };

    BlendedKalman2DBandPass(float rho = 0.975f,
                            float q = 1e-4f, float r = 0.01f,
                            float freq_q = 1e-6f, float freq_r = 0.01f)
        : rho(rho), rho_sq(rho * rho), q(q), r(r),
          freq_q(freq_q), freq_r(freq_r)
    {
        reset();
    }

    void reset() {
        z_prev1 = z_prev2 = cfloat(0.0f, 0.0f);

        A_est_vec = Eigen::Vector2f(1.0f, 0.0f);
        A_prev_vec = A_est_vec;

        P = Eigen::Matrix2f::Identity();
        Q = Eigen::Matrix2f::Identity() * q;
        R = Eigen::Matrix2f::Identity() * r;

        // Frequency parameter and covariance init (like KalmANF 'a' and p_cov)
        a_freq = 1.0f;
        p_cov_freq = 1.0f;

        omega = 2.0f * M_PI * 0.3f;
        theta = 0.0f;
    }

    void setInitialFrequency(float freq_hz, float delta_t) {
        omega = 2.0f * M_PI * freq_hz;
        a_freq = 2.0f * std::cos(omega * delta_t);
        p_cov_freq = 1.0f;
        theta = 0.0f;
    }

    /// Step 1: Resonator-only
    cfloat stepResonator(const Eigen::Vector2f& input_xy, float delta_t) {
        cfloat input(input_xy.x(), input_xy.y());
        cfloat a_complex = std::polar(1.0f, omega * delta_t);
        cfloat z = input + rho * a_complex * z_prev1 - rho_sq * z_prev2;

        z_prev2 = z_prev1;
        z_prev1 = z;

        return z;
    }

    /// Step 2: Kalman amplitude + frequency update
    Output stepKalman(const cfloat& z, float delta_t) {
        // Update oscillator phase wrapped to [-π, π]
        theta += omega * delta_t;
        theta = std::fmod(theta + M_PI, 2.0f * M_PI);
        if (theta < 0) theta += 2.0f * M_PI;
        theta -= M_PI;

        // Measurement matrix H: 2D rotation by theta
        Eigen::Matrix2f H;
        float c = std::cos(theta);
        float s = std::sin(theta);
        H << c, -s,
             s,  c;

        Eigen::Vector2f z_vec(z.real(), z.imag());
        Eigen::Vector2f pred = H * A_est_vec;
        Eigen::Vector2f innovation = z_vec - pred;
        Eigen::Matrix2f S = H * P * H.transpose() + R;
        Eigen::Matrix2f K = P * H.transpose() * S.inverse();

        A_prev_vec = A_est_vec;
        A_est_vec += K * innovation;

        Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

        // --- Frequency scalar Kalman inspired by KalmANF ---
        float s_prev1 = z_prev1.real();
        float s_prev2 = z_prev2.real();

        // Prediction step for frequency covariance
        p_cov_freq += freq_q;

        // Kalman gain for frequency parameter
        float denom = s_prev1 * s_prev1 + freq_r / p_cov_freq;
        float K_freq = s_prev1 / denom;

        // Innovation for frequency parameter (scalar)
        float e_freq = s_prev1 * a_freq - s_prev2 - z.real();

        // Update frequency parameter
        a_freq += K_freq * e_freq;

        // Clamp frequency parameter to valid range for acos()
        a_freq = std::clamp(a_freq, -1.9999f, 1.9999f);

        // Update frequency covariance
        p_cov_freq = (1.0f - K_freq * s_prev1) * p_cov_freq;

        // Update frequency from frequency parameter
        float omega_hat = std::acos(a_freq / 2.0f);
        omega = omega_hat / delta_t;

        // Prepare output
        Output out;
        std::complex<float> filtered = std::complex<float>(A_est_vec.x(), A_est_vec.y()) * std::polar(1.0f, theta);
        out.filtered_xy = Eigen::Vector2f(filtered.real(), filtered.imag());
        out.amplitude = std::abs(filtered);
        out.phase = std::arg(filtered);
        out.frequency = omega / (2.0f * M_PI);

        float trace_P = P.trace();
        float trace_R = R.trace();
        out.confidence = 1.0f - (trace_P / (trace_P + trace_R));

        return out;
    }

    /// Convenience combined step
    Output process(const Eigen::Vector2f& input_xy, float delta_t) {
        cfloat z = stepResonator(input_xy, delta_t);
        return stepKalman(z, delta_t);
    }

private:
    // Resonator state
    float rho, rho_sq;
    cfloat z_prev1, z_prev2;

    // Kalman amplitude state
    Eigen::Vector2f A_est_vec, A_prev_vec;
    Eigen::Matrix2f P, Q, R;
    float q, r;

    // Frequency scalar Kalman state
    float a_freq;
    float p_cov_freq;
    float freq_q;
    float freq_r;

    // Oscillator tracking
    float omega;
    float theta;
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

  BlendedKalman2DBandPass filt(0.975f, 0.001f, 0.1f, 1e-6f, 0.01f);
  filt.setInitialFrequency(freq, dt);

  std::ofstream out("bandpass.csv");
  out << "t,ax,ay,filtered_ax,filtered_ay,frequency,amplitude,phase,confidence\n";
  for (int i = 0; i < N; ++i) {
    float t = i * dt;
    float ax, ay;
    KalmanBandpass_test_signal(t, freq, ax, ay);

    Eigen::Vector2f input_xy(ax, ay);
    BlendedKalman2DBandPass::Output output = filt.process(input_xy, dt);

    out << t << "," 
        << ax << "," << ay << ","
        << output.filtered_xy.x() << "," << output.filtered_xy.y() << ","
        << output.frequency << "," << output.amplitude << "," << output.phase << "," << output.confidence << "\n";
  }
  out.close();
}

#endif

#endif // BLENDED_KALMAN_2D_BANDPASS_H

