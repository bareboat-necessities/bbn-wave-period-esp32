

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
        float phase = 0.0f;     // Phase of A_est, in [-π, π]
        float frequency = 0.0f; // Hz, instantaneous from phase drift
        float confidence = 0.0f;
    };

    BlendedKalman2DBandPass(float rho = 0.985f, float q = 1e-4f, float r = 0.01f)
        : rho(rho), rho_sq(rho * rho), q(q), r(r)
    {
        reset();
    }

    void reset() {
        z_prev1 = z_prev2 = cfloat(0.0f, 0.0f);

        // Initialize state vector (real, imag)
        A_est_vec = Eigen::Vector2f(1.0f, 0.0f);
        A_prev_vec = A_est_vec;

        // Initialize covariance matrices
        P = Eigen::Matrix2f::Identity();
        Q = Eigen::Matrix2f::Identity() * q; // process noise covariance
        R = Eigen::Matrix2f::Identity() * r; // measurement noise covariance

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

    /// Step 2: Kalman + instantaneous frequency estimation (2D Kalman)
    Output stepKalman(const cfloat& z, float delta_t) {
        // Update theta with wrapping to [-π, π]
        theta += omega * delta_t;
        theta = std::fmod(theta + M_PI, 2.0f * M_PI);
        if (theta < 0) theta += 2.0f * M_PI;
        theta -= M_PI;

        // Measurement matrix H: rotation matrix by theta
        Eigen::Matrix2f H;
        float c = std::cos(theta);
        float s = std::sin(theta);
        H << c, -s,
             s,  c;

        // Convert measurement z to vector form
        Eigen::Vector2f z_vec(z.real(), z.imag());

        // Predicted measurement
        Eigen::Vector2f pred = H * A_est_vec;

        // Innovation (measurement residual)
        Eigen::Vector2f innovation = z_vec - pred;

        // Innovation covariance
        Eigen::Matrix2f S = H * P * H.transpose() + R;

        // Kalman gain
        Eigen::Matrix2f K = P * H.transpose() * S.inverse();

        // Save previous state for frequency calculation
        A_prev_vec = A_est_vec;

        // State update
        A_est_vec += K * innovation;

        // Covariance update (Joseph form for numerical stability)
        Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

        // Instantaneous frequency from phase difference
        std::complex<float> A_est_cplx(A_est_vec.x(), A_est_vec.y());
        std::complex<float> A_prev_cplx(A_prev_vec.x(), A_prev_vec.y());

        std::complex<float> ratio = A_est_cplx / A_prev_cplx;
        float dphi = std::arg(ratio);

        float freq_hz;
        if (std::abs(A_prev_cplx) < 1e-4f) {
            freq_hz = omega / (2.0f * M_PI);
        } else {
            freq_hz = dphi / (2.0f * M_PI * delta_t);
        }

        // Directly update frequency (no smoothing)
        omega = 2.0f * M_PI * freq_hz;

        // Output structure
        Output out;
        std::complex<float> filtered = A_est_cplx * std::polar(1.0f, theta);
        out.filtered_xy = Eigen::Vector2f(filtered.real(), filtered.imag());
        out.amplitude = std::abs(A_est_cplx);
        out.phase = std::arg(A_est_cplx);
        out.frequency = freq_hz;

        // Confidence metric based on covariance trace
        float trace_P = P.trace();
        float trace_R = R.trace();
        out.confidence = 1.0f - (trace_P / (trace_P + trace_R));

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

    // Kalman state: complex state as 2D vector + covariance matrix
    Eigen::Vector2f A_est_vec, A_prev_vec;
    Eigen::Matrix2f P;  // covariance matrix
    Eigen::Matrix2f Q;  // process noise covariance
    Eigen::Matrix2f R;  // measurement noise covariance

    // Noise parameters
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

  BlendedKalman2DBandPass filt(0.975f, 0.001f, 0.01f);
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

