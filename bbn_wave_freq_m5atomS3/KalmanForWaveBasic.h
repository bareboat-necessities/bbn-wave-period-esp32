#ifndef KalmanForWaveBasic_h
#define KalmanForWaveBasic_h

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter to double integrate vertical acceleration in wave
  into vertical displacement, correct for accelerometer bias,
  estimate accelerometer bias, correct integral for zero average displacement.
  The third integral (responsible for zero average vertical displacement)
  is taken as a measurement of zero.

  Process model:

  velocity:
  v(k) = v(k-1) + a*T - a_hat(k-1)*T

  displacement:
  y(k) = y(k-1) + v(k-1)*T + 1/2*a*T^2 - 1/2*a_hat(k-1)*T^2

  displacement_integral:
  z(k) = z(k-1) + y(k-1)*T + 1/2*v(k-1)*T^2 + 1/6*a*T^3 - 1/6*a_hat(k-1)*T^3

  accelerometer bias:
  a_hat(k) = a_hat(k-1)

  State vector:
  
  x = [ z, 
        y,
        v,
        a_hat ]

  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = a 

  Input a - vertical acceleration from accelerometer

  Measurement - z = 0 (displacement integral) used as pseudo measurement for soft drift correction

  Observation matrix:
  H = [ 1, 0, 0, 0 ]  

  F = [[ 1,  T,  1/2*T^2, -1/6*T^3 ],
       [ 0,  1,  T,       -1/2*T^2 ],
       [ 0,  0,  1,       -T       ],
       [ 0,  0,  0,        1       ]]

  B = [  1/6*T^3,
         1/2*T^2,
         T,
         0       ]
*/


#include <ArduinoEigenDense.h>
#include <cmath>
#include <algorithm>

#define MIN_DIVISOR_VALUE 1e-12f

class EIGEN_ALIGN_MAX KalmanForWaveBasic {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Vector4f = Eigen::Matrix<float, 4, 1>;
    using Matrix4f = Eigen::Matrix<float, 4, 4>;

    struct State {
        float displacement_integral;
        float heave;
        float vert_speed;
        float accel_bias;
    };

    KalmanForWaveBasic(float q0 = 5.0f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 1e-5f, 
                       float observation_noise = 1e-3f) {
        initialize(q0, q1, q2, q3);
        initMeasurementNoise(observation_noise);
    }

    void initialize(float q0, float q1, float q2, float q3) {
        x.setZero();
        P.setIdentity();
        H << 1, 0, 0, 0;
        Q.setZero();
        Q(0,0) = q0;
        Q(1,1) = q1;
        Q(2,2) = q2;
        Q(3,3) = q3;
        enforcePositiveDefiniteness(Q);
        I.setIdentity();
    }

    void initMeasurementNoise(float r0) { R = r0; }

    void initState(const State& state) {
        x(0) = state.displacement_integral;
        x(1) = state.heave;
        x(2) = state.vert_speed;
        x(3) = state.accel_bias;
    }

    void predict(float accel, float delta_t) {
        const float T = delta_t;
        const float T2 = T * T;
        const float T3 = T2 * T;
        F << 1.0f, T, 0.5f * T2, (-1.0f / 6.0f) * T3,
             0.0f, 1.0f, T,     -0.5f * T2,
             0.0f, 0.0f, 1.0f,  -T,
             0.0f, 0.0f, 0.0f,   1.0f;
        B << (1.0f / 6.0f) * T3,
              0.5f * T2,
              T,
              0.0f;
        x = F * x + B * accel;
        P = (F * P).eval() * F.transpose() + Q;
        enforcePositiveDefiniteness(P);
    }

    void correct(float /*delta_t*/) {
        float z = 0.0f;
        float y = z - H * x;
        float S = (H * P * H.transpose())(0, 0) + R;
        if (fabs(S) > MIN_DIVISOR_VALUE) {
            Vector4f K = (P * H.transpose()) / S;
            x = x + K * y;
            Matrix4f I_KH = I - K * H;
            P = (I_KH * P).eval() * I_KH.transpose() + K * R * K.transpose();
            enforcePositiveDefiniteness(P);
        }
    }

    void step(float accel, float delta_t, State& state) {
        predict(accel, delta_t);
        correct(delta_t);
        state.displacement_integral = x(0);
        state.heave = x(1);
        state.vert_speed = x(2);
        state.accel_bias = x(3);
    }

    State getState() const {
        return State{x(0), x(1), x(2), x(3)};
    }

    // Calculate theoretical process noise Q matrix using IMU specs and Allan variance parameters
    // Defaults are from MPU6886 specs.
    // This method assumes that Kalman filter is in SI units and R is not scaled.
    // These values will be too low for practical use
    // but can provide a starting point for tuning Q for the production filter.
    Matrix4f calculateTheoreticalProcessNoise(float sample_rate_hz, float sigma_a_density = 0.004f,
                                              float sigma_b = 1.0f, float tau_b = 100.0f) const {
        const float BW = sample_rate_hz / 2.0f;
        const float sigma_a2 = sigma_a_density * sigma_a_density * BW;
        const float T = 1.0f / sample_rate_hz;
        const float q_bias = (sigma_b * sigma_b * T) / tau_b;
        const float q_z = (4.0f / 36.0f) * sigma_a2 * powf(T, 6);
        const float q_y = (1.0f / 4.0f) * sigma_a2 * powf(T, 4);
        const float q_v = sigma_a2 * powf(T, 2);

        Matrix4f theoretical_Q;
        theoretical_Q.setZero();
        theoretical_Q(0,0) = q_z;
        theoretical_Q(1,1) = q_y;
        theoretical_Q(2,2) = q_v;
        theoretical_Q(3,3) = q_bias;
        enforcePositiveDefiniteness(theoretical_Q);
        return theoretical_Q;
    }

private:
    Vector4f x;
    Matrix4f F;
    Vector4f B;
    Matrix4f Q;
    Eigen::RowVector4f H;
    float R;
    Matrix4f P;
    Matrix4f I;

    void enforceSymmetry(Matrix4f& mat) const {
        mat = 0.5f * (mat + mat.transpose());
    }

    void enforcePositiveDefiniteness(Matrix4f& mat) const {
        enforceSymmetry(mat);
        Eigen::LLT<Matrix4f> llt(mat);
        float epsilon = 1e-7f;
        while (llt.info() == Eigen::NumericalIssue && epsilon < 0.01f) {
            mat += epsilon * Matrix4f::Identity();
            llt.compute(mat);
            epsilon *= 10;
        }
    }
};

typedef KalmanForWaveBasic::State KalmanForWaveBasicState;

#endif
