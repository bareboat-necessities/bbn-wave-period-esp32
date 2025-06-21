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

  Measurement - z = 0 (displacement integral)

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

  The code below also uses zero acceleration correction. 
  When acceleration is zero then vertical displacement is zero 
  too (at least in trochoidal wave model), and vertical velocity is at
  its max or min. The code of the filter
  below uses it for an additional drift correction. 
         
*/

#include <ArduinoEigenDense.h>

class KalmanForWaveBasic {
private:
    Eigen::Vector4f x;
    Eigen::Matrix4f F;
    Eigen::Vector4f B;
    Eigen::Matrix4f Q;
    Eigen::RowVector4f H;
    float R;
    Eigen::Matrix4f P;
    Eigen::Matrix4f I;

    // Zero-correction parameters
    float zero_accel_threshold;
    float zero_correction_gain;  // [0-1] how strongly to correct
    int zero_counter = 0;
    const int zero_counter_threshold = 3; // require N consecutive low-accel samples
    
    // Separate observation noise for zero-correction
    float R_heave = 10.0f;
    float R_velocity = 100.0f;

public:
    struct State {
        float displacement_integral;
        float heave;
        float vert_speed;
        float accel_bias;
    };

    KalmanForWaveBasic(float q0, float q1, float q2, float q3, 
                       float observation_noise = 0.01f, float zero_threshold = 0.05f, float correction_gain = 0.5f)
                       : zero_accel_threshold(zero_threshold), zero_correction_gain(correction_gain) {
        initialize(q0, q1, q2, q3, observation_noise);
    }

    void initialize(float q0, float q1, float q2, float q3, float r0) {            
        // Initialize state vector
        x.setZero();
        
        // Initialize state covariance
        P.setIdentity();
        
        // Initialize observation matrix
        H << 1, 0, 0, 0;
        
        // Initialize observation noise
        R = r0;
        
        // Initialize process noise
        Q.setZero();
        Q(0,0) = q0;
        Q(1,1) = q1;
        Q(2,2) = q2;
        Q(3,3) = q3;
        
        // Initialize identity matrix
        I.setIdentity();
    }

    void initState(const State& state) {
        x(0) = state.displacement_integral;
        x(1) = state.heave;
        x(2) = state.vert_speed;
        x(3) = state.accel_bias;
    }

    void predict(float accel, float delta_t) {
        F << 1.0f, delta_t, 0.5f * delta_t * delta_t, (-1.0f/6.0f) * delta_t * delta_t * delta_t,
             0.0f, 1.0f,    delta_t,                  -0.5f * delta_t * delta_t,
             0.0f, 0.0f,    1.0f,                      -delta_t,
             0.0f, 0.0f,    0.0f,                      1.0f;
        
        B << (1.0f/6.0f) * delta_t * delta_t * delta_t,
             0.5f * delta_t * delta_t,
             delta_t,
             0.0f;
        
        x = F * x + B * accel;
        P = F * P * F.transpose() + Q;
    }

    void correct(float accel) {
        if (std::fabs(accel) < zero_accel_threshold) {
            zero_counter++;
        } else {
            zero_counter = 0;
        }

        if (zero_counter >= zero_counter_threshold) {
            // Soft correction - only move partially toward zero
            Eigen::Matrix<float, 2, 4> H_special;
            H_special << 0, 1, 0, 0,  // Observe heave
                         0, 0, 1, 0;  // Observe velocity
            
            // Target values (partial correction toward zero)
            Eigen::Vector2f z;
            z << (1.0f - zero_correction_gain) * x(1),  // Target: reduce heave by gain%
                 x(2);                                  // Target: no change to velocity%
            
            Eigen::Vector2f y = z - H_special * x;
            Eigen::Matrix2f S = H_special * P * H_special.transpose();
            S(0,0) += R_heave;
            S(1,1) += R_velocity;
            
            Eigen::Matrix<float, 4, 2> K = P * H_special.transpose() * S.inverse();
            x = x + K * y;
            
            // Joseph form update for covariance
            Eigen::Matrix4f I_KH = I - K * H_special;
            P = I_KH * P * I_KH.transpose() + K * S * K.transpose();
        }
        
        // Always do the standard correction with Joseph form
        float z = 0.0f;
        float y = z - H * x;
        float S = H * P * H.transpose() + R;
        Eigen::Vector4f K = P * H.transpose() / S;
        x = x + K * y;
        
        // Joseph form update for covariance
        Eigen::Matrix4f I_KH = I - K * H;
        P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    }

    void step(float accel, float delta_t, State& state) {
        predict(accel, delta_t);
        correct(accel);
        
        state.displacement_integral = x(0);
        state.heave = x(1);
        state.vert_speed = x(2);
        state.accel_bias = x(3);
    }

    const State getState() const {
        return State{x(0), x(1), x(2), x(3)};
    }

    void setZeroCorrectionParams(float threshold, float gain, float r_heave, float r_velocity) {
        zero_accel_threshold = threshold;
        zero_correction_gain = gain;
        R_heave = r_heave;
        R_velocity = r_velocity;
    }
};

typedef KalmanForWaveBasic::State KalmanForWaveBasicState;

#endif
