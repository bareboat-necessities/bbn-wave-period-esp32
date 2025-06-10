#ifndef KALMAN_WAVE_NUM_STABLE_ALT_H
#define KALMAN_WAVE_NUM_STABLE_ALT_H

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter to estimate vertical displacement in wave using accelerometer, 
  correct for accelerometer bias, estimate accelerometer bias. This method
  assumes that displacement follows trochoidal model and the frequency of
  wave is known. Frequency can be estimated using another step with Aranovskiy filter.

  In trochoidal wave model there is simple linear dependency between displacement and 
  acceleration.

  y - displacement (at any time):
  y = - L / (2 *pi) * (a/g),  g - acceleration of free fall constant, a - vertical acceleration

  wave length L: 
  L = g * period^2 / (2 *pi)

  wave period via frequency:
  period = 1 / f

  a = - (2 * pi * f)^2 * y

  let
  k_hat = - (2 * pi * f)^2

  Process model:

  displacement_integral:
  z(k) = z(k-1) + y(k-1)*T + 1/2*v(k-1)*T^2 + 1/6*a(k-1)*T^3 - 1/6*a_hat(k-1)*T^3
  
  displacement:
  y(k) = y(k-1) + v(k-1)*T + 1/2*a(k-1)*T^2 - 1/2*a_hat(k-1)*T^2

  velocity:
  v(k) = v(k-1) + a(k-1)*T - a_hat(k-1)*T

  acceleration (from trochoidal wave model):
  a(k) = k_hat*y(k-1) + k_hat*v(k-1)*T + k_hat*1/2*a(k-1)*T^2 - k_hat*1/2*a_hat(k-1)*T^2

  accelerometer bias:
  a_hat(k) = a_hat(k-1)

  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = 0 
  
  State vector:
  
  x = [ z,
        y,
        v,
        a,
        a_hat ]

  Input a - vertical acceleration from accelerometer

  Measurements:
    a (vertical acceleration), z = 0

  Observation matrix:
  H = [[ 1, 0 ],
       [ 0, 0 ],
       [ 0, 0 ],
       [ 0, 1 ],
       [ 0, 0 ]]  

  F = [[ 1,      T,    1/2*T^2,       1/6*T^3,         -1/6*T^3         ],
       [ 0,      1,    T,             1/2*T^2,         -1/2*T^2         ],
       [ 0,      0,    1,             T,               -T               ],
       [ 0,  k_hat,    k_hat*T,       1/2*k_hat*T^2,   -1/2*k_hat*T^2   ],
       [ 0,      0,    0,             0,               1                ]]
         
*/

#include <ArduinoEigenDense.h>

class KalmanWaveNumStableAlt {
public:
    // Type aliases
    using Vector5f = Eigen::Matrix<float, 5, 1>;
    using Matrix5f = Eigen::Matrix<float, 5, 5>;
    using Vector2f = Eigen::Matrix<float, 2, 1>;
    using Matrix2f = Eigen::Matrix<float, 2, 2>;
    using Matrix25f = Eigen::Matrix<float, 2, 5>;
    using Matrix52f = Eigen::Matrix<float, 5, 2>;

    struct State {
        float displacement_integral = 0.0f;
        float heave = 0.0f;
        float vert_speed = 0.0f;
        float vert_accel = 0.0f;
        float accel_bias = 0.0f;
        
        void print() const {
            Serial.print("Heave: "); Serial.print(heave, 4);
            Serial.print("m, Speed: "); Serial.print(vert_speed, 4);
            Serial.print("m/s, Accel: "); Serial.print(vert_accel, 4);
            Serial.println("m/s²");
        }
    };

    KalmanWaveNumStableAlt(float q0 = 1e+1f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 5.0f, float q4 = 1e-5f) {
        initialize(q0, q1, q2, q3, q4);
    }

    void initialize(float q0, float q1, float q2, float q3, float q4) {
        x.setZero();
        
        // Initialize UD factors (U = identity, D = initial variances)
        U.setIdentity();
        D << 10.0f, 10.0f, 10.0f, 10.0f, 10.0f; // Large initial uncertainty
        
        // Process noise UD factors (diagonal)
        Q_U.setIdentity();
        Q_D << q0, q1, q2, q3, q4;
        
        // Measurement noise
        R << 0.01f, 0.0f,
             0.0f, 1.0f;
        
        H << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
    }

    void update(float measured_accel, float wave_frequency, float delta_t) {
        const float k_hat = -pow(2.0f * M_PI * wave_frequency, 2);
        updateStateTransition(k_hat, delta_t);
        predictUD();
        correctUD(Vector2f(0.0f, measured_accel));
    }

    State getState() const {
        State s;
        s.displacement_integral = x(0);
        s.heave = x(1);
        s.vert_speed = x(2);
        s.vert_accel = x(3);
        s.accel_bias = x(4);
        return s;
    }

private:
    // State variables
    Vector5f x;     // State vector
    Matrix5f U;     // U factor (unit upper triangular)
    Vector5f D;     // D factor (diagonal)
    Matrix5f Q_U;   // Process noise U
    Vector5f Q_D;   // Process noise D
    Matrix2f R;     // Measurement noise
    Matrix25f H;    // Measurement model
    Matrix5f F;     // State transition

    void updateStateTransition(float k_hat, float delta_t) {
        const float T = delta_t;
        const float T2 = T * T;
        const float T3 = T2 * T;
        const float kT = k_hat * T;
        const float kT2 = k_hat * T2;

        F << 1.0f,    T,    0.5f*T2,    (1.0f/6.0f)*T3,    -(1.0f/6.0f)*T3,
             0.0f,    1.0f,    T,        0.5f*T2,          -0.5f*T2,
             0.0f,    0.0f,    1.0f,     T,                -T,
             0.0f,    k_hat,   kT,       0.5f*kT2,         -0.5f*kT2,
             0.0f,    0.0f,    0.0f,     0.0f,             1.0f;
    }

    void predictUD() {
        // State prediction
        x = F * x;
        
        // Covariance prediction: F*U*D*U'*F' + Q
        // Using specialized UD update for prediction
        Matrix5f A = F * U;
        Vector5f new_D = D;
        Matrix5f new_U = Matrix5f::Identity();
        
        // Bierman-Thornton UD update
        for (int j = 4; j >= 0; --j) {
            new_D(j) = Q_D(j);
            for (int i = 0; i <= j; ++i) {
                float sigma = A.row(i).head(j+1).dot(new_D.head(j+1).cwiseProduct(A.row(j).head(j+1)));
                if (i < j) {
                    new_U(i,j) = sigma / new_D(j);
                } else {
                    new_D(j) += sigma;
                }
            }
        }
        
        U = new_U;
        D = new_D;
    }

    void correctUD(const Vector2f& z) {
        // Measurement update using UD factorization
        Matrix25f H_U = H * U;
        Vector2f y = z - H * x;
        
        // Calculate Kalman gain using UD factors
        Vector2f f;
        Matrix52f K;
        Vector5f v;
        
        // Thornton's UD measurement update
        for (int i = 0; i < 2; ++i) {
            f(i) = H_U.row(i).dot(D.cwiseProduct(H_U.row(i).transpose())) + R(i,i);
            v = D.cwiseProduct(H_U.row(i).transpose());
            
            for (int j = 0; j < 5; ++j) {
                K(j,i) = v(j);
                for (int k = 0; k < j; ++k) {
                    K(j,i) -= U(k,j) * v(k);
                }
                K(j,i) /= f(i);
            }
            
            // Update U and D
            for (int j = 0; j < 5; ++j) {
                float save = H_U(i,j);
                for (int k = 0; k < j; ++k) {
                    H_U(i,j) -= H_U(i,k) * U(k,j);
                }
                v(j) = save + H_U(i,j);
            }
            
            for (int j = 0; j < 5; ++j) {
                float save = U.row(j).head(j).dot(v.head(j));
                D(j) -= K(j,i) * K(j,i) * f(i);
                for (int k = 0; k < j; ++k) {
                    U(k,j) -= K(k,i) * (v(j) + save);
                }
            }
        }
        
        // State update
        x += K * y;
        
        // Ensure positive definiteness
        for (int i = 0; i < 5; ++i) {
            if (D(i) <= 0.0f) {
                D(i) = 1e-8f;
            }
        }
    }
};

typedef KalmanWaveNumStableAltState KalmanWaveNumStableAlt::State; 

#endif // KALMAN_WAVE_NUM_STABLE_ALT_H
