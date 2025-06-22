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
  H = [[ 1, 0, 0, 0, 0 ],
       [ 0, 0, 0, 1, 1 ]]   (since measurement includes bias and is not 'true' a)

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
    };

    struct FilterMetrics {
        float innovation_magnitude = 0.0f;       // Magnitude of innovation vector (z - Hx)
        float innovation_normalized = 0.0f;      // Normalized innovation squared (y'*S^-1*y)
        float covariance_trace = 0.0f;           // Trace of covariance matrix (sum of variances)
        float max_covariance = 0.0f;             // Maximum diagonal element of covariance matrix
        float condition_number = 0.0f;           // Condition number of covariance matrix
        float position_std_dev = 0.0f;           // Standard deviation of position estimate
        float velocity_std_dev = 0.0f;           // Standard deviation of velocity estimate
        float acceleration_std_dev = 0.0f;       // Standard deviation of acceleration estimate
        float bias_std_dev = 0.0f;               // Standard deviation of bias estimate
        float residual_accel = 0.0f;             // Acceleration measurement residual
    };

    KalmanWaveNumStableAlt(float q0 = 1e+1f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 5.0f, float q4 = 1e-5f) {
        initialize(q0, q1, q2, q3, q4);
    }

    void initialize(float q0, float q1, float q2, float q3, float q4) {
        // State vector initialization
        x.setZero();

        // Initial covariance - large uncertainty
        P.setIdentity();
        P *= 1.0f;  // Initial uncertainty

        // Process noise covariance (diagonal)
        Q.setZero();
        Q.diagonal() << q0, q1, q2, q3, q4;

        // Measurement noise covariance
        R << 0.001f,  0.0f,   // Displacement integral noise
             0.0f,    1.0f;   // Acceleration noise (m/s²)²

        // Measurement model
        H << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // Measures displacement integral
             0.0f, 0.0f, 0.0f, 1.0f, 1.0f;  // Measures acceleration

        // Reset metrics
        resetMetrics();
    }

    void update(float measured_accel, float k_hat, float delta_t) {
      
        // Update state transition matrix
        updateStateTransition(k_hat, delta_t);

        // Prediction step
        predict();

        // Create measurement vector [displacement_integral, acceleration]
        // Note: We assume displacement integral measurement is 0 (reset each cycle)
        const Vector2f z(0.0f, measured_accel);

        // Correction step with Joseph form
        Vector2f innovation;
        Matrix2f S;
        correctJoseph(z, innovation, S);

        // Update metrics after correction
        updateMetrics(z, innovation, S);
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

    FilterMetrics getMetrics() const {
        return metrics;
    }

    void initState(const State& s0) {
        x(0) = s0.displacement_integral;
        x(1) = s0.heave;
        x(2) = s0.vert_speed;
        x(3) = s0.vert_accel;
        x(4) = s0.accel_bias;
    }

    void initMeasurementNoise(float r0, float r1) {
        // Measurement noise covariance
        R.setZero();
        R.diagonal() << r0, r1;  // Displacement integral noise, Acceleration noise (m/s²)²
    }

    float calculate_k_hat(float linear_freq) const {
      float k_hat = - pow(2.0 * M_PI * linear_freq, 2);
      return k_hat;
    }

    void resetMetrics() {
        metrics = FilterMetrics();
    }

private:
    Vector5f x;     // State vector
    Matrix5f P;     // Covariance matrix
    Matrix5f Q;     // Process noise
    Matrix2f R;     // Measurement noise
    Matrix25f H;    // Measurement model
    Matrix5f F;     // State transition matrix
    FilterMetrics metrics; // Filter performance metrics

    void updateStateTransition(float k_hat, float delta_t) {
        const float T = delta_t;
        const float T2 = T * T;
        const float T3 = T2 * T;
        const float kT = k_hat * T;
        const float kT2 = k_hat * T2;

        // clang-format off
        F << 1.0f,    T,    0.5f*T2,    (1.0f/6.0f)*T3,    -(1.0f/6.0f)*T3,
             0.0f,    1.0f,    T,        0.5f*T2,          -0.5f*T2,
             0.0f,    0.0f,    1.0f,     T,                -T,
             0.0f,    k_hat,   kT,       0.5f*kT2,         -0.5f*kT2,
             0.0f,    0.0f,    0.0f,     0.0f,             1.0f;
        // clang-format on
    }

    void predict() {
        // State prediction: x = F * x
        x = F * x;
        
        // Covariance prediction: P = F * P * F' + Q
        // Using .eval() to force immediate evaluation
        P = (F * P * F.transpose()).eval() + Q;
        
        // Enforce symmetry
        enforceSymmetry(P);
    }

    void correctJoseph(const Vector2f& z, Vector2f& innovation, Matrix2f& S) {
        // Innovation: y = z - H * x
        innovation = z - H * x;
        
        // Innovation covariance: S = H * P * H' + R
        S = (H * P * H.transpose() + R).eval();
        
        // Kalman gain: K = P * H' * S^-1
        const Matrix52f K = P * H.transpose() * S.inverse();
        
        // State update: x = x + K * y
        x += K * innovation;
        
        // Joseph form covariance update: 
        // P = (I-KH) * P * (I-KH)' + K * R * K'
        const Matrix5f I = Matrix5f::Identity();
        const Matrix5f KH = K * H;
        P = ((I - KH) * P * (I - KH).transpose() + K * R * K.transpose()).eval();
        
        // Numerical stabilization
        enforceSymmetry(P);
        ensurePositiveDefinite(P);
    }

    void updateMetrics(const Vector2f& z, const Vector2f& innovation, const Matrix2f& S) {
        // Innovation vector innovation = (z - Hx)
        // Innovation covariance S = H * P * H.transpose() + R;
        
        // Update metrics
        metrics.innovation_magnitude = innovation.norm();

        // innovation_normalized 
        // This is the Mahalanobis distance squared. For a well-tuned filter, this value should ideally be 
        // around the dimension of the innovation vector (which is 2 in this case, innovation.size()). 
        // If it's consistently much larger, R might be too small, or Q too large, or the model is incorrect. 
        // If it's consistently much smaller, R might be too large
        metrics.innovation_normalized = (innovation.transpose() * S.inverse() * innovation)(0,0);
        metrics.covariance_trace = P.trace();
        metrics.max_covariance = P.diagonal().maxCoeff();
        
        // Compute condition number of covariance matrix
        // A very high condition number for P indicates that the state variables are highly
        // correlated or that the matrix is close to singular, which can lead to numerical instability.
        Eigen::JacobiSVD<Matrix5f> svd(P);
        float singular_max = svd.singularValues()(0);
        float singular_min = svd.singularValues()(svd.singularValues().size()-1);
        metrics.condition_number = singular_max / singular_min;
        
        // Standard deviations (uncertainties) of state estimates
        metrics.position_std_dev = sqrt(P(1,1));        // heave
        metrics.velocity_std_dev = sqrt(P(2,2));        // vertical speed
        metrics.acceleration_std_dev = sqrt(P(3,3));    // vertical acceleration
        metrics.bias_std_dev = sqrt(P(4,4));            // accelerometer bias
        
        // Acceleration measurement residual (actual - predicted)
        // If the residuals are consistently large, it suggests issues with the accelerometer model or the filter's state
        metrics.residual_accel = innovation(1);
    }

    void enforceSymmetry(Matrix5f& mat) {
        // Average upper and lower triangular parts
        Matrix5f symm = 0.5f * (mat + mat.transpose());
        mat = symm;
    }

    void ensurePositiveDefinite(Matrix5f& mat) {
        Eigen::LLT<Matrix5f> llt(mat);  // Cholesky
        float epsilon = 1e-7f;
        while (llt.info() == Eigen::NumericalIssue && epsilon < 0.01f) {
            mat += epsilon * Matrix5f::Identity();
            llt.compute(mat);
            epsilon *= 10;
        }      
    }
};

typedef KalmanWaveNumStableAlt::State KalmanWaveNumStableAltState; 
typedef KalmanWaveNumStableAlt::FilterMetrics KalmanWaveNumStableAltMetrics;

#endif // KALMAN_WAVE_NUM_STABLE_ALT_H
