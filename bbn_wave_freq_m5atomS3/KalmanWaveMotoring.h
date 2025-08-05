#ifndef KALMAN_WAVE_MOTORING_H
#define KALMAN_WAVE_MOTORING_H

/*
  Copyright 2025, Mikhail Grushinskiy

  Kalman filter to estimate vertical displacement in wave using accelerometer, 
  correct for accelerometer bias, estimate accelerometer bias. This method
  assumes that displacement follows trochoidal model and the frequency of acceleration is known. 
  Augmented with a low‑pass acceleration state (vib_noise) to suppress high‑frequency
  engine vibration, based on the colored‑noise state augmentation approach
  from Crassidis & Kumar (2007), AIAA GNC.

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
    // In implementation, temperature‑dependent bias drift is modeled as:
    // a_hat(k) ← a_hat(k) + temperature_coefficient * ( T_celsius(k) - T_celsius(k-1) )
    // where:
    //   temperature_coefficient  = bias change per degree Celsius [m/s² / °C]
    //   T_celsius(k)             = current temperature in degrees Celsius
    
  Process model in matrix form:
  
  x(k) = F*x(k-1) + B*u(k) + w(k)

  w(k) - zero mean noise,
  u(k) = 0 
  
  State vector:
  
  x = [ z,             // displacement_integral
        y,             // heave
        v,             // vertical speed
        a,             // vertical acceleration (trochoidal model)
        a_hat,         // accelerometer bias
        vib_noise ]    // low‑pass acceleration (internal, unmeasured)

  low‑pass acceleration (colored‑noise augmentation):
  vib_noise(k) = phi * vib_noise(k-1) + w_vib(k)

    where:
      w_vib(k) ~ N(0, sigma_vib² * (1 - phi²))
      phi = exp(-T / tau)
      tau = 1 / (2 * pi * f_c)  // f_c is cutoff frequency in Hz

  Added low‑pass acceleration state (vib_noise) to suppress high‑frequency engine noise.
  Based on Crassidis & Kumar (2007) colored‑noise augmentation.
  
  Input measured vertical acceleration from accelerometer (includes bias)

  Measurements:
    a_measured (vertical acceleration with a bias), z = 0 (using it as a soft constraint to anchor drift)

  Process model in matrix form (augmented F):

  F = [[ 1,      T,    1/2*T^2,       1/6*T^3,         -1/6*T^3,       0            ],
       [ 0,      1,    T,             1/2*T^2,         -1/2*T^2,       0            ],
       [ 0,      0,    1,             T,               -T,             0            ],
       [ 0,  k_hat,    k_hat*T,       1/2*k_hat*T^2,   -1/2*k_hat*T^2, 0            ],
       [ 0,      0,    0,             0,                1,             0            ],
       [ 0,      0,    0,             0,                0,             phi          ]]

  Note: There could be a variation of the filter where F(5,3)=(1-phi) & F(5,4)=(1-phi).

  Measurement model (unchanged physical measurements):

  H = [[ 1, 0, 0, 0, 0, 0 ],     // displacement integral (soft constraint)
       [ 0, 0, 0, 1, 1, 1 ]]     // raw accelerometer = a + a_hat + vib_noise

  The low‑pass state vib_noise is updated through the process model,
  but is not directly measured.
         
*/

#include <ArduinoEigenDense.h>

#ifndef TWO_PI
#define TWO_PI 6.283185307179586476925286766559
#endif

class KalmanWaveMotoring {
public:
    // Type aliases
    using Vector6f = Eigen::Matrix<float, 6, 1>;
    using Matrix6f = Eigen::Matrix<float, 6, 6>;
    using Vector2f = Eigen::Matrix<float, 2, 1>;
    using Matrix2f = Eigen::Matrix<float, 2, 2>;
    using Matrix26f = Eigen::Matrix<float, 2, 6>;
    using Matrix62f = Eigen::Matrix<float, 6, 2>;

    struct State {
        float displacement_integral = 0.0f;
        float heave = 0.0f;
        float vert_speed = 0.0f;
        float vert_accel = 0.0f;
        float accel_bias = 0.0f;
        float vib_noise  = 0.0f;
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
        float vib_noise_std_dev = 0.0f;          // Standard deviation of vibration noise estimate
        float residual_accel = 0.0f;             // Acceleration measurement residual
    };

    KalmanWaveMotoring(float q0 = 2.0f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 1e+5f, float q4 = 1e-5f, float q5 = 1e-3f,
            float temperature_drift_coeff = 0.007f) {
        initialize(q0, q1, q2, q3, q4, q5, temperature_drift_coeff);

        // Measurement noise covariance
        initMeasurementNoise(
            1e-3f,  // Displacement integral noise
            1e-2f   // Acceleration noise (m/s²)²
        );
      
        setCutoffHz(4.0f); // default LPF cutoff
    }

    void initialize(float q0, float q1, float q2, float q3, float q4, float q5, float temperature_drift_coeff) {
        temperature_coefficient = temperature_drift_coeff;
      
        // State vector initialization
        x.setZero();

        // Initial covariance - large uncertainty
        P.setIdentity();
        P *= 10.0f;  // Initial uncertainty

        // Process noise covariance (diagonal)
        Q.setZero();
        Q.diagonal() << q0,  // third accel integral
                        q1,  // displacement
                        q2,  // velocity
                        q3,  // accel (high noise due to square noisy frequency term)
                        q4,  // accel bias
                        q5;  // low-passed accel

        // Measurement model, H: measure displacement integral (z) and raw acceleration (a + bias)
        H << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // Measures displacement integral
             0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f;  // Measures acceleration

        // Reset metrics
        resetMetrics();
    }

    void setCutoffHz(float fc) {
        cutoff_hz = fc;
        tau = 1.0f / (2.0f * M_PI * cutoff_hz);
    }

    void update(float measured_accel, float k_hat, float delta_t, float temperature_celsius = NAN) {
        if (delta_t < 1e-10f) return;

        // compute phi for LPF
        phi = std::exp(-delta_t / tau);

        // update Q for LPF state
        Q(5,5) = sigm_vib_noise2 * (1.0f - phi * phi);
      
        // Update state transition matrix
        updateStateTransition(k_hat, delta_t);

        // Temperature compensation of bias
        if (std::isnan(last_temperature_celsius)) {
          if (!std::isnan(temperature_celsius)) {
            last_temperature_celsius = temperature_celsius;
          }
        } else if (!std::isnan(temperature_celsius)) {
          float delta_Temp = temperature_celsius - last_temperature_celsius;
          x(4) += temperature_coefficient * delta_Temp;
          last_temperature_celsius = temperature_celsius;
        }
      
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
        s.vib_noise = x(5);
        return s;
    }

    FilterMetrics getMetrics() const {
        return metrics;
    }

    void initState(const State& s0) {
        x(0) = s0.displacement_integral; // m*s
        x(1) = s0.heave;                 // m
        x(2) = s0.vert_speed;            // m/s
        x(3) = s0.vert_accel;            // m/s^2
        x(4) = s0.accel_bias;            // m/s^2
        x(5) = s0.vib_noise;             // m/s^2
    }

    void initMeasurementNoise(const float r0, const float r1) {
        // Measurement noise covariance
        R.setZero();
        R.diagonal() << r0, r1;  // Displacement integral noise, Acceleration noise (m/s²)²
    }

    static float calculate_k_hat(const float linear_freq) {
        float k_hat = - (TWO_PI * linear_freq) * (TWO_PI * linear_freq);
        return k_hat;
    }

    void resetMetrics() {
        metrics = FilterMetrics();
    }

private:
    Vector6f x;     // State vector
    Matrix6f P;     // Covariance matrix
    Matrix6f Q;     // Process noise
    Matrix2f R;     // Measurement noise
    Matrix26f H;    // Measurement model
    Matrix6f F;     // State transition matrix

    float cutoff_hz = 4.0f;
    float tau = 1.0f / (2.0f * M_PI * cutoff_hz);
    float phi = 0.0f;
    float sigm_vib_noise2 = 1e-3f;

    FilterMetrics metrics; // Filter performance metrics

    float last_temperature_celsius = NAN;    // degC
    float temperature_coefficient = 0.007f;  // m/s^2/degC

    void updateStateTransition(float k_hat, float delta_t) {
        if (delta_t < 1e-10f) {
            return;
        }
        const float T = delta_t;
        const float T2 = T * T;
        const float T3 = T2 * T;
        const float kT = k_hat * T;
        const float kT2 = k_hat * T2;

        // clang-format off
        F <<
          1,   T,       0.5f * T2,  (1.0f/6.0f) * T3,    -(1.0f/6.0f) * T3,      0,
          0,   1,       T,           0.5f * T2,          -0.5f * T2,             0,
          0,   0,       1,           T,                  -T,                     0,
          0,   k_hat,   kT,          0.5f * kT2,         -0.5f * kT2,            0,
          0,   0,       0,           0,                   1,                     0,
          0,   0,       0,          (1.0f - phi) /*0*/,  (1.0f - phi) /*0*/,     phi; 
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
        const Matrix62f K = P * H.transpose() * S.inverse();
        
        // State update: x = x + K * y
        x += K * innovation;
        
        // Joseph form covariance update: 
        // P = (I-KH) * P * (I-KH)' + K * R * K'
        const Matrix6f I = Matrix6f::Identity();
        const Matrix6f KH = K * H;
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
        Eigen::JacobiSVD<Matrix6f> svd(P);
        float singular_max = svd.singularValues().maxCoeff();
        float singular_min = svd.singularValues().minCoeff();
        metrics.condition_number = singular_max / std::max(singular_min, 1e-12f);
        
        // Standard deviations (uncertainties) of state estimates
        metrics.position_std_dev = sqrt(P(1,1));        // heave
        metrics.velocity_std_dev = sqrt(P(2,2));        // vertical speed
        metrics.acceleration_std_dev = sqrt(P(3,3));    // vertical acceleration
        metrics.bias_std_dev = sqrt(P(4,4));            // accelerometer bias
        metrics.vib_noise_std_dev = sqrt(P(5,5));       // vibration noise
        
        // Acceleration measurement residual (actual - predicted)
        // If the residuals are consistently large, it suggests issues with the accelerometer model or the filter's state
        metrics.residual_accel = innovation(1);
    }

    void enforceSymmetry(Matrix6f& mat) const {
        // Average upper and lower triangular parts
        Matrix6f symm = 0.5f * (mat + mat.transpose());
        mat = symm;
    }

    void ensurePositiveDefinite(Matrix6f& mat) const {
        Eigen::LLT<Matrix6f> llt(mat);  // Cholesky
        float epsilon = 1e-7f;
        while (llt.info() == Eigen::NumericalIssue && epsilon < 0.01f) {
            mat += epsilon * Matrix6f::Identity();
            llt.compute(mat);
            epsilon *= 10;
        }      
    }
};

typedef KalmanWaveMotoring::State KalmanWaveMotoringState; 
typedef KalmanWaveMotoring::FilterMetrics KalmanWaveMotoringMetrics;

#endif // KALMAN_WAVE_MOTORING_H
