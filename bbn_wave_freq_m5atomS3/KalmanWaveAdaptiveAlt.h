#ifndef KALMAN_WAVE_ADAPTIVE_ALT_H
#define KALMAN_WAVE_ADAPTIVE_ALT_H

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


  Innovation-Based R Tuning

    The measurement noise covariance (R) is adapted based on the actual innovation statistics.

  Allan Variance-Inspired Q Tuning

    For process noise (Q), we use Allan variance principles to estimate IMU noise parameters.
         
*/

#include <ArduinoEigenDense.h>

// Static configuration for Allan variance calculation
static constexpr size_t AV_WINDOW_SIZE = 256;          // Power of 2 for better Allan variance calculation
static constexpr size_t AV_MIN_CLUSTER_SIZE = 4;       // Minimum samples for variance calculation
static constexpr size_t AV_MAX_CLUSTER_SIZE = 64;     // Maximum cluster size to check

static constexpr size_t INNOVATION_WINDOW_SIZE = 80;  // Fixed window size

class KalmanWaveAdaptiveAlt {
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

    KalmanWaveAdaptiveAlt(float q0 = 1e+1f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 5.0f, float q4 = 1e-5f) {
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
        R << 0.01f,  0.0f,    // Displacement integral noise
             0.0f,   1.0f;    // Acceleration noise (m/s²)²

        // Measurement model
        H << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // Measures displacement integral
             0.0f, 0.0f, 0.0f, 1.0f, 1.0f;  // Measures acceleration
    }

    void update(float measured_accel, float k_hat, float delta_t) {

        // Allan Variance-Inspired Q Tuning
        updateProcessNoise(measured_accel, delta_t);
      
        // Update state transition matrix
        updateStateTransition(k_hat, delta_t);

        // Prediction step
        predict();

        // Create measurement vector [displacement_integral, acceleration]
        // Note: We assume displacement integral measurement is 0 (reset each cycle)
        const Vector2f z(0.0f, measured_accel);

        // Correction step with Joseph form
        correctJoseph(z);
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

private:
    Vector5f x;     // State vector
    Matrix5f P;     // Covariance matrix
    Matrix5f Q;     // Process noise
    Matrix2f R;     // Measurement noise
    Matrix25f H;    // Measurement model
    Matrix5f F;     // State transition matrix

    Eigen::Matrix<float, 2, INNOVATION_WINDOW_SIZE> innovation_history;
    size_t innovation_count = 0;

    // Static storage for acceleration history
    Eigen::Matrix<float, AV_WINDOW_SIZE, 1> accel_history;
    size_t history_index = 0;
    bool history_filled = false;

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

    void updateMeasurementNoise(const Vector2f& innovation) {
        // Circular buffer implementation
        size_t current_index = innovation_count % INNOVATION_WINDOW_SIZE;
        innovation_history.col(current_index) = innovation;
        innovation_count++;
        
        // Only update R after we have enough samples
        size_t valid_samples = std::min(innovation_count, INNOVATION_WINDOW_SIZE);
        if (valid_samples >= 10) {  // Minimum samples before updating
            // Calculate mean
            Vector2f mean = Vector2f::Zero();
            for (size_t i = 0; i < valid_samples; ++i) {
                mean += innovation_history.col(i);
            }
            mean /= valid_samples;
            
            // Calculate covariance
            Matrix2f cov = Matrix2f::Zero();
            for (size_t i = 0; i < valid_samples; ++i) {
                Vector2f diff = innovation_history.col(i) - mean;
                cov += diff * diff.transpose();
            }
            cov /= (valid_samples - 1);
            
            // Blend between initial R and measured innovation covariance
            R = 0.95f * R + 0.05f * cov;
            
            // Ensure positive definiteness
            enforceSymmetry(R);
            ensurePositiveDefinite(R);
        }
    }

    void updateProcessNoise(float accel_measurement, float delta_t) {
        float sample_time = delta_t;
        
        // Store acceleration in circular buffer
        accel_history(history_index) = accel_measurement;
        history_index = (history_index + 1) % AV_WINDOW_SIZE;
        if (history_index == 0) history_filled = true;

        // Update Q periodically (every 80 samples)
        static size_t update_counter = 0;
        if (++update_counter >= 80) {
            update_counter = 0;
            
            // Only proceed if we have enough data
            size_t available_samples = history_filled ? AV_WINDOW_SIZE : history_index;
            if (available_samples >= AV_MIN_CLUSTER_SIZE * 2) {
                // Estimate noise parameters using static buffers
                float q_angle = estimateAngleRandomWalk(available_samples, sample_time);
                float q_bias = estimateBiasInstability(available_samples, sample_time);
                
                // Update Q diagonal elements
                Q(1,1) = q_angle;        // Heave position noise
                Q(2,2) = q_angle * 10.0f; // Velocity noise
                Q(3,3) = q_angle * 100.0f;// Acceleration noise
                Q(4,4) = q_bias;         // Bias noise
                
                enforceSymmetry(Q);
                ensurePositiveDefinite(Q);
            }
        }
    }

    // Angle random walk estimation
    float estimateAngleRandomWalk(size_t available_samples, float sample_time) {
        float variance = 0.0f;
        size_t count = 0;
        
        // Calculate first differences
        for (size_t i = 1; i < available_samples; i++) {
            size_t idx1 = (history_index - i - 1) % AV_WINDOW_SIZE;
            size_t idx2 = (history_index - i) % AV_WINDOW_SIZE;
            float diff = accel_history(idx2) - accel_history(idx1);
            variance += diff * diff;
            count++;
        }
        
        variance /= count;
        return variance * sample_time;  // Angle random walk coefficient
    }

    // Bias instability estimation
    float estimateBiasInstability(size_t available_samples, float sample_time) {
        float min_var = std::numeric_limits<float>::max();
        
        // Check cluster sizes from AV_MIN_CLUSTER_SIZE to AV_MAX_CLUSTER_SIZE
        for (size_t m = AV_MIN_CLUSTER_SIZE; m <= std::min(AV_MAX_CLUSTER_SIZE, available_samples/2); m++) {
            size_t num_clusters = available_samples / m;
            
            for (size_t cluster = 0; cluster < num_clusters; cluster++) {
                float sum = 0.0f;
                
                // Calculate cluster mean
                for (size_t i = 0; i < m; i++) {
                    size_t idx = (history_index - (cluster * m + i) - 1) % AV_WINDOW_SIZE;
                    sum += accel_history(idx);
                }
                float mean = sum / m;
                
                // Calculate cluster variance
                float var = 0.0f;
                for (size_t i = 0; i < m; i++) {
                    size_t idx = (history_index - (cluster * m + i) - 1) % AV_WINDOW_SIZE;
                    float diff = accel_history(idx) - mean;
                    var += diff * diff;
                }
                var /= (m - 1);
                
                if (var < min_var) min_var = var;
            }
        }
        return min_var * sample_time * sample_time;  // Bias instability coefficient
    }

    void correctJoseph(const Vector2f& z) {
        // Innovation: y = z - H * x
        Vector2f y = z - H * x;
        updateMeasurementNoise(y); 
        
        // Innovation covariance: S = H * P * H' + R
        const Matrix2f S = (H * P * H.transpose() + R).eval();
        
        // Kalman gain: K = P * H' * S^-1
        const Matrix52f K = P * H.transpose() * S.inverse();
        
        // State update: x = x + K * y
        x += K * y;
        
        // Joseph form covariance update: 
        // P = (I-KH) * P * (I-KH)' + K * R * K'
        const Matrix5f I = Matrix5f::Identity();
        const Matrix5f KH = K * H;
        P = ((I - KH) * P * (I - KH).transpose() + K * R * K.transpose()).eval();
        
        // Numerical stabilization
        enforceSymmetry(P);
        ensurePositiveDefinite(P);
    }

    template<int Rows, int Cols>
    void enforceSymmetry(Eigen::Matrix<float, Rows, Cols>& mat) {
        // Average upper and lower triangular parts
        Eigen::Matrix<float, Rows, Cols> symm = 0.5f * (mat + mat.transpose());
        mat = symm;
    }

    template<int Size>
    void ensurePositiveDefinite(Eigen::Matrix<float, Size, Size>& mat) {
        // Check for positive definiteness via LDLT
        Eigen::LDLT<Eigen::Matrix<float, Size, Size>> ldlt(mat);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
            // Add small regularization to diagonal
            mat.diagonal().array() += 1e-9f;
        }
    }
};

typedef KalmanWaveAdaptiveAlt::State KalmanWaveAdaptiveAltState; 

#endif // KALMAN_WAVE_ADAPTIVE_ALT_H
