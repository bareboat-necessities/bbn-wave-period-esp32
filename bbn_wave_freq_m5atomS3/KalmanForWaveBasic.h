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

  The code below now uses Schmitt trigger-like zero-crossing detection
  for additional drift correction. When acceleration crosses zero with
  sufficient hysteresis, we assume vertical displacement is near zero
  and vertical velocity is at its max or min.
*/

#include <ArduinoEigenDense.h>

#define ZERO_CROSSINGS_HYSTERESIS_KF             0.04f
#define ZERO_CROSSINGS_VELOCITY_THRESHOLD_KF     0.6f
#define ZERO_CROSSINGS_DEBOUNCE_TIME_KF          0.15f
#define MIN_DIVISOR_VALUE                        1e-12f  // Minimum allowed value for division operations

class KalmanForWaveBasic {

public:
    // Type aliases
    using Vector4f = Eigen::Matrix<float, 4, 1>;
    using Matrix4f = Eigen::Matrix<float, 4, 4>;
    using Vector2f = Eigen::Matrix<float, 2, 1>;
    using Matrix2f = Eigen::Matrix<float, 2, 2>;
    using Matrix24f = Eigen::Matrix<float, 2, 4>;
    using Matrix42f = Eigen::Matrix<float, 4, 2>;

    struct State {
        float displacement_integral;
        float heave;
        float vert_speed;
        float accel_bias;
    };

    enum class SchmittTriggerState {
        SCHMITT_LOW = 0,
        SCHMITT_HIGH = 1
    };

    KalmanForWaveBasic(float q0 = 5.0f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 1e-5f, 
                       float observation_noise = 1e-3f, 
                       float positive_threshold = ZERO_CROSSINGS_HYSTERESIS_KF, 
                       float negative_threshold = -ZERO_CROSSINGS_HYSTERESIS_KF,
                       float velocity_threshold = ZERO_CROSSINGS_VELOCITY_THRESHOLD_KF,
                       float debounce_time = ZERO_CROSSINGS_DEBOUNCE_TIME_KF,
                       float correction_gain = 1.0f)
                       : schmitt_positive_threshold(positive_threshold),
                         schmitt_negative_threshold(negative_threshold),
                         schmitt_velocity_threshold(velocity_threshold),
                         schmitt_debounce_time(debounce_time),
                         zero_correction_gain(correction_gain) {
        initialize(q0, q1, q2, q3);
        initMeasurementNoise(observation_noise);
    }

    void initialize(float q0, float q1, float q2, float q3) {            
        // Initialize state vector
        x.setZero();
        
        // Initialize state covariance - symmetric and positive definite
        P.setIdentity();
        
        // Initialize observation matrix
        H << 1, 0, 0, 0;
                
        // Initialize process noise - symmetric and positive definite
        Q.setZero();
        Q(0,0) = q0;
        Q(1,1) = q1;
        Q(2,2) = q2;
        Q(3,3) = q3;
        enforcePositiveDefiniteness(Q);
        
        // Initialize identity matrix
        I.setIdentity();
        
        // Initialize Schmitt trigger state
        schmitt_state = SchmittTriggerState::SCHMITT_LOW;
    }

    void initMeasurementNoise(float r0) {
        // Initialize observation noise covariance
        // Displacement integral noise (m*s)²
        R = r0;  
    }
    
    void initState(const State& state) {
        x(0) = state.displacement_integral;
        x(1) = state.heave;
        x(2) = state.vert_speed;
        x(3) = state.accel_bias;
    }

    void predict(float accel, float delta_t) {
        // Precompute powers of delta_t
        const float T = delta_t;    // T
        const float T2 = T * T;     // T squared
        const float T3 = T2 * T;    // T cubed
        
        F << 1.0f,      T,    0.5f * T2,  (-1.0f/6.0f) * T3,
             0.0f,   1.0f,    T,          -0.5f * T2,
             0.0f,   0.0f,    1.0f,       -T,
             0.0f,   0.0f,    0.0f,       1.0f;
        
        B << (1.0f/6.0f) * T3,
             0.5f * T2,
             T,
             0.0f;
        
        x = F * x + B * accel;
        P = F * P * F.transpose() + Q;
        enforcePositiveDefiniteness(P);  // Ensure P remains symmetric and positive definite
        
        // Update Schmitt trigger state
        updateSchmittTrigger(accel, x(2), delta_t);
    }

    void updateSchmittTrigger(float accel, float velocity, float delta_t) {
        zero_crossing_time_since += delta_t;
        if (schmitt_state == SchmittTriggerState::SCHMITT_LOW) {
            // Currently in low state, check if we should switch to high
            if (accel > schmitt_positive_threshold && abs(velocity) > schmitt_velocity_threshold && zero_crossing_time_since > schmitt_debounce_time) {
                schmitt_state = SchmittTriggerState::SCHMITT_HIGH;
                zero_crossing_detected = true;
                zero_crossing_last_interval = zero_crossing_time_since;
                zero_crossing_time_since = 0.0f;
            }
        } else {
            // Currently in high state, check if we should switch to low
            if (accel < schmitt_negative_threshold && abs(velocity) > schmitt_velocity_threshold && zero_crossing_time_since > schmitt_debounce_time) {
                schmitt_state = SchmittTriggerState::SCHMITT_LOW;
                zero_crossing_detected = true;
                zero_crossing_last_interval = zero_crossing_time_since;
                zero_crossing_time_since = 0.0f;
            }
        }
    }

    void correct(float delta_t) {
        if (zero_crossing_detected) {
            zeroCrossingCorrection(delta_t);
        }
        
        // Always do the standard correction with Joseph form
        float z = 0.0f;
        float y = z - H * x;
        float S = (H * P * H.transpose())(0, 0) + R;
        
        // Check for numerical stability before division
        if (fabs(S) > MIN_DIVISOR_VALUE) {
            Vector4f K = P * H.transpose() / S;
            x = x + K * y;
            
            // Joseph form update for covariance
            Matrix4f I_KH = I - K * H;
            P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
            enforcePositiveDefiniteness(P);  // Ensure P remains symmetric and positive definite
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

    float getEstimatedWavePeriod() const { 
        return 2.0f * zero_crossing_last_interval; 
    }

    void setZeroCorrectionParams(float positive_thresh, float negative_thresh, float velocity_thresh,
                                 float debounce_time, float gain, float r_heave, float r_velocity) {
        schmitt_positive_threshold = positive_thresh;
        schmitt_negative_threshold = negative_thresh;
        schmitt_velocity_threshold = velocity_thresh;
        schmitt_debounce_time = debounce_time;
        zero_correction_gain = gain;
        R_heave = r_heave;
        R_velocity = r_velocity;
    }

    // Calculate theoretical process noise Q matrix using IMU specs and Allan variance parameters
    // Defaults are from MPU6886 specs.
    // This method assumes that Kalman filter is in SI units and R is not scaled.
    // These values will be too low for practical use
    // but can provide a starting point for tuning Q for the production filter.
    Matrix4f calculateTheoreticalProcessNoise(
        float sample_rate_hz,               // Accelerometer sample rate (Hz)
        float sigma_a_density = 0.002943f,  // Accelerometer noise density (m/s²/√Hz)
        float sigma_b = 1.962e-4f,          // Accelerometer bias instability (m/s²)
        float tau_b = 100.0f                // Accelerometer bias time constant (sec), typically 100 for MEMS IMUs
    ) const {
        // Calculate acceleration noise component
        const float BW = sample_rate_hz / 2.0f;  // Noise bandwidth
        const float sigma_a2 = sigma_a_density * sigma_a_density * BW;
        const float T = 1.0f / sample_rate_hz;   // Sample time

        // Calculate bias process noise from Allan variance
        const float q_bias = (sigma_b * sigma_b * T) / tau_b;

        // Calculate process noise components for each state
        const float q_z = (4.0f / 36.0f) * sigma_a2 * powf(T, 6);  // z: displacement integral
        const float q_y = (1.0f / 4.0f) * sigma_a2 * powf(T, 4);   // y: displacement
        const float q_v = sigma_a2 * powf(T, 2);                   // v: velocity

        // Construct the Q matrix
        Matrix4f theoretical_Q;
        theoretical_Q.setZero();
        theoretical_Q(0,0) = q_z;
        theoretical_Q(1,1) = q_y;
        theoretical_Q(2,2) = q_v;
        theoretical_Q(3,3) = q_bias;

        // Ensure the matrix is positive definite
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

    // Schmitt trigger parameters
    float schmitt_positive_threshold;          // Threshold for switching from low to high state
    float schmitt_negative_threshold;          // Threshold for switching from high to low state
    float schmitt_velocity_threshold;          // Threshold for switching from high to low state, abs(velocity)
    float schmitt_debounce_time;               // Debounce time (sec)
    float zero_correction_gain;                // [0-1] how strongly to correct
    SchmittTriggerState schmitt_state;         // Current state of the Schmitt trigger
    bool zero_crossing_detected = false;
    float zero_crossing_last_interval = 300.0f;  // Last time period between two zero crossings (sec)
    float zero_crossing_time_since = 0.0f;       // Time since last zero crossing (sec)
    
    // Separate observation noise for zero-correction
    float R_heave = 500.0f;
    float R_velocity = 1000.0f;

    void zeroCrossingCorrection(float delta_t) {
        // Soft correction - only move partially toward zero (controlled by zero_correction_gain)
        Matrix24f H_special;
        H_special << 0, 1, 0, 0,  // Observe heave
                     0, 0, 1, 0;  // Observe velocity
        
        // Target values (partial correction toward zero)
        Vector2f z;
        const float freq_guess = zero_crossing_last_interval > 3.0f ? 
            M_PI / zero_crossing_last_interval :
            2.0f * M_PI * 0.07f;  //  rad/s
        float new_y = x(1);    
        float new_v = sqrtf(x(2) * x(2) + (freq_guess * x(1)) * (freq_guess * x(1)));  // energy conservation
        if (new_v > 3.0f) {  // clamp velocity to reasonable limit (m/s) for ocean waves
            new_v = 3.0f;
        }
        if (x(2) < 0.0f) {
            new_v = -new_v;
        }
        
        z << new_y * (1.0f - zero_correction_gain),         // Target: reduce heave by gain%
             x(2) + (new_v - x(2)) * zero_correction_gain;  // Target: increase speed by gain%
        
        Vector2f y = z - H_special * x;
        Matrix2f Sz = H_special * P * H_special.transpose();
        Sz(0,0) += R_heave;
        Sz(1,1) += R_velocity;
        enforcePositiveDefiniteness(Sz);  // Ensure Sz remains symmetric and positive definite
        
        // Check for numerical stability before inversion
        if (Sz.determinant() > MIN_DIVISOR_VALUE) {
            Matrix2f Sz_inv = Sz.inverse();

            // Mahalanobis gating
            float mahalanobis_distance_sq = y.transpose() * Sz_inv * y;
            const float GATING_THRESHOLD = 13.0f;  // low rejection rate for 2D

            if (mahalanobis_distance_sq < GATING_THRESHOLD) {
                // Accept correction
                Matrix42f K = P * H_special.transpose() * Sz_inv;
                x = x + K * y;

                // Joseph form update for covariance
                Matrix4f JI_KH = I - K * H_special;
                P = JI_KH * P * JI_KH.transpose() + K * Sz * K.transpose();
                enforcePositiveDefiniteness(P);  // Ensure P remains symmetric and positive definite
             }
        }
        // Reset detection flag
        zero_crossing_detected = false;
    }

    // Helper function to enforce symmetry on a matrix
    void enforceSymmetry(Matrix4f& mat) const {
        mat = 0.5f * (mat + mat.transpose());
    }

    void enforceSymmetry(Matrix2f& mat) const {
        mat = 0.5f * (mat + mat.transpose());
    }

    // Helper function to enforce positive definiteness on a matrix
    void enforcePositiveDefiniteness(Matrix4f& mat) const {
        // First ensure symmetry
        enforceSymmetry(mat);

        Eigen::LLT<Matrix4f> llt(mat);  // Cholesky
        float epsilon = 1e-7f;
        while (llt.info() == Eigen::NumericalIssue && epsilon < 0.01f) {
            mat += epsilon * Matrix4f::Identity();
            llt.compute(mat);
            epsilon *= 10;
        }
    }

    void enforcePositiveDefiniteness(Matrix2f& mat) const {
        // First ensure symmetry
        enforceSymmetry(mat);

        Eigen::LLT<Matrix2f> llt(mat);  // Cholesky
        float epsilon = 1e-7f;
        while (llt.info() == Eigen::NumericalIssue && epsilon < 0.01f) {
            mat += epsilon * Matrix2f::Identity();
            llt.compute(mat);
            epsilon *= 10;
        }
    }
};

typedef KalmanForWaveBasic::State KalmanForWaveBasicState;

#endif
