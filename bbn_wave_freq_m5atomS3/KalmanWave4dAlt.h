#ifndef KALMAN_WAVE_4D_ALT_H
#define KALMAN_WAVE_4D_ALT_H

/*
  Modified Kalman filter for wave displacement estimation
  Now with proper bias estimation by removing acceleration from state vector
  while maintaining physical relationship a = k_hat*y
*/

#include <ArduinoEigenDense.h>

class KalmanWave4dAlt {
public:
    // Updated type aliases for 4-state system
    using Vector4f = Eigen::Matrix<float, 4, 1>;
    using Matrix4f = Eigen::Matrix<float, 4, 4>;
    using Vector2f = Eigen::Matrix<float, 2, 1>;
    using Matrix2f = Eigen::Matrix<float, 2, 2>;
    using Matrix24f = Eigen::Matrix<float, 2, 4>;
    using Matrix42f = Eigen::Matrix<float, 4, 2>;

    struct State {
        float displacement_integral = 0.0f;
        float heave = 0.0f;
        float vert_speed = 0.0f;
        float accel_bias = 0.0f;  // Only bias remains as estimated parameter

        // Helper method to get acceleration (calculated from wave model)
        float get_acceleration(float k_hat) const {
            return k_hat * heave;
        }
    };

    struct FilterMetrics {
        float innovation_magnitude = 0.0f;
        float innovation_normalized = 0.0f;
        float covariance_trace = 0.0f;
        float max_covariance = 0.0f;
        float condition_number = 0.0f;
        float position_std_dev = 0.0f;
        float velocity_std_dev = 0.0f;
        float bias_std_dev = 0.0f;
        float residual_accel = 0.0f;
    };

    KalmanWave4dAlt(float q0 = 1e+1f, float q1 = 1e-4f, float q2 = 1e-2f, float q3 = 1e-5f) {
        initialize(q0, q1, q2, q3);
    }

    void initialize(float q0, float q1, float q2, float q3) {
        // State vector initialization
        x.setZero();

        // Initial covariance - large uncertainty
        P.setIdentity();
        P *= 1.0f;

        // Process noise covariance (diagonal)
        Q.setZero();
        Q.diagonal() << q0, q1, q2, q3;

        // Measurement noise covariance
        R << 0.01f,  0.0f,   // Displacement integral noise
             0.0f,   1.0f;    // Acceleration noise (m/s²)²

        // Measurement model will be updated in update() with current k_hat
        // H << 1.0f, 0.0f, 0.0f, 0.0f,   // Measures displacement integral
        //      0.0f, k_hat, 0.0f, 1.0f;  // Measures acceleration (k_hat*y + bias)

        // Reset metrics
        resetMetrics();
    }

    void update(float measured_accel, float k_hat, float delta_t) {
        measured_accel -= x(3);  // trick to accelerate convergence of displacement

        // 1. Subtract current bias estimate from measurement
        float corrected_accel = measured_accel - x(3);  // x(3) = bias estimate

        // 2. Update state transition matrix
        updateStateTransition(k_hat, delta_t);

        // 3. Prediction step
        predict();

        // 4. Prepare measurements:
        Vector2f z;
        z << 0.0f,                      // Position reference (reset)
             corrected_accel;           // Bias-corrected acceleration

        // 5. Measurement matrix - now only observes wave dynamics
        Matrix24f H_mod;
        H_mod << 1.0f, 0.0f, 0.0f, 0.0f,   // Position measurement
             0.0f, k_hat, 0.0f, 0.0f;   // Wave acceleration observation

        // 6. Correction with Joseph form
        correctJoseph(z, H_mod);

        // 7. Now update bias estimate separately
        updateBiasEstimate(measured_accel, k_hat);

        // 8. Update metrics
        updateMetrics(z, k_hat, H_mod);
    }

    void updateBiasEstimate(float measured_accel, float k_hat) {
        // Simple IIR filter for bias estimation
        float wave_accel = k_hat * x(1);
        float bias_innovation = measured_accel - wave_accel - x(3);

        // Adaptive gain - larger when uncertainty is high
        float bias_gain = P(3,3) / (P(3,3) + R(1,1));
        x(3) += bias_gain * bias_innovation;

        // Reduce covariance for bias
        P(3,3) *= 0.99f;  // Slowly reduce uncertainty
    }

    State getState() const {
        State s;
        s.displacement_integral = x(0);
        s.heave = x(1);
        s.vert_speed = x(2);
        s.accel_bias = x(3);
        return s;
    }

    FilterMetrics getMetrics() const {
        return metrics;
    }

    void initState(const State& s0) {
        x(0) = s0.displacement_integral;
        x(1) = s0.heave;
        x(2) = s0.vert_speed;
        x(3) = s0.accel_bias;
    }

    void initMeasurementNoise(float r0, float r1) {
        R.setZero();
        R.diagonal() << r0, r1;
    }

    float calculate_k_hat(float linear_freq) const {
        return -pow(2.0f * M_PI * linear_freq, 2);
    }

    void resetMetrics() {
        metrics = FilterMetrics();
    }

private:
    Vector4f x;     // State vector [z, y, v, a_hat]
    Matrix4f P;     // Covariance matrix
    Matrix4f Q;     // Process noise
    Matrix2f R;     // Measurement noise
    Matrix4f F;     // State transition matrix
    FilterMetrics metrics;

    void updateStateTransition(float k_hat, float delta_t) {
        const float T = delta_t;
        const float T2 = T * T;
        const float T3 = T2 * T;
        const float kT = k_hat * T;
        const float kT2 = k_hat * T2;
        const float kT3 = k_hat * T3;

        // State transition matrix with substituted a = k_hat*y
        F << 1.0f,    T + (1.0f/6.0f)*kT3,    0.5f*T2,    0.0f,
             0.0f,    1.0f + 0.5f*kT2,         T,          0.0f,
             0.0f,    kT,                      1.0f,       -T,
             0.0f,    0.0f,                    0.0f,       1.0f;
    }

    void predict() {
        x = F * x;
        P = (F * P * F.transpose()).eval() + Q;
        enforceSymmetry(P);
    }

    void correctJoseph(const Vector2f& z, const Matrix24f& H) {
        const Vector2f y = z - H * x;
        const Matrix2f S = (H * P * H.transpose() + R).eval();
        const Matrix42f K = P * H.transpose() * S.inverse();

        x += K * y;

        const Matrix4f I = Matrix4f::Identity();
        const Matrix4f KH = K * H;
        P = ((I - KH) * P * (I - KH).transpose() + K * R * K.transpose()).eval();

        enforceSymmetry(P);
        ensurePositiveDefinite(P);
    }

    void updateMetrics(const Vector2f& z, float k_hat, const Matrix24f& H) {
        const Vector2f innovation = z - H * x;
        const Matrix2f S = H * P * H.transpose() + R;

        metrics.innovation_magnitude = innovation.norm();
        metrics.innovation_normalized = (innovation.transpose() * S.inverse() * innovation)(0,0);
        metrics.covariance_trace = P.trace();
        metrics.max_covariance = P.diagonal().maxCoeff();

        Eigen::JacobiSVD<Matrix4f> svd(P);
        float singular_max = svd.singularValues()(0);
        float singular_min = svd.singularValues()(svd.singularValues().size()-1);
        metrics.condition_number = singular_max / singular_min;

        metrics.position_std_dev = sqrt(P(1,1));        // heave
        metrics.velocity_std_dev = sqrt(P(2,2));        // vertical speed
        metrics.bias_std_dev = sqrt(P(3,3));            // accelerometer bias
        metrics.residual_accel = innovation(1);          // acceleration residual
    }

    void enforceSymmetry(Matrix4f& mat) {
        Matrix4f symm = 0.5f * (mat + mat.transpose());
        mat = symm;
    }

    void ensurePositiveDefinite(Matrix4f& mat) {
        Eigen::LDLT<Matrix4f> ldlt(mat);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
            mat.diagonal().array() += 1e-9f;
        }
    }
};

typedef KalmanWave4dAlt::State KalmanWave4dAltState;
typedef KalmanWave4dAlt::FilterMetrics KalmanWave4dAltMetrics;

#endif // KALMAN_WAVE_4D_ALT_H
