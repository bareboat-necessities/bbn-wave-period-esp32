#ifndef KalmanForWaveBasic_h
#define KalmanForWaveBasic_h

#include <ArduinoEigenDense.h>

class KalmanForWaveBasic {
private:
    // State vector: [displacement_integral, heave, vert_speed, accel_bias]
    Eigen::Vector4f x;
    
    // State transition matrix
    Eigen::Matrix4f F;
    
    // Input transition matrix
    Eigen::Vector4f B;
    
    // Process noise covariance
    Eigen::Matrix4f Q;
    
    // Observation matrix
    Eigen::RowVector4f H;
    
    // Observation noise
    float R;
    
    // State covariance
    Eigen::Matrix4f P;
    
    // Identity matrix
    Eigen::Matrix4f I;

public:
    struct State {
        float displacement_integral; // displacement integral
        float heave;                // vertical displacement
        float vert_speed;           // vertical velocity
        float accel_bias;           // accel bias
    };

    KalmanForWaveBasic(float q0, float q1, float q2, float q3, float observation_noise = 0.01f) {
        initialize(q0, q1, q2, q3, observation_noise);
    }

    void initialize(float q0, float q1, float q2, float q3, float observation_noise) {
        // Initialize state vector
        x.setZero();
        
        // Initialize state covariance
        P.setIdentity();
        
        // Initialize observation matrix
        H << 1, 0, 0, 0;
        
        // Initialize observation noise
        R = observation_noise;
        
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
        // Update state transition matrix
        F << 1.0f, delta_t, 0.5f * delta_t * delta_t, (-1.0f/6.0f) * delta_t * delta_t * delta_t,
             0.0f, 1.0f,    delta_t,                  -0.5f * delta_t * delta_t,
             0.0f, 0.0f,    1.0f,                      -delta_t,
             0.0f, 0.0f,    0.0f,                      1.0f;
        
        // Update input transition matrix
        B << (1.0f/6.0f) * delta_t * delta_t * delta_t,
             0.5f * delta_t * delta_t,
             delta_t,
             0.0f;
        
        // Predict state
        x = F * x + B * accel;
        
        // Predict state covariance
        P = F * P * F.transpose() + Q;
    }

    void correct() {
        // Measurement is always zero for displacement integral
        float z = 0.0f;
        
        // Calculate innovation
        float y = z - H * x;
        
        // Calculate innovation covariance
        float S = H * P * H.transpose() + R;
        
        // Calculate Kalman gain
        Eigen::Vector4f K = P * H.transpose() / S;
        
        // Update state estimate
        x = x + K * y;
        
        // Update state covariance
        P = (I - K * H) * P;
    }

    void step(float accel, float delta_t, State& state) {
        predict(accel, delta_t);
        correct();
        
        state.displacement_integral = x(0);
        state.heave = x(1);
        state.vert_speed = x(2);
        state.accel_bias = x(3);
    }

    const State getState() const {
        return State{x(0), x(1), x(2), x(3)};
    }
};


typedef KalmanForWaveBasic::State KalmanForWaveBasicState;

#endif
