#pragma once

/*

  Kalman filter which estimates vector direction from X and Y components

*/

class VectorDirectionKF {
private:
    float theta;    // Estimated angle (radians)
    float P;        // Estimation error variance (scalar)
    float Q;        // Process noise variance
    float R;        // Measurement noise variance

public:
    VectorDirectionKF(float initial_angle = 0.0f, 
                     float initial_uncertainty = 1.0f,
                     float process_noise = 0.01f,
                     float measurement_noise = 1.0f) :
        theta(initial_angle),
        P(initial_uncertainty),
        Q(process_noise),
        R(measurement_noise) {}

    void predict() {
        P += Q;
    }

    void update(float x, float y) {
        // Normalize measurements to unit vector
        float meas_mag = sqrt(x*x + y*y);
        float nx = x/meas_mag;
        float ny = y/meas_mag;
        
        // Measurement prediction
        float pred_x = cos(theta);
        float pred_y = sin(theta);
        
        // Angle difference (innovation)
        float delta_theta = atan2(nx*pred_y - ny*pred_x, nx*pred_x + ny*pred_y);
        
        // Simplified Kalman update
        float K = P / (P + R);
        theta += K * delta_theta;
        P *= (1 - K);
        
        // Normalize angle
        theta = atan2(sin(theta), cos(theta));
    }

    float getAngle() const { return theta; }
    float getAngleDegrees() const { return theta * 180.0f / M_PI; }

    // Set process noise (affects smoothing)
    void setProcessNoise(float q) {
        Q = q;
    }

    // Set measurement noise (affects smoothing)
    void setMeasurementNoise(float r) {
        R = r;
    }
};
