#pragma once
#include <cmath>
#include <algorithm>

template <typename Real = float>
class FrequencySmoother {
public:
    /**
     * Constructor with tuned parameters for ocean waves
     * @param q_short Process noise for short periods (fast waves)
     * @param q_long Process noise for long periods (slow waves)
     * @param T_min Minimum wave period in seconds
     * @param T_max Maximum wave period in seconds
     * @param measurement_noise Measurement noise (period jitter)
     * @param estimated_error Initial estimation uncertainty
     * @param rel_threshold Relative deviation threshold for outlier detection
     * @param noise_scale Factor to inflate R for suspicious samples
     */
    FrequencySmoother(Real q_short_ = Real(0.0001),
                      Real q_long_  = Real(0.0025),
                      Real T_min_   = Real(0.33),
                      Real T_max_   = Real(15.0),
                      Real measurement_noise_ = Real(0.05),
                      Real estimated_error_   = Real(10.0),
                      Real rel_threshold_     = Real(0.3),
                      Real noise_scale_       = Real(15.0))
        : q_short(q_short_), q_long(q_long_),
          T_min(T_min_), T_max(T_max_),
          r(measurement_noise_),
          p(estimated_error_),
          T(Real(0)),
          rel_thresh(rel_threshold_),
          r_scale(noise_scale_)
    {}

    /// Set initial frequency in Hz
    void setInitial(Real freq_hz) {
        if (freq_hz > Real(0)) {
            T = Real(1) / freq_hz;
        } else {
            T = Real(0);
        }
    }

    /// Update with new frequency measurement (Hz) and return smoothed frequency (Hz)
    Real update(Real freq_measured_hz) {
        if (freq_measured_hz <= Real(0)) {
            return (T > Real(0)) ? (Real(1) / T) : Real(0);
        }

        Real T_meas = Real(1) / freq_measured_hz;

        // Interpolate process noise q based on current period estimate T
        Real q = interpolateProcessNoise(T);

        // Determine if suspicious measurement: relative difference in period
        Real r_eff = r;
        if (T > Real(0)) {
            Real rel_diff = std::fabs(T_meas - T) / T;
            if (rel_diff > rel_thresh) {
                r_eff *= r_scale; // inflate measurement noise for outlier
            }
        }

        // Kalman update
        Real k_gain = p / (p + r_eff);
        Real T_est = T + k_gain * (T_meas - T);
        p = (Real(1) - k_gain) * p + std::fabs(T - T_est) * q;
        T = T_est;

        return Real(1) / T; // return frequency
    }

private:
    Real q_short;
    Real q_long;
    Real T_min;
    Real T_max;

    Real r;           // measurement noise
    Real p;           // estimation error covariance
    Real T;           // current period estimate

    Real rel_thresh;  // relative outlier threshold
    Real r_scale;     // outlier noise scaling factor

    Real interpolateProcessNoise(Real period) const {
        if (period <= T_min) return q_short;
        if (period >= T_max) return q_long;
        // Linear interpolation
        Real alpha = (period - T_min) / (T_max - T_min);
        return q_short + alpha * (q_long - q_short);
    }
};
