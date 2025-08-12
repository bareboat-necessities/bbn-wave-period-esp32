#pragma once
#include <cmath>
#include <algorithm>

template <typename Real = float>
class FrequencySmoother {
public:
    /**
     * Constructor (defaults tuned for ocean wave dynamics)
     * @param process_noise_calm Process noise for calm seas (small, e.g., 0.0005)
     * @param process_noise_storm Process noise for storm seas (larger, e.g., 0.005)
     * @param measurement_noise Expected zero-crossing jitter (seconds)
     * @param estimated_error Initial estimation uncertainty
     * @param rel_threshold Relative deviation threshold for outlier detection (0.3 = 30%)
     * @param noise_scale Factor to inflate R for suspicious samples (e.g., 15.0)
     * @param storm_var_threshold Variance threshold for switching to storm mode
     * @param calm_var_threshold Variance threshold for switching back to calm mode
     */
    FrequencySmoother(Real process_noise_calm  = Real(0.0005),
                      Real process_noise_storm = Real(0.005),
                      Real measurement_noise   = Real(0.05),
                      Real estimated_error     = Real(10.0),
                      Real rel_threshold       = Real(0.3),
                      Real noise_scale         = Real(15.0),
                      Real storm_var_threshold = Real(0.04),  // Hz^2 variance
                      Real calm_var_threshold  = Real(0.01))  // Hz^2 variance
        : q_calm(process_noise_calm),
          q_storm(process_noise_storm),
          r(measurement_noise),
          p(estimated_error),
          T(0),
          rel_thresh(rel_threshold),
          r_scale(noise_scale),
          var_storm(storm_var_threshold),
          var_calm(calm_var_threshold),
          q(process_noise_calm),
          modeStorm(false),
          freqMean(0),
          freqM2(0),
          sampleCount(0)
    {}

    /// Set initial frequency in Hz
    void setInitial(Real freq_hz) {
        if (freq_hz > Real(0)) {
            T = Real(1) / freq_hz;
            resetVarianceTracking(freq_hz);
        } else {
            T = Real(0);
            resetVarianceTracking(Real(0));
        }
    }

    /// Update with new frequency measurement (Hz) and return smoothed frequency (Hz)
    Real update(Real freq_measured_hz) {
        if (freq_measured_hz <= Real(0)) {
            return (T > Real(0)) ? (Real(1) / T) : Real(0);
        }

        // Update rolling variance tracker
        updateVarianceTracking(freq_measured_hz);
        adjustProcessNoise();

        Real T_meas = Real(1) / freq_measured_hz; // convert to period

        // Determine if suspicious measurement
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
    // Process noise settings
    Real q_calm;
    Real q_storm;
    Real q; // active process noise

    // Kalman filter state
    Real r;
    Real p;
    Real T;

    // Outlier handling
    Real rel_thresh;
    Real r_scale;

    // Variance-based adaptation
    Real var_storm;
    Real var_calm;
    bool modeStorm;

    // Welford's algorithm variables for variance estimation
    Real freqMean;
    Real freqM2;
    unsigned sampleCount;

    void resetVarianceTracking(Real initialFreq) {
        freqMean = initialFreq;
        freqM2 = Real(0);
        sampleCount = 1;
    }

    void updateVarianceTracking(Real f) {
        sampleCount++;
        Real delta = f - freqMean;
        freqMean += delta / sampleCount;
        Real delta2 = f - freqMean;
        freqM2 += delta * delta2;
    }

    Real getVariance() const {
        return (sampleCount > 1) ? (freqM2 / (sampleCount - 1)) : Real(0);
    }

    void adjustProcessNoise() {
        Real var = getVariance();
        if (!modeStorm && var > var_storm) {
            // Enter storm mode
            q = q_storm;
            modeStorm = true;
        } else if (modeStorm && var < var_calm) {
            // Back to calm mode
            q = q_calm;
            modeStorm = false;
        }
    }
};
