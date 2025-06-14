#pragma once

/*
  A particle filter which tracks two most characteristic frequencies of a signal.

  See: https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/43
  
*/

#include <ArduinoEigenDense.h>

static constexpr int PF_NUM_PARTICLES = 500;
static constexpr float PF_FREQ_MIN = 0.04f;  // 0.04 Hz ~ 25s waves
static constexpr float PF_FREQ_MAX = 0.5f;   // 0.5 Hz ~ 2s waves
static constexpr float PF_AMP_MIN = 0.01f;   // Minimum amplitude (m/s²)
static constexpr float PF_AMP_MAX = 10.0f;    // Maximum amplitude (m/s²)
static constexpr int PF_SEED = 777;

// State: [f1, f2, B1, C1, B2, C2] (6D, but effectively 4D since B/C coupled)
typedef Eigen::Matrix<float, PF_NUM_PARTICLES, 6> ParticleMatrix;
typedef Eigen::Matrix<float, PF_NUM_PARTICLES, 1> WeightVector;
typedef Eigen::Matrix<float, 2, 1> Vector2f;

class FrequencyTrackingParticleFilter {
private:
    ParticleMatrix particles;
    WeightVector weights;
    uint32_t noise_state = PF_SEED;

    // Deterministic random numbers (for reproducibility)
    float uniformRand() {
        noise_state = (1664525 * noise_state + 1013904223);
        return (noise_state / 4294967296.0f);
    }

    float normalRand() {
        static bool hasSpare = false;
        static float spare;
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        hasSpare = true;
        float u, v, s;
        do {
            u = uniformRand() * 2.0f - 1.0f;
            v = uniformRand() * 2.0f - 1.0f;
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        s = sqrtf(-2.0f * logf(s) / s);
        spare = v * s;
        return u * s;
    }

    void enforceFrequencyOrdering(int i) {
        // Always enforce ordering with minimum separation
        if (particles(i, 0) > particles(i, 1)) {
            std::swap(particles(i, 0), particles(i, 1));
            std::swap(particles(i, 2), particles(i, 4));
            std::swap(particles(i, 3), particles(i, 5));
        }
        
        // Enforce minimum separation
        if (particles(i, 1) - particles(i, 0) < 0.05f) {
            particles(i, 1) = particles(i, 0) + 0.05f;
        }
    }

    float constraint(float value, float min, float max) {
        return (value < min) ? min : (value > max) ? max : value;
    }

public:
    FrequencyTrackingParticleFilter() : 
        particles(ParticleMatrix::Zero()),
        weights(WeightVector::Constant(1.0f / PF_NUM_PARTICLES)) {
        initializeParticles();
    }

    void initializeParticles() {
        noise_state = PF_SEED;
        float log_min = logf(PF_FREQ_MIN);
        float log_max = logf(PF_FREQ_MAX);

        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Initialize fundamental frequency (log-uniform)
            particles(i, 0) = expf(uniformRand() * (log_max - log_min) + log_min);
            
            // Initialize second frequency with higher probability of being harmonic
            if (uniformRand() < 0.7f) { // 70% chance of harmonic
                particles(i, 1) = particles(i, 0) * (1 + round(uniformRand() * 3)); // 1x-4x
            } else {
                particles(i, 1) = expf(uniformRand() * (log_max - log_min) + log_min);
            }
            
            // Enforce minimum frequency separation
            if (fabs(particles(i, 0) - particles(i, 1)) < 0.1f) {
                particles(i, 1) = particles(i, 0) + 0.1f;
            }
            enforceFrequencyOrdering(i);

            // Initialize quadrature amplitudes (B_i, C_i)
            for (int j = 2; j < 6; j += 2) {
                float amp = uniformRand() * (PF_AMP_MAX - PF_AMP_MIN) + PF_AMP_MIN;
                float phi = uniformRand() * 2 * M_PI;
                particles(i, j)   = amp * cosf(phi);  // B_i
                particles(i, j+1) = amp * sinf(phi);  // C_i
            }
        }
    }

    void resample() {
        // Cumulative weights
        Eigen::Matrix<float, PF_NUM_PARTICLES, 1> cum_weights;
        cum_weights(0) = weights(0);
        for (int i = 1; i < PF_NUM_PARTICLES; ++i) {
            cum_weights(i) = cum_weights(i-1) + weights(i);
        }

        // Systematic resampling
        ParticleMatrix new_particles;
        float step = 1.0f / PF_NUM_PARTICLES;
        float u = uniformRand() * step;
        int i = 0;

        for (int j = 0; j < PF_NUM_PARTICLES; ++j) {
            while (u > cum_weights(i) && i < PF_NUM_PARTICLES - 1) i++;
            new_particles.row(j) = particles.row(i);
            u += step;
        }

        particles = new_particles;
        weights.setConstant(1.0f / PF_NUM_PARTICLES);
    }

    void process(float measurement, float time, float dt,
                float sigma_f = 0.02f,      // Frequency noise
                float sigma_bc = 0.01f,     // Amplitude (B/C) noise
                float measurement_noise_std = 0.08f) {
        
        // --- Prediction Step ---
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Add noise to frequencies (log-space for better behavior)
            float log_f1 = logf(particles(i, 0)) + normalRand() * sigma_f;
            float log_f2 = logf(particles(i, 1)) + normalRand() * sigma_f;
            particles(i, 0) = expf(constraint(log_f1, logf(PF_FREQ_MIN), logf(PF_FREQ_MAX)));
            particles(i, 1) = expf(constraint(log_f2, logf(PF_FREQ_MIN), logf(PF_FREQ_MAX)));
            
            // Only enforce ordering if frequencies are too close
            if (fabs(particles(i, 0) - particles(i, 1)) < 0.05f) {
                enforceFrequencyOrdering(i);
            }
    
            // Add noise to quadrature amplitudes with constraints
            for (int j = 2; j < 6; ++j) {
                particles(i, j) = constraint(
                    particles(i, j) + normalRand() * sigma_bc,
                    -PF_AMP_MAX, PF_AMP_MAX
                );
            }
        }
    
        // --- Update Step ---
        float max_weight = 0.0f;
        
        // First pass: measurement update
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Quadrature signal model
            float y_pred = 
                particles(i, 2) * sinf(2 * M_PI * particles(i, 0) * time) +
                particles(i, 3) * cosf(2 * M_PI * particles(i, 0) * time) +
                particles(i, 4) * sinf(2 * M_PI * particles(i, 1) * time) +
                particles(i, 5) * cosf(2 * M_PI * particles(i, 1) * time);
    
            float residual = measurement - y_pred;  
            weights(i) = expf(-0.5f * residual * residual / 
                             (measurement_noise_std * measurement_noise_std));
            max_weight = fmaxf(max_weight, weights(i));
        }
    
        // Second pass: apply harmonic constraints
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            float ratio = particles(i, 1) / particles(i, 0);
            float nearest_integer = roundf(ratio);
            float harmonicity = fabs(ratio - nearest_integer);
            
            // Reward harmonic relationships
            if (harmonicity < 0.2f) {
                weights(i) *= (1.0f + (0.2f - harmonicity)); // Moderate boost
            }
            
            // Penalize frequencies being too close
            float freq_diff = particles(i, 1) - particles(i, 0);
            if (freq_diff < 0.05f) {
                weights(i) *= (0.2f + 0.8f * (freq_diff / 0.05f)); // Smooth penalty
            }
        }
        
        // Normalization with log-sum-exp trick
        float weight_sum_exp = 0.0f;
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            weights(i) = expf(weights(i) - max_weight);
            weight_sum_exp += weights(i);
        }
        
        // Normalize weights
        if (weight_sum_exp < 1e-10f) {
            weights.setConstant(1.0f / PF_NUM_PARTICLES);
        } else {
            weights /= weight_sum_exp;
        }
    
        // Less aggressive resampling condition
        float effective_sample_size = 1.0f / weights.array().square().sum();
        if (effective_sample_size < PF_NUM_PARTICLES / 2.0f) { // Changed from /3.0f
            resample();
        }
    }
    
    void estimate(Vector2f& freqs, Vector2f& displacement_amps) {
        // Find indices of top weighted particles
        std::vector<size_t> indices(PF_NUM_PARTICLES);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 100, indices.end(),
            [&](size_t a, size_t b) { return weights(a) > weights(b); });
    
        // Calculate weighted mean using top particles
        freqs = Vector2f::Zero();
        Vector2f accel_amps = Vector2f::Zero();
        float total_weight = 0.0f;
        
        for (int j = 0; j < 100; ++j) {
            int i = indices[j];
            float w = weights(i);
            freqs += w * particles.block<1, 2>(i, 0).transpose();
            
            float a1 = sqrtf(particles(i, 2)*particles(i, 2) + particles(i, 3)*particles(i, 3));
            float a2 = sqrtf(particles(i, 4)*particles(i, 4) + particles(i, 5)*particles(i, 5));
            accel_amps += w * Vector2f(a1, a2);
            total_weight += w;
        }
        
        // Normalize
        if (total_weight > 1e-10f) {
            freqs /= total_weight;
            accel_amps /= total_weight;
        } else {
            // Fallback: use simple average if weights are degenerate
            freqs = particles.block<100, 2>(0, 0).colwise().mean();
            accel_amps = Vector2f(
                sqrtf(particles.block<100, 1>(0, 2).array().square().mean() + 
                     particles.block<100, 1>(0, 3).array().square().mean()),
                sqrtf(particles.block<100, 1>(0, 4).array().square().mean() + 
                     particles.block<100, 1>(0, 5).array().square().mean())
            );
        }
    
        // Convert to displacement amplitudes with regularization
        float omega1_sq = powf(2 * M_PI * std::max(freqs(0), PF_FREQ_MIN), 2) + 1e-5f;
        float omega2_sq = powf(2 * M_PI * std::max(freqs(1), PF_FREQ_MIN), 2) + 1e-5f;
        displacement_amps = Vector2f(
            accel_amps(0) / omega1_sq,
            accel_amps(1) / omega2_sq
        );
    
        // Sort by displacement amplitude (descending)
        if (displacement_amps(0) < displacement_amps(1)) {
            std::swap(displacement_amps(0), displacement_amps(1));
            std::swap(freqs(0), freqs(1));
        }
    }
};
