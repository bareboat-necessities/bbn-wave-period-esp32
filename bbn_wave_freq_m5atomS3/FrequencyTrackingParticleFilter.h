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
        // Only swap if frequencies cross with some minimum separation
        if (particles(i, 0) > particles(i, 1) && 
            fabs(particles(i, 0) - particles(i, 1)) > 0.05f) {
            std::swap(particles(i, 0), particles(i, 1));
            std::swap(particles(i, 2), particles(i, 4));
            std::swap(particles(i, 3), particles(i, 5));
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
            // Initialize frequencies (log-uniform)
            particles(i, 0) = expf(uniformRand() * (log_max - log_min) + log_min);  // f1
            particles(i, 1) = expf(uniformRand() * (log_max - log_min) + log_min);  // f2
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
            if (fabs(particles(i, 0) - particles(i, 1)) < 0.01f) {
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

        // Harmonic Constraints
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Penalize particles where f2 is not roughly a multiple of f1
            float ratio = particles(i, 1) / particles(i, 0);
            float nearest_integer = roundf(ratio);
            if (fabs(ratio - nearest_integer) > 0.1f) {
                weights(i) *= 0.1f; // Strongly downweight non-harmonic particles
            }
        }     
      
        // --- Update Step ---
        float sum_weights = 0.0f;
        float max_weight = 0.0f;
        
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Quadrature signal model: B1*sin(2πf1t) + C1*cos(2πf1t) + B2*sin(2πf2t) + C2*cos(2πf2t)
            float y_pred = 
                particles(i, 2) * sinf(2 * M_PI * particles(i, 0) * time) +  // B1*sin(2πf1t)
                particles(i, 3) * cosf(2 * M_PI * particles(i, 0) * time) +  // C1*cos(2πf1t)
                particles(i, 4) * sinf(2 * M_PI * particles(i, 1) * time) +  // B2*sin(2πf2t)
                particles(i, 5) * cosf(2 * M_PI * particles(i, 1) * time);   // C2*cos(2πf2t)

            // Add a term for potential harmonic relationship
            if (fabs(particles(i, 1)/particles(i, 0) - 2.0f) < 0.2f) {
                y_pred += 0.5f * ( // Example: second harmonic might be weaker
                    particles(i, 4) * sinf(4 * M_PI * particles(i, 0) * time) +
                    particles(i, 5) * cosf(4 * M_PI * particles(i, 0) * time)
                );
            }
          
            float residual = measurement - y_pred;  
            float normalized_residual = residual / measurement_noise_std;
            weights(i) = 1.0f / (1.0f + 0.5f * normalized_residual * normalized_residual);
            max_weight = fmaxf(max_weight, weights(i));
        }
        
        // Log-sum-exp trick for numerical stability
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
        if (effective_sample_size < PF_NUM_PARTICLES / 3.0f) {
            resample();
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

    void estimate(Vector2f& freqs, Vector2f& displacement_amps) {
        freqs = Vector2f::Zero();
        displacement_amps = Vector2f::Zero();
        Vector2f accel_amps = Vector2f::Zero();
    
        // First calculate weighted means
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Frequencies
            freqs += weights(i) * particles.block<1, 2>(i, 0).transpose();
            
            // Acceleration amplitudes (A_accel = sqrt(B² + C²))
            float a1 = sqrtf(particles(i, 2)*particles(i, 2) + particles(i, 3)*particles(i, 3));
            float a2 = sqrtf(particles(i, 4)*particles(i, 4) + particles(i, 5)*particles(i, 5));
            accel_amps += weights(i) * Vector2f(a1, a2);
        }
    
        // Convert to from vertical acceleration to displacement amplitudes (A_disp = A_accel/ω²)
        float omega1 = 2 * M_PI * freqs(0);
        float omega2 = 2 * M_PI * freqs(1);
        displacement_amps = Vector2f(
            accel_amps(0) / (omega1 * omega1),
            accel_amps(1) / (omega2 * omega2)
        );
    
        // Sort by displacement amplitude (descending)
        if (displacement_amps(0) < displacement_amps(1)) {
            std::swap(displacement_amps(0), displacement_amps(1));
            std::swap(freqs(0), freqs(1));
        }
    }
};
