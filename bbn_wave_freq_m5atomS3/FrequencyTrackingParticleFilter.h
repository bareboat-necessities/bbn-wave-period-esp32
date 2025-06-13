#pragma once

/*
  A particle filter which tracks three most energy carrying frequencies of a signal.

  See: https://github.com/bareboat-necessities/bbn-wave-period-esp32/issues/43
  
*/

#include <ArduinoEigenDense.h>

static constexpr int PF_NUM_PARTICLES = 250;
static constexpr float PF_FREQ_MIN = 0.04f;
static constexpr float PF_FREQ_MAX = 2.0f;
static constexpr float PF_AMP_MIN = 0.1f;
static constexpr float PF_AMP_MAX = 10.0f;
static constexpr float PF_BIAS_MIN = -1.0f;  // Expected bias range
static constexpr float PF_BIAS_MAX = 1.0f;

// Static matrices with bias term added
typedef Matrix<float, PF_NUM_PARTICLES, 10> ParticleMatrix;  // Now 10 params: [f1,f2,f3,A1,A2,A3,φ1,φ2,φ3,bias]
typedef Matrix<float, PF_NUM_PARTICLES, 1> WeightVector;
typedef Matrix<float, 3, 1> Vector3f;

class FrequencyTrackingParticleFilter {
private:
    
    ParticleMatrix particles;
    WeightVector weights;
    uint32_t noise_state;
    uint32_t seed;
    uint32_t noise_state;

    // Deterministic random number generators
    float uniformRand() {
        noise_state = (1664525 * noise_state + 1013904223);
        return (noise_state / 4294967296.0);
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
            u = uniformRand() * 2.0 - 1.0;
            v = uniformRand() * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        return u * s;
    }

    void enforceFrequencyOrdering(int particle_idx) {
        if (particles(particle_idx, 0) > particles(particle_idx, 1)) {
            std::swap(particles(particle_idx, 0), particles(particle_idx, 1));
        }
        if (particles(particle_idx, 1) > particles(particle_idx, 2)) {
            std::swap(particles(particle_idx, 1), particles(particle_idx, 2));
        }
        if (particles(particle_idx, 0) > particles(particle_idx, 1)) {
            std::swap(particles(particle_idx, 0), particles(particle_idx, 1));
        }
    }
    
    float constrain(float value, float min_val, float max_val) {
        return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
    }

public:
    void initializeParticles() {
        noise_state = seed;
        
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Initialize frequencies (sorted)
            particles(i,0) = uniformRand() * (PF_FREQ_MAX - PF_FREQ_MIN) + PF_;
            particles(i,1) = uniformRand() * (PF_FREQ_MAX - PF_FREQ_MIN) + PF_;
            particles(i,2) = uniformRand() * (PF_FREQ_MAX - PF_FREQ_MIN) + PF_;
            enforceFrequencyOrdering(i);
            
            // Initialize amplitudes
            particles(i,3) = uniformRand() * (PF_AMP_MAX - PF_AMP_MIN) + PF_AMP_MIN;
            particles(i,4) = uniformRand() * (PF_AMP_MAX - PF_AMP_MIN) + PF_AMP_MIN;
            particles(i,5) = uniformRand() * (PF_AMP_MAX - PF_AMP_MIN) + PF_AMP_MIN;
            
            // Initialize phases
            particles(i,6) = uniformRand() * 2 * M_PI;
            particles(i,7) = uniformRand() * 2 * M_PI;
            particles(i,8) = uniformRand() * 2 * M_PI;
            
            // Initialize bias
            particles(i,9) = uniformRand() * (PF_BIAS_MAX - PF_BIAS_MIN) + PF_BIAS_MIN;
        }
        
        weights.setConstant(1.0f / PF_NUM_PARTICLES);
    }

    void process(float measurement, float time, float dt,
                float sigma_f = 0.01f, float sigma_a = 0.1f, 
                float sigma_phi = 0.01f, float sigma_bias = 0.005f,
                float measurement_noise_std = 1.0f) {
        // Prediction step
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Add process noise to frequencies
            for (int j = 0; j < 3; ++j) {
                particles(i, j) += normalRand() * sigma_f;
                particles(i, j) = constrain(particles(i, j), PF_FREQ_MIN, PF_FREQ_MAX);
            }
            enforceFrequencyOrdering(i);
            
            // Add noise to amplitudes
            for (int j = 3; j < 6; ++j) {
                particles(i, j) += normalRand() * sigma_a;
                particles(i, j) = constrain(particles(i, j), PF_AMP_MIN, PF_AMP_MAX);
            }
            
            // Add noise to phases
            for (int j = 6; j < 9; ++j) {
                particles(i, j) += normalRand() * sigma_phi;
                particles(i, j) = fmod(particles(i, j), 2 * M_PI);
                if (particles(i, j) < 0) particles(i, j) += 2 * M_PI;
            }
            
            // Add noise to bias (slow drift)
            particles(i,9) += normalRand() * sigma_bias;
            particles(i,9) = constrain(particles(i,9), PF_BIAS_MIN, PF_BIAS_MAX);
        }
        
        // Update step
        float sum_weights = 0.0f;
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // Signal model with bias term
            float y_pred = particles(i,9);  // Start with bias
            for (int j = 0; j < 3; ++j) {
                y_pred += particles(i,j+3) * 
                         fast_sin(2 * M_PI * particles(i,j) * time + particles(i,j+6));
            }
            
            float residual = measurement - y_pred;
            weights(i) *= expf(-0.5f * residual * residual / 
                              (measurement_noise_std * measurement_noise_std));
            sum_weights += weights(i);
        }
        
        // Normalize weights
        if (sum_weights < 1e-300) {
            weights.setConstant(1.0 / PF_NUM_PARTICLES);
        } else {
            weights /= sum_weights;
        }
        
        // Resampling if needed
        float neff = 1.0 / weights.array().square().sum();
        if (neff < PF_NUM_PARTICLES / 2.0) {
            resample();
        }
    }

    void resample() {
        // Calculate cumulative weights
        Matrix<float, PF_NUM_PARTICLES, 1> cum_weights;
        cum_weights(0) = weights(0);
        for (int i = 1; i < NUM_PARTICLES; ++i) {
            cum_weights(i) = cum_weights(i-1) + weights(i);
        }
        
        // Systematic resampling
        ParticleMatrix new_particles;
        float step = 1.0 / PF_NUM_PARTICLES;
        float u = uniformRand() * step;
        
        int i = 0;
        for (int j = 0; j < PF_NUM_PARTICLES; ++j) {
            while (u > cum_weights(i) && i < PF_NUM_PARTICLES-1) {
                i++;
            }
            new_particles.row(j) = particles.row(i);
            u += step;
        }
        
        particles = new_particles;
        weights.setConstant(1.0 / PF_NUM_PARTICLES);
    }

    void estimate(Vector3f& freqs, Vector3f& amps, Vector3f& energies, float& estimated_bias) {
        // Weighted mean calculation
        Vector3f mean_freqs = Vector3f::Zero();
        Vector3f mean_amps = Vector3f::Zero();
        float mean_bias = 0.0f;
        
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            mean_freqs += weights(i) * particles.block<1,3>(i,0).transpose();
            mean_amps += weights(i) * particles.block<1,3>(i,3).transpose();
            mean_bias += weights(i) * particles(i,9);
        }
        
        // Energy calculation (excluding bias)
        energies = mean_amps.array().square() * mean_freqs.array();
        
        // Sort frequencies by energy
        std::vector<std::pair<float, int>> energy_index;
        for (int i = 0; i < 3; ++i) {
            energy_index.emplace_back(energies(i), i);
        }
        std::sort(energy_index.begin(), energy_index.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Prepare outputs
        Vector3f sorted_freqs, sorted_amps, sorted_energies;
        for (int i = 0; i < 3; ++i) {
            int idx = energy_index[i].second;
            sorted_freqs(i) = mean_freqs(idx);
            sorted_amps(i) = mean_amps(idx);
            sorted_energies(i) = energies(idx);
        }
        
        freqs = sorted_freqs;
        amps = sorted_amps;
        energies = sorted_energies;
        estimated_bias = mean_bias;
    }
};
