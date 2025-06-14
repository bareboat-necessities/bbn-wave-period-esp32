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
                float sigma_f = 0.01f,      // Reduced frequency noise
                float sigma_bc = 0.005f,    // Reduced amplitude noise
                float measurement_noise_std = 0.08f) {
        
        // --- Prediction Step ---
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            // More conservative frequency updates
            particles(i, 0) = constraint(
                particles(i, 0) * expf(normalRand() * sigma_f),
                PF_FREQ_MIN, PF_FREQ_MAX);
            particles(i, 1) = constraint(
                particles(i, 1) * expf(normalRand() * sigma_f),
                PF_FREQ_MIN, PF_FREQ_MAX);
            
            // Stronger enforcement of frequency ordering
            if (particles(i, 0) >= particles(i, 1) - 0.05f) {
                particles(i, 1) = particles(i, 0) + 0.05f;
                // Randomize the higher frequency more when forced to separate
                if (uniformRand() < 0.3f) {
                    particles(i, 1) = constraint(
                        particles(i, 0) + 0.05f + uniformRand() * 0.3f,
                        PF_FREQ_MIN, PF_FREQ_MAX);
                }
            }
    
            // Slower amplitude changes
            for (int j = 2; j < 6; ++j) {
                particles(i, j) = constraint(
                    particles(i, j) * (1.0f + normalRand() * sigma_bc),
                    -PF_AMP_MAX, PF_AMP_MAX);
            }
        }
    
        // --- Update Step ---
        float max_log_weight = -FLT_MAX;
        Eigen::Array<float, PF_NUM_PARTICLES, 1> log_weights;
    
        // Measurement update only - removed harmonic constraints from update
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            float y_pred = 
                particles(i, 2) * sinf(2 * M_PI * particles(i, 0) * time) +
                particles(i, 3) * cosf(2 * M_PI * particles(i, 0) * time) +
                particles(i, 4) * sinf(2 * M_PI * particles(i, 1) * time) +
                particles(i, 5) * cosf(2 * M_PI * particles(i, 1) * time);
    
            float residual = measurement - y_pred;
            log_weights(i) = -0.5f * residual * residual / 
                           (measurement_noise_std * measurement_noise_std);
            max_log_weight = fmaxf(max_log_weight, log_weights(i));
        }
    
        // Convert log weights to regular weights
        weights = (log_weights - max_log_weight).exp();
        float sum_weights = weights.sum();
    
        // Normalize weights
        if (sum_weights > 1e-10f) {
            weights /= sum_weights;
        } else {
            weights.setConstant(1.0f / PF_NUM_PARTICLES);
        }
    
        // Very conservative resampling
        float effective_sample_size = 1.0f / weights.array().square().sum();
        if (effective_sample_size < PF_NUM_PARTICLES / 5.0f) {  // Much less frequent
            resample();
        }
    }
    
    void estimate(Vector2f& freqs, Vector2f& displacement_amps) {
        // Cluster particles by frequency pairs to find dominant modes
        struct Cluster {
            Vector2f freq_sum = Vector2f::Zero();
            Vector2f amp_sum = Vector2f::Zero();
            float weight_sum = 0;
            int count = 0;
        };
        std::vector<Cluster> clusters;
    
        // Find significant frequency pairs using weighted k-means
        for (int i = 0; i < PF_NUM_PARTICLES; ++i) {
            if (weights(i) < 1.0f / (10.0f * PF_NUM_PARTICLES)) continue;
    
            Vector2f current_freqs = particles.block<1, 2>(i, 0).transpose();
            bool assigned = false;
    
            // Try to assign to existing cluster
            for (auto& cluster : clusters) {
                Vector2f cluster_mean = cluster.freq_sum / cluster.weight_sum;
                if ((current_freqs - cluster_mean).norm() < 0.05f) {
                    cluster.freq_sum += weights(i) * current_freqs;
                    cluster.amp_sum += weights(i) * Vector2f(
                        sqrtf(particles(i, 2)*particles(i, 2) + particles(i, 3)*particles(i, 3)),
                        sqrtf(particles(i, 4)*particles(i, 4) + particles(i, 5)*particles(i, 5)));
                    cluster.weight_sum += weights(i);
                    cluster.count++;
                    assigned = true;
                    break;
                }
            }
    
            // Create new cluster if no match
            if (!assigned && clusters.size() < 5) {
                Cluster new_cluster;
                new_cluster.freq_sum = weights(i) * current_freqs;
                new_cluster.amp_sum = weights(i) * Vector2f(
                    sqrtf(particles(i, 2)*particles(i, 2) + particles(i, 3)*particles(i, 3)),
                    sqrtf(particles(i, 4)*particles(i, 4) + particles(i, 5)*particles(i, 5)));
                new_cluster.weight_sum = weights(i);
                new_cluster.count = 1;
                clusters.push_back(new_cluster);
            }
        }
    
        // Sort clusters by total weight
        std::sort(clusters.begin(), clusters.end(),
            [](const Cluster& a, const Cluster& b) {
                return a.weight_sum > b.weight_sum;
            });
    
        // Get top two clusters
        if (clusters.size() >= 2) {
            freqs(0) = clusters[0].freq_sum(0) / clusters[0].weight_sum;
            freqs(1) = clusters[1].freq_sum(1) / clusters[1].weight_sum;
            
            Vector2f accel_amps(
                clusters[0].amp_sum(0) / clusters[0].weight_sum,
                clusters[1].amp_sum(1) / clusters[1].weight_sum);
    
            // Convert to displacement amplitudes
            float omega1 = 2 * M_PI * freqs(0);
            float omega2 = 2 * M_PI * freqs(1);
            displacement_amps = Vector2f(
                accel_amps(0) / (omega1 * omega1),
                accel_amps(1) / (omega2 * omega2));
            
            // Ensure proper ordering by displacement amplitude
            if (displacement_amps(0) < displacement_amps(1)) {
                std::swap(freqs(0), freqs(1));
                std::swap(displacement_amps(0), displacement_amps(1));
            }
        } else if (clusters.size() == 1) {
            // Fallback for single cluster case
            freqs(0) = clusters[0].freq_sum(0) / clusters[0].weight_sum;
            freqs(1) = freqs(0) + 0.1f;  // Add dummy second frequency
            displacement_amps = Vector2f::Constant(0.1f);
        } else {
            // Fallback when no clusters found
            freqs = Vector2f(0.1f, 0.2f);
            displacement_amps = Vector2f(0.1f, 0.05f);
        }
    }
};
