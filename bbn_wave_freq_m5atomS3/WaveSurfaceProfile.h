#ifndef WAVE_SURFACE_PROFILE_H
#define WAVE_SURFACE_PROFILE_H

#include <math.h>

/*
  Copyright 2025, Mikhail Grushinskiy
  
  WaveSurfaceProfile - Tracks a rolling buffer of heave samples
  and computes wave phase, crest sharpness, asymmetry, and future prediction.

  - Anchors phase to last zero-upcrossing (heave < 0 → ≥ 0)
  - Uses frequency for phase and prediction
  - Tracks N samples for wave shape reconstruction
*/

struct WaveSample {
  float heave;  // meters
  float time;   // sec
};

template<int N = 128>
class WaveSurfaceProfile {
private:
  WaveSample samples[N];  // circular buffer
  int head = 0;
  int count = 0;
  float freq = 1.0f;       // wave frequency in Hz
  float last_wave_profile_update = 0.0f;  // sec
  float zc_time = 0.0f;  // zero crossing time (sec)

public:
  void reset() {
    head = 0;
    count = 0;
    freq = 1.0f;
  }

  void update(float heave, float new_freq, float t) {
    samples[head] = {heave, t};
    head = (head + 1) % N;
    if (count < N) count++;
    if (new_freq > 0.01f && new_freq < 2.0f) {
      freq = new_freq;
    }
  }

  void updateIfNeeded(float heave, float new_freq, float t) {
    if (new_freq > 0.01f && new_freq < 2.0f) {
      freq = new_freq;
    }
    float period_sec = 1.0f / freq;
    float target_dt = 2.0f * period_sec / N; // seconds per sample
    if (last_wave_profile_update == 0.0f || (t - last_wave_profile_update >= target_dt)) {
      update(heave, freq, t);
      last_wave_profile_update = t;
    }
  }

  float getFrequency() const {
    return freq;
  }

  float getPeriod() const {
    return 1.0f / freq;
  }

  float getPhase(float t) {
    if (!findLatestZeroUpcrossing(t)) return 0.0f;
    float elapsed = t - zc_time;
    float phase = fmodf(elapsed * freq, 1.0f); // phase = t / T = t * f
    return (phase >= 0.0f) ? phase : (phase + 1.0f);
  }

  float getPhaseDegrees(float t) {
    return 360.0f * getPhase(t);
  }

  bool findLatestZeroUpcrossing(float t) {
    return findLatestZeroCrossing(true, t);
  }

  bool findLatestZeroDowncrossing(float t) {
    return findLatestZeroCrossing(false, t);
  }

  bool findLatestZeroCrossing(bool upcrossing, float t) {
    const float EPSILON = 1e-6f; // Small threshold for floating-point comparisons

    for (int i = 0; i < count - 1; ++i) {
      int idx1 = (head - 1 - i + N) % N;
      int idx0 = (head - 2 - i + N) % N;

      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      // Check for upcrossing with epsilon threshold
      if (upcrossing) {
        if (s0.heave < -EPSILON && s1.heave >= -EPSILON) {
          float denominator = s0.heave - s1.heave;
          if (fabsf(denominator) < EPSILON) {
            // Near-vertical line case - take midpoint
            zc_time = (s0.time + s1.time) * 0.5f;
          } else {
            float frac = s0.heave / denominator;
            float dt = s1.time - s0.time;
            zc_time = s0.time + (frac * dt);
          }
          return true;
        }
      }
      // Check for downcrossing with epsilon threshold
      else {
        if (s0.heave > EPSILON && s1.heave <= EPSILON) {
          float denominator = s0.heave - s1.heave;
          if (fabsf(denominator) < EPSILON) {
            // Near-vertical line case - take midpoint
            zc_time = (s0.time + s1.time) * 0.5f;
          } else {
            float frac = s0.heave / denominator;
            float dt = s1.time - s0.time;
            zc_time = s0.time + (frac * dt);
          }
          return true;
        }
      }
    }
    return false;
  }


  // Crest sharpness: max heave divided by time-to-next-downcrossing
  float computeCrestSharpness() const {
    float maxHeave = -INFINITY;
    float crestTime = 0.0f, downTime = 0.0f;
    bool crestFound = false;

    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.time;
        crestFound = true;
      }
    }

    if (!crestFound) return 0.0f;

    bool downFound = false;
    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (head - 2 - i + N) % N;
      int idx1 = (head - 1 - i + N) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime) {
        float frac = s0.heave / (s0.heave - s1.heave);
        float dt = s1.time - s0.time;
        downTime = s0.time + (frac * dt);
        downFound = true;
        break;
      }
    }

    if (!downFound) return 0.0f;

    float dt_sec = downTime - crestTime;
    return (dt_sec > 0.0f) ? maxHeave / dt_sec : 0.0f;
  }

  // Asymmetry: time from upcross to crest vs crest to downcross
  float computeAsymmetry() {
    float upTime = 0.0f, crestTime = 0.0f, downTime = 0.0f;
    float maxHeave = -INFINITY;

    if (!findLatestZeroUpcrossing(0.0f)) return 0.0f;
    upTime = zc_time;

    bool crestFound = false;
    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.time <= upTime) break;
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.time;
        crestFound = true;
      }
    }

    if (!crestFound) return 0.0f;

    bool downFound = false;
    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (head - 2 - i + N) % N;
      int idx1 = (head - 1 - i + N) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime) {
        float frac = s0.heave / (s0.heave - s1.heave);
        float dt = s1.time - s0.time;
        downTime = s0.time + (frac * dt);
        downFound = true;
        break;
      }
    }

    if (!downFound) return 0.0f;

    float rise = (crestTime - upTime);
    float fall = (downTime - crestTime);
    return (rise + fall > 0.0f) ? (rise - fall) / (rise + fall) : 0.0f;
  }

  // Predict heave at future phase [0–1)
  float predictAtPhase(float phase, float t) {
    if (count < 2) return 0.0f;
    phase = fmodf(phase, 1.0f);
    if (phase < 0) phase += 1.0f;

    float now_phase = getPhase(t);
    float target_phase = fmodf(now_phase + phase, 1.0f);
    int target_idx = (int)(target_phase * count);

    // Sample from buffer at target index (older data)
    int idx = (head - count + target_idx + N) % N;
    return samples[idx].heave;
  }

  float computeWaveTrainSlope() const {
    float crest = -INFINITY, trough = INFINITY;
    float crestTime = 0.0f, troughTime = 0.0f;

    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.heave > crest) {
        crest = s.heave;
        crestTime = s.time;
      }
      if (s.heave < trough) {
        trough = s.heave;
        troughTime = s.time;
      }
    }

    if (crestTime == 0 || troughTime == 0 || crestTime == troughTime) return 0.0f;

    float dt = fabs(crestTime - troughTime);
    float dh = crest - trough;
    float wave_speed = 9.81f / (2.0f * PI * freq); // trochoidal approx

    return (dt > 0.0f) ? (dh / dt) * wave_speed : 0.0f;
  }

  float computeWaveEnergy(float water_density = 1025.0f, float gravity = 9.81f) const {
    if (count < 1) return 0.0f;
  
    float sumSq = 0.0f;
    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      float h = samples[idx].heave;
      sumSq += h * h;
    }
  
    float meanSq = sumSq / count; // RMS^2
    // E = 0.5 * rho * g * RMS^2
    return 0.5f * water_density * gravity * meanSq;
  }

  // Raw buffer access (read-only)
  const WaveSample* getSamples() const { return samples; }
  int getCount() const { return count; }
  int getCapacity() const { return N; }
};

#endif // WAVE_SURFACE_PROFILE_H
