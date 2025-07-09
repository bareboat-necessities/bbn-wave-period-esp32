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

constexpr float EPSILON = 1e-6f;
constexpr float GRAVITY = 9.81f;
constexpr float TWO_PI = 6.28318530718f;

struct WaveSample {
  float heave;  // meters
  float time;   // seconds
};

template<int N = 128>
class WaveSurfaceProfile {
private:
  WaveSample samples[N];  // circular buffer
  int head = 0;
  int count = 0;
  float freq = 1.0f;
  float last_wave_profile_update = 0.0f;
  float zc_time = 0.0f;

public:
  void reset() {
    head = 0;
    count = 0;
    freq = 1.0f;
  }

  void update(float heave, float new_freq, float t) {
    if (count > 0 && fabsf(samples[(head - 1 + N) % N].time - t) < EPSILON) return;

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
    float period_sec = (freq > EPSILON) ? (1.0f / freq) : 1.0f;
    float target_dt = 2.0f * period_sec / N;
    if (last_wave_profile_update == 0.0f || (t - last_wave_profile_update >= target_dt)) {
      update(heave, freq, t);
      last_wave_profile_update = t;
    }
  }

  bool isReady() const { return count >= 3; }
  float getFrequency() const { return freq; }
  float getPeriod() const { return (freq > EPSILON) ? (1.0f / freq) : 0.0f; }

  float getPhase(float t) {
    if (!findLatestZeroUpcrossing(t)) return 0.0f;
    float elapsed = t - zc_time;
    float phase = elapsed * freq - floorf(elapsed * freq);
    return phase;
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

  bool findLatestZeroCrossing(bool upcrossing, float /*t*/) {
    for (int i = 0; i < count - 1; ++i) {
      int idx1 = (head - 1 - i + N) % N;
      int idx0 = (head - 2 - i + N) % N;

      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (s0.time >= s1.time) continue;

      if (upcrossing) {
        if (s0.heave < 0.0f && s1.heave >= 0.0f) {
          float denominator = s0.heave - s1.heave;
          float frac = (fabsf(denominator) < EPSILON) ? 0.5f : s0.heave / denominator;
          zc_time = s0.time + frac * (s1.time - s0.time);
          return true;
        }
      } else {
        if (s0.heave > 0.0f && s1.heave <= 0.0f) {
          float denominator = s0.heave - s1.heave;
          float frac = (fabsf(denominator) < EPSILON) ? 0.5f : s0.heave / denominator;
          zc_time = s0.time + frac * (s1.time - s0.time);
          return true;
        }
      }
    }
    return false;
  }

  // Crest sharpness = average heave slope to and from the crest
  float computeCrestSharpness() const {
    float maxHeave = -INFINITY;
    float crestTime = 0.0f;
    float upTime = 0.0f, downTime = 0.0f;
    bool crestFound = false, upFound = false, downFound = false;

    int start = (head - count + N) % N;

    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (start + i) % N;
      int idx1 = (start + i + 1) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (!upFound && s0.heave < 0 && s1.heave >= 0) {
        float frac = s0.heave / (s0.heave - s1.heave);
        upTime = s0.time + frac * (s1.time - s0.time);
        upFound = true;
      }

      if (s1.heave > maxHeave) {
        maxHeave = s1.heave;
        crestTime = s1.time;
        crestFound = true;
      }

      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime && !downFound) {
        float frac = s0.heave / (s0.heave - s1.heave);
        downTime = s0.time + frac * (s1.time - s0.time);
        downFound = true;
      }
    }

    if (!crestFound || !upFound || !downFound) return 0.0f;

    float rise = crestTime - upTime;
    float fall = downTime - crestTime;

    float riseSlope = (rise > 0.0f) ? maxHeave / rise : 0.0f;
    float fallSlope = (fall > 0.0f) ? maxHeave / fall : 0.0f;

    return 0.5f * (riseSlope + fallSlope);
  }

  // Asymmetry = normalized rise vs fall time difference around crest
  float computeAsymmetry() {
    if (count < 3) return 0.0f;
    float latestT = samples[(head - 1 + N) % N].time;
    if (!findLatestZeroUpcrossing(latestT)) return 0.0f;

    float upTime = zc_time;
    float crestTime = 0.0f, downTime = 0.0f;
    float maxHeave = -INFINITY;
    bool crestFound = false, downFound = false;

    int start = (head - count + N) % N;

    for (int i = 0; i < count; ++i) {
      int idx = (start + i) % N;
      const WaveSample& s = samples[idx];
      if (s.time <= upTime) continue;
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.time;
        crestFound = true;
      }
    }

    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (start + i) % N;
      int idx1 = (start + i + 1) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime) {
        float frac = s0.heave / (s0.heave - s1.heave);
        downTime = s0.time + frac * (s1.time - s0.time);
        downFound = true;
        break;
      }
    }

    if (!crestFound || !downFound) return 0.0f;

    float rise = crestTime - upTime;
    float fall = downTime - crestTime;
    return (rise + fall > 0.0f) ? (rise - fall) / (rise + fall) : 0.0f;
  }

  // Predict heave value at given future phase [0..1]
  float predictAtPhase(float phase, float t) {
    if (count < 2) return 0.0f;

    phase = phase - floorf(phase);  // wrap to [0,1)
    float now_phase = getPhase(t);
    float target_phase = now_phase + phase;
    target_phase = target_phase - floorf(target_phase);

    float fidx = target_phase * count;
    if (fidx >= count) fidx = count - 1;  // bound fidx

    int i0 = (int)fidx;
    int i1 = (i0 + 1) % count;
    float alpha = fidx - i0;

    int start = (head - count + N) % N;
    int idx0 = (start + i0) % N;
    int idx1 = (start + i1) % N;

    return (1.0f - alpha) * samples[idx0].heave + alpha * samples[idx1].heave;
  }

  // Slope (m/s * wave speed) of crest-to-trough heave change
  float computeWaveTrainVelocityGradient() const {
    float crest = -INFINITY, trough = INFINITY;
    float crestTime = 0.0f, troughTime = 0.0f;
    bool crestFound = false, troughFound = false;

    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.heave > crest) {
        crest = s.heave;
        crestTime = s.time;
        crestFound = true;
      }
      if (s.heave < trough) {
        trough = s.heave;
        troughTime = s.time;
        troughFound = true;
      }
    }

    float dt = fabsf(crestTime - troughTime);
    if (!crestFound || !troughFound || dt < EPSILON) return 0.0f; 

    float dh = crest - trough;
    float wave_speed = GRAVITY / (TWO_PI * freq);

    return (dh / dt) * wave_speed;
  }

  // Wave energy per unit area (J/m²) using linear approximation
  float computeWaveEnergy(float water_density = 1025.0f, float gravity = GRAVITY) const {
    if (count < 1) return 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      float h = samples[idx].heave;
      sumSq += h * h;
    }
    float meanSq = sumSq / count;
    return 0.5f * water_density * gravity * meanSq;
  }

  const WaveSample* getSamples() const { return samples; }
  int getCount() const { return count; }
  int getCapacity() const { return N; }

  WaveSample getLatestSample() const {
    return samples[(head - 1 + N) % N];
  }
};

#endif // WAVE_SURFACE_PROFILE_H
