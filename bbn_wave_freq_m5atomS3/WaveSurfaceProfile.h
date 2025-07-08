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
  float heave;
  uint32_t timeMicros;
};

template<int N = 128>
class WaveSurfaceProfile {
private:
  WaveSample samples[N];  // circular buffer
  int head = 0;
  int count = 0;
  float freq = 1.0f;       // wave frequency in Hz

public:
  void reset() {
    head = 0;
    count = 0;
    freq = 1.0f;
  }

  void update(float heave, float new_freq, uint32_t timeMicros) {
    samples[head] = {heave, timeMicros};
    head = (head + 1) % N;
    if (count < N) count++;

    if (new_freq > 0.01f && new_freq < 2.0f) {
      freq = new_freq;
    }
  }

  float getFrequency() const {
    return freq;
  }

  float getPeriod() const {
    return 1.0f / freq;
  }

  float getPhase(uint32_t nowMicros) const {
    uint32_t zc_time;
    if (!findLatestZeroUpcrossing(zc_time)) return 0.0f;
    float elapsed = (nowMicros - zc_time) / 1e6f;
    float phase = fmodf(elapsed * freq, 1.0f); // phase = t / T = t * f
    return (phase >= 0.0f) ? phase : (phase + 1.0f);
  }

  float getPhaseDegrees(uint32_t nowMicros) const {
    return 360.0f * getPhase(nowMicros);
  }

  bool findLatestZeroUpcrossing(uint32_t& zc_time_micros) const {
    return findLatestZeroCrossing(true, zc_time_micros);
  }

  bool findLatestZeroDowncrossing(uint32_t& zc_time_micros) const {
    return findLatestZeroCrossing(false, zc_time_micros);
  }

  bool findLatestZeroCrossing(bool upcrossing, uint32_t& zc_time_micros) const {
    for (int i = 0; i < count - 1; ++i) {
      int idx1 = (head - 1 - i + N) % N;
      int idx0 = (head - 2 - i + N) % N;

      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (upcrossing && s0.heave < 0 && s1.heave >= 0) {
        float frac = s0.heave / (s0.heave - s1.heave);
        uint32_t dt = s1.timeMicros - s0.timeMicros;
        zc_time_micros = s0.timeMicros + (uint32_t)(frac * dt);
        return true;
      }

      if (!upcrossing && s0.heave > 0 && s1.heave <= 0) {
        float frac = s0.heave / (s0.heave - s1.heave);
        uint32_t dt = s1.timeMicros - s0.timeMicros;
        zc_time_micros = s0.timeMicros + (uint32_t)(frac * dt);
        return true;
      }
    }
    return false;
  }

  // Crest sharpness: max heave divided by time-to-next-downcrossing
  float computeCrestSharpness() const {
    float maxHeave = -INFINITY;
    uint32_t crestTime = 0, downTime = 0;
    bool crestFound = false;

    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.timeMicros;
        crestFound = true;
      }
    }

    if (!crestFound) return 0.0f;

    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (head - 2 - i + N) % N;
      int idx1 = (head - 1 - i + N) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (s0.heave > 0 && s1.heave <= 0 && s0.timeMicros > crestTime) {
        float frac = s0.heave / (s0.heave - s1.heave);
        uint32_t dt = s1.timeMicros - s0.timeMicros;
        downTime = s0.timeMicros + (uint32_t)(frac * dt);
        float dt_sec = (downTime - crestTime) / 1e6f;
        return dt_sec > 0 ? maxHeave / dt_sec : 0.0f;
      }
    }
    return 0.0f;
  }

  // Asymmetry: time from upcross to crest vs crest to downcross
  float computeAsymmetry() const {
    uint32_t upTime = 0, crestTime = 0, downTime = 0;
    float maxHeave = -INFINITY;

    // find last upcrossing
    if (!findLatestZeroUpcrossing(upTime)) return 0.0f;

    // find crest after upcrossing
    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.timeMicros <= upTime) break;
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.timeMicros;
      }
    }

    // find next downcrossing after crest
    for (int i = 0; i < count - 1; ++i) {
      int idx0 = (head - 2 - i + N) % N;
      int idx1 = (head - 1 - i + N) % N;
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.heave > 0 && s1.heave <= 0 && s0.timeMicros > crestTime) {
        float frac = s0.heave / (s0.heave - s1.heave);
        uint32_t dt = s1.timeMicros - s0.timeMicros;
        downTime = s0.timeMicros + (uint32_t)(frac * dt);
        break;
      }
    }

    if (upTime == 0 || crestTime == 0 || downTime == 0) return 0.0f;

    float rise = (crestTime - upTime) / 1e6f;
    float fall = (downTime - crestTime) / 1e6f;
    return (rise + fall > 0.0f) ? (rise - fall) / (rise + fall) : 0.0f;
  }

  // Predict heave at future phase [0–1)
  float predictAtPhase(float phase, uint32_t nowMicros) const {
    if (count < 2) return 0.0f;
    phase = fmodf(phase, 1.0f);
    if (phase < 0) phase += 1.0f;

    float now_phase = getPhase(nowMicros);
    float target_phase = fmodf(now_phase + phase, 1.0f);
    int target_idx = (int)(target_phase * count);

    // Sample from buffer at target index (older data)
    int idx = (head - count + target_idx + N) % N;
    return samples[idx].heave;
  }

  float computeWaveTrainSlope() const {
    float crest = -INFINITY, trough = INFINITY;
    uint32_t crestTime = 0, troughTime = 0;

    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
      const WaveSample& s = samples[idx];
      if (s.heave > crest) {
        crest = s.heave;
        crestTime = s.timeMicros;
      }
      if (s.heave < trough) {
        trough = s.heave;
        troughTime = s.timeMicros;
      }
    }

    if (crestTime == 0 || troughTime == 0 || crestTime == troughTime) return 0.0f;

    float dt = fabs((int32_t)(crestTime - troughTime)) / 1e6f;
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
