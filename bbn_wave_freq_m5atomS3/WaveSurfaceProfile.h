#ifndef WAVE_SURFACE_PROFILE_H
#define WAVE_SURFACE_PROFILE_H

/*
  Copyright 2025, Mikhail Grushinskiy

  WaveSurfaceProfile - Tracks a rolling buffer of heave samples
  and computes wave phase, crest sharpness, asymmetry, and future prediction.

  - Anchors phase to last zero-upcrossing (heave < 0 → ≥ 0)
  - Uses frequency for phase and prediction
  - Tracks N samples for wave shape reconstruction
*/

#include <math.h>

constexpr float EPSILON = 1e-6f;
constexpr float GRAVITY = 9.81f;
constexpr int STORE_PERIODS = 2;

struct WaveSample {
  float heave;  // meters
  float time;   // seconds
};

template<int N = 128>
class WaveSurfaceProfile {
private:
  WaveSample samples[N];
  int head = 0;
  int count = 0;
  float freq = 1.0f;
  float lastWaveProfileUpdate = 0.0f;
  float lastZcTime = 0.0f;

  inline int wrapIdx(int i) const {
    return (i + N) % N;
  }

  inline bool isValidFrequency(float f) const {
    return f > 0.01f && f < 2.0f && isFinite(f);
  }

  inline bool isFinite(float x) const {
    return isfinite(x);
  }

  inline float interpolateZeroCrossingTime(const WaveSample& s0, const WaveSample& s1) const {
    float denom = s0.heave - s1.heave;
    float frac = (fabsf(denom) < EPSILON) ? 0.5f : s0.heave / denom;
    frac = fminf(fmaxf(frac, 0.0f), 1.0f);
    return s0.time + frac * (s1.time - s0.time);
  }

  inline float normalizePhase(float p) const {
    p = fmodf(p, 1.0f);
    return (p < 0.0f) ? (p + 1.0f) : p;
  }

  inline float wrappedPhaseDistance(float a, float b) const {
    float d = fabsf(a - b);
    return (d > 0.5f) ? (1.0f - d) : d;
  }

  bool findZeroCrossing(bool upcrossing, float& crossingTime) {
    for (int i = 0; i < count - 1; ++i) {
      int idx1 = wrapIdx(head - 1 - i);
      int idx0 = wrapIdx(head - 2 - i);
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.time >= s1.time) continue;

      if (upcrossing && s0.heave < 0 && s1.heave >= 0) {
        crossingTime = interpolateZeroCrossingTime(s0, s1);
        return true;
      }
      if (!upcrossing && s0.heave > 0 && s1.heave <= 0) {
        crossingTime = interpolateZeroCrossingTime(s0, s1);
        return true;
      }
    }
    return false;
  }

  struct CrestMetrics {
    float upTime = 0, crestTime = 0, downTime = 0, maxHeave = -INFINITY;
    bool crestFound = false, upFound = false, downFound = false;
  };

  CrestMetrics findCrestMetrics() const {
    CrestMetrics m;
    int start = wrapIdx(head - count);

    for (int i = 0; i < count - 1; ++i) {
      int idx0 = wrapIdx(start + i);
      int idx1 = wrapIdx(start + i + 1);
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (!m.upFound && s0.heave < 0 && s1.heave >= 0) {
        m.upTime = interpolateZeroCrossingTime(s0, s1);
        m.upFound = true;
      }

      if (s1.heave > m.maxHeave) {
        m.maxHeave = s1.heave;
        m.crestTime = s1.time;
        m.crestFound = true;
      }

      if (s0.heave > 0 && s1.heave <= 0 && s0.time > m.crestTime && !m.downFound) {
        m.downTime = interpolateZeroCrossingTime(s0, s1);
        m.downFound = true;
      }
    }
    return m;
  }

public:
  inline void reset() {
    head = 0;
    count = 0;
    freq = 1.0f;
    lastWaveProfileUpdate = 0.0f;
    lastZcTime = 0.0f;
  }

  void update(float heave, float newFreq, float t) {
    if (!isFinite(heave) || !isFinite(t)) return;
    if (count > 0 && fabsf(samples[wrapIdx(head - 1)].time - t) < EPSILON) return;

    samples[head] = {heave, t};
    head = wrapIdx(head + 1);
    if (count < N) count++;
    if (isValidFrequency(newFreq)) {
      freq = newFreq;
    }
  }

  void updateIfNeeded(float heave, float newFreq, float t) {
    if (!isFinite(heave) || !isFinite(t)) return;
    if (isValidFrequency(newFreq)) {
      freq = newFreq;
    }
    float periodSec = (freq > EPSILON) ? (1.0f / freq) : 1.0f;
    float targetDt = STORE_PERIODS * periodSec / N;
    if (lastWaveProfileUpdate == 0.0f || (t - lastWaveProfileUpdate >= targetDt)) {
      update(heave, freq, t);
      lastWaveProfileUpdate = t;
    }
  }

  [[nodiscard]] inline bool isReady() const { return count >= 3; }
  [[nodiscard]] inline float getFrequency() const { return freq; }
  [[nodiscard]] inline float getPeriod() const { return (freq > EPSILON) ? (1.0f / freq) : 0.0f; }

  float getPhase(float t) {
    if (!isFinite(t)) return 0.0f;
    if (!findLatestZeroUpcrossing()) return 0.0f;
    float elapsed = t - lastZcTime;
    return normalizePhase(elapsed * freq);
  }

  float getPhaseDegrees(float t) {
    return 360.0f * getPhase(t);
  }

  bool findLatestZeroUpcrossing() {
    float zc;
    if (findZeroCrossing(true, zc)) {
      lastZcTime = zc;
      return true;
    }
    return false;
  }

  bool findLatestZeroDowncrossing() {
    float zc;
    return findZeroCrossing(false, zc);
  }

  float computeCrestSharpness() const {
    CrestMetrics m = findCrestMetrics();
    if (!m.crestFound || !m.upFound || !m.downFound) return 0.0f;

    float rise = m.crestTime - m.upTime;
    float fall = m.downTime - m.crestTime;
    float riseSlope = (rise > 0.0f) ? m.maxHeave / rise : 0.0f;
    float fallSlope = (fall > 0.0f) ? m.maxHeave / fall : 0.0f;
    return 0.5f * (riseSlope + fallSlope);
  }

  float computeAsymmetry() {
    if (count < 3) return 0.0f;
    if (!findLatestZeroUpcrossing()) return 0.0f;

    float upTime = lastZcTime, crestTime = 0.0f, downTime = 0.0f;
    float maxHeave = -INFINITY;
    bool crestFound = false, downFound = false;
    int start = wrapIdx(head - count);

    for (int i = 0; i < count; ++i) {
      int idx = wrapIdx(start + i);
      const WaveSample& s = samples[idx];
      if (s.time <= upTime) continue;
      if (s.heave > maxHeave) {
        maxHeave = s.heave;
        crestTime = s.time;
        crestFound = true;
      }
    }

    for (int i = 0; i < count - 1; ++i) {
      int idx0 = wrapIdx(start + i);
      int idx1 = wrapIdx(start + i + 1);
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime) {
        downTime = interpolateZeroCrossingTime(s0, s1);
        downFound = true;
        break;
      }
    }

    if (!crestFound || !downFound) return 0.0f;
    float rise = crestTime - upTime;
    float fall = downTime - crestTime;
    if (rise + fall < EPSILON) return 0.0f;
    return (rise - fall) / (rise + fall);
  }

  float predictAtPhase(float phase, float t) {
    if (count < 3 || !isFinite(t) || !isFinite(phase)) return 0.0f;
    if (!findLatestZeroUpcrossing()) return 0.0f;

    float targetPhase = normalizePhase(getPhase(t) + phase);
    int samplesPerPeriod = count / STORE_PERIODS;
    int start = wrapIdx(head - samplesPerPeriod);

    float bestHeave = 0.0f;
    float minPhaseDist = 10.0f;

    for (int i = 0; i < samplesPerPeriod - 1; ++i) {
      int idx0 = wrapIdx(start + i);
      int idx1 = wrapIdx(start + i + 1);
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      float p0 = normalizePhase((s0.time - lastZcTime) * freq);
      float p1 = normalizePhase((s1.time - lastZcTime) * freq);

      float dist0 = wrappedPhaseDistance(p0, targetPhase);
      float dist1 = wrappedPhaseDistance(p1, targetPhase);

      if (dist0 < minPhaseDist) {
        bestHeave = s0.heave;
        minPhaseDist = dist0;
      }
      if (dist1 < minPhaseDist) {
        bestHeave = s1.heave;
        minPhaseDist = dist1;
      }

      bool crosses = (p0 <= targetPhase && targetPhase <= p1) ||
                     (p1 < p0 && (targetPhase >= p0 || targetPhase <= p1));
      if (crosses) {
        float dp = p1 - p0;
        if (dp < 0.0f) dp += 1.0f;
        float alpha = (dp < EPSILON) ? 0.5f : (targetPhase - p0) / dp;
        if (alpha < 0.0f) alpha += 1.0f;
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
        return s0.heave + alpha * (s1.heave - s0.heave);
      }
    }
    return bestHeave;
  }

  float computeWaveTrainVelocityGradient() const {
    float crest = -INFINITY, trough = INFINITY;
    float crestTime = 0.0f, troughTime = 0.0f;
    bool crestFound = false, troughFound = false;

    for (int i = 0; i < count; ++i) {
      int idx = wrapIdx(head - 1 - i);
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
    float waveSpeed = GRAVITY / (2 * M_PI * freq);
    return (dh / dt) * waveSpeed;
  }

  float computeWaveEnergy(float waterDensity = 1025.0f, float gravity = GRAVITY) const {
    if (count < 1) return 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < count; ++i) {
      int idx = wrapIdx(head - 1 - i);
      float h = samples[idx].heave;
      if (isFinite(h)) {
        sumSq += h * h;
      }
    }
    float meanSq = sumSq / count;
    return 0.5f * waterDensity * gravity * meanSq;
  }

  [[nodiscard]] const WaveSample* getSamples() const { return samples; }
  [[nodiscard]] int getCount() const { return count; }
  [[nodiscard]] int getCapacity() const { return N; }

  [[nodiscard]] WaveSample getLatestSample() const {
    return samples[wrapIdx(head - 1)];
  }
};

#endif // WAVE_SURFACE_PROFILE_H
