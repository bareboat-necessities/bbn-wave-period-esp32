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
  float lastWaveProfileUpdate = 0.0f;
  float lastZcTime = 0.0f;

  inline bool isValidFrequency(float f) const {
    return f > 0.01f && f < 2.0f && isFinite(f);
  }

  inline bool isFinite(float x) const {
    return isfinite(x);
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
    if (count > 0 && fabsf(samples[(head - 1 + N) % N].time - t) < EPSILON) return;

    samples[head] = {heave, t};
    head = (head + 1) % N;
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
    float targetDt = 2.0f * periodSec / N;
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
    float phase = elapsed * freq - floorf(elapsed * freq);
    return phase;
  }

  float getPhaseDegrees(float t) {
    return 360.0f * getPhase(t);
  }

  bool findLatestZeroUpcrossing() {
    return findLatestZeroCrossing(true);
  }

  bool findLatestZeroDowncrossing() {
    return findLatestZeroCrossing(false);
  }

  bool findLatestZeroCrossing(bool upcrossing) {
    for (int i = 0; i < count - 1; ++i) {
      int idx1 = (head - 1 - i + N) % N;
      int idx0 = (head - 2 - i + N) % N;

      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      if (s0.time >= s1.time) continue;

      float denominator = s0.heave - s1.heave;
      float frac = (fabsf(denominator) < EPSILON) ? 0.5f : s0.heave / denominator;
      frac = fminf(fmaxf(frac, 0.0f), 1.0f); // Clamp [0,1]
      float zcTime = s0.time + frac * (s1.time - s0.time);

      if (upcrossing && s0.heave < 0.0f && s1.heave >= 0.0f) {
        lastZcTime = zcTime;
        return true;
      } else if (!upcrossing && s0.heave > 0.0f && s1.heave <= 0.0f) {
        lastZcTime = zcTime;
        return true;
      }
    }
    return false;
  }

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
        float denom = s0.heave - s1.heave;
        float frac = (fabsf(denom) < EPSILON) ? 0.5f : s0.heave / denom;
        frac = fminf(fmaxf(frac, 0.0f), 1.0f);
        upTime = s0.time + frac * (s1.time - s0.time);
        upFound = true;
      }

      if (s1.heave > maxHeave) {
        maxHeave = s1.heave;
        crestTime = s1.time;
        crestFound = true;
      }

      if (s0.heave > 0 && s1.heave <= 0 && s0.time > crestTime && !downFound) {
        float denom = s0.heave - s1.heave;
        float frac = (fabsf(denom) < EPSILON) ? 0.5f : s0.heave / denom;
        frac = fminf(fmaxf(frac, 0.0f), 1.0f);
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

  float computeAsymmetry() {
    if (count < 3) return 0.0f;
    if (!findLatestZeroUpcrossing()) return 0.0f;

    float upTime = lastZcTime;
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
        float denom = s0.heave - s1.heave;
        float frac = (fabsf(denom) < EPSILON) ? 0.5f : s0.heave / denom;       
        frac = fminf(fmaxf(frac, 0.0f), 1.0f);
        downTime = s0.time + frac * (s1.time - s0.time);
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

    float targetPhase = fmodf(getPhase(t) + phase, 1.0f);
    if (targetPhase < 0.0f) targetPhase += 1.0f;

    int samplesPerPeriod = count / 2;
    int start = (head - samplesPerPeriod + N) % N;

    float bestHeave = 0.0f;
    float minPhaseDist = 10.0f; // larger than max phase diff (wraps around at 1)

    for (int i = 0; i < samplesPerPeriod - 1; ++i) {
      int idx0 = (start + i) % N;
      int idx1 = (start + i + 1) % N;

      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];

      float p0 = fmodf((s0.time - lastZcTime) * freq, 1.0f);
      float p1 = fmodf((s1.time - lastZcTime) * freq, 1.0f);
      if (p0 < 0.0f) p0 += 1.0f;
      if (p1 < 0.0f) p1 += 1.0f;

      // Compute distance from targetPhase to segment
      float dist0 = fabsf(p0 - targetPhase);
      float dist1 = fabsf(p1 - targetPhase);
      if (dist0 > 0.5f) dist0 = 1.0f - dist0;
      if (dist1 > 0.5f) dist1 = 1.0f - dist1;

      // Track nearest sample as fallback
      if (dist0 < minPhaseDist) {
        bestHeave = s0.heave;
        minPhaseDist = dist0;
      }
      if (dist1 < minPhaseDist) {
        bestHeave = s1.heave;
        minPhaseDist = dist1;
      }

      // Check if target phase lies between p0 and p1 (wrap-aware)
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
    // No segment crossing target phase found → fallback to best match
    return bestHeave;
  }

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
    float waveSpeed = GRAVITY / (2 * M_PI * freq);
    return (dh / dt) * waveSpeed;
  }

  float computeWaveEnergy(float waterDensity = 1025.0f, float gravity = GRAVITY) const {
    if (count < 1) return 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < count; ++i) {
      int idx = (head - 1 - i + N) % N;
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
    return samples[(head - 1 + N) % N];
  }
};

#endif // WAVE_SURFACE_PROFILE_H
