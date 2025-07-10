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
#include <algorithm>
#include <cassert>

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

  enum class CrossingType {
    Upcrossing,
    Downcrossing
  };

  inline int wrapIdx(int i) const {
    return (i + N) % N;
  }

  inline bool isFinite(float x) const {
    return isfinite(x);
  }

  inline bool isValidFrequency(float f) const {
    return f > 0.01f && f < 2.0f && isFinite(f);
  }

  template<typename Func>
  bool scanSamples(int startOffset, int endOffset, Func&& callback) const {
    for (int i = startOffset; i < endOffset; ++i) {
      int idx0 = wrapIdx(head + i);
      int idx1 = wrapIdx(head + i + 1);
      const WaveSample& s0 = samples[idx0];
      const WaveSample& s1 = samples[idx1];
      if (s0.time >= s1.time) continue;
      if (callback(s0, s1)) return true;
    }
    return false;
  }

  template<typename Func>
  void forEachSample(Func&& callback) const {
    for (int i = 0; i < count; ++i) {
      int idx = wrapIdx(head - 1 - i);
      callback(samples[idx]);
    }
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

  bool detectCrossing(CrossingType type, float& crossingTime) const {
    return scanSamples(-count, -1, [&](const WaveSample& s0, const WaveSample& s1) {
      if (type == CrossingType::Upcrossing && s0.heave < 0 && s1.heave >= 0) {
        crossingTime = interpolateZeroCrossingTime(s0, s1);
        return true;
      }
      if (type == CrossingType::Downcrossing && s0.heave > 0 && s1.heave <= 0) {
        crossingTime = interpolateZeroCrossingTime(s0, s1);
        return true;
      }
      return false;
    });
  }

  float interpolateHeaveBetween(const WaveSample& s0, const WaveSample& s1, float p0, float p1, float targetPhase) const {
    if (p0 <= p1) {
      float dp = p1 - p0;
      float alpha = (dp < EPSILON) ? 0.5f : (targetPhase - p0) / dp;
      return s0.heave + alpha * (s1.heave - s0.heave);
    } else {
      float adjustedP1 = p1 + 1.0f;
      float adjustedTarget = (targetPhase < p0) ? (targetPhase + 1.0f) : targetPhase;
      float dp = adjustedP1 - p0;
      float alpha = (dp < EPSILON) ? 0.5f : (adjustedTarget - p0) / dp;
      return s0.heave + alpha * (s1.heave - s0.heave);
    }
  }

  float nearestHeaveOfPair(const WaveSample& s0, const WaveSample& s1,
                           float p0, float p1, float targetPhase, float& minDist) const {
    float dist0 = wrappedPhaseDistance(p0, targetPhase);
    float dist1 = wrappedPhaseDistance(p1, targetPhase);
  
    float bestHeave = 0.0f;
    bool updated = false;
    if (dist0 < minDist) {
      minDist = dist0;
      bestHeave = s0.heave;
      updated = true;
    }
    if (dist1 < minDist) {
      minDist = dist1;
      bestHeave = s1.heave;
      updated = true;
    }
    return updated ? bestHeave : 0.0f;
  }

  struct CrestMetrics {
    float upTime = 0, crestTime = 0, downTime = 0, maxHeave = -INFINITY;
    bool crestFound = false, upFound = false, downFound = false;

    [[nodiscard]] bool isComplete() const {
      return upFound && crestFound && downFound;
    }
  };

  CrestMetrics computeCrestMetrics() const {
    CrestMetrics m;
    scanSamples(-count, -1, [&](const WaveSample& s0, const WaveSample& s1) {
      if (!m.upFound && s0.heave < 0 && s1.heave >= 0) {
        m.upTime = interpolateZeroCrossingTime(s0, s1);
        m.upFound = true;
      }
      if (s1.heave > m.maxHeave) {
        m.maxHeave = s1.heave;
        m.crestTime = s1.time;
        m.crestFound = true;
      }
      if (!m.downFound && s0.heave > 0 && s1.heave <= 0 && s0.time > m.crestTime) {
        m.downTime = interpolateZeroCrossingTime(s0, s1);
        m.downFound = true;
      }
      return m.isComplete();
    });
    return m;
  }


public:
  /**
   * Clears internal sample buffer and resets frequency and timestamps.
   */
  inline void reset() {
    head = 0;
    count = 0;
    freq = 1.0f;
    lastWaveProfileUpdate = 0.0f;
    lastZcTime = 0.0f;
  }

  /**
   * Inserts a new sample and updates the frequency if valid.
   * Assumes sample is temporally after the last.
   */
  void update(float heave, float newFreq, float t) {
    if (!isFinite(heave) || !isFinite(t)) return;
    if (count > 0 && fabsf(samples[wrapIdx(head - 1)].time - t) < EPSILON) return;

    samples[head] = {heave, t};
    head = wrapIdx(head + 1);
    if (count < N) count++;
    if (isValidFrequency(newFreq)) freq = newFreq;
  }

  /**
   * Adds a sample only if enough time has elapsed since last insertion,
   * preserving temporal spacing based on wave period.
   */
  void updateIfNeeded(float heave, float newFreq, float t) {
    if (!isFinite(heave) || !isFinite(t)) return;
    if (isValidFrequency(newFreq)) freq = newFreq;

    float periodSec = (freq > EPSILON) ? (1.0f / freq) : 1.0f;
    float targetDt = STORE_PERIODS * periodSec / N;
    if (lastWaveProfileUpdate < EPSILON || (t - lastWaveProfileUpdate >= targetDt)) {
      update(heave, freq, t);
      lastWaveProfileUpdate = t;
    }
  }

  /**
   * Returns true if enough samples are available for wave analysis.
   */
  [[nodiscard]] inline bool isReady() const { return count >= 3; }

  /**
   * Returns the most recent wave frequency in Hz.
   */
  [[nodiscard]] inline float getFrequency() const { return freq; }

  /**
   * Returns the wave period in seconds (1/frequency).
   */
  [[nodiscard]] inline float getPeriod() const { return (freq > EPSILON) ? (1.0f / freq) : 0.0f; }

  /**
   * Computes normalized phase [0,1) based on last zero-upcrossing before `t`.
   * Phase = 0 corresponds to wave passing through mean level upward.
   */
  float getPhase(float t) {
    if (!isFinite(t)) return 0.0f;
    float zc;
    if (!findLatestZeroUpcrossing(zc)) return 0.0f;
    float elapsed = t - zc;
    return normalizePhase(elapsed * freq);
  }

  /**
   * Same as getPhase(), but returns phase in degrees [0, 360).
   */
  float getPhaseDegrees(float t) {
    return 360.0f * getPhase(t);
  }

  /**
   * Finds time of most recent zero-upcrossing and outputs it.
   * Returns false if not found.
   */
  bool findLatestZeroUpcrossing(float& zcOut) {
    return detectCrossing(CrossingType::Upcrossing, zcOut);
  }

  /**
   * Updates `lastZcTime` with most recent zero-upcrossing, if found.
   */
  bool detectLatestZeroUpcrossing() {
    return findLatestZeroUpcrossing(lastZcTime);
  }

  /**
   * Computes average slope of crest: how sharply wave rises and falls around the peak.
   * - Sharpness = ½ × (ascent slope + descent slope)
   * - Slope = maxHeave / timeToRiseOrFall
   */
  float computeCrestSharpness() const {
    if (count < 3) return 0.0f;
    CrestMetrics m = computeCrestMetrics();
    if (!m.isComplete()) return 0.0f;
    float rise = m.crestTime - m.upTime;
    float fall = m.downTime - m.crestTime;
    float riseSlope = (rise > 0.0f) ? m.maxHeave / rise : 0.0f;
    float fallSlope = (fall > 0.0f) ? m.maxHeave / fall : 0.0f;
    return 0.5f * (riseSlope + fallSlope);
  }

  /**
   * Computes vertical asymmetry of the wave crest:
   * - Asymmetry = (rise - fall) / (rise + fall)
   * - Positive = sharper rise, flatter fall
   * - Negative = gentle rise, steep fall
   */
  float computeAsymmetry() {
    if (count < 3 || !detectLatestZeroUpcrossing()) return 0.0f;
    CrestMetrics m = computeCrestMetrics();
    if (!m.isComplete()) return 0.0f;
    float rise = m.crestTime - m.upTime;
    float fall = m.downTime - m.crestTime;
    if (rise + fall < EPSILON) return 0.0f;
    return (rise - fall) / (rise + fall);
  }

  /**
   * Predicts heave (vertical displacement) at a given wave-relative phase.
   * - `phase` is relative phase from current time [0,1)
   * - Uses interpolation of samples around last upcrossing
   * - Falls back to nearest sample match if interpolation fails
   */
  float predictAtPhase(float phase, float t) {
    if (count < 3 || !isFinite(t) || !isFinite(phase)) return 0.0f;
    float zc;
    if (!findLatestZeroUpcrossing(zc)) return 0.0f;

    float currentPhase = normalizePhase((t - zc) * freq);
    float targetPhase = normalizePhase(currentPhase + phase);

    int samplesPerPeriod = std::max(3, count / STORE_PERIODS);
    int start = -samplesPerPeriod;

    float bestHeave = 0.0f;
    float minPhaseDist = 10.0f;

    scanSamples(start, -1, [&](const WaveSample& s0, const WaveSample& s1) {
      float p0 = normalizePhase((s0.time - zc) * freq);
      float p1 = normalizePhase((s1.time - zc) * freq);

      bool inSegment = (p0 <= p1)
                       ? (p0 <= targetPhase && targetPhase <= p1)
                       : (targetPhase >= p0 || targetPhase <= p1);
      if (inSegment) {
        bestHeave = interpolateHeaveBetween(s0, s1, p0, p1, targetPhase);
        return true;
      }
      float h = nearestHeaveOfPair(s0, s1, p0, p1, targetPhase, minPhaseDist);
      if (h != 0.0f) bestHeave = h;
      return false;
    });
    return bestHeave;
  }

  /**
   * Estimates horizontal velocity gradient of wave train using max crest/trough.
   * Useful for studying wave propagation intensity.
   * - Gradient = (crest - trough) / time × waveSpeed
   */
  float computeWaveTrainVelocityGradient() const {
    float crest = -INFINITY, trough = INFINITY;
    float crestTime = 0.0f, troughTime = 0.0f;
    bool crestFound = false, troughFound = false;

    forEachSample([&](const WaveSample& s) {
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
    });

    float dt = fabsf(crestTime - troughTime);
    if (!crestFound || !troughFound || dt < EPSILON) return 0.0f;
    float dh = crest - trough;
    float waveSpeed = GRAVITY / (2 * M_PI * freq);
    return (dh / dt) * waveSpeed;
  }

  /**
   * Computes average wave energy per unit area (J/m²).
   * Formula: E = ½ ρ g (heave² avg)
   * - `waterDensity`: default is seawater (~1025 kg/m³)
   */
  float computeWaveEnergy(float waterDensity = 1025.0f, float gravity = GRAVITY) const {
    if (count < 1) return 0.0f;
    float sumSq = 0.0f;
    forEachSample([&](const WaveSample& s) {
      if (isFinite(s.heave)) {
        sumSq += s.heave * s.heave;
      }
    });
    return 0.5f * waterDensity * gravity * (sumSq / count);
  }

  /**
   * Returns raw circular buffer of all samples.
   */
  [[nodiscard]] const WaveSample* getSamples() const { return samples; }

  /**
   * Returns number of valid samples in buffer.
   */
  [[nodiscard]] int getCount() const { return count; }

  /**
   * Returns buffer capacity (maximum samples storable).
   */
  [[nodiscard]] int getCapacity() const { return N; }

  /**
   * Returns most recently added sample.
   */
  [[nodiscard]] WaveSample getLatestSample() const {
    return samples[wrapIdx(head - 1)];
  }
};

#endif // WAVE_SURFACE_PROFILE_H
