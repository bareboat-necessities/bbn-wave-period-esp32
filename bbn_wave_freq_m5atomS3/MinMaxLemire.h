#ifndef MinMaxLemire_h
#define MinMaxLemire_h

#include "MonoWedge.h"

#include <deque>
#include <cmath>

typedef struct Sample {
  float value;
  uint32_t timeMicroSec;
  bool operator<(const Sample& o) const {
    return value < o.value;
  }
  bool operator>(const Sample& o) const {
    return value > o.value;
  }
} SampleType;

typedef struct min_max_lemire {
  Sample min;
  Sample max;
  std::deque<Sample> min_wedge;
  std::deque<Sample> max_wedge;
} MinMaxLemire;

/**
 * Running min max
 */
void min_max_lemire_update(MinMaxLemire* minMax, Sample sample, uint32_t window_size_micro_sec);

void min_max_lemire_update(MinMaxLemire* minMax, Sample sample, uint32_t window_size_micro_sec) {
  mono_wedge::max_wedge_update(minMax->max_wedge, sample);
  // Get rid of old samples outside our rolling range
  while (sample.timeMicroSec - minMax->max_wedge.front().timeMicroSec > window_size_micro_sec) {
    auto front = minMax->max_wedge.front();
    minMax->max_wedge.pop_front();
  }
  // The maximum value is at the front of the wedge.
  auto maximumInRange = minMax->max_wedge.front();
  minMax->max = maximumInRange;

  mono_wedge::min_wedge_update(minMax->min_wedge, sample);
  // Get rid of old samples outside our rolling range
  while (sample.timeMicroSec - minMax->min_wedge.front().timeMicroSec > window_size_micro_sec) {
    auto front = minMax->min_wedge.front();
    minMax->min_wedge.pop_front();
  }
  // The minimum value is at the front of the wedge.
  auto minimumInRange = minMax->min_wedge.front();
  minMax->min = minimumInRange;
}

#endif
