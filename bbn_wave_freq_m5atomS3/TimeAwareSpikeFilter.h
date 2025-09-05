#ifndef TIME_AWARE_SPIKE_FILTER_H
#define TIME_AWARE_SPIKE_FILTER_H

/**
 * Copyright 2025, Mikhail Grushinskiy
 */

class TimeAwareSpikeFilter {
  private:
    struct Sample {
      float value;
      float deltaTime; // Time since previous sample
    };

    Sample *window;      // Circular buffer for previous values and time deltas
    int windowSize;      // Size of the moving window
    int currentIndex;    // Current position in the circular buffer
    float threshold;     // Derivative threshold for spike detection
    bool initialized;    // Flag to indicate if filter is initialized

    float *derivatives;
    float *values;
    float *temp;
      
  public:
    /**
       @brief Construct a new Spike Filter object
       @param size Size of the moving window (recommend 3-5)
       @param thr Threshold for spike detection (tune based on your signal)
    */
    TimeAwareSpikeFilter(int size, float thr) : windowSize(size), threshold(thr), initialized(false) {
      window = new Sample[windowSize];
      values = new float[windowSize];
      derivatives = new float[windowSize];
      temp = new float[windowSize];
      currentIndex = 0;
    }

    ~TimeAwareSpikeFilter() {
      delete[] temp;
      delete[] derivatives;
      delete[] values;
      delete[] window;
    }

    /**
       @brief Filtering when you already have the delta time
       @param input Raw input value
       @param deltaT Time since last sample
       @return Filtered output value
    */
    float filterWithDelta(float input, float deltaT) {
      if (!initialized) {
        // Initialize the window with the first value
        for (int i = 0; i < windowSize; i++) {
          window[i].value = input;
          window[i].deltaTime = deltaT; // Initialize
        }
        initialized = true;
        return input;
      }

      // Store the new value and time delta
      window[currentIndex].value = input;
      window[currentIndex].deltaTime = deltaT;
      currentIndex = (currentIndex + 1) % windowSize;

      // Calculate time-weighted derivatives
      for (int i = 0; i < windowSize; i++) {
        int currentIdx = (currentIndex + i) % windowSize;
        int prevIdx = (currentIndex + i - 1 + windowSize) % windowSize;

        float dt = window[currentIdx].deltaTime;

        derivatives[i] = (window[currentIdx].value - window[prevIdx].value) / dt;
      }

      // Find the median derivative (more robust than average)
      float medianDerivative = computeMedian(derivatives, windowSize);

      // Check if current derivative is a spike
      int prevIdx = (currentIndex - 1 + windowSize) % windowSize;
      float currentDeltaT = window[currentIndex].deltaTime;
      float currentDerivative = (window[currentIndex].value - window[prevIdx].value) / currentDeltaT;

      if (fabs(currentDerivative - medianDerivative) > threshold) {
        // Spike detected - replace with median of window values
        for (int i = 0; i < windowSize; i++) {
          values[i] = window[i].value;
        }
        float medianValue = computeMedian(values, windowSize);
        return medianValue;
      }

      // No spike - return the original value
      return input;
    }

  private:
    /**
       @brief Computes the median of an array
       @param arr Input array
       @param n Size of array
       @return Median value
    */
    float computeMedian(float *arr, int n) {
      // Create a copy to avoid modifying original
      for (int i = 0; i < n; i++) {
        temp[i] = arr[i];
      }

      // Sort the array (bubble sort for simplicity)
      for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
          if (temp[j] > temp[j + 1]) {
            float swap = temp[j];
            temp[j] = temp[j + 1];
            temp[j + 1] = swap;
          }
        }
      }

      // Return median
      if (n % 2 == 1) {
        return temp[n / 2];
      } else {
        return (temp[n / 2 - 1] + temp[n / 2]) / 2.0f;
      }
    }
};

#endif
