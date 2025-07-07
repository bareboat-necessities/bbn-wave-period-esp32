#pragma once

enum WaveDirection {
  BACKWARD = -1,
  UNCERTAIN = 0,
  FORWARD = 1
};

class WaveDirectionDetector {
private:
  const float alpha;
  const float threshold;
  float prevVertAccel = NAN;
  float filteredP = 0.0f;

public:
  // waveAngle in radians (0=positive X, PI/2=positive Y)
  WaveDirectionDetector(float smoothing = 0.002f, 
                        float sensitivity = 0.005f)
    : alpha(smoothing), 
      threshold(sensitivity) {
  }

  // Processes X,Y,Z accelerations
  WaveDirection update(float accelX, float accelY, float accelZ, float delta_t) {
    float mag_a = sqrtf(accelX * accelX + accelY * accelY);
    if (std::isnan(prevVertAccel)) {
      prevVertAccel = accelZ;
      return UNCERTAIN;
    }
    if (mag_a > 1e-8f) {
      // Project X/Y onto wave direction axis
      float aHoriz = accelY > 0 ? mag_a : -mag_a;
      
      // Compute vertical slope
      float vertSlope = (accelZ - prevVertAccel) / delta_t;
      prevVertAccel = accelZ;
      
      // Update EMA filter
      filteredP += alpha * (aHoriz * vertSlope - filteredP);
    }
    
    // Threshold decision
    if (filteredP > threshold) return FORWARD;
    if (filteredP < -threshold) return BACKWARD;
    return UNCERTAIN;
  }

  float getFilteredP() const {
    return filteredP;
  }
};

