#pragma once

enum WaveDirection {
  UNCERTAIN,
  FORWARD,
  BACKWARD
};

class WaveDirectionDetector {
private:
  const float alpha;
  const float threshold;
  const float waveAngle; // Known wave axis angle (radians)
  float prevVertAccel = 0;
  float filteredP = 0;
  float cosAngle, sinAngle; // Pre-computed trig values

public:
  // waveAngle in radians (0=positive X, PI/2=positive Y)
  WaveDirectionDetector(float smoothing = 0.1f, 
                       float sensitivity = 0.03f,
                       float waveAngleRad = 0.0f)
    : alpha(smoothing), 
      threshold(sensitivity),
      waveAngle(waveAngleRad) {
    cosAngle = cos(waveAngleRad);
    sinAngle = sin(waveAngleRad);
  }

  // Processes X,Y,Z accelerations (all in G units)
  WaveDirection update(float accelX, float accelY, float accelZ, float delta_t) {
    // Project X/Y onto wave direction axis
    float aHoriz = accelX * cosAngle + accelY * sinAngle;
    
    // Compute vertical slope
    float vertSlope = (accelZ - prevVertAccel) / delta_t;
    prevVertAccel = accelZ;
    
    // Update EMA filter
    filteredP += alpha * (aHoriz * vertSlope - filteredP);
    
    // Threshold decision
    if (filteredP > threshold) return FORWARD;
    if (filteredP < -threshold) return BACKWARD;
    return UNCERTAIN;
  }

  float getFilteredP() const {
    return filteredP;
  }
};

