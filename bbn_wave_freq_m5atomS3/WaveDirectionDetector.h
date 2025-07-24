#pragma once

/*
  Copyright 2025, Mikhail Grushinskiy
*/

enum WaveDirection {
  BACKWARD = -1,
  UNCERTAIN = 0,
  FORWARD = 1
};

template <typename Real = float>
class WaveDirectionDetector {
private:
  const Real alpha;
  const Real threshold;
  Real prevVertAccel = NAN;
  Real filteredP = Real(0);

public:
  // waveAngle in radians (0=positive X, PI/2=positive Y)
  WaveDirectionDetector(Real smoothing = Real(0.002), 
                        Real sensitivity = Real(0.005))
    : alpha(smoothing), 
      threshold(sensitivity) {
  }

  // Processes X,Y,Z accelerations
  WaveDirection update(Real accelX, Real accelY, Real accelZ, Real delta_t) {
    Real mag_a = sqrtf(accelX * accelX + accelY * accelY);
    if (std::isnan(prevVertAccel)) {
      prevVertAccel = accelZ;
      return UNCERTAIN;
    }
    if (mag_a > Real(1e-8)) {
      // Project X/Y onto wave direction axis
      Real aHoriz = accelY > Real(0) ? mag_a : -mag_a;
      
      // Compute vertical slope
      Real vertSlope = (accelZ - prevVertAccel) / delta_t;
      prevVertAccel = accelZ;
      
      // Update EMA filter
      filteredP += alpha * (aHoriz * vertSlope - filteredP);
    }
    
    // Threshold decision
    if (filteredP > threshold) return FORWARD;
    if (filteredP < -threshold) return BACKWARD;
    return UNCERTAIN;
  }

  Real getFilteredP() const {
    return filteredP;
  }
};

