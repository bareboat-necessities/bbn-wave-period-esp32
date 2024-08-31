#ifndef TrochoidalWave_h
#define TrochoidalWave_h

/*

 See https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
 
 */

float trochoid_wave_length(float periodSec);

const float g_std = 9.80665; // standard gravity acceleration m/s2

float trochoid_wave_length(float periodSec) {
  float lengthMeters = g_std * periodSec * periodSec / (2 * PI);
  return lengthMeters;
}

#endif
