#ifndef TrochoidalWave_h
#define TrochoidalWave_h

/*

 See https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
 
 */

float trochoid_wave_length(float periodSec);
float trochoid_wave_period(float wave_height, float amp_range);

const float g_std = 9.80665; // standard gravity acceleration m/s2

float trochoid_wave_length(float periodSec) {
  float lengthMeters = g_std * periodSec * periodSec / (2 * PI);
  return lengthMeters;
}

float trochoid_wave_period(float wave_height, float amp_range) {
  float wave_period = 2.0 * PI * sqrt(fabs(wave_height / amp_range));
  return wave_period;
}


#endif
