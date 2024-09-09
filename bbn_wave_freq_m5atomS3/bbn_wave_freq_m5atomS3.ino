#ifndef TrochoidalWave_h
#define TrochoidalWave_h

/*

 See https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
 
 */

float trochoid_wave_length(float periodSec);
float trochoid_wave_period(float height, float accel);
float trochoid_wave_freq(float height, float accel);

const float g_std = 9.80665; // standard gravity acceleration m/s2

float trochoid_wave_length(float periodSec) {
  float lengthMeters = g_std * periodSec * periodSec / (2 * PI);
  return lengthMeters;
}

float trochoid_wave_period(float height, float accel) {
  float wave_period = 2.0 * PI * sqrt(fabs(height / accel));
  return wave_period;
}

float trochoid_wave_freq(float height, float accel) {
  float wave_freq = sqrt(fabs(accel / height)) / (2.0 * PI);
  return wave_freq;
}


#endif
