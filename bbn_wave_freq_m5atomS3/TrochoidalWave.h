#ifndef TrochoidalWave_h
#define TrochoidalWave_h

/*

 See https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
 
 */

double trochoid_wave_length(double periodSec);

const double g_std = 9.80665; // standard gravity acceleration m/s2

double trochoid_wave_length(double periodSec) {
  double lengthMeters = g_std * periodSec * periodSec / (2 * PI);
  return lengthMeters;
}

#endif
