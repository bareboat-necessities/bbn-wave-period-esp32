#ifndef TrochoidalWave_h
#define TrochoidalWave_h

/*

 See https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
 
 */

float trochoid_wave_displacement(float displacement_amplitude, float frequency, float phase_rad, float t);
float trochoid_wave_vert_speed(float displacement_amplitude, float frequency, float phase_rad, float t);
float trochoid_wave_vert_accel(float displacement_amplitude, float frequency, float phase_rad, float t);

float trochoid_wave_length(float periodSec);
float trochoid_wave_period(float displacement, float accel);
float trochoid_wave_freq(float displacement, float accel);

float trochoid_wavelength(float omega);
float trochoid_wavenumber(float wavelength);
float trochoid_wave_speed(float k);

const float g_std = 9.80665; // standard gravity acceleration m/s2

// Function to compute wavelength (λ) from angular frequency (ω)
float trochoid_wavelength(float omega) {
  return (2 * M_PI * g_std) / (omega * omega);
}

// Function to compute wavenumber (k) from wavelength (λ)
float trochoid_wavenumber(float wavelength) {
  return (2 * M_PI) / wavelength;
}

// Function to compute wave speed (c) from wavenumber (k)
float trochoid_wave_speed(float k) {
  return sqrt(g_std / k);
}

// Function to compute wavelength (λ) from wave period (T)
float trochoid_wave_length(float periodSec) {
  float lengthMeters = g_std * periodSec * periodSec / (2 * PI);
  return lengthMeters;
}

// Function to compute wave period (T) from vertical displacement and vertical acceleration
float trochoid_wave_period(float displacement, float accel) {
  return 2.0 * PI * sqrt(fabs(displacement / accel));
}

// Function to compute wave frequency (f) from vertical displacement and vertical acceleration
float trochoid_wave_freq(float displacement, float accel) {
  return sqrt(fabs(accel / displacement)) / (2.0 * PI);
}

float trochoid_wave_displacement(float displacement_amplitude, float frequency, float phase_rad, float t) {
  return -displacement_amplitude * cos(2.0 * PI * frequency * t + phase_rad);
}

float trochoid_wave_vert_speed(float displacement_amplitude, float frequency, float phase_rad, float t) {
  return 2.0 * PI * frequency * displacement_amplitude * sin(2.0 * PI * frequency * t + phase_rad);
}

float trochoid_wave_vert_accel(float displacement_amplitude, float frequency, float phase_rad, float t) {
  return pow(2.0 * PI * frequency, 2) * displacement_amplitude * cos(2.0 * PI * frequency * t + phase_rad);
}

#endif
