/*

  Ocean wave frequency estimator using esp32 (m5atomS3)

  See: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html

  Instead of FFT method for finding main wave frequency we could use Aranovskiy frequency estimator which is a simple on-line filter.

  Ref:

  Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy

  The New Algorithm of Sinusoidal Signal Frequency Estimation.

  11th IFAC International Workshop on Adaptation and Learning in Control and Signal Processing July 3-5, 2013. Caen, France

*/

#include <M5Unified.h>
#include "AranovskiyFilter.h"
#include "KalmanSmoother.h"

unsigned long now = 0UL, last_refresh = 0UL, last_update = 0UL;
int got_samples = 0;

AranovskiyParams params;
AranovskiyState  state;

double omega_up = 5.0 * (2 * PI);  // upper frequency Hz * 2 * PI
double k_gain = 2.0;

double t_0 = 0.0;
double x1_0 = 0.0;
double theta_0 = - (omega_up * omega_up / 4.0);
double sigma_0 = theta_0;

double delta_t;  // time step sec

KalmanSmootherVars kalman;

int first = 1;

void repeatMe() {
  auto imu_update = M5.Imu.update();
  if (imu_update) {
    m5::imu_3d_t accel;
    M5.Imu.getAccelData(&accel.x, &accel.y, &accel.z);
    got_samples++;

    now = micros();
    delta_t = ((now - last_update) / 1000000.0);
    last_update = now;

    double y = accel.z - 1.0 /* since it includes g */;
    //double y = sin(2 * PI * state.t * 0.25); // dummy test data

    aranovskiy_update(&params, &state, y, delta_t);

    if (first) {
      kalman_smoother_set_initial(&kalman, state.f);
      first = 0;
    }
    double freq_adj = kalman_smoother_update(&kalman, state.f);

    if (now - last_refresh >= 1000000) {
      M5.Lcd.setCursor(0, 10);
      M5.Lcd.clear();  // Delay 100ms

      M5.Lcd.printf("IMU:\n\n");
      M5.Lcd.printf("sec: %d\n\n", now / 1000000);
      M5.Lcd.printf("period sec: %0.4f\n\n", (state.f > 0 ? 1.0 / state.f : 9999.0));
      M5.Lcd.printf("period adj: %0.4f\n\n", (freq_adj > 0 ? 1.0 / freq_adj : 9999.0));
      M5.Lcd.printf("samples: %d\n\n", got_samples);
      M5.Lcd.printf("%0.3f %0.3f %0.3f\n\n", accel.x, accel.y, accel.z - 1.0);

      last_refresh = now;
      got_samples = 0;
    }
  }
}

void setup(void) {
  auto cfg = M5.config();
  M5.begin(cfg);

  aranovskiy_default_params(&params, omega_up, k_gain);
  aranovskiy_init_state(&state, t_0, x1_0, theta_0, sigma_0);
  kalman_smoother_init(&kalman, 0.003, 10.0, 100.0);

  last_update = micros();
}

void loop(void) {
  M5.update();
  delay(3);
  repeatMe();
}

