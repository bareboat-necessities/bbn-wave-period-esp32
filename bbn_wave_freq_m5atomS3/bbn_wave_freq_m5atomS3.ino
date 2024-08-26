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
#include <utility/imu/MPU6886_Class.hpp>

unsigned long now = 0UL, last_refresh = 0UL, last_update = 0UL;
int got_samples = 0;

double omega_up = 1.0 * (2 * PI);  // upper frequency Hz * 2 * PI
double a = omega_up;
double b = a;
double k = 2.0;
double theta_0 = - omega_up * omega_up / 4.0;
double x1_0 = 0.0;
double sigma_0 = theta_0;
double delta_t;  // time step sec

// initialize variables
double t = 0.0;
double x1 = x1_0;
double theta = theta_0;
double sigma = sigma_0;
double y, x1_dot, sigma_dot, omega, f;

void repeatMe() {
  auto imu_update = M5.Imu.update();
  if (imu_update) {
    m5::imu_3d_t accel;
    M5.Imu.getAccel(&accel.x, &accel.y, &accel.z);
    got_samples++;

    now = millis();
    delta_t = ((now - last_update) / 1000.0f);
    last_update = now;

    y = accel.z - 1.0 /* since it includes g */;
    //y = sin(2*PI*t*0.25); // dummy test data
    x1_dot = - a * x1 + b * y;
    sigma_dot = - k * x1 * x1 * theta - k * a * x1 * x1_dot - k * b * x1_dot * y;
    theta = sigma + k * b * x1 * y;
    omega = sqrt(abs(theta));
    f = omega / (2.0 * PI);

    x1 = x1 + x1_dot * delta_t;
    sigma = sigma + sigma_dot * delta_t;
    t = t + delta_t;

    if (now - last_refresh >= 200) {
      M5.Lcd.setCursor(0, 10);
      M5.Lcd.clear();  // Delay 100ms

      M5.Lcd.printf("IMU:\n\n");
      M5.Lcd.printf("sec: %d\n\n", now / 1000);
      M5.Lcd.printf("period sec: %0.4f\n\n", (f > 0 ? 1.0/f : 9999.0));
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
  //auto imu6886 = ((m5::MPU6886_Class*) &M5.Imu);
  //imu6886->enableFIFO(imu6886->ODR_1kHz); 
  //imu6886->setAccelFsr(imu6886->AFS_2G);
  last_update = millis();
}

void loop(void) {
  M5.update();
  delay(2);
  repeatMe();
}
