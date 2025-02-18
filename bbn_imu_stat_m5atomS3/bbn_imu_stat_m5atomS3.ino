/*
  Measure IMU standard deviations

  Copyright 2024, Mikhail Grushinskiy
*/

#include <M5Unified.h>
#include <Arduino.h>

#include "Statistic.h"

statistic::Statistic<float, uint32_t, true> accelX;
statistic::Statistic<float, uint32_t, true> accelY;
statistic::Statistic<float, uint32_t, true> accelZ;
statistic::Statistic<float, uint32_t, true> gyroX;
statistic::Statistic<float, uint32_t, true> gyroY;
statistic::Statistic<float, uint32_t, true> gyroZ;

const char* imu_name;

unsigned long last_update = 0UL, now = 0UL;

#define SAMPLES_COUNT 2500

int update_stat(statistic::Statistic<float, uint32_t, true> *stats, float value, const char* name) {
  stats->add(value);
  if (stats->count() == SAMPLES_COUNT) {
    now = micros();
    Serial.println(name);
    Serial.print("        Count: ");
    Serial.println(stats->count());
    Serial.print("  Sample Rate: ");
    Serial.println(stats->count() / ((now - last_update) / 1000000.0));
    Serial.print("          Min: ");
    Serial.println(stats->minimum(), 7);
    Serial.print("          Max: ");
    Serial.println(stats->maximum(), 7);
    Serial.print("      Average: ");
    Serial.println(stats->average(), 7);
    Serial.print("     variance: ");
    Serial.println(stats->variance(), 7);
    Serial.print("    pop stdev: ");
    Serial.println(stats->pop_stdev(), 7);
    Serial.print(" unbias stdev: ");
    Serial.println(stats->unbiased_stdev(), 7);
    Serial.print(" time(micros): ");
    Serial.printf("%d\n", now);
    Serial.println("=====================================");
    stats->clear();
  }
  return stats->count();
}

void read_and_processIMU_data() {
  m5::imu_3d_t accel;
  M5.Imu.getAccelData(&accel.x, &accel.y, &accel.z);

  m5::imu_3d_t gyro;
  M5.Imu.getGyroData(&gyro.x, &gyro.y, &gyro.z);

  int count = 
  update_stat(&gyroX, gyro.x, "gyroX");
  update_stat(&gyroY, gyro.y, "gyroY");
  update_stat(&gyroZ, gyro.z, "gyroZ");

  update_stat(&accelX, accel.x, "accelX");
  update_stat(&accelY, accel.y, "accelY");
  update_stat(&accelZ, accel.z, "accelZ");

  if (count == 0) {
    last_update = micros();
  }
}

void repeatMe() {
  static uint32_t prev_sec = 0;
  auto imu_update = M5.Imu.update();
  if (imu_update) {
    read_and_processIMU_data();
  }
  int32_t sec = millis() / 1000;
  if (prev_sec != sec) {
    prev_sec = sec;
    if ((sec & 7) == 0) {
      // prevent WDT.
      vTaskDelay(1);
    }
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);

  auto imu_type = M5.Imu.getType();
  switch (imu_type) {
    case m5::imu_none:        imu_name = "not found";   break;
    case m5::imu_sh200q:      imu_name = "sh200q";      break;
    case m5::imu_mpu6050:     imu_name = "mpu6050";     break;
    case m5::imu_mpu6886:     imu_name = "mpu6886";     break;
    case m5::imu_mpu9250:     imu_name = "mpu9250";     break;
    case m5::imu_bmi270:      imu_name = "bmi270";      break;
    default:                  imu_name = "unknown";     break;
  };

  if (imu_type == m5::imu_none) {
    for (;;) {
      delay(1);
    }
  }

  // Read calibration values from NVS.
  if (!M5.Imu.loadOffsetFromNVS()) {
    Serial.println("Could not load calibration data!");
  }

  gyroX.clear();
  gyroY.clear();
  gyroZ.clear();

  accelX.clear();
  accelY.clear();
  accelZ.clear();

  delay(200);
  last_update = micros();
}

void loop(void) {
  M5.update();
  repeatMe();
  delayMicroseconds(3000);
}
