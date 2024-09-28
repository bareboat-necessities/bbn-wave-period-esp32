# BBN Boat Heave Sensor using IMU

Boat Heave Sensor on m5stack atomS3.

m5stack atomS3 is ESP32 microcontroller with built-in IMU MPU6886 (accelerometer and gyroscope)

Use arduino IDE to compile and upload sketch to esp32

Check required libraries and version in .github/workflows/build.yaml

Produces NMEA-0183 XDR sentences over USB (See NmeaXDR.h). Baud rate: 115200.

Mount atomS3 with LCD facing up.

## Estimating Boat Heave using IMU
Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32

The method for estimating wave height and heave from a moving boat implemented here is using the following on-line algorithm:

1. Sample MPU6886 3D acceleration and 3D gyroscope (angular velocities) measurements at about 250 Hz.
2. Estimate attitude and get attitude quaternion using Mahony agorithm. Using acceleration and gyroscope is enough. No magnetometer required because we only interested vertical acceleration for the next steps.
3. Double integrate vertical acceleration into vertical displacement using specially designed Kalman filter which corrects for integral drift in wave and corrects for the constant accelerometer bias.
4. Estimate observed heave frequency with Aranovskiy on-line filter. The correction for accelerometer bias is important for this step.
5. Smooth frequency produced by Aranovskiy filter with Kalman smoother.
6. Use another specially designed Kalman filter knowing the frequency and fusing model with trochoidal wave model to double integrate vertical acceleration. Assuming convergence of frequency this method would give real-time phase correction of heave compared to the first Kalman method. Doppler effect due to a boat moving relating to waves has no impact on displacement amplitude.

## Implementation Notes

* Rolling min/max algorithm with window of about couple wave periods samples to produce wave height measurement.
Algorithm:
https://github.com/lemire/runningmaxmin from Daniel Lemire paper, and improvements from: https://github.com/EvanBalster/STL_mono_wedge

## TODO

* Try to find a way for mpu6886 to sample at higher (than 250Hz) frequency
