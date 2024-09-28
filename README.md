# bbn-wave-period-esp32

Boat Heave Sensor on m5stack atomS3.

m5stack atomS3 is ESP32 microcontroller with built-in IMU MPU6886 (accelerometer and gyroscope)

Use arduino IDE to compile and upload sketch to esp32

Check required libraries and version in .github/workflows/build.yaml

Produces NMEA-0183 XDR sentences over USB (See NmeaXDR.h). Baud rate: 115200.

Mount atomS3 with LCD facing up.

## Estimating Wave Height using IMU
Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32

The method for estimating wave height and heave from a moving boat implemented here is using the following algorithm:

1. Estimate observed wave frequency using vertical boat acceleration measurements from IMU with Aranovskiy on-line filter. A frequency of acceleration sampling is important. This code seems able to achieve about 250Hz acceleration sampling from MPU6886. Lower frequencies wouldn’t give reasonable results from Aranovskiy filter. 
1. Assume the waves follow trochoidal model. Trochoidal wave would still be observed as trochoidal from a vessel moving with constant speed. I.e. vertical movement is harmonic over time 
1. Calculate observed wave period and calculate wave length using trochoidal wave model. 
1. In trochoidal model horizontal wave displacement would be simply proportional to measured vertical acceleration. Doppler effect has no impact on displacement (amplitude). Although though it does on observed wave frequency and vertical acceleration in it. 
1. Calculate the coefficient for previous step using known wave length in trochoidal model. 
1. This gives a simple formula to produce vertical displacement from measured vertical acceleration. Valid in the considered models even by observations from a moving vessel (assuming speed doesn’t change much)
1. So this method avoids double integration of acceleration. It approximates a wave with trochoidal wave parameterized by two variables (frequency and max vertical acceleration in it) by observations from IMU on a moving boat.
1. There is another way to use Kalman filter (with drift correction) which I might add later. See: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
However due to high accelerometer noise and low sample frequency scaling of heave is an issue. Plugging in some values periodically from trochoidal model to create sensor fusion will make it more accurate. 

## Implementation Notes

* Tilt compensation. No magnetometer needed as we only need vertical projection of acceleation. Mahony algorithm using accel and gyro (without mag) is enough. Quaternion rotation is done for esimating vertical acceleration
* Rolling min/max algorithm with window of about three wave periods samples to produce wave height measurement.
Algorithm:
https://github.com/lemire/runningmaxmin from Daniel Lemire paper, and improvements from: https://github.com/EvanBalster/STL_mono_wedge

## TODO

* Try to find a way for mpu6886 to sample at higher (than 250Hz) 
