# bbn-wave-period-esp32

## Estimating Wave Height using IMU
Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32

The method for estimating wave height and heave from a moving boat implemented here is using the following algorithm:

1. Estimate observed wave frequency using vertical boat acceleration measurements from IMU with Aranovskiy on-line filter. A frequency of acceleration sampling is important. This code seems able to achieve about 250Hz acceleration sampling from MPU6886. Lower frequencies wouldn’t give reasonable results from Aranovskiy filter. 
1. Assume the waves follow trochoidal model. Trochoidal wave would still be observed as trochoidal from a vessel moving with constant speed. I.e. vertical movement is harmonic over time 
1. Calculate observed wave period and calculate wave length using trochoidal wave model. 
1. In trochoidal model horizontal wave displacement would be simply proportional to measured vertical acceleration. Doppler effect has no impact on displacement (amplitude). Although though it does on observed wave frequency and vertical acceleration in it. 
1. Calculate the coefficient for previous step using known wave length in trochoidal model. 
1. This gives a simple formula to produce vertical displacement from measured vertical acceleration. Valid in the considered models even by observations from a moving vessel (assuming speed doesn’t change much)
1. So this method avoids double integration of acceleration. It approximates a wave with trochoidal wave parameterized by two variables (frequency and max vertical acceleration in it) by observations from IMU on a moving boat. There is another way to use Kalman filter (with drift correction) which I might add later. See: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html
However due to high accelerometer noise and low sample frequency I do not have high hopes for that Kalman filter. 

## Implementation Notes

* Tilt compensation. No magnetometer needed as we only need vertical projection of acceleation. Mahony algorithm using accel and gyro (without mag) is enough. Quaternion rotation is done for esimating vertical acceleration
* Rolling min/max algorithm with window of about three wave periods samples to produce wave height measurement.
Algorithm: https://github.com/lemire/runningmaxmin from Daniel Lemire paper

## TODO

* Try to find a way for mpu6886 to sample at higher (than 250Hz) frequency

* Implement Kalman alorithm for wave estimation (the one mentioned here: https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html) And compare with trochoidal model results.

* Generate some reference data, test cases with it and produce some graphical charts. There are many parameterized filters used and gains and coefficients
need to be fine tuned for better results.

* Produce NMEA 0183 XDR sentences for heave, wave height, etc to serial port of m5atomS3
