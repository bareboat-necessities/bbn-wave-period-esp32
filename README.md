# BBN Boat Heave Sensor using IMU

Boat Heave Sensor on m5stack atomS3.

m5stack atomS3 is ESP32 microcontroller with built-in IMU MPU6886 (accelerometer and gyroscope)

Use arduino IDE to compile and upload sketch to esp32

Check required libraries and version in .github/workflows/build.yaml

Produces NMEA-0183 XDR sentences over USB (See NmeaXDR.h). Baud rate: 115200.

Mount atomS3 with LCD facing up.

## Estimating Boat Heave using IMU
Estimate vessel heave (vertical displacement) in ocean waves using IMU on esp32

The method for estimating wave height and heave from a moving boat implemented here using the following on-line algorithm:

1. Sample MPU6886 3D acceleration and 3D gyroscope (angular velocities) measurements at about 250 Hz.
2. Estimate attitude and get attitude quaternion using Mahony algorithm. Using acceleration and gyroscope is enough. No magnetometer is required because we are only interested in vertical acceleration for the next steps.
3. Double integrate vertical acceleration into vertical displacement using specially designed Kalman filter which corrects for integral drift in wave and corrects for the constant accelerometer bias.
4. Estimate observed heave frequency with Aranovskiy on-line filter. The correction for accelerometer bias is important for this step.
5. Smooth frequency produced by Aranovskiy filter with Kalman smoother.
6. Use another specially designed Kalman filter knowing the frequency and fusing model with trochoidal wave model to double integrate vertical acceleration. Assuming convergence of frequency, this method would give real-time phase correction of heave compared to the first Kalman method. Doppler effect due to boat movement in waves has no impact on displacement amplitude.

### Kalman Filter #1

Kalman filter to double integrate vertical acceleration in wave
into vertical displacement, correct for accelerometer bias,
estimate accelerometer bias, correct integral for zero average displacement.
The third integral (responsible for zero average vertical displacement)
is taken as a measurement of zero.

Process model:

velocity:

$$  v_k = v_{k-1} + aT - \hat{a}_{k-1}T  $$

displacement:

$$  y_k = y_{k-1} + v_{k-1}T + {a \over 2}T^2 - {\hat{a}_{k-1} \over 2}T^2  $$

displacement_integral:

$$  z_k = z_{k-1} + y_{k-1}T + {v_{k-1} \over 2}T^2 + {a \over 6}T^3 - {\hat{a}_{k-1} \over 6}T^3  $$

accelerometer bias:

$$  \hat {a}_k = \hat {a}_{k-1}  $$


### Implementation Notes

* Rolling min/max algorithm with window of about a couple wave period samples to produce wave height measurements.
Algorithm:
https://github.com/lemire/runningmaxmin from Daniel Lemire paper, and improvements from: https://github.com/EvanBalster/STL_mono_wedge

### Results on Reference Data

![BBN Boat Heave Sensor Results](bbn_wave_freq_m5atomS3/tests/wave_results.png?raw=true "BBN Boat Heave Sensor Results")

## Integration with Bareboat Necessities (BBN) OS

![BBN Boat Heave Sensor Display](bbn_wave_freq_m5atomS3/tests/bbn_heave.png?raw=true "BBN Boat Heave Sensor Display")

## TODO

* Try to find a way for mpu6886 to sample at higher (than 250Hz) frequency

# References

1. Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy: The New Algorithm of Sinusoidal Signal Frequency Estimation. [11th IFAC International Workshop on
Adaptation and Learning in Control and Signal Processing, 2013](https://www.sciencedirect.com/science/article/pii/S1474667016329421)

2. Sharkh S. M., Hendijanizadeh2 M., Moshrefi-Torbati3 M., Abusara M. A.: A Novel Kalman Filter Based Technique for Calculating the Time History of Vertical Displacement of a Boat from Measured Acceleration, [Marine Engineering Frontiers Volume 2, 2014](https://www.researchgate.net/profile/Mehdi-Hendijanizadeh/publication/264713649_A_Novel_Kalman_Filter_Based_Technique_for_Calculating_the_Time_History_of_Vertical_Displacement_of_a_Boat_from_Measured_Acceleration/links/53ec88db0cf24f241f1584c5/A-Novel-Kalman-Filter-Based-Technique-for-Calculating-the-Time-History-of-Vertical-Displacement-of-a-Boat-from-Measured-Acceleration.pdf "Marine Engineering Frontiers Volume 2, 2014")

3. Daniel Lemire, [Streaming Maximum-Minimum Filter Using No More than 
Three Comparisons per Element](http://arxiv.org/abs/cs.DS/0610046). Nordic Journal of Computing, 13 (4), pages 328-339, 2006.

4. Trochoidal Wave [Wikipedia](https://en.wikipedia.org/wiki/Trochoidal_wave)

