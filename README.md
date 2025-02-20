# BBN Boat Heave Sensor using IMU

Boat Heave Sensor on m5stack atomS3.

m5stack atomS3 is ESP32 microcontroller with built-in IMU MPU6886 (accelerometer and gyroscope)

<p align="center" style="width: 50vw;" >
<img src="./images/BBN-heave-sensor-enclosure.jpg?raw=true" alt="BBN Heave Sensor" />
</p>

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
4. Estimate observed heave frequency with Aranovskiy on-line filter (without need for FFT). The correction for accelerometer bias is important for this step.
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

$$
\begin{flalign}
& \large v _k = v _{k-1} + aT - \hat{a} _{k-1}T &
\end{flalign}
$$

displacement:

$$
\begin{flalign}
& \large y _k = y _{k-1} + v _{k-1}T + {a \over 2}T^2 - {\hat{a} _{k-1} \over 2}T^2 &
\end{flalign}
$$

displacement integral:

$$
\begin{flalign}
& \large z _k = z _{k-1} + y _{k-1}T + {v _{k-1} \over 2}T^2 + {a \over 6}T^3 - {\hat{a} _{k-1} \over 6}T^3 &
\end{flalign}
$$

accelerometer bias:

$$
\begin{flalign}
& \large \hat {a} _k = \hat {a} _{k-1} &
\end{flalign}
$$

State vector:

$$
\begin{flalign}
&
\large
x = 
\begin{bmatrix}
z \\
y \\
v \\
\hat {a}
\end{bmatrix}
&
\end{flalign}
$$


Process model in matrix form:

$$
\begin{flalign}
& 
\large 
x _k = Fx _{k-1} + B u _k + w _k
&
\end{flalign}
$$

$w _k$ - zero mean noise,
$u _k = a$


Input $a$ - vertical acceleration from accelerometer

Measurement $z = 0$ (displacement integral)

Observation matrix:

$$
\begin{flalign}
&
\large
H = 
\begin{bmatrix}
1 \\
0 \\
0 \\
0
\end{bmatrix}
&
\end{flalign}
$$

Process matrix:

$$
\begin{flalign}
& \large
F = 
\begin{bmatrix}
1, & T, & {1 \over 2}T^2, & -{1 \over 6}T^3\\
0, & 1, & T, &       -{1 \over 2}T^2\\
0, & 0, & 1, &       -T\\
0, & 0, & 0, &       1
\end{bmatrix}
&
\end{flalign}
$$

Transition matrix:

$$
\begin{flalign}
& \large
B = 
\begin{bmatrix}
{1 \over 6}T^3\\
{1 \over 2}T^2\\
T\\
0
\end{bmatrix}
&
\end{flalign}
$$

### Kalman Filter #2

Kalman filter to estimate vertical displacement in wave using accelerometer, 
correct for accelerometer bias, estimate accelerometer bias. This method
assumes that displacement follows trochoidal model and the frequency of
wave is known. Frequency can be estimated using another step with Aranovskiy filter.

In trochoidal wave model there is simple linear dependency between displacement and 
acceleration.

$y$ - displacement (at any time):

$$
\begin{flalign}
& \large
y = - {L \over {2 \pi}}  {a \over g}
&
\end{flalign}
$$

$g$ - acceleration of free fall constant, 
$a$ - vertical acceleration

wave length L: 

$$
\begin{flalign}
& \large
L = { {period^2} g \over {2 \pi}}
&
\end{flalign}
$$


wave period via frequency:

$$
\begin{flalign}
& \large
period = {1 \over f}
&
\end{flalign}
$$

acceleration:

$$
\begin{flalign}
& \large
a = - (2  \pi  f)^2  y
&
\end{flalign}
$$

let

$$
\begin{flalign}
& \large
\hat{k} = - (2 \pi f)^2
&
\end{flalign}
$$

Process model:

displacement integral:

$$
\begin{flalign}
& \large z _k = z _{k-1} + y _{k-1}T + {v _{k-1} \over 2}T^2 + {a _{k-1} \over 6}T^3 - {\hat{a} _{k-1} \over 6}T^3 &
\end{flalign}
$$

displacement:

$$
\begin{flalign}
& \large y _k = y _{k-1} + v _{k-1}T + {a _{k-1} \over 2}T^2 - {\hat{a} _{k-1} \over 2}T^2 &
\end{flalign}
$$

velocity:

$$
\begin{flalign}
& \large v _k = v _{k-1} + a _{k-1}T - \hat{a} _{k-1}T &
\end{flalign}
$$


acceleration (from trochoidal wave model):

$$
\begin{flalign}
& \large  a _k = \hat{k}y _{k-1} + \hat{k} v _{k-1}T + \hat{k}{a _{k-1} \over 2}T^2 - \hat{k}{\hat{a} _{k-1} \over 2}T^2  &
\end{flalign}
$$


accelerometer bias:

$$
\begin{flalign}
& \large \hat {a} _k = \hat {a} _{k-1} &
\end{flalign}
$$


Process model in matrix form:

$$
\begin{flalign}
& 
\large 
x _k = Fx _{k-1} + B u _k + w _k
&
\end{flalign}
$$

$w _k$ - zero mean noise,
$u _k = 0$

State vector:

$$
\begin{flalign}
&
\large
x = 
\begin{bmatrix}
z \\
y \\
v \\
a \\
\hat {a}
\end{bmatrix}
&
\end{flalign}
$$


Input $a$ - vertical acceleration from accelerometer

Measurements:
    
$a$ (vertical acceleration), $z$ = 0

Observation matrix:

$$
\begin{flalign}
&
\large
H = 
\begin{bmatrix}
1, & 0 \\
0, & 0 \\
0, & 0 \\
0, & 1 \\
0, & 0
\end{bmatrix}
&
\end{flalign}
$$

Process matrix:

$$
\begin{flalign}
& \large
F = 
\begin{bmatrix}
1, & T,       & {1 \over 2}T^2, & {1 \over 6}T^3,        & -{1 \over 6}T^3\\
0, & 1,       & T,              & {1 \over 2}T^2,        & -{1 \over 2}T^2\\
0, & 0,       & 1,              & T,                     & -T\\
0, & \hat{k}, & \hat{k}T,       & {1 \over 2}\hat{k}T^2, & -{1 \over 2}\hat{k}T^2\\
0, & 0,       & 0,              & 0,                     & 1
\end{bmatrix}
&
\end{flalign}
$$

       
### Implementation Notes

* Rolling min/max algorithm with window of about a couple wave period samples to produce wave height measurements.
Algorithm:
https://github.com/lemire/runningmaxmin from Daniel Lemire paper, and improvements from: https://github.com/EvanBalster/STL_mono_wedge

### Results on Reference Data

![BBN Boat Heave Sensor Results](bbn_wave_freq_m5atomS3/tests/wave_results.png?raw=true "BBN Boat Heave Sensor Results")

## Integration with Bareboat Necessities (BBN) OS

![BBN Boat Heave Sensor Display](bbn_wave_freq_m5atomS3/tests/bbn_heave.png?raw=true "BBN Boat Heave Sensor Display")

## Applications

- Provide real-time heave data for ships with active heave compensation system (pipe laying ships, crane platforms, etc)
- Recording sea state
- Prediction of likelihood of sea sickness
- Estimating risk of breaking from anchor
- Autotuning gains of autopilots

## Flashing Firmware

````
wget https://github.com/bareboat-necessities/bbn-wave-period-esp32/releases/download/v1.0.2/bbn_wave_freq_m5atomS3_bin-2025-02-19.zip
unzip bbn_wave_freq_m5atomS3_bin-2025-02-19.zip 
/srv/esphome/bin/esptool.py  --chip esp32s3 --port "/dev/ttyACM0" --baud 921600  --before default_reset --after hard_reset write_flash 0x0 bbn_wave_freq_m5atomS3_firmware.bin
````

### On Bareboat Necessities OS

````
# shutdown signalk
sudo systemctl stop signalk

if [ -f bbn-flash-m5-wave.sh ]; then rm bbn-flash-m5-wave.sh; fi
wget https://raw.githubusercontent.com/bareboat-necessities/my-bareboat/refs/heads/master/m5stack-tools/bbn-flash-m5-wave.sh
chmod +x bbn-flash-m5-wave.sh
./bbn-flash-m5-wave.sh -p /dev/ttyACM0
````

## TODO

* Instead of chaining Kalman filters
it would probably be better to implement it as a single state one Kalman filter
* Try to find a way for mpu6886 to sample at higher (than 250Hz) frequency
* The method can be improved for less regular waves by splitting original signal with band pass filter into several signals
of different frequencies bands. Then it would be possible to apply the method separately to each band and sum up the results
received for each band. Effectively it would approximate the signal spectrum and identify main frequency within each band,
giving possibly better results. Coefficients used by filters can be tuned better for each band to give faster convergence.

## Project Home

https://bareboat-necessities.github.io/

## Other sensors by Bareboat Necessities

https://github.com/bareboat-necessities/bbn-m5atomS3-lite

# References

1. Alexey A. Bobtsov, Nikolay A. Nikolaev, Olga V. Slita, Alexander S. Borgul, Stanislav V. Aranovskiy: The New Algorithm of Sinusoidal Signal Frequency Estimation. [11th IFAC International Workshop on
Adaptation and Learning in Control and Signal Processing, 2013](https://www.sciencedirect.com/science/article/pii/S1474667016329421)

2. Sharkh S. M., Hendijanizadeh2 M., Moshrefi-Torbati3 M., Abusara M. A.: A Novel Kalman Filter Based Technique for Calculating the Time History of Vertical Displacement of a Boat from Measured Acceleration, [Marine Engineering Frontiers Volume 2, 2014](https://www.researchgate.net/profile/Mehdi-Hendijanizadeh/publication/264713649_A_Novel_Kalman_Filter_Based_Technique_for_Calculating_the_Time_History_of_Vertical_Displacement_of_a_Boat_from_Measured_Acceleration/links/53ec88db0cf24f241f1584c5/A-Novel-Kalman-Filter-Based-Technique-for-Calculating-the-Time-History-of-Vertical-Displacement-of-a-Boat-from-Measured-Acceleration.pdf "Marine Engineering Frontiers Volume 2, 2014")

3. Daniel Lemire, [Streaming Maximum-Minimum Filter Using No More than 
Three Comparisons per Element](http://arxiv.org/abs/cs.DS/0610046). Nordic Journal of Computing, 13 (4), pages 328-339, 2006.

4. Trochoidal Wave [Wikipedia](https://en.wikipedia.org/wiki/Trochoidal_wave)

