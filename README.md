# bbn-wave-period-esp32
Estimate ocean wave displacement using IMU on esp32

The method for estimating wave height and heave from a moving boat implemented here is using the following algorithm:

1. Estimate observed wave frequency using vertical boat acceleration measurements from IMU with Aranovskiy on-line filter. A frequency of acceleration sampling is important. This code seems able to achieve about 250Hz acceleration sampling from MPU6886. Lower frequencies wouldn’t give reasonable results from Aranovskiy filter. 
1. Assume the waves follow trochoidal model. Trochoidal wave would still be observed as trochoidal from a vessel moving with constant speed. I.e. vertical movement is harmonic over time 
1. Calculate observed wave period and calculate wave length using trochoidal wave model. 
1. In trochoidal model horizontal wave displacement would be simply proportional to measured vertical acceleration. Doppler effect has no impact on displacement (amplitude). Although though it does on observed wave frequency and vertical acceleration in it. 
1. Calculate the coefficient for previous step using known wave length in trochoidal model. 
1. This gives a simple formula to produce vertical displacement from measured vertical acceleration. Valid in the considered models even by observations from a moving vessel (assuming speed doesn’t change much)

