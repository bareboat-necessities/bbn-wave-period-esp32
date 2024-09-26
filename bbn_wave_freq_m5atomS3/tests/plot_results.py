import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt(fname="results.csv", delimiter=",",
                  usecols=(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27),
                  skiprows=0)

Time = Data[:, [0]]
AccX = Data[:, [1]]
RefVelX = Data[:, [2]]
RefPosX = Data[:, [3]]
Heave = Data[:, [4]]
HeaveAlt = Data[:, [5]]
Freq = Data[:, [10]]
FreqAdj = Data[:, [11]]

f, axarr = plt.subplots(3, sharex="all")

axarr[0].set_title('Acceleration')
axarr[0].plot(Time, AccX, label="Input AccX")
axarr[0].grid()
axarr[0].legend()

axarr[1].set_title('Freq')
axarr[1].plot(Time, Freq, "r-", label="Freq")
axarr[1].plot(Time, FreqAdj, "g-", label="FreqAdj")
axarr[1].grid()
axarr[1].legend()

axarr[2].set_title('Position')
axarr[2].plot(Time, RefPosX, label="Reference PosX")
axarr[2].plot(Time, Heave, "r-", label="Heave")
axarr[2].plot(Time, HeaveAlt, "g-", label="HeaveAlt")
axarr[2].grid()
axarr[2].legend()

plt.show()
