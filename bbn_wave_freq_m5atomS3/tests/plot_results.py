import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt(fname="results.csv", delimiter=",",
                  usecols=(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31),
                  skiprows=0)

Time = Data[:, [0]]
AccX = Data[:, [1]]
RefVelX = Data[:, [2]]
RefPosX = Data[:, [3]]
Heave = Data[:, [4]]
HeaveAlt = Data[:, [5]]
Freq = Data[:, [10]]
FreqAdj = Data[:, [11]]
RefFreq = Data[:, [14]]
HeaveAltErr = Data[:, [15]]

rms = np.sqrt(np.mean(HeaveAltErr[-250 * 60:] ** 2))

fig, axarr = plt.subplots(4, sharex="all")

axarr[0].set_title('Acceleration (m/s^2)')
axarr[0].plot(Time, AccX, label="Input AccX")
axarr[0].grid()
axarr[0].legend()

axarr[1].set_title('Freq (Hz)')
axarr[1].plot(Time, Freq, "r-", label="Freq")
axarr[1].plot(Time, FreqAdj, "g-", label="FreqAdj")
axarr[1].plot(Time, RefFreq, "b-", label="RefFreq")
axarr[1].grid()
axarr[1].legend()

axarr[2].set_title('Heave (m)')
axarr[2].plot(Time, RefPosX, label="Reference PosX")
#axarr[2].plot(Time, Heave, "r-", label="Heave")
axarr[2].plot(Time, HeaveAlt, "g-", label="HeaveAlt")
axarr[2].grid()
axarr[2].legend()

axarr[3].set_title('HeaveAlt Err (m) rms=' + np.str(rms))
axarr[3].plot(Time, HeaveAltErr, "r-", label="HeaveAltErr")
axarr[3].grid()
axarr[3].legend()

#fig.savefig("results.png", dpi=600)
#fig.close()

plt.show()
