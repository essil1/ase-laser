import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('taverage19dyns.csv')

time = data[:,0]
T_ads = data[:,1]
T_latt = data[:,2]
T_el = data[:,3]
T_ph = data[:,4]

plt.plot(time, T_ads)
plt.plot(time, T_latt)
plt.plot(time, T_el)
plt.plot(time, T_ph)

plt.xlabel('$t[fs]$', fontsize=16)
plt.ylabel('$T[K]$', fontsize=16)
plt.legend(['$T_{ads}$', '$T_{latt}$', '$T_{el}$', '$T_{ph}$'], fontsize='large')
plt.show()
