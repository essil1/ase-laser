import numpy as np
import matplotlib.pyplot as plt
import sys

poscar = str(sys.argv[1])
data = np.loadtxt('temperatures_' + poscar)

time = data[:,0]
T_ads = data[:,1]/2.
T_latt = data[:,2]
T_el = data[:,3]
T_ph = data[:,4]
E_ads = data[:,5]
E_latt = data[:,6]

plt.plot(time, T_ads)
plt.plot(time, T_latt)
plt.plot(time, T_el)
plt.plot(time, T_ph)

plt.xlabel('$t[fs]$')
plt.ylabel('$T[K]$')
plt.legend(['$T_{ads}$', '$T_{latt}$', '$T_{el}$', '$T_{ph}$'])
plt.show()
