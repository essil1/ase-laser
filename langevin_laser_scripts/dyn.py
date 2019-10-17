from __future__ import print_function

from ase.md.langevin_laser import LangevinLaser
from ase.io.trajectory import Trajectory
from ase.io import read
from ase import units
from ase.calculators.emt import EMT  # Way too slow with ase.EMT !

#To use amp calculators remove comments
from amp import Amp
from amp.descriptor import *

import numpy as np
import sys

"""End import"""

i = sys.argv[1]
#Create files in which temperatures and frictions will be written
f1 = open('temperatures_' + str(i), 'w+')
f2 = open('frictions_' + str(i), 'w+')
f1.close()
f2.close()

T = 'T_el_ph.dat'

#Set friction for surface atoms
lattice_friction = 0.2

#Read a configuration
#conf = Trajectory('t4x2.traj', 'r')
#atoms = conf[0]
atoms = read('POSCAR-' + str(i), format='vasp')

#Set initial positions and velocities
#positions = np.loadtxt('pos1')
velocities_fs = np.loadtxt('POSCAR-' + str(i), skiprows=54, usecols=(0,1,2), max_rows=44)
velocities = velocities_fs/units.fs
#atoms.set_positions(positions)
atoms.set_velocities(velocities)

#To use AMP calculator remove comment and change EMT() to calc
calc = Amp.load('735.amp', label='label' + str(i))
atoms.set_calculator(calc)

# We want to run MD with the Langevin algorithm
#time step of 1 fs, the temperatures T, electron densities file, constant lattice friction.
dyn = LangevinLaser(atoms, units.fs, T, lattice_friction, str(i))

def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every interval-th time step.
traj = Trajectory('moldyn4x2_' + str(i) + '.traj', 'w', atoms)
dyn.attach(traj.write, interval=1)

# Now run the dynamics
printenergy()
dyn.run(3500)
