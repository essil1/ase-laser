"""LangevinLaser dynamics class."""

from ase.md.md import MolecularDynamics
from ase.parallel import world
from ase import units

from scipy import interpolate
import numpy as np
import sys


class LangevinLaser(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.

    Usage: Langevin(atoms, dt, temperature, friction)

    atoms
        The list of atoms.

    dt
        The time step.

    temperature
        Input as np.loadtxt(temperature_data)

    el_dens_vs_r
        Input as np.loadtxt(density_data)

    lattice_friction
        Optional, constant friction for surface atoms

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    rng
        Random number generator, by default numpy.random.  Must have a
        standard_normal method matching the signature of
        numpy.random.standard_normal.

    The temperature and friction are normally scalars, but in principle one
    quantity per atom could be specified by giving an array.

    RATTLE constraints can be used with these propagators, see:
    E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

    The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
    of the above reference.  That reference also contains another
    propagator in Eq. 21/34; but that propagator is not quasi-symplectic
    and gives a systematic offset in the temperature at large time steps.

    This dynamics accesses the atoms using Cartesian coordinates."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 3

    def __init__(self, atoms, timestep, temperature, density, lattice_friction=0.002, fixcm=True,
                 trajectory=None, logfile=None, loginterval=1,
                 communicator=world, rng=np.random):

        """ Read from temperature T_el_ph.dat into 1D arrays and generate interpolated functions"""

        temperature_file = np.loadtxt(temperature)
        temp_t = temperature_file[:, 0] * 1000 * units.fs
        temp_el = temperature_file[:, 1] * units.kB
        temp_ph = temperature_file[:, 2] * units.kB

        self.interpolated_el_temp = interpolate.interp1d(temp_t, temp_el, kind='cubic')
        self.interpolated_ph_temp = interpolate.interp1d(temp_t, temp_ph, kind='cubic')

        """ Read from density.txt into 1D arrays, generate interpolated function, and get cutoff"""

        density_file = np.loadtxt(density)

        r_array = density_file[:, 0]
        n_array = density_file[:, 1]

        self.interpolated_density = interpolate.interp1d(r_array, n_array, kind='cubic')

        self.cutoff = r_array[-1]

        """ Constant friction for surface atoms """

        self.lattice_friction = lattice_friction

        """Make empty arrays for the coefficients """  # find journal reference

        natoms = atoms.get_number_of_atoms()

        self.c1 = np.zeros([natoms])
        self.c2 = np.zeros([natoms])
        self.c3 = np.zeros([natoms])
        self.c4 = np.zeros([natoms])
        self.c5 = np.zeros([natoms])

        """ Get indices of adsorbate and surface atoms """

        self.C_indices = np.empty(0, dtype=int)
        self.O_indices = np.empty(0, dtype=int)
        self.lattice_indices = np.empty(0, dtype=int)
        self.symbols = atoms.get_chemical_symbols()

        index = 0

        for symbol in self.symbols:
            if symbol == 'C':
                self.C_indices = np.append(self.C_indices, index)

            elif symbol == 'O':
                self.O_indices = np.append(self.O_indices, index)

            elif symbol == 'Pd':
                self.lattice_indices = np.append(self.lattice_indices, index)
            index += 1

        self.adsorbate_indices = np.concatenate((self.C_indices, self.O_indices))

        """ Make empty arrays for atom temperatures and frictions """

        self.T = np.zeros(natoms)
        self.friction = np.zeros(natoms)

        """ Set friction for surface atoms immediately since it is constant during the simulation """

        np.put(self.friction, self.lattice_indices, lattice_friction)

        """ Get atomic masses """
        self.mass = atoms.get_masses()

        self.fixcm = fixcm  # will the center of mass be held fixed?
        self.communicator = communicator
        self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile, loginterval)
        self.updatevars()

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature': self.temp,
                  'friction': self.fr,
                  'fix-cm': self.fixcm})
        return d

    def set_temperature(self, temperature):
        self.temp = temperature
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

        """ Function that calculates friction of an atom for which self.atoms.index == index"""

    def calculate_friction(self, symbol, index):

        friction = 0.

        if symbol == 'Pd':
            friction = self.lattice_friction

        elif symbol == 'C' or symbol == 'O':

            distances = self.atoms.get_distances(index, self.lattice_indices, mic=True)
            electron_density = self.interpolated_density(distances[np.where(distances < self.cutoff)]).sum()

            if electron_density > 0.:

                rs = (3. / (4. * np.pi * electron_density)) ** (1./3.)

                if symbol == 'C':
                    friction = 22.654 * rs ** (2.004) * np.exp(-3.134 * rs) + 2.497 * rs ** (-2.061) * np.exp(0.0793 * rs)

                elif symbol == 'O':
                    friction = 1.36513 * rs ** (-1.8284) * np.exp(-0.0820301 * rs) + 50.342 * rs ** (0.490785) * np.exp(-2.70429 * rs)

            else:
                friction = 0.

        else:
            sys.exit('Error while calculating friction for atom no. ' + str(index) + '. Invalid chemical symbol ' + str(symbol) + '.')

        return friction

    def calculate_friction2(self):

        for index in self.adsorbate_indices:

            symbol = self.symbols[index]

            distances = self.atoms.get_distances(index, self.lattice_indices, mic=True)
            electron_density = self.interpolated_density(distances[np.where(distances < self.cutoff)]).sum()

            if electron_density > 0.:

                rs = (3. / (4. * np.pi * electron_density)) ** (1./3.)

                if symbol == 'C':
                    self.friction[index] = 22.654 * rs ** (2.004) * np.exp(-3.134 * rs) + 2.497 * rs ** (-2.061) * np.exp(0.0793 * rs)

                elif symbol == 'O':
                    self.friction[index] = 1.36513 * rs ** (-1.8284) * np.exp(-0.0820301 * rs) + 50.342 * rs ** (0.490785) * np.exp(-2.70429 * rs)

                else:
                    sys.exit('Error while calculating friction for atom no. ' + str(index) + '. Invalid chemical symbol ' + str(symbol) + '.')

            else:
                self.friction[index] = 0.

    def updatevars(self):

        dt = self.dt
        masses = self.mass

        """Get electronic and phonon temperatures at current time"""

        current_time = self.get_time()

        T_el = self.interpolated_el_temp(current_time)
        T_ph = self.interpolated_ph_temp(current_time)

        np.put(self.T, self.adsorbate_indices, T_el)
        np.put(self.T, self.lattice_indices, T_ph)

        """Calculate adsorbate friction at current time"""

        self.calculate_friction2()
        fr = self.friction

        sigma = np.sqrt(2. * self.T * fr / masses)
        self.c1 = (dt / 2. - dt * dt * fr / 8.)
        self.c2 = (dt * fr / 2 - dt * dt * fr * fr / 8.)
        self.c3 = (np.sqrt(dt) * sigma / 2. - dt ** 1.5 * fr * sigma / 8.)
        self.c5 = (dt ** 1.5 * sigma / (2 * np.sqrt(3.)))
        self.c4 = (fr / 2. * self.c5)

        # Works in parallel Asap, #GLOBAL number of atoms:
        self.natoms = self.atoms.get_number_of_atoms()

    def step(self, f):

        self.updatevars()
        print(self.get_time()/units.fs/1000)

        atoms = self.atoms
        natoms = len(atoms)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(natoms, 3))
        self.eta = self.rng.standard_normal(size=(natoms, 3))

        if self.communicator is not None:
            self.communicator.broadcast(self.xi, 0)
            self.communicator.broadcast(self.eta, 0)

        # First halfstep in the velocity.
        self.v += (self.c1[:, None] * f / self.masses - self.c2[:, None] * self.v + self.xi * self.c3[:, None] - self.c4[:, None] * self.eta)

        # Full step in positions
        x = atoms.get_positions()
        if self.fixcm:
            old_cm = atoms.get_center_of_mass()
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.c5[:, None] * self.eta)
        if self.fixcm:
            new_cm = atoms.get_center_of_mass()
            d = old_cm - new_cm
            # atoms.translate(d)  # Does not respect constraints
            atoms.set_positions(atoms.get_positions() + d)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x -
                  self.c5[:, None] * self.eta) / self.dt
        f = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (self.c1[:, None] * f / self.masses - self.c2[:, None] * self.v +
                   self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)

        if self.fixcm:  # subtract center of mass vel
            v_cm = self._get_com_velocity()
            self.v -= v_cm

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return f

    def _get_com_velocity(self):
        """Return the center of mass velocity.

        Internal use only.  This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.flatten(), self.v) / self.masses.sum()
