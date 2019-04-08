"""LangevinLaser dynamics class."""

from ase.md.md import MolecularDynamics
from ase.parallel import world
from ase import units

from scipy import interpolate
import numpy as np


class LangevinLaser(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.

    Usage: Langevin(atoms, dt, temperature, friction)

    atoms
        The list of atoms.

    dt
        The time step.

    temperature
        String, name of file which contains time versus electronic temperatures versus phonon temperatures.

    density
        String, name of file which contains distance from lattice atom versus electron density.

    lattice_friction
        Optional, constant friction for lattice atoms

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    rng
        Random number generator, by default numpy.random.  Must have a
        standard_normal method matching the signature of
        numpy.random.standard_normal.

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

        """ Read from density.txt into 1D arrays, generate interpolated function and get cutoff"""

        density_file = np.loadtxt(density)

        r_array = density_file[:, 0]
        n_array = density_file[:, 1]

        self.interpolated_density = interpolate.interp1d(r_array, n_array, kind='cubic')

        self.cutoff = r_array[-1]

        """Make empty arrays for the coefficients """  # find journal reference

        self.natoms = atoms.get_number_of_atoms()
        self.c1 = self.c2 = self.c3 = self.c4 = self.c5 = np.zeros(self.natoms)

        """ Get indices of adsorbate and surface atoms """

        self.symbols = np.asarray(atoms.get_chemical_symbols(), dtype=str)

        self.C_indices = np.flatnonzero(self.symbols == 'C')
        self.O_indices = np.flatnonzero(self.symbols == 'O')
        self.lattice_indices = np.flatnonzero(self.symbols == 'Pd')
        self.adsorbate_indices = np.concatenate((self.C_indices, self.O_indices))

        """ Make empty arrays for atom temperatures and atom frictions. """

        self.T = np.zeros(self.natoms)
        self.friction = np.zeros(self.natoms)

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

    def friction_c(self, rs_c):
        fric_c = np.where(rs_c > 0., 22.654*rs_c**(2.004)*np.exp(-3.134*rs_c)+2.497*rs_c**(-2.061)*np.exp(0.0793*rs_c), 0.)
        return fric_c

    def friction_o(self, rs_o):
        fric_o = np.where(rs_o > 0., 1.36513 * rs_o ** (-1.8284) * np.exp(-0.0820301 * rs_o) + 50.342 * rs_o ** (0.490785) * np.exp(-2.70429 * rs_o), 0.)
        return fric_o

    def calculate_friction(self):

        distances = self.atoms.get_distances_list(self.adsorbate_indices, self.lattice_indices, mic=True)
        distances[distances >= self.cutoff] = np.nan
        density = np.nansum(self.interpolated_density(distances), axis=1)

        rs = np.where(density > 0., (3. / (4. * np.pi * density) ) ** (1./3.), 0.)

        np.put(self.friction, self.C_indices, self.friction_c(rs[self.C_indices]))
        np.put(self.friction, self.O_indices, self.friction_o(rs[self.O_indices]))

    def calculate_temperature(self):

        current_time = self.get_time()

        np.put(self.T, self.adsorbate_indices, self.interpolated_el_temp(current_time))
        np.put(self.T, self.lattice_indices, self.interpolated_ph_temp(current_time))

    def updatevars(self):

        dt = self.dt

        """Get electronic and phonon temperatures at current time"""

        self.calculate_temperature()

        """Calculate adsorbate friction at current time"""

        self.calculate_friction()

        sigma = np.sqrt(2. * self.T * self.friction / self.mass)
        self.c1 = (dt / 2. - dt * dt * self.friction / 8.)
        self.c2 = (dt * self.friction / 2 - dt * dt * self.friction * self.friction / 8.)
        self.c3 = (np.sqrt(dt) * sigma / 2. - dt ** 1.5 * self.friction * sigma / 8.)
        self.c5 = (dt ** 1.5 * sigma / (2 * np.sqrt(3.)))
        self.c4 = (self.friction / 2. * self.c5)

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
