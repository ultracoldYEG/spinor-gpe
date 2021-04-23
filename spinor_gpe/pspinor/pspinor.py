"""Base class for pseudospinor GPE propagation."""

# Classes & Modules:
#  - Spinors
#  - PropResult
#  - TensorPropagator
#  - tensor_tools
#  - constants

import os
import shutil

import numpy as np
from scipy.ndimage import fourier_shift
# from matplotlib import pyplot as plt
# import torch

from definitions import ROOT_DIR
import constants as const
from pspinor import tensor_tools as ttools
from pspinor import plotting_tools as ptools
from pspinor import tensor_propagator as tprop


# pylint: disable=too-many-public-methods
class PSpinor:
    """A GPU-compatible simulator of the pseudospin-1/2 GPE.

    Contains the functionality to run a real- or imaginary-time propataion of
    the pseudospin-1/2 Gross-Pitaevskii equation. Contains methods to generate
    the required energy and spatial grids. Also has methods to generate the
    grids for the momentum-(in)dependent coupling between spin components,
    corresponding to an (RF) Raman coupling interaction.

    The dominant length scale is in terms of the harmonic oscillator length
    along the x-direction `a_x`. The dominant energy scale is the harmonic
    trapping energy along the x-direction [hbar * `omeg['x']`].

    # TODO: Reduce the number of public attributes in this class

    Attributes
    ----------
    paths : :obj:`str`

    atom_num : :obj:`int`
    pop_frac : :obj:`tuple`
    omeg : :obj:`dict`
    g_sc : :obj:`dict`
    a_x :
    rad_tf :
    time_scale :

    mesh_points :
    r_sizes :
    delta_r :
    k_sizes :
    delta_k :
    x_lin :
    y_lin :
    kx_lin :
    ky_lin :
    x_mesh :
    y_mesh :
    kx_mesh :
    ky_mesh :
    dv_r :
    dv_k :

    pot_eng :
    kin_eng :
    psi :
    psik :
    is_coupling :
    kin_eng_spin :
    pot_eng_spin :
    kL_recoil :
    EL_recoil :
    mom_shift :
    prop :
    n_steps :
    rand_seed :

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, path, omeg=None, g_sc=None, mesh_points=(256, 256),
                 r_sizes=(16, 16), atom_num=1e4, pop_frac=(0.5, 0.5),
                 **kwargs):
        # pylint: disable=too-many-arguments
        """Instantiate a Spinor object.

        Generates the parameters and
        basic energy grids required for propagation.

        Parameters
        ----------
        path : :obj:`str`
            The path to the subdirectory /data/`path` where the data and
            propagation results will be saved. This path may take the form
            "project_name/trial_name".
        omeg : :obj:`dict`, optional
            Trapping frequencies, {x, y, z} [rad/s].
        g_sc : :obj:`dict`, optional
            Relative coupling strengths for scattering interactions,
            {uu, dd, ud}. Intercomponent interaction are assummed to be
            symmetric, i.e. ud == du.
        mesh_points : :obj:`iterable` of :obj:`int`, optional
            The number of grid points along the x- and y-axes, respectively.
        r_sizes : :obj:`iterable` of :obj:`int`, optional
            The half size of the real space grid along the x- and y-axes,
            respectively, in units of [a_x].
        atom_num : :obj:`int`, optional
            Total atom number.
        pop_frac : :obj:`array_like` of :obj:`float`, optional
            Starting population fraction in each spin component.

        Other Parameters
        ----------------
        phase_factor : :obj:`complex`, optional
            Unit complex number; initial relative phase factor between the two
            spin components.
        overwrite : :obj:`bool`, optional
            By default, the simulation will halt and raise an error if it
            attempts to overwrite a directory `path` already containing data.
            `overwrite` gives the user the option to overwrite the data with
            every new instance.


        """
        phase_factor = kwargs.get('phase_factor', 1)
        overwrite = kwargs.get('overwrite', False)
        # pylint: disable=too-many-arguments
        self.setup_data_path(path, overwrite)

        self.atom_num = atom_num

        assert sum(pop_frac) == 1.0, "Total population must equal 1."
        self.pop_frac = pop_frac  #: Spins' initial population fraction

        if omeg is None:
            omeg0 = 2*np.pi*50
            #: dict: Angular trapping frequencies
            self.omeg = {'x': omeg0, 'y': omeg0, 'z': 40 * omeg0}
            # ??? Maybe make self.omeg (& g_sc) object @properties with methods
            # for dynamic updating.
        else:
            omeg_names = {'x', 'y', 'z'}
            assert omeg_names == omeg.keys(), ("Keys for `omeg` must have "
                                               f"the form: {omeg_names}.")
            self.omeg = omeg

        if g_sc is None:
            #: dict: Relative scattering interaction strengths
            self.g_sc = {'uu': 1.0, 'dd': 0.995, 'ud': 0.995}
        else:
            g_names = {'uu', 'dd', 'ud'}
            assert g_names == g_sc.keys(), ("Keys for `g_sc` must have "
                                            f"the form: {g_names}.")
            self.g_sc = g_sc

        self.compute_tf_params()
        self.compute_spatial_grids(mesh_points, r_sizes)
        self.compute_energy_grids()
        self.compute_tf_psi(phase_factor)
        self.no_coupling_setup()

        self.prop = None
        self.n_steps = None
        self.rand_seed = None

    def setup_data_path(self, path, overwrite):
        """Create new data directory to store simulation data & results.

        Parameters
        ----------
        path : :obj:`str`
            The name of the directory to save the simulation. If `path` 
            does not represent an absolute path, then the data is stored
            in spinor-gpe/data/`path`.
        overwrite : :obj:`bool`
            Gives the option to overwrite existing data sub-directories

        """
        # TODO: Make it so the user can use any arbitrary data path.
        #: Path to the directory containing all the simulation data & results
        if not os.path.isabs(path):
            data_path = ROOT_DIR + '/data/' + path + '/'
        else:
            data_path = path
        #: Path to the subdirectory containing the raw trial data
        trial_data_path = data_path + 'trial_data/'
        #: Path to the subdirectory containing the trial code.
        code_data_path = data_path + 'code/'
        if os.path.isdir(data_path):
            if not overwrite:
                raise FileExistsError(
                    f"The directory {data_path} already exists. "
                    "To overwrite this directory, supply the parameter "
                    "`overwrite=True`.")

            shutil.rmtree(data_path)  # Deletes the data directory
        # Create the directories and sub-directories
        os.makedirs(data_path)
        os.makedirs(code_data_path)
        os.makedirs(trial_data_path)

        self.paths = {'data': data_path, 'trial': trial_data_path,
                      'code': code_data_path}

    def compute_tf_psi(self, phase_factor):
        """Compute the intial pseudospinor wavefunction `psi` and FFT `psik`.

        `psi` is a list of 2D numpy arrays.

        """
        assert abs(phase_factor) == 1.0, ("Relative phase factor must have "
                                          "unit magnitude.")
        g_bare = [self.g_sc['uu'], self.g_sc['dd']]
        profile = np.real(np.sqrt(self.atom_num * (self.chem_pot
                                                   - self.pot_eng + 0.j)))
        #: Initial Thomas-Fermi wavefunction for the two spin components
        self.psi = [profile * np.real(np.sqrt(pop / g + 0.j)) for pop, g
                    in zip(self.pop_frac, g_bare)]
        self.psi[1] *= phase_factor

        self.psi, _ = ttools.norm(self.psi, self.dv_r, self.atom_num)
        self.psik = ttools.fft_2d(self.psi, self.delta_r)

        # Saves the real- and k-space versions of the Thomas-Fermi wavefunction
        np.savez(self.paths['trial'] + 'tf-wf', psi=self.psi, psik=self.psik)

    def compute_tf_params(self, species='Rb87'):
        """Compute parameters and scales for the Thomas-Fermi solution."""
        #: Relative size of y-axis trapping frequency relative to x-axis.
        y_trap = self.omeg['y'] / self.omeg['x']
        #: Relative size of z-axis trapping frequency relative to x-axis.
        z_trap = self.omeg['z'] / self.omeg['x']
        #: float: Harmonic oscillator length scale [m].
        self.a_x = np.sqrt(const.hbar / (const.Rb87['m'] * self.omeg['x']))

        #: Dimensionless scattering length, [a_x]
        if species == 'Rb87':
            a_sc = const.Rb87['a_sc'] / self.a_x
        else:
            a_sc = 1
        #: Chemical potential for an asymmetric harmonic BEC, [hbar*omeg_x].
        self.chem_pot = ((4 * self.atom_num * a_sc * y_trap
                          * np.sqrt(z_trap / (2 * np.pi)))**(1/2))

        g_scale = np.sqrt(8 * z_trap * np.pi) * a_sc
        self.g_sc.update({k: g_scale * self.g_sc[k] for k in self.g_sc.keys()})
        self.rad_tf = np.sqrt(2 * self.chem_pot)  #: Thomas-Fermi radius [a_x].

        self.time_scale = 1 / self.omeg['x']  #: Time scale [1/omeg_x]

    def compute_spatial_grids(self, mesh_points, r_sizes):
        """Compute the real and momentum space grids.

        Parameters
        ----------
        mesh_points : :obj:`list` of :obj:`int`
            The number of grid points along the x- and y-axes, respectively.
        r_sizes : :obj:`list` of :obj:`int`, optional
            The half size of the grid along the real x- and y-axes,
            respectively,in units of [a_x].

        """
        assert all(point % 2 == 0 for point in mesh_points), (
            f"Number of mesh points {mesh_points} should be powers of 2.")
        self.mesh_points = np.array(mesh_points)
        self.r_sizes = np.array(r_sizes)

        #: Spacing between real-space mesh points [a_x]
        self.delta_r = 2 * self.r_sizes / self.mesh_points
        #: Half size of the grid along the kx- and ky- axes [1/a_x]
        self.k_sizes = np.pi / self.delta_r
        #: Spacing between momentum-space mesh points [1/a_x]
        self.delta_k = np.pi / self.r_sizes

        #: Linear arrays for real- [a_x] and k-space [1/a_x], x- and y-axes
        self.x_lin = self._compute_lin(self.r_sizes, self.mesh_points, axis=0)
        self.y_lin = self._compute_lin(self.r_sizes, self.mesh_points, axis=1)
        self.kx_lin = self._compute_lin(self.r_sizes, self.mesh_points, axis=0)
        self.ky_lin = self._compute_lin(self.r_sizes, self.mesh_points, axis=1)

        #: 2D meshes for computing the energy grids [a_x] and [1/a_x]
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_lin, self.y_lin)
        self.kx_mesh, self.ky_mesh = np.meshgrid(self.x_lin, self.y_lin)

        # ??? Add functionality for Tukey filter window?

        #: Real-space volume element used for normalization [a_x^2]
        self.dv_r = np.prod(self.delta_r)
        #: k-space volume element used for normalization [1/a_x^2]
        self.dv_k = np.prod(self.delta_k)

    @classmethod
    def _compute_lin(cls, sizes, points, axis=0):
        """Compute linear 1D arrays of real or momentum space mesh points.

        Parameters
        ----------
        sizes : array
            The half sizes of the mesh
        points : array
            The number of points in the mesh
        axis : :obj:`int`, optional
            The axis along which to generate: 0 -> 'x'; 1 -> 'y'

        """
        return np.linspace(-sizes[axis], sizes[axis], num=points[axis],
                           endpoint=False)

    def compute_energy_grids(self):
        """Compute basic potential and kinetic energy grids.

        Assumes that the BEC is in a harmonic trap. This harmonic potential
        determines the initial 'Thomas-Fermi' density profile of the BEC.
        `pot_eng` can be modified prior to progation to have any arbitrary
        potential energy landscape.

        Assumes that the BEC has a simple free-particle kinetic energy
        dispersion. If using a momentum-dependent spin coupling, this grid
        will be modified later.

        """
        y_trap = self.omeg['y'] / self.omeg['x']
        #: Potential energy grid [hbar*omeg_x]
        self.pot_eng = (self.x_mesh**2 + (y_trap * self.y_mesh)**2) / 2
        #: Kinetic energy grid [hbar*omeg_x]
        self.kin_eng = (self.kx_mesh**2 + self.ky_mesh**2) / 2

    def _calc_atoms(self, psi=None, space='r'):
        """Given a list of wavefunctions, calculates the total atom number.

        May need to consider the difference between numpy and tensor versions.

        Parameters
        ----------
        psi : :obj:`list` of Numpy :obj:`array`, optional.
            The pseudospinor wavefunction.
        space : {'r', 'k'}, optional

        Returns
        -------
        atom_num : :obj:`float`
        """
        if psi is None:
            psi = self.psi

        if space == 'r':
            vol_elem = self.dv_r
        elif space == 'k':
            vol_elem = self.dv_k

        atom_num = ttools.calc_atoms(psi, vol_elem)
        return atom_num

    def no_coupling_setup(self):
        """Calculate the kinetic & potential energy grids for no coupling."""
        self.is_coupling = False
        self.kin_eng_spin = [self.kin_eng] * 2
        self.pot_eng_spin = [self.pot_eng] * 2
        # pylint: disable=invalid-name
        self.kL_recoil = None
        self.EL_recoil = None
        self.mom_shift = None
        # self.coupling = None
        # self.detuning = None

    def coupling_setup(self, wavel=790.1e-9, scale=1, mom_shift=False):
        """Calculate parameters for the momentum-(in)dependent coupling.

        Parameters
        ----------
        wavel : :obj:`float`
            Wavelength of Raman coupling (in [m])
        scale : :obj:`float`
            Relative scale of recoil momentum
        mom_shift : :obj:`bool`
            Option for a momentum-(in)dependent coupling.
        """
        # pass wavelength, relative scaling of k_L, momentum-(in)dependency
        # pylint: disable=attribute-defined-outside-init
        #: Designator attribute for the presence of coupling
        self.is_coupling = True
        #: Recoil momentum of the coupling interaction [a_x].
        # pylint: disable=invalid-name
        self.kL_recoil = scale * (np.sqrt(2) * np.pi / wavel * self.a_x)
        #: Recoil energy of the coupling interaction [hbar*omeg_x]
        # pylint: disable=invalid-name
        self.EL_recoil = self.kL_recoil**2 / 2
        #: Momentum shift option
        self.mom_shift = mom_shift
        if self.mom_shift:
            shift = self.kx_mesh * self.kL_recoil
        else:
            shift = 0

        self.kin_eng_spin = [self.kin_eng + shift, self.kin_eng - shift]
        self.kin_eng_spin = [k - np.min(k) for k in self.kin_eng_spin]

    def shift_momentum(self, psik=None, kshift_val=1, frac=(0.5, 0.5)):
        """Shifts the momentum components pf `psi` by +/- kL_recoil."""
        assert self.is_coupling, ("The `is_coupling` option is "
                                  f"{self.is_coupling}. Initialize coupling "
                                  "with `coupling_setup()`.")
        if psik is None:
            psik = self.psik

        shift = kshift_val * self.kL_recoil / self.delta_k[0]
        input_ = ttools.fft_2d(psik, self.delta_r)
        result = [np.zeros_like(pk) for pk in psik]
        for i in range(len(psik)):
            positive = fourier_shift(input_[i], shift=[0, shift], axis=1)
            negative = fourier_shift(input_[i], shift=[0, -shift], axis=1)
            result[i] = frac[0]*positive + frac[1]*negative
            frac = np.flip(frac)
        psik_shift = ttools.ifft_2d(result, self.delta_r)

        return psik_shift

    @property
    def coupling(self):
        """Get the `coupling` attribute."""
        return self._coupling

    @coupling.setter
    def coupling(self, array):
        """Set the `coupling` attribute."""
        if not self.is_coupling:
            raise Exception(f"The `is_coupling` option is {self.is_coupling}. \
                            Initialize coupling with `coupling_setup()`.")
        self._coupling = array

    @property
    def detuning(self):
        """Get the `detuning` attribute."""
        return self._detuning

    @detuning.setter
    def detuning(self, array):
        """Set the `detuning` attribute."""
        if not self.is_coupling:
            raise Exception(f"The `is_coupling` option is {self.is_coupling}. \
                            Initialize coupling with `coupling_setup()`.")
        self._detuning = array
        self.pot_eng_spin = [self.pot_eng + self._detuning / 2,
                             self.pot_eng - self._detuning / 2]

    def coupling_grad(self, slope, offset, axis=1):
        """Generate a linear gradient of the interspin coupling strength.

        Convenience function for generating linear gradients of the coupling.
        `coupling` can also be set to any arbitrary numpy array directly:

        >>> ps = PSpinor()
        >>> ps.coupling_setup()
        >>> ps.coupling = np.exp(-ps.x_mesh**2 / 2)  # Gaussian function

        .. note:: When working with Raman recoil units [E_L], they will first
        need to be converted to [hbar*omeg_x] units before.

        Parameters
        ----------
        slope : :obj:`float`
            The slope of the coupling gradient, in [hbar*omeg_x/a_x].
        offset : :obj:`float`
            The origin offset of the coupling gradient, in [hbar*omeg_x].
        axis : :obj:`int`, optional
            The axis along which the coupling gradient runs.

        """
        if axis == 0:
            mesh = self.x_mesh
        elif axis == 1:
            mesh = self.y_mesh

        self.coupling = mesh * slope + offset

    def coupling_uniform(self, value):
        """Generate a uniform interspin coupling strength.

        Convenience function for generating unirom gradients of the coupling.
        `coupling` can also be set to any arbitrary numpy array directly.

        Parameters
        ----------
        value : :obj:`float`
            The value of the coupling, in [hbar*omega_x].

        See Also
        --------
        coupling_grad : Coupling gradient

        """
        assert value >= 0, f"Cannot have a negative coupling value: {value}."
        self.coupling = np.ones_like(self.x_mesh) * value

    def detuning_grad(self, slope, offset, axis=1):
        """Generate a linear gradient of the interspin coupling strength.

        Convenience function for generating linear gradients of the coupling.
        `detuning` can also be set to any arbitrary numpy array directly:

        >>> ps = PSpinor()
        >>> ps.coupling_setup()
        >>> ps.detuning = np.sin(2 * np.pi * ps.x_mesh)  # Sin function

        **Note**: when working with Raman recoil units [E_L], they will first
        need to be converted to [hbar*omeg_x] units.

        Parameters
        ----------
        slope : :obj:`float`
            The slope of the detuning gradient, in [hbar*omeg_x/a_x].
        offset : :obj:`float`
            The origin offset of the detuning gradient, in [hbar*omeg_x].
        axis : :obj:`int`, optional
            The axis along which the detuning gradient runs.

        """
        if axis == 0:
            mesh = self.x_mesh
        elif axis == 1:
            mesh = self.y_mesh

        self.detuning = mesh * slope + offset

    def detuning_uniform(self, value):
        """Generate a uniform coupling detuning.

        Convenience function for generating unirom gradients of the coupling.
        `detuning` can also be set to any arbitrary numpy array directly.

        Parameters
        ----------
        value : :obj:`float`
            The value of the coupling, in [hbar*omega_x].

        See Also
        --------
        PSpinor.detuning_grad : Detuning gradient

        """
        self.detuning = np.ones_like(self.x_mesh) * value

    def seed_regular_vortices(self):
        """Seed regularly-arranged vortices into the wavefunction.

        These seed-vortex functions might be moved to the ttools module.
        """

    def seed_random_vortices(self):
        """Seed randomly-arranged vortices into the wavefunction."""

    def plot_rdens(self, psi=None, spin=None, cmap='viridis', scale=1.):
        """Plot the real-space density of the wavefunction.

        Plots either the up (`spin=0`), down (`spin=1`), or both (`spin=None`)
        spin components. If no `psi` is supplied, then it uses the
        object attribute `self.psi`.

        See Also
        --------
        plotting_tools.plot_dens : Density plots.

        """
        if psi is None:
            psi = self.psi

        extent = np.ravel(np.vstack((-self.r_sizes, self.r_sizes)).T) / scale
        ptools.plot_dens(psi, spin, cmap, scale, extent=extent)

    def plot_kdens(self, psik=None, spin=None, cmap='viridis', scale=1.):
        """Plot the k-space density of the wavefunction.

        Plots either the up (`spin=0`), down (`spin=1`), or both (`spin=None`)
        spin components. If no `psik` is supplied, then it uses the
        object attribute `self.psik`.

        See Also
        --------
        plotting_tools.plot_dens : Density plots.

        """
        if psik is None:
            psik = self.psik

        extent = np.ravel(np.vstack((-self.k_sizes, self.k_sizes)).T) / scale
        ptools.plot_dens(psik, spin, cmap, scale, extent)

    def plot_rphase(self, psi=None, spin=None, cmap='twilight_shifted',
                    scale=1.):
        """Plot the real-space phase of the wavefunction.

        Plots either the up (`spin=0`), down (`spin=1`), or both (`spin=None`)
        spin components. If no `psi` is supplied, then it uses the
        object attribute `self.psi`.

        See Also
        --------
        plotting_tools.plot_phase : Phase plots.

        """
        if psi is None:
            psi = self.psi

        extent = np.ravel(np.vstack((-self.r_sizes, self.r_sizes)).T) / scale
        ptools.plot_phase(psi, spin, cmap, scale, extent)

    # pylint: disable=too-many-arguments
    def imaginary(self, t_step, n_steps=1000, device='cpu',
                  is_sampling=False, n_samples=0,
                  is_annealing=False, n_anneals=0):
        """Perform imaginary-time propagation."""
        # Pass PSpinor object instance `self` as the first parameter of
        # TensorPropagator.__init__.
        self.prop = tprop.TensorPropagator(self, t_step, n_steps, device,
                                           is_sampling=is_sampling,
                                           n_samples=n_samples,
                                           is_annealing=is_annealing,
                                           n_anneals=n_anneals)
        return PropResult()

    def real(self, t_step, n_steps=1000, device='cpu', is_sampling=False,
             n_samples=0):
        """Perform real-time propagation."""
        self.prop = tprop.TensorPropagator(self, t_step, n_steps, device,
                                           is_sampling=is_sampling,
                                           n_samples=n_samples)
        return PropResult()


class PropResult:
    """Results of propagation, along with plotting and analysis tools."""

    def __init__(self):
        pass

    def plot_spins(self):
        """Plot the densities (real & k) and phases of spin components."""

    def plot_total(self):
        """Plot the total real-space density and phase of the wavefunction."""

    def plot_eng(self):
        """Plot the sampled energy expectation values."""

    def plot_pops(self):
        """Plot the spin populations as a function of propagation time."""

    def analyze_vortex(self):
        """Compute the total vorticity in each spin component."""

    def make_movie(self):
        """Generate a movie of the wavefunctions' densities and phases."""


# ----- DOCUMENTATION -----
#  - `sphinx`
#  - `sphinx.ext.autodoc`; this website was helpful:
#  - `sphinx.ext.napoleon` --> for using NumPy documentation style;
#    alternatively, use `numpydoc`; here is their style guide:
#    https://numpydoc.readthedocs.io/en/latest/format.html
#  - ReadTheDocs, for hosting the documentation once it's good
