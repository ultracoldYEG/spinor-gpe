"""Base class for pseudospinor GPE propagation."""

import os
import shutil
import warnings

import numpy as np
from scipy.ndimage import fourier_shift
# from matplotlib import pyplot as plt
# import torch

from definitions import ROOT_DIR
# pylint: disable=import-error
import spinor_gpe.constants as const
from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor import plotting_tools as ptools
from spinor_gpe.pspinor import tensor_propagator as tprop


# pylint: disable=too-many-public-methods
class PSpinor:
    r"""A GPU-compatible simulator of the pseudospin-1/2 GPE.

    Contains the functionality to run a real- or imaginary-time propataion of
    the pseudospin-1/2 Gross-Pitaevskii equation. Contains methods to generate
    the required energy and spatial grids. Also has methods to generate the
    grids for the momentum-(in)dependent coupling between spin components,
    corresponding to an (RF) Raman coupling interaction.

    The dominant length scale is in terms of the harmonic oscillator length
    along the x-direction `a_x`. The dominant energy scale is the harmonic
    trapping energy along the x-direction [hbar * `omeg['x']`].

    Attributes
    ----------
    paths : :obj:`dict`
        Essential paths and directory names for a given trial.

        - 'folder' : The name of the topmost trial directory.
        - 'data' : Path to the simulation data & results.
        - 'code' : Path to the subdirectory containing the trial code.
        - 'trial' : Path to the subdirectory containing raw trial data.

    atom_num : :obj:`float`
        Total atom number.
    space : :obj:`dict`
        Spatial arrays, meshes, spacings, volume elements, and sizes:

        +--------------------+----------+----------+-----------+-----------+
        | **KEYS**                                                         |
        +====================+==========+==========+===========+===========+
        | Arrays:            |    'x'   |   'y'    |    'kx'   |    'ky'   |
        +--------------------+----------+----------+-----------+-----------+
        | Meshes:            | 'x_mesh' | 'y_mesh' | 'kx_mesh' | 'ky_mesh' |
        +--------------------+----------+----------+-----------+-----------+
        | Spacings:          |        'dr'         |          'dk'         |
        +--------------------+----------+----------+-----------+-----------+
        | Vol. elem.:        |       'dv_r'        |         'dv_k'        |
        +--------------------+----------+----------+-----------+-----------+
        | Sizes:             |      'r_sizes'      |       'k_sizes'       |
        +--------------------+----------+----------+-----------+-----------+
        | Other:             |                'mesh_points'                |
        +--------------------+----------+----------+-----------+-----------+

    pop_frac : :obj:`iterable`
        The initial population fraction in each spin component.
    omeg : :obj:`dict`
        Angular trapping frequencies, {'x', 'y', 'z'}. omeg['x'] multiplied
        by \\hbar is the characteristic energy scale.
    g_sc : :obj:`dict`
        Relative scattering interaction strengths, {'uu', 'dd', 'ud'}.
    a_x : :obj:`float`
        Harmonic oscillator length along the x-axis; this is the
        characteristic length scale.
    chem_pot : :obj:`float`
        Chemical potential, [\\hbar * omeg['x']].
    rad_tf : :obj:`float`
        Thomas-Fermi radius along the x-axis, [a_x].
    time_scale : :obj:`float`
        Inverse x-axis trapping frequency - the characteristic time scale,
        [1 / omeg['x']].
    pot_eng_spin : :obj:`list` of :obj:`array`
        A :obj:`list` of 2D potential energy grids for each spin component,
        [\\hbar * omeg['x']].
    kin_eng_spin : :obj:`list` of :obj:`array`
        A :obj:`list` of 2D kinetic energy grids for each spin component,
        [\\hbar * omeg['x']].
    psi : :obj:`list` of :obj:`array`
        A :obj:`list` of 2D real-space spin component wavefunctions; generally
        complex.
    psik : :obj:`list` of :obj:`array`
        A :obj:`list` of 2D momentum-space spin component wavefunctions;
        generally complex.
    is_coupling : :obj:`bool`
        Signals the presence of direct coupling between spin components.
    kL_recoil : :obj:`float`
        The value of the single-photon recoil momentum, [1 / a_x].
    EL_recoil : :obj:`float`
        The energy scale corresponding to the single-photon recoil momentum,
        [1 / omeg['x']].
    rand_seed : :obj:`int`
        Value to seed the pseudorandom number generator.
    prop : :obj:`PropResult`
        Object containing the results of propagation, along with analysis
        methods.
    rot_coupling : :obj:`bool`, default=True
        Option to place coupling in a rotating reference frame, i.e. no
        momentum shift on the coupling operation.

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
        mesh_points : :obj:`iterable` of :obj:`int`, default=(256, 256)
            The number of grid points along the x- and y-axes, respectively.
        r_sizes : :obj:`iterable` of :obj:`int`, default=(16, 16)
            The half size of the real space grid along the x- and y-axes,
            respectively, in units of [a_x].
        atom_num : :obj:`int`, default=1e4
            Total atom number.
        pop_frac : :obj:`array_like` of :obj:`float`, default=(0.5, 0.5)
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
        self.space = {}

        assert sum(pop_frac) == 1.0, "Total population must equal 1."
        self.pop_frac = pop_frac

        if omeg is None:
            omeg0 = 2*np.pi*50
            self.omeg = {'x': omeg0, 'y': omeg0, 'z': 40 * omeg0}
            # ??? Maybe make self.omeg (& g_sc) object @properties with methods
            # for dynamic updating.
        else:
            omeg_names = {'x', 'y', 'z'}
            assert omeg_names == omeg.keys(), ("Keys for `omeg` must have "
                                               f"the form: {omeg_names}.")
            self.omeg = omeg

        if g_sc is None:
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

        self.rand_seed = None
        self.prop = None
        self.coupling = np.zeros(np.flip(mesh_points))
        self.detuning = np.zeros(np.flip(mesh_points))
        self.rot_coupling = True

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
        # TODO: Copy the code from the script file to the /code subfolder
        # Path to the directory containing all the simulation data & results
        if not os.path.isabs(path):
            data_path = ROOT_DIR + '/data/' + path + '/'
        else:
            data_path = path
        # Path to the subdirectory containing the raw trial data
        trial_data_path = data_path + 'trial_data/'

        # Path to the subdirectory containing the trial code.
        code_data_path = data_path + 'code/'

        if os.path.isdir(data_path):
            if not overwrite:
                raise FileExistsError(
                    f"The directory {data_path} already exists. "
                    "To overwrite this directory, supply the parameter "
                    "`overwrite=True`.")

            # warnings.warn(f"The directory {data_path} is being deleted and "
            #               "all containing data will be lost.")
            shutil.rmtree(data_path)  # Deletes the data directory

        # Create the directories and sub-directories
        data_path = f'{os.path.normpath(data_path)}{os.sep}'
        code_data_path = f'{os.path.normpath(code_data_path)}{os.sep}'
        trial_data_path = f'{os.path.normpath(trial_data_path)}{os.sep}'

        os.makedirs(data_path, exist_ok=True)
        os.makedirs(code_data_path, exist_ok=True)
        os.makedirs(trial_data_path, exist_ok=True)

        folder_name = os.path.basename(os.path.normpath(data_path))

        self.paths = {'data': data_path, 'trial': trial_data_path,
                      'code': code_data_path, 'folder': folder_name}

    def compute_tf_psi(self, phase_factor=1.0):
        """Compute the intial pseudospinor wavefunction `psi` and FFT `psik`.

        The psuedospinor wavefunction `psi` that is generated is a :obj:`list`
        of 2D NumPy arrays. The Thomas-Fermi solution is real and has the
        form of an inverted parabaloid. `psik` is a :obj:`list` of the 2D FFT
        of `psi`'s components.

        Parameters
        ----------
        phase_factor : :obj:`complex`, default=1.0
            Unit complex number; initial relative phase factor between the two
            spin components.

        """
        assert abs(phase_factor) == 1.0, ("Relative phase factor must have "
                                          "unit magnitude.")
        g_bare = [self.g_sc['uu'], self.g_sc['dd']]
        profile = np.real(np.sqrt((self.chem_pot - self.pot_eng + 0.j)))

        # Initial Thomas-Fermi wavefunction for the two spin components
        self.psi = [profile * np.sqrt(pop / abs(g)) for pop, g
                    in zip(self.pop_frac, g_bare)]
        self.psi[1] = self.psi[1] * phase_factor

        self.psi, _ = ttools.norm(self.psi, self.space['dv_r'], self.atom_num)
        self.psik = ttools.fft_2d(self.psi, self.space['dr'])

        # Saves the real- and k-space versions of the Thomas-Fermi wavefunction
        np.savez(self.paths['trial'] + 'tf_wf-' + self.paths['folder'],
                 psi=self.psi, psik=self.psik)

    def compute_tf_params(self, species='Rb87'):
        """Compute parameters and scales for the Thomas-Fermi solution.

        Parameters
        ----------
        sepecies : :obj:`str`, default='Rb87'
            Designates the atomic species and corresponding physical data
            used in the simulations.

        """
        # Relative size of y-axis trapping frequency relative to x-axis.
        y_trap = self.omeg['y'] / self.omeg['x']
        # Relative size of z-axis trapping frequency relative to x-axis.
        z_trap = self.omeg['z'] / self.omeg['x']

        self.a_x = np.sqrt(const.hbar / (const.Rb87['m'] * self.omeg['x']))

        # Dimensionless scattering length, [a_x]
        if species == 'Rb87':
            a_sc = const.Rb87['a_sc'] / self.a_x
        else:
            a_sc = 1

        self.chem_pot = ((4 * self.atom_num * a_sc * y_trap
                          * np.sqrt(z_trap / (2 * np.pi)))**(1/2))

        g_scale = np.sqrt(8 * z_trap * np.pi) * a_sc
        self.g_sc.update({k: g_scale * self.g_sc[k] for k in self.g_sc.keys()})
        self.rad_tf = np.sqrt(2 * self.chem_pot)

        self.time_scale = 1 / self.omeg['x']

    def compute_spatial_grids(self, mesh_points=(256, 256), r_sizes=(16, 16)):
        """Compute the real and momentum space grids.

        Stored in the :obj:`dict` `space` are the real- and momentum-space
        mesh grids, mesh sizes, mesh spacings, volume elements, and the
        corresponding linear arrays.

        Parameters
        ----------
        mesh_points : :obj:`iterable` of :obj:`int`, default=(256, 256)
            The number of grid points along the x- and y-axes, respectively.
        r_sizes : :obj:`iterable` of :obj:`int`, default=(16, 16)
            The half size of the grid along the real x- and y-axes,
            respectively,in units of [a_x].

        """
        assert all(point % 2 == 0 for point in mesh_points), (
            f"Number of mesh points {mesh_points} should be powers of 2.")
        mesh_points = np.array(mesh_points)
        r_sizes = np.array(r_sizes)

        # Spacing between real-space mesh points [a_x]
        self.space['dr'] = 2 * r_sizes / mesh_points
        # Half size of the grid along the kx- and ky- axes [1/a_x]
        k_sizes = np.pi / self.space['dr']
        # Spacing between momentum-space mesh points [1/a_x]
        self.space['dk'] = np.pi / r_sizes

        # Linear arrays for real- [a_x] and k-space [1/a_x], x- and y-axes
        self.space['x'] = self._compute_lin(r_sizes, mesh_points, axis=0)
        self.space['y'] = self._compute_lin(r_sizes, mesh_points, axis=1)
        self.space['kx'] = self._compute_lin(k_sizes, mesh_points, axis=0)
        self.space['ky'] = self._compute_lin(k_sizes, mesh_points, axis=1)

        # 2D meshes for computing the energy grids [a_x] and [1/a_x]
        x_mesh, y_mesh = np.meshgrid(self.space['x'], self.space['y'])
        kx_mesh, ky_mesh = np.meshgrid(self.space['kx'], self.space['ky'])
        self.space.update({'x_mesh': x_mesh, 'y_mesh': y_mesh})
        self.space.update({'kx_mesh': kx_mesh, 'ky_mesh': ky_mesh})

        # ??? Add functionality for Tukey filter window?

        # Real-space volume element used for normalization [a_x^2]
        self.space['dv_r'] = np.prod(self.space['dr'])
        # k-space volume element used for normalization [1/a_x^2]
        self.space['dv_k'] = np.prod(self.space['dk'])

        self.space['mesh_points'] = mesh_points
        self.space['r_sizes'] = r_sizes
        self.space['k_sizes'] = k_sizes

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

    @property
    def pot_eng(self):
        r"""Get the `pot_eng` attribute.

        2D potential energy grid, [\\hbar * omeg['x']].

        """
        return self._pot_eng

    @pot_eng.setter
    def pot_eng(self, array):
        """Set the `pot_eng` attribute."""
        self._pot_eng = array
        self.pot_eng_spin = [self._pot_eng] * 2

    @property
    def kin_eng(self):
        r"""Get the `kin_eng` attribute.

        2D kinetic energy grid, [\\hbar * omeg['x']]0

        """
        return self._kin_eng

    @kin_eng.setter
    def kin_eng(self, array):
        """Set the `kin_eng` attribute."""
        self._kin_eng = array
        self.kin_eng_spin = [self._kin_eng] * 2

    def compute_energy_grids(self):
        """Compute the initial potential and kinetic energy grids.

        Assumes that the BEC is in a harmonic trap. This harmonic potential
        determines the initial 'Thomas-Fermi' density profile of the BEC.
        `pot_eng` can be modified prior to progation to have any arbitrary
        potential energy landscape.

        Also assumes that the BEC has a simple free-particle kinetic energy
        dispersion. If using a momentum-dependent spin coupling, this grid
        will be modified later.

        """
        y_trap = self.omeg['y'] / self.omeg['x']
        self.pot_eng = (self.space['x_mesh']**2
                        + (y_trap * self.space['y_mesh'])**2) / 2
        self.kin_eng = (self.space['kx_mesh']**2
                        + self.space['ky_mesh']**2) / 2

    def _calc_atoms(self, psi=None, space='r'):
        """Given a list of wavefunctions, calculates the total atom number.

        Parameters
        ----------
        psi : :obj:`list` of NumPy :obj:`array`, optional.
            The pseudospinor wavefunction. If `psi` is not supplied, and
            depending on `space`, calculates based on the current instance
            attributes `self.psi` or `self.psik`.
        space : {'r', 'k'}, default='r'
            Specifies if the passed wavefunction is in real-space ('r') or
            momentum-space ('k').

        Returns
        -------
        atom_num : :obj:`float`
            The cumulative atom number in both spin components.
        """
        if space == 'r':
            if psi is None:
                psi = self.psi
            vol_elem = self.space['dv_r']
        elif space == 'k':
            if psi is None:
                psi = self.psik
            vol_elem = self.space['dv_k']

        atom_num = ttools.calc_atoms(psi, vol_elem)
        return atom_num

    def no_coupling_setup(self):
        """Provide the default parameters for no coupling."""
        self.is_coupling = False
        # pylint: disable=invalid-name
        self.kL_recoil = 1.0
        self.EL_recoil = 1.0

    def coupling_setup(self, wavel=790.1e-9, scale=1.0, kin_shift=False):
        """Calculate parameters for the momentum-(in)dependent coupling.

        If `kin_shift`=True, the kinetic energy grid receives a spin-dependent
        shift in momentum.

        Parameters
        ----------
        wavel : :obj:`float`, default=790.1e-9
            Wavelength of Raman coupling. Note that you must specify the
            wavelength in meters.
        scale : :obj:`float`, default=1.0
            The relative scale of recoil momentum. Interesting physics may be
            simulated by considering a recoil momentum that is hypothetically
            much larger or much smaller than the native wavelength recoil
            momentum.
        kin_shift : :obj:`bool`, default=False
            Option for a momentum-(in)dependent coupling.

        """
        # pylint: disable=attribute-defined-outside-init
        self.is_coupling = True
        # pylint: disable=invalid-name
        self.kL_recoil = scale * (np.sqrt(2) * np.pi / wavel * self.a_x)
        # pylint: disable=invalid-name
        self.EL_recoil = self.kL_recoil**2 / 2
        # Momentum shift option
        if kin_shift:
            shift = self.space['kx_mesh'] * self.kL_recoil
        else:
            shift = 0

        self.kin_eng_spin = [self.kin_eng + shift, self.kin_eng - shift]
        self.kin_eng_spin = [k - np.min(k) for k in self.kin_eng_spin]

    def shift_momentum(self, psik=None, scale=1.0, frac=(0.5, 0.5)):
        """Shifts momentum components of `psi` by a fracion of +/- kL_recoil.

        The ground-state solutions of Raman-coupled spinor systems in general
        have spinor components with both left- and right-moving momentum
        peaks. Providing a manual shift on the momentum-space wavefunction
        components better approximates these solutions, i.e. faster convergence
        in imaginary time propagation.

        Parameters
        ----------
        psik : :obj:`list` of NumPy :obj:`array`, optional.
            The momentum-space pseudospinor wavefunction. If `psik` is not
            provided, then this function uses the current class attribute
            `self.psik`.
        scale : :obj:`float`, default=1.0
            By default, the function shifts the momentum peaks by a single
            unit of recoil momenta `kL_recoil`. `scale` gives the option of
            scaling the shift larger or smaller for faster convergence.
        frac : :obj:`iterable`, default=(0.5, 0.5)
            The fraction of each spinor component's momentum peak to shift in
            either direction. frac=(0.5, 0.5) splits into two equal peaks,
            while (0.0, 1.0) and (1.0, 0.0) move the entire peak one
            direction or the other.

        """
        assert self.is_coupling, ("The `is_coupling` option is "
                                  f"{self.is_coupling}. Initialize coupling "
                                  "with `coupling_setup()`.")
        if psik is None:
            psik = self.psik

        shift = scale * self.kL_recoil / self.space['dk'][0]
        input_ = ttools.fft_2d(psik, self.space['dr'])
        result = [np.zeros_like(pk) for pk in psik]

        for i in range(len(psik)):
            positive = fourier_shift(input_[i], shift=[0, shift], axis=1)
            negative = fourier_shift(input_[i], shift=[0, -shift], axis=1)
            result[i] = frac[0]*positive + frac[1]*negative
            frac = np.flip(frac)
        self.psik = ttools.ifft_2d(result, self.space['dr'])
        self.psi = ttools.ifft_2d(self.psik, self.space['dr'])

    @property
    def coupling(self):
        r"""Get the `coupling` attribute.

        2D coupling array [\\hbar * omeg['x']].

        """
        return self._coupling

    @coupling.setter
    def coupling(self, array):
        """Set the `coupling` attribute."""
        self._coupling = array

    @property
    def detuning(self):
        r"""Get the `detuning` attribute.

        2D detuning array [\\hbar * omeg['x']].

        """
        return self._detuning

    @detuning.setter
    def detuning(self, array):
        """Set the `detuning` attribute."""
        self._detuning = array
        self.pot_eng_spin = [self.pot_eng + self._detuning / 2,
                             self.pot_eng - self._detuning / 2]

    def coupling_grad(self, slope, offset, axis=1):
        """Generate a linear gradient of the interspin coupling strength.

        Convenience function for generating linear gradients of the coupling.
        `coupling` can also be set to any arbitrary NumPy array directly:

        >>> ps = PSpinor()
        >>> ps.coupling_setup()
        >>> ps.coupling = np.exp(-ps.x_mesh**2 / 2)  # Gaussian function


        .. note:: When working with Raman recoil units [E_L], they will first
          need to be converted to [hbar*omeg_x] units.

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
            mesh = self.space['x_mesh']
        elif axis == 1:
            mesh = self.space['y_mesh']

        self.coupling = mesh * slope + offset

    def coupling_uniform(self, value):
        """Generate a uniform interspin coupling strength.

        Convenience function for generating unirom gradients of the coupling.
        `coupling` can also be set to any arbitrary NumPy array directly.

        Parameters
        ----------
        value : :obj:`float`
            The value of the coupling, in [hbar*omega_x].

        See Also
        --------
        coupling_grad : Coupling gradient

        """
        assert value >= 0, f"Cannot have a negative coupling value: {value}."
        self.coupling = np.ones_like(self.space['x_mesh']) * value

    def detuning_grad(self, slope, offset=0.0, axis=1):
        """Generate a linear gradient of the interspin coupling strength.

        Convenience function for generating linear gradients of the coupling.
        `detuning` can also be set to any arbitrary NumPy array directly:

        >>> ps = PSpinor()
        >>> ps.coupling_setup()
        >>> ps.detuning = np.sin(2 * np.pi * ps.x_mesh)  # Sin function


        .. note:: when working with Raman recoil units [E_L], they will first
          need to be converted to [hbar*omeg_x] units.

        Parameters
        ----------
        slope : :obj:`float`
            The slope of the detuning gradient, in [hbar*omeg_x/a_x].
        offset : :obj:`float`
            The origin offset of the detuning gradient, in [hbar*omeg_x].
        axis : :obj:`int`, optional
            The axis along which the detuning gradient runs.

        See Also
        --------
        coupling_grad : Coupling gradient

        """
        if axis == 0:
            mesh = self.space['x_mesh']
        elif axis == 1:
            mesh = self.space['y_mesh']

        self.detuning = mesh * slope + offset

    def detuning_uniform(self, value):
        """Generate a uniform coupling detuning.

        Convenience function for generating unirom gradients of the coupling.
        `detuning` can also be set to any arbitrary NumPy array directly.

        Parameters
        ----------
        value : :obj:`float`
            The value of the coupling, in [hbar*omega_x].

        See Also
        --------
        coupling_grad : Coupling gradient

        """
        self.detuning = np.ones_like(self.space['x_mesh']) * value

    def seed_regular_vortices(self):
        """Seed regularly-arranged vortices into the wavefunction.

        These seed-vortex functions might be moved to the ttools module.
        """
        raise NotImplementedError()

    def seed_random_vortices(self):
        """Seed randomly-arranged vortices into the wavefunction."""
        raise NotImplementedError()

    def plot_rdens(self, psi=None, spin=None, cmap='viridis', scale=1.0):
        """Plot the real-space density of the wavefunction.

        Shows the real-space density of either the up (`spin=0`),
        down (`spin=1`), or both (`spin=None`) spin components.

        Parameters
        ----------
        psi : :obj:`list` of Numpy :obj:`array`, optional.
            The wavefunction to plot. If no `psi` is supplied, then it uses the
            object attribute `self.psi`.
        spin : :obj:`int` or `None`, optional
            Which spin to plot. `None` plots both spins. 0 or 1 plots only the
            up or down spin, respectively.
        cmap : :obj:`str`, default='viridis'
            The matplotlib colormap to use for the plots.
        scale : :obj:`float`, default=1.0
            A factor by which to scale the spatial dimensions, e.g. Thomas-
            Fermi radius.

        See Also
        --------
        plotting_tools.plot_dens : Density plots.

        """
        if psi is None:
            psi = self.psi
        sizes = self.space['r_sizes']
        extent = np.ravel(np.vstack((-sizes, sizes)).T) / scale
        ptools.plot_dens(psi, spin, cmap, scale, extent=extent)

    def plot_kdens(self, psik=None, spin=None, cmap='viridis', scale=1.0):
        """Plot the momentum-space density of the wavefunction.

        Shows the momentum-space density of either the up (`spin=0`),
        down (`spin=1`), or both (`spin=None`) spin components.

        Parameters
        ----------
        psik : :obj:`list` of Numpy :obj:`array`, optional.
            The wavefunction to plot. If no `psik` is supplied, then it uses
            the object attribute `self.psik`.
        spin : :obj:`int` or `None`, optional
            Which spin to plot. `None` plots both spins. 0 or 1 plots only the
            up or down spin, respectively.
        cmap : :obj:`str`, default='viridis'
            The matplotlib colormap to use for the plots.
        scale : :obj:`float`, default=1.0
            A factor by which to scale the spatial dimensions, e.g. Thomas-
            Fermi radius.

        See Also
        --------
        plotting_tools.plot_dens : Density plots.

        """
        if psik is None:
            psik = self.psik
        sizes = self.space['k_sizes']
        extent = np.ravel(np.vstack((-sizes, sizes)).T) / scale
        ptools.plot_dens(psik, spin, cmap, scale, extent)

    def plot_rphase(self, psi=None, spin=None, cmap='twilight_shifted',
                    scale=1.0):
        """Plot the real-space phase of the wavefunction.

        Shows the real-space phase of either the up (`spin=0`),
        down (`spin=1`), or both (`spin=None`) spin components.

        Parameters
        ----------
        psi : :obj:`list` of Numpy :obj:`array`, optional.
            The wavefunction to plot. If no `psi` is supplied, then it uses
            the object attribute `self.psi`.
        spin : :obj:`int` or `None`, optional
            Which spin to plot. `None` plots both spins. 0 or 1 plots only the
            up or down spin, respectively.
        cmap : :obj:`str`, default='twilight_shifted'
            The matplotlib colormap to use for the plots.
        scale : :obj:`float`, default=1.0
            A factor by which to scale the spatial dimensions, e.g. Thomas-
            Fermi radius.

        See Also
        --------
        plotting_tools.plot_phase : Phase plots.

        """
        if psi is None:
            psi = self.psi
        sizes = self.space['r_sizes']
        extent = np.ravel(np.vstack((-sizes, sizes)).T) / scale
        ptools.plot_phase(psi, spin, cmap, scale, extent)

    # pylint: disable=too-many-arguments
    def plot_spins(self, rscale=1.0, kscale=1.0, cmap='viridis', save=True,
                   ext='.pdf', zoom=1.0):
        """Plot the densities (both real & k) and phases of spin components.

        Parameters
        ----------
        rscale : :obj:`float`, optional
            Real-space length scale. The default of 1.0 corresponds to the
            naturatl harmonic length scale along the x-axis.
        kscale : :obj:`float`, optional
            Momentum-space length scale. The default of 1.0 corresponds to the
            inverse harmonic length scale along the x-axis.
        cmap : :obj:`str`, optional
            Color map name for the real- and momentum-space density plots.
        save : :obj:`bool`, optional
            Saves the figure as a .pdf file (default). The filename has the
            format "/`data_path`/pop_evolution%s-`trial_name`.pdf".
        ext : :obj:`str`, optional
            Saved plot image file extension.
        zoom : :obj:`float`, optional
            A zoom factor for the k-space density plot.

        """
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale

        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale

        extents = {'r': r_extent, 'k': k_extent}

        fig, all_plots = ptools.plot_spins(self.psi, self.psik, extents,
                                           self.paths, cmap=cmap, save=save,
                                           ext=ext, zoom=zoom)
        return fig, all_plots

    # pylint: disable=too-many-arguments
    def imaginary(self, t_step, n_steps=1000, device='cpu',
                  is_sampling=False, n_samples=1):
        """Perform imaginary-time propagation.

        Propagation is carried out in a `TensorPropagator` object. The
        results are stored and returned in the `PropResult` object for further
        analysis.

        Parameters
        ----------
        t_step : :obj:`float`
            The propagation time step.
        n_steps : :obj:`int`, optional
            The total number of propagation steps.
        device : :obj:`str`, optional
            {'cpu', 'cuda'}
        is_sampling : :obj:`bool`, optional
            Option to sample wavefunctions throughout the propagation.
        n_samples : :obj:`int`, optional
            The number of samples to collect.

        """
        prop = tprop.TensorPropagator(self, t_step, n_steps, device,
                                      time='imag',
                                      is_sampling=is_sampling,
                                      n_samples=n_samples)
        result = prop.prop_loop(prop.n_steps)

        # Include PSpinor attributes with the result object
        result.paths = self.paths
        result.time_scale = self.time_scale
        result.space = self.space

        self.psik = result.psik
        self.psi = result.psi
        return result, prop

    def real(self, t_step, n_steps=1000, device='cpu', is_sampling=False,
             n_samples=1):
        """Perform real-time propagation.

        Propagation is carried out in a `TensorPropagator` object. The
        results are stored and returned in the `PropResult` object for further
        analysis.

        Parameters
        ----------
        t_step : :obj:`float`
            The propagation time step.
        n_steps : :obj:`int`, optional
            The total number of propagation steps.
        device : :obj:`str`, optional
            {'cpu', 'cuda'}
        is_sampling : :obj:`bool`, optional
            Option to sample wavefunctions throughout the propagation.
        n_samples : :obj:`int`, optional
            The number of samples to collect.

        """
        prop = tprop.TensorPropagator(self, t_step, n_steps, device,
                                      time='real',
                                      is_sampling=is_sampling,
                                      n_samples=n_samples)
        result = prop.prop_loop(prop.n_steps)

        # Include PSpinor attributes with the result object
        result.paths = self.paths
        result.time_scale = self.time_scale
        result.space = self.space

        self.psik = result.psik
        self.psi = result.psi
        return result, prop
