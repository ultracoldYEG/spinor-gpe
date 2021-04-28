"""Placeholder for the prop_result.py module."""
import numpy as np
from matplotlib import pyplot as plt

from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor import plotting_tools as ptools


class PropResult:
    """Results of propagation, along with plotting and analysis tools."""

    def __init__(self, psi_final, psik_final, eng_final, pops,
                 sampled_path=None):
        """Generate a PropResult instance.

        Parameters
        ----------
        psi : :obj:`list` of NumPy :obj:`array`
            The finished propagated real-space wavefunction.
        psik : :obj:`list` of NumPy :obj:`array`
            The finished propagated k-space wavefunction.
        eng_final : :obj:`float`
            The final energy expectation value.
        pops : :obj:`dict`
            dict of {str: NumPy :obj:`array`}. Contains the 'times' and 'vals'
            of the spin components' populations throughout the propagation.
        sampled_path : :obj:`str`, optional
            The path to the .npz file where the sampled wavefunctions and
            times are stored for this result.
        """
        self.psi = psi_final
        self.psik = psik_final
        self.eng_final = eng_final
        self.pops = pops
        self.sampled_path = sampled_path

        self.dens = ttools.density(self.psi)
        self.densk = ttools.density(self.psik)
        self.phase = ttools.phase(self.psi)

        self.paths = None
        self.r_scale = None
        self.k_scale = None
        self.t_scale = None

    def calc_separation(self):
        """Calculate the phase separation of the two spin components."""
        s = 1 - np.sum(ttools.prod(self.dens))  # FIXME
        return s

    def plot_spins(self):
        """Plot the densities (real & k) and phases of spin components."""

    def plot_total(self):
        """Plot the total real-space density and phase of the wavefunction."""

    def plot_eng(self):
        """Plot the sampled energy expectation values."""

    def plot_pops(self, scaled=True, save=True, ext='.pdf'):
        """Plot the spin populations as a function of propagation time.

        Parameters
        ----------
        scaled : :obj:`bool`, optional
            If `scaled` is True then the time-axis will be rescaled into
            proper time units. Otherwise, it's left in dimensionless time
            units.
        save : :obj:`bool`, optional
            Saves the figure as a .pdf file (default). The filename has the
            format "/`data_path`/pop_evolution%s-`trial_name`.pdf".
        """
        if scaled:
            xlabel = 'Time [s]'
            scale = self.t_scale
        else:
            xlabel = 'Time [$1/\\omega_x$]'
            scale = 1.0
        plt.figure()
        plt.plot(self.pops['times'] * scale, self.pops['vals'])
        plt.ylabel('Population')
        plt.xlabel(xlabel)
        plt.grid(alpha=0.25)
        if save:
            test_name = self.paths['data'] + 'pop_evolution'
            file_name = ptools.next_available_path(test_name,
                                                   self.paths['folder'], ext)
            plt.savefig(file_name)
        plt.show()

    def analyze_vortex(self):
        """Compute the total vorticity in each spin component."""

    def make_movie(self):
        """Generate a movie of the wavefunctions' densities and phases."""
