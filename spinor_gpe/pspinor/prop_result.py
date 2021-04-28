"""Placeholder for the prop_result.py module."""
import numpy as np

from spinor_gpe.pspinor import tensor_tools as ttools


class PropResult:
    """Results of propagation, along with plotting and analysis tools."""

    def __init__(self, psi_final, psik_final, eng_final, pops,
                 sampled_path=None):
        """Generate a PropResult instance.

        Parameters
        ----------
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
