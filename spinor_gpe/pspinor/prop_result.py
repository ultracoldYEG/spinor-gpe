"""Placeholder for the prop_result.py module."""
import numpy as np

from spinor_gpe.pspinor import tensor_tools as ttools


class PropResult:
    """Results of propagation, along with plotting and analysis tools."""

    def __init__(self, psik, ):
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
