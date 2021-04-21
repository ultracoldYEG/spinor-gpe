"""plotting_tools.py module."""
import numpy as np
from matplotlib import pyplot as plt

from pspinor import tensor_tools as ttools


def plot_dens(psi=None, spin=None, cmap='viridis', scale=1,
              extent=None):
    """Plot the real or k-space density of the wavefunction.

    Based on the value passed to `spin`, this function will plot either
    the up (0), down (1), or both (None) spin components.

    """
    if spin is None:
        n_plots = 2
    else:
        assert spin in (0, 1), f"The `spin` parameter should be 0 or 1, \
            not {spin}."
        n_plots = 1
        psi = [psi[spin]]

    dens = ttools.density(psi)

    fig, axs = plt.subplots(1, n_plots, sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):  # Makes single axs an array
        axs = np.array([axs])

    for i, d in enumerate(dens):
        axs[i].imshow(d, cmap=cmap, extent=extent)
    plt.show()
