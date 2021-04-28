"""plotting_tools.py module."""
import numpy as np
from matplotlib import pyplot as plt

from spinor_gpe.pspinor import tensor_tools as ttools


def plot_dens(psi, spin=None, cmap='viridis', scale=1.,
              extent=None):
    """Plot the real or k-space density of the wavefunction.

    Based on the value of the `spin` parameter, this function will plot either
    the up (0), down (1), or both (None) spin components of the spinor
    wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array`, optional.
        The wavefunction to plot. If no `psi` is supplied, then it uses the
        object attribute `self.psi`.
    spin : :obj:`int` or `None`, optional
        Which spin to plot. `None` plots both spins. 0 or 1 plots only the
        up or down spin, respectively.
    cmap : :obj:`str`, optional
        The colormap to use for the plots.
    scale : :obj:`float`, optional
        A factor to scale the spatial dimensions by, e.g. Thomas-Fermi radius.
    extent : iterable
        The spatial extent of the wavefunction components, in the format
        np.array([x_min, x_max, y_min, y_max]). Determines the natural spatial
        scale of the plot.

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

    for i, den in enumerate(dens):
        axs[i].imshow(den, cmap=cmap, extent=extent)

    plt.show()


def plot_phase(psi=None, spin=None, cmap='twilight_shifted', scale=1,
               extent=None):
    """Plot the phase of the real wavefunction.

    Based on the value of the `spin` parameter, this function will plot either
    the up (0), down (1), or both (None) spin components of the spinor
    wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array`, optional.
        The wavefunction to plot. If no `psi` is supplied, then it uses the
        object attribute `self.psi`.
    spin : :obj:`int` or `None`, optional
        Which spin to plot. `None` plots both spins. 0 or 1 plots only the
        up or down spin, respectively.
    cmap : :obj:`str`, optional
        The colormap to use for the plots.
    scale : :obj:`float`, optional
        A factor to scale the spatial dimensions by, e.g. Thomas-Fermi radius.
    extent : iterable
        The spatial extent of the wavefunction components, in the format
        np.array([x_min, x_max, y_min, y_max]). Determines the natural spatial
        scale of the plot.

    """
    if spin is None:
        n_plots = 2
    else:
        assert spin in (0, 1), f"The `spin` parameter should be 0 or 1, \
            not {spin}."
        n_plots = 1
        psi = [psi[spin]]

    phase = ttools.phase(psi)
    dens = ttools.density(psi)

    fig, axs = plt.subplots(1, n_plots, sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):  # Makes single axs an array
        axs = np.array([axs])

    for i, phz in enumerate(phase):
        phz[dens[i] < 1e-6 * np.max(dens[i])] = 0
        axs[i].imshow(phz, cmap=cmap, extent=extent)

    plt.show()
