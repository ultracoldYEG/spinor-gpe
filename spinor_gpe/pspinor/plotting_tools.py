"""plotting_tools.py module."""
import os
import sys
import time as t

import numpy as np
from matplotlib import pyplot as plt

from spinor_gpe.pspinor import tensor_tools as ttools


def next_available_path(file_name, trial_name, ext=''):
    """
    Test for the next available path for a given file.

    Parameters
    ----------
    path_pattern : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    i = 1
    test_path = file_name + str(i) + '-' + trial_name + ext
    while os.path.exists(test_path):
        i += 1
        test_path = file_name + str(i) + '-' + trial_name + ext
    return test_path


def progress_message(frame, n_total):
    """Display an updating progress message while the animation is saving."""
    global timelast
    if frame != 0:
        timethis = t.time()
        itertime = 1 / (timethis - timelast)
        timelast = timethis
        remaining = time_remaining(frame, n_total, itertime)

        message = ('\r' + str(frame) + '/' + str(n_total) + ', '
                   + remaining + f', {itertime:.2f} it/sec')
        sys.stdout.write(message)
        sys.stdout.flush()
    else:
        timelast = t.time()
    return timelast


def time_remaining(frame, n_total, its):
    """Calculate completion time in the progressMessage function."""
    totaltime = (n_total - frame) / its
    hours = int(np.floor(totaltime / 3600))
    minutes = int(np.floor((totaltime - hours * 3600) / 60))
    seconds = int(np.mod(totaltime, 60))
    if len(str(hours)) < 2:
        hour_str = '0' + str(hours)
    else:
        hour_str = str(hours)

    if len(str(minutes)) < 2:
        minute_str = '0' + str(minutes)
    else:
        minute_str = str(minutes)

    if len(str(seconds)) < 2:
        second_str = '0' + str(seconds)
    else:
        second_str = str(seconds)
    return f'[{hour_str}:{minute_str}:{second_str}]'


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
