"""plotting_tools.py module."""
import os
import sys
import time as t

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

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


def plot_spins(psi, psik, extents, paths, cmap='viridis', save=True,
               ext='.pdf'):
    """Plot the densities (real & k) and phases of spin components.

    Parameters
    ----------
    psi :
    psik :
    extents :
    paths :
    cmap : :obj:`str`, optional
        Color map name for the real- and momentum-space density plots.
    save : :obj:`bool`, optional
        Saves the figure as a .pdf file (default). The filename has the
        format "/`data_path`/pop_evolution%s-`trial_name`.pdf".
    ext : :obj:`str`, optional
        Saved plot image file extension.

    Returns
    -------
    fig :
    all_plots :

    """
    # pylint: disable=unused-variable
    dens = ttools.density(psi)
    phase = ttools.phase(psi, uwrap=True, dens=dens)
    densk = ttools.density(psik)

    widths = [1] * 4
    heights = [1] * 6
    fig = plt.figure(figsize=(5.5, 6.4))
    gsp = gridspec.GridSpec(6, 4, width_ratios=widths,
                            height_ratios=heights)
    r_axs = [fig.add_subplot(gsp[0:2, 0:2]),
             fig.add_subplot(gsp[0:2, 2:])]
    ph_axs = [fig.add_subplot(gsp[2:4, 0:2]),
              fig.add_subplot(gsp[2:4, 2:])]
    k_axs = [fig.add_subplot(gsp[4:6, 0:2]),
             fig.add_subplot(gsp[4:6, 2:])]

    # Real-space density plot
    r_plots = [ax.imshow(d, cmap=cmap, origin='lower', extent=extents['r'],
                         vmin=0)
               for ax, d in zip(r_axs, dens)]
    r_cb = [fig.colorbar(plot, ax=ax) for plot, ax in zip(r_plots, r_axs)]
    any(ax.set_xlabel('$x$') for ax in r_axs)
    any(ax.set_ylabel('$y$') for ax in r_axs)

    # Real-space phase plot
    ph_plots = [ax.imshow(phz, cmap='twilight_shifted', origin='lower',
                          extent=extents['r'], vmin=-np.pi, vmax=np.pi)
                for ax, phz in zip(ph_axs, phase)]
    ph_cb = [fig.colorbar(plot, ax=ax)
             for plot, ax in zip(ph_plots, ph_axs)]
    any(cb.set_ticks(np.linspace(-np.pi, np.pi, 5)) for cb in ph_cb)
    any(cb.set_ticklabels(['$-\\pi$', '', '$0$', '', '$\\pi$'])
        for cb in ph_cb)
    any(ax.set_xlabel('$x$') for ax in ph_axs)
    any(ax.set_ylabel('$y$') for ax in ph_axs)

    # Momentum-space density plot
    k_plots = [ax.imshow(d, cmap=cmap, origin='lower', extent=extents['k'],
                         vmin=0)
               for ax, d in zip(k_axs, densk)]
    k_cb = [fig.colorbar(plot, ax=ax) for plot, ax in zip(k_plots, k_axs)]
    any(ax.set_xlabel('$k_x$') for ax in k_axs)
    any(ax.set_ylabel('$k_y$') for ax in k_axs)

    plt.tight_layout()

    # Save figure
    if save:
        test_name = paths['data'] + 'spin_dens_phase'
        file_name = next_available_path(test_name, paths['folder'], ext)
        plt.savefig(file_name)
    plt.show()

    all_plots = {'r': r_plots, 'ph': ph_plots, 'k': k_plots}
    return fig, all_plots
