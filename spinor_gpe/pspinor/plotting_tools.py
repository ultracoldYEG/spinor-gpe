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
    file_name : :obj:`str`
        The base file path to test.
    trial_name : :obj:`str`
        The name of the trial to append to the end of the file name.
    ext : :obj:`str`, default=''
        File extention.

    Returns
    -------
    test_path : :obj:`str`
        The file path with the next available index.

    """
    i = 1
    test_path = file_name + str(i) + '-' + trial_name + ext
    while os.path.exists(test_path):
        i += 1
        test_path = file_name + str(i) + '-' + trial_name + ext
    return test_path


def progress_message(frame, n_total):
    """Display an updating progress message while the animation is saving.

    This function produces an output similar to what the ``tqdm`` package
    gives. This one works in situations where ``tqdm`` cannot be applied.

    Parameters
    ----------
    frame : :obj:`int`
        The current frame/index number in the loop.
    n_total : :obj:`int`
        The total number of frames in the loop.

    """
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
    """Calculate completion time in the progress_message function.

    Parameters
    ----------
    frame : :obj:`int`
        The current frame/index number in the loop.
    n_total : :obj:`int`
        The total number of frames in the loop.
    its : :obj:`float`
        The time in seconds between successive iterations.

    """
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
    cmap : :obj:`str`, default='viridis'
        The matplotlib colormap to use for the plots.
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


def plot_phase(psi, spin=None, cmap='twilight_shifted', scale=1,
               extent=None):
    """Plot the phase of the real wavefunction.

    Based on the value of the `spin` parameter, this function will plot either
    the up (0), down (1), or both (None) spin components of the spinor
    wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array`, optional.
        The wavefunction to plot.
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
               ext='.pdf', show=True, zoom=1.0):
    """Plot the densities (real & k) and phases of spin components.

    In total, six subplots are generated. Each pair of axes are stored together
    in a list, which is returned in `all_plots`.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array`, optional.
        The real-space wavefunction to plot.
    psik : :obj:`list` of Numpy :obj:`array`, optional.
        The momentum-space wavefunction to plot.
    extents : :obj:`dict` of :obj:`iterable`
        The dictionary keys are {'r', 'k'}, and each value is a 4-element
        iterables giving the x- (kx-) and y- (ky-) spatial extents of the plot
        area, e.g. [x_min, x_max, y_min, y_max]
    paths : :obj:`dict` of :obj:`str`
        The dictionary keys contain {'data', 'folder'}, and the values are
        absolute paths to the saved data path and its containing folder.
    cmap : :obj:`str`, default='viridis'
        Matplotlib color map name for the real- and momentum-space density
        plots.
    save : :obj:`bool`, default=True
        Saves the figure as a .pdf file (default). The filename has the
        format "/`data_path`/spin_dens_phase%s-`trial_name`.pdf".
    ext : :obj:`str`, default='.pdf'
        File extension for the saved density plots.
    zoom : :obj:`float`, default=1.0
        A zoom factor for the k-space density plot.

    Returns
    -------
    fig : :obj:`plt.Figure`
        The matplotlib figure for the plot.
    all_plots : :obj:`dict` of :obj:`list`
        The keys are {'r', 'ph', 'k'}. Each value is a pair of
        :obj:`matplotlib.image.AxesImage` for both spins.

    """
    # pylint: disable=unused-variable
    dens = ttools.density(psi)
    phase = ttools.phase(psi, uwrap=False, dens=dens)
    densk = ttools.density(psik)

    widths = [1] * 4
    heights = [1] * 6
    fig = plt.figure(figsize=(5.5, 6.4))
    gsp = gridspec.GridSpec(6, 4, width_ratios=widths, height_ratios=heights)

    r_axs = [fig.add_subplot(gsp[0:2, 0:2]), fig.add_subplot(gsp[0:2, 2:])]
    ph_axs = [fig.add_subplot(gsp[2:4, 0:2]), fig.add_subplot(gsp[2:4, 2:])]
    k_axs = [fig.add_subplot(gsp[4:6, 0:2]), fig.add_subplot(gsp[4:6, 2:])]

    # Real-space density plot
    r_plots = [ax.imshow(d, cmap=cmap, origin='lower', extent=extents['r'],
                         vmin=0, aspect='equal')
               for ax, d in zip(r_axs, dens)]
    r_cb = [fig.colorbar(plot, ax=ax) for plot, ax in zip(r_plots, r_axs)]
    all(ax.set_xlabel('$x$') for ax in r_axs)
    all(ax.set_ylabel('$y$') for ax in r_axs)

    # Real-space phase plot
    ph_plots = [ax.imshow(phz, cmap='twilight_shifted', origin='lower',
                          extent=extents['r'], vmin=-np.pi, vmax=np.pi,
                          aspect='equal')
                for ax, phz in zip(ph_axs, phase)]
    ph_cb = [fig.colorbar(plot, ax=ax)
             for plot, ax in zip(ph_plots, ph_axs)]
    any(cb.set_ticks(np.linspace(-np.pi, np.pi, 5)) for cb in ph_cb)
    any(cb.set_ticklabels(['$-\\pi$', '', '$0$', '', '$\\pi$'])
        for cb in ph_cb)
    all(ax.set_xlabel('$x$') for ax in ph_axs)
    all(ax.set_ylabel('$y$') for ax in ph_axs)

    # Momentum-space density plot
    k_plots = [ax.imshow(d, cmap=cmap, origin='lower', extent=extents['k'],
                         vmin=0, aspect='equal')
               for ax, d in zip(k_axs, densk)]
    k_cb = [fig.colorbar(plot, ax=ax) for plot, ax in zip(k_plots, k_axs)]
    all(ax.set_xlabel('$k_x$') for ax in k_axs)
    all(ax.set_ylabel('$k_y$') for ax in k_axs)
    zoom_kext = extents['k'] / zoom
    all(ax.set_xlim(zoom_kext[:2]) for ax in k_axs)
    all(ax.set_ylim(zoom_kext[2:]) for ax in k_axs)

    plt.tight_layout()

    # Save figure
    if save:
        test_name = paths['data'] + 'spin_dens_phase'
        file_name = next_available_path(test_name, paths['folder'], ext)
        plt.savefig(file_name)
    if show:
        plt.show()

    all_plots = {'r': r_plots, 'ph': ph_plots, 'k': k_plots}
    return fig, all_plots


def plot_total(psi, psik, extents, paths, cmap='viridis', save=True,
               ext='.pdf', show=True, zoom=1.0):
    """Plot the total densities and phase of the wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array`, optional.
        The real-space wavefunction to plot.
    psik : :obj:`list` of Numpy :obj:`array`, optional.
        The momentum-space wavefunction to plot.
    extents : :obj:`dict` of :obj:`iterable`
        The dictionary keys are {'r', 'k'}, and each value is a 4-element
        iterables giving the x- (kx-) and y- (ky-) spatial extents of the plot
        area, e.g. [x_min, x_max, y_min, y_max]
    paths : :obj:`dict` of :obj:`str`
        The dictionary keys contain {'data', 'folder'}, and the values are
        absolute paths to the saved data path and its containing folder.
    cmap : :obj:`str`, default='viridis'
        Matplotlib color map name for the real- and momentum-space density
        plots.
    save : :obj:`bool`, default=True
        Saves the figure as a .pdf file (default). The filename has the
        format "/`data_path`/spin_dens_phase%s-`trial_name`.pdf".
    ext : :obj:`str`, default='.pdf'
        File extension for the saved density plots.
    zoom : :obj:`float`, default=1.0
        A zoom factor for the k-space density plot.

    Returns
    -------
    fig : :obj:`plt.Figure`
        The matplotlib figure for the plot.
    all_plots : :obj:`dict` of :obj:`matplotlib.image.AxesImage`
        The keys are {'r', 'ph', 'k'}. Each value is a separate
        :obj:`matplotlib.image.AxesImage`.

    """
    dens_tot_r = sum(ttools.density(psi))
    ph_tot_r = ttools.phase(sum(psi), uwrap=False, dens=dens_tot_r)
    dens_tot_k = sum(ttools.density(psik))

    widths = [1] * 4
    heights = [1] * 4
    fig = plt.figure()
    gsp = gridspec.GridSpec(4, 4, width_ratios=widths,
                            height_ratios=heights)
    r_ax = fig.add_subplot(gsp[0:2, 0:2])
    ph_ax = fig.add_subplot(gsp[0:2, 2:])
    k_ax = fig.add_subplot(gsp[2:, 1:3])

    # Real-space density plot
    r_plot = r_ax.imshow(dens_tot_r, cmap=cmap, origin='lower',
                         extent=extents['r'], vmin=0)
    fig.colorbar(r_plot, ax=r_ax)
    r_ax.set_xlabel('$x$')
    r_ax.set_ylabel('$y$')

    # Real-space phase plot
    ph_plot = ph_ax.imshow(ph_tot_r, cmap='twilight_shifted',
                           origin='lower', extent=extents['r'],
                           vmin=-np.pi, vmax=np.pi)

    ph_cb = fig.colorbar(ph_plot, ax=ph_ax)
    ph_cb.set_ticks(np.linspace(-np.pi, np.pi, 5))
    ph_cb.set_ticklabels(['$-\\pi$', '', '$0$', '', '$\\pi$'])
    ph_ax.set_xlabel('$x$')
    ph_ax.set_ylabel('$y$')

    # Momentum-space density plot
    k_plot = k_ax.imshow(dens_tot_k, cmap=cmap, origin='lower',
                         extent=extents['k'], vmin=0)
    fig.colorbar(k_plot, ax=k_ax)
    k_ax.set_xlabel('$k_x$')
    k_ax.set_ylabel('$k_y$')
    k_ax.set_xlim(extents['k'][:2] / zoom)
    k_ax.set_ylim(extents['k'][2:] / zoom)

    plt.tight_layout()

    # Save figure
    if save:
        test_name = paths['data'] + 'total_dens_phase'
        file_name = next_available_path(test_name,
                                        paths['folder'], ext)
        plt.savefig(file_name)
    if show:
        plt.show()

    all_plots = {'r': r_plot, 'ph': ph_plot, 'k': k_plot}
    return fig, all_plots
