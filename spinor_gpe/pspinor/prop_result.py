"""prop_result.py module."""
import os
import warnings
import sys
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as ani

from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor import plotting_tools as ptools

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


class PropResult:
    """The result of propagation, with plotting and analysis tools.

    Attributes
    ----------
    psi : :obj:`list` of :obj:`array`
        The final real-space wavefunctions.
    psik : :obj:`list` of :obj:`array`
        The final momentum-space wavefunctions.
    eng_final : :obj:`list`
        The energy expectation values: [<total>, <kin.>, <pot.>, <int.>].
    pops : :obj:`dict` of :obj:`array`
        Times and populations at every time step, {'times', 'vals'}.
    sampled_path : :obj:`str`
        Path to the .npz file where the sampled wavefunctions and times are
        stored for this result.
    dens : :obj:`list` of :obj:`array`
        The final real-space densities.
    densk : :obj:`list` of :obj:`array`
        The final momentum-space densities.
    phase : :obj:`list` of :obj:`array`
        The final real-space phases.
    paths : :obj:`dict`
        See ``pspinor.PSpinor``.
    time_scale : :obj:`float`
        See ``pspinor.PSpinor``.
    space : :obj:`dict` of :obj:`array`
        See ``tensor_propagator.TensorPropagator``.

    """

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
        self.phase = ttools.phase(self.psi, uwrap=False, dens=self.dens)

        self.paths = dict()
        self.time_scale = None
        self.space = dict()

    def calc_separation(self):
        """Calculate the phase separation of the two spin components."""
        s = 1 - (np.sum(ttools.prod(self.dens))
                 / np.sqrt(np.sum(self.dens[0]**2) * np.sum(self.dens[1]**2)))
        return s

    def plot_spins(self, rscale=1.0, kscale=1.0, cmap='viridis', save=True,
                   ext='.pdf', show=True, zoom=1.0):
        """Plot the densities (real & k) and phases of spin components.

        Parameters
        ----------
        rscale : :obj:`float`, default=1.0
            Real-space length scale. The default of 1.0 corresponds to the
            naturatl harmonic length scale along the x-axis.
        kscale : :obj:`float`, default=1.0
            Momentum-space length scale. The default of 1.0 corresponds to the
            inverse harmonic length scale along the x-axis.
        cmap : :obj:`str`, default='viridis'
            Matplotlib color map name for the real- and momentum-space
            density plots.
        save : :obj:`bool`, default=True
            Saves the figure as a .pdf file (default). The filename has the
            format "/`data_path`/spin_dens_phase%s-`trial_name`.pdf".
        ext : :obj:`str`, default='.pdf'
            File extension for the saved plot image.
        zoom : :obj:`float`, default=1.0
            A zoom factor for the k-space density plot.

        """
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale

        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale

        extents = {'r': r_extent, 'k': k_extent}

        fig, all_plots = ptools.plot_spins(self.psi, self.psik, extents,
                                           self.paths, cmap=cmap, save=save,
                                           ext=ext, show=show, zoom=zoom)
        return fig, all_plots

    def plot_total(self, rscale=1.0, kscale=1.0, cmap='viridis', save=True,
                   ext='.pdf', show=True, zoom=1.0):
        """Plot the total real-space density and phase of the wavefunction.

        Parameters
        ----------
        rscale : :obj:`float`, default=1.0
            Real-space length scale. The default of 1.0 corresponds to the
            naturatl harmonic length scale along the x-axis.
        kscale : :obj:`float`, default=1.0
            Momentum-space length scale. The default of 1.0 corresponds to the
            inverse harmonic length scale along the x-axis.
        cmap : :obj:`str`, default='viridis'
            Color map name for the real- and momentum-space density plots.
        save : :obj:`bool`, default=True
            Saves the figure as a .pdf file (default). The filename has the
            format "/`data_path`/pop_evolution%s-`trial_name`.pdf".
        ext : :obj:`str`, default='.pdf'
            File extension for the saved plot image.
        show : :obj:`bool`, default=True
            Option to display the generated image.
        zoom : :obj:`float`, default = 1.0
            A zoom factor for the k-space density plot.

        """
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale

        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale

        extents = {'r': r_extent, 'k': k_extent}

        fig, all_plots = ptools.plot_total(self.psi, self.psik, extents,
                                           self.paths, cmap=cmap, save=save,
                                           ext=ext, show=show, zoom=zoom)
        return fig, all_plots

    def plot_eng(self):
        """Plot the sampled energy expectation values."""
        raise NotImplementedError()

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
        ext : :obj:`str`, optional
            File extension for the saved plot image.
        """
        if scaled:
            xlabel = 'Time [s]'
            scale = self.time_scale
        else:
            xlabel = 'Time [$1/\\omega_x$]'
            scale = 1.0
        diff = np.abs(np.diff(self.pops['vals']))

        fig = plt.figure(figsize=(12, 4))
        ax0 = fig.add_subplot(121)
        lines = ax0.plot(self.pops['times'] * scale, self.pops['vals'])
        ax0.set_ylabel('Population')
        ax0.set_xlabel(xlabel)
        ax0.grid(alpha=0.5)
        ax0.legend(lines, ('Pop. $| \\uparrow\\rangle$',
                           'Pop. $| \\downarrow\\rangle$'))

        ax1 = fig.add_subplot(122)
        ax1.plot(self.pops['times'] * scale, diff)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Abs. Population Difference')
        ax1.grid(alpha=0.5)
        ax1.set_yscale('log')
        ax1.set_ylim(2e-16, None)

        if save:
            test_name = self.paths['data'] + 'pop_evolution'
            file_name = ptools.next_available_path(test_name,
                                                   self.paths['folder'], ext)
            plt.savefig(file_name)
        plt.show()

    def analyze_vortex(self):
        """Compute the total vorticity in each spin component."""
        raise NotImplementedError()

    def make_movie(self, rscale=1.0, kscale=1.0, cmap='viridis', play=False,
                   zoom=1.0, norm_type='all'):
        """Generate a movie of the wavefunctions' densities and phases.

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
        play : :obj:`bool`, default=False
            If True, the movie is opened in the computer's default media
            player after it is saved.
        kzoom : :obj:`float`, optional
            A zoom factor for the k-space density plot.
        norm_type : :obj:`str`, optional
            {'all', 'half'} Normalizes the colormaps to the full or half sum
            of the max densites. 'half' is useful for visualizing situations
            where the population is equally divided between the two spins.

        """
        def animate(frame, n_total, val):
            global timelast, timethis
            psik = psiks[frame]
            psi = ttools.ifft_2d(psik, self.space['dr'])

            dens = ttools.density(psi)
            phase = ttools.phase(psi, uwrap=False, dens=dens)
            densk = ttools.density(psik)
            max_r = [np.max(d) for d in dens]
            max_k = [np.max(d) for d in densk]

            any(plot.set_data(d) for plot, d in zip(all_plots['r'], dens))
            any(plot.set_data(ph) for plot, ph in zip(all_plots['ph'], phase))
            any(plot.set_data(dk) for plot, dk in zip(all_plots['k'], densk))

            any(plot.set_clim(0, sum(max_r) / val) for plot in all_plots['r'])
            any(plot.set_clim(0, sum(max_k) / val) for plot in all_plots['k'])

            ptools.progress_message(frame, n_total)

        if not os.path.exists(str((self.sampled_path))):
            warnings.warn("Cannot generate propagation movie. No sampled "
                          "wavefuntion data exists.")
            return

        if norm_type == 'all':
            norm_val = 1.0
        elif norm_type == 'half':
            norm_val = 2.0

        with np.load(self.sampled_path) as sampled:
            times = sampled['times']
            psiks = sampled['psiks']
            # ??? Need to rebin grids for speed?

        n_samples = len(times)
        writer = ani.writers['ffmpeg'](fps=5, bitrate=-1)
        fig, all_plots = self.plot_spins(rscale, kscale, cmap, save=False,
                                         show=False, zoom=zoom)

        # Create and then save the animation
        anim = ani.FuncAnimation(fig, animate, frames=n_samples, blit=False,
                                 fargs=(n_samples, norm_val,))

        # Save animation
        test_name = self.paths['data'] + 'prop_movie'
        file_name = ptools.next_available_path(test_name,
                                               self.paths['folder'],
                                               '.mp4')
        anim.save(file_name, writer=writer)
        plt.close(fig)

        if play:
            if sys.platform == "win32":
                os.startfile(file_name)
            else:
                opener ="open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, file_name])

    def rebin(self, arr, new_shape=(256, 256)):
        """Rebin a 2D `arr` to shape `new_shape` by averaging.

        This may be used when generating movies of sampled wavefunctions.
        By down-sampling the density grids, the movie is generated much faster.

        Parameters
        ----------
        arr : 2D :obj:`list` or NumPy :obj:`array`
            The input 2D array to rebin.
        new_shape : :obj:`iterable`, default=(256, 256)
            The target rebinned shape.

        """
        assert arr[0].shape == arr[1].shape
        new_arr = [0, 0]
        curr_shape = arr[0].shape
        if new_shape < curr_shape:
            for i, a in enumerate(arr):
                shape = (new_shape[0], a.shape[0] // new_shape[0],
                         new_shape[1], a.shape[1] // new_shape[1])
                new_arr[i] = a.reshape(shape).mean(-1).mean(1)
        else:
            new_arr = arr
        return new_arr
