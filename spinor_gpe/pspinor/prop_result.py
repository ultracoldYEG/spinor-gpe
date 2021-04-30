"""prop_result.py module."""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation

from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor import plotting_tools as ptools

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


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
        self.phase = ttools.phase(self.psi, uwrap=True, dens=self.dens)

        self.paths = dict()
        self.t_scale = None
        self.space = dict()

    def calc_separation(self):
        """Calculate the phase separation of the two spin components."""
        s = 1 - np.sum(ttools.prod(self.dens))  # FIXME
        return s

    def plot_spins(self, rscale=1.0, kscale=1.0, cmap='viridis', save=True,
                   ext='.pdf', show=True):
        """Plot the densities (real & k) and phases of spin components.

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

        """
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale

        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale

        extents = {'r': r_extent, 'k': k_extent}

        fig, all_plots = ptools.plot_spins(self.psi, self.psik, extents,
                                           self.paths, cmap=cmap, save=save,
                                           ext=ext, show=show)
        return fig, all_plots

    def plot_total(self, rscale=1.0, kscale=1.0, cmap='viridis', save=True,
                   ext='.pdf', show=True):
        """Plot the total real-space density and phase of the wavefunction.

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

        """
        # TODO: Move this to the ptools module and make it callable from here.
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale

        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale

        extents = {'r': r_extent, 'k': k_extent}

        dens_tot_r = sum(self.dens)
        ph_tot_r = ttools.phase(sum(self.psi), uwrap=True, dens=dens_tot_r)
        dens_tot_k = sum(self.densk)

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
                             extent=r_extent, vmin=0)
        fig.colorbar(r_plot, ax=r_ax)
        r_ax.set_xlabel('$x$')
        r_ax.set_ylabel('$y$')

        # Real-space phase plot
        ph_plot = ph_ax.imshow(ph_tot_r, cmap='twilight_shifted',
                               origin='lower', extent=r_extent,
                               vmin=-np.pi, vmax=np.pi)

        ph_cb = fig.colorbar(ph_plot, ax=ph_ax)
        ph_cb.set_ticks(np.linspace(-np.pi, np.pi, 5))
        ph_cb.set_ticklabels(['$-\\pi$', '', '$0$', '', '$\\pi$'])
        ph_ax.set_xlabel('$x$')
        ph_ax.set_ylabel('$y$')

        # Momentum-space density plot
        k_plot = k_ax.imshow(dens_tot_k, cmap=cmap, origin='lower',
                             extent=k_extent, vmin=0)
        fig.colorbar(k_plot, ax=k_ax)
        k_ax.set_xlabel('$k_x$')
        k_ax.set_ylabel('$k_y$')

        plt.tight_layout()

        # Save figure
        if save:
            test_name = self.paths['data'] + 'total_dens_phase'
            file_name = ptools.next_available_path(test_name,
                                                   self.paths['folder'], ext)
            plt.savefig(file_name)
        plt.show()

    # def plot_eng(self):
    #     """Plot the sampled energy expectation values."""

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
            Saved plot image file extension.
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

    def make_movie(self, rscale=1.0, kscale=1.0, cmap='viridis', play=False):
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
        """
        with np.load(self.sampled_path) as sampled:
            times = sampled['times']
            psiks = sampled['psiks']
            # ??? Need to rebin for speed?
        n_samples = len(times)
        writer = animation.writers['ffmpeg'](fps=5, bitrate=-1)
        fig, all_plots = self.plot_spins(rscale, kscale, cmap, save=False,
                                         show=False)

        def animate(frame, n_total):
            global timelast, timethis
            psik = psiks[frame]
            psi = ttools.ifft_2d(psik, self.space['dr'])

            dens = ttools.density(psi)
            phase = ttools.phase(psi, uwrap=True, dens=dens)
            densk = ttools.density(psik)

            any(plot.set_data(d) for plot, d in zip(all_plots['r'], dens))
            any(plot.set_data(ph) for plot, ph in zip(all_plots['ph'], phase))
            any(plot.set_data(dk) for plot, dk in zip(all_plots['k'], densk))

            ptools.progress_message(frame, n_total)

        # create and then save the animation
        anim = animation.FuncAnimation(fig, animate, frames=n_samples,
                                       blit=False, fargs=(n_samples,))

        # Save animation
        test_name = self.paths['data'] + 'prop_movie'
        file_name = ptools.next_available_path(test_name,
                                               self.paths['folder'], '.mp4')
        anim.save(file_name, writer=writer)
        plt.close(fig)

        if play:
            os.startfile(file_name)


def rebin(arr, new_shape=(256, 256)):
    """Rebin 2D array arr to shape new_shape by averaging."""
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
