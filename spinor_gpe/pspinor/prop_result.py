"""Placeholder for the prop_result.py module."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        self.paths = dict()
        self.r_scale = None
        self.k_scale = None
        self.t_scale = None
        self.space = dict()

    def calc_separation(self):
        """Calculate the phase separation of the two spin components."""
        s = 1 - np.sum(ttools.prod(self.dens))  # FIXME
        return s

    def plot_spins(self):
        """Plot the densities (real & k) and phases of spin components."""

    def plot_total(self, rscale=1.0, kscale=1.0, cmap='viridis'):
        """Plot the total real-space density and phase of the wavefunction."""
        dens_tot_r = sum(self.dens)
        ph_tot_r = ttools.phase(sum(self.psi), uwrap=True)
        dens_tot_k = sum(self.densk)
        widths = [1, 1, 1, 1]
        heights = [1, 1, 1, 1]
        fig = plt.figure()
        gsp = gridspec.GridSpec(4, 4, width_ratios=widths,
                                height_ratios=heights)

        r_ax = fig.add_subplot(gsp[0:2, 0:2])
        ph_ax = fig.add_subplot(gsp[0:2, 2:])
        k_ax = fig.add_subplot(gsp[2:, 1:3])
        # fig.add_subplot(r_plot, ph_plot, k_plot)

        # Real-space density plot
        r_sizes = self.space['r_sizes']
        r_extent = np.ravel(np.vstack((-r_sizes, r_sizes)).T) / rscale
        r_plot = r_ax.imshow(dens_tot_r, cmap=cmap, origin='lower',
                             extent=r_extent, vmin=0)
        fig.colorbar(r_plot, ax=r_ax)
        # r_ax.set_title('Total real-space density')
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
        # ph_divider = make_axes_locatable(ph_plot)
        # ph_cb = ph_divider.append_axes('right', '5%', pad=0.1)
        # ph_cb_ax = fig.colorbar(ph_plot, cax=ph_cb, orientation='vertical',
        #                         format='%.0e')
        # ph_ax.set_title('Total phase')

        # Momentum-space density plot
        k_sizes = self.space['k_sizes']
        k_extent = np.ravel(np.vstack((-k_sizes, k_sizes)).T) / kscale
        k_plot = k_ax.imshow(dens_tot_k, cmap=cmap, origin='lower',
                             extent=k_extent, vmin=0)
        fig.colorbar(k_plot, ax=k_ax)
        k_ax.set_xlabel('$k_x$')
        k_ax.set_ylabel('$k_y$')
        # k_ax.set_title('Total k-space density')
        plt.tight_layout()
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
