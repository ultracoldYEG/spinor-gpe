"""Placeholder for the tensor_propagator.py module."""
import numpy as np
import torch
from tqdm import tqdm

from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor.plotting_tools import next_available_path
from spinor_gpe.pspinor import prop_result


class TensorPropagator:
    """CPU- or GPU-compatible propagator of the GPE, with tensors.

    Attributes
    ----------
    n_steps : :obj:`int`
        The total number of full time steps in propagation.
    device : :obj:`str`
        The computing device on which propagation is performed
    paths : :obj:`dict`
        See ``pspinor.Pspinor``.
    t_step : :obj:`float` or :obj:`complex`
        Duration of the full time step.
    dt_out : :obj:`float` or :obj:`complex`
        Duration of the outer time sub-step.
    dt_in : :obj:`float` or :obj:`complex`
        Duration of the inner time sub-step.
    rand_seed : :obj:`int`
        See ``pspinor.Pspinor``.
    is_sampling : :obj:`bool`
        Option to sample the wavefunction periodically throughout propagation.
    atom_num : :obj:`float`
        See ``pspinor.Pspinor``.
    is_coupling : :obj:`bool`
        See ``pspinor.Pspinor``.
    g_sc : :obj:`dict` of :obj:`Tensor`
        See `pspinor.Pspinor`.
    kin_eng_spin : :obj:`list` of :obj:`Tensor`
        See ``pspinor.Pspinor``.
    pot_eng_spin : :obj:`list` of :obj:`Tensor`
        See ``pspinor.Pspinor``.
    psik : :obj:`list` of :obj:`Tensor`
        See `pspinor.Pspinor`.
    space : :obj:`dict` of :obj:`Tensor`
        See `pspinor.Pspinor`. Contains only keys:
            {'dr', 'dk', 'x_mesh', 'y_mesh', 'dv_r', 'dv_k'}
    coupling : :obj:`Tensor`
        See `pspinor.Pspinor`.
    kL_recoil : :obj:`float`
        See ``pspinor.Pspinor``.
    expon : :obj:`Tensor`
        The exponential argument on the coupling operator off-diagonals. If
        the coupling is in a rotated reference frame, then `expon`=0.0.
    sample_rate : :obj:`int`
        How often wavefunctions are sampled.
    eng_out : :obj:`dict` of :obj:`Tensor`
        Pre-computed energy evolution operators for the outer time sub-step.
    eng_in : :obj:`dict` of :obj:`Tensor`
        Pre-computed energy evolution operators for the inner time sub-step.

    """

    # Ideally it would be great to keep this class agnostic as to the
    # wavefunction structure, i.e. pseudospinors vs. scalars vs. spin-1.

    # pylint: disable=too-many-instance-attributes
    def __init__(self, spin, t_step, n_steps, device='cpu', time='imag',
                 is_sampling=False, n_samples=1):
        """Begin a propagation loop.

        Parameters
        ----------
        spin :  :obj:`PSpinor`
            The energy and spatial grids are taken from this object and
            converted to PyTorch :obj:`Tensor` objects.
        t_step : :obj:`float`
            Propagation time step.
        n_steps : :obj:`int`
            The number of steps to propagate in time.
        device : :obj:`str`, default='cpu'
            The device on which to compute: {'cpu', 'cuda'}.
        time : :obj:`str`, optional
            Whether propagation occurs in real or imaginary time:
            {'real', 'imag'}.
        is_sampling : :obj:`bool`, default=False
            Option to sample and save wavefunctions throughout the propagation.
        n_samples : :obj:`int`, default=1
            The number of samples to save.

        """
        self.n_steps = n_steps
        self.device = device
        self.paths = spin.paths

        # Calculate the time step intervals
        if time == 'imag':
            self.t_step = -1.0j * t_step
        elif time == 'real':
            self.t_step = t_step

        magic_gamma = 1 / (2 + 2**(1 / 3))
        self.dt_out = self.t_step * magic_gamma
        self.dt_in = self.t_step * (1 - 2 * magic_gamma)

        self.rand_seed = spin.rand_seed
        if self.rand_seed is not None:
            torch.manual_seed(self.rand_seed)
        self.is_sampling = is_sampling

        # Load in data from PSpinor object as tensors
        self.atom_num = spin.atom_num
        self.is_coupling = spin.is_coupling
        self.g_sc = spin.g_sc
        self.kin_eng_spin = ttools.to_tensor(spin.kin_eng_spin,
                                             dev=self.device)
        self.pot_eng_spin = ttools.to_tensor(spin.pot_eng_spin,
                                             dev=self.device)
        self.psik = ttools.to_tensor(spin.psik, dev=self.device, dtype=128)
        keys_space = ['dr', 'dk', 'x_mesh', 'y_mesh', 'dv_r', 'dv_k']
        self.space = {k: torch.tensor(spin.space[k], device=self.device)
                      for k in keys_space}
        self.coupling = ttools.to_tensor(spin.coupling, dev=self.device)

        # pylint: disable=invalid-name
        self.kL_recoil = spin.kL_recoil
        if spin.rot_coupling:
            self.expon = torch.tensor(0.0)
        else:
            self.expon = 2 * self.kL_recoil * self.space['x_mesh']
        # Calculate the sampling and annealing rates, as needed.
        if self.is_sampling:
            assert self.n_steps % n_samples == 0, (
                f"The number of samples requested {n_samples} does not evenly "
                f"divide the total number of steps {self.n_steps}.")

        self.sample_rate = self.n_steps / n_samples
        # Pre-compute several evolution operators
        self.eng_out = {'kin': ttools.evolution_op(self.dt_out / 2,
                                                   self.kin_eng_spin),
                        'pot': ttools.evolution_op(self.dt_out,
                                                   self.pot_eng_spin),
                        'coupl': ttools.coupling_op(self.dt_out,
                                                    self.coupling / 2, self.expon)}
        self.eng_in = {'kin': ttools.evolution_op(self.dt_in / 2,
                                                  self.kin_eng_spin),
                       'pot': ttools.evolution_op(self.dt_in,
                                                  self.pot_eng_spin),
                       'coupl': ttools.coupling_op(self.dt_in / 2,
                                                   self.coupling, self.expon)}

    def prop_loop(self, n_steps):
        """Evaluate the propagation steps in a for-loop.

        Saves the spin populations at every time step. If wavefunctions are
        sampled throughout the propagation, they are saved with the associated
        sampled times in `trial_data/psik_sampled%s_`folder_name`.npz.

        Parameters
        ----------
        n_steps : :obj:`int`
            The number of propagation steps.

        Returns
        -------
        result : :obj:`PropResult`
            Contains the propagation results and analysis methods.

        See Also
        --------
        spinor_gpe.prop_results : Propagation results

        """
        pop_times = np.linspace(0, self.n_steps * np.abs(self.t_step), n_steps)
        pops = {'times': pop_times, 'vals': np.empty((n_steps, 2))}

        # Pre-allocate arrays for efficient sampling.
        if self.is_sampling:
            n_samples = int(n_steps / self.sample_rate)
            sampled_psik = np.empty((n_samples, 2, *self.psik[0].shape),
                                    dtype=np.complex128)
            sampled_times = np.linspace(0, self.n_steps * np.abs(self.t_step),
                                        n_samples)

        # Main propagation loop
        for _i in tqdm(range(n_steps)):
            if self.is_sampling:
                if _i % self.sample_rate == 0:
                    idx = int(_i / self.sample_rate)
                    sampled_psik[idx] = np.array(ttools.to_numpy(self.psik))

            self.full_step()

            # Calculate and store populations
            pops['vals'][_i] = ttools.calc_pops(self.psik, self.space['dv_k'])

        energy = self.eng_expect(self.psik)

        if self.is_sampling:
            # Saves sampled wavefunctions to file; times are in dimensionless
            # time units
            test_name = self.paths['trial'] + 'psik_sampled'
            file_name = next_available_path(test_name,
                                            self.paths['folder'], '.npz')
            np.savez(file_name, psiks=sampled_psik, times=sampled_times)
        else:
            file_name = None

        psik = ttools.to_numpy(self.psik)
        psi = ttools.ifft_2d(psik, ttools.to_numpy(self.space['dr']))

        result = prop_result.PropResult(psi, psik, energy, pops, file_name)
        return result

    def full_step(self):
        """Full step forward in real or imaginary time.

        For accuracy, divide the full propagation step into three single steps
        using the magic gamma time steps.
        """
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step
        self.single_step(self.dt_in, self.eng_in)  # Inner sub-time step
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step

    def single_step(self, t_step, eng):
        """Single step forward in real or imaginary time with spectral method.

        The kinetic, interaction, and coupling time-evolution operators are
        symmetrically split into two half-single steps around the full-single
        step potential energy operator.

        Parameters
        ----------
        t_step : :obj:`float`
            The sub-time step.
        eng : :obj:`dict`
            The kinetic and potential energy evolution operators corresponding
            to the given sub-time step.

        """
        # First half step of the kinetic energy operator
        # psik = self.psik
        psik = [eng * pk for eng, pk in zip(eng['kin'], self.psik)]
        psi = ttools.ifft_2d(psik, delta_r=self.space['dr'])
        psi, dens = ttools.norm(psi, self.space['dv_r'], self.atom_num)

        # First half step of the interaction energy operator
        int_eng = [self.g_sc['uu'] * dens[0] + self.g_sc['ud'] * dens[1],
                   self.g_sc['dd'] * dens[1] + self.g_sc['ud'] * dens[0]]
        int_op = ttools.evolution_op(t_step / 2, int_eng)
        psi = [op * p for op, p in zip(int_op, psi)]
        # First half step of the coupling energy operator
        if self.is_coupling:
            psi = [sum([elem * p for elem, p in zip(row, psi)])
                   for row in eng['coupl']]
        # Full step of the potential energy operator
        psi = [eng * p for eng, p in zip(eng['pot'], psi)]
        # Second half step of the coupling energy operator
        if self.is_coupling:
            psi = [sum([elem * p for elem, p in zip(row, psi)])
                   for row in eng['coupl']]
        # Second half step of the interaction energy operator
        # ??? Is renormalization needed? It's not in previous code versions.
        # psi, dens = ttools.norm(psi, self.space['dv_r'], self.atom_num)
        # int_eng = [self.g_sc['uu'] * dens[0] + self.g_sc['ud'] * dens[1],
        #            self.g_sc['dd'] * dens[1] + self.g_sc['ud'] * dens[0]]
        # int_op = ttools.evolution_op(t_step / 2, int_eng)
        psi = [op * p for op, p in zip(int_op, psi)]
        # Second half step of the kintetic energy operator
        psik = ttools.fft_2d(psi, delta_r=self.space['dr'])
        psik = [eng * pk for eng, pk in zip(eng['kin'], psik)]
        self.psik, _ = ttools.norm(psik, self.space['dv_k'], self.atom_num)

    def eng_expect(self, psik):
        """Compute the energy expectation value of the wavefunction.

        To calculate the kinetic energy potion of the energy expectation value,
        spatial gradients of the phase and square root density must be
        obtained.

        Parameters
        ----------
        psik : :obj:`list` of NumPy :obj:`array`
            The k-space representation of wavefunction to evaluate.

        Notes
        -----
        While spatial gradients of the wavefunction's phase can be computed
        with PyTorch tensors, there is currently not an implementation of the
        2D phase-unwrapping algorithm. For this this reason, the energy
        expectation value needs to be computed with NumPy :obj:`arrays`.

        """
        assert len(psik) == 2, ("Requires two spinor components to calculate "
                                "the energy expectation value.")
        # FIXME: I don't have a lot of confidence in the values generated.
        if not isinstance(psik[0], np.ndarray):
            psik = ttools.to_numpy(psik)
        delta_r = ttools.to_numpy(self.space['dr'])

        psi = ttools.ifft_2d(psik, delta_r)
        dens = ttools.density(psi)
        dens_sqrt = [np.sqrt(d) for d in dens]

        phase = ttools.phase(psi, uwrap=True, dens=dens)
        phase_gradx = [ph_x for ph_x, _ in ttools.grad(phase, delta_r)]
        phase_gradsq = ttools.grad_sq(phase, delta_r)

        kin = (sum(ttools.grad_sq(dens_sqrt, delta_r))
               + sum([d * p for d, p in zip(dens, phase_gradsq)])
               + sum([d * p for d, p in zip(dens, phase_gradx)])
               * 2 * self.kL_recoil * self.is_coupling) / 2

        pot = sum([d * pot for d, pot in
                   zip(dens, ttools.to_numpy(self.pot_eng_spin))])

        int_e = (self.g_sc['uu'] * dens[0]**2 + self.g_sc['dd'] * dens[1]**2
                 + self.g_sc['ud'] * dens[0] * dens[1])

        coupl_e = ((np.conj(psi[0]) * psi[1] + np.conj(psi[1]) * psi[0])
                   * ttools.to_numpy(self.coupling) / 2)

        total_eng = np.real((kin + pot + int_e + coupl_e).sum())
        return [total_eng, np.real(kin).sum(), np.real(pot).sum(),
                np.real(int_e).sum()]
