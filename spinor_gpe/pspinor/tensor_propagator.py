"""Placeholder for the tensor_propagator.py module."""
import numpy as np
import torch
from tqdm import tqdm

from spinor_gpe.pspinor import tensor_tools as ttools
from spinor_gpe.pspinor import prop_result
from skimage.restoration import unwrap_phase


class TensorPropagator:
    """CPU- or GPU-compatible propagator of the GPE, with tensors.

    Attributes
    ----------
    t_step :

    """

    # Object that sucks in the needed energy grids and parameters for
    # propagation, converts them to tensors, & performs the propagation.
    #  - It means that two copies of the grids aren't carried in the main class
    #  - However, it restricts access to the tensor form of the grids; unless
    #    I keep the Propagation object as a class "attribute".
    #  - I can directly pass `self` to this class and access class attributes,
    #    methods, esp. energy grids. Only do this in the __init__ function
    #    so as to not store the main Spinor object in the class

    # --> Should it be a class or just a pure function??
    #      - Maybe a class because then it can store the grids it needs, and
    #        then access them from the different functions for free.
    #      - It would allow these operations to reside in a separate module.
    # BUT, then I have two classes who are attributes of each other, and
    #     THAT's a weird structure.
    #    - Maybe this class doesn't have to attribute the other one; it just
    #      sucks in the data it needs.

    # Will need to calculate certain data throughout the loop, and then
    #     create and populate the PropResult object.

    # Ideally it would be great to keep this class agnostic as to the
    # wavefunction structure, i.e. pseudospinors vs. scalars vs. spin-1.

    # pylint: disable=too-many-instance-attributes
    def __init__(self, spin, t_step, n_steps, device='cpu', time='imag',
                 **kwargs):
        """Begin a propagation.

        Parameters
        ----------
        spin :
        t_step :
        n_steps :
        device :
        time :

        Other Parameters
        ----------------
        is_sampling :
        n_samples :
        is_annealing :
        n_anneals :

        """
        # Needs:
        #  spin
        #   - Energy grids -> tensor
        #     - Kinetic spinor
        #     - Potential + detuning spinor
        #   - Raman grids -> tensor
        #     - coupling
        #   - Interaction parameters -> dict

        #   - Atom number
        #   - space parameters -> tensor [maybe ok as dict of tensors]
        #     - delta_r
        #     - delta_k
        #     - x_mesh
        #     - y_mesh
        #     - volume elements
        #   - psi
        #   - psik
        #
        #  imaginary/real:
        #   + Number of steps
        #   + dt
        #   + is_sampling (bool)
        #   + wavefunction sample frequency
        #   + wavefunction anneal frequency (imaginary time)
        #   + device (cpu vs. gpu)

        self.n_steps = n_steps
        self.device = device
        self.paths = spin.paths

        # Calculate the time step intervals
        if time == 'imag':
            self.t_step = -1.0j * t_step
        elif time == 'real':
            self.t_step = t_step

        magic_gamma = 1/(2 + 2**(1/3))
        magic_gamma_diff = 1 - 2*magic_gamma
        self.dt_out = self.t_step * magic_gamma
        self.dt_in = self.t_step * magic_gamma_diff

        self.rand_seed = kwargs.get('rand_seed', None)
        if self.rand_seed is not None:
            torch.manual_seed(self.rand_seed)
        self.is_sampling = kwargs.get('is_sampling', False)
        self.is_annealing = kwargs.get('is_sampling', False)
        n_samples = kwargs.get('n_samples', 1)
        n_anneals = kwargs.get('n_anneals', 1)

        # Load in data from PSpinor object as tensors
        self.atom_num = spin.atom_num
        self.is_coupling = spin.is_coupling
        self.g_sc = spin.g_sc
        kin_eng = ttools.to_tensor(spin.kin_eng_spin, dev=self.device)
        self.pot_eng = ttools.to_tensor(spin.pot_eng_spin, dev=self.device)

        # self.psi = ttools.to_tensor(spin.psi, dev=self.device, dtype=128)
        self.psik = ttools.to_tensor(spin.psik, dev=self.device, dtype=128)

        keys_space = ['dr', 'dk', 'x_mesh', 'y_mesh', 'dv_r', 'dv_k']
        self.space = {k: torch.tensor(spin.space[k], device=self.device)
                      for k in keys_space}

        if self.is_coupling:
            self.coupling = ttools.to_tensor(spin.coupling, dev=self.device)
            # pylint: disable=invalid-name
            self.kL_recoil = spin.kL_recoil
            expon = 2 * self.kL_recoil * self.space['x_mesh']
        else:
            self.coupling = None
            self.kL_recoil = 0

        # Calculate the sampling and annealing rates, as needed.
        if self.is_sampling:
            if n_samples == 1:
                n_samples = 100
            assert self.n_steps % n_samples == 0, (
                f"The number of samples requested {n_samples} does not evenly "
                f"divide the total number of steps {self.n_steps}.")

        if self.is_annealing:
            assert self.n_steps % n_anneals == 0, (
                f"The number of annealings requested {n_anneals} does not "
                f"evenly divide the total number of steps {self.n_steps}.")

        self.anneal_rate = self.n_steps / n_anneals
        self.sample_rate = self.n_steps / n_samples

        # Pre-compute several evolution operators
        self.eng_out = {'kin': ttools.evolution_op(kin_eng, self.dt_out / 2),
                        'pot': ttools.evolution_op(self.pot_eng, self.dt_out),
                        'coupl': ttools.coupling_op(self.dt_out / 2,
                                                    self.coupling, expon)}
        self.eng_in = {'kin': ttools.evolution_op(kin_eng, self.dt_in / 2),
                       'pot': ttools.evolution_op(self.pot_eng, self.dt_in),
                       'coupl': ttools.coupling_op(self.dt_in / 2,
                                                   self.coupling, expon)}

    def single_step(self, t_step, eng):
        """Single step forward in real or imaginary time.

        Parameters
        ----------
        t_step : :obj:`float`
            The sub-time step.
        eng : :obj:`dict`
            The kinetic and potential energy evolution operators corresponding
            to the given sub-time step.

        """
        # First half step of the kinetic energy operator
        psik = [eng * pk for eng, pk in zip(eng['kin'], self.psik)]
        psi = ttools.ifft_2d(psik, delta_r=self.space['dr'])
        psi, dens = ttools.norm(psi, self.space['dv_r'], self.atom_num)

        # First half step of the interaction energy operator
        int_eng = [self.g_sc['uu'] * dens[0] + self.g_sc['ud'] * dens[1],
                   self.g_sc['dd'] * dens[1] + self.g_sc['ud'] * dens[0]]
        int_op = ttools.evolution_op(int_eng, t_step / 2)
        psi = [op * p for op, p in zip(int_op, psi)]

        # First half step of the coupling energy operator
        if self.is_coupling:
            psi = [sum([r * p for r, p in zip(row, psi)])
                   for row in eng['coupl']]

        # Full step of the potential energy operator
        psi = [eng * p for eng, p in zip(eng['pot'], psi)]

        # Second half step of the coupling energy operator
        if self.is_coupling:
            psi = [sum([r * p for r, p in zip(row, psi)])
                   for row in eng['coupl']]

        # Second half step of the interaction energy operator
        # ??? Is renormalization needed? We don't have it in the previous code.
        # psi, dens = ttools.norm(psi, self.space['dv_r'], self.atom_num)
        # int_eng = [self.g_sc['uu'] * dens[0] + self.g_sc['ud'] * dens[1],
        #            self.g_sc['dd'] * dens[1] + self.g_sc['ud'] * dens[0]]
        # int_op = ttools.evolution_op(int_eng, t_step / 2)
        psi = [op * p for op, p in zip(int_op, psi)]

        # Second half step of the kintetic energy operator
        psik = ttools.fft_2d(psi, delta_r=self.space['dr'])
        psik = [eng * pk for eng, pk in zip(eng['kin'], self.psik)]
        self.psik, _ = ttools.norm(psik, self.space['dv_k'], self.atom_num)

    def full_step(self):
        """Full step forward in real or imaginary time.

        Divide the full propagation step into three single steps using
        the magic gamma for accuracy.
        """
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step
        self.single_step(self.dt_in, self.eng_in)  # Inner sub-time step
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step

    def prop_loop(self, n_steps):
        """Contains the actual propagation for-loop.

        Parameters
        ----------
        n_steps : :obj:`int`
            The number of propagation steps.

        """

        # Every sample_rate steps, need to save a copy of the wavefunction to a
        #    Numpy array in memory.
        # Every anneal_rate steps, need to
        n_samples = int(n_steps / self.sample_rate)
        sampled_psik = np.empty((n_samples, 2, *self.psik[0].shape),
                                dtype=np.complex128())
        pops = np.empty((n_steps, 2))
        # if self.is_annealing:
        #     best_config = {'psik': self.psik,
        #                    'eng': self.eng_expect(self.psik)}

        for _i in tqdm(range(n_steps)):
            if (self.is_sampling and (_i % self.sample_rate) == 0):
                # sampled_psik.append(ttools.to_numpy(self.psik))
                idx = int(_i / self.sample_rate)
                sampled_psik[idx] = np.array(ttools.to_numpy(self.psik))
                continue

            self.full_step()

            # if (_i % self.anneal_rate) == 0:  # Measure the energy & anneal
            #     eng = self.eng_expect(self.psik)
            #     if self.is_annealing:
            #         if eng < best_config['eng']:
            #             best_config.update({'psik': self.psik, 'eng': eng})
            #         else:
            #             self.psik = best_config['psik']
            #         # ADD: Annealing stuff here.
            #     # Save energy value(s)

            # TODO: For some reason the first element is not saved properly
            pops[_i] = ttools.calc_pops(self.psik, self.space['dv_k'])
            # Save population values, pre-allocate these ones too.

        energy = self.eng_expect(self.psik)
        print('\n', energy)
        sampled_times = np.linspace(0, self.n_steps * np.abs(self.t_step),
                                    n_samples)

        # Save sampled wavefunctions; times are in dimensionless time units
        sampled_psik_path = (self.paths['trial'] + 'psik_sampled-'
                             + self.paths['folder'] + '.npy')
        sampled_times_path = (self.paths['trial'] + 'times_sampled-'
                              + self.paths['folder'] + '.npy')
        print(pops)
        np.save(sampled_psik_path, sampled_psik)
        np.save(sampled_times_path, sampled_times)
        return prop_result.PropResult(self.psik)

    def eng_expect(self, psik):
        """Compute the energy expectation value.

        Needs to be computed on the fly. Energies are computed with NumPy
        on the CPU.
        """
        assert len(psik) == 2, ("Requires two spinor components to calculate "
                                "the energy expectation value.")
        # TODO: Debug this later when running a Raman configuration.
        if not isinstance(psik[0], np.ndarray):
            psik = ttools.to_numpy(psik)
        delta_r = ttools.to_numpy(self.space['dr'])

        psi = ttools.ifft_2d(psik, delta_r)
        dens = ttools.density(psi)
        dens_sqrt = [np.sqrt(d) for d in dens]

        phase = ttools.phase(psi)
        phase = [unwrap_phase(phz) for phz in phase]
        phase_gradx = [ph_x for ph_x, _ in ttools.grad(phase, delta_r)]
        phase_gradsq = ttools.grad_sq(phase, delta_r)

        kin = (sum(ttools.grad_sq(dens_sqrt, delta_r))
               + sum([d * p for d, p in zip(dens, phase_gradsq)])
               + sum([d * p for d, p in zip(dens, phase_gradx)])
               * 2 * self.kL_recoil) / 2

        pot = sum([d * pot for d, pot in
                   zip(dens, ttools.to_numpy(self.pot_eng))])

        int_e = (self.g_sc['uu'] * dens[0]**2 + self.g_sc['dd'] * dens[1]**2
                 + self.g_sc['ud'] * dens[0] * dens[1])

        if self.is_coupling:
            coupl_e = ((np.conj(psi[0]) * psi[1] + np.conj(psi[1]) * psi[0])
                       * ttools.to_numpy(self.coupling) / 2)
        else:
            coupl_e = 0

        total_eng = np.real((kin + pot + int_e + coupl_e).sum())
        print(total_eng)
        return total_eng
