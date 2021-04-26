"""Placeholder for the tensor_propagator.py module."""
import torch
from tqdm import tqdm

from pspinor import tensor_tools as ttools


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

        # Calculate the time step intervals
        if time == 'imag':
            self.t_step = -1.0j * t_step
        elif time == 'real':
            self.t_step = t_step

        magic_gamma = 1/(2 + 2**(1/3))
        magic_gamma_diff = 1 - 2*magic_gamma
        self.dt_out = self.t_step * magic_gamma
        self.dt_in = self.t_step * magic_gamma_diff

        self.is_sampling = kwargs.get('is_sampling', False)
        self.is_annealing = kwargs.get('is_sampling', False)
        n_samples = kwargs.get('n_samples', 1)
        n_anneals = kwargs.get('n_anneals', 1)

        # Load in data from PSpinor object as tensors
        self.atom_num = spin.atom_num
        self.is_coupling = spin.is_coupling
        self.g_sc = spin.g_sc
        kin_eng = ttools.to_tensor(spin.kin_eng_spin, dev=self.device)
        pot_eng = ttools.to_tensor(spin.pot_eng_spin, dev=self.device)
        if self.is_coupling:
            self.coupling = ttools.to_tensor(spin.coupling, dev=self.device)
        else:
            self.coupling = None
        # self.psi = ttools.to_tensor(spin.psi, dev=self.device, dtype=128)
        self.psik = ttools.to_tensor(spin.psik, dev=self.device, dtype=128)

        keys_space = ['dr', 'dk', 'x_mesh', 'y_mesh', 'dv_r', 'dv_k']
        self.space = {k: torch.tensor(spin.space[k], device=self.device)
                      for k in keys_space}

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
                        'pot': ttools.evolution_op(pot_eng, self.dt_out)}
        self.eng_in = {'kin': ttools.evolution_op(kin_eng, self.dt_in / 2),
                       'pot': ttools.evolution_op(pot_eng, self.dt_in)}

    def apply_coupling_op(self, psi, shifted=False):
        """Apply the time-evolution operator for the coupling matrix."""

    def coupling_op(self, coupling):
        """Compute the time-evolution operator for the coupling term.

        - May not be needed :)
        """

    def single_step(self, t_step, eng):
        """Single step forward in real or imaginary time."""
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
            psi = self.apply_coupling_op(psi, shifted=False)

        # Full step of the potential energy operator
        psi = [eng * p for eng, p in zip(eng['pot'], psi)]

        # Second half step of the coupling energy operator
        if self.is_coupling:
            psi = self.apply_coupling_op(psi, shifted=False)

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
        self.psik, densk = ttools.norm(psik, self.space['dv_k'], self.atom_num)

    def full_step(self):
        """Full step forward in real or imaginary time.

        Divide the full propagation step into three single steps using
        the magic gamma for accuracy.
        """
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step
        self.single_step(self.dt_in, self.eng_in)  # Inner sub-time step
        self.single_step(self.dt_out, self.eng_out)  # Outer sub-time step

    def prop_loop(self, n_steps):
        """Contains the actual propagation for-loop."""
        for _i in tqdm(range(n_steps)):
            self.full_step()
            atom_num = ttools.calc_atoms(self.psik, self.space['dv_k'])

    def run_prop(self):
        """Start and processes a propagation cycle."""

    def eng_expect(self):
        """Compute the energy expectation value."""

    def expect_val(self):
        """Compute the expectation value of the supplied spatial operator."""
