"""Placeholder for the tensor_propagator.py module."""


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
    def __init__(self, spin, t_step, n_steps, device='cpu', **kwargs):
        """Begin a propagation.

        Parameters
        ----------
        spin :
        t_step :
        n_steps :
        device :

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
        #   - grid parameters -> tensor [maybe ok as dict of tensors]
        #     - delta_r
        #     - delta_k
        #     - x_mesh
        #     - y_mesh
        #   - volume elements
        #
        #  imaginary/real:
        #   + Number of steps
        #   + dt
        #   + is_sampling (bool)
        #   + wavefunction sample frequency
        #   + wavefunction anneal frequency (imaginary time)
        #   + device (cpu vs. gpu)

        self.t_step = t_step
        self.n_steps = n_steps
        self.device = device

        self.is_sampling = kwargs.get('is_sampling', False)
        self.is_annealing = kwargs.get('is_sampling', False)
        n_samples = kwargs.get('n_samples', 1)
        n_anneals = kwargs.get('n_anneals', 1)

        self.is_coupling = spin.is_coupling
        self.g_sc = spin.g_sc
        self.kin_eng = spin.kin_eng_spin
        self.pot_eng = spin.pot_eng_spin
        if self.is_coupling:
            self.coupling = spin.coupling
        else:
            self.coupling = None

        if (self.is_sampling and n_samples == 1):
            n_samples = 100
        assert self.n_steps % n_samples == 0, ("The number of samples "
                                               f"requested {n_samples} does "
                                               "not evenly divide the total "
                                               "number of steps "
                                               f"{self.n_steps}.")
        self.sample_rate = self.n_steps / n_samples

        assert self.n_steps % n_anneals == 0, ("The number of samples "
                                               f"requested {n_samples} does "
                                               "not evenly divide the total "
                                               "number of steps "
                                               f"{self.n_steps}.")
        self.anneal_rate = self.n_steps / n_anneals

    def evolution_op(self):
        """Compute the time-evolution operator for a given energy term."""

    def coupling_op(self):
        """Compute the time-evolution operator for the coupling term.

        - May not be needed :)
        """

    def single_step(self):
        """Single step forward in real or imaginary time."""

    def full_step(self):
        """Full step forward in real or imaginary time.

        Divide the full propagation step into three single steps using
        the magic gamma for accuracy.
        """
        self.single_step()
        self.single_step()
        self.single_step()

    def propagation(self, n_steps):
        """Contains the actual propagation for-loop."""
        for _i in range(n_steps):
            self.full_step()

    def eng_expect(self):
        """Compute the energy expectation value."""

    def expect_val(self):
        """Compute the expectation value of the supplied spatial operator."""
