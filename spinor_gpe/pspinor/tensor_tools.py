"""tensor_tools.py module."""
import numpy as np
import torch

# ??? How should the individual FFT operations be normalized? Should they
# remain as norm="backward", or, because of the nature of our operations,
# changed to norm="ortho"?


def fft_1d(psi, delta_r=(1, 1), axis=0) -> list:
    """Compute the forward 1D FFT of `psi` along a single axis.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input wavefunction.
    delta_r : Numpy :obj:`array`
        A two-element list of the x- and y-mesh spacings, respectively.
    axis : :obj:`int`, optional
        The axis along which to transform; note that 0 -> y-axis, and
        1 -> x-axis.

    Returns
    -------
    psik_axis : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The FFT of psi along `axis`.

    """
    true_ax = np.array([1, 0])
    normalization = delta_r[axis] / np.sqrt(2 * np.pi)

    if isinstance(psi[0], np.ndarray):
        psik_axis = [np.fft.fftn(p, axes=[true_ax[axis]]) * normalization
                     for p in psi]
        psik_axis = [np.fft.fftshift(pk, axes=true_ax[axis])
                     for pk in psik_axis]
    elif isinstance(psi[0], torch.Tensor):
        psik_axis = [torch.fft.fftn(p, dim=[true_ax[axis]]) * normalization
                     for p in psi]
        psik_axis = [torch.fft.fftshift(pk, dim=true_ax[axis])
                     for pk in psik_axis]

    return psik_axis


def ifft_1d(psik, delta_r=(1, 1), axis=0) -> list:
    """Compute the inverse 1D FFT of `psi` along a single axis.

    Parameters
    ----------
    psik : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input wavefunction.
    delta_r : Numpy :obj:`array`
        A two-element list of the x- and y-mesh spacings, respectively.
    axis : :obj:`int`, optional
        The axis along which to transform; note that 0 -> x-axis, and
        1 -> y-axis.

    Returns
    -------
    psi_axis : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The FFT of psi along `axis`.

    """
    true_ax = np.array([1, 0])  # Makes the x/y axes correspond to 0/1
    normalization = delta_r[axis] / np.sqrt(2 * np.pi)
    if isinstance(psik[0], np.ndarray):
        psi_axis = [np.fft.ifftshift(pk, axes=true_ax[axis]) for pk in psik]
        psi_axis = [np.fft.ifftn(p, axes=[true_ax[axis]]) / normalization
                    for p in psi_axis]
    elif isinstance(psik[0], torch.Tensor):
        psi_axis = [torch.fft.ifftshift(pk, dim=true_ax[axis]) for pk in psik]
        psi_axis = [torch.fft.ifftn(p, dim=true_ax[axis]) / normalization
                    for p in psi_axis]

    return psi_axis


def fft_2d(psi, delta_r=(1, 1)) -> list:
    """Compute the forward 2D FFT of `psi`.

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input wavefunction.
    delta_r : Numpy :obj:`array`
        A two-element list of the x- and y-mesh spacings, respectively.

    Returns
    -------
    psik : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The k-space FFT of the input wavefunction.

    """
    normalization = np.prod(delta_r) / (2 * np.pi)  #: FFT normalization factor

    if isinstance(psi[0], np.ndarray):
        psik = [np.fft.fftn(p) * normalization for p in psi]
        psik = [np.fft.fftshift(pk) for pk in psik]

    elif isinstance(psi[0], torch.Tensor):
        psik = [torch.fft.fftn(p) * normalization for p in psi]
        psik = [torch.fft.fftshift(pk) for pk in psik]

    return psik


def ifft_2d(psik, delta_r=(1, 1)) -> list:
    """Compute the inverse 2D FFT of `psik`.

    Parameters
    ----------
    psik : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input wavefunction.
    delta_r : Numpy :obj:`array`
        A two-element list of the x- and y-mesh spacings, respectively.

    Returns
    -------
    psi : :obj:`list` of Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The real-space FFT of the input wavefunction.

    """
    normalization = np.prod(delta_r) / (2 * np.pi)  #: FFT normalization factor

    if isinstance(psik[0], np.ndarray):
        psik = [np.fft.ifftshift(pk) for pk in psik]
        psi = [np.fft.ifftn(p) / normalization for p in psik]

    elif isinstance(psik[0], torch.Tensor):
        psik = [torch.fft.ifftshift(pk) for pk in psik]
        psi = [torch.fft.ifftn(p) / normalization for p in psik]

    return psi


def to_numpy(input_tens):
    """Convert from PyTorch Tensor to Nunmpy arrays.

    Accepts a single PyTorch Tensor, or a :obj:`list` of PyTorch Tensor,
    as in the wavefunction objects.

    Parameters
    ----------
    input_tens : PyTorch :obj:`Tensor`, or :obj:`list` of PyTorch :obj:`Tensor`
        Input tensor, or list of tensor, to be converted to :obj:`array`,
        on CPU memory.

    Returns
    -------
    output_arr : Numpy :obj:`array` or :obj:`list` of Numpy :obj:`array`
        Output array stored on CPU memory.
    """
    if isinstance(input_tens, list):
        output_tens = [inp.cpu() for inp in input_tens]

    elif isinstance(input_tens, torch.Tensor):
        output_tens = input_tens.cpu().numpy()

    return output_tens


def to_tensor(input_arr, dev='cpu', dtype=64):
    """Convert from Numpy arrays to Tensors.

    Accepts a single Numpy array, or a :obj:`list` of Numpy arrays, as in the
    wavefunction objects.

    Parameters
    ----------
    input_arr : Numpy :obj:`array`,  or :obj:`list` of Numpy :obj:`array`
        Input array, or list of arrays, to be converted to a :obj:`Tensor`,
        on either CPU or GPU memory.
    dev : :obj:`str`, optional
        The name of the input device, e.g. {'cpu', 'cuda', 'cuda:0'}
    dtype : :obj:`int`, optional
        Designator for the torch dtype -
        32  : :obj:`float32`;
        64  : :obj:`float64`;
        128 : :obj:`complex128`

    Returns
    -------
    output_tens : PyTorch :obj:`Tensor` or :obj:`list` of PyTorch :obj:`Tensor`
        Output tensor of `dtype` stored on `dev` memory.
    """
    all_dtypes = {32: torch.float32, 64: torch.float64, 128: torch.complex128}
    if isinstance(input_arr, list):
        output_tens = [torch.as_tensor(inp, dtype=all_dtypes[dtype],
                                       device=dev) for inp in input_arr]

    elif isinstance(input_arr, np.ndarray):
        output_tens = torch.as_tensor(input_arr, dtype=all_dtypes[dtype],
                                      device=dev)

    return output_tens


def to_cpu(input_tens):
    """Transfers `input_tens` from gpu to cpu memory.

    Parameters
    ----------
    input_tens : PyTorch :obj:`Tensor` or :obj:`list` of PyTorch :obj:`Tensor`
        Input tensor stored on GPU memory.

    Returns
    -------
    output_tens : PyTorch :obj:`Tensor` or :obj:`list` of PyTorch :obj:`Tensor`
        Output tensor stored on CPU memory.
    """
    if isinstance(input_tens, list):
        output_tens = [inp.cpu() for inp in input_tens]

    elif isinstance(input_tens, torch.Tensor):
        output_tens = input_tens.cpu()

    return output_tens


def to_gpu(input_tens, dev='cuda'):
    """Transfers `input_tens` from gpu to cpu memory.

    Parameters
    ----------
    input_tens : PyTorch :obj:`Tensor` or :obj:`list` of PyTorch :obj:`Tensor`
        Input tensor stored on GPU memory.

    Returns
    -------
    output_tens : PyTorch :obj:`Tensor` or :obj:`list` of PyTorch :obj:`Tensor`

    """
    if isinstance(input_tens, list):
        assert isinstance(input_tens[0], torch.Tensor), f"Cannot move a \
            non-Tensor object of dtype `{type(input_tens[0])}` to GPU memory."
        output_tens = [inp.to(dev) for inp in input_tens]

    elif isinstance(input_tens.to(dev), torch.Tensor):
        output_tens = input_tens

    return output_tens


def mult(input_tens, psi):
    """Elementwise multiplication of `input` onto the elements of `psi`."""
    return input_tens * psi


def norm_sq(psi_comp):
    """Compute the density (norm-squared) of a single wavefunction component.

    Parameters
    ----------
    psi_comp : Numpy :obj:`array` or PyTorch :obj:`Tensor`
        A single wavefunction component.

    Returns
    -------
    psi_sq : Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The norm-square of the wavefunction.

    """
    if isinstance(psi_comp, np.ndarray):
        psi_sq = np.abs(psi_comp)**2

    elif isinstance(psi_comp, torch.Tensor):
        psi_sq = torch.abs(psi_comp)**2

    return psi_sq


def angle(psi_comp):
    """Compute the phase (angle) of a single complex wavefunction component.

    Parameters
    ----------
    psi_comp : Numpy :obj:`array` or PyTorch :obj:`Tensor`
        A single wavefunction component.

    Returns
    -------
    angle : Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The phase (angle) of the component's wavefunction.

    """
    if isinstance(psi_comp, np.ndarray):
        ang = np.angle(psi_comp)
    elif isinstance(psi_comp, torch.Tensor):
        ang = torch.angle(psi_comp)

    return ang


def cosh():
    """Hyperbolic cosine of a complex tensor."""


def sinh():
    """Hyperbolic sine of a complex tensor."""


def grad():
    """Take a list of tensors or np arrays; checks type."""


def grad__sq():
    """Take a list of tensors or np arrays; checks type."""


def conj():
    """Complex conjugate of a complex tensor."""


def norm(psi, vol_elem, atom_num, pop_frac=None):
    """
    Normalize spinor wavefunction to the expected atom numbers and populations.

    This function normalizes to the total expected atom number `atom_num`,
    and to the expected population fractions `pop_frac`. Normalization is
    essential in processes where the total atom number is not conserved,
    (e.g. imaginary time propagation).

    Parameters
    ----------
    psi : :obj:`list` of Numpy :obj:`arrays` or PyTorch :obj:`Tensors`.
        The wavefunction to normalize.
    vol_elem : :obj:`float`
        Volume element for either real- or k-space.
    atom_num : :obj:`int`
        The total expected atom number.
    pop_frac : array-like, optional
        The expected population fractions in each spin component.

    Returns
    -------
    psi_norm : :obj:`list` of Numpy :obj:`arrays` or PyTorch :obj:`Tensors`.

    """
    dens = density(psi)
    if isinstance(dens[0], np.ndarray):
        if pop_frac is None:
            norm_factor = np.sum(dens[0] + dens[1]) * vol_elem / atom_num
            psi_norm = [p / np.sqrt(norm_factor) for p in psi]
            dens_norm = [d / norm_factor for d in dens]
        else:
            # TODO: Implement population fraction normalization.
            raise NotImplementedError("""Normalizing to the expected population
                                      fractions is not yet implemented for
                                      Numpy arrays.""")

    elif isinstance(dens[0], torch.Tensor):
        if pop_frac is None:
            norm_factor = torch.sum(dens[0] + dens[1]) * vol_elem / atom_num
            psi_norm = [p / np.sqrt(norm_factor.item()) for p in psi]
            dens_norm = [d / norm_factor.item() for d in dens]
        else:
            raise NotImplementedError("""Normalizing to the expected population
                                      fractions is not implemented for
                                      PyTorch tensors.""")

    return psi_norm, dens_norm


def density(psi):
    """
    Compute the density of a spinor wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of 2D Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input spinor wavefunction.

    Returns
    -------
    dens : Numpy :obj:`array`, PyTorch :obj:`Tensor`, or :obj:`list` thereof
        The density of each component's wavefunction.

    """
    if isinstance(psi, list):
        dens = [norm_sq(p) for p in psi]
    else:
        dens = norm_sq(psi)
    return dens


def phase(psi):
    """Compute the phase of a real-space spinor wavefunction.

    Parameters
    ----------
    psi : :obj:`list` of 2D Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input spinor wavefunction.

    Returns
    -------
    phase : Numpy :obj:`array`, PyTorch :obj:`Tensor`, or :obj:`list` thereof
        The phase of each component's wavefunction.
    """
    if isinstance(psi, list):
        phase_psi = [angle(p) for p in psi]
    else:
        phase_psi = angle(psi)

    return phase_psi


def calc_atoms(psi, vol_elem=1.0):
    """Calculate the total number of atoms.

    Parameters
    ----------
    psi : :obj:`list` of 2D Numpy :obj:`array` or PyTorch :obj:`Tensor`
        The input spinor wavefunction.
    vol_elem : :obj:`float`
        2D volume element of the space.

    Returns
    -------
    atom_num : :obj:`float`
        The total atom number in both spin components.

    """
    dens = density(psi)
    atom_num = float(sum(dens).sum() * vol_elem)
    return atom_num


def inner_prod():
    """Calculate the inner product of two wavefunctions."""
