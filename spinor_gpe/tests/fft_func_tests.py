"""Test script for FFT functions.

Functions are tested self-consistently against each other. FFT functions
should:
 - Preserve normalization/atom number
 - Forward and then inverse FFT should return the original (1D & 2D)
 - Not do anything funny to the shape of the wavefunction
 - Work the same for np or torch
 - Two 1D transforms should be the same as a single 2D transform
"""

# pylint: disable=wrong-import-position
import os
import sys
import itertools
import math
sys.path.insert(0, os.path.abspath('..'))

import numpy as np  # noqa: E402
# from matplotlib import pyplot as plt  # noqa: E402
import torch  # noqa: E402

# from pspinor import pspinor as spin  # noqa: E402
from pspinor import tensor_tools as ttools  # noqa: E402


def for_back_torch_2d():
    """Compute forward 2D FFT and then backward 2D FFT. Assert same as initial.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        grid_k = ttools.fft_2d(grid, delta_r)
        grid_r = ttools.ifft_2d(grid_k, delta_r)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}"
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `for_back_torch_2d` passed.")


def for_back_np_2d():
    """Compute forward 2D FFT and then backward 2D FFT.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k = ttools.fft_2d(grid, delta_r)
        grid_r = ttools.ifft_2d(grid_k, delta_r)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}"
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `for_back_np_2d` passed.")


def for_back_np_1d(axis=0):
    """Compute forward 1D FFT and then backward 1D FFT along the x-direction.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k = ttools.fft_1d(grid, delta_r, axis=axis)
        grid_r = ttools.ifft_1d(grid_k, delta_r, axis=axis)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}"
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print(f"Test `for_back_np_1d` for axis={axis} passed.")


def for_back_torch_1d(axis=0):
    """Compute forward 1D FFT and then backward 1D FFT along the x-direction.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        grid_k = ttools.fft_1d(grid, delta_r, axis=axis)
        grid_r = ttools.ifft_1d(grid_k, delta_r, axis=axis)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}"
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print(f"Test `for_back_torch_1d` for axis={axis} passed.")


def compare1_torch_1d_2d():
    """Compute two forward 1D FFTs and then a single inverse 2D FFT.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        grid_k_half = ttools.fft_1d(grid, delta_r, axis=0)
        grid_k = ttools.fft_1d(grid_k_half, delta_r, axis=1)
        grid_r = ttools.ifft_2d(grid_k, delta_r)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare1_torch_1d_2d` passed.")


def compare1_np_1d_2d():
    """Compute two forward 1D FFTs and then a single inverse 2D FFT.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k_half = ttools.fft_1d(grid, delta_r, axis=0)
        grid_k = ttools.fft_1d(grid_k_half, delta_r, axis=1)
        grid_r = ttools.ifft_2d(grid_k, delta_r)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare1_np_1d_2d` passed.")


def compare2_np_1d_2d():
    """Compute a single forward 2D FFT and then two inverse 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k = ttools.fft_2d(grid, delta_r)
        grid_r_half = ttools.ifft_1d(grid_k, delta_r, axis=1)
        grid_r = ttools.ifft_1d(grid_r_half, delta_r, axis=0)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare2_np_1d_2d` passed.")


def compare2_torch_1d_2d():
    """Compute a single forward 2D FFT and then two inverse 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        grid_k = ttools.fft_2d(grid, delta_r)
        grid_r_half = ttools.ifft_1d(grid_k, delta_r, axis=1)
        grid_r = ttools.ifft_1d(grid_r_half, delta_r, axis=0)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare2_torch_1d_2d` passed.")


def compare3_torch_1d_2d():
    """Compute a single forward 2D FFT and two forward 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        grid_r_half = ttools.fft_1d(grid, delta_r, axis=1)
        grid_r = ttools.fft_1d(grid_r_half, delta_r, axis=0)
        grid = ttools.fft_2d(grid, delta_r)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare3_torch_1d_2d` passed.")


def compare3_np_1d_2d():
    """Compute a single forward 2D FFT and two forward 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_r_half = ttools.fft_1d(grid, delta_r, axis=1)
        grid_r = ttools.fft_1d(grid_r_half, delta_r, axis=0)
        grid = ttools.fft_2d(grid, delta_r)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"Max errors: \
            {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare3_np_1d_2d` passed.")


def compare4_torch_1d_2d():
    """Compute a single inverse 2D FFT and two inverse 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype, device='cuda')] * 2
        # grid = ttools.fft_2d(grid, delta_r)
        grid_r_half = ttools.ifft_1d(grid, delta_r, axis=0)
        grid_r = ttools.ifft_1d(grid_r_half, delta_r, axis=1)
        grid = ttools.ifft_2d(grid, delta_r)

        max_diff0 = torch.max(abs(grid_r[0] - grid[0]))
        max_diff1 = torch.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"\
            Max errors: {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare4_torch_1d_2d` passed.")


def compare4_np_1d_2d():
    """Compute a single inverse 2D FFT and two inverse 1D FFTs.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid = ttools.fft_2d(grid, delta_r)
        grid_r_half = ttools.ifft_1d(grid, delta_r, axis=0)
        grid_r = ttools.ifft_1d(grid_r_half, delta_r, axis=1)
        grid = ttools.ifft_2d(grid, delta_r)

        max_diff0 = np.max(abs(grid_r[0] - grid[0]))
        max_diff1 = np.max(abs(grid_r[1] - grid[1]))
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_r)]

        assert (max_diff0 <= eps and max_diff1 <= eps), f"\
            Max errors: {max_diff0}, {max_diff1}."
        assert len(grid_r) == 2
        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare4_np_1d_2d` passed.")


def compare_norm_np():
    """Compare the total norm of a grid and its FFT.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = np.complex128
    eps = np.finfo(dtype).eps
    eps *= 10
    func = np.ones
    delta_r = np.array([1, 1])
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k = ttools.fft_2d(grid, delta_r)

        vol_elem = 4 * np.pi**2 / math.prod(shape)
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_k, vol_elem)]

        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare_norm_np` passed.")


def compare_norm_torch():
    """Compare the total norm of a grid and its FFT.

    Compute this for all possible grid sizes with lengths powers of 2.

    """
    length = [2**s for s in range(7, 11)]
    all_shapes = list(itertools.product(length, length))
    dtype = torch.complex128
    eps = torch.finfo(dtype).eps
    eps *= 10
    func = torch.ones
    delta_r = (1, 1)
    for shape in all_shapes:
        grid = [func(shape, dtype=dtype)] * 2
        grid_k = ttools.fft_2d(grid, delta_r)

        vol_elem = 4 * np.pi**2 / math.prod(shape)
        atoms = [ttools.calc_atoms(grid), ttools.calc_atoms(grid_k, vol_elem)]

        assert abs(atoms[0] - atoms[1]) <= eps * math.prod(shape), \
            f"\nAtom num. before/after: {atoms[0]}, {atoms[1]}; \
            \nDifference: {abs(atoms[0] - atoms[1])}; \
            \n2*eps*N: {eps * math.prod(shape)}."
    print("Test `compare_norm_torch` passed.")


if __name__ == "__main__":
    for_back_torch_2d()  # 2D forward & 2D back, pytorch
    for_back_np_2d()  # 2D forward & 2D back, numpy
    for_back_torch_1d(axis=0)  # 1D forward & 1D back, pytorch
    for_back_torch_1d(axis=1)  # 1D forward & 1D back, pytorch
    for_back_np_1d(axis=0)  # 1D forward & 1D back, numpy
    for_back_np_1d(axis=1)  # 1D forward & 1D back, numpy
    compare1_torch_1d_2d()  # 1D forward (x2) & 2D back, pytorch
    compare1_np_1d_2d()  # 1D forward (x2) & 2D back, numpy
    compare2_torch_1d_2d()  # 2D forward & 1D back (x2), pytorch
    compare2_np_1d_2d()  # 2D forward & 1D back (x2), numpy
    compare3_torch_1d_2d()  # 2D forward & 1D forward (x2), pytorch
    compare3_np_1d_2d()  # 2D forward & 1D forward (x2), numpy
    compare4_torch_1d_2d()  # 2D back & 1D back (x2), pytorch
    compare4_np_1d_2d()  # 2D back & 1D back (x2), numpy
    compare_norm_torch()  # Real- & k-space populations, pytorch
    compare_norm_np()  # Real- & k-space populations, numpy
