"""
FFT and iFFT Time
=================

On a given system and hardware configuration, times the FFT and iFFT function
calls for increasing mesh grid sizes.

"""
import os
import sys
import timeit
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np
import torch
from scipy.stats import median_abs_deviation as mad

from spinor_gpe.pspinor import pspinor as spin
from spinor_gpe.pspinor import tensor_tools as ttools

torch.cuda.empty_cache()

grids = [(64, 64),
         (64, 128),
         (128, 128),
         (128, 256),
         (256, 256),
         (256, 512),
         (512, 512),
         (512, 1024),
         (1024, 1024),
         (1024, 2048),
         (2048, 2048),
         (2048, 4096),
         (4096, 4096)]
n_grids = len(grids)
meas_times = [[0] for i in range(n_grids)]
repeats = np.zeros(n_grids)
size = np.zeros(n_grids)


DATA_PATH = 'benchmarks/Bench_001'  # Default data path is in the /data/ folder

W = 2 * np.pi * 50
ATOM_NUM = 1e2
OMEG = {'x': W, 'y': W, 'z': 40 * W}
G_SC = {'uu': 1, 'dd': 1, 'ud': 1.04}

DEVICE = 'cuda'
COMPUTER = 'Acer Aspire'

for i, grid in enumerate(grids):
    print(i)
    try:
        ps = spin.PSpinor(DATA_PATH, overwrite=True,
                          atom_num=ATOM_NUM, omeg=OMEG, g_sc=G_SC,
                          pop_frac=(0.5, 0.5), r_sizes=(8, 8),
                          mesh_points=grid)

        ps.coupling_setup(wavel=790.1e-9, kin_shift=False)

        res, prop = ps.imaginary(1/50, 1, DEVICE, is_sampling=False)

        stmt = """ttools.fft_2d(prop.psik, prop.space['dr'])"""

        timer = timeit.Timer(stmt=stmt, globals=globals())

        N = timer.autorange()[0]
        if N < 10:
            N *= 10
        vals = timer.repeat(N, 1)
        meas_times[i] = vals
        repeats[i] = N
        size[i] = np.log2(np.prod(grid))

        torch.cuda.empty_cache()
    except RuntimeError as ex:
        print(ex)
        break

median = np.array([np.median(times) for times in meas_times])
med_ab_dev = np.array([mad(times, scale='normal') for times in meas_times])

tag = 'fft\\' + COMPUTER + '_' + DEVICE + '_fft'
np.savez(ps.paths['data'] + '..\\' + tag, computer=COMPUTER, device=DEVICE,
         size=size, n_repeats=repeats, med=median, mad=med_ab_dev)

np.save(ps.paths['data'] + '..\\' + tag, np.array(meas_times, dtype='object'))


# %%

for i, grid in enumerate(grids):
    print(i)
    try:
        ps = spin.PSpinor(DATA_PATH, overwrite=True,
                          atom_num=ATOM_NUM, omeg=OMEG, g_sc=G_SC,
                          pop_frac=(0.5, 0.5), r_sizes=(8, 8),
                          mesh_points=grid)

        ps.coupling_setup(wavel=790.1e-9, kin_shift=False)

        res, prop = ps.imaginary(1/50, 1, DEVICE, is_sampling=False)

        stmt = """ttools.ifft_2d(prop.psik, prop.space['dr'])"""

        timer = timeit.Timer(stmt=stmt, globals=globals())

        N = timer.autorange()[0] * 10
        vals = timer.repeat(N, 1)
        meas_times[i] = vals
        repeats[i] = N
        size[i] = np.log2(np.prod(grid))

        torch.cuda.empty_cache()
    except RuntimeError as ex:
        print(ex)
        break

median = np.array([np.median(times) for times in meas_times])
med_ab_dev = np.array([mad(times, scale='normal') for times in meas_times])

tag = COMPUTER + '_' + DEVICE + '_ifft'
np.savez('data\\' + tag, computer=COMPUTER, device=DEVICE,
         size=size, n_repeats=repeats, med=median, mad=med_ab_dev)

np.save(ps.paths['data'] + '..\\' + tag, np.array(meas_times, dtype='object'))
