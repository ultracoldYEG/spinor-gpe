"""Test script for FFT and other functions in the tensor_tools.py module.

The working directory for this script needs to be in the project root
directory, i.e. /spinor-gpe/.

"""
# pylint: disable=wrong-import-position
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from pspinor import pspinor as spin  # noqa: E402
from pspinor import tensor_tools as ttools  # noqa: E402
# import torch

DATA_PATH = 'ground_state/Trial_000'

FREQ = 50
W = 2*np.pi*FREQ
GAMMA = 1.0
ETA = 40.0

ATOM_NUM = 1e2
omeg = {'x': W, 'y': GAMMA*W, 'z': ETA*W}
g_sc = {'uu': 1.0, 'dd': 1.0, 'ud': 1.04}
pop_frac = (0.5, 0.5)
ps = spin.PSpinor(DATA_PATH, overwrite=True, atom_num=ATOM_NUM, omeg=omeg,
                  g_sc=g_sc, phase_factor=-1, is_coupling=False,
                  pop_frac=pop_frac, r_sizes=(8, 8), mesh_points=(256, 512))

plt.figure()
# plt.imshow(ttools.density(ttools.fft_2d(ps.psi, ps.delta_r))[0])
plt.imshow(ttools.density(ps.psik)[0])
plt.show()

ps.coupling_setup(wavel=790.1e-9)
ps.coupling_grad()

psi = ps.psi
psik = ttools.fft_2d(psi, ps.delta_r)
psi_prime = ttools.ifft_2d(psik, ps.delta_r)
print((np.abs(psi[0])**2 - np.abs(psi_prime[0])**2).max())

# --------- 2. RUN (Imaginary) ----
ps.N_STEPS = 1000
ps.dt = 1/50
ps.is_sampling = True
ps.device = 'cuda:0'

res0 = ps.imaginary()
# `res0` is an object containing the final wavefunctions, the energy exp.
# values, populations, average positions, and a directory path to sampled
# wavefunctions. It also has class methods for plotting and analysis.

psik_shifted = ps.shift_momentum(ps.psik, frac=(0.0, 1.0))
plt.figure()
plt.imshow(ttools.density(psik_shifted[0]))
plt.show()

psi_shifted = ttools.ifft_2d(psik_shifted, ps.delta_r)
plt.figure()
plt.imshow(ttools.density(psi_shifted[0]))
plt.show()

ps.plot_rdens(psi_shifted, spin=0, scale=ps.rad_tf)
ps.plot_rphase(psi_shifted, spin=0, scale=ps.rad_tf)
ps.plot_kdens(psik_shifted, spin=0, scale=ps.kL_recoil)

# %%


def fft_test(f=1, size=1024):
    x = np.linspace(-5, 5, size)
    diff_x = np.diff(x)[0]
    y = np.ones_like(x)  # np.sin(2 * np.pi * f * x)
    freq = np.fft.fftfreq(size, diff_x)
    fft = np.fft.fft(y, norm='ortho')
    fft = np.fft.fftshift(fft)
    freq = np.fft.fftshift(freq)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    ax1.plot(x, y)
    ax2.plot(freq, fft, '-o', lw=0.5)
    plt.show()


fft_test(6, 64)
