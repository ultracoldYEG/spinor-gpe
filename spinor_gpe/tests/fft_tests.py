"""Test script for FFT and other functions in the tensor_tools.py module.

The working directory for this script needs to be in the project root
directory, i.e. /spinor-gpe/.

"""
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# pylint: disable=wrong-import-position
import os, sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from matplotlib import pyplot as plt

from pspinor import pspinor as spin
from pspinor import tensor_tools as ttools
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
                  pop_frac=pop_frac, r_sizes=(8, 8))

plt.figure()
plt.imshow(ttools.density(ttools.fft_2d(ps.psi, ps.delta_r))[0])
plt.show()

ps.coupling_setup(wavel=790.1)
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

# --------- 3. ANALYZE ------------
res0.plot_spins()
res0.plot_total()
res0.plot_eng()
res0.plot_pops()
res0.make_movie()

# --------- 4. SETUP --------------



# --------- 5. RUN (Real) ---------
ps.N_STEPS = 2000
ps.dt = 1/5000
ps.is_sampling = True

res1 = ps.real()

# --------- 6. ANALYZE ------------
res1.plot_spins()
res1.plot_total()
res1.plot_eng()
res1.plot_pops()
res1.make_movie()
