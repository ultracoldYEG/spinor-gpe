"""Example script for 2D spinor GPE propagation on a GPU.

Example 1: Ground State
    Starting with the Thomas-Fermi solution, propagate in imaginary time,
    before reaching the true ground state.

"""
# pylint: disable=wrong-import-position
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt

from spinor_gpe.pspinor import pspinor as spin  # noqa: E402


# 1. SETUP

DATA_PATH = 'examples/Trial_004'

FREQ = 50
W = 2*np.pi*FREQ
GAMMA = 1
ETA = 40.0

ATOM_NUM = 1e4
omeg = {'x': W, 'y': GAMMA*W, 'z': ETA*W}
g_sc = {'uu': 1, 'dd': 0.995, 'ud': 0.995}
pop_frac = (0.5, 0.5)
ps = spin.PSpinor(DATA_PATH, overwrite=True, atom_num=ATOM_NUM, omeg=omeg,
                  g_sc=g_sc, phase_factor=1,
                  pop_frac=pop_frac, r_sizes=(16, 16), mesh_points=(256, 256))

ps.coupling_setup(wavel=804e-9, mom_shift=True)
ps.coupling_uniform(5 * ps.EL_recoil)
ps.detuning_grad(-12)
ps.shift_momentum(kshift_val=0.6, frac=(0.5, 0.5))
ZOOM = 2
ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)

# %%
# 2. RUN (Imaginary)

N_STEPS = 1000
DT = 1/50
IS_SAMPLING = True
DEVICE = 'cuda'
ps.rand_seed = 99999
N_SAMPLES = 50

res, t_prop = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=IS_SAMPLING,
                           n_samples=N_SAMPLES)


# 3. ANALYZE

res.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res.plot_pops()
res.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM)
print(f'\nFinal energy: {res.eng_final[0]} [hbar * omeg]')
