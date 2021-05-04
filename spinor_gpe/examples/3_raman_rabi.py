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

DATA_PATH = 'examples/Trial_003'

FREQ = 50
W = 2*np.pi*FREQ
GAMMA = 1
ETA = 40.0

ATOM_NUM = 1e4
omeg = {'x': W, 'y': GAMMA*W, 'z': ETA*W}
g_sc = {'uu': 1, 'dd': 1, 'ud': 0.0}
pop_frac = (1.0, 0.0)
ps = spin.PSpinor(DATA_PATH, overwrite=True, atom_num=ATOM_NUM, omeg=omeg,
                  g_sc=g_sc, phase_factor=1,
                  pop_frac=pop_frac, r_sizes=(16, 16), mesh_points=(256, 256))

ps.coupling_setup(wavel=790.1e-9, mom_shift=True)
# ps.coupling_uniform(0 * ps.EL_recoil)
# ps.detuning_uniform(0)
ps.shift_momentum(kshift_val=1, frac=(0, 1.0))
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

res0, t_prop0 = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=IS_SAMPLING,
                             n_samples=N_SAMPLES)


# 3. ANALYZE

res0.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_pops()
# res0.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM)
print(f'\nFinal energy: {res0.eng_final[0]} [hbar * omeg]')


# 4. RUN (Real)
ps.coupling_uniform(1.0 * ps.EL_recoil)

N_STEPS = 2000
DT = 1/5000
IS_SAMPLING = True
N_SAMPLES = 100
res1, t_prop0 = ps.real(DT, N_STEPS, DEVICE, is_sampling=IS_SAMPLING,
                        n_samples=N_SAMPLES)


# 5. ANALYZE

res1.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_total(kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_pops()
res1.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM/2)
print(f'\nFinal energy: {res1.eng_final[0]} [hbar * omeg]')
