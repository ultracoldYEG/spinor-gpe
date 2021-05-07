"""
Example 2: Anisotropic Time-of-Flight
=====================================

Starts with the Thomas-Fermi solution for a highly anisotropic trap.
Propagates in imaginary time tor reach the ground state. The trapping
potential is suddenly removed and both components expand and experience
an inversion of their aspect ratio.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin


# 1. SETUP

DATA_PATH = 'examples/Trial_002'  # Default data path is in the /data/ folder

FREQ = 50
W = 2*np.pi*FREQ
Y_SCALE = 4
Z_SCALE = 40.0

ATOM_NUM = 1e4
OMEG = {'x': W, 'y': Y_SCALE * W, 'z':  Z_SCALE * W}
G_SC = {'uu': 1, 'dd': 1, 'ud': 0.5}

ps = spin.PSpinor(DATA_PATH, overwrite=True,
                  atom_num=ATOM_NUM,
                  omeg=OMEG,
                  g_sc=G_SC,
                  phase_factor=1,  # Complex unit phase factor on down spin
                  pop_frac=(0.5, 0.5),
                  r_sizes=(32, 32),
                  mesh_points=(512, 512))

ps.coupling_setup(wavel=790.1e-9)

ZOOM = 4  # Zooms the momentum-space density plots by a constant factor

# Plot real- and momentum-space density & real-space phase of both components
ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


# 2. RUN (Imaginary-time)

N_STEPS = 1000
DT = 1/50
DEVICE = 'cuda'
ps.rand_seed = 99999

# Run propagation loop:
# - Returns `PropResult` & `TensorPropagator` objects
res0, t_prop0 = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=False,
                             n_samples=50)


# 3. ANALYZE

res0.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_pops()


# 4. RUN (Real-time)

N_STEPS = 1000
DT = 1/500
ps.pot_eng = np.zeros_like(ps.pot_eng)  # Removes trapping potential

# Run propagation loop
res1, t_prop0 = ps.real(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=50)


# 5. ANALYZE

res1.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_total(kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_pops()
res1.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM/2,
                norm_type='half')
