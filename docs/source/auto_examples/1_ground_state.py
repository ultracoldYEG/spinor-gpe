"""
Example 1: Ground State
=======================

Starting with the Thomas-Fermi solution, propagate in imaginary time to
reach the ground state. Propagation smooths out the sharp edges
on both components' densities.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin
# sphinx_gallery_thumbnail_path = '_static/1_ground.png'


# 1. SETUP

DATA_PATH = 'examples/Trial_011'  # Default data path is in the /data/ folder

FREQ = 50
W = 2*np.pi*FREQ
Y_SCALE = 1
Z_SCALE = 40.0

ATOM_NUM = 1e2
OMEG = {'x': W, 'y': Y_SCALE * W, 'z': Z_SCALE * W}
G_SC = {'uu': 1, 'dd': 1, 'ud': 1.04}

ps = spin.PSpinor(DATA_PATH, overwrite=True,  # Initialize PSpinor object
                  atom_num=ATOM_NUM,
                  omeg=OMEG,
                  g_sc=G_SC,
                  pop_frac=(0.5, 0.5),
                  r_sizes=(8, 8),
                  mesh_points=(256, 256))

ps.coupling_setup(wavel=790.1e-9, kin_shift=False)

ZOOM = 4  # Zooms the momentum-space density plots by a constant factor

# Plot real- and momentum-space density & real-space phase of both components
ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


# 2. RUN (Imaginary-time)

DT = 1/50
N_STEPS = 1000
DEVICE = 'cuda'
ps.rand_seed = 99999

# Run propagation loop:
# - Returns `PropResult` & `TensorPropagator` objects
res = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=50)


# 3. ANALYZE

res.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)  # Plot total density & phase
res.plot_pops()  # Plot how the spins' populations evolves
res.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM,
               norm_type='half')
