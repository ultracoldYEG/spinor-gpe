"""
Benchmarks
==========

On a given system and hardware configuration, times the propagation loop for
increasing grid sizes.

"""
import os
import sys
import timeit
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin
# sphinx_gallery_thumbnail_path = '_static/1_ground.png'


# 1. SETUP

DATA_PATH = 'examples/Bench_001'  # Default data path is in the /data/ folder

W = 2 * np.pi * 50
Y_SCALE, Z_SCALE = 1, 40.0

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

DT = 1/50
N_STEPS = 1
DEVICE = 'cuda'

# Run propagation loop:
# - Returns `PropResult` & `TensorPropagator` objects
res, prop = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=False)

stmt = """prop.full_step()"""

timer = timeit.Timer(stmt=stmt, globals=globals())
vals = timer.repeat(5, 100)
print(vals)
