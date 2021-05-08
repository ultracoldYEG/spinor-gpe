"""
Example 4: Raman Detuning Gradient Ground State
===============================================

Starts with the Thomas-Fermi solution. Configures a uniform Raman coupling
and a linear gradient in the Raman detuning. Propagates in imaginary time,
before reaching the ground state of this configuration. The detuning
gradient separates the two components vertically, and the line where they
interfere is a row of vortices.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin


# 1. SETUP

DATA_PATH = 'examples/Trial_004'  # Default data path is in the /data/ folder

FREQ = 50
W = 2*np.pi*FREQ
Y_SCALE = 1
Z_SCALE = 40.0

ATOM_NUM = 1e4
OMEG = {'x': W, 'y': Y_SCALE * W, 'z': Z_SCALE * W}
G_SC = {'uu': 1, 'dd': 0.995, 'ud': 0.995}

ps = spin.PSpinor(DATA_PATH, overwrite=True,  # Initialize PSpinor object
                  atom_num=ATOM_NUM,
                  omeg=OMEG,
                  g_sc=G_SC,
                  pop_frac=(0.5, 0.5),
                  r_sizes=(16, 16),
                  mesh_points=(256, 256))

ps.coupling_setup(wavel=804e-9, kin_shift=True)

# Shifts the k-space density momentum peaks by `kshift_val` [`kL_recoil` units]
ps.shift_momentum(kshift_val=0.6, frac=(0.5, 0.5))
ps.coupling_uniform(5 * ps.EL_recoil)
ps.detuning_grad(-12)

# Selects the form of the coupling operator in the rotated reference frame
ps.rot_coupling = True

ZOOM = 2  # Zooms the momentum-space density plots by a constant factor

ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


# 2. RUN (Imaginary-time)

N_STEPS = 1000
DT = 1/50
DEVICE = 'cuda'
ps.rand_seed = 99999

res, t_prop = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True,
                           n_samples=50)


# 3. ANALYZE

res.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res.plot_pops()
res.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM,
               norm_type='half')
print(f'\nFinal energy: {res.eng_final[0]} [hbar * omeg]')
