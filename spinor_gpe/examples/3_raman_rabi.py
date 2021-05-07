"""
Example 3: Raman Rabi Flopping
==============================

Starts with the Thomas-Fermi solution with all the population in one spin
component. Propagates in imaginary time to the ground state. From here,
configures a uniform Raman coupling which drives the population on resonance
between the two components.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin


# 1. SETUP

DATA_PATH = 'examples/Trial_003'  # Default data path is in the /data/ folder

FREQ = 50
W = 2*np.pi*FREQ
Y_SCALE = 1
Z_SCALE = 40.0

ATOM_NUM = 1e4
OMEG = {'x': W, 'y': Y_SCALE * W, 'z': Z_SCALE * W}
G_SC = {'uu': 1, 'dd': 1, 'ud': 0.0}
POP_FRAC = (1.0, 0.0)

ps = spin.PSpinor(DATA_PATH, overwrite=True,
                  atom_num=ATOM_NUM,
                  omeg=OMEG,
                  g_sc=G_SC,
                  pop_frac=POP_FRAC,
                  r_sizes=(16, 16),
                  mesh_points=(256, 256))

ps.coupling_setup(wavel=790.1e-9, kin_shift=True)

# Shifts the k-space density momentum peaks by `kshift_val` [`kL_recoil` units]
ps.shift_momentum(kshift_val=1.0, frac=(0, 1.0))

# Selects the form of the coupling operator in the non-rotated reference frame
ps.rot_coupling = False

ZOOM = 2  # Zooms the momentum-space density plots by a constant factor

ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


# 2. RUN (Imaginary-time)

N_STEPS = 1000
DT = 1/50
DEVICE = 'cuda'
ps.rand_seed = 99999

res0, t_prop0 = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True,
                             n_samples=50)


# 3. ANALYZE

res0.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_pops()
# res0.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM)
print(f'\nFinal energy: {res0.eng_final[0]} [hbar * omeg]')


# 4. RUN (Real-time)

# Initializes a uniform Raman coupling (scaled in `EL_recoil` units)
ps.coupling_uniform(1.0 * ps.EL_recoil)

N_STEPS = 2000
DT = 1/5000
res1, t_prop0 = ps.real(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=100)


# 5. ANALYZE

res1.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_total(kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_pops()
res1.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM/2)
print(f'\nFinal energy: {res1.eng_final[0]} [hbar * omeg]')
