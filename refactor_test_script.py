# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:25:01 2021

@author: benjamin
"""

import numpy as np
from matplotlib import pyplot as plt
import spinor_gpe.pspinor.pspinor as spin

### BASIC STRUCTURE OF A SIMULATION:

# --------- 1. SETUP --------------
# [ ] Instantiate some sort of spinor object
# [ ] Set directory information (optional; default is the package directory)
# [ ] Set up trap parameters (default values available)
# [ ] Set up interaction parameters (default values available)
# [ ] Set up Raman parameters
# [ ] Additional functions (e.g. seed vortices, shift momentum)
# [ ] Specify sampling, time step duration (needs a default)
# --------- 2. RUN ----------------
# [ ] Propagate (imaginary or real time; should be independent of spinor vs. scalar)
# --------- 3. ANALYZE ------------
# [ ] Post analysis from final wavefunction (plots, vortex)
# [ ] Post analysis from sampled wavefunctions (e.g. energy exp., populations, max density)
# --------- 4. REPEAT -------------



# -----------------------------------------------------------------------------
# Test Case #1: Simple imaginary time propagation to the ground state

# --------- 1. SETUP --------------

'''All of the wavefunctions and simulation parameters (e.g. psi, psik,
TF parameters, trap frequencies, Raman parameters, directory paths) will
be contained in a Spinors object, with class methods for propagation
(real & imaginary).
'''

DATA_PATH = 'ground_state/Trial_000'
# The directory might look like:
#     spinor_gpe
#     ├── pspinors
#     |    ├── __init__.py
#     |    ├── pspinor.py
#     |    ├── tensor_tools.py
#     |    ├── tensor_propagator.py
#     |    └── prop_result.py
#     ├── constants.py
#     ├── data
#     |    ├── {project_name1}
#     |    |    ├── {Trial_000}
#     |    |    |   ├── code
#     |    |    |   |   └── this_script.py
#     |    |    |   ├── trial_data
#     |    |    |   |   ├── sampled_wavefunctions.npy
#     |    |    |   |   └── initial_wavefunction.npy
#     |    |    |   ├── description.txt
#     |    |    |   ├── assorted_images.png
#     |    |    |   └── assorted_videos.mp4
#     |    |    ├── {Trial_001}
#     |    |    |   └── ...
#     |    |    ├── {Trial_002}
#     |    |    |   └── ...
#     |    |    └── ...
#     |    ├── {project_name2}
#     |    |    ├── {Trial_000}
#     |    |    ├── {Trial_001}
#     |    |    ├── {Trial_002}
#     |    |    └── ...
#     |    ├── ...

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
plt.imshow(spin.density(spin.fft_1d(ps.psi, ps.delta_r))[0])
plt.show()

ps.coupling_setup(lam=790.1)
ps.omega_grad()

# --------- 2. RUN (Imaginary) ----
ps.N_STEPS = 1000
ps.dt = 1/50
ps.is_sampling = True
ps.device = 'cuda:0'

res0 = ps.imaginary()
''' `res0` is an object containing the final wavefunctions, the energy exp.
values, populations, average positions, and a directory path to sampled
wavefunctions. It also has class methods for plotting and analysis.
'''

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
