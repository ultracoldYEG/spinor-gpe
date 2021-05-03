"""General test script for GPE propagation on GPU."""
# pylint: disable=wrong-import-position
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np  # noqa: E402
# import torch  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from spinor_gpe.pspinor import pspinor as spin  # noqa: E402
# from spinor_gpe.pspinor import tensor_tools as ttools  # noqa: E402
# from spinor_gpe.pspinor import prop_result as result  # noqa: E402


# BASIC STRUCTURE OF A SIMULATION:

# --------- 1. SETUP --------------
# [ ] Instantiate some sort of spinor object
# [ ] Set directory information (optional; default is the package directory)
# [ ] Set up trap parameters (default values available)
# [ ] Set up interaction parameters (default values available)
# [ ] Set up Raman parameters
# [ ] Additional functions (e.g. seed vortices, shift momentum)
# [ ] Specify sampling, time step duration (needs a default)
# --------- 2. RUN ----------------
# [ ] Propagate (imaginary or real time; should be independent of spinor vs.
#     scalar)
# --------- 3. ANALYZE ------------
# [ ] Post analysis from final wavefunction (plots, vortex)
# [ ] Post analysis from sampled wavefunctions (e.g. energy exp., populations,
#                                               max density)
# --------- 4. REPEAT -------------

# -------------------------------------------------------------------
# Test Case #1: Simple imaginary time propagation to the ground state

# --------- 1. SETUP --------------

# All of the wavefunctions and simulation parameters (e.g. psi, psik,
# TF parameters, trap frequencies, Raman parameters, directory paths) will
# be contained in a PSpinors object, with class methods for propagation
# (real & imaginary).

DATA_PATH = 'ground_state/Trial_003'
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
#     |    |    |   |   ├── sampled_psik.npy
#     |    |    |   |   ├── sampled_times.npy
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
GAMMA = 1
ETA = 40.0

ATOM_NUM = 1e4
omeg = {'x': W, 'y': GAMMA*W, 'z': ETA*W}
g_sc = {'uu': 1, 'dd': 1, 'ud': 0.0}
pop_frac = (0.5, 0.5)
# pop_frac = (1.0, 0.0)
ps = spin.PSpinor(DATA_PATH, overwrite=True, atom_num=ATOM_NUM, omeg=omeg,
                  g_sc=g_sc, phase_factor=-1,
                  pop_frac=pop_frac, r_sizes=(16, 16), mesh_points=(256, 256))
print(ps._calc_atoms(space='k'))
# ps.plot_rdens()
# ps.plot_rphase()
# ps.plot_kdens()
# ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil)

ps.coupling_setup(wavel=790.1e-9)
ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil)
# ps.detuning_grad(20, 0)
# ps.shift_momentum()

# --------- 2. RUN (Imaginary) ----

N_STEPS = 1000
DT = 1/50
IS_SAMPLING = True
DEVICE = 'cuda'
ps.rand_seed = 99999
N_SAMPLES = 10

res0, t_prop = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=IS_SAMPLING,
                            n_samples=N_SAMPLES)
# print(ps.prop.space)
# `res0` is an object containing the final wavefunctions, the energy exp.
# values, populations, average positions, and a directory path to sampled
# wavefunctions. It also has class methods for plotting and analysis.
print(ps._calc_atoms(space='r'))
# --------- 3. ANALYZE ------------
res0.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil)
# res0.plot_total(kscale=ps.kL_recoil)
res0.plot_pops()
# res0.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=False)

# --------- 4. SETUP --------------


# # --------- 5. RUN (Real) ---------
# N_STEPS = 1000
# DT = 1/500
# IS_SAMPLING = True

# res1 = ps.real(DT, N_STEPS, DEVICE, is_sampling=IS_SAMPLING,
#                n_samples=N_SAMPLES)

# # --------- 6. ANALYZE ------------
# res1.plot_spins(kscale=ps.kL_recoil)
# # res1.plot_total(kscale=ps.kL_recoil)
# # res1.plot_pops()
# res1.make_movie(kscale=ps.kL_recoil, play=True)
