# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:25:01 2021

@author: benjamin
"""

import numpy as np
#from matplotlib import pyplot as plt
import spinors

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
sp = spinors.Spinors()
'''All of the wavefunctions and simulation parameters (e.g. psi, psik,
TF parameters, trap frequencies, Raman parameters, directory paths) will
be contained in a Spinors object, with class methods for propagation
(real & imaginary).
'''

sp.TRIAL = '001'

FREQ = 50
W = 2*np.pi*FREQ
GAMMA = 1.0
ETA = 40.0

sp.ATOM_NUM = 1e4
sp.w = {'wx' : W, 'wy' : GAMMA*W, 'wz' : ETA*W}
sp.g = {'00' : 1.0, '11' : 0.995, '01' : 0.995}
sp.is_coupling = False

sp.coupling_setup(lam=790.1)
sp.omega_grad()

# --------- 2. RUN (Imaginary) ----
sp.N_STEPS = 1000
sp.dt = 1/50
sp.is_sampling = True
sp.device = 'cuda:0'

res0 = sp.imaginary()
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


# --------- 4. RUN (Real) ---------
sp.N_STEPS = 2000
sp.dt = 1/5000
sp.is_sampling = True

res1 = sp.real()

# --------- 5. ANALYZE ------------
res1.plot_spins()
res1.plot_total()
res1.plot_eng()
res1.plot_pops()
res1.make_movie()
