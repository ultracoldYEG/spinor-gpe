"""
Example 3: Raman Rabi Flopping
==============================

Starts with the Thomas-Fermi solution with all the population in one spin
component. Propagates in imaginary time to the ground state. From here,
configures a uniform Raman coupling which drives the population on resonance
between the two components.

Physical Parameters
-------------------
.. topic:: Atom number

    :math:`\\quad N_{\\rm at} = 10,000`

.. topic:: Atomic mass, Rubidium-87

    :math:`\\quad m = 1.4442 \\times 10^{-25}~[\\rm kg]`

.. topic:: Trap frequencies

    :math:`\\quad (\\omega_x, \\omega_y, \\omega_z) = 2 \\pi \\times (50, 200, 2000)~[{\\rm Hz}]`

    :math:`\\quad (\\omega_x, \\omega_y, \\omega_z) = \\omega_x \\times (1, \\gamma, \\eta) = (1, 1, 40)~[\\omega_x]`

.. topic:: Harmonic oscillator length, x-axis

    :math:`\\quad a_x = \\sqrt{\\hbar / m \\omega_x} = 1.525~[{\\mu\\rm m}]`

.. topic:: 3D scattering length, Rubidium-87

    | :math:`\\quad a = 5.313~[{\\rm nm}]`

    | :math:`\\quad a_{\\rm sc} = a / a_x = 0.00348~[a_x]`

.. topic:: Scattering 2D scale

    | :math:`\\quad g_{\\rm sc}^{2\\rm D} = \\sqrt{8\\pi\\eta}~a_{\\rm sc} = 0.1105~[\\omega_x a_x^2]`

.. topic:: Scattering coupling

    | :math:`\\quad (g_{\\rm uu}, g_{\\rm dd}, g_{\\rm ud}) = g_{\\rm sc}^{2 \\rm D} \\times (1, 1, 0.0)~[\\omega_x a_x^2]`

.. topic:: Chemical potential

    :math:`\\quad \\mu = \\sqrt{4 N_{\\rm at} a_{\\rm sc} \\gamma \\sqrt{\\eta / 2 \\pi}} = 18.754~[\\omega_x]`

.. topic:: Thomas-Fermi radius

    :math:`\\quad R_{\\rm TF} = \\sqrt{2 \\mu} = 6.124~[a_x]`

.. topic:: Initial population fractions

    :math:`\\quad (p_0, p_1) = (1.0, 0.0)`

.. topic:: Raman wavelength

    :math:`\\quad \\lambda_L = 790.1~[{\\rm nm}]`

Numerical Parameters
--------------------

.. topic:: Number of grid points

    :math:`\\quad (N_x, N_y) = (256, 256)`

.. topic:: r-grid half-size

    :math:`\\quad (x^{\\rm max}, y^{\\rm max}) = (16, 16)~[a_x]`

.. topic:: r-grid spacing

    :math:`\\quad (\\Delta x, \\Delta y) = (0.125, 0.125)~[a_x]`

.. topic:: k-grid half-size

    :math:`\\quad (k_x^{\\rm max}, k_y^{\\rm max}) = \\pi / (\\Delta x, \\Delta y)`

    :math:`\\quad (k_x^{\\rm max}, k_y^{\\rm max}) = (25.133, 25.133)~[a_x^{-1}]`

.. topic:: k-grid spacing

    :math:`\\quad (\\Delta k_x, \\Delta k_y) = \\pi / (x^{\\rm max}, y^{\\rm max})`

    :math:`\\quad (\\Delta k_x, \\Delta k_y) = (0.1963, 0.1963)~[a_x^{-1}]`

.. topic:: Time scale

    :math:`\\quad \\tau_0 = 1 / \\omega_x = 0.00318~[{\\rm s/rad}]`

    :math:`\\quad \\tau_0 = 1~[\\omega_x^{-1}]`

.. topic:: Time step duration, imaginary

    :math:`\\quad \\Delta \\tau_{\\rm im} = 1 / 50~[-i \\tau_0]`

.. topic:: Number of time steps, imaginary

    :math:`\\quad N_{\\rm im} = 1000`

.. topic:: Time step duration, real

    :math:`\\quad \\Delta \\tau_{\\rm real} = 1 / 5000~[\\tau_0]`

.. topic:: Number of time steps, real

    :math:`\\quad N_{\\rm real} = 2000`



"""
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

import numpy as np

from spinor_gpe.pspinor import pspinor as spin
# sphinx_gallery_thumbnail_path = '_static/3_rabi.png'


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
ps.shift_momentum(scale=1.0, frac=(0, 1.0))

# Selects the form of the coupling operator in the non-rotated reference frame
ps.rot_coupling = False

ZOOM = 2  # Zooms the momentum-space density plots by a constant factor

ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


# 2. RUN (Imaginary-time)

DT = 1/50
N_STEPS = 1000
DEVICE = 'cuda'
ps.rand_seed = 99999

res0, prop0 = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=50)


# 3. ANALYZE

res0.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)
res0.plot_pops()
# res0.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM)
print(f'\nFinal energy: {res0.eng_final[0]} [hbar * omeg]')


# 4. RUN (Real-time)

# Initializes a uniform Raman coupling (scaled in `EL_recoil` units)
ps.coupling_uniform(1.0 * ps.EL_recoil)

DT = 1/5000
N_STEPS = 2000
res1, prop1 = ps.real(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=100)


# 5. ANALYZE

res1.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_total(kscale=ps.kL_recoil, zoom=ZOOM/2)
res1.plot_pops()
res1.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM/2)
print(f'\nFinal energy: {res1.eng_final[0]} [hbar * omeg]')
