# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:34:46 2021

@author: benjamin

Physical constants used in the pseudospin-1/2 GPE simulations. All units are
in SI.
"""

import scipy.constants as const

# Universal constants:
h = const.h  #: Planck's constant
hbar = const.hbar  #: Reduced Planck's constant
c = const.c
eps0 = const.epsilon_0
a0 = const.physical_constants['Bohr radius'][0]
e = const.elementary_charge

# Rubidium-87 constants --------------------:
Rb87 = {}  #: dict: Atomic data for Rubidium-87
Rb87['m'] = 87 * 1.66e-27  #: Mass of Rb87 atom [kg]
Rb87['D2'] = 780.1e-9  #: Rb87 D2 line wavelength [m]
Rb87['a_sc'] = 100.4 * a0  #: Rb87 Scattering length [m]
#: Scattering interaction coupling strength [m^5 kg/s^2]
Rb87['g'] = 4 * const.pi * (hbar**2) * Rb87['a_sc'] / Rb87['m']
Rb87['k_r'] = 2 * const.pi / Rb87['D2']  #: Resonant recoil wavenumber [1/m]
Rb87['E_r'] = (h / (2*Rb87['m'])) * (1/Rb87['D2'])**2  #: Recoil Energy [Hz]
#: D1-line Scalar polarizability [Hz m^2/V^2]
Rb87['D1_alpha'] = h * 1.22306e-5
Rb87['gJ_S12'] = 2.002331  #: 5S-1/2 fine structure constant
Rb87['gJ_P12'] = 0.666  #: 5P-1/2 fine structure constant
Rb87['gJ_P32'] = 1.3362  #: 5P-3/2 fine structure constant
Rb87['delta_FS'] = 6.834682e9  #: Ground state hyperfine splitting [Hz]

# Possibly add more atomic species data later
