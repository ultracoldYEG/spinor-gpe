# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:38:56 2021

@author: benjamin
"""

#import numpy as np
#from matplotlib import pyplot as plt

class Spinors(object):
    def __init__(self):
        pass

    def imaginary(self):
        ''' Also generates a dictionary of the tensor versions of all the energy
        grids; should be accessible to the user.'''
        return PropResult()

    def real(self):
        return PropResult()

    def coupling_setup(self, **kwargs):
        pass

    def omega_grad(self):
        pass

    def detuning_grad(self):
        pass



class PropResult(object):
    def __init__(self):
        pass

    def plot_spins(self):
        pass

    def plot_total(self):
        pass

    def plot_eng(self):
        pass

    def plot_pops(self):
        pass

    def analyze_vortex(self):
        pass

    def make_movie(self):
        pass

class SpinCoupling(object):
    def __init__(self):
        pass
    