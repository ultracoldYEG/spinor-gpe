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
        self.prop = TensorPropagator(self)
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



class TensorPropagator(object):
    # Object that sucks in the needed energy grids and parameters for
    # propagation, converts them to tensors, & performs the propagation.
    #  - It means that two copies of the grids aren't carried in the main class
    #  - However, it restricts access to the tensor form of the grids; unless
    #    I keep the Propagation object as a class "attribute".
    #  - I can directly pass `self` to this class and access class attributes,
    #    methods, esp. energy grids. Only do this in the __init__ function
    #    so as to not store the main Spinor object in the class
    
    # --> Should it be a class or just a pure function??
    #      - Maybe a class because then it can store the grids it needs, and 
    #        then access them from the different functions for free.
    #      - It would allow these operations to reside in a separate module.
    # BUT, then I have two classes who are attributes of each other, and
    #     THAT's a weird structure.
    #    - Maybe this class doesn't have to attribute the other one; it just
    #      sucks in the data it needs.
    
    # Will need to calculate certain data throughout the loop, and then
    #     create and populate the PropResult object.
    
    def __init__(self, spin):
        # Needs:
        #  - Energy grids
        #  - Raman grids
        #  - Atom number
        #  - Number of steps
        #  - grid parameters [don't keep tensor versions in a dict, not stable]
        #  - dt
        #  - sample (bool)
        #  - wavefunction sample frequency
        #  - wavefunction anneal frequency (imaginary time)
        #  - device (cpu vs. gpu)
        #  - 
        
        print(spin.N_STEPS)
        pass


    def evolution_op(self):
        pass
    
    def coupling_op(self):
        pass
    
    def single_step(self):
        pass
    
    def full_step(self):
        ''' Divides the full propagation step into three single steps using
        the magic gamma for accuracy.
        '''
        self.single_step()
        self.single_step()
        self.single_step()
        pass
    

    def propagation(self, N_STEPS):
        '''Contains the actual propagation for-loop.'''
        for i in range(N_STEPS):
            self.full_step()
        pass
    
    def energy_exp(self):
        pass
    
    def normalize(self):
        pass
    
    def density(self):
        pass
    
    def inner_prod(self):
        pass
    
    def expect_val(self):
        pass
    
    
    

    
    
#  ----------------- tensor_tools MODULE ---------------
# Would it be a good idea to allow all these functions to accept both arrays 
# and tensors? Maybe, for completeness it's a good idea.
        
def fft_1d():
    '''takes a list of tensors or np arrays; checks type.'''
    pass

def fft_2d():
    '''takes a list of tensors or np arrays; checks type.'''
    pass

def fft_shift():
    pass

def to_numpy():
    pass

def to_tensor():
    pass

def t_mult(a, b):
    '''Assert that a and b are tensors.'''
    pass

def norm_sq():
    '''takes a list of tensors or np arrays; checks type.'''
    pass

def t_cosh():
    pass

def t_sinh():
    pass

def grad():
    '''takes a list of tensors or np arrays; checks type.'''
    pass

def grad__sq():
    '''takes a list of tensors or np arrays; checks type.'''
    pass

def conj():
    pass