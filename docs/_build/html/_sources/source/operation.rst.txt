Basic Operation
===============
This package has a simple, object-oriented interface for imaginary- and real-time propagations of the pseudospinor-GPE. While there are other operations and features to this package, all simulations will have the following basic structure:

1. Setup: Data path and PSpinor object
######################################

.. code-block:: python

  >>> import pspinor as spin
  >>> DATA_PATH = '<project_name>/Trial_###'
  >>> ps = spin.PSpinor(DATA_PATH)
  
The program will create a new directory ``DATA_PATH``, in which the data and results from this simulation trial will be saved. If ``DATA_PATH`` is a relative path, as shown above, then the trial data will be located in the ``/data/`` folder. When working with multiple simulation projects, it can be helpful to specify a ``<project_name>`` directory; furthermore, the form ``Trial_###`` is convenient, but not strictly required. 


2. Run: Begin Propagation
#########################
The example below demonstrates imaginary-time propagation. The method ``PSpinor.imaginary`` performs the propagation loop and returns a ``PropResult`` object. This object contains the results, including the final wavefunctions and populations, and analysis and plotting methods (described below).

.. code-block:: python

  >>> DT = 1/50
  >>> N_STEPS = 1000
  >>> DEVICE = 'cuda'  
  >>> res = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=50)
  
For real-time propagation, use the method ``PSpinor.real``.


3. Analyze: Plot the results
############################
``PropResult`` provides several methods for viewing and understanding the final results. The code block below demonstrates several of them:

.. code-block:: python

  >>> res.plot_spins()  # Plots the spin-dependent densities and phases.
  >>> res.plot_total()  # Plots the total densities and phases.
  >>> res.plot_pops()   # Plots the spin populations throughout the propagation. 
  >>> res.make_movie()  # Generates a movie from the sampled wavefunctions.
  
Note that ``PSpinor`` also exposes methods to plot the spin and total densities. These can be used independent of ``PropResult``:

.. code-block:: python

  >>> ps.plot_spins()
  
4. Repeat
#########
Likely you will want to repeat or chain together different segments of this structure. Demonstrations of this are shown in the ``Examples`` gallery.
