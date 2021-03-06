
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "source\auto_examples\1_ground_state.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_source_auto_examples_1_ground_state.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_source_auto_examples_1_ground_state.py:


Example 1: Ground State
=======================

Starting with the Thomas-Fermi solution, propagate in imaginary time to
reach the ground state. Propagation smooths out the sharp edges
on both components' densities.

Physical Parameters
-------------------
.. topic:: Atom number

    :math:`\quad N_{\rm at} = 100`

.. topic:: Atomic mass, Rubidium-87

    :math:`\quad m = 1.4442 \times 10^{-25}~[\rm kg]`

.. topic:: Trap frequencies

    :math:`\quad (\omega_x, \omega_y, \omega_z) = 2 \pi \times (50, 50, 2000)~[{\rm Hz}]`

    :math:`\quad (\omega_x, \omega_y, \omega_z) = \omega_x \times (1, \gamma, \eta) = (1, 1, 40)~[\omega_x]`

.. topic:: Harmonic oscillator length, x-axis

    :math:`\quad a_x = \sqrt{\hbar / m \omega_x} = 1.525~[{\mu\rm m}]`

.. topic:: 3D scattering length, Rubidium-87

    | :math:`\quad a = 5.313~[{\rm nm}]`

    | :math:`\quad a_{\rm sc} = a / a_x = 0.00348~[a_x]`

.. topic:: Scattering 2D scale

    | :math:`\quad g_{\rm sc}^{2\rm D} = \sqrt{8\pi\eta}~a_{\rm sc} = 0.1105~[\omega_x a_x^2]`

.. topic:: Scattering coupling

    | :math:`\quad (g_{\rm uu}, g_{\rm dd}, g_{\rm ud}) = g_{\rm sc}^{2 \rm D} \times (1, 1, 1.04)~[\omega_x a_x^2]`

.. topic:: Chemical potential

    :math:`\quad \mu = \sqrt{4 N_{\rm at} a_{\rm sc} \gamma \sqrt{\eta / 2 \pi}} = 1.875~[\omega_x]`

.. topic:: Thomas-Fermi radius

    :math:`\quad R_{\rm TF} = \sqrt{2 \mu} = 1.937~[a_x]`

.. topic:: Initial population fractions

    :math:`\quad (p_0, p_1) = (0.5, 0.5)`

.. topic:: Raman wavelength

    :math:`\quad \lambda_L = 790.1~[{\rm nm}]`

Numerical Parameters
--------------------

.. topic:: Number of grid points

    :math:`\quad (N_x, N_y) = (64, 64)`

.. topic:: r-grid half-size

    :math:`\quad (x^{\rm max}, y^{\rm max}) = (8, 8)~[a_x]`

.. topic:: r-grid spacing

    :math:`\quad (\Delta x, \Delta y) = (0.25, 0.25)~[a_x]`

.. topic:: k-grid half-size

    :math:`\quad (k_x^{\rm max}, k_y^{\rm max}) = \pi / (\Delta x, \Delta y)`

    :math:`\quad (k_x^{\rm max}, k_y^{\rm max}) = (12.566, 12.566)~[a_x^{-1}]`

.. topic:: k-grid spacing

    :math:`\quad (\Delta k_x, \Delta k_y) = \pi / (x^{\rm max}, y^{\rm max})`

    :math:`\quad (\Delta k_x, \Delta k_y) = (0.3927, 0.3927)~[a_x^{-1}]`

.. topic:: Time scale

    :math:`\quad \tau_0 = 1 / \omega_x = 0.00318~[{\rm s/rad}]`

    :math:`\quad \tau_0 = 1~[\omega_x^{-1}]`

.. topic:: Time step duration, imaginary

    :math:`\quad \Delta \tau_{\rm im} = 1 / 50~[-i \tau_0]`

.. topic:: Number of time steps, imaginary

    :math:`\quad N_{\rm im} = 100`

.. GENERATED FROM PYTHON SOURCE LINES 102-159

.. code-block:: default

    import os
    import sys
    sys.path.insert(0, os.path.abspath('../..'))  # Adds project root to the PATH

    import numpy as np

    from spinor_gpe.pspinor import pspinor as spin


    # 1. SETUP

    DATA_PATH = 'examples/Trial_011'  # Default data path is in the /data/ folder

    FREQ = 50
    W = 2*np.pi*FREQ
    Y_SCALE = 1
    Z_SCALE = 40.0

    ATOM_NUM = 1e2
    OMEG = {'x': W, 'y': Y_SCALE * W, 'z': Z_SCALE * W}
    G_SC = {'uu': 1, 'dd': 1, 'ud': 1.04}

    ps = spin.PSpinor(DATA_PATH, overwrite=True,  # Initialize PSpinor object
                      atom_num=ATOM_NUM,
                      omeg=OMEG,
                      g_sc=G_SC,
                      pop_frac=(0.5, 0.5),
                      r_sizes=(8, 8),
                      mesh_points=(64, 64))

    ps.coupling_setup(wavel=790.1e-9, kin_shift=False)

    ZOOM = 4  # Zooms the momentum-space density plots by a constant factor

    # Plot real- and momentum-space density & real-space phase of both components
    ps.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)


    # 2. RUN (Imaginary-time)

    DT = 1/50
    N_STEPS = 100
    DEVICE = 'cpu'
    ps.rand_seed = 99999

    # Run propagation loop:
    # - Returns `PropResult` & `TensorPropagator` objects
    res, prop = ps.imaginary(DT, N_STEPS, DEVICE, is_sampling=True, n_samples=50)


    # 3. ANALYZE

    res.plot_spins(rscale=ps.rad_tf, kscale=ps.kL_recoil, zoom=ZOOM)
    res.plot_total(kscale=ps.kL_recoil, zoom=ZOOM)  # Plot total density & phase
    res.plot_pops()  # Plot how the spins' populations evolves
    res.make_movie(rscale=ps.rad_tf, kscale=ps.kL_recoil, play=True, zoom=ZOOM,
                   norm_type='half')


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_source_auto_examples_1_ground_state.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: 1_ground_state.py <1_ground_state.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: 1_ground_state.ipynb <1_ground_state.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
