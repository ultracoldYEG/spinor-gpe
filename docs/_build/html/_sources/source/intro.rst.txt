Introduction
============
``spinor-gpe`` is high-level, object-oriented Python package for numerically solving the quasi-2D, psuedospinor (two component) Gross-Piteavskii equation (GPE), for both ground state solutions and real-time dynamics. This project grew out of a desire to make high-performance simulations of the GPE more accessible to the entering researcher. 

While this package is primarily built on NumPy, the main computational heavy-lifting is performed using PyTorch, a deep neural network library commonly used in machine learning applications. PyTorch has a NumPy-like interface, but a backend that can run either on a conventional processor or a CUDA-enabled NVIDIA(R) graphics card. Accessing a CUDA device will provide a significant hardware acceleration of the simulations.

This package has been tested on Windows, Mac, and Linux systems. 

View the documentation on `ReadTheDocs <https://spinor-gpe.readthedocs.io/en/latest/>`_
