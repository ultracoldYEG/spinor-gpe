Introduction
============
``spinor-gpe`` is high-level, object-oriented Python package for numerically solving the quasi-2D, psuedospinor (two component) Gross-Piteavskii equation (GPE), for both ground state solutions and real-time dynamics. While this package is primarily built on NumPy, computational heavy-lifting is done with PyTorch, a deep-neural network library commonly used in machine learning applications. This implementation provide significant hardware acceleration of of GPE solutions using NVIDIA(R) graphics processing units (GPUs).

This project grew out of a desire to make high-performance GPU simulations of the GPE more accessible to the entering researcher.