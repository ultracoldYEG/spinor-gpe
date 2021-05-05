Introduction
============
``spinro-gpe`` is high-level, object-oriented Python package for numerically solving the psuedospinor (two spinor components) Gross-Piteavskii equation (GPE) for ground state solutions, or real-time dynamics. While this package is built on NumPy, the actual computations take place using PyTorch, a deep-neural network library commonly used in machine learning applications. This implementations for significant hardware acceleration of the computations using NVIDIA(R) graphics processing units (GPUs).

This project grew out of a desire to make high-performance GPU acceleration more accessible for the entering GPE researcher.

Conditions
**********
Since this is a graduate school project, this package is neither guaranteed to be stable nor maintained.