<img src="docs/_static/test_logo.png" width="500px">

A python package simulating the quasi-2D pseudospin-1/2 Gross-Pitaevskii equation with NVIDIA GPU acceleration.

## Dependencies:
* Python 3.7.1
* Numpy 1.15.4
* Matplotlib 3.0.2
* PyTorch


## Getting Started:
1. Create a Python environment with the above listed dependencies.
    1. To install a version PyTorch, you also need to install a 
    corresponding version of CUDA. [This page](https://pytorch.org/get-started/locally/)
    gives installation instructions for PyTorch. The installation may take a while.
    1. Verify that Pytorch was installed correctly. You should be able to import it:
    ```
    >>>import torch
    >>>x = torch.rand(5, 3)
    >>>print(x)
    ```
    Also, if you have an NVIDIA GPU, you can test that it is accessible to the interpreter:
    ```
    >>>torch.cuda.is_available()
    True
    ```
1. Clone the `spinor-gpe` repository, and navigate to the directory.
1. Start playing around with some of the test files. Sit back, have fun, and enjoy the GPU acceleration!

Further documentation is available \<in the docstrings\>.