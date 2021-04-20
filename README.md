<img src="docs/_static/test_logo.png" width="500px">

A python package simulating the quasi-2D pseudospin-1/2 Gross-Pitaevskii equation with NVIDIA GPU acceleration.

## Dependencies:
* Python      3.8.8
* CUDA        11.3
* cuDNN       8.1.1
* Numpy       1.19.2
* Matplotlib  3.3.4
* PyTorch     1.8.1


## Getting Started:
1. Install the NVIDIA CUDA Toolkit. [Here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) is their installation guide. If you know that the NVIDIA device on your computer is already CUDA-compatible, then you can go directly to the [download page](https://developer.nvidia.com/cuda-downloads) for the most recent version of CUDA. Select the operating system options and installer type. Download the installer and install it via the wizard on the screen. This part may take a while.
1. (May be optional) Download the [cuDNN library](https://developer.nvidia.com/cudnn) corresponding to your CUDA installation. To do this you will need to create an account with NVIDIA and - for ethical purposs - specify for what you will be using the deep neural network library.
    1. Unzip the download file
    1. Move all the folders in the unzipped sub-directory '/cuda' to the `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3` directory (for Windows).
1. (May be optional) Download the corrct drivers for your NVIDIA device. These can be accessed [here](https://www.nvidia.com/Download/index.aspx). Once the driver is installed, you will have on your computer the NVIDIA Control Panel.
1. With `conda` or `pip`, create a Python environment with the above listed dependencies.
    1. Install the version of PyTorch corresponding to your CUDA installation. [This page](https://pytorch.org/get-started/locally/) gives installation instructions for PyTorch.
    1. Verify that Pytorch was installed correctly. You should be able to import it:
    ```
    >>>import torch
    >>>x = torch.rand(5, 3)
    >>>print(x)
    tensor([[0.2757, 0.3957, 0.9074],
            [0.6304, 0.1279, 0.7565],
            [0.0946, 0.7667, 0.2934],
            [0.9395, 0.4782, 0.9530],
            [0.2400, 0.0020, 0.9569]])
    ```
    Also, if you have an NVIDIA GPU, you can test that it is available for GPU computing:
    ```
    >>>torch.cuda.is_available()
    True
    ```
1. Clone the `spinor-gpe` repository, and navigate to the directory.
1. Start playing around with some of the test files. Sit back, have fun, and enjoy the GPU acceleration!

Further documentation is available \<in the docstrings\>.