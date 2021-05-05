.. image:: docs/_static/test_logo.png
   :align: right

A python package simulating the quasi-2D pseudospin-1/2 Gross-Pitaevskii equation with NVIDIA GPU acceleration.

.. include:: docs/source/intro.rst

Dependencies
############

* Python         3.8.8
* CUDA           11.3
* cuDNN          8.1.1
* NumPy          1.19.2
* Matplotlib     3.3.4
* PyTorch        1.8.1
* tqdm           4.59.0
* scikit-image   0.18.1
* ffmpeg         4.3.1


CUDA Installation
#################

#. Install the NVIDIA CUDA Toolkit.
   `Here <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_ is their installation guide. If you know that the NVIDIA device on your computer is already CUDA-compatible, then you can go directly to the [download page](https://developer.nvidia.com/cuda-downloads) for the most recent version of CUDA. Select the operating system options and installer type. Download the installer and install it via the wizard on the screen. This part may take a while.
#. (Optional?) Download the `cuDNN library <https://developer.nvidia.com/cudnn>`_ corresponding to your CUDA installation. To do this you will need to create an account with NVIDIA and - for ethical purposes - specify for what you will be using the deep neural network library.

    #. Unzip the download file
    #. Move all the folders in the unzipped sub-directory ``/cuda`` to the ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`` directory (for Windows).

#. (May be optional) Download the correct drivers for your NVIDIA device. These can be accessed at `this site <https://www.nvidia.com/Download/index.aspx>`_. Once the driver is installed, you will have on your computer the NVIDIA Control Panel.


PyTorch Installation & Getting Started
######################################

#. With ``conda`` or `pip`, create a Python environment with the above-listed dependencies.

    #. Install the version of PyTorch corresponding to your CUDA installation. `This page <https://pytorch.org/get-started/locally/>`_ gives installation instructions for PyTorch.
    #. Verify that Pytorch was installed correctly. You should be able to import it:

       .. code-block:: python

          >>>import torch
          >>>x = torch.rand(5, 3)
          >>>print(x)
          tensor([[0.2757, 0.3957, 0.9074],
                  [0.6304, 0.1279, 0.7565],
                  [0.0946, 0.7667, 0.2934],
                  [0.9395, 0.4782, 0.9530],
                  [0.2400, 0.0020, 0.9569]])



       Also, if you have an NVIDIA GPU, you can test that it is available for GPU computing:

       .. code-block:: python

          >>>torch.cuda.is_available()
          True

#. Clone the ``spinor-gpe`` repository, and navigate to the directory.
#. Start playing around with some of the test files. Sit back, have fun, and enjoy the GPU acceleration!

Further documentation is available \<in the docstrings\>