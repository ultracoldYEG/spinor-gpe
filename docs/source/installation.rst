Installation
============

Dependencies
############

Main packages:
--------------

* PyTorch > 1.8.0
* cudatoolkit > 11.1
* NumPy

Other packages:
---------------

* matplotlib (visualizing results)
* tqdm (progress messages)
* scikit-image (matrix processing)
* ffmpeg (animation generation)

Installing Dependencies
#######################
The dependencies for ``spinor-gpe`` can be installed manually using ``conda`` into a virtual environment: ::

   conda activate <new_virt_env>
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   conda install numpy matplotlib tqdm scikit-image ffmpeg spyder

.. note:: For more information on installing PyTorch, see its `installation instructions <https://pytorch.org/get-started/locally/>`_ page.

Alternatively, the dependencies can be installed via `pip` and the ``requirements.txt`` file included with this package: ::

   pip install -r /spinor-gpe/requirements.txt

Verify that Pytorch was installed correctly. You should be able to import it:

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

CUDA Installation
#################

CUDA is the API that interfaces with the computing resources on NVIDIA graphics cards, and it can be accessed through the PyTorch package. If your computer has an NVIDIA graphics card, start by verifying that it is CUDA-compatible. `This page <https://developer.nvidia.com/cuda-gpus#compute>`_ lists out the compute capability of many NVIDIA devices. (Note: yours may still be CUDA-compatible even if it is not listed here.)

Given that your graphics card can run CUDA, the following are the steps to install CUDA on a Windows computer:

#. Install the NVIDIA CUDA Toolkit.
   Go to the [CUDA download page](https://developer.nvidia.com/cuda-downloads) for the most recent version. Select the operating system options and installer type. Download the installer and install it via the wizard on the screen. This may take a while. For reference, here is the CUDA Toolkit `installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.
#. (Optional?) Download the `cuDNN library <https://developer.nvidia.com/cudnn>`_ corresponding to your CUDA installation version. To do this you will need to create an account with NVIDIA and - for ethical purposes - specify for what you will be using the deep neural network library. To install:

    #. Unzip the download file
    #. Move all the folders in the unzipped sub-directory ``/cuda`` to the ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`` directory.

#. `Download <https://www.nvidia.com/Download/index.aspx>`_ the correct drivers for your NVIDIA device. Once the driver is installed, you will have the NVIDIA Control Panel installed on your computer.


Getting Started
###############
#. Clone the repository.
#. Navigate to the ``spinor_gpe/examples/`` directory, and start to experiment with the examples there.