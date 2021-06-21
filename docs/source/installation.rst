Installation
============

Dependencies
############

Primary packages:
-----------------

1. PyTorch >= 1.8.0
2. cudatoolkit >= 11.1
3. NumPy

Other packages:
---------------

4. matplotlib (visualizing results)
5. tqdm (progress messages)
6. scikit-image (matrix signal processing)
7. ffmpeg = 4.3.1 (animation generation)

Installing Dependencies
#######################
The dependencies for ``spinor-gpe`` can be installed directly into the new ``conda`` virtual environment ``spinor`` using the `environment.yml` file included with the package: ::

   conda env create --file environment.yml

This installation may take a while.

.. note::
   The version of CUDA used in this package does not support macOS. Users on these computers may still install PyTorch and run the examples on their CPU. To install correctly on macOS, remove the `- cudatoolkit=11.1` line from the `environment.yml` file. After installation, you will need to modify the example code to run on the `cpu` device instead of the `cuda` device.


The above dependencies can also be installed manually using ``conda`` into a virtual environment: ::

   conda activate <new_virt_env_name>
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   conda install numpy matplotlib tqdm scikit-image ffmpeg spyder

.. note:: For more information on installing PyTorch, see its `installation instructions page <https://pytorch.org/get-started/locally/>`_.

To verify that Pytorch was installed correctly, you should be able to import it:

.. code-block:: python

  >>> import torch
  >>> x = torch.rand(5, 3)
  >>> print(x)
  tensor([[0.2757, 0.3957, 0.9074],
          [0.6304, 0.1279, 0.7565],
          [0.0946, 0.7667, 0.2934],
          [0.9395, 0.4782, 0.9530],
          [0.2400, 0.0020, 0.9569]])

Also, if you have an NVIDIA GPU, you can test that it is available for GPU computing:

.. code-block:: python

  >>> torch.cuda.is_available()
  True

CUDA Installation
#################

CUDA is the API that interfaces with the computing resources on NVIDIA graphics cards, and it can be accessed through the PyTorch package. If your computer has an NVIDIA graphics card, start by verifying that it is CUDA-compatible. `This page <https://developer.nvidia.com/cuda-gpus#compute>`_ lists out the compute capability of many NVIDIA devices. (Note: yours may still be CUDA-compatible even if it is not listed here.)

Given that your graphics card can run CUDA, the following are the steps to install CUDA on a Windows computer:

#. Install the NVIDIA CUDA Toolkit.
   Go to the `CUDA download page <https://developer.nvidia.com/cuda-downloads>`_ for the most recent version. Select the operating system options and installer type. Download the installer and install it via the wizard on the screen. This may take a while. For reference, here is the Windows CUDA Toolkit `installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.
   
   To test that CUDA is installed, run `which nvcc`, and, if instlled correctly, will return the installation path. Also run `nvcc --version` to verify that the version of CUDA matches the PyTorch CUDA toolkit version (>=11.1).

#. `Download <https://www.nvidia.com/Download/index.aspx>`_ the correct drivers for your NVIDIA device. Once the driver is installed, you will have the NVIDIA Control Panel installed on your computer.

..
   #. (Optional) Download the `cuDNN library <https://developer.nvidia.com/cudnn>`_ corresponding to your CUDA installation version. To do this you will need to create an account with NVIDIA and - for ethical purposes - specify for what you will be using the deep neural network library. To install:

..
       #. Unzip the download file
       #. Move all the folders in the unzipped sub-directory ``/cuda`` to the ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`` directory.


Getting Started
###############
#. Clone the repository.
#. Navigate to the ``spinor_gpe/examples/`` directory, and start to experiment with the examples there.