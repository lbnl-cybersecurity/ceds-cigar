..    include:: <isonum.txt>
.. contents:: Table of contents

Local Installation of PyCIGAR
==================

To get PyCIGAR running, you need two things: PyCIGAR, Simulator (e.g. OpenDSS,
Gridlab-D & OMF...) and a reinforcement learning library (RLlib).

**It is highly recommended that users install**
`Anaconda <https://www.anaconda.com/download>`_ **or**
`Miniconda <https://conda.io/miniconda.html>`_
**for Python and the setup instructions will assume that you are
doing so.**

Installing PyCIGAR
------------------------

In this section we install PyCIGAR.

If you have not done so already, download the PyCIGAR github repository.

::

    git clone https://github.com/toanngosy/pycigar.git
    cd pycigar

We begin by creating a conda environment and installing PyCIGAR and its
dependencies within the environment. This can be done by running the below
script. Be sure to run the below commands from ``/path/to/pycigar``.

::

    # create a conda environment
    conda env create -f environment.yml
    source activate pycigar

If the conda install fails, you can also install the requirements using pip by calling

::

    # install PyCIGAR within the environment
    pip install -e .

The option `-e` will allow you to modify PyCIGAR module and it will directly change the module
without the need for reinstallation.

Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the above modules have been successfully installed, we can test the
installation by running a few examples. Before trying to run any examples, be
sure to enter your conda environment by typing:

::

    source activate pycigar

Let’s see some traffic action:

::

    python examples/opendss/test_env.py

(Optional) Install Ray RLlib
----------------------------

PyCIGAR has used RLlib as a reinforcement learning library.
First visit <https://github.com/ray-project/ray/blob/master/doc/source/installation.rst> and
install the required packages.

If you are not intending to develop RL algorithms or customize rllib you don't need to do anything,
Ray was installed when you created the conda environment.

If you are intending to modify Ray, the installation process for this library is as follows:

::

    cd ~
    git clone https://github.com/ray-project/ray.git
    cd ray/python/
    python setup.py develop

If missing libraries cause errors, please also install additional
required libraries as specified at
<http://ray.readthedocs.io/en/latest/installation.html> and
then follow the setup instructions.

Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

See `getting started with RLlib <http://ray.readthedocs.io/en/latest/rllib.html#getting-started>`_ for sample commands.

To run any of the RL examples, make sure to run

::

    source activate pycigar

In order to test run an PyCIGAR experiment in RLlib, try the following command:

::

    python examples/rllib/test_rllib.py

If it does not fail, this means that you have PyCIGAR properly configured with
RLlib.

To visualize the training progress:

::

    tensorboard --logdir=~/ray_results

If tensorboard is not installed, you can install with pip:

::

    pip install tensorboard

For information on how to deploy a cluster, refer to the `Ray instructions <http://ray.readthedocs.io/en/latest/autoscaling.html>`_.
The basic workflow is running the following locally, ssh-ing into the host machine, and starting
jobs from there.

::

    pip install boto3
    ray create-or-update scripts/ray_autoscale.yaml
    ray teardown scripts/ray_autoscale.yaml

