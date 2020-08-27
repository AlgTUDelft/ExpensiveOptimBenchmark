.. _dependency-management:

Dependency Management
=====================

When adding new approaches (:ref:`adding-approaches`) or new problems (:ref:`adding-problems`), a common requirement is to add new python libraries into the environment.

Furthermore, when running experiments it is required to have an environment defined in which all dependencies are installed.

This project utilizes `Poetry <https://python-poetry.org/>`__ to handle the configuration and management of these environments.

In addition, the requirements in the container are installed from the ``requirements.txt`` file. Which can be created/updated by poetry by running::

    poetry export -E julia -E smac -f requirements.txt > requirements.txt

A short summary of the most important commands of poetry is as follows. See also `Poetry's CLI Documentation <https://python-poetry.org/docs/cli/>`__.

.. highlight:: shell

**Setting up the environment**::

    poetry install

**Updating the environment**::

    poetry update

**Adding a new dependency** (via PyPI or git respectively)::

    poetry add pendulum
    poetry add git+https://github.com/sdispater/pendulum.git#develop 

**Running a single command inside the environment**::

    poetry run python ...

**Opening a shell (command prompt) inside the environment**::

    poetry shell

Docker and Singularity
----------------------

Not all dependencies are neccesarily restricted to Python. For example the CFD Simulator OpenFOAM is a binary that cannot be installed though poetry.

Take a look at ``CFD.Dockerfile`` and ``CFD.Singularity`` for the specifications of these containers.

Docker
^^^^^^

.. note:: Windows does not use sudo as a prefix.

.. warning:: 
    Note that any files written inside the docker container, stay in the container, unless the directory is mounted in the host filesystem.

    If you want to export any files, for example the results of any experiments ran, `create a bind mount <https://docs.docker.com/storage/bind-mounts/>`__.


**Building the container**::

    sudo docker build -t cfdbench . -f ./CFD.Dockerfile

**Running a command inside the container**::

    sudo docker run cfdbench ...

**Obtaining a shell inside the container**::

    sudo docker run -it cfdbench bash

Singularity
^^^^^^^^^^^

.. note:: Singularity only works on Linux

.. note::
    Unlike docker, Singularity by default mounts the current users' home directory and the current working directory.

.. warning:: 
    Singularity's filesystem is by default read-only, apart from mounted directories. Any problems or approaches that neccesitate the creation of new files within, may fail due to this.

    Refer to `Persistent Overlays <https://sylabs.io/guides/3.6/user-guide/persistent_overlays.html>`__ in the Singularity documentation for solutions.

**Building the container**::

    sudo singularity build ./CFD.sif ./CFD.Singularity

**Running a command inside the container**::

    singularity run ./CFD.sif ...

**Obtaining a shell inside the container**::

    singularity shell ./CFD.sif.