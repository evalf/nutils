Getting Started
===============

Nutils can be installed via the `Python Package Index
<https://pypi.org/project/nutils/>`_ or cloned from `Github
<https://github.com/evalf/nutils>`_. Once properly configured, the best way to
get going is by reading the :ref:`tutorial` and by studying the :ref:`examples`
that demonstrate implementations of several solid and fluid mechanics problems.


Installation
------------

Nutils is platform independent and is known to work on Linux, Windows and OS X.

A working installation of Python 3.5 or higher is required. Many different
installers exist and there are no known issues with any of them. When in doubt
about which to use, a safe option is to go with the `official installer
<https://www.python.org/downloads/>`_.

With Python installed, the recommended way to install the latest stable version
of Nutils is through `pip <https://github.com/pypa/pip>`_ (included in the
standard installer):

.. code:: sh

    python3 -m pip install --user nutils

By default and without explicitly specifying the source of the given packages,
pip installs packages from the `Python Package Index
<https://pypi.org/project/nutils/>`_. To install the latest development version
of Nutils, pass a zip of branch master of the `official repository
<https://github.com/evalf/nutils>`_ to pip:

.. code:: sh

    python3 -m pip install --user https://github.com/evalf/nutils/archive/master.zip

To view which version of Nutils is currently installed, run:

.. code:: sh

    python3 -m pip show nutils


First steps
-----------

To confirm Nutils and its dependencies are installed correctly, try to run the
Laplace example or any of the other examples included in this repostitory. Make
sure to use the same version of an example as the version of Nutils that is
currently installed.

When running an example from a terminal or editor with Python console, log
messages should appear in the terminal or console. Simulateneously, a html file
``log.html`` and any produced figures are written to
``public_html/<script_name>/yyyy/mm/dd/hh-mm-ss`` in the home directory. In case a
webserver is running and configured for user directories this automatically
makes simulations remotely accessible. For convenience,
``public_html/log.html`` always redirects to the most recent simulation.


Docker
------

`Docker <https://www.docker.com/>`_ container images with the latest and recent
stable versions of Nutils preinstalled are available from `ghcr.io/evalf/nutils
<https://github.com/orgs/evalf/packages/container/package/nutils>`_. The
container images includes all examples in this repository. To run an example,
add the name of the example and any additional arguments to the command line.
For example, you can run example ``laplace`` using the latest version of Nutils
with

.. code:: sh

    docker run --rm -it ghcr.io/evalf/nutils:latest laplace

HTML log files are generated in the ``/log`` directory of the container. If you
want to store the log files in ``/path/to/log`` on the host, add ``-v
/path/to/log:/log`` to the command line before the name of the image. Extending
the previous example:

.. code:: sh

    docker run --rm -it -v /path/to/log:/log ghcr.io/evalf/nutils:latest laplace

To run a Python script in this container, bind mount the directory
containing the script, including all files necessary to run the script,
to ``/app`` in the container and add the relative path to the script and
any arguments to the command line. For example, you can run
``/path/to/script/example.py`` with Docker using

.. code:: sh

    docker run --rm -it -v /path/to/script:/app:ro ghcr.io/evalf/nutils:latest example.py


Next steps and support
----------------------

For the numerical background of all examples as well as line by line
documentation see the overview of :ref:`examples`. Documentation of individual
functions can be found in the :ref:`api_reference`.

Most simulations will have components in common with the example scripts, so a
mix-and-match approach is a good way to start building your own script. For
questions that are not answered by the API reference there is the nutils-users
support channel at `#nutils-users:matrix.org
<https://matrix.to/#/#nutils-users:matrix.org>`_. Note that you will need to
create an account at any Matrix server in order to join this channel.

If you are using Nutils in academic research, please consider `citing
Nutils <https://doi.org/10.5281/zenodo.822369>`_.
