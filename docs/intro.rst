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

With Python installed, the recommended way to install Nutils is to clone `the
repository <https://github.com/evalf/nutils>`_, followed by an editable
installation using `pip <https://github.com/pypa/pip>`_ (included in the
standard installer)::

    $ git clone https://github.com/evalf/nutils.git
    $ python3 -m pip install --user --editable nutils

This will install Nutils locally along with all dependencies. Afterward a
simple ``git pull`` in the project directory will suffice to update Nutils with
no reinstallation required.

Alternatively it is possible to install Nutils directly::

    $ python3 -m pip install --user nutils

This will download the latest stable version from the `Python Package Index
<https://pypi.org/project/nutils/>`_ and install it along with dependencies.
However, since this installation leaves no access to examples or unit tests, in
the following is is assumed that the former approach was used.


First steps
-----------

A good first step after installing Nutils is to confirm that all unit tests are
passing. With the current working directory at the root of the repository::

    $ python3 -m unittest -b

Note that this might take a long time. After that, try to run any of the
scripts in the examples directory, such as the Laplace problem::

    $ python3 examples/laplace.py

Log messages should appear in the terminal during operation. Simulateneously, a
html file ``log.html`` and any produced figures are written to
``public_html/laplace.py/yyyy/mm/dd/hh-mm-ss`` in the home directory. In case a
webserver is running and configured for user directories this automatically
makes simulations remotely accessible. For convenience,
``public_html/log.html`` always redirects to the most recent simulation.


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
