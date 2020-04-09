Nutils
======

[![Test Status](https://github.com/joostvanzwieten/nutils/workflows/test/badge.svg?branch=master)](https://github.com/evalf/nutils/actions?query=workflow%3Atest+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/evalf/nutils/branch/master/graph/badge.svg)](https://codecov.io/gh/evalf/nutils/branch/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.822369.svg)](https://doi.org/10.5281/zenodo.822369)

Nutils is a Free and Open Source Python programming library for Finite Element
Method computations, developed by [Evalf Computing][1] and distributed under
the permissive MIT license. Key features are a readable, math centric syntax,
an object oriented design, strict separation of topology and geometry, and high
level function manipulations with support for automatic differentiation.

Nutils provides the tools required to construct a typical simulation workflow
in just a few lines of Python code, while at the same time leaving full
flexibility to build novel workflows or interact with third party tools. With
native support for Isogeometric Analysis (IGA), the Finite Cell method (FCM),
multi-physics, mixed methods, and hierarchical refinement, Nutils is at the
forefront of numerical discretization science. Efficient under-the-hood
vectorization and built-in parallellisation provide for an effortless
transition from academic research projects to full scale, real world
applications.


Installation
------------

Nutils is platform independent and is known to work on Linux, Windows and OS X.

A working installation of Python 3.5 or higher is required. Many different
installers exist and there are no known issues with any of them. When in doubt
about which to use, a safe option is to go with the [official installer][2].

With Python installed, the recommended way to install Nutils is to clone [the
repository][3], followed by an editable installation using [pip][4] (included
in the standard installer):

    $ git clone https://github.com/nutils/nutils.git
    $ python3 -m pip install --user --editable nutils

This will install Nutils locally along with all dependencies. Afterward a
simple `git pull` in the project directory will suffice to update Nutils with
no reinstallation required.

Alternatively it is possible to install Nutils directly:

    $ python3 -m pip install --user nutils

This will download the latest stable version from the [Python Package Index][5]
and install it along with dependencies. However, since this installation leaves
no access to examples or unit tests, in the following is is assumed that the
former approach was used.


First steps
-----------

A good first step after installing Nutils is to confirm that all unit tests are
passing. With the current working directory at the root of the repository:

    $ python3 -m unittest -b

Note that this might take a long time. After that, try to run any of the
scripts in the examples directory, such as the Laplace problem:

    $ python3 examples/laplace.py

Log messages should appear in the terminal during operation. Simulateneously, a
html file `log.html` and any produced figures are written to
`public_html/laplace.py/yyyy/mm/dd/hh-mm-ss` in the home directory. In case a
webserver is running and configured for user directories this automatically
makes simulations remotely accessible. For convenience, `public_html/log.html`
always redirects to the most recent simulation.


Next steps and support
----------------------

For the numerical background of all examples as well as line by line
documentation see the [overview of examples][6]. Documentation of individual
functions can be found in the [API reference][7].

Most simulations will have components in common with the example scripts, so a
mix-and-match approach is a good way to start building your own script. For
questions that are not answered by the API reference there is the nutils-users
support channel at [#nutils-users:matrix.org][8]. Note that you will need to
create an account at any Matrix server in order to join this channel.

If you are using Nutils in academic research, please consider [citing
Nutils][9].


[1]: http://evalf.com/
[2]: https://www.python.org/downloads/
[3]: https://github.com/nutils/nutils
[4]: https://github.com/pypa/pip
[5]: https://pypi.org/project/nutils/
[6]: http://docs.nutils.org/en/examples/
[7]: http://docs.nutils.org/en/nutils/
[8]: https://matrix.to/#/#nutils-users:matrix.org
[9]: https://doi.org/10.5281/zenodo.822369
