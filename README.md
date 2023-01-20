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

Nutils is platform independent and is known to work on Linux, Windows and macOS.

A working installation of Python 3.5 or higher is required. Many different
installers exist and there are no known issues with any of them. When in doubt
about which to use, a safe option is to go with the [official installer][2].

With Python installed, the recommended way to install the latest stable version
of Nutils is through [pip][4] (included in the standard installer):

    python3 -m pip install --user nutils

By default and without explicitly specifying the source of the given packages,
pip installs packages from the [Python Package Index][5]. To install the latest
development version of Nutils, pass a zip of branch master of the [official
repository][3] to pip:

    python3 -m pip install --user https://github.com/evalf/nutils/archive/master.zip

To view which version of Nutils is currently installed, run:

    python3 -m pip show nutils

Nutils can be installed in a Windows machine using WSL environment.
If you want to assemble matrices in parallel using nutils on a Windows machine, 
then WSL is the way to go. Instructions to setup WSL are available [here][13]. 
After setting up WSL, nutils can be installed using the above instructions. 

First steps
-----------

To confirm Nutils and its dependencies are installed correctly, try to run the
Laplace example or any of the other examples included in this repostitory. Make
sure to use the same version of an example as the version of Nutils that is
currently installed.

When running an example from a terminal or editor with Python console, log
messages should appear in the terminal or console. Simulateneously, a html file
`log.html` and any produced figures are written to
`public_html/<script_name>/yyyy/mm/dd/hh-mm-ss` in the home directory. In case a
webserver is running and configured for user directories this automatically
makes simulations remotely accessible. For convenience, `public_html/log.html`
always redirects to the most recent simulation.


Docker
------

[Docker][10] container images with the latest and recent stable versions of
Nutils preinstalled are available from [`ghcr.io/evalf/nutils`][11]. The
container images includes all examples in this repository. To run an example,
add the name of the example and any additional arguments to the command line.
For example, you can run example `laplace` using the latest version of Nutils
with

    docker run --rm -it ghcr.io/evalf/nutils:latest laplace

HTML log files are generated in the `/log` directory of the container. If
you want to store the log files in `/path/to/log` on the
host, add `-v /path/to/log:/log` to the command line before the
name of the image. Extending the previous example:

    docker run --rm -it -v /path/to/log:/log ghcr.io/evalf/nutils:latest laplace

To run a Python script in this container, bind mount the directory
containing the script, including all files necessary to run the script,
to `/app` in the container and add the relative path to the script and
any arguments to the command line. For example, you can run
`/path/to/script/example.py` with Docker using

    docker run --rm -it -v /path/to/script:/app:ro ghcr.io/evalf/nutils:latest example.py


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
[3]: https://github.com/evalf/nutils
[4]: https://github.com/pypa/pip
[5]: https://pypi.org/project/nutils/
[6]: http://docs.nutils.org/en/latest/examples/
[7]: http://docs.nutils.org/en/latest/nutils/
[8]: https://matrix.to/#/#nutils-users:matrix.org
[9]: https://doi.org/10.5281/zenodo.822369
[10]: https://www.docker.com/
[11]: https://github.com/orgs/evalf/packages/container/package/nutils
[12]: https://raw.githubusercontent.com/evalf/nutils/master/examples/laplace.py
[13]: https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview
