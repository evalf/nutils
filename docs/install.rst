Installation
============

Nutils requires a working installation of Python 3.5 or higher. Many different
installers exist and there are no known issues with any of them. When in doubt
about which to use, a safe option is to go with the `official installer
<https://www.python.org/downloads/>`_. From there on Nutils can be installed
following the steps below.

Depending on your system the Python executable may be installed as either
``python`` or ``python3``, or both, not to mention alternative implementations
such as ``pypy`` or ``pyston``. In the following instructions, ``python``
is to be replaced with the relevant executable name.


Installing Nutils
-----------------

Nutils is installed via Python's `Pip <https://pip.pypa.io/en/stable/>`_
package installer, which most Python distributions install by default. In the
following instructions we add the flag ``--user`` for a local installation that
does not require system privileges, which is recommended but not required.

The following command installs the stable version of Nutils from the package
archive, along with its dependencies `Numpy <https://numpy.org/>`_, `Treelog
<https://github.com/evalf/treelog>`_ and `Stringly
<https://github.com/evalf/stringly>`_::

    python -m pip install --user nutils

To install the most recent development version we use Github's ability to
generate zip balls::

    python -m pip install --user --force-reinstall \
      https://github.com/evalf/nutils/archive/refs/heads/master.zip

Alternatively, if the `Git <https://git-scm.com/>`_ version control system is
installed, we can use pip's ability to interact with it directly to install the
same version as follows::

    python -m pip install --user --force-reinstall \
      git+https://github.com/evalf/nutils.git@master

This notation has the advantage that even a specific commit (rather than a
branch) can be installed directly by specifying it after the ``@``.

Finally, if we do desire a checkout of Nutils' source code, for instance to
make changes to it, then we can instruct pip to install directly from the
location on disk::

    git clone https://github.com/evalf/nutils.git
    cd nutils
    python -m pip install --user .

In this scenario it is possible to add the ``--editable`` flag to install
Nutils by reference, rather than by making a copy, which is useful in
situations of active development. Note, however, that pip requires manual
intervention to revert back to a subsequent installation by copy.


Installing a matrix backend
---------------------------

Nutils currently supports three matrix backends: Numpy, Scipy and MKL. Since
Numpy is a primary dependency this backend is always available. Unfortunately
it is also the least performant of the three because of its inability to
exploit sparsity. It is therefore strongly recommended to install one of the
other two backends via the instructions below.

By default, Nutils automatically activates the best available matrix backend:
MKL, Scipy or Numpy, in that order. A consequence of this is that a faulty
installation may easily go unnoticed as Nutils will silently fall back on a
lesser backend. As such, to make sure that the installation was successful it
is recommended to force the backend at least once by setting the
``NUTILS_MATRIX`` environment variable. In Linux::

    NUTILS_MATRIX=MKL python myscript.py

Scipy
~~~~~

The Scipy matrix backend becomes available when `Scipy
<https://www.scipy.org/>`_ is installed, either using the platform's package
manager or via pip::

    python -m pip install --user scipy

In addition to a sparse direct solver, the Scipy backend provides many
iterative solvers such as CG, CGS and GMRES, as well as preconditioners. The
direct solver can optionally be made more performant by additionally installing
the ``scikit-umfpack`` module.

MKL
~~~

Intel's oneAPI Math Kernel Library provides the Pardiso sparse direct solver,
which is easily the most powerful direct solver that is currently supported.
It is installed via the `official instructions
<https://software.intel.com/oneapi/onemkl>`_, or, if applicable, by any of the
steps below.

On a Debian based Linux system (such as Ubuntu) the libraries can be directly
installed via the package manager::

    sudo apt install libmkl-rt

For Fedora or Centos Linux, Intel maintains its own repository that can be
added with the following steps::

    sudo dnf config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
    sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    sudo dnf install intel-mkl
    sudo tee /etc/ld.so.conf.d/mkl.conf << EOF > /dev/null
    /opt/intel/lib/intel64/
    /opt/intel/mkl/lib/intel64/
    EOF
    sudo ldconfig -v


Improving performance
---------------------

While Nutils is not (yet) the fastest tool in its class, with some effort it is
possible to achieve sufficient performance to allow simulations of over a
million degrees of freedom. The matrix backend is the most important thing to
get right, but there are a few other factors that are worth considering.

Enable parallel processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

On multi-core architectures, the most straightforward acceleration path
available is to use parallel assembly, activated using the ``NUTILS_NPROCS``
environment variable. Both Linux and OS X both are supported. Unfortunately,
the feature is currently disabled on Windows as it does not support the
``fork`` system call that is used by the current implementation.

On Windows, the easiest way to enjoy parallel speedup is to make use of the new
Windows Subsystem for Linux (WSL2), which is complete Linux environment running
on top of Windows. To install it simply select one of the many Linux
distributions from the Windows store, such as `Ubuntu 20.04 LTS
<https://www.microsoft.com/store/apps/9n6svws3rx71>`_ or `Debian GNU/Linux
<https://www.microsoft.com/store/apps/9MSVKQC78PK6>`_.

Disable threads
~~~~~~~~~~~~~~~

Many Numpy installations default to using the openBLAS library to provide its
linear algebra routines, which supports multi-threading using the openMP
parallelization standard. While this is useful in general, it is in fact
detrimental in case Nutils is using parallel assembly, in which case the
numerical operations are best performed sequentially. This can be achieved by
setting the ``OMP_NUM_THREADS`` environment variable.

In Linux this can be done permanently by adding the following line to the
shell's configuration file. In Linux this is typically ``~/.bashrc``::

    export OMP_NUM_THREADS=1

The downside to this approach is that multithreading is disabled for all
applications that use openBLAS, not just Nutils. Alternatively in Linux the
setting can be specified one-off in the form of a prefix::

    OMP_NUM_THREADS=1 NUTILS_NPROCS=8 python myscript.py

Consider a faster interpretor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most commonly used Python interpretor is without doubt the `CPython
<https://github.com/python/cpython>`_ reference implementation, but it is not
the only option. Before taking an application in production it may be worth
testing if `other implementations
<https://www.python.org/download/alternatives/>`_ have useful performance
benefits.

One interpretor of note is `Pyston <https://www.pyston.org/>`_, which brings
just-in-time compilation enhancements that in a typical application can yield a
20% speed improvement. After Pyston is installed, Nutils and dependencies can
be installed as before simply replacing ``python`` by ``pyston3``. As packages
will be installed from source some development libraries may need to be
installed, but what is missing can usually be inferred from the error messages.


Using Docker, Podman
--------------------

As an alternative to installing Nutils, it is possible to download a
preinstalled system image with all the above considerations taken care of.
Nutils provides `OCI <https://opencontainers.org/>`_ compatible containers for
all releases, as well as the current developement version, which can be run
using tools such as `Docker <https://www.docker.com/>`_ or `Podman
<https://podman.io/>`_. The images are hosted in `Github's container repository
<https://github.com/evalf/nutils/pkgs/container/nutils>`_.

The container images include all the :ref:`examples`. To run one, add the name
of the example and any additional arguments to the command line. For example,
you can run example ``laplace`` using the latest version of Nutils with::

    docker run --rm -it ghcr.io/evalf/nutils:latest laplace

HTML log files are generated in the ``/log`` directory of the container. If you
want to store the log files in ``/path/to/log`` on the host, add ``-v
/path/to/log:/log`` to the command line before the name of the image. Extending
the previous example::

    docker run --rm -it -v /path/to/log:/log ghcr.io/evalf/nutils:latest laplace

To run a Python script in this container, bind mount the directory containing
the script, including all files necessary to run the script, to ``/app`` in the
container and add the relative path to the script and any arguments to the
command line. For example, you can run ``/path/to/myscript.py`` with Docker
using::

    docker run --rm -it -v /path/to:/app:ro ghcr.io/evalf/nutils:latest myscript.py


Remote Computing
----------------

Computations beyond a certain size are usually moved to a remote computing
facility, typically accessed using tools such as `Secure Shell
<https://en.wikipedia.org/wiki/Secure_Shell>`_ or `Mosh <https://mosh.org>`_,
combined with a terminal multiplexer such as `GNU Screen
<https://www.gnu.org/software/screen/>`_ or `Tmux
<https://github.com/tmux/tmux/wiki>`_. In this scenario it is useful to install
a webserver for remote viewing of the html logs.

The standard ``~/public_html`` output directory is configured with the scenario
in mind, as the `Apache <https://httpd.apache.org/>`_ webserver uses this as
the default `user directory
<https://httpd.apache.org/docs/2.4/howto/public_html.html>`_. As this is
disabled by default, the module needs to be enabled by editing the relevant
configuration file or, in Debian Linux, by using the ``a2enmod`` utility::

    sudo a2enmod userdir

Similar behaviour can be achieved with the `Nginx <https://www.nginx.com/>`_ by
configuring a location pattern in the appropriate server block::

    location ~ ^/~(.+?)(/.*)?$ {
      alias /home/$1/public_html$2;
    }

Finally, the terminal output can be made to show the http address rather than
the local uri by adding the following line to the ``~/.nutilsrc`` configuration
file::

    outrooturi = 'https://mydomain.tld/~myusername/'
