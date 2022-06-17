# Installing a matrix backend

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
`NUTILS_MATRIX` environment variable. In Linux:

```sh
NUTILS_MATRIX=MKL python myscript.py
```

## Scipy

The Scipy matrix backend becomes available when [Scipy](https://www.scipy.org/)
is installed, either using the platform's package manager or via pip:

```sh
python -m pip install --user scipy
```

In addition to a sparse direct solver, the Scipy backend provides many
iterative solvers such as CG, CGS and GMRES, as well as preconditioners. The
direct solver can optionally be made more performant by additionally installing
the `scikit-umfpack` module.

## MKL

Intel's oneAPI Math Kernel Library provides the Pardiso sparse direct solver,
which is easily the most powerful direct solver that is currently supported. It
is installed via the [official
instructions](https://software.intel.com/oneapi/onemkl), or, if applicable, by
any of the steps below.

On a Debian based Linux system (such as Ubuntu) the libraries can be directly
installed via the package manager:

```sh
sudo apt install libmkl-rt
```

For Fedora or Centos Linux, Intel maintains its own repository that can be
added with the following steps:

```sh
sudo dnf config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo dnf install intel-mkl
sudo tee /etc/ld.so.conf.d/mkl.conf << EOF > /dev/null
/opt/intel/lib/intel64/
/opt/intel/mkl/lib/intel64/
EOF
sudo ldconfig -v
```
