# Installing Nutils

Nutils is installed via Python's [Pip](https://pip.pypa.io/en/stable/) package
installer, which most Python distributions install by default. In the following
instructions we add the flag `--user` for a local installation that does not
require system privileges, which is recommended but not required.

The following command installs the stable version of Nutils from the package
archive, along with its dependencies [Numpy](https://numpy.org/),
[Treelog](https://github.com/evalf/treelog) and
[Stringly](https://github.com/evalf/stringly):

```sh
python -m pip install --user nutils
```

To install the most recent development version we use Github's ability to
generate zip balls:

```sh
python -m pip install --user --force-reinstall \
  https://github.com/evalf/nutils/archive/refs/heads/master.zip
```

Alternatively, if the [Git](https://git-scm.com/) version control system is
installed, we can use pip's ability to interact with it directly to install the
same version as follows:

```sh
python -m pip install --user --force-reinstall \
  git+https://github.com/evalf/nutils.git@master
```

This notation has the advantage that even a specific commit (rather than a
branch) can be installed directly by specifying it after the `@`.

Finally, if we do desire a checkout of Nutils' source code, for instance to
make changes to it, then we can instruct pip to install directly from the
location on disk::

```sh
git clone https://github.com/evalf/nutils.git
cd nutils
python -m pip install --user .
```

In this scenario it is possible to add the `--editable` flag to install Nutils
by reference, rather than by making a copy, which is useful in situations of
active development. Note, however, that pip requires manual intervention to
revert back to a subsequent installation by copy.
