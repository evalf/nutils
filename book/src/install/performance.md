# Improving performance

While Nutils is not (yet) the fastest tool in its class, with some effort it is
possible to achieve sufficient performance to allow simulations of over a
million degrees of freedom. The matrix backend is the most important thing to
get right, but there are a few other factors that are worth considering.

## Enable parallel processing

On multi-core architectures, the most straightforward acceleration path
available is to use parallel assembly, activated using the `NUTILS_NPROCS`
environment variable. Both Linux and OS X both are supported. Unfortunately,
the feature is currently disabled on Windows as it does not support the `fork`
system call that is used by the current implementation.

On Windows, the easiest way to enjoy parallel speedup is to make use of the new
Windows Subsystem for Linux (WSL2), which is complete Linux environment running
on top of Windows. To install it simply select one of the many Linux
distributions from the Windows store, such as [Ubuntu 20.04
LTS](https://www.microsoft.com/store/apps/9n6svws3rx71) or [Debian
GNU/Linux](https://www.microsoft.com/store/apps/9MSVKQC78PK6).

## Disable threads

Many Numpy installations default to using the openBLAS library to provide its
linear algebra routines, which supports multi-threading using the openMP
parallelization standard. While this is useful in general, it is in fact
detrimental in case Nutils is using parallel assembly, in which case the
numerical operations are best performed sequentially. This can be achieved by
setting the `OMP_NUM_THREADS` environment variable.

In Linux this can be done permanently by adding the following line to the
shell's configuration file. In Linux this is typically `~/.bashrc`:

```sh
export OMP_NUM_THREADS=1
```

The downside to this approach is that multithreading is disabled for all
applications that use openBLAS, not just Nutils. Alternatively in Linux the
setting can be specified one-off in the form of a prefix::

```sh
OMP_NUM_THREADS=1 NUTILS_NPROCS=8 python myscript.py
```

## Consider a faster interpreter

The most commonly used Python interpreter is without doubt the
[CPython](https://github.com/python/cpython) reference implementation, but it
is not the only option. Before taking an application in production it may be
worth testing if [other
implementations](https://www.python.org/download/alternatives) have useful
performance benefits.

One interpreter of note is [Pyston](https://www.pyston.org), which brings
just-in-time compilation enhancements that in a typical application can yield a
20% speed improvement. After Pyston is installed, Nutils and dependencies can
be installed as before simply replacing `python` by `pyston3`. As packages will
be installed from source some development libraries may need to be installed,
but what is missing can usually be inferred from the error messages.

