"""
The matrix module defines an abstract :class:`Matrix` object and several
implementations.  Matrix objects support basic addition and subtraction
operations and provide a consistent insterface for solving linear systems.
Matrices can be converted into other forms suitable for external processing via
the ``export`` method.
"""

from .. import _util as util, sparse, warnings
import numpy
import importlib
import os

from ._base import Matrix, MatrixError, BackendNotAvailable, ToleranceNotReached
for cls in Matrix, MatrixError, BackendNotAvailable, ToleranceNotReached:
    cls.__module__ = __name__  # make it appear as if cls was defined here
del cls  # clean up for sphinx


@util.set_current
@util.defaults_from_env
def backend(matrix: str = 'auto'):
    return importlib.import_module('._'+matrix.lower(), __name__)


def assemble(data, index, shape):
    if not isinstance(data, numpy.ndarray) or data.ndim != 1 or len(index) != 2 or len(shape) != 2:
        raise MatrixError('assemble received invalid input')
    n, = (index[0][1:] <= index[0][:-1]).nonzero()  # index[0][n+1] <= index[0][n]
    if (index[0][n+1] < index[0][n]).any() or (index[1][n+1] <= index[1][n]).any():
        raise MatrixError('assemble input must be sorted')
    return backend.current.assemble(data, index, shape)


def fromsparse(data, inplace=False):
    indices, values, shape = sparse.extract(sparse.prune(sparse.dedup(data, inplace=inplace), inplace=True))
    return backend.current.assemble(values, indices, shape)


def empty(shape):
    return backend.current.assemble(data=numpy.empty([0], dtype=float), index=numpy.empty([len(shape), 0], dtype=int), shape=shape)


def diag(d):
    assert d.ndim == 1
    return backend.current.assemble(d, index=numpy.arange(len(d))[numpy.newaxis].repeat(2, axis=0), shape=d.shape*2)


def eye(n):
    return diag(numpy.ones(n))


def _helper(name):
    warnings.deprecation("matrix.{0}(...) is deprecated; use matrix.backend('{0}', ...) instead".format(name))
    try:
        return backend(name)
    except BackendNotAvailable:
        return None


def Numpy():
    return _helper('Numpy')


def Scipy():
    return _helper('Scipy')


def MKL():
    return _helper('MKL')

# vim:sw=2:sts=2:et
