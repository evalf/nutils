"""
The matrix module defines an abstract :class:`Matrix` object and several
implementations.  Matrix objects support basic addition and subtraction
operations and provide a consistent insterface for solving linear systems.
Matrices can be converted into other forms suitable for external processing via
the ``export`` method.
"""

from .. import _util as util, sparse, warnings, numeric
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


def assemble_csr(values, rowptr, colidx, ncols):
    '''Create sparse matrix from CSR sparse data.

    Parameters
    ----------
    values
        Matrix values in the row and column positions of the subsequent
        parameters; one dimensional array of any data type.
    rowptr
        Compressed row indices; one dimensonal integer array of a length one up
        from the number of matrix rows.
    colidx
        Column indices; one dimensional integer array of the same length as the
        values parameter.
    ncols
        Number of matrix columns.
    '''

    values = numpy.asarray(values)
    rowptr = numpy.asarray(rowptr)
    colidx = numpy.asarray(colidx)
    ncols = ncols.__index__()
    if not values.ndim == 1:
        raise MatrixError('assemble received invalid values')
    if not (rowptr.ndim == 1 and
            rowptr.dtype.kind in 'ui' and
            rowptr[0] == 0 and
            all(rowptr[1:] >= rowptr[:-1]) and
            rowptr[-1] == len(values)):
        raise MatrixError('assemble received invalid row indices')
    if not (colidx.ndim == 1 and
            colidx.dtype.kind in 'ui' and
            len(colidx) == rowptr[-1] and
            all(colidx < ncols)):
        raise MatrixError('assemble received invalid column indices')
    colidx_is_increasing = numpy.empty((len(colidx)+1,), bool)
    numpy.greater_equal(colidx[1:], colidx[:-1], out=colidx_is_increasing[1:-1])
    colidx_is_increasing[rowptr] = True
    if not colidx_is_increasing.all():
        raise MatrixError('column indices are not stricty increasing')
    return backend.current.assemble(values, rowptr, colidx, ncols)


def assemble_coo(values, rowidx, nrows, colidx, ncols):
    '''Create sparse matrix from COO sparse data.

    Parameters
    ----------
    values
        Matrix values in the row and column positions of the subsequent
        parameters; one dimensional array of any data type.
    rowptr
        Row indices; one dimensonal integer array of the same length as the
        values parameter.
    nrows
        Number of matrix rows.
    colidx
        Column indices; one dimensional integer array of the same length as the
        values parameter.
    ncols
        Number of matrix columns.
    '''

    return assemble_csr(values, numeric.compress_indices(rowidx, nrows), colidx, ncols)


def assemble(data, index, shape):
    # for backwards compatibility
    rowidx, colidx = index
    nrows, ncols = shape
    return assemble_coo(data, rowidx, nrows, colidx, ncols)


def fromsparse(data, inplace=False):
    indices, values, shape = sparse.extract(sparse.prune(sparse.dedup(data, inplace=inplace), inplace=True))
    return assemble(values, indices, shape)


def empty(shape):
    return assemble(data=numpy.empty([0], dtype=float), index=numpy.empty([len(shape), 0], dtype=int), shape=shape)


def diag(d):
    assert d.ndim == 1
    return assemble(d, index=numpy.arange(len(d))[numpy.newaxis].repeat(2, axis=0), shape=d.shape*2)


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
