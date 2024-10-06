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
    if isinstance(matrix, str):
        return importlib.import_module('._'+matrix.lower(), __name__)
    if hasattr(matrix, 'assemble'):
        return matrix
    raise ValueError('matrix backend does not have an assemble function')


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
            len(colidx) == len(values) and
            all(all(colidx[i+1:j] >= colidx[i:j-1]) for i, j in zip(rowptr, rowptr[1:]) if i != j) and
            all(colidx < ncols)):
        raise MatrixError('assemble received invalid column indices')
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
    warnings.deprecation('matrix.assemble is deprecated, use matrix.assemble_coo instead')
    rowidx, colidx = index
    nrows, ncols = shape
    return assemble_coo(data, rowidx, nrows, colidx, ncols)


def fromsparse(data, inplace=False):
    (rowidx, colidx), values, (nrows, ncols) = sparse.extract(sparse.prune(sparse.dedup(data, inplace=inplace), inplace=True))
    return assemble_coo(values, rowidx, nrows, colidx, ncols)


def empty(shape):
    nrows, ncols = shape
    return assemble_csr(numpy.zeros(0, dtype=float), numpy.zeros(nrows+1, dtype=int), numpy.zeros(0, dtype=int), ncols)


def diag(d):
    n = len(d)
    return assemble_csr(d, numpy.arange(n+1), numpy.arange(n), n)


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
