from ._base import Matrix, MatrixError
from .. import numeric
import numpy
import functools


def assemble(data, index, shape):
    array = numpy.zeros(shape, dtype=data.dtype)
    if len(data):
        array[tuple(index)] = data
    return NumpyMatrix(array)


class NumpyMatrix(Matrix):
    '''matrix based on numpy array'''

    def __init__(self, core):
        assert numeric.isarray(core)
        self.core = core
        super().__init__(core.shape, core.dtype)

    def convert(self, mat):
        if not isinstance(mat, Matrix):
            raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
        if self.shape != mat.shape:
            raise MatrixError('non-matching shapes')
        if isinstance(mat, NumpyMatrix):
            return mat
        return NumpyMatrix(mat.export('dense'))

    def __add__(self, other):
        return NumpyMatrix(self.core + self.convert(other).core)

    def __sub__(self, other):
        return NumpyMatrix(self.core - self.convert(other).core)

    def __mul__(self, other):
        if not numeric.isnumber(other):
            raise TypeError
        return NumpyMatrix(self.core * other)

    def __matmul__(self, other):
        if not isinstance(other, numpy.ndarray):
            raise TypeError
        if other.shape[0] != self.shape[1]:
            raise MatrixError
        return numpy.einsum('ij,j...->i...', self.core, other)

    def __neg__(self):
        return NumpyMatrix(-self.core)

    @property
    def T(self):
        return NumpyMatrix(self.core.T)

    def export(self, form):
        if form == 'dense':
            return self.core
        if form == 'coo':
            ij = self.core.nonzero()
            return self.core[ij], ij
        if form == 'csr':
            rows, cols = self.core.nonzero()
            return self.core[rows, cols], cols, rows.searchsorted(numpy.arange(self.shape[0]+1))
        raise NotImplementedError('cannot export NumpyMatrix to {!r}'.format(form))

    def rowsupp(self, tol=0):
        return numpy.greater(abs(self.core), tol).any(axis=1)

    def _precon_direct(self):
        return functools.partial(numpy.linalg.solve, self.core)

    def _submatrix(self, rows, cols):
        return NumpyMatrix(self.core[numpy.ix_(rows, cols)])

# vim:sw=4:sts=4:et
