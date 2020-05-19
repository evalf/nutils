# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from ._base import Matrix, MatrixError, refine_to_tolerance
from .. import numeric
import numpy

def setassemble(sets):
  return sets(assemble)

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
    super().__init__(core.shape)

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

  @refine_to_tolerance
  def _solver_direct(self, rhs):
    return numpy.linalg.solve(self.core, rhs)

  def _submatrix(self, rows, cols):
    return NumpyMatrix(self.core[numpy.ix_(rows, cols)])

# vim:sw=2:sts=2:et
