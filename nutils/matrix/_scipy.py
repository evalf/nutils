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

from ._base import Matrix, MatrixError, BackendNotAvailable
from .. import numeric
import treelog as log
import numpy
try:
  import scipy.sparse.linalg
except ImportError:
  raise BackendNotAvailable('the Scipy matrix backend requires scipy to be installed (try: pip install scipy)')

def assemble(data, index, shape):
  return ScipyMatrix(scipy.sparse.csr_matrix((data, index), shape))

class ScipyMatrix(Matrix):
  '''matrix based on any of scipy's sparse matrices'''

  def __init__(self, core):
    self.core = core
    super().__init__(core.shape, core.dtype)

  def convert(self, mat):
    if not isinstance(mat, Matrix):
      raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
    if self.shape != mat.shape:
      raise MatrixError('non-matching shapes')
    if isinstance(mat, ScipyMatrix):
      return mat
    return ScipyMatrix(scipy.sparse.csr_matrix(mat.export('csr'), self.shape), scipy)

  def __add__(self, other):
    return ScipyMatrix(self.core + self.convert(other).core)

  def __sub__(self, other):
    return ScipyMatrix(self.core - self.convert(other).core)

  def __mul__(self, other):
    if not numeric.isnumber(other):
      raise TypeError
    return ScipyMatrix(self.core * other)

  def __matmul__(self, other):
    if not isinstance(other, numpy.ndarray):
      raise TypeError
    if other.shape[0] != self.shape[1]:
      raise MatrixError
    return self.core * other

  def __neg__(self):
    return ScipyMatrix(-self.core)

  def export(self, form):
    if form == 'dense':
      return self.core.toarray()
    if form == 'csr':
      csr = self.core.tocsr()
      return csr.data, csr.indices, csr.indptr
    if form == 'coo':
      coo = self.core.tocoo()
      return coo.data, (coo.row, coo.col)
    raise NotImplementedError('cannot export NumpyMatrix to {!r}'.format(form))

  @property
  def T(self):
    return ScipyMatrix(self.core.transpose())

  _solver_bicg = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'bicg', atol, **kwargs)
  _solver_bicgstab = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'bicgstab', atol, **kwargs)
  _solver_cg = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'cg', atol, **kwargs)
  _solver_cgs = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'cgs', atol, **kwargs)
  _solver_gmres = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'gmres', atol, callback_type='pr_norm', **kwargs)
  _solver_lgmres = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'lgmres', atol, **kwargs)
  _solver_minres = lambda self, rhs, atol, **kwargs: self._solver_scipy(rhs, 'minres', atol, **kwargs)

  def _solver_scipy(self, rhs, method, atol, callback=None, precon=None, preconargs={}, **solverargs):
    rhsnorm = numpy.linalg.norm(rhs)
    solverfun = getattr(scipy.sparse.linalg, method)
    myrhs = rhs / rhsnorm # normalize right hand side vector for best control over scipy's stopping criterion
    mytol = atol / rhsnorm
    if precon is not None:
      precon = scipy.sparse.linalg.LinearOperator(self.shape, self.getprecon(precon, **preconargs), dtype=float)
    with log.context(method + ' {:.0f}%', 0) as reformat:
      def mycallback(arg):
        # some solvers provide the residual, others the left hand side vector
        res = numpy.linalg.norm(myrhs - self @ arg) if numpy.ndim(arg) == 1 else float(arg)
        if callback:
          callback(res)
        reformat(100 * numpy.log10(max(mytol, res)) / numpy.log10(mytol))
      mylhs, status = solverfun(self.core, myrhs, M=precon, tol=mytol, callback=mycallback, **solverargs)
    if status != 0:
      raise Exception('status {}'.format(status))
    return mylhs * rhsnorm

  def _precon_direct(self):
    return scipy.sparse.linalg.factorized(self.core.tocsc())

  def _precon_splu(self):
    return scipy.sparse.linalg.splu(self.core.tocsc()).solve

  def _precon_spilu(self, **kwargs):
    return scipy.sparse.linalg.spilu(self.core.tocsc(), **kwargs).solve

  def _submatrix(self, rows, cols):
    return ScipyMatrix(self.core[rows,:][:,cols])

  def diagonal(self):
    return self.core.diagonal()

# vim:sw=2:sts=2:et
