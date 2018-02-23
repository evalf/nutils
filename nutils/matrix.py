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

"""
The matrix module defines a number of 2D matrix objects, notably the
:func:`ScipyMatrix` and :func:`NumpyMatrix`. Matrix objects support basic
addition and subtraction operations and provide a consistent insterface for
solving linear systems. Matrices can be converted into other forms suitable for
external processing via the ``export`` method.
"""

from . import numpy, log, numeric, warnings
import abc


class Backend(metaclass=abc.ABCMeta):
  'backend base class'

  def __enter__(self):
    if hasattr(self, '_old_backend'):
      raise RuntimeError('This context manager is not reentrant.')
    global _current_backend
    self._old_backend = _current_backend
    _current_backend = self
    return self

  def __exit__(self, etype, value, tb):
    if not hasattr(self, '_old_backend'):
      raise RuntimeError('This context manager is not yet entered.')
    global _current_backend
    _current_backend = self._old_backend
    del self._old_backend

  @abc.abstractmethod
  def assemble(self, data, index, shape):
    '''Assemble a (sparse) tensor based on index-value pairs.

    .. Note:: This function is abstract.
    '''

class Matrix(metaclass=abc.ABCMeta):
  'matrix base class'

  def __init__(self, shape):
    assert len(shape) == 2
    self.shape = shape

  @abc.abstractmethod
  def __add__(self, other):
    'add two matrices'

  @abc.abstractmethod
  def __mul__(self, other):
    'multiply matrix with a scalar'

  @abc.abstractmethod
  def __neg__(self, other):
    'negate matrix'

  def __sub__(self, other):
    return self.__add__(-other)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    return self.__mul__(1/other)

  @property
  @abc.abstractmethod
  def T(self):
    'transpose matrix'

  @property
  def size(self):
    return numpy.prod(self.shape)

  def rowsupp(self, tol=0):
    'return row indices with nonzero/non-small entries'

    data, (row, col) = self.export('coo')
    supp = numpy.zeros(self.shape[0], dtype=bool)
    supp[row[data > tol]] = True
    return supp

  @abc.abstractmethod
  def solve(self, rhs=None, constrain=None, lconstrain=None, rconstrain=None):
    'solve system given right hand side vector and/or constraints'

  @abc.abstractmethod
  def submatrix(self, rows, cols):
    '''Create submatrix from selected rows, columns.

    Args
    ----
    rows : boolean/int array selecting rows for keeping
    cols : boolean/int array selecting columns for keeping

    Returns
    -------
    Matrix instance of reduced dimensions
    '''

  def export(self, form):
    '''Export matrix data to any of supported forms.

    Args
    ----
    form : :class:`str`
      - "dense" : return matrix as a single dense array
      - "csr" : return matrix as 3-tuple of (data, indices, indptr)
      - "coo" : return matrix as 2-tuple of (data, (row, col))
    '''

    raise NotImplementedError('cannot export {} to {!r}'.format(self.__class__.__name__, form))

  def toarray(self):
    warnings.deprecation('M.toarray is deprecated; use M.export("dense") instead')
    return self.export('dense')

  def toscipy(self):
    warnings.deprecation('M.toscipy is deprecated; use scipy.sparse.csr_matrix(M.export("csr")) instead')
    return scipy.sparse.csr_matrix(self.export('csr'))


## NUMPY BACKEND

class Numpy(Backend):
  '''matrix backend based on numpy array'''

  def assemble(self, data, index, shape):
    array = numeric.accumulate(data, index, shape)
    return NumpyMatrix(array) if len(shape) == 2 else array

class NumpyMatrix(Matrix):
  '''matrix based on numpy array'''

  def __init__(self, core):
    assert numeric.isarray(core)
    self.core = core
    super().__init__(core.shape)

  def __add__(self, other):
    if not isinstance(other, NumpyMatrix) or self.shape != other.shape:
      return NotImplemented
    return NumpyMatrix(self.core + other.core)

  def __mul__(self, other):
    if not numeric.isnumber(other):
      return NotImplemented
    return NumpyMatrix(self.core * other)

  def __neg__(self):
    return NumpyMatrix(-self.core)

  @property
  def T(self):
    return NumpyMatrix(self.core.T)

  def matvec(self, vec):
    return numpy.dot(self.core, vec)

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

  @log.title
  def solve(self, b=None, constrain=None, lconstrain=None, rconstrain=None):

    if b is None:
      b = numpy.zeros(self.shape[0])
    else:
      b = numpy.asarray(b, dtype=float)
      assert b.ndim == 1, 'right-hand-side has shape {}, expected a vector'.format(b.shape)
      assert b.shape == self.shape[:1]

    x, I, J = parsecons(constrain, lconstrain, rconstrain, self.shape)
    if I.all() and J.all():
      return numpy.linalg.solve(self.core, b)

    data = self.core[I]
    x[J] = numpy.linalg.solve(data[:,J], b[I] - numpy.dot(data[:,~J], x[~J]))
    return x

  def submatrix(self, rows, cols):
    return NumpyMatrix(self.core[numpy.ix_(rows, cols)])


## SCIPY BACKEND

try:
  import scipy.sparse.linalg
except ImportError:
  pass
else:

  class Scipy(Backend):
    '''matrix backend based on scipy's sparse matrices'''

    def assemble(self, data, index, shape):
      if len(shape) < 2:
        return numeric.accumulate(data, index, shape)
      if len(shape) == 2:
        csr = scipy.sparse.csr_matrix((data, index), shape)
        return ScipyMatrix(csr)
      raise Exception('{}d data not supported by scipy backend'.format(len(shape)))

  class ScipyMatrix(Matrix):
    '''matrix based on any of scipy's sparse matrices'''

    def __init__(self, core):
      self.core = core
      super().__init__(core.shape)

    def __add__(self, other):
      if not isinstance(other, ScipyMatrix) or self.shape != other.shape:
        return NotImplemented
      return ScipyMatrix(self.core + other.core)

    def __sub__(self, other):
      if not isinstance(other, ScipyMatrix) or self.shape != other.shape:
        return NotImplemented
      return ScipyMatrix(self.core - other.core)

    def __mul__(self, other):
      if not numeric.isnumber(other):
        return NotImplemented
      return ScipyMatrix(self.core * other)

    def __neg__(self):
      return ScipyMatrix(-self.core)

    def matvec(self, vec):
      return self.core.dot(vec)

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

    @log.title
    def solve(self, rhs=None, constrain=None, lconstrain=None, rconstrain=None, solver='spsolve', tol=0, lhs0=None, callback=None, precon=None, **solverargs):

      lhs, I, J = parsecons(constrain, lconstrain, rconstrain, self.shape)
      A = self.core
      if not I.all():
        A = A[I,:]
      b = (rhs[I] if rhs is not None else 0) - A * lhs
      if not J.all():
        A = A[:,J]

      if not b.any():
        log.info('right hand side is zero')
        return lhs

      x0 = lhs0[J] if lhs0 is not None else numpy.zeros(A.shape[0])
      res0 = numpy.linalg.norm(b - A * x0) / numpy.linalg.norm(b)
      if res0 < tol:
        log.info('initial residual is below tolerance')
        x = x0
      elif solver == 'spsolve':
        log.info('solving system using sparse direct solver')
        x = scipy.sparse.linalg.spsolve(A, b)
      else:
        assert tol, 'tolerance must be specified for iterative solver'
        log.info('solving system using {} iterative solver'.format(solver))
        solverfun = getattr(scipy.sparse.linalg, solver)
        # keep scipy from making things circular by shielding the nature of A
        A = scipy.sparse.linalg.LinearOperator(A.shape, A.__mul__, dtype=float)
        if isinstance(precon, str):
          M = self.getprecon(precon, constrain, lconstrain, rconstrain)
        elif callable(precon):
          M = precon(A)
        elif precon is None: # create identity operator, because scipy's native identity operator has circular references
          M = scipy.sparse.linalg.LinearOperator(A.shape, matvec=lambda x:x, rmatvec=lambda x:x, matmat=lambda x:x, dtype=float)
        else:
          M = precon
        assert isinstance(M, scipy.sparse.linalg.LinearOperator)
        norm = max(numpy.linalg.norm(b), 1) # scipy terminates on minimum of relative and absolute residual
        niter = numpy.array(0)
        def mycallback(arg):
          niter[...] += 1
          # some solvers provide the residual, others the left hand side vector
          res = float(numpy.linalg.norm(b - A * arg) / norm if numpy.ndim(arg) == 1 else arg)
          if callback:
            callback(res)
          with log.context('residual {:.2e} ({:.0f}%)'.format(res, 100. * numpy.log10(res) / numpy.log10(tol) if res > 0 else 0)):
            pass
        x, status = solverfun(A, b, M=M, tol=tol, x0=x0, callback=mycallback, **solverargs)
        assert status == 0, '{} solver failed with status {}'.format(solver, status)
        res = numpy.linalg.norm(b - A * x) / norm
        log.info('solver converged in {} iterations to residual {:.1e}'.format(niter, res))

      lhs[J] = x
      return lhs

    def getprecon(self, name='SPLU', constrain=None, lconstrain=None, rconstrain=None):
      name = name.lower()
      x, I, J = parsecons(constrain, lconstrain, rconstrain, self.shape)
      A = self.core[I,:][:,J]
      assert A.shape[0] == A.shape[1], 'constrained matrix must be square'
      log.info('building {} preconditioner'.format(name))
      if name == 'splu':
        precon = scipy.sparse.linalg.splu(A.tocsc()).solve
      elif name == 'spilu':
        precon = scipy.sparse.linalg.spilu(A.tocsc(), drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None).solve
      elif name == 'diag':
        precon = numpy.reciprocal(A.diagonal()).__mul__
      else:
        raise Exception('invalid preconditioner {!r}'.format(name))
      return scipy.sparse.linalg.LinearOperator(A.shape, precon, dtype=float)

    def submatrix(self, rows, cols):
      return ScipyMatrix(self.core[rows,:][:,cols])


## INTERNALS

def parsecons(constrain, lconstrain, rconstrain, shape):
  'parse constraints'

  I = numpy.ones(shape[0], dtype=bool)
  x = numpy.empty(shape[1])
  x[:] = numpy.nan
  if constrain is not None:
    assert lconstrain is None
    assert rconstrain is None
    assert numeric.isarray(constrain)
    I[:] = numpy.isnan(constrain)
    x[:] = constrain
  if lconstrain is not None:
    assert numeric.isarray(lconstrain)
    x[:] = lconstrain
  if rconstrain is not None:
    assert numeric.isarray(rconstrain)
    I[:] = rconstrain
  J = numpy.isnan(x)
  assert numpy.sum(I) == numpy.sum(J), 'constrained matrix is not square: {}x{}'.format(numpy.sum(I), numpy.sum(J))
  x[J] = 0
  return x, I, J


## MODULE METHODS

_current_backend = Numpy()

def backend(names):
  for name in names.lower().split(','):
    for cls in Backend.__subclasses__():
      if cls.__name__.lower() == name:
        return cls()
  raise RuntimeError('matrix backend {!r} is not available'.format(names))

def assemble(data, index, shape):
  return _current_backend.assemble(data, index, shape)


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
