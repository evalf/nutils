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

from .. import numeric
import abc, treelog, functools, numpy, itertools

class MatrixError(Exception):
  '''
  General error message for matrix-related failure.
  '''

class BackendNotAvailable(MatrixError):
  '''
  Error message reporting that the selected matrix backend is not available on
  the system.
  '''

class ToleranceNotReached(MatrixError):
  '''
  Error message reporting that the configured linear solver tolerance was not
  reached. The ``.best`` attribute carries the non-conforming solution.
  '''
  def __init__(self, best):
    super().__init__('solver failed to reach tolerance')
    self.best = best

class Matrix:
  'matrix base class'

  def __init__(self, shape):
    assert len(shape) == 2
    self.shape = shape
    self._precon_name = None

  def __reduce__(self):
    from . import assemble
    data, index = self.export('coo')
    return assemble, (data, index, self.shape)

  @abc.abstractmethod
  def __add__(self, other):
    'add two matrices'

    raise NotImplementedError

  @abc.abstractmethod
  def __mul__(self, other):
    'multiply matrix with a scalar'

    raise NotImplementedError

  @abc.abstractmethod
  def __matmul__(self, other):
    'multiply matrix with a dense tensor'

    raise NotImplementedError

  @abc.abstractmethod
  def __neg__(self):
    'negate matrix'

    raise NotImplementedError

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

    raise NotImplementedError

  @property
  def size(self):
    return numpy.prod(self.shape)

  def rowsupp(self, tol=0):
    'return row indices with nonzero/non-small entries'

    data, (row, col) = self.export('coo')
    supp = numpy.zeros(self.shape[0], dtype=bool)
    supp[row[abs(data) > tol]] = True
    return supp

  @treelog.withcontext
  def solve(self, rhs=None, *, lhs0=None, constrain=None, rconstrain=None, solver='direct', atol=0., rtol=0., **solverargs):
    '''Solve system given right hand side vector and/or constraints.

    Args
    ----
    rhs : :class:`float` vector or :any:`None`
        Right hand side vector. `None` implies all zeros.
    lhs0 : class:`float` vector or :any:`None`
        Initial values. `None` implies all zeros.
    constrain : :class:`float` or :class:`bool` array, or :any:`None`
        Column constraints. For float values, a number signifies a constraint,
        NaN signifies a free dof. For boolean, a True value signifies a
        constraint to the value in `lhs0`, a False value signifies a free dof.
        `None` implies no constraints.
    rconstrain : :class:`bool` array or :any:`None`
        Row constrains. A True value signifies a constrains, a False value a free
        dof. `None` implies that the constraints follow those defined in
        `constrain` (by implication the matrix must be square).
    solver : :class:`str`
        Name of the solver algorithm. The set of available solvers depends on
        the type of the matrix (i.e. the active backend), although all matrices
        should implement at least the 'direct' solver.
    **kwargs :
        All remaining arguments are passed on to the selected solver method.

    Returns
    -------
    :class:`numpy.ndarray`
        Left hand side vector.
    '''
    nrows, ncols = self.shape
    if rhs is None:
      rhs = numpy.zeros(nrows)
    if lhs0 is None:
      x = numpy.zeros((ncols,)+rhs.shape[1:])
    else:
      x = numpy.array(lhs0, dtype=float)
      while x.ndim < rhs.ndim:
        x = x[...,numpy.newaxis].repeat(rhs.shape[x.ndim], axis=x.ndim)
      assert x.shape == (ncols,)+rhs.shape[1:]
    if constrain is None:
      J = numpy.ones(ncols, dtype=bool)
    else:
      assert constrain.shape == (ncols,)
      if constrain.dtype == bool:
        J = ~constrain
      else:
        J = numpy.isnan(constrain)
        x[~J] = constrain[~J]
    if rconstrain is None:
      assert nrows == ncols
      I = J
    else:
      assert rconstrain.shape == (nrows,) and constrain.dtype == bool
      I = ~rconstrain
    n = I.sum()
    if J.sum() != n:
      raise MatrixError('constrained matrix is not square: {}x{}'.format(I.sum(), J.sum()))
    b = (rhs - self @ x)[J]
    bnorm = numpy.linalg.norm(b)
    atol = max(atol, rtol * bnorm)
    if bnorm > atol:
      treelog.info('solving {} dof system to {} using {} solver'.format(n, 'tolerance {:.0e}'.format(atol) if atol else 'machine precision', solver))
      try:
        x[J] += getattr(self.submatrix(I, J), 'solve_'+solver)(b, atol=atol, **solverargs)
      except Exception as e:
        raise MatrixError('solver failed with error: {}'.format(e)) from e
      if not numpy.isfinite(x).all():
        raise MatrixError('solver returned non-finite left hand side')
      resnorm = numpy.linalg.norm((rhs - self @ x)[J])
      treelog.info('solver returned with residual {:.0e}'.format(resnorm))
      if resnorm > atol > 0:
        raise ToleranceNotReached(x)
    else:
      treelog.info('skipping solver because initial vector is within tolerance')
    return x

  def solve_leniently(self, *args, **kwargs):
    '''
    Identical to :func:`nutils.matrix.Matrix.solve`, but emit a warning in case
    tolerances are not met rather than an exception, while returning the
    obtained solution vector.
    '''
    try:
      return self.solve(*args, **kwargs)
    except ToleranceNotReached as e:
      treelog.warning(e)
      return e.best

  def submatrix(self, rows, cols):
    '''Create submatrix from selected rows, columns.

    Args
    ----
    rows : :class:`bool`/:class:`int` array selecting rows for keeping
    cols : :class:`bool`/:class:`int` array selecting columns for keeping

    Returns
    -------
    :class:`Matrix`
        Matrix instance of reduced dimensions
    '''

    rows = numeric.asboolean(rows, self.shape[0])
    cols = numeric.asboolean(cols, self.shape[1])
    if rows.all() and cols.all():
      return self

    return self._submatrix(rows, cols)

  @abc.abstractmethod
  def _submatrix(self, rows, cols):
    raise NotImplementedError

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

  def diagonal(self):
    nrows, ncols = self.shape
    if nrows != ncols:
      raise MatrixError('failed to extract diagonal: matrix is not square')
    data, indices, indptr = self.export('csr')
    diag = numpy.empty(nrows)
    for irow in range(nrows):
      icols = indices[indptr[irow]:indptr[irow+1]]
      idiag = numpy.searchsorted(icols, irow)
      diag[irow] = data[indptr[irow]+idiag] if idiag < len(icols) and icols[idiag] == irow else 0
    return diag

  def precon_diag(self):
    diag = self.diagonal()
    if not diag.all():
      raise MatrixError("building 'diag' preconditioner: diagonal has zero entries")
    return numpy.reciprocal(diag).__mul__

  def getprecon(self, name):
    if not isinstance(name, str):
      raise MatrixError('invalid preconditioner {!r}'.format(name))
    if self.shape[0] != self.shape[1]:
      raise MatrixError('matrix must be square')
    if self._precon_name != name:
      treelog.info('creating {} solver'.format(name))
      self._precon = getattr(self, 'precon_'+name)()
    return self._precon

  def __repr__(self):
    return '{}<{}x{}>'.format(type(self).__qualname__, *self.shape)

def refine_to_tolerance(solve):
  @functools.wraps(solve)
  def wrapped(self, rhs, atol, **kwargs):
    lhs = solve(self, rhs, **kwargs)
    res = rhs - self @ lhs
    resnorm = numpy.linalg.norm(res)
    if not numpy.isfinite(resnorm) or resnorm <= atol:
      return lhs
    with treelog.iter.plain('refinement iteration', itertools.count(start=1)) as count:
      for iiter in count:
        newlhs = lhs + solve(self, res, **kwargs)
        newres = rhs - self @ newlhs
        newresnorm = numpy.linalg.norm(newres)
        if not numpy.isfinite(resnorm) or newresnorm >= resnorm:
          treelog.debug('residual increased to {:.0e} (discarding)'.format(newresnorm))
          return lhs
        treelog.debug('residual decreased to {:.0e}'.format(newresnorm))
        lhs, res, resnorm = newlhs, newres, newresnorm
        if resnorm <= atol:
          return lhs
  return wrapped

# vim:sw=2:sts=2:et
