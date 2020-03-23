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
The matrix module defines an abstract :class:`Matrix` object and several
implementations.  Matrix objects support basic addition and subtraction
operations and provide a consistent insterface for solving linear systems.
Matrices can be converted into other forms suitable for external processing via
the ``export`` method.
"""

from . import numpy, numeric, warnings, cache, types, util
import abc, sys, ctypes, enum, treelog as log, functools, itertools, typing


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


class Backend(metaclass=abc.ABCMeta):
  'backend base class'

  def __enter__(self):
    if hasattr(self, '_old_backend'):
      raise RuntimeError('This context manager is not reentrant.')
    if not self:
      raise BackendNotAvailable
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

class Matrix(metaclass=types.CacheMeta):
  'matrix base class'

  def __init__(self, shape):
    assert len(shape) == 2
    self.shape = shape

  def __reduce__(self):
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

  def matvec(self, vec):
    warnings.deprecation('A.matvec(x) is deprecated; use A @ x instead')
    return self.__matmul__(vec)

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

  @log.withcontext
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
      log.info('solving {} dof system to {} using {} solver'.format(n, 'tolerance {:.0e}'.format(atol) if atol else 'machine precision', solver))
      try:
        x[J] += getattr(self.submatrix(I, J), 'solve_'+solver)(b, atol=atol, **solverargs)
      except Exception as e:
        raise MatrixError('solver failed with error: {}'.format(e)) from e
      if not numpy.isfinite(x).all():
        raise MatrixError('solver returned non-finite left hand side')
      resnorm = numpy.linalg.norm((rhs - self @ x)[J])
      log.info('solver returned with residual {:.0e}'.format(resnorm))
      if resnorm > atol > 0:
        raise ToleranceNotReached(x)
    else:
      log.info('skipping solver because initial vector is within tolerance')
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
      log.warning(e)
      return e.best

  @abc.abstractmethod
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

    raise NotImplementedError


    rows = numeric.asboolean(rows, self.shape[0])
    cols = numeric.asboolean(cols, self.shape[1])
    if rows.all() and cols.all():
      return self
    data, (I,J) = self.export('coo')
    keep = numpy.logical_and(rows[I], cols[J])
    csI = rows.cumsum()
    csJ = cols.cumsum()
    return assemble(data[keep], numpy.array([csI[I[keep]]-1, csJ[J[keep]]-1]), shape=(csI[-1], csJ[-1]))

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
    with log.iter.plain('refinement iteration', itertools.count(start=1)) as count:
      for iiter in count:
        newlhs = lhs + solve(self, res, **kwargs)
        newres = rhs - self @ newlhs
        newresnorm = numpy.linalg.norm(newres)
        if not numpy.isfinite(resnorm) or newresnorm >= resnorm:
          log.debug('residual increased to {:.0e} (discarding)'.format(newresnorm))
          return lhs
        log.debug('residual decreased to {:.0e}'.format(newresnorm))
        lhs, res, resnorm = newlhs, newres, newresnorm
        if resnorm <= atol:
          return lhs
  return wrapped

## NUMPY BACKEND

class Numpy(Backend):
  '''matrix backend based on numpy array'''

  @staticmethod
  def assemble(data, index, shape):
    array = numeric.accumulate(data, index, shape)
    return NumpyMatrix(array) if len(shape) == 2 else array

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
  def solve_direct(self, rhs):
    return numpy.linalg.solve(self.core, rhs)

  def submatrix(self, rows, cols):
    return NumpyMatrix(self.core[numpy.ix_(rows, cols)])


## SCIPY BACKEND

class Scipy(Backend):
  '''matrix backend based on scipy's sparse matrices'''

  def __init__(self):
    try:
      import scipy.sparse.linalg
    except ImportError:
      self.scipy = None
    else:
      self.scipy = scipy

  def __bool__(self):
    return self.scipy is not None

  def assemble(self, data, index, shape):
    if len(shape) < 2:
      return numeric.accumulate(data, index, shape)
    if len(shape) == 2:
      csr = self.scipy.sparse.csr_matrix((data, index), shape)
      return ScipyMatrix(csr, scipy=self.scipy)
    raise MatrixError('{}d data not supported by scipy backend'.format(len(shape)))

class ScipyMatrix(Matrix):
  '''matrix based on any of scipy's sparse matrices'''

  def __init__(self, core, scipy):
    self.core = core
    self.scipy = scipy
    super().__init__(core.shape)

  def convert(self, mat):
    if not isinstance(mat, Matrix):
      raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
    if self.shape != mat.shape:
      raise MatrixError('non-matching shapes')
    if isinstance(mat, ScipyMatrix):
      return mat
    return ScipyMatrix(self.scipy.sparse.csr_matrix(mat.export('csr'), self.shape), self.scipy)

  def __add__(self, other):
    return ScipyMatrix(self.core + self.convert(other).core, scipy=self.scipy)

  def __sub__(self, other):
    return ScipyMatrix(self.core - self.convert(other).core, scipy=self.scipy)

  def __mul__(self, other):
    if not numeric.isnumber(other):
      raise TypeError
    return ScipyMatrix(self.core * other, scipy=self.scipy)

  def __matmul__(self, other):
    if not isinstance(other, numpy.ndarray):
      raise TypeError
    if other.shape[0] != self.shape[1]:
      raise MatrixError
    return self.core * other

  def __neg__(self):
    return ScipyMatrix(-self.core, scipy=self.scipy)

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
    return ScipyMatrix(self.core.transpose(), scipy=self.scipy)

  @refine_to_tolerance
  def solve_direct(self, rhs):
    return self.scipy.sparse.linalg.spsolve(self.core, rhs)

  def solve_scipy(self, rhs, solver, atol, callback=None, precon=None, **solverargs):
    rhsnorm = numpy.linalg.norm(rhs)
    solverfun = getattr(self.scipy.sparse.linalg, solver)
    myrhs = rhs / rhsnorm # normalize right hand side vector for best control over scipy's stopping criterion
    mytol = atol / rhsnorm
    M = self.getprecon(precon) if isinstance(precon, str) else precon(self.core) if callable(precon) else precon
    with log.context(solver + ' {:.0f}%', 0) as reformat:
      def mycallback(arg):
        # some solvers provide the residual, others the left hand side vector
        res = numpy.linalg.norm(myrhs - self @ arg) if numpy.ndim(arg) == 1 else float(arg)
        if callback:
          callback(res)
        reformat(100 * numpy.log10(max(mytol, res)) / numpy.log10(mytol))
      mylhs, status = solverfun(self.core, myrhs, M=M, tol=mytol, callback=mycallback, **solverargs)
    if status != 0:
      raise Exception('status {}'.format(status))
    return mylhs * rhsnorm

  solve_bicg     = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'bicg',     **kwargs)
  solve_bicgstab = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'bicgstab', **kwargs)
  solve_cg       = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'cg',       **kwargs)
  solve_cgs      = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'cgs',      **kwargs)
  solve_gmres    = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'gmres',    **kwargs)
  solve_lgmres   = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'lgmres',   **kwargs)
  solve_minres   = lambda self, rhs, **kwargs: self.solve_scipy(rhs, 'minres',   **kwargs)

  def getprecon(self, name):
    name = name.lower()
    assert self.shape[0] == self.shape[1], 'constrained matrix must be square'
    log.info('building {} preconditioner'.format(name))
    if name == 'splu':
      try:
        precon = self.scipy.sparse.linalg.splu(self.core.tocsc()).solve
      except RuntimeError as e:
        raise MatrixError(e) from e
    elif name == 'spilu':
      try:
        precon = self.scipy.sparse.linalg.spilu(self.core.tocsc(), drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None).solve
      except RuntimeError as e:
        raise MatrixError(e) from e
    elif name == 'diag':
      diag = self.core.diagonal()
      if not diag.all():
        raise MatrixError("building 'diag' preconditioner: diagonal has zero entries")
      precon = numpy.reciprocal(diag).__mul__
    else:
      raise MatrixError('invalid preconditioner {!r}'.format(name))
    return self.scipy.sparse.linalg.LinearOperator(self.shape, precon, dtype=float)

  def submatrix(self, rows, cols):
    return ScipyMatrix(self.core[rows,:][:,cols], scipy=self.scipy)

  def diagonal(self):
    return self.core.diagonal()


## INTEL MKL BACKEND

c_int = types.c_array[numpy.int32]
c_long = types.c_array[numpy.int64]
c_double = types.c_array[numpy.float64]

class MKL(Backend):
  '''matrix backend based on Intel's Math Kernel Library'''

  class Threading(enum.Enum):
    INTEL = 0
    SEQUENTIAL = 1
    PGI = 2
    GNU = 3
    TBB = 4

  def __init__(self, threading:typing.Optional[Threading]=None):
    self.current_threading = -1
    self.threading = threading
    self.libmkl = util.loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')

  def __bool__(self):
    return self.libmkl is not None

  def __enter__(self):
    super().__enter__()
    threading = self.threading
    if threading is None:
      from . import parallel
      threading = self.Threading.TBB if parallel._maxprocs > 1 else self.Threading.SEQUENTIAL
    self.current_threading = self.libmkl.mkl_set_threading_layer(c_long(threading.value))

  def __exit__(self, *exc):
    if self.current_threading != -1:
      self.libmkl.mkl_set_threading_layer(c_long(self.current_threading))
    super().__exit__(*exc)

  def assemble(self, data, index, shape):
    if len(shape) < 2:
      return numeric.accumulate(data, index, shape)
    if len(shape) == 2:
      if not len(data):
        return MKLMatrix([], [1]*(shape[0]+1), [], shape[1], libmkl=self.libmkl)
      # sort rows, columns
      reorder = numpy.lexsort(index[::-1])
      index = index[:,reorder]
      data = data[reorder]
      # sum duplicate entries
      keep = numpy.empty(len(reorder), dtype=bool)
      keep[0] = True
      numpy.not_equal(index[:,1:], index[:,:-1]).any(axis=0, out=keep[1:])
      if not keep.all():
        index = index[:,keep]
        data = numeric.accumulate(data, [keep.cumsum()-1], [index.shape[1]])
      if not data.all():
        nz = data.astype(bool)
        data = data[nz]
        index = index[:,nz]
      return MKLMatrix(data, index[0].searchsorted(numpy.arange(shape[0]+1))+1, index[1]+1, shape[1], libmkl=self.libmkl)
    raise MatrixError('{}d data not supported by MKL backend'.format(len(shape)))

class Pardiso:
  '''simple wrapper for libmkl.pardiso

  https://software.intel.com/en-us/mkl-developer-reference-c-pardiso
  '''

  _errorcodes = {
    -1: 'input inconsistent',
    -2: 'not enough memory',
    -3: 'reordering problem',
    -4: 'zero pivot, numerical factorization or iterative refinement problem',
    -5: 'unclassified (internal) error',
    -6: 'reordering failed (matrix types 11 and 13 only)',
    -7: 'diagonal matrix is singular',
    -8: '32-bit integer overflow problem',
    -9: 'not enough memory for OOC',
   -10: 'error opening OOC files',
   -11: 'read/write error with OOC files',
   -12: 'pardiso_64 called from 32-bit library',
  }

  def __init__(self, libmkl):
    self._pardiso = libmkl.pardiso
    self.pt = numpy.zeros(64, numpy.int64) # handle to data structure

  @types.apply_annotations
  def __call__(self, *, phase:c_int, iparm:c_int, maxfct:c_int=1, mnum:c_int=1, mtype:c_int=0, n:c_int=0, a:c_double=None, ia:c_int=None, ja:c_int=None, perm:c_int=None, nrhs:c_int=0, msglvl:c_int=0, b:c_double=None, x:c_double=None):
    error = ctypes.c_int32(1)
    self._pardiso(self.pt.ctypes, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, ctypes.byref(error))
    if error.value:
      raise MatrixError(self._errorcodes.get(error.value, 'unknown error {}'.format(error.value)))

  def __del__(self):
    if self.pt.any(): # release all internal memory for all matrices
      self(phase=-1, iparm=numpy.zeros(64, dtype=numpy.int32))
      assert not self.pt.any(), 'it appears that Pardiso failed to release its internal memory'

class MKLMatrix(Matrix):
  '''matrix implementation based on sorted coo data'''

  _factors = False

  def __init__(self, data, rowptr, colidx, ncols, libmkl):
    assert len(data) == len(colidx) == rowptr[-1]-1
    self.data = numpy.ascontiguousarray(data, dtype=numpy.float64)
    self.rowptr = numpy.ascontiguousarray(rowptr, dtype=numpy.int32)
    self.colidx = numpy.ascontiguousarray(colidx, dtype=numpy.int32)
    self.libmkl = libmkl
    super().__init__((len(rowptr)-1, ncols))

  def convert(self, mat):
    if not isinstance(mat, Matrix):
      raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
    if self.shape != mat.shape:
      raise MatrixError('non-matching shapes')
    if isinstance(mat, MKLMatrix):
      return mat
    data, colidx, rowptr = mat.export('csr')
    return MKLMatrix(data, rowptr+1, colidx+1, self.shape[1], self.libmkl)

  def __add__(self, other):
    other = self.convert(other)
    assert self.shape == other.shape
    request = ctypes.c_int32(1)
    info = ctypes.c_int32()
    rowptr = numpy.empty(self.shape[0]+1, dtype=numpy.int32)
    args = ["N", ctypes.byref(request), ctypes.byref(ctypes.c_int32(0)),
      ctypes.byref(ctypes.c_int32(self.shape[0])), ctypes.byref(ctypes.c_int32(self.shape[1])),
      self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, ctypes.byref(ctypes.c_double(1.)),
      other.data.ctypes, other.colidx.ctypes, other.rowptr.ctypes,
      None, None, rowptr.ctypes, None, ctypes.byref(info)]
    self.libmkl.mkl_dcsradd(*args)
    assert info.value == 0
    colidx = numpy.empty(rowptr[-1]-1, dtype=numpy.int32)
    data = numpy.empty(rowptr[-1]-1, dtype=numpy.float64)
    request.value = 2
    args[12:14] = data.ctypes, colidx.ctypes
    self.libmkl.mkl_dcsradd(*args)
    assert info.value == 0
    return MKLMatrix(data, rowptr, colidx, self.shape[1], libmkl=self.libmkl)

  def __mul__(self, other):
    if not numeric.isnumber(other):
      raise TypeError
    return MKLMatrix(self.data * other, self.rowptr, self.colidx, self.shape[1], libmkl=self.libmkl)

  def __matmul__(self, other):
    if not isinstance(other, numpy.ndarray):
      raise TypeError
    if other.shape[0] != self.shape[1]:
      raise MatrixError
    x = numpy.ascontiguousarray(other.T, dtype=numpy.float64)
    y = numpy.empty(x.shape[:-1] + self.shape[:1], dtype=numpy.float64)
    if other.ndim == 1:
      self.libmkl.mkl_dcsrgemv('N', ctypes.byref(ctypes.c_int32(self.shape[0])),
        self.data.ctypes, self.rowptr.ctypes, self.colidx.ctypes, x.ctypes, y.ctypes)
    else:
      self.libmkl.mkl_dcsrmm('N', ctypes.byref(ctypes.c_int32(self.shape[0])),
        ctypes.byref(ctypes.c_int32(other.size//other.shape[0])),
        ctypes.byref(ctypes.c_int32(self.shape[1])), ctypes.byref(ctypes.c_double(1.)), 'GXXFXX',
        self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, self.rowptr[1:].ctypes,
        x.ctypes, ctypes.byref(ctypes.c_int32(other.shape[0])), ctypes.byref(ctypes.c_double(0.)),
        y.ctypes, ctypes.byref(ctypes.c_int32(other.shape[0])))
    return y.T

  def __neg__(self):
    return MKLMatrix(-self.data, self.rowptr, self.colidx, self.shape[1], libmkl=self.libmkl)

  @property
  def T(self):
    if self.shape[0] != self.shape[1]:
      raise NotImplementedError('MKLMatrix does not yet support transpose of non-square matrices')
    job = numpy.array([0, 1, 1, 0, 0, 1], numpy.int32)
    data = numpy.empty_like(self.data)
    rowptr = numpy.empty_like(self.rowptr)
    colidx = numpy.empty_like(self.colidx)
    info = ctypes.c_int32()
    self.libmkl.mkl_dcsrcsc(job.ctypes,
      ctypes.byref(ctypes.c_int32(self.shape[0])), self.data.ctypes,
      self.colidx.ctypes, self.rowptr.ctypes, data.ctypes, colidx.ctypes,
      rowptr.ctypes, ctypes.byref(info))
    return MKLMatrix(data, rowptr, colidx, self.shape[1], libmkl=self.libmkl)

  def submatrix(self, rows, cols):
    rows = numeric.asboolean(rows, self.shape[0])
    cols = numeric.asboolean(cols, self.shape[1])
    keep = (rows.all() or rows.repeat(numpy.diff(self.rowptr))) & (cols.all() or cols[self.colidx-1])
    if keep is True: # all rows and all columns are kept
      return self
    elif keep.all(): # all nonzero entries are kept
      rowptr = self.rowptr[numpy.hstack([True, rows])]
      keep = slice(None) # avoid array copies
    else:
      rowptr = numpy.cumsum([1] + [keep[i:j].sum() for i, j in numeric.overlapping(self.rowptr-1)[rows]], dtype=numpy.int32)
    data = self.data[keep]
    assert rowptr[-1] == len(data)+1
    colidx = (self.colidx if cols.all() else cols.cumsum(dtype=numpy.int32)[self.colidx-1])[keep]
    return MKLMatrix(data, rowptr, colidx, cols.sum(), libmkl=self.libmkl)

  def export(self, form):
    if form == 'dense':
      dense = numpy.zeros(self.shape)
      for row, i, j in zip(dense, self.rowptr[:-1]-1, self.rowptr[1:]-1):
        row[self.colidx[i:j]-1] = self.data[i:j]
      return dense
    if form == 'csr':
      return self.data, self.colidx-1, self.rowptr-1
    if form == 'coo':
      return self.data, numpy.array([numpy.arange(self.shape[0]).repeat(self.rowptr[1:]-self.rowptr[:-1]), self.colidx-1])
    raise NotImplementedError('cannot export MKLMatrix to {!r}'.format(form))

  @refine_to_tolerance
  def solve_direct(self, rhs):
    log.debug('solving system using MKL Pardiso')
    if self._factors:
      log.debug('reusing existing factorization')
      pardiso, iparm, mtype = self._factors
      phase = 33 # solve, iterative refinement
    else:
      pardiso = Pardiso(self.libmkl)
      iparm = numpy.zeros(64, dtype=numpy.int32) # https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
      iparm[0] = 1 # supply all values in components iparm[1:64]
      iparm[1] = 2 # fill-in reducing ordering for the input matrix: nested dissection algorithm from the METIS package
      iparm[9] = 13 # pivoting perturbation threshold 1e-13 (default for nonsymmetric)
      iparm[10] = 1 # enable scaling vectors (default for nonsymmetric)
      iparm[12] = 1 # enable improved accuracy using (non-) symmetric weighted matching (default for nonsymmetric)
      iparm[34] = 0 # one-based indexing
      mtype = 11 # real and nonsymmetric
      phase = 13 # analysis, numerical factorization, solve, iterative refinement
      self._factors = pardiso, iparm, mtype
    rhsflat = numpy.ascontiguousarray(rhs.reshape(rhs.shape[0], -1).T, dtype=numpy.float64)
    lhsflat = numpy.empty((rhsflat.shape[0], self.shape[1]), dtype=numpy.float64)
    pardiso(phase=phase, mtype=mtype, iparm=iparm, n=self.shape[0], nrhs=rhsflat.shape[0], b=rhsflat, x=lhsflat, a=self.data, ia=self.rowptr, ja=self.colidx)
    log.debug('solver returned after {} refinement steps; peak memory use {:,d}k'.format(iparm[6], max(iparm[14], iparm[15]+iparm[16])))
    return lhsflat.T.reshape(lhsflat.shape[1:] + rhs.shape[1:])

  def solve_fgmres(self, rhs, atol, maxiter=0, restart=150, precon=None, ztol=1e-12):
    rci = ctypes.c_int32(0)
    n = ctypes.c_int32(len(rhs))
    b = numpy.array(rhs, dtype=numpy.float64)
    x = numpy.zeros_like(b)
    ipar = numpy.zeros(128, dtype=numpy.int32)
    ipar[0] = len(rhs) # problem size
    ipar[1] = 6 # output on screen
    ipar[2] = 1 # current stage of the RCI FGMRES computations; the initial value is 1
    ipar[3] = 0 # current iteration number; the initial value is 0
    ipar[4] = 0 # maximum number of iterations
    ipar[5] = 1 # output error messages in accordance with the parameter ipar[1]
    ipar[6] = 1 # output warning messages in accordance with the parameter ipar[1]
    ipar[7] = 0 # do not perform the stopping test for the maximum number of iterations: ipar[3] <= ipar[4]
    ipar[8] = 0 # do not perform the residual stopping test: dpar[4] <= dpar[3]
    ipar[9] = 1 # perform the user-defined stopping test by setting RCI_request=2
    if precon is None:
      ipar[10] = 0 # run the non-preconditioned version of the FGMRES method
    else:
      ipar[10] = 1 # run the preconditioned version of the FGMRES method
      if precon == 'lu':
        precon = self.solve_direct
      elif precon == 'diag':
        diag = self.diagonal()
        if not diag.all():
          raise MatrixError("building 'diag' preconditioner: diagonal has zero entries")
        precon = numpy.reciprocal(diag).__mul__
      elif not callable(precon):
        raise MatrixError('invalid preconditioner {!r}'.format(precon))
    ipar[11] = 0 # do not perform the automatic test for zero norm of the currently generated vector: dpar[6] <= dpar[7]
    ipar[12] = 1 # update the solution to the vector b according to the computations done by the dfgmres routine
    ipar[13] = 0 # internal iteration counter that counts the number of iterations before the restart takes place; the initial value is 0
    ipar[14] = min(restart, len(rhs)) # the number of non-restarted FGMRES iterations
    dpar = numpy.zeros(128, dtype=numpy.float64)
    tmp = numpy.zeros((2*ipar[14]+1)*ipar[0]+(ipar[14]*(ipar[14]+9))//2+1, dtype=numpy.float64)
    self.libmkl.dfgmres_check(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
    if rci.value != 0:
      raise MatrixError('dgmres check failed with error code {}'.format(rci.value))
    with log.context('fgmres {:.0f}%', 0, 0) as format:
      while True:
        self.libmkl.dfgmres(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
        if rci.value == 1: # multiply the matrix
          tmp[ipar[22]-1:ipar[22]+n.value-1] = self @ tmp[ipar[21]-1:ipar[21]+n.value-1]
        elif rci.value == 2: # perform the stopping test
          if dpar[4] < atol:
            self.libmkl.dfgmres_get(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, ctypes.byref(ctypes.c_int32(0)))
            if numpy.linalg.norm(self @ b - rhs) < atol:
              break
            b[:] = rhs # reset rhs vector for restart
          format(100 * numpy.log(dpar[2]/dpar[4]) / numpy.log(dpar[2]/atol))
          if ipar[3] > maxiter > 0:
            break
        elif rci.value == 3: # apply the preconditioner
          tmp[ipar[22]-1:ipar[22]+n.value-1] = precon(tmp[ipar[21]-1:ipar[21]+n.value-1])
        elif rci.value == 4: # check if the norm of the current orthogonal vector is zero
          if dpar[6] < ztol:
            self.libmkl.dfgmres_get(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, ctypes.byref(ctypes.c_int32(0)))
            if numpy.linalg.norm(self @ b - rhs) < atol:
              break
            raise MatrixError('singular matrix')
        else:
          raise MatrixError('this should not have occurred: rci={}'.format(rci.value))
    log.debug('performed {} fgmres iterations, {} restarts'.format(ipar[3], ipar[3]//ipar[14]))
    return b

## MODULE METHODS

_current_backend = Numpy()

def assemble(data, index, shape):
  return _current_backend.assemble(data, index, shape)

def empty(shape):
  return assemble(data=numpy.empty([0], dtype=float), index=numpy.empty([len(shape), 0], dtype=int), shape=shape)

def diag(d):
  assert d.ndim == 1
  return assemble(d, index=numpy.arange(len(d))[numpy.newaxis].repeat(2, axis=0), shape=d.shape*2)

def eye(n):
  return diag(numpy.ones(n))

backend = typing.Union[Numpy,Scipy,MKL]
auto = MKL() or Scipy() or Numpy()

# vim:sw=2:sts=2:et
