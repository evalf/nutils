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

from . import numpy, log, numeric, warnings, cache, util
import abc, sys, ctypes


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
  def solve(self, rhs=None, *, lhs0=None, constrain=None, rconstrain=None, **solverargs):
    '''Solve system given right hand side vector and/or constraints.

    Args
    ----
    rhs : float vector or None
        Right hand side vector. `None` implies all zeros.
    lhs0 : float vector or None
        Initial values. `None` implies all zeros.
    constrain : float or boolean array, or None
        Column constraints. For float values, a number signifies a constraint,
        NaN signifies a free dof. For boolean, a True value signifies a
        constraint to the value in `lhs0`, a False value signifies a free dof.
        `None` implies no constraints.
    rconstrain : boolean array or None
        Row constrains. A True value signifies a constrains, a False value a free
        dof. `None` implies that the constraints follow those defined in
        `constrain` (by implication the matrix must be square).

    Returns
    -------
    Left hand side vector.
    '''

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

def preparesolvearguments(wrapped):
  '''Make rhs optional, add lhs0, constrain, rconstrain arguments.

  See Matrix.solve.'''

  def solve(self, rhs=None, *, lhs0=None, constrain=None, rconstrain=None, **solverargs):
    nrows, ncols = self.shape
    if lhs0 is None:
      x = numpy.zeros(ncols)
    else:
      x = numpy.array(lhs0, dtype=float)
      assert x.shape == (ncols,)
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
    assert I.sum() == J.sum(), 'constrained matrix is not square: {}x{}'.format(I.sum(), J.sum())
    if rhs is None:
      rhs = 0.
    b = (rhs - self.matvec(x))[J]
    if b.any():
      x[J] += wrapped(self if I.all() and J.all() else self.submatrix(I, J), b, **solverargs)
      log.info('solver returned with residual {:.0e}'.format(numpy.linalg.norm((rhs - self.matvec(x))[J])))
    else:
      log.info('skipping solver because initial vector is exact')
    return x
  return log.title(solve)


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

  @preparesolvearguments
  def solve(self, rhs):
    return numpy.linalg.solve(self.core, rhs)

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

    @preparesolvearguments
    def solve(self, rhs, atol=0, solver='spsolve', callback=None, precon=None, **solverargs):
      if solver == 'spsolve':
        log.info('solving system using sparse direct solver')
        return scipy.sparse.linalg.spsolve(self.core, rhs)
      assert atol, 'tolerance must be specified for iterative solver'
      rhsnorm = numpy.linalg.norm(rhs)
      if rhsnorm <= atol:
        return numpy.zeros(self.shape[1])
      log.info('solving system using {} iterative solver'.format(solver))
      solverfun = getattr(scipy.sparse.linalg, solver)
      myrhs = rhs / rhsnorm # normalize right hand side vector for best control over scipy's stopping criterion
      mytol = atol / rhsnorm
      niter = numpy.array(0)
      def mycallback(arg):
        niter[...] += 1
        # some solvers provide the residual, others the left hand side vector
        res = numpy.linalg.norm(myrhs - self.matvec(arg)) if numpy.ndim(arg) == 1 else float(arg)
        if callback:
          callback(res)
        with log.context('residual {:.2e} ({:.0f}%)'.format(res, 100. * numpy.log10(res) / numpy.log10(mytol) if res > 0 else 0)):
          pass
      M = self.getprecon(precon) if isinstance(precon, str) else precon(self.core) if callable(precon) else precon
      mylhs, status = solverfun(self.core, myrhs, M=M, tol=mytol, callback=mycallback, **solverargs)
      assert status == 0, '{} solver failed with status {}'.format(solver, status)
      log.info('solver converged in {} iterations'.format(niter))
      return mylhs * rhsnorm

    def getprecon(self, name):
      name = name.lower()
      assert self.shape[0] == self.shape[1], 'constrained matrix must be square'
      log.info('building {} preconditioner'.format(name))
      if name == 'splu':
        precon = scipy.sparse.linalg.splu(self.core.tocsc()).solve
      elif name == 'spilu':
        precon = scipy.sparse.linalg.spilu(self.core.tocsc(), drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None).solve
      elif name == 'diag':
        precon = numpy.reciprocal(self.core.diagonal()).__mul__
      else:
        raise Exception('invalid preconditioner {!r}'.format(name))
      return scipy.sparse.linalg.LinearOperator(self.shape, precon, dtype=float)

    def submatrix(self, rows, cols):
      return ScipyMatrix(self.core[rows,:][:,cols])


## INTEL MKL BACKEND

try:
  libmkl = ctypes.CDLL({'linux': 'libmkl_rt.so', 'darwin': 'libmkl_rt.dylib', 'win32': 'mkl_rt.dll'}[sys.platform])
except (OSError, KeyError):
  pass
else:

  class c_array:
    def __init__(self, dtype):
      self.dtype = dtype
    def __call__(self, obj):
      if obj is not None:
        if not isinstance(obj, numpy.ndarray):
          obj = numpy.array(obj, dtype=self.dtype)
        assert obj.flags.c_contiguous and obj.dtype == self.dtype
        return obj.ctypes

  # typedefs
  c_int = c_array(numpy.int32)
  c_long = c_array(numpy.int64)
  c_double = c_array(numpy.float64)

  class MKL(Backend):
    '''matrix backend based on Intel's Math Kernel Library'''

    def __init__(self):
      libmkl.mkl_set_threading_layer(c_long(4)) # use Intel Threading Building Blocks instead op OpenMP

    @staticmethod
    def assemble(data, index, shape):
      if len(shape) < 2:
        return numeric.accumulate(data, index, shape)
      if len(shape) == 2:
        return MKLMatrix(data, index, shape)
      raise Exception('{}d data not supported by scipy backend'.format(len(shape)))

  class Pardiso:
    '''simple wrapper for libmkl.pardiso

    https://software.intel.com/en-us/mkl-developer-reference-c-pardiso
    '''

    _pardiso = libmkl.pardiso
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

    class PardisoError(Exception):
      pass

    def __init__(self):
      self.pt = numpy.zeros(64, numpy.int64) # handle to data structure

    @util.enforcetypes
    def __call__(self, *, phase:c_int, iparm:c_int, maxfct:c_int=1, mnum:c_int=1, mtype:c_int=0, n:c_int=0, a:c_double=None, ia:c_int=None, ja:c_int=None, perm:c_int=None, nrhs:c_int=0, msglvl:c_int=0, b:c_double=None, x:c_double=None):
      error = ctypes.c_int32(1)
      self._pardiso(self.pt.ctypes, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, ctypes.byref(error))
      if error.value:
        raise self.PardisoError(self._errorcodes.get(error.value, 'unknown error {}'.format(error.value)))

    def __del__(self):
      if self.pt.any(): # release all internal memory for all matrices
        self(phase=-1, iparm=numpy.zeros(64, dtype=numpy.int32))
        assert not self.pt.any(), 'it appears that Pardiso failed to release its internal memory'

  class MKLMatrix(Matrix):
    '''matrix implementation based on sorted coo data'''

    _factors = False

    def __init__(self, data, index, shape):
      assert index.shape == (2, len(data))
      if len(data):
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
      self.data = numpy.ascontiguousarray(data, dtype=numpy.float64)
      self.index = numpy.ascontiguousarray(index, dtype=numpy.int32)
      super().__init__(shape)

    @cache.property
    def indptr(self):
      return self.index[0].searchsorted(numpy.arange(self.shape[0]+1)).astype(numpy.int32, copy=False)

    def __add__(self, other):
      if not isinstance(other, MKLMatrix) or self.shape != other.shape:
        return NotImplemented
      return MKLMatrix(numpy.concatenate([self.data, other.data]), numpy.concatenate([self.index, other.index], axis=1), self.shape)

    def __sub__(self, other):
      if not isinstance(other, MKLMatrix) or self.shape != other.shape:
        return NotImplemented
      return MKLMatrix(numpy.concatenate([self.data, -other.data]), numpy.concatenate([self.index, other.index], axis=1), self.shape)

    def __mul__(self, other):
      if not numeric.isnumber(other):
        return NotImplemented
      return MKLMatrix(self.data * other, self.index, self.shape)

    def __neg__(self):
      return MKLMatrix(-self.data, self.index, self.shape)

    @property
    def T(self):
      return MKLMatrix(self.data, self.index[::-1], self.shape[::-1])

    def matvec(self, vec):
      rows, cols = self.index
      return numeric.accumulate(self.data * vec[cols], [rows], [self.shape[0]])

    def export(self, form):
      if form == 'dense':
        return numeric.accumulate(self.data, self.index, self.shape)
      if form == 'csr':
        return self.data, self.index[1], self.indptr
      if form == 'coo':
        return self.data, self.index
      raise NotImplementedError('cannot export MKLMatrix to {!r}'.format(form))

    def submatrix(self, rows, cols):
      I, J = self.index
      keep = numpy.logical_and(rows[I], cols[J])
      csI = rows.cumsum()
      csJ = cols.cumsum()
      return MKLMatrix(self.data[keep], numpy.array([csI[I[keep]]-1, csJ[J[keep]]-1]), shape=(csI[-1], csJ[-1]))

    @preparesolvearguments
    def solve(self, rhs):
      log.info('solving {0}x{0} system using MKL Pardiso'.format(self.shape[0]))
      if self._factors:
        log.info('reusing existing factorization')
        pardiso, iparm, mtype = self._factors
        phase = 33 # solve, iterative refinement
      else:
        pardiso = Pardiso()
        iparm = numpy.zeros(64, dtype=numpy.int32) # https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
        iparm[0] = 1 # supply all values in components iparm[1:64]
        iparm[1] = 2 # fill-in reducing ordering for the input matrix: nested dissection algorithm from the METIS package
        iparm[9] = 13 # pivoting perturbation threshold 1e-13 (default for nonsymmetric)
        iparm[10] = 1 # enable scaling vectors (default for nonsymmetric)
        iparm[12] = 1 # enable improved accuracy using (non-) symmetric weighted matching (default for nonsymmetric)
        iparm[34] = 1 # zero base indexing
        mtype = 11 # real and nonsymmetric
        phase = 13 # analysis, numerical factorization, solve, iterative refinement
        self._factors = pardiso, iparm, mtype
      lhs = numpy.empty(self.shape[1], dtype=numpy.float64)
      pardiso(phase=phase, mtype=mtype, iparm=iparm, n=self.shape[0], nrhs=1, b=rhs, x=lhs, a=self.data, ia=self.indptr, ja=self.index[1])
      return lhs


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
