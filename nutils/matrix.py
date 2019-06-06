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

from . import numpy, numeric, warnings, cache, types, config, util
import abc, sys, ctypes, treelog as log


class MatrixError(Exception): pass


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

  @classmethod
  def get(cls, name):
    for subcls in cls.__subclasses__():
      if subcls.__name__.lower() == name.lower():
        return subcls

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

  def __add__(self, other):
    'add two matrices'
    if not isinstance(other, Matrix):
      return NotImplemented
    if self.shape != other.shape:
      raise MatrixError('incompatible shapes')
    data1, index1 = self.export('coo')
    data2, index2 = other.export('coo')
    return assemble(numpy.concatenate([data1, data2]), numpy.concatenate([index1, index2], axis=1), self.shape)

  def __mul__(self, other):
    'multiply matrix with a scalar'
    if not numeric.isnumber(other):
      return NotImplemented
    data, index = self.export('coo')
    return assemble(data * other, index, self.shape)

  def __matmul__(self, other):
    'multiply matrix with a dense tensor'
    if not isinstance(other, numpy.ndarray):
      return NotImplemented
    if other.shape[0] != self.shape[1]:
      raise MatrixError('incompatible shapes')
    retval = numpy.zeros(self.shape[:1] + other.shape[1:])
    data, index = self.export('coo')
    numpy.add.at(retval, index[0], data[(slice(None),)+(numpy.newaxis,)*(other.ndim-1)] * other[index[1]])
    return retval

  def __neg__(self):
    'negate matrix'
    data, index = self.export('coo')
    return assemble(-data, index, self.shape)

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
  def T(self):
    'transpose matrix'
    data, index = self.export('coo')
    return assemble(data, index[::-1], self.shape[::-1])

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
  def solve(self, rhs=None, *, lhs0=None, constrain=None, rconstrain=None, solver='direct', **solverargs):
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
    assert I.sum() == J.sum(), 'constrained matrix is not square: {}x{}'.format(I.sum(), J.sum())
    b = (rhs - self @ x)[J]
    if b.any():
      A = self if I.all() and J.all() else self.submatrix(I, J)
      log.info('solving {0[0]}x{0[1]} system using {1} solver'.format(A.shape, solver))
      try:
        x[J] += getattr(A, 'solve_'+solver)(b, **solverargs)
      except Exception as e:
        raise MatrixError('solver failed with error: {}'.format(e)) from e
      if not numpy.isfinite(x).all():
        raise MatrixError('solver returned non-finite left hand side')
      log.info('solver returned with residual {:.0e}'.format(numpy.linalg.norm((rhs - self @ x)[J])))
    else:
      log.info('skipping solver because initial vector is exact')
    return x

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

  def __repr__(self):
    return '{}<{}x{}>'.format(type(self).__qualname__, *self.shape)


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

  def __add__(self, other):
    if isinstance(other, NumpyMatrix) and self.shape == other.shape:
      return NumpyMatrix(self.core + other.core)
    return super().__add__(other)

  def __mul__(self, other):
    if numeric.isnumber(other):
      return NumpyMatrix(self.core * other)
    return super().__mul__(other)

  def __matmul__(self, other):
    if isinstance(other, numpy.ndarray) and other.shape[0] == self.shape[1]:
      return numpy.einsum('ij,j...->i...', self.core, other)
    return super().__matmul__(other)

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

  def solve_direct(self, rhs):
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

    @staticmethod
    def assemble(data, index, shape):
      if len(shape) < 2:
        return numeric.accumulate(data, index, shape)
      if len(shape) == 2:
        csr = scipy.sparse.csr_matrix((data, index), shape)
        return ScipyMatrix(csr)
      raise MatrixError('{}d data not supported by scipy backend'.format(len(shape)))

  class ScipyMatrix(Matrix):
    '''matrix based on any of scipy's sparse matrices'''

    def __init__(self, core):
      self.core = core
      super().__init__(core.shape)

    def __add__(self, other):
      if isinstance(other, ScipyMatrix) and self.shape == other.shape:
        return ScipyMatrix(self.core + other.core)
      return super().__add__(other)

    def __sub__(self, other):
      if isinstance(other, ScipyMatrix) and self.shape == other.shape:
        return ScipyMatrix(self.core - other.core)
      return super().__sub__(other)

    def __mul__(self, other):
      if numeric.isnumber(other):
        return ScipyMatrix(self.core * other)
      return super().__mul__(other)

    def __matmul__(self, other):
      if isinstance(other, numpy.ndarray) and other.shape[0] == self.shape[1]:
        return self.core * other
      return super().__matmul__(other)

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

    def solve_direct(self, rhs):
      return scipy.sparse.linalg.spsolve(self.core, rhs)

    def solve_scipy(self, rhs, solver, atol=1e-6, callback=None, precon=None, **solverargs):
      rhsnorm = numpy.linalg.norm(rhs)
      if rhsnorm <= atol:
        return numpy.zeros(self.shape[1])
      solverfun = getattr(scipy.sparse.linalg, solver)
      myrhs = rhs / rhsnorm # normalize right hand side vector for best control over scipy's stopping criterion
      mytol = atol / rhsnorm
      niter = numpy.array(0)
      def mycallback(arg):
        niter[...] += 1
        # some solvers provide the residual, others the left hand side vector
        res = numpy.linalg.norm(myrhs - self @ arg) if numpy.ndim(arg) == 1 else float(arg)
        if callback:
          callback(res)
        with log.context('residual {:.2e} ({:.0f}%)'.format(res, 100. * numpy.log10(res) / numpy.log10(mytol) if res > 0 else 0)):
          pass
      M = self.getprecon(precon) if isinstance(precon, str) else precon(self.core) if callable(precon) else precon
      mylhs, status = solverfun(self.core, myrhs, M=M, tol=mytol, callback=mycallback, **solverargs)
      if status != 0:
        raise Exception('status {}'.format(status))
      if numpy.linalg.norm(myrhs - self @ mylhs) > atol:
        raise Exception('failed to reach tolerance')
      log.info('solver converged in {} iterations'.format(niter))
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
          precon = scipy.sparse.linalg.splu(self.core.tocsc()).solve
        except RuntimeError as e:
          raise MatrixError(e) from e
      elif name == 'spilu':
        try:
          precon = scipy.sparse.linalg.spilu(self.core.tocsc(), drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None).solve
        except RuntimeError as e:
          raise MatrixError(e) from e
      elif name == 'diag':
        diag = self.core.diagonal()
        if not diag.all():
          raise MatrixError("building 'diag' preconditioner: diagonal has zero entries")
        precon = numpy.reciprocal(diag).__mul__
      else:
        raise MatrixError('invalid preconditioner {!r}'.format(name))
      return scipy.sparse.linalg.LinearOperator(self.shape, precon, dtype=float)

    def submatrix(self, rows, cols):
      return ScipyMatrix(self.core[rows,:][:,cols])


## INTEL MKL BACKEND

libmkl = util.loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')
if libmkl is not None:

  # typedefs
  c_int = types.c_array[numpy.int32]
  c_long = types.c_array[numpy.int64]
  c_double = types.c_array[numpy.float64]

  class MKL(Backend):
    '''matrix backend based on Intel's Math Kernel Library'''

    def __enter__(self):
      super().__enter__()
      usethreads = config.nprocs > 1
      libmkl.mkl_set_threading_layer(c_long(4 if usethreads else 1)) # 1:SEQUENTIAL, 4:TBB
      return self

    @staticmethod
    def assemble(data, index, shape):
      if len(shape) < 2:
        return numeric.accumulate(data, index, shape)
      if len(shape) == 2:
        if not len(data):
          return MKLMatrix([], [1]*(shape[0]+1), [], shape[1])
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
        return MKLMatrix(data, index[0].searchsorted(numpy.arange(shape[0]+1))+1, index[1]+1, shape[1])
      raise MatrixError('{}d data not supported by MKL backend'.format(len(shape)))

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

    def __init__(self):
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

    def __init__(self, data, rowptr, colidx, ncols):
      assert len(data) == len(colidx) == rowptr[-1]-1
      self.data = numpy.ascontiguousarray(data, dtype=numpy.float64)
      self.rowptr = numpy.ascontiguousarray(rowptr, dtype=numpy.int32)
      self.colidx = numpy.ascontiguousarray(colidx, dtype=numpy.int32)
      super().__init__((len(rowptr)-1, ncols))

    def __add__(self, other):
      if isinstance(other, MKLMatrix) and self.shape == other.shape:
        request = ctypes.c_int32(1)
        info = ctypes.c_int32()
        rowptr = numpy.empty(self.shape[0]+1, dtype=numpy.int32)
        args = ["N", ctypes.byref(request), ctypes.byref(ctypes.c_int32(0)),
          ctypes.byref(ctypes.c_int32(self.shape[0])), ctypes.byref(ctypes.c_int32(self.shape[1])),
          self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, ctypes.byref(ctypes.c_double(1.)),
          other.data.ctypes, other.colidx.ctypes, other.rowptr.ctypes,
          None, None, rowptr.ctypes, None, ctypes.byref(info)]
        libmkl.mkl_dcsradd(*args)
        assert info.value == 0
        colidx = numpy.empty(rowptr[-1]-1, dtype=numpy.int32)
        data = numpy.empty(rowptr[-1]-1, dtype=numpy.float64)
        request.value = 2
        args[12:14] = data.ctypes, colidx.ctypes
        libmkl.mkl_dcsradd(*args)
        assert info.value == 0
        return MKLMatrix(data, rowptr, colidx, self.shape[1])
      return super().__add__(other)

    def __mul__(self, other):
      if numeric.isnumber(other):
        return MKLMatrix(self.data * other, self.rowptr, self.colidx, self.shape[1])
      return super().__mul__(other)

    def __matmul__(self, other):
      if isinstance(other, numpy.ndarray) and other.shape[0] == self.shape[1]:
        x = numpy.ascontiguousarray(other.T, dtype=numpy.float64)
        y = numpy.empty(x.shape[:-1] + self.shape[:1], dtype=numpy.float64)
        if other.ndim == 1:
          libmkl.mkl_dcsrgemv('N', ctypes.byref(ctypes.c_int32(self.shape[0])),
            self.data.ctypes, self.rowptr.ctypes, self.colidx.ctypes, x.ctypes, y.ctypes)
        else:
          libmkl.mkl_dcsrmm('N', ctypes.byref(ctypes.c_int32(self.shape[0])),
            ctypes.byref(ctypes.c_int32(other.size//other.shape[0])),
            ctypes.byref(ctypes.c_int32(self.shape[1])), ctypes.byref(ctypes.c_double(1.)), 'GXXFXX',
            self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, self.rowptr[1:].ctypes,
            x.ctypes, ctypes.byref(ctypes.c_int32(other.shape[0])), ctypes.byref(ctypes.c_double(0.)),
            y.ctypes, ctypes.byref(ctypes.c_int32(other.shape[0])))
        return y.T
      return super().__matmul__(other)

    def __neg__(self):
      return MKLMatrix(-self.data, self.rowptr, self.colidx, self.shape[1])

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

    def solve_direct(self, rhs):
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
        iparm[34] = 0 # one-based indexing
        mtype = 11 # real and nonsymmetric
        phase = 13 # analysis, numerical factorization, solve, iterative refinement
        self._factors = pardiso, iparm, mtype
      rhsflat = numpy.ascontiguousarray(rhs.reshape(rhs.shape[0], -1).T, dtype=numpy.float64)
      lhsflat = numpy.empty((rhsflat.shape[0], self.shape[1]), dtype=numpy.float64)
      pardiso(phase=phase, mtype=mtype, iparm=iparm, n=self.shape[0], nrhs=rhsflat.shape[0], b=rhsflat, x=lhsflat, a=self.data, ia=self.rowptr, ja=self.colidx)
      return lhsflat.T.reshape(lhsflat.shape[1:] + rhs.shape[1:])

    def solve_fgmres(self, rhs, maxiter=0, atol=1e-6, restart=150):
      rci = ctypes.c_int32(0)
      n = ctypes.c_int32(len(rhs))
      b = numpy.array(rhs, dtype=numpy.float64)
      x = numpy.zeros_like(b)
      ipar = numpy.zeros(128, dtype=numpy.int32)
      ipar[0] = len(rhs) # problem size
      ipar[1] = 6 # output on screen
      ipar[2] = 1 # current stage of the RCI FGMRES computations; the initial value is 1
      ipar[3] = 0 # current iteration number; the initial value is 0
      ipar[4] = maxiter # maximum number of iterations
      ipar[5] = 1 # output error messages in accordance with the parameter ipar[1]
      ipar[6] = 1 # output warning messages in accordance with the parameter ipar[1]
      ipar[7] = 1 if maxiter > 0 else 0 # perform the stopping test for the maximum number of iterations: ipar[3] <= ipar[4]
      ipar[8] = 1 # perform the residual stopping test: dpar[4] <= dpar[3]
      ipar[9] = 0 # the user-defined stopping test should not be performed by setting RCI_request=2
      ipar[10] = 0 # run the non-preconditioned version of the FGMRES method
      ipar[11] = 1 # perform the automatic test for zero norm of the currently generated vector: dpar[6] <= dpar[7]
      ipar[12] = 0 # update the solution to the vector x according to the computations done by the dfgmres routine
      ipar[13] = 0 # internal iteration counter that counts the number of iterations before the restart takes place; the initial value is 0
      ipar[14] = min(restart, len(rhs)) # the number of non-restarted FGMRES iterations
      dpar = numpy.zeros(128, dtype=numpy.float64)
      dpar[0] = 0 # relative tolerance
      dpar[1] = atol # absolute tolerance
      dpar[7] = 1e-12 # tolerance for the zero norm of the currently generated vector
      tmp = numpy.zeros((2*ipar[14]+1)*ipar[0]+(ipar[14]*(ipar[14]+9))//2+1, dtype=numpy.float64)
      libmkl.dfgmres_check(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
      assert rci.value == 0
      while True:
        with log.context('iter {} ({:.0f}%)'.format(ipar[13], 100 * numpy.log(dpar[2]/dpar[4]) / numpy.log(dpar[2]/atol) if dpar[4] else 0)):
          libmkl.dfgmres(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
          if rci.value == 0:
            break
          elif rci.value == 1:
            tmp[ipar[22]-1:ipar[22]+n.value-1] = self @ tmp[ipar[21]-1:ipar[21]+n.value-1]
          else:
            raise NotImplementedError
      itercount = ctypes.c_int32(0)
      libmkl.dfgmres_get(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, ctypes.byref(itercount))
      if numpy.linalg.norm(self @ x - b) > atol:
        raise MatrixError('fgmres solver failed to reach tolerance')
      return x

## MODULE METHODS

_current_backend = Numpy()

def backend(names):
  for cls in map(Backend.get, names.split(',')):
    if cls is not None:
      return cls()
  raise RuntimeError('matrix backend {!r} is not available'.format(names))

def assemble(data, index, shape):
  return _current_backend.assemble(data, index, shape)

def diag(d):
  assert d.ndim == 1
  return assemble(d, index=numpy.arange(len(d))[numpy.newaxis].repeat(2, axis=0), shape=d.shape*2)

def eye(n):
  return diag(numpy.ones(n))

# vim:sw=2:sts=2:et
