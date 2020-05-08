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

from ._base import Matrix, MatrixError, BackendNotAvailable, refine_to_tolerance
from .. import numeric, util, types
from contextlib import contextmanager
import treelog as log
import numpy, typing, ctypes

libmkl = util.loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')
if not libmkl:
  raise BackendNotAvailable('the Intel MKL matrix backend requires libmkl to be installed (try: pip install mkl)')

c_int = types.c_array[numpy.int32]
c_long = types.c_array[numpy.int64]
c_double = types.c_array[numpy.float64]

@contextmanager
def setassemble(sets, threading:str=None):
  if not threading:
    from ..parallel import _maxprocs
    threading = 'tbb' if _maxprocs.value > 1 else 'sequential'
  value = dict(intel=0, sequential=1, pgi=2, gnu=3, tbb=4)[threading.lower()]
  _threading = libmkl.mkl_set_threading_layer(c_long(value))
  try:
    with sets(assemble):
      yield
  finally:
    libmkl.mkl_set_threading_layer(c_long(_threading))

def assemble(data, index, shape):
  return MKLMatrix(data, index[0].searchsorted(numpy.arange(shape[0]+1))+1, index[1]+1, shape[1])

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

  def __init__(self):
    self.pt = numpy.zeros(64, numpy.int64) # handle to data structure

  @types.apply_annotations
  def __call__(self, *, phase:c_int, iparm:c_int, maxfct:c_int=1, mnum:c_int=1, mtype:c_int=0, n:c_int=0, a:c_double=None, ia:c_int=None, ja:c_int=None, perm:c_int=None, nrhs:c_int=0, msglvl:c_int=0, b:c_double=None, x:c_double=None):
    error = ctypes.c_int32(1)
    libmkl.pardiso(self.pt.ctypes, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, ctypes.byref(error))
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

  def convert(self, mat):
    if not isinstance(mat, Matrix):
      raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
    if self.shape != mat.shape:
      raise MatrixError('non-matching shapes')
    if isinstance(mat, MKLMatrix):
      return mat
    data, colidx, rowptr = mat.export('csr')
    return MKLMatrix(data, rowptr+1, colidx+1, self.shape[1])

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
    libmkl.mkl_dcsradd(*args)
    assert info.value == 0
    colidx = numpy.empty(rowptr[-1]-1, dtype=numpy.int32)
    data = numpy.empty(rowptr[-1]-1, dtype=numpy.float64)
    request.value = 2
    args[12:14] = data.ctypes, colidx.ctypes
    libmkl.mkl_dcsradd(*args)
    assert info.value == 0
    return MKLMatrix(data, rowptr, colidx, self.shape[1])

  def __mul__(self, other):
    if not numeric.isnumber(other):
      raise TypeError
    return MKLMatrix(self.data * other, self.rowptr, self.colidx, self.shape[1])

  def __matmul__(self, other):
    if not isinstance(other, numpy.ndarray):
      raise TypeError
    if other.shape[0] != self.shape[1]:
      raise MatrixError
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

  def __neg__(self):
    return MKLMatrix(-self.data, self.rowptr, self.colidx, self.shape[1])

  @property
  def T(self):
    if self.shape[0] != self.shape[1]:
      raise NotImplementedError('MKLMatrix does not yet support transpose of non-square matrices')
    job = numpy.array([0, 1, 1, 0, 0, 1], numpy.int32)
    data = numpy.empty_like(self.data)
    rowptr = numpy.empty_like(self.rowptr)
    colidx = numpy.empty_like(self.colidx)
    info = ctypes.c_int32()
    libmkl.mkl_dcsrcsc(job.ctypes,
      ctypes.byref(ctypes.c_int32(self.shape[0])), self.data.ctypes,
      self.colidx.ctypes, self.rowptr.ctypes, data.ctypes, colidx.ctypes,
      rowptr.ctypes, ctypes.byref(info))
    return MKLMatrix(data, rowptr, colidx, self.shape[1])

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
    return MKLMatrix(data, rowptr, colidx, cols.sum())

  def export(self, form):
    if form == 'dense':
      dense = numpy.zeros(self.shape)
      for row, i, j in zip(dense, self.rowptr[:-1]-1, self.rowptr[1:]-1):
        row[self.colidx[i:j]-1] = self.data[i:j]
      return dense
    if form == 'csr':
      return self.data, self.colidx-1, self.rowptr-1
    if form == 'coo':
      return self.data, (numpy.arange(self.shape[0]).repeat(self.rowptr[1:]-self.rowptr[:-1]), self.colidx-1)
    raise NotImplementedError('cannot export MKLMatrix to {!r}'.format(form))

  @refine_to_tolerance
  def solve_direct(self, rhs):
    log.debug('solving system using MKL Pardiso')
    if self._factors:
      log.debug('reusing existing factorization')
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
    libmkl.dfgmres_check(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
    if rci.value != 0:
      raise MatrixError('dgmres check failed with error code {}'.format(rci.value))
    with log.context('fgmres {:.0f}%', 0, 0) as format:
      while True:
        libmkl.dfgmres(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
        if rci.value == 1: # multiply the matrix
          tmp[ipar[22]-1:ipar[22]+n.value-1] = self @ tmp[ipar[21]-1:ipar[21]+n.value-1]
        elif rci.value == 2: # perform the stopping test
          if dpar[4] < atol:
            libmkl.dfgmres_get(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, ctypes.byref(ctypes.c_int32(0)))
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
            libmkl.dfgmres_get(ctypes.byref(n), x.ctypes, b.ctypes, ctypes.byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, ctypes.byref(ctypes.c_int32(0)))
            if numpy.linalg.norm(self @ b - rhs) < atol:
              break
            raise MatrixError('singular matrix')
        else:
          raise MatrixError('this should not have occurred: rci={}'.format(rci.value))
    log.debug('performed {} fgmres iterations, {} restarts'.format(ipar[3], ipar[3]//ipar[14]))
    return b

# vim:sw=2:sts=2:et
