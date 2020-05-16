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
from .. import numeric, util, warnings
from contextlib import contextmanager
from ctypes import c_long, c_int, c_double, byref
import treelog as log
import numpy

libmkl = util.loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')
if not libmkl:
  raise BackendNotAvailable('the Intel MKL matrix backend requires libmkl to be installed (try: pip install mkl)')

@contextmanager
def setassemble(sets, threading:str=None):
  if not threading:
    from ..parallel import _maxprocs
    threading = 'tbb' if _maxprocs.value > 1 else 'sequential'
  value = dict(intel=0, sequential=1, pgi=2, gnu=3, tbb=4)[threading.lower()]
  oldvalue = libmkl.mkl_set_threading_layer(byref(c_long(value)))
  try:
    with sets(assemble):
      yield
  finally:
    libmkl.mkl_set_threading_layer(byref(c_long(oldvalue)))

def assemble(data, index, shape):
  return MKLMatrix(data, index[0].searchsorted(numpy.arange(shape[0]+1))+1, index[1]+1, shape[1])

class Pardiso:
  '''Wrapper for libmkl.pardiso.

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

  def __init__(self, mtype, a, ia, ja, checkmatrix=False, verbose=False):
    self.pt = numpy.zeros(64, numpy.int64) # handle to data structure
    self.maxfct = c_int(1)
    self.mnum = c_int(1)
    self.mtype = c_int(mtype)
    self.n = c_int(len(ia)-1)
    self.a = a.ctypes
    self.ia = ia.ctypes
    self.ja = ja.ctypes
    self.perm = None
    self.iparm = numpy.zeros(64, dtype=numpy.int32) # https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
    self.msglvl = c_int(verbose)
    libmkl.pardisoinit(self.pt.ctypes, byref(self.mtype), self.iparm.ctypes) # initialize iparm based on mtype
    assert self.iparm[0] == 1, 'pardiso init failed'
    self.iparm[26] = checkmatrix
    self.iparm[27] = 0 # double precision data
    self.iparm[34] = 0 # one-based indexing
    self.iparm[36] = 0 # csr matrix format
    self._phase(12) # analysis, numerical factorization
    log.debug('peak memory use {:,d}k'.format(max(self.iparm[14], self.iparm[15]+self.iparm[16])))

  def __call__(self, rhs):
    rhsflat = numpy.ascontiguousarray(rhs.reshape(rhs.shape[0], -1).T, dtype=numpy.float64)
    lhsflat = numpy.empty_like(rhsflat)
    self._phase(33, rhsflat.shape[0], rhsflat.ctypes, lhsflat.ctypes) # solve, iterative refinement
    return lhsflat.T.reshape(rhs.shape)

  def _phase(self, phase, nrhs=0, b=None, x=None):
    error = c_int(1)
    libmkl.pardiso(self.pt.ctypes, byref(self.maxfct), byref(self.mnum), byref(self.mtype),
      byref(c_int(phase)), byref(self.n), self.a, self.ia, self.ja, self.perm,
      byref(c_int(nrhs)), self.iparm.ctypes, byref(self.msglvl), b, x, byref(error))
    if error.value:
      raise MatrixError(self._errorcodes.get(error.value, 'unknown error {}'.format(error.value)))

  def __del__(self):
    self._phase(-1) # release all internal memory for all matrices
    if self.pt.any():
      warnings.warn('Pardiso failed to release its internal memory')

class MKLMatrix(Matrix):
  '''matrix implementation based on sorted coo data'''

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
    request = c_int(1)
    info = c_int()
    rowptr = numpy.empty(self.shape[0]+1, dtype=numpy.int32)
    args = ["N", byref(request), byref(c_int(0)),
      byref(c_int(self.shape[0])), byref(c_int(self.shape[1])),
      self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, byref(c_double(1.)),
      other.data.ctypes, other.colidx.ctypes, other.rowptr.ctypes,
      None, None, rowptr.ctypes, None, byref(info)]
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
      libmkl.mkl_dcsrgemv('N', byref(c_int(self.shape[0])),
        self.data.ctypes, self.rowptr.ctypes, self.colidx.ctypes, x.ctypes, y.ctypes)
    else:
      libmkl.mkl_dcsrmm('N', byref(c_int(self.shape[0])),
        byref(c_int(other.size//other.shape[0])),
        byref(c_int(self.shape[1])), byref(c_double(1.)), 'GXXFXX',
        self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, self.rowptr[1:].ctypes,
        x.ctypes, byref(c_int(other.shape[0])), byref(c_double(0.)),
        y.ctypes, byref(c_int(other.shape[0])))
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
    info = c_int()
    libmkl.mkl_dcsrcsc(job.ctypes,
      byref(c_int(self.shape[0])), self.data.ctypes,
      self.colidx.ctypes, self.rowptr.ctypes, data.ctypes, colidx.ctypes,
      rowptr.ctypes, byref(info))
    return MKLMatrix(data, rowptr, colidx, self.shape[1])

  def _submatrix(self, rows, cols):
    keep = rows.repeat(numpy.diff(self.rowptr))
    keep &= cols[self.colidx-1]
    if keep.all(): # all nonzero entries are kept
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

  def _solver_fgmres(self, rhs, atol, maxiter=0, restart=150, precon=None, ztol=1e-12):
    rci = c_int(0)
    n = c_int(len(rhs))
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
      precon = self.getprecon(precon)
    ipar[11] = 0 # do not perform the automatic test for zero norm of the currently generated vector: dpar[6] <= dpar[7]
    ipar[12] = 1 # update the solution to the vector b according to the computations done by the dfgmres routine
    ipar[13] = 0 # internal iteration counter that counts the number of iterations before the restart takes place; the initial value is 0
    ipar[14] = min(restart, len(rhs)) # the number of non-restarted FGMRES iterations
    dpar = numpy.zeros(128, dtype=numpy.float64)
    tmp = numpy.zeros((2*ipar[14]+1)*ipar[0]+(ipar[14]*(ipar[14]+9))//2+1, dtype=numpy.float64)
    libmkl.dfgmres_check(byref(n), x.ctypes, b.ctypes, byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
    if rci.value != 0:
      raise MatrixError('dgmres check failed with error code {}'.format(rci.value))
    with log.context('fgmres {:.0f}%', 0, 0) as format:
      while True:
        libmkl.dfgmres(byref(n), x.ctypes, b.ctypes, byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes)
        if rci.value == 1: # multiply the matrix
          tmp[ipar[22]-1:ipar[22]+n.value-1] = self @ tmp[ipar[21]-1:ipar[21]+n.value-1]
        elif rci.value == 2: # perform the stopping test
          if dpar[4] < atol:
            libmkl.dfgmres_get(byref(n), x.ctypes, b.ctypes, byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, byref(c_int(0)))
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
            libmkl.dfgmres_get(byref(n), x.ctypes, b.ctypes, byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes, byref(c_int(0)))
            if numpy.linalg.norm(self @ b - rhs) < atol:
              break
            raise MatrixError('singular matrix')
        else:
          raise MatrixError('this should not have occurred: rci={}'.format(rci.value))
    log.debug('performed {} fgmres iterations, {} restarts'.format(ipar[3], ipar[3]//ipar[14]))
    return b

  def _precon_direct(self):
    return Pardiso(mtype=11, a=self.data, ia=self.rowptr, ja=self.colidx)

# vim:sw=2:sts=2:et
