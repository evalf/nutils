from ._base import Matrix, MatrixError, BackendNotAvailable
from .. import numeric, _util as util, warnings
from contextlib import contextmanager
from ctypes import c_int, byref, CDLL
import treelog as log
import os
import numpy

libmkl_path = os.environ.get('NUTILS_MATRIX_MKL_LIB', None)
if libmkl_path:
    libmkl = CDLL(libmkl_path)
else:
    for v in '.2', '.1', '':
        libmkl = util.loadlib(linux=f'libmkl_rt.so{v}', darwin=f'libmkl_rt{v}.dylib', win32=f'mkl_rt{v}.dll')
        if libmkl:
            break
    else:
        raise BackendNotAvailable('the Intel MKL matrix backend requires libmkl to be installed (try: pip install mkl)')


def assemble(data, index, shape):
    # In the increments below the output dtype is set to int32 not only to avoid
    # an additional allocation, but crucially also to avoid truncation in case
    # the incremented index overflows the original type.
    return MKLMatrix(data, ncols=shape[1],
                     rowptr=numpy.add(index[0].searchsorted(numpy.arange(shape[0]+1)), 1, dtype=numpy.int32),
                     colidx=numpy.add(index[1], 1, dtype=numpy.int32))


class Pardiso:
    '''Wrapper for libmkl.pardiso.

    https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/
      sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html
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

    def __init__(self, mtype, a, ia, ja, verbose=False, iparm={}):
        self.dtype = a.dtype
        self.pt = numpy.zeros(64, numpy.int64)  # handle to data structure
        self.maxfct = c_int(1)
        self.mnum = c_int(1)
        self.mtype = c_int(mtype)
        self.n = c_int(len(ia)-1)
        self.a = a.ctypes
        self.ia = ia.ctypes
        self.ja = ja.ctypes
        self.perm = None
        self.iparm = numpy.zeros(64, dtype=numpy.int32)  # https://software.intel.com/en-us/mkl-developer-reference-c-pardiso-iparm-parameter
        self.msglvl = c_int(verbose)
        libmkl.pardisoinit(self.pt.ctypes, byref(self.mtype), self.iparm.ctypes)  # initialize iparm based on mtype
        if self.iparm[0] != 1:
            raise MatrixError('pardiso init failed')
        for n, v in iparm.items():
            self.iparm[n] = v
        self.iparm[10] = 1 # enable scaling (default for nonsymmetric matrices, recommended for highly indefinite symmetric matrices)
        self.iparm[12] = 1 # enable matching (default for nonsymmetric matrices, recommended for highly indefinite symmetric matrices)
        self.iparm[27] = 0 # double precision data
        self.iparm[34] = 0 # one-based indexing
        self.iparm[36] = 0 # csr matrix format
        self._phase(12)  # analysis, numerical factorization
        log.debug('peak memory use {:,d}k'.format(max(self.iparm[14], self.iparm[15]+self.iparm[16])))

    def __call__(self, rhs):
        rhsflat = numpy.ascontiguousarray(rhs.reshape(rhs.shape[0], -1).T, dtype=self.dtype)
        lhsflat = numpy.empty_like(rhsflat)
        self._phase(33, rhsflat.shape[0], rhsflat.ctypes, lhsflat.ctypes)  # solve, iterative refinement
        return lhsflat.T.reshape(rhs.shape)

    def _phase(self, phase, nrhs=0, b=None, x=None):
        error = c_int(1)
        libmkl.pardiso(self.pt.ctypes, byref(self.maxfct), byref(self.mnum), byref(self.mtype),
                       byref(c_int(phase)), byref(self.n), self.a, self.ia, self.ja, self.perm,
                       byref(c_int(nrhs)), self.iparm.ctypes, byref(self.msglvl), b, x, byref(error))
        if error.value:
            raise MatrixError(self._errorcodes.get(error.value, 'unknown error {}'.format(error.value)))

    def __del__(self):
        self._phase(-1)  # release all internal memory for all matrices
        if self.pt.any():
            warnings.warn('Pardiso failed to release its internal memory')


class MKLMatrix(Matrix):
    '''matrix implementation based on sorted coo data'''

    def __init__(self, data, rowptr, colidx, ncols):
        assert len(data) == len(colidx) == rowptr[-1]-1
        self.data = numpy.ascontiguousarray(data, dtype=numpy.complex128 if data.dtype.kind == 'c' else numpy.float64)
        self.rowptr = numpy.ascontiguousarray(rowptr, dtype=numpy.int32)
        self.colidx = numpy.ascontiguousarray(colidx, dtype=numpy.int32)
        super().__init__((len(rowptr)-1, ncols), self.data.dtype)

    def mkl_(self, name, *args):
        attr = 'mkl_' + dict(f='d', c='z')[self.dtype.kind] + name
        return getattr(libmkl, attr)(*args)

    def convert(self, mat):
        if not isinstance(mat, Matrix):
            raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
        if self.shape != mat.shape:
            raise MatrixError('non-matching shapes')
        if isinstance(mat, MKLMatrix) and mat.dtype == self.dtype:
            return mat
        data, colidx, rowptr = mat.export('csr')
        return MKLMatrix(data.astype(self.dtype, copy=False), rowptr+1, colidx+1, self.shape[1])

    def __add__(self, other):
        other = self.convert(other)
        assert self.shape == other.shape and self.dtype == other.dtype
        request = c_int(1)
        info = c_int()
        rowptr = numpy.empty(self.shape[0]+1, dtype=numpy.int32)
        one = numpy.array(1, dtype=self.dtype)
        args = ["N", byref(request), byref(c_int(0)),
                byref(c_int(self.shape[0])), byref(c_int(self.shape[1])),
                self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, one.ctypes,
                other.data.ctypes, other.colidx.ctypes, other.rowptr.ctypes,
                None, None, rowptr.ctypes, None, byref(info)]
        self.mkl_('csradd', *args)
        assert info.value == 0
        colidx = numpy.empty(rowptr[-1]-1, dtype=numpy.int32)
        data = numpy.empty(rowptr[-1]-1, dtype=self.dtype)
        request.value = 2
        args[12:14] = data.ctypes, colidx.ctypes
        self.mkl_('csradd', *args)
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
        x = numpy.ascontiguousarray(other.T, dtype=self.dtype)
        y = numpy.empty(x.shape[:-1] + self.shape[:1], dtype=self.dtype)
        if other.ndim == 1:
            self.mkl_('csrgemv', 'N', byref(c_int(self.shape[0])),
                      self.data.ctypes, self.rowptr.ctypes, self.colidx.ctypes, x.ctypes, y.ctypes)
        else:
            zero = numpy.array(0, dtype=self.dtype)
            one = numpy.array(1, dtype=self.dtype)
            self.mkl_('csrmm', 'N', byref(c_int(self.shape[0])),
                      byref(c_int(other.size//other.shape[0])),
                      byref(c_int(self.shape[1])), one.ctypes, 'GXXFXX',
                      self.data.ctypes, self.colidx.ctypes, self.rowptr.ctypes, self.rowptr[1:].ctypes,
                      x.ctypes, byref(c_int(other.shape[0])), zero.ctypes,
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
        self.mkl_('csrcsc', job.ctypes,
                  byref(c_int(self.shape[0])), self.data.ctypes,
                  self.colidx.ctypes, self.rowptr.ctypes, data.ctypes, colidx.ctypes,
                  rowptr.ctypes, byref(info))
        return MKLMatrix(data, rowptr, colidx, self.shape[1])

    def _submatrix(self, rows, cols):
        keep = rows.repeat(numpy.diff(self.rowptr))
        keep &= cols[self.colidx-1]
        if keep.all():  # all nonzero entries are kept
            rowptr = self.rowptr[numpy.hstack([True, rows])]
            keep = slice(None)  # avoid array copies
        else:
            rowptr = numpy.cumsum([1] + [keep[i:j].sum() for i, j in numeric.overlapping(self.rowptr-1)[rows]], dtype=numpy.int32)
        data = self.data[keep]
        assert rowptr[-1] == len(data)+1
        colidx = (self.colidx if cols.all() else cols.cumsum(dtype=numpy.int32)[self.colidx-1])[keep]
        return MKLMatrix(data, rowptr, colidx, cols.sum())

    def export(self, form):
        if form == 'dense':
            dense = numpy.zeros(self.shape, self.dtype)
            for row, i, j in zip(dense, self.rowptr[:-1]-1, self.rowptr[1:]-1):
                row[self.colidx[i:j]-1] = self.data[i:j]
            return dense
        if form == 'csr':
            return self.data, self.colidx-1, self.rowptr-1
        if form == 'coo':
            return self.data, (numpy.arange(self.shape[0]).repeat(self.rowptr[1:]-self.rowptr[:-1]), self.colidx-1)
        raise NotImplementedError('cannot export MKLMatrix to {!r}'.format(form))

    def _solver_fgmres(self, rhs, atol, maxiter=0, restart=150, precon=None, ztol=1e-12, preconargs={}, **args):
        if self.dtype.kind == 'c':
            raise MatrixError("MKL's fgmres does not support complex data")
        rci = c_int(0)
        n = c_int(len(rhs))
        b = numpy.array(rhs, dtype=numpy.float64, copy=False)
        x = numpy.zeros_like(b)
        N = min(restart, len(rhs))
        ipar = numpy.empty(128, dtype=numpy.int32)
        dpar = numpy.empty(128, dtype=numpy.float64)
        tmp = numpy.empty((2*N+1)*len(rhs)+(N*(N+9))//2+1, dtype=numpy.float64)
        dfgmres_args = byref(n), x.ctypes, b.ctypes, byref(rci), ipar.ctypes, dpar.ctypes, tmp.ctypes
        itercount = c_int(0)
        libmkl.dfgmres_init(*dfgmres_args)
        ipar[7] = 0  # do not perform the stopping test for the maximum number of iterations
        ipar[8] = 0  # do not perform the residual stopping test
        ipar[9] = 1  # perform the user-defined stopping test by setting RCI_request=2
        if precon is not None:
            ipar[10] = 1  # run the preconditioned version of the FGMRES method
            precon = self.getprecon(precon, **args, **preconargs)
        ipar[11] = 0  # do not perform the automatic test for zero norm of the currently generated vector
        ipar[12] = 0  # update the solution to the vector x according to the computations done by the dfgmres routine
        ipar[14] = N  # the number of non-restarted FGMRES iterations
        libmkl.dfgmres_check(*dfgmres_args)
        if rci.value in (-1001, -1010, -1011):
            warnings.warn('dgmres ' + ' and '.join(['wrote some warnings to stdout', 'changed some parameters to make them consistent or correct'][1 if rci.value == -1010 else 0:1 if rci.value == -1001 else 2]))
        elif rci.value != 0:
            raise MatrixError('dgmres check failed with error code {}'.format(rci.value))
        with log.context('fgmres {:.0f}%', 0, 0) as format:
            while True:
                libmkl.dfgmres(*dfgmres_args)
                if rci.value == 1:  # multiply the matrix
                    tmp[ipar[22]-1:ipar[22]+n.value-1] = self @ tmp[ipar[21]-1:ipar[21]+n.value-1]
                elif rci.value == 2:  # perform the stopping test
                    if dpar[4] < atol:
                        libmkl.dfgmres_get(*dfgmres_args, byref(itercount))
                        if numpy.linalg.norm(self @ x - b) < atol:
                            break
                    format(100 * numpy.log(dpar[2]/dpar[4]) / numpy.log(dpar[2]/atol))
                    if ipar[3] > maxiter > 0:
                        break
                elif rci.value == 3:  # apply the preconditioner
                    tmp[ipar[22]-1:ipar[22]+n.value-1] = precon(tmp[ipar[21]-1:ipar[21]+n.value-1])
                elif rci.value == 4:  # check if the norm of the current orthogonal vector is zero
                    if dpar[6] < ztol:
                        libmkl.dfgmres_get(*dfgmres_args, byref(itercount))
                        if numpy.linalg.norm(self @ x - b) < atol:
                            break
                        raise MatrixError('singular matrix')
                else:
                    raise MatrixError('this should not have occurred: rci={}'.format(rci.value))
        log.debug('performed {} fgmres iterations, {} restarts'.format(ipar[3], ipar[3]//ipar[14]))
        return x

    def _precon_direct(self, **args):
        return Pardiso(mtype=dict(f=11, c=13)[self.dtype.kind], a=self.data, ia=self.rowptr, ja=self.colidx, **args)

    def _precon_sym_direct(self, **args):
        upper = numpy.zeros(len(self.data), dtype=bool)
        rowptr = numpy.empty_like(self.rowptr)
        rowptr[0] = 1
        diagdom = True
        for irow, (n, m) in enumerate(numeric.overlapping(self.rowptr-1), start=1):
            d = n + self.colidx[n:m].searchsorted(irow)
            upper[d:m] = True
            rowptr[irow] = rowptr[irow-1] + (m-d)
            diagdom = diagdom and d < m and self.colidx[d] == irow and abs(self.data[n:m]).sum() < 2 * abs(self.data[d])
        if diagdom:
            log.debug('matrix is diagonally dominant, solving as SPD')
            mtype = dict(f=2, c=4)
        else:
            mtype = dict(f=-2, c=6)
        return Pardiso(mtype=mtype[self.dtype.kind], a=self.data[upper], ia=rowptr, ja=self.colidx[upper], **args)

# vim:sw=4:sts=4:et
