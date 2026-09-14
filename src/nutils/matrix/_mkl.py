from ._base import Matrix, MatrixError, BackendNotAvailable
from .. import numeric, _util as util, warnings
from ctypes import c_int, c_double, c_void_p, byref, sizeof, cast, POINTER, Structure
import treelog as log
import numpy


libmkl = util.loadlib('mkl_rt')
if libmkl is None:
    raise BackendNotAvailable('the Intel MKL matrix backend requires libmkl to be installed (try: pip install mkl)')


def assemble(data, rowptr, colidx, ncols):
    return MKLMatrix(data, rowptr, colidx, ncols)


class Pardiso:
    '''Wrapper for libmkl.pardiso.

    https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/
      onemkl-pardiso-parallel-direct-sparse-solver-iface.html
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
        self.iparm[34] = 1 # zero-based indexing
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


# The following defines two matrix types: HandleMatrix, which maintains an MKL
# sparse matrix handle, and MKLMatrix, which maintains the CSR data triplet.
#
# Reasons for MKLMatrix:
#
#     MKL does not appear to support 0 x n or n x 0 matrices. The MKLMatrix is
#     therefore the more generic object that is able to serve edge cases.
#     Furthermore, Pardiso requires a CSR triplet, not an MKL matrix handle.
#     Even the newer Direct Sparse Solver API, which uses handles to refer to
#     data, uses a _different_ handle with no conversion between the two
#     provided. And since it is very common for a MKLMatrix to be created just
#     to forward its CSR data to Pardiso, the creation of a matrix handle would
#     be unnecessary overhead.
#
# Reasons for HandleMatrix:
#
#     The Inspector-Executor sparse BLAS routines require that MKL maintains
#     its own array data which therefore needs to be explicitly destroyed.
#     While we could use a context for this, destroying the handle after every
#     operation, this leaves the problem of export, which returns pointer to
#     MKL managed memory that needs to be kept alive with the arrays. For this
#     reason the HandleMatrix uses a destructor to release memory, and any
#     exported arrays carry the matrix instance in their .base attribute.
#
# While it is technically possible to fold one matrix into the other (using a
# lazily cached handle, conditional destructor, etc.) the distinctness of the
# above considerations make for cleaner code when kept separate.


sparse_matrix_t = c_void_p


class MatrixDescr(Structure):
    _fields_ = [("type", c_int), ("mode", c_int), ("diag", c_int)]


class MKL_Complex16(Structure):
    _fields_ = [("real", c_double), ("imag", c_double)]


SPARSE_INDEX_BASE_ZERO = c_int(0)
SPARSE_OPERATION_NON_TRANSPOSE = c_int(10)
SPARSE_OPERATION_CONJUGATE_TRANSPOSE = c_int(12)
SPARSE_MATRIX_TYPE_GENERAL = c_int(20)
SPARSE_LAYOUT_ROW_MAJOR = c_int(101) # NOTE: this used to be 60 until (probably) MKL 2019 Update 4


class MklType:
    def __init__(self, kind, numpy_dtype, c_zero, c_one):
        self.kind = kind
        self.numpy_dtype = numpy_dtype
        self.c_zero = c_zero
        self.c_one = c_one

    def __str__(self):
        return self.kind


D_FLOAT64 = MklType(
    kind="d",
    numpy_dtype=numpy.dtype("float64"),
    c_zero=c_double(0.0),
    c_one=c_double(1.0),
)

D_COMPLEX128 = MklType(
    kind="z",
    numpy_dtype=numpy.dtype("complex128"),
    c_zero=MKL_Complex16(0.0, 0.0),
    c_one=MKL_Complex16(1.0, 0.0),
)


class HandleMatrix:
    """Interface to MKL's `Sparse BLAS Routines`_.

    .. _Sparse BLAS Routines: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/inspector-executor-sparse-blas-routines.html
    """

    @classmethod
    def create(cls, rowptr, colidx, values, ncols):
        handle = sparse_matrix_t(0)
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-sparse-create-csr.html
        if values.dtype == numpy.float64:
            dtype = D_FLOAT64
        elif values.dtype == numpy.complex128:
            dtype = D_COMPLEX128
        else:
            raise ValueError(f"unsupported data type {values.dtype}")
        f = getattr(libmkl, f"mkl_sparse_{dtype}_create_csr")
        nrows = len(rowptr) - 1
        ncols = ncols.__index__()
        status = f(
            byref(handle),
            SPARSE_INDEX_BASE_ZERO,
            c_int(nrows),
            c_int(ncols),
            rowptr[:-1].ctypes,
            rowptr[1:].ctypes,
            colidx.ctypes,
            values.ctypes,
        )
        if status != 0:
            raise RuntimeError(f"MKL sparse create csr failed with error code {status}")
        m = cls(handle, nrows, ncols, dtype)
        m._keep_alive = rowptr, colidx, values
        return m

    def __init__(self, handle, nrows, ncols, dtype):
        assert isinstance(handle, sparse_matrix_t)
        assert isinstance(nrows, int) and nrows > 0
        assert isinstance(ncols, int) and ncols > 0
        assert isinstance(dtype, MklType)
        self._handle = handle
        self._nrows = nrows
        self._ncols = ncols
        self._dtype = dtype

    def __del__(self):
        libmkl.mkl_sparse_destroy(self._handle)

    def export(self):
        """Generate rowptr, colidx and values arrays."""

        out_base = c_int(0)
        out_rows = c_int(0)
        out_cols = c_int(0)
        p_rows_start = POINTER(c_int)()
        p_rows_end = POINTER(c_int)()
        p_col_indx = POINTER(c_int)()
        p_values = POINTER(c_double)()
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-sparse-export-csr.html
        f = getattr(libmkl, f"mkl_sparse_{self._dtype}_export_csr")
        status = f(
            self._handle,
            byref(out_base),
            byref(out_rows),
            byref(out_cols),
            byref(p_rows_start),
            byref(p_rows_end),
            byref(p_col_indx),
            byref(p_values),
        )
        if status != 0:
            raise RuntimeError(f"MKL sparse export csr failed with error code {status}")
        assert out_rows.value == self._nrows
        assert out_cols.value == self._ncols
        assert p_rows_start[0] == 0
        assert out_base.value == 0
        assert cast(p_rows_end, c_void_p).value == cast(
            p_rows_start, c_void_p
        ).value + sizeof(c_int)
        nnz = p_rows_end[self._nrows - 1]

        # The __array_interface__ approach below achieves that all three arrays
        # have self as their .base attribute, so that the referenced memory
        # will not be deallocated until all arrays are garbage collected.

        # 1. rowptr
        self.__array_interface__ = {
            "data": (cast(p_rows_start, c_void_p).value, False),
            "typestr": numpy.dtype("int32").str,
            "shape": (self._nrows + 1,),
            "version": 3,
        }
        yield numpy.asarray(self)

        # 2. colidx
        self.__array_interface__ = {
            "data": (cast(p_col_indx, c_void_p).value, False),
            "typestr": numpy.dtype("int32").str,
            "shape": (nnz,),
            "version": 3,
        }
        yield numpy.asarray(self)

        # 3. values
        self.__array_interface__ = {
            "data": (cast(p_values, c_void_p).value, False),
            "typestr": self._dtype.numpy_dtype.str,
            "shape": (nnz,),
            "version": 3,
        }
        yield numpy.asarray(self)

        del self.__array_interface__

    def transpose(self):
        handle = sparse_matrix_t(0)
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-sparse-convert-csr.html
        libmkl.mkl_sparse_convert_csr(
            self._handle, SPARSE_OPERATION_CONJUGATE_TRANSPOSE, byref(handle)
        )
        return HandleMatrix(handle, self._ncols, self._nrows, self._dtype)

    def __add__(self, other):
        assert isinstance(other, HandleMatrix)
        assert other._dtype is self._dtype
        assert other._nrows == self._nrows
        assert other._ncols == self._ncols
        handle = sparse_matrix_t(0)
        f = getattr(libmkl, f"mkl_sparse_{self._dtype}_add")
        status = f(
            SPARSE_OPERATION_NON_TRANSPOSE,
            self._handle,
            self._dtype.c_one,
            other._handle,
            byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"MKL sparse add failed with error code {status}")
        # Make column indices increasing
        status = libmkl.mkl_sparse_order(handle)
        if status != 0:
            raise RuntimeError(f"MKL sparse order failed with error code {status}")
        return HandleMatrix(handle, self._nrows, self._ncols, self._dtype)

    def __matmul__(self, other):
        if not isinstance(other, numpy.ndarray):
            raise TypeError
        if other.shape[0] != self._ncols:
            raise MatrixError(
                f"cannot multiply {self._nrows}x{self._ncols} matrix with array of length {other.shape[0]}"
            )
        x = numpy.ascontiguousarray(other, dtype=self._dtype.numpy_dtype)
        if x.size == 0:
            return x.copy()
        y = numpy.empty((self._nrows, *x.shape[1:]), dtype=self._dtype.numpy_dtype)
        descr = MatrixDescr(SPARSE_MATRIX_TYPE_GENERAL, 0, 0)
        nvecs = x.size // x.shape[0]
        if nvecs == 1:
            # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-sparse-mv.html
            f = getattr(libmkl, f"mkl_sparse_{self._dtype}_mv")
            status = f(
                SPARSE_OPERATION_NON_TRANSPOSE,
                self._dtype.c_one,
                self._handle,
                descr,
                x.ctypes,
                self._dtype.c_zero,
                y.ctypes,
            )
            if status != 0:
                raise RuntimeError(f"MKL sparse mv failed with error code {status}")
        else:
            # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-sparse-mm.html
            f = getattr(libmkl, f"mkl_sparse_{self._dtype}_mm")
            n = c_int(nvecs)
            status = f(
                SPARSE_OPERATION_NON_TRANSPOSE,
                self._dtype.c_one,
                self._handle,
                descr,
                SPARSE_LAYOUT_ROW_MAJOR,
                x.ctypes,
                n,
                n,
                self._dtype.c_zero,
                y.ctypes,
                n,
            )
            if status != 0:
                raise RuntimeError(f"MKL sparse mm failed with error code {status}")
        return y


class MKLMatrix(Matrix):
    '''matrix implementation based on sorted coo data'''

    def __init__(self, data, rowptr, colidx, ncols):
        assert len(data) == len(colidx) == rowptr[-1]
        self.data = numpy.ascontiguousarray(data, dtype=numpy.complex128 if data.dtype.kind == 'c' else numpy.float64)
        self.rowptr = numpy.ascontiguousarray(rowptr, dtype=numpy.int32)
        self.colidx = numpy.ascontiguousarray(colidx, dtype=numpy.int32)
        super().__init__((len(rowptr)-1, ncols), self.data.dtype)

    def _as_handle_matrix(self):
        return HandleMatrix.create(self.rowptr, self.colidx, self.data, self.shape[1])

    def convert(self, mat):
        if not isinstance(mat, Matrix):
            raise TypeError('cannot convert {} to Matrix'.format(type(mat).__name__))
        if self.shape != mat.shape:
            raise MatrixError('non-matching shapes')
        if isinstance(mat, MKLMatrix) and mat.dtype == self.dtype:
            return mat
        data, colidx, rowptr = mat.export('csr')
        return MKLMatrix(data.astype(self.dtype, copy=False), rowptr, colidx, self.shape[1])

    def __add__(self, other):
        if not all(self.shape):
            return self
        m = self._as_handle_matrix() + self.convert(other)._as_handle_matrix()
        rowptr, colidx, values = m.export()
        return MKLMatrix(values, rowptr, colidx, self.shape[1])

    def __mul__(self, other):
        if not numeric.isnumber(other):
            raise TypeError
        return MKLMatrix(self.data * other, self.rowptr, self.colidx, self.shape[1])

    def __matmul__(self, other):
        if not all(self.shape):
            return numpy.empty((self.shape[0], *other.shape[1:]), dtype=self.dtype)
        return self._as_handle_matrix() @ other

    def __neg__(self):
        return MKLMatrix(-self.data, self.rowptr, self.colidx, self.shape[1])

    @property
    def T(self):
        if not all(self.shape):
            rowptr = numpy.zeros(self.shape[1] + 1, dtype=numpy.int32)
            colidx = numpy.empty(0, dtype=numpy.int32)
            values = numpy.empty(0, dtype=self.dtype)
        else:
            rowptr, colidx, values = self._as_handle_matrix().transpose().export()
        return MKLMatrix(values, rowptr, colidx, self.shape[0])

    def _submatrix(self, rows, cols):
        keep = rows.repeat(numpy.diff(self.rowptr))
        keep &= cols[self.colidx]
        if keep.all():  # all nonzero entries are kept
            rowptr = self.rowptr[numpy.hstack([True, rows])]
            keep = slice(None)  # avoid array copies
        else:
            rowptr = numpy.cumsum([0] + [keep[i:j].sum() for i, j in numeric.overlapping(self.rowptr)[rows]], dtype=numpy.int32)
        data = self.data[keep]
        assert rowptr[-1] == len(data)
        colidx = (self.colidx if cols.all() else cols.cumsum(dtype=numpy.int32)[self.colidx] - 1)[keep]
        return MKLMatrix(data, rowptr, colidx, cols.sum())

    def export(self, form):
        if form == 'dense':
            dense = numpy.zeros(self.shape, self.dtype)
            for row, i, j in zip(dense, self.rowptr[:-1], self.rowptr[1:]):
                row[self.colidx[i:j]] = self.data[i:j]
            return dense
        if form == 'csr':
            return self.data, self.colidx, self.rowptr
        if form == 'coo':
            return self.data, (numpy.arange(self.shape[0]).repeat(self.rowptr[1:]-self.rowptr[:-1]), self.colidx)
        raise NotImplementedError('cannot export MKLMatrix to {!r}'.format(form))

    def _solver_fgmres(self, rhs, atol, maxiter=0, restart=150, precon=None, ztol=1e-12, preconargs={}, **args):
        if self.dtype.kind == 'c':
            raise MatrixError("MKL's fgmres does not support complex data")
        rci = c_int(0)
        n = c_int(len(rhs))
        b = numpy.ascontiguousarray(rhs, dtype=numpy.float64)
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
        if not len(self.data):
            raise MatrixError('matrix is exactly zero')
        if self.shape[0] == self.shape[1] == 1:
            # workaround for MKL 2025.1.0 "out of memory" bug for 1x1 matrices
            v, = self.data
            return (1./v).__mul__
        return Pardiso(mtype=dict(f=11, c=13)[self.dtype.kind], a=self.data, ia=self.rowptr, ja=self.colidx, **args)

    def _precon_sym_direct(self, **args):
        if not len(self.data):
            raise MatrixError('matrix is exactly zero')
        if self.shape[0] == self.shape[1] == 1:
            # workaround for MKL 2025.1.0 "out of memory" bug for 1x1 matrices
            v, = self.data
            return (1./v).__mul__
        upper = numpy.zeros(len(self.data), dtype=bool)
        rowptr = numpy.empty_like(self.rowptr)
        rowptr[0] = 0
        diagdom = True
        for irow, (n, m) in enumerate(numeric.overlapping(self.rowptr)):
            d = n + self.colidx[n:m].searchsorted(irow)
            upper[d:m] = True
            rowptr[irow+1] = rowptr[irow] + (m-d)
            diagdom = diagdom and d < m and self.colidx[d] == irow and abs(self.data[n:m]).sum() < 2 * abs(self.data[d])
        if diagdom:
            log.debug('matrix is diagonally dominant, solving as SPD')
            mtype = dict(f=2, c=4)
        else:
            mtype = dict(f=-2, c=6)
        return Pardiso(mtype=mtype[self.dtype.kind], a=self.data[upper], ia=rowptr, ja=self.colidx[upper], **args)

# vim:sw=4:sts=4:et
