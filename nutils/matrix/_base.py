from .. import numeric
import abc
import treelog
import functools
import numpy
import collections


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

    def __init__(self, shape, dtype):
        assert len(shape) == 2
        self.shape = shape
        self.dtype = dtype
        self._precon_args = None
        self._cached_submatrix = None

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
    def solve(self, rhs=None, *, lhs0=None, constrain=None, rconstrain=None, solver='arnoldi', atol=0., rtol=0., **solverargs):
        '''Solve system given right hand side vector and/or constraints.

        Args
        ----
        rhs : :class:`float` vector or :any:`None`
            Right hand side vector. A :any:`None` value implies the zero vector.
        lhs0 : class:`float` vector or :any:`None`
            Initial values: compute the solution by solving ``A dx = b - A lhs0``.
            A :any:`None` value implies the zero vector, i.e. solving ``A x = b``
            directly.
        constrain : :class:`float` or :class:`bool` array, or :any:`None` Column
            constraints. For float values, a number signifies a constraint, NaN
            signifies a free dof. For boolean, a :any:`True` value signifies a
            constraint to the value in ``lhs0``, a :any:`False` value signifies a
            free dof. A :any:`None` value implies no constraints.
        rconstrain : :class:`bool` array or :any:`None`
            Row constrains. A True value signifies a constrains, a False value a free
            dof. A :any:`None` value implies that the constraints follow those
            defined in ``constrain`` (by implication the matrix must be square).
        solver : :class:`str`
            Name of the solver algorithm. The set of available solvers depends on
            the type of the matrix (i.e. the active backend), although the 'direct'
            and 'arnoldi' solvers are always available.
        rtol : :class:`float`
            Relative tolerance: see ``atol``.
        atol : :class:`float`
            Absolute tolerance: require that ``|A x - b| <= max(atol, rtol |b|)``
            after applying constraints and the initial value. In case ``atol`` and
            ``rtol`` are both zero (the defaults) solve to machine precision.
            Otherwise fail with :class:`nutils.matrix.ToleranceNotReached` if the
            requirement is not reached.
        **kwargs :
            All remaining arguments are passed on to the selected solver method.

        Returns
        -------
        :class:`numpy.ndarray`
            Left hand side vector.
        '''

        # absent an initial guess and constraints we can directly forward to _solver
        if lhs0 is constrain is rconstrain is None:
            return self._solver(rhs, solver, atol=atol, rtol=rtol, **solverargs)

        # otherwise we need to do some pre- and post-processing
        nrows, ncols = self.shape
        if rhs is None:
            rhs = numpy.zeros(nrows, self.dtype)
        if lhs0 is None:
            lhs = numpy.zeros((ncols,)+rhs.shape[1:], self.dtype)
        else:
            lhs = numpy.array(lhs0, dtype=self.dtype)
            while lhs.ndim < rhs.ndim:
                lhs = lhs[..., numpy.newaxis].repeat(rhs.shape[lhs.ndim], axis=lhs.ndim)
            assert lhs.shape == (ncols,)+rhs.shape[1:]
        if constrain is None:
            J = numpy.ones(ncols, dtype=bool)
        else:
            assert constrain.shape == (ncols,)
            if constrain.dtype == bool:
                J = ~constrain
            else:
                J = numpy.isnan(constrain)
                lhs[~J] = constrain[~J]
        if rconstrain is None:
            assert nrows == ncols
            I = J
        else:
            assert rconstrain.shape == (nrows,) and constrain.dtype == bool
            I = ~rconstrain
        lhs[J] += self.submatrix(I, J)._solver((rhs - self @ lhs)[I], solver, atol=atol, rtol=rtol, **solverargs)
        return lhs

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

    def _method(self, prefix, attr):
        if callable(attr):
            return functools.partial(attr, self), getattr(attr, '__name__', 'user defined')
        if isinstance(attr, str):
            fullattr = '_' + prefix + '_' + attr
            if hasattr(self, fullattr):
                return getattr(self, fullattr), attr
        raise MatrixError('invalid {} {!r} for {}'.format(prefix, attr, self.__class__.__name__))

    def _solver(self, rhs, solver, *, atol, rtol, **solverargs):
        if self.shape[0] != self.shape[1]:
            raise MatrixError('constrained matrix is not square: {}x{}'.format(*self.shape))
        if rhs.shape[0] != self.shape[0]:
            raise MatrixError('right-hand size shape does not match matrix shape')
        rhsnorm = numpy.linalg.norm(rhs, axis=0).max()
        atol = max(atol, rtol * rhsnorm)
        if rhsnorm <= atol:
            treelog.info('skipping solver because initial vector is within tolerance')
            return numpy.zeros_like(rhs)
        solver_method, solver_name = self._method('solver', solver)
        treelog.info('solving {} dof system to {} using {} solver'.format(self.shape[0], 'tolerance {:.0e}'.format(atol) if atol else 'machine precision', solver_name))
        try:
            lhs = solver_method(rhs, atol=atol, **solverargs)
        except MatrixError:
            raise
        except Exception as e:
            raise MatrixError('solver failed with error: {}'.format(e)) from e
        if not numpy.isfinite(lhs).all():
            raise MatrixError('solver returned non-finite left hand side')
        resnorm = numpy.linalg.norm(rhs - self @ lhs, axis=0).max()
        treelog.info('solver returned with residual {:.0e}'.format(resnorm))
        if resnorm > atol > 0:
            raise ToleranceNotReached(lhs)
        return lhs

    def _solver_direct(self, rhs, atol, precon='direct', preconargs={}, **args):
        solve = self.getprecon(precon, **args, **preconargs)
        return solve(rhs)

    def _solver_arnoldi(self, rhs, atol, precon='direct', truncate=None, preconargs={}, **args):
        solve = self.getprecon(precon, **args, **preconargs)
        lhs = numpy.zeros_like(rhs)
        res = rhs
        resnorm = numpy.linalg.norm(res, axis=0).max()
        krylov = collections.deque(maxlen=truncate)  # unlimited if truncate is None
        while resnorm > atol:
            k = solve(res)
            v = self @ k
            for k_, v_, v2_ in krylov:  # orthogonolize v (modified Gramm-Schmidt)
                c = _vdot(v, v_) / v2_
                k -= k_ * c
                v -= v_ * c
            v2 = _vdot(v)
            c = _vdot(v, res) / v2  # min_c |res - c v| => c = v.res / v.v
            newlhs = lhs + k * c
            res = rhs - self @ newlhs  # recompute rather than update to avoid drift
            newresnorm = numpy.linalg.norm(res, axis=0).max()
            if not numpy.isfinite(newresnorm) or newresnorm >= resnorm:
                break
            if newresnorm == 0.0:
                treelog.debug('solution is exact')
            else:
                treelog.debug('residual decreased by {:.1f} orders using {} krylov vectors'.format(numpy.log10(resnorm/newresnorm), len(krylov)))
            lhs = newlhs
            resnorm = newresnorm
            krylov.append((k, v, v2))
        return lhs

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

        if self._cached_submatrix is None or (rows != self._cached_rows).any() or (cols != self._cached_cols).any():
            self._cached_rows = rows
            self._cached_cols = cols
            self._cached_submatrix = self._submatrix(rows, cols)

        return self._cached_submatrix

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
        diag = numpy.empty(nrows, self.dtype)
        for irow in range(nrows):
            icols = indices[indptr[irow]:indptr[irow+1]]
            idiag = numpy.searchsorted(icols, irow)
            diag[irow] = data[indptr[irow]+idiag] if idiag < len(icols) and icols[idiag] == irow else 0
        return diag

    def getprecon(self, precon, **args):
        if (precon, args) == self._precon_args:
            return self._precon_object
        if self.shape[0] != self.shape[1]:
            raise MatrixError('matrix must be square')
        precon_args = args.copy()  # keep original args for caching purposes
        if precon_args.pop('symmetric', False) and isinstance(precon, str) and hasattr(self, '_precon_sym_' + precon):
            precon_method = getattr(self, '_precon_sym_' + precon)
            precon_name = 'symmetric ' + precon
        else:
            precon_method, precon_name = self._method('precon', precon)
        try:
            with treelog.context('constructing {} preconditioner'.format(precon_name)):
                precon_object = precon_method(**precon_args)
        except MatrixError:
            raise
        except Exception as e:
            raise MatrixError('failed to create preconditioner: {}'.format(e)) from e
        self._precon_args = precon, args
        self._precon_object = precon_object
        return precon_object

    def _precon_diag(self):
        diag = self.diagonal()
        if not diag.all():
            raise MatrixError("building 'diag' preconditioner: diagonal has zero entries")
        return numpy.reciprocal(diag).__mul__

    def __repr__(self):
        return '{}<{}x{}>'.format(type(self).__qualname__, *self.shape)


def _vdot(a, b=None):
    # Complex dot product that uses numpy.sum rather than a direct reduction for
    # slightly higher accuracy due to partial pairwise summation, see
    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    a = numpy.asarray(a)
    if b is None:
        ab = numpy.square(a.real, order='F')
        if a.dtype.kind == 'c':
            ab += numpy.square(a.imag, order='F')
    else:
        ab = numpy.multiply(a.conj(), b, order='F')
    return ab.sum(0)

# vim:sw=4:sts=4:et
