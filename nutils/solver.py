"""
The solver module defines solvers for problems of the kind ``res = 0`` or
``∂inertia/∂t + res = 0``, where ``res`` is a
:class:`nutils.evaluable.AsEvaluableArray`.  To demonstrate this consider the
following setup:

>>> from nutils import mesh, function, solver
>>> ns = function.Namespace()
>>> domain, ns.x = mesh.rectilinear([4,4])
>>> ns.basis = domain.basis('spline', degree=2)
>>> cons = domain.boundary['left,top'].project(0, onto=ns.basis, geometry=ns.x, ischeme='gauss4')
project > constrained 11/36 dofs, error 0.00e+00/area
>>> ns.u = 'basis_n ?lhs_n'

Function ``u`` represents an element from the discrete space but cannot not
evaluated yet as we did not yet establish values for ``?lhs``. It can,
however, be used to construct a residual functional ``res``. Aiming to solve
the Poisson problem ``u_,kk = f`` we define the residual functional ``res = v,k
u,k + v f`` and solve for ``res == 0`` using ``solve_linear``:

>>> res = domain.integral('(basis_n,i u_,i + basis_n) d:x' @ ns, degree=2)
>>> lhs = solver.solve_linear('lhs', residual=res, constrain=cons)
solve > solving for argument lhs (36) using direct method
solve > residual norm: 1.5e-15

The coefficients ``lhs`` represent the solution to the Poisson problem.

In addition to ``solve_linear`` the solver module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, evaluable, cache, numeric, types, _util as util, matrix, warnings, sparse
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any, Iterator, Callable
import abc
import numpy
import itertools
import functools
import collections
import math
import treelog as log


ArrayDict = Dict[str, numpy.ndarray]


# EXCEPTIONS

class SolverError(Exception):
    pass


# LINE SEARCH

@dataclass(eq=True, frozen=True)
class NormBased:
    '''
    Line search abstraction for Newton-like iterations, computing relaxation
    values that correspond to greatest reduction of the residual norm.

    Parameters
    ----------
    minscale : :class:`float`
        Minimum relaxation scaling per update. Must be strictly greater than
        zero.
    acceptscale : :class:`float`
        Relaxation scaling that is considered close enough to optimality to
        to accept the current Newton update. Must lie between minscale and one.
    maxscale : :class:`float`
        Maximum relaxation scaling per update. Must be greater than one, and
        therefore always coincides with acceptance, determining how fast
        relaxation values rebound to one if not bounded by optimality.
    '''

    minscale: float = .01
    acceptscale: float = 2/3
    maxscale: float = 2.

    def __post_init__(self):
        assert isinstance(self.minscale, float), f'minscale={self.minscale!r}'
        assert isinstance(self.acceptscale, float), f'acceptscale={self.acceptscale!r}'
        assert isinstance(self.maxscale, float), f'maxscale={self.maxscale!r}'
        assert 0 < self.minscale < self.acceptscale < 1 < self.maxscale

    def __call__(self, res0, dres0, res1, dres1):
        if not numpy.isfinite(res1).all():
            log.info('non-finite residual')
            return self.minscale, False
        # To determine optimal relaxation we minimize a polynomial estimation for
        # the residual norm: P(x) = p0 + q0 x + c x^2 + d x^3
        p0 = res0@res0
        q0 = 2*res0@dres0
        p1 = res1@res1
        q1 = 2*res1@dres1
        if q0 >= 0:
            raise SolverError('search vector does not reduce residual')
        c = math.fsum([-3*p0, 3*p1, -2*q0, -q1])
        d = math.fsum([2*p0, -2*p1, q0, q1])
        # To minimize P we need to determine the roots for P'(x) = q0 + 2 c x + 3 d x^2
        # For numerical stability we use Citardauq's formula: x = -q0 / (c +/- sqrt(D)),
        # with D the discriminant
        D = c**2 - 3 * q0 * d
        # If D <= 0 we have at most one duplicate root, which we ignore. For D > 0,
        # taking into account that q0 < 0, we distinguish three situations:
        # - d > 0 => sqrt(D) > abs(c): one negative, one positive root
        # - d = 0 => sqrt(D) = abs(c): one negative root
        # - d < 0 => sqrt(D) < abs(c): two roots of same sign as c
        scale = -q0 / (c + math.sqrt(D)) if D > 0 and (c > 0 or d > 0) else math.inf
        if scale >= 1 and p1 > p0:  # this should not happen, but just in case
            log.info('failed to estimate scale factor')
            return self.minscale, False
        log.info('estimated residual minimum at {:.0f}% of update vector'.format(scale*100))
        return min(max(scale, self.minscale), self.maxscale), scale >= self.acceptscale and p1 < p0


@dataclass(eq=True, frozen=True)
class MedianBased:
    '''
    Line search abstraction for Newton-like iterations, computing relaxation
    values such that half (or any other configurable quantile) of the residual
    vector has its optimal reduction beyond it. Unline the :class:`NormBased`
    approach this is invariant to constant scaling of the residual items.

    Parameters
    ----------
    minscale : :class:`float`
        Minimum relaxation scaling per update. Must be strictly greater than
        zero.
    acceptscale : :class:`float`
        Relaxation scaling that is considered close enough to optimality to
        to accept the current Newton update. Must lie between minscale and one.
    maxscale : :class:`float`
        Maximum relaxation scaling per update. Must be greater than one, and
        therefore always coincides with acceptance, determining how fast
        relaxation values rebound to one if not bounded by optimality.
    quantile : :class:`float`
        Fraction of the residual vector that is aimed to have its optimal
        reduction at a smaller relaxation value. The default value of one half
        corresponds to the median. A value close to zero means tighter control,
        resulting in strong relaxation.
    '''

    minscale: float = .01
    acceptscale: float = 2/3
    maxscale: float = 2.
    quantile: float = .5

    def __post_init__(self):
        assert isinstance(self.minscale, float), f'minscale={self.minscale!r}'
        assert isinstance(self.acceptscale, float), f'acceptscale={self.acceptscale!r}'
        assert isinstance(self.maxscale, float), f'maxscale={self.maxscale!r}'
        assert isinstance(self.quantile, float), f'quantile={self.quantile!r}'
        assert 0 < self.minscale < self.acceptscale < 1 < self.maxscale
        assert 0 < self.quantile < 1

    def __call__(self, res0, dres0, res1, dres1):
        if not numpy.isfinite(res1).all():
            log.info('non-finite residual')
            return self.minscale, False
        # To determine optimal relaxation we minimize a polynomial estimation for
        # the squared residual: P(x) = p0 + q0 x + c x^2 + d x^3
        dp = res1**2 - res0**2
        q0 = 2*res0*dres0
        q1 = 2*res1*dres1
        mask = q0 <= 0  # ideally this mask is all true, but solver inaccuracies can result in some positive slopes
        n = round(len(res0)*self.quantile) - (~mask).sum()
        if n < 0:
            raise SolverError('search vector fails to reduce more than {}-quantile of residual vector'.format(self.quantile))
        c = 3*dp - 2*q0 - q1
        d = -2*dp + q0 + q1
        D = c**2 - 3 * q0 * d
        mask &= D > 0
        numer = -q0[mask]
        denom = c[mask] + numpy.sqrt(D[mask])
        mask = denom > 0
        if n < mask.sum():
            scales = numer[mask] / denom[mask]
            scales.sort()
            scale = scales[n]
        else:
            scale = numpy.inf
        log.info('estimated {}-quantile at {:.0f}% of update vector'.format(self.quantile, scale*100))
        return min(max(scale, self.minscale), self.maxscale), scale >= self.acceptscale


# SYSTEM

class System:
    '''System of one or more variables.

    The System class represents a linear or nonlinear problem of one or more
    variables, and offers several methods to solve it. The main solution method
    is ``solve``, which serves as a general entry point that offloads the
    problem to specialized solvers. The other prominent method is ``step``,
    which adds functionality to deal with time dependent problems.
    '''

    def __init__(self, residual: Union[evaluable.AsEvaluableArray,Tuple[evaluable.AsEvaluableArray,...]], /, trial: Union[str,Tuple[str,...]], test: Optional[Union[str,Tuple[str,...]]] = None):
        '''Construct a system.

        Parameters
        ----------
        residual :
            Any object that supports conversion to a scalar evaluable Array via
            the AsEvaluableArray protocol. This is the function that the trial
            arguments will seek to make constant in all the test arguments, or
            zero in all the derivatives. If provided as a tuple of vectors then
            these are already the derivatives to be made zero, and no test
            arguments may be specified.
        trial :
            Names of the trial arguments, provided as a string ('arg1,arg2') or
            tuple of strings (('arg1', 'arg2')).
        test :
            Names of the test arguments (optional) provided as a string or
            tuple of strings. Must be omitted if residual is a tuple of
            vectors; otherwise, may be omitted if test and trial are equal.
        '''

        self.trials = tuple(trial.split(',') if isinstance(trial, str) else trial)

        if isinstance(residual, (tuple, list)):
            if test is not None:
                raise ValueError('a test argument is not allowed in combination with a residual vector')
            residuals = [res.as_evaluable_array for res in residual]
            self.dtype = residuals[0].dtype
            if not all(v.dtype == self.dtype for v in residuals):
                raise ValueError('inconsistent data types in residual vector')
            argobjects = _dict((arg.name, arg) for res in residuals for arg in res.arguments if isinstance(arg, evaluable.Argument))
            self.is_symmetric = False
        else:
            value = residual.as_evaluable_array
            self.dtype = value.dtype
            argobjects = {arg.name: arg for arg in value.arguments if isinstance(arg, evaluable.Argument)}
            tests = self.trials if test is None else tuple(test.split(',') if isinstance(test, str) else test)
            residuals = [evaluable.derivative(value, argobjects[t]) for t in tests]
            self.is_symmetric = self.trials == tests

        self.arguments = frozenset(argobjects)

        self.trial_args = tuple(argobjects[t] for t in self.trials)
        self.trial_shapes = evaluable.eval_once([arg.shape for arg in self.trial_args])

        trial_sizes = tuple(numpy.prod(shape, dtype=int) for shape in self.trial_shapes)
        trial_offsets = numpy.cumsum([0, *trial_sizes])

        self.__trial_total = trial_offsets[-1]
        self.__trial_slices = tuple(map(slice, trial_offsets, trial_offsets[1:]))

        block_residual = [evaluable._flat(res) for res in residuals]
        block_jacobian = [[evaluable._flat(evaluable.derivative(res, arg).simplified, 2) for arg in self.trial_args] for res in block_residual]

        self.is_linear = not any(arg.name in self.trials for row in block_jacobian for col in row for arg in col.arguments)
        self.is_constant_matrix = self.is_linear and not any(col.arguments for row in block_jacobian for col in row)

        self.__block_jacobian = tuple(tuple(map(evaluable.as_csr, row)) for row in block_jacobian)
        self.__block_residual = tuple(block_residual)
        if self.is_symmetric:
            self.__value = value

        self.__cache = {}

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('System', self.trials, self.__value if self.is_symmetric else self.__block_residual))

    def deconstruct(self, arguments, constrain):
        arguments = arguments.copy()
        xparts = []
        for t, shape in zip(self.trials, self.trial_shapes):
            a = arguments.get(t)
            c = constrain.get(t)
            if a is None:
                if c is None:
                    a = numpy.full(shape, numpy.nan, dtype=self.dtype)
                elif c.dtype == bool:
                    a = numpy.full(shape, numpy.nan, dtype=self.dtype)
                    a[c] = 0.
                else:
                    a = c
                x = numpy.zeros(numpy.isnan(a).sum(), dtype=self.dtype)
            else:
                if c is None:
                    x = a.ravel()
                    a = numpy.full(shape, numpy.nan, dtype=self.dtype)
                elif c.dtype == bool:
                    x = a[~c]
                    a = a.copy()
                    a[~c] = numpy.nan
                else:
                    x = a[numpy.isnan(c)]
                    a = c
                assert numpy.isfinite(x).all()
            assert a.dtype == self.dtype
            assert x.dtype == self.dtype
            arguments[t] = a
            xparts.append(x)
        return arguments, numpy.concatenate(xparts, dtype=self.dtype)

    def construct(self, arguments, x=None, return_free=False):
        v = numpy.concatenate([arguments[t].ravel() for t in self.trials])
        free = numpy.isnan(v)
        if x is None:
            assert not free.any()
            free = return_free and numpy.ones_like(free)
        else:
            v[free] = x
            arguments = arguments | {t: v[s].reshape(shape) for t, shape, s in zip(self.trials, self.trial_shapes, self.__trial_slices)}
        return (arguments, free) if return_free else arguments

    def assemble_jacobian(self, arguments: ArrayDict, x=None):
        arguments, free = self.construct(arguments, x, return_free=True)
        key = 'jacobian'
        jacobian = self.__cache.get(key)
        if jacobian is None:
            if self.is_constant_matrix:
                jacobian = evaluable.eval_once(self.__block_jacobian)
                self.__cache[key] = jacobian
            else:
                eval_ = evaluable.compile(self.__block_jacobian)
                jacobian = matrix.assemble_block_csr(eval_(arguments))
                self.__cache[key] = eval_
        elif not self.is_constant_matrix:
            jacobian = matrix.assemble_block_csr(jacobian(arguments))
        return jacobian.submatrix(free, free)

    def assemble_residual(self, arguments: ArrayDict, x=None):
        arguments, free = self.construct(arguments, x, return_free=True)
        key = 'residual'
        residual = self.__cache.get(key)
        if residual is None:
            residual = evaluable.compile(self.__block_residual)
            self.__cache[key] = residual
        res = numpy.concatenate(residual(arguments))
        return res[free]

    def assemble_value(self, arguments: ArrayDict, x=None):
        if not self.is_symmetric:
            raise Exception('value is not defined')
        arguments = self.construct(arguments, x)
        key = 'value'
        value = self.__cache.get(key)
        if value is None:
            value = evaluable.compile(self.__value)
            self.__cache[key] = value
        return value(arguments)

    def assemble_jacobian_residual(self, arguments: ArrayDict, x=None):
        arguments, free = self.construct(arguments, x, return_free=True)
        key = 'jacobian_residual'
        f = self.__cache.get(key)
        if self.is_linear:
            if f is None:
                z = dict(zip(self.trials, map(evaluable.zeros_like, self.trial_args)))
                block_residual = tuple(evaluable.replace_arguments(vector, z).simplified for vector in self.__block_residual)
                f = evaluable.compile((self.__block_jacobian, block_residual))
                self.__cache[key] = f
            jac_blocks, res_blocks = f(arguments)
            if self.is_constant_matrix:
                key = 'jacobian'
                jac = self.__cache.get(key)
                if jac is None:
                    jac = matrix.assemble_block_csr(jac_blocks)
                    self.__cache[key] = jac
            else:
                jac = matrix.assemble_block_csr(jac_blocks)
            res = numpy.concatenate(res_blocks)
            res += jac @ numpy.concatenate([arguments[t].ravel() for t in self.trials])
        else:
            if f is None:
                f = evaluable.compile((self.__block_jacobian, self.__block_residual))
                self.__cache[key] = f
            jac_blocks, res_blocks = f(arguments)
            jac = matrix.assemble_block_csr(jac_blocks)
            res = numpy.concatenate(res_blocks)
        return jac.submatrix(free, free), res[free]

    def assemble_jacobian_residual_value(self, arguments: ArrayDict, x=None):
        if not self.is_symmetric:
            raise Exception('value is not defined')
        arguments, free = self.construct(arguments, x, return_free=True)
        key = 'jacobian_residual_value'
        f = self.__cache.get(key)
        if self.is_linear:
            if f is None:
                z = dict(zip(self.trials, map(evaluable.zeros_like, self.trial_args)))
                f = evaluable.compile((
                    self.__block_jacobian,
                    tuple(evaluable.replace_arguments(vector, z) for vector in self.__block_residual),
                    evaluable.replace_arguments(self.__value, z)))
                self.__cache[key] = f
            jac_blocks, res_blocks, val = f(arguments)
            if self.is_constant_matrix:
                key = 'jacobian'
                jac = self.__cache.get(key)
                if jac is None:
                    jac = matrix.assemble_block_csr(jac_blocks)
                    self.__cache[key] = jac
            else:
                jac = matrix.assemble_block_csr(jac_blocks)
            res = numpy.concatenate(res_blocks)
            x = numpy.concatenate([arguments[t].ravel() for t in self.trials])
            jac_x = jac @ x
            val = self.dtype(val + res @ x + .5 * (x @ jac_x))
            res += jac_x
        else:
            if f is None:
                f = evaluable.compile((self.__block_jacobian, self.__block_residual, self.__value))
                self.__cache[key] = f
            jac_blocks, res_blocks, val = f(arguments)
            jac = matrix.assemble_block_csr(jac_blocks)
            res = numpy.concatenate(res_blocks)
            val = self.dtype(val)
        return jac.submatrix(free, free), res[free], val

    def assemble(self, arguments: ArrayDict, x=None):
        if self.is_symmetric:
            return self.assemble_jacobian_residual_value(arguments, x)
        else:
            return *self.assemble_jacobian_residual(arguments, x), None

    @property
    def _trial_info(self):
        return ' and '.join(t + ' (' + ','.join(map(str, shape)) + ')' for t, shape in zip(self.trials, self.trial_shapes))

    MethodValue = Tuple[ArrayDict, float]
    MethodIter = Iterator[MethodValue]

    @cache.function
    @log.withcontext
    def solve(self, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}, tol: float = 0., miniter: int = 0, maxiter: Optional[int] = None, method: Optional[Callable[...,Union[MethodIter,MethodValue]]] = None) -> Tuple[ArrayDict, float]:
        '''Solve the system.

        Determines the trial arguments for which the derivatives of the
        system's scalar functional are zero in all the test arguments.

        Parameters
        ----------
        arguments
            Arguments required to evaluate the system. Arguments that coincide
            with the system's trial arguments serve as initial guess.
        constrain
            Constrained values for any of the trial arguments, supplied as
            float vectors with NaN entries or boolean vectors, the latter
            holding values at those of the initial guess.
        tol
            Maximum residual norm, stopping criterion for the iterative solver.
            Required for nonlinear problems.
        miniter
            Minimum number of iterations for the iterative solver, overruling
            to ``tol`` based stopping criterion.
        maxiter
            Maximum number of iterations, after which a ``SolverError`` is
            raised.
        method
            Iterative solution method, in the form of a callable that returns
            an iterator of (arguments, resnorm, value) triplets.
        '''

        if method is None:
            method = (Direct if self.is_linear else Newton)(atol=tol)
        log.info(f'{"optimizing" if self.is_symmetric else "solving"} for argument {self._trial_info} using {method} method')

        m = method(self, arguments=arguments, constrain=constrain)
        if isinstance(m, tuple):
            arguments, resnorm = m
            log.info(f'residual norm: {resnorm:.1e}')
            if resnorm > tol > 0:
                raise SolverError(f'failed to reach desired tolerance of {tol:.0e}')
        else:
            if tol <= 0:
                raise ValueError('iterative solver requires a strictly positive tolerance')
            arguments, resnorm = next(m)
            with log.context(f'iter 0'):
                log.info(f'residual norm: {resnorm:.1e}')
            resnorm0 = resnorm
            iiter = 0
            while iiter < miniter or resnorm > tol:
                if maxiter is not None and iiter >= maxiter:
                    raise SolverError(f'failed to converge in {maxiter} iterations')
                iiter += 1
                with log.context(f'iter {iiter}'):
                    arguments, resnorm = next(m)
                    progress = numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol) if resnorm > tol else 1
                    log.info(f'residual norm: {resnorm:.1e} ({100*progress:.0f}%)')

        if self.is_symmetric:
            val = self.assemble_value(arguments)
            log.info(f'optimal value: {val:.1e}')
        return arguments

    def step(self, *, arguments: ArrayDict, suffix: str, timearg: Optional[str] = None, timesteparg: Optional[str] = None, timestep: Optional[float] = None, maxretry: int = 2, **solveargs) -> ArrayDict:
        '''Advance a time step.

        This method is best described by an example. Let ``timearg`` equal 't'
        and ``suffix`` '0', and let our system's trial arguments be 'u' and 'v'.
        This method then creates argument copies 't0', 'u0', 'v0', advances 't'
        by ``timestep``, and solves for the new 'u' and 'v'.

        Parameters
        ----------
        arguments
            Arguments required to evaluate the system. Arguments that coincide
            with the system's trial arguments serve as initial value.
        suffix
            String suffix to add to argument names to denote their value at the
            beginning of the time step.
        timearg
            Name of the scalar argument that tracks the time (optional).
        timesteparg
            Name of the scalar argument that tracks the timestep (optional).
        timestep
            Size of the time increment (required if either timearg or
            timesteparg are specified).
        maxretry
            If either timearg or timesteparg are specified and affecting the
            system, then this positive integer determines how many levels of
            timestep bisections will be considered to recover from solver or
            matrix errors.
        solveargs
            Remaining keyword arguments are passed on to the ``solver`` method.
        '''

        arguments = arguments.copy()
        for trial in self.trials:
            if trial in arguments:
                arguments[trial + suffix] = arguments[trial]
        if timearg or timesteparg:
            if timestep is None:
                raise ValueError('timearg and timesteparg require timestep to be specified')
            if timesteparg:
                arguments[timesteparg] = timestep
            if timearg:
                time = arguments.get(timearg, 0.)
                arguments[timearg + suffix] = time
                arguments[timearg] = time + timestep
        try:
            return self.solve(arguments=arguments, **solveargs)
        except (SolverError, matrix.MatrixError) as e:
            if timearg not in self.arguments and timesteparg not in self.arguments or maxretry <= 0:
                raise
            log.error(f'error: {e}; retrying with timestep {timestep/2}')
            halfstep_args = dict(solveargs, timestep=timestep/2, timearg=timearg, timesteparg=timesteparg, suffix=suffix, maxretry=maxretry-1)
            with log.context('retry 1/2'):
                halfway_arguments = self.step(arguments=arguments, **halfstep_args)
            with log.context('retry 2/2'):
                return self.step(arguments=halfway_arguments, **halfstep_args)

    @cache.function
    @log.withcontext
    def solve_constraints(self, *, droptol: Optional[float], arguments: ArrayDict = {}, constrain: ArrayDict = {}, linargs: Dict[str, Any] = {}) -> Tuple[ArrayDict, float]:
        '''Solve for Dirichlet constraints.

        This method is similar to ``solve``, but with two key differences.

        The method is limited to linear systems, but adds the ability to solve
        a limited class of singular systems. It does so by isolating the subset
        of arguments that contribute (up to droptol) to the residual, and
        solving the corresponding submatrix. The remaining argument values are
        returned as NaN (not a number).

        The second key difference with solve is that the returned dictionary
        is augmented with the remaining _constrain_ items, rather than those
        from _arguments_, reflecting the method's main utility of forming
        Dirichlet constraints. This allows for the aggregation of constraints
        by calling the method multiple times in series.

        Parameters
        ----------
        arguments
            Arguments required to evaluate the system. Any arguments that
            coincide with the system's trial arguments serve as initial guess.
        constrain
            Constrained values for any of the trial arguments, supplied as
            float vectors with NaN entries or boolean vectors, the latter
            holding values at those of the initial guess.
        linargs
            Keyword arguments to be passed on to the linear solver.
        droptol
            Minimum absolute value if matrix entries for rows and columns to
            participate in the optimization problem.
        '''

        log.info(f'{"optimizing" if self.is_symmetric else "solving"} for argument {self._trial_info} with drop tolerance {droptol:.0e}')
        if not self.is_linear:
            raise ValueError('system is not linear')
        arguments, x = self.deconstruct(arguments, constrain)
        jac, res, val = self.assemble(arguments, x)
        data, colidx, _ = jac.export('csr')
        mycons = numpy.ones(res.shape, dtype=bool)
        mycons[colidx[abs(data) > droptol]] = False # unconstrain dofs with nonzero columns
        dx = -jac.solve(res, constrain=mycons, **_copy_with_defaults(linargs, symmetric=self.is_symmetric))
        log.info(f'residual norm: {numpy.linalg.norm((jac @ dx + res)[~mycons]):.1e}')
        if self.is_symmetric:
            log.info(f'optimal value: {val+.5*(res@dx):.1e}') # val(x + dx) = val(x) + res(x) dx + .5 dx jac dx
        x += dx
        for trial, s in zip(self.trials, self.__trial_slices):
            log.info(f'constrained {len(mycons[s])-mycons[s].sum()} degrees of freedom of {trial}')
        x[mycons] = numpy.nan
        arguments = self.construct(arguments, x)
        return constrain | {t: arguments[t] for t in self.trials}


class Direct:
    '''Direct solution of a linear system.'''

    def __init__(self, **linargs):
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('Direct', self.linargs))

    def __str__(self):
        return 'direct'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodValue:
        if not system.is_linear:
            raise ValueError('problem is not linear')
        arguments, x = system.deconstruct(arguments, constrain)
        linargs = _copy_with_defaults(self.linargs, symmetric=system.is_symmetric)
        jac, res = system.assemble_jacobian_residual(arguments, x)
        dx = jac.solve(res, **linargs)
        x -= dx
        return system.construct(arguments, x), numpy.linalg.norm(res - jac @ dx)


class Newton:
    '''Newton solver.

    This solver generates the iterates ``x_n+1 = x_n - J(x_n)^-1 r(x_n)``
    consistent with a standard Newton process.'''

    def __init__(self, **linargs):
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('Newton', self.linargs))

    def __str__(self):
        return 'newton'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        arguments, x = system.deconstruct(arguments, constrain)
        linargs = _copy_with_defaults(self.linargs, rtol=1-3, symmetric=system.is_symmetric)
        while True:
            jac, res = system.assemble_jacobian_residual(arguments, x)
            yield system.construct(arguments, x), numpy.linalg.norm(res)
            x -= jac.solve_leniently(res, **linargs)


class ReuseNewton:
    '''Newton with approximate Jacobians.

    This is a modification of the vanilla Newton process, in which a new
    Jacobian matrix is NOT created for every update, but an existing matrix is
    reused instead for as long as ``|res(x - J^-1 res(x))| < require
    |res(x)|``.

    When the condition no longer holds, a new Jacobian matrix is assembled and
    a standard Newton update is performed without regard for the ``require``
    parameter.
    '''

    def __init__(self, require: float = .5, **linargs):
        self.require = require
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('ReuseNewton', self.require, self.linargs))

    def __str__(self):
        return 'reuse-newton'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        arguments, x = system.deconstruct(arguments, constrain)
        linargs = _copy_with_defaults(self.linargs, rtol=1-3, symmetric=system.is_symmetric)

        res = system.assemble_residual(arguments, x)
        resnorm = numpy.linalg.norm(res)

        yield system.construct(arguments, x), resnorm

        update_jacobian = True

        while True:

            if update_jacobian:
                log.info('updating jacobian matrix')
                jac = system.assemble_jacobian(arguments, x)

            newx = x - jac.solve_leniently(res, **linargs)
            newres = system.assemble_residual(arguments, newx)
            newresnorm = numpy.linalg.norm(newres)

            if update_jacobian or newresnorm < self.require * resnorm:
                resnorm = newresnorm
                res = newres
                x = newx
                yield system.construct(arguments, x), resnorm
                update_jacobian = False
            else:
                update_jacobian = True


class LinesearchNewton:
    '''Newton solver with automatic relaxation.

    This solver generates the iterates ``x_n+1 = x_n - α J(x_n)^-1 r(x_n)``,
    where ``0 < α <= 1`` is a relaxation value. The value for ``α`` is
    initially ``relax0``, and subsequently determined by a configurable
    ``strategy``. If the value falls below ``failrelax`` then the solver fails
    with a ``SolverError``.

    The default strategy is ``NormBased``, which employs a polynomial
    interpolation of the residual norm between the current value of ``x`` and
    the prospective next value to determine at what value of ``α`` the residual
    norm is likely minimal.
    '''

    def __init__(self, strategy: Callable = NormBased(), failrelax: float = 1e-6, relax0: float = 1., **linargs):
        assert callable(strategy), f'invalid linesearch strategy {strategy!r}'
        self.strategy = strategy
        self.failrelax = failrelax
        self.relax0 = relax0
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('LinesearchNewton', self.strategy, self.failrelax, self.relax0, self.linargs))

    def __str__(self):
        return 'linesearch-newton'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        arguments, x = system.deconstruct(arguments, constrain)
        linargs = _copy_with_defaults(self.linargs, rtol=1-3, symmetric=system.is_symmetric)
        jac, res = system.assemble_jacobian_residual(arguments, x)
        relax = self.relax0
        while True: # newton iterations
            yield system.construct(arguments, x), numpy.linalg.norm(res)
            dx = -jac.solve_leniently(res, **linargs)
            res0 = res
            jac0dx = jac@dx # == res0 if dx was solved to infinite precision
            while True: # line search
                newx = x + dx * relax
                jac, res = system.assemble_jacobian_residual(arguments, newx)
                scale, accept = self.strategy(res0, (jac0dx)*relax, res, (jac@dx)*relax)
                if accept:
                    log.info('update accepted at relaxation', round(relax, 5))
                    relax = min(relax * scale, 1)
                    break
                assert scale < 1
                relax *= scale
                if relax <= self.failrelax:
                    raise SolverError('stuck in local minimum')
            x = newx


class Minimize:
    '''Direct minimization of scalar functional.

    This solver constructs a trajectory of steepest descent to find the
    minimizer of a locally convex scalar function.
    '''

    def __init__(self, rampup: float = .5, rampdown: float = -1., failrelax: float = -10., **linargs):
        self.rampup = rampup
        self.rampdown = rampdown
        self.failrelax = failrelax
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('Minimize', self.rampup, self.rampdown, self.failrelax, self.linargs))

    def __str__(self):
        return 'minimize'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        if not system.is_symmetric:
            raise ValueError('problem is not symmetric')
        arguments, x = system.deconstruct(arguments, constrain)
        jac, res, val = system.assemble_jacobian_residual_value(arguments, x)
        linargs = _copy_with_defaults(self.linargs, rtol=1-3, symmetric=system.is_symmetric)
        relax = 0.
        while True:
            yield system.construct(arguments, x), numpy.linalg.norm(res)
            dx = -jac.solve_leniently(res, **linargs)
            x += dx # baseline: vanilla Newton
            # compute first two ritz values to determine approximate path of steepest descent
            dxnorm = numpy.linalg.norm(dx)
            k0 = dx / dxnorm
            k1 = -res / dxnorm # == jac @ k0
            a = k1 @ k0
            k1 -= k0 * a # orthogonalize
            c = numpy.linalg.norm(k1)
            k1 /= c # normalize
            b = k1 @ (jac @ k1)
            # at this point k0 and k1 are orthonormal, and [k0 k1]^T jac [k0 k1] = [a c; c b]
            D = numpy.hypot(b-a, 2*c)
            L = numpy.array([a+b-D, a+b+D]) / 2 # 2nd order ritz values: eigenvalues of [a c; c b]
            v0, v1 = res + dx * L[:, numpy.newaxis]
            V = numpy.stack([v1, -v0], axis=1) / D # ritz vectors times dx -- note: V @ L == -res, V.sum() == dx
            log.info('spectrum: {:.1e}..{:.1e} ({}definite)'.format(*L, 'positive ' if L[0] > 0 else 'negative ' if L[-1] < 0 else 'in'))
            val0 = val
            while True: # line search along steepest descent curve
                r = numpy.exp(relax - numpy.log(D)) # == exp(relax) / D
                eL = numpy.exp(-r*L)
                newx = x - V @ eL
                jac, res, val = system.assemble_jacobian_residual_value(arguments, newx)
                slope = res @ (V @ (eL*L))
                log.info('energy {:+.2e} / e{:+.1f} and {}creasing'.format(val - val0, relax, 'in' if slope > 0 else 'de'))
                if numpy.isfinite(val) and numpy.isfinite(res).all() and val <= val0 and slope <= 0:
                    relax += self.rampup
                    break
                relax += self.rampdown
                if relax <= self.failrelax:
                    raise SolverError('stuck in local minimum')
            x = newx


class Arnoldi:
    '''Arnoldi iterations for linear systems.

    Upon first use, the Arnoldi solver behaves identically to Direct, in that
    it returns the solution to the linear system ``A x = b``.

    The difference arises when the method is reused. If the matrix and residual
    depend on a second set of arguments, as in ``A(y) x = b(y)``, then the
    Arnoldi solver will aim to benefit from an existing factorization of ``P =
    A(y_previous)^-1`` by projecting ``x`` on the growing subspace ``[P b(y),
    P^2 b(y), P^3 b(y), ..)``.

    If the desired tolerance is not reached at a configurable subspace
    dimension of ``maxiter``, then the system is solved with ``A(y)`` and the
    cached matrix updated for subsequent reuse.
    '''

    def __init__(self, maxiter: int = 2, **linargs):
        self.maxiter = maxiter
        self.linargs = linargs
        self.__cached_matrix = None

    def __str__(self):
        return 'arnoldi'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        if not system.is_linear:
            raise ValueError('problem is not linear')

        arguments, x = system.deconstruct(arguments, constrain)
        jac, res = system.assemble_jacobian_residual(arguments, x)
        yield system.construct(arguments, x), numpy.linalg.norm(res)

        linargs = _copy_with_defaults(self.linargs, solver='direct', symmetric=system.is_symmetric)
        approx_jac = self.__cached_matrix
        if approx_jac is not None and approx_jac.shape == jac.shape:
            dx_space, dres_space = numpy.empty((2, self.maxiter, len(res)))
            for i in range(self.maxiter):
                try:
                    dx_space[i] = approx_dx = approx_jac.solve(res, **linargs)
                    dres_space[i] = jac @ approx_dx
                    w, (res2,), *_ = numpy.linalg.lstsq(dres_space[:i+1].T, res, rcond=None)
                    resnorm = numpy.sqrt(res2)
                except Exception as e:
                    log.warning('solution failed:', e)
                    break
                x -= w @ dx_space[:i+1]
                yield system.construct(arguments, x), resnorm
                res -= w @ dres_space[:i+1]

        log.info(f'updating jacobian')
        self.__cached_matrix = jac

        linargs = _copy_with_defaults(self.linargs, symmetric=system.is_symmetric)
        dx = jac.solve(res, **linargs)
        x -= dx
        yield system.construct(arguments, x), numpy.linalg.norm(res - jac @ dx)


class Pseudotime:
    '''Inertia assisted Newton.

    This solver is a variant of the Newton solver for problems that do not
    converge along the standard Newton path. For problems that represent the
    steady state solution of a dynamic system, the pseudo-time solver uses a
    provided inertia matrix to progress towards the steady state along the
    (approximately) physical path using ever larger time steps.

    Concretely, rather than forming updates of the form ``J(x) dx = r(x)``, the
    pseudo-time solver solves ``(J(x) + dJ / dt) dx = r(x)``, where ``dJ`` is
    the provided inertia matrix and ``dt`` is the configured initial
    ``timestep`` scaled by the ratio of ``|r(x0)| / |r(x)|``.
    '''

    def __init__(self, inertia: Tuple[evaluable.AsEvaluableArray,...], timestep: float, **linargs):
        self.inertia = inertia
        self.timestep = timestep
        self.linargs = linargs

    @property
    def __nutils_hash__(self):
        return types.nutils_hash(('Pseudotime', self.inertia, self.timestep, self.linargs))

    def __str__(self):
        return 'pseudotime'

    def __call__(self, system, *, arguments: ArrayDict = {}, constrain: ArrayDict = {}) -> System.MethodIter:
        arguments, x = system.deconstruct(arguments, constrain)
        free = numpy.concatenate([numpy.isnan(arguments[t]).ravel() for t in system.trials])
        linargs = _copy_with_defaults(self.linargs, rtol=1-3)
        djac = matrix.assemble_block_csr(evaluable.eval_once([[evaluable.as_csr(evaluable._flat(evaluable.derivative(evaluable._flat(res), arg).simplified, 2)) for arg in system.trial_args] for res in self.inertia], arguments=arguments)).submatrix(free, free)

        first = _First()
        while True:
            jac, res = system.assemble_jacobian_residual(arguments, x)
            resnorm = numpy.linalg.norm(res)
            yield system.construct(arguments, x), resnorm
            timestep = self.timestep * (first(resnorm) / resnorm)
            log.info(f'timestep: {timestep:.0e}')
            x -= (jac + djac / timestep).solve_leniently(res, **linargs)


# SOLVERS

def solve_linear(target, residual, *, constrain = None, lhs0: types.arraydata = None, arguments = {}, **kwargs):
    '''solve linear problem

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    residual : :class:`nutils.evaluable.AsEvaluableArray`
        Residual integral, depends on ``target``
    constrain : :class:`numpy.ndarray` with dtype :class:`float`
        Defines the fixed entries of the coefficient vector
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`.  The ``target`` should not be present in ``arguments``.
        Optional.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of ``target`` values for which ``residual == 0``'''

    if isinstance(target, str) and ',' not in target and ':' not in target:
        return solve_linear([target], [residual], constrain={} if constrain is None else {target: constrain},
            lhs0=lhs0, arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, **kwargs)[target]
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
    linargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    system = System(residual, *_split_trial_test(target))
    if not system.is_linear:
        raise SolverError('problem is not linear')
    return system.solve(arguments=arguments, constrain=constrain or {}, method=Direct(**linargs))


def newton(target, residual, *, jacobian = None, lhs0 = None, relax0: float = 1., constrain = None, linesearch=NormBased(), failrelax: float = 1e-6, arguments = {}, **kwargs):
    '''iteratively solve nonlinear problem by gradient descent

    Generates targets such that residual approaches 0 using Newton procedure with
    line search based on the residual norm. Suitable to be used inside ``solve``.

    An optimal relaxation value is computed based on the following cubic
    assumption::

        |res(lhs + r * dlhs)|^2 = A + B * r + C * r^2 + D * r^3

    where ``A``, ``B``, ``C`` and ``D`` are determined based on the current
    residual and tangent, the new residual, and the new tangent. If this value is
    found to be close to 1 then the newton update is accepted.

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    residual : :class:`nutils.evaluable.AsEvaluableArray`
    relax0 : :class:`float`
        Initial relaxation value.
    constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
        Masks the free vector entries as ``False`` (boolean) or NaN (float). In
        the remaining positions the values of ``lhs0`` are returned unchanged
        (boolean) or overruled by the values in `constrain` (float).
    linesearch : Callable[[float, float, float, float], Tuple[float, bool]]
        Callable that defines relaxation logic. The callable takes four
        arguments: the current residual and directional derivative, and the
        candidate residual and directional derivative, with derivatives
        normalized to unit length; and returns the optimal scaling and a
        boolean flag that marks whether the candidate should be accepted.
    failrelax : :class:`float`
        Fail with exception if relaxation reaches this lower limit.
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`. If ``target`` is present in ``arguments`` then it is used
        as the initial guess for the iterative procedure.

    Yields
    ------
    :class:`numpy.ndarray`
        Coefficient vector that approximates residual==0 with increasing accuracy
    '''

    if isinstance(target, str) and ',' not in target and ':' not in target:
        return newton([target], [residual], jacobian=None if jacobian is None else [jacobian],
            relax0=relax0, constrain={} if constrain is None else {target: constrain}, linesearch=linesearch,
            failrelax=failrelax, arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, **kwargs)[target]
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
    if jacobian is not None:
        warnings.warn('jac argument is no longer in use')
    linargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    system = System(residual, *_split_trial_test(target))
    method = Newton(**linargs) if not linesearch \
        else LinesearchNewton(strategy=linesearch, relax0=relax0, failrelax=failrelax, **linargs)
    return _with_solve(system, method, arguments, constrain or {})


def minimize(target, energy: evaluable.asarray, *, lhs0: types.arraydata = None, constrain = None, rampup: float = .5, rampdown: float = -1., failrelax: float = -10., arguments = {}, **kwargs):
    '''iteratively minimize nonlinear functional by gradient descent

    Generates targets such that residual approaches 0 using Newton procedure with
    line search based on the energy. Suitable to be used inside ``solve``.

    An optimal relaxation value is computed based on the following assumption::

        energy(lhs + r * dlhs) = A + B * r + C * r^2 + D * r^3 + E * r^4 + F * r^5

    where ``A``, ``B``, ``C``, ``D``, ``E`` and ``F`` are determined based on the
    current and new energy, residual and tangent. If this value is found to be
    close to 1 then the newton update is accepted.

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    residual : :class:`nutils.evaluable.AsEvaluableArray`
    constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
        Masks the free vector entries as ``False`` (boolean) or NaN (float). In
        the remaining positions the values of ``lhs0`` are returned unchanged
        (boolean) or overruled by the values in `constrain` (float).
    rampup : :class:`float`
        Value to increase the relaxation power by in case energy is decreasing.
    rampdown : :class:`float`
        Value to decrease the relaxation power by in case energy is increasing.
    failrelax : :class:`float`
        Fail with exception if relaxation reaches this lower limit.
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`. If ``target`` is present in ``arguments`` then it is used
        as the initial guess for the iterative procedure.

    Yields
    ------
    :class:`numpy.ndarray`
        Coefficient vector that approximates residual==0 with increasing accuracy
    '''

    if isinstance(target, str) and ',' not in target:
        return minimize([target], energy, constrain={} if constrain is None else {target: constrain}, rampup=rampup, rampdown=rampdown,
            failrelax=failrelax, arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, **kwargs)[target]
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
    linargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    system = System(energy, *_split_trial_test(target))
    method = Minimize(rampup=rampup, rampdown=rampdown, failrelax=failrelax, **linargs)
    return _with_solve(system, method, arguments, constrain or {})


def pseudotime(target, residual, inertia, timestep: float, *, lhs0: types.arraydata = None, constrain = None, arguments = {}, **kwargs):
    '''iteratively solve nonlinear problem by pseudo time stepping

    Generates targets such that residual approaches 0 using hybrid of Newton and
    time stepping. Requires an inertia term and initial timestep. Suitable to be
    used inside ``solve``.

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    residual : :class:`nutils.evaluable.AsEvaluableArray`
    inertia : :class:`nutils.evaluable.AsEvaluableArray`
    timestep : :class:`float`
        Initial time step, will scale up as residual decreases
    constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
        Masks the free vector entries as ``False`` (boolean) or NaN (float). In
        the remaining positions the values of ``lhs0`` are returned unchanged
        (boolean) or overruled by the values in `constrain` (float).
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`. If ``target`` is present in ``arguments`` then it is used
        as the initial guess for the iterative procedure.

    Yields
    ------
    :class:`numpy.ndarray` with dtype :class:`float`
        Tuple of coefficient vector and residual norm
    '''

    if isinstance(target, str) and ',' not in target and ':' not in target:
        return pseudotime([target], [residual], [inertia], timestep, constrain={} if constrain is None else {target: constrain},
            arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, **kwargs)[target]
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
    target, residual, inertia = _target_helper(target, residual, inertia)
    linargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    system = System(residual, *_split_trial_test(target))
    method = Pseudotime(inertia=inertia, timestep=timestep, **linargs)
    return _with_solve(system, method, arguments, constrain or {})


def thetamethod(target, residual, inertia, timestep: float, theta: float, *, lhs0: types.arraydata = None, constrain = None, newtontol: float = 1e-10, arguments = {}, newtonargs: types.frozendict = {}, timetarget: str = '_thetamethod_time', time0: float = 0., historysuffix: str = '0'):
    '''solve time dependent problem using the theta method

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    residual : :class:`nutils.evaluable.AsEvaluableArray`
    inertia : :class:`nutils.evaluable.AsEvaluableArray`
    timestep : :class:`float`
        The time step.
    theta : :class:`float`
        Theta value (theta=1 for implicit Euler, theta=0.5 for Crank-Nicolson)
    residual0 : :class:`nutils.evaluable.AsEvaluableArray`
        Optional additional residual component evaluated in previous timestep
    constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
        Masks the free vector entries as ``False`` (boolean) or NaN (float). In
        the remaining positions the values of ``lhs0`` are returned unchanged
        (boolean) or overruled by the values in `constrain` (float).
    newtontol : :class:`float`
        Residual tolerance of individual timesteps
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`. If ``target`` is present in ``arguments`` then it is used
        as the initial condition.
    timetarget : :class:`str`
        Name of the :class:`nutils.function.Argument` that represents time.
        Optional.
    time0 : :class:`float`
        The intial time.  Default: ``0.0``.

    Yields
    ------
    :class:`numpy.ndarray`
        Coefficient vector for all timesteps after the initial condition.
    '''
    if isinstance(target, str) and ',' not in target and ':' not in target:
        return (res[target] for res in thetamethod([target], [residual], [inertia], timestep, theta,
            constrain={} if constrain is None else {target: constrain}, newtontol=newtontol,
            arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, newtonargs=newtonargs,
            timetarget=timetarget, time0=time0, historysuffix=historysuffix))
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial condition via arguments instead')
    target, residual, inertia = _target_helper(target, residual, inertia)

    argobjs = _dict((arg.name, arg) for func in residual + inertia if func is not None for arg in func.arguments if isinstance(arg, evaluable.Argument))
    argobjs.setdefault(timetarget, evaluable.Argument(timetarget, ()))

    old_new = [(t+historysuffix, t) for t in target]
    old_new.append((timetarget+historysuffix, timetarget))

    subs0 = {new: evaluable.Argument(old, argobjs[new].shape) for old, new in old_new}
    dt = evaluable.Argument(timetarget, ()) - subs0[timetarget]
    theta = evaluable.constant(float(theta))
    residuals = tuple(res * theta + evaluable.replace_arguments(res, subs0) * (1.-theta)
        + (inert - evaluable.replace_arguments(inert, subs0)) / dt for res, inert in zip(residual, inertia))

    arguments = arguments.copy()
    arguments.setdefault(timetarget, time0)
    system = System(residuals, *_split_trial_test(target))
    newtonargs = dict(newtonargs)
    linesearch = newtonargs.pop('linesearch', NormBased())
    method = None if system.is_linear else Newton() if linesearch is None else LinesearchNewton(strategy=linesearch, **newtonargs)
    return _thetamethod(system, arguments,
        timestep=timestep, timearg=timetarget, suffix=historysuffix, constrain=constrain or {}, tol=newtontol, method=method)


def _thetamethod(system, arguments, **stepargs):
    with log.iter.plain('timestep', itertools.count()) as steps:
        yield arguments
        for _ in steps:
            arguments = system.step(arguments=arguments, **stepargs)
            yield arguments


impliciteuler = functools.partial(thetamethod, theta=1)
cranknicolson = functools.partial(thetamethod, theta=0.5)


def optimize(target, functional: evaluable.asarray, *, tol: float = 0., arguments = {}, droptol: float = None, constrain = None, lhs0: types.arraydata = None, relax0: float = 1., linesearch=NormBased(), failrelax: float = 1e-6, **kwargs):
    '''find the minimizer of a given functional

    Parameters
    ----------
    target : :class:`str`
        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
    functional : scalar :class:`nutils.evaluable.AsEvaluableArray`
        The functional the should be minimized by varying target
    tol : :class:`float`
        Target residual norm.
    arguments : :class:`collections.abc.Mapping`
        Defines the values for :class:`nutils.function.Argument` objects in
        `residual`. If ``target`` is present in ``arguments`` then it is used
        as the initial guess.
    droptol : :class:`float`
        Threshold for leaving entries in the return value at NaN if they do not
        contribute to the value of the functional.
    constrain : :class:`numpy.ndarray` with dtype :class:`float`
        Defines the fixed entries of the coefficient vector
    relax0 : :class:`float`
        Initial relaxation value.
    linesearch : Callable[[float, float, float, float], Tuple[float, bool]]
        Callable that defines relaxation logic. The callable takes four
        arguments: the current residual and directional derivative, and the
        candidate residual and directional derivative, with derivatives
        normalized to unit length; and returns the optimal scaling and a
        boolean flag that marks whether the candidate should be accepted.
    failrelax : :class:`float`
        Fail with exception if relaxation reaches this lower limit.

    Yields
    ------
    :class:`numpy.ndarray`
        Coefficient vector corresponding to the functional optimum
    '''

    if isinstance(target, str) and ',' not in target:
        return optimize([target], functional, tol=tol, arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, droptol=droptol,
            constrain={} if constrain is None else {target: constrain}, relax0=relax0, linesearch=linesearch, failrelax=failrelax, **kwargs)[target]
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
    linargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    system = System(functional, *_split_trial_test(target))
    if droptol is not None:
        return system.solve_constraints(arguments=arguments, constrain=constrain or {}, linargs=linargs, droptol=droptol)
    method = Direct(**linargs) if system.is_linear \
        else Newton(**linargs) if linesearch is None \
        else LinesearchNewton(strategy=linesearch, relax0=relax0, failrelax=failrelax, **linargs)
    return system.solve(arguments=arguments, constrain=constrain or {}, method=method, tol=tol)


# HELPER FUNCTIONS


def _strip(kwargs, prefix):
    return {key[len(prefix):]: kwargs.pop(key) for key in list(kwargs) if key.startswith(prefix)}


def _dict(items):
    '''construct dictionary from items while checking that duplicate keys have matching values'''

    d = {}
    for k, v in items:
        v_ = d.setdefault(k, v)
        if v_ is not v and v_ != v: # cheap test first to avoid equality checks
            raise ValueError('incompatible items')
    return d


def _copy_with_defaults(d, **kwargs):
    kwargs.update(d)
    return kwargs


def _parse_lhs_cons(constrain, targets, argobjs, arguments):
    arguments = {t: numpy.asarray(a) for t, a in arguments.items()}
    constrain = {t: numpy.asarray(c) for t, c in constrain.items()}
    for target in targets:
        if target not in argobjs:
            raise SolverError('target does not occur in functional: {!r}'.format(target))
        shape = tuple(map(int, argobjs[target].shape))
        if target not in arguments:
            arguments[target] = numpy.zeros(shape)
        elif arguments[target].shape != shape:
            raise SolverError('invalid argument shape for {}: {} != {}'.format(target, arguments[target].shape, shape))
        if target not in constrain:
            constrain[target] = numpy.zeros(shape, dtype=bool)
        elif constrain[target].shape != shape:
            raise SolverError('invalid constrain shape for {}: {} != {}'.format(target, constrain[target].shape, shape))
        if constrain[target].dtype != bool:
            isnan = numpy.isnan(constrain[target])
            arguments[target] = numpy.choose(isnan, [constrain[target], arguments[target]])
            constrain[target] = ~isnan
    return arguments, constrain


def _split_trial_test(target):
    if isinstance(target, str):
        target = target.rstrip(',')
        target = target.split(',') if target else []
    if not target:
        raise ValueError('no targets specified')
    target = [item.split(':') if isinstance(item, str) else item for item in target]
    n = len(target[0])
    if not all(len(t) == n for t in target):
        raise ValueError('inconsistent targets')
    if n == 1:
        trial, = zip(*target)
        test = None
    elif n == 2:
        trial, test = zip(*target)
    else:
        raise ValueError('invalid targets')
    return trial, test


def _target_helper(target, *args):
    trial, test = _split_trial_test(target)
    if test is not None:
        arguments = function.arguments_for(*args)
        args = [[arg.derivative(arguments[t]) for t in test] for arg in args]
    elif len(args) > 1:
        shapes = [{f.shape for f in ziparg if f is not None} for ziparg in zip(*args)]
        if any(len(arg) != len(shapes) for arg in args) or any(len(shape) != 1 for shape in shapes):
            raise ValueError('inconsistent residuals')
        args = [[function.zeros(shape) if f is None else f for f, (shape,) in zip(arg, shapes)] for arg in args]
    return trial, *[tuple(f.as_evaluable_array for f in arg) for arg in args]


class _First:
    def __call__(self, value):
        try:
            return self.value
        except AttributeError:
            self.value = value
            return value


@dataclass(frozen=True)
class _with_solve:
    '''add a .solve method to iterables'''

    system: System
    method: Any
    arguments: Any
    constrain: Any
    item: Optional[str] = None

    def __iter__(self):
        iters = self.method(self.system, arguments=self.arguments, constrain=self.constrain)
        for arguments, resnorm in iters:
            yield arguments if self.item is None else arguments[self.item], types.attributes(resnorm=resnorm)

    def __getitem__(self, item):
        assert self.item is None
        return _with_solve(self.system, self.method, self.arguments, self.constrain, item)

    def solve(self, tol, maxiter=float('inf'), miniter=0):
        '''execute nonlinear solver, return lhs

        Iterates over nonlinear solver until tolerance is reached. Example::

            lhs = newton(target, residual).solve(tol=1e-5)

        Parameters
        ----------
        tol : :class:`float`
            Target residual norm
        maxiter : :class:`int`
            Maximum number of iterations
        miniter : :class:`int`
            Minimum number of iterations

        Returns
        -------
        :class:`numpy.ndarray`
            Coefficient vector that corresponds to a smaller than ``tol`` residual.
        '''

        lhs, info = self.solve_withinfo(tol=tol, maxiter=maxiter, miniter=miniter)
        return lhs

    @cache.function
    def solve_withinfo(self, tol, maxiter=float('inf'), miniter=0):
        '''execute nonlinear solver, return lhs and info

        Like :func:`solve`, but return a 2-tuple of the solution and the
        corresponding info object which holds information about the final residual
        norm and other generator-dependent information.
        '''

        if miniter > maxiter:
            raise ValueError('The minimum number of iterations cannot be larger than the maximum.')
        with log.context(str(self.method)):
            with log.context('iter {}', 0) as recontext:
                it = enumerate(self)
                iiter, (lhs, info) = next(it)
                resnorm0 = info.resnorm
                while info.resnorm > tol or iiter < miniter:
                    if iiter >= maxiter:
                        raise SolverError(f'failed to reach target tolerance in {maxiter} iterations')
                    recontext(f'{iiter+1} ({100 * numpy.log(resnorm0 / max(info.resnorm, tol)) / numpy.log(resnorm0 / tol):.0f}%)')
                    iiter, (lhs, info) = next(it)
            log.info(f'converged in {iiter} iterations to residual {info.resnorm:.1e}')
        info.niter = iiter
        return lhs, info


# vim:sw=4:sts=4:et
