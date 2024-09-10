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
solve > solving 25 dof system to machine precision using arnoldi solver
solve > solver returned with residual ...

The coefficients ``lhs`` represent the solution to the Poisson problem.

In addition to ``solve_linear`` the solver module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, evaluable, cache, numeric, types, _util as util, matrix, warnings, sparse
from dataclasses import dataclass
from typing import Optional
import abc
import numpy
import itertools
import functools
import collections
import math
import treelog as log


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

class BaseSystem:

    def __init__(self, trials, block_vector, block_matrix):
        self.trials = tuple(trials)
        self.dtype = block_vector[0].dtype
        if not all(v.dtype == self.dtype for v in block_vector) or not all(m.dtype == self.dtype for row in block_matrix for m in row):
            raise ValueError('inconsistent data types in block matrix, vector')
        _shapes = _dict((arg.name, arg.shape) for vec in block_vector for arg in vec.arguments if isinstance(arg, evaluable.Argument))
        self.argshapes = dict(zip(trials, evaluable.compile(tuple(_shapes[t] for t in trials))()))
        self.argsizes = {t: numpy.prod(shape, dtype=int) for t, shape in self.argshapes.items()}
        self.eval = evaluable.compile((tuple(block_vector), tuple(tuple(map(evaluable.as_csr, row)) for row in block_matrix)))
        self.is_constant = not any(col.arguments for row in block_matrix for col in row)
        self.matrix_cache = None

    @functools.cached_property
    def __nutils_hash__(self):
        from linecache import cache # we abuse the line cache to hash self.eval
        return types.nutils_hash((self.trials, ''.join(cache[self.eval.__code__.co_filename][2])))

    def assemble(self, arguments):
        vec_blocks, mat_blocks = self.eval(**arguments)
        vec = numpy.concatenate(vec_blocks)
        if self.matrix_cache is not None:
            mat = self.matrix_cache
        else:
            mat = matrix.assemble_block_csr(mat_blocks)
            if self.is_constant:
                self.matrix_cache = mat
        return vec, mat

    def prepare_solution_vector(self, arguments: dict, constrain: dict):
        arguments = arguments.copy()
        trial_offsets = numpy.cumsum([0] + [self.argsizes[trial] for trial in self.trials])
        x = numpy.empty(trial_offsets[-1], self.dtype)
        iscons = numpy.empty(trial_offsets[-1], dtype=bool)
        for trial, i, j in zip(self.trials, trial_offsets, trial_offsets[1:]):
            trialshape = self.argshapes[trial]
            trialarg = x[i:j].reshape(trialshape)
            trialarg[...] = arguments.get(trial, 0)
            c = constrain.get(trial, False)
            if c is not False:
                assert c.shape == trialshape
                if c.dtype != bool:
                    c, v = ~numpy.isnan(c), c
                    trialarg[c] = v[c]
            trialcons = iscons[i:j].reshape(trialshape)
            trialcons[...] = c
            arguments[trial] = trialarg # IMPORTANT: arguments share memory with x
        return x, iscons, arguments


class System(BaseSystem):

    def __init__(self, trials: tuple, residuals: tuple):
        if isinstance(trials, str):
            trials = trials.rstrip(',').split(',')
        if isinstance(residuals, (tuple, list)):
            residuals = [res.as_evaluable_array for res in residuals]
            argobjects = _dict((arg.name, arg) for res in residuals for arg in res.arguments if isinstance(arg, evaluable.Argument))
            is_symmetric = False
        else:
            functional = residuals.as_evaluable_array
            argobjects = {arg.name: arg for arg in functional.arguments if isinstance(arg, evaluable.Argument)}
            trials, tests = zip(*[s if isinstance(s, (tuple, list)) else s.split(':') if ':' in s else (s, s) for s in trials])
            residuals = [evaluable.derivative(functional, argobjects[t]) for t in tests]
            is_symmetric = trials == tests

        residuals = [evaluable._flat(res) for res in residuals]
        jacobians = [[evaluable._flat(evaluable.derivative(res, argobjects[t]).simplified, 2) for t in trials] for res in residuals]

        self.is_linear = not any(arg.name in trials for row in jacobians for col in row for arg in col.arguments)
        self.is_symmetric = is_symmetric

        super().__init__(trials, residuals, jacobians)

    def solve_linear(self, *, arguments: dict = {}, constrain = None, linargs = None):
        if not self.is_linear:
            raise SolverError('problem is not linear')
        x, iscons, arguments = self.prepare_solution_vector(arguments, constrain or {})
        res, jac = self.assemble(arguments)
        x -= jac.solve(res, constrain=iscons, **_copy_with_defaults(linargs, symmetric=self.is_symmetric))
        return arguments, numpy.linalg.norm((res - jac @ x)[~iscons])

    def iter_newton(self, *, arguments: dict = {}, constrain = None, linargs = None, linesearch = None, failrelax: float = 1e-6, relax: float = 1.):
        x, iscons, arguments = self.prepare_solution_vector(arguments, constrain or {})
        linargs = _copy_with_defaults(linargs, rtol=1-3, symmetric=self.is_symmetric)
        res, jac = self.assemble(arguments)
        while True: # newton iterations
            yield arguments, numpy.linalg.norm(res[~iscons])
            dx = -jac.solve_leniently(res, constrain=iscons, **linargs)
            x += dx * relax
            if linesearch is None:
                res, jac = self.assemble(arguments)
            else:
                res0 = res
                jac0dx = jac@dx # == res0 if dx was solved to infinite precision
                while True: # line search
                    res, jac = self.assemble(arguments)
                    relax, adjust = _linesearch_helper(linesearch, res0, jac0dx, res, jac@dx, iscons, relax)
                    if not adjust:
                        break
                    if relax <= failrelax:
                        raise SolverError('stuck in local minimum')
                    x += dx * adjust

    def iter_minimize(self, *, arguments: dict = {}, constrain = None, linargs = None, rampup: float = .5, rampdown: float = -1., failrelax: float = -10.):
        if not self.is_symmetric:
            raise SolverError('problem is not symmetric')
        x, iscons, arguments = self.prepare_solution_vector(arguments, constrain or {})
        res, jac = self.assemble(arguments)
        linargs = _copy_with_defaults(linargs, rtol=1-3, symmetric=self.is_symmetric)
        relax = 0.
        while True:
            yield arguments, numpy.linalg.norm(res[~iscons])
            dx = -jac.solve_leniently(res, constrain=iscons, **linargs) # baseline: vanilla Newton
            # compute first two ritz values to determine approximate path of steepest descent
            zres = res * ~iscons
            dxnorm = numpy.linalg.norm(dx)
            k0 = dx / dxnorm
            k1 = -zres / dxnorm # == jac @ k0
            a = k1 @ k0
            k1 -= k0 * a # orthogonalize
            c = numpy.linalg.norm(k1)
            k1 /= c # normalize
            b = k1 @ (jac @ k1)
            # at this point k0 and k1 are orthonormal, and [k0 k1]^T jac [k0 k1] = [a c; c b]
            D = numpy.hypot(b-a, 2*c)
            L = numpy.array([a+b-D, a+b+D]) / 2 # 2nd order ritz values: eigenvalues of [a c; c b]
            v0, v1 = zres + dx * L[:, numpy.newaxis]
            V = numpy.stack([v1, -v0], axis=1) / D # ritz vectors times dx -- note: V @ L == -zres, V.sum() == dx
            log.info('spectrum: {:.1e}..{:.1e} ({}definite)'.format(*L, 'positive ' if L[0] > 0 else 'negative ' if L[-1] < 0 else 'in'))
            while True: # line search along steepest descent curve
                r = numpy.exp(relax - numpy.log(D)) # == exp(relax) / D
                eL = numpy.exp(-r*L)
                dx -= V @ eL
                x += dx
                res, jac = self.assemble(arguments)
                slope = res @ (V @ (eL*L))
                log.info(f'slope is {slope:+.1e} at relaxation {r:.1e}')
                if slope < 0:
                    relax += rampup
                    break
                relax += rampdown
                if relax <= failrelax:
                    raise SolverError('stuck in local minimum')
                dx = V @ eL # return to baseline

    def iter_pseudotime(self, *, inertia, timestep: float, arguments: dict = {}, constrain = None, linargs = None):
        x, iscons, arguments = self.prepare_solution_vector(arguments, constrain or {})
        linargs = _copy_with_defaults(linargs, rtol=1-3) #, symmetric=self.is_symmetric)

        argobjs = {t: evaluable.Argument(t, tuple(map(evaluable.constant, self.argshapes[t])), float) for t in self.trials}
        djacobians = [[evaluable._flat(evaluable.derivative(evaluable._flat(res), argobjs[t]).simplified, 2) for t in self.trials] for res in inertia]
        djac_blocks = evaluable.compile(tuple(tuple(map(evaluable.as_csr, row)) for row in djacobians))
        djac = matrix.assemble_block_csr(djac_blocks(**arguments))

        timestep0 = timestep
        first = _First()
        while True:
            res, jac = self.assemble(arguments)
            resnorm = numpy.linalg.norm(res[~iscons])
            yield arguments, resnorm
            x -= (jac + djac / timestep).solve_leniently(res, constrain=iscons, **linargs)
            timestep = timestep0 * (first(resnorm) / resnorm)
            log.info(f'timestep: {timestep:.0e}')

    @cache.function
    def solve_withnorm(self, *, arguments: dict = {}, constrain = None, linargs = None, tol: float = 0., miniter: int = 0, maxiter: Optional[int] = None, method = 'newton', **methodargs):
        if self.is_linear:
            return self.solve_linear(arguments=arguments, constrain=constrain, linargs=linargs)
        if tol <= 0:
            raise SolverError('nonlinear problem requires a strictly positive tolerance')
        iter_method = getattr(self, 'iter_' + method)
        first = _First()
        with log.iter.plain(method, itertools.count()) as steps:
            for arguments, resnorm in iter_method(arguments=arguments, constrain=constrain, linargs=linargs, **methodargs):
                progress = numpy.log(first(resnorm)/resnorm) / numpy.log(first(resnorm)/tol) if resnorm > tol else 1
                log.info(f'residual: {resnorm:.0e} ({100*progress:.0f}%)')
                iiter = next(steps) # opens new log context
                if iiter >= miniter and resnorm <= tol:
                    return arguments, resnorm
                if maxiter is not None and iiter >= maxiter:
                    raise SolverError(f'failed to converge in {maxiter} iterations')

    def solve(self, **kwargs):
        arguments, resnorm = self.solve_withnorm(**kwargs)
        return arguments

    def step(self, *, timestep, timetarget, historysuffix, arguments, **solveargs):
        arguments = arguments.copy()
        for trial in self.trials:
            if trial in arguments:
                arguments[trial + historysuffix] = arguments[trial]
        time = arguments.get(timetarget, 0.)
        arguments[timetarget + historysuffix] = time
        arguments[timetarget] = time + timestep
        try:
            return self.solve(arguments=arguments, **solveargs)
        except (SolverError, matrix.MatrixError) as e:
            log.error(f'error: {e}; retrying with timestep {timestep/2}')
            with log.context('tic'):
                halfway_arguments = self.step(timestep=timestep/2, timetarget=timetarget, historysuffix=historysuffix, arguments=arguments, **solveargs)
            with log.context('toc'):
                return self.step(timestep=timestep/2, timetarget=timetarget, historysuffix=historysuffix, arguments=halfway_arguments, **solveargs)


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
    return System(target, residual).solve(arguments=arguments, constrain=constrain, linargs=linargs)


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
    return _with_solve('newton', System(target, residual),
        arguments=arguments, constrain=constrain, linesearch=linesearch, relax=relax0, failrelax=failrelax, linargs=linargs)


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
    return _with_solve('minimize', System(target, energy),
        arguments=arguments, constrain=constrain, rampup=rampup, rampdown=rampdown, failrelax=failrelax, linargs=linargs)

    #iterations = system.iter_minimize(arguments=arguments, constrain=constrain, rampup=rampup, rampdown=rampdown, failrelax=failrelax, linargs=linargs)
    #eval_energy = evaluable.compile(energy.as_evaluable_array)
    #return _with_solve(((arguments, types.attributes(resnorm=resnorm, energy=eval_energy(**arguments))) for arguments, resnorm in iterations), name='minimize')


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
    linargs = _strip(kwargs, 'lin', rtol=1e-3)
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    return _with_solve('pseudotime', System(target, residual),
        arguments=arguments, constrain=constrain, inertia=inertia, timestep=timestep)


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
    return _thetamethod(System(target, residuals), arguments,
        timestep=timestep, timetarget=timetarget, historysuffix=historysuffix,
        constrain=constrain, tol=newtontol, **_copy_with_defaults(newtonargs,
        linesearch=NormBased()))


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
    with log.context('optimize'):
        system = System(target, functional)
        eval_functional = evaluable.compile(functional.as_evaluable_array)
        if droptol is None:
            arguments = system.solve(arguments=arguments, constrain=constrain, tol=tol, relax=relax0, failrelax=failrelax, linesearch=linesearch)
            value = eval_functional(**arguments)
        elif system.is_linear:
            x, iscons, arguments = system.prepare_solution_vector(arguments, constrain or {})
            res, jac = system.assemble(arguments)
            nosupp = ~jac.rowsupp(droptol)
            x -= jac.solve(res, constrain=iscons|nosupp, **dict(symmetric=True, **linargs))
            value = eval_functional(**arguments)
            x[nosupp & ~iscons] = numpy.nan
        else:
            raise ValueError('drop tolerance is only accepted for linear systems')
        log.info(f'optimum value: {value:.1e}')
        return arguments


# HELPER FUNCTIONS

def _strip(kwargs, prefix, **return_args):
    for key in list(kwargs):
        if key.startswith(prefix):
            return_args[key[len(prefix):]] = kwargs.pop(key)
    return return_args


def _dict(items):
    '''construct dictionary from items while checking that duplicate keys have matching values'''

    d = {}
    for k, v in items:
        v_ = d.setdefault(k, v)
        if v_ is not v and v_ != v: # cheap test first to avoid equality checks
            raise ValueError('incompatible items')
    return d


def _copy_with_defaults(d, **kwargs):
    d = d.copy() if d else {}
    for k, v in kwargs.items():
        d.setdefault(k, v)
    return d


def _target_helper(target, *args):
    targets = target.rstrip(',').split(',') if isinstance(target, str) else list(target)
    is_functional = [':' in target for target in targets]
    if all(is_functional):
        targets, tests = zip(*[t.split(':', 1) for t in targets])
        arguments = function._join_arguments(arg.arguments for arg in args)
        testargs = [function.Argument(t, *arguments[t]) for t in tests]
        args = [map(arg.derivative, testargs) for arg in args]
    elif any(is_functional):
        raise ValueError('inconsistent targets')
    elif len(args) > 1:
        shapes = [{f.shape for f in ziparg if f is not None} for ziparg in zip(*args)]
        if any(len(arg) != len(shapes) for arg in args) or any(len(shape) != 1 for shape in shapes):
            raise ValueError('inconsistent residuals')
        args = [[function.zeros(shape) if f is None else f for f, (shape,) in zip(arg, shapes)] for arg in args]
    return (tuple(targets), *[tuple(f.as_evaluable_array for f in arg) for arg in args])


def _linesearch_helper(linesearch, res0, dres0, res1, dres1, iscons, relax):
    isdof = ~iscons
    scale, accept = linesearch(res0[isdof], dres0[isdof] * relax, res1[isdof], dres1[isdof] * relax)
    if accept:
        log.info('update accepted at relaxation', round(relax, 5))
        return min(relax * scale, 1), 0
    assert scale < 1
    return relax * scale, relax * (scale-1)


class _First:
    def __call__(self, value):
        try:
            return self.value
        except AttributeError:
            self.value = value
            return value


class _with_solve:

    def __init__(self, method, system, item=None, **kwargs):
        self.system = system
        self.method = method
        self.kwargs = kwargs
        self._item = item

    def __getitem__(self, item):
        assert self._item is None
        return _with_solve(self.method, self.system, item, **self.kwargs)

    def _map(self, result):
        arguments, resnorm = result
        return arguments if self._item is None else arguments[self._item], types.attributes(resnorm=resnorm)

    def __iter__(self):
        iter_method = getattr(self.system, 'iter_' + self.method)
        return map(self._map, iter_method(**self.kwargs))

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

    def solve_withinfo(self, tol, maxiter=float('inf'), miniter=0):
        '''execute nonlinear solver, return lhs and info

        Like :func:`solve`, but return a 2-tuple of the solution and the
        corresponding info object which holds information about the final residual
        norm and other generator-dependent information.
        '''

        if maxiter == float('inf'):
            maxiter = None
        return self._map(self.system.solve_withnorm(tol=tol, miniter=miniter, maxiter=maxiter, method=self.method, **self.kwargs))


# vim:sw=4:sts=4:et
