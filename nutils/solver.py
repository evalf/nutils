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

    @classmethod
    def legacy(cls, kwargs):
        args = {}
        if 'searchrange' in kwargs:
            args['minscale'], args['acceptscale'] = kwargs.pop('searchrange')
        if 'rebound' in kwargs:
            args['maxscale'] = kwargs.pop('rebound')
        if args:
            warnings.deprecation('the searchrange and rebound arguments are deprecated; use linesearch=solver.NormBased(minscale=searchrange[0], acceptscale=searchrange[1], maxscale=rebound) instead')
        return cls(**args)

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
    target, residual = _target_helper(target, residual)
    solveargs = _strip(kwargs, 'lin')
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    return _solve_linear(target, residual,
        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
        types.frozendict(solveargs))


@cache.function
def _solve_linear(target, residual: tuple, constraints: dict, arguments: dict, solveargs: dict):
    arguments, constraints = _parse_lhs_cons(constraints, target, _argobjs(residual), arguments)
    jacobians = _derivative(residual, target)
    if not set(target).isdisjoint(_argobjs(jacobians)):
        raise SolverError('problem is not linear')
    dtype = _determine_dtype(target, residual, arguments, constraints)
    lhs, vlhs = _redict(arguments, target, dtype)
    mask, vmask = _invert(constraints, target)
    res, jac = _integrate_blocks(residual, jacobians, arguments=lhs, mask=mask)
    vlhs[vmask] -= jac.solve(res, **solveargs)
    return lhs


def newton(target, residual, *, jacobian = None, lhs0 = None, relax0: float = 1., constrain = None, linesearch='__legacy__', failrelax: float = 1e-6, arguments = {}, **kwargs):
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
    target, residual = _target_helper(target, residual)
    if linesearch == '__legacy__':
        linesearch = NormBased.legacy(kwargs)
    solveargs = _strip(kwargs, 'lin')
    solveargs.setdefault('rtol', 1e-3)
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    return _with_solve(_newton(target, residual, None if jacobian is None else tuple(jacobian),
        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
        linesearch, relax0, failrelax, types.frozendict(solveargs)))


class _newton(cache.Recursion, length=1):

    def __init__(self, target, residual, jacobian, constrain, arguments, linesearch, relax0: float, failrelax: float, solveargs):
        super().__init__()
        self.target = target
        self.residual = residual
        self.jacobian = _derivative(residual, target, jacobian)
        self.lhs0, self.constrain = _parse_lhs_cons(constrain, target, _argobjs(residual), arguments)
        self.dtype = _determine_dtype(target, residual, self.lhs0, self.constrain)
        self.relax0 = relax0
        self.linesearch = linesearch
        self.failrelax = failrelax
        self.solveargs = solveargs

    def _eval(self, lhs, mask):
        return _integrate_blocks(self.residual, self.jacobian, arguments=lhs, mask=mask)

    def resume(self, history):
        mask, vmask = _invert(self.constrain, self.target)
        if history:
            lhs, info = history[-1]
            lhs, vlhs = _redict(lhs, self.target, self.dtype)
            res, jac = self._eval(lhs, mask)
            assert numpy.linalg.norm(res) == info.resnorm
            relax = info.relax
        else:
            lhs, vlhs = _redict(self.lhs0, self.target, self.dtype)
            res, jac = self._eval(lhs, mask)
            relax = self.relax0
            yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), relax=relax)
        while True:
            dlhs = -jac.solve_leniently(res, **self.solveargs)  # compute new search vector
            res0 = res
            dres = jac@dlhs  # == -res if dlhs was solved to infinite precision
            vlhs[vmask] += relax * dlhs
            res, jac = self._eval(lhs, mask)
            if self.linesearch:
                scale, accept = self.linesearch(res0, relax*dres, res, relax*(jac@dlhs))
                while not accept:  # line search
                    assert scale < 1
                    oldrelax = relax
                    relax *= scale
                    if relax <= self.failrelax:
                        raise SolverError('stuck in local minimum')
                    vlhs[vmask] += (relax - oldrelax) * dlhs
                    res, jac = self._eval(lhs, mask)
                    scale, accept = self.linesearch(res0, relax*dres, res, relax*(jac@dlhs))
                log.info('update accepted at relaxation', round(relax, 5))
                relax = min(relax * scale, 1)
            yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), relax=relax)


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
    if isinstance(target, str):
        target = target.rstrip(',').split(',')
    solveargs = _strip(kwargs, 'lin')
    solveargs['symmetric'] = True
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    return _with_solve(_minimize(tuple(target), energy.as_evaluable_array,
        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
        rampup, rampdown, failrelax, types.frozendict(solveargs)))


class _minimize(cache.Recursion, length=1, version=3):

    def __init__(self, target, energy: evaluable.asarray, constrain, arguments, rampup: float, rampdown: float, failrelax: float, solveargs):
        super().__init__()
        if energy.shape != ():
            raise ValueError('`energy` should be scalar')
        self.target = target
        self.energy = energy
        self.residual = _derivative((energy,), target)
        self.jacobian = _derivative(self.residual, target)
        self.lhs0, self.constrain = _parse_lhs_cons(constrain, target, _argobjs((energy,)), arguments)
        self.dtype = _determine_dtype(target, (energy,), self.lhs0, self.constrain)
        self.rampup = rampup
        self.rampdown = rampdown
        self.failrelax = failrelax
        self.solveargs = solveargs

    def _eval(self, lhs, mask):
        return _integrate_blocks(self.energy, self.residual, self.jacobian, arguments=lhs, mask=mask)

    def resume(self, history):
        mask, vmask = _invert(self.constrain, self.target)
        if history:
            lhs, info = history[-1]
            lhs, vlhs = _redict(lhs, self.target, self.dtype)
            nrg, res, jac = self._eval(lhs, mask)
            assert nrg == info.energy
            assert numpy.linalg.norm(res) == info.resnorm
            relax = info.relax
        else:
            lhs, vlhs = _redict(self.lhs0, self.target, self.dtype)
            nrg, res, jac = self._eval(lhs, mask)
            relax = 0
            yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), energy=nrg, relax=relax)

        while True:
            nrg0 = nrg
            dlhs = -jac.solve_leniently(res, **self.solveargs)
            vlhs[vmask] += dlhs  # baseline: vanilla Newton

            # compute first two ritz values to determine approximate path of steepest descent
            dlhsnorm = numpy.linalg.norm(dlhs)
            k0 = dlhs / dlhsnorm
            k1 = -res / dlhsnorm  # = jac @ k0
            a = k1 @ k0
            k1 -= k0 * a  # orthogonalize
            c = numpy.linalg.norm(k1)
            k1 /= c  # normalize
            b = k1 @ (jac @ k1)
            # at this point k0 and k1 are orthonormal, and [k0 k1]^T jac [k0 k1] = [a c; c b]
            D = numpy.hypot(b-a, 2*c)
            L = numpy.array([a+b-D, a+b+D]) / 2  # 2nd order ritz values: eigenvalues of [a c; c b]
            v0, v1 = res + dlhs * L[:, numpy.newaxis]
            V = numpy.array([v1, -v0]).T / D  # ritz vectors times dlhs -- note: V.dot(L) = -res, V.sum() = dlhs
            log.info('spectrum: {:.1e}..{:.1e} ({}definite)'.format(*L, 'positive ' if L[0] > 0 else 'negative ' if L[-1] < 0 else 'in'))

            eL = 0
            for irelax in itertools.count():  # line search along steepest descent curve
                r = numpy.exp(relax - numpy.log(D))  # = exp(relax) / D
                eL0 = eL
                eL = numpy.exp(-r*L)
                vlhs[vmask] -= V.dot(eL - eL0)
                nrg, res, jac = self._eval(lhs, mask)
                slope = res.dot(V.dot(eL*L))
                log.info('energy {:+.2e} / e{:+.1f} and {}creasing'.format(nrg - nrg0, relax, 'in' if slope > 0 else 'de'))
                if numpy.isfinite(nrg) and numpy.isfinite(res).all() and nrg <= nrg0 and slope <= 0:
                    relax += self.rampup
                    break
                relax += self.rampdown
                if relax <= self.failrelax:
                    raise SolverError('stuck in local minimum')

            yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), energy=nrg, relax=relax)


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
    solveargs = _strip(kwargs, 'lin')
    solveargs.setdefault('rtol', 1e-3)
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    return _with_solve(_pseudotime(target, residual, inertia, timestep,
        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
        types.frozendict(solveargs)))


class _pseudotime(cache.Recursion, length=1):

    def __init__(self, target, residual, inertia, timestep: float, constrain, arguments, solveargs: dict):
        super().__init__()
        if target in arguments:
            raise ValueError('`target` should not be defined in `arguments`')
        if len(residual) != len(inertia):
            raise Exception('length of residual and inertia do no match')
        for inert, res in zip(inertia, residual):
            if inert and not evaluable.equalshape(inert.shape, res.shape):
                raise ValueError('expected `inertia` with shape {} but got {}'.format(res.shape, inert.shape))
        self.target = target
        self.timesteptarget = '_pseudotime_timestep'
        dt = evaluable.Argument(self.timesteptarget, ())
        self.residuals = residual
        self.jacobians = _derivative(tuple(res + (inert/dt if inert else 0) for res, inert in zip(residual, inertia)), target)
        self.lhs0, self.constrain = _parse_lhs_cons(constrain, target, _argobjs(residual+inertia), arguments)
        self.dtype = _determine_dtype(target, residual+inertia, self.lhs0, self.constrain)
        self.timestep = timestep
        self.solveargs = solveargs

    def _eval(self, lhs, mask, timestep):
        return _integrate_blocks(self.residuals, self.jacobians, arguments=dict({self.timesteptarget: timestep}, **lhs), mask=mask)

    def resume(self, history):
        mask, vmask = _invert(self.constrain, self.target)
        if history:
            lhs, info = history[-1]
            lhs, vlhs = _redict(lhs, self.target, self.dtype)
            resnorm0 = info.resnorm0
            timestep = info.timestep
            res, jac = self._eval(lhs, mask, timestep)
            resnorm = numpy.linalg.norm(res)
            assert resnorm == info.resnorm
        else:
            lhs, vlhs = _redict(self.lhs0, self.target, self.dtype)
            timestep = self.timestep
            res, jac = self._eval(lhs, mask, timestep)
            resnorm = resnorm0 = numpy.linalg.norm(res)
            yield lhs, types.attributes(resnorm=resnorm, timestep=timestep, resnorm0=resnorm0)

        while True:
            vlhs[vmask] -= jac.solve_leniently(res, **self.solveargs)
            timestep = self.timestep * (resnorm0/resnorm)
            log.info('timestep: {:.0e}'.format(timestep))
            res, jac = self._eval(lhs, mask, timestep)
            resnorm = numpy.linalg.norm(res)
            yield lhs, types.attributes(resnorm=resnorm, timestep=timestep, resnorm0=resnorm0)


def thetamethod(target, residual, inertia, timestep: float, theta: float, *, lhs0: types.arraydata = None, target0: str = None, constrain = None, newtontol: float = 1e-10, arguments = {}, newtonargs: types.frozendict = {}, timetarget: str = '_thetamethod_time', time0: float = 0., historysuffix: str = '0'):
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
        return (res[target] for res in thetamethod([target], [residual], [inertia], timestep, theta, target0=target0,
            constrain={} if constrain is None else {target: constrain}, newtontol=newtontol,
            arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, newtonargs=newtonargs,
            timetarget=timetarget, time0=time0, historysuffix=historysuffix))
    if lhs0 is not None:
        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial condition via arguments instead')
    target, residual, inertia = _target_helper(target, residual, inertia)
    return _thetamethod(target, residual, inertia, timestep,
        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
        theta, target0, newtontol, types.frozendict(newtonargs), timetarget, time0, historysuffix)


class _thetamethod(cache.Recursion, length=1, version=1):

    def __init__(self, target, residual, inertia, timestep: float, constrain, arguments, theta: float, target0: str, newtontol: float, newtonargs: types.frozendict, timetarget: str, time0: float, historysuffix: str):
        super().__init__()
        if len(residual) != len(inertia):
            raise Exception('length of residual and inertia do no match')
        for inert, res in zip(inertia, residual):
            if not evaluable.equalshape(inert.shape, res.shape):
                raise ValueError('expected `inertia` with shape {} but got {}'.format(res.shape, inert.shape))
        self.target = target
        self.newtonargs = newtonargs
        self.newtontol = newtontol
        self.timestep = timestep
        self.timetarget = timetarget
        self.lhs0, self.constrain = _parse_lhs_cons(constrain, target, _argobjs(residual+inertia), arguments)
        self.lhs0[timetarget] = numpy.array(time0)
        if target0 is None:
            self.old_new = [(t+historysuffix, t) for t in target]
        elif len(target) == 1:
            warnings.deprecation('target0 is deprecated; use historysuffix instead (target0=target+historysuffix)')
            self.old_new = [(target0, target[0])]
        else:
            raise Exception('target0 is not supported in combination with multiple targets; use historysuffix instead')
        self.old_new.append((timetarget+historysuffix, timetarget))
        subs0 = {new: evaluable.Argument(old, tuple(map(evaluable.constant, self.lhs0[new].shape))) for old, new in self.old_new}
        dt = evaluable.Argument(timetarget, ()) - subs0[timetarget]
        self.residuals = tuple(res * theta + evaluable.replace_arguments(res, subs0) * (1-theta) + (inert - evaluable.replace_arguments(inert, subs0)) / dt for res, inert in zip(residual, inertia))
        self.jacobians = _derivative(self.residuals, target)

    def _step(self, lhs0, dt):
        arguments = lhs0.copy()
        arguments.update((old, lhs0[new]) for old, new in self.old_new)
        arguments[self.timetarget] = lhs0[self.timetarget] + dt
        try:
            return newton(self.target, residual=self.residuals, jacobian=self.jacobians, constrain=self.constrain, arguments=arguments, **self.newtonargs).solve(tol=self.newtontol)
        except (SolverError, matrix.MatrixError) as e:
            log.error('error: {}; retrying with timestep {}'.format(e, dt/2))
            return self._step(self._step(lhs0, dt/2), dt/2)

    def resume(self, history):
        if history:
            lhs, = history
        else:
            lhs = self.lhs0
            yield lhs
        while True:
            lhs = self._step(lhs, self.timestep)
            yield lhs


impliciteuler = functools.partial(thetamethod, theta=1)
cranknicolson = functools.partial(thetamethod, theta=0.5)


def optimize(target, functional: evaluable.asarray, *, tol: float = 0., arguments = {}, droptol: float = None, constrain = None, lhs0: types.arraydata = None, relax0: float = 1., linesearch=None, failrelax: float = 1e-6, **kwargs):
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
    if isinstance(target, str):
        target = target.rstrip(',').split(',')
    if linesearch is None:
        linesearch = NormBased.legacy(kwargs)
    solveargs = _strip(kwargs, 'lin')
    solveargs['symmetric'] = True
    if kwargs:
        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
    with log.context('optimize'):
        return _optimize(tuple(target), functional.as_evaluable_array,
            types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
            types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
            tol, droptol, relax0, linesearch, failrelax, types.frozendict(solveargs))


@cache.function(version=1)
def _optimize(target, functional: evaluable.asarray, constrain, arguments, tol: float, droptol: float, relax0: float, linesearch, failrelax: float, solveargs):
    argobjs = _argobjs((functional,))
    if any(t not in argobjs for t in target):
        if not droptol:
            raise ValueError('target {} does not occur in integrand; consider setting droptol>0'.format(', '.join(t for t in target if t not in argobjs)))
        target = [t for t in target if t in argobjs]
        if not target:
            return {}
    residual = _derivative((functional,), target)
    jacobian = _derivative(residual, target)
    lhs0, constrain = _parse_lhs_cons(constrain, target, argobjs, arguments)
    dtype = _determine_dtype(target, (functional,), lhs0, constrain)
    mask, vmask = _invert(constrain, target)
    lhs, vlhs = _redict(lhs0, target, dtype)
    val, res, jac = _integrate_blocks(functional, residual, jacobian, arguments=lhs, mask=mask)
    if droptol is not None:
        supp = jac.rowsupp(droptol)
        res = res[supp]
        jac = jac.submatrix(supp, supp)
        nan = numpy.zeros_like(vmask)
        nan[vmask] = ~supp  # return value is set to nan if dof is not supported and not constrained
        vmask[vmask] = supp  # dof is computed if it is supported and not constrained
        assert vmask.sum() == len(res)
    resnorm = numpy.linalg.norm(res)
    solveargs = dict(solveargs)
    if not set(target).isdisjoint(_argobjs(jacobian)):
        if tol <= 0:
            raise ValueError('nonlinear optimization problem requires a nonzero "tol" argument')
        solveargs.setdefault('rtol', 1e-3)
        firstresnorm = resnorm
        relax = relax0
        accept = True
        with log.context('newton {:.0f}%', 0) as reformat:
            while not numpy.isfinite(resnorm) or resnorm > tol:
                if accept:
                    reformat(100 * numpy.log(firstresnorm/resnorm) / numpy.log(firstresnorm/tol))
                    dlhs = -jac.solve_leniently(res, **solveargs)
                    res0 = res
                    dres = jac@dlhs  # == -res0 if dlhs was solved to infinite precision
                    relax0 = 0
                vlhs[vmask] += (relax - relax0) * dlhs
                relax0 = relax  # currently applied relaxation
                val, res, jac = _integrate_blocks(functional, residual, jacobian, arguments=lhs, mask=mask)
                resnorm = numpy.linalg.norm(res)
                scale, accept = linesearch(res0, relax*dres, res, relax*(jac@dlhs))
                relax = min(relax * scale, 1)
                if relax <= failrelax:
                    raise SolverError('stuck in local minimum')
            log.info('converged with residual {:.1e}'.format(resnorm))
    elif resnorm > tol:
        solveargs.setdefault('atol', tol)
        dlhs = -jac.solve(res, **solveargs)
        vlhs[vmask] += dlhs
        val += (res + jac@dlhs/2).dot(dlhs)
    if droptol is not None:
        vlhs[nan] = numpy.nan
        log.info('constrained {}/{} dofs'.format(len(vlhs)-nan.sum(), len(vlhs)))
    log.info('optimum value {:.2e}'.format(val))
    return lhs


# HELPER FUNCTIONS

def _strip(kwargs, prefix):
    return {key[len(prefix):]: kwargs.pop(key) for key in list(kwargs) if key.startswith(prefix)}


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


def _derivative(residual, target, jacobian=None):
    argobjs = _argobjs(residual)
    if jacobian is None:
        jacobian = tuple(evaluable.derivative(res, argobjs[t]).simplified for res in residual for t in target)
    elif len(jacobian) != len(residual) * len(target):
        raise ValueError('jacobian has incorrect length')
    elif not all(evaluable.equalshape(jacobian[i*len(target)+j].shape, res.shape + argobjs[t].shape) for i, res in enumerate(residual) for j, t in enumerate(target)):
        raise ValueError('jacobian has incorrect shape')
    return jacobian


def _redict(lhs, targets, dtype=float):
    '''copy argument dictionary referencing a newly allocated contiguous array'''

    vlhs = numpy.empty(sum(lhs[target].size for target in targets), dtype)
    lhs = lhs.copy()
    offset = 0
    for target in targets:
        old = lhs[target]
        nextoffset = offset + old.size
        new = vlhs[offset:nextoffset].reshape(old.shape)
        new[...] = old
        new.flags.writeable = False
        lhs[target] = new
        offset = nextoffset
    assert offset == len(vlhs)
    return lhs, vlhs


def _invert(cons, targets):
    '''invert constraints dictionary to tuple referencing a contiguous array'''

    mask = []
    vmask = numpy.empty(sum(cons[target].size for target in targets), dtype=bool)
    offset = 0
    for target in targets:
        c = cons[target]
        nextoffset = offset + c.size
        mask.append(numpy.invert(c, out=vmask[offset:nextoffset].reshape(c.shape)))
        offset = nextoffset
    assert offset == len(vmask)
    return tuple(mask), vmask


def _integrate_blocks(*blocks, arguments, mask):
    '''helper function for blockwise integration'''

    *scalars, residuals, jacobians = blocks
    assert len(residuals) == len(mask)
    assert len(jacobians) == len(mask)**2
    data = iter(evaluable.eval_sparse((*scalars, *residuals, *jacobians), **arguments))
    nrg = [sparse.toarray(next(data)) for _ in range(len(scalars))]
    res = [sparse.take(next(data), [m]) for m in mask]
    jac = [[sparse.take(next(data), [mi, mj]) for mj in mask] for mi in mask]
    assert not list(data)
    return nrg + [sparse.toarray(sparse.block(res)), matrix.fromsparse(sparse.block(jac), inplace=True)]


def _argobjs(funcs):
    '''get :class:`evaluable.Argument` dependencies of multiple functions'''

    argobjs = {}
    for func in filter(None, funcs):
        for arg in func.arguments:
            if isinstance(arg, evaluable.Argument):
                if arg._name in argobjs:
                    if argobjs[arg._name] != arg:
                        raise ValueError('shape or dtype mismatch for argument {}: {} != {}'.format(arg._name, argobjs[arg._name], arg))
                else:
                    argobjs[arg._name] = arg
    return argobjs


def _determine_dtype(targets, residuals, lhs0, constrain):
    argobjs = _argobjs(residuals)
    dtype = complex if (
        any(argobjs[target].dtype == complex for target in targets)
        or any(residual.dtype == complex for residual in residuals if residual is not None)
        or any(vec.dtype.kind == 'c' for vec in lhs0.values())
        or any(vec.dtype.kind == 'c' for vec in constrain.values())
    ) else float
    if not all(argobjs[target].dtype == dtype for target in targets):
        raise ValueError('All targets must have dtype {}.'.format(dtype.__name__))
    return dtype


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


class _with_solve(types.Immutable):
    '''add a .solve method to iterables'''

    def __init__(self, wrapped, item = None):
        self._wrapped = wrapped
        self._item = item

    def __iter__(self):
        return iter(self._wrapped) if self._item is None else ((res[self._item], info) for (res, info) in self._wrapped)

    def __getitem__(self, item):
        assert self._item is None
        return _with_solve(self._wrapped, item)

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
        with log.context(self._wrapped.__class__.__name__.strip('_')):
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
