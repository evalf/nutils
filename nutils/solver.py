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
The solver module defines solvers for problems of the kind ``res = 0`` or
``∂inertia/∂t + res = 0``, where ``res`` is a
:class:`nutils.sample.Integral`.  To demonstrate this consider the following
setup:

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
solve > solving 25 dof system to machine precision using direct solver
solve > solver returned with residual ...

The coefficients ``lhs`` represent the solution to the Poisson problem.

In addition to ``solve_linear`` the solver module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, cache, numeric, sample, types, util, matrix, warnings
import abc, numpy, itertools, functools, numbers, collections, math, treelog as log


argdict = types.frozendict[types.strictstr,types.frozenarray]


class SolverError(Exception): pass


@types.apply_annotations
@cache.function
def solve_linear(target:types.strictstr, residual:sample.strictintegral, constrain:types.frozenarray=None, *, arguments:argdict={}, solveargs:types.frozendict={}, **linargs):
  '''solve linear problem

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.sample.Integral`
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

  solveargs = _striplin(linargs, solveargs)
  jacobian = residual.derivative(target)
  if jacobian.contains(target):
    raise SolverError('problem is not linear')
  assert target not in arguments, '`target` should not be defined in `arguments`'
  argshape = residual._argshape(target)
  res, jac = sample.eval_integrals(residual, jacobian, **{target: numpy.zeros(argshape)}, **arguments)
  return jac.solve(-res, constrain=constrain, **solveargs)

def solve(gen_lhs_resnorm, tol=0., maxiter=float('inf')):
  warnings.deprecation('solve(x, ...) is deprecated, use x.solve(...) instead')
  return gen_lhs_resnorm.solve(tol, maxiter)

def solve_withinfo(gen_lhs_resnorm, tol:types.strictfloat=0., maxiter:types.strictfloat=float('inf')):
  warnings.deprecation('solve_withinfo(x, ...) is deprecated, use x.solve_withinfo(...) instead')
  return gen_lhs_resnorm.solve_withinfo(tol, maxiter)


class RecursionWithSolve(cache.Recursion):
  '''add a .solve method to (lhs,resnorm) iterators'''

  __slots__ = ()

  def solve(self, tol=0., maxiter=float('inf')):
    '''execute nonlinear solver, return lhs
  
    Iterates over nonlinear solver until tolerance is reached. Example::
  
        lhs = newton(target, residual).solve(tol=1e-5)
  
    Parameters
    ----------
    tol : :class:`float`
        Target residual norm
    maxiter : :class:`int`
        Maximum number of iterations
  
    Returns
    -------
    :class:`numpy.ndarray`
        Coefficient vector that corresponds to a smaller than ``tol`` residual.
    '''
  
    lhs, info = self.solve_withinfo(tol=tol, maxiter=maxiter)
    return lhs

  @types.apply_annotations
  @cache.function
  def solve_withinfo(self, tol=0., maxiter=float('inf')):
    '''execute nonlinear solver, return lhs and info

    Like :func:`solve`, but return a 2-tuple of the solution and the
    corresponding info object which holds information about the final residual
    norm and other generator-dependent information.
    '''
  
    if not tol:
      warnings.deprecation('solve with zero tolerance is deprecated and will be removed; proceeding with tol=1e-12')
      tol = 1e-12
    with log.iter.wrap(_progress(self.__class__.__name__, tol), self) as items:
      i = 0
      for lhs, info in items:
        if info.resnorm <= tol:
          break
        if i > maxiter:
          raise SolverError('failed to reach target tolerance')
        i += 1
      log.info('converged in {} steps to residual {:.1e}'.format(i, info.resnorm))
    return lhs, info


class newton(RecursionWithSolve, length=1):
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
  residual : :class:`nutils.sample.Integral`
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  relax0 : :class:`float`
      Initial relaxation value.
  constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  linesearch : :class:`nutils.solver.LineSearch`
      Callable that defines relaxation logic.
  failrelax : :class:`float`
      Fail with exception if relaxation reaches this lower limit.
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector that approximates residual==0 with increasing accuracy
  '''

  @types.apply_annotations
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, jacobian:sample.strictintegral=None, lhs0:types.frozenarray[types.strictfloat]=None, relax0:float=1., constrain:types.frozenarray=None, linesearch=None, failrelax:types.strictfloat=1e-6, arguments:argdict={}, solveargs:types.frozendict={}, **linargs):
    super().__init__()
    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    self.target = target
    self.residual = residual
    self.jacobian = _derivative(residual, target, jacobian)
    self.lhs0, constrain = _parse_lhs_cons(lhs0, constrain, residual.shape)
    self.relax0 = relax0
    self.free = ~constrain
    self.linesearch = linesearch or NormBased.legacy(linargs)
    self.failrelax = failrelax
    self.arguments = arguments
    self.solveargs = _striplin(linargs, solveargs)
    self.solveargs.setdefault('rtol', 1e-3)

  def _eval(self, lhs):
    res, jac = sample.eval_integrals(self.residual, self.jacobian, **{self.target: lhs}, **self.arguments)
    return res[self.free], jac.submatrix(self.free, self.free)

  def resume(self, history):
    if history:
      lhs, info = history[-1]
      res, jac = self._eval(lhs)
      assert numpy.linalg.norm(res) == info.resnorm
      relax = info.relax
    else:
      lhs = self.lhs0
      res, jac = self._eval(lhs)
      relax = self.relax0
      yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), relax=relax)
    while True:
      dlhs = -jac.solve_leniently(res, **self.solveargs) # compute new search vector
      dres = jac@dlhs # == -res if dlhs was solved to infinite precision
      while True: # line search
        newlhs = lhs.copy()
        newlhs[self.free] += relax * dlhs
        newres, newjac = self._eval(newlhs)
        scale, accept = self.linesearch(res, relax*dres, newres, relax*newjac@dlhs)
        if accept:
          break
        relax *= scale
        if relax <= self.failrelax:
          raise SolverError('stuck in local minimum')
      log.info('update accepted at relaxation', round(relax, 5))
      lhs = newlhs
      res = newres
      jac = newjac
      relax = min(relax * scale, 1)
      yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), relax=relax)


class LineSearch(types.Immutable):
  '''
  Line search abstraction for gradient based optimization.

  A line search object is a callable that takes four arguments: the current
  residual and directional derivative, and the candidate residual and
  directional derivative, with derivatives normalized to unit length; and
  returns the optimal scaling and a boolean flag that marks whether the
  candidate should be accepted.
  '''

  @abc.abstractmethod
  def __call__(self, res0, dres0, res1, dres1):
    raise NotImplementedError


class NormBased(LineSearch):
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

  @types.apply_annotations
  def __init__(self, minscale:float=.01, acceptscale:float=2/3, maxscale:float=2.):
    assert 0 < minscale < acceptscale < 1 < maxscale
    self.minscale = minscale
    self.acceptscale = acceptscale
    self.maxscale = maxscale

  @classmethod
  def legacy(cls, kwargs):
    minscale, acceptscale = kwargs.pop('searchrange', (.01, 2/3))
    maxscale = kwargs.pop('rebound', 2.)
    return cls(minscale=minscale, acceptscale=acceptscale, maxscale=maxscale)

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
    if scale >= 1 and p1 > p0: # this should not happen, but just in case
      log.info('failed to estimate scale factor')
      return self.minscale, False
    log.info('estimated residual minimum at {:.0f}% of update vector'.format(scale*100))
    return min(max(scale, self.minscale), self.maxscale), scale >= self.acceptscale and p1 < p0


class MedianBased(LineSearch, version=1):
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

  @types.apply_annotations
  def __init__(self, minscale:float=.01, acceptscale:float=2/3, maxscale:float=2., quantile:float=.5):
    assert 0 < minscale < acceptscale < 1 < maxscale
    assert 0 < quantile < 1
    self.minscale = minscale
    self.acceptscale = acceptscale
    self.maxscale = maxscale
    self.quantile = quantile

  def __call__(self, res0, dres0, res1, dres1):
    if not numpy.isfinite(res1).all():
      log.info('non-finite residual')
      return self.minscale, False
    # To determine optimal relaxation we minimize a polynomial estimation for
    # the squared residual: P(x) = p0 + q0 x + c x^2 + d x^3
    dp = res1**2 - res0**2
    q0 = 2*res0*dres0
    q1 = 2*res1*dres1
    mask = q0 <= 0 # ideally this mask is all true, but solver inaccuracies can result in some positive slopes
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


class minimize(RecursionWithSolve, length=1, version=3):
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
  residual : :class:`nutils.sample.Integral`
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  rampup : :class:`float`
      Value to increase the relaxation power by in case energy is decreasing.
  rampdown : :class:`float`
      Value to decrease the relaxation power by in case energy is increasing.
  failrelax : :class:`float`
      Fail with exception if relaxation reaches this lower limit.
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector that approximates residual==0 with increasing accuracy
  '''

  @types.apply_annotations
  def __init__(self, target:types.strictstr, energy:sample.strictintegral, lhs0:types.frozenarray[types.strictfloat]=None, constrain:types.frozenarray=None, rampup:types.strictfloat=.5, rampdown:types.strictfloat=-1., failrelax:types.strictfloat=-10., arguments:argdict={}, solveargs:types.frozendict={}, **linargs):
    super().__init__()
    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    if energy.shape != ():
      raise ValueError('`energy` should be scalar')
    self.target = target
    self.energy = energy
    self.residual = energy.derivative(target)
    self.jacobian = _derivative(self.residual, target)
    self.lhs0, self.constrain = _parse_lhs_cons(lhs0, constrain, self.residual.shape)
    self.rampup = rampup
    self.rampdown = rampdown
    self.failrelax = failrelax
    self.arguments = arguments
    self.solveargs = _striplin(linargs, solveargs)

  def _eval(self, lhs):
    nrg, res, jac = sample.eval_integrals(self.energy, self.residual, self.jacobian, **{self.target: lhs}, **self.arguments)
    return nrg, res[~self.constrain], jac.submatrix(~self.constrain, ~self.constrain)

  def resume(self, history):
    if history:
      lhs, info = history[-1]
      nrg, res, jac = self._eval(lhs)
      assert nrg == info.energy
      assert numpy.linalg.norm(res) == info.resnorm
      relax = info.relax
    else:
      lhs = self.lhs0
      nrg, res, jac = self._eval(lhs)
      relax = 0
      yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), energy=nrg, relax=relax)

    while True:
      nrg0 = nrg
      lhs0 = lhs
      dlhs = -jac.solve_leniently(res, **self.solveargs)

      # compute first two ritz values to determine approximate path of steepest descent
      dlhsnorm = numpy.linalg.norm(dlhs)
      k0 = dlhs / dlhsnorm
      k1 = -res / dlhsnorm # = jac @ k0
      a = k1 @ k0
      k1 -= k0 * a # orthogonalize
      c = numpy.linalg.norm(k1)
      k1 /= c # normalize
      b = k1 @ (jac @ k1)
      # at this point k0 and k1 are orthonormal, and [k0 k1]^T jac [k0 k1] = [a c; c b]
      D = numpy.hypot(b-a, 2*c)
      L = numpy.array([a+b-D, a+b+D]) / 2 # 2nd order ritz values: eigenvalues of [a c; c b]
      v0, v1 = res + dlhs * L[:,numpy.newaxis]
      V = numpy.array([v1, -v0]).T / D # ritz vectors times dlhs -- note: V.dot(L) = -res, V.sum() = dlhs
      log.info('spectrum: {:.1e}..{:.1e} ({}definite)'.format(*L, 'positive ' if L[0] > 0 else 'negative ' if L[-1] < 0 else 'in'))

      for irelax in itertools.count():
        r = numpy.exp(relax - numpy.log(D)) # = exp(relax) / D
        eL = numpy.exp(-r*L)
        lhs = lhs0.copy()
        lhs[~self.constrain] += dlhs - V.dot(eL)
        nrg, res, jac = self._eval(lhs)
        slope = res.dot(V.dot(eL*L))
        log.info('energy {:+.2e} / e{:+.1f} and {}creasing'.format(nrg - nrg0, relax, 'in' if slope > 0 else 'de'))
        if numpy.isfinite(nrg) and numpy.isfinite(res).all() and nrg <= nrg0 and slope <= 0:
          relax += self.rampup
          break
        relax += self.rampdown
        if relax <= self.failrelax:
          raise SolverError('stuck in local minimum')

      yield lhs, types.attributes(resnorm=numpy.linalg.norm(res), energy=nrg, relax=relax)


class pseudotime(RecursionWithSolve, length=1):
  '''iteratively solve nonlinear problem by pseudo time stepping

  Generates targets such that residual approaches 0 using hybrid of Newton and
  time stepping. Requires an inertia term and initial timestep. Suitable to be
  used inside ``solve``.

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.sample.Integral`
  inertia : :class:`nutils.sample.Integral`
  timestep : :class:`float`
      Initial time step, will scale up as residual decreases
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Yields
  ------
  :class:`numpy.ndarray` with dtype :class:`float`
      Tuple of coefficient vector and residual norm
  '''

  @types.apply_annotations
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, inertia:sample.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray[types.strictfloat]=None, constrain:types.frozenarray=None, arguments:argdict={}, solveargs:types.frozendict={}, **linargs):
    super().__init__()
    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    if inertia.shape != residual.shape:
      raise ValueError('expected `inertia` with shape {} but got {}'.format(residual.shape, inertia.shape))
    self.target = target
    self.residual = residual
    self.jacobian = _derivative(residual, target)
    self.inertia = inertia
    self.jacobiant = _derivative(inertia, target)
    self.lhs0, self.constrain = _parse_lhs_cons(lhs0, constrain, residual.shape)
    self.timestep = timestep
    self.arguments = arguments
    self.solveargs = _striplin(linargs, solveargs)
    self.solveargs.setdefault('rtol', 1e-3)

  def _eval(self, lhs, timestep):
    return sample.eval_integrals(self.residual, self.jacobian+self.jacobiant/timestep, **{self.target: lhs}, **self.arguments)

  def resume(self, history):
    if history:
      lhs, info = history[-1]
      resnorm0 = info.resnorm0
      timestep = info.timestep
      res, jac = self._eval(lhs, timestep)
      resnorm = numpy.linalg.norm(res[~self.constrain])
      assert resnorm == info.resnorm
    else:
      lhs = self.lhs0
      timestep = self.timestep
      res, jac = self._eval(lhs, timestep)
      resnorm = resnorm0 = numpy.linalg.norm(res[~self.constrain])
      yield numpy.array(lhs), types.attributes(resnorm=resnorm, timestep=timestep, resnorm0=resnorm0)

    lhs = numpy.array(lhs)
    while True:
      lhs -= jac.solve_leniently(res, constrain=self.constrain, **self.solveargs)
      timestep = self.timestep * (resnorm0/resnorm)
      log.info('timestep: {:.0e}'.format(timestep))
      res, jac = self._eval(lhs, timestep)
      resnorm = numpy.linalg.norm(res[~self.constrain])
      yield lhs.copy(), types.attributes(resnorm=resnorm, timestep=timestep, resnorm0=resnorm0)


class thetamethod(RecursionWithSolve, length=1, version=1):
  '''solve time dependent problem using the theta method

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.sample.Integral`
  inertia : :class:`nutils.sample.Integral`
  timestep : :class:`float`
      Initial time step, will scale up as residual decreases
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  theta : :class:`float`
      Theta value (theta=1 for implicit Euler, theta=0.5 for Crank-Nicolson)
  residual0 : :class:`nutils.sample.Integral`
      Optional additional residual component evaluated in previous timestep
  constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  newtontol : :class:`float`
      Residual tolerance of individual timesteps
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.
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

  __cache__ = '_res_jac'

  @types.apply_annotations
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, inertia:sample.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray, theta:types.strictfloat, target0:types.strictstr='_thetamethod_target0', constrain:types.frozenarray=None, newtontol:types.strictfloat=1e-10, arguments:argdict={}, newtonargs:types.frozendict={}, timetarget:types.strictstr=None, time0:types.strictfloat=0.):
    super().__init__()

    assert target != target0, '`target` should not be equal to `target0`'
    assert target not in arguments, '`target` should not be defined in `arguments`'
    assert target0 not in arguments, '`target0` should not be defined in `arguments`'
    self.target = target
    self.target0 = target0
    self.lhs0 = lhs0
    self.constrain = constrain
    self.newtonargs = newtonargs
    self.newtontol = newtontol
    self.arguments = arguments
    self.residual = residual
    self.inertia = inertia
    self.theta = theta
    self.timestep = timestep
    self.timetarget = timetarget or '_thetamethod_dummy'
    self.time0 = time0

  def _res_jac(self, timestep):
    res = (self.residual * self.theta + self.inertia / timestep).replace({self.timetarget: function.Argument(self.timetarget, ())+timestep}) \
        + (self.residual * (1-self.theta) - self.inertia / timestep).replace({self.target: function.Argument(self.target0, self.lhs0.shape)})
    return res, res.derivative(self.target)

  def _step(self, lhs, t, timestep):
    res, jac = self._res_jac(timestep)
    try:
      return newton(self.target, residual=res, jacobian=jac, lhs0=lhs, constrain=self.constrain,
        arguments=collections.ChainMap(self.arguments, {self.target0: lhs, self.timetarget: t}), **self.newtonargs).solve(tol=self.newtontol)
    except (SolverError, matrix.MatrixError) as e:
      log.error('error: {}; retrying with timestep {}'.format(e, timestep/2))
      return self._step(self._step(lhs, t, timestep/2), t+timestep/2, timestep/2)

  def resume_index(self, history, index):
    if history:
      lhs, = history
    else:
      lhs = self.lhs0
      yield lhs
    while True:
      lhs = self._step(lhs, self.time0+index*self.timestep, self.timestep)
      index += 1
      yield lhs

impliciteuler = functools.partial(thetamethod, theta=1)
cranknicolson = functools.partial(thetamethod, theta=0.5)


@log.withcontext
@types.apply_annotations
@cache.function
def optimize(target:types.strictstr, functional:sample.strictintegral, *, tol:types.strictfloat=0., arguments:argdict={}, droptol:float=None, constrain:types.frozenarray=None, lhs0:types.frozenarray[types.strictfloat]=None, relax0:float=1., solveargs:types.frozendict={}, linesearch=None, failrelax:types.strictfloat=1e-6, **linargs):
  '''find the minimizer of a given functional

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  functional : scalar :class:`nutils.sample.Integral`
      The functional the should be minimized by varying target
  tol : :class:`float`
      Target residual norm.
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.
  droptol : :class:`float`
      Threshold for leaving entries in the return value at NaN if they do not
      contribute to the value of the functional.
  constrain : :class:`numpy.ndarray` with dtype :class:`float`
      Defines the fixed entries of the coefficient vector
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  relax0 : :class:`float`
      Initial relaxation value.
  linesearch : :class:`nutils.solver.LineSearch`
      Callable that defines relaxation logic.
  failrelax : :class:`float`
      Fail with exception if relaxation reaches this lower limit.

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector corresponding to the functional optimum
  '''

  if 'newtontol' in linargs:
    warnings.deprecation('argument "newtontol" is deprecated, use "tol" instead')
    tol = linargs.pop('newtontol')
  if linesearch is None:
    linesearch = NormBased.legacy(linargs)
  solveargs = _striplin(linargs, solveargs)
  residual = functional.derivative(target)
  jacobian = residual.derivative(target)
  lhs, cons = _parse_lhs_cons(lhs0, constrain, residual.shape)
  val, res, jac = sample.eval_integrals(functional, residual, jacobian, **{target: lhs}, **arguments)
  if droptol is not None:
    nan = ~(cons|jac.rowsupp(droptol))
    cons = cons | nan
  resnorm = numpy.linalg.norm(res[~cons])
  if jacobian.contains(target):
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
          lhs0 = lhs
          dlhs = -jac.solve_leniently(res, constrain=cons, **solveargs)
          res0 = res[~cons]
          dres0 = (jac@dlhs)[~cons] # == -res0 if dlhs was solved to infinite precision
          resnorm0 = resnorm
        lhs = lhs0 + relax * dlhs
        val, res, jac = sample.eval_integrals(functional, residual, jacobian, **{target: lhs}, **arguments)
        resnorm = numpy.linalg.norm(res[~cons])
        scale, accept = linesearch(res0, relax*dres0, res[~cons], relax*(jac@dlhs)[~cons])
        relax = min(relax * scale, 1)
        if relax <= failrelax:
          raise SolverError('stuck in local minimum')
      log.info('converged with residual {:.1e}'.format(resnorm))
  elif resnorm > tol:
    solveargs.setdefault('atol', tol)
    dlhs = -jac.solve(res, constrain=cons, **solveargs)
    lhs = lhs + dlhs
    val += (res + jac@dlhs/2).dot(dlhs)
  if droptol is not None:
    lhs = numpy.choose(nan, [lhs, numpy.nan])
    log.info('constrained {}/{} dofs'.format(len(lhs)-nan.sum(), len(lhs)))
  log.info('optimum value {:.2e}'.format(val))
  return lhs


## HELPER FUNCTIONS

def _striplin(linargs, solveargs):
  if solveargs:
    warnings.deprecation('solveargs={"key": value} is deprecated, use linkey=value instead')
    solveargs = solveargs.copy()
  else:
    solveargs = {}
  for key, value in linargs.items():
    if not key.startswith('lin'):
      raise TypeError('unexpected keyword argument {!r}'.format(key))
    if key[3:] in solveargs:
      raise TypeError('duplicate solver argument: {!r}'.format(key))
    solveargs[key[3:]] = value
  return solveargs

def _parse_lhs_cons(lhs0, constrain, shape):
  if lhs0 is None:
    lhs0 = types.frozenarray.full(shape, fill_value=0.)
  elif lhs0.shape == shape:
    lhs0 = types.frozenarray(lhs0)
  else:
    raise ValueError('expected `lhs0` with shape {} but got {}'.format(shape, lhs0.shape))
  if constrain is None:
    constrain = types.frozenarray.full(shape, fill_value=False)
  elif constrain.shape != shape:
    raise ValueError('expected `constrain` with shape {} but got {}'.format(shape, constrain.shape))
  elif constrain.dtype == float:
    isnan = numpy.isnan(constrain)
    lhs0 = types.frozenarray(numpy.choose(isnan, [constrain, lhs0]), copy=False)
    constrain = types.frozenarray(~isnan, copy=False)
  elif constrain.dtype == bool:
    constrain = types.frozenarray(constrain)
  else:
    raise ValueError('`constrain` should have dtype bool or float but got {}'.format(constrain.dtype))
  return lhs0, constrain

def _derivative(residual, target, jacobian=None):
  if jacobian is None:
    jacobian = residual.derivative(target)
  if jacobian.shape != residual.shape * 2:
    raise ValueError('expected `jacobian` with shape {} but got {}'.format(residual.shape * 2, jacobian.shape))
  return jacobian

def _progress(name, tol):
  '''helper function for iter.wrap'''

  lhs, info = yield name
  resnorm0 = info.resnorm
  while True:
    lhs, info = yield (name + ' {:.0f}%').format(100 * numpy.log(resnorm0/max(info.resnorm,tol)) / numpy.log(resnorm0/tol) if tol else 0 if info.resnorm else 100)

# vim:sw=2:sts=2:et
