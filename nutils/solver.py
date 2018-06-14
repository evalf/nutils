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

>>> res = domain.integral('basis_n,i u_,i + basis_n' @ ns, geometry=ns.x, degree=2)
>>> lhs = solver.solve_linear('lhs', residual=res, constrain=cons)
solve > solver returned with residual ...

The coefficients ``lhs`` represent the solution to the Poisson problem.

In addition to ``solve_linear`` the solver module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, cache, log, numeric, sample, types, util, matrix
import numpy, itertools, functools, numbers, collections


argdict = types.frozendict[types.strictstr,types.frozenarray]


class ModelError(Exception): pass


@types.apply_annotations
@cache.function
def solve_linear(target:types.strictstr, residual:sample.strictintegral, constrain:types.frozenarray=None, *, arguments:argdict={}, solveargs:types.frozendict={}):
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

  jacobian = residual.derivative(target)
  if jacobian.contains(target):
    raise ModelError('problem is not linear')
  assert target not in arguments, '`target` should not be defined in `arguments`'
  argshape = residual._argshape(target)
  arguments = collections.ChainMap(arguments, {target: numpy.zeros(argshape)})
  res, jac = sample.eval_integrals(residual, jacobian, arguments=arguments)
  return jac.solve(-res, constrain=constrain, **solveargs)


def solve(gen_lhs_resnorm, tol=0., maxiter=float('inf')):
  '''execute nonlinear solver, return lhs

  Iterates over nonlinear solver until tolerance is reached. Example::

      lhs = solve(newton(target, residual), tol=1e-5)

  Parameters
  ----------
  gen_lhs_resnorm : :class:`collections.abc.Generator`
      Generates (lhs, resnorm) tuples
  tol : :class:`float`
      Target residual norm
  maxiter : :class:`int`
      Maximum number of iterations

  Returns
  -------
  :class:`numpy.ndarray`
      Coefficient vector that corresponds to a smaller than ``tol`` residual.
  '''

  lhs, info = solve_withinfo(gen_lhs_resnorm, tol=tol, maxiter=maxiter)
  return lhs


@types.apply_annotations
@cache.function
def solve_withinfo(gen_lhs_resnorm, tol:types.strictfloat=0., maxiter:types.strictfloat=float('inf')):
  '''execute nonlinear solver, return lhs and info

  Like :func:`solve`, but return a 2-tuple of the solution and the
  corresponding info object which holds information about the final residual
  norm and other generator-dependent information.
  '''

  for iiter, (lhs, info) in log.enumerate('iter', gen_lhs_resnorm):
    resnorm = info.resnorm
    if resnorm <= tol or iiter > maxiter:
      break
    if not iiter:
      resnorm0 = resnorm
    elif tol:
      log.info('residual: {:.2e} ({:.0f}%)'.format(resnorm, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol)))
    else:
      raise ModelError('nonlinear problem requires a nonzero tolerance')
  if resnorm > tol:
    raise ModelError('failed to reach target tolerance')
  elif resnorm:
    log.info('tolerance reached in {} iterations with residual {:.2e}'.format(iiter, resnorm))
  return lhs, info


class RecursionWithSolve(cache.Recursion):
  '''add a .solve method to (lhs,resnorm) iterators

  Introduces the convenient form::

      newton(target, residual).solve(tol)

  Shorthand for::

      solve(newton(target, residual), tol)
  '''

  __slots__ = ()

  solve_withinfo = solve_withinfo
  solve = solve


class newton(RecursionWithSolve, length=1):
  '''iteratively solve nonlinear problem by gradient descent

  Generates targets such that residual approaches 0 using Newton procedure with
  line search based on the residual norm. Suitable to be used inside ``solve``.

  An optimal relaxation value is computed based on the following cubic
  assumption::

      |res(lhs + r * dlhs)|^2 = A + B * r + C * r^2 + D * r^3

  where ``A``, ``B``, ``C`` and ``D`` are determined based on the current
  residual and tangent, the new residual, and the new tangent. If this value is
  found to be close to 1 (``> maxrelax``) then the newton update is accepted.

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
  nrelax : :class:`int`
      Maximum number of relaxation steps before proceding with the updated
      coefficient vector (by default unlimited).
  minrelax : :class:`float`
      Lower bound for the relaxation value, to force re-evaluating the
      functional in situation where the parabolic assumption would otherwise
      result in unreasonably small steps.
  maxrelax : :class:`float`
      Relaxation value below which relaxation continues, unless ``nrelax`` is
      reached; should be a value less than or equal to 1.
  rebound : :class:`float`
      Factor by which the relaxation value grows after every update until it
      reaches unity.
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
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, jacobian:sample.strictintegral=None, lhs0:types.frozenarray[types.strictfloat]=None, constrain:types.frozenarray=None, nrelax:types.strictint=None, minrelax:types.strictfloat=.01, maxrelax:types.strictfloat=2/3, rebound:types.strictfloat=2., arguments:argdict={}, solveargs:types.frozendict={}):
    super().__init__()

    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    self.target = target
    argshape = residual._argshape(self.target)
    if residual.shape != argshape:
      raise ValueError('expected `residual` with shape {} but got {}'.format(argshape, residual.shape))
    self.residual = residual
    if jacobian is None:
      self.jacobian = residual.derivative(target)
    elif jacobian.shape == argshape * 2:
      self.jacobian = jacobian
    else:
      raise ValueError('expected `jacobian` with shape {} but got {}'.format(argshape * 2, jacobian.shape))
    if lhs0 is None:
      self.lhs0 = types.frozenarray.full(argshape, fill_value=0.)
    elif lhs0.shape == argshape:
      self.lhs0 = lhs0
    else:
      raise ValueError('expected `lhs0` with shape {} but got {}'.format(argshape, lhs0.shape))
    if constrain is None:
      self.constrain = types.frozenarray.full(argshape, fill_value=False)
    elif constrain.shape != argshape:
      raise ValueError('expected `constrain` with shape {} but got {}'.format(argshape, constrain.shape))
    elif constrain.dtype == float:
      self.constrain = types.frozenarray(~numpy.isnan(constrain), copy=False)
      self.lhs0 = types.frozenarray(numpy.choose(self.constrain, [self.lhs0, constrain]), copy=False)
    elif constrain.dtype == bool:
      self.constrain = constrain
    else:
      raise ValueError('`constrain` should have dtype bool or float but got {}'.format(constrain.dtype))
    self.nrelax = nrelax
    self.minrelax = minrelax
    self.maxrelax = maxrelax
    self.rebound = rebound
    self.arguments = arguments
    self.solveargs = solveargs
    self.islinear = not self.jacobian.contains(self.target)

  def _eval(self, lhs):
    arguments = {self.target: lhs}
    arguments.update(self.arguments)
    return sample.eval_integrals(self.residual, self.jacobian, arguments=arguments)

  def resume(self, history):
    if history:
      lhs, info = history[-1]
      res, jac = self._eval(lhs)
      resnorm = numpy.linalg.norm(res[~self.constrain])
      assert resnorm == info.resnorm
      relax = info.relax
    else:
      lhs = self.lhs0
      res, jac = self._eval(lhs)
      resnorm = numpy.linalg.norm(res[~self.constrain])
      relax = 1
      yield numpy.array(lhs), types.attributes(resnorm=resnorm, relax=relax)

    while resnorm:
      dlhs = -jac.solve(res, constrain=self.constrain, **self.solveargs) # dlhs corresponds to an unrelaxed newton update
      for irelax in itertools.count() if self.nrelax is None else range(self.nrelax):
        newlhs = lhs+relax*dlhs
        if self.islinear:
          newresnorm = 0.
          break
        res, jac = self._eval(newlhs)
        newresnorm = numpy.linalg.norm(res[~self.constrain])
        if not numpy.isfinite(newresnorm):
          log.info('residual norm {} / {}'.format(newresnorm, round(relax, 5)))
          relax *= self.minrelax
          continue
        # To determine optimal relaxation we create a polynomial estimation for the residual norm:
        #   P(x) = A + B (x/relax) + C (x/relax)^2 + D (x/relax)^3 ~= |res(lhs+x*dlhs)|^2
        # We determine A, B, C and D based on the following constraints:
        #   P(0) = |res(lhs)|^2
        #   P'(0) = 2 res(lhs).jac(lhs).dlhs = -2 |res(lhs)|^2
        #   P(relax) = |res(lhs+relax*dlhs)|^2
        #   P'(relax) = 2 res(lhs+relax*dlhs).jac(lhs+relax*dlhs).dlhs
        A = resnorm**2
        B = -2 * A * relax
        C = 3 * newresnorm**2 - 2 * numpy.dot(jac.matvec(dlhs)[~self.constrain], res[~self.constrain]) * relax - 3 * A - 2 * B
        D = newresnorm**2 - A - B - C
        # Minimizing P:
        #   B + 2 C (x/relax) + 3 D (x/relax)^2 = 0 => x/relax = (-C +/- sqrt(C^2 - 3 B D)) / (3 D)
        # Special case 1: largest root is negative
        #   -C / (3 D) + sqrt(C^2 - 3 B D) / abs(3 D) < 0 <=> sqrt(C^2 - 3 B D) < C * sign(D) <=> D < 0 & C < 0
        # Special case 2: smallest root is positive
        #   -C / (3 D) - sqrt(C^2 - 3 B D) / abs(3 D) > 0 <=> sqrt(C^2 - 3 B D) < -C * sign(D) <=> D < 0 & C > 0
        discriminant = C**2 - 3 * B * D
        scale = numpy.inf if discriminant < 0 or D < 0 and C < 0 else (numpy.sqrt(discriminant) - C) / (3 * D) if D else -B / (2 * C)
        log.info('residual norm {:+.2f}% / {} with minimum at x{}'.format(100*(newresnorm/resnorm-1), round(relax, 5), round(scale, 2)))
        if newresnorm < resnorm and scale > self.maxrelax:
          relax = min(relax * min(scale, self.rebound), 1)
          break
        relax *= max(scale, self.minrelax)
      else:
        log.warning('failed to', 'decrease' if newresnorm > resnorm else 'optimize', 'residual')
      lhs, resnorm = newlhs, newresnorm
      yield numpy.array(lhs), types.attributes(resnorm=resnorm, relax=relax)


class minimize(RecursionWithSolve, length=1):
  '''iteratively minimize nonlinear functional by gradient descent

  Generates targets such that residual approaches 0 using Newton procedure with
  line search based on the energy. Suitable to be used inside ``solve``.

  An optimal relaxation value is computed based on the following assumption::

      energy(lhs + r * dlhs) = A + B * r + C * r^2 + D * r^3 + E * r^4 + F * r^5

  where ``A``, ``B``, ``C``, ``D``, ``E`` and ``F`` are determined based on the
  current and new energy, residual and tangent. If this value is found to be
  close to 1 (``> maxrelax``) then the newton update is accepted.

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
  nrelax : :class:`int`
      Maximum number of relaxation steps before proceding with the updated
      coefficient vector (by default unlimited).
  minrelax : :class:`float`
      Lower bound for the relaxation value, to force re-evaluating the
      functional in situation where the parabolic assumption would otherwise
      result in unreasonably small steps.
  maxrelax : :class:`float`
      Relaxation value below which relaxation continues, unless ``nrelax`` is
      reached; should be a value less than or equal to 1.
  rebound : :class:`float`
      Factor by which the relaxation value grows after every update until it
      reaches unity.
  droptol : :class:`float`
      Threshold for leaving entries in the return value at NaN if they do not
      contribute to the value of the functional.
  minderiv : :class:`float`
      Only consider local minima for which ``f'' > f * minderiv``.
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
  def __init__(self, target:types.strictstr, energy:sample.strictintegral, lhs0:types.frozenarray[types.strictfloat]=None, constrain:types.frozenarray=None, nrelax:types.strictint=None, minrelax:types.strictfloat=.01, maxrelax:types.strictfloat=2/3, rebound:types.strictfloat=2., droptol:types.strictfloat=None, minderiv=1e-10, arguments:argdict={}, solveargs:types.frozendict={}):
    super().__init__()

    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    self.target = target
    if energy.shape != ():
      raise ValueError('`energy` should be scalar')
    argshape = energy._argshape(self.target)
    self.energy = energy
    self.residual = energy.derivative(target)
    self.jacobian = self.residual.derivative(target)
    if lhs0 is None:
      self.lhs0 = types.frozenarray.full(argshape, fill_value=0.)
    elif lhs0.shape == argshape:
      self.lhs0 = lhs0
    else:
      raise ValueError('expected `lhs0` with shape {} but got {}'.format(argshape, lhs0.shape))
    if constrain is None:
      self.constrain = types.frozenarray.full(argshape, fill_value=False)
    elif constrain.shape != argshape:
      raise ValueError('expected `constrain` with shape {} but got {}'.format(argshape, constrain.shape))
    elif constrain.dtype == float:
      self.constrain = types.frozenarray(~numpy.isnan(constrain), copy=False)
      self.lhs0 = types.frozenarray(numpy.choose(self.constrain, [self.lhs0, constrain]), copy=False)
    elif constrain.dtype == bool:
      self.constrain = constrain
    else:
      raise ValueError('`constrain` should have dtype bool or float but got {}'.format(constrain.dtype))
    self.nrelax = nrelax
    self.minrelax = minrelax
    self.maxrelax = maxrelax
    self.rebound = rebound
    self.droptol = droptol
    self.minderiv = minderiv
    self.arguments = arguments
    self.solveargs = solveargs
    self.islinear = not self.jacobian.contains(self.target)

  def _eval(self, lhs):
    arguments = {self.target: lhs}
    arguments.update(self.arguments)
    return sample.eval_integrals(self.energy, self.residual, self.jacobian, arguments=arguments)

  def resume(self, history):
    if history:
      lhs, info = history[-1]
      if self.droptol is not None:
        lhs = numpy.choose(numpy.isnan(lhs), [lhs, self.lhs0])
      nrg, res, jac = self._eval(lhs)
      assert nrg == info.energy
      resnorm = numpy.linalg.norm(res[~self.constrain])
      assert resnorm == info.resnorm
      relax = info.relax
    else:
      lhs = self.lhs0
      nrg, res, jac = self._eval(lhs)
      resnorm = numpy.linalg.norm(res[~self.constrain])
      relax = 1
      nosupp = self.droptol is not None and ~(jac.rowsupp(self.droptol)|self.constrain)
      yield _nan_at(lhs, nosupp), types.attributes(resnorm=resnorm, energy=nrg, relax=relax, shift=0)

    while resnorm:
      nosupp = self.droptol is not None and ~(jac.rowsupp(self.droptol)|self.constrain)
      dlhs = -jac.solve(res, constrain=self.constrain|nosupp)
      if self.islinear:
        yield _nan_at(lhs+dlhs, nosupp), types.attributes(resnorm=0, energy=nrg+.5*res.dot(dlhs), relax=1, shift=0)
        return
      shift = 0
      while res.dot(dlhs) > 0:
        shift += res.dot(res) / res.dot(dlhs)
        log.warning('negative eigenvalue detected; shifting spectrum by {:.2e}'.format(shift))
        dlhs = -(jac + shift * matrix.eye(len(dlhs))).solve(res, constrain=self.constrain|nosupp)
      for irelax in itertools.count() if self.nrelax is None else range(self.nrelax):
        newlhs = lhs+relax*dlhs
        newnrg, newres, newjac = self._eval(newlhs)
        resnorm = numpy.linalg.norm(newres[~self.constrain])
        if not numpy.isfinite(newnrg):
          log.info('energy {} / {}'.format(newnrg, round(relax, 5)))
          relax *= self.minrelax
          continue
        # To determine optimal relaxation we create a polynomial estimation for the energy:
        #   P(x) = A + B (x/relax) + C (x/relax)^2 + D (x/relax)^3 + E (x/relax)^4 + F (x/relax)^5 ~= nrg(lhs+x*dlhs)
        # We determine A, B, C, D, E and F based on the following constraints:
        #   P(0) = nrg(lhs)
        #   P'(0) = res(lhs).dlhs
        #   P''(0) = dlhs.jac(lhs).dlhs
        #   P(relax) = nrg(lhs+relax*dlhs)
        #   P'(relax) = res(lhs+relax*dlhs).dlhs
        #   P''(relax) = dlhs.jac(lhs+relax*dlhs).dlhs
        A = nrg
        B = res.dot(dlhs) * relax
        C = .5 * jac.matvec(dlhs).dot(dlhs) * relax**2
        D = 10 * newnrg - 4 * newres.dot(dlhs) * relax + 0.25 * newjac.matvec(dlhs).dot(dlhs) * relax**2 - 10 * A - 6 * B - 3 * C
        E = 5 * newnrg - newres.dot(dlhs) * relax - 5 * A - 4 * B - 3 * C - 2 * D
        F = newnrg - A - B - C - D - E
        # Minimizing P:
        #   B + 2 C (x/relax) + 3 D (x/relax)^2 + 4 E (x/relax)^3 + 5 F (x/relax)^4 = 0
        #   C + 3 D (x/relax) + 6 E (x/relax)^2 + 10 F (x/relax)^3 > eps
        roots = [r.real for r in numpy.roots([5*F,4*E,3*D,2*C,B]) if not r.imag and r > 0 and C+r*(3*D+r*(6*E+r*10*F)) > A*self.minderiv]
        scale = min(roots) if roots else numpy.inf
        log.info('energy {:+.2f}% / {} with minimum at x{}'.format(100*(newnrg/nrg-1), round(relax, 5), round(scale, 2)))
        if scale > self.maxrelax:
          relax = min(relax * min(scale, self.rebound), 1)
          break
        relax *= max(scale, self.minrelax)
      else:
        log.warning('failed to', 'decrease' if newnrg > nrg else 'optimize', 'energy')
      yield _nan_at(newlhs, nosupp), types.attributes(resnorm=resnorm, energy=newnrg, relax=relax, shift=shift)
      nrg, res, jac = newnrg, newres, newjac
      lhs = numpy.choose(nosupp, [newlhs, self.lhs0])


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
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, inertia:sample.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray[types.strictfloat], residual0:sample.strictintegral=None, constrain:types.frozenarray=None, arguments:argdict={}, solveargs:types.frozendict={}):
    super().__init__()

    assert target not in arguments, '`target` should not be defined in `arguments`'

    self.target = target
    self.residual = residual
    self.inertia = inertia
    self.timestep = timestep
    self.lhs0 = lhs0
    self.residual0 = residual0
    self.constrain = constrain
    self.arguments = arguments
    self.solveargs = solveargs

  def resume(self, history):

    jacobian0 = self.residual.derivative(self.target)
    jacobiant = self.inertia.derivative(self.target)
    residual = self.residual
    if self.residual0 is not None:
      residual += self.residual0
    inertia = self.inertia

    argshape = residual._argshape(self.target)
    assert len(argshape) == 1

    lhs0 = self.lhs0.copy()
    constrain = self.constrain
    if constrain is None:
      constrain = numpy.zeros(residual.shape, dtype=bool)
    else:
      assert numeric.isarray(constrain) and constrain.dtype in (bool,float) and constrain.shape == residual.shape, 'invalid constrain argument'
      if constrain.dtype == float:
        lhs0 = numpy.choose(numpy.isnan(constrain), [constrain, lhs0])
        constrain = ~numpy.isnan(constrain)
    constrain = types.frozenarray(constrain)

    if history:
      lhs, info = history[-1]
      res0 = residual.eval(arguments=collections.ChainMap(self.arguments, {self.target: lhs0}))
      resnorm0 = numpy.linalg.norm(res0[~constrain])
      res, jac = sample.eval_integrals(residual, jacobian0+jacobiant/info.timestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
      resnorm = numpy.linalg.norm(res[~constrain])
      assert resnorm == info.resnorm
    else:
      res, jac = sample.eval_integrals(residual, jacobian0+jacobiant/self.timestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs0}))
      lhs = lhs0
      resnorm = resnorm0 = numpy.linalg.norm(res[~constrain])
      yield lhs, types.attributes(resnorm=resnorm, timestep=self.timestep)

    while True:
      lhs -= jac.solve(res, constrain=constrain, **self.solveargs)
      thistimestep = self.timestep * (resnorm0/resnorm)
      log.info('timestep: {:.0e}'.format(thistimestep))
      res, jac = sample.eval_integrals(residual, jacobian0+jacobiant/thistimestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
      resnorm = numpy.linalg.norm(res[~constrain])
      yield lhs, types.attributes(resnorm=resnorm, timestep=thistimestep)


class thetamethod(RecursionWithSolve, length=1):
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

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector for all timesteps after the initial condition.
  '''

  @types.apply_annotations
  def __init__(self, target:types.strictstr, residual:sample.strictintegral, inertia:sample.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray, theta:types.strictfloat, target0:types.strictstr='_thetamethod_target0', constrain:types.frozenarray=None, newtontol:types.strictfloat=1e-10, arguments:argdict={}, newtonargs:types.frozendict={}):
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
    res0 = residual * theta + inertia / timestep
    res1 = residual * (1-theta) - inertia / timestep
    self.res = res0 + res1.replace({target: function.Argument(target0, lhs0.shape)})
    self.jac = self.res.derivative(target)

  def resume(self, history):
    if history:
      lhs, = history
    else:
      lhs = self.lhs0
      yield lhs
    while True:
      lhs = newton(self.target, residual=self.res, jacobian=self.jac, lhs0=lhs, constrain=self.constrain, arguments=collections.ChainMap(self.arguments, {self.target0: lhs}), **self.newtonargs).solve(tol=self.newtontol)
      yield lhs


impliciteuler = functools.partial(thetamethod, theta=1)
cranknicolson = functools.partial(thetamethod, theta=0.5)


@log.title
def optimize(target:types.strictstr, functional:sample.strictintegral, *, newtontol:types.strictfloat=0., **kwargs):
  '''find the minimizer of a given functional

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  functional : scalar :class:`nutils.sample.Integral`
      The functional the should be minimized by varying target
  newtontol : :class:`float`
      Residual tolerance of Newton procedure (if applicable)
  **kwargs
      Additional arguments for :class:`minimize`

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector corresponding to the functional optimum
  '''

  lhs = minimize(target, functional, **kwargs).solve(newtontol)
  log.info('constrained {}/{} dofs'.format(len(lhs)-numpy.isnan(lhs).sum(), len(lhs)))
  return lhs


## HELPER FUNCTIONS

def _nan_at(vec, where):
  copy = vec.copy()
  if where is not False:
    copy[where] = numpy.nan
  return copy
