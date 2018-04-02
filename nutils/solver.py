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
:class:`nutils.topology.Integral`.  To demonstrate this consider the following
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

from . import function, cache, log, numeric, topology, types
import numpy, itertools, functools, numbers, collections


argdict = types.frozendict[types.strictstr,types.frozenarray]


class ModelError(Exception): pass


@types.apply_annotations
@cache.function
def solve_linear(target:types.strictstr, residual:topology.strictintegral, constrain:types.frozenarray=None, *, arguments:argdict={}, solveargs:types.frozendict={}):
  '''solve linear problem

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.topology.Integral`
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
  res, jac = topology.eval_integrals(residual, jacobian, arguments=arguments)
  return jac.solve(-res, constrain=constrain, **solveargs)


@types.apply_annotations
@cache.function
def solve(gen_lhs_resnorm, tol:types.strictfloat=1e-10, maxiter:types.strictfloat=float('inf')):
  '''execute nonlinear solver

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

  gen_lhs_resnorm = iter(gen_lhs_resnorm)
  try:
    lhs, resnorm = next(gen_lhs_resnorm)
    resnorm0 = resnorm
    inewton = 0
    while resnorm > tol:
      if inewton >= maxiter:
        raise ModelError('tolerance not reached in {} iterations'.format(maxiter))
      with log.context('iter {0} ({1:.0f}%)'.format(inewton, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol))):
        log.info('residual: {:.2e}'.format(resnorm))
        lhs, resnorm = next(gen_lhs_resnorm)
      inewton += 1
  except StopIteration:
    raise ModelError('generator stopped before reaching target tolerance')
  else:
    log.info('tolerance reached in {} iterations with residual {:.2e}'.format(inewton, resnorm))
    return lhs


class RecursionWithSolve(cache.Recursion):
  '''add a .solve method to (lhs,resnorm) iterators

  Introduces the convenient form::

      newton(target, residual).solve(tol)

  Shorthand for::

      solve(newton(target, residual), tol)
  '''

  __slots__ = ()

  solve = solve


class newton(RecursionWithSolve, length=1):
  '''iteratively solve nonlinear problem by gradient descent

  Generates targets such that residual approaches 0 using Newton procedure with
  line search based on a residual integral. Suitable to be used inside
  ``solve``.

  An optimal relaxation value is computed based on the following cubic
  assumption::

      | res(lhs + r * dlhs) |^2 = A + B * r + C * r^2 + D * r^3

  where ``A``, ``B``, ``C`` and ``D`` are determined based on the current and
  updated residual and tangent.

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.topology.Integral`
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
  def __init__(self, target:types.strictstr, residual:topology.strictintegral, jacobian:topology.strictintegral=None, lhs0:types.frozenarray[types.strictfloat]=None, constrain:types.frozenarray=None, nrelax:types.strictint=None, minrelax:types.strictfloat=.1, maxrelax:types.strictfloat=.9, rebound:types.strictfloat=2**.5, arguments:argdict={}, solveargs:types.frozendict={}):
    super().__init__()

    if target in arguments:
      raise ValueError('`target` should not be defined in `arguments`')
    self.target = target
    self.residual = residual
    self.jacobian = jacobian
    self.lhs0 = lhs0
    self.constrain = constrain
    self.nrelax = float('inf') if nrelax is None else nrelax
    self.minrelax = minrelax
    self.maxrelax = maxrelax
    self.rebound = rebound
    self.arguments = arguments
    self.solveargs = solveargs

  def resume(self, history):

    lhs0 = self.lhs0
    if lhs0 is None:
      lhs0 = numpy.zeros(self.residual.shape)
    elif lhs0.shape != self.residual.shape:
      raise ValueError('expected `lhs0` with shape {} but got {}'.format(self.residual.shape, lhs0.shape))

    constrain = self.constrain
    if constrain is None:
      constrain = numpy.zeros(self.residual.shape, dtype=bool)
    else:
      if constrain.dtype not in (bool, float):
        raise ValueError('`constrain` should have dtype bool or float but got {}'.format(constrain.dtype))
      if constrain.shape != self.residual.shape:
        raise ValueError('expected `constrain` with shape {} but got {}'.format(self.residual.shape, constrain.shape))
      if constrain.dtype == float:
        lhs0 = numpy.choose(numpy.isnan(constrain), [constrain, lhs0])
        constrain = ~numpy.isnan(constrain)
    constrain = types.frozenarray(constrain)

    argshape = self.residual._argshape(self.target)
    jacobian = self.residual.derivative(self.target) if self.jacobian is None else self.jacobian

    if not jacobian.contains(self.target):
      if history:
        return
      log.info('problem is linear')
      res, jac = topology.eval_integrals(self.residual, jacobian, arguments=collections.ChainMap(self.arguments, {self.target: numpy.zeros(argshape)}))
      lhs = jac.solve(-res, lhs0=lhs0, constrain=constrain, **self.solveargs)
      yield lhs, 0, 1
      return

    if history:
      (lhs, resnorm, relax), = history
      res, jac = topology.eval_integrals(self.residual, jacobian, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
    else:
      lhs = lhs0
      res, jac = topology.eval_integrals(self.residual, jacobian, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
      resnorm = numpy.linalg.norm(res[~constrain])
      relax = 1
      yield lhs, resnorm, relax

    while True:
      dlhs = -jac.solve(res, constrain=constrain, **self.solveargs)
      relax = min(relax * self.rebound, 1)
      for irelax in itertools.count():
        newlhs = lhs+relax*dlhs
        res, jac = topology.eval_integrals(self.residual, jacobian, arguments=collections.ChainMap(self.arguments, {self.target: newlhs}))
        newresnorm = numpy.linalg.norm(res[~constrain])
        if irelax >= self.nrelax:
          if newresnorm > resnorm:
            log.warning('failed to decrease residual')
            return
          break
        if not numpy.isfinite(newresnorm):
          log.info('failed to evaluate residual ({})'.format(newresnorm))
          newrelax = 0 # replaced by self.minrelax later
        else:
          r0 = resnorm**2
          d0 = -2 * r0
          r1 = newresnorm**2
          d1 = 2 * numpy.dot(jac.matvec(dlhs)[~constrain], res[~constrain])
          log.info('line search: 0[{}]{} {}creased by {:.0f}%'.format('---+++' if d1 > 0 else '--++--' if r1 > r0 else '------', round(relax,5), 'in' if newresnorm > resnorm else 'de', 100*abs(newresnorm/resnorm-1)))
          if r1 <= r0 and d1 <= 0:
            break
          D = 2*r0 - 2*r1 + d0 + d1
          if D > 0:
            C = 3*r1 - 3*r0 - 2*d0 - d1
            newrelax = (numpy.sqrt(C**2-3*d0*D) - C) / (3*D)
            log.info('minimum based on 3rd order estimation: {:.3f}'.format(newrelax))
          else:
            C = r1 - r0 - d0
            # r1 > r0 => C > 0
            # d1 > 0  => C = r1 - r0 - d0/2 - d0/2 > r1 - r0 - d0/2 - d1/2 = -D/2 > 0
            newrelax = -.5 * d0 / C
            log.info('minimum based on 2nd order estimation: {:.3f}'.format(newrelax))
          if newrelax > self.maxrelax:
            break
        relax *= max(newrelax, self.minrelax)
      lhs, resnorm = newlhs, newresnorm
      yield lhs, resnorm, relax

  def __iter__(self):
    for lhs, resnorm, relax in super().__iter__():
      yield lhs, resnorm


class pseudotime(RecursionWithSolve, length=1):
  '''iteratively solve nonlinear problem by pseudo time stepping

  Generates targets such that residual approaches 0 using hybrid of Newton and
  time stepping. Requires an inertia term and initial timestep. Suitable to be
  used inside ``solve``.

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.topology.Integral`
  inertia : :class:`nutils.topology.Integral`
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
  def __init__(self, target:types.strictstr, residual:topology.strictintegral, inertia:topology.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray[types.strictfloat], residual0:topology.strictintegral=None, constrain:types.frozenarray=None, arguments:argdict={}, solveargs:types.frozendict={}):
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
      (lhs, resnorm, thistimestep), = history
      res0 = residual.eval(arguments=collections.ChainMap(self.arguments, {self.target: lhs0}))
      resnorm0 = numpy.linalg.norm(res0[~constrain])
      res, jac = topology.eval_integrals(residual, jacobian0+jacobiant/thistimestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
    else:
      res, jac = topology.eval_integrals(residual, jacobian0+jacobiant/self.timestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs0}))
      lhs = lhs0
      resnorm = resnorm0 = numpy.linalg.norm(res[~constrain])
      yield lhs, resnorm, self.timestep

    while True:
      lhs -= jac.solve(res, constrain=constrain, **self.solveargs)
      thistimestep = self.timestep * (resnorm0/resnorm)
      log.info('timestep: {:.0e}'.format(thistimestep))
      res, jac = topology.eval_integrals(residual, jacobian0+jacobiant/thistimestep, arguments=collections.ChainMap(self.arguments, {self.target: lhs}))
      resnorm = numpy.linalg.norm(res[~constrain])
      yield lhs, resnorm, thistimestep

  def __iter__(self):
    for lhs, resnorm, timestep in super().__iter__():
      yield lhs, resnorm


class thetamethod(RecursionWithSolve, length=1):
  '''solve time dependent problem using the theta method

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : :class:`nutils.topology.Integral`
  inertia : :class:`nutils.topology.Integral`
  timestep : :class:`float`
      Initial time step, will scale up as residual decreases
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure.
  theta : :class:`float`
      Theta value (theta=1 for implicit Euler, theta=0.5 for Crank-Nicolson)
  residual0 : :class:`nutils.topology.Integral`
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
  def __init__(self, target:types.strictstr, residual:topology.strictintegral, inertia:topology.strictintegral, timestep:types.strictfloat, lhs0:types.frozenarray, theta:types.strictfloat, target0:types.strictstr='_thetamethod_target0', constrain:types.frozenarray=None, newtontol:types.strictfloat=1e-10, arguments:argdict={}, newtonargs:types.frozendict={}):
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
@types.apply_annotations
@cache.function
def optimize(target:types.strictstr, functional:topology.strictintegral, droptol:types.strictfloat=None, lhs0:types.frozenarray=None, constrain:types.frozenarray=None, newtontol:types.strictfloat=None, *, arguments:argdict={}):
  '''find the minimizer of a given functional

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  functional : scalar :class:`nutils.topology.Integral`
      The functional the should be minimized by varying target
  droptol : :class:`float`
      Threshold for leaving entries in the return value at NaN if they do not
      contribute to the value of the functional.
  lhs0 : :class:`numpy.ndarray`
      Coefficient vector, starting point of the iterative procedure (if
      applicable).
  constrain : :class:`numpy.ndarray` with dtype :class:`bool` or :class:`float`
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  newtontol : :class:`float`
      Residual tolerance of Newton procedure (if applicable)

  Yields
  ------
  :class:`numpy.ndarray`
      Coefficient vector corresponding to the functional optimum
  '''

  assert target not in arguments, '`target` should not be defined in `arguments`'
  assert len(functional.shape) == 0, 'functional should be scalar'
  argshape = functional._argshape(target)
  if lhs0 is None:
    lhs0 = numpy.zeros(argshape)
  else:
    assert numeric.isarray(lhs0) and lhs0.dtype == float and lhs0.shape == argshape, 'invalid lhs0 argument'
  if constrain is None:
    constrain = numpy.zeros(argshape, dtype=bool)
  else:
    assert numeric.isarray(constrain) and constrain.dtype in (bool,float) and constrain.shape == argshape, 'invalid constrain argument'
    if constrain.dtype == float:
      lhs0 = numpy.choose(numpy.isnan(constrain), [constrain, lhs0])
      constrain = ~numpy.isnan(constrain)
  residual = functional.derivative(target)
  jacobian = residual.derivative(target)
  f0, res, jac = topology.eval_integrals(functional, residual, jacobian, arguments=collections.ChainMap(arguments, {target: lhs0}))
  freezedofs = constrain if droptol is None else constrain | ~jac.rowsupp(droptol)
  log.info('optimizing for {}/{} degrees of freedom'.format(len(res)-freezedofs.sum(), len(res)))
  lhs = lhs0 - jac.solve(res, constrain=freezedofs) # residual(lhs0) + jacobian(lhs0) dlhs = 0
  if not jacobian.contains(target): # linear: functional(lhs0+dlhs) = functional(lhs0) + residual(lhs0) dlhs + .5 dlhs jacobian(lhs0) dlhs
    value = f0 + .5 * res.dot(lhs-lhs0)
  else: # nonlinear
    assert newtontol is not None, 'newton tolerance `newtontol` must be specified for nonlinear problems'
    lhs = newton(target, residual, lhs0=lhs, constrain=freezedofs, arguments=arguments).solve(newtontol)
    value = functional.eval(arguments=collections.ChainMap(arguments, {target: lhs}))
  assert numpy.isfinite(lhs).all(), 'optimization failed (forgot droptol?)'
  log.info('optimum: {:.2e}'.format(value))
  lhs[freezedofs & ~constrain] = numpy.nan
  return lhs
