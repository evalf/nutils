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
The solver module defines the :class:`Integral` class, which represents an
unevaluated integral. This is useful for fully automated solution procedures
such as Newton, that require functional derivatives of an entire functional.

To demonstrate this consider the following setup:

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
solve > solving system using sparse direct solver

The coefficients ``lhs`` represent the solution to the Poisson problem.

In addition to ``solve_linear`` the solver module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, cache, log, util, numeric
import numpy, itertools, functools, numbers, collections


class Integral:
  '''Postponed integral, used for derivative purposes'''

  def __init__(self, integrands):
    self._integrands = util.hashlessdict(integrands)
    shapes = {integrand.shape for integrand in self._integrands.values()}
    assert len(shapes) == 1, 'incompatible shapes: {}'.format(' != '.join(str(shape) for shape in shapes))
    self.shape, = shapes

  @classmethod
  def multieval(cls, *integrals, fcache=None, arguments=None):
    assert all(isinstance(integral, cls) for integral in integrals)
    if fcache is None:
      fcache = cache.WrapperCache()
    gather = util.hashlessdict()
    for iint, integral in enumerate(integrals):
      for di in integral._integrands:
        gather.setdefault(di, []).append(iint)
    retvals = [None] * len(integrals)
    for (domain, ischeme), iints in gather.items():
      for iint, retval in zip(iints, domain.integrate([integrals[iint]._integrands[domain, ischeme] for iint in iints], ischeme=ischeme, fcache=fcache, arguments=arguments)):
        if retvals[iint] is None:
          retvals[iint] = retval
        else:
          retvals[iint] += retval
    return retvals

  def eval(self, **kwargs):
    retval, = self.multieval(self, **kwargs)
    return retval

  def derivative(self, target):
    argshape = self._argshape(target)
    arg = function.Argument(target, argshape)
    seen = {}
    return Integral([di, function.derivative(integrand, var=arg, seen=seen)] for di, integrand in self._integrands.items())

  def replace(self, arguments):
    return Integral([di, function.replace_arguments(integrand, arguments)] for di, integrand in self._integrands.items())

  def contains(self, name):
    try:
      self._argshape(name)
    except KeyError:
      return False
    else:
      return True

  def __add__(self, other):
    if not isinstance(other, Integral):
      return NotImplemented
    assert self.shape == other.shape
    integrands = self._integrands.copy()
    for di, integrand in other._integrands.items():
      try:
        integrands[di] += integrand
      except KeyError:
        integrands[di] = integrand
    return Integral(integrands)

  def __neg__(self):
    return Integral([di, -integrand] for di, integrand in self._integrands.items())

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return Integral([di, integrand * other] for di, integrand in self._integrands.items())

  __rmul__ = __mul__

  def __truediv__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return self.__mul__(1/other)

  def _argshape(self, name):
    assert isinstance(name, str)
    shapes = {func.shape[:func.ndim-func._nderiv]
      for func in function.Tuple(self._integrands.values()).simplified.dependencies
        if isinstance(func, function.Argument) and func._name == name}
    if not shapes:
      raise KeyError(name)
    assert len(shapes) == 1, 'inconsistent shapes for argument {!r}'.format(name)
    shape, = shapes
    return shape


class ModelError(Exception): pass


def solve_linear(target, residual, constrain=None, *, arguments=None, **solveargs):
  '''solve linear problem

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : Integral
      Residual integral, depends on ``target``
  constrain : float vector
      Defines the fixed entries of the coefficient vector
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Returns
  -------
  vector
      Array of ``target`` values for which ``residual == 0``'''

  jacobian = residual.derivative(target)
  if jacobian.contains(target):
    raise ModelError('problem is not linear')
  assert target not in (arguments or {}), '`target` should not be defined in `arguments`'
  argshape = residual._argshape(target)
  arguments = collections.ChainMap(arguments or {}, {target: numpy.zeros(argshape)})
  res, jac = Integral.multieval(residual, jacobian, arguments=arguments)
  return jac.solve(-res, constrain=constrain, **solveargs)


def solve(gen_lhs_resnorm, tol=1e-10, maxiter=numpy.inf):
  '''execute nonlinear solver

  Iterates over nonlinear solver until tolerance is reached. Example::

      lhs = solve(newton(target, residual), tol=1e-5)

  Parameters
  ----------
  gen_lhs_resnorm : generator
      Generates (lhs, resnorm) tuples
  tol : float
      Target residual norm
  maxiter : int
      Maximum number of iterations

  Returns
  -------
  vector
      Coefficient vector that corresponds to a smaller than ``tol`` residual.
  '''

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


def withsolve(f):
  '''add a .solve method to (lhs,resnorm) iterators

  Introduces the convenient form::

      newton(target, residual).solve(tol)

  Shorthand for::

      solve(newton(target, residual), tol)
  '''

  @functools.wraps(f, updated=())
  class wrapper:
    def __init__(self, *args, **kwargs):
      self.iter = f(*args, **kwargs)
    def __next__(self):
      return next(self.iter)
    def __iter__(self):
      return self.iter
    def solve(self, *args, **kwargs):
      return solve(self.iter, *args, **kwargs)
  return wrapper


@withsolve
def newton(target, residual, jacobian=None, lhs0=None, constrain=None, nrelax=numpy.inf, minrelax=.1, maxrelax=.9, rebound=2**.5, *, arguments=None, **solveargs):
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
  residual : Integral
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  constrain : boolean or float vector
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  nrelax : int
      Maximum number of relaxation steps before proceding with the updated
      coefficient vector (by default unlimited).
  minrelax : float
      Lower bound for the relaxation value, to force re-evaluating the
      functional in situation where the parabolic assumption would otherwise
      result in unreasonably small steps.
  maxrelax : float
      Relaxation value below which relaxation continues, unless ``nrelax`` is
      reached; should be a value less than or equal to 1.
  rebound : float
      Factor by which the relaxation value grows after every update until it
      reaches unity.
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Yields
  ------
  vector
      Coefficient vector that approximates residual==0 with increasing accuracy
  '''

  assert target not in (arguments or {}), '`target` should not be defined in `arguments`'
  argshape = residual._argshape(target)

  if lhs0 is None:
    lhs0 = numpy.zeros(residual.shape)
  else:
    assert numeric.isarray(lhs0) and lhs0.dtype == float and lhs0.shape == residual.shape, 'invalid lhs0 argument'

  if constrain is None:
    constrain = numpy.zeros(residual.shape, dtype=bool)
  else:
    assert numeric.isarray(constrain) and constrain.dtype in (bool,float) and constrain.shape == residual.shape, 'invalid constrain argument'
    if constrain.dtype == float:
      lhs0 = numpy.choose(numpy.isnan(constrain), [constrain, lhs0])
      constrain = ~numpy.isnan(constrain)

  if jacobian is None:
    jacobian = residual.derivative(target)

  if not jacobian.contains(target):
    log.info('problem is linear')
    res, jac = Integral.multieval(residual, jacobian, arguments=collections.ChainMap(arguments or {}, {target: numpy.zeros(argshape)}))
    cons = lhs0.copy()
    cons[~constrain] = numpy.nan
    lhs = jac.solve(-res, constrain=cons, **solveargs)
    yield lhs, 0
    return

  lhs = lhs0.copy()
  fcache = cache.WrapperCache()
  res, jac = Integral.multieval(residual, jacobian, fcache=fcache, arguments=collections.ChainMap(arguments or {}, {target: lhs}))
  zcons = numpy.zeros(argshape)
  zcons[~constrain] = numpy.nan
  relax = 1
  while True:
    resnorm = numpy.linalg.norm(res[~constrain])
    yield lhs, resnorm
    dlhs = -jac.solve(res, constrain=zcons, **solveargs)
    relax = min(relax * rebound, 1)
    for irelax in itertools.count():
      res, jac = Integral.multieval(residual, jacobian, fcache=fcache, arguments=collections.ChainMap(arguments or {}, {target: lhs+relax*dlhs}))
      newresnorm = numpy.linalg.norm(res[~constrain])
      if irelax >= nrelax:
        if newresnorm > resnorm:
          log.warning('failed to decrease residual')
          return
        break
      if not numpy.isfinite(newresnorm):
        log.info('failed to evaluate residual ({})'.format(newresnorm))
        newrelax = 0 # replaced by minrelax later
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
        if newrelax > maxrelax:
          break
      relax *= max(newrelax, minrelax)
    lhs += relax * dlhs


@withsolve
def pseudotime(target, residual, inertia, timestep, lhs0, residual0=None, constrain=None, *, arguments=None, **solveargs):
  '''iteratively solve nonlinear problem by pseudo time stepping

  Generates targets such that residual approaches 0 using hybrid of Newton and
  time stepping. Requires an inertia term and initial timestep. Suitable to be
  used inside ``solve``.

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : Integral
  inertia : Integral
  timestep : float
      Initial time step, will scale up as residual decreases
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  constrain : boolean or float vector
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
  vector, float
      Tuple of coefficient vector and residual norm
  '''

  assert target not in (arguments or {}), '`target` should not be defined in `arguments`'

  jacobian0 = residual.derivative(target)
  jacobiant = inertia.derivative(target)
  if residual0 is not None:
    residual += residual0

  if constrain is None:
    constrain = numpy.zeros(residual.shape, dtype=bool)
  else:
    assert numeric.isarray(constrain) and constrain.dtype in (bool,float) and constrain.shape == residual.shape, 'invalid constrain argument'
    if constrain.dtype == float:
      lhs0 = numpy.choose(numpy.isnan(constrain), [constrain, lhs0])
      constrain = ~numpy.isnan(constrain)

  argshape = residual._argshape(target)
  assert len(argshape) == 1
  zcons = util.NanVec(argshape[0])
  zcons[constrain] = 0
  lhs = lhs0.copy()
  fcache = cache.WrapperCache()
  res, jac = Integral.multieval(residual, jacobian0+jacobiant/timestep, fcache=fcache, arguments=collections.ChainMap(arguments or {}, {target: lhs}))
  resnorm = resnorm0 = numpy.linalg.norm(res[~constrain])
  while True:
    yield lhs, resnorm
    lhs -= jac.solve(res, constrain=zcons, **solveargs)
    thistimestep = timestep * (resnorm0/resnorm)
    log.info('timestep: {:.0e}'.format(thistimestep))
    res, jac = Integral.multieval(residual, jacobian0+jacobiant/thistimestep, fcache=fcache, arguments=collections.ChainMap(arguments or {}, {target: lhs}))
    resnorm = numpy.linalg.norm(res[~constrain])


def thetamethod(target, residual, inertia, timestep, lhs0, theta, target0='_thetamethod_target0', constrain=None, newtontol=1e-10, *, arguments=None, **newtonargs):
  '''solve time dependent problem using the theta method

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  residual : Integral
  inertia : Integral
  timestep : float
      Initial time step, will scale up as residual decreases
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  theta : float
      Theta value (theta=1 for implicit Euler, theta=0.5 for Crank-Nicolson)
  residual0 : Integral
      Optional additional residual component evaluated in previous timestep
  constrain : boolean or float vector
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  newtontol : float
      Residual tolerance of individual timesteps
  arguments : :class:`collections.abc.Mapping`
      Defines the values for :class:`nutils.function.Argument` objects in
      `residual`.  The ``target`` should not be present in ``arguments``.
      Optional.

  Yields
  ------
  vector
      Coefficient vector for all timesteps after the initial condition.
  '''

  assert target != target0, '`target` should not be equal to `target0`'
  assert target not in (arguments or {}), '`target` should not be defined in `arguments`'
  assert target0 not in (arguments or {}), '`target0` should not be defined in `arguments`'
  lhs = lhs0
  res0 = residual * theta + inertia / timestep
  res1 = residual * (1-theta) - inertia / timestep
  res = res0 + res1.replace({target: function.Argument(target0, lhs.shape)})
  jac = res.derivative(target)
  while True:
    yield lhs
    lhs = newton(target, residual=res, jacobian=jac, lhs0=lhs, constrain=constrain, arguments=collections.ChainMap(arguments or {}, {target0: lhs}), **newtonargs).solve(tol=newtontol)


impliciteuler = functools.partial(thetamethod, theta=1)
cranknicolson = functools.partial(thetamethod, theta=0.5)


@log.title
def optimize(target, functional, droptol=None, lhs0=None, constrain=None, newtontol=None, *, arguments=None):
  '''find the minimizer of a given functional

  Parameters
  ----------
  target : :class:`str`
      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
  functional : scalar Integral
      The functional the should be minimized by varying target
  droptol : :class:`float`
      Threshold for leaving entries in the return value at NaN if they do not
      contribute to the value of the functional.
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure (if
      applicable).
  constrain : boolean or float vector
      Equal length to ``lhs0``, masks the free vector entries as ``False``
      (boolean) or NaN (float). In the remaining positions the values of
      ``lhs0`` are returned unchanged (boolean) or overruled by the values in
      `constrain` (float).
  newtontol : float
      Residual tolerance of Newton procedure (if applicable)

  Yields
  ------
  vector
      Coefficient vector corresponding to the functional optimum
  '''

  assert target not in (arguments or {}), '`target` should not be defined in `arguments`'
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
  f0, res, jac = Integral.multieval(functional, residual, jacobian, arguments=collections.ChainMap(arguments or {}, {target: lhs0}))
  usedofs = ~constrain
  if droptol is not None:
    usedofs &= jac.rowsupp(droptol)
  log.info('optimizing for {}/{} degrees of freedom'.format(usedofs.sum(), len(usedofs)))
  cons = numpy.zeros(residual.shape)
  cons[usedofs] = numpy.nan
  lhs = lhs0 - jac.solve(res, constrain=cons) # residual(lhs0) + jacobian(lhs0) dlhs = 0
  if not jacobian.contains(target): # linear: functional(lhs0+dlhs) = functional(lhs0) + residual(lhs0) dlhs + .5 dlhs jacobian(lhs0) dlhs
    value = f0 + .5 * res.dot(lhs-lhs0)
  else: # nonlinear
    assert newtontol is not None, 'newton tolerance `newtontol` must be specified for nonlinear problems'
    lhs = newton(target, residual, lhs0=lhs, constrain=~usedofs, arguments=arguments).solve(newtontol)
    value = functional.eval(arguments=collections.ChainMap(arguments or {}, {target: lhs}))
  assert not numpy.isnan(lhs[usedofs]).any(), 'optimization failed (forgot droptol?)'
  log.info('optimum: {:.2e}'.format(value))
  lhs[~(usedofs|constrain)] = numpy.nan
  return lhs
