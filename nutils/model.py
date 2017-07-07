# -*- coding: utf8 -*-
#
# Module MODEL
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The model module defines the :class:`Integral` class, which represents an
unevaluated integral. This is useful for fully automated solution procedures
such as Newton, that require functional derivatives of an entire functional.

To demonstrate this consider the following setup:

>>> domain, geom = mesh.rectilinear( [4,4] )
>>> basis = domain.basis( 'spline', degree=2 )
>>> cons = domain.boundary['left,top'].project( 0, onto=basis, geometry=geom, ischeme='gauss4' )
>>> target = function.DerivativeTarget( [len(basis)] )
>>> u = basis.dot( target )

Function ``u`` represents an element from the discrete space but cannot not
evaluated yet as we did not yet establish values for ``target``. It can,
however, be used to construct a residual functional ``res``. Aiming to solve
the Poisson problem u_,kk = f we define the residual functional res = v,k u,k +
v f and solve for res == 0 using ``solve_linear``:

>>> res = model.Integral( basis['n,i']*u[',i']+basis['n'], domain=domain, geometry=geom, degree=2 )
>>> lhs = model.solve_linear( target, residual=res, constrain=cons )
>>> u = basis.dot( lhs )

The new function ``u`` represents the solution to the Poisson problem.

In addition to ``solve_linear`` the model module defines ``newton`` and
``pseudotime`` for solving nonlinear problems, as well as ``impliciteuler`` for
time dependent problems.
"""

from . import function, cache, log, util, numeric
import numpy, itertools, functools, numbers


class Integral( dict ):
  '''Postponed integral, used for derivative purposes'''

  def __init__( self, integrand, domain, geometry, degree, edit=None ):
    integrand *= function.J( geometry, domain.ndims )
    if edit is not None:
      integrand = edit( integrand )
    self[ cache.HashableAny(domain) ] = integrand, degree
    self.shape = integrand.shape

  @classmethod
  def empty( self, shape ):
    empty = dict.__new__( Integral )
    empty.shape = tuple(shape)
    return empty

  @classmethod
  def concatenate( cls, integrals ):
    assert all( integral.shape[1:] == integrals[0].shape[1:] for integral in integrals[1:] )
    concatenate = cls.empty( ( sum( integral.shape[0] for integral in integrals ), ) + integrals[0].shape[1:] )
    for domain in set.union( *[ set(integral) for integral in integrals ] ):
      integrands, degrees = zip( *[ integral.get(domain,(function.zeros(integral.shape),0)) for integral in integrals ] )
      concatenate[domain] = function.concatenate( integrands, axis=0 ), max(degrees)
    return concatenate

  @classmethod
  def multieval( cls, *integrals, fcache=None ):
    if fcache is None:
      fcache = cache.WrapperCache()
    assert all( isinstance( integral, cls ) for integral in integrals )
    domains = set( domain for integral in integrals for domain in integral )
    retvals = []
    for i, domain in enumerate(domains):
      integrands, degrees = zip( *[ integral.get( domain, (function.zeros(integral.shape),0) ) for integral in integrals ] )
      retvals.append( domain.obj.integrate( integrands, ischeme='gauss{}'.format(max(degrees)), fcache=fcache ) )
    return numpy.sum( retvals, axis=0 )

  def eval( self, fcache=None ):
    if fcache is None:
      fcache = cache.WrapperCache()
    values = [ domain.obj.integrate( integrand, ischeme='gauss{}'.format(degree), fcache=fcache ) for domain, (integrand,degree) in self.items() ]
    return numpy.sum( values, axis=0 )

  def derivative(self, target):
    assert target.ndim == 1
    seen = {}
    derivative = self.empty(self.shape+target.shape)
    for domain, (integrand, degree) in self.items():
      derivative[domain] = function.derivative(integrand, var=target, seen=seen), degree
    return derivative

  def replace( self, target, replacement ):
    edit = functools.partial( function.replace, target, replacement )
    replace = self.empty( self.shape )
    for domain, (integrand,degree) in self.items():
      replace[domain] = edit(integrand), degree
    return replace

  def contains( self, target ):
    return any( target in integrand.serialized[0] for integrand, degree in self.values() )

  def __add__( self, other ):
    assert isinstance( other, Integral ) and self.shape == other.shape
    add = self.empty( self.shape )
    add.update( self )
    for domain, integrand_degree in other.items():
      try:
        integrand, degree = add.pop(domain)
      except KeyError:
        add[domain] = integrand_degree
      else:
        add[domain] = integrand_degree[0] + integrand, max( integrand_degree[1], degree )
    return add

  def __neg__( self ):
    return self * -1

  def __sub__( self, other ):
    return self + (-other)

  def __mul__( self, other ):
    if not isinstance( other, numbers.Number ):
      return NotImplemented
    mul = self.empty( self.shape )
    mul.update({ domain: (integrand*other,degree) for domain, (integrand,degree) in self.items() })
    return mul

  def __rmul__( self, other ):
    if not isinstance( other, numbers.Number ):
      return NotImplemented
    return self * other

  def __truediv__( self, other ):
    if not isinstance( other, numbers.Number ):
      return NotImplemented
    return self * (1/other)


class ModelError( Exception ): pass


def solve_linear( target, residual, constrain ):
  '''solve linear problem

  Parameters
  ----------
  target : DerivativeTarget
      Representation of coefficient vector
  residual : Integral
      Residual integral, depends on ``target``
  constrain : util.NanVec
      Defines the fixed entries of the coefficient vector

  Returns
  -------
  vector
      Array of ``target`` values for which ``residual == 0``'''

  jacobian = residual.derivative( target )
  if jacobian.contains( target ):
    raise ModelError( 'problem is not linear' )
  res, jac = Integral.multieval( residual.replace(target,function.zeros_like(target)), jacobian )
  return jac.solve( -res, constrain=constrain )


def solve( gen_lhs_resnorm, tol=1e-10, maxiter=numpy.inf ):
  '''execute nonlinear solver
  
  Iterates over nonlinear solver until tolerance is reached. Example:

  >>> lhs = solve( newton( target, residual ), tol=1e-5 )
  
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
        raise ModelError( 'tolerance not reached in {} iterations'.format(maxiter) )
      with log.context( 'iter {0} ({1:.0f}%)'.format( inewton, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol) ) ):
        log.info( 'residual: {:.2e}'.format(resnorm) )
        lhs, resnorm = next(gen_lhs_resnorm)
      inewton += 1
  except StopIteration:
    raise ModelError( 'generator stopped before reaching target tolerance' )
  else:
    log.info( 'tolerance reached in {} iterations with residual {:.2e}'.format(inewton, resnorm) )
    return lhs


def withsolve( f ):
  '''add a .solve method to (lhs,resnorm) iterators

  Introduces the convenient form:

  >>> newton( target, residual ).solve( tol )

  Shorthand for

  >>> solve( newton( target, residual ), tol )
  '''

  @functools.wraps( f, updated=() )
  class wrapper:
    def __init__( self, *args, **kwargs ):
      self.iter = f( *args, **kwargs )
    def __next__( self ):
      return next( self.iter )
    def __iter__( self ):
      return self.iter
    def solve( self, *args, **kwargs ):
      return solve( self.iter, *args, **kwargs )
  return wrapper


@withsolve
def newton( target, residual, lhs0=None, freezedofs=None, nrelax=numpy.inf, minrelax=.1, maxrelax=.9, rebound=2**.5 ):
  '''iteratively solve nonlinear problem by gradient descent

  Generates targets such that residual approaches 0 using Newton procedure with
  line search based on a residual integral. Suitable to be used inside
  ``solve``.

  An optimal relaxation value is computed based on the following cubic
  assumption:

      | res( lhs + r * dlhs ) |^2 = A + B * r + C * r^2 + D * r^3

  where A, B, C and D are determined based on the current and updated residual
  and tangent.

  Parameters
  ----------
  target : DerivativeTarget
  residual : Integral
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  freezedofs : boolean vector
      Equal length to lhs0, masks the non-free vector entries as True. In the
      positions where ``freezedofs`` is True the values of ``lhs0`` are returned
      unchanged.
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

  Yields
  ------
  vector
      Coefficient vector that approximates residual==0 with increasing accuracy
  '''

  if freezedofs is None:
    freezedofs = numpy.zeros( len(target), dtype=bool )

  if lhs0 is None:
    lhs0 = numpy.zeros( len(target) )

  jacobian = residual.derivative( target )
  if not jacobian.contains( target ):
    log.info( 'problem is linear' )
    res, jac = Integral.multieval( residual.replace(target,function.zeros_like(target)), jacobian )
    cons = lhs0.copy()
    cons[~freezedofs] = numpy.nan
    lhs = jac.solve( -res, constrain=cons )
    yield lhs, 0
    return

  lhs = lhs0.copy()
  fcache = cache.WrapperCache()
  res, jac = Integral.multieval( residual.replace(target,lhs), jacobian.replace(target,lhs), fcache=fcache )
  zcons = numpy.zeros( len(target) )
  zcons[~freezedofs] = numpy.nan
  relax = 1
  while True:
    resnorm = numpy.linalg.norm( res[~freezedofs] )
    yield lhs, resnorm
    dlhs = -jac.solve( res, constrain=zcons )
    relax = min( relax * rebound, 1 )
    for irelax in itertools.count():
      res, jac = Integral.multieval( residual.replace(target,lhs+relax*dlhs), jacobian.replace(target,lhs+relax*dlhs), fcache=fcache )
      newresnorm = numpy.linalg.norm( res[~freezedofs] )
      if irelax >= nrelax:
        if newresnorm > resnorm:
          log.warning( 'failed to decrease residual' )
          return
        break
      if not numpy.isfinite( newresnorm ):
        log.info( 'failed to evaluate residual ({})'.format( newresnorm ) )
        newrelax = 0 # replaced by minrelax later
      else:
        r0 = resnorm**2
        d0 = -2 * r0
        r1 = newresnorm**2
        d1 = 2 * numpy.dot( jac.matvec(dlhs)[~freezedofs], res[~freezedofs] )
        log.info( 'line search: 0[{}]{} {}creased by {:.0f}%'.format( '---+++' if d1 > 0 else '--++--' if r1 > r0 else '------', round(relax,5), 'in' if newresnorm > resnorm else 'de', 100*abs(newresnorm/resnorm-1) ) )
        if r1 <= r0 and d1 <= 0:
          break
        D = 2*r0 - 2*r1 + d0 + d1
        if D > 0:
          C = 3*r1 - 3*r0 - 2*d0 - d1
          newrelax = ( numpy.sqrt(C**2-3*d0*D) - C ) / (3*D)
          log.info( 'minimum based on 3rd order estimation: {:.3f}'.format(newrelax) )
        else:
          C = r1 - r0 - d0
          # r1 > r0 => C > 0
          # d1 > 0  => C = r1 - r0 - d0/2 - d0/2 > r1 - r0 - d0/2 - d1/2 = -D/2 > 0
          newrelax = -.5 * d0 / C
          log.info( 'minimum based on 2nd order estimation: {:.3f}'.format(newrelax) )
        if newrelax > maxrelax:
          break
      relax *= max( newrelax, minrelax )
    lhs += relax * dlhs


@withsolve
def pseudotime( target, residual, inertia, timestep, lhs0, residual0=None, freezedofs=None ):
  '''iteratively solve nonlinear problem by pseudo time stepping

  Generates targets such that residual approaches 0 using hybrid of Newton and
  time stepping. Requires an inertia term and initial timestep. Suitable to be
  used inside ``solve``.

  Parameters
  ----------
  target : DerivativeTarget
  residual : Integral
  inertia : Integral
  timestep : float
      Initial time step, will scale up as residual decreases
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  freezedofs : boolean vector
      Equal length to lhs0, masks the non-free vector entries as True. In the
      positions where ``freezedofs`` is True the values of ``lhs0`` are returned
      unchanged.

  Yields
  ------
  vector, float
      Tuple of coefficient vector and residual norm
  '''

  jacobian0 = residual.derivative( target )
  jacobiant = inertia.derivative( target )
  if residual0 is not None:
    residual += residual0
  if freezedofs is None:
    freezedofs = numpy.zeros( len(target), dtype=bool )
  zcons = util.NanVec( len(target) )
  zcons[freezedofs] = 0
  lhs = lhs0.copy()
  fcache = cache.WrapperCache()
  res, jac = Integral.multieval( residual.replace(target,lhs), (jacobian0+jacobiant/timestep).replace(target,lhs), fcache=fcache )
  resnorm = resnorm0 = numpy.linalg.norm( res[~freezedofs] )
  while True:
    yield lhs, resnorm
    lhs -= jac.solve( res, constrain=zcons )
    thistimestep = timestep * (resnorm0/resnorm)
    log.info( 'timestep: {:.0e}'.format(thistimestep) )
    res, jac = Integral.multieval( residual.replace(target,lhs), (jacobian0+jacobiant/thistimestep).replace(target,lhs), fcache=fcache )
    resnorm = numpy.linalg.norm( res[~freezedofs] )


def impliciteuler( target, residual, inertia, timestep, lhs0, residual0=None, freezedofs=None, tol=1e-10, **newtonargs ):
  '''solve time dependent problem using implicit euler time stepping

  Parameters
  ----------
  target : DerivativeTarget
  residual : Integral
  inertia : Integral
  timestep : float
      Initial time step, will scale up as residual decreases
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  residual0 : Integral
      Optional additional residual component evaluated in previous timestep
  freezedofs : boolean vector
      Equal length to lhs0, masks the non-free vector entries as True. In the
      positions where ``freezedofs`` is True the values of ``lhs0`` are returned
      unchanged.
  tol : float
      Residual tolerance of individual timesteps

  Yields
  ------
  vector
      Coefficient vector for all timesteps after the initial condition.
  '''

  res = residual + inertia / timestep
  res0 = -inertia / timestep
  if residual0 is not None:
    res0 += residual0
  lhs = lhs0
  while True:
    yield lhs
    lhs = solve( newton( target, residual=res0.replace(target,lhs) + res, lhs0=lhs, freezedofs=freezedofs, **newtonargs ), tol=tol )
