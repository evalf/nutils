from . import function, index, cache, log, util, numeric
import numpy, itertools, functools, numbers


class Integral( dict ):
  '''Postponed integral, used for derivative purposes'''

  def __init__( self, integrand, domain, geometry, degree, edit=None ):
    if isinstance( integrand, index.IndexedArray ):
      integrand = integrand.unwrap( geometry )
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

  def derivative( self, target ):
    assert target.ndim == 1
    seen = {}
    derivative = self.empty( self.shape+target.shape )
    for domain, (integrand,degree) in self.items():
      derivative[domain] = function.derivative( integrand, var=target, axes=[0], seen=seen ), degree
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


def solve_linear( target, residual, constrain ):
  '''solve linear problem

  Parameters
  ----------
  target : DerivativeTarget
  residual : Integral
  constrain : util.NanVec

  Returns
  -------
  vector
      Solution of residual(target) == 0'''

  jacobian = residual.derivative( target )
  assert not jacobian.contains( target ), 'problem is not linear'
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
      Coefficient vector that corresponds to a smaller than `tol` residual.
  '''

  try:
    lhs, resnorm = next(gen_lhs_resnorm)
    resnorm0 = resnorm
    inewton = 0
    while resnorm > tol:
      if inewton >= maxiter:
        raise Exception( 'tolerance not reached in {} iterations'.format(maxiter) )
      with log.context( 'iter {0} ({1:.0f}%)'.format( inewton, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol) ) ):
        log.info( 'residual: {:.2e}'.format(resnorm) )
        lhs, resnorm = next(gen_lhs_resnorm)
      inewton += 1
  except StopIteration:
    raise Exception( 'generator stopped before reaching target tolerance' )
  else:
    return lhs
  

def newton( target, residual, lhs0=None, freezedofs=None, nrelax=5, maxrelax=.9 ):
  '''iteratively solve nonlinear problem by gradient descent

  Generates targets such that residual approaches 0 using Newton procedure with
  line search based on a residual integral. Suitable to be used inside `solve`.

  An optimal relaxation value is computed based on the following parabolic
  assumption:

      | res( lhs + r * dlhs ) |^2 = A + B * r + C * r^2 + D * r^3

  where A, B, C and D are determined based on the current and updated residual
  value and tangent.

  Parameters
  ----------
  target : DerivativeTarget
  residual : Integral
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  freezedofs : boolean vector
      Equal length to lhs0, masks the non-free vector entries as True. In the
      positions where `freezedofs' is True the values of `lhs0' are returned
      unchanged.
  nrelax : int
      Maximum number of relaxation steps before proceding with the updated
      coefficient vector.
  maxrelax : float
      Relaxation value below which relaxation continues, unless `nrelax' is
      reached; should be a value less than or equal to 1.

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
  while True:
    resnorm = numpy.linalg.norm( res[~freezedofs] )
    yield lhs, resnorm
    dlhs = -jac.solve( res, constrain=zcons )
    relax = 1
    for irelax in itertools.count():
      res, jac = Integral.multieval( residual.replace(target,lhs+relax*dlhs), jacobian.replace(target,lhs+relax*dlhs), fcache=fcache )
      newresnorm = numpy.linalg.norm( res[~freezedofs] )
      if irelax >= nrelax:
        assert newresnorm < resnorm, 'stuck in local minimum'
        break
      # endpoint values, derivatives
      r0 = resnorm**2
      d0 = -2 * relax * resnorm**2
      r1 = newresnorm**2
      d1 = 2 * relax * numpy.dot( jac.matvec(dlhs)[~freezedofs], res[~freezedofs] )
      # polynomial coefficients
      A = r0
      B = d0
      C = 3*r1 - 3*r0 - 2*d0 - d1
      D = d0 + d1 + 2*r0 - 2*r1
      # optimization
      discriminant = C**2 - 3*B*D
      if discriminant < 0: # monotomously decreasing
        break
      malpha = -C / (3*D)
      dalpha = numpy.sqrt(discriminant) / abs(3*D)
      newrelax = malpha + dalpha if malpha < dalpha else malpha - dalpha # smallest positive root
      if newrelax > maxrelax:
        break
      assert newrelax > 0, 'newrelax should be strictly positive, computed {!r}'.format(newrelax)
      log.info( 'relaxation {0:}: scaling by {1:.3f}'.format( irelax+1, newrelax ) )
      relax *= newrelax
    lhs += relax * dlhs


def pseudotime( target, residual, inertia, timestep, lhs0, residual0=None, freezedofs=None ):
  '''iteratively solve nonlinear problem by pseudo time stepping

  Generates targets such that residual approaches 0 using hybrid of Newton and
  time stepping. Requires an inertia term and initial timestep. Suitable to be
  used inside `solve`.

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
      positions where `freezedofs' is True the values of `lhs0' are returned
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
    freezedofs = numpy.zeros( len(target) )
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
      positions where `freezedofs' is True the values of `lhs0' are returned
      unchanged.
  tol : float
      Residual tolerance of individual timesteps

  Yields
  ------
  vector
      Coefficient vector that approximates residual==0 with increasing accuracy
  '''

  res = residual + inertia / timestep
  res0 = -inertia / timestep
  if residual0 is not None:
    res0 += residual0
  lhs = lhs0
  while True:
    yield lhs
    lhs = solve( newton( target, residual=res0.replace(target,lhs) + res, lhs0=lhs, freezedofs=freezedofs, **newtonargs ), tol=tol )
