from . import function, index, cache, log, util
import numpy, itertools, functools, numbers


@log.title
def newton( lhs0, isdof, eval_res_jac, tol=1e-10, nrelax=5, maxrelax=.9, callback=None ):
  '''iteratively solve nonlinear problem by gradient descent

  Newton procedure with line search based on a residual and tangent generating
  function. An optimal relaxation value is computed based on the following
  parabolic assumption:

      | res( lhs + r * dlhs ) |^2 = A + B * r + C * r^2 + D * r^3

  where A, B, C and D are determined based on the current and updated residual
  value and tangent.

  Parameters
  ----------
  lhs0 : vector
      Coefficient vector, starting point of the iterative procedure.
  isdof : boolean vector
      Equal length to lhs0, masks the free vector entries as True. In the
      positions where `isdof' is False the values of `lhs0' are returned
      unchanged.
  eval_res_jac : function
      Takes as argument a coefficient vector, and returns the corresponding
      residual vector and tangent matrix.
  tol : float
      The residual value at which iterations stop.
  nrelax : int
      Maximum number of relaxation steps before proceding with the updated
      coefficient vector.
  maxrelax : float
      Relaxation value below which relaxation continues, unless `nrelax' is
      reached; should be a value less than or equal to 1.

  Returns
  -------
  vector
      Coefficient vector that corresponds to a smaller than `tol` residual.
  '''

  zcons = numpy.zeros( lhs0.shape )
  zcons[isdof] = numpy.nan
  lhs = lhs0.copy()
  res, jac = eval_res_jac( lhs )
  newresnorm = resnorm0 = numpy.linalg.norm( res[isdof] )
  for inewton in itertools.count():
    resnorm = newresnorm
    if resnorm < tol:
      break
    with log.context( 'iter {0} ({1:.0f}%)'.format( inewton, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol) ) ):
      log.info( 'residual: {:.2e}'.format(resnorm) )
      if callback is not None:
        callback( lhs )
      dlhs = -jac.solve( res, constrain=zcons )
      relax = 1
      for irelax in itertools.count():
        res, jac = eval_res_jac( lhs + relax * dlhs )
        newresnorm = numpy.linalg.norm( res[isdof] )
        if irelax >= nrelax:
          assert newresnorm < resnorm, 'stuck in local minimum'
          break
        # endpoint values, derivatives
        r0 = resnorm**2
        d0 = -2 * relax * resnorm**2
        r1 = newresnorm**2
        d1 = 2 * relax * numpy.dot( jac.matvec(dlhs)[isdof], res[isdof] )
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

  return lhs


class AttrDict( dict ):
  '''Container for key/value pairs. Items can be get/set as either dictonary
  items or object attributes. Dictionary items cannot be accessed as attributes
  and vice versa.'''
  def __init__( self, *args, **kwargs ):
    nitems = 0
    nattrs = 0
    for arg in args:
      self.update( arg )
      nitems += len(arg)
      if isinstance( arg, AttrDict ):
        self.__dict__.update( arg.__dict__ )
        nattrs += len(arg.__dict__)
    self.__dict__.update( kwargs )
    nattrs += len(kwargs)
    assert len(self) == nitems, 'duplicate items'
    assert len(self.__dict__) == nattrs, 'duplicate attributes'


class Integral( dict ):
  '''Postponed integral, used for derivative purposes'''

  def __init__( self, integrand, domain, geometry, degree ):
    if isinstance( integrand, index.IndexedArray ):
      integrand = integrand.unwrap( geometry )
    self[ cache.HashableAny(domain) ] = integrand * function.J(geometry,domain.ndims), degree
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
    assert target.shape == replacement.shape
    edit = lambda f: replacement if f is target else function.edit( f, edit )
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

  def __sub__( self, other ):
    return self + other * -1

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


class Model:
  '''Model base class

  Model classes define a discretized physical problem by implementing two
  member functions:
    - namespace, returns a dictionary or AttrDict
    - residual, returns an Integral object
  and optionally
    - residual0, returns an Integral object
    - inertia, returns an Integral object
    - constraints, returns a util.NanVec

  Models can be combined using the or-operator.
  '''

  def __init__( self, ndofs ):
    self.ndofs = ndofs

  def __or__( self, other ):
    return MultiModel( self, other )

  def namespace( self, coeffs ):
    raise NotImplementedError( 'Model subclass needs to implement namespace' )

  def constraints( self, geom ):
    return util.NanVec( self.ndofs )

  def residual( self, geom, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement residual' )

  def residual0( self, geom, namespace ):
    return Integral.empty( [self.ndofs] )

  def inertia( self, geom, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement inertia' )

  def initial( self, geom, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement inertia' )

  def get_initial_condition( self, geom, cons=None ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    if cons is None:
      cons = self.constraints( geom )
    try:
      init = self.initial( geom, ns )
    except NotImplementedError:
      lhs0 = cons|0
    else:
      with log.context( 'initial condition' ):
        initjac = init.derivative( target )
        assert not initjac.contains( target )
        b, A = Integral.multieval( init.replace(target,function.zeros_like(target)), initjac )
        lhs0 = A.solve( -b, constrain=cons )
    return lhs0

  @log.title
  def solve( self, geom, lhs0=None, **newtonargs ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    cons = self.constraints( geom )
    res = self.residual( geom, ns ) + self.residual0( geom, ns )
    jac = res.derivative( target )
    if not jac.contains( target ):
      log.user( 'problem is linear' )
      b, A = Integral.multieval( res.replace(target,function.zeros_like(target)), jac )
      return A.solve( -b, constrain=cons )
    if lhs0 is None:
      lhs0 = self.get_initial_condition( geom, cons )
    fcache = cache.WrapperCache()
    eval_res_jac = lambda lhs: Integral.multieval( res.replace(target,lhs), jac.replace(target,lhs), fcache=fcache )
    return newton( lhs0, numpy.isnan(cons), eval_res_jac, **newtonargs )

  def solve_namespace( self, *args, **kwargs ):
    coeffs = self.solve( *args, **kwargs )
    return self.namespace( coeffs )

  @log.title
  def timestep( self, geom, timestep, lhs0=None, **newtonargs ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    cons = self.constraints( geom )
    coeffs = lhs0 if lhs0 is not None else self.get_initial_condition( geom, cons )
    res = self.residual( geom, ns ) + self.inertia( geom, ns ) / timestep
    jac = res.derivative( target )
    fcache = cache.WrapperCache()
    islinear = not jac.contains( target )
    if islinear:
      log.user( 'problem is linear' )
      b, A = Integral.multieval( res.replace(target,function.zeros_like(target)), jac, fcache=fcache )
    while True:
      yield coeffs
      ns0 = self.namespace( coeffs )
      res0 = self.residual0( geom, ns0 ) - self.inertia( geom, ns0 ) / timestep
      if islinear:
        coeffs = -A.solve( b + res0.eval(fcache), constrain=cons )
      else:
        eval_res_jac = lambda lhs: Integral.multieval( res0 + res.replace(target,lhs), jac.replace(target,lhs), fcache=fcache )
        coeffs = newton( coeffs, numpy.isnan(cons), eval_res_jac, **newtonargs )

  def timestep_namespace( self, *args, **kwargs ):
    return ( self.namespace( coeffs ) for coeffs in self.timestep( *args, **kwargs ) )

class MultiModel( Model ):
  '''Two models combined'''

  def __init__( self, m1, m2 ):
    self.models = m1, m2
    Model.__init__( self, ndofs=m1.ndofs+m2.ndofs )

  def namespace( self, coeffs ):
    assert len(coeffs) == self.ndofs
    m1, m2 = self.models
    return AttrDict( m1.namespace( coeffs[:m1.ndofs] ), m2.namespace( coeffs[m1.ndofs:] ) )

  def constraints( self, geom ):
    return numpy.concatenate([ m.constraints(geom) for m in self.models ]).view( util.NanVec )

  def initial( self, geom, namespace ):
    return Integral.concatenate([ m.initial(geom,namespace) for m in self.models ])

  def residual0( self, geom, namespace ):
    return Integral.concatenate([ m.residual0(geom,namespace) for m in self.models ])

  def residual( self, geom, namespace ):
    return Integral.concatenate([ m.residual(geom,namespace) for m in self.models ])

  def inertia( self, geom, namespace ):
    return Integral.concatenate([ m.inertia(geom,namespace) for m in self.models ])

  def get_initial_condition( self, geom, cons=None ):
    if cons is None:
      cons = self.constraints( geom )
    m1, m2 = self.models
    return numpy.concatenate([ m1.get_initial_condition( geom, cons[:m1.ndofs] ),
                               m2.get_initial_condition( geom, cons[m1.ndofs:] ) ])

if __name__ == '__main__':

  class Laplace( Model ):
    def __init__( self, domain ):
      self.domain = domain
      self.basis = self.domain.basis( 'std', degree=1 )
      Model.__init__( self, ndofs=len(self.basis) )
    def namespace( self, coeffs ):
      return AttrDict( u=self.basis.dot(coeffs) )
    def constraints( self, geom ):
      return self.domain.boundary['left'].project( 0, onto=self.basis, geometry=geom, ischeme='gauss2' )
    def residual( self, geom, ns ):
      return Integral( ( self.basis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=self.domain, geometry=geom, degree=2 ) \
           + Integral( self.basis, domain=self.domain.boundary['top'], geometry=geom, degree=2 )

  from nutils import mesh, plot
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace( domain )
  ns = model.solve_namespace( geom )
  geom_, u_ = domain.elem_eval( [ geom, ns.u ], ischeme='bezier2' )
  with plot.PyPlot( 'model_demo', ndigits=0 ) as plot:
    plot.mesh( geom_, u_ )
