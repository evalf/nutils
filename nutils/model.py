from . import function, index, cache, log, util, numeric
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


def pseudotime( lhs0, isdof, eval_res_jac, timestep, tol=1e-10 ):
  zcons = numpy.zeros( lhs0.shape )
  zcons[isdof] = numpy.nan
  lhs = lhs0.copy()
  with log.context( 'pseudotime' ):
    b, A = eval_res_jac( lhs, timestep )
    resnorm = resnorm0 = numpy.linalg.norm( b[isdof] )
    for istep in itertools.count():
      with log.context( 'iter {0} ({1:.0f}%)'.format( istep, 100 * numpy.log(resnorm0/resnorm) / numpy.log(resnorm0/tol) ) ):
        thistimestep = timestep * (resnorm0/resnorm)
        log.info( 'residual: {:.2e} (time step {:.0e})'.format(resnorm,thistimestep) )
        yield lhs
        if resnorm < tol:
          break
        lhs -= A.solve( b, constrain=zcons )
        b, A = eval_res_jac( lhs, timestep * (resnorm0/resnorm) )
        resnorm = numpy.linalg.norm( b[isdof] )


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
    edit = function.replace( target, replacement )
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

  Models can be combined using the or-operator.
  '''

  def __init__( self, ndofs ):
    if numeric.isint( ndofs ):
      self.ndofs = int(ndofs)
      self.constraints = util.NanVec( self.ndofs )
    else:
      assert isinstance( ndofs, numpy.ndarray )
      self.ndofs = len(ndofs)
      self.constraints = ndofs.copy().view( util.NanVec )

  def __or__( self, other ):
    return MultiModel( self, other )

  def namespace( self, coeffs ):
    raise NotImplementedError( 'Model subclass needs to implement namespace' )

  def residual( self, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement residual' )

  def residual0( self, namespace ):
    return Integral.empty( [self.ndofs] )

  def inertia( self, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement inertia' )

  def initial( self, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement inertia' )

  def get_initial_condition( self ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    try:
      init = self.initial( ns )
    except NotImplementedError:
      lhs0 = self.constraints|0
    else:
      with log.context( 'initial condition' ):
        initjac = init.derivative( target )
        assert not initjac.contains( target )
        b, A = Integral.multieval( init.replace(target,function.zeros_like(target)), initjac )
        lhs0 = A.solve( -b, constrain=self.constraints )
    return lhs0

  @log.title
  def solve( self, lhs0=None, **newtonargs ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    res = self.residual( ns ) + self.residual0( ns )
    jac = res.derivative( target )
    if not jac.contains( target ):
      log.user( 'problem is linear' )
      b, A = Integral.multieval( res.replace(target,function.zeros_like(target)), jac )
      return A.solve( -b, constrain=self.constraints )
    if lhs0 is None:
      lhs0 = self.get_initial_condition()
    fcache = cache.WrapperCache()
    eval_res_jac = lambda lhs: Integral.multieval( res.replace(target,lhs), jac.replace(target,lhs), fcache=fcache )
    return newton( lhs0, numpy.isnan(self.constraints), eval_res_jac, **newtonargs )

  def solve_namespace( self, *args, **kwargs ):
    coeffs = self.solve( *args, **kwargs )
    return self.namespace( coeffs )

  def timestep( self, timestep, lhs0=None, **newtonargs ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    coeffs = lhs0 if lhs0 is not None else self.get_initial_condition()
    res = self.residual( ns ) + self.inertia( ns ) / timestep
    jac = res.derivative( target )
    fcache = cache.WrapperCache()
    islinear = not jac.contains( target )
    if islinear:
      log.user( 'problem is linear' )
      b, A = Integral.multieval( res.replace(target,function.zeros_like(target)), jac, fcache=fcache )
    while True:
      yield coeffs
      ns0 = self.namespace( coeffs )
      res0 = self.residual0( ns0 ) - self.inertia( ns0 ) / timestep
      if islinear:
        coeffs = -A.solve( b + res0.eval(fcache), constrain=self.constraints )
      else:
        eval_res_jac = lambda lhs: Integral.multieval( res0 + res.replace(target,lhs), jac.replace(target,lhs), fcache=fcache )
        coeffs = newton( coeffs, numpy.isnan(self.constraints), eval_res_jac, **newtonargs )

  def timestep_namespace( self, *args, **kwargs ):
    return ( self.namespace( coeffs ) for coeffs in self.timestep( *args, **kwargs ) )

  def pseudo_timestep( self, timestep, lhs0=None, tol=1e-10 ):
    target = function.DerivativeTarget( [self.ndofs] )
    ns = self.namespace( target )
    res = self.residual0( ns ) + self.residual( ns )
    jac0 = self.residual( ns ).derivative( target )
    jact = self.inertia( ns ).derivative( target )
    if lhs0 is None:
      lhs0 = self.get_initial_condition()
    fcache = cache.WrapperCache()
    eval_res_jac = lambda lhs, dt: Integral.multieval( res.replace(target,lhs), (jac0+jact/dt).replace(target,lhs), fcache=fcache )
    yield from pseudotime( lhs0, isdof=numpy.isnan(self.constraints), eval_res_jac=eval_res_jac, timestep=timestep, tol=tol )

  def pseudo_timestep_namespace( self, *args, **kwargs ):
    return ( self.namespace( coeffs ) for coeffs in self.pseudo_timestep( *args, **kwargs ) )


class MultiModel( Model ):
  '''Two models combined'''

  def __init__( self, m1, m2 ):
    self.models = m1, m2
    Model.__init__( self, numpy.concatenate([ m1.constraints, m2.constraints ]) )

  def namespace( self, coeffs ):
    assert len(coeffs) == self.ndofs
    m1, m2 = self.models
    return AttrDict( m1.namespace( coeffs[:m1.ndofs] ), m2.namespace( coeffs[m1.ndofs:] ) )

  def initial( self, namespace ):
    return Integral.concatenate([ m.initial(namespace) for m in self.models ])

  def residual0( self, namespace ):
    return Integral.concatenate([ m.residual0(namespace) for m in self.models ])

  def residual( self, namespace ):
    return Integral.concatenate([ m.residual(namespace) for m in self.models ])

  def inertia( self, namespace ):
    return Integral.concatenate([ m.inertia(namespace) for m in self.models ])

  def get_initial_condition( self ):
    return numpy.concatenate([ m.get_initial_condition() for m in self.models ])

if __name__ == '__main__':

  class Laplace( Model ):
    def __init__( self, domain, geom ):
      self.domain = domain
      self.geom = geom
      self.basis = self.domain.basis( 'std', degree=1 )
      cons = domain.boundary['left'].project( 0, onto=self.basis, geometry=geom, ischeme='gauss2' )
      super().__init__( cons )
    def namespace( self, coeffs ):
      return AttrDict( u=self.basis.dot(coeffs) )
    def residual( self, ns ):
      return Integral( ( self.basis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=self.domain, geometry=geom, degree=2 ) \
           + Integral( self.basis, domain=self.domain.boundary['top'], geometry=geom, degree=2 )

  from nutils import mesh, plot
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace( domain, geom )
  ns = model.solve_namespace()
  geom_, u_ = domain.elem_eval( [ geom, ns.u ], ischeme='bezier2' )
  with plot.PyPlot( 'model_demo', ndigits=0 ) as plot:
    plot.mesh( geom_, u_ )
