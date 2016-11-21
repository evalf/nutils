from . import function, index, cache, log
import numpy, itertools


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


class AttrDict:
  '''Dictionary-like object which values can also be accessed as attributes'''
  def __getitem__( self, key ):
    return self.__dict__[key]
  def __setitem__( self, key, value ):
    self.__dict__[key] = value
  def __contains__( self, key ):
    return key in self.__dict__
  def __iter__( self ):
    return iter( self.__dict__ )
  def __str__( self ):
    return str( self.__dict__ )


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
    empty.shape = shape
    return empty

  @classmethod
  def concatenate( cls, integrals ):
    assert all( integral.shape[1:] == integrals[0].shape[1:] for integral in integrals[1:] )
    concatenate = cls.empty( ( sum( integral.shape[0] for integral in integrals ), ) + integrals[0].shape[1:] )
    for domain in set.union( *[ set(integral) for integral in integrals ] ):
      integrands, degrees = zip( *[ integral.get(domain,(function.zeros(integral.shape),0)) for integral in integrals ] )
      concatenate[domain] = function.concatenate( integrands, axis=0 ), max(degrees)
    return concatenate

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

  def eval( self ):
    values = [ domain.obj.integrate( integrand, ischeme='gauss{}'.format(degree) ) for domain, (integrand,degree) in self.items() ]
    return numpy.sum( values, axis=0 )

  def solve( self, target, cons, lhs0, **newtonargs ):

    seen = {}
    res_jac = [ ( domain.obj, integrand, function.derivative( integrand, var=target, axes=[0], seen=seen ), 'gauss{}'.format(degree) )
      for domain, (integrand,degree) in self.items() ]

    islinear = all( target not in jac.serialized[0] for domain, res, jac, ischeme in res_jac )
    if islinear:
      log.user( 'problem is linear' )

    fcache = cache.WrapperCache()
    def eval_res_jac( lhs ):
      lhs = function.asarray( lhs )
      edit = lambda f: lhs if f is target else function.edit( f, edit )
      values = [ domain.integrate( [res,jac], ischeme=ischeme, edit=edit, fcache=fcache ) for domain, res, jac, ischeme in res_jac ]
      return numpy.sum( values, dtype=object, axis=0 )

    if islinear:
      res, jac = eval_res_jac( function.zeros(target.shape) )
      return jac.solve( -res, constrain=cons )

    if lhs0 is None:
      lhs0 = cons|0

    return newton( lhs0, numpy.isnan(cons), eval_res_jac, **newtonargs )


class Model:
  '''Model base class

  Model classes define a discretized physical problem by implementing two
  member functions:
    - bases, yields tuples of (name,basis) pairs that define the trial space
    - evalres, returns an Integral object and constraint vector that are used
      to construct the residual and tangent matrix

  Models can be combined using the or-operator.
  '''

  def bases( self, domain ):
    raise NotImplementedError( 'Model subclass needs to implement bases' )

  def evalres( self, domain, geom, namespace ):
    raise NotImplementedError( 'Model subclass needs to implement evalres' )

  def evalres0( self, domain, geom, namespace ):
    return self.evalres( domain, geom, namespace )

  def __or__( self, other ):
    return ChainModel( self, other )

  def chained( self, domain ):
    return function.chain( [ basis for name, basis in self.bases(domain) ] )

  def namespace( self, domain, coeffs ):
    namespace = AttrDict()
    i = 0
    for name, basis in self.bases( domain ):
      j = i + len(basis)
      assert name not in namespace, 'duplicate variable name: {!r}'.format(name)
      namespace[name] = basis.dot( coeffs[i:j] )
      i = j
    assert i == len(coeffs)
    return namespace

  @log.title
  def solve( self, domain, geom, lhs0=None, **newtonargs ):
    ndofs = sum( len(basis) for name, basis in self.bases(domain) )
    target = function.DerivativeTarget( [ndofs] )
    namespace = self.namespace( domain, target )
    cons = self.constraints( domain, geom )
    res0 = self.evalres0( domain, geom, namespace )
    res = self.evalres( domain, geom, namespace )
    if lhs0 is None and res != res0:
      with log.context( 'initial condition' ):
        lhs0 = res0.solve( target, cons=cons, lhs0=None, **newtonargs )
    return res.solve( target, cons=cons, lhs0=lhs0, **newtonargs )

  def solve_namespace( self, domain, geom, **newtonargs ):
    coeffs = self.solve( domain, geom, **newtonargs )
    return self.namespace( domain, coeffs )


class ChainModel( Model ):
  '''Two models combined'''

  def __init__( self, m1, m2 ):
    self.models = m1, m2

  def bases( self, domain ):
    for m in self.models:
      yield from m.bases( domain )

  def constraints( self, domain, geom ):
    return numpy.concatenate([ m.constraints(domain,geom) for m in self.models ])

  def evalres0( self, domain, geom, namespace ):
    return Integral.concatenate([ m.evalres0(domain,geom,namespace) for m in self.models ])

  def evalres( self, domain, geom, namespace ):
    return Integral.concatenate([ m.evalres(domain,geom,namespace) for m in self.models ])


if __name__ == '__main__':

  class Laplace( Model ):
    def bases( self, domain ):
      yield 'u', domain.basis( 'std', degree=1 )
    def constraints( self, domain, geom ):
      ubasis, = self.chained( domain )
      return domain.boundary['left'].project( 0, onto=ubasis, geometry=geom, ischeme='gauss2' )
    def evalres( self, domain, geom, ns ):
      ubasis, = self.chained( domain )
      return Integral( ( ubasis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=domain, geometry=geom, degree=2 ) \
           + Integral( ubasis, domain=domain.boundary['top'], geometry=geom, degree=2 )

  from nutils import mesh, plot
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace()
  ns = model.solve_namespace( domain, geom )
  geom_, u_ = domain.elem_eval( [ geom, ns.u ], ischeme='bezier2' )
  with plot.PyPlot( 'model_demo', ndigits=0 ) as plot:
    plot.mesh( geom_, u_ )
