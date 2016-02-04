from nutils import *
from . import register, unittest

@register( '2d', 2, False )
@register( '2dcurved', 2, True )
@register( '3d', 3, False )
def check( ndims, curved ):

  domain, geom = mesh.rectilinear( [[1,1.5,2],[-1,0],[0,2,4]][:ndims] )
  if curved:
    x, y = geom
    geom = function.stack([ x**2-y, x-y**2 ])

  @unittest
  def zero():
    zero = domain.boundary.integrate( geom.normal(), geometry=geom, ischeme='gauss2' )
    numpy.testing.assert_almost_equal( zero, 0 )

  @unittest
  def volume():
    volume = domain.integrate( 1, geometry=geom, ischeme='gauss2' )
    volumes = domain.boundary.integrate( geom * geom.normal(), geometry=geom, ischeme='gauss2' )
    numpy.testing.assert_almost_equal( volume, volumes )

  @unittest
  def interfaces():
    funcsp = domain.discontfunc( degree=2 )
    f = ( funcsp[:,_] * numpy.arange(funcsp.shape[0]*ndims).reshape(-1,ndims) ).sum(0)
    g = funcsp.dot( numpy.arange(funcsp.shape[0]) )

    fg1 = domain.integrate( ( f * g.grad(geom) ).sum(), geometry=geom, ischeme='gauss2' )
    fg2 = domain.boundary.integrate( (f*g).dotnorm(geom), geometry=geom, ischeme='gauss2' ) \
        - domain.interfaces.integrate( function.jump(f*g).dotnorm(geom), geometry=geom, ischeme='gauss2' ) \
        - domain.integrate( f.div(geom) * g, geometry=geom, ischeme='gauss2' )

    numpy.testing.assert_almost_equal( fg1, fg2 )

  if curved:
    return

  @unittest
  def boundaries():
    normal = geom.normal()
    boundary = domain.boundary
    for name, n in zip( ['right','top','back'][:ndims], numpy.eye(ndims) ):
      numpy.testing.assert_almost_equal( boundary[name].elem_eval( normal, ischeme='gauss1', separate=False )-n, 0 )
    for name, n in zip( ['left','bottom','front'][:ndims], -numpy.eye(ndims) ):
      numpy.testing.assert_almost_equal( boundary[name].elem_eval( normal, ischeme='gauss1', separate=False )-n, 0 )
