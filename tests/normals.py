from nutils import *
from . import register, unittest

@register( '2d', 2, False )
@register( '2dcurved', 2, True )
@register( '3d', 3, False )
def check( ndims, curved ):

  if not curved:
    domain, geom = mesh.rectilinear( [[1,1.5,2],[-1,0],[0,2,4]][:ndims] )
    curv = 0
  else:
    assert ndims == 2
    nodes = numpy.linspace( -.25*numpy.pi, .25*numpy.pi, 3 )
    domain, (xi,eta) = mesh.rectilinear([ nodes, nodes ])
    geom = numpy.sqrt(2) * function.stack([ function.sin(xi) * function.cos(eta), function.cos(xi) * function.sin(eta) ])
    curv = 1

  @unittest
  def zero():
    zero = domain.boundary.integrate( geom.normal(), geometry=geom, ischeme='gauss9' )
    numpy.testing.assert_almost_equal( zero, 0 )

  @unittest
  def volume():
    volume = domain.integrate( 1, geometry=geom, ischeme='gauss9' )
    volumes = domain.boundary.integrate( geom * geom.normal(), geometry=geom, ischeme='gauss9' )
    numpy.testing.assert_almost_equal( volume, volumes )

  @unittest
  def interfaces():
    funcsp = domain.discontfunc( degree=2 )
    f = ( funcsp[:,_] * numpy.arange(funcsp.shape[0]*ndims).reshape(-1,ndims) ).sum(0)
    g = funcsp.dot( numpy.arange(funcsp.shape[0]) )

    fg1 = domain.integrate( ( f * g.grad(geom) ).sum(-1), geometry=geom, ischeme='gauss9' )
    fg2 = domain.boundary.integrate( (f*g).dotnorm(geom), geometry=geom, ischeme='gauss9' ) \
        - domain.interfaces.integrate( function.jump(f*g).dotnorm(geom), geometry=geom, ischeme='gauss9' ) \
        - domain.integrate( f.div(geom) * g, geometry=geom, ischeme='gauss9' )

    numpy.testing.assert_almost_equal( fg1, fg2 )

  @unittest
  def curvature():
    c = domain.boundary.elem_eval( geom.curvature(), ischeme='uniform1', separate=False )
    numpy.testing.assert_almost_equal( c, curv )

  if curved:
    return

  @unittest
  def boundaries():
    normal = geom.normal()
    boundary = domain.boundary
    for name, n in zip( ['right','top','back'][:ndims], numpy.eye(ndims) ):
      numpy.testing.assert_almost_equal( boundary[name].elem_eval( normal, ischeme='gauss9', separate=False )-n, 0 )
    for name, n in zip( ['left','bottom','front'][:ndims], -numpy.eye(ndims) ):
      numpy.testing.assert_almost_equal( boundary[name].elem_eval( normal, ischeme='gauss9', separate=False )-n, 0 )
