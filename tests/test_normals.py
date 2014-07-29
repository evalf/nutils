from nutils import *
import numpy

class NormalTest( object ):

  def __init__( self, domain, geom ):
    self.domain = domain
    self.geom = geom

  def test_zero( self ):
    zero = self.domain.boundary.integrate( self.geom.normal(), geometry=self.geom, ischeme='gauss2' )
    numpy.testing.assert_almost_equal( zero, 0 )

  def test_volume( self ):
    volume = self.domain.integrate( 1, geometry=self.geom, ischeme='gauss2' )
    volumes = self.domain.boundary.integrate( self.geom * self.geom.normal(), geometry=self.geom, ischeme='gauss2' )
    numpy.testing.assert_almost_equal( volume, volumes )

  def test_interfaces( self ):
    funcsp = self.domain.discontfunc( degree=2 )
    f = ( funcsp[:,_] * numpy.arange(funcsp.shape[0]*self.domain.ndims).reshape(-1,self.domain.ndims) ).sum(0)
    g = funcsp.dot( numpy.arange(funcsp.shape[0]) )

    fg1 = self.domain.integrate( ( f * g.grad(self.geom) ).sum(), geometry=self.geom, ischeme='gauss2' )
    fg2 = self.domain.boundary.integrate( (f*g).dotnorm(self.geom), geometry=self.geom, ischeme='gauss2' ) \
        + self.domain.interfaces.integrate( function.jump(f*g).dotnorm(self.geom), geometry=self.geom, ischeme='gauss2' ) \
        - self.domain.integrate( f.div(self.geom) * g, geometry=self.geom, ischeme='gauss2' )

    numpy.testing.assert_almost_equal( fg1, fg2 )


class Test2D( NormalTest ):

  def __init__( self ):
    domain, geom = mesh.rectilinear( [[1,1.5,2],[-1,0]] )
    NormalTest.__init__( self, domain, geom )

class Test2DCurve( NormalTest ):

  def __init__( self ):
    domain, (x,y) = mesh.rectilinear( [[1,1.5,2],[-1,0]] )
    geom = function.stack([ x**2-y, x-y**2 ])
    NormalTest.__init__( self, domain, geom )

class Test3D( NormalTest ):

  def __init__( self ):
    domain, geom = mesh.rectilinear( [[1,1.5,2],[-1,0],[0,2,4]] )
    NormalTest.__init__( self, domain, geom )
