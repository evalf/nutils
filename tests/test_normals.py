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
