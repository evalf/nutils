#!/usr/bin/env python

from nutils import *
import numpy

class FiniteCellTestBase( object ):

  def __init__ ( self ):
    domain, self.geom = mesh.rectilinear( (numpy.linspace(-1.,1.,self.nelems+1),)*self.ndims )
    self.fdomain = domain.trim( levelset=self.levelset, maxrefine=self.maxrefine, finestscheme=self.finestscheme )

  def test_volume ( self ):
    vol = self.fdomain.integrate( 1., geometry=self.geom, ischeme='gauss1' )
    log.info( 'Volume =', vol, '(%5.4f)' % self.vol_exact )
    numpy.testing.assert_almost_equal( vol, self.vol_exact, decimal=self.vol_decimal )

### Temporarily disabled:
#
#   topo = topology.UnstructuredTopology( self.fdomain.get_trimmededges( self.maxrefine ), ndims=self.ndims-1 )
#   vol_gauss = (1./float(self.ndims))*topo.integrate( sum(self.geom*self.geom.normal()), geometry=self.geom, ischeme='gauss1' )
#   log.info( 'Volume (Gauss)=', vol_gauss, '(%5.4f)' % vol )
#
#   numpy.testing.assert_almost_equal( vol_gauss, vol, decimal=14 )
#
# def test_surfacearea ( self ):
#   topo = topology.UnstructuredTopology( self.fdomain.get_trimmededges( self.maxrefine ), ndims=self.ndims-1 )
#   surf = topo.integrate( 1., geometry=self.geom, ischeme='gauss1' )
#   log.info( 'Surface area =', surf, '(%5.4f)' % self.surf_exact )
#
#   if __name__ == '__main__':
#     plot.writevtu( 'surface.vtu', topo, self.geom )
#
#   numpy.testing.assert_almost_equal( surf, self.surf_exact, decimal=self.surf_decimal )

class TestCircle( FiniteCellTestBase ):

  ndims  = 2
  nelems = 5
  
  @property
  def levelset ( self ):
    return -sum(self.geom*self.geom)+1./numpy.pi

  maxrefine    = 4
  finestscheme = 'simplex1'

  vol_decimal = 3
  vol_exact   = 1.

  surf_exact   = 2.*numpy.sqrt(numpy.pi)
  surf_decimal = 3

class TestSphere( FiniteCellTestBase ):

  ndims  = 3
  nelems = 7
  
  r2 = numpy.power(3./(4.*numpy.pi),2./3.)

  @property
  def levelset ( self ):
    return -sum(self.geom*self.geom)+self.r2

  maxrefine    = 2
  finestscheme = 'simplex1'

  vol_decimal = 2
  vol_exact   = 1.

  surf_exact   = 4.8091571139 #4.*numpy.pi*r2
  surf_decimal = 10
  

def two_D ():
  test_obj = TestCircle()
  test_obj.test_volume()
  test_obj.test_surfacearea()

def three_D ():
  test_obj = TestSphere()
  test_obj.test_volume()
  test_obj.test_surfacearea()

if __name__ == '__main__':
  util.run( two_D, three_D )
