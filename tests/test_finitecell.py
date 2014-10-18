#!/usr/bin/env python

from nutils import *
import numpy

class FiniteCellTestBase( object ):

  def __init__ ( self ):
    domain, self.geom = mesh.rectilinear( (numpy.linspace(-1.,1.,self.nelems+1),)*self.ndims )
    self.fdomain, complement = domain.trim( levelset=self.levelset, maxrefine=self.maxrefine )

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
  nelems = 6
  
  @property
  def levelset ( self ):
    return -sum(self.geom*self.geom)+1./numpy.pi

  maxrefine    = 4

  vol_decimal = 3
  vol_exact   = 1.

  surf_exact   = 2.*numpy.sqrt(numpy.pi)
  surf_decimal = 3

class TestSphere( FiniteCellTestBase ):

  ndims  = 3
  nelems = 6
  
  r2 = numpy.power(3./(4.*numpy.pi),2./3.)

  @property
  def levelset ( self ):
    return -sum(self.geom*self.geom)+self.r2

  maxrefine    = 3

  vol_decimal = 2
  vol_exact   = 1.

  surf_exact   = 4.8091571139 #4.*numpy.pi*r2
  surf_decimal = 10
  

def two_D ():
  test_obj = TestCircle()
  test_obj.test_volume()
  #test_obj.test_surfacearea()

def three_D ():
  test_obj = TestSphere()
  test_obj.test_volume()
  #test_obj.test_surfacearea()

class TestHierarchical():

  def test_hierarchical( self, makeplots=False ):

    # Topologies:
    # ref0    [  .  .  .  |  .  .  .  ]
    # ref1    [  .  .  .  |  .  |  .  ]
    # ref2    [  .  .  .  |  |  |  .  ]
    # trimmed [  .  .  .  |]

    ref0, geom = mesh.rectilinear( [[0,1,2]] )
    e1, e2 = ref0
    ref1 = ref0.refined_by( [e2] )
    e1, e2, e3 = ref1
    ref2 = ref1.refined_by( [e2] )

    basis = ref2.basis( 'std', degree=1 )
    assert basis.shape == (5,)
    x, y = ref2.elem_eval( [ geom[0], basis ], ischeme='bezier2', separate=False )
    assert numpy.all( y == .25 * numpy.array(
      [[4,0,0,0,0],
       [0,4,0,0,0],
       [0,3,2,0,4],
       [0,2,4,0,0],
       [0,0,0,4,0]] )[[0,1,1,2,2,3,3,4]] )

    if makeplots:
      with plot.PyPlot( 'basis' ) as plt:
        plt.plot( x, y )

    levelset = 1.125 - geom[0]
    trimmed, complement = ref2.trim( levelset, maxrefine=3 )
    trimbasis = trimmed.basis( 'std', degree=1 )
    x, y = trimmed.simplex.elem_eval( [ geom[0], trimbasis ], ischeme='bezier2', separate=False )
    assert numpy.all( y == .125 * numpy.array(
      [[8,0,0],
       [0,8,0],
       [0,7,4]] )[[0,1,1,2]] )

    if makeplots:
      with plot.PyPlot( 'basis' ) as plt:
        plt.plot( x, y )


if __name__ == '__main__':
  util.run( TestHierarchical().test_hierarchical, two_D, three_D )
