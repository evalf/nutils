#!/usr/bin/env python

from nutils import *
import numpy

class FiniteCellTestBase( object ):

  def __init__ ( self, ndims, nelems, maxrefine, errtol ):
    domain, self.geom = mesh.rectilinear( (numpy.linspace(0,1,nelems+1),)*ndims )
    self.radius = numpy.sqrt( .5 )
    levelset = self.radius**2 - ( self.geom**2 ).sum()
    self.trimdomain, complement = domain.trim( levelset=levelset, maxrefine=maxrefine, check=False )
    V = 1.
    Vprev = 1. / (numpy.pi*self.radius)
    for idim in range( ndims ):
      S = Vprev * (2*numpy.pi*self.radius)
      Vprev = V
      V = S * (self.radius/(idim+1))
    self.volume = V / 2**ndims
    self.trimsurface = S / 2**ndims
    self.totalsurface = self.trimsurface + Vprev / (2**(ndims-1)) * ndims
    self.errtol = errtol

  def all( self ):
    self.test_volume()
    self.test_surface()

  def test_volume( self ):
    vol = self.trimdomain.volume( self.geom )
    volerr = abs( vol - self.volume ) / self.volume
    log.user( 'volume error:', volerr )
    assert volerr < self.errtol, 'volume tolerance not met: {:.2e} > {:.2e}'.format( volerr, self.errtol )

  def test_divergence( self ):
    self.trimdomain.volume_check( self.geom, decimal=15 )
 
  def test_surface( self ):
    trimsurface = self.trimdomain.boundary['trimmed'].volume( self.geom )
    trimerr = abs( trimsurface - self.trimsurface ) / self.trimsurface
    log.user( 'trim surface error:', trimerr )
    totalsurface = self.trimdomain.boundary.volume( self.geom )
    totalerr = abs( totalsurface - self.totalsurface ) / self.totalsurface
    log.user( 'total surface error:', totalerr )
    assert trimerr < self.errtol, 'trim surface tolerance not met: {:.2e} > {:.2e}'.format( trimerr, self.errtol )
    assert totalerr < self.errtol, 'total surface tolerance not met: {:.2e} > {:.2e}'.format( totalerr, self.errtol )

class TestCircle( FiniteCellTestBase ):

  def __init__( self ):
    FiniteCellTestBase.__init__( self, ndims=2, nelems=2, maxrefine=5, errtol=2.1e-4 )
  
class TestSphere( FiniteCellTestBase ):

  def __init__( self ):
    FiniteCellTestBase.__init__( self, ndims=3, nelems=2, maxrefine=3, errtol=6e-3 )
  
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


class TestSpecialCases( object ):

  def test_almost_all_positive( self ):
    domain, geom = mesh.rectilinear( [[0,.5,1]]*2 )
    x, y = geom
    domain.trim( (x-y) * (x-y+.25), maxrefine=1, check=True )

  def test_intra_elem_2d( self ):
    domain, geom = mesh.rectilinear( [[0,.5,1]]*2 )
    eps = .0001
    x, y = geom
    for maxrefine in 0, 1:
      for perturb in 0, 1, -1, x-.5, x-y:
        pos, neg = domain.trim( y-.5+eps*perturb, maxrefine=maxrefine, check=True )
        pos.volume_check( geom )
        neg.volume_check( geom )
      for perturb in 0, 1, -1, y-.5, x-y:
        pos, neg = domain.trim( x-.5+eps*perturb, maxrefine=maxrefine, check=True )
        pos.volume_check( geom )
        neg.volume_check( geom )

  def test_inter_elem_2d( self ):
    domain, geom = mesh.rectilinear( [[0,.5,1]]*2 )
    eps = .0001
    x, y = geom
    for maxrefine in 0, 1, 2:
      for perturb in 0, 1, -1, x-.5, x-y:
        pos, neg = domain.trim( y-.75+eps*perturb, maxrefine=maxrefine, check=True )
      for perturb in 0, 1, -1, y-.5, x-y:
        pos, neg = domain.trim( x-.75+eps*perturb, maxrefine=maxrefine, check=True )

  #def test_inter_elem_3d( self ):
  #  domain, geom = mesh.rectilinear( [[0,.5],[0,.5],[0,.5,1]] )
  #  eps = .0001
  #  x, y, z = geom
  #  for maxrefine in 0, 1, 2:
  #    for perturb in 0, 1, -1, x-.5, y-.5, x-y:
  #      pos, neg = domain.trim( z-.75+eps*perturb, maxrefine=maxrefine, check=True )

  def all( self ):
    #self.test_inter_elem_3d()
    #return
    self.test_inter_elem_2d()
    self.test_intra_elem_2d()
    self.test_almost_all_positive()


if __name__ == '__main__':
  def special(): return TestSpecialCases().all()
  def hierarchical(): return TestHierarchical().test_hierarchical()
  def two_D(): return TestCircle().all()
  def three_D(): return TestSphere().all()
  util.run( special, hierarchical, two_D, three_D )
