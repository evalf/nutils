#!/usr/bin/env python

from nutils import *
from . import register, unittest


@register( 'sphere', 3, 2, 3, 6e-3 )
@register( 'circle', 2, 2, 5, 2.1e-4 )
def cutdomain( ndims, nelems, maxrefine, errtol ):

  domain, geom = mesh.rectilinear( (numpy.linspace(0,1,nelems+1),)*ndims )
  radius = numpy.sqrt( .5 )
  levelset = radius**2 - ( geom**2 ).sum()
  pos, neg = domain.trim( levelset=levelset, maxrefine=maxrefine )
  V = 1.
  Vprev = 1. / (numpy.pi*radius)
  for idim in range( ndims ):
    S = Vprev * (2*numpy.pi*radius)
    Vprev = V
    V = S * (radius/(idim+1))
  exact_volume = V / 2**ndims
  exact_trimsurface = S / 2**ndims
  exact_totalsurface = exact_trimsurface + Vprev / (2**(ndims-1)) * ndims
  errtol = errtol

  @unittest
  def volume():
    volume = pos.volume( geom )
    volerr = abs( volume - exact_volume ) / exact_volume
    log.user( 'volume error:', volerr )
    assert volerr < errtol, 'volume tolerance not met: {:.2e} > {:.2e}'.format( volerr, errtol )

  for name, dom in ('pos',pos), ('neg',neg):
    @unittest( name )
    def div():
      dom.volume_check( geom, decimal=15 )
      dom.volume_check( geom, decimal=15 )
 
  @unittest
  def surface():
    trimsurface = pos.boundary['trimmed'].volume( geom )
    trimerr = abs( trimsurface - exact_trimsurface ) / exact_trimsurface
    log.user( 'trim surface error:', trimerr )
    totalsurface = pos.boundary.volume( geom )
    totalerr = abs( totalsurface - exact_totalsurface ) / exact_totalsurface
    log.user( 'total surface error:', totalerr )
    assert trimerr < errtol, 'trim surface tolerance not met: {:.2e} > {:.2e}'.format( trimerr, errtol )
    assert totalerr < errtol, 'total surface tolerance not met: {:.2e} > {:.2e}'.format( totalerr, errtol )

@register
def hierarchical():

  ref0, geom = mesh.rectilinear( [[0,1,2]] )
  e1, e2 = ref0
  ref1 = ref0.refined_by( [e2] )
  e1, e2, e3 = ref1
  ref2 = ref1.refined_by( [e2] )

  # Topologies:
  # ref0    [  .  .  .  |  .  .  .  ]
  # ref1    [  .  .  .  |  .  |  .  ]
  # ref2    [  .  .  .  |  |  |  .  ]
  # trimmed [  .  .  .  |]

  @unittest
  def untrimmed( makeplots=False ):
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

  @unittest
  def trimmed( makeplots=False ):
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


@register
def specialcases():

  domain, geom = mesh.rectilinear( [[0,.5,1]]*2 )
  x, y = geom

  @unittest
  def almost_all_positive():
    domain.trim( (x-y) * (x-y+.25), maxrefine=1 )

  eps = .0001

  for xi, eta, direction in (x,y,'x'), (y,x,'y'):
    for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (xi-.5,'ramp'), (xi-eta,'tilt'):
      for maxrefine in 0, 1, 2:

        @unittest( direction + how + str(maxrefine) )
        def inter_elem_2d():
          pos, neg = domain.trim( eta-.75+eps*perturb, maxrefine=maxrefine )

      for maxrefine in 0, 1:

        @unittest( ('x' if xi is x else 'y') + how + str(maxrefine) )
        def intra_elem_2d():
          pos, neg = domain.trim( eta-.5+eps*perturb, maxrefine=maxrefine )
          pos.volume_check( geom )
          neg.volume_check( geom )

  domain, geom = mesh.rectilinear( [[0,.5],[0,.5],[0,.5,1]] )
  x, y, z = geom

  for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (x-.5,'ramp'), (x-y,'tilt'):
    for maxrefine in 0, 1, 2:

      @unittest( how + str(maxrefine) )
      def inter_elem_3d():
        pos, neg = domain.trim( z-.75+eps*perturb, maxrefine=maxrefine )
