
from nutils import *
from . import register, unittest


@register
def hierarchical():

  ref0, geom = mesh.rectilinear( [[0,1,2]] )
  e1, e2 = ref0
  ref1 = ref0.refined_by( [e2] )
  e3, e4, e5 = ref1
  ref2 = ref1.refined_by( [e4] )

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
    trimmed = ref0.trim( levelset, maxrefine=3 ).refined_by( [e2] ).refined_by( [e4] )
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
def hierarchicalboundary():

  domain, geom = mesh.rectilinear( [[0,1,2],[0,1,2]] )
  left = domain[:1].withboundary( leftbnd=... )
  leftbasis = left.basis( 'std', degree=1 )
  right = domain[1:].withboundary( rightbnd=... )
  rightbasis = right.basis( 'std', degree=1 )
  trimmed = domain - right

  @unittest
  def volume():
    assert trimmed.boundary.integrate( 1, geometry=geom, ischeme='gauss1' ) == 6

  @unittest
  def boundary():
    assert trimmed.boundary['rightbnd'].integrate( 1, geometry=geom, ischeme='gauss1' ) == 2

  @unittest
  def left_boundary():
    left = trimmed.boundary['rightbnd']
    assert numpy.any( left.elem_eval( leftbasis, ischeme='gauss1', separate=False ) )
    assert not numpy.any( left.elem_eval( function.opposite(leftbasis), ischeme='gauss1', separate=False ) )
    assert numpy.any( left.elem_eval( function.opposite(rightbasis), ischeme='gauss1', separate=False ) )
    assert not numpy.any( left.elem_eval( rightbasis, ischeme='gauss1', separate=False ) )


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

        @unittest( name=direction+how+str(maxrefine) )
        def inter_elem_2d():
          domain.trim( eta-.75+eps*perturb, maxrefine=maxrefine )

      for maxrefine in 0, 1:

        @unittest( name=('x' if xi is x else 'y')+how+str(maxrefine) )
        def intra_elem_2d():
          pos = domain.trim( eta-.5+eps*perturb, maxrefine=maxrefine )
          pos.volume_check( geom )
          neg = domain - pos
          neg.volume_check( geom )

  domain, geom = mesh.rectilinear( [[0,.5],[0,.5],[0,.5,1]] )
  x, y, z = geom

  for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (x-.5,'ramp'), (x-y,'tilt'):
    for maxrefine in 0, 1, 2:

      @unittest( name=how+str(maxrefine) )
      def inter_elem_3d():
        domain.trim( z-.75+eps*perturb, maxrefine=maxrefine )


@register
def setoperations():

  domain, geom = mesh.rectilinear( [[-.5,-1./6,1./6,.5]]*2 ) # unit square
  x, y = geom
  bottomright = domain.trim( x-y, maxrefine=0, name='trim1' )
  right = bottomright.trim( x+y, maxrefine=0, name='trim2' )
  bottom = bottomright - right
  topleft = domain - bottomright
  top = topleft.trim( x+y, maxrefine=0, name='trim2' )
  left = topleft - top

  Lexact = 1+numpy.sqrt(2)
  for name, dom in ('left',left), ('top',top), ('right',right), ('bottom',bottom):
    @unittest( name=name )
    def boundary():
      L = dom.boundary.integrate( 1, geometry=geom, ischeme='gauss1' )
      assert numpy.isclose( L, 1+numpy.sqrt(2)  ), 'full boundary: wrong length: {} != {}'.format( L, 1+numpy.sqrt(2) )
      L = dom.boundary[name].integrate( 1, geometry=geom, ischeme='gauss1' )
      assert numpy.isclose( L, 1  ), '{}: wrong length: {} != {}'.format( name, L, 1 )
      L = dom.boundary['trim1' if name not in ('left','top') else 'trim1'].integrate( 1, geometry=geom, ischeme='gauss1' )
      assert numpy.isclose( L, .5*numpy.sqrt(2)  ), 'trim1: wrong length: {} != {}'.format( L, .5*numpy.sqrt(2) )
      L = dom.boundary['trim2' if name not in ('left','bottom') else 'trim2'].integrate( 1, geometry=geom, ischeme='gauss1' )
      assert numpy.isclose( L, .5*numpy.sqrt(2)  ), 'trim2: wrong length: {} != {}'.format( L, .5*numpy.sqrt(2) )

  @unittest
  def union():
    assert (top|left) | (right|bottom) == domain
    union = (right|left) | (top|bottom)
    assert isinstance( union, topology.UnionTopology )
    assert set(union) == set(domain)


@register( 'sphere', 3, 2, 3, 6e-3 )
@register( 'circle', 2, 2, 5, 2.1e-4 )
def cutdomain( ndims, nelems, maxrefine, errtol ):

  domain, geom = mesh.rectilinear( (numpy.linspace(0,1,nelems+1),)*ndims )
  radius = numpy.sqrt( .5 )
  levelset = radius**2 - ( geom**2 ).sum(-1)
  pos = domain.trim( levelset=levelset, maxrefine=maxrefine )
  neg = domain - pos
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
    @unittest( name=name )
    def div():
      dom.volume_check( geom, decimal=14 )
 
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
def multitrim():
  domain, geom = mesh.rectilinear( [[-1,1],[-1,1]] )
  geom_rel = ( function.rotmat(numpy.pi/6) * geom ).sum(-1)
  for itrim in range(4):
    domain = domain.trim( .7+(1-itrim%2*2)*geom_rel[itrim//2], maxrefine=1, name='trim{}'.format(itrim), ndivisions=16 )
  for itrim in range(4):
    L = domain.boundary['trim{}'.format(itrim)].integrate( 1, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( L, 1.4, decimal=4 )
  L = domain.boundary.integrate( 1, geometry=geom, ischeme='gauss1' )
  numpy.testing.assert_almost_equal( L, 5.6, decimal=4 )


@register
def trim_conforming():

  domain, geom = mesh.rectilinear( [4,4] )

  @unittest
  def untrimmed():
    assert len(domain.interfaces) == 24
    assert len(domain.boundary) == 16

  domain1 = domain.trim( 3-geom[0], maxrefine=2, name='trimright' )

  @unittest
  def trimright():
    assert len(domain1.interfaces) == 17
    assert len(domain1.boundary) == 14
    assert len(domain1.boundary['trimright']) == 4

  domain2 = domain1.trim( 3-geom[1], maxrefine=2, name='trimtop' )

  @unittest
  def trimtop():
    assert len(domain2.interfaces) == 12
    assert len(domain2.boundary) == 12
    assert len(domain2.boundary['trimright']) == 3
    assert len(domain2.boundary['trimtop']) == 3

  domain3 = domain2.trim( geom[0]-1, maxrefine=2, name='trimleft' )

  @unittest
  def trimleft():
    assert len(domain3.interfaces) == 7
    assert len(domain3.boundary) == 10
    assert len(domain3.boundary['trimright']) == 3
    assert len(domain3.boundary['trimtop']) == 2
    assert len(domain3.boundary['trimleft']) == 3

  domain4 = domain3.trim( geom[1]-1, maxrefine=2, name='trimbottom' )

  @unittest
  def trimbottom():
    assert len(domain4.interfaces) == 4
    assert len(domain4.boundary) == 8
    assert len(domain4.boundary['trimright']) == 2
    assert len(domain4.boundary['trimtop']) == 2
    assert len(domain4.boundary['trimleft']) == 2
    assert len(domain4.boundary['trimbottom']) == 2

  domain5 = domain.trim( 3-geom[0], maxrefine=2, name='trimtopright' ).trim( 3-geom[1], maxrefine=2, name='trimtopright' )

  @unittest
  def trimtopright():
    assert len(domain5.interfaces) == 12
    assert len(domain5.boundary) == 12
    assert len(domain5.boundary['trimtopright']) == 6
