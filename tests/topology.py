
from nutils import *
from . import register, unittest
import numpy, copy, sys, pickle, subprocess, base64

grid = numpy.linspace( 0., 1., 4 )

def neighbor( elem1, elem2 ):
  elem1_vertices = set(elem1.vertices)
  ncommon = sum( v in elem1_vertices for v in elem2.vertices )
  if not ncommon:
    return -1
  if elem1.reference == elem2.reference == element.getsimplex(1):
    return {2:0,1:1}[ncommon]
  if elem1.reference == elem2.reference == element.getsimplex(1)**2:
    return {4:0,2:1,1:2}[ncommon]
  if elem1.reference == elem2.reference == element.getsimplex(1)**3:
    return {8:0,4:1,2:2,1:3}[ncommon]
  raise NotImplementedError( '%s, %s' % ( elem1.reference, elem2.reference ) )

def verify_connectivity( structure, geom ):
  (e00,e01), (e10,e11) = structure

  a0 = geom.eval( e00, numpy.array([[0,1]]) )
  a1 = geom.eval( e01, numpy.array([[0,0]]) )
  numpy.testing.assert_array_almost_equal( a0, a1 )

  b0 = geom.eval( e10, numpy.array([[1,1]]) )
  b1 = geom.eval( e11, numpy.array([[1,0]]) )
  numpy.testing.assert_array_almost_equal( b0, b1 )

  c0 = geom.eval( e00, numpy.array([[1,0]]) )
  c1 = geom.eval( e10, numpy.array([[0,0]]) )
  numpy.testing.assert_array_almost_equal( c0, c1 )

  d0 = geom.eval( e01, numpy.array([[1,1]]) )
  d1 = geom.eval( e11, numpy.array([[0,1]]) )
  numpy.testing.assert_array_almost_equal( d0, d1 )

  x00 = geom.eval( e00, numpy.array([[1,1]]) )
  x01 = geom.eval( e01, numpy.array([[1,0]]) )
  x10 = geom.eval( e10, numpy.array([[0,1]]) )
  x11 = geom.eval( e11, numpy.array([[0,0]]) )
  numpy.testing.assert_array_almost_equal( x00, x01 )
  numpy.testing.assert_array_almost_equal( x10, x11 )
  numpy.testing.assert_array_almost_equal( x00, x11 )

def verify_boundaries( domain, geom ):
  # Test ∫_Ω f_,i = ∫_∂Ω f n_i.
  f = ((0.5 - geom)**2).sum(axis=0)
  lhs = domain.integrate( f.grad(geom), ischeme='gauss2', geometry=geom )
  rhs = domain.boundary.integrate( f*function.normal(geom), ischeme='gauss2', geometry=geom )
  numpy.testing.assert_array_almost_equal( lhs, rhs )

def verify_interfaces( domain, geom, periodic ):
  x1, x2, n1, n2 = domain.interfaces.elem_eval( [ geom, function.opposite(geom), geom.normal(), function.opposite(geom.normal()) ], 'gauss2', separate=False )
  if not periodic:
    assert numpy.all( x1 == x2 )
  assert numpy.all( n1 == -n2 )

  # Test ∫_E f_,i = ∫_∂E f n_i ∀ E in `domain`.
  f = ((0.5 - geom)**2).sum(axis=0)
  elemindicator = domain.basis( 'discont', degree=0 ).vector( domain.ndims )
  lhs = domain.integrate( (elemindicator*f.grad(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom )
  rhs = domain.interfaces.integrate( (-function.jump(elemindicator)*f*function.normal(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom )
  if not periodic:
    rhs += domain.boundary.integrate( (elemindicator*f*function.normal(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom )
  numpy.testing.assert_array_almost_equal( lhs, rhs )

@register( 'periodic', periodic=True )
@register( 'nonperiodic', periodic=False )
def connectivity( periodic ):

  @unittest
  def check1d():
    domain = mesh.rectilinear( 1*(grid,), periodic=[0] if periodic else [] )[0]
    elem = domain.structure
    assert neighbor( elem[0], elem[0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert neighbor( elem[1], elem[2] ) ==  1, 'Failed to identify codim 1 neighbors'
    if periodic:
      assert neighbor( elem[0], elem[2] ) ==  1, 'Failed to identify periodicity neighbors'
    else:
      assert neighbor( elem[0], elem[2] ) == -1, 'Failed to identify non-neighbors'

  @unittest
  def check2d():
    domain = mesh.rectilinear( 2*(grid,), periodic=[0] if periodic else [] )[0]
    elem = domain.structure
    assert neighbor( elem[0,0], elem[0,0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert neighbor( elem[1,1], elem[1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert neighbor( elem[0,0], elem[1,1] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert neighbor( elem[1,1], elem[0,0] ) ==  2, 'Failed to identify codim 2 neighbors'
    if periodic:
      assert neighbor( elem[2,1], elem[0,1] ) ==  1, 'Failed to identify periodicity neighbors'
      assert neighbor( elem[2,1], elem[0,0] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert neighbor( elem[2,1], elem[0,1] ) == -1, 'Failed to identify non-neighbors'

  @unittest
  def check3d():
    domain = mesh.rectilinear( 3*(grid,), periodic=[0] if periodic else [] )[0]
    elem = domain.structure
    assert neighbor( elem[1,1,1], elem[1,1,1] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert neighbor( elem[1,1,1], elem[1,1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert neighbor( elem[1,1,1], elem[1,2,2] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert neighbor( elem[1,1,1], elem[2,2,2] ) ==  3, 'Failed to identify codim 3 neighbors'
    if periodic:
      assert neighbor( elem[0,2,2], elem[2,2,2] ) ==  1, 'Failed to identify periodicity neighbors'
      assert neighbor( elem[0,2,2], elem[2,1,2] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert neighbor( elem[0,2,2], elem[2,2,2] ) == -1, 'Failed to identify non-neighbors'

@register
def structure2d():

  @unittest
  def domain():
    domain, geom = mesh.rectilinear( [[-1,0,1]]*2 )
    verify_connectivity( domain.structure, geom )

  @unittest
  def boundaries():
    domain, geom = mesh.rectilinear( [[-1,0,1]]*3 )
    for grp in 'left', 'right', 'top', 'bottom', 'front', 'back':
      bnd = domain.boundary[grp]
      # DISABLED: what does this check? -GJ 14/07/28
      #verify_connectivity( bnd.structure, geom )
      xn = bnd.elem_eval( geom.dotnorm(geom), ischeme='gauss1', separate=False )
      numpy.testing.assert_array_less( 0, xn, 'inward pointing normals' )

  @unittest
  def interfaces():
    domain, geom = mesh.rectilinear( [[-1,0,1]]*3 )
    verify_interfaces( domain, geom, periodic=False )

def _test_pickle_dump_load( data ):
  script = b'from nutils import *\nimport pickle, base64\npickle.loads( base64.decodebytes( b"""' \
    + base64.encodebytes( pickle.dumps( data ) ) \
    + b'""" ) )'
  p = subprocess.Popen( [ sys.executable ], stdin=subprocess.PIPE )
  p.communicate( script )
  assert p.wait() == 0, 'unpickling failed'

@register
def picklability():

  @unittest
  def domain():
    domain, geom = mesh.rectilinear( [[0,1,2]]*2 )
    _test_pickle_dump_load( domain )

  @unittest
  def geom():
    domain, geom = mesh.rectilinear( [[0,1,2]]*2 )
    _test_pickle_dump_load( geom )

  @unittest
  def basis():
    domain, geom = mesh.rectilinear( [[0,1,2]]*2 )
    basis = domain.basis( 'spline', degree=2 )
    _test_pickle_dump_load( basis )

@register
def common_refine():

  dom, geom = mesh.rectilinear( [[0,1,2],[0,1,2]] )

  dom1 = dom.refined_by( list(dom)[:1] )
  fun1 = dom1.basis( 'std', degree=1 )
  vals1 = .5,.25,.5,1,.5,.25,.5,.25,.0625,.125,.125,.25

  dom234 = dom.refined_by( list(dom)[1:] )
  fun234 = dom234.basis( 'std', degree=1 )
  vals234 = .25,.5,.5,1,.125,.0625,.25,.125,.25,.125,.125,.25,.25,.25,.125,.0625,.125,.125,.125,.0625

  dom123 = dom.refined_by( list(dom)[:-1] )
  fun123 = dom123.basis( 'std', degree=1 )
  vals123 = 1,.5,.5,.25,.0625,.125,.125,.125,.0625,.125,.25,.25,.25,.125,.125,.25,.125,.25,.0625,.125

  dom4 = dom.refined_by( list(dom)[-1:] )
  fun4 = dom4.basis( 'std', degree=1 )
  vals4 = .25,.5,.25,.5,1,.5,.25,.5,.25,.125,.125,.0625

  @unittest
  def ref1vs234():
    common = topology.common_refine( dom1, dom234 )
    assert len(common) == 16
    vals = common.integrate( fun1, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals1 )
    vals = common.integrate( fun234, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals234 )

  @unittest
  def ref1vs4():
    common = topology.common_refine( dom1, dom4 )
    assert len(common) == 10
    vals = common.integrate( fun1, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals1 )
    vals = common.integrate( fun4, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals4 )

  @unittest
  def ref123vs234():
    common = topology.common_refine( dom123, dom234 )
    assert len(common) == 16
    vals = common.integrate( fun123, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals123 )
    vals = common.integrate( fun234, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( vals, vals234 )

@register
def revolved():

  rdomain, rgeom = mesh.rectilinear( [2] )
  domain, geom, simplify = rdomain.revolved( rgeom )

  @unittest
  def simplified():
    integrand = function.norm2(geom) * function.jacobian( geom, ndims=1 )
    assert integrand != simplify(integrand)
    vals1, vals2 = domain.elem_eval( [ integrand, simplify(integrand) ], ischeme='uniform2' )
    numpy.testing.assert_array_almost_equal( vals1, vals2 )

  @unittest
  def circle_area():
    vol = domain.integrate( 1, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
    numpy.testing.assert_array_almost_equal( vol, 4 )

  @unittest
  def circle_circumference():
    surf = domain.boundary.integrate( 1, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
    numpy.testing.assert_array_almost_equal( surf, 4 )

  rzdomain, rzgeom = mesh.rectilinear( [1,2] )
  domain, geom, simplify = rzdomain.revolved( rzgeom )

  @unittest
  def cylinder_volume():
    vol = domain.integrate( 1, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
    numpy.testing.assert_array_almost_equal( vol, 2 )

  for name, exact in ('full',6), ('right',4), ('left',0):
    @unittest( name=name )
    def cylinder_surface():
      surf = domain.boundary[name if name != 'full' else ()].integrate( 1, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
      numpy.testing.assert_array_almost_equal( surf, exact )

  rzdomain, rzgeom = mesh.rectilinear( [[.5,1],2] )
  domain, geom, simplify = rzdomain.revolved( rzgeom )

  @unittest
  def hollowcylinder_volume():
    v = domain.integrate( 1, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
    numpy.testing.assert_array_almost_equal( v, 1.5 )

  @unittest
  def hollowcylinder_volume2():
    v = domain.boundary.integrate( geom.dotnorm(geom)/3, geometry=geom, ischeme='gauss1', edit=simplify ) / numpy.pi
    numpy.testing.assert_array_almost_equal( v, 1.5 )

  for name, exact in ('full',7.5), ('right',4), ('left',2):
    @unittest
    def hollowcylinder_surface():
      surf = domain.boundary[name if name != 'full' else ()].integrate( 1, geometry=geom, ischeme='gauss9', edit=simplify ) / numpy.pi
      numpy.testing.assert_array_almost_equal( surf, exact )

  basis = domain.basis( 'std', degree=2 )

  @unittest
  def hollowcylinder_basistype():
    assert isinstance( basis, function.Inflate )

  @unittest
  def hollowcylinder_dofcount():
    assert len(basis) == 3*5


@register( 'unstructured_periodic2', False, 2 )
@register( 'unstructured_periodic1', False, 1 )
@register( 'unstructured_periodic0', False, 0 )
@register( 'unstructured', False, False )
@register( 'structured_periodic2', True, 2 )
@register( 'structured_periodic1', True, 1 )
@register( 'structured_periodic0', True, 0 )
@register( 'structured', True, False )
def general( isstructured, periodic ):

  domain, geom = mesh.rectilinear( [3,4,5], periodic=[] if periodic is False else [periodic] )
  if not isstructured:
    domain = topology.UnstructuredTopology( domain.ndims, tuple(domain) )

  @unittest
  def connectivity():
    nboundaries = 0
    ninterfaces = 0
    for ielem, ioppelems in enumerate(domain.connectivity):
      for iedge, ioppelem in enumerate(ioppelems):
        if ioppelem == -1:
          nboundaries += 1
        else:
          ioppedge = tuple( domain.connectivity[ioppelem] ).index( ielem )
          edge = domain.elements[ielem].edge(iedge)
          oppedge = domain.elements[ioppelem].edge(ioppedge)
          assert sorted(edge.vertices) == sorted(oppedge.vertices), 'edges do not match: {}, {}'.format( edge, oppedge )
          ninterfaces += .5
    assert nboundaries == len(domain.boundary), 'incompatible number of boundaries'
    assert ninterfaces == len(domain.interfaces), 'incompatible number of interfaces'

  @unittest
  def boundary():
    for elem in domain.boundary:
      ielem, tail = elem.transform.lookup_item( domain.edict )
      iedge = domain.elements[ielem].reference.edge_transforms.index( tail )
      assert domain.connectivity[ielem][iedge] == -1

  @unittest
  def interfaces():
    for elem in domain.interfaces:
      ielem, tail = elem.transform.lookup_item( domain.edict )
      iedge = domain.elements[ielem].reference.edge_transforms.index( tail )
      ioppelem, opptail = elem.opposite.lookup_item( domain.edict )
      ioppedge = domain.elements[ioppelem].reference.edge_transforms.index( opptail )
      assert domain.connectivity[ielem][iedge] == ioppelem
      assert domain.connectivity[ioppelem][ioppedge] == ielem


@register( 'structured', structured=True )
@register( 'unstructured', structured=False )
def locate( structured ):

  @unittest
  def test():
    domain, geom = mesh.rectilinear( [numpy.linspace(0,1,3)]*2 ) if structured else mesh.demo()
    geom += .1 * function.sin( geom * numpy.pi ) # non-polynomial geometry
    target = numpy.array([ (.2,.3), (.1,.9), (0,1) ])
    ltopo = domain.locate( geom, target, eps=1e-15 )
    located = ltopo.elem_eval( geom, ischeme='gauss1' )
    numpy.testing.assert_array_almost_equal( located, target )


@register( '3d', ndims=3 )
@register( '2d_periodic', ndims=2, periodic=True )
@register( '2d', ndims=2 )
@register( '1d_periodic', ndims=1, periodic=True )
#@register( '1d', ndims=1 ) # disabled, see issue #193
def hierarchical( ndims, periodic=False ):

  domain, geom = mesh.rectilinear( [numpy.linspace(0, 1, 5)]*ndims, periodic=range(ndims) if periodic else [] )
  # Refine `domain`.
  indicator = 1
  for dim in range( ndims ):
      indicator = function.min( indicator, 1 - geom[dim] )
  for threshold in 0.5, 0.75:
      domain = domain.refined_by( elem for elem, value in zip( domain, domain.elem_mean( [indicator], ischeme='gauss1', geometry=geom )[0] ) if value >= threshold )

  if not periodic:
    @unittest
    def boundaries():
      verify_boundaries( domain, geom )

  @unittest
  def interfaces():
    verify_interfaces( domain, geom, periodic )
