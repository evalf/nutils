#!/usr/bin/env python

from nutils import *
import numpy, copy

grid = numpy.linspace( 0., 1., 4 )

class ConnectivityStructuredBase( object ):
  'Tests StructuredTopology.neighbor(), also handles periodicity.'

  def test_1DConnectivity( self ):
    domain = mesh.rectilinear( 1*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert element.neighbor( elem[0], elem[0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert element.neighbor( elem[1], elem[2] ) ==  1, 'Failed to identify codim 1 neighbors'
    if self.periodic:
      assert element.neighbor( elem[0], elem[2] ) ==  1, 'Failed to identify periodicity neighbors'
    else:
      assert element.neighbor( elem[0], elem[2] ) == -1, 'Failed to identify non-neighbors'

  def test_2DConnectivity( self ):
    domain = mesh.rectilinear( 2*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert element.neighbor( elem[0,0], elem[0,0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert element.neighbor( elem[1,1], elem[1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert element.neighbor( elem[0,0], elem[1,1] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert element.neighbor( elem[1,1], elem[0,0] ) ==  2, 'Failed to identify codim 2 neighbors'
    if self.periodic:
      assert element.neighbor( elem[2,1], elem[0,1] ) ==  1, 'Failed to identify periodicity neighbors'
      assert element.neighbor( elem[2,1], elem[0,0] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert element.neighbor( elem[2,1], elem[0,1] ) == -1, 'Failed to identify non-neighbors'

  def test_3DConnectivity( self ):
    domain = mesh.rectilinear( 3*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert element.neighbor( elem[1,1,1], elem[1,1,1] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert element.neighbor( elem[1,1,1], elem[1,1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert element.neighbor( elem[1,1,1], elem[1,2,2] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert element.neighbor( elem[1,1,1], elem[2,2,2] ) ==  3, 'Failed to identify codim 3 neighbors'
    if self.periodic:
      assert element.neighbor( elem[0,2,2], elem[2,2,2] ) ==  1, 'Failed to identify periodicity neighbors'
      assert element.neighbor( elem[0,2,2], elem[2,1,2] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert element.neighbor( elem[0,2,2], elem[2,2,2] ) == -1, 'Failed to identify non-neighbors'

class TestConnectivityStructured( ConnectivityStructuredBase ):
  periodic = False

class TestConnectivityStructuredPeriodic( ConnectivityStructuredBase ):
  periodic = True

class TestStructure2D( object ):
  'Test coordinate evaluation for StructuredTopology.'

  def verify_connectivity( self, structure, geom ):
    (e00,e01), (e10,e11) = structure

    geom = geom.compiled()

    a0 = geom.eval( e00, numpy.array([0,1]) )
    a1 = geom.eval( e01, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( a0, a1 )

    b0 = geom.eval( e10, numpy.array([1,1]) )
    b1 = geom.eval( e11, numpy.array([1,0]) )
    numpy.testing.assert_array_almost_equal( b0, b1 )

    c0 = geom.eval( e00, numpy.array([1,0]) )
    c1 = geom.eval( e10, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( c0, c1 )

    d0 = geom.eval( e01, numpy.array([1,1]) )
    d1 = geom.eval( e11, numpy.array([0,1]) )
    numpy.testing.assert_array_almost_equal( d0, d1 )

    x00 = geom.eval( e00, numpy.array([1,1]) )
    x01 = geom.eval( e01, numpy.array([1,0]) )
    x10 = geom.eval( e10, numpy.array([0,1]) )
    x11 = geom.eval( e11, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( x00, x01 )
    numpy.testing.assert_array_almost_equal( x10, x11 )
    numpy.testing.assert_array_almost_equal( x00, x11 )

  def testMesh( self ):
    domain, geom = mesh.rectilinear( [[-1,0,1]]*2 )
    self.verify_connectivity( domain.structure, geom )

  def testBoundaries( self ):
    domain, geom = mesh.rectilinear( [[-1,0,1]]*3 )
    for grp in 'left', 'right', 'top', 'bottom', 'front', 'back':
      bnd = domain.boundary[grp]
      self.verify_connectivity( bnd.structure, geom )
      xn = bnd.elem_eval( geom.dotnorm(geom), ischeme='gauss1', separate=False )
      numpy.testing.assert_array_less( 0, xn, 'inward pointing normals' )

class TestTopologyGlueing( object ):
  'Test glueing of compatible topologies along prespecified boundary.'

  def __init__( self ):
    'Create half dome geometry for glueing.'
    # Aliases
    pi, sqrt, sin, cos, abs = numpy.pi, function.sqrt, function.sin, function.cos, function.abs
    grid = numpy.linspace( -.25*pi, .25*pi, 5 )

    # Half dome
    self.topo0, (xi, eta) = mesh.rectilinear( 2*(grid,) )
    x, y = sqrt(2)*sin(xi)*cos(eta), sqrt(2)*sin(eta)*cos(xi)
    self.geom0 = function.stack( [x, y, abs(1-x**2-y**2)] ) # Don't take sqrt, upsets BEM conv.

    # Plane, rotated to ensure singular-integration-compatible connectivity
    self.topo1, (xi, eta) = mesh.rectilinear( 2*(grid,) )
    for elem in self.topo1: # relabel vertices
      elem.vertices = tuple( vertex+"/" for vertex in elem.vertices )
    x, y = sin(xi)*cos(eta)-sin(eta)*cos(xi), sin(xi)*cos(eta)+sin(eta)*cos(xi)
    self.geom1 = function.stack( [x, -y, 0] ) # minus to get normal downwards

    # Merged function object and coordinate function
    # For one single merged coordinate system we need the cascades to match up, so we project, this
    # turns out to preserve the matching of element edges up to errors of 10^-4 at the vertices.
    splines_on = lambda topo: topo.splinefunc(degree=4)
    self.funcsp = function.merge( [splines_on(self.topo0), splines_on(self.topo1)] ).vector(3)
    dofs = self.topo0.project( self.geom0, self.funcsp, self.geom0, exact_boundaries=True, ischeme='gauss8' ) \
         | self.topo1.project( self.geom1, self.funcsp, self.geom1, exact_boundaries=True, ischeme='gauss8' )
    self.geom = self.funcsp.dot( dofs )

    # Glue boundary definition
    self.topo0.boundary.groups['__glue__'] = self.topo0.boundary
    self.topo1.boundary.groups['__glue__'] = self.topo1.boundary
    self.topo = topology.glue( self.topo0, self.topo1, self.geom, tol=1e-4 )

  def plot_gauss_on_circle( self, elem, ischeme='singular2', title='' ):
    'Given a product element on our 4x4 circular domain (see __init__), plot gauss points'
    dom, coo = mesh.rectilinear( 2*([0,1],) )
    circumf = dom.boundary.elem_eval( coo, 'bezier5', separate=True )
    quadpoints = elem.eval( ischeme )[0]
    with plot.PyPlot( 'quad', figsize=(6,5) ) as fig:
      # Glued grids
      for partition, style in (('__master__', 'r-'), ('__slave__', 'g-')):
        for cell in self.topo.groups[partition]:
          pts = self.geom( cell, circumf )
          fig.plot( pts[:,0], pts[:,1], style )
      # Quad points on elem pair
      for element, points in zip( (elem.elem1, elem.elem2), (quadpoints[:,:2], quadpoints[:,2:]) ):
        style = 'rx' if element in self.topo.groups['__master__'] else 'g+'
        pts = self.geom( element, points )
        fig.plot( pts[:,0], pts[:,1], style )

      fig.title( title + ' | n:%i, t:%i, %i'%elem.orientation )

  def _integrate( self, func, ecoll, qset=range(1,9), qmax=16, slopes=None, plot_quad_points=False ):
    '''Test convergence of approximation on all product element types.
    I: func,    integrand,
       ddomain, product domain over which to perform integration test,
       dcoords, tuple of corresponding coordinate functions,
       ecoll,   product elements over which to perform integration test,
       qset,    set of quadrature orders, length (1,2, >2) determines type of test,
       qmax,    reference quadrature level,
       slopes,  expected rate of convergence.'''
    devel = len(qset) > 2

    # This could be handled underwater by Topology.integrate only if geom can be glued.
    iw = function.iwscale( self.geom, 2 )
    iweights = iw * function.opposite( iw ) * function.IWeights()

    # integrands and primitives
    for neighbor, elems in enumerate( ecoll ):
      if devel: errs = {}
      for key, elem in elems.iteritems():
        topo = topology.UnstructuredTopology( [elem], ndims=2 )
        integrate = lambda q: topo.integrate( func, iweights=iweights, ischeme='singular%i'%q )
        F = integrate( qmax )

        if devel:
          # Devel mode (default), visual check of convergence
          errs[key] = []
          for q in qset:
            Fq = integrate( q )
            errs[key].append( numpy.abs(F/Fq-1) )

        elif len(qset) == 1:
          # Test assertions on exact quadrature
          Fq = integrate( qset[0] )
          err = numpy.abs(F/Fq-1)
          assert err < 1.e-12, 'Nonexact quadrature, err = %.1e' % err

        elif len(qset) == 2:
          # Test assertions on convergence rate of quadrature
          q0, q1 = tuple( qset )
          F0 = integrate( q0 )
          F1 = integrate( q1 )
          err0 = numpy.abs(F/F0-1)
          err1 = numpy.abs(F/F1-1)
          slope = numpy.log10(err1/err0)/(q1-q0)
          assert slope <= (-2. if slopes is None else slopes[neighbor]) or err1 < 1.e-12, \
              'Insufficient quadrature convergence (is func analytic?), slope = %.2f' % slope

        else:
          raise ValueError( 'Range of quadrature orders should contain >=1 value.' )

      if devel and len(elems):
        with plot.PyPlot( 'conv' ) as fig:
          for val in errs.itervalues():
            fig.semilogy( qset, val )
          i = len(qset)//2
          slope = fig.slope_triangle( qset[i::i-1][::-1], val[i::i-1][::-1], slopefmt='{0:.2f}' )
          fig.title( 'n-type: %i'%(-1 if neighbor is 3 else neighbor) )

  def test_2DNodeRelabelingCorrect( self, visual=False ):
    'Topology glueing should not raise any errors.'
    # 0. Test if glue passes without errors: done in __init__(), is the resulting glued topology up to specs?
    assert len(self.topo) == 32
    keys_provided = set( self.topo.groups )
    keys_required = set(['master', 'slave'])
    assert keys_provided == keys_required, 'Something went awry with copying groups into union topology.'

    bkeys_provided = set( self.topo.boundary.groups )
    bkeys_required = set(['master', 'master_bottom', 'master_left', 'master_right', 'master_top',
                          'slave', 'slave_bottom', 'slave_left', 'slave_right', 'slave_top' ])
    assert bkeys_provided == bkeys_required, 'Something went awry with copying boundary groups into union topology.'

    # 1. The connectivity should still be correct, cf. test_quadrature.TestSingularQuadrature.test_connectivity
    elem = self.topo.elements
    neighbor_tests = [[(0,1),1], [(0,2),1], [(0,0),2], [(0,3),2], [(1,1),-1]]
    index = lambda alpha, offset=16: offset+numpy.ravel_multi_index( alpha, (4,4) ) # Return index of multiindex
    for alpha, n in neighbor_tests:
      assert element.neighbor( elem[3], elem[index(alpha)] ) == n, \
          'Failed to identify codim %i neighbor over boundary' % n

    # 2. Orientation information should also be preserved, cf. test_quadrature.TestSingularQuadrature.test_orientations
    ddom = topology.UnstructuredTopology( elem[3:4], ndims=2 ) * \
           topology.UnstructuredTopology( elem[16:], ndims=2 )
    orientations = {str( elem[index((0,3))] ): [2, (0,0), (0,7), (7,0), (0,7)],
                    str( elem[index((0,2))] ): [1, (3,7), (7,3)],
                    str( elem[index((0,1))] ): [1, (2,7), (6,3)],
                    str( elem[index((0,0))] ): [2, (2,3), (2,6), (5,3), (5,6)]}
    ecoll = [{}, {}, {}, {}] # Collect product elements to be used in integration test below.
    for alpha, pelem in enumerate( ddom ):
      orientation = orientations.get( str( pelem.elem2 ), [-1, (0,0)] )
      ecoll[orientation[0]][pelem.__repr__()] = pelem
      if visual: self.plot_gauss_on_circle( pelem )
      assert pelem.reference.neighborhood == orientation[0], 'Incorrect neighbor type.'
      assert pelem.reference.transf in orientation[1:], 'Incorrect transformation.'

    # 3. Integration should converge, cf. test_quadrature.TestSingularQuadrature.test_stronglysingularfunc
    kwargs = {'qset':(2,4), 'qmax':8, 'slopes':(-0.0, -1.0, -1.0, -1.0)}
    ddom = topology.UnstructuredTopology( elem[:16], ndims=2 ) * \
           topology.UnstructuredTopology( elem[16:], ndims=2 )
    func = function.norm2( self.geom-function.opposite(self.geom) )**-2
    if visual:
      kwargs.update( {'qset':range(1,10), 'qmax':16, 'plot_quad_points':True} )
      # Fill inspection set ecoll and count number of product elements for each neighbor type
      ecoll = [{}, {}, {}, {}]
      for pelem in ddom: ecoll[pelem.orientation[0]][pelem.__repr__()] = pelem
      for i, coll in enumerate(ecoll): log.warning( 'n: %i, #el: %i' % (i, len(coll)) )
    self._integrate( func, ecoll, **kwargs )

  def test_2DNodeRelabelingBigMaster( self ):
    'This should raise an AssertionError, as there are too many master elements.'
    topo0 = copy.deepcopy( self.topo0 ) # Don't modify the original object
    topo0.boundary.groups['__glue__'] = topo0.boundary + topo0[:1,:1].boundary['right'] # For some strange reason deepcopy skips boundary['__glue__']
    args = topo0, self.topo1, self.geom
    numpy.testing.assert_raises( AssertionError, topology.glue, *args )

  def test_2DNodeRelabelingBigSlave( self ):
    'This should raise an AssertionError, as there are too many slave elements.'
    topo1 = copy.deepcopy( self.topo1 )
    topo1.boundary.groups['__glue__'] = topo1.boundary + topo1[:1,:1].boundary['right']
    args = self.topo0, topo1, self.geom
    numpy.testing.assert_raises( AssertionError, topology.glue, *args )

  def StokesBEM( self, visual=False ):
    'The singular integration scheme depends on the correct functioning of Topology.glue().'
    # Aliases and definitions
    pi, sqrt, sin, cos, abs = numpy.pi, function.sqrt, function.sin, function.cos, function.abs
    def V( x, y ):
      rInv = function.norm2( x-y )**-1.
      return 0.125*pi**-1. * (function.eye(3)*rInv + (x-y)[:,_]*(x-y)[_,:]*rInv**3)
    def K( x, y ):
      rInv = function.norm2( x-y )**-1.
      return 0.75*pi**-1. * (x-y)[:,_]*(x-y)[_,:] * ((x-y)*y.normal()).sum() * rInv**5
    l2norm = lambda func: sqrt( sum( self.topo.integrate( func**2., 'gauss6', self.geom ) ) )

    # Boundary data
    velo = function.stack( [self.geom[2], 0., 0.] )
    trac = function.stack( [self.geom.normal()[2], 0., self.geom.normal()[0]] )

    # Matrix/vector assembly (integration schemes optimized)
    prod_topo = self.topo*self.topo
    iw = function.iwscale(self.geom,2)
    iweights = iw * function.opposite(iw) * function.IWeights()
    x, y = self.geom, function.opposite(self.geom)
    kwargs = {'iweights':iweights, 'ischeme':'singular6', 'force_dense':True}
    integrand = (self.funcsp*(V(x,y)*function.opposite(self.funcsp)[:,_,_,:]).sum()).sum()
    mat = prod_topo.integrate_symm( integrand, title='int[mat]', **kwargs )
    integrand = (self.funcsp*(K(x,y)*function.opposite(velo)).sum()).sum()
    vec = 0.5 * self.topo.integrate( (self.funcsp*velo).sum(), geometry=x, ischeme='gauss4' ) \
        + prod_topo.integrate( integrand, title='int[vec]', **kwargs )

    # Solve
    sol = mat.solve( vec, tol=0, symmetric=True )
    trach = self.funcsp.dot( sol )
    if visual:
      plot.writevtu( './dome.vtu', self.topo.refine(2), self.geom, sdv=1.e-3,
          pointdata={'trac0':trac, 'trach':trach} )
    relerr = l2norm(trach-trac)/l2norm(trac)
    log.info( 'rel. L2 err: %.2e' % relerr )
    assert relerr < 1.e-2, 'Traction computed in BEM example exceeds tolerance.'

def visualinspect():
  'Visual inspection of StokesBEM test case.'
  visual = TestTopologyGlueing()
  # visual.test_2DNodeRelabelingCorrect()
  visual.StokesBEM( visual=True )

util.run( visualinspect )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
