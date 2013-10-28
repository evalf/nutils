#!/usr/bin/env python

from nutils import *
import numpy

grid = numpy.linspace( 0., 1., 4 )

class ConnectivityStructuredBase( object ):
  'Tests StructuredTopology.neighbor(), also handles periodicity.'

  def test_1DConnectivity( self ):
    domain = mesh.rectilinear( 1*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert domain.neighbor( elem[0], elem[0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert domain.neighbor( elem[1], elem[2] ) ==  1, 'Failed to identify codim 1 neighbors'
    if self.periodic:
      assert domain.neighbor( elem[0], elem[2] ) ==  1, 'Failed to identify periodicity neighbors'
    else:
      assert domain.neighbor( elem[0], elem[2] ) == -1, 'Failed to identify non-neighbors'

  def test_2DConnectivity( self ):
    domain = mesh.rectilinear( 2*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert domain.neighbor( elem[0,0], elem[0,0] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert domain.neighbor( elem[1,1], elem[1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert domain.neighbor( elem[0,0], elem[1,1] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert domain.neighbor( elem[1,1], elem[0,0] ) ==  2, 'Failed to identify codim 2 neighbors'
    if self.periodic:
      assert domain.neighbor( elem[2,1], elem[0,1] ) ==  1, 'Failed to identify periodicity neighbors'
      assert domain.neighbor( elem[2,1], elem[0,0] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert domain.neighbor( elem[2,1], elem[0,1] ) == -1, 'Failed to identify non-neighbors'

  def test_3DConnectivity( self ):
    domain = mesh.rectilinear( 3*(grid,), periodic=[0] if self.periodic else [] )[0]
    elem = domain.structure
    assert domain.neighbor( elem[1,1,1], elem[1,1,1] ) ==  0, 'Failed to identify codim 0 neighbors'
    assert domain.neighbor( elem[1,1,1], elem[1,1,2] ) ==  1, 'Failed to identify codim 1 neighbors'
    assert domain.neighbor( elem[1,1,1], elem[1,2,2] ) ==  2, 'Failed to identify codim 2 neighbors'
    assert domain.neighbor( elem[1,1,1], elem[2,2,2] ) ==  3, 'Failed to identify codim 3 neighbors'
    if self.periodic:
      assert domain.neighbor( elem[0,2,2], elem[2,2,2] ) ==  1, 'Failed to identify periodicity neighbors'
      assert domain.neighbor( elem[0,2,2], elem[2,1,2] ) ==  2, 'Failed to identify periodicity neighbors'
    else:
      assert domain.neighbor( elem[0,2,2], elem[2,2,2] ) == -1, 'Failed to identify non-neighbors'

class TestConnectivityStructured( ConnectivityStructuredBase ):
  periodic = False

class TestConnectivityStructuredPeriodic( ConnectivityStructuredBase ):
  periodic = True

class TestStructure2D( object ):

  def verify_connectivity( self, structure, coords ):
    (e00,e01), (e10,e11) = structure

    a0 = coords( e00, numpy.array([0,1]) )
    a1 = coords( e01, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( a0, a1 )

    b0 = coords( e10, numpy.array([1,1]) )
    b1 = coords( e11, numpy.array([1,0]) )
    numpy.testing.assert_array_almost_equal( b0, b1 )

    c0 = coords( e00, numpy.array([1,0]) )
    c1 = coords( e10, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( c0, c1 )

    d0 = coords( e01, numpy.array([1,1]) )
    d1 = coords( e11, numpy.array([0,1]) )
    numpy.testing.assert_array_almost_equal( d0, d1 )

    x00 = coords( e00, numpy.array([1,1]) )
    x01 = coords( e01, numpy.array([1,0]) )
    x10 = coords( e10, numpy.array([0,1]) )
    x11 = coords( e11, numpy.array([0,0]) )
    numpy.testing.assert_array_almost_equal( x00, x01 )
    numpy.testing.assert_array_almost_equal( x10, x11 )
    numpy.testing.assert_array_almost_equal( x00, x11 )

  def testMesh( self ):
    domain, coords = mesh.rectilinear( [[-1,0,1]]*2 )
    self.verify_connectivity( domain.structure, coords )

  def testBoundaries( self ):
    domain, coords = mesh.rectilinear( [[-1,0,1]]*3 )
    for grp in 'left', 'right', 'top', 'bottom', 'front', 'back':
      bnd = domain.boundary[grp]
      self.verify_connectivity( bnd.structure, coords )
      xn = bnd.elem_eval( coords.dotnorm(coords), ischeme='gauss1', separate=False )
      numpy.testing.assert_array_less( 0, xn, 'inward pointing normals' )
