#!/usr/bin/env python

from nutils import *
from . import register, unittest
import numpy, copy

grid = numpy.linspace( 0., 1., 4 )

def neighbor( elem1, elem2 ):
  elem1_vertices = set(elem1.vertices)
  ncommon = sum( v in elem1_vertices for v in elem2.vertices )
  if not ncommon:
    return -1
  if elem1.reference == elem2.reference == element.SimplexReference(1):
    return {2:0,1:1}[ncommon]
  if elem1.reference == elem2.reference == element.SimplexReference(1)**2:
    return {4:0,2:1,1:2}[ncommon]
  if elem1.reference == elem2.reference == element.SimplexReference(1)**3:
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
