#!/usr/bin/env python

from nutils import *
from . import unittest, register


gmsh_elementary = '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
13
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.499999999998694 0 0
6 1 0.499999999998694 0
7 0.499999999998694 1 0
8 0 0.5000000000020591 0
9 0.5 0.5 0
10 0.7499999999993471 0.249999999999347 0
11 0.249999999999347 0.7500000000010296 0
12 0.2499999999996735 0.2500000000005148 0
13 0.750000000000653 0.749999999999347 0
$EndNodes
$Elements
29
1 15 2 0 1 1
2 15 2 0 2 2
3 15 2 0 3 3
4 15 2 0 4 4
5 1 2 0 1 1 5
6 1 2 0 1 2 5
7 1 2 0 2 2 6
8 1 2 0 2 3 6
9 1 2 0 3 4 7
10 1 2 0 3 7 3
11 1 2 0 4 4 8
12 1 2 0 4 8 1
13 2 2 0 6 9 11 8
14 2 2 0 6 12 9 8
15 2 2 0 6 13 10 6
16 2 2 0 6 9 10 13
17 2 2 0 6 1 12 8
18 2 2 0 6 10 2 5
19 2 2 0 6 7 11 9
20 2 2 0 6 3 13 6
21 2 2 0 6 3 7 13
22 2 2 0 6 5 9 10
23 2 2 0 6 5 9 12
24 2 2 0 6 7 9 13
25 2 2 0 6 4 11 7
26 2 2 0 6 1 5 12
27 2 2 0 6 2 6 10
28 2 2 0 6 4 8 11
29 15 2 0 7 9
$EndElements'''

gmsh_physical = '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
13
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.499999999998694 0 0
6 1 0.499999999998694 0
7 0.499999999998694 1 0
8 0 0.5000000000020591 0
9 0.5 0.5 0
10 0.7499999999993471 0.249999999999347 0
11 0.249999999999347 0.7500000000010296 0
12 0.2499999999996735 0.2500000000005148 0
13 0.750000000000653 0.749999999999347 0
$EndNodes
$Elements
25
1 1 2 7 1 1 5
2 1 2 7 1 5 2
3 1 2 7 2 2 6
4 1 2 7 2 3 6
5 1 2 8 3 4 7
6 1 2 8 3 7 3
7 1 2 8 4 4 8
8 1 2 8 4 8 1
9 2 2 9 6 9 11 8
10 2 2 9 6 12 9 8
11 2 2 9 6 13 10 6
12 2 2 9 6 9 10 13
13 2 2 9 6 1 12 8
14 2 2 9 6 2 10 5
15 2 2 9 6 7 11 9
16 2 2 9 6 13 3 6
17 2 2 9 6 3 7 13
18 2 2 9 6 5 10 9
19 2 2 9 6 5 9 12
20 2 2 9 6 9 7 13
21 2 2 9 6 4 11 7
22 2 2 9 6 1 5 12
23 2 2 9 6 2 6 10
24 2 2 9 6 4 8 11
25 15 2 3 7 9
$EndElements'''

@register( 'elementary', gmsh_elementary, {}, 1, 4, True )
@register( 'physical', gmsh_physical, {7:'dirichlet',8:'neumann',9:'interior'}, 1, 4, False )
def check( data, tags, exact_volume, exact_length, use_elementary ):

  domain, geom = mesh.gmesh( data.splitlines(), tags=tags, use_elementary=use_elementary )

  @unittest
  def volume():
    volume = domain.integrate( 1., geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( volume, exact_volume, decimal=10 )

  @unittest
  def length():
    length = domain.boundary.integrate( 1., geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( length, exact_length, decimal=10 )

  @unittest
  def divergence():
    volumes = domain.boundary.integrate( geom*geom.normal(), geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( volumes, 1., decimal=10 )

  @unittest
  def pointeval():
    xy = domain.points.elem_eval( geom, ischeme='gauss1' )
    assert xy.shape == ( 7 if not use_elementary else 0, 2 )
    assert numpy.all( xy == .5 )
