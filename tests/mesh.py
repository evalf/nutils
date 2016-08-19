from nutils import *
from . import unittest, register


gmsh_physical = '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
1 7 "dirichlet"
1 8 "neumann"
2 9 "interior"
$EndPhysicalNames
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

@register
def gmsh():

  domain, geom = mesh.gmesh( gmsh_physical.splitlines() )
  exact_volume = 1
  exact_length = 4

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
    assert xy.shape == ( 7, 2 )
    assert numpy.all( xy == .5 )
