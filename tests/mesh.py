from nutils import *
from . import unittest, register

# gmsh geo:
#
# Point(1) = {0,0,0,.5};
# Point(2) = {1,0,0,.5};
# Point(3) = {1,1,0,.5};
# Point(4) = {0,1,0,.5};
# Line(5) = {1,2};
# Line(6) = {2,3};
# Line(7) = {3,4};
# Line(8) = {4,1};
# Line Loop(9) = {5,6,7,8};
# Plane Surface(10) = {9};
# Physical Point("corner") = {2};
# Physical Line("neumann") = {5};
# Physical Line("dirichlet") = {6,7,8};
# Physical Surface("interior") = {10};

gmshdata = '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
0 1 "corner"
1 2 "neumann"
1 3 "dirichlet"
2 4 "interior"
$EndPhysicalNames
$Nodes
13
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0.499999999998694 0 0
6 1 0.499999999998694 0
7 0.5000000000020591 1 0
8 0 0.5000000000020591 0
9 0.5 0.5 0
10 0.2500000000010295 0.7500000000010295 0
11 0.7500000000005148 0.7499999999996735 0
12 0.749999999999432 0.249999999999432 0
13 0.2499999999989704 0.2500000000010296 0
$EndNodes
$Elements
25
1 15 2 1 2 2
2 1 2 2 5 1 5
3 1 2 2 5 5 2
4 1 2 3 6 2 6
5 1 2 3 6 6 3
6 1 2 3 7 3 7
7 1 2 3 7 7 4
8 1 2 3 8 4 8
9 1 2 3 8 8 1
10 2 2 4 10 13 10 8
11 2 2 4 10 9 10 13
12 2 2 4 10 4 10 7
13 2 2 4 10 1 13 8
14 2 2 4 10 3 11 6
15 2 2 4 10 2 12 5
16 2 2 4 10 7 10 9
17 2 2 4 10 7 9 11
18 2 2 4 10 6 11 9
19 2 2 4 10 5 9 13
20 2 2 4 10 5 12 9
21 2 2 4 10 6 9 12
22 2 2 4 10 1 5 13
23 2 2 4 10 2 6 12
24 2 2 4 10 4 8 10
25 2 2 4 10 3 7 11
$EndElements
'''

@register
def gmsh():

  domain, geom = mesh.gmesh( gmshdata.splitlines() )

  @unittest
  def volume():
    volume = domain.integrate( 1, geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( volume, 1, decimal=10 )

  for group, exact_length in ('neumann',1), ('dirichlet',3), ((),4):
    @unittest( name=group or 'all' )
    def length():
      length = domain.boundary[group].integrate( 1., geometry=geom, ischeme='gauss1' )
      numpy.testing.assert_almost_equal( length, exact_length, decimal=10 )

  @unittest
  def divergence():
    volumes = domain.boundary.integrate( geom*geom.normal(), geometry=geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( volumes, 1., decimal=10 )

  @unittest
  def pointeval():
    xy = domain.points.elem_eval( geom, ischeme='gauss1' )
    assert xy.shape == ( 2, 2 )
    assert numpy.all( xy == [1,0] )
