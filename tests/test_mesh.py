#!/usr/bin/env python

from nutils import *

elementary_mesh = '''$MeshFormat
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
                     28
                     1 15 2 0 1 1
                     2 15 2 0 2 2
                     3 15 2 0 3 3
                     4 15 2 0 4 4
                     5 1 2 0 1 1 5
                     6 1 2 0 1 5 2
                     7 1 2 0 2 2 6
                     8 1 2 0 2 6 3
                     9 1 2 0 3 4 7
                     10 1 2 0 3 7 3
                     11 1 2 0 4 4 8
                     12 1 2 0 4 8 1
                     13 2 2 0 6 9 11 8
                     14 2 2 0 6 12 9 8
                     15 2 2 0 6 13 10 6
                     16 2 2 0 6 9 10 13
                     17 2 2 0 6 1 12 8
                     18 2 2 0 6 2 10 5
                     19 2 2 0 6 7 11 9
                     20 2 2 0 6 3 13 6
                     21 2 2 0 6 3 7 13
                     22 2 2 0 6 5 10 9
                     23 2 2 0 6 5 9 12
                     24 2 2 0 6 7 9 13
                     25 2 2 0 6 4 11 7
                     26 2 2 0 6 1 5 12
                     27 2 2 0 6 2 6 10
                     28 2 2 0 6 4 8 11
                     $EndElements'''

physical_mesh = '''$MeshFormat
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
                   24
                   1 1 2 7 1 1 5
                   2 1 2 7 1 5 2
                   3 1 2 7 2 2 6
                   4 1 2 7 2 6 3
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
                   16 2 2 9 6 3 13 6
                   17 2 2 9 6 3 7 13
                   18 2 2 9 6 5 10 9
                   19 2 2 9 6 5 9 12
                   20 2 2 9 6 7 9 13
                   21 2 2 9 6 4 11 7
                   22 2 2 9 6 1 5 12
                   23 2 2 9 6 2 6 10
                   24 2 2 9 6 4 8 11
                   $EndElements'''

class GmshBase ( object ):
  def __init__ ( self ):
    self.topo, self.geom = mesh.gmesh( self.data.splitlines(), tags=self.tags, use_elementary=self.use_elementary )

  def test_area( self ):
    area = self.topo.integrate( 1., geometry=self.geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( area, 1., decimal=10 )

  def test_length( self ):
    length = self.topo.boundary.integrate( 1., geometry=self.geom, ischeme='gauss1' )
    numpy.testing.assert_almost_equal( length, 4., decimal=10 )

  def test_gauss( self ):
    points, normals =  self.topo.boundary.elem_eval( [self.geom, self.geom.normal()], ischeme='gauss2', separate=True )
    for i,j in zip(points,normals):
      print i, j
    #area = self.topo.boundary.integrate( 0.5*self.geom.dotnorm(self.geom), geometry=self.geom, ischeme='gauss1' )
    #numpy.testing.assert_almost_equal( area, 1., decimal=10 )

class TestGmshElementary ( GmshBase ):
  data = elementary_mesh
  use_elementary = True
  tags = {}

class TestGmshPhysical ( GmshBase ):
  data = physical_mesh
  use_elementary = False
  tags = {7:'dirichlet',8:'neumann',9:'interior'}

def elementary ():
  test = TestGmshElementary()
  test.test_gauss()

def physical ():
  test = TestGmshPhysical()
  test.test_gauss()

if __name__ == '__main__':
  util.run( elementary, physical )
