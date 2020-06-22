from nutils import *
from nutils.testing import *

class specialcases(TestCase):

  def test_tensoredge_swapup_identifier(self):
    lineedge = transform.SimplexEdge(1, 0, False)
    for edge in transform.TensorEdge1(lineedge, 1), transform.TensorEdge2(1, lineedge):
      with self.subTest(type(edge).__name__):
        idnt = transform.Identifier(1, 'test')
        self.assertEqual(edge.swapup(idnt), None)

class TestTransform(TestCase):

  def setUp(self, trans, linear, offset):
    self.trans = trans
    self.linear = linear
    self.offset = offset

  def test_fromdims(self):
    self.assertEqual(self.trans.fromdims, numpy.shape(self.linear)[1])

  def test_todims(self):
    self.assertEqual(self.trans.todims, numpy.shape(self.linear)[0])

  def test_linear(self):
    self.assertAllEqual(self.trans.linear, self.linear)

  def test_offset(self):
    self.assertAllEqual(self.trans.offset, self.offset)

  def test_apply(self):
    coords = numpy.array([[0]*self.trans.fromdims, numpy.arange(.5,self.trans.fromdims)/self.trans.fromdims])
    a, b = self.trans.apply(coords)
    self.assertAllAlmostEqual(a, self.offset)
    self.assertAllAlmostEqual(b, numpy.dot(self.linear, coords[1]) + self.offset)

class TestInvertible(TestTransform):

  def test_invapply(self):
    coords = numpy.array([self.offset, numpy.arange(.5,self.trans.fromdims)/self.trans.fromdims])
    a, b = self.trans.invapply(coords)
    self.assertAllAlmostEqual(a, 0)
    self.assertAllAlmostEqual(b, numpy.linalg.solve(self.linear, (coords[1] - self.offset)))

class TestUpdim(TestTransform):

  def test_ext(self):
    ext = numeric.ext(self.linear)
    self.assertAllAlmostEqual(ext, self.trans.ext)

class Matrix(TestTransform):

  def setUp(self):
    super().setUp(trans=transform.Matrix([[1.],[2]], [3.,4]), linear=[[1],[2]], offset=[3,4])

class Qquare(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Square([[1.,2],[1,3]], [5.,6]), linear=[[1,2],[1,3]], offset=[5,6])

class Shift(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Shift([1.,2]), linear=[[1,0],[0,1]], offset=[1,2])

class Identity(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Identity(2), linear=[[1,0],[0,1]], offset=[0,0])

class Scale(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Scale(2, offset=[1.,2]), linear=[[2,0],[0,2]], offset=[1,2])

class SimplexEdge(TestUpdim):

  def setUp(self):
    super().setUp(trans=transform.SimplexEdge(3, 0), linear=[[-1.,-1],[1,0],[0,1]], offset=[1,0,0])

class SimplexChild(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.SimplexChild(3, 1), linear=numpy.eye(3)/2, offset=[.5,0,0])

del TestTransform, TestInvertible, TestUpdim
