from nutils import *
from nutils.testing import *

class specialcases(TestCase):

  def test_tensoredge_swapup_index(self):
    lineedge = transform.SimplexEdge(1, 0, False)
    for edge in transform.TensorEdge1(lineedge, 1), transform.TensorEdge2(1, lineedge):
      with self.subTest(type(edge).__name__):
        idnt = transform.Index(1, 0)
        self.assertEqual(edge.swapup(idnt), None)

class TestTransform(TestCase):

  def setUp(self, trans, linear, offset):
    super().setUp()
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
    self.assertAllAlmostEqual(a, numpy.zeros((self.trans.todims,)))
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

class Identity(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Identity(2), linear=[[1,0],[0,1]], offset=[0,0])

class Index(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Index(2, 3), linear=[[1,0],[0,1]], offset=[0,0])

class SimplexEdge(TestUpdim):

  def setUp(self):
    super().setUp(trans=transform.SimplexEdge(3, 0), linear=[[-1.,-1],[1,0],[0,1]], offset=[1,0,0])

class SimplexChild(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.SimplexChild(3, 1), linear=numpy.eye(3)/2, offset=[.5,0,0])

class Point(TestTransform):

  def setUp(self):
    super().setUp(trans=transform.Point(numpy.array([1.,2.,3.])), linear=numpy.zeros((3,0)), offset=[1.,2.,3.])

del TestTransform, TestInvertible, TestUpdim


class EvaluableTransformChainArgument(TestCase):

  def test_evalf(self):
    chain = transform.SimplexEdge(2, 0),
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertEqual(echain.eval(chain=chain), chain)

  def test_todims(self):
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertEqual(echain.todims, 2)

  def test_fromdims(self):
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertEqual(echain.fromdims, 1)

  def test_linear(self):
    chain = transform.SimplexEdge(2, 0),
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertAllAlmostEqual(echain.linear.eval(chain=chain), numpy.array([[-1.],[1.]]))

  def test_linear_derivative(self):
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertTrue(evaluable.iszero(evaluable.derivative(echain.linear, evaluable.Argument('test', ())).simplified))

  def test_basis(self):
    chain = transform.SimplexEdge(2, 0),
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertAllAlmostEqual(echain.basis.eval(chain=chain), numpy.array([[-1.,1.],[1.,1.]]))

  def test_basis_derivative(self):
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    self.assertTrue(evaluable.iszero(evaluable.derivative(echain.basis, evaluable.Argument('test', ())).simplified))

  def test_apply(self):
    chain = transform.SimplexEdge(2, 0),
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    ecoords = evaluable.Argument('coords', (5, echain.fromdims), float)
    coords = numpy.linspace(0, 1, 5*echain.fromdims).reshape(5, echain.fromdims)
    self.assertAllAlmostEqual(echain.apply(ecoords).eval(chain=chain, coords=coords), transform.apply(chain, coords))

  def test_apply_derivative(self):
    chain = transform.SimplexEdge(2, 0),
    echain = transform.EvaluableTransformChain.from_argument('chain', 2, 1)
    ecoords = evaluable.Argument('coords', (5, echain.fromdims), float)
    coords = numpy.linspace(0, 1, 5*echain.fromdims).reshape(5, echain.fromdims)
    actual = evaluable.derivative(echain.apply(ecoords), ecoords).eval(chain=chain)
    desired = numpy.einsum('jk,iklm->ijlm', numpy.array([[-1.],[1.]]), numpy.eye(5*echain.fromdims).reshape(5, echain.fromdims, 5, echain.fromdims))
    self.assertAllAlmostEqual(actual, desired)

class EmptyEvaluableTransformChain(TestCase):

  def setUp(self):
    super().setUp()
    self.chain = transform.EvaluableTransformChain.empty(2)

  def test_evalf(self):
    self.assertEqual(self.chain.evalf(), ())

  def test_todims(self):
    self.assertEqual(self.chain.todims, 2)

  def test_fromdims(self):
    self.assertEqual(self.chain.fromdims, 2)

  def test_linear(self):
    self.assertAllAlmostEqual(self.chain.linear.eval(), numpy.diag([1,1]))

  def test_basis(self):
    self.assertAllAlmostEqual(self.chain.basis.eval(), numpy.diag([1,1]))

  def test_apply(self):
    coords = numpy.array([1.,2.])
    self.assertAllAlmostEqual(self.chain.apply(evaluable.Argument('coords', (2,))).eval(coords=coords),  coords)
