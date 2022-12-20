from nutils import transform, evaluable, numeric, types
from nutils.testing import TestCase
import numpy


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
        coords = numpy.array([[0]*self.trans.fromdims, numpy.arange(.5, self.trans.fromdims)/self.trans.fromdims])
        a, b = self.trans.apply(coords)
        self.assertAllAlmostEqual(a, self.offset)
        self.assertAllAlmostEqual(b, numpy.dot(self.linear, coords[1]) + self.offset)


class TestInvertible(TestTransform):

    def test_invapply(self):
        coords = numpy.array([self.offset, numpy.arange(.5, self.trans.fromdims)/self.trans.fromdims])
        a, b = self.trans.invapply(coords)
        self.assertAllAlmostEqual(a, numpy.zeros((self.trans.todims,)))
        self.assertAllAlmostEqual(b, numpy.linalg.solve(self.linear, (coords[1] - self.offset)))


class TestUpdim(TestTransform):

    def test_ext(self):
        ext = numeric.ext(self.linear)
        self.assertAllAlmostEqual(ext, self.trans.ext)


class Matrix(TestTransform):

    def setUp(self):
        super().setUp(trans=transform.Matrix(types.arraydata([[1.], [2]]), types.arraydata([3., 4])), linear=[[1], [2]], offset=[3, 4])


class Qquare(TestInvertible):

    def setUp(self):
        super().setUp(trans=transform.Square(types.arraydata([[1., 2], [1, 3]]), types.arraydata([5., 6])), linear=[[1, 2], [1, 3]], offset=[5, 6])


class Identity(TestInvertible):

    def setUp(self):
        super().setUp(trans=transform.Identity(2), linear=[[1, 0], [0, 1]], offset=[0, 0])


class Index(TestInvertible):

    def setUp(self):
        super().setUp(trans=transform.Index(2, 3), linear=[[1, 0], [0, 1]], offset=[0, 0])


class SimplexEdge(TestUpdim):

    def setUp(self):
        super().setUp(trans=transform.SimplexEdge(3, 0), linear=[[-1., -1], [1, 0], [0, 1]], offset=[1, 0, 0])


class SimplexChild(TestInvertible):

    def setUp(self):
        super().setUp(trans=transform.SimplexChild(3, 1), linear=numpy.eye(3)/2, offset=[.5, 0, 0])


class Point(TestTransform):

    def setUp(self):
        super().setUp(trans=transform.Point(types.arraydata([1., 2., 3.])), linear=numpy.zeros((3, 0)), offset=[1., 2., 3.])


class swaps(TestCase):

    def setUp(self):
        self.chain = transform.SimplexChild(3, 2), transform.SimplexEdge(3, 0), transform.SimplexChild(2, 1), transform.SimplexChild(2, 1), transform.SimplexEdge(2, 0)

    def assertMidpoint(self, chain):
        midpoint = transform.apply(self.chain, numpy.array([.5]))
        self.assertEqual(midpoint.tolist(), [0, 0.9375, 0.0625])

    def test_canonical(self):
        canonical = transform.SimplexEdge(3, 0), transform.SimplexEdge(2, 0), transform.SimplexChild(1, 0), transform.SimplexChild(1, 0), transform.SimplexChild(1, 0)
        self.assertEqual(transform.canonical(self.chain), canonical)
        self.assertMidpoint(canonical)
        self.assertTrue(transform.iscanonical(canonical))

    def test_promote(self):
        promote = transform.SimplexEdge(3, 0), transform.SimplexChild(2, 1), transform.SimplexChild(2, 1), transform.SimplexChild(2, 1), transform.SimplexEdge(2, 0)
        self.assertEqual(transform.promote(self.chain, 2), promote)
        self.assertMidpoint(promote)
        self.assertFalse(transform.iscanonical(promote))

    def test_uppermost(self):
        uppermost = transform.SimplexChild(3, 2), transform.SimplexChild(3, 2), transform.SimplexChild(3, 2), transform.SimplexEdge(3, 0), transform.SimplexEdge(2, 0)
        self.assertEqual(transform.uppermost(self.chain), uppermost)
        self.assertMidpoint(uppermost)
        self.assertFalse(transform.iscanonical(uppermost))


del TestTransform, TestInvertible, TestUpdim
