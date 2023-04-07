from nutils import element, points, transform, numeric
from nutils.testing import TestCase, parametrize
import numpy


class gauss(TestCase):

    def test_line(self):
        line = element.getsimplex(1)
        for degree in range(1, 8):
            points = line.getpoints('gauss', degree)
            self.assertEqual(points.npoints, degree//2+1)
            self.assertLess(abs(points.weights.sum()-1), 2e-15)

    def test_quad(self):
        quad = element.getsimplex(1)**2
        for degree in range(1, 8):
            points = quad.getpoints('gauss', degree)
            self.assertEqual(points.npoints, (degree//2+1)**2)
            self.assertLess(abs(points.weights.sum()-1), 2e-15)

    def test_hexahedron(self):
        hex = element.getsimplex(1)**3
        for degree in range(1, 8):
            points = hex.getpoints('gauss', degree)
            self.assertEqual(points.npoints, (degree//2+1)**3)
            self.assertLess(abs(points.weights.sum()-1), 2e-15)

    def test_triangle(self):
        tri = element.getsimplex(2)
        for degree in range(1, 8):
            points = tri.getpoints('gauss', degree)
            self.assertLess(abs(points.weights.sum()-.5), 2e-15)

    def test_pyramid(self):
        pyramid12 = element.getsimplex(1)*element.getsimplex(2)
        pyramid21 = element.getsimplex(2)*element.getsimplex(1)
        for degree in range(1, 8):
            points12 = pyramid12.getpoints('gauss', degree)
            points21 = pyramid21.getpoints('gauss', degree)
            self.assertEqual(points12.npoints, points21.npoints)
            self.assertLess(abs(points12.weights.sum()-.5), 2e-15)
            self.assertLess(abs(points21.weights.sum()-.5), 2e-15)

    def test_tetrahedron(self):
        tet = element.getsimplex(3)
        for degree in range(1, 9):
            points = tet.getpoints('gauss', degree)
            self.assertLess(abs(points.weights.sum()-1/6), 2e-15)


class bezier(TestCase):

    def test_line(self):
        line = element.getsimplex(1)
        for n in range(2, 8):
            bezier = line.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, n)
            self.assertEqual(len(bezier.tri), n-1)
            self.assertEqual(len(bezier.hull), 2)

    def test_quad(self):
        quad = element.getsimplex(1)**2
        for n in range(2, 8):
            bezier = quad.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, n**2)
            self.assertEqual(len(bezier.tri), 2*(n-1)**2)
            self.assertEqual(len(bezier.hull), 4*(n-1))

    def test_hexahedron(self):
        hex = element.getsimplex(1)**3
        for n in range(2, 8):
            bezier = hex.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, n**3)
            self.assertEqual(len(bezier.tri), 6*(n-1)**3)
            self.assertEqual(len(bezier.hull), 12*(n-1)**2)

    def test_triangle(self):
        tri = element.getsimplex(2)
        for n in range(2, 8):
            bezier = tri.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, (n*(n+1))//2)
            self.assertEqual(len(bezier.tri), (n-1)**2)
            self.assertEqual(len(bezier.hull), 3*(n-1))

    def test_tetrahedron(self):
        tet = element.getsimplex(3)
        for n in range(2, 8):
            bezier = tet.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, (n*(n+1)*(n+2))//6)
            self.assertEqual(len(bezier.tri), (n-1)**3)
            self.assertEqual(len(bezier.hull), 4*(n-1)**2)

    def test_pyramid(self):
        pyramid = element.getsimplex(1)*element.getsimplex(2)
        for n in range(2, 8):
            bezier = pyramid.getpoints('bezier', n)
            self.assertEqual(bezier.npoints, n*(n*(n+1))//2)
            self.assertEqual(len(bezier.tri), 3*(n-1)**3)
            self.assertEqual(len(bezier.hull), 8*(n-1)**2)
            fullhull = points.Points.hull.func(bezier).tolist() # contains additional internal faces for n >= 3
            for h in bezier.hull:  # assert that hull is a subset of fullfull
                self.assertIn(sorted(h), fullhull)


class trimmed(TestCase):

    def setUp(self):
        super().setUp()
        quad = element.getsimplex(1)**2
        levels = numeric.overlapping(numpy.arange(-1, 16, 2), n=5)  # linear ramp cutting at x + y == .125
        trimmed = quad.trim(levels.ravel(), maxrefine=2, ndivisions=16)
        self.bezier = trimmed.getpoints('bezier', 5)
        self.gauss = trimmed.getpoints('gauss', 3)
        self.uniform = trimmed.getpoints('uniform', 3)

    def test_type(self):
        for pnt in self.bezier, self.gauss, self.uniform:
            self.assertIsInstance(pnt, points.ConcatPoints)
            for i, subpoints in enumerate(pnt.allpoints):
                self.assertIsInstance(subpoints, points.TransformPoints)
                self.assertIsInstance(subpoints.points, points.TensorPoints if i else points.ConcatPoints)

    def test_weights(self):
        exact = 1-.5*.125**2
        for pnt in self.gauss, self.uniform:
            self.assertLess(abs(pnt.weights.sum()-exact), 1e-15)

    def test_points(self):
        self.assertEqual(self.bezier.npoints, 26)
        for x in [0., .25, .5, .75, 1.]:
            for y in [0., .25, .5, .75, 1.]:
                if x or y:
                    self.assertIn([x, y], self.bezier.coords.tolist())
        self.assertIn([0., .125], self.bezier.coords.tolist())
        self.assertIn([.125, 0.], self.bezier.coords.tolist())

    def test_tri(self):
        self.assertEqual(len(self.bezier.tri), 33)

    def test_hull(self):
        self.assertEqual(len(self.bezier.hull), 17)
