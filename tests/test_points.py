from nutils import *
from nutils.testing import *

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
      fullhull = super(points.TensorPoints, bezier).hull # contains additional internal faces for n >= 3
      for h in bezier.hull: # assert that hull is a subset of fullfull
        self.assertIn(h, fullhull)

@parametrize
class cone(TestCase):

  def setUp(self):
    if self.shape == 'square':
      self.edgeref = element.getsimplex(1)**2
    elif self.shape == 'triangle':
      self.edgeref = element.getsimplex(2)
    else:
      raise Exception('invalid shape: {!r}'.format(self.shape))
    self.etrans = transform.Updim(linear=[[-1.,0],[0,-3],[0,0]], offset=[1.,3,1], isflipped=False)
    self.cone = element.Cone(edgeref=self.edgeref, etrans=self.etrans, tip=[1.,3,0])

  def test_volume(self):
    numpy.testing.assert_almost_equal(actual=self.cone.volume, desired=self.edgeref.volume)

  def _test_points(self, *args):
    points = self.cone.getpoints(*args)
    if hasattr(points, 'weights'):
      numpy.testing.assert_almost_equal(actual=self.cone.volume, desired=points.weights.sum())
    # check that all points lie within pyramid/prism
    x, y, z = points.coords.T
    self.assertTrue(numpy.all(numpy.greater_equal(x, 1-z) & numpy.less_equal(x, 1) & numpy.greater_equal(y, 1-z) & numpy.less_equal(y, 3)))
    if self.shape == 'triangle':
      self.assertTrue(numpy.less_equal(2-x-y/3, z).all())

  def test_gauss(self):
    self._test_points('gauss', 3)

  def test_uniform(self):
    self._test_points('uniform', 3)

cone(shape='square')
cone(shape='triangle')


class trimmed(TestCase):

  def setUp(self):
    quad = element.getsimplex(1)**2
    levels = numeric.overlapping(numpy.arange(-1, 16, 2), n=5) # linear ramp cutting at x + y == .125
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
    self.assertEqual(self.bezier.npoints, 27)
    for x in [0., .25, .5, .75, 1.]:
      for y in [0., .25, .5, .75, 1.]:
        if x or y:
          self.assertIn(types.frozenarray([x,y]), self.bezier.coords)
    self.assertIn(types.frozenarray([0.,.125]), self.bezier.coords)
    self.assertIn(types.frozenarray([.0625,.0625]), self.bezier.coords)
    self.assertIn(types.frozenarray([.125,0.]), self.bezier.coords)

  def test_tri(self):
    self.assertEqual(len(self.bezier.tri), 34)

  def test_hull(self):
    self.assertEqual(len(self.bezier.hull), 18)
