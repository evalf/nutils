from nutils import *
from . import *

@parametrize
class elem(TestCase):

  def setUp(self):
    super().setUp()
    self.ref = element.getsimplex(self.ndims[0])
    for ndim in self.ndims[1:]:
      self.ref *= element.getsimplex(ndim)

  def test_ndims(self):
    self.assertEqual(self.ref.ndims, sum(self.ndims))

  def test_centroid(self):
    numpy.testing.assert_almost_equal(self.ref.centroid, self.exactcentroid, decimal=15)

  @parametrize.enable_if(lambda ndims, **kwargs: sum(ndims) >= 1)
  def test_children(self):
    childvol = sum(abs(trans.det) * child.volume for trans, child in self.ref.children)
    numpy.testing.assert_almost_equal(childvol, self.ref.volume)

  @parametrize.enable_if(lambda ndims, **kwargs: sum(ndims) >= 1)
  def test_childdivide(self):
    for n in 1, 2, 3:
      with self.subTest(n=n):
        points, weights = self.ref.getischeme('vertex{}'.format(n))
        for (ctrans, cref), vals in zip(self.ref.children, self.ref.child_divide(points, n)):
          cpoints, cweights = cref.getischeme('vertex{}'.format(n-1))
          numpy.testing.assert_equal(ctrans.apply(cpoints), vals)

  @parametrize.enable_if(lambda ndims, **kwargs: sum(ndims) >= 2)
  def test_ribbons(self):
    self.ref.ribbons

elem('point', ndims=[0], exactcentroid=numpy.zeros((0,)))
elem('line', ndims=[1], exactcentroid=[.5])
elem('triangle', ndims=[2], exactcentroid=[1/3]*2)
elem('tetrahedron', ndims=[3], exactcentroid=[1/4]*3)
elem('square', ndims=[1,1], exactcentroid=[.5]*2)
elem('hexagon', ndims=[1,1,1], exactcentroid=[.5]*3)
elem('prism1', ndims=[2,1], exactcentroid=[1/3,1/3,1/2])
elem('prism2', ndims=[1,2], exactcentroid=[1/2,1/3,1/3])
