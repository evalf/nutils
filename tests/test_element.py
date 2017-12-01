from nutils import *
from . import *
import itertools

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

  def test_dof_transpose_map(self):
    nverts = tuple(element.getsimplex(ndim).nverts for ndim in self.ndims)
    i = numpy.arange(util.product(nverts)).reshape(nverts)
    for perm_ref_order, *perms_refs in itertools.product(*(itertools.permutations(range(n)) for n in (len(self.ndims),)+nverts)):
        j = i
        for k, perm in enumerate(perms_refs):
          j = numpy.take(j, perm, axis=k)
        j = tuple(numpy.transpose(j, perm_ref_order).ravel())
        with self.subTest(perm_ref_order=perm_ref_order, perms_refs=perms_refs):
          self.assertEqual(j, tuple(self.ref.get_dof_transpose_map(1, j)))

elem('point', ndims=[0], exactcentroid=numpy.zeros((0,)))
elem('point2', ndims=[0,0], exactcentroid=numpy.zeros((0,)))
elem('line', ndims=[1], exactcentroid=[.5])
elem('triangle', ndims=[2], exactcentroid=[1/3]*2)
elem('tetrahedron', ndims=[3], exactcentroid=[1/4]*3)
elem('square', ndims=[1,1], exactcentroid=[.5]*2)
elem('hexagon', ndims=[1,1,1], exactcentroid=[.5]*3)
elem('prism1', ndims=[2,1], exactcentroid=[1/3,1/3,1/2])
elem('prism2', ndims=[1,2], exactcentroid=[1/2,1/3,1/3])
