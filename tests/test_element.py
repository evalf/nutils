from nutils import *
from nutils.testing import *
import itertools

@parametrize
class elem(TestCase):

  def test_ndims(self):
    self.assertEqual(self.ref.ndims, len(self.exactcentroid))

  def test_centroid(self):
    numpy.testing.assert_almost_equal(self.ref.centroid, self.exactcentroid, decimal=15)

  @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
  def test_children(self):
    childvol = sum(abs(trans.det) * child.volume for trans, child in self.ref.children)
    numpy.testing.assert_almost_equal(childvol, self.ref.volume)

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
  def test_edges(self):
    self.ref.check_edges(print=self.fail)
    for etrans in self.ref.edge_transforms:
      assert etrans.flipped.flipped == etrans

  @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
  def test_childdivide(self):
    for n in 1, 2, 3:
      with self.subTest(n=n):
        points, weights = self.ref.getischeme('vertex{}'.format(n))
        for (ctrans, cref), vals in zip(self.ref.children, self.ref.child_divide(points, n)):
          if cref:
            cpoints, cweights = cref.getischeme('vertex{}'.format(n-1))
            numpy.testing.assert_equal(ctrans.apply(cpoints), vals)

  @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
  def test_swap(self):
    for iedge, (etrans, eref) in enumerate(self.ref.edges):
      for ichild, (ctrans, cref) in enumerate(eref.children):
        swapped_up = etrans.swapup(ctrans)
        self.assertNotEqual(swapped_up, None)
        ctrans_, etrans_ = swapped_up
        self.assertEqual(etrans * ctrans, ctrans_ * etrans_)
        swapped_down = etrans_.swapdown(ctrans_)
        self.assertEqual(swapped_down, (etrans, ctrans))

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 2)
  def test_ribbons(self):
    self.ref.ribbons

  @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
  def test_connectivity(self):
    for ichild, edges in enumerate(self.ref.connectivity):
      for iedge, ioppchild in enumerate(edges):
        if ioppchild != -1:
          self.assertIn(ichild, self.ref.connectivity[ioppchild])
          ioppedge = self.ref.connectivity[ioppchild].index(ichild)
          self.assertEqual(
            self.ref.child_transforms[ichild] * self.ref.child_refs[ichild].edge_transforms[iedge],
            (self.ref.child_transforms[ioppchild] * self.ref.child_refs[ioppchild].edge_transforms[ioppedge]).flipped)

  @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
  def test_edgechildren(self):
    for iedge, edgechildren in enumerate(self.ref.edgechildren):
      for ichild, (jchild, jedge) in enumerate(edgechildren):
        self.assertEqual(
          self.ref.edge_transforms[iedge] * self.ref.edge_refs[iedge].child_transforms[ichild],
          self.ref.child_transforms[jchild] * self.ref.child_refs[jchild].edge_transforms[jedge])

  def test_inside(self):
    self.assertTrue(self.ref.inside(self.exactcentroid))
    if self.ref.ndims:
      self.assertFalse(self.ref.inside(-numpy.ones(self.ref.ndims)))

elem('point', ref=element.PointReference(), exactcentroid=numpy.zeros((0,)))
elem('point2', ref=element.PointReference()**2, exactcentroid=numpy.zeros((0,)))
elem('line', ref=element.LineReference(), exactcentroid=[.5])
elem('triangle', ref=element.TriangleReference(), exactcentroid=[1/3]*2)
elem('tetrahedron', ref=element.TetrahedronReference(), exactcentroid=[1/4]*3)
elem('square', ref=element.LineReference()**2, exactcentroid=[.5]*2)
elem('hexagon', ref=element.LineReference()**3, exactcentroid=[.5]*3)
elem('prism1', ref=element.TriangleReference()*element.LineReference(), exactcentroid=[1/3,1/3,1/2])
elem('prism2', ref=element.LineReference()*element.TriangleReference(), exactcentroid=[1/2,1/3,1/3])
line = element.LineReference()
quad = line**2
elem('withchildren1', ref=element.WithChildrenReference(quad, [quad,quad.empty,quad.empty,quad.empty]), exactcentroid=[1/4,1/4])
elem('withchildren2', ref=element.WithChildrenReference(quad, [quad,quad,quad.empty,quad.empty]), exactcentroid=[1/4,1/2])
elem('mosaic', ref=element.MosaicReference(quad, [line,line.empty,line,line.empty], [.25,.75]), exactcentroid=[2/3,2/3])
