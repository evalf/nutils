from nutils import *
from . import *
import itertools

@parametrize
class elem(TestCase):

  def test_ndims(self):
    self.assertEqual(self.ref.ndims, len(self.exactcentroid))

  def test_centroid(self):
    numpy.testing.assert_almost_equal(self.ref.centroid, self.exactcentroid, decimal=15)

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
  def test_children(self):
    childvol = sum(abs(trans.det) * child.volume for trans, child in self.ref.children)
    numpy.testing.assert_almost_equal(childvol, self.ref.volume)

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
  def test_childdivide(self):
    for n in 1, 2, 3:
      with self.subTest(n=n):
        points, weights = self.ref.getischeme('vertex{}'.format(n))
        for (ctrans, cref), vals in zip(self.ref.children, self.ref.child_divide(points, n)):
          if cref:
            cpoints, cweights = cref.getischeme('vertex{}'.format(n-1))
            numpy.testing.assert_equal(ctrans.apply(cpoints), vals)

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
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

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
  def test_connectivity(self):
    for ichild, edges in enumerate(self.ref.connectivity):
      for iedge, ioppchild in enumerate(edges):
        if ioppchild != -1:
          ioppedge = self.ref.connectivity[ioppchild].index(ichild)
          self.assertEqual(
            self.ref.child_transforms[ichild] * self.ref.child_refs[ichild].edge_transforms[iedge],
            (self.ref.child_transforms[ioppchild] * self.ref.child_refs[ioppchild].edge_transforms[ioppedge]).flipped)

  @parametrize.enable_if(lambda ref, **kwargs: isinstance(ref, (element.SimplexReference, element.TensorReference)))
  def test_dof_transpose_map(self):
    nverts = []
    ref = self.ref
    while isinstance(ref, element.TensorReference):
      nverts.append(ref.ref1.nverts)
      ref = ref.ref2
    nverts.append(ref.nverts)
    for perm_ref_order, *perms_refs in itertools.product(*(itertools.permutations(range(n)) for n in [len(nverts)]+nverts)):
      j = numpy.arange(self.ref.nverts).reshape(nverts)
      for k, perm in enumerate(perms_refs):
        j = numpy.take(j, perm, axis=k)
      j = tuple(numpy.transpose(j, perm_ref_order).ravel())
      with self.subTest(perm_ref_order=perm_ref_order, perms_refs=perms_refs):
        self.assertEqual(j, tuple(self.ref.get_dof_transpose_map(1, j)))

elem('point', ref=element.PointReference(), exactcentroid=numpy.zeros((0,)))
elem('point2', ref=element.PointReference()**2, exactcentroid=numpy.zeros((0,)))
elem('line', ref=element.LineReference(), exactcentroid=[.5])
elem('triangle', ref=element.TriangleReference(), exactcentroid=[1/3]*2)
elem('tetrahedron', ref=element.TetrahedronReference(), exactcentroid=[1/4]*3)
elem('square', ref=element.LineReference()**2, exactcentroid=[.5]*2)
elem('hexagon', ref=element.LineReference()**3, exactcentroid=[.5]*3)
elem('prism1', ref=element.TriangleReference()*element.LineReference(), exactcentroid=[1/3,1/3,1/2])
elem('prism2', ref=element.LineReference()*element.TriangleReference(), exactcentroid=[1/2,1/3,1/3])
elem('withchildren1', ref=element.WithChildrenReference(element.LineReference()**2, [element.LineReference()**2,element.EmptyReference(2),element.EmptyReference(2),element.EmptyReference(2)]), exactcentroid=[1/4,1/4])
elem('withchildren2', ref=element.WithChildrenReference(element.LineReference()**2, [element.LineReference()**2,element.LineReference()**2,element.EmptyReference(2),element.EmptyReference(2)]), exactcentroid=[1/4,1/2])
