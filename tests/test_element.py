from nutils import *
from nutils.testing import *
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
  def test_edges(self):
    self.ref.check_edges(print=self.fail)

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
          self.assertIn(ichild, self.ref.connectivity[ioppchild])
          ioppedge = self.ref.connectivity[ioppchild].index(ichild)
          self.assertEqual(
            self.ref.child_transforms[ichild] * self.ref.child_refs[ichild].edge_transforms[iedge],
            (self.ref.child_transforms[ioppchild] * self.ref.child_refs[ioppchild].edge_transforms[ioppedge]).flipped)

  @parametrize.enable_if(lambda ref, **kwargs: ref.ndims >= 1)
  def test_edgechildren(self):
    for iedge, edgechildren in enumerate(self.ref.edgechildren):
      for ichild, (jchild, jedge) in enumerate(edgechildren):
        self.assertEqual(
          self.ref.edge_transforms[iedge] * self.ref.edge_refs[iedge].child_transforms[ichild],
          self.ref.child_transforms[jchild] * self.ref.child_refs[jchild].edge_transforms[jedge])

elem('point', ref=element.PointReference(), exactcentroid=numpy.zeros((0,)))
elem('point2', ref=element.PointReference()**2, exactcentroid=numpy.zeros((0,)))
elem('line', ref=element.LineReference(), exactcentroid=[.5])
elem('triangle', ref=element.TriangleReference(), exactcentroid=[1/3]*2)
elem('tetrahedron', ref=element.TetrahedronReference(), exactcentroid=[1/4]*3)
elem('square', ref=element.LineReference()**2, exactcentroid=[.5]*2)
elem('hexagon', ref=element.LineReference()**3, exactcentroid=[.5]*3)
elem('prism1', ref=element.TriangleReference()*element.LineReference(), exactcentroid=[1/3,1/3,1/2])
elem('prism2', ref=element.LineReference()*element.TriangleReference(), exactcentroid=[1/2,1/3,1/3])
quad = element.LineReference()**2
elem('withchildren1', ref=element.WithChildrenReference(quad, [quad,quad.empty,quad.empty,quad.empty]), exactcentroid=[1/4,1/4])
elem('withchildren2', ref=element.WithChildrenReference(quad, [quad,quad,quad.empty,quad.empty]), exactcentroid=[1/4,1/2])

class Element(TestCase):

  def test_withopposite(self):
    ref = element.PointReference()
    trans = transform.Identifier(0, 'a'),
    opp = transform.Identifier(0, 'b'),
    elem1 = element.Element(ref, trans)
    elem2 = element.Element(ref, opp)
    elem3 = elem1.withopposite(elem2)
    self.assertEqual(elem3.reference, ref)
    self.assertEqual(elem3.transform, trans)
    self.assertEqual(elem3.opposite, opp)

  def test_mul(self):
    lref, rref = element.LineReference(), element.LineReference()*element.LineReference()
    ltrans, rtrans = (transform.Identifier(1, 'l'),), (transform.Identifier(2, 'r'),)
    elem = element.Element(lref, ltrans)*element.Element(rref, rtrans)
    self.assertEqual(elem.reference, lref*rref)
    self.assertEqual(elem.transform, (transform.Bifurcate(ltrans, rtrans),))
    self.assertEqual(elem.opposite, elem.transform)

  def test_mul_iface(self):
    lref, rref = element.LineReference(), element.LineReference()*element.LineReference()
    ltrans, lopp, rtrans = (transform.Identifier(1, 'l'),), (transform.Identifier(1, 'o'),), (transform.Identifier(2, 'r'),)
    elem = element.Element(lref, ltrans, lopp)*element.Element(rref, rtrans)
    self.assertEqual(elem.reference, lref*rref)
    self.assertEqual(elem.transform, (transform.Bifurcate(ltrans, rtrans),))
    self.assertEqual(elem.opposite, (transform.Bifurcate(lopp, rtrans),))

  def test_vertices(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'), transform.Identity(1)
    elem = element.Element(ref, trans)
    self.assertEqual(ref.vertices.tolist(), [[0.],[1.]])

  def test_ndims(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'), transform.Identity(1)
    elem = element.Element(ref, trans)
    self.assertEqual(elem.ndims, 1)

  def test_nverts(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'), transform.Identity(1)
    elem = element.Element(ref, trans)
    self.assertEqual(elem.nverts, ref.nverts)

  def test_edges(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'),
    opp = transform.Identifier(1, 'b'),
    elem = element.Element(ref, trans, opp)
    self.assertEqual(elem.nedges, ref.nedges)
    self.assertEqual(elem.edges, [element.Element(eref, trans+(etrans,), opp+(etrans,)) if eref else None for etrans, eref in ref.edges])
    for i, (etrans, eref) in enumerate(ref.edges):
      self.assertEqual(elem.edge(i), element.Element(eref, trans+(etrans,), opp+(etrans,)) if eref else None)

  def test_children(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'),
    opp = transform.Identifier(1, 'b'),
    elem = element.Element(ref, trans, opp)
    self.assertEqual(elem.children, [element.Element(cref, trans+(ctrans,), opp+(ctrans,)) if cref else None for ctrans, cref in ref.children])

  def test_flipped(self):
    ref = element.LineReference()
    trans = transform.Identifier(1, 'a'),
    opp = transform.Identifier(1, 'b'),
    elem = element.Element(ref, trans, opp).flipped
    self.assertEqual(elem.reference, ref)
    self.assertEqual(elem.transform, opp)
    self.assertEqual(elem.opposite, trans)

  def test_simplices(self):
    ref = element.LineReference()*element.LineReference()
    trans = transform.Identifier(2, 'a'), transform.Identity(1)
    elem = element.Element(ref, trans)
    self.assertEqual(elem.simplices, [element.Element(sref, trans+(strans,), trans+(strans,)) for strans, sref in ref.simplices])
