from nutils import element, _util as util, types
from nutils.testing import TestCase, parametrize
import numpy, math


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
                        self.assertAllEqual(ctrans.apply(cpoints), vals)

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

    @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.MosaicReference) and ref.ndims >= 1)
    def test_connectivity(self):
        for ichild, edges in enumerate(self.ref.connectivity):
            for iedge, ioppchild in enumerate(edges):
                if ioppchild != -1:
                    self.assertIn(ichild, self.ref.connectivity[ioppchild])
                    ioppedge = util.index(self.ref.connectivity[ioppchild], ichild)
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

    @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.WithChildrenReference) and ref.ndims >= 1)
    def test_edge_vertices(self):
        for etrans, eref, everts in zip(self.ref.edge_transforms, self.ref.edge_refs, self.ref.edge_vertices):
            self.assertAllEqual(self.ref.vertices[everts], etrans.apply(eref.vertices))

    @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.WithChildrenReference) and ref.ndims >= 1)
    def test_simplices(self):
        volume = 0
        centroid = 0
        for simplex in self.ref.vertices[self.ref.simplices]:
            simplex_volume = numpy.linalg.det(simplex[1:] - simplex[0]) / math.factorial(self.ref.ndims)
            self.assertGreater(simplex_volume, 0)
            volume += simplex_volume
            centroid += simplex.mean(axis=0) * simplex_volume
        centroid /= volume
        self.assertAlmostEqual(volume, self.ref.volume)
        self.assertAllAlmostEqual(centroid, self.exactcentroid)

    @parametrize.enable_if(lambda ref, **kwargs: not isinstance(ref, element.WithChildrenReference) and ref.ndims >= 1)
    def test_simplex_transforms(self):
        for simplex, strans in zip(self.ref.vertices[self.ref.simplices], self.ref.simplex_transforms):
            self.assertAllEqual(strans.linear, (simplex[1:] - simplex[0]).T)
            self.assertAllEqual(strans.offset, simplex[0])

elem('point', ref=element.PointReference(), exactcentroid=numpy.zeros((0,)))
elem('point2', ref=element.PointReference()**2, exactcentroid=numpy.zeros((0,)))
elem('line', ref=element.LineReference(), exactcentroid=numpy.array([.5]))
elem('triangle', ref=element.TriangleReference(), exactcentroid=numpy.array([1/3]*2))
elem('tetrahedron', ref=element.TetrahedronReference(), exactcentroid=numpy.array([1/4]*3))
elem('square', ref=element.LineReference()**2, exactcentroid=numpy.array([.5]*2))
elem('hexagon', ref=element.LineReference()**3, exactcentroid=numpy.array([.5]*3))
elem('prism1', ref=element.TriangleReference()*element.LineReference(), exactcentroid=numpy.array([1/3, 1/3, 1/2]))
elem('prism2', ref=element.LineReference()*element.TriangleReference(), exactcentroid=numpy.array([1/2, 1/3, 1/3]))
line = element.LineReference()
quad = line**2
elem('withchildren1', ref=element.WithChildrenReference(quad, (quad, quad.empty, quad.empty, quad.empty)), exactcentroid=numpy.array([1/4, 1/4]))
elem('withchildren2', ref=element.WithChildrenReference(quad, (quad, quad, quad.empty, quad.empty)), exactcentroid=numpy.array([1/4, 1/2]))
elem('mosaic', ref=element.MosaicReference(quad, (line, line.empty, line, line.empty), types.arraydata([.25, .75])), exactcentroid=numpy.array([2/3, 2/3]))
