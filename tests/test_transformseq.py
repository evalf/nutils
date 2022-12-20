from nutils import types, transform
from nutils.testing import TestCase, parametrize
import nutils.transformseq
import nutils.element
import nutils.evaluable
from nutils.elementseq import References
import unittest
import numpy
import itertools
import functools


class Common:

    def test_todims(self):
        self.assertEqual(self.seq.todims, self.checktodims)
        for trans in self.seq:
            self.assertEqual(trans[0].todims, self.checktodims)

    def test_fromdims(self):
        self.assertEqual(self.seq.fromdims, self.checkfromdims)
        for trans in self.seq:
            self.assertEqual(trans[-1].fromdims, self.checkfromdims)

    def test_len(self):
        self.assertEqual(len(self.seq), len(self.check))

    def test_getitem_scalar_pos(self):
        for i in range(len(self.check)):
            self.assertEqual(self.seq[i], self.check[i])

    def test_getitem_scalar_neg(self):
        for i in range(-len(self.check), 0):
            self.assertEqual(self.seq[i], self.check[i])

    def test_getitem_scalar_invalid(self):
        for i in [len(self.check), -len(self.check)-1]:
            with self.assertRaises(IndexError):
                self.seq[i]

    def test_getitem_slice(self):
        for i in range(len(self.check)):
            for j in range(i, len(self.check)+1):
                with self.subTest('{}:{}'.format(i, j)):
                    self.assertEqual(tuple(self.seq[i:j]), tuple(self.check[i:j]))

    def test_getitem_intarray(self):
        for mask in itertools.product(*[[False, True]]*len(self.check)):
            mask = numpy.array(mask, dtype=bool)
            indices, = numpy.where(mask)
            with self.subTest(tuple(indices)):
                self.assertEqual(tuple(self.seq[indices]), tuple(self.check[i] for i in indices))

    def test_getitem_intarray_outofbounds(self):
        for i in [-1, len(self.check)]:
            with self.assertRaises(IndexError):
                self.seq[numpy.array([i], dtype=int)]

    def test_getitem_intarray_invalidndim(self):
        with self.assertRaises(IndexError):
            self.seq[numpy.array([[0]], dtype=int)]

    def test_getitem_boolarray(self):
        for mask in itertools.product(*[[False, True]]*len(self.check)):
            mask = numpy.array(mask, dtype=bool)
            indices, = numpy.where(mask)
            with self.subTest(tuple(indices)):
                self.assertEqual(tuple(self.seq[mask]), tuple(self.check[i] for i in indices))

    def test_getitem_boolarray_invalidshape(self):
        with self.assertRaises(IndexError):
            self.seq[numpy.array([True]*(len(self.check)+1), dtype=bool)]
        with self.assertRaises(IndexError):
            self.seq[numpy.array([[True]*len(self.check)], dtype=bool)]

    def test_iter(self):
        self.assertEqual(tuple(self.seq), tuple(self.check))

    def test_add(self):
        self.assertEqual(tuple(self.seq+nutils.transformseq.EmptyTransforms(self.checktodims, self.checkfromdims)), tuple(self.check))
        self.assertEqual(tuple(self.seq+self.seq), tuple(self.check)+tuple(self.check))

    def test_index_with_tail(self):
        assert len(self.check) == len(self.checkrefs)
        for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
            self.assertEqual(self.seq.index_with_tail(trans), (i, ()))
            for ctrans in ref.child_transforms:
                self.assertEqual(self.seq.index_with_tail(trans+(ctrans,)), (i, (ctrans,)))
            if self.checkfromdims > 0:
                for etrans in ref.edge_transforms:
                    for shuffle in lambda t: t, nutils.transform.canonical:
                        self.assertEqual(self.seq.index_with_tail(shuffle(trans+(etrans,))), (i, (etrans,)))

    def test_index_with_tail_missing(self):
        for trans in self.checkmissing:
            with self.assertRaises(ValueError):
                self.seq.index_with_tail(trans)

    def test_index(self):
        for i, trans in enumerate(self.check):
            self.assertEqual(self.seq.index(trans), i)

    def test_index_missing(self):
        for trans in self.checkmissing:
            with self.assertRaises(ValueError):
                self.seq.index(trans)
        assert len(self.check) == len(self.checkrefs)
        for trans, ref in zip(self.check, self.checkrefs):
            for ctrans in ref.child_transforms:
                with self.assertRaises(ValueError):
                    self.seq.index(trans+(ctrans,))

    def test_contains_with_tail(self):
        assert len(self.check) == len(self.checkrefs)
        for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
            self.assertEqual(self.seq.index_with_tail(trans), (i, ()))
            for ctrans in ref.child_transforms:
                self.assertTrue(self.seq.contains_with_tail(trans+(ctrans,)))
            if self.checkfromdims > 0:
                for etrans in ref.edge_transforms:
                    for shuffle in lambda t: t, nutils.transform.canonical:
                        self.assertTrue(self.seq.contains_with_tail(shuffle(trans+(etrans,))))

    def test_contains_with_tail_missing(self):
        for trans in self.checkmissing:
            self.assertFalse(self.seq.contains_with_tail(trans))

    def test_contains(self):
        for i, trans in enumerate(self.check):
            self.assertTrue(self.seq.contains(trans))

    def test_contains_missing(self):
        for trans in self.checkmissing:
            self.assertFalse(self.seq.contains(trans))
        assert len(self.check) == len(self.checkrefs)
        for trans, ref in zip(self.check, self.checkrefs):
            for ctrans in ref.child_transforms:
                self.assertFalse(self.seq.contains(trans+(ctrans,)))

    def test_refined(self):
        refined = self.seq.refined(self.checkrefs)
        assert len(self.check) == len(self.checkrefs)
        ctransforms = (trans+(ctrans,) for trans, ref in zip(self.check, self.checkrefs) for ctrans in ref.child_transforms)
        for i, trans in enumerate(ctransforms):
            self.assertEqual(refined.index(trans), i)


class Edges:

    def test_edges(self):
        edges = self.seq.edges(self.checkrefs)
        assert len(self.check) == len(self.checkrefs)
        etransforms = (trans+(etrans,) for trans, ref in zip(self.check, self.checkrefs) for etrans in ref.edge_transforms)
        for i, trans in enumerate(etransforms):
            self.assertEqual(edges.index(trans), i)


point = nutils.element.PointReference()
line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()

l1, x1, r1 = nutils.transform.Index(1, 100), nutils.transform.Index(1, 101), nutils.transform.Index(1, 102)

i10 = nutils.transform.Index(1, 0)
i11 = nutils.transform.Index(1, 1)
i12 = nutils.transform.Index(1, 2)
i13 = nutils.transform.Index(1, 3)
i14 = nutils.transform.Index(1, 4)
i20 = nutils.transform.Index(2, 0)
i21 = nutils.transform.Index(2, 1)
i22 = nutils.transform.Index(2, 2)
i23 = nutils.transform.Index(2, 3)

c0, c1 = line.child_transforms
e0, e1 = line.edge_transforms

l2, x2, r2 = nutils.transform.Index(2, 100), nutils.transform.Index(2, 101), nutils.transform.Index(2, 102)

c00, c01, c10, c11 = square.child_transforms
e00, e01, e10, e11 = square.edge_transforms


class EmptyTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.EmptyTransforms(todims=1, fromdims=1)
        self.check = ()
        self.checkmissing = (l1, i10), (x1, i14), (r1, i10)
        self.checkrefs = References.empty(1)
        self.checktodims = 1
        self.checkfromdims = 1


class PlainTransforms1D(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11), (x1, i12), (x1, i13)), 1, 1)
        self.check = (x1, i10), (x1, i11), (x1, i12), (x1, i13)
        self.checkmissing = (l1, i10), (x1, i14), (r1, i10)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class PlainTransforms2D(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.PlainTransforms(((x2, i20, i20), (x2, i20, i21), (x2, i21, i20), (x2, i21, i21)), 2, 2)
        self.check = (x2, i20, i20), (x2, i20, i21), (x2, i21, i20), (x2, i21, i21)
        self.checkmissing = (l2, i20, i20), (x2, i20, i22), (x2, i21, i22), (r2, i20, i20)
        self.checkrefs = References.from_iter((square, square, triangle, triangle), 2)
        self.checktodims = 2
        self.checkfromdims = 2


class MaskedTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.MaskedTransforms(nutils.transformseq.PlainTransforms(((x2, i20, i20), (x2, i20, i21), (x2, i21, i20), (x2, i21, i21)), 2, 2), types.arraydata([0, 2]))
        self.check = (x2, i20, i20), (x2, i21, i20)
        self.checkmissing = (l2, i20, i20), (x2, i20, i21), (x2, i21, i21), (x2, i20, i22), (x2, i21, i22), (r2, i20, i20)
        self.checkrefs = References.from_iter((square, triangle), 2)
        self.checktodims = 2
        self.checkfromdims = 2


class ReorderedTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.ReorderedTransforms(nutils.transformseq.PlainTransforms(((x2, i20, i20), (x2, i20, i21), (x2, i21, i20), (x2, i21, i21)), 2, 2), types.arraydata([0, 2, 3, 1]))
        self.check = (x2, i20, i20), (x2, i21, i20), (x2, i21, i21), (x2, i20, i21)
        self.checkmissing = (l2, i20, i20), (x2, i20, i22), (x2, i21, i22), (r2, i20, i20)
        self.checkrefs = References.uniform(square, 4)
        self.checktodims = 2
        self.checkfromdims = 2


class DerivedTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.DerivedTransforms(nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1), References.uniform(line, 2), 'child_transforms', 1)
        self.check = (x1, i10, c0), (x1, i10, c1), (x1, i11, c0), (x1, i11, c1)
        self.checkmissing = (l1, i10), (x1, i10), (x1, i11), (r1, i10)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class UniformDerivedTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.UniformDerivedTransforms(nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1), line, 'child_transforms', 1)
        self.check = (x1, i10, c0), (x1, i10, c1), (x1, i11, c0), (x1, i11, c1)
        self.checkmissing = (l1, i10), (x1, i10), (x1, i11), (r1, i10)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class ChainedTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.ChainedTransforms((nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1), nutils.transformseq.PlainTransforms(((x1, i12), (x1, i13)), 1, 1)))
        self.check = (x1, i10), (x1, i11), (x1, i12), (x1, i13)
        self.checkmissing = (l1, i10), (x1, i14), (r1, i10)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class StructuredTransforms1D(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.DimAxis(0, 4, 0, False),), 0)
        self.check = (x1, i10), (x1, i11), (x1, i12), (x1, i13)
        self.checkmissing = (l1, i10), (x1,), (x1, i14), (r1, i10), (x1, c1)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class StructuredTransforms1DRefined(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.DimAxis(0, 4, 0, False),), 1)
        self.check = (x1, i10, c0), (x1, i10, c1), (x1, i11, c0), (x1, i11, c1)
        self.checkmissing = (l1, i10), (x1,), (x1, i10), (x1, i11), (x1, i10, i11), (r1, i10)
        self.checkrefs = References.uniform(line, 4)
        self.checktodims = 1
        self.checkfromdims = 1


class StructuredTransforms1DLeft(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(3, 4, 9, 0, False),), 0)
        self.check = (x1, i13, e1),
        self.checkmissing = (x1, i10, e0), (x1, i12, e0), (x1, i14, e0)
        self.checkrefs = References.uniform(point, 1)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms1DRight(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(2, 3, 9, 0, True),), 0)
        self.check = (x1, i12, e0),
        self.checkmissing = (x1, i10, e0), (x1, i13, e1), (x1, i14, e0)
        self.checkrefs = References.uniform(point, 1)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms1DInterfacesLeft(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(1, 4, 9, 0, False),), 0)
        self.check = (x1, i11, e1), (x1, i12, e1), (x1, i13, e1)
        self.checkmissing = (x1, i10, e1), (x1, i10, e0), (x1, i11, e0), (x1, i12, e0), (x1, i13, e0)
        self.checkrefs = References.uniform(point, 3)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms1DInterfacesRight(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(0, 3, 9, 0, True),), 0)
        self.check = (x1, i10, e0), (x1, i11, e0), (x1, i12, e0)
        self.checkmissing = (x1, i13, e0), (x1, i10, e1), (x1, i11, e1), (x1, i12, e1), (x1, i13, e1)
        self.checkrefs = References.uniform(point, 3)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms1DPeriodicInterfacesLeft(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(1, 5, 4, 0, False),), 0)
        self.check = (x1, i11, e1), (x1, i12, e1), (x1, i13, e1), (x1, i10, e1)
        self.checkmissing = (x1, i10, e0), (x1, i11, e0), (x1, i12, e0), (x1, i13, e0), (x1, i14, e0)
        self.checkrefs = References.uniform(point, 4)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms1DPeriodicInterfacesRight(TestCase, Common):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x1, (nutils.transformseq.IntAxis(0, 4, 4, 0, True),), 0)
        self.check = (x1, i10, e0), (x1, i11, e0), (x1, i12, e0), (x1, i13, e0)
        self.checkmissing = (x1, i10, e1), (x1, i11, e1), (x1, i12, e1), (x1, i13, e1), (x1, i14, e1)
        self.checkrefs = References.uniform(point, 4)
        self.checktodims = 1
        self.checkfromdims = 0


class StructuredTransforms2D(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x2, (nutils.transformseq.DimAxis(0, 2, 0, False), nutils.transformseq.DimAxis(2, 4, 0, False)), 0)
        self.check = (x2, i20, i22), (x2, i20, i23), (x2, i21, i22), (x2, i21, i23)
        self.checkmissing = (x2,), (x2, i20), (x2, i20, i20), (x2, i20, i21), (x2, i21, i20), (x2, i21, i21)
        self.checkrefs = References.uniform(square, 4)
        self.checktodims = 2
        self.checkfromdims = 2


class StructuredTransforms2DRefined(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.StructuredTransforms(x2, (nutils.transformseq.DimAxis(0, 2, 0, False), nutils.transformseq.DimAxis(2, 4, 0, False)), 1)
        self.check = (x2, i20, i21, c00), (x2, i20, i21, c01), (x2, i20, i21, c10), (x2, i20, i21, c11)
        self.checkmissing = (x2, i20, i20, c00),
        self.checkrefs = References.uniform(square, 4)
        self.checktodims = 2
        self.checkfromdims = 2


class IndexTransforms(TestCase, Common, Edges):
    def setUp(self):
        super().setUp()
        self.seq = nutils.transformseq.IndexTransforms(ndims=2, length=4, offset=2)
        self.check = [(nutils.transform.Index(2, i),) for i in range(2, 6)]
        self.checkmissing = (nutils.transform.Index(1, 0),), (nutils.transform.Index(2, -1),), (nutils.transform.Index(2, 6,),)
        self.checkrefs = References.uniform(triangle, 4)
        self.checktodims = 2
        self.checkfromdims = 2


class exceptions(TestCase):

    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, '^invalid dimensions$'):
            nutils.transformseq.EmptyTransforms(1, -1)
        with self.assertRaisesRegex(ValueError, '^invalid dimensions$'):
            nutils.transformseq.EmptyTransforms(1, 2)

    def test_PlainTransforms_invalid_todims(self):
        with self.assertRaisesRegex(ValueError, 'expected transforms with todims=2, but got .*'):
            nutils.transformseq.PlainTransforms(((x1, c0), (x1, c1)), 2, 1)

    def test_PlainTransforms_invalid_fromdims(self):
        with self.assertRaisesRegex(ValueError, 'expected transforms with fromdims=2, but got .*'):
            nutils.transformseq.PlainTransforms(((x2, e00), (x2, e11)), 2, 2)

    def test_PlainTransforms_multiple_fromdims(self):
        with self.assertRaisesRegex(ValueError, 'expected transforms with fromdims=2, but got .*'):
            nutils.transformseq.PlainTransforms(tuple(transform.canonical((x2, *tail)) for tail in ((c00, e00), (c01, i20, i20))), 2, 2)
            nutils.transformseq.PlainTransforms(tuple(transform.canonical((x2, *tail)) for tail in ((c00, e00), (c01, i20, i20))), 2, 2)

    def test_DerivedTransforms_length_mismatch(self):
        transforms = nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1)
        references = References.uniform(line, 3)
        with self.assertRaisesRegex(ValueError, '`parent` and `parent_references` should have the same length'):
            nutils.transformseq.DerivedTransforms(transforms, references, 'child_transforms', 1)

    def test_DerivedTransforms_ndims_mismatch(self):
        transforms = nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1)
        references = References.uniform(square, 2)
        with self.assertRaisesRegex(ValueError, '`parent` and `parent_references` have different dimensions'):
            nutils.transformseq.DerivedTransforms(transforms, references, 'child_transforms', 1)

    def test_UniformDerivedTransforms_ndims_mismatch(self):
        transforms = nutils.transformseq.PlainTransforms(((x1, i10), (x1, i11)), 1, 1)
        with self.assertRaisesRegex(ValueError, '`parent` and `parent_reference` have different dimensions'):
            nutils.transformseq.UniformDerivedTransforms(transforms, square, 'child_transforms', 1)

    def test_ChainedTransforms_no_items(self):
        with self.assertRaisesRegex(ValueError, 'Empty chain.'):
            nutils.transformseq.ChainedTransforms(())

    def test_ChainedTransforms_multiple_todims(self):
        a = nutils.transformseq.PlainTransforms(tuple(transform.canonical((x2, c00, e)) for e in (e00, e01)), 2, 1)
        b = nutils.transformseq.PlainTransforms(((x1, c0), (x1, c1)), 1, 1)
        with self.assertRaisesRegex(ValueError, 'Cannot chain Transforms with different todims.'):
            nutils.transformseq.ChainedTransforms((a, b))

    def test_ChainedTransforms_multiple_fromdims(self):
        a = nutils.transformseq.PlainTransforms(tuple(transform.canonical((x2, c00, e)) for e in (e00, e01)), 2, 1)
        b = nutils.transformseq.PlainTransforms(((x2, c10), (x2, c11)), 2, 2)
        with self.assertRaisesRegex(ValueError, 'Cannot chain Transforms with different fromdims.'):
            nutils.transformseq.ChainedTransforms((a, b))

# vim:sw=2:sts=2:et
