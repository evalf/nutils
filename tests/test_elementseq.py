from nutils.testing import TestCase, parametrize
import nutils.elementseq
import nutils.element
import nutils.warnings
import nutils.types
import unittest
import numpy
import itertools
import operator
import functools

line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()


class Common:

    def test_fromdims(self):
        self.assertEqual(self.seq.ndims, self.checkndims)

    def test_len(self):
        self.assertEqual(len(self.seq), len(self.check))

    def test_iter(self):
        self.assertEqual(tuple(self.seq), tuple(self.check))

    def _op_or_meth(self, op, name):
        for name, func in (op, functools.partial(getattr(operator, op), self.seq)), (name, getattr(self.seq, name)):
            with self.subTest(name):
                yield func

    def test_getitem_invalid(self):
        with self.assertRaisesRegex(IndexError, 'invalid index: .*'):
            self.seq['a']

    def test_add_invalid(self):
        with self.assertRaises(TypeError):
            self.seq + 'a'

    def test_mul_invalid(self):
        with self.assertRaises(TypeError):
            self.seq * 'a'

    def test_get_pos(self):
        for get in self._op_or_meth('__getitem__', 'get'):
            for i in range(len(self.check)):
                self.assertEqual(get(i), self.check[i])

    def test_get_neg(self):
        for get in self._op_or_meth('__getitem__', 'get'):
            for i in range(-len(self.check), 0):
                self.assertEqual(get(i), self.check[i])

    def test_get_invalid(self):
        for get in self._op_or_meth('__getitem__', 'get'):
            for i in [len(self.check), -len(self.check)-1]:
                with self.assertRaises(IndexError):
                    get(i)

    def test_getitem_slice(self):
        for i in range(len(self.check)):
            for j in range(i, len(self.check)+1):
                with self.subTest('{}:{}'.format(i, j)):
                    self.assertEqual(tuple(self.seq[i:j]), tuple(self.check[i:j]))

    def test_take(self):
        for take in self._op_or_meth('__getitem__', 'take'):
            for mask in itertools.product(*[[False, True]]*len(self.check)):
                mask = numpy.array(mask, dtype=bool)
                indices, = numpy.where(mask)
                with self.subTest(tuple(indices)):
                    self.assertEqual(tuple(take(indices)), tuple(self.check[i] for i in indices))

    def test_take_outofbounds(self):
        for take in self._op_or_meth('__getitem__', 'take'):
            for i in [-1, len(self.check)]:
                with self.assertRaises(IndexError):
                    take(numpy.array([i], dtype=int))

    def test_take_invalidndim(self):
        for take in self._op_or_meth('__getitem__', 'take'):
            with self.assertRaises(IndexError):
                take(numpy.array([[0]], dtype=int))

    def test_compress(self):
        for compress in self._op_or_meth('__getitem__', 'compress'):
            for mask in itertools.product(*[[False, True]]*len(self.check)):
                mask = numpy.array(mask, dtype=bool)
                indices, = numpy.where(mask)
                with self.subTest(tuple(indices)):
                    self.assertEqual(tuple(compress(mask)), tuple(self.check[i] for i in indices))

    def test_compress_invalidshape(self):
        for compress in self._op_or_meth('__getitem__', 'compress'):
            with self.assertRaises(IndexError):
                compress(numpy.array([True]*(len(self.check)+1), dtype=bool))
            with self.assertRaises(IndexError):
                compress(numpy.array([[True]*len(self.check)], dtype=bool))

    def test_repeat(self):
        for repeat in self._op_or_meth('__mul__', 'repeat'):
            for m in range(4):
                with self.subTest(m):
                    self.assertEqual(tuple(repeat(m)), tuple(self.check)*m)

    def test_repeat_invalid_count(self):
        for repeat in self._op_or_meth('__mul__', 'repeat'):
            with self.assertRaises(ValueError):
                repeat(-1)

    def test_product(self):
        for product in self._op_or_meth('__mul__', 'product'):
            for other in [], [square], [square]*2, [square, triangle]:
                with self.subTest(other=other):
                    self.assertEqual(tuple(product(nutils.elementseq.References.from_iter(other, 2))), tuple(l*r for l, r in itertools.product(self.check, other)))

    def test_chain_empty(self):
        for chain in self._op_or_meth('__add__', 'chain'):
            self.assertEqual(tuple(chain(nutils.elementseq._Empty(self.checkndims))), tuple(self.check))

    def test_chain_self(self):
        for chain in self._op_or_meth('__add__', 'chain'):
            self.assertEqual(tuple(chain(self.seq)), tuple(self.check)*2)

    def test_chain_repeat(self):
        self.assertEqual(tuple(self.seq.chain(self.seq.repeat(2))), tuple(self.check)*3)
        self.assertEqual(tuple(self.seq.repeat(2).chain(self.seq)), tuple(self.check)*3)
        self.assertEqual(tuple(self.seq.repeat(2).chain(self.seq.repeat(3))), tuple(self.check)*5)

    def test_chain_other(self):
        other = tuple(self.check)[::-1]+(self.check[0],) if self.check else ()
        for chain in self._op_or_meth('__add__', 'chain'):
            self.assertEqual(tuple(chain(nutils.elementseq.References.from_iter(other, self.checkndims))), tuple(self.check)+other)

    def test_children(self):
        self.assertEqual(tuple(self.seq.children), tuple(itertools.chain.from_iterable(ref.child_refs for ref in self.check)))

    def test_edges(self):
        self.assertEqual(tuple(self.seq.edges), tuple(itertools.chain.from_iterable(ref.edge_refs for ref in self.check)))

    def test_getpoints(self):
        self.assertEqual(tuple(self.seq.getpoints('bezier', 2)), tuple(ref.getpoints('bezier', 2) for ref in self.check))


class Empty(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Empty(2)
        self.check = []
        self.checkndims = 2
        super().setUp()


class Plain(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Plain((square, triangle), 2)
        self.check = [square, triangle]
        self.checkndims = 2
        super().setUp()


class Uniform(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Uniform(square, 3)
        self.check = [square]*3
        self.checkndims = 2
        super().setUp()


class Take(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Take(nutils.elementseq.References.from_iter([square, triangle, square], 2), nutils.types.arraydata([1, 2]))
        self.check = [triangle, square]
        self.checkndims = 2
        super().setUp()


class Chain(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Chain(nutils.elementseq.References.from_iter([square, triangle], 2), nutils.elementseq.References.from_iter([square, square, triangle], 2))
        self.check = [square, triangle, square, square, triangle]
        self.checkndims = 2
        super().setUp()


class Repeat(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Repeat(nutils.elementseq.References.from_iter([square, triangle, square], 2), 2)
        self.check = [square, triangle, square]*2
        self.checkndims = 2
        super().setUp()


class Product(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Product(nutils.elementseq.References.from_iter([square, triangle], 2), nutils.elementseq.References.uniform(line, 3))
        self.check = [square*line]*3+[triangle*line]*3
        self.checkndims = 3
        super().setUp()


class Derived(TestCase, Common):

    def setUp(self):
        self.seq = nutils.elementseq._Derived(nutils.elementseq.References.from_iter([square, triangle], 2), 'child_refs', 2)
        self.check = [square]*4+[triangle]*4
        self.checkndims = 2
        super().setUp()


class from_iter(TestCase):

    def test_empty(self):
        self.assertEqual(nutils.elementseq.References.from_iter([], 2), nutils.elementseq._Empty(2))

    def test_uniform(self):
        self.assertEqual(nutils.elementseq.References.from_iter([square]*3, 2), nutils.elementseq._Uniform(square, 3))

    def test_plain(self):
        self.assertEqual(nutils.elementseq.References.from_iter([square, triangle], 2), nutils.elementseq._Plain((square, triangle), 2))

    def test_invalid_ndims(self):
        with self.assertRaisesRegex(ValueError, '^not all `Reference` objects in the sequence have ndims equal to 1$'):
            nutils.elementseq.References.from_iter([line, square], 1)


class uniform(TestCase):

    def test_empty(self):
        self.assertEqual(nutils.elementseq.References.uniform(square, 0), nutils.elementseq._Empty(2))

    def test_uniform(self):
        self.assertEqual(nutils.elementseq.References.uniform(square, 1), nutils.elementseq._Uniform(square, 1))

    def test_invalid_length(self):
        with self.assertRaisesRegex(ValueError, '^expected nonnegative `length` but got -1$'):
            nutils.elementseq.References.uniform(square, -1)


class empty(TestCase):

    def test_empty(self):
        self.assertEqual(nutils.elementseq.References.empty(1), nutils.elementseq._Empty(1))

    def test_invalid_ndims(self):
        with self.assertRaisesRegex(ValueError, '^expected nonnegative `ndims` but got -1$'):
            nutils.elementseq.References.empty(-1)

# vim:sw=2:sts=2:et
