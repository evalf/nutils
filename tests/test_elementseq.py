from nutils.testing import TestCase, parametrize
import nutils.elementseq, nutils.element
import unittest, numpy, itertools

line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()

class Common:

  def test_fromdims(self):
    self.assertEqual(self.seq.ndims, self.checkndims)

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
        with self.subTest('{}:{}'.format(i,j)):
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
    self.assertEqual(tuple(self.seq+nutils.elementseq.EmptyReferences(self.checkndims)), tuple(self.check))
    self.assertEqual(tuple(self.seq+self.seq), tuple(self.check)+tuple(self.check))

  def test_mul_int(self):
    for m in range(4):
      with self.subTest(m):
        self.assertEqual(tuple(self.seq*m), tuple(self.check)*m)

  def test_mul_references(self):
    other = [square, triangle]
    self.assertEqual(tuple(self.seq*nutils.elementseq.PlainReferences(other, 2)), tuple(l*r for l, r in itertools.product(self.check, other)))

  def test_children(self):
    self.assertEqual(tuple(self.seq.children), tuple(itertools.chain.from_iterable(ref.child_refs for ref in self.check)))

  def test_edges(self):
    self.assertEqual(tuple(self.seq.edges), tuple(itertools.chain.from_iterable(ref.edge_refs for ref in self.check)))

  def test_getpoints(self):
    self.assertEqual(self.seq.getpoints('bezier', 2), tuple(ref.getpoints('bezier', 2) for ref in self.check))

class EmptyReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.EmptyReferences(2)
    self.check = []
    self.checkndims = 2

class PlainReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.PlainReferences([square, triangle], 2)
    self.check = [square, triangle]
    self.checkndims = 2

class UniformReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.UniformReferences(square, 3)
    self.check = [square]*3
    self.checkndims = 2

class SelectedReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.SelectedReferences(nutils.elementseq.PlainReferences([square, triangle, square], 2), [1, 2])
    self.check = [triangle, square]
    self.checkndims = 2

class ChainedReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.ChainedReferences([nutils.elementseq.PlainReferences([square, triangle], 2)]*2)
    self.check = [square, triangle]*2
    self.checkndims = 2

class RepeatedReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.RepeatedReferences(nutils.elementseq.PlainReferences([square, triangle, square], 2), 2)
    self.check = [square, triangle, square]*2
    self.checkndims = 2

class ProductReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.ProductReferences(nutils.elementseq.PlainReferences([square, triangle], 2), nutils.elementseq.PlainReferences([line, line], 1))
    self.check = [square*line]*2+[triangle*line]*2
    self.checkndims = 3

class DerivedReferences(TestCase, Common):
  def setUp(self):
    self.seq = nutils.elementseq.DerivedReferences(nutils.elementseq.PlainReferences([square, triangle], 2), 'child_refs', 2)
    self.check = [square]*4+[triangle]*4
    self.checkndims = 2


class exceptions(TestCase):

  def test_PlainReferences_invalid_ndims(self):
    with self.assertRaisesRegex(ValueError, 'expected references with ndims=2, but got .*'):
      nutils.elementseq.PlainReferences([line, line], 2)

  def test_PlainReferences_multiple_ndims(self):
    with self.assertRaisesRegex(ValueError, 'expected references with ndims=2, but got .*'):
      nutils.elementseq.PlainReferences([line, square], 2)

  def test_UniformReferences_negative_length(self):
    with self.assertRaisesRegex(ValueError, 'length should be strict positive, but got .*'):
      nutils.elementseq.UniformReferences(line, -1)

  def test_SelectedReferences_invalid_indices(self):
    parent = nutils.elementseq.PlainReferences([square, triangle], 2)
    with self.assertRaisesRegex(IndexError, '`indices` out of range'):
      nutils.elementseq.SelectedReferences(parent, [-1])
    with self.assertRaisesRegex(IndexError, '`indices` out of range'):
      nutils.elementseq.SelectedReferences(parent, [2])

  def test_ChainedReferences_no_items(self):
    with self.assertRaisesRegex(ValueError, 'Empty chain.'):
      nutils.elementseq.ChainedReferences([])

  def test_ChainedReferences_multiple_ndims(self):
    a = nutils.elementseq.PlainReferences([line], 1)
    b = nutils.elementseq.PlainReferences([square], 2)
    with self.assertRaisesRegex(ValueError, 'Cannot chain References with different ndims.'):
      nutils.elementseq.ChainedReferences([a, b])

  def test_RepeatedReferences_negative_count(self):
    parent = nutils.elementseq.PlainReferences([square, triangle], 2)
    with self.assertRaisesRegex(ValueError, 'count should be strict positive, but got .*'):
      nutils.elementseq.RepeatedReferences(parent, -1)

class asreferences(TestCase):

  def test_References(self):
    value = nutils.elementseq.UniformReferences(line, 2)
    self.assertEqual(nutils.elementseq.asreferences(value, 1), value)

  def test_Reference_invalid_ndims(self):
    value = nutils.elementseq.UniformReferences(line, 2)
    with self.assertRaisesRegex(ValueError, 'expected References object with ndims=2, but got 1'):
      nutils.elementseq.asreferences(value, 2)

  def test_References_list_empty(self):
    self.assertEqual(nutils.elementseq.asreferences([], 2), nutils.elementseq.EmptyReferences(2))

  def test_References_list_pluriform(self):
    self.assertEqual(nutils.elementseq.asreferences([square, triangle], 2), nutils.elementseq.PlainReferences([square, triangle], 2))

  def test_References_list_uniform(self):
    self.assertEqual(nutils.elementseq.asreferences([square]*3, 2), nutils.elementseq.UniformReferences(square, 3))

  def test_invalid(self):
    with self.assertRaisesRegex(ValueError, 'cannot convert .* to a References object'):
      nutils.elementseq.asreferences(None, 2)

# vim:sw=2:sts=2:et
