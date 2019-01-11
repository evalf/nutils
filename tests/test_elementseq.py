from nutils.testing import TestCase, parametrize
import nutils.elementseq, nutils.element
import unittest, numpy, itertools

line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()

@parametrize
class common(TestCase):

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
    self.assertEqual(tuple(self.seq+nutils.elementseq.PlainReferences(())), tuple(self.check))
    self.assertEqual(tuple(self.seq+self.seq), tuple(self.check)+tuple(self.check))

  def test_mul_int(self):
    for m in range(4):
      with self.subTest(m):
        self.assertEqual(tuple(self.seq*m), tuple(self.check)*m)

  def test_mul_references(self):
    other = [square, triangle]
    self.assertEqual(tuple(self.seq*nutils.elementseq.PlainReferences(other)), tuple(l*r for l, r in itertools.product(self.check, other)))

  def test_children(self):
    self.assertEqual(tuple(self.seq.children), tuple(itertools.chain.from_iterable(ref.child_refs for ref in self.check)))

  def test_getpoints(self):
    self.assertEqual(self.seq.getpoints('bezier', 2), tuple(ref.getpoints('bezier', 2) for ref in self.check))

common(
  'PlainReferences',
  seq=nutils.elementseq.PlainReferences([square, triangle]),
  check=[square, triangle])
common(
  'UniformReferences',
  seq=nutils.elementseq.UniformReferences(square, 3),
  check=[square]*3)
common(
  'SelectedReferences',
  seq=nutils.elementseq.SelectedReferences(nutils.elementseq.PlainReferences([square, triangle, square]), [1, 2]),
  check=[triangle, square])
common(
  'ChainedReferences',
  seq=nutils.elementseq.ChainedReferences([nutils.elementseq.PlainReferences([square, triangle])]*2),
  check=[square, triangle]*2)
common(
  'RepeatedReferences',
  seq=nutils.elementseq.RepeatedReferences(nutils.elementseq.PlainReferences([square, triangle]), 2),
  check=[square, triangle]*2)
common(
  'ProductReferences',
  seq=nutils.elementseq.ProductReferences(nutils.elementseq.PlainReferences([square, triangle]), nutils.elementseq.PlainReferences([line, line])),
  check=[square*line]*2+[triangle*line]*2)
common(
  'ChildReferences',
  seq=nutils.elementseq.ChildReferences(nutils.elementseq.PlainReferences([square, triangle])),
  check=[square]*4+[triangle]*4)

# vim:sw=2:sts=2:et
