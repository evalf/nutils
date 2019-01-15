from nutils.testing import TestCase, parametrize
import nutils.transformseq, nutils.elementseq, nutils.element
import unittest, numpy, itertools

@parametrize
class common(TestCase):

  def test_fromdims(self):
    self.assertEqual(self.seq.fromdims, self.checkfromdims)

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
    self.assertEqual(tuple(self.seq+nutils.transformseq.PlainTransforms((), self.checkfromdims)), tuple(self.check))
    self.assertEqual(tuple(self.seq+self.seq), tuple(self.check)+tuple(self.check))

  def test_index_with_tail(self):
    for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
      self.assertEqual(self.seq.index_with_tail(trans), (i, ()))
      for ctrans in ref.child_transforms:
        self.assertEqual(self.seq.index_with_tail(trans+(ctrans,)), (i, (ctrans,)))
      if self.checkfromdims > 0:
        for etrans in ref.edge_transforms:
          for shuffle in lambda t: t, nutils.transform.canonical:
            self.assertEqual(self.seq.index_with_tail(shuffle(trans+(ctrans,))), (i, (ctrans,)))

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
    for trans, ref in zip(self.check, self.checkrefs):
      for ctrans in ref.child_transforms:
        with self.assertRaises(ValueError):
          self.seq.index(trans+(ctrans,))

  def test_contains_with_tail(self):
    for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
      self.assertEqual(self.seq.index_with_tail(trans), (i, ()))
      for ctrans in ref.child_transforms:
        self.assertTrue(self.seq.contains_with_tail(trans+(ctrans,)))
      if self.checkfromdims > 0:
        for etrans in ref.edge_transforms:
          for shuffle in lambda t: t, nutils.transform.canonical:
            self.assertTrue(self.seq.contains_with_tail(trans+(etrans,)))

  def test_contains_with_tail_missing(self):
    for trans in self.checkmissing:
      self.assertFalse(self.seq.contains_with_tail(trans))

  def test_contains(self):
    for i, trans in enumerate(self.check):
      self.assertTrue(self.seq.contains(trans))

  def test_contains_missing(self):
    for trans in self.checkmissing:
      self.assertFalse(self.seq.contains(trans))
    for trans, ref in zip(self.check, self.checkrefs):
      for ctrans in ref.child_transforms:
        self.assertFalse(self.seq.contains(trans+(ctrans,)))

  def test_refined(self):
    refined = self.seq.refined(self.checkrefs)
    ctransforms = (trans+(ctrans,) for trans, ref in zip(self.check, self.checkrefs) for ctrans in ref.child_transforms)
    for i, trans in enumerate(ctransforms):
      self.assertEqual(refined.index(trans), i)

point = nutils.element.PointReference()
line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()

l1, x1, r1 = sorted([nutils.transform.Identifier(1, n) for n in ('l1', 'x1', 'r1')], key=id)

s0 = nutils.transform.Shift([0.])
s1 = nutils.transform.Shift([1.])
s2 = nutils.transform.Shift([2.])
s3 = nutils.transform.Shift([3.])
s4 = nutils.transform.Shift([4.])

c0,c1 = line.child_transforms
e0,e1 = line.edge_transforms

l2, x2, r2 = sorted([nutils.transform.Identifier(2, n) for n in ('l2', 'x2', 'r2')], key=id)
s00 = nutils.transform.Shift([0.,0.])
s01 = nutils.transform.Shift([0.,1.])
s02 = nutils.transform.Shift([0.,2.])
s03 = nutils.transform.Shift([0.,3.])
s10 = nutils.transform.Shift([1.,0.])
s11 = nutils.transform.Shift([1.,1.])
s12 = nutils.transform.Shift([1.,2.])
s13 = nutils.transform.Shift([1.,3.])

c00,c01,c10,c11 = square.child_transforms

common(
  'PlainTransforms:1D',
  seq=nutils.transformseq.PlainTransforms([(x1,s0),(x1,s1),(x1,s2),(x1,s3)], fromdims=1),
  check=[(x1,s0),(x1,s1),(x1,s2),(x1,s3)],
  checkmissing=[(l1,s0),(x1,s4),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)
common(
  'PlainTransforms:2D',
  seq=nutils.transformseq.PlainTransforms([(x2,s00),(x2,s01),(x2,s10),(x2,s11)], fromdims=2),
  check=[(x2,s00),(x2,s01),(x2,s10),(x2,s11)],
  checkmissing=[(l2,s00),(x2,s02),(x2,s12),(r2,s00)],
  checkrefs=nutils.elementseq.asreferences([square,square,triangle,triangle], 2),
  checkfromdims=2)
common(
  'MaskedTransforms',
  seq=nutils.transformseq.MaskedTransforms(nutils.transformseq.PlainTransforms([(x2,s00),(x2,s01),(x2,s10),(x2,s11)], fromdims=2), [0,2]),
  check=[(x2,s00),(x2,s10)],
  checkmissing=[(l2,s00),(x2,s01),(x2,s11),(x2,s02),(x2,s12),(r2,s00)],
  checkrefs=nutils.elementseq.asreferences([square,triangle], 2),
  checkfromdims=2)
common(
  'RefinedTransforms',
  seq=nutils.transformseq.RefinedTransforms(nutils.transformseq.PlainTransforms([(x1,s0),(x1,s1)], fromdims=1), nutils.elementseq.asreferences([line,line], 1)),
  check=[(x1,s0,c0),(x1,s0,c1),(x1,s1,c0),(x1,s1,c1)],
  checkmissing=[(l1,s0),(x1,s0),(x1,s1),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)
common(
  'UniformRefinedTransforms',
  seq=nutils.transformseq.UniformRefinedTransforms(nutils.transformseq.PlainTransforms([(x1,s0),(x1,s1)], fromdims=1), line),
  check=[(x1,s0,c0),(x1,s0,c1),(x1,s1,c0),(x1,s1,c1)],
  checkmissing=[(l1,s0),(x1,s0),(x1,s1),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)
common(
  'ChainedTransforms',
  seq=nutils.transformseq.ChainedTransforms([nutils.transformseq.PlainTransforms([(x1,s0),(x1,s1)], fromdims=1), nutils.transformseq.PlainTransforms([(x1,s2),(x1,s3)], fromdims=1)]),
  check=[(x1,s0),(x1,s1),(x1,s2),(x1,s3)],
  checkmissing=[(l1,s0),(x1,s4),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)

common(
  'StructuredTransforms:1D',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.DimAxis(0,4,False)], 0),
  check=[(x1,s0),(x1,s1),(x1,s2),(x1,s3)],
  checkmissing=[(l1,s0),(x1,s4),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)
common(
  'StructuredTransforms:1D,refined',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.DimAxis(0,4,False)], 1),
  check=[(x1,s0,c0),(x1,s0,c1),(x1,s1,c0),(x1,s1,c1)],
  checkmissing=[(l1,s0),(x1,s0),(x1,s1),(x1,s0,s1),(r1,s0)],
  checkrefs=nutils.elementseq.asreferences([line]*4, 1),
  checkfromdims=1)
for i, side, s, e in (3,False,s3,e1), (3,True,s2,e0):
  trans = (x1,s,e)
  common(
    'StructuredTransforms:1D,boundary,side={}'.format(side),
    seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.BndAxis(i,i,0,side)], 0),
    check=[trans],
    checkmissing=[t for t in [(x1,s0,e0),(x1,s3,e1),(x1,s4,e0)] if t != trans],
    checkrefs=nutils.elementseq.asreferences([point], 0),
    checkfromdims=0)
common(
  'StructuredTransforms:1D,interfaces,side=False',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.IntAxis(0,4,0,False)], 0),
  check=[(x1,s1,e1),(x1,s2,e1),(x1,s3,e1)],
  checkmissing=[(x1,s0,e1),(x1,s0,e0),(x1,s1,e0),(x1,s2,e0),(x1,s3,e0)],
  checkrefs=nutils.elementseq.asreferences([point]*3, 0),
  checkfromdims=0)
common(
  'StructuredTransforms:1D,interfaces,side=True',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.IntAxis(0,4,0,True)], 0),
  check=[(x1,s0,e0),(x1,s1,e0),(x1,s2,e0)],
  checkmissing=[(x1,s3,e0),(x1,s0,e1),(x1,s1,e1),(x1,s2,e1),(x1,s3,e1)],
  checkrefs=nutils.elementseq.asreferences([point]*3, 0),
  checkfromdims=0)
common(
  'StructuredTransforms:1D,periodic,interfaces,side=False',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.PIntAxis(0,4,0,False)], 0),
  check=[(x1,s1,e1),(x1,s2,e1),(x1,s3,e1),(x1,s0,e1)],
  checkmissing=[(x1,s0,e0),(x1,s1,e0),(x1,s2,e0),(x1,s3,e0),(x1,s4,e0)],
  checkrefs=nutils.elementseq.asreferences([point]*3, 0),
  checkfromdims=0)
common(
  'StructuredTransforms:1D,periodic,interfaces,side=True',
  seq=nutils.transformseq.StructuredTransforms(x1, [nutils.transformseq.PIntAxis(0,4,0,True)], 0),
  check=[(x1,s0,e0),(x1,s1,e0),(x1,s2,e0),(x1,s3,e0)],
  checkmissing=[(x1,s0,e1),(x1,s1,e1),(x1,s2,e1),(x1,s3,e1),(x1,s4,e1)],
  checkrefs=nutils.elementseq.asreferences([point]*3, 0),
  checkfromdims=0)

common(
  'StructuredTransforms:2D',
  seq=nutils.transformseq.StructuredTransforms(x2, [nutils.transformseq.DimAxis(0,2,False),nutils.transformseq.DimAxis(2,4,False)], 0),
  check=[(x2,s02),(x2,s03),(x2,s12),(x2,s13)],
  checkmissing=[(x2,s00),(x2,s01),(x2,s10),(x2,s11)],
  checkrefs=nutils.elementseq.asreferences([square]*4, 2),
  checkfromdims=2)
common(
  'StructuredTransforms:2D,refined',
  seq=nutils.transformseq.StructuredTransforms(x2, [nutils.transformseq.DimAxis(0,2,False),nutils.transformseq.DimAxis(2,4,False)], 1),
  check=[(x2,s01,c00),(x2,s01,c01),(x2,s01,c10),(x2,s01,c11)],
  checkmissing=[(x2,s00,c00)],
  checkrefs=nutils.elementseq.asreferences([square]*4, 2),
  checkfromdims=2)

# vim:sw=2:sts=2:et
