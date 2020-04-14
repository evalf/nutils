from nutils.testing import TestCase, parametrize
import nutils.transformseq, nutils.elementseq, nutils.element
import unittest, numpy, itertools, functools, types

class Common:

  def test_todims(self):
    self.assertEqual(self.seq.todims, self.checktodims)
    for trans in self.seq:
      self.assertEqual(tuple(t[0].todims for t in trans), self.checktodims)

  def test_fromdims(self):
    self.assertEqual(len(self.seq), len(self.checkrefs))
    for trans, ref in zip(self.seq, self.checkrefs):
      self.assertEqual(sum(t[-1].fromdims for t in trans), ref.ndims)

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
    self.assertEqual(tuple(self.seq+nutils.transformseq.EmptyTransforms(self.checktodims)), tuple(self.check))
    self.assertEqual(tuple(self.seq+self.seq), tuple(self.check)+tuple(self.check))

  def test_index_with_tail(self):
    assert len(self.check) == len(self.checkrefs)
    for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
      self.assertEqual(self.seq.index_with_tail(trans), (i, ((),)))
      for ctrans in nutils.transform.child_transforms(trans, ref):
        self.assertEqual(self.seq.index_with_tail(ctrans), (i, tuple(t[-1:] for t in ctrans)))
      if ref.ndims > 0:
        for etrans in nutils.transform.edge_transforms(trans, ref):
          for variant in etrans, nutils.transform.canonical(etrans):
            self.assertEqual(self.seq.index_with_tail(variant), (i, tuple(t[-1:] for t in etrans)))

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
      for ctrans in nutils.transform.child_transforms(trans, ref):
        with self.assertRaises(ValueError):
          self.seq.index(ctrans)

  def test_contains_with_tail(self):
    assert len(self.check) == len(self.checkrefs)
    for i, (trans, ref) in enumerate(zip(self.check, self.checkrefs)):
      self.assertEqual(self.seq.index_with_tail(trans), (i, ((),)))
      for ctrans in nutils.transform.child_transforms(trans, ref):
        self.assertTrue(self.seq.contains_with_tail(ctrans))
      if ref.ndims > 0:
        for etrans in nutils.transform.edge_transforms(trans, ref):
          for variant in etrans, nutils.transform.canonical(etrans):
            self.assertTrue(self.seq.contains_with_tail(variant))

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
      for ctrans in nutils.transform.child_transforms(trans, ref):
        self.assertFalse(self.seq.contains(ctrans))

  def test_refined(self):
    refined = self.seq.refined(self.checkrefs)
    assert len(self.check) == len(self.checkrefs)
    for i, ctrans in enumerate((ctrans for trans, ref in zip(self.check, self.checkrefs) for ctrans in nutils.transform.child_transforms(trans, ref))):
      self.assertEqual(refined.index(ctrans), i)

class Edges:

  def test_edges(self):
    edges = self.seq.edges(self.checkrefs)
    assert len(self.check) == len(self.checkrefs)
    for i, etrans in enumerate((etrans for trans, ref in zip(self.check, self.checkrefs) for etrans in nutils.transform.edge_transforms(trans, ref))):
      self.assertEqual(edges.index(etrans), i)

point = nutils.element.PointReference()
line = nutils.element.LineReference()
square = line*line
triangle = nutils.element.TriangleReference()

s0 = nutils.transform.Shift([0.])
s1 = nutils.transform.Shift([1.])
s2 = nutils.transform.Shift([2.])
s3 = nutils.transform.Shift([3.])
s4 = nutils.transform.Shift([4.])

c0,c1 = line.child_transforms
e0,e1 = line.edge_transforms

s00 = nutils.transform.Shift([0.,0.])
s01 = nutils.transform.Shift([0.,1.])
s02 = nutils.transform.Shift([0.,2.])
s03 = nutils.transform.Shift([0.,3.])
s10 = nutils.transform.Shift([1.,0.])
s11 = nutils.transform.Shift([1.,1.])
s12 = nutils.transform.Shift([1.,2.])
s13 = nutils.transform.Shift([1.,3.])

c00,c01,c10,c11 = square.child_transforms

class EmptyTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.EmptyTransforms((1,))
    self.check = ()
    self.checkmissing = ((s0,),),((s4,),)
    self.checkrefs = nutils.elementseq.EmptyReferences(1)
    self.checktodims = 1,

class PlainTransforms1D(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.PlainTransforms([(s1,),(s2,),(s3,)], 1, 1)
    self.check = ((s1,),),((s2,),),((s3,),)
    self.checkmissing = ((s0,),),((s4,),),((c0,),)
    self.checkrefs = nutils.elementseq.asreferences([line]*3, 1)
    self.checktodims = 1,

class PlainTransforms2D(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.PlainTransforms([(s00,),(s01,),(s10,),(s11,)], 2, 2)
    self.check = ((s00,),),((s01,),),((s10,),),((s11,),)
    self.checkmissing = ((s02,),),((s12,),),((c00,),)
    self.checkrefs = nutils.elementseq.asreferences([square,square,triangle,triangle], 2)
    self.checktodims = 2,

class MaskedTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.MaskedTransforms(nutils.transformseq.PlainTransforms([(s00,),(s01,),(s10,),(s11,)], 2, 2), [0,2])
    self.check = ((s00,),),((s10,),)
    self.checkmissing = ((s01,),),((s11,),),((s02,),),((s12,),),((c00,),)
    self.checkrefs = nutils.elementseq.asreferences([square,triangle], 2)
    self.checktodims = 2,

class ReorderedTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.ReorderedTransforms(nutils.transformseq.PlainTransforms([(s00,),(s01,),(s10,),(s11,)], 2, 2), [0,2,3,1])
    self.check = ((s00,),),((s10,),),((s11,),),((s01,),)
    self.checkmissing = ((s02,),),((s12,),),((c00,),)
    self.checkrefs = nutils.elementseq.asreferences([square]*4, 2)
    self.checktodims = 2,

class DerivedTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.DerivedTransforms(nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 1), nutils.elementseq.asreferences([line,line], 1), 'child_transforms', False)
    self.check = ((s0,c0),),((s0,c1),),((s1,c0),),((s1,c1),)
    self.checkmissing = ((s0,),),((s1,),),((s2,c0),)
    self.checkrefs = nutils.elementseq.asreferences([line]*4, 1)
    self.checktodims = 1,

class UniformDerivedTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.UniformDerivedTransforms(nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 1), line, 'child_transforms', False)
    self.check = ((s0,c0),),((s0,c1),),((s1,c0),),((s1,c1),)
    self.checkmissing = ((s0,),),((s1,),),((s2,c0),)
    self.checkrefs = nutils.elementseq.asreferences([line]*4, 1)
    self.checktodims = 1,

class ChainedTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.ChainedTransforms([nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 1), nutils.transformseq.PlainTransforms([(s2,),(s3,)], 1, 1)])
    self.check = ((s0,),),((s1,),),((s2,),),((s3,),)
    self.checkmissing = ((s4,),),((c0,),)
    self.checkrefs = nutils.elementseq.asreferences([line]*4, 1)
    self.checktodims = 1,

class StructuredTransforms1D(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.DimAxis(1,4,False)], 0)
    self.check = ((s1,),),((s2,),),((s3,),)
    self.checkmissing = ((s0,),),((s4,),),((c0,),)
    self.checkrefs = nutils.elementseq.asreferences([line]*3, 1)
    self.checktodims = 1,

class StructuredTransforms1DRefined(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.DimAxis(1,4,False)], 1)
    self.check = ((s0,c1),),((s1,c0),),((s1,c1),)
    self.checkmissing = ((s0,),),((s0,c0),),((s2,),),((s2,c0),)
    self.checkrefs = nutils.elementseq.asreferences([line]*3, 1)
    self.checktodims = 1,

class StructuredTransforms1DLeft(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.BndAxis(3,3,0,False)], 0)
    self.check = ((s3,e1),),
    self.checkmissing = ((s0,e0),),((s2,e0),),((s4,e0),)
    self.checkrefs = nutils.elementseq.asreferences([point], 0)
    self.checktodims = 1,

class StructuredTransforms1DRight(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.BndAxis(3,3,0,True)], 0)
    self.check = ((s2,e0),),
    self.checkmissing = ((s0,e0),),((s3,e1),),((s4,e0),)
    self.checkrefs = nutils.elementseq.asreferences([point], 0)
    self.checktodims = 1,

class StructuredTransforms1DInterfacesLeft(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.IntAxis(0,4,0,False)], 0)
    self.check = ((s1,e1),),((s2,e1),),((s3,e1),)
    self.checkmissing = ((s0,e1),),((s0,e0),),((s1,e0),),((s2,e0),),((s3,e0),)
    self.checkrefs = nutils.elementseq.asreferences([point]*3, 0)
    self.checktodims = 1,

class StructuredTransforms1DInterfacesRight(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.IntAxis(0,4,0,True)], 0)
    self.check = ((s0,e0),),((s1,e0),),((s2,e0),)
    self.checkmissing = ((s3,e0),),((s0,e1),),((s1,e1),),((s2,e1),),((s3,e1),)
    self.checkrefs = nutils.elementseq.asreferences([point]*3, 0)
    self.checktodims = 1,

class StructuredTransforms1DPeriodicInterfacesLeft(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.PIntAxis(0,4,0,False)], 0)
    self.check = ((s1,e1),),((s2,e1),),((s3,e1),),((s0,e1),)
    self.checkmissing = ((s0,e0),),((s1,e0),),((s2,e0),),((s3,e0),),((s4,e0),)
    self.checkrefs = nutils.elementseq.asreferences([point]*4, 0)
    self.checktodims = 1,

class StructuredTransforms1DPeriodicInterfacesRight(TestCase, Common):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.PIntAxis(0,4,0,True)], 0)
    self.check = ((s0,e0),),((s1,e0),),((s2,e0),),((s3,e0),)
    self.checkmissing = ((s0,e1),),((s1,e1),),((s2,e1),),((s3,e1),),((s4,e1),)
    self.checkrefs = nutils.elementseq.asreferences([point]*4, 0)
    self.checktodims = 1,

class StructuredTransforms2D(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.DimAxis(0,2,False),nutils.transformseq.DimAxis(2,4,False)], 0)
    self.check = ((s02,),),((s03,),),((s12,),),((s13,),)
    self.checkmissing = ((s00,),),((s01,),),((s10,),),((s11,),)
    self.checkrefs = nutils.elementseq.asreferences([square]*4, 2)
    self.checktodims = 2,

class StructuredTransforms2DRefined(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.StructuredTransforms([nutils.transformseq.DimAxis(0,2,False),nutils.transformseq.DimAxis(2,4,False)], 1)
    self.check = ((s01,c00),),((s01,c01),),((s01,c10),),((s01,c11),)
    self.checkmissing = ((s00,c00),),
    self.checkrefs = nutils.elementseq.asreferences([square]*4, 2)
    self.checktodims = 2,

class IdentifierTransforms(TestCase, Common, Edges):
  def setUp(self):
    self.seq = nutils.transformseq.IdentifierTransforms(ndims=2, name='foo', length=4)
    self.check = [((nutils.transform.Identifier(2, ('foo', i)),),) for i in range(4)]
    self.checkmissing = ((nutils.transform.Identifier(1, ('foo', 0)),),), ((nutils.transform.Identifier(2, ('foo', -1)),),), ((nutils.transform.Identifier(2, ('foo', 4)),),), ((nutils.transform.Identifier(2, ('bar', 0)),),)
    self.checkrefs = nutils.elementseq.asreferences([triangle]*4, 2)
    self.checktodims = 2,

class exceptions(TestCase):

  def test_PlainTransforms_invalid_todims(self):
    with self.assertRaisesRegex(ValueError, 'expected transforms with todims=2, but got .*'):
      nutils.transformseq.PlainTransforms([(s0,),(s1,)], 2, 1)

  def test_PlainTransforms_invalid_fromdims(self):
    with self.assertRaisesRegex(ValueError, 'expected transforms with fromdims=2, but got .*'):
      nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 2)

  def test_DerivedTransforms_length_mismatch(self):
    transforms = nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 1)
    references = nutils.elementseq.PlainReferences([line]*3, 1)
    with self.assertRaisesRegex(ValueError, '`parent` and `parent_references` should have the same length'):
      nutils.transformseq.DerivedTransforms(transforms, references, 'child_transforms', False)

  def test_ChainedTransforms_no_items(self):
    with self.assertRaisesRegex(ValueError, 'Empty chain.'):
      nutils.transformseq.ChainedTransforms([])

  def test_ChainedTransforms_multiple_ndims(self):
    a = nutils.transformseq.PlainTransforms([(s0,),(s1,)], 1, 1)
    b = nutils.transformseq.PlainTransforms([(s00,),(s01,)], 2, 2)
    with self.assertRaisesRegex(ValueError, 'Cannot chain Transforms with different todims.'):
      nutils.transformseq.ChainedTransforms([a, b])

# vim:sw=2:sts=2:et
