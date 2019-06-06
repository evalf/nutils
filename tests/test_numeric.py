from nutils import numeric
import numpy
from nutils.testing import *

@parametrize
class pack(TestCase):

  def setUp(self):
    assert self.nbits in (8, 16, 32)
    self.dtype = numpy.dtype('int{}'.format(self.nbits))
    self.nnan, self.nnil, self.nmin, self.nmax, self.ninf = n = numpy.array(
      [-128, 0, 1, 126, 127] if self.nbits == 8 else
      [-32768, 0, 1, 32766, 32767] if self.nbits == 16 else
      [-2147483648, 0, 1, 2147483646, 2147483647], dtype=self.dtype)
    self.amin, self.amax, self.aclip = numpy.sinh(n[2:]*self.rtol)*(self.atol/self.rtol)

  def pack(self, a):
    return numeric.pack(a, atol=self.atol, rtol=self.rtol, dtype=self.dtype)

  def unpack(self, n):
    return numeric.unpack(n, atol=self.atol, rtol=self.rtol)

  def test_decode(self):
    self.assertTrue(numpy.isnan(self.unpack(self.nnan)))
    self.assertEqual(self.unpack(-self.ninf), -numpy.inf)
    self.assertEqual(self.unpack(-self.nmax), -self.amax)
    self.assertEqual(self.unpack(-self.nmin), -self.amin)
    self.assertEqual(self.unpack(self.nnil), 0)
    self.assertEqual(self.unpack(self.nmin), self.amin)
    self.assertEqual(self.unpack(self.nmax), self.amax)
    self.assertEqual(self.unpack(self.ninf), numpy.inf)

  def test_encode(self):
    self.assertEqual(self.pack(numpy.nan), self.nnan)
    self.assertEqual(self.pack(-numpy.inf), -self.ninf)
    self.assertEqual(self.pack(-self.amax), -self.nmax)
    self.assertEqual(self.pack(-self.amin), -self.nmin)
    self.assertEqual(self.pack(0), self.nnil)
    self.assertEqual(self.pack(self.amin), self.nmin)
    self.assertEqual(self.pack(self.amax), self.nmax)
    self.assertEqual(self.pack(numpy.inf), self.ninf)

  def test_clip(self):
    with self.assertWarns(RuntimeWarning):
      self.assertEqual(self.pack(-self.aclip), -self.ninf)
    with self.assertWarns(RuntimeWarning):
      self.assertEqual(self.pack(self.aclip), self.ninf)

  def test_round(self):
    b01 = numpy.sinh(0.5*self.rtol)*(self.atol/self.rtol)
    b12 = numpy.sinh(1.5*self.rtol)*(self.atol/self.rtol)
    a = -b12*1.001, -b12*.999, -b01*1.001, -b01*.999, b01*.999, b01*1.001, b12*.999, b12*1.001
    n = self.pack(a)
    self.assertEqual(tuple(n), (-2, -1, -1, 0, 0, 1, 1, 2))

  def test_spacing(self):
    for a in -1., 0., 1.:
      n = self.pack(a)
      da = numpy.sqrt(self.atol**2 + (a*self.rtol)**2)
      self.assertLess(self.unpack(n-1), a - da/2)
      self.assertGreater(self.unpack(n), a - da/2)
      self.assertLess(self.unpack(n), a + da/2)
      self.assertGreater(self.unpack(n+1), a + da/2)

pack('int8', atol=2e-6, rtol=2e-1, nbits=8)
pack('int16', atol=2e-15, rtol=2e-3, nbits=16)
pack('int32', atol=2e-96, rtol=2e-7, nbits=32)

class sorted_index(TestCase):

  def test_None(self):
    for a, b in ([], []), ([1], []), ([1,2], [2,1]):
      self.assertEqual(numeric.sorted_index(numpy.array(a, int), b).tolist(), [a.index(v) for v in b if v in a])

  def test_None_exception(self):
    for a, b in ([], [1]), ([1], [2]), ([1,2], [3,1]):
      with self.assertRaises(ValueError):
        numeric.sorted_index(numpy.array(a, int), b)

  def test_int(self):
    for a, b in ([], []), ([1], []), ([1,2], [2,1]), ([], [1]), ([1], [2]), ([1,2], [3,1]):
      self.assertEqual(numeric.sorted_index(numpy.array(a, int), b, missing=-1).tolist(), [a.index(v) if v in a else -1 for v in b])

  def test_mask(self):
    for a, b in ([], []), ([1], []), ([1,2], [2,1]), ([], [1]), ([1], [2]), ([1,2], [3,1]):
      self.assertEqual(numeric.sorted_index(numpy.array(a, int), b, missing='mask').tolist(), [a.index(v) for v in b if v in a])

  def test_invalid(self):
    with self.assertRaises(ValueError):
      numeric.sorted_index(numpy.array([1], int), [1], missing='foo')

class sorted_contains(TestCase):

  def test(self):
    for a, b in ([], []), ([1], []), ([1,2], [2,1]), ([], [1]), ([1], [2]), ([1,2], [3,1]):
      self.assertEqual(numeric.sorted_contains(numpy.array(a, int), b).tolist(), [v in a for v in b])

class asboolean(TestCase):

  def test_bool(self):
    self.assertAllEqual(numeric.asboolean([True, False], 2), [True, False])

  def test_int(self):
    self.assertAllEqual(numeric.asboolean([1], 2), [False, True])
    self.assertAllEqual(numeric.asboolean([0], 2), [True, False])

  def test_none(self):
    self.assertAllEqual(numeric.asboolean(None, 2), [False, False])
    self.assertAllEqual(numeric.asboolean([], 2), [False, False])
    self.assertAllEqual(numeric.asboolean((), 2), [False, False])

  def test_float(self):
    with self.assertRaises(Exception):
      numeric.asboolean([1.5, 2.5], 2)

  def test_wrongsize(self):
    with self.assertRaises(Exception):
      numeric.asboolean([True, False, True], 2)

  def test_wrongdimension(self):
    with self.assertRaises(Exception):
      numeric.asboolean([[True, False, True], [False, True, True]], 2)

  def test_outofbounds(self):
    with self.assertRaises(Exception):
      numeric.asboolean([-1], 2)
    with self.assertRaises(Exception):
      numeric.asboolean([2], 2)

  def test_unordered(self):
    self.assertAllEqual(numeric.asboolean([2,1], 3, ordered=False), [False, True, True])
    with self.assertRaises(Exception):
      numeric.asboolean([2,1], 3)
