from nutils import *
import unittest
from . import parametrize

@parametrize
class ortho_complement(unittest.TestCase):

  def setUp(self):
    self.A = numpy.sin(numpy.arange(self.m*self.n)).reshape(self.m,self.n) # just a matrix
    self.B = numeric.ortho_complement(self.A)

  def test_shape(self):
    self.assertEqual(self.B.shape, (self.m, self.m-self.n))

  def test_orthogonal(self):
    numpy.testing.assert_almost_equal(numpy.dot(self.A.T, self.B), 0, decimal=15)

ortho_complement('10x7', m=10, n=7)
ortho_complement('10x0', m=10, n=0)
ortho_complement('10x10', m=10, n=10)


class searchsorted(unittest.TestCase):

  def test_insertion(self):
    for i in range(4):
      with self.subTest(i):
        assert numeric.searchsorted([.5,1.5,2.5], i) == i

  def test_lookup(self):
    for i in range(3):
      with self.subTest(i):
        assert numeric.searchsorted( [0,1,2], i ) == i

  def test_repeats(self):
    self.assertEqual(numeric.searchsorted( [.5,.5,.5], 0 ), 0)
    self.assertEqual(numeric.searchsorted( [.5,.5,.5], 1 ), 3)

  def test_strings(self):
    self.assertEqual(numeric.searchsorted( ['bar','foo','fool'], 'food' ), 2)
