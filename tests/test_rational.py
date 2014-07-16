from __future__ import division
from nutils import *
import numpy

class TestRational( object ):

  nums = { n: rational.factor(n) for n in ( 1, 2, 7, 27, 49, 137 ) }

  def test_int( self ):
    for n, N in self.nums.items():
      assert int(N) == n

  def test_float( self ):
    for n, N in self.nums.items():
      assert float(N) == n

  def test_multiply( self ):
    for n, N in self.nums.items():
      for m, M in self.nums.items():
        assert float(N*M) == n*m

  def test_divide( self ):
    for n, N in self.nums.items():
      for m, M in self.nums.items():
        assert float(N/M) == n/m 

  def test_power( self ):
    for n, N in self.nums.items():
      for i in range(3):
        assert float(N**i) == n**i

  def test_factor( self ):
    for n, N in self.nums.items():
      assert rational.factor(n) == N

  def test_gcd( self ):
    nums, factor = rational.gcd([ 2, 4, 8 ])
    assert all( nums == [ 1, 2, 4 ] )
    assert int(factor) == 2
    nums, factor = rational.gcd([ 0, 42, 21 ])
    assert all( nums == [ 0, 2, 1 ] )
    assert int(factor) == 21
