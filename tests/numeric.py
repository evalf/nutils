from nutils import *
from . import register, unittest

@register( '10x7', 10, 7 )
@register( '10x0', 10, 0 )
@register( '10x10', 10, 10 )
def ortho_complement( m, n ):
  A = numpy.sin(numpy.arange(m*n)).reshape(m,n) # just a matrix
  B = numeric.ortho_complement( A )

  @unittest
  def shape():
    assert B.shape == ( m, m-n )

  @unittest
  def orthogonal():
    numpy.testing.assert_almost_equal( numpy.dot( A.T, B ), 0, decimal=15 )


@register
def searchsorted():
  for i in range(4):
    @unittest( name=i )
    def insertion():
      assert numeric.searchsorted( [.5,1.5,2.5], i ) == i
  for i in range(3):
    @unittest( name=i )
    def lookup():
      assert numeric.searchsorted( [0,1,2], i ) == i
  @unittest
  def repeats():
    assert numeric.searchsorted( [.5,.5,.5], 0 ) == 0
    assert numeric.searchsorted( [.5,.5,.5], 1 ) == 3
  @unittest
  def strings():
    assert numeric.searchsorted( ['bar','foo','fool'], 'food' ) == 2
