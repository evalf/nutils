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
