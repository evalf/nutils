
from nutils import *
from . import register, unittest

def _test( ref, ptype, target_points, target_weights=None ):
  points, weights = ref.getischeme( ptype )
  assert points.ndim == 2
  assert points.shape[1] == ref.ndims
  numpy.testing.assert_almost_equal( points, target_points )
  if target_weights is None:
    assert weights is None
  else:
    assert weights.ndim == 1
    assert weights.shape[0] == points.shape[0]
    numpy.testing.assert_almost_equal( weights, target_weights )

@register
def check():

  @unittest
  def line():
    ref = element.getsimplex(1)
    a = numpy.sqrt(1/3.)
    b = numpy.sqrt(3/5.)
    _test( ref, 'gauss1', [[.5]], [1.] )
    _test( ref, 'gauss2', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    _test( ref, 'gauss3', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    _test( ref, 'gauss4', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    _test( ref, 'gauss5', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    _test( ref, 'uniform1', [[.5]], [1.] )
    _test( ref, 'uniform2', [[.25],[.75]], [.5,.5] )
    _test( ref, 'uniform3', [[1/6.],[1/2.],[5/6.]], [1/3.]*3 )

  def quad():
    ref = element.getsimplex(1)**2
    a = numpy.sqrt(1/3.)
    b = numpy.sqrt(3/5.)
    _test( ref, 'gauss1', [[.5,.5]], [1.] )
    _test( ref, 'gauss2', [[.5-.5*a,.5-.5*a],[.5-.5*a,.5+.5*a],[.5+.5*a,.5-.5*a],[.5+.5*a,.5+.5*a]], [1/4.]*4 )
    _test( ref, 'gauss1,4', [[.5,.5-.5*b],[.5,.5],[.5,.5+.5*b]], [5/18.,4/9.,5/18.] )
    _test( ref, 'uniform1', [[.5,.5]], [1.] )
    _test( ref, 'uniform2', [[.25,.25],[.25,.75],[.75,.25],[.75,.75]], [1/4.]*4 )
    _test( ref, 'uniform1,3', [[.5,1/6.],[.5,1/2.],[.5,5/6.]], [1/3.]*3 )
    _test( ref, 'uniform1*gauss1', [[.5,.5]], [1.] )
    _test( ref, 'uniform2*gauss1', [[.25,.5],[.75,.5]], [.5,.5] )
    _test( ref, 'uniform1*gauss2', [[.5,.5-.5*a],[.5,.5+.5*a]], [.5,.5] )
    _test( ref, 'uniform2*gauss2', [[.25,.5-.5*a],[.25,.5+.5*a],[.75,.5-.5*a],[.75,.5+.5*a]], [1/4.]*4 )

  def testTriangle():
    ref = element.getsimplex(2)
    _test( ref, 'gauss1', [[1/3.,1/3.]], [.5] )
    _test( ref, 'gauss2', [[2/3.,1/6.],[1/6.,2/3.],[1/6.,1/6.]], [1/6.]*3 )
    _test( ref, 'gauss3', [[1/3.,1/3.],[3/5.,1/5.],[1/5.,3/5.],[1/5.,1/5.]], [-9/32.,25/96.,25/96.,25/96.] )
    _test( ref, 'uniform1', [[1/3.,1/3.]], [.5] )
    _test( ref, 'uniform2', [[1/6.,1/6.],[1/6.,2/3.],[2/3.,1/6.],[1/3.,1/3.]], [1/8.]*4 )

  def testHexagon():
    ref = element.getsimplex(1)**3
    _test( ref, 'uniform1', [[.5,.5,.5]], [1.] )

  def testPrism():
    ref = element.getsimplex(1)*element.getsimplex(2)
    _test( ref, 'uniform1', [[.5,1/3.,1/3.]], [.5] )
    _test( ref, 'uniform2', [[.25,1/6.,1/6.],[.25,1/6.,2/3.],[.25,2/3.,1/6.],[.25,1/3.,1/3.],[.75,1/6.,1/6.],[.75,1/6.,2/3.],[.75,2/3.,1/6.],[.75,1/3.,1/3.]], [1/16.]*8 )
    _test( ref, 'uniform1*gauss2', [[.5,2/3.,1/6.],[.5,1/6.,2/3.],[.5,1/6.,1/6.]], [1/6.]*3 )
