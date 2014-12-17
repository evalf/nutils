#!/usr/bin/env python

from nutils import *

class TestIscheme( object ):

  def _test( self, ref, ptype, target_points, target_weights=None ):
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

  def testLine( self ):
    ref = element.SimplexReference(1)
    a = numpy.sqrt(1/3.)
    b = numpy.sqrt(3/5.)
    self._test( ref, 'gauss1', [[.5]], [1.] )
    self._test( ref, 'gauss2', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    self._test( ref, 'gauss3', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    self._test( ref, 'gauss4', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( ref, 'gauss5', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( ref, 'uniform1', [[.5]], [1.] )
    self._test( ref, 'uniform2', [[.25],[.75]], [.5,.5] )
    self._test( ref, 'uniform3', [[1/6.],[1/2.],[5/6.]], [1/3.]*3 )

  def testQuad( self ):
    ref = element.SimplexReference(1)**2
    a = numpy.sqrt(1/3.)
    b = numpy.sqrt(3/5.)
    self._test( ref, 'gauss1', [[.5,.5]], [1.] )
    self._test( ref, 'gauss2', [[.5-.5*a,.5-.5*a],[.5-.5*a,.5+.5*a],[.5+.5*a,.5-.5*a],[.5+.5*a,.5+.5*a]], [1/4.]*4 )
    self._test( ref, 'gauss1,4', [[.5,.5-.5*b],[.5,.5],[.5,.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( ref, 'uniform1', [[.5,.5]], [1.] )
    self._test( ref, 'uniform2', [[.25,.25],[.25,.75],[.75,.25],[.75,.75]], [1/4.]*4 )
    self._test( ref, 'uniform1,3', [[.5,1/6.],[.5,1/2.],[.5,5/6.]], [1/3.]*3 )
    self._test( ref, 'uniform1*gauss1', [[.5,.5]], [1.] )
    self._test( ref, 'uniform2*gauss1', [[.25,.5],[.75,.5]], [.5,.5] )
    self._test( ref, 'uniform1*gauss2', [[.5,.5-.5*a],[.5,.5+.5*a]], [.5,.5] )
    self._test( ref, 'uniform2*gauss2', [[.25,.5-.5*a],[.25,.5+.5*a],[.75,.5-.5*a],[.75,.5+.5*a]], [1/4.]*4 )

  def testTriangle( self ):
    ref = element.SimplexReference(2)
    self._test( ref, 'gauss1', [[1/3.,1/3.]], [.5] )
    self._test( ref, 'gauss2', [[2/3.,1/6.],[1/6.,2/3.],[1/6.,1/6.]], [1/6.]*3 )
    self._test( ref, 'gauss3', [[1/3.,1/3.],[3/5.,1/5.],[1/5.,3/5.],[1/5.,1/5.]], [-9/32.,25/96.,25/96.,25/96.] )
    self._test( ref, 'uniform1', [[1/3.,1/3.]], [.5] )
    self._test( ref, 'uniform2', [[1/6.,1/6.],[1/6.,2/3.],[2/3.,1/6.],[1/3.,1/3.]], [1/8.]*4 )

  def testHexagon( self ):
    ref = element.SimplexReference(1)**3
    self._test( ref, 'uniform1', [[.5,.5,.5]], [1.] )

  def testPrism( self ):
    ref = element.SimplexReference(1)*element.SimplexReference(2)
    self._test( ref, 'uniform1', [[.5,1/3.,1/3.]], [.5] )
    self._test( ref, 'uniform2', [[.25,1/6.,1/6.],[.25,1/6.,2/3.],[.25,2/3.,1/6.],[.25,1/3.,1/3.],[.75,1/6.,1/6.],[.75,1/6.,2/3.],[.75,2/3.,1/6.],[.75,1/3.,1/3.]], [1/16.]*8 )
    self._test( ref, 'uniform1*gauss2', [[.5,2/3.,1/6.],[.5,1/6.,2/3.],[.5,1/6.,1/6.]], [1/6.]*3 )
