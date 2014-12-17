#!/usr/bin/env python

from nutils import *

class TestIscheme( object ):

  def __init__( self ):
    self.line = element.SimplexReference(1)
    self.triangle = element.SimplexReference(2)
    self.tetrahedron = element.SimplexReference(3)
    self.quad = element.SimplexReference(1)**2
    self.hexagon = element.SimplexReference(1)**3
    self.prism = element.SimplexReference(1)*element.SimplexReference(2)

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

  def testGauss( self ):
    a = numpy.sqrt(1/3.)
    b = numpy.sqrt(3/5.)
    self._test( self.line, 'gauss1', [[.5]], [1.] )
    self._test( self.line, 'gauss2', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    self._test( self.line, 'gauss3', [[.5-.5*a],[.5+.5*a]], [.5,.5] )
    self._test( self.line, 'gauss4', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( self.line, 'gauss5', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( self.quad, 'gauss1', [[.5,.5]], [1.] )
    self._test( self.quad, 'gauss2', [[.5-.5*a,.5-.5*a],[.5-.5*a,.5+.5*a],[.5+.5*a,.5-.5*a],[.5+.5*a,.5+.5*a]], [1/4.]*4 )
    self._test( self.quad, 'gauss1,4', [[.5,.5-.5*b],[.5,.5],[.5,.5+.5*b]], [5/18.,4/9.,5/18.] )
    self._test( self.triangle, 'gauss1', [[1/3.,1/3.]], [.5] )
    self._test( self.triangle, 'gauss2', [[2/3.,1/6.],[1/6.,2/3.],[1/6.,1/6.]], [1/6.]*3 )
    self._test( self.triangle, 'gauss3', [[1/3.,1/3.],[3/5.,1/5.],[1/5.,3/5.],[1/5.,1/5.]], [-9/32.,25/96.,25/96.,25/96.] )

  def testUniform( self ):
    self._test( self.line, 'uniform1', [[.5]], [1.] )
    self._test( self.line, 'uniform2', [[.25],[.75]], [.5,.5] )
    self._test( self.line, 'uniform3', [[1/6.],[1/2.],[5/6.]], [1/3.]*3 )
    self._test( self.triangle, 'uniform1', [[1/3.,1/3.]], [.5] )
    self._test( self.triangle, 'uniform2', [[1/6.,1/6.],[1/6.,2/3.],[2/3.,1/6.],[1/3.,1/3.]], [1/8.]*4 )
    self._test( self.quad, 'uniform1', [[.5,.5]], [1.] )
    self._test( self.quad, 'uniform2', [[.25,.25],[.25,.75],[.75,.25],[.75,.75]], [1/4.]*4 )
    self._test( self.quad, 'uniform1,3', [[.5,1/6.],[.5,1/2.],[.5,5/6.]], [1/3.]*3 )
    self._test( self.hexagon, 'uniform1', [[.5,.5,.5]], [1.] )
    self._test( self.prism, 'uniform1', [[.5,1/3.,1/3.]], [.5] )
    self._test( self.prism, 'uniform2', [[.25,1/6.,1/6.],[.25,1/6.,2/3.],[.25,2/3.,1/6.],[.25,1/3.,1/3.],[.75,1/6.,1/6.],[.75,1/6.,2/3.],[.75,2/3.,1/6.],[.75,1/3.,1/3.]], [1/16.]*8 )

  def testMixed( self ):
    a = numpy.sqrt(1/3.)
    self._test( self.quad, 'uniform1*gauss1', [[.5,.5]], [1.] )
    self._test( self.quad, 'uniform2*gauss1', [[.25,.5],[.75,.5]], [.5,.5] )
    self._test( self.quad, 'uniform1*gauss2', [[.5,.5-.5*a],[.5,.5+.5*a]], [.5,.5] )
    self._test( self.quad, 'uniform2*gauss2', [[.25,.5-.5*a],[.25,.5+.5*a],[.75,.5-.5*a],[.75,.5+.5*a]], [1/4.]*4 )
    self._test( self.prism, 'uniform1*gauss2', [[.5,2/3.,1/6.],[.5,1/6.,2/3.],[.5,1/6.,1/6.]], [1/6.]*3 )
