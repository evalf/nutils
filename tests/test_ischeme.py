from nutils import *
from nutils.testing import *

@parametrize
class check(TestCase):

  def test(self):
    points, weights = self.ref.getischeme(self.ptype)
    points.ndim == 2
    self.assertEqual(points.shape[1], self.ref.ndims)
    numpy.testing.assert_almost_equal(points, self.target_points)
    if self.target_weights is None:
      self.assertIsNone(weights)
    else:
      self.assertEqual(weights.ndim, 1)
      self.assertEqual(weights.shape[0], points.shape[0])
      numpy.testing.assert_almost_equal(weights, self.target_weights)

_check = lambda refname, ref, ptype, target_points, target_weights=None: check(refname+','+ptype, ref=ref, ptype=ptype, target_points=target_points, target_weights=target_weights)

ref = element.getsimplex(1)
a = numpy.sqrt(1/3.)
b = numpy.sqrt(3/5.)
_check('line', ref, 'gauss1', [[.5]], [1.])
_check('line', ref, 'gauss2', [[.5-.5*a],[.5+.5*a]], [.5,.5])
_check('line', ref, 'gauss3', [[.5-.5*a],[.5+.5*a]], [.5,.5])
_check('line', ref, 'gauss4', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.])
_check('line', ref, 'gauss5', [[.5-.5*b],[.5],[.5+.5*b]], [5/18.,4/9.,5/18.])
_check('line', ref, 'uniform1', [[.5]], [1.])
_check('line', ref, 'uniform2', [[.25],[.75]], [.5,.5])
_check('line', ref, 'uniform3', [[1/6.],[1/2.],[5/6.]], [1/3.]*3)

ref = element.getsimplex(1)**2
a = numpy.sqrt(1/3.)
b = numpy.sqrt(3/5.)
_check('quad', ref, 'gauss1', [[.5,.5]], [1.])
_check('quad', ref, 'gauss2', [[.5-.5*a,.5-.5*a],[.5-.5*a,.5+.5*a],[.5+.5*a,.5-.5*a],[.5+.5*a,.5+.5*a]], [1/4.]*4)
_check('quad', ref, 'gauss1,4', [[.5,.5-.5*b],[.5,.5],[.5,.5+.5*b]], [5/18.,4/9.,5/18.])
_check('quad', ref, 'uniform1', [[.5,.5]], [1.])
_check('quad', ref, 'uniform2', [[.25,.25],[.25,.75],[.75,.25],[.75,.75]], [1/4.]*4)
_check('quad', ref, 'uniform1,3', [[.5,1/6.],[.5,1/2.],[.5,5/6.]], [1/3.]*3)
_check('quad', ref, 'uniform1*gauss1', [[.5,.5]], [1.])
_check('quad', ref, 'uniform2*gauss1', [[.25,.5],[.75,.5]], [.5,.5])
_check('quad', ref, 'uniform1*gauss2', [[.5,.5-.5*a],[.5,.5+.5*a]], [.5,.5])
_check('quad', ref, 'uniform2*gauss2', [[.25,.5-.5*a],[.25,.5+.5*a],[.75,.5-.5*a],[.75,.5+.5*a]], [1/4.]*4)

ref = element.getsimplex(2)
_check('triangle', ref, 'gauss1', [[1/3.,1/3.]], [.5])
_check('triangle', ref, 'gauss2', [[1/6.,1/6.],[2/3.,1/6.],[1/6.,2/3.]], [1/6.]*3)
_check('triangle', ref, 'gauss3', [[1/3.,1/3.],[1/5.,1/5.],[3/5.,1/5.],[1/5.,3/5.]], [-9/32.,25/96.,25/96.,25/96.])
_check('triangle', ref, 'uniform1', [[1/3.,1/3.]], [.5])
_check('triangle', ref, 'uniform2', [[1/6.,1/6.],[1/6.,2/3.],[2/3.,1/6.],[1/3.,1/3.]], [1/8.]*4)

ref = element.getsimplex(1)**3
_check('hexagon', ref, 'uniform1', [[.5,.5,.5]], [1.])

ref = element.getsimplex(1)*element.getsimplex(2)
_check('prism', ref, 'uniform1', [[.5,1/3.,1/3.]], [.5])
_check('prism', ref, 'uniform2', [[.25,1/6.,1/6.],[.25,1/6.,2/3.],[.25,2/3.,1/6.],[.25,1/3.,1/3.],[.75,1/6.,1/6.],[.75,1/6.,2/3.],[.75,2/3.,1/6.],[.75,1/3.,1/3.]], [1/16.]*8)
_check('prism', ref, 'uniform1*gauss2', [[.5,1/6.,1/6.],[.5,2/3.,1/6.],[.5,1/6.,2/3.]], [1/6.]*3)
