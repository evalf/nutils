from nutils import *
from . import *
import scipy.special, re


@parametrize
class gauss(TestCase):
  # Gaussian quadrature and exact integration on different element types
  maxdegree=7
  eps=1e-12

  def setUp(self):
    super().setUp()
    self.monomials = numpy.mgrid[ (slice(self.maxdegree),)*self.ndims ].reshape(self.ndims,-1).T
    if self.istensor:
      self.ref = element.getsimplex(1)**self.ndims
      self.integrals = numpy.reciprocal((self.monomials+1.).prod(-1))
    else:
      self.ref = element.getsimplex(self.ndims)
      self.integrals = scipy.special.gamma(self.monomials+1.).prod(-1) / scipy.special.gamma(self.ndims+1+self.monomials.sum(-1))

  def test_degree(self):
    for degree in range(1, self.maxdegree+1):
      with self.subTest(degree=degree):
        points, weights = self.ref.getischeme('gauss{}'.format(degree))
        for monomial, integral in zip(self.monomials, self.integrals):
          result = numpy.dot(weights, numpy.prod(points**monomial,axis=-1))
          error = abs(result-integral) / integral
          expect_exact = degree // 2 >= max(monomial) // 2 if self.istensor else degree >= sum(monomial)
          if expect_exact:
            self.assertLess(error, self.eps, 'integration should be exact')
          else:
            self.assertGreater(error, self.eps, 'integration should not be exact')
            # Counterexamples can be constructed, but in the case of monomials with maxdegree<8 this assert is verified

gauss('line', ndims=1, istensor=True)
gauss('quad', ndims=2, istensor=True)
gauss('hex', ndims=3, istensor=True)
gauss('tri', ndims=2, istensor=False)
gauss('tet', ndims=3, istensor=False, maxdegree=8)
