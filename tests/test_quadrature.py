from nutils import *
from nutils.testing import *
import math, re


@parametrize
class gauss(TestCase):
  # Gaussian quadrature and exact integration on different element types
  maxdegree=7

  def setUp(self):
    super().setUp()
    self.monomials = numpy.mgrid[ (slice(self.maxdegree),)*self.ndims ].reshape(self.ndims,-1).T
    if self.istensor:
      self.ref = element.getsimplex(1)**self.ndims
      self.integrals = numpy.reciprocal((self.monomials+1.).prod(-1))
    else:
      self.ref = element.getsimplex(self.ndims)
      gamma = numpy.vectorize(math.gamma)
      self.integrals = gamma(self.monomials+1.).prod(-1) / gamma(self.ndims+1+self.monomials.sum(-1))

  def test_gauss(self):
    for degree in range(1, self.maxdegree+1):
      with self.subTest(degree=degree):
        points = self.ref.getpoints('gauss', degree)
        for monomial, integral in zip(self.monomials, self.integrals):
          result = numpy.dot(points.weights, numpy.prod(points.coords**monomial, axis=-1))
          expect_exact = degree // 2 >= max(monomial) // 2 if self.istensor else degree >= sum(monomial)
          if expect_exact:
            self.assertAlmostEqual(result/integral, 1, msg='integration should be exact', places=12)
          else:
            self.assertNotAlmostEqual(result/integral, 1, msg='integration should not be exact', places=12)
            # Counterexamples can be constructed, but in the case of monomials with maxdegree<8 this assert is verified

gauss('line', ndims=1, istensor=True)
gauss('quad', ndims=2, istensor=True)
gauss('hex', ndims=3, istensor=True)
gauss('tri', ndims=2, istensor=False)
gauss('tet', ndims=3, istensor=False, maxdegree=8)
