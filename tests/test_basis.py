from nutils import *
from . import *

@parametrize
class basis(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,1,2]]*self.ndims) if self.ndims else mesh.demo()
    self.basis = self.domain.basis(self.btype, degree=self.degree)
    self.gauss = 'gauss{}'.format(2*self.degree)

  def test_pum(self):
    error = numpy.sqrt(self.domain.integrate( (1-self.basis.sum(0))**2, geometry=self.geom, ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=14)

  def test_poly(self):
    target = (self.geom**self.degree).sum(-1)
    projection = self.domain.projection(target, onto=self.basis, geometry=self.geom, ischeme=self.gauss, droptol=0)
    error = numpy.sqrt(self.domain.integrate((target-projection)**2, geometry=self.geom, ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

basis('tri:discont0', btype='discont', degree=0, ndims=None)
basis('tri:discont1', btype='discont', degree=1, ndims=None)
basis('tri:std1', btype='std', degree=1, ndims=None)
for ndims in range(1, 4):
  for btype in 'discont', 'std', 'spline':
    for degree in range(0 if btype == 'discont' else 1, 4):
      basis(btype=btype, degree=degree, ndims=ndims)
