from nutils import *
from . import *

@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    if not self.curved:
      self.domain, self.geom = mesh.rectilinear([[1,1.5,2],[-1,0],[0,2,4]][:self.ndims])
      self.curv = 0
    else:
      assert self.ndims == 2
      nodes = numpy.linspace(-.25*numpy.pi, .25*numpy.pi, 3)
      self.domain, (xi,eta) = mesh.rectilinear([nodes, nodes])
      self.geom = numpy.sqrt(2) * function.stack([function.sin(xi) * function.cos(eta), function.cos(xi) * function.sin(eta)])
      self.curv = 1

  def zero(self):
    zero = self.domain.boundary.integrate(self.geom.normal(), geometry=self.geom, ischeme='gauss9')
    numpy.testing.assert_almost_equal(zero, 0)

  def volume(self):
    volume = self.domain.integrate(1, geometry=self.geom, ischeme='gauss9')
    volumes = self.domain.boundary.integrate(self.geom * self.geom.normal(), geometry=self.geom, ischeme='gauss9')
    numpy.testing.assert_almost_equal(volume, volumes)

  def interfaces(self):
    funcsp = self.domain.discontfunc(degree=2)
    f = (funcsp[:,_] * numpy.arange(funcsp.shape[0]*self.ndims).reshape(-1,self.ndims)).sum(0)
    g = funcsp.dot(numpy.arange(funcsp.shape[0]))

    fg1 = self.domain.integrate((f * g.grad(self.geom)).sum(-1), geometry=self.geom, ischeme='gauss9')
    fg2 = self.domain.boundary.integrate((f*g).dotnorm(self.geom), geometry=self.geom, ischeme='gauss9') \
        - self.domain.interfaces.integrate(function.jump(f*g).dotnorm(self.geom), geometry=self.geom, ischeme='gauss9') \
        - self.domain.integrate(f.div(self.geom) * g, geometry=self.geom, ischeme='gauss9')

    numpy.testing.assert_almost_equal(fg1, fg2)

  def curvature(self):
    c = self.domain.boundary.elem_eval(self.geom.curvature(), ischeme='uniform1', separate=False)
    numpy.testing.assert_almost_equal(c, self.curv)

  @parametrize.enable_if(lambda curved, **params: not curved)
  def test_boundaries(self):
    normal = self.geom.normal()
    boundary = self.domain.boundary
    for name, n in zip(['right','top','back'][:self.ndims], numpy.eye(self.ndims)):
      numpy.testing.assert_almost_equal(boundary[name].elem_eval(normal, ischeme='gauss9', separate=False)-n, 0)
    for name, n in zip(['left','bottom','front'][:self.ndims], -numpy.eye(self.ndims)):
      numpy.testing.assert_almost_equal(boundary[name].elem_eval(normal, ischeme='gauss9', separate=False)-n, 0)

check('2d', ndims=2, curved=False)
check('2dcurved', ndims=2, curved=True)
check('3d', ndims=3, curved=False)
