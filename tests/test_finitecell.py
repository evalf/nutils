from nutils import *
from . import *


class hierarchical(TestCase):

  def setUp(self):
    super().setUp()
    self.ref0, self.geom = mesh.rectilinear([[0,1,2]])
    self.e1, self.e2 = self.ref0
    self.ref1 = self.ref0.refined_by([self.e2])
    self.e3, self.e4, self.e5 = self.ref1
    self.ref2 = self.ref1.refined_by([self.e4])

    # Topologies:
    # ref0    [  .  .  .  |  .  .  .  ]
    # ref1    [  .  .  .  |  .  |  .  ]
    # ref2    [  .  .  .  |  |  |  .  ]
    # trimmed [  .  .  .  |]

  def test_untrimmed(self, makeplots=False):
    basis = self.ref2.basis('std', degree=1)
    self.assertEqual(basis.shape, (5,))
    x, y = self.ref2.elem_eval([self.geom[0], basis], ischeme='bezier2', separate=False)
    self.assertTrue(numpy.equal(y, .25 * numpy.array(
      [[4,0,0,0,0],
       [0,4,0,0,0],
       [0,3,2,0,4],
       [0,2,4,0,0],
       [0,0,0,4,0]])[[0,1,1,2,2,3,3,4]]).all())
    if makeplots:
      with plot.PyPlot('basis') as plt:
        plt.plot(x, y)

  def test_trimmed(self, makeplots=False):
    levelset = 1.125 - self.geom[0]
    trimmed = self.ref0.trim(levelset, maxrefine=3).refined_by([self.e2]).refined_by([self.e4])
    trimbasis = trimmed.basis('std', degree=1)
    x, y = trimmed.simplex.elem_eval([self.geom[0], trimbasis], ischeme='bezier2', separate=False)
    self.assertTrue(numpy.equal(y, .125 * numpy.array(
      [[8,0,0],
       [0,8,0],
       [0,7,4]])[[0,1,1,2]]).all())
    if makeplots:
      with plot.PyPlot('basis') as plt:
        plt.plot(x, y)


class trimmedboundary(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,1,2],[0,1,2]])
    self.left = self.domain[:1].withboundary(leftbnd=...)
    self.leftbasis = self.left.basis('std', degree=1)
    self.right = self.domain[1:].withboundary(rightbnd=...)
    self.rightbasis = self.right.basis('std', degree=1)
    self.trimmed = self.domain - self.right

  def test_boundary_length(self):
    self.assertEqual(self.trimmed.boundary.integrate(1, geometry=self.geom, ischeme='gauss1'), 6)

  def test_trimmed_boundary_length(self):
    self.assertEqual(self.trimmed.boundary['rightbnd'].integrate(1, geometry=self.geom, ischeme='gauss1'), 2)

  def test_trimmed_boundary(self):
    left = self.trimmed.boundary['rightbnd']
    self.assertTrue(numpy.any(left.elem_eval(self.leftbasis, ischeme='gauss1', separate=False)))
    with self.assertRaises(function.EvaluationError):
      left.elem_eval(function.opposite(self.leftbasis), ischeme='gauss1', separate=False)
    self.assertTrue(numpy.any(left.elem_eval(function.opposite(self.rightbasis), ischeme='gauss1', separate=False)))
    with self.assertRaises(function.EvaluationError):
      left.elem_eval(self.rightbasis, ischeme='gauss1', separate=False)


class specialcases_2d(TestCase):

  eps = .0001

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,.5,1]]*2)

  def test_almost_all_positive(self):
    x, y = self.geom
    self.domain.trim((x-y) * (x-y+.25), maxrefine=1)

  def test_inter_elem(self):
    x, y = self.geom
    for xi, eta, direction in (x,y,'x'), (y,x,'y'):
      for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (xi-.5,'ramp'), (xi-eta,'tilt'):
        for maxrefine in 0, 1, 2:
          with self.subTest(direction=direction, how=how, maxrefine=maxrefine):
            self.domain.trim(eta-.75+self.eps*perturb, maxrefine=maxrefine).check_boundary(self.geom, elemwise=True, print=self.fail)

  def test_intra_elem(self):
    x, y = self.geom
    for xi, eta, direction in (x,y,'x'), (y,x,'y'):
      for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (xi-.5,'ramp'), (xi-eta,'tilt'):
        for maxrefine in 0, 1:
          with self.subTest(direction=('x' if xi is x else 'y'), how=how, maxrefine=maxrefine):
            pos = self.domain.trim(eta-.5+self.eps*perturb, maxrefine=maxrefine)
            pos.check_boundary(self.geom, elemwise=True, print=self.fail)
            neg = self.domain - pos
            neg.check_boundary(self.geom, elemwise=True, print=self.fail)

class specialcases_3d(TestCase):

  eps = .0001

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,.5],[0,.5],[0,.5,1]])

  def test_inter_elem(self):
    x, y, z = self.geom
    for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (x-.5,'ramp'), (x-y,'tilt'):
      for maxrefine in 0, 1, 2:
        with self.subTest(how=how, maxrefine=maxrefine):
          self.domain.trim(z-.75+self.eps*perturb, maxrefine=maxrefine).check_boundary(self.geom, elemwise=True, print=self.fail)


class setoperations(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[-.5,-1./6,1./6,.5]]*2) # unit square
    x, y = self.geom
    bottomright = self.domain.trim(x-y, maxrefine=0, name='trim1')
    self.right = bottomright.trim(x+y, maxrefine=0, name='trim2')
    self.bottom = bottomright - self.right
    topleft = self.domain - bottomright
    self.top = topleft.trim(x+y, maxrefine=0, name='trim2')
    self.left = topleft - self.top

  def test_boundary(self):
    Lexact = 1+numpy.sqrt(2)
    for name, dom in ('left',self.left), ('top',self.top), ('right',self.right), ('bottom',self.bottom):
      with self.subTest(name):
        L = dom.boundary.integrate(1, geometry=self.geom, ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, 1+numpy.sqrt(2)), 'full boundary: wrong length: {} != {}'.format(L, 1+numpy.sqrt(2)))
        L = dom.boundary[name].integrate(1, geometry=self.geom, ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, 1), '{}: wrong length: {} != {}'.format(name, L, 1))
        L = dom.boundary['trim1' if name not in ('left','top') else 'trim1'].integrate(1, geometry=self.geom, ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, .5*numpy.sqrt(2)), 'trim1: wrong length: {} != {}'.format(L, .5*numpy.sqrt(2)))
        L = dom.boundary['trim2' if name not in ('left','bottom') else 'trim2'].integrate(1, geometry=self.geom, ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, .5*numpy.sqrt(2)), 'trim2: wrong length: {} != {}'.format(L, .5*numpy.sqrt(2)))

  def test_union(self):
    self.assertEqual((self.top|self.left) | (self.right|self.bottom), self.domain)
    union = (self.right|self.left) | (self.top|self.bottom)
    self.assertIsInstance(union, topology.UnionTopology)
    self.assertEqual(set(union), set(self.domain))


@parametrize
class cutdomain(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear((numpy.linspace(0,1,self.nelems+1),)*self.ndims)
    self.radius = numpy.sqrt(.5)
    levelset = self.radius**2 - (self.geom**2).sum(-1)
    self.pos = self.domain.trim(levelset=levelset, maxrefine=self.maxrefine)
    self.neg = self.domain - self.pos
    V = 1.
    Vprev = 1. / (numpy.pi*self.radius)
    for idim in range(self.ndims):
      S = Vprev * (2*numpy.pi*self.radius)
      Vprev = V
      V = S * (self.radius/(idim+1))
    self.exact_volume = V / 2**self.ndims
    self.exact_trimsurface = S / 2**self.ndims
    self.exact_totalsurface = self.exact_trimsurface + Vprev / (2**(self.ndims-1)) * self.ndims

  def test_volume(self):
    volume = self.pos.volume(self.geom)
    volerr = abs(volume - self.exact_volume) / self.exact_volume
    log.user('volume error:', volerr)
    self.assertLess(volerr, self.errtol, 'volume tolerance not met')

  def test_div(self):
    for name, dom in ('pos',self.pos), ('neg',self.neg):
      with self.subTest(name):
        dom.check_boundary(self.geom, elemwise=True, print=self.fail)

  def test_surface(self):
    trimsurface = self.pos.boundary['trimmed'].volume(self.geom)
    trimerr = abs(trimsurface - self.exact_trimsurface) / self.exact_trimsurface
    log.user('trim surface error:', trimerr)
    totalsurface = self.pos.boundary.volume(self.geom)
    totalerr = abs(totalsurface - self.exact_totalsurface) / self.exact_totalsurface
    log.user('total surface error:', totalerr)
    self.assertLess(trimerr, self.errtol, 'trim surface tolerance not met')
    self.assertLess(totalerr, self.errtol, 'total surface tolerance not met')

  def test_locate(self):
    curvegeom = self.geom * (1 + .1 * function.sin(function.norm2(self.geom)*numpy.pi/self.radius)) # interface preserving non-polynomial scaling
    for p in numpy.linspace(.001, .999, 20):
      with self.subTest(p=p):
        point = p * .5**numpy.arange(self.domain.ndims)
        r = numpy.linalg.norm(point)
        try:
          ptopo = self.pos.locate(curvegeom, points=[point])
        except topology.LocateError:
          self.assertGreater(r, self.radius)
        else:
          self.assertLessEqual(r, self.radius)
          x, = ptopo.elem_eval(curvegeom, 'gauss1')
          numpy.testing.assert_almost_equal(x, point)

cutdomain('sphere', ndims=3, nelems=2, maxrefine=3, errtol=6e-3)
cutdomain('circle', ndims=2, nelems=2, maxrefine=5, errtol=2.1e-4)

class multitrim(TestCase):

  def test(self):
    domain, geom = mesh.rectilinear([[-1,1],[-1,1]])
    geom_rel = (function.rotmat(numpy.pi/6) * geom).sum(-1)
    for itrim in range(4):
      domain = domain.trim(.7+(1-itrim%2*2)*geom_rel[itrim//2], maxrefine=1, name='trim{}'.format(itrim), ndivisions=16)
    domain.check_boundary(geom, elemwise=True, print=self.fail)
    for itrim in range(4):
      L = domain.boundary['trim{}'.format(itrim)].integrate(1, geometry=geom, ischeme='gauss1')
      numpy.testing.assert_almost_equal(L, 1.4, decimal=4)
    L = domain.boundary.integrate(1, geometry=geom, ischeme='gauss1')
    numpy.testing.assert_almost_equal(L, 5.6, decimal=4)


class leveltopo(TestCase):

  def setUp(self):
    super().setUp()
    self.domain0, self.geom = mesh.rectilinear([2,2])
    self.domain1 = self.domain0.refined

  def test_uniform(self):
    domain2 = self.domain1.refined
    basis = self.domain0.basis('std', degree=1)
    level = basis.dot((numpy.arange(len(basis))%2)-.5)
    trimtopoA = self.domain0.trim(level, maxrefine=2)
    trimtopoB = self.domain0.trim(level, maxrefine=2, leveltopo=domain2)
    self.assertEqual(trimtopoA.elements, trimtopoB.elements)

  def test_uniformfail(self):
    with self.assertRaises(Exception):
      domain2 = self.domain1.refined
      basis = self.domain0.basis('std', degree=1)
      level = basis.dot((numpy.arange(len(basis))%2)-.5)
      trimtopo = self.domain0.trim(level, maxrefine=1, leveltopo=domain2)

  def test_hierarchical(self):
    domain2 = self.domain1.refined_by(self.domain1.elements[:1])
    basis = domain2.basis('std', degree=1)
    level = basis.dot((numpy.arange(len(basis))%2)-.5)
    trimtopo = self.domain0.trim(level, maxrefine=2, leveltopo=domain2)

  def test_hierarchicalfail(self):
    with self.assertRaises(Exception):
      domain2 = self.domain1.refined_by(self.domain1.elements[:1])
      basis = domain2.basis('std', degree=1)
      level = basis.dot((numpy.arange(len(basis))%2)-.5)
      trimtopo = self.domain0.trim(level, maxrefine=1, leveltopo=domain2)


class trim_conforming(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.rectilinear([4,4])
    self.domain1 = self.domain.trim(3-geom[0], maxrefine=2, name='trimright')
    self.domain2 = self.domain1.trim(3-geom[1], maxrefine=2, name='trimtop')
    self.domain3 = self.domain2.trim(geom[0]-1, maxrefine=2, name='trimleft')
    self.domain4 = self.domain3.trim(geom[1]-1, maxrefine=2, name='trimbottom')
    self.domain5 = self.domain.trim(3-geom[0], maxrefine=2, name='trimtopright').trim(3-geom[1], maxrefine=2, name='trimtopright')

  def test_untrimmed(self):
    self.assertEqual(len(self.domain.interfaces), 24)
    self.assertEqual(len(self.domain.boundary), 16)

  def test_trimright(self):
    self.assertEqual(len(self.domain1.interfaces), 17)
    self.assertEqual(len(self.domain1.boundary), 14)
    self.assertEqual(len(self.domain1.boundary['trimright']), 4)

  def test_trimtop(self):
    self.assertEqual(len(self.domain2.interfaces), 12)
    self.assertEqual(len(self.domain2.boundary), 12)
    self.assertEqual(len(self.domain2.boundary['trimright']), 3)
    self.assertEqual(len(self.domain2.boundary['trimtop']), 3)

  def test_trimleft(self):
    self.assertEqual(len(self.domain3.interfaces), 7)
    self.assertEqual(len(self.domain3.boundary), 10)
    self.assertEqual(len(self.domain3.boundary['trimright']), 3)
    self.assertEqual(len(self.domain3.boundary['trimtop']), 2)
    self.assertEqual(len(self.domain3.boundary['trimleft']), 3)

  def test_trimbottom(self):
    self.assertEqual(len(self.domain4.interfaces), 4)
    self.assertEqual(len(self.domain4.boundary), 8)
    self.assertEqual(len(self.domain4.boundary['trimright']), 2)
    self.assertEqual(len(self.domain4.boundary['trimtop']), 2)
    self.assertEqual(len(self.domain4.boundary['trimleft']), 2)
    self.assertEqual(len(self.domain4.boundary['trimbottom']), 2)

  def test_trimtopright(self):
    self.assertEqual(len(self.domain5.interfaces), 12)
    self.assertEqual(len(self.domain5.boundary), 12)
    self.assertEqual(len(self.domain5.boundary['trimtopright']), 6)
