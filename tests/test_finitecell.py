from nutils import *
from nutils.testing import *


class hierarchical(TestCase):

  def setUp(self):
    super().setUp()
    self.ref0, self.geom = mesh.rectilinear([[0,1,2]])
    self.ref1 = self.ref0.refined_by([1])
    self.ref2 = self.ref1.refined_by([1])

    # Topologies:
    # ref0    [  .  .  .  |  .  .  .  ]
    # ref1    [  .  .  .  |  .  |  .  ]
    # ref2    [  .  .  .  |  |  |  .  ]
    # trimmed [  .  .  .  |]

  def test_untrimmed(self, makeplots=False):
    basis = self.ref2.basis('h-std', degree=1)
    self.assertEqual(basis.shape, (5,))
    x, y = self.ref2.sample('bezier', 2).eval([self.geom[0], basis])
    self.assertTrue((abs(y - .25 * numpy.array(
      [[4,0,0,0,0],
       [0,4,0,0,0],
       [0,3,2,0,4],
       [0,2,4,0,0],
       [0,0,0,4,0]])[[0,1,1,2,2,3,3,4]]) < 1e15).all())
    if makeplots:
      with plot.PyPlot('basis') as plt:
        plt.plot(x, y)

  def test_trimmed(self, makeplots=False):
    levelset = 1.125 - self.geom[0]
    trimmed = self.ref0.trim(levelset, maxrefine=3).refined_by([1]).refined_by([1])
    trimbasis = trimmed.basis('h-std', degree=1)
    x, y = trimmed.sample('bezier', 2).eval([self.geom[0], trimbasis])
    self.assertTrue((abs(y - .125 * numpy.array(
      [[8,0,0],
       [0,8,0],
       [0,7,4]])[[0,1,1,2]]) < 1e15).all())
    if makeplots:
      with plot.PyPlot('basis') as plt:
        plt.plot(x, y)


@parametrize
class trimmedboundary(TestCase):

  def setUp(self):
    super().setUp()
    if self.boundary:
      domain0, self.geom = mesh.rectilinear([2,2,2])
      self.domain0 = domain0.boundary['front']
    else:
      self.domain0, self.geom = mesh.rectilinear([2,2])
    if self.gridline:
      self.domain1 = self.domain0 - self.domain0[1:].withboundary(trimmed='left')
    else:
      self.domain1 = self.domain0.trim(1.25 - self.geom[0], maxrefine=1)
    self.domain2 = self.domain1.refined_by(filter(self.domain1.transforms.contains, self.domain0[:,1:].transforms))

  def test_boundary_length(self):
    self.assertEqual(self.domain2.boundary.integrate(function.J(self.geom), ischeme='gauss1'), 6 if self.gridline else 6.5)

  def test_trimmed_boundary_length(self):
    self.assertEqual(self.domain2.boundary['trimmed'].integrate(function.J(self.geom), ischeme='gauss1'), 2)

  @parametrize.enable_if(lambda gridline, **params: gridline)
  def test_trimmed_boundary(self):
    trimmed = self.domain2.boundary['trimmed']
    gauss1 = trimmed.sample('gauss', 1)
    leftbasis = self.domain0[:1].basis('std', degree=1)
    self.assertTrue(numpy.any(gauss1.eval(leftbasis)))
    with self.assertRaises(function.EvaluationError):
      gauss1.eval(function.opposite(leftbasis))
    rightbasis = self.domain0[1:].basis('std', degree=1)
    self.assertTrue(numpy.any(gauss1.eval(function.opposite(rightbasis))))
    with self.assertRaises(function.EvaluationError):
      gauss1.eval(rightbasis)

for boundary in True, False:
  for gridline in True, False:
    trimmedboundary(boundary=boundary, gridline=gridline)


class specialcases_2d(TestCase):

  eps = .0001

  def check_connectivity(self, connectivity):
    for ielem, ioppelems in enumerate(connectivity):
      for ioppelem in ioppelems:
        if ioppelem != -1:
          self.assertIn(ielem, connectivity[ioppelem])

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,.5,1]]*2)

  def test_almost_all_positive(self):
    x, y = self.geom
    dom = self.domain.trim((x-y) * (x-y+.25), maxrefine=1)
    self.check_connectivity(dom.connectivity)

  def test_inter_elem(self):
    x, y = self.geom
    for xi, eta, direction in (x,y,'x'), (y,x,'y'):
      for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (xi-.5,'ramp'), (xi-eta,'tilt'):
        for maxrefine in 0, 1, 2:
          with self.subTest(direction=direction, how=how, maxrefine=maxrefine):
            dom = self.domain.trim(eta-.75+self.eps*perturb, maxrefine=maxrefine)
            dom.check_boundary(self.geom, elemwise=True, print=self.fail)
            self.check_connectivity(dom.connectivity)

  def test_intra_elem(self):
    x, y = self.geom
    for xi, eta, direction in (x,y,'x'), (y,x,'y'):
      for perturb, how in (0,'mid'), (1,'pos'), (-1,'neg'), (xi-.5,'ramp'), (xi-eta,'tilt'):
        for maxrefine in 0, 1:
          with self.subTest(direction=('x' if xi is x else 'y'), how=how, maxrefine=maxrefine):
            pos = self.domain.trim(eta-.5+self.eps*perturb, maxrefine=maxrefine)
            pos.check_boundary(self.geom, elemwise=True, print=self.fail)
            self.check_connectivity(pos.connectivity)
            neg = self.domain - pos
            neg.check_boundary(self.geom, elemwise=True, print=self.fail)
            self.check_connectivity(neg.connectivity)

  def test_inter_intra1(self):
    x, y = self.geom
    ttopo = self.domain.trim((x-.5)*(x-.75), maxrefine=2)
    self.check_connectivity(ttopo.connectivity)
    self.assertEqual(len(ttopo.boundary), 14)
    self.assertEqual(len(ttopo.interfaces), 2)

  def test_inter_intra2(self):
    x, y = self.geom
    ttopo = self.domain.trim((x-.25)*(x-.5), maxrefine=2)
    self.check_connectivity(ttopo.connectivity)
    self.assertEqual(len(ttopo.boundary), 14)
    self.assertEqual(len(ttopo.interfaces), 2)

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
        L = dom.boundary.integrate(function.J(self.geom), ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, 1+numpy.sqrt(2)), 'full boundary: wrong length: {} != {}'.format(L, 1+numpy.sqrt(2)))
        L = dom.boundary[name].integrate(function.J(self.geom), ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, 1), '{}: wrong length: {} != {}'.format(name, L, 1))
        L = dom.boundary['trim1' if name not in ('left','top') else 'trim1'].integrate(function.J(self.geom), ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, .5*numpy.sqrt(2)), 'trim1: wrong length: {} != {}'.format(L, .5*numpy.sqrt(2)))
        L = dom.boundary['trim2' if name not in ('left','bottom') else 'trim2'].integrate(function.J(self.geom), ischeme='gauss1')
        self.assertTrue(numpy.isclose(L, .5*numpy.sqrt(2)), 'trim2: wrong length: {} != {}'.format(L, .5*numpy.sqrt(2)))

  def test_union(self):
    self.assertEqual((self.top|self.left) | (self.right|self.bottom), self.domain)
    union = topology.UnionTopology([self.right, self.left, self.top, self.bottom])
    self.assertIsInstance(union, topology.UnionTopology)
    self.assertEqual(set(union.references), set(self.domain.references))
    self.assertEqual(set(union.transforms), set(self.domain.transforms))
    self.assertEqual(set(union.opposites), set(self.domain.opposites))


@parametrize
class cutdomain(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear((numpy.linspace(0,1,self.nelems+1),)*3)
    if self.ndims == 3:
      self.domain = domain
      self.geom = geom
    elif self.ndims == 2: # create a 3d boundary instead of a plain 2d domain for added complexity
      self.domain = domain.boundary['back']
      self.geom = geom[:2]
    else:
      raise Exception('invalid dimension: ndims={}'.format(self.ndims))
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
          sample = self.pos.locate(curvegeom, [point])
        except topology.LocateError:
          self.assertGreater(r, self.radius)
        else:
          self.assertLessEqual(r, self.radius)
          x, = sample.eval(curvegeom)
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
      L = domain.boundary['trim{}'.format(itrim)].integrate(function.J(geom), ischeme='gauss1')
      numpy.testing.assert_almost_equal(L, 1.4, decimal=4)
    L = domain.boundary.integrate(function.J(geom), ischeme='gauss1')
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
    self.assertEqual(tuple(trimtopoA.references), tuple(trimtopoB.references))
    self.assertEqual(tuple(trimtopoA.transforms), tuple(trimtopoB.transforms))
    self.assertEqual(tuple(trimtopoA.opposites), tuple(trimtopoB.opposites))

  def test_uniformfail(self):
    with self.assertRaises(Exception):
      domain2 = self.domain1.refined
      basis = self.domain0.basis('std', degree=1)
      level = basis.dot((numpy.arange(len(basis))%2)-.5)
      trimtopo = self.domain0.trim(level, maxrefine=1, leveltopo=domain2)

  def test_hierarchical(self):
    domain2 = self.domain1.refined_by([0])
    basis = domain2.basis('h-std', degree=1)
    level = basis.dot((numpy.arange(len(basis))%2)-.5)
    trimtopo = self.domain0.trim(level, maxrefine=2, leveltopo=domain2)

  def test_hierarchicalfail(self):
    with self.assertRaises(Exception):
      domain2 = self.domain1.refined_by([0])
      basis = domain2.basis('h-std', degree=1)
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


class partialtrim(TestCase):

  # Test setup:
  # +-----+-----+
  # |         A |
  # .           |
  # + '.  +     +
  # |    '.     |
  # | B   |<----|--half of original A-B interface element
  # +-----+-----+

  def setUp(self):
    self.topo, geom = mesh.rectilinear([2,2])
    self.topoA = self.topo.trim(geom[0]-1+geom[1]*(geom[1]-.5), maxrefine=1)
    self.topoB = self.topo - self.topoA

  def test_topos(self):
    self.assertEqual(len(self.topoA), 4)
    self.assertEqual(len(self.topoB), 2)

  def test_boundaries(self):
    self.assertEqual(len(self.topoA.boundary), 11)
    self.assertEqual(len(self.topoB.boundary), 8)
    self.assertEqual(len(self.topoA.boundary['trimmed']), 5)
    self.assertEqual(len(self.topoB.boundary['trimmed']), 5)

  def test_interfaces(self):
    self.assertEqual(len(self.topoA.interfaces), 4)
    self.assertEqual(len(self.topoB.interfaces), 1)

  def test_transforms(self):
    self.assertEqual(set(self.topoA.boundary['trimmed'].transforms), set(self.topoB.boundary['trimmed'].opposites))
    self.assertEqual(set(self.topoB.boundary['trimmed'].transforms), set(self.topoA.boundary['trimmed'].opposites))

  def test_opposites(self):
    ielem = function.elemwise(self.topo.transforms, numpy.arange(4))
    sampleA = self.topoA.boundary['trimmed'].sample('uniform', 1)
    sampleB = self.topoB.boundary['trimmed'].sample('uniform', 1)
    self.assertEqual(set(sampleB.eval(ielem)), {0,1})
    self.assertEqual(set(sampleB.eval(function.opposite(ielem))), {0,1,2})
    self.assertEqual(set(sampleA.eval(ielem)), {0,1,2})
    self.assertEqual(set(sampleA.eval(function.opposite(ielem))), {0,1})

  def test_baseboundaries(self):
    # the base implementation should create the correct boundary topology but
    # without interface opposites and without the trimmed group
    for topo in self.topoA, self.topoB:
      alttopo = topology.ConnectedTopology(topo.references, topo.transforms, topo.opposites, topo.connectivity)
      self.assertEqual(dict(zip(alttopo.boundary.transforms, alttopo.boundary.references)), dict(zip(topo.boundary.transforms, topo.boundary.references)))
