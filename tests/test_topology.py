from nutils import *
from . import *
import numpy, copy, sys, pickle, subprocess, base64, itertools, os

grid = numpy.linspace(0., 1., 4)

def verify_connectivity(structure, geom):
  (e00,e01), (e10,e11) = structure

  a0 = geom.eval(_transforms=[e00.transform], _points=numpy.array([[0,1]]))
  a1 = geom.eval(_transforms=[e01.transform], _points=numpy.array([[0,0]]))
  numpy.testing.assert_array_almost_equal(a0, a1)

  b0 = geom.eval(_transforms=[e10.transform], _points=numpy.array([[1,1]]))
  b1 = geom.eval(_transforms=[e11.transform], _points=numpy.array([[1,0]]))
  numpy.testing.assert_array_almost_equal(b0, b1)

  c0 = geom.eval(_transforms=[e00.transform], _points=numpy.array([[1,0]]))
  c1 = geom.eval(_transforms=[e10.transform], _points=numpy.array([[0,0]]))
  numpy.testing.assert_array_almost_equal(c0, c1)

  d0 = geom.eval(_transforms=[e01.transform], _points=numpy.array([[1,1]]))
  d1 = geom.eval(_transforms=[e11.transform], _points=numpy.array([[0,1]]))
  numpy.testing.assert_array_almost_equal(d0, d1)

  x00 = geom.eval(_transforms=[e00.transform], _points=numpy.array([[1,1]]))
  x01 = geom.eval(_transforms=[e01.transform], _points=numpy.array([[1,0]]))
  x10 = geom.eval(_transforms=[e10.transform], _points=numpy.array([[0,1]]))
  x11 = geom.eval(_transforms=[e11.transform], _points=numpy.array([[0,0]]))
  numpy.testing.assert_array_almost_equal(x00, x01)
  numpy.testing.assert_array_almost_equal(x10, x11)
  numpy.testing.assert_array_almost_equal(x00, x11)

def verify_boundaries(domain, geom):
  # Test ∫_Ω f_,i = ∫_∂Ω f n_i.
  f = ((0.5 - geom)**2).sum(axis=0)
  lhs = domain.integrate(f.grad(geom)*function.J(geom), ischeme='gauss2')
  rhs = domain.boundary.integrate(f*function.normal(geom)*function.J(geom), ischeme='gauss2')
  numpy.testing.assert_array_almost_equal(lhs, rhs)

def verify_interfaces(domain, geom, periodic, interfaces=None, elemindicator=None):
  # If `periodic` is true, the domain should be a unit hypercube or this test
  # might fail.  The function `f` defined below is C0 continuous on a periodic
  # hypercube and Cinf continuous inside the hypercube.
  if interfaces is None:
    interfaces = domain.interfaces
  x1, x2, n1, n2 = interfaces.elem_eval([ geom, function.opposite(geom), geom.normal(), function.opposite(geom.normal()) ], 'gauss2', separate=False)
  if not periodic:
    numpy.testing.assert_array_almost_equal(x1, x2)
  numpy.testing.assert_array_almost_equal(n1, -n2)

  # Test ∫_E f_,i = ∫_∂E f n_i ∀ E in `domain`.
  f = ((0.5 - geom)**2).sum(axis=0)
  if elemindicator is None:
    elemindicator = domain.basis('discont', degree=0)
  elemindicator = elemindicator.vector(domain.ndims)
  lhs = domain.integrate((elemindicator*f.grad(geom)[None]).sum(axis=1)*function.J(geom), ischeme='gauss2')
  rhs = interfaces.integrate((-function.jump(elemindicator)*f*function.normal(geom)[None]).sum(axis=1)*function.J(geom), ischeme='gauss2')
  if len(domain.boundary):
    rhs += domain.boundary.integrate((elemindicator*f*function.normal(geom)[None]).sum(axis=1)*function.J(geom), ischeme='gauss2')
  numpy.testing.assert_array_almost_equal(lhs, rhs)


@parametrize
class elem_project(TestCase):

  def test_extraction(self):
    topo, geom = mesh.rectilinear([numpy.linspace(-1,1,4)]*self.ndims)

    splinebasis = topo.basis('spline', degree=self.degree)
    bezierbasis = topo.basis('spline', degree=self.degree, knotmultiplicities=[numpy.array([self.degree+1]+[self.degree]*(n-1)+[self.degree+1]) for n in topo.shape])

    splinevals, beziervals = topo.elem_eval([splinebasis,bezierbasis], ischeme='uniform2', separate=True)
    sextraction = topo.elem_project(splinebasis, degree=self.degree, check_exact=True)
    bextraction = topo.elem_project(bezierbasis, degree=self.degree, check_exact=True)
    for svals, (sien,sext), bvals, (bien,bext) in zip(splinevals,sextraction,beziervals,bextraction):
      sien, bien = sien[0][0], bien[0][0]
      self.assertEqual(len(sien), len(bien))
      self.assertEqual(len(sien), sext.shape[0])
      self.assertEqual(len(sien), sext.shape[1])
      self.assertEqual(len(sien), bext.shape[0])
      self.assertEqual(len(sien), bext.shape[1])
      self.assertEqual(len(sien), (self.degree+1)**self.ndims)
      numpy.testing.assert_array_almost_equal(bext, numpy.eye((self.degree+1)**self.ndims))
      numpy.testing.assert_array_almost_equal(svals[:,sien], bvals[:,bien].dot(sext))

for ndims in range(1, 4):
  for degree in [2] if ndims == 3 else range(1, 4):
    elem_project(ndims=ndims, degree=degree)


class structure2d(TestCase):

  def test_domain(self):
    domain, geom = mesh.rectilinear([[-1,0,1]]*2)
    verify_connectivity(domain.structure, geom)

  def test_boundaries(self):
    domain, geom = mesh.rectilinear([[-1,0,1]]*3)
    for grp in 'left', 'right', 'top', 'bottom', 'front', 'back':
      bnd = domain.boundary[grp]
      # DISABLED: what does this check? -GJ 14/07/28
      #verify_connectivity(bnd.structure, geom)
      xn = bnd.elem_eval(geom.dotnorm(geom), ischeme='gauss1', separate=False)
      numpy.testing.assert_array_less(0, xn, 'inward pointing normals')

  def test_interfaces(self):
    domain, geom = mesh.rectilinear([[-1,0,1]]*3)
    verify_interfaces(domain, geom, periodic=False)


@parametrize
class structured_prop_periodic(TestCase):

  def test(self):
    bnames = 'left', 'top', 'front'
    side = bnames[self.sdim]
    domain, geom = mesh.rectilinear([2]*self.ndim, periodic=self.periodic)
    self.assertEqual(list(domain.boundary[side].periodic), [i if i < self.sdim else i-1 for i in self.periodic if i != self.sdim])

structured_prop_periodic('2d_1_0', ndim=2, periodic=[1], sdim=0)
structured_prop_periodic('2d_0_1', ndim=2, periodic=[0], sdim=1)
structured_prop_periodic('3d_0,2_1', ndim=3, periodic=[0,2], sdim=1)


class picklability(TestCase):

  def assert_pickle_dump_load(self, data):
    script = b'from nutils import *\nimport pickle, base64\npickle.loads(base64.decodebytes(b"""' \
      + base64.encodebytes(pickle.dumps(data)) \
      + b'"""))'
    p = subprocess.Popen([sys.executable], stdin=subprocess.PIPE)
    p.communicate(script)
    self.assertEqual(p.wait(), 0, 'unpickling failed')

  def test_domain(self):
    domain, geom = mesh.rectilinear([[0,1,2]]*2)
    self.assert_pickle_dump_load(domain)

  def test_geom(self):
    domain, geom = mesh.rectilinear([[0,1,2]]*2)
    self.assert_pickle_dump_load(geom)

  def test_basis(self):
    domain, geom = mesh.rectilinear([[0,1,2]]*2)
    basis = domain.basis('spline', degree=2)
    self.assert_pickle_dump_load(basis)


class common_refine(TestCase):

  def test(self):
    dom, geom = mesh.rectilinear([[0,1,2],[0,1,2]])
    doms, funs, vals = {}, {}, {}

    doms['1'] = dom.refined_by(list(dom)[:1])
    funs['1'] = doms['1'].basis('th-std', degree=1)
    vals['1'] = 0.375,0.25,0.375,0.9375,0.5,0.25,0.5,0.25,0.0625,0.125,0.125,0.25

    doms['234'] = dom.refined_by(list(dom)[1:])
    funs['234'] = doms['234'].basis('th-std', degree=1)
    vals['234'] = 0.25,0.375,0.375,0.5625,0.125,0.0625,0.25,0.125,0.25,0.125,0.125,0.25,0.25,0.25,0.125,0.0625,0.125,0.125,0.125,0.0625

    doms['123'] = dom.refined_by(list(dom)[:-1])
    funs['123'] = doms['123'].basis('th-std', degree=1)
    vals['123'] = 0.5625,0.375,0.375,0.25,0.0625,0.125,0.125,0.125,0.0625,0.125,0.25,0.25,0.25,0.125,0.125,0.25,0.125,0.25,0.0625,0.125

    doms['4'] = dom.refined_by(list(dom)[-1:])
    funs['4'] = doms['4'].basis('th-std', degree=1)
    vals['4'] = 0.25,0.5,0.25,0.5,0.9375,0.375,0.25,0.375,0.25,0.125,0.125,0.0625

    for a, b, n in ('1', '234', 16), ('1', '4', 10), ('123', '234', 16):
      with self.subTest('ref{}vs{}'.format(a, b)):
        common = doms[a] & doms[b]
        self.assertEqual(len(common), n)
        for c in a, b:
          testvals = common.integrate(funs[c]*function.J(geom), ischeme='gauss1')
          numpy.testing.assert_array_almost_equal(testvals, vals[c])

@parametrize
class revolved(TestCase):

  def setUp(self):
    super().setUp()
    if self.domtype == 'circle':
      self.domain0, self.geom0 = mesh.rectilinear([2])
      self.exact_volume = 4 * numpy.pi
      self.exact_surface = 4 * numpy.pi
      self.exact_groups = {}
    elif self.domtype == 'cylinder':
      self.domain0, self.geom0 = mesh.rectilinear([1,2])
      self.exact_volume = 2 * numpy.pi
      self.exact_surface = 6 * numpy.pi
      self.exact_groups = dict(right=4*numpy.pi, left=0)
    elif self.domtype == 'hollowcylinder':
      self.domain0, self.geom0 = mesh.rectilinear([[.5,1],2])
      self.exact_volume = 1.5 * numpy.pi
      self.exact_surface = 7.5 * numpy.pi
      self.exact_groups = dict(right=4*numpy.pi, left=2*numpy.pi)
    else:
      raise Exception('unknown domain type {!r}'.format(self.domtype))
    self.domain, self.geom, self.simplify = self.domain0.revolved(self.geom0)
    if self.refined:
      self.domain = self.domain.refined
      self.domain0 = self.domain0.refined

  def test_revolved(self):
    self.assertEqual(len(self.domain), len(self.domain0))

  def test_volume(self):
    vol = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_array_almost_equal(vol, self.exact_volume)

  def test_volume_bydiv(self):
    boundary = self.domain.boundary
    if self.domtype != 'hollowcylinder':
      boundary = boundary['bottom,right,top']
    v = boundary.integrate(self.geom.dotnorm(self.geom)*function.J(self.geom), ischeme='gauss1') / self.domain.ndims
    numpy.testing.assert_array_almost_equal(v, self.exact_volume)

  def test_surface(self):
    surf = self.domain.boundary.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_array_almost_equal(surf, self.exact_surface)

  def test_surface_groups(self):
    for name, exact_surface in self.exact_groups.items():
      surf = self.domain.boundary[name].integrate(function.J(self.geom), ischeme='gauss1')
      numpy.testing.assert_array_almost_equal(surf, exact_surface)

  def test_basis(self):
    basis = self.domain.basis('std', degree=1)
    values = self.domain.sample('uniform', 2).eval(basis).sum(1)
    numpy.testing.assert_array_almost_equal(values, 1)

  def test_trim(self):
    r = function.norm2(self.geom[:2])
    trimmed = self.domain.trim(r - .75, maxrefine=1)
    volume = trimmed.integrate(function.J(self.geom), degree=1)
    self.assertGreater(volume, 0)
    self.assertLess(volume, self.exact_volume)

for domtype in 'circle', 'cylinder', 'hollowcylinder':
  for refined in False, True:
    revolved(domtype=domtype, refined=refined)


_refined_refs = dict(
  line=element.LineReference(),
  quadrilateral=element.LineReference()**2,
  hexahedron=element.LineReference()**3,
  triangle=element.TriangleReference(),
  tetrahedron=element.TetrahedronReference())

@parametrize
class refined(TestCase):

  def test_boundary_gradient(self):
    ref = _refined_refs[self.etype]
    elem = element.Element(ref, (transform.Identifier(ref.ndims),))
    domain = topology.ConnectedTopology(ref.ndims, (elem,), ((-1,)*ref.nedges,)).refine(self.ref0)
    geom = function.rootcoords(ref.ndims)
    basis = domain.basis('std', degree=1)
    u = domain.projection(geom.sum(), onto=basis, geometry=geom, degree=2)
    bpoints = domain.refine(self.ref1).boundary.refine(self.ref2).sample('uniform', 1)
    g = bpoints.eval(u.grad(geom))
    numpy.testing.assert_allclose(g, 1)

for etype in _refined_refs:
  for ref0 in 0, 1:
    for ref1 in 0, 1:
      for ref2 in 0, 1:
        refined(etype=etype, ref0=ref0, ref1=ref1, ref2=ref2)


@parametrize
class general(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([3,4,5], periodic=[] if self.periodic is False else [self.periodic])
    if not self.isstructured:
      self.domain = topology.ConnectedTopology(self.domain.ndims, self.domain.elements, self.domain.connectivity)

  def test_connectivity(self):
    nboundaries = 0
    ninterfaces = 0
    for ielem, ioppelems in enumerate(self.domain.connectivity):
      for iedge, ioppelem in enumerate(ioppelems):
        if ioppelem == -1:
          nboundaries += 1
        else:
          ioppedge = tuple(self.domain.connectivity[ioppelem]).index(ielem)
          edge = self.domain.elements[ielem].edge(iedge)
          oppedge = self.domain.elements[ioppelem].edge(ioppedge)
          self.assertEqual(edge.reference, oppedge.reference)
          ninterfaces += .5
    self.assertEqual(nboundaries, len(self.domain.boundary), 'incompatible number of boundaries')
    self.assertEqual(ninterfaces, len(self.domain.interfaces), 'incompatible number of interfaces')

  def test_boundary(self):
    for elem in self.domain.boundary:
      ielem, tail = transform.lookup_item(elem.transform, self.domain.edict)
      etrans, = tail
      iedge = self.domain.elements[ielem].reference.edge_transforms.index(etrans)
      self.assertEqual(self.domain.connectivity[ielem][iedge], -1)

  def test_interfaces(self):
    for elem in self.domain.interfaces:
      ielem, tail = transform.lookup_item(elem.transform, self.domain.edict)
      etrans, = tail
      iedge = self.domain.elements[ielem].reference.edge_transforms.index(etrans)
      ioppelem, opptail = transform.lookup_item(elem.opposite, self.domain.edict)
      eopptrans, = opptail
      ioppedge = self.domain.elements[ioppelem].reference.edge_transforms.index(eopptrans)
      self.assertEqual(self.domain.connectivity[ielem][iedge], ioppelem)
      self.assertEqual(self.domain.connectivity[ioppelem][ioppedge], ielem)

for isstructured in True, False:
  for periodic in False, 0, 1, 2:
    general(isstructured=isstructured, periodic=periodic)


@parametrize
class locate(TestCase):

  @parametrize.skip_if(lambda nprocs, **kwargs: nprocs > 1 and not hasattr(os, 'fork'), 'nprocs > 1 not supported on this platform')
  def test(self):
    with config(nprocs=self.nprocs):
      domain, geom = mesh.unitsquare(4, etype=self.etype)
      geom += .1 * function.sin(geom * numpy.pi) # non-polynomial geometry
      target = numpy.array([(.2,.3), (.1,.9), (0,1)])
      sample = domain.locate(geom, target, eps=1e-15)
      located = sample.eval(geom)
      numpy.testing.assert_array_almost_equal(located, target)

for etype in 'square', 'triangle', 'mixed':
  for nprocs in 1, 2:
    locate(etype=etype, nprocs=nprocs)


@parametrize
class hierarchical(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([numpy.linspace(0, 1, 7)]*self.ndims, periodic=self.periodic)
    # Refine `self.domain` near `self.pos`.
    distance = ((self.geom-self.pos)**2).sum(0)**0.5
    for threshold in 0.3, 0.15:
      self.domain = self.domain.refined_by(elem for elem, value in zip(self.domain, self.domain.elem_mean([distance], ischeme='gauss1', geometry=self.geom)[0]) if value <= threshold)

  @parametrize.enable_if(lambda periodic, **params: not periodic)
  def test_boundaries(self):
    verify_boundaries(self.domain, self.geom)

  def test_interfaces(self):
    verify_interfaces(self.domain, self.geom, self.periodic)

hierarchical('3d_l_rrr', pos=0, ndims=3, periodic=[])
hierarchical('3d_l_rpr', pos=0, ndims=3, periodic=[1])
hierarchical('2d_l_pp', pos=0, ndims=2, periodic=[0,1])
hierarchical('2d_l_pr', pos=0, ndims=2, periodic=[0])
hierarchical('2d_c_pr', pos=0.5, ndims=2, periodic=[0])
hierarchical('2d_r_pr', pos=1, ndims=2, periodic=[0])
hierarchical('2d_l_rr', pos=0, ndims=2, periodic=[])
hierarchical('1d_l_p', pos=0, ndims=1, periodic=[0])
hierarchical('1d_c_p', pos=0.5, ndims=1, periodic=[0])
#hierarchical('1d_l_r', pos=0, ndims=1, periodic=[]) # disabled, see issue #193


@parametrize
class multipatch_hyperrect(TestCase):

  def setUp(self):
    super().setUp()
    npatches = numpy.array(self.npatches)
    indices = numpy.arange((npatches+1).prod()).reshape(npatches+1)

    self.domain, self.geom = mesh.multipatch(
      patches=[indices[tuple(map(slice, i, numpy.array(i)+2))].ravel().tolist() for i in itertools.product(*map(range, npatches))],
      patchverts=tuple(itertools.product(*map(range, npatches+1))),
      nelems=4,
    )

  def test_spline_basis(self):
    basis = self.domain.basis('spline', degree=2)
    coeffs = self.domain.elem_eval(basis.sum(0), ischeme='gauss4', separate=False)
    numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

  def test_discont_basis(self):
    basis = self.domain.basis('discont', degree=2)
    coeffs = self.domain.elem_eval(basis.sum(0), ischeme='gauss4', separate=False)
    numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

  def test_boundaries(self):
    verify_boundaries(self.domain, self.geom)

  def test_interfaces(self):
    verify_interfaces(self.domain, self.geom, periodic=False)

  def test_interpatch_interfaces(self):
    verify_interfaces(self.domain, self.geom, periodic=False, interfaces=self.domain.interfaces['interpatch'], elemindicator=self.domain.basis('patch'))

multipatch_hyperrect('3', npatches=(3,))
multipatch_hyperrect('2x2', npatches=(2,2))
multipatch_hyperrect('3x3', npatches=(3,3))
multipatch_hyperrect('2x2x3', npatches=(2,2,3))


class multipatch_L(TestCase):

  def setUp(self):
    # 2---5
    # |   |
    # 1---4------7
    # |   |      |
    # 0---3------6

    super().setUp()
    self.domain, self.geom = mesh.multipatch(
      patches=[[0,1,3,4], [1,2,4,5], [3,4,6,7]],
      patchverts=[[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [3,0], [3,1]],
      nelems={None: 4, (3,6): 8, (4,7): 8})

  def test_spline_basis(self):
    for continuity in (-1, 0):
      basis = self.domain.basis('spline', degree=2, continuity=continuity)
      with self.subTest('partition of unity', continuity=continuity):
        sample = self.domain.sample('bezier', 5)
        coeffs = sample.eval(basis.sum(0))
        numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))
      with self.subTest('interpatch continuity', continuity=continuity):
        sample = self.domain.interfaces['interpatch'].sample('bezier', 5)
        jump = sample.eval(function.jump(basis))
        numpy.testing.assert_array_almost_equal(jump, numpy.zeros_like(jump))

  def test_nonuniform_spline_basis(self):
    knots_01 = 0, 0.25, 0.5, 0.75, 1
    mults_01 = 3, 1, 2, 1, 3
    knots_12 = 0, 0.2, 0.4, 0.6, 1
    knotvalues = {None: None, (1,2): knots_12, (5,4): knots_12[::-1], (0,1): knots_01, (3,4): knots_01, (6,7): knots_01}
    knotmultiplicities = {None: None, (0,1): mults_01, (4,3): mults_01[::-1], (6,7): mults_01}
    basis = self.domain.basis('spline', degree=2, knotvalues=knotvalues, knotmultiplicities=knotmultiplicities)
    coeffs = self.domain.project(1, onto=basis, geometry=self.geom, ischeme='gauss4')
    numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

  def test_discont_basis(self):
    basis = self.domain.basis('discont', degree=2)
    with self.subTest('partition of unity'):
      sample = self.domain.sample('bezier', 5)
      coeffs = sample.eval(basis.sum(0))
      numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

  def test_patch_basis(self):
    patch_index = self.domain.basis('patch').dot([0, 1, 2])
    for ipatch in range(3):
      with self.subTest(ipatch=ipatch):
        sample = self.domain['patch{}'.format(ipatch)].sample('gauss', 1)
        numpy.testing.assert_array_almost_equal(sample.eval(patch_index), ipatch)


class elem_eval(TestCase):

  def setUp(self):
    self.domain, self.geom = mesh.rectilinear([[0,1,2],[0,1,2]])

  def test_separate(self):
    retvals = self.domain.elem_eval([self.geom[0], self.geom[1]], ischeme='bezier3', separate=True)
    self.assertEqual(len(retvals), 2)
    for arrays in retvals:
      self.assertIsInstance(arrays, list)
      self.assertEqual(len(arrays), 4)
      for array in arrays:
        self.assertTrue(numeric.isarray(array))
        self.assertEqual(array.shape, (9,))

  def test_noseparate(self):
    retvals = self.domain.elem_eval([self.geom[0], self.geom[1]], ischeme='bezier3', separate=False)
    self.assertEqual(len(retvals), 2)
    for array in retvals:
      self.assertTrue(numeric.isarray(array))
      self.assertEqual(array.shape, (36,))

  def test_asfunction(self):
    x, y = self.domain.elem_eval([self.geom[0], self.geom[1]], ischeme='gauss3', asfunction=True)
    err = self.domain.integrate((x-self.geom[0])**2+(y-self.geom[1])**2, ischeme='gauss3')
    self.assertEqual(err, 0)
    with self.assertRaises(function.EvaluationError):
      self.domain.integrate(x, ischeme='gauss4')

  def test_failures(self):
    f = function.inverse(function.Guard(function.diagonalize(self.geom))) # fails at x=0 and y=0
    arrays = self.domain.elem_eval(f, ischeme='bezier3', separate=True)
    self.assertIsInstance(arrays, list)
    self.assertEqual(len(arrays), 4)
    isnan = numpy.zeros((4, 4), dtype=bool) # boolean matrix marking all failing bezier points
    isnan[0,:] = True
    isnan[:,0] = True
    for i, array in enumerate(arrays):
      self.assertTrue(numpy.all(numpy.isnan(array).any(axis=(1,2)).reshape(3,3) == isnan[i//2:,i%2:][:3,:3]))
