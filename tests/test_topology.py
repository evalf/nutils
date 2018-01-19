from nutils import *
from . import *
import numpy, copy, sys, pickle, subprocess, base64, itertools, os

grid = numpy.linspace(0., 1., 4)

def neighbor(elem1, elem2):
  elem1_vertices = set(elem1.vertices)
  ncommon = sum(v in elem1_vertices for v in elem2.vertices)
  if not ncommon:
    return -1
  if elem1.ndims == elem2.ndims == 1:
    return {2:0,1:1}[ncommon]
  if elem1.ndims == elem2.ndims == 2:
    return {4:0,2:1,1:2}[ncommon]
  if elem1.ndims == elem2.ndims == 3:
    return {8:0,4:1,2:2,1:3}[ncommon]
  raise NotImplementedError('%s, %s' % (elem1.reference, elem2.reference))

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
  lhs = domain.integrate(f.grad(geom), ischeme='gauss2', geometry=geom)
  rhs = domain.boundary.integrate(f*function.normal(geom), ischeme='gauss2', geometry=geom)
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
  lhs = domain.integrate((elemindicator*f.grad(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom)
  rhs = interfaces.integrate((-function.jump(elemindicator)*f*function.normal(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom)
  if len(domain.boundary):
    rhs += domain.boundary.integrate((elemindicator*f*function.normal(geom)[None]).sum(axis=1), ischeme='gauss2', geometry=geom)
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


@parametrize
class connectivity(TestCase):

  def test_check1d(self):
    domain = mesh.rectilinear(1*(grid,), periodic=[0] if self.periodic else [])[0]
    elem = domain.structure
    self.assertEqual(neighbor(elem[0], elem[0]), 0, 'Failed to identify codim 0 neighbors')
    self.assertEqual(neighbor(elem[1], elem[2]), 1, 'Failed to identify codim 1 neighbors')
    if self.periodic:
      self.assertEqual(neighbor(elem[0], elem[2]), 1, 'Failed to identify periodicity neighbors')
    else:
      self.assertEqual(neighbor(elem[0], elem[2]), -1, 'Failed to identify non-neighbors')

  def test_check2d(self):
    domain = mesh.rectilinear(2*(grid,), periodic=[0] if self.periodic else [])[0]
    elem = domain.structure
    self.assertEqual(neighbor(elem[0,0], elem[0,0]), 0, 'Failed to identify codim 0 neighbors')
    self.assertEqual(neighbor(elem[1,1], elem[1,2]), 1, 'Failed to identify codim 1 neighbors')
    self.assertEqual(neighbor(elem[0,0], elem[1,1]), 2, 'Failed to identify codim 2 neighbors')
    self.assertEqual(neighbor(elem[1,1], elem[0,0]), 2, 'Failed to identify codim 2 neighbors')
    if self.periodic:
      self.assertEqual(neighbor(elem[2,1], elem[0,1]), 1, 'Failed to identify periodicity neighbors')
      self.assertEqual(neighbor(elem[2,1], elem[0,0]), 2, 'Failed to identify periodicity neighbors')
    else:
      self.assertEqual(neighbor(elem[2,1], elem[0,1]), -1, 'Failed to identify non-neighbors')

  def test_check3d(self):
    domain = mesh.rectilinear(3*(grid,), periodic=[0] if self.periodic else [])[0]
    elem = domain.structure
    self.assertEqual(neighbor(elem[1,1,1], elem[1,1,1]), 0, 'Failed to identify codim 0 neighbors')
    self.assertEqual(neighbor(elem[1,1,1], elem[1,1,2]), 1, 'Failed to identify codim 1 neighbors')
    self.assertEqual(neighbor(elem[1,1,1], elem[1,2,2]), 2, 'Failed to identify codim 2 neighbors')
    self.assertEqual(neighbor(elem[1,1,1], elem[2,2,2]), 3, 'Failed to identify codim 3 neighbors')
    if self.periodic:
      self.assertEqual(neighbor(elem[0,2,2], elem[2,2,2]), 1, 'Failed to identify periodicity neighbors')
      self.assertEqual(neighbor(elem[0,2,2], elem[2,1,2]), 2, 'Failed to identify periodicity neighbors')
    else:
      self.assertEqual(neighbor(elem[0,2,2], elem[2,2,2]), -1, 'Failed to identify non-neighbors')

connectivity(periodic=True)
connectivity(periodic=False)


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
    funs['1'] = doms['1'].basis('std', degree=1)
    vals['1'] = .5,.25,.5,1,.5,.25,.5,.25,.0625,.125,.125,.25

    doms['234'] = dom.refined_by(list(dom)[1:])
    funs['234'] = doms['234'].basis('std', degree=1)
    vals['234'] = .25,.5,.5,1,.125,.0625,.25,.125,.25,.125,.125,.25,.25,.25,.125,.0625,.125,.125,.125,.0625

    doms['123'] = dom.refined_by(list(dom)[:-1])
    funs['123'] = doms['123'].basis('std', degree=1)
    vals['123'] = 1,.5,.5,.25,.0625,.125,.125,.125,.0625,.125,.25,.25,.25,.125,.125,.25,.125,.25,.0625,.125

    doms['4'] = dom.refined_by(list(dom)[-1:])
    funs['4'] = doms['4'].basis('std', degree=1)
    vals['4'] = .25,.5,.25,.5,1,.5,.25,.5,.25,.125,.125,.0625

    for a, b, n in ('1', '234', 16), ('1', '4', 10), ('123', '234', 16):
      with self.subTest('ref{}vs{}'.format(a, b)):
        common = doms[a] & doms[b]
        self.assertEqual(len(common), n)
        for c in a, b:
          testvals = common.integrate(funs[c], geometry=geom, ischeme='gauss1')
          numpy.testing.assert_array_almost_equal(testvals, vals[c])


class revolved_circle(TestCase):

  def setUp(self):
    super().setUp()
    rdomain, rgeom = mesh.rectilinear([2])
    self.domain, self.geom, self.simplify = rdomain.revolved(rgeom)

  def test_simplified(self):
    integrand = function.norm2(self.geom) * function.jacobian(self.geom, ndims=1)
    self.assertNotEqual(integrand, self.simplify(integrand))
    vals1, vals2 = self.domain.elem_eval([ integrand, self.simplify(integrand)], ischeme='uniform2')
    numpy.testing.assert_array_almost_equal(vals1, vals2)

  def test_circle_area(self):
    vol = self.domain.integrate(1, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
    numpy.testing.assert_array_almost_equal(vol, 4)

  def test_circle_circumference(self):
    surf = self.domain.boundary.integrate(1, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
    numpy.testing.assert_array_almost_equal(surf, 4)


class revolved_cylinder(TestCase):

  def setUp(self):
    super().setUp()
    rzdomain, rzgeom = mesh.rectilinear([1,2])
    self.domain, self.geom, self.simplify = rzdomain.revolved(rzgeom)

  def test_volume(self):
    vol = self.domain.integrate(1, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
    numpy.testing.assert_array_almost_equal(vol, 2)

  def test_surface(self):
    for name, exact in ('full',6), ('right',4), ('left',0):
      with self.subTest(name):
        surf = self.domain.boundary[name if name != 'full' else ()].integrate(1, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
        numpy.testing.assert_array_almost_equal(surf, exact)


class revolved_hollowcylinder(TestCase):

  def setUp(self):
    super().setUp()
    rzdomain, rzgeom = mesh.rectilinear([[.5,1],2])
    self.domain, self.geom, self.simplify = rzdomain.revolved(rzgeom)
    self.basis = self.domain.basis('std', degree=2)

  def test_volume(self):
    v = self.domain.integrate(1, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
    numpy.testing.assert_array_almost_equal(v, 1.5)

  def test_volume2(self):
    v = self.domain.boundary.integrate(self.geom.dotnorm(self.geom)/3, geometry=self.geom, ischeme='gauss1', edit=self.simplify) / numpy.pi
    numpy.testing.assert_array_almost_equal(v, 1.5)

  def test_surface(self):
    for name, exact in ('full',7.5), ('right',4), ('left',2):
      with self.subTest(name):
        surf = self.domain.boundary[name if name != 'full' else ()].integrate(1, geometry=self.geom, ischeme='gauss9', edit=self.simplify) / numpy.pi
        numpy.testing.assert_array_almost_equal(surf, exact)

  def test_basistype(self):
    self.assertIsInstance(self.basis.simplified, function.Inflate)

  def test_dofcount(self):
    self.assertEqual(len(self.basis), 3*5)


@parametrize
class general(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([3,4,5], periodic=[] if self.periodic is False else [self.periodic])
    if not self.isstructured:
      self.domain = topology.UnstructuredTopology(self.domain.ndims, tuple(self.domain))

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
          self.assertCountEqual(edge.vertices, oppedge.vertices, 'edges do not match: {}, {}'.format(edge, oppedge))
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
      domain, geom = mesh.rectilinear([numpy.linspace(0,1,3)]*2) if self.structured else mesh.demo()
      geom += .1 * function.sin(geom * numpy.pi) # non-polynomial geometry
      target = numpy.array([(.2,.3), (.1,.9), (0,1)])
      ltopo = domain.locate(geom, target, eps=1e-15)
      located = ltopo.elem_eval(geom, ischeme='gauss1')
      numpy.testing.assert_array_almost_equal(located, target)

for nprocs in 1, 2:
  locate(structured=True, nprocs=nprocs)
  locate(structured=False, nprocs=nprocs)


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
    basis = self.domain.basis('spline', degree=2)
    coeffs = self.domain.elem_eval(basis.sum(0), ischeme='gauss4', separate=False)
    numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

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
    coeffs = self.domain.elem_eval(basis.sum(0), ischeme='gauss4', separate=False)
    numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

  def test_patch_basis(self):
    patch_index = self.domain.basis('patch').dot([0, 1, 2])
    for ipatch in range(3):
      vals = self.domain['patch{}'.format(ipatch)].elem_eval(patch_index, ischeme='gauss1')
      numpy.testing.assert_array_almost_equal(vals, ipatch)


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
    self.assertTrue(len(arrays), 4)
    for i, array in enumerate(arrays):
      if i < 3:
        self.assertTrue(numpy.isnan(array).all())
      else:
        self.assertFalse(numpy.isnan(array).any())
