from nutils import *
import random, itertools, functools
from nutils.testing import *

@parametrize
class basis(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([[0,1,2]]*self.ndims)
    for iref in range(self.nrefine):
      self.domain = self.domain.refined_by([0])
    self.basis = self.domain.basis(self.btype, degree=self.degree)
    self.gauss = 'gauss{}'.format(2*self.degree)

  @parametrize.enable_if(lambda btype, **params: btype != 'discont')
  def test_continuity(self):
    funcsp = self.basis
    for regularity in (range(self.degree) if self.btype=='spline' else [0]):
      elem_jumps = self.domain.interfaces.sample('gauss', 2).eval(function.jump(funcsp))
      numpy.testing.assert_almost_equal(elem_jumps,0,decimal=10)
      funcsp = function.grad(funcsp, self.geom)

  @parametrize.enable_if(lambda btype, **params: not btype.startswith('h-'))
  def test_pum(self):
    error = numpy.sqrt(self.domain.integrate((1-self.basis.sum(0))**2*function.J(self.geom), ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

  def test_poly(self):
    target = (self.geom**self.degree).sum(-1)
    projection = self.domain.projection(target, onto=self.basis, geometry=self.geom, ischeme=self.gauss, droptol=0)
    error = numpy.sqrt(self.domain.integrate((target-projection)**2*function.J(self.geom), ischeme=self.gauss))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

for ndims in range(1, 4):
  for btype in 'discont', 'h-std', 'th-std', 'h-spline', 'th-spline':
    for degree in range(0 if btype == 'discont' else 1, 4):
      for nrefine in 0, 2:
        basis(btype=btype, degree=degree, ndims=ndims, nrefine=nrefine)

class NNZ(matrix.Backend):
  def assemble(self, data, index, shape):
    return type('nnzmatrix', (), dict(nnz=len(numpy.unique(numpy.ravel_multi_index(index, shape))), shape=shape))()

@parametrize
class sparsity(TestCase):

  vals = {
    1: [60,66,70],
    2: [3012,3216,3424],
  }

  def test_sparsity(self):
    topo, geom = mesh.rectilinear([6]*self.ndim)
    topo = topo.refined_by(set(map(topo.transforms.index, itertools.chain(topo[1:3].transforms, topo[-2:].transforms))))

    ns = function.Namespace()
    ns.x = geom
    ns.tbasis = topo.basis('th-spline', degree=2, truncation_tolerance=1e-14)
    ns.tnotol = topo.basis('th-spline', degree=2, truncation_tolerance=0)
    ns.hbasis = topo.basis('h-spline', degree=2)

    with matrix.backend('nnz'):
      tA, tA_tol, hA = topo.integrate([ns.eval_ij('tbasis_i,k tbasis_j,k'), ns.eval_ij('tnotol_i,k tnotol_j,k'), ns.eval_ij('hbasis_i,k hbasis_j,k')], degree=5)

    tA_nnz, tA_tol_nnz, hA_nnz = self.vals[self.ndim]
    self.assertEqual(tA.nnz, tA_nnz)
    self.assertEqual(tA_tol.nnz, tA_tol_nnz)
    self.assertEqual(hA.nnz, hA_nnz)

for ndim in 1, 2:
  sparsity(ndim=ndim)

@parametrize
class structured(TestCase):

  def setUp(self):
    if self.product:
      self.domain, geom = mesh.rectilinear([2,3])
    else:
      domain1, geom1 = mesh.rectilinear([2])
      domain2, geom2 = mesh.rectilinear([3])
      self.domain = domain1 * domain2

  def test_std_equalorder(self):
    for p in range(1, 3):
      basis = self.domain.basis('std', degree=p)
      self.assertEqual(len(basis), (3+2*(p-1))*(4+3*(p-1)))

  def test_spline_equalorder(self):
    for p in range(1, 3):
      basis = self.domain.basis('spline', degree=p)
      self.assertEqual(len(basis), (2+p)*(3+p))

  def test_std_mixedorder(self):
    basis = self.domain.basis('std', degree=(1,2))
    self.assertEqual(len(basis), 3*7)
    basis = self.domain.basis('std', degree=(2,1))
    self.assertEqual(len(basis), 5*4)

  def test_spline_mixedorder(self):
    basis = self.domain.basis('spline', degree=(1,2))
    self.assertEqual(len(basis), 3*5)
    basis = self.domain.basis('spline', degree=(2,1))
    self.assertEqual(len(basis), 4*4)

  def test_knotvalues(self):
    # test refinement of knotvalues[0] -> [0,1/2,1]
    basis = self.domain.basis('spline', degree=2, knotvalues=[[0,1],[0,1/3,2/3,1]])

  def test_knotmultiplicities(self):
    # test refinement of knotmultiplicities[0] -> [3,1,3]
    basis = self.domain.basis('spline', degree=2, knotmultiplicities=[[3,3],[3,1,1,3]])
    self.assertEqual(len(basis), 4*5)
    basis = self.domain.basis('spline', degree=2, knotmultiplicities=[[3,3],[3,2,1,3]])
    self.assertEqual(len(basis), 4*6)

  def test_continuity(self):
    # test refinement of knotmultiplicities[0] -> [3,1,3]
    basis = self.domain.basis('spline', degree=2, continuity=0)
    self.assertEqual(len(basis), 5*7)
    basis = self.domain.basis('spline', degree=2, continuity=0, knotmultiplicities=[[3,3],None])
    self.assertEqual(len(basis), 5*7)

structured(product=False)
structured(product=True)


@parametrize
class structured_line(TestCase):

  def setUp(self):
    self.periodic = {'normal': False, 'periodic': True}[self.variant]
    verts = numpy.linspace(0, 1, self.nelems+1)
    self.domain, self.x = mesh.line(verts, periodic=self.periodic)

    vl = function.elemwise(self.domain.transforms, verts[:-1])
    vr = function.elemwise(self.domain.transforms, verts[1:])
    j = numpy.arange(self.degree+1)
    self.Bbernstein = numpy.vectorize(numeric.binom)(self.degree,j)*(self.x[0]-vl)**j*(vr-self.x[0])**(self.degree-j)/(vr-vl)**self.degree

    if self.periodic:
      t = numpy.linspace(-self.degree/self.nelems, 1+self.degree/self.nelems, self.nelems+1+2*self.degree)
    else:
      t = numpy.concatenate([[0]*self.degree, verts, [1]*self.degree], axis=0)
    self.Bspline = function.heaviside(self.x[0]-t[:-1])*function.heaviside(t[1:]-self.x[0])
    for p in range(1, self.degree+1):
      dt = numpy.array([t[i+p]-t[i] if t[i+p] != t[i] else 1 for i in range(len(self.Bspline))])
      self.Bspline = (self.x[0]-t[:-p-1])/dt[:-1]*self.Bspline[:-1] + (t[p+1:]-self.x[0])/dt[1:]*self.Bspline[1:]

  def test_coeffs(self):
    numpy.random.seed(0)
    if self.btype == 'spline':
      c = numpy.random.random(self.nelems if self.periodic else len(self.Bspline))
      f = self.Bspline.dot(numpy.array([c[i%self.nelems] for i in range(len(self.Bspline))]) if self.periodic else c)
    elif self.btype == 'std':
      ndofs = self.nelems*self.degree+(1 if not self.periodic else 0) if self.degree else self.nelems
      c = numpy.random.random((ndofs,))
      f = self.Bbernstein.dot(function.elemwise(self.domain.transforms, types.frozenarray([[c[(i*self.degree+j)%ndofs if self.degree else i] for j in range(self.degree+1)] for i in range(self.nelems)])))
    elif self.btype == 'discont':
      ndofs = self.nelems*(self.degree+1)
      c = types.frozenarray(numpy.random.random((ndofs,)))
      f = self.Bbernstein.dot(function.elemwise(self.domain.transforms, c.reshape(self.nelems, self.degree+1)))
    basis = self.domain.basis(self.btype, degree=self.degree)
    pc = self.domain.project(f, onto=basis, geometry=self.x, ischeme='gauss', degree=2*self.degree)
    numpy.testing.assert_array_almost_equal(c, pc)

for btype in ['discont', 'spline', 'std']:
  for variant in ['normal', 'periodic']:
    for nelems in range(1, 4):
      for degree in range(0 if btype == 'discont' else 1, 4):
        if btype == 'spline' and variant == 'periodic' and nelems < 2*degree:
          continue
        structured_line(variant=variant, btype=btype, degree=degree, nelems=nelems)
structured_line(variant='periodic', btype='spline', degree=0, nelems=1)


@parametrize
class unstructured_topology(TestCase):

  def setUp(self):
    if self.ndims == 2:
      domain, geom = mesh.unitsquare(4, self.variant)
      nverts = 25
    elif self.variant == 'tensor':
      structured, geom = mesh.rectilinear([numpy.linspace(0, 1, 5-i) for i in range(self.ndims)])
      domain = topology.ConnectedTopology(structured.references, structured.transforms, structured.opposites, structured.connectivity)
      nverts = numpy.product([5-i for i in range(self.ndims)])
    elif self.variant == 'simplex':
      numpy.random.seed(0)
      nverts = 20
      simplices = numeric.overlapping(numpy.arange(nverts), n=self.ndims+1)
      coords = numpy.random.normal(size=(nverts, self.ndims))
      root = transform.Identifier(self.ndims, 'test')
      domain = topology.SimplexTopology(simplices, [(root, transform.Simplex(c)) for c in coords[simplices]])
      geom = function.rootcoords(self.ndims)
    else:
      raise NotImplementedError
    self.domain = domain
    self.basis = domain.basis(self.btype) if self.btype == 'bubble' else domain.basis(self.btype, degree=self.degree)
    self.geom = geom
    self.nverts = nverts

  @parametrize.enable_if(lambda btype, **params: btype in {'bubble', 'discont'} or degree == 1)
  def test_ndofs(self):
    if self.btype == 'bubble':
      ndofs = self.nverts + len(self.domain)
    elif self.btype == 'discont':
      ndofs_by_ref = {
        element.getsimplex(1)**self.ndims: (self.degree+1)**self.ndims,
        element.getsimplex(self.ndims): numpy.product(self.degree+numpy.arange(self.ndims)+1) // numpy.product(numpy.arange(self.ndims)+1)}
      ndofs = sum(ndofs_by_ref[reference] for reference in self.domain.references)
    elif self.degree == 1:
      ndofs = self.nverts
    else:
      raise NotImplementedError
    self.assertEqual(len(self.basis), ndofs)

  def test_pum_sum(self):
    # Note that this test holds for btype 'lagrange' as well, although the
    # basis functions are not confined to range [0,1].
    error = numpy.sqrt(self.domain.integrate((1-self.basis.sum(0))**2*function.J(self.geom), ischeme='gauss', degree=2*self.degree))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

  @parametrize.enable_if(lambda btype, **params: btype != 'lagrange')
  def test_pum_range(self):
    values = self.domain.sample('gauss', 2*self.degree).eval(self.basis)
    self.assertTrue((values > 0-1e-10).all())
    self.assertTrue((values < 1+1e-10).all())

  def test_poly(self):
    target = (self.geom**self.degree).sum(-1)
    if self.btype == 'discont':
      target += function.TransformsIndexWithTail(self.domain.transforms, function.TRANS).index
    projection = self.domain.projection(target, onto=self.basis, geometry=self.geom, ischeme='gauss', degree=2*self.degree, droptol=0)
    error = numpy.sqrt(self.domain.integrate((target-projection)**2*function.J(self.geom), ischeme='gauss', degree=2*self.degree))
    numpy.testing.assert_almost_equal(error, 0, decimal=12)

for ndims in 1, 2, 3:
  for variant in ['simplex', 'tensor'] if ndims != 2 else ['triangle', 'square', 'mixed']:
    for btype in ['discont', 'bernstein', 'lagrange', 'std', 'bubble'][:5 if variant in ('simplex', 'triangle') else 4]:
      for degree in [0,1,2,3] if btype == 'discont' else [1] if btype == 'bubble' else [1,2,3]:
        unstructured_topology(ndims=ndims, btype=btype, degree=degree, variant=variant)
