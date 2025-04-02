from nutils import function, mesh, sparse, transformseq, topology, element, numeric
from nutils.testing import TestCase, parametrize
import random
import itertools
import numpy


class basisTest(TestCase):

    def assertContinuous(self, topo, geom, basis, continuity):
        for regularity in range(continuity+1):
            smpl = topo.sample('gauss', 2)
            elem_jumps = smpl.eval(function.jump(basis))
            self.assertAllAlmostEqual(elem_jumps, numpy.zeros((smpl.npoints, *basis.shape)), places=10)
            basis = function.grad(basis, geom)[..., 0]

    def assertPartitionOfUnity(self, topo, basis):
        smpl = topo.sample('uniform', 2)
        sumbasis = smpl.eval(basis.sum(0))
        self.assertAllAlmostEqual(sumbasis, numpy.ones((smpl.npoints,)), places=10)

    def assertPolynomial(self, topo, geom, basis, degree):
        target = (geom**degree).sum(-1)
        matrix, rhs, target2 = topo.integrate([basis[:, numpy.newaxis] * basis, basis * target, target**2], degree=degree*2, legacy=False)
        lhs = numpy.linalg.solve(matrix, rhs)
        error = target2 - rhs @ lhs
        self.assertAlmostEqual(error, 0, places=10)


@parametrize
class basis(basisTest):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.rectilinear([max(1, self.nelems-n) for n in range(self.ndims)], periodic=[0] if self.periodic else [])
        if self.refine == 'uniform':
            self.domain = self.domain.refined
        elif self.refine == 'hierarchical':
            for i in range(2):
                self.domain = self.domain.refined_by([len(self.domain)-1])
        else:
            assert not self.refine
        if self.boundary:
            self.domain = self.domain.boundary[self.boundary]
        self.basis = self.domain.basis(self.btype, degree=self.degree)
        self.gauss = 'gauss{}'.format(2*self.degree)

    @parametrize.enable_if(lambda btype, degree, boundary, **params: btype != 'discont' and degree != 0 and not boundary)
    def test_continuity(self):
        topos = dict(interfaces=self.domain.interfaces)
        if self.refine != 'hierarchical':
            # NOTE: the reason for excluding hierarchical refinements is that the
            # HierarchicalTopology does not (yet) take opposites into account when
            # forming its boundary, possibly resulting in elements that are coarser
            # than their opposing neighbours.
            if self.nelems > 1:
                topos['rightinterior'] = self.domain[:1].boundary['right']
                topos['leftinterior'] = self.domain[-1:].boundary['left']
            if self.periodic:
                topos['leftboundary'] = self.domain[:1].boundary['left']
                topos['rightboundary'] = self.domain[-1:].boundary['right']
        for name, topo in topos.items():
            with self.subTest(name):
                self.assertContinuous(topo=topo, geom=self.geom, basis=self.basis,
                                      continuity=0 if self.btype.endswith('std') or self.nelems == 1 and not self.refine and self.periodic else self.degree-1)

    @parametrize.enable_if(lambda btype, **params: not btype.startswith('h-'))
    def test_pum(self):
        self.assertPartitionOfUnity(topo=self.domain, basis=self.basis)

    @parametrize.enable_if(lambda periodic, **params: not periodic)
    def test_poly(self):
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=self.basis, degree=self.degree)


for ndims in range(1, 4):
    for btype in 'discont', 'h-std', 'th-std', 'h-spline', 'th-spline':
        for degree in range(0 if 'std' not in btype else 1, 4):
            for refine in False, 'uniform', 'hierarchical':
                for boundary in [None, 'bottom'] if ndims > 1 else [None]:
                    for periodic in False, True:
                        for nelems in range(1, 5-ndims):
                            basis(btype=btype, degree=degree, ndims=ndims, refine=refine, boundary=boundary, periodic=periodic, nelems=nelems)


@parametrize
class sparsity(TestCase):

    vals = {
        1: [60, 66, 70],
        2: [3012, 3216, 3424],
    }

    def test_sparsity(self):
        topo, geom = mesh.rectilinear([6]*self.ndim)
        topo = topo.refined_by(set(map(topo.transforms.index, itertools.chain(topo[1:3].transforms, topo[-2:].transforms))))

        ns = function.Namespace()
        ns.x = geom
        ns.tbasis = topo.basis('th-spline', degree=2, truncation_tolerance=1e-14)
        ns.tnotol = topo.basis('th-spline', degree=2, truncation_tolerance=0)
        ns.hbasis = topo.basis('h-spline', degree=2)

        gauss = topo.sample('gauss', 5)
        tA, tA_tol, hA = function.eval([
            function.as_coo(gauss.integral(ns.eval_ij('tbasis_i,k tbasis_j,k'))),
            function.as_coo(gauss.integral(ns.eval_ij('tnotol_i,k tnotol_j,k'))),
            function.as_coo(gauss.integral(ns.eval_ij('hbasis_i,k hbasis_j,k')))])

        tA_nnz, tA_tol_nnz, hA_nnz = self.vals[self.ndim]
        self.assertEqual(len(tA[0]), tA_nnz)
        self.assertEqual(len(tA_tol[0]), tA_tol_nnz)
        self.assertEqual(len(hA[0]), hA_nnz)


for ndim in 1, 2:
    sparsity(ndim=ndim)


class structured(basisTest):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.rectilinear([2, 3])

    def test_std_equalorder(self):
        for p in range(1, 3):
            basis = self.domain.basis('std', degree=p)
            self.assertEqual(len(basis), (3+2*(p-1))*(4+3*(p-1)))
            self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=0)
            self.assertPartitionOfUnity(topo=self.domain, basis=basis)
            self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=p)

    def test_spline_equalorder(self):
        for p in range(1, 3):
            basis = self.domain.basis('spline', degree=p)
            self.assertEqual(len(basis), (2+p)*(3+p))
            self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=p-1)
            self.assertPartitionOfUnity(topo=self.domain, basis=basis)
            self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=p)

    def test_std_mixedorder(self):
        basis = self.domain.basis('std', degree=(1, 2))
        self.assertEqual(len(basis), 3*7)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        basis = self.domain.basis('std', degree=(2, 1))
        self.assertEqual(len(basis), 5*4)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)

    def test_spline_mixedorder(self):
        basis = self.domain.basis('spline', degree=(1, 2))
        self.assertEqual(len(basis), 3*5)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        basis = self.domain.basis('spline', degree=(2, 1))
        self.assertEqual(len(basis), 4*4)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)

    def test_knotvalues(self):
        # test refinement of knotvalues[0] -> [0,1/2,1]
        basis = self.domain.basis('spline', degree=2, knotvalues=[[0, 1], [0, 1/3, 2/3, 1]])
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=1)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=2)

    def test_knotmultiplicities(self):
        # test refinement of knotmultiplicities[0] -> [3,1,3]
        basis = self.domain.basis('spline', degree=2, knotmultiplicities=[[3, 3], [3, 1, 1, 3]])
        self.assertEqual(len(basis), 4*5)
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=1)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=2)
        basis = self.domain.basis('spline', degree=2, knotmultiplicities=[[3, 3], [3, 2, 1, 3]])
        self.assertEqual(len(basis), 4*6)
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=0)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=2)

    def test_continuity(self):
        # test refinement of knotmultiplicities[0] -> [3,1,3]
        basis = self.domain.basis('spline', degree=2, continuity=0)
        self.assertEqual(len(basis), 5*7)
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=0)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=2)
        basis = self.domain.basis('spline', degree=2, continuity=0, knotmultiplicities=[[3, 3], None])
        self.assertEqual(len(basis), 5*7)
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=basis, continuity=0)
        self.assertPartitionOfUnity(topo=self.domain, basis=basis)
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=basis, degree=2)


@parametrize
class structured_line(basisTest):

    def setUp(self):
        super().setUp()
        verts = numpy.linspace(0, 1, self.nelems+1)
        self.domain, self.geom = mesh.line(verts, periodic=self.periodic)
        self.geom = self.geom[None]
        self.basis = self.domain.basis(self.btype, degree=self.degree)

    @parametrize.enable_if(lambda btype, **params: btype != 'discont')
    def test_continuity(self):
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=self.basis, continuity=0 if self.btype == 'std' else self.degree-1)

    def test_pum(self):
        self.assertPartitionOfUnity(topo=self.domain, basis=self.basis)

    @parametrize.enable_if(lambda periodic, **params: not periodic)
    def test_poly(self):
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=self.basis, degree=self.degree)


for btype in ['discont', 'spline', 'std']:
    for periodic in False, True:
        for degree in range(0 if btype == 'discont' else 1, 4):
            for nelems in range(2*degree or 1, 2*degree + 3):
                structured_line(periodic=periodic, btype=btype, degree=degree, nelems=nelems)


@parametrize
class structured_rect1d(basisTest):

    def setUp(self):
        super().setUp()
        verts = numpy.linspace(0, 1, self.nelems+1)
        self.domain, self.geom = mesh.rectilinear([verts], periodic=(0,) if self.periodic else ())
        self.basis = self.domain.basis(self.btype, degree=self.degree) if self.btype != 'spline' \
            else self.domain.basis(self.btype, degree=self.degree, continuity=self.continuity)

    @parametrize.enable_if(lambda continuity, **params: continuity >= 0)
    def test_continuity(self):
        self.assertContinuous(topo=self.domain.interfaces, geom=self.geom, basis=self.basis, continuity=self.continuity)

    def test_pum(self):
        self.assertPartitionOfUnity(topo=self.domain, basis=self.basis)

    @parametrize.enable_if(lambda periodic, **params: not periodic)
    def test_poly(self):
        self.assertPolynomial(topo=self.domain, geom=self.geom, basis=self.basis, degree=self.degree)


for btype in ['discont', 'spline', 'std']:
    for periodic in False, True:
        for nelems in range(1, 4):
            for degree in range(0 if btype == 'discont' else 1, 4):
                for continuity in [-1] if btype == 'discont' else [0] if btype == 'std' else range(degree):
                    structured_rect1d(periodic=periodic, btype=btype, degree=degree, nelems=nelems, continuity=continuity)


class structured_rect1d_periodic_knotmultiplicities(basisTest):

    def test(self):
        for knotmultiplicities, ndofs in [([3, 1, 3], 4), ([3, 2, 1, 3], 6)]:
            domain, geom = mesh.rectilinear([len(knotmultiplicities)-1], periodic=[0])
            basis = domain.basis('spline', degree=3, knotmultiplicities=[knotmultiplicities])
            self.assertEqual(len(basis), ndofs)
            self.assertContinuous(topo=domain.interfaces, geom=geom, basis=basis, continuity=0)
            self.assertPartitionOfUnity(topo=domain, basis=basis)

    def test_discontinuous(self):
        pdomain, pgeom = mesh.rectilinear([3], periodic=[0])
        rdomain, rgeom = mesh.rectilinear([3])
        pbasis = pdomain.basis('spline', degree=2, knotmultiplicities=[[3, 1, 2, 3]])
        rbasis = rdomain.basis('spline', degree=2, knotmultiplicities=[[1, 1, 2, 1]])
        psampled = pdomain.sample('gauss', 2).eval(pbasis)
        rsampled = rdomain.sample('gauss', 2).eval(rbasis)
        self.assertAllAlmostEqual(psampled, rsampled)


@parametrize
class unstructured_topology(TestCase):

    def setUp(self):
        super().setUp()
        if self.ndims == 2:
            domain, geom = mesh.unitsquare(4, self.variant)
            nverts = 25
        elif self.variant == 'tensor':
            structured, geom = mesh.rectilinear([numpy.linspace(0, 1, 5-i) for i in range(self.ndims)])
            domain = topology.ConnectedTopology(structured.space, structured.references, structured.transforms, structured.opposites, structured.connectivity)
            nverts = numpy.prod([5-i for i in range(self.ndims)])
        elif self.variant == 'simplex':
            numpy.random.seed(0)
            nverts = 20
            simplices = numeric.overlapping(numpy.arange(nverts), n=self.ndims+1)
            coords = numpy.random.normal(size=(nverts, self.ndims))
            space = 'test'
            transforms = transformseq.IndexTransforms(self.ndims, len(simplices))
            domain = topology.SimplexTopology(space, simplices, transforms, transforms)
            geom = (domain.basis('std', degree=1) * coords.T).sum(-1)
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
                element.getsimplex(self.ndims): numpy.prod(self.degree+numpy.arange(self.ndims)+1) // numpy.prod(numpy.arange(self.ndims)+1)}
            ndofs = sum(ndofs_by_ref[reference] for reference in self.domain.references)
        elif self.degree == 1:
            ndofs = self.nverts
        else:
            raise NotImplementedError
        self.assertEqual(len(self.basis), ndofs)

    def test_pum_sum(self):
        # Note that this test holds for btype 'lagrange' as well, although the
        # basis functions are not confined to range [0,1].
        error2 = self.domain.integrate((1-self.basis.sum(0))**2*function.J(self.geom), ischeme='gauss', degree=2*self.degree)
        numpy.testing.assert_almost_equal(error2, 0, decimal=22)

    @parametrize.enable_if(lambda btype, **params: btype != 'lagrange')
    def test_pum_range(self):
        values = self.domain.sample('gauss', 2*self.degree).eval(self.basis)
        self.assertTrue((values > 0-1e-10).all())
        self.assertTrue((values < 1+1e-10).all())

    def test_poly(self):
        target = self.geom.sum(-1) if self.btype == 'bubble' \
            else (self.geom**self.degree).sum(-1) + self.domain.f_index if self.btype == 'discont' \
            else (self.geom**self.degree).sum(-1)
        projection = self.domain.projection(target, onto=self.basis, geometry=self.geom, ischeme='gauss', degree=2*self.degree, droptol=0)
        error2 = self.domain.integrate((target-projection)**2*function.J(self.geom), ischeme='gauss', degree=2*self.degree)
        numpy.testing.assert_almost_equal(error2, 0, decimal=24)


for ndims in 1, 2, 3:
    for variant in ['simplex', 'tensor'] if ndims != 2 else ['triangle', 'square', 'mixed']:
        for btype in ['discont', 'bernstein', 'lagrange', 'std', 'bubble'][:5 if variant in ('simplex', 'triangle') else 4]:
            for degree in [0, 1, 2, 3] if btype == 'discont' else [2] if btype == 'bubble' else [1, 2, 3]:
                unstructured_topology(ndims=ndims, btype=btype, degree=degree, variant=variant)
