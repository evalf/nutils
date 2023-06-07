from nutils import mesh, function, element, transform, topology
from nutils.testing import TestCase, parametrize, requires
import pathlib
import numpy


@parametrize
class gmsh(TestCase):

    def setUp(self):
        super().setUp()
        path = pathlib.Path(__file__).parent/'test_mesh'/'mesh{0.ndims}d_p{0.degree}_v{0.version}.msh'.format(self)
        self.domain, self.geom = mesh.gmsh(path)

    @requires('meshio')
    def test_volume(self):
        for group, exact_volume in ((), 2), ('left', 1), ('right', 1):
            with self.subTest(group or 'all'):
                volume = self.domain[group].integrate(function.J(self.geom), ischeme='gauss1')
                self.assertAllAlmostEqual(volume, exact_volume, places=10)

    @requires('meshio')
    def test_divergence(self):
        for group, exact_volume in ((), 2), ('left', 1), ('right', 1):
            with self.subTest(group or 'all'):
                volumes = self.domain[group].boundary.integrate(self.geom*self.geom.normal()*function.J(self.geom), ischeme='gauss1')
                self.assertAllAlmostEqual(volumes[:2], [exact_volume]*2, places=10)
                self.assertAllAlmostEqual(volumes[2:], numpy.zeros((self.domain.ndims-2,)), places=10)

    @requires('meshio')
    def test_length(self):
        for name, boundary, exact_length in (('full', self.domain.boundary, 6),
                                             ('neumann', self.domain.boundary['neumann'], 2),
                                             ('dirichlet', self.domain.boundary['dirichlet'], 4),
                                             ('extra', self.domain.boundary['extra'], 2),
                                             ('extraneumann', self.domain.boundary['extra'] & self.domain.boundary['neumann'], 1),
                                             ('extradirichlet', self.domain.boundary['extra'] & self.domain.boundary['dirichlet'], 1),
                                             ('left', self.domain['left'].boundary, 4),
                                             ('right', self.domain['right'].boundary, 4)):
            with self.subTest(name):
                length = boundary.integrate(function.J(self.geom), ischeme='gauss1')
                self.assertAllAlmostEqual(length, exact_length, places=10)

    @requires('meshio')
    def test_interfaces(self):
        a, b = self.domain.interfaces.sample('bezier', 2).eval([self.geom, function.opposite(self.geom)])
        self.assertAllAlmostEqual(a[:, :2], b[:, :2], places=11)  # the third dimension (if present) is discontinuous at the periodic boundary

    @requires('meshio')
    def test_ifacegroup(self):
        for name in 'iface', 'left', 'right':
            with self.subTest(name):
                topo = (self.domain.interfaces if name == 'iface' else self.domain[name].boundary)['iface']
                smpl = topo.sample('uniform', 2)
                x1, x2 = smpl.eval([self.geom, function.opposite(self.geom)])
                self.assertAllAlmostEqual(x1[:, 0], numpy.ones((smpl.npoints,)), places=13)
                self.assertAllAlmostEqual(x2[:, 0], numpy.ones((smpl.npoints,)), places=13)
                self.assertAllAlmostEqual(x1, x2, places=13)

    @requires('meshio')
    def test_pointeval(self):
        smpl = self.domain.points.sample('gauss', 1)
        x = smpl.eval(self.geom)
        self.assertAllAlmostEqual(x[:, 0], numpy.ones((smpl.npoints,)), places=15)
        self.assertAllAlmostEqual(x[:, 1], numpy.zeros((smpl.npoints,)), places=15)

    @requires('meshio')
    def test_refine(self):
        boundary1 = self.domain.refined.boundary
        boundary2 = self.domain.boundary.refined
        assert len(boundary1) == len(boundary2) == len(self.domain.boundary) * element.getsimplex(self.domain.ndims-1).nchildren
        assert set(map(transform.canonical, boundary1.transforms)) == set(map(transform.canonical, boundary2.transforms))
        assert all(boundary2.references[boundary2.transforms.index(trans)] == ref for ref, trans in zip(boundary1.references, boundary1.transforms))

    @requires('meshio')
    def test_refinesubset(self):
        domain = topology.SubsetTopology(self.domain, [ref if ielem % 2 else ref.empty for ielem, ref in enumerate(self.domain.references)])
        boundary1 = domain.refined.boundary
        boundary2 = domain.boundary.refined
        assert len(boundary1) == len(boundary2) == len(domain.boundary) * element.getsimplex(domain.ndims-1).nchildren
        assert set(map(transform.canonical, boundary1.transforms)) == set(map(transform.canonical, boundary2.transforms))
        assert all(boundary2.references[boundary2.transforms.index(trans)] == ref for ref, trans in zip(boundary1.references, boundary1.transforms))


for ndims in 2, 3:
    for version in 2, 4:
        for degree in range(1, 5 if ndims == 2 else 3):
            gmsh(ndims=ndims, version=version, degree=degree)


@parametrize
class gmshmanifold(TestCase):

    def setUp(self):
        super().setUp()
        path = pathlib.Path(__file__).parent/'test_mesh'/'mesh3dmani_p{0.degree}_v{0.version}.msh'.format(self)
        self.domain, self.geom = mesh.gmsh(path)

    @requires('meshio')
    def test_volume(self):
        volume = self.domain.integrate(function.J(self.geom), degree=self.degree)
        self.assertAllAlmostEqual(volume, 2*numpy.pi, places=0 if self.degree == 1 else 1)

    @requires('meshio')
    def test_length(self):
        length = self.domain.boundary.integrate(function.J(self.geom), degree=self.degree)
        self.assertAllAlmostEqual(length, 2*numpy.pi, places=1 if self.degree == 1 else 3)


for version in 2, 4:
    for degree in 1, 2:
        gmshmanifold(version=version, degree=degree)


class rectilinear(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.rectilinear([4, 5])

    def test_volume(self):
        volume = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(volume, 20, decimal=15)

    def divergence(self):
        self.domain.check_boundary(geometry=self.geom)

    def test_length(self):
        for group, exact_length in ('right', 5), ('left', 5), ('top', 4), ('bottom', 4), ((), 18):
            with self.subTest(group or 'all'):
                length = self.domain.boundary[group].integrate(function.J(self.geom), ischeme='gauss1')
                numpy.testing.assert_almost_equal(length, exact_length, decimal=10)

    def test_interface(self):
        geomerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom - function.opposite(self.geom))
        numpy.testing.assert_almost_equal(geomerr, 0, decimal=15)
        normalerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom.normal() + function.opposite(self.geom.normal()))
        numpy.testing.assert_almost_equal(normalerr, 0, decimal=15)

    def test_pum(self):
        for basistype in 'discont', 'std', 'spline':
            for degree in 1, 2, 3:
                with self.subTest(basistype+str(degree)):
                    basis = self.domain.basis(basistype, degree=degree)
                    values = self.domain.interfaces.sample('uniform', 2).eval(basis*function.J(self.geom))
                    numpy.testing.assert_almost_equal(values.sum(1), 1)


@parametrize
class unitsquare(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.unitsquare(nelems=4, etype=self.etype)

    def test_volume(self):
        self.assertAllAlmostEqual(self.domain.volume(self.geom), 1)

    def test_boundaries(self):
        self.assertAllAlmostEqual(self.domain.boundary.volume(self.geom), 4)
        self.domain.check_boundary(geometry=self.geom)

    def test_boundary_groups(self):
        numpy.testing.assert_almost_equal(self.domain.boundary['left'].sample('gauss', 0).eval(self.geom[0]), 0)
        numpy.testing.assert_almost_equal(self.domain.boundary['bottom'].sample('gauss', 0).eval(self.geom[1]), 0)
        numpy.testing.assert_almost_equal(self.domain.boundary['right'].sample('gauss', 0).eval(self.geom[0]), 1)
        numpy.testing.assert_almost_equal(self.domain.boundary['top'].sample('gauss', 0).eval(self.geom[1]), 1)

    def test_interface(self):
        geomerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom - function.opposite(self.geom))
        numpy.testing.assert_almost_equal(geomerr, 0, decimal=15)
        normalerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom.normal() + function.opposite(self.geom.normal()))
        numpy.testing.assert_almost_equal(normalerr, 0, decimal=15)

unitsquare(etype='square')
unitsquare(etype='triangle')
unitsquare(etype='mixed')
unitsquare(etype='multipatch')


@parametrize
class unitcircle(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.unitcircle(nelems=8, variant=self.variant)

    def test_volume(self):
        self.assertAllAlmostEqual(self.domain.volume(self.geom, degree=6), numpy.pi)

    def test_boundaries(self):
        self.assertAllAlmostEqual(self.domain.boundary.volume(self.geom, degree=6), 2 * numpy.pi)
        self.domain.check_boundary(geometry=self.geom, degree=8)

    def test_interface(self):
        geomerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom - function.opposite(self.geom))
        numpy.testing.assert_almost_equal(geomerr, 0, decimal=15)
        normalerr = self.domain.interfaces.sample('uniform', 2).eval(self.geom.normal() + function.opposite(self.geom.normal()))
        numpy.testing.assert_almost_equal(normalerr, 0, decimal=14)

unitcircle(variant='rectilinear')
unitcircle(variant='multipatch')
