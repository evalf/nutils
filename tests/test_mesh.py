from nutils import *
import tempfile, pathlib, os, io
from nutils.testing import *

MESHDIR = pathlib.Path(__file__).parent/'test_mesh'

class gmsh_init(TestCase):

  geo = [ # single element triangle mesh
    '$MeshFormat', '2.2 0 8', '$EndMeshFormat',
    '$PhysicalNames', '1', '2 1 "v"', '$EndPhysicalNames',
    '$Nodes', '3', '1 0 0 0', '2 1 0 0', '3 0 1 0', '$EndNodes',
    '$Elements', '1', '1 2 2 1 8 1 2 3', '$EndElements']

  def test_valid(self):
    domain, geom = mesh.gmsh(io.StringIO('\n'.join(self.geo)))
    self.assertEqual(len(domain), 1)

  def test_missing_section(self):
    with self.assertRaises(ValueError):
      mesh.gmsh(io.StringIO('\n'.join(self.geo[3:]))) # missing meshformat
    with self.assertRaises(ValueError):
      mesh.gmsh(io.StringIO('\n'.join(self.geo[:-4]))) # missing elements

  def test_meshformat(self):
    geo = self.geo.copy()
    geo[1] = '9.0 0 8' # imaginary future version
    with self.assertRaises(ValueError):
      mesh.gmsh(io.StringIO('\n'.join(geo)))
    geo[1] = '2.2 1 8' # binary data
    with self.assertRaises(ValueError):
      mesh.gmsh(io.StringIO('\n'.join(geo)))

@parametrize
class gmsh(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.gmsh(MESHDIR/'{0.data}_p{0.degree}.msh'.format(self))

  def test_volume(self):
    volume = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volume, 1, decimal=10)

  def test_length(self):
    for group, exact_length in ('neumann',1), ('dirichlet',3), ((),2*self.domain.ndims):
      with self.subTest(group or 'all'):
        length = self.domain.boundary[group].integrate(function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(length, exact_length, decimal=10)

  def test_interfaces(self):
    err = self.domain.interfaces.sample('uniform', 2).eval(self.geom - function.opposite(self.geom))
    numpy.testing.assert_almost_equal(err, 0, decimal=15)

  def test_divergence(self):
    volumes = self.domain.boundary.integrate(self.geom*self.geom.normal()*function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volumes, 1, decimal=10)

  def test_pointeval(self):
    xy = self.domain.points.sample('gauss', 1).eval(self.geom)
    self.assertEqual(xy.shape, (2, 2) if self.domain.ndims==2 else (4, 3))
    self.assertTrue(numpy.equal(xy, ([1,0] if self.domain.ndims==2 else [1,0,0])).all())

  def test_refine(self):
    boundary1 = self.domain.refined.boundary
    boundary2 = self.domain.boundary.refined
    assert len(boundary1) == len(boundary2) == len(self.domain.boundary) * element.getsimplex(self.domain.ndims-1).nchildren
    assert set(boundary1.transforms) == set(boundary2.transforms)
    assert all(boundary2.references[boundary2.transforms.index(trans)] == ref for ref, trans in zip(boundary1.references, boundary1.transforms))

  def test_refinesubset(self):
    domain = topology.SubsetTopology(self.domain, [ref if ielem % 2 else ref.empty for ielem, ref in enumerate(self.domain.references)])
    boundary1 = domain.refined.boundary
    boundary2 = domain.boundary.refined
    assert len(boundary1) == len(boundary2) == len(domain.boundary) * element.getsimplex(domain.ndims-1).nchildren
    assert set(boundary1.transforms) == set(boundary2.transforms)
    assert all(boundary2.references[boundary2.transforms.index(trans)] == ref for ref, trans in zip(boundary1.references, boundary1.transforms))

gmsh(data='square', degree=1)
gmsh(data='square', degree=2)
gmsh(data='cube', degree=1)

class gmshrect(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.gmsh(MESHDIR/'rectangle.msh')

  def test_volume(self):
    volume = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volume, 2, decimal=10)

  def test_length(self):
    length = self.domain.boundary.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(length, 6, decimal=10)

  def test_divergence(self):
    volumes = self.domain.boundary.integrate(self.geom*self.geom.normal()*function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volumes, 2, decimal=10)

  def test_subvolume(self):
    for group in 'left', 'right':
      with self.subTest(group):
        subdom = self.domain[group]
        volume = subdom.integrate(function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(volume, 1, decimal=10)

  def test_sublength(self):
    for group in 'left', 'right':
      with self.subTest(group):
        subdom = self.domain[group]
        length = subdom.boundary.integrate(function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(length, 4, decimal=10)

  def test_subdivergence(self):
    for group in 'left', 'right':
      with self.subTest(group):
        subdom = self.domain[group]
        volumes = subdom.boundary.integrate(self.geom*self.geom.normal()*function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(volumes, 1, decimal=10)

  def test_iface(self):
    ax, ay = self.domain.interfaces['iface'].sample('uniform', 1).eval(self.geom).T
    bx, by = self.domain['left'].boundary['iface'].sample('uniform', 1).eval(self.geom).T
    cx, cy = self.domain['right'].boundary['iface'].sample('uniform', 1).eval(self.geom).T
    self.assertTrue(numpy.equal(ax, 1).all())
    self.assertTrue(numpy.equal(bx, 1).all())
    self.assertTrue(numpy.equal(cx, 1).all())
    self.assertTrue(min(ay) == min(by) == min(cy))
    self.assertTrue(max(ay) == max(by) == max(cy))

class gmshperiodic(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.gmsh(MESHDIR/'periodic.msh')

  def test_volume(self):
    volume = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volume, 1, decimal=10)

  def test_length(self):
    for group, exact_length in ('right',1), ('left',1), ((),2):
      with self.subTest(group or 'all'):
        length = self.domain.boundary[group].integrate(function.J(self.geom), ischeme='gauss1')
        numpy.testing.assert_almost_equal(length, exact_length, decimal=10)

  def test_interface(self):
    err = self.domain.interfaces['periodic'].sample('uniform', 2).eval(function.opposite(self.geom) - self.geom)
    numpy.testing.assert_almost_equal(abs(err)-[0,1], 0, decimal=15)

  def test_basis(self):
    basis = self.domain.basis('std', degree=1)
    err = self.domain.interfaces.integrate((basis - function.opposite(basis))*function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(err, 0, decimal=15)

@parametrize
class rectilinear(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = getattr(mesh, self.method)([4,5])

  def test_volume(self):
    volume = self.domain.integrate(function.J(self.geom), ischeme='gauss1')
    numpy.testing.assert_almost_equal(volume, 20, decimal=15)

  def divergence(self):
    self.domain.check_boundary(geometry=self.geom)

  def test_length(self):
    for group, exact_length in ('right',5), ('left',5), ('top',4), ('bottom',4), ((),18):
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

rectilinear('new', method='newrectilinear')
rectilinear('old', method='rectilinear')
