from nutils import *
import tempfile, pathlib, os
from nutils.testing import *

@parametrize
class gmsh(TestCase):

  def setUp(self):
    path = pathlib.Path(__file__).parent/'test_mesh'/'mesh{0.ndims}d_p{0.degree}_v{0.version}.msh'.format(self)
    self.domain, self.geom = mesh.gmsh(path)

  @requires('meshio')
  def test_volume(self):
    for group, exact_volume in ((),2), ('left',1), ('right',1):
      with self.subTest(group or 'all'):
        volume = self.domain[group].integrate(function.J(self.geom), ischeme='gauss1')
        self.assertAllAlmostEqual(volume, exact_volume, places=10)

  @requires('meshio')
  def test_divergence(self):
    for group, exact_volume in ((),2), ('left',1), ('right',1):
      with self.subTest(group or 'all'):
        volumes = self.domain[group].boundary.integrate(self.geom*self.geom.normal()*function.J(self.geom), ischeme='gauss1')
        self.assertAllAlmostEqual(volumes[:2], exact_volume, places=10)
        self.assertAllAlmostEqual(volumes[2:], 0, places=10)

  @requires('meshio')
  def test_length(self):
    for vgroup, bgroup, exact_length in ((),(),6), ((),'neumann',2), ((),'dirichlet',4), ('left',(),4), ('right',(),4):
      with self.subTest('{},{}'.format(vgroup or 'all', bgroup or 'all')):
        length = self.domain[vgroup].boundary[bgroup].integrate(function.J(self.geom), ischeme='gauss1')
        self.assertAllAlmostEqual(length, exact_length, places=10)

  @requires('meshio')
  def test_interfaces(self):
    err = self.domain.interfaces.sample('uniform', 2).eval(self.geom - function.opposite(self.geom))
    self.assertAllAlmostEqual(err[:,:2], 0, places=13) # the third dimension (if present) is discontinuous at the periodic boundary

  @requires('meshio')
  def test_ifacegroup(self):
    for name in 'iface', 'left', 'right':
      with self.subTest(name):
        topo = (self.domain.interfaces if name == 'iface' else self.domain[name].boundary)['iface']
        x1, x2 = topo.sample('uniform', 2).eval([self.geom, function.opposite(self.geom)])
        self.assertAllAlmostEqual(x1[:,0], 1, places=13)
        self.assertAllAlmostEqual(x2[:,0], 1, places=13)
        self.assertAllAlmostEqual(x1, x2, places=13)

  @requires('meshio')
  def test_pointeval(self):
    x = self.domain.points.sample('gauss', 1).eval(self.geom)
    self.assertAllAlmostEqual(x[:,0], 1, places=15)
    self.assertAllAlmostEqual(x[:,1], 0, places=15)

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
