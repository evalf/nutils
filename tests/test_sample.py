from nutils import *
import random, itertools, functools, warnings as _builtin_warnings
from nutils.testing import *
from nutils.sample import Sample
from nutils.pointsseq import PointsSequence
from nutils.transformseq import IdentifierTransforms

class Common:

  @property
  def desired_npoints(self):
    return sum(util.product(p.npoints for p in points) for points in self.desired_points)

  def test_spaces(self):
    self.assertEqual(self.sample.spaces, self.desired_spaces)

  def test_ndims(self):
    self.assertEqual(self.sample.ndims, self.desired_ndims)

  def test_nelems(self):
    self.assertEqual(self.sample.nelems, self.desired_nelems)

  def test_npoints(self):
    self.assertEqual(self.sample.npoints, self.desired_npoints)

  def test_getindex(self):
    assert len(self.desired_indices) == self.desired_nelems
    for ielem, desired in enumerate(self.desired_indices):
      self.assertEqual(self.sample.getindex(ielem).tolist(), numpy.ravel(desired).tolist())
    with self.assertRaises(IndexError):
      self.sample.getindex(-1)
    with self.assertRaises(IndexError):
      self.sample.getindex(self.desired_nelems)

  def test_index(self):
    actual = [indices.tolist() for indices in self.sample.index]
    desired = [numpy.ravel(i).tolist() for i in self.desired_indices]
    self.assertEqual(actual, desired)

  def test_get_evaluable_indices(self):
    assert len(self.desired_indices) == self.desired_nelems
    actual = self.sample.get_evaluable_indices(evaluable.Argument('ielem', (), int))
    for ielem, desired in enumerate(self.desired_indices):
      self.assertEqual(actual.eval(ielem=ielem).tolist(), desired)

  def test_get_evaluable_weights(self):
    assert len(self.desired_points) == self.desired_nelems
    actual = self.sample.get_evaluable_weights(evaluable.Argument('ielem', (), int))
    for ielem, points in enumerate(self.desired_points):
      desired = functools.reduce(lambda l, r: numpy.einsum('...,i->...i', l, r), (p.weights for p in points))
      self.assertAllAlmostEqual(actual.eval(ielem=ielem).tolist(), desired.tolist())

  def test_update_lower_args(self):
    assert len(self.desired_transform_chains) == len(self.desired_points) == self.desired_nelems
    points_shape, transform_chains, coords = self.sample.update_lower_args(evaluable.Argument('ielem', (), int), (), {}, {})
    for ielem, (desired_chains, desired_points) in enumerate(zip(self.desired_transform_chains, self.desired_points)):
      assert len(desired_chains) == len(desired_points) == len(self.desired_spaces)
      desired_shape = tuple(p.coords.shape[0] for p in desired_points)
      actual_shape = tuple(n.__index__() for n in evaluable.Tuple(points_shape).eval(ielem=ielem))
      self.assertEqual(actual_shape, desired_shape)
      offset = 0
      for space, desired_chain, desired_point in zip(self.desired_spaces, desired_chains, desired_points):
        self.assertEqual(transform_chains[space][0].eval(ielem=ielem), desired_chain)
        desired_coords = desired_point.coords
        desired_coords = numpy.lib.stride_tricks.as_strided(desired_coords, shape=(*desired_shape, desired_point.ndims,), strides=(0,)*offset+desired_coords.strides[:-1]+(0,)*(len(points_shape)-offset-desired_coords.ndim+1)+desired_coords.strides[-1:])
        actual_coords = coords[space].eval(ielem=ielem)
        self.assertEqual(actual_coords.shape, desired_coords.shape)
        self.assertAllAlmostEqual(actual_coords, desired_coords)
        offset += desired_point.coords.ndim - 1
    with self.assertRaisesRegex(ValueError, '^Nested'):
      self.sample.update_lower_args(evaluable.Argument('ielem2', (), int), points_shape, transform_chains, coords)

  @property
  def _desired_element_tri(self):
    assert len(self.desired_points) == self.desired_nelems
    for p in self.desired_points:
      yield util.product(p).tri.tolist()

  def test_get_element_tri(self):
    for ielem, desired in enumerate(self._desired_element_tri):
      self.assertEqual(self.sample.get_element_tri(ielem).tolist(), desired)
    with self.assertRaises(IndexError):
      self.sample.get_element_tri(-1)
    with self.assertRaises(IndexError):
      self.sample.get_element_tri(self.desired_nelems)

  def test_tri(self):
    assert len(self.desired_indices) == self.desired_nelems
    desired = []
    for idx, tri in zip(self.desired_indices, self._desired_element_tri):
      desired.extend(numpy.take(numpy.ravel(idx), tri).tolist())
    actual = self.sample.tri
    self.assertEqual(actual.shape, (len(desired), self.desired_ndims+1))
    self.assertEqual(sorted(actual.tolist()), sorted(desired))

  @property
  def _desired_element_hull(self):
    assert len(self.desired_points) == self.desired_nelems
    for p in self.desired_points:
      yield util.product(p).hull.tolist()

  def test_get_element_hull(self):
    if self.desired_ndims > 1:
      for ielem, desired in enumerate(self._desired_element_hull):
        self.assertEqual(sorted(self.sample.get_element_hull(ielem).tolist()), sorted(desired))
      with self.assertRaises(IndexError):
        self.sample.get_element_hull(-1)
      with self.assertRaises(IndexError):
        self.sample.get_element_hull(self.desired_nelems)

  def test_hull(self):
    assert len(self.desired_indices) == self.desired_nelems
    desired = []
    for idx, hull in zip(self.desired_indices, self._desired_element_hull):
      desired.extend(numpy.take(numpy.ravel(idx), hull).tolist())
    actual = self.sample.hull
    self.assertEqual(actual.shape, (len(desired), self.desired_ndims))
    self.assertEqual(sorted(actual.tolist()), sorted(desired))

  def test_take_elements_single(self):
    for ielem in range(self.desired_nelems):
      take = self.sample.take_elements(numpy.array([ielem]))
      self.assertEqual(take.nelems, 1)
      self.assertEqual(take.ndims, self.desired_ndims)
      points_shape, transform_chains, coords = take.update_lower_args(evaluable.Argument('ielem', (), int), (), {}, {})
      for space, desired_chain in zip(self.desired_spaces, self.desired_transform_chains[ielem]):
        self.assertEqual(transform_chains[space][0].eval(ielem=0), desired_chain)

  def test_take_elements_empty(self):
    take = self.sample.take_elements(numpy.array([], int))
    self.assertEqual(take.nelems, 0)
    self.assertEqual(take.npoints, 0)

  def test_ones_at(self):
    self.assertEqual(self.sample(function.ones((), int)).eval().tolist(), [1]*self.desired_npoints)

  def test_at_in_integral(self):
    topo, geom = mesh.line(2, space='parent-integral')
    actual = topo.integral(self.sample(function.jacobian(geom)), degree=0)
    self.assertEqual(actual.eval().round(5).tolist(), [2]*self.desired_npoints)

  def test_asfunction(self):
    func = self.sample.asfunction(numpy.arange(self.sample.npoints))
    self.assertEqual(self.sample(func).eval().tolist(), numpy.arange(self.desired_npoints).tolist())

class Empty(TestCase, Common):

  def setUp(self):
    super().setUp()
    self.sample = Sample.empty(('a', 'b'), 2)
    self.desired_spaces = 'a', 'b'
    self.desired_ndims = 2
    self.desired_nelems = 0
    self.desired_transform_chains = ()
    self.desired_points = ()
    self.desired_indices = ()
    self.desired_tri = ()
    self.desired_hull = ()

  def test_update_lower_args(self):
    pass

class Add(TestCase, Common):

  def setUp(self):
    super().setUp()
    points1 = [element.getsimplex(2).getpoints('bezier', 2)]*3
    points2 = [element.getsimplex(2).getpoints('bezier', 3)]*2
    transforms1 = IdentifierTransforms(2, 'a', 3)
    transforms2 = IdentifierTransforms(2, 'b', 2)
    sample1 = Sample.new('a', (transforms1, transforms1), PointsSequence.from_iter(points1, 2))
    sample2 = Sample.new('a', (transforms2, transforms2), PointsSequence.from_iter(points2, 2))
    self.sample = sample1 + sample2
    self.desired_spaces = 'a',
    self.desired_ndims = 2
    self.desired_nelems = 5
    self.desired_transform_chains = [[t] for t in (*transforms1, *transforms2)]
    self.desired_points = [[p] for p in points1 + points2]
    self.desired_indices = [0,1,2],[3,4,5],[6,7,8],[9,10,11,12,13,14],[15,16,17,18,19,20]

  def test_get_evaluable_indices(self):
    with self.assertRaises(NotImplementedError):
      super().test_get_evaluable_indices()

  def test_get_evaluable_weights(self):
    with self.assertRaises(NotImplementedError):
      super().test_get_evaluable_weights()

  def test_update_lower_args(self):
    with self.assertRaises(NotImplementedError):
      super().test_update_lower_args()

  def test_asfunction(self):
    with self.assertRaises(NotImplementedError):
      super().test_asfunction()

class Mul(TestCase, Common):

  def setUp(self):
    super().setUp()
    points1 = [element.getsimplex(1).getpoints('bezier', 2)]*2
    points2 = [(element.getsimplex(1)**2).getpoints('bezier', 2)] + [element.getsimplex(2).getpoints('bezier', 2)]*2
    transforms1 = IdentifierTransforms(1, 'a', 2)
    transforms2 = IdentifierTransforms(2, 'b', 3)
    sample1 = Sample.new('a', (transforms1, transforms1), PointsSequence.from_iter(points1, 1))
    sample2 = Sample.new('b', (transforms2, transforms2), PointsSequence.from_iter(points2, 2))
    self.sample = sample1 * sample2
    self.desired_spaces = 'a', 'b'
    self.desired_ndims = 3
    self.desired_nelems = 6
    self.desired_transform_chains = [(c1, c2) for c1 in transforms1 for c2 in transforms2]
    self.desired_points = [[p1, p2] for p1 in points1 for p2 in points2]
    self.desired_indices = tuple([[j+i*10 for j in J] for i in I] for I in [[0,1],[2,3]] for J in [[0,1,2,3],[4,5,6],[7,8,9]])

class Mul_left0d(TestCase, Common):

  def setUp(self):
    super().setUp()
    points1 = [element.getsimplex(0).getpoints('bezier', 2)]
    points2 = [(element.getsimplex(1)**2).getpoints('bezier', 2)] + [element.getsimplex(2).getpoints('bezier', 2)]*2
    transforms1 = IdentifierTransforms(0, 'a', 1)
    transforms2 = IdentifierTransforms(2, 'b', 3)
    sample1 = Sample.new('a', (transforms1, transforms1), PointsSequence.from_iter(points1, 0))
    sample2 = Sample.new('b', (transforms2, transforms2), PointsSequence.from_iter(points2, 2))
    self.sample = sample1 * sample2
    self.desired_spaces = 'a', 'b'
    self.desired_ndims = 2
    self.desired_nelems = 3
    self.desired_transform_chains = [(c1, c2) for c1 in transforms1 for c2 in transforms2]
    self.desired_points = [[p1, p2] for p1 in points1 for p2 in points2]
    self.desired_indices = tuple([[[0,1,2,3]],[[4,5,6]],[[7,8,9]]])

class Mul_right0d(TestCase, Common):

  def setUp(self):
    super().setUp()
    points1 = [(element.getsimplex(1)**2).getpoints('bezier', 2)] + [element.getsimplex(2).getpoints('bezier', 2)]*2
    points2 = [element.getsimplex(0).getpoints('bezier', 2)]
    transforms1 = IdentifierTransforms(2, 'a', 3)
    transforms2 = IdentifierTransforms(0, 'b', 1)
    sample1 = Sample.new('a', (transforms1, transforms1), PointsSequence.from_iter(points1, 2))
    sample2 = Sample.new('b', (transforms2, transforms2), PointsSequence.from_iter(points2, 0))
    self.sample = sample1 * sample2
    self.desired_spaces = 'a', 'b'
    self.desired_ndims = 2
    self.desired_nelems = 3
    self.desired_transform_chains = [(c1, c2) for c1 in transforms1 for c2 in transforms2]
    self.desired_points = [[p1, p2] for p1 in points1 for p2 in points2]
    self.desired_indices = tuple([[[0],[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])

@parametrize
class Zip(TestCase):

  def setUp(self):
    topoY, self.geomY = mesh.line(numpy.linspace(0,1,6), space='Y')
    topoX, self.geomX = mesh.unitsquare(nelems=3, etype=self.etype)
    self.sampleY = topoY.sample('uniform', 3)
    self.slope = numpy.array([1, .5]) # geomX == geomY * slope
    self.sampleX = topoX.locate(self.geomX, self.sampleY.eval(self.geomY * self.slope), tol=1e-10)
    self.stitched = self.sampleY.zip(self.sampleX)

  def test_eval(self):
    geomY, geomX = self.stitched.eval([self.geomY, self.geomX])
    self.assertAllAlmostEqual(geomY, self.sampleY.eval(self.geomY))
    self.assertAllAlmostEqual(geomX, geomY[:,numpy.newaxis] * self.slope)

  def test_integrate(self):
    self.assertAlmostEqual(self.stitched.integrate(function.J(self.geomY)), 1)
    self.assertAlmostEqual(self.stitched.integrate(function.J(self.geomX)), 5/9) # NOTE: != norm(slope)

  def test_nested(self):
    with self.assertRaisesRegex(ValueError, 'Nested integrals or samples in the same space are not supported.'):
      self.stitched.integral(self.stitched.integral(1)).eval()
    topoZ, geomZ = mesh.line(2, space='Z')
    inner = self.stitched.integral((geomZ - self.geomX) * function.J(self.geomY))
    outer = topoZ.integral(inner * function.J(geomZ), degree=2)
    self.assertAllAlmostEqual(outer.eval(), 2 - self.slope) # ∫_0^2 dz ∫_0^1 (z - α x) dx = ∫_0^2 (z - .5 α) dz = 2 - α

  def test_triplet(self):
    topoZ, geomZ = mesh.line(3, space='Z')
    sampleZ = topoZ.sample('uniform', 5)
    triplet = Sample.zip(self.sampleY, self.sampleX, sampleZ)
    geomX, geomY, geomZ = triplet.eval([self.geomX, self.geomY, geomZ])
    self.assertAllAlmostEqual(geomX, geomY[:,numpy.newaxis] * self.slope)
    self.assertAllAlmostEqual(geomY, geomZ / 3)

Zip(etype='square')
Zip(etype='triangle')
Zip(etype='mixed')

class TakeElements(TestCase, Common):

  def setUp(self):
    super().setUp()
    points1 = [element.getsimplex(1).getpoints('bezier', 2)]*2
    points2 = [element.getsimplex(1).getpoints('bezier', 2), element.getsimplex(1).getpoints('bezier', 3), element.getsimplex(1).getpoints('bezier', 4)]
    points3 = [element.getsimplex(2).getpoints('bezier', 2)] + [(element.getsimplex(1)**2).getpoints('bezier', 2)]
    transforms1 = IdentifierTransforms(1, 'a', 2)
    transforms2 = IdentifierTransforms(1, 'b', 3)
    transforms3 = IdentifierTransforms(2, 'c', 2)
    sample1 = Sample.new('a', (transforms1, transforms1), PointsSequence.from_iter(points1, 1))
    sample2 = Sample.new('b', (transforms2, transforms2), PointsSequence.from_iter(points2, 1))
    sample3 = Sample.new('c', (transforms3, transforms3), PointsSequence.from_iter(points3, 2))
    indices = [[0,0,0],[0,2,1],[1,1,0]]
    self.sample = (sample1 * (sample2 * sample3)).take_elements(numpy.einsum('ij,j->i', indices, [6,2,1]))
    self.desired_spaces = 'a','b','c'
    self.desired_ndims = 4
    self.desired_nelems = len(indices)
    self.desired_transform_chains = [[t[i] for t, i in zip((transforms1, transforms2, transforms3), I)] for I in indices]
    self.desired_points = [[p[i] for p, i in zip((points1, points2, points3), I)] for I in indices]
    self.desired_indices = []
    offset = 0
    for I in indices:
      shape = [p[i].npoints for p, i in zip((points1, points2, points3), I)]
      size = numpy.prod(shape)
      self.desired_indices.append((offset + numpy.arange(size).reshape(shape)).tolist())
      offset += size

  def test_asfunction(self):
    with self.assertRaises(NotImplementedError):
      super().test_asfunction()

class DefaultIndex(TestCase, Common):

  def setUp(self):
    super().setUp()
    line = element.getsimplex(1)
    triangle = element.getsimplex(2)
    points = [ref.getpoints('bezier', 2) for ref in (line**2, triangle, line**2)]
    self.transforms = IdentifierTransforms(2, 'test', 3)
    self.sample = Sample.new('a', (self.transforms, self.transforms), PointsSequence.from_iter(points, 2))
    self.desired_spaces = 'a',
    self.desired_ndims = 2
    self.desired_nelems = 3
    self.desired_transform_chains = [[t] for t in self.transforms]
    self.desired_points = [[p] for p in points]
    self.desired_indices = [0,1,2,3],[4,5,6],[7,8,9,10]

  def test_at(self):
    self.geom = function.rootcoords('a', 2) + numpy.array([0,2]) * function.transforms_index('a', self.transforms)
    actual = self.sample(self.geom).as_evaluable_array.eval()
    desired = numpy.array([[0,0],[0,1],[1,0],[1,1],[0,2],[1,2],[0,3],[0,4],[0,5],[1,4],[1,5]])
    self.assertAllAlmostEqual(actual, desired)

  def test_basis(self):
    with _builtin_warnings.catch_warnings():
      _builtin_warnings.simplefilter('ignore', category=evaluable.ExpensiveEvaluationWarning)
      self.assertAllAlmostEqual(self.sample(self.sample.basis()).as_evaluable_array.eval(), numpy.eye(11))

class CustomIndex(TestCase, Common):

  def setUp(self):
    super().setUp()
    line = element.getsimplex(1)
    triangle = element.getsimplex(2)
    points = [ref.getpoints('bezier', 2) for ref in (line**2, triangle, line**2)]
    self.transforms = IdentifierTransforms(2, 'test', 3)
    self.desired_indices = [5,10,4,9],[2,0,6],[7,8,3,1]
    self.sample = Sample.new('a', (self.transforms, self.transforms), PointsSequence.from_iter(points, 2), tuple(map(numpy.array, self.desired_indices)))
    self.desired_spaces = 'a',
    self.desired_ndims = 2
    self.desired_nelems = 3
    self.desired_transform_chains = [[t] for t in self.transforms]
    self.desired_points = [[p] for p in points]

  def test_at(self):
    self.geom = function.rootcoords('a', 2) + numpy.array([0,2]) * function.transforms_index('a', self.transforms)
    actual = self.sample(self.geom).as_evaluable_array.eval()
    desired = numpy.array([[0,0],[0,1],[1,0],[1,1],[0,2],[1,2],[0,3],[0,4],[0,5],[1,4],[1,5]])
    desired = numpy.take(desired, numpy.argsort(numpy.concatenate(self.desired_indices), axis=0), axis=0)
    self.assertAllAlmostEqual(actual, desired)

  def test_basis(self):
    with _builtin_warnings.catch_warnings():
      _builtin_warnings.simplefilter('ignore', category=evaluable.ExpensiveEvaluationWarning)
      self.assertAllAlmostEqual(self.sample(self.sample.basis()).as_evaluable_array.eval(), numpy.eye(11))

class Special(TestCase):

  def test_add_different_spaces(self):
    class Dummy(Sample): pass
    with self.assertRaisesRegex(ValueError, '^Cannot add .* different spaces.$'):
      Dummy(('a','b'), 1, 1, 1) + Dummy(('b','c'), 1, 1, 1)

  def test_mul_common_spaces(self):
    class Dummy(Sample): pass
    with self.assertRaisesRegex(ValueError, '^Cannot multiply .* common spaces.$'):
      Dummy(('a','b'), 1, 1, 1) * Dummy(('b','c'), 1, 1, 1)

class rectilinear(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([2,1])
    self.bezier2 = self.domain.sample('bezier', 2)
    self.bezier3 = self.domain.sample('bezier', 3)
    self.gauss2 = self.domain.sample('gauss', 2)

  def test_integrate(self):
    area = self.gauss2.integrate(1)
    self.assertLess(abs(area-2), 1e-15)

  def test_integral(self):
    area = self.gauss2.integral(function.asarray(1)).eval()
    self.assertLess(abs(area-2), 1e-15)

  def test_eval(self):
    x = self.bezier3.eval(self.geom)
    self.assertEqual(x.shape, (self.bezier3.npoints,)+self.geom.shape)

  def test_tri(self):
    self.assertEqual(len(self.bezier2.tri), 4)
    self.assertEqual(len(self.bezier3.tri), 16)

  def test_hull(self):
    self.assertEqual(len(self.bezier2.hull), 8)
    self.assertEqual(len(self.bezier3.hull), 16)

  def test_subset(self):
    subset1 = self.bezier2.subset(numpy.eye(8)[0])
    subset2 = self.bezier2.subset(numpy.eye(8)[1])
    self.assertEqual(subset1.npoints, 4)
    self.assertEqual(subset2.npoints, 4)
    self.assertEqual(subset1, subset2)

  def test_asfunction(self):
    func = self.geom[0]**2 - self.geom[1]**2
    values = self.gauss2.eval(func)
    sampled = self.gauss2.asfunction(values)
    with self.assertRaises(evaluable.EvaluationError):
      self.bezier2.eval(sampled)
    self.assertAllEqual(self.gauss2.eval(sampled), values)
    arg = function.Argument('dofs', [2,3])
    self.assertTrue(evaluable.iszero(evaluable.asarray(self.gauss2(function.derivative(sampled, arg)))))

class integral(TestCase):

  def setUp(self):
    super().setUp()
    self.ns = function.Namespace()
    self.topo, self.ns.x = mesh.rectilinear([5])
    self.ns.basis = self.topo.basis('std', degree=1)
    self.ns.v = 'basis_n ?lhs_n'
    self.lhs = numpy.sin(numpy.arange(len(self.ns.basis)))

  def test_eval(self):
    self.assertAllAlmostEqual(
      self.topo.integrate('basis_n d:x' @ self.ns, degree=2),
      self.topo.integral('basis_n d:x' @ self.ns, degree=2).eval(),
      places=15)

  def test_args(self):
    self.assertAlmostEqual(
      self.topo.integrate('v d:x' @ self.ns, degree=2, arguments=dict(lhs=self.lhs)),
      self.topo.integral('v d:x' @ self.ns, degree=2).eval(lhs=self.lhs),
      places=15)

  def test_derivative(self):
    self.assertAllAlmostEqual(
      self.topo.integrate('2 basis_n v d:x' @ self.ns, degree=2, arguments=dict(lhs=self.lhs)),
      self.topo.integral('v^2 d:x' @ self.ns, degree=2).derivative('lhs').eval(lhs=self.lhs),
      places=15)

  def test_transpose(self):
    with self.assertWarns(evaluable.ExpensiveEvaluationWarning):
      self.assertAllAlmostEqual(
        self.topo.integrate(self.ns.eval_nm('basis_n (basis_m + 1_m) d:x'), degree=2).export('dense').T,
        self.topo.integral(self.ns.eval_nm('basis_n (basis_m + 1_m) d:x'), degree=2).T.eval().export('dense'),
        places=15)

  def test_empty(self):
    shape = 2, 3
    empty = function.zeros(shape, float)
    array = empty.eval().export('dense')
    self.assertAllEqual(array, numpy.zeros((2,3)))
