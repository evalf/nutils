from nutils import element, mesh, topology, function, transformseq, evaluable, transform, _util as util
from nutils.testing import TestCase, parametrize
from nutils.elementseq import References
import numpy
import copy
import sys
import pickle
import subprocess
import base64
import itertools
import os
import unittest


def as_rounded_list(data):
    return numpy.round(data, 5).tolist()


def pairwise(items):
    return [[i, j] for i, j in zip(items[:-1], items[1:])]


def subdiv(V):
    V = iter(V)
    items = [next(V)]
    for v in V:
        items += [(items[-1] + v) / 2, v]
    return items


class CommonAssertions:

    def assertVertices(self, topo, desired_coords):
        assert len(desired_coords) == len(topo)
        bezier = topo.sample('bezier', 2)
        actual_coords_flat = as_rounded_list(bezier.eval(self.geom))
        for ielem, desired_elem_coords in enumerate(desired_coords):
            actual_elem_coords = numpy.take(actual_coords_flat, bezier.getindex(ielem), axis=0)
            self.assertEqual(actual_elem_coords.tolist(), desired_elem_coords)

    def assertTake(self, topo, selection):
        # Like `assertVertices` but using `self.desired_vertices[selection]`.
        selection = numpy.asarray(selection, dtype=int).ravel()
        verts = [self.desired_vertices[i] for i in selection]
        self.assertVertices(topo, verts)

    def assertCompressed(self, topo, mask):
        # Like `assertVertices` but using `self.desired_vertices[mask]`.
        mask = numpy.asarray(mask, dtype=bool).ravel()
        assert mask.size == self.desired_nelems
        verts = [verts for select, verts in zip(mask, self.desired_vertices) if select]
        self.assertVertices(topo, verts)

    def assertUnorderedVertices(self, topo, desired_coords):
        assert len(desired_coords) == len(topo)
        bezier = topo.sample('bezier', 2)
        actual_coords_flat = as_rounded_list(bezier.eval(self.geom))
        actual_coords = []
        for ielem, desired_elem_coords in enumerate(desired_coords):
            actual_elem_coords = numpy.take(actual_coords_flat, bezier.getindex(ielem), axis=0)
            actual_coords.append(actual_elem_coords.tolist())
        self.assertEqual(sorted(actual_coords), sorted(desired_coords))

    def assertEquivalent(self, topo1, topo2):
        self.assertEqual(type(topo1), type(topo2))
        self.assertEqual(topo1.spaces, topo2.spaces)
        self.assertEqual(topo1.space_dims, topo2.space_dims)
        self.assertEqual(topo1.references, topo2.references)


class CommonTests(CommonAssertions):

    def test_empty_like(self):
        empty = self.topo.empty_like()
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.spaces, self.desired_spaces)
        self.assertEqual(empty.space_dims, self.desired_space_dims)
        self.assertEqual(empty.ndims, self.desired_ndims)

    def test_spaces(self):
        self.assertEqual(self.topo.spaces, self.desired_spaces)

    def test_space_dims(self):
        self.assertEqual(self.topo.space_dims, self.desired_space_dims)

    def test_ndims(self):
        self.assertEqual(self.topo.ndims, self.desired_ndims)

    def test_len(self):
        self.assertEqual(len(self.topo), self.desired_nelems)

    def test_references(self):
        assert len(self.desired_references) == self.desired_nelems
        self.assertSequenceEqual(self.topo.references, self.desired_references)

    def test_elements(self):
        # This sort of tests the `self.topo.transforms` by evaluating `self.geom`
        # and comparing with `self.desired_vertices`.
        self.assertVertices(self.topo, self.desired_vertices)

    def test_get_groups_nonexistent(self):
        self.assertFalse(self.topo.get_groups('nonexistent'))

    def test_getitem_empty(self):
        with self.assertRaises(KeyError):
            self.topo['nonexistent']

    def test_take(self):
        self.assertFalse(self.topo.take([]))
        for ielem in range(self.desired_nelems):
            self.assertTake(self.topo.take([ielem]), [ielem])
            self.assertTake(self.topo[[ielem]], [ielem])

    def test_take_invalid_indices(self):
        with self.assertRaisesRegex(ValueError, '^expected a one-dimensional array$'):
            self.topo.take(numpy.array(0))
        with self.assertRaisesRegex(ValueError, '^expected a one-dimensional array$'):
            self.topo.take(numpy.array([[0, 1], [2, 3]]))

    def test_compress(self):
        self.assertFalse(self.topo.compress([False]*self.desired_nelems))
        for ielem in range(self.desired_nelems):
            self.assertTake(self.topo.compress([i == ielem for i in range(self.desired_nelems)]), [ielem])
            self.assertTake(self.topo[numpy.arange(self.desired_nelems) == ielem], [ielem])

    def test_slice_invalid_dim(self):
        with self.assertRaisesRegex(IndexError, '^dimension index out of range$'):
            self.topo.slice(slice(None), self.desired_ndims)

    def test_f_index(self):
        self.assertEqual(self.topo.sample('gauss', 0).eval(self.topo.f_index).tolist(), list(range(self.desired_nelems)))

    def test_unit_integral(self):
        self.assertAlmostEqual(self.topo.integral(function.J(self.geom), degree=0).eval(), sum(self.desired_volumes))

    def test_unit_integrate(self):
        self.assertAlmostEqual(self.topo.integrate(function.J(self.geom), degree=0), sum(self.desired_volumes))

    def test_unit_integrate_elementwise(self):
        self.assertEqual(as_rounded_list(self.topo.integrate_elementwise(function.J(self.geom), degree=0)), self.desired_volumes)

    def test_refine_spaces_none(self):
        self.assertEquivalent(self.topo.refine_spaces([]), self.topo)

    def test_invalid_intersections(self):
        with self.assertRaises(ValueError):
            self.topo & topology.Topology.empty(tuple('other' + space for space in self.desired_spaces), self.desired_space_dims, self.desired_ndims)
        with self.assertRaises(ValueError):
            self.topo & topology.Topology.empty(self.desired_spaces, tuple(dim + 1 for dim in self.desired_space_dims), self.desired_ndims)
        with self.assertRaises(ValueError):
            self.topo & topology.Topology.empty(self.desired_spaces, self.desired_space_dims, self.desired_ndims + 1)

    def test_invalid_unions(self):
        with self.assertRaises(ValueError):
            self.topo | topology.Topology.empty(tuple('other' + space for space in self.desired_spaces), self.desired_space_dims, self.desired_ndims)
        with self.assertRaises(ValueError):
            self.topo | topology.Topology.empty(self.desired_spaces, tuple(dim + 1 for dim in self.desired_space_dims), self.desired_ndims)
        with self.assertRaises(ValueError):
            self.topo | topology.Topology.empty(self.desired_spaces, self.desired_space_dims, self.desired_ndims + 1)

    def test_select(self):
        if self.desired_ndims == 0:
            return
        if self.desired_nelems:
            centers = numpy.stack([numpy.mean(v, axis=0) for v in self.desired_vertices])
            center = numpy.mean(centers, axis=0)
            center = centers[numpy.argmin(((centers - center)**2).sum(1).round(5))]
        else:
            center = numpy.zeros(self.desired_ndims)
        direction = 1 / (self.desired_ndims - numpy.arange(self.desired_ndims))
        for i in range(self.desired_ndims):
            desired_selection, = numpy.where([(numpy.sum((numpy.array(v) - center) * direction, axis=1) > 0).any() for v in self.desired_vertices])
            desired_vertices = [self.desired_vertices[i] for i in desired_selection]
            self.assertVertices(self.topo.select(((self.geom - center) * direction).sum()), desired_vertices)
            direction = numpy.roll(direction, shift=1)


class ConformingTests:

    @property
    def edge_map(self):
        # Mapping from edge vertices to pairs of element and edge indices based on
        # `self.desired_references` and `self.desired_vertices`.
        assert len(self.desired_references) == len(self.desired_vertices) == self.desired_nelems
        edge_map = {}
        for ielem, (ref, verts) in enumerate(zip(self.desired_references, self.desired_vertices)):
            local_verts = as_rounded_list(ref.vertices)
            for iedge, (trans, edge) in enumerate(ref.edges):
                local_edge_verts = as_rounded_list(trans.apply(edge.vertices))
                edge_verts = tuple(tuple(verts[local_verts.index(v)]) for v in local_edge_verts)
                edge_map.setdefault(edge_verts, set()).add((ielem, iedge))
        return edge_map

    @property
    def connectivity(self):
        assert len(self.desired_references) == self.desired_nelems
        connectivity = [[-1] * ref.nedges for ref in self.desired_references]
        for sides in self.edge_map.values():
            assert len(sides) <= 2
            if len(sides) == 2:
                (ielem1, iedge1), (ielem2, iedge2) = sides
                connectivity[ielem1][iedge1] = ielem2
                connectivity[ielem2][iedge2] = ielem1
        return connectivity

    def test_connectivity(self):
        self.assertEqual(list(map(list, self.topo.connectivity)), self.connectivity)

    def test_boundary_all_spaces(self):
        boundary_vertices = [list(map(list, verts)) for verts, sides in self.edge_map.items() if len(sides) == 1]
        self.assertUnorderedVertices(self.topo.boundary, boundary_vertices)

    def test_interfaces_all_spaces(self):
        interface_vertices = [list(map(list, verts)) for verts, sides in self.edge_map.items() if len(sides) == 2]
        self.assertUnorderedVertices(self.topo.interfaces, interface_vertices)

    def test_basis_std_degree1(self):
        basis = self.topo.basis('std', degree=1)
        values, verts = self.topo.sample('bezier', 2).eval([basis, self.geom])
        dofs_to_verts = {}
        verts_to_dofs = {}
        for val, vert in zip(map(as_rounded_list, values), (tuple(as_rounded_list(v)) for v in verts)):
            self.assertCountEqual(val, [1]+[0]*(len(val)-1))
            dof = val.index(1)
            if dof in dofs_to_verts:
                self.assertEqual(dofs_to_verts[dof], vert)
            else:
                dofs_to_verts[dof] = vert
            if vert in verts_to_dofs:
                self.assertEqual(verts_to_dofs[vert], dof)
            else:
                verts_to_dofs[vert] = dof
        self.assertEqual(sorted(dofs_to_verts), list(range(len(basis))))
        self.assertEqual(sorted(verts_to_dofs), sorted(set(tuple(v) for e in self.desired_vertices for v in e)))


class NewTopologyRefine(TestCase, CommonAssertions):
    # Tests for default implementations of `Topology.refine_*`.

    def setUp(self):
        super().setUp()

        class TestTopo(topology.Topology):
            def __init__(self, real):
                self.real = real
                super().__init__(real.spaces, real.space_dims, real.references)

            def refine_spaces(self, spaces):
                return TestTopo(self.real.refine_spaces(spaces))

            def sample(self, ischeme, degree):
                return self.real.sample(ischeme, degree)
        topo, self.geom = mesh.newrectilinear([4, 2], spaces=['X', 'Y'])
        self.topo = TestTopo(topo)

    @staticmethod
    def mkverts(XX, YY):
        return [[[x, y] for x in X for y in Y] for X in pairwise(XX) for Y in pairwise(YY)]

    def test_refine_count_iter(self):
        refine_iter = iter(self.topo.refine_iter)
        X, Y = range(5), range(3)
        for i in range(3):
            desired = self.mkverts(X, Y)
            self.assertVertices(self.topo.refine_count(i), desired)
            self.assertVertices(self.topo.refine(i), desired)
            self.assertVertices(next(refine_iter), desired)
            X, Y = subdiv(X), subdiv(Y)

    def test_refine_spaces(self):
        # We only test `Topology.refine` because `Topology.refine_spaces` is
        # abstract.
        self.assertVertices(self.topo.refine(['X']), self.mkverts(subdiv(range(5)), range(3)))
        self.assertVertices(self.topo.refine(['Y']), self.mkverts(range(5), subdiv(range(3))))

    def test_refine_spaces_count(self):
        self.assertVertices(self.topo.refine(dict(X=1, Y=2)), self.mkverts(subdiv(range(5)), subdiv(subdiv(range(3)))))

    def test_refine_count_negative(self):
        with self.assertRaisesRegex(ValueError, '^Negative counts are invalid.$'):
            self.topo.refine_count(-1)
        with self.assertRaisesRegex(ValueError, '^Negative counts are invalid.$'):
            self.topo.refine(-1)
        with self.assertRaisesRegex(ValueError, '^Negative counts are invalid.$'):
            self.topo.refine_spaces_count(dict(X=-1))

    def test_refine_unknown_space(self):
        with self.assertRaisesRegex(ValueError, '^This topology does not have space Z.$'):
            self.topo.refine_spaces(['Z'])


class NewTopologyTake(TestCase, CommonAssertions):
    # Tests for default implementations of `Topology.take` and
    # `Topology.compress`.

    def setUp(self):
        super().setUp()

        class TestTopo(topology.Topology):
            def __init__(self, real):
                self.real = real
                super().__init__(real.spaces, real.space_dims, real.references)

            def sample(self, ischeme, degree):
                return self.real.sample(ischeme, degree)
        topo, self.geom = mesh.newrectilinear([4, 2], spaces=['X', 'Y'])
        self.topo = TestTopo(topo)
        self.desired_vertices = [[[x, y] for x in X for y in Y] for X in pairwise(range(5)) for Y in pairwise(range(3))]

    def test_take(self):
        self.assertTake(self.topo.take([1, 3, 4]), [1, 3, 4])
        self.assertTake(self.topo.take(numpy.array([1, 3, 4])), [1, 3, 4])

    def test_getitem(self):
        self.assertTake(self.topo[[1, 3, 4]], [1, 3, 4])
        self.assertTake(self.topo[numpy.array([1, 3, 4])], [1, 3, 4])

    def test_take_empty(self):
        self.assertTake(self.topo.take([]), [])
        self.assertTake(self.topo.take(numpy.array([], dtype=int)), [])
        # Test whether an empty float array is allowed.
        self.assertTake(self.topo.take(numpy.array([])), [])

    def test_take_invalid_array(self):
        with self.assertRaisesRegex(ValueError, '^expected a one-dimensional array$'):
            self.topo.take(numpy.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            self.topo.take(numpy.array([1, 2], dtype=float))

    def test_compress(self):
        self.assertTake(self.topo.compress([False, True, False, True, True, False, False, False]), [1, 3, 4])

    def test_compress_invalid_array(self):
        with self.assertRaisesRegex(ValueError, '^expected a one-dimensional array$'):
            self.topo.compress([[False, True]]*4)
        with self.assertRaisesRegex(ValueError, '^length of mask does not match number of elements$'):
            self.topo.compress([False])


class NewTopologySlice(TestCase, CommonAssertions):
    # Tests for default implementation of `Topology.__getitem__`.

    def setUp(self):
        super().setUp()
        self.topo, self.geom = mesh.newrectilinear([4, 2], spaces=['X', 'Y'])
        self.desired_vertices = [[[x, y] for x in X for y in Y] for X in pairwise(range(5)) for Y in pairwise(range(3))]
        self.idx = numpy.arange(8).reshape(4, 2)

    def test_slice(self):
        self.assertTake(self.topo.slice(slice(None), 0), self.idx)
        self.assertTake(self.topo.slice(slice(None), 1), self.idx)
        self.assertTake(self.topo.slice(slice(2, None), 0), self.idx[2:])
        self.assertTake(self.topo.slice(slice(None, 1), 1), self.idx[:, :1])

    def test_getitem(self):
        self.assertTake(self.topo[:], self.idx)
        self.assertTake(self.topo[:, :], self.idx)
        self.assertTake(self.topo[..., :], self.idx)
        self.assertTake(self.topo[..., :, :], self.idx)
        self.assertTake(self.topo[:, ..., :], self.idx)
        self.assertTake(self.topo[:, :, ...], self.idx)
        self.assertTake(self.topo[2:], self.idx[2:])
        self.assertTake(self.topo[2:, 1:], self.idx[2:, 1:])

    def test_getitem_multiple_ellipsis(self):
        with self.assertRaisesRegex(Exception, '^only one ellipsis is allowed$'):
            self.topo[..., :, ...]

    def test_getitem_too_many_indices(self):
        with self.assertRaisesRegex(Exception, '^too many indices'):
            self.topo[:, :, :]

    def test_slice_invalid_dimensions(self):
        with self.assertRaises(IndexError):
            self.topo.slice(slice(None), -1)
        with self.assertRaises(IndexError):
            self.topo.slice(slice(None), 2)


class NewTopologyBoundaryInterfaces(TestCase):

    def setUp(self):
        super().setUp()
        self.topo1, self.geom = mesh.line([0, 1, 2], space='X')
        self.topo0 = self.topo1.boundary_spaces(['X'])

    def test_boundary_0d(self):
        with self.assertRaisesRegex(ValueError, '^A 0D topology has no boundary.$'):
            self.topo0.boundary_spaces(['X'])

    def test_interfaces_0d(self):
        with self.assertRaisesRegex(ValueError, '^A 0D topology has no interfaces.$'):
            self.topo0.interfaces_spaces(['X'])

    def test_boundary_empty_spaces(self):
        with self.assertRaisesRegex(ValueError, '^A 0D topology has no boundary.$'):
            self.topo0.boundary_spaces([])

    def test_interfaces_empty_spaces(self):
        with self.assertRaisesRegex(ValueError, '^A 0D topology has no interfaces.$'):
            self.topo0.interfaces_spaces([])

    def test_boundary_unknown_space(self):
        with self.assertRaisesRegex(ValueError, '^This topology does not have space Y.$'):
            self.topo1.boundary_spaces(['Y'])

    def test_interfaces_unknown_space(self):
        with self.assertRaisesRegex(ValueError, '^This topology does not have space Y.$'):
            self.topo1.interfaces_spaces(['Y'])

    def test_basis_0d(self):
        basis = self.topo0.basis('std', degree=0)
        sampled = self.topo0.sample('bezier', 2).eval(basis)
        self.assertEqual(as_rounded_list(sampled), [[1.], [1.]])


class NewEmpty(TestCase, CommonTests, ConformingTests):

    def setUp(self):
        super().setUp()
        self.desired_spaces = 'a', 'b'
        self.desired_space_dims = 1, 2
        self.desired_ndims = 3
        self.topo = topology.Topology.empty(self.desired_spaces, self.desired_space_dims, self.desired_ndims)
        self.geom = numpy.concatenate([function.rootcoords(space, dim) for space, dim in zip(self.desired_spaces, self.desired_space_dims)])
        self.desired_nelems = 0
        self.desired_volumes = []
        self.desired_references = []
        self.desired_vertices = []

    def test_opposite(self):
        self.assertEqual(len(~self.topo), 0)

    def test_intersection(self):
        atrans = transformseq.IndexTransforms(1, 1, 0)
        btrans = transformseq.IndexTransforms(2, 1, 1)
        other = topology.SimplexTopology('a', numpy.array([[0, 1]]), atrans, atrans) * topology.SimplexTopology('b', numpy.array([[0, 1, 2]]), btrans, btrans)
        self.assertEquivalent(self.topo & other, self.topo)
        self.assertEquivalent(other & self.topo, self.topo)

    def test_union(self):
        atrans = transformseq.IndexTransforms(1, 1, 0)
        btrans = transformseq.IndexTransforms(2, 1, 1)
        other = topology.SimplexTopology('a', numpy.array([[0, 1]]), atrans, atrans) * topology.SimplexTopology('b', numpy.array([[0, 1, 2]]), btrans, btrans)
        self.assertEqual(self.topo | other, other)
        self.assertEqual(other | self.topo, other)

    def test_indicator(self):
        self.assertEqual(self.topo.indicator('group').shape, ())

    def test_f_coords(self):
        self.assertEqual(self.topo.f_coords.shape, (3,))


class NewDisjointUnion(TestCase, CommonTests, ConformingTests):

    def setUp(self):
        super().setUp()
        topo, self.geom = mesh.newrectilinear([8, 3], spaces='XY')
        self.topo = topology.Topology.disjoint_union(topo.slice(slice(0, 3), 0), topo.slice(slice(4, 8), 0).slice(slice(0, 2), 1))
        self.desired_spaces = 'X', 'Y'
        self.desired_space_dims = 1, 1
        self.desired_ndims = 2
        self.desired_nelems = 17
        self.desired_volumes = [1] * 17
        self.desired_references = [element.LineReference()**2]*17
        self.desired_vertices = self.mkverts(pairwise(range(4)), pairwise(range(4))) + self.mkverts(pairwise(range(4, 9)), pairwise(range(3)))

    @staticmethod
    def mkverts(XX, YY):
        return [[[x, y] for x in X for y in Y] for X in XX for Y in YY]

    def test_refine(self):
        self.assertVertices(self.topo.refine_spaces([]), self.mkverts(pairwise(range(4)), pairwise(range(4))) + self.mkverts(pairwise(range(4, 9)), pairwise(range(3))))
        self.assertVertices(self.topo.refine_spaces(['X']), self.mkverts(pairwise(subdiv(range(4))), pairwise(range(4))) + self.mkverts(pairwise(subdiv(range(4, 9))), pairwise(range(3))))
        self.assertVertices(self.topo.refine_spaces(['Y']), self.mkverts(pairwise(range(4)), pairwise(subdiv(range(4)))) + self.mkverts(pairwise(range(4, 9)), pairwise(subdiv(range(3)))))
        self.assertVertices(self.topo.refine_spaces(['X', 'Y']), self.mkverts(pairwise(subdiv(range(4))), pairwise(subdiv(range(4)))) + self.mkverts(pairwise(subdiv(range(4, 9))), pairwise(subdiv(range(3)))))

    def test_take(self):
        self.assertVertices(self.topo.take([0]), self.mkverts([[0, 1]], [[0, 1]]))
        self.assertVertices(self.topo.take([9, 10]), self.mkverts([[4, 5]], pairwise([0, 1, 2])))
        self.assertVertices(self.topo.take([0, 9, 10]), self.mkverts([[0, 1]], [[0, 1]]) + self.mkverts([[4, 5]], pairwise([0, 1, 2])))

    def test_slice_unstructured(self):
        for i in range(self.desired_ndims):
            with self.assertRaisesRegex(ValueError, '^cannot slice'):
                self.topo.slice(slice(None), i)

    def test_f_index(self):
        with self.assertRaises(NotImplementedError):
            self.topo.f_index

    def test_basis_std_degree1(self):
        with self.assertRaises(Exception):
            self.topo.basis('std', degree=1)

    def test_trim(self):
        topo, x = mesh.line([0, 1, 2, 3], space='X')
        topo = topology.Topology.disjoint_union(topo.slice(slice(0, 1), 0), topo.slice(slice(2, 3), 0))
        self.assertEqual(as_rounded_list(topo.trim(x-0.5, maxrefine=0).volume(x[None])), 1.5)
        self.assertEqual(as_rounded_list(topo.trim(x-2.5, maxrefine=0).volume(x[None])), 0.5)
        self.assertEqual(as_rounded_list(topo.trim(0.5-x, maxrefine=0).volume(x[None])), 0.5)


class NewMul(TestCase, CommonTests, ConformingTests):

    def setUp(self):
        super().setUp()
        self.topo1, self.x = mesh.line([0, 1, 2], bnames=('a', 'b'), space='X')
        self.topo2, self.y = mesh.line([0, 1, 2, 3], bnames=('c', 'd'), space='Y')
        self.topo = self.topo1 * self.topo2
        self.geom = numpy.stack([self.x, self.y])
        self.desired_spaces = 'X', 'Y'
        self.desired_space_dims = 1, 1
        self.desired_ndims = 2
        self.desired_nelems = 6
        self.desired_volumes = [1]*6
        self.desired_references = [element.LineReference()**2]*6
        self.desired_vertices = self.mkverts(pairwise(range(3)), pairwise(range(4)))

    @staticmethod
    def mkverts(XX, YY):
        return [[[x, y] for x in X for y in Y] for X in XX for Y in YY]

    def test_f_coords(self):
        self.assertEqual(as_rounded_list(self.topo.sample('bezier', 2).eval(self.topo.f_coords)), ([[0., 0.], [0., 1.]]*3+[[1., 0.], [1., 1.]]*3)*2)

    def test_refine_spaces(self):
        self.assertVertices(self.topo.refine_spaces([]), self.mkverts(pairwise(range(3)), pairwise(range(4))))
        self.assertVertices(self.topo.refine_spaces(['X']), self.mkverts(pairwise(subdiv(range(3))), pairwise(range(4))))
        self.assertVertices(self.topo.refine_spaces(['Y']), self.mkverts(pairwise(range(3)), pairwise(subdiv(range(4)))))
        self.assertVertices(self.topo.refine_spaces(['X', 'Y']), self.mkverts(pairwise(subdiv(range(3))), pairwise(subdiv(range(4)))))

    def test_boundary_spaces(self):
        bX = self.mkverts([[0], [2]], pairwise(range(4)))
        bY = self.mkverts(pairwise(range(3)), [[0], [3]])
        self.assertVertices(self.topo.boundary_spaces(['X']), bX)
        self.assertVertices(self.topo.boundary_spaces(['Y']), bY)
        self.assertVertices(self.topo.boundary_spaces(['X', 'Y']), bY+bX)

    def test_boundary_0d(self):
        topo0 = mesh.line([0, 1, 2], space='Z')[0].boundary
        for topo in self.topo * topo0, topo0 * self.topo:
            with self.assertRaisesRegex(ValueError, '^A 0D topology has no boundary.$'):
                topo.boundary_spaces('Z')

    def test_interfaces_spaces(self):
        iX = self.mkverts([[1]], pairwise(range(4)))
        iY = self.mkverts(pairwise(range(3)), [[1], [2]])
        self.assertVertices(self.topo.interfaces_spaces(['X']), iX)
        self.assertVertices(self.topo.interfaces_spaces(['Y']), iY)
        self.assertVertices(self.topo.interfaces_spaces(['X', 'Y']), iY+iX)

    def test_interfaces_0d(self):
        topo0 = mesh.line([0, 1, 2], space='Z')[0].boundary
        for topo in self.topo * topo0, topo0 * self.topo:
            with self.assertRaisesRegex(ValueError, '^A 0D topology has no interfaces.$'):
                topo.interfaces_spaces('Z')

    def test_slice(self):
        self.assertVertices(self.topo.slice(slice(0, 1), 0), self.mkverts([[0, 1]], pairwise(range(4))))
        self.assertVertices(self.topo.slice(slice(1, 3), 1), self.mkverts(pairwise(range(3)), pairwise(range(1, 4))))

    def test_get_groups(self):
        topo = self.topo1.withsubdomain(e=self.topo1[:1], g=self.topo1[:1]) * self.topo2.withsubdomain(f=self.topo2[:1], g=self.topo2[:1], h=self.topo2[2:])
        self.assertVertices(topo.get_groups('e'), self.mkverts([[0, 1]], pairwise(range(0, 4))))
        self.assertVertices(topo.get_groups('f'), self.mkverts(pairwise(range(3)), pairwise(range(0, 2))))
        self.assertVertices(topo.get_groups('h'), self.mkverts(pairwise(range(3)), pairwise(range(2, 4))))
        self.assertVertices(topo.get_groups('f', 'h'), self.mkverts(pairwise(range(3)), [[0, 1], [2, 3]]))
        with self.assertRaises(NotImplementedError):
            topo.get_groups('g')

    def test_indicator(self):
        topo = self.topo1.withsubdomain(e=self.topo1[:1], g=self.topo1[:1]) * self.topo2.withsubdomain(f=self.topo2[:1], g=self.topo2[:1], h=self.topo2[2:])
        self.assertEqual(as_rounded_list(self.topo.sample('gauss', 0).eval(topo.indicator('e'))), [1, 1, 1, 0, 0, 0])
        self.assertEqual(as_rounded_list(self.topo.sample('gauss', 0).eval(topo.indicator('f'))), [1, 0, 0, 1, 0, 0])
        self.assertEqual(as_rounded_list(self.topo.sample('gauss', 0).eval(topo.indicator('h'))), [0, 0, 1, 0, 0, 1])
        self.assertEqual(as_rounded_list(self.topo.sample('gauss', 0).eval(topo.indicator('f,h'))), [1, 0, 1, 1, 0, 1])
        self.assertEqual(as_rounded_list(self.topo.sample('gauss', 0).eval(topo.indicator('nonexistent'))), [0, 0, 0, 0, 0, 0])
        with self.assertRaises(NotImplementedError):
            topo.indicator('g')

    def test_common_spaces(self):
        with self.assertRaisesRegex(ValueError, '^Cannot multiply'):
            topology.Topology.empty(['X'], [1], 1) * topology.Topology.empty(['X', 'Y'], [1, 2], 3)

    def test_basis(self):
        self.assertEqual(len(self.topo.basis('spline', degree=1)), 3*4)
        self.assertEqual(len(self.topo.basis('spline', degree=1, periodic=[0])), 2*4)
        self.assertEqual(len(self.topo.basis('spline', degree=1, periodic=[1])), 3*3)
        self.assertEqual(len(self.topo.basis('spline', degree=1, periodic=[0, 1])), 2*3)
        self.assertEqual(len(self.topo.basis('spline', degree=[0, 1])), 2*4)
        self.assertEqual(len(self.topo.basis('spline', degree=[1, 0])), 3*3)
        self.assertEqual(len(self.topo.basis('spline', continuity=-2, degree=1)), (2*2)*(2*3))
        self.assertEqual(len(self.topo.basis('spline', continuity=[-2, -1], degree=1)), (2*2)*4)
        self.assertEqual(len(self.topo.basis('spline', continuity=[-1, -2], degree=1)), 3*(2*3))
        self.assertEqual(len(self.topo.basis('spline', knotmultiplicities=[None, [1, 2, 1, 1]], degree=1)), 3*5)
        self.assertEqual(len(self.topo.basis('spline', knotmultiplicities=[[1, 2, 1], [1, 2, 1, 1]], degree=1)), 4*5)
        with self.assertRaisesRegex(ValueError, '^argument `degree` must have length'):
            self.topo.basis('spline', degree=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, '^argument `degree` must be'):
            self.topo.basis('spline', degree='a')
        with self.assertRaisesRegex(ValueError, '^argument `continuity` must have length'):
            self.topo.basis('spline', degree=1, continuity=[-1, -2, -1])
        with self.assertRaisesRegex(ValueError, '^argument `continuity` must be'):
            self.topo.basis('spline', degree=1, continuity='a')
        with self.assertRaisesRegex(ValueError, '^argument `periodic` must be'):
            self.topo.basis('spline', degree=1, periodic=['a', 'b'])
        with self.assertRaisesRegex(ValueError, '^argument `knotmultiplicities` must have length'):
            self.topo.basis('spline', degree=1, knotmultiplicities=[[1, 1, 1], [1, 1, 1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            self.topo.basis('spline', degree=1, knotmultiplicities=['a', 'b'])


class NewWithGroupAliases(TestCase, CommonTests, ConformingTests):

    def setUp(self):
        super().setUp()
        self.topo1, self.x = mesh.line([0, 1, 2], bnames=('a', 'b'), space='X')
        self.topo1 = self.topo1.withsubdomain(e=self.topo1[:1])
        self.topo2, self.y = mesh.line([0, 1, 2, 3], bnames=('c', 'd'), space='Y')
        self.topo2 = self.topo2.withsubdomain(f=self.topo2[:1], g=self.topo2[2:])
        self.topo = (self.topo1 * self.topo2).withgroups(vgroups=dict(ealias='e', falias='f', galias='g', fgalias='f,g'))
        self.geom = numpy.stack([self.x, self.y])
        self.desired_spaces = 'X', 'Y'
        self.desired_space_dims = 1, 1
        self.desired_ndims = 2
        self.desired_nelems = 6
        self.desired_volumes = [1]*6
        self.desired_references = [element.LineReference()**2]*6
        self.desired_vertices = [[[x, y] for x in X for y in Y] for X in pairwise(range(3)) for Y in pairwise(range(4))]
        self.masks = dict(e=[1, 1, 1, 0, 0, 0], f=[1, 0, 0, 1, 0, 0], g=[0, 0, 1, 0, 0, 1])

    def test_slice(self):
        self.assertCompressed(self.topo.slice(slice(0, 1), 0), [1, 1, 1, 0, 0, 0])

    def test_indicator(self):
        e, f, g, fg = self.topo.sample('gauss', 0).eval(list(self.topo.indicator(g+'alias') for g in ('e', 'f', 'g', 'fg')))
        self.assertEqual(as_rounded_list(e), [1, 1, 1, 0, 0, 0])
        self.assertEqual(as_rounded_list(f), [1, 0, 0, 1, 0, 0])
        self.assertEqual(as_rounded_list(g), [0, 0, 1, 0, 0, 1])
        self.assertEqual(as_rounded_list(fg), [1, 0, 1, 1, 0, 1])


class TopologyAssertions:

    def assertConnectivity(self, domain, geom):
        boundary = domain.boundary
        interfaces = domain.interfaces
        bmask = numpy.zeros(len(boundary), dtype=int)
        imask = numpy.zeros(len(interfaces), dtype=int)
        coordinates = evaluable.Points(evaluable.NPoints(), evaluable.constant(boundary.ndims))
        edges = domain.transforms.edges(domain.references)
        iedge = evaluable.Argument('_iedge', (), int)
        lowered_geom = geom.lower(function.LowerArgs.for_space(domain.space, (edges,), iedge, coordinates)).simplified
        for ielem, ioppelems in enumerate(domain.connectivity):
            for iedge, ioppelem in enumerate(ioppelems):
                etrans, eref = domain.references[ielem].edges[iedge]
                trans = domain.transforms[ielem] + (etrans,)
                if ioppelem == -1:
                    index = boundary.transforms.index(trans)
                    bmask[index] += 1
                else:
                    ioppedge = util.index(domain.connectivity[ioppelem], ielem)
                    oppetrans, opperef = domain.references[ioppelem].edges[ioppedge]
                    opptrans = domain.transforms[ioppelem] + (oppetrans,)
                    try:
                        index = interfaces.transforms.index(trans)
                    except ValueError:
                        index = interfaces.transforms.index(opptrans)
                        self.assertEqual(interfaces.opposites[index], trans)
                    else:
                        self.assertEqual(interfaces.opposites[index], opptrans)
                    imask[index] += 1
                    self.assertEqual(eref, opperef)
                    points = eref.getpoints('gauss', 2)
                    a0 = lowered_geom.eval(_iedge=edges.index(trans), _points=points)
                    a1 = lowered_geom.eval(_iedge=edges.index(opptrans), _points=points)
                    numpy.testing.assert_array_almost_equal(a0, a1)
        self.assertTrue(numpy.equal(bmask, 1).all())
        self.assertTrue(numpy.equal(imask, 2).all())

    def assertBoundaries(self, domain, geom):
        # Test ∫_Ω f_,i = ∫_∂Ω f n_i.
        f = ((0.5 - geom)**2).sum(axis=0)
        lhs = domain.integrate(f.grad(geom)*function.J(geom), ischeme='gauss2')
        rhs = domain.boundary.integrate(f*function.normal(geom)*function.J(geom), ischeme='gauss2')
        numpy.testing.assert_array_almost_equal(lhs, rhs)

    def assertInterfaces(self, domain, geom, periodic, interfaces=None, elemindicator=None):
        # If `periodic` is true, the domain should be a unit hypercube or this test
        # might fail.  The function `f` defined below is C0 continuous on a periodic
        # hypercube and Cinf continuous inside the hypercube.
        if interfaces is None:
            interfaces = domain.interfaces
        x1, x2, n1, n2 = interfaces.sample('gauss', 2).eval([geom, function.opposite(geom), geom.normal(), function.opposite(geom.normal())])
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
class structure(TestCase, TopologyAssertions):

    def setUp(self):
        super().setUp()
        domain, self.geom = mesh.rectilinear([[-1, 0, 1]]*self.ndims)
        self.domain = domain.refine(self.refine)

    def test_domain(self):
        self.assertConnectivity(self.domain, self.geom)

    def test_boundaries(self):
        for grp in ['left', 'right', 'top', 'bottom', 'front', 'back'][:self.ndims*2]:
            bnd = self.domain.boundary[grp]
            xn = bnd.sample('gauss', 1).eval(self.geom.dotnorm(self.geom))
            numpy.testing.assert_array_less(0, xn, 'inward pointing normals')
            self.assertConnectivity(bnd, self.geom)


structure(ndims=2, refine=0)
structure(ndims=3, refine=0)
structure(ndims=2, refine=1)
structure(ndims=3, refine=1)


class picklability(TestCase):

    def assert_pickle_dump_load(self, data):
        script = b'import pickle, base64\npickle.loads(base64.decodebytes(b"""' \
            + base64.encodebytes(pickle.dumps(data)) \
            + b'"""))'
        p = subprocess.Popen([sys.executable], stdin=subprocess.PIPE)
        p.communicate(script)
        self.assertEqual(p.wait(), 0, 'unpickling failed')

    def test_domain(self):
        domain, geom = mesh.rectilinear([[0, 1, 2]]*2)
        self.assert_pickle_dump_load(domain)

    def test_geom(self):
        domain, geom = mesh.rectilinear([[0, 1, 2]]*2)
        self.assert_pickle_dump_load(geom)

    def test_basis(self):
        domain, geom = mesh.rectilinear([[0, 1, 2]]*2)
        basis = domain.basis('spline', degree=2)
        self.assert_pickle_dump_load(basis)


@parametrize
class common_refine(TestCase, CommonAssertions):

    def _shield(self, topo):
        return topo if self.special \
            else topology.TransformChainsTopology(space=topo.space, references=topo.references, transforms=topo.transforms, opposites=topo.opposites)

    def test(self):
        dom, geom = mesh.rectilinear([[0, 1, 2], [0, 1, 2]])
        doms, funs, vals = {}, {}, {}
        indices = tuple(range(len(dom.transforms)))

        doms['1'] = dom.refined_by(indices[:1])
        funs['1'] = doms['1'].basis('th-std', degree=1)
        vals['1'] = 0.375, 0.25, 0.375, 0.9375, 0.5, 0.25, 0.5, 0.25, 0.0625, 0.125, 0.125, 0.25

        doms['234'] = dom.refined_by(indices[1:])
        funs['234'] = doms['234'].basis('th-std', degree=1)
        vals['234'] = 0.25, 0.375, 0.375, 0.5625, 0.125, 0.0625, 0.25, 0.125, 0.25, 0.125, 0.125, 0.25, 0.25, 0.25, 0.125, 0.0625, 0.125, 0.125, 0.125, 0.0625

        doms['123'] = dom.refined_by(indices[:-1])
        funs['123'] = doms['123'].basis('th-std', degree=1)
        vals['123'] = 0.5625, 0.375, 0.375, 0.25, 0.0625, 0.125, 0.125, 0.125, 0.0625, 0.125, 0.25, 0.25, 0.25, 0.125, 0.125, 0.25, 0.125, 0.25, 0.0625, 0.125

        doms['4'] = dom.refined_by(indices[-1:])
        funs['4'] = doms['4'].basis('th-std', degree=1)
        vals['4'] = 0.25, 0.5, 0.25, 0.5, 0.9375, 0.375, 0.25, 0.375, 0.25, 0.125, 0.125, 0.0625

        for a, b, n in ('1', '234', 16), ('1', '4', 10), ('123', '234', 16):
            with self.subTest('ref{}vs{}'.format(a, b)):
                common = doms[a] & self._shield(doms[b])
                self.assertEqual(len(common), n)
                for c in a, b:
                    testvals = common.integrate(funs[c]*function.J(geom), ischeme='gauss1')
                    numpy.testing.assert_array_almost_equal(testvals, vals[c])

    def test_bnd(self):
        dom, geom = mesh.rectilinear([2, 3])
        dom1 = dom[:1]
        dom2 = dom[1:]
        with self.subTest('equal'):
            iface = dom1.boundary & self._shield(~dom2.boundary)
            self.assertEqual(len(iface), 3)
            self.assertAlmostEqual(iface.integrate(function.J(geom), degree=1), 3)
        with self.subTest('refined-left'):
            iface = dom1.refined.boundary['right'] & self._shield(~dom2.boundary)
            self.assertIs(iface, dom1.refined.boundary['right'])
        with self.subTest('refined-right'):
            iface = self._shield(dom1.boundary) & ~dom2.refined.boundary['left']
            self.assertEquivalent(iface, ~dom2.refined.boundary['left'])
        with self.subTest('partial-refined-both'):
            iface = dom1.refined_by([0]).boundary & self._shield(~dom2.refined_by([2]).boundary)
            self.assertEqual(len(iface), 5)
            self.assertAlmostEqual(iface.integrate(function.J(geom), degree=1), 3)


common_refine(special=True)  # use special case union implementation
common_refine(special=False)  # use fallback union implementation


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
        space = 'test'
        domain = topology.ConnectedTopology(space, References.uniform(ref, 1), transformseq.IndexTransforms(ref.ndims, 1), transformseq.IndexTransforms(ref.ndims, 1), [numpy.repeat(-1, ref.nedges)]).refine(self.ref0)
        geom = function.rootcoords(space, ref.ndims)
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
        self.domain, self.geom = mesh.rectilinear([3, 4, 5], periodic=[] if self.periodic is False else [self.periodic])
        if not self.isstructured:
            self.domain = topology.ConnectedTopology(self.domain.space, self.domain.references, self.domain.transforms, self.domain.opposites, self.domain.connectivity)

    def test_connectivity(self):
        nboundaries = 0
        ninterfaces = 0
        for ielem, ioppelems in enumerate(self.domain.connectivity):
            for iedge, ioppelem in enumerate(ioppelems):
                if ioppelem == -1:
                    nboundaries += 1
                else:
                    ioppedge = tuple(self.domain.connectivity[ioppelem]).index(ielem)
                    edgeref = self.domain.references[ielem].edge_refs[iedge]
                    oppedgeref = self.domain.references[ioppelem].edge_refs[ioppedge]
                    self.assertEqual(edgeref, oppedgeref)
                    ninterfaces += .5
        self.assertEqual(nboundaries, len(self.domain.boundary), 'incompatible number of boundaries')
        self.assertEqual(ninterfaces, len(self.domain.interfaces), 'incompatible number of interfaces')

    def test_boundary(self):
        for trans in self.domain.boundary.transforms:
            ielem, tail = self.domain.transforms.index_with_tail(trans)
            etrans, = tail
            iedge = self.domain.references[ielem].edge_transforms.index(etrans)
            self.assertEqual(self.domain.connectivity[ielem][iedge], -1)

    def test_interfaces(self):
        itopo = self.domain.interfaces
        for trans, opptrans in zip(itopo.transforms, itopo.opposites):
            ielem, tail = self.domain.transforms.index_with_tail(trans)
            etrans, = tail
            iedge = self.domain.references[ielem].edge_transforms.index(etrans)
            ioppelem, opptail = self.domain.transforms.index_with_tail(opptrans)
            eopptrans, = opptail
            ioppedge = self.domain.references[ioppelem].edge_transforms.index(eopptrans)
            self.assertEqual(self.domain.connectivity[ielem][iedge], ioppelem)
            self.assertEqual(self.domain.connectivity[ioppelem][ioppedge], ielem)

    def test_integrate_elementwise(self):
        v = self.domain.integrate_elementwise(self.geom, degree=2)
        self.assertEqual(v.shape, (len(self.domain), len(self.geom)))
        vtot = self.domain.integrate(self.geom, degree=2)
        self.assertAllAlmostEqual(v.sum(axis=0), vtot)
        vf = self.domain.integrate_elementwise(self.geom, degree=2, asfunction=True)
        v_ = self.domain.sample('uniform', 1).eval(vf)
        self.assertAllAlmostEqual(v, v_)


for isstructured in True, False:
    for periodic in False, 0, 1, 2:
        general(isstructured=isstructured, periodic=periodic)


@parametrize
class locate(TestCase):

    def setUp(self):
        super().setUp()
        domain, geom = mesh.unitsquare(4, etype=self.etype)
        if self.mode == 'nonlinear':
            geom = numpy.sin(geom * numpy.pi / 2)  # nonlinear map from [0,1] to [0,1]
        geom = geom * (.32 if self.mode == 'trimmed' else .2, .7) + (0, .3)  # trimmed: keep 2.5 elements in x-direction
        if self.mode == 'trimmed':
            domain = domain.trim(.2 - geom[0], maxrefine=0)
        self.domain = domain
        self.geom = geom * (function.Argument('scale', ()) / .123)

    def test(self):
        target = numpy.array([(.2, .3), (.1, .9), (0, 1), (.1, .3)])
        sample = self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))
        located = sample.eval(self.geom, scale=.123)
        self.assertAllAlmostEqual(located, target)

    @parametrize.enable_if(lambda etype, mode, **kwargs: etype != 'square' or mode == 'nonlinear')
    def test_maxdist(self):
        target = numpy.array([(.2, .3), (.1, .9), (0, 1), (.1, .3)])
        with self.assertRaises(topology.LocateError):
            self.domain.locate(self.geom, [(0, .3)], eps=1e-15, tol=1e-12, maxdist=.001, arguments=dict(scale=.123))
        sample = self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, maxdist=.5, arguments=dict(scale=.123))
        located = sample.eval(self.geom, scale=.123)
        self.assertAllAlmostEqual(located, target)

    def test_invalidargs(self):
        target = numpy.array([(.2,), (.1,), (0,)])
        with self.assertRaises(Exception):
            self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))

    def test_invalidpoint(self):
        target = numpy.array([(.3, 1), (.2, .3), (.1, .9), (0, 1), (.1, .3)])
        # the first point is outside the domain, but inside basetopo for mode==trimmed
        with self.subTest('skip_missing=False'), self.assertRaises(topology.LocateError):
            self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))
        with self.subTest('skip_missing=True'):
            sample = self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123), skip_missing=True)
            self.assertEqual(sample.npoints, 4)

    def test_boundary(self):
        target = numpy.array([(.2,), (.1,), (0,)])
        sample = self.domain.boundary['bottom'].locate(self.geom[:1], target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))
        located = sample.eval(self.geom[:1], scale=.123)
        self.assertAllAlmostEqual(located, target)

    def test_boundary_scalar(self):
        target = numpy.array([.3, .9, 1])
        sample = self.domain.boundary['left'].locate(self.geom[1], target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))
        located = sample.eval(self.geom[1], scale=.123)
        self.assertAllAlmostEqual(located, target)

    def test_missing_argument(self):
        target = numpy.array([(.2, .3)])
        with self.assertRaises(Exception):
            self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12)

    @parametrize.enable_if(lambda etype, mode, **kwargs: etype == 'square' and mode != 'nonlinear')
    def test_detect_linear(self):
        target = numpy.array([(.2, .3)])
        with self.assertLogs('nutils', level='DEBUG') as cm:
            self.domain.locate(self.geom, target, eps=1e-15, tol=1e-12, arguments=dict(scale=.123))
        self.assertRegex(cm.output[0], 'locate detected linear geometry')


for etype in 'square', 'triangle', 'mixed':
    for mode in 'linear', 'nonlinear', 'trimmed':
        locate(etype=etype, mode=mode, tol=1e-12)


@parametrize
class hierarchical(TestCase, TopologyAssertions):

    def setUp(self):
        super().setUp()
        self.nelems = 6
        self.domain, self.geom = mesh.rectilinear([numpy.linspace(0, 1, self.nelems+1)]*self.ndims, periodic=self.periodic)
        # Refine `self.domain` near `self.pos`.
        distance = ((self.geom-self.pos)**2).sum(0)**0.5
        for threshold in 0.3, 0.15:
            selection = self.domain.select(distance < threshold, ischeme='gauss1')
            self.domain = self.domain.refined_by(selection)

    @parametrize.enable_if(lambda periodic, **params: not periodic)
    def test_boundaries(self):
        self.assertBoundaries(self.domain, self.geom)

    def test_interfaces(self):
        self.assertInterfaces(self.domain, self.geom, self.periodic)

    def test_slice(self):
        for idim in range(self.ndims):
            self.assertAlmostEqual(self.domain[(slice(None),)*idim+(slice(1),)].integrate(function.J(self.geom), degree=1), 1/self.nelems)
        self.assertAlmostEqual(self.domain[(slice(1),)*self.ndims].integrate(function.J(self.geom), degree=1), 1/self.nelems**self.ndims)


hierarchical('3d_l_rrr', pos=0, ndims=3, periodic=[])
hierarchical('3d_l_rpr', pos=0, ndims=3, periodic=[1])
hierarchical('2d_l_pp', pos=0, ndims=2, periodic=[0, 1])
hierarchical('2d_l_pr', pos=0, ndims=2, periodic=[0])
hierarchical('2d_c_pr', pos=0.5, ndims=2, periodic=[0])
hierarchical('2d_r_pr', pos=1, ndims=2, periodic=[0])
hierarchical('2d_l_rr', pos=0, ndims=2, periodic=[])
hierarchical('1d_l_p', pos=0, ndims=1, periodic=[0])
hierarchical('1d_c_p', pos=0.5, ndims=1, periodic=[0])
# hierarchical('1d_l_r', pos=0, ndims=1, periodic=[]) # disabled, see issue #193


@parametrize
class trimmedhierarchical(TestCase, TopologyAssertions):

    def setUp(self):
        super().setUp()
        self.domain0, self.geom = mesh.rectilinear([2]*self.ndims)
        self.domain1 = self.domain0.trim((1.1 - self.geom).sum(), maxrefine=2)
        self.domain2 = self.domain1.refined_by(filter(self.domain1.transforms.contains, self.domain0[:1].transforms))
        self.domain3 = self.domain2.refined_by(filter(self.domain2.transforms.contains, self.domain0.refined[:1].transforms))

    def test_boundaries(self):
        self.assertBoundaries(self.domain1, self.geom)
        self.assertBoundaries(self.domain2, self.geom)
        self.assertBoundaries(self.domain3, self.geom)

    def test_interfaces(self):
        self.assertInterfaces(self.domain1, self.geom, periodic=False)
        self.assertInterfaces(self.domain2, self.geom, periodic=False)
        self.assertInterfaces(self.domain3, self.geom, periodic=False)


trimmedhierarchical('1d', ndims=1)
trimmedhierarchical('2d', ndims=2)
trimmedhierarchical('3d', ndims=3)


@parametrize
class multipatch_hyperrect(TestCase, TopologyAssertions):

    def setUp(self):
        super().setUp()
        npatches = numpy.array(self.npatches)
        indices = numpy.arange((npatches+1).prod()).reshape(npatches+1)

        self.domain, self.geom = mesh.multipatch(
            patches=[indices[tuple(map(slice, i, numpy.array(i)+2))].ravel().tolist() for i in itertools.product(*map(range, npatches))],
            patchverts=tuple(itertools.product(*map(range, npatches+1))),
            nelems=4,
        )
        if getattr(self, 'refined', False):
            self.domain = self.domain.refined

    def test_spline_basis(self):
        basis = self.domain.basis('spline', degree=2)
        coeffs = self.domain.sample('gauss', 4).eval(basis.sum(0))
        numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

    def test_discont_basis(self):
        basis = self.domain.basis('discont', degree=2)
        coeffs = self.domain.sample('gauss', 4).eval(basis.sum(0))
        numpy.testing.assert_array_almost_equal(coeffs, numpy.ones(coeffs.shape))

    def test_boundaries(self):
        self.assertBoundaries(self.domain, self.geom)

    def test_interfaces(self):
        self.assertInterfaces(self.domain, self.geom, periodic=False)

    def test_interpatch_interfaces(self):
        self.assertInterfaces(self.domain, self.geom, periodic=False, interfaces=self.domain.interfaces['interpatch'], elemindicator=self.domain.basis('patch'))


multipatch_hyperrect('3', npatches=(3,))
multipatch_hyperrect('2x2', npatches=(2, 2))
multipatch_hyperrect('2x2-refined', npatches=(2, 2), refined=True)
multipatch_hyperrect('3x3', npatches=(3, 3))
multipatch_hyperrect('2x2x3', npatches=(2, 2, 3))


class multipatch_L(TestCase):

    def setUp(self):
        # 2---5
        # |   |
        # 1---4------7
        # |   |      |
        # 0---3------6

        super().setUp()
        self.domain, self.geom = mesh.multipatch(
            patches=[[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7]],
            patchverts=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [3, 0], [3, 1]],
            nelems={None: 4, (3, 6): 8, (4, 7): 8})

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
        knotvalues = {None: None, (1, 2): knots_12, (5, 4): knots_12[::-1], (0, 1): knots_01, (3, 4): knots_01, (6, 7): knots_01}
        knotmultiplicities = {None: None, (0, 1): mults_01, (4, 3): mults_01[::-1], (6, 7): mults_01}
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

    def test_connectivity(self):
        interfaces1 = self.domain.interfaces
        interfaces2 = topology.ConnectedTopology(self.domain.space, self.domain.references, self.domain.transforms, self.domain.opposites, self.domain.connectivity).interfaces
        self.assertEqual(len(interfaces1), len(interfaces2))
        for trans1, opp1 in zip(interfaces1.transforms, interfaces1.opposites):
            try:
                i2 = interfaces2.transforms.index(trans1)
            except ValueError:
                i2 = interfaces2.opposites.index(trans1)
                trans1, opp1 = opp1, trans1
            self.assertEqual(trans1, interfaces2.transforms[i2])
            self.assertEqual(opp1, interfaces2.opposites[i2])


class multipatch_misc(TestCase):

    def test_basis_merge_dofs(self):
        patches = (0, 4, 2, 5), (9, 1, 5, 3), (2, 5, 6, 7), (5, 3, 7, 8)
        domain, geom = mesh.multipatch(patches=patches, nelems=1)
        basis = domain.basis('spline', 1)
        actual = domain.sample('bezier', 2).eval(basis)
        desired = numpy.zeros((16, 10))
        desired[numpy.arange(16), [0, 1, 2, 3, 4, 5, 3, 6, 2, 3, 7, 8, 3, 6, 8, 9]] = 1
        numpy.testing.assert_allclose(actual, desired)


class multipatch_errors(TestCase):

    def test_reverse(self):
        with self.assertRaises(NotImplementedError):
            mesh.multipatch(
                patches=[[0, 1, 2, 3], [3, 2, 5, 4]],
                patchverts=tuple(itertools.product([0, 1, 2], [0, 1])),
                nelems=1)

    def test_transpose(self):
        with self.assertRaises(NotImplementedError):
            mesh.multipatch(
                patches=[[0, 1, 2, 3, 4, 5, 6, 7], [4, 6, 5, 7, 8, 10, 9, 11]],
                patchverts=tuple(itertools.product([0, 1, 2], [0, 1], [0, 1])),
                nelems=1)


class groups(TestCase):

    def setUp(self):
        super().setUp()
        self.topo, geom = mesh.rectilinear([2, 2])

    def test_subdomain(self):
        topo1 = self.topo.withsubdomain(ll=self.topo[:1, :1], ur=self.topo[1:, 1:])
        self.assertEqual(len(topo1['ll']), 1)
        self.assertEqual(len(topo1['ur']), 1)
        topo2 = topo1.withsubdomain(diag='ll,ur')
        self.assertEqual(len(topo1['ll']), 1)
        self.assertEqual(len(topo1['ur']), 1)
        self.assertEqual(len(topo2['diag']), 2)

    def test_boundary(self):
        topo1 = self.topo.withboundary(ll='left,bottom', ur='top,right')
        self.assertEqual(len(topo1.boundary['ll']), 4)
        self.assertEqual(len(topo1.boundary['ur']), 4)
        topo2 = topo1.withboundary(full='ll,ur')
        self.assertEqual(len(topo2.boundary['ll']), 4)
        self.assertEqual(len(topo2.boundary['ur']), 4)
        self.assertEqual(len(topo2.boundary['full']), 8)
        topo3 = topo1.withboundary(cup='ll,right')
        self.assertEqual(len(topo3.boundary['cup']), 6)

    def test_interfaces(self):
        topo1 = self.topo.withinterfaces(hor=self.topo[:, :1].boundary['top'], ver=self.topo[:1, :].boundary['right'])
        self.assertEqual(len(topo1.interfaces['hor']), 2)
        self.assertEqual(len(topo1.interfaces['ver']), 2)
        topo2 = topo1.withinterfaces(full='hor,ver')
        self.assertEqual(len(topo2.interfaces['hor']), 2)
        self.assertEqual(len(topo2.interfaces['ver']), 2)
        self.assertEqual(len(topo2.interfaces['full']), 4)


class TransformChainsTests:

    def test_refined(self):
        refined = self.topo.refined
        checkreferences = self.topo.references.children
        checktransforms = self.topo.transforms.refined(self.topo.references)
        self.assertEqual(len(refined), len(checktransforms))
        self.assertEqual(set(refined.transforms), set(checktransforms))
        for ref, trans in zip(refined.references, refined.transforms):
            self.assertEqual(ref, checkreferences[checktransforms.index(trans)])

    def test_refine_iter(self):
        level_iter = iter(self.topo.refine_iter)
        check = self.topo
        for i in range(4):
            level = next(level_iter)
            self.assertIs(level, check)
            check = check.refined


class TransformChainsBoundaryTests:

    def test_border_transforms(self):
        border = set(map(self.topo.transforms.index, self.topo.border_transforms))
        check = set(self.topo.transforms.index_with_tail(btrans)[0] for btrans in self.topo.boundary.transforms)
        self.assertEqual(border, check)


class TransformChainsTopology(TestCase, CommonTests, TransformChainsTests):

    def setUp(self):
        super().setUp()
        references = References.uniform(element.PointReference(), 1)
        transforms = transformseq.IndexTransforms(0, 1)
        self.topo = topology.TransformChainsTopology('test', references, transforms, transforms)
        self.geom = self.topo.f_coords
        self.desired_nelems = 1
        self.desired_spaces = 'test',
        self.desired_space_dims = 0,
        self.desired_ndims = 0
        self.desired_volumes = [1]
        self.desired_references = list(references)
        self.desired_vertices = [[[]]]


class StructuredTopology2D(TestCase, CommonTests, ConformingTests, TransformChainsTests, TransformChainsBoundaryTests):

    def setUp(self):
        super().setUp()
        self.topo, self.geom = mesh._oldrectilinear([2, 2])
        self.desired_nelems = 4
        self.desired_spaces = 'X',
        self.desired_space_dims = 2,
        self.desired_ndims = 2
        self.desired_volumes = [1]*4
        self.desired_references = [element.LineReference()**2]*4
        self.desired_vertices = [[[x, y] for x in X for y in Y] for X in pairwise(range(3)) for Y in pairwise(range(3))]


class UnionTopology(TestCase, CommonTests, TransformChainsTests):

    def setUp(self):
        super().setUp()
        self.line, self.geom = mesh._oldrectilinear([8])
        part1 = self.line[0:4]
        part2 = self.line[2:6].withsubdomain(a=self.line[5:6])
        self.topo = topology.UnionTopology([part1, part2], ['b'])
        self.desired_nelems = 6
        self.desired_spaces = 'X',
        self.desired_space_dims = 1,
        self.desired_ndims = 1
        self.desired_volumes = [1]*6
        self.desired_references = [element.LineReference()]*6
        self.desired_vertices = [[[x] for x in X] for X in pairwise(range(7))]

    def test_nested_unions(self):
        topo = self.topo | topology.UnionTopology([self.line[2:4], self.line[5:7]])
        self.assertEqual(len(topo), 7)

    def test_union_other(self):
        topo = self.topo | self.line[5:7]
        self.assertEqual(len(topo), 7)

    def test_get_groups_self(self):
        self.assertEqual(len(self.topo.get_groups('b')), 4)

    def test_get_groups_parent(self):
        self.assertEqual(len(self.topo.get_groups('a')), 1)


class DisjointUnionTopology(TestCase, CommonTests, TransformChainsTests):

    def setUp(self):
        super().setUp()
        self.line, self.geom = mesh._oldrectilinear([8])
        part1 = self.line[0:2].withsubdomain(a=self.line[1:2])
        part2 = self.line[4:6].withsubdomain(a=self.line[5:6])
        self.topo = topology.DisjointUnionTopology([part1, part2], ['b'])
        self.desired_nelems = 4
        self.desired_spaces = 'X',
        self.desired_space_dims = 1,
        self.desired_ndims = 1
        self.desired_volumes = [1]*4
        self.desired_references = [element.LineReference()]*4
        self.desired_vertices = [[[x] for x in X] for X in pairwise(range(3)) + pairwise(range(4, 7))]

    def test_get_groups_self(self):
        self.assertEqual(len(self.topo.get_groups('b')), 2)

    def test_get_groups_parent(self):
        self.assertEqual(len(self.topo.get_groups('a')), 2)


class SimplexTopology(TestCase, CommonTests, TransformChainsTests, ConformingTests):

    def setUp(self):
        super().setUp()
        coords = numpy.array([[0,0],[0,1],[1,0],[1,1],[.5,.5]])
        simplices = numpy.array([[0,1,4],[0,2,4],[1,3,4],[2,3,4]])
        transforms = transformseq.IndexTransforms(2, len(simplices))
        self.topo = topology.SimplexTopology('X', simplices, transforms, transforms)
        self.geom = self.topo.basis('std', degree=1) @ coords
        self.desired_nelems = 4
        self.desired_spaces = 'X',
        self.desired_space_dims = 2,
        self.desired_ndims = 2
        self.desired_volumes = [.25]*4
        self.desired_references = [element.TriangleReference()]*4
        self.desired_vertices = coords[simplices].tolist()

    def test_boundary(self):
        self.assertIsInstance(self.topo.boundary, topology.SimplexTopology)

    def test_getitem(self):
        self.assertIsInstance(self.topo[numpy.arange(4) < 2], topology.SimplexTopology)
        self.assertIsInstance(self.topo[numpy.arange(2)], topology.SimplexTopology)


class project(TestCase):

    def setUp(self):
        super().setUp()
        self.topo, self.geom = mesh.rectilinear([8])
        self.basis = self.topo.basis('std', degree=1)

    def test_lsqr(self):
        exact = numpy.sin(numpy.arange(9))
        projected = self.topo.project(self.basis.dot(exact), onto=self.basis, geometry=self.geom, ptype='lsqr', degree=2)
        self.assertAllAlmostEqual(exact, projected)

    def test_nodal(self):
        exact = numpy.sin(numpy.arange(9))
        projected = self.topo.project(self.basis.dot(exact), onto=self.basis, geometry=self.geom, ptype='nodal', degree=2)
        self.assertAllAlmostEqual(exact, projected)

    def test_convolute(self):
        exact = numpy.repeat(numpy.pi, 9)
        projected = self.topo.project(self.basis.dot(exact), onto=self.basis, geometry=self.geom, ptype='convolute', degree=2)
        self.assertAllAlmostEqual(exact, projected)
