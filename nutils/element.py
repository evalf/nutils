"""
The element module defines reference elements such as the
:class:`LineReference` and :class:`TriangleReference`, but also more exotic
objects like the :class:`MosaicReference`. A set of (interconnected) elements
together form a :class:`nutils.topology.Topology`. Elements have edges and
children (for refinement), which are in turn elements and map onto self by an
affine transformation. They also have a well defined reference coordinate
system, and provide pointsets for purposes of integration and sampling.
"""

from . import _util as util, numeric, cache, transform, warnings, types, points
import numpy
import re
import math
import itertools
import operator
import functools
_ = numpy.newaxis


# REFERENCE ELEMENTS

class Reference(types.Singleton):
    'reference element'

    __slots__ = 'ndims',
    __cache__ = 'connectivity', 'edgechildren', 'ribbons', 'volume', 'centroid', '_linear_bernstein', 'getpoints'

    @types.apply_annotations
    def __init__(self, ndims: int):
        super().__init__()
        self.ndims = ndims

    @property
    def nverts(self):
        return len(self.vertices)

    @property
    def vertices(self):
        '''Vertex coordinates.'''

        raise NotImplementedError(self)

    @property
    def simplices(self):
        '''Partition of the element consisting of simplices.

        The `simplices` attribute is a sequence of integer arrays that specify
        per simplex the indices of the vertices in :attr:`vertices`.
        '''

        raise NotImplementedError(self)

    @property
    def simplex_transforms(self):
        '''Sequence of transforms from simplex to parent element.

        The `simplex_transforms` attribute is a sequence of objects of type
        :class:`nutils.transform.TransformItem` that provide per simplex the
        coordinate mapping from the simplex to the parent element. The origin
        of the simplex-local coordinate system maps to its first vertex, the
        first unit vector to the second, the second to the third, and so on.
        '''

        return tuple(transform.Square((vertices[1:] - vertices[0]).T, vertices[0]) for vertices in self.vertices[self.simplices])

    def inside(self, point, eps=0):
        for strans in self.simplex_transforms:
            spoint = strans.invapply(point) # point in simplex coordinates
            tol = -eps / strans.det # account for simplex scale
            if all(bary >= tol for bary in (*spoint, 1-spoint.sum())):
                return True
        return False

    @property
    def edge_vertices(self):
        '''Relation between edge and volume vertices.

        Given a volume reference `vol`, `vol.edge_vertices` is a sequence of
        integer arrays that specifies per edge (outer sequence, corresponding
        to `vol.edges`) for each vertex (inner sequence, corresponding to
        `vol.edges[iedge].vertices`) its index in `vol` (corresponding to
        `vol.vertices`).
        '''

        # naive implementation; will be removed in a later commit
        edge_vertices = []
        for etrans, eref in self.edges:
            dist = numpy.linalg.norm(self.vertices[:,_,:] - etrans.apply(eref.vertices), axis=2)
            emap = numpy.argmin(dist, 0)
            assert (dist[emap, numpy.arange(eref.nverts)] < 1e-15).all(), dist
            edge_vertices.append(types.frozenarray(emap, copy=False))
        return tuple(edge_vertices)

    __and__ = lambda self, other: self if self == other else NotImplemented
    __or__ = lambda self, other: self if self == other else NotImplemented
    __rand__ = lambda self, other: self.__and__(other)
    __ror__ = lambda self, other: self.__or__(other)
    __sub__ = __rsub__ = lambda self, other: self.empty if self == other else NotImplemented
    __bool__ = __nonzero__ = lambda self: bool(self.volume)

    @property
    def empty(self):
        return EmptyLike(self)

    def __mul__(self, other):
        '''Return ``self*other``.'''

        if not isinstance(other, Reference):
            return NotImplemented
        return self.product(other)

    def product(self, other):
        return self if not other.ndims else other if not self.ndims else TensorReference(self, other)

    def __pow__(self, n):
        assert numeric.isint(n) and n >= 0
        return getsimplex(0) if n == 0 \
            else self if n == 1 \
            else self * self**(n-1)

    @property
    def nedges(self):
        return len(self.edge_transforms)

    @property
    def nchildren(self):
        return len(self.child_transforms)

    @property
    def edges(self):
        return list(zip(self.edge_transforms, self.edge_refs))

    @property
    def children(self):
        return list(zip(self.child_transforms, self.child_refs))

    @property
    def connectivity(self):
        # Nested tuple with connectivity information about edges of children:
        # connectivity[ichild][iedge] = ioppchild (interface) or -1 (boundary).
        connectivity = [-numpy.ones(child.nedges, dtype=int) for child in self.child_refs]
        vmap = {}
        for ichild, (ctrans, cref) in enumerate(self.children):
            for iedge, etrans in enumerate(cref.edge_transforms):
                v = ctrans * etrans
                try:
                    jchild, jedge = vmap.pop(v.flipped)
                except KeyError:
                    vmap[v] = ichild, iedge
                else:
                    connectivity[jchild][jedge] = ichild
                    connectivity[ichild][iedge] = jchild
        for etrans, eref in self.edges:
            for ctrans in eref.child_transforms:
                vmap.pop(etrans * ctrans, None)
        assert not any(self.child_refs[ichild].edge_refs[iedge] for ichild, iedge in vmap.values()), 'not all boundary elements recovered'
        return tuple(types.frozenarray(c, copy=False) for c in connectivity)

    @property
    def edgechildren(self):
        edgechildren = []
        for iedge, (etrans, eref) in enumerate(self.edges):
            children = []
            for ichild, ctrans in enumerate(eref.child_transforms):
                ctrans_, etrans_ = etrans.swapup(ctrans)
                ichild_ = self.child_transforms.index(ctrans_)
                iedge_ = self.child_refs[ichild].edge_transforms.index(etrans_)
                children.append((ichild_, iedge_))
            edgechildren.append(types.frozenarray(children))
        return tuple(edgechildren)

    @property
    def ribbons(self):
        # tuples of (iedge1,jedge1), (iedge2,jedge2) pairs
        assert self.ndims >= 2
        transforms = {}
        ribbons = []
        for iedge1, (etrans1, edge1) in enumerate(self.edges):
            if edge1:
                for iedge2, (etrans2, edge2) in enumerate(edge1.edges):
                    if edge2:
                        key = tuple(sorted(tuple(p) for p in (etrans1 * etrans2).apply(edge2.vertices)))
                        try:
                            jedge1, jedge2 = transforms.pop(key)
                        except KeyError:
                            transforms[key] = iedge1, iedge2
                        else:
                            assert self.edge_refs[jedge1].edge_refs[jedge2] == edge2
                            ribbons.append(((iedge1, iedge2), (jedge1, jedge2)))
        assert not transforms
        return tuple(ribbons)

    def getischeme(self, ischeme):
        ischeme, degree = parse_legacy_ischeme(ischeme)
        points = self.getpoints(ischeme, degree)
        return points.coords, getattr(points, 'weights', None)

    def getpoints(self, ischeme, degree):
        if ischeme == '_centroid':
            gauss = self.getpoints('gauss', 1)
            if gauss.npoints == 1:
                return gauss
            volume = gauss.weights.sum()
            return points.CoordsUniformPoints(gauss.coords.T[_] @ gauss.weights / volume, volume)
        raise Exception('unsupported ischeme for {}: {!r}'.format(self.__class__.__name__, ischeme))

    def with_children(self, child_refs):
        child_refs = tuple(child_refs)
        if not any(child_refs):
            return self.empty
        if child_refs == self.child_refs:
            return self
        return WithChildrenReference(self, child_refs)

    @property
    def volume(self):
        volume, = self.getpoints('_centroid', None).weights
        return volume

    @property
    def centroid(self):
        centroid, = self.getpoints('_centroid', None).coords
        return centroid

    def trim(self, levels, maxrefine, ndivisions):
        'trim element along levelset'

        assert len(levels) == self._nlinear_by_level(maxrefine)
        return self if not self or numpy.greater_equal(levels, 0).all() \
            else self.empty if numpy.less_equal(levels, 0).all() \
            else self.with_children(cref.trim(clevels, maxrefine-1, ndivisions)
                                    for cref, clevels in zip(self.child_refs, self.child_divide(levels, maxrefine))) if maxrefine > 0 \
            else self.slice(numeric.poly_eval(self._linear_bernstein, self.vertices) @ levels, ndivisions)

    @property
    def _linear_bernstein(self):
        return self.get_poly_coeffs('bernstein', degree=1)

    def slice(self, levels, ndivisions):
        # slice along levelset by recursing over dimensions

        assert len(levels) == len(self.vertices)
        if numpy.greater_equal(levels, 0).all():
            return self
        if numpy.less_equal(levels, 0).all():
            return self.empty
        assert self.ndims >= 1

        refs = [edgeref.slice(levels[edgeverts], ndivisions) for edgeverts, edgeref in zip(self.edge_vertices, self.edge_refs)]
        if sum(ref != baseref for ref, baseref in zip(refs, self.edge_refs)) < self.ndims:
            return self
        if sum(bool(ref) for ref in refs) < self.ndims:
            return self.empty

        if self.ndims == 1:

            # For 1D elements a midpoint is introduced through linear
            # interpolation of the vertex levels, followed by a binning step to
            # remove near-vertex cuts and improve robustness for topology-wide
            # connectivity.

            iedge = [i for (i,), edge in zip(self.edge_vertices, self.edge_refs) if edge]
            l0, l1 = levels[iedge]
            nbins = 2**ndivisions
            xi = numpy.round(l0/(l0-l1) * nbins)
            if xi in (0, nbins):
                return self.empty if xi == 0 and l1 < 0 or xi == nbins and l0 < 0 else self
            v0, v1 = self.vertices[iedge]
            midpoint = v0 + (xi/nbins) * (v1-v0)

        else:

            # For higher-dimensional elements, the first vertex that is newly
            # introduced by an edge slice is selected to serve as 'midpoint'.
            # In case no new vertices are introduced (all edges are either
            # fully retained or fully removed) then the first vertex is
            # selected that occurs in only one of the edges. Either situation
            # guarantees that the selected vertex lies on the exterior hull.

            for trans, edge, emap, newedge in zip(self.edge_transforms, self.edge_refs, self.edge_vertices, refs):
                if newedge.nverts > edge.nverts:
                    midpoint = trans.apply(newedge.vertices[edge.nverts])
                    break
            else:
                count = numpy.zeros(self.nverts, dtype=int)
                for emap, eref in zip(self.edge_vertices, refs):
                    count[emap[eref.simplices]] += 1
                midpoint = self.vertices[count==1][0]

        return MosaicReference(self, refs, midpoint)

    def cone(self, trans, tip):
        return Cone(self, trans, tip)

    def check_edges(self, tol=1e-15, print=print):
        volume = 0
        zero = 0
        for trans, edge in self.edges:
            if edge:
                gauss = edge.getpoints('gauss', 1)
                w_normal = gauss.weights[:, _] * trans.ext
                zero += w_normal.sum(0)
                volume += numeric.contract(trans.apply(gauss.coords), w_normal, axis=0)
        if numpy.greater(abs(zero), tol).any():
            print('divergence check failed: {} != 0'.format(zero))
        if numpy.greater(abs(volume - self.volume), tol).any():
            print('divergence check failed: {} != {}'.format(volume, self.volume))

    def _linear_cover(self, ctransforms, maxrefine):
        if maxrefine < 0:
            raise Exception('maxrefine is too low')
        npoints = self._nlinear_by_level(maxrefine)
        allindices = numpy.arange(npoints)
        if len(ctransforms) == 1:
            ctrans, = ctransforms
            assert not ctrans
            return ((), self.getpoints('vertex', maxrefine), allindices),
        if maxrefine == 0:
            raise Exception('maxrefine is too low')
        cbins = [set() for ichild in range(self.nchildren)]
        for ctrans in ctransforms:
            ichild = self.child_transforms.index(ctrans[0])
            cbins[ichild].add(ctrans[1:])
        if not all(cbins):
            raise Exception('transformations to not form an element cover')
        fcache = cache.WrapperCache()
        return tuple(((ctrans,) + trans, points, cindices[indices])
                     for ctrans, cref, cbin, cindices in zip(self.child_transforms, self.child_refs, cbins, self.child_divide(allindices, maxrefine))
                     for trans, points, indices in fcache[cref._linear_cover](frozenset(cbin), maxrefine-1))

    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__

    def get_ndofs(self, degree):
        raise NotImplementedError

    def get_poly_coeffs(self, basis, **kwargs):
        raise NotImplementedError

    def get_edge_dofs(self, degree, iedge):
        raise NotImplementedError


strictreference = types.strict[Reference]


class EmptyLike(Reference):
    'inverse reference element'

    __slots__ = 'baseref',

    volume = 0

    @property
    def empty(self):
        return self

    @types.apply_annotations
    def __init__(self, baseref: strictreference):
        self.baseref = baseref
        super().__init__(baseref.ndims)

    @property
    def vertices(self):
        return self.baseref.vertices

    @property
    def edge_vertices(self):
        return self.baseref.edge_vertices

    @property
    def simplices(self):
        return types.frozenarray(numpy.empty([0, self.ndims+1], dtype=int), copy=False)

    @property
    def edge_transforms(self):
        return self.baseref.edge_transforms

    @property
    def edge_refs(self):
        return tuple(eref.empty for eref in self.baseref.edge_refs)

    @property
    def child_transforms(self):
        return self.baseref.child_transforms

    @property
    def child_refs(self):
        return tuple(cref.empty for cref in self.baseref.child_refs)

    __and__ = __sub__ = lambda self, other: self if other.ndims == self.ndims else NotImplemented
    __or__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented
    __rsub__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented

    def trim(self, levels, maxrefine, ndivisions):
        return self


class SimplexReference(Reference):
    'simplex reference'

    __slots__ = ()
    __cache__ = 'edge_refs', 'edge_transforms', 'edge_vertices', 'ribbons', '_get_poly_coeffs_bernstein', '_get_poly_coeffs_lagrange', '_integer_barycentric_coordinates'

    @property
    def vertices(self):
        return types.frozenarray(numpy.eye(self.ndims+1)[1:].T, copy=False) # first vertex in origin

    @property
    def edge_vertices(self):
        return tuple(types.frozenarray(numpy.arange(self.ndims+1).repeat(self.ndims).reshape(self.ndims,self.ndims+1).T[::-1], copy=False))

    @property
    def simplices(self):
        return types.frozenarray(numpy.arange(self.ndims+1)[numpy.newaxis], copy=False)

    @property
    def simplex_transforms(self):
        # The definition of self.vertices is such that the conventions of
        # Reference.simplex_transforms result in the identity map.
        return transform.Identity(self.ndims),

    @property
    def edge_refs(self):
        assert self.ndims > 0
        return (getsimplex(self.ndims-1),) * (self.ndims+1)

    @property
    def edge_transforms(self):
        assert self.ndims > 0
        return tuple(transform.SimplexEdge(self.ndims, i) for i in range(self.ndims+1))

    @property
    def child_refs(self):
        return tuple([self] * (2**self.ndims))

    @property
    def child_transforms(self):
        return tuple(transform.SimplexChild(self.ndims, ichild) for ichild in range(2**self.ndims))

    @property
    def ribbons(self):
        return tuple(((iedge1, iedge2), (iedge2+1, iedge1)) for iedge1 in range(self.ndims+1) for iedge2 in range(iedge1, self.ndims))

    def getpoints(self, ischeme, degree):
        if ischeme == 'gauss':
            return points.SimplexGaussPoints(self.ndims, degree if numeric.isint(degree) else sum(degree))
        if ischeme == 'vtk':
            return points.SimplexBezierPoints(self.ndims, 2)
        if ischeme == 'vertex':
            return points.SimplexBezierPoints(self.ndims, 2**(degree or 0) + 1)
        if ischeme == 'bezier':
            return points.SimplexBezierPoints(self.ndims, degree)
        return super().getpoints(ischeme, degree)

    def get_ndofs(self, degree):
        prod = lambda start, stop: functools.reduce(operator.mul, range(start, stop), 1)
        return prod(degree+1, degree+1+self.ndims) // prod(1, self.ndims+1)

    def get_poly_coeffs(self, basis, **kwargs):
        f = getattr(self, '_get_poly_coeffs_{}'.format(basis), None)
        if f:
            return f(**kwargs)
        else:
            raise ValueError('basis {!r} undefined on {}'.format(basis, type(self).__qualname__))

    def _integer_barycentric_coordinates(self, degree):
        return tuple(
            (degree-sum(i), *i[::-1])
            for i in itertools.product(*[range(degree+1)]*self.ndims)
            if sum(i) <= degree)

    def _get_poly_coeffs_bernstein(self, degree):
        ndofs = self.get_ndofs(degree)
        coeffs = numpy.zeros((ndofs,)+(degree+1,)*self.ndims)
        for i, p in enumerate(self._integer_barycentric_coordinates(degree)):
            p = p[1:]
            for q in itertools.product(*[range(degree+1)]*self.ndims):
                if sum(p+q) <= degree:
                    coeffs[(i,)+tuple(map(operator.add, p, q))] = (-1)**sum(q)*math.factorial(degree)//util.product(map(math.factorial, (degree-sum(p+q), *p, *q)))
        assert i == ndofs - 1
        return types.frozenarray(coeffs, copy=False)

    def _get_poly_coeffs_lagrange(self, degree):
        if self.ndims == 0:
            coeffs = numpy.ones((1,))
        elif degree == 0:
            coeffs = numpy.ones((1, *[1]*self.ndims))
        else:
            P = numpy.array(tuple(self._integer_barycentric_coordinates(degree)), dtype=int)[:, 1:]
            coeffs_ = numpy.linalg.inv(((P[:, _, :]/degree)**P[_, :, :]).prod(-1))
            coeffs = numpy.zeros((len(P), *[degree+1]*self.ndims), dtype=float)
            for i, p in enumerate(P):
                coeffs[(slice(None), *p)] = coeffs_[i]
        return types.frozenarray(coeffs, copy=False)

    def get_edge_dofs(self, degree, iedge):
        return types.frozenarray(tuple(i for i, j in enumerate(self._integer_barycentric_coordinates(degree)) if j[iedge] == 0), dtype=int)


class PointReference(SimplexReference):
    '0D simplex'

    __slots__ = ()
    __cache__ = 'getpoints',

    def __init__(self):
        super().__init__(ndims=0)

    def getpoints(self, ischeme, degree):
        return points.CoordsWeightsPoints(numpy.empty([1, 0]), [1.])

    def _nlinear_by_level(self, n):
        return 1

    def child_divide(self, vals, n):
        return vals,


class LineReference(SimplexReference):
    '1D simplex'

    __slots__ = '_bernsteincache',

    def __init__(self):
        self._bernsteincache = []  # TEMPORARY
        super().__init__(ndims=1)

    def getpoints(self, ischeme, degree):
        if ischeme == 'uniform':
            return points.CoordsUniformPoints(numpy.arange(.5, degree)[:, _] / degree, 1)
        return super().getpoints(ischeme, degree)

    def _nlinear_by_level(self, n):
        return 2**n + 1

    def child_divide(self, vals, n):
        assert n > 0
        assert len(vals) == self._nlinear_by_level(n)
        m = (len(vals)+1) // 2
        return vals[:m], vals[m-1:]


class TriangleReference(SimplexReference):
    '2D simplex'

    __slots__ = ()

    def __init__(self):
        super().__init__(ndims=2)

    def getpoints(self, ischeme, degree):
        if ischeme == 'uniform':
            p = numpy.arange(1./3, degree) / degree
            C = numpy.empty([2, degree, degree])
            C[0] = p[:, _]
            C[1] = p[_, :]
            coords = C.reshape(2, -1)
            flip = numpy.greater(coords.sum(0), 1)
            coords[:, flip] = 1 - coords[::-1, flip]
            return points.CoordsUniformPoints(coords.T, .5)
        return super().getpoints(ischeme, degree)

    def _nlinear_by_level(self, n):
        m = 2**n + 1
        return ((m+1)*m) // 2

    def child_divide(self, vals, n):
        assert len(vals) == self._nlinear_by_level(n)
        np = 1 + 2**n  # points along parent edge
        mp = 1 + 2**(n-1)  # points along child edge
        cvals = []
        for i in range(mp):
            j = numpy.arange(mp-i)
            cvals.append([vals[b+a*np-(a*(a-1))//2] for a, b in [(i, j), (i, mp-1+j), (mp-1+i, j), (i+j, mp-1-j)]])
        return numpy.concatenate(cvals, axis=1)


class TetrahedronReference(SimplexReference):
    '3D simplex'

    # TETRAHEDRON:
    # c\d
    # a-b
    #
    # EDGES:
    # d\  d\  d\  c\
    # b-c a-c a-b a-b

    # SUBDIVIDED TETRAHEDRON:
    # f\  i\j
    # d-e\g-h
    # a-b-c
    #
    # SUBDIVIDED EDGES:
    # j\    j\    j\    f\
    # h-i\  g-i\  g-h\  d-e\
    # c-e-f a-d-f a-b-c a-b-c
    #
    # CHILDREN:
    # d\g e\h f\i i\j e\g g\h g\i h\i
    # a-b b-c d-e g-h b-d b-e d-e e-g

    __slots__ = ()

    _children_vertices = [0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9], [1, 3, 4, 6], [1, 4, 6, 7], [3, 4, 6, 8], [4, 6, 7, 8]

    def __init__(self):
        super().__init__(ndims=3)

    def getindices_vertex(self, n):
        m = 2**n+1
        indis = numpy.arange(m)
        return numpy.array([[i, j, k] for k in indis for j in indis[:m-k] for i in indis[:m-j-k]])

    def _nlinear_by_level(self, n):
        m = 2**n+1
        return ((m+2)*(m+1)*m)//6

    def child_divide(self, vals, n):
        assert len(vals) == self._nlinear_by_level(n)

        child_indices = self.getindices_vertex(1)

        offset = numpy.array([1, 0, 0, 0])
        linear = numpy.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        m = 2**n+1
        cvals = []
        for child_ref, child_vertices in zip(self.child_refs, self._children_vertices):
            V = child_indices[child_vertices]

            child_offset = (2**(n-1))*V.T.dot(offset)
            child_linear = V.T.dot(linear)

            original = child_ref.getindices_vertex(n-1)
            transformed = original.dot(child_linear.T) + child_offset

            i, j, k = transformed.T
            cvals.append(vals[((k-1)*k*(2*k-1)//6 - (1+2*m)*(k-1)*k//2 + m*(m+1)*k)//2 + ((2*(m-k)+1)*j-j**2)//2 + i])

        return numpy.array(cvals)


class TensorReference(Reference):
    'tensor reference'

    __slots__ = 'ref1', 'ref2'
    __cache__ = 'vertices', 'edge_transforms', 'edge_vertices', 'ribbons', 'child_transforms', 'getpoints', 'get_poly_coeffs', 'centroid', 'simplices'

    def __init__(self, ref1, ref2):
        assert not isinstance(ref1, TensorReference)
        self.ref1 = ref1
        self.ref2 = ref2
        super().__init__(ref1.ndims + ref2.ndims)

    def product(self, other):
        return self.ref1.product(self.ref2.product(other))

    @property
    def vertices(self):
        vertices = numpy.empty((self.ref1.nverts, self.ref2.nverts, self.ndims), dtype=float)
        vertices[:, :, :self.ref1.ndims] = self.ref1.vertices[:, _]
        vertices[:, :, self.ref1.ndims:] = self.ref2.vertices[_, :]
        return types.frozenarray(vertices.reshape(self.ref1.nverts*self.ref2.nverts, self.ndims), copy=False)

    @property
    def edge_vertices(self):
        n1 = self.ref1.nverts
        n2 = self.ref2.nverts
        edge_vertices = [everts[:,_] * n2 + numpy.arange(n2) for everts in self.ref1.edge_vertices] \
                      + [numpy.arange(n1)[:,_] * n2 + everts for everts in self.ref2.edge_vertices]
        return tuple(types.frozenarray(e.ravel(), copy=False) for e in edge_vertices)

    @property
    def simplices(self):
        if self.ref1.ndims != 1 and self.ref2.ndims != 1:
            raise NotImplementedError((self.ref1, self.ref2))
        # For an n-dimensional simplex with vertices a0,a1,..,an, the extruded
        # element has vertices a0,a1,..,an,b0,b1,..,bn. These can be divided in
        # simplices by selecting a0,a1,..,an,b0; a1,..,an,b0,n1; and so on until
        # an,b0,b1,..,bn; resulting in n+1 n+1-dimensional simplices.
        indices = self.ref1.simplices[:,numpy.newaxis,:,numpy.newaxis] * self.ref2.nverts \
                + self.ref2.simplices[numpy.newaxis,:,numpy.newaxis,:]
        if self.ref1.ndims != 1:
            indices = indices.swapaxes(2,3) # simplex strips require penultimate axis to be of length 2
        assert indices.shape[2] == 2
        indices = numeric.overlapping(indices.reshape(-1, 2*self.ndims), n=self.ndims+1).copy() # nsimplex x nstrip x ndims+1
        # to see determinants: X = self.vertices[indices]; numpy.linalg.det(X[...,1:,:] - X[...,:1,:])
        if self.ndims % 2 == 0: # simplex strips of even dimension (e.g. triangles) have alternating orientation
            indices[:,::2,:2] = indices[:,::2,1::-1].copy() # flip every other simplex
        return types.frozenarray(indices.reshape(-1, self.ndims+1), copy=False)

    @property
    def centroid(self):
        return types.frozenarray(numpy.concatenate([self.ref1.centroid, self.ref2.centroid]), copy=False)

    def _nlinear_by_level(self, n):
        return self.ref1._nlinear_by_level(n) * self.ref2._nlinear_by_level(n)

    def child_divide(self, vals, n):
        np1 = self.ref1._nlinear_by_level(n)
        np2 = self.ref2._nlinear_by_level(n)
        return [v2.swapaxes(0, 1).reshape((-1,)+vals.shape[1:])
                for v1 in self.ref1.child_divide(vals.reshape((np1, np2)+vals.shape[1:]), n)
                for v2 in self.ref2.child_divide(v1.swapaxes(0, 1), n)]

    def __str__(self):
        return '{}*{}'.format(self.ref1, self.ref2)

    def getpoints(self, ischeme, degree):
        if self.ref1.ndims == 0:
            return self.ref2.getpoints(ischeme, degree)
        if self.ref2.ndims == 0:
            return self.ref1.getpoints(ischeme, degree)
        if ischeme != 'vtk':
            ischeme1, ischeme2 = ischeme.split('*', 1) if '*' in ischeme else (ischeme, ischeme)
            degree1 = degree if not isinstance(degree, tuple) else degree[0]
            degree2 = degree if not isinstance(degree, tuple) else degree[1] if len(degree) == 2 else degree[1:]
            return self.ref1.getpoints(ischeme1, degree1) * self.ref2.getpoints(ischeme2, degree2)
        if self.ref1.ndims == self.ref2.ndims == 1:
            coords = numpy.empty([2, 2, 2])
            coords[..., :1] = self.ref1.vertices[:, _]
            coords[0, :, 1:] = self.ref2.vertices
            coords[1, :, 1:] = self.ref2.vertices[::-1]
        elif self.ref1.ndims <= 1 and self.ref2.ndims >= 1:
            coords = numpy.empty([self.ref1.nverts, self.ref2.nverts, self.ndims])
            coords[..., :self.ref1.ndims] = self.ref1.vertices[:, _]
            coords[..., self.ref1.ndims:] = self.ref2.vertices[_, :]
        elif self.ref1.ndims >= 1 and self.ref2.ndims <= 1:
            coords = numpy.empty([self.ref2.nverts, self.ref1.nverts, self.ndims])
            coords[..., :self.ref1.ndims] = self.ref1.vertices[_, :]
            coords[..., self.ref1.ndims:] = self.ref2.vertices[:, _]
        else:
            raise NotImplementedError
        return points.CoordsPoints(coords.reshape(self.nverts, self.ndims))

    @property
    def edge_transforms(self):
        edge_transforms = []
        if self.ref1.ndims:
            edge_transforms.extend(transform.TensorEdge1(trans1, self.ref2.ndims) for trans1 in self.ref1.edge_transforms)
        if self.ref2.ndims:
            edge_transforms.extend(transform.TensorEdge2(self.ref1.ndims, trans2) for trans2 in self.ref2.edge_transforms)
        return tuple(edge_transforms)

    @property
    def edge_refs(self):
        edge_refs = []
        if self.ref1.ndims:
            edge_refs.extend(edge1 * self.ref2 for edge1 in self.ref1.edge_refs)
        if self.ref2.ndims:
            edge_refs.extend(self.ref1 * edge2 for edge2 in self.ref2.edge_refs)
        return tuple(edge_refs)

    @property
    def ribbons(self):
        if self.ref1.ndims == 0:
            return self.ref2.ribbons
        if self.ref2.ndims == 0:
            return self.ref1.ribbons
        ribbons = []
        for iedge1 in range(self.ref1.nedges):
            #iedge = self.ref1.edge_refs[iedge] * self.ref2
            for iedge2 in range(self.ref2.nedges):
                #jedge = self.ref1 * self.ref2.edge_refs[jedge]
                jedge1 = self.ref1.nedges + iedge2
                jedge2 = iedge1
                if self.ref1.ndims > 1:
                    iedge2 += self.ref1.edge_refs[iedge1].nedges
                ribbons.append(((iedge1, iedge2), (jedge1, jedge2)))
        if self.ref1.ndims >= 2:
            ribbons.extend(self.ref1.ribbons)
        if self.ref2.ndims >= 2:
            ribbons.extend(((iedge1+self.ref1.nedges, iedge2+self.ref1.nedges),
                            (jedge1+self.ref1.nedges, jedge2+self.ref1.nedges)) for (iedge1, iedge2), (jedge1, jedge2) in self.ref2.ribbons)
        return tuple(ribbons)

    @property
    def child_transforms(self):
        return tuple(transform.TensorChild(trans1, trans2) for trans1 in self.ref1.child_transforms for trans2 in self.ref2.child_transforms)

    @property
    def child_refs(self):
        return tuple(child1 * child2 for child1 in self.ref1.child_refs for child2 in self.ref2.child_refs)

    def inside(self, point, eps=0):
        return self.ref1.inside(point[:self.ref1.ndims], eps) and self.ref2.inside(point[self.ref1.ndims:], eps)

    def get_ndofs(self, degree):
        return self.ref1.get_ndofs(degree)*self.ref2.get_ndofs(degree)

    def get_poly_coeffs(self, basis, **kwargs):
        return numeric.poly_outer_product(self.ref1.get_poly_coeffs(basis, **kwargs), self.ref2.get_poly_coeffs(basis, **kwargs))

    def get_edge_dofs(self, degree, iedge):
        if not numeric.isint(iedge) or iedge < 0 or iedge >= self.nedges:
            raise IndexError('edge index out of range')
        nd2 = self.ref2.get_ndofs(degree)
        if iedge < self.ref1.nedges:
            dofs1 = self.ref1.get_edge_dofs(degree, iedge)
            dofs2 = range(self.ref2.get_ndofs(degree))
        else:
            dofs1 = range(self.ref1.get_ndofs(degree))
            dofs2 = self.ref2.get_edge_dofs(degree, iedge-self.ref1.nedges)
        return types.frozenarray(tuple(d1*nd2+d2 for d1, d2 in itertools.product(dofs1, dofs2)), dtype=int)

    @property
    def _flat_refs(self):
        for ref in self.ref1, self.ref2:
            if isinstance(ref, TensorReference):
                yield from ref._flat_refs
            else:
                yield ref


class Cone(Reference):
    'cone'

    __slots__ = 'edgeref', 'etrans', 'tip', 'extnorm', 'height'
    __cache__ = 'vertices', 'edge_transforms', 'edge_refs'

    @types.apply_annotations
    def __init__(self, edgeref, etrans, tip: types.arraydata):
        assert etrans.fromdims == edgeref.ndims
        assert etrans.todims == tip.shape[0]
        super().__init__(etrans.todims)
        self.edgeref = edgeref
        self.etrans = etrans
        self.tip = numpy.asarray(tip)
        ext = etrans.ext
        self.extnorm = numpy.linalg.norm(ext)
        self.height = numpy.dot(etrans.offset - self.tip, ext) / self.extnorm
        assert self.height >= 0, 'tip is positioned at the negative side of edge'

    @property
    def vertices(self):
        return types.frozenarray(numpy.vstack([[self.tip], self.etrans.apply(self.edgeref.vertices)]), copy=False)

    @property
    def simplices(self):
        return types.frozenarray([[0, *emap[simplex]] for emap, eref in zip(self.edge_vertices, self.edge_refs) for simplex in eref.simplices])

    @property
    def edge_transforms(self):
        edge_transforms = [self.etrans]
        if self.edgeref.ndims > 0:
            for trans, edge in self.edgeref.edges:
                if edge:
                    b = self.etrans.apply(trans.offset)
                    A = numpy.hstack([numpy.dot(self.etrans.linear, trans.linear), (self.tip-b)[:, _]])
                    newtrans = transform.Updim(A, b, isflipped=self.etrans.isflipped ^ trans.isflipped ^ (self.ndims % 2 == 1))  # isflipped logic tested up to 3D
                    edge_transforms.append(newtrans)
        else:
            edge_transforms.append(transform.Updim(numpy.zeros((1, 0)), self.tip, isflipped=not self.etrans.isflipped))
        return tuple(edge_transforms)

    @property
    def edge_refs(self):
        edge_refs = [self.edgeref]
        if self.edgeref.ndims > 0:
            extrudetrans = transform.Updim(numpy.eye(self.ndims-1)[:, :-1], numpy.zeros(self.ndims-1), isflipped=self.ndims % 2 == 0)
            tip = numpy.array([0]*(self.ndims-2)+[1], dtype=float)
            edge_refs.extend(edge.cone(extrudetrans, tip) for edge in self.edgeref.edge_refs if edge)
        else:
            edge_refs.append(getsimplex(0))
        return tuple(edge_refs)

    def getpoints(self, ischeme, degree):
        if ischeme == 'gauss':
            if self.nverts == self.ndims+1:  # use optimal gauss schemes for simplex-like cones
                pnts = getsimplex(self.ndims).getpoints(ischeme, degree)
                linearT = self.etrans.apply(self.edgeref.vertices) - self.tip
                coords = pnts.coords @ linearT + self.tip
                weights = pnts.weights * abs(numpy.linalg.det(linearT))
                return points.CoordsWeightsPoints(coords, weights)
            epoints = self.edgeref.getpoints('gauss', degree)
            tx, tw = points.gauss((degree + self.ndims - 1)//2)
            wx = tx**(self.ndims-1) * tw * self.extnorm * self.height
            return points.CoordsWeightsPoints((tx[:, _, _] * (self.etrans.apply(epoints.coords)-self.tip)[_, :, :] + self.tip).reshape(-1, self.ndims), (epoints.weights[_, :] * wx[:, _]).ravel())
        if ischeme in ('_centroid', 'uniform'):
            layers = [(i+1, (i+.5)/degree) for i in range(degree)] if ischeme == 'uniform' \
                else [(None, self.ndims/(self.ndims+1))]  # centroid
            coords = numpy.concatenate([self.etrans.apply(self.edgeref.getpoints(ischeme, p).coords) * w + self.tip * (1-w) for p, w in layers])
            volume = self.edgeref.volume * self.extnorm * self.height / self.ndims
            return points.CoordsUniformPoints(coords, volume)
        if ischeme == 'vtk' and self.nverts == 5 and self.ndims == 3:  # pyramid
            return points.CoordsPoints(self.vertices[[1, 2, 4, 3, 0]])
        return points.ConePoints(self.edgeref.getpoints(ischeme, degree), self.etrans, self.tip)

    def inside(self, point, eps=0):
        # point = etrans.apply(epoint) * xi + tip * (1-xi) => etrans.apply(epoint) = tip + (point-tip) / xi
        xi = numpy.dot(self.etrans.ext, point-self.tip) / numpy.dot(self.etrans.ext, self.etrans.offset-self.tip)
        return -eps <= xi <= 1+eps and self.edgeref.inside(numpy.linalg.solve(
            numpy.dot(self.etrans.linear.T, self.etrans.linear),
            numpy.dot(self.etrans.linear.T, self.tip + (point-self.tip)/xi - self.etrans.offset)), eps=eps)


class OwnChildReference(Reference):
    'forward self as child'

    __slots__ = 'baseref', 'child_refs', 'child_transforms'

    def __init__(self, baseref):
        self.baseref = baseref
        self.child_refs = baseref,
        self.child_transforms = transform.Identity(baseref.ndims),
        super().__init__(baseref.ndims)

    @property
    def vertices(self):
        return self.baseref.vertices

    @property
    def edge_transforms(self):
        return self.baseref.edge_transforms

    @property
    def edge_refs(self):
        return [OwnChildReference(edge) for edge in self.baseref.edge_refs]

    def getpoints(self, ischeme, degree):
        return self.baseref.getpoints(ischeme, degree)

    def get_ndofs(self, degree):
        return self.baseref.get_ndofs(degree)

    def get_poly_coeffs(self, basis, **kwargs):
        return self.baseref.get_poly_coeffs(basis, **kwargs)

    def get_edge_dofs(self, degree, iedge):
        return self.baseref.get_edge_dofs(degree, iedge)


class WithChildrenReference(Reference):
    'base reference with explicit children'

    __slots__ = 'baseref', 'child_transforms', 'child_refs'
    __cache__ = '__extra_edges', 'edge_transforms', 'edge_refs', 'connectivity'

    @types.apply_annotations
    def __init__(self, baseref, child_refs: tuple):
        assert len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
        assert all(isinstance(child_ref, Reference) for child_ref in child_refs)
        assert all(child_ref.ndims == baseref.ndims for child_ref in child_refs)
        self.baseref = baseref
        self.child_transforms = baseref.child_transforms
        self.child_refs = child_refs
        super().__init__(baseref.ndims)

    def check_edges(self, tol=1e-15, print=print):
        super().check_edges(tol=tol, print=print)
        for cref in self.child_refs:
            cref.check_edges(tol=tol, print=print)

    @property
    def vertices(self):
        return self.baseref.vertices

    def _nlinear_by_level(self, n):
        return self.baseref._nlinear_by_level(n)

    def child_divide(self, vals, n):
        return self.baseref.child_divide(vals, n)

    __sub__ = lambda self, other: self.empty if other in (self, self.baseref) else self.baseref.with_children(self_child-other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref in (self, self.baseref) else NotImplemented
    __rsub__ = lambda self, other: self.baseref.with_children(other_child - self_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if other == self.baseref else NotImplemented
    __and__ = lambda self, other: self if other == self.baseref else other if isinstance(other, WithChildrenReference) and self == other.baseref else self.baseref.with_children(self_child & other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented
    __or__ = lambda self, other: other if other == self.baseref else self.baseref.with_children(self_child | other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented

    @property
    def __extra_edges(self):
        extra_edges = [(ichild, iedge, cref.edge_refs[iedge])
                       for ichild, cref in enumerate(self.child_refs) if cref
                       for iedge in range(self.baseref.child_refs[ichild].nedges, cref.nedges)]
        for ichild, edges in enumerate(self.baseref.connectivity):
            cref = self.child_refs[ichild]
            if not cref:
                continue  # new child is empty
            for iedge, jchild in enumerate(edges):
                if jchild == -1:
                    continue  # iedge already is an external boundary
                coppref = self.child_refs[jchild]
                if coppref == self.baseref.child_refs[jchild]:
                    continue  # opposite is complete, so iedge cannot form a new external boundary
                eref = cref.edge_refs[iedge]
                if coppref:  # opposite new child is not empty
                    eref -= coppref.edge_refs[util.index(self.baseref.connectivity[jchild], ichild)]
                if eref:
                    extra_edges.append((ichild, iedge, eref))
        return tuple(extra_edges)

    def subvertex(self, ichild, i):
        assert 0 <= ichild < self.nchildren
        npoints = 0
        for childindex, child in enumerate(self.child_refs):
            if child:
                points = child.getpoints('vertex', i-1).coords
                if childindex == ichild:
                    rng = numpy.arange(npoints, npoints+len(points))
                npoints += len(points)
            elif ichild == childindex:
                rng = numpy.array([], dtype=int)
        return npoints, rng

    def getpoints(self, ischeme, degree):
        if ischeme == 'vertex':
            return self.baseref.getpoints(ischeme, degree)
        if ischeme == '_centroid':
            return super().getpoints(ischeme, degree)
        if ischeme == 'bezier':
            degree = degree//2+1  # modify child degree to keep (approximate) uniformity
            dedup = True
        else:
            dedup = False
        childpoints = [points.TransformPoints(ref.getpoints(ischeme, degree), trans) for trans, ref in self.children if ref]
        return points.ConcatPoints(childpoints, points.find_duplicates(childpoints) if dedup else ())

    @property
    def edge_transforms(self):
        return tuple(self.baseref.edge_transforms) \
            + tuple(transform.ScaledUpdim(self.child_transforms[ichild], self.child_refs[ichild].edge_transforms[iedge]) for ichild, iedge, ref in self.__extra_edges)

    @property
    def edge_refs(self):
        refs = []
        for etrans, eref in self.baseref.edges:
            children = []
            if eref:
                for ctrans, cref in eref.children:
                    ctrans_, etrans_ = etrans.swapup(ctrans)
                    ichild = self.baseref.child_transforms.index(ctrans_)
                    cref = self.child_refs[ichild]
                    children.append(cref.edge_refs[cref.edge_transforms.index(etrans_)])
            refs.append(eref.with_children(children))
        for ichild, iedge, ref in self.__extra_edges:
            refs.append(OwnChildReference(ref))
        return tuple(refs)

    @property
    def connectivity(self):
        return tuple(types.frozenarray(edges.tolist() + [-1] * (self.child_refs[ichild].nedges - len(edges))) for ichild, edges in enumerate(self.baseref.connectivity))

    def inside(self, point, eps=0):
        return any(cref.inside(ctrans.invapply(point), eps=eps) for ctrans, cref in self.children)

    def get_ndofs(self, degree):
        return self.baseref.get_ndofs(degree)

    def get_poly_coeffs(self, basis, **kwargs):
        return self.baseref.get_poly_coeffs(basis, **kwargs)

    def get_edge_dofs(self, degree, iedge):
        return self.baseref.get_edge_dofs(degree, iedge)


class MosaicReference(Reference):
    'triangulation'

    __slots__ = 'baseref', '_edge_refs', '_midpoint', 'edge_refs', 'edge_transforms', 'vertices', '_imidpoint', 'edge_vertices'
    __cache__ = 'simplices'

    @types.apply_annotations
    def __init__(self, baseref, edge_refs: tuple, midpoint: types.arraydata):
        assert len(edge_refs) == baseref.nedges
        assert edge_refs != tuple(baseref.edge_refs)
        assert midpoint.shape == (baseref.ndims,)
        assert all(numpy.all(edge.vertices == newedge.vertices[:edge.nverts])
            for edge, newedge in zip(baseref.edge_refs, edge_refs))

        vertices = list(baseref.vertices)
        match, = (baseref.vertices == midpoint).all(1).nonzero()
        if match.size:
            imidpoint, = match
        else:
            imidpoint = len(vertices)
            vertices.append(numpy.asarray(midpoint))

        # The remainder of this constructor is concerned with establishing the
        # edges of the mosaic, and setting up the corresponding edge-vertex
        # relations.

        if baseref.ndims == 1:

            # For 1D elements the situation is simple: the midpoint represents
            # the only new vertex (already added to vertices) as well as the
            # only new edge, with a trivial edge-vertex relationship.

            assert not match.size, '1D mosaic must introduce a new vertex'
            edge_vertices = (*baseref.edge_vertices, types.frozenarray([imidpoint]))
            orientation = [not trans.isflipped for trans, edge in zip(baseref.edge_transforms, edge_refs) if edge]
            assert len(orientation) == 1, 'invalid 1D mosaic: exactly one edge should be non-empty'

        else:

            # For higher-dimensional elements the situation is more complex.
            # Firstly, the new edge_refs may introduce new vertices. Luckily
            # here the previously asserted convention applies that the vertices
            # of the original edge are repeated in the modified edge, so we can
            # focus our attention on the new ones, having only to deduplicate
            # between them.

            edge_vertices = []
            for trans, edge, emap, newedge in zip(baseref.edge_transforms, baseref.edge_refs, baseref.edge_vertices, edge_refs):
                for v in trans.apply(newedge.vertices[edge.nverts:]):
                    for i, v_ in enumerate(vertices[baseref.nverts:], start=baseref.nverts):
                        if (v == v_).all():
                            break # the new vertex was already added by a previous edge
                    else:
                        i = len(vertices)
                        vertices.append(v)
                    emap = types.frozenarray([*emap, i])
                edge_vertices.append(emap)

            # Secondly, new edges (which will be pulled to the midpoint) can
            # originate either from new edge-edges, or from existing ones that
            # find themselves without a counterpart. The former situation is
            # trivial, following the convention that existing edge transforms
            # are copied over in the modified edge.

            assert all(edge.edge_transforms == newedge.edge_transforms[:edge.nverts]
                for edge, newedge in zip(baseref.edge_refs, edge_refs))

            # The latter, however, is more tricky. This is the situation that
            # occurs, for instance, when two out of four edges of a square are
            # cleanly removed, making two existing edge-edges the new exterior.
            # Identifying these situations requires an examination of all the
            # modified edges, that is, edge-edges in locations that pre-existed
            # in baseref. Knowing that the edges of baseref form a watertight
            # hull, we employ the strategy of first identifying all edge-edge
            # counterparts, and then comparing the new references in the
            # identified locations to see if one of the two disappeared: in
            # this case the other reference is added to the exterior set.

            # NOTE: establishing edge-edge relations could potentially be
            # cached for reuse at the level of baseref. However, since this is
            # the only place that the information is used and all edge pairs
            # need to anyhow be examined for gaps, it is not clear that the
            # gains merit the additional complexity.

            orientation = []
            seen = {}
            for edge1, newemap1, etrans1, newedge1 in zip(baseref.edge_refs, edge_vertices, baseref.edge_transforms, edge_refs):
                newedge1_edge = zip(newedge1.edge_vertices, newedge1.edge_transforms, newedge1.edge_refs)
                trimmed = [] # trimmed will be populated with a subset of newedge1_edge
                for edge2, (newemap2, etrans2, newedge2) in zip(edge1.edge_refs, newedge1_edge):
                    if edge2: # existing non-empty edge

                        # To identify matching edge-edges we map their vertices
                        # to the numbering of the baseref for comparison. Since
                        # matching edge-edges have must have equal references,
                        # and by construction have matching orientation, the
                        # vertex ordering will be consistent between them.

                        key = tuple(newemap1[newemap2])

                        # NOTE: there have been anecdotal reports that suggest
                        # the assumption of matching edges may be violated, but
                        # it is not clear in what scenario this can occur. If
                        # the 'not seen' assertion below fails, please provide
                        # the developers with a reproducable issue for study.

                        try:
                            newedge2_ = seen.pop(key)
                        except KeyError:
                            seen[key] = newedge2
                        else: # a counterpart is found, placing newedge2 against newedge2_
                            if not newedge2:
                                trimmed.append((newemap2, etrans2.flipped, newedge2_))
                            elif not newedge2_:
                                trimmed.append((newemap2, etrans2, newedge2))
                            elif newedge2 != newedge2_:
                                raise NotImplementedError

                # Since newedge1_edge was zipped against the shorter list of
                # original edge1.edge_refs, what remains are the new edge-edges
                # that can be added without further examination.

                trimmed.extend(newedge1_edge)

                # What remains is only to extend the edge-vertex relations and
                # to track if the new edges are left- or right-handed.

                for newemap2, etrans2, newedge2 in trimmed:
                    for simplex in newemap1[newemap2[newedge2.simplices]]:
                        if imidpoint not in simplex:
                            edge_vertices.append(types.frozenarray([imidpoint, *simplex]))
                            orientation.append(not etrans1.isflipped^etrans2.isflipped)

            assert not seen, f'leftover unmatched edges ({seen}) indicate the edges of baseref ({baseref}) are not watertight!'

        self.baseref = baseref
        self._edge_refs = edge_refs
        self.vertices = types.frozenarray(vertices, copy=False)
        self._imidpoint = imidpoint
        self._midpoint = midpoint
        self.edge_vertices = tuple(edge_vertices)
        self.edge_refs = edge_refs + (getsimplex(baseref.ndims-1),) * len(orientation)
        self.edge_transforms = baseref.edge_transforms + tuple(transform.Updim((vertices[1:] - vertices[0]).T, vertices[0], isflipped)
          for vertices, isflipped in zip(self.vertices[numpy.array(edge_vertices[baseref.nedges:])], orientation))

        super().__init__(baseref.ndims)

    def _with_edges(self, edge_refs):
        edge_refs = tuple(edge_refs)
        return self.baseref if edge_refs == self.baseref.edge_refs \
          else self.empty if not any(edge_refs) \
          else MosaicReference(self.baseref, edge_refs, self._midpoint)

    def __and__(self, other):
        if other in (self, self.baseref):
            return self
        if isinstance(other, MosaicReference) and other.baseref == self:
            return other
        if isinstance(other, MosaicReference) and self.baseref == other.baseref and other._midpoint == self._midpoint:
            return self._with_edges(selfedge & otheredge for selfedge, otheredge in zip(self._edge_refs, other._edge_refs))
        return NotImplemented

    def __or__(self, other):
        if other in (self, self.baseref):
            return other
        if isinstance(other, MosaicReference) and self.baseref == other.baseref and other._midpoint == self._midpoint:
            return self._with_edges(selfedge | otheredge for selfedge, otheredge in zip(self._edge_refs, other._edge_refs))
        return NotImplemented

    def __sub__(self, other):
        if other in (self, self.baseref):
            return self.empty
        if isinstance(other, MosaicReference) and other.baseref == self:
            return other._with_edges(baseedge - edge for baseedge, edge in zip(self.edge_refs, other._edge_refs))
        return NotImplemented

    def __rsub__(self, other):
        if other == self.baseref:
            return self._with_edges(baseedge - edge for baseedge, edge in zip(other.edge_refs, self._edge_refs))
        return NotImplemented

    def _nlinear_by_level(self, n):
        return self.baseref._nlinear_by_level(n)

    @property
    def simplices(self):
        indices = []
        for vmap, etrans, eref in zip(self.edge_vertices, self.baseref.edge_transforms, self._edge_refs):
            for index in vmap[eref.simplices]:
                if self._imidpoint not in index:
                    indices.append([self._imidpoint, *index] if not etrans.isflipped else [index[0], self._imidpoint, *index[1:]])
        return types.frozenarray(indices, dtype=int)

    def getpoints(self, ischeme, degree):
        if ischeme == 'vertex':
            return self.baseref.getpoints(ischeme, degree)
        elif ischeme in ('gauss', 'uniform', 'bezier'):
            simplexpoints = getsimplex(self.ndims).getpoints(ischeme, degree)
            subpoints = [points.TransformPoints(simplexpoints, strans) for strans in self.simplex_transforms]
            dups = points.find_duplicates(subpoints) if ischeme == 'bezier' else ()
            return points.ConcatPoints(subpoints, dups)
        else:
            return super().getpoints(ischeme, degree)

    def get_ndofs(self, degree):
        return self.baseref.get_ndofs(degree)

    def get_poly_coeffs(self, basis, **kwargs):
        return self.baseref.get_poly_coeffs(basis, **kwargs)

    def get_edge_dofs(self, degree, iedge):
        return self.baseref.get_edge_dofs(degree, iedge)


# UTILITY FUNCTIONS

def parse_legacy_ischeme(ischeme):
    matches = list(map(re.compile('^([a-zA-Z]+)(.*)$').match, ischeme.split('*')))
    assert all(matches), 'cannot parse integration scheme {!r}'.format(ischeme)
    ischeme = '*'.join(match.group(1) for match in matches)
    degree = eval(','.join(match.group(2) or 'None' for match in matches))
    return ischeme, degree


def getsimplex(ndims):
    Simplex_by_dim = PointReference, LineReference, TriangleReference, TetrahedronReference
    return Simplex_by_dim[ndims]()


def index_or_append(items, item):
    try:
        index = items.index(item)
    except ValueError:
        index = len(items)
        items.append(item)
    return index


def arglexsort(triangulation):
    return numpy.argsort(numeric.asobjvector(tuple(tri) for tri in triangulation))


# vim:sw=4:sts=4:et
