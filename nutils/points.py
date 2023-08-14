'''
The points module defines the :class:`Points` base class, which bundles point
coordinates, point weights, a local triangulation and a hull triangulation. The
module provides several different implementations such as :class:`TensorPoints`
and :class:`SimplexGaussPoints` that reflect the variety of elements in the
:mod:`nutils.element` module.
'''

from . import types, transform, numeric, _util as util
from ._backports import cached_property
from typing import Tuple, FrozenSet
from numbers import Integral
import numpy
import functools
import itertools
import warnings
import math
_ = numpy.newaxis


class Points(types.Singleton):
    '''Collection of points on an element.

    The :class:`Points` base class bundles point coordinates, point weights,
    a local triangulation and hull triangulation. Of these only the coordinates
    are mandatory, and should be provided by the derived class in the form of the
    ``coords`` attribute. Of the remaining properties only :func:`hull` has a
    functional base implementation that relies on the availability of ``tri``.

    .. attribute:: coords

      Coordinates of the points as a :class:`float` array.

    .. attribute:: weights

      Weights of the points as a :class:`float` array.

    Args
    ----
    npoints : :class:`int`
      Number of discrete points.
    ndims : :class:`int`
      Number of spatial dimensions.
    '''

    def __init__(self, npoints: Integral, ndims: Integral):
        assert isinstance(npoints, Integral), f'npoints={npoints!r}'
        assert isinstance(ndims, Integral), f'ndims={ndims!r}'
        self.npoints = npoints
        self.ndims = ndims

    def __mul__(self, other):
        '''Return ``self*other``.'''

        if not isinstance(other, Points):
            return NotImplemented
        return self.product(other)

    def product(self, other):
        '''Return the product with ``other``.

        Parameters
        ----------
        other : :class:`Points`

        Returns
        -------
        product : :class:`Points`
            The product.
        '''

        return TensorPoints(self, other)

    @cached_property
    def tri(self):
        '''Triangulation of interior.

        A two-dimensional integer array with ``ndims+1`` columns, of which every
        row defines a simplex by mapping vertices into the list of points.
        '''

        if self.ndims == 0 and self.npoints == 1:
            return types.frozenarray([[0]])
        raise Exception('tri not defined for {}'.format(self))

    @cached_property
    def hull(self):
        '''Triangulation of the exterior hull.

        A two-dimensional integer array with ``ndims`` columns, of which every row
        defines a simplex by mapping vertices into the list of points.
        '''

        assert self.ndims > 0, 'hull is not defined for a zero-dimensional point set'
        edge_vertices = numpy.arange(self.ndims+1).repeat(self.ndims).reshape(self.ndims, self.ndims+1).T  # ndims+1 x ndims
        edge_simplices = numpy.sort(self.tri, axis=1)[:, edge_vertices]  # nelems x ndims+1 x ndims
        elems, edges = divmod(numpy.lexsort(edge_simplices.reshape(-1, self.ndims).T), self.ndims+1)
        sorted_edge_simplices = edge_simplices[elems, edges]  # (nelems x ndims+1) x ndims; matching edges are now adjacent
        notequal = numpy.not_equal(sorted_edge_simplices[1:], sorted_edge_simplices[:-1]).any(axis=1)
        return types.frozenarray(sorted_edge_simplices[numpy.hstack([True, notequal]) & numpy.hstack([notequal, True])], copy=False)

    @cached_property
    def onhull(self):
        '''Boolean mask marking boundary points.

        The array of length ``npoints`` is ``True`` where the corresponding point
        is part of the :attr:`hull`, and ``False`` where it is not.
        '''

        onhull = numpy.zeros(self.npoints, dtype=bool)
        onhull[numpy.ravel(self.hull)] = True  # not clear why ravel is necessary but setitem seems to require it
        return types.frozenarray(onhull, copy=False)


class CoordsPoints(Points):
    '''Manually supplied points.'''

    def __init__(self, coords: types.arraydata):
        assert isinstance(coords, types.arraydata) and coords.ndim == 2 and coords.dtype == float, f'coords={coords!r}'
        self.coords = numpy.asarray(coords)
        super().__init__(*coords.shape)


class CoordsWeightsPoints(CoordsPoints):
    '''Manually supplied points and weights.'''

    def __init__(self, coords: types.arraydata, weights: types.arraydata):
        assert isinstance(coords, types.arraydata) and coords.ndim == 2 and coords.dtype == float, f'coords={coords!r}'
        assert isinstance(weights, types.arraydata) and weights.ndim == 1 and weights.dtype == float, f'weights={weights!r}'
        self.weights = numpy.asarray(weights)
        super().__init__(coords)


class CoordsUniformPoints(CoordsPoints):
    '''Manually supplied points with uniform weights.'''

    def __init__(self, coords: types.arraydata, volume: float):
        assert isinstance(coords, types.arraydata) and coords.ndim == 2 and coords.dtype == float, f'coords={coords!r}'
        assert isinstance(volume, float), f'volume={volume!r}'
        self.weights = numeric.full(coords.shape[:1], fill_value=volume/coords.shape[0], dtype=float)
        super().__init__(coords)


class TensorPoints(Points):
    '''Tensor product of two Points instances.'''

    def __init__(self, points1: Points, points2: Points):
        assert isinstance(points1, Points), f'points1={points1!r}'
        assert isinstance(points2, Points), f'points2={points2!r}'
        self.points1 = points1
        self.points2 = points2
        super().__init__(points1.npoints * points2.npoints, points1.ndims + points2.ndims)

    @cached_property
    def coords(self):
        coords = numpy.empty((self.points1.npoints, self.points2.npoints, self.ndims))
        coords[:, :, :self.points1.ndims] = self.points1.coords[:, _, :]
        coords[:, :, self.points1.ndims:] = self.points2.coords[_, :, :]
        return types.frozenarray(coords.reshape(self.npoints, self.ndims), copy=False)

    @cached_property
    def weights(self):
        return types.frozenarray((self.points1.weights[:, _] * self.points2.weights[_, :]).ravel(), copy=False)

    @cached_property
    def tri(self):
        if self.points1.npoints == 1:
            return self.points2.tri
        elif self.points2.npoints == 1:
            return self.points1.tri
        elif self.points1.ndims == 1:
            # For an n-dimensional simplex with vertices a0,a1,..,an, the extruded
            # element has vertices a0,a1,..,an,b0,b1,..,bn. These can be divided in
            # simplices by selecting a0,a1,..,an,b0; a1,..,an,b0,n1; and so on until
            # an,b0,b1,..,bn; resulting in n+1 n+1-dimensional simplices. In the
            # algorithm below this is achieved by first taking the tensorial product
            # of triangulations and raveling, effectively achieving vectorized
            # concatenation. The overlapping vertex subsets then follow directly from
            # numeric.overlapping.
            tri12 = self.points1.tri[:, _, :, _] * self.points2.npoints + self.points2.tri[_, :, _, :]  # ntri1 x ntri2 x 2 x ndims
            return types.frozenarray(numeric.overlapping(tri12.reshape(-1, 2*self.ndims), n=self.ndims+1).reshape(-1, self.ndims+1), copy=False)
        else:
            return super().tri.func()

    @cached_property
    def hull(self):
        if self.points1.npoints == 1:
            return self.points2.hull
        elif self.points2.npoints == 1:
            return self.points1.hull
        elif self.points1.ndims == 1:
            hull1 = self.points1.hull[:, _, :, _] * self.points2.npoints + self.points2.tri[_, :, _, :]  # 2 x ntri2 x 1 x ndims
            hull2 = self.points1.tri[:, _, :, _] * self.points2.npoints + self.points2.hull[_, :, _, :]  # ntri1 x nhull2 x 2 x ndims-1
            # The subdivision of hull2 into simplices follows identical logic to that
            # used in the construction of self.tri.
            hull = numpy.concatenate([hull1.reshape(-1, self.ndims), numeric.overlapping(hull2.reshape(-1, 2*(self.ndims-1)), n=self.ndims).reshape(-1, self.ndims)])
            return types.frozenarray(hull, copy=False)
        else:
            return super().hull.func()

    def product(self, other):
        return self.points1.product(self.points2.product(other))


class SimplexGaussPoints(CoordsWeightsPoints):
    '''Gauss quadrature points on a simplex.'''

    def __init__(self, ndims: Integral, degree: Integral):
        assert isinstance(ndims, Integral), f'ndims={ndims!r}'
        assert isinstance(degree, Integral), f'degree={degree!r}'
        super().__init__(*gaussn[ndims](degree))


class SimplexBezierPoints(CoordsUniformPoints):
    '''Bezier points on a simplex.'''

    def __init__(self, ndims: Integral, n: Integral):
        assert isinstance(ndims, Integral), f'ndims={ndims!r}'
        assert isinstance(n, Integral), f'n={n!r}'
        self.n = n
        self._indices = numpy.array([index[::-1] for index in numpy.ndindex(*[n] * ndims) if sum(index) < n])
        super().__init__(types.arraydata(self._indices/(n-1)), 1/math.factorial(ndims))

    @property
    def _indexgrid(self):
        grid = numpy.full([self.n]*self.ndims, self.npoints, dtype=int)  # initialize with out of bounds
        grid[tuple(self._indices.T)] = numpy.arange(self.npoints)
        return grid

    @cached_property
    def tri(self):
        if self.n == 2:
            tri = numpy.arange(self.npoints)[_]
        elif self.ndims == 1:
            tri = numeric.overlapping(self._indexgrid)
        elif self.ndims == 2:
            grid = self._indexgrid
            tri = [grid[[i, i+1, i], [j, j, j+1]] for i in range(self.n-1) for j in range(self.n-i-1)] \
                + [grid[[i+1, i+1, i], [j, j+1, j+1]] for i in range(self.n-1) for j in range(self.n-i-2)]
        elif self.ndims == 3:
            grid = self._indexgrid
            aaa, aab, aba, abb, baa, bab, bba, bbb = [grid[i:i+self.n-1, j:j+self.n-1, k:k+self.n-1] for i in range(2) for j in range(2) for k in range(2)]
            tri = numpy.array([  # we devide every cube in 6 tetrahedra, subject to some constraints
                [bba, abb, bab, bbb],  # 1: required in order to introduce the x+y+z=1 cutting plane
                [aaa, aab, baa, aba],  # 2: required for opposing plane symmetry
                [bba, abb, bab, aab],  # 3-6: arbitrarily connected to vertex aab
                [baa, bab, bba, aab],
                [bba, aba, baa, aab],
                [aba, abb, bba, aab]])
            ws, wx, wy, wz = (tri < self.npoints).all(axis=1).nonzero()
            tri = tri[ws, :, wx, wy, wz]  # remove all x+y+z>1 simplices
        else:
            return super().tri.func()
        return types.frozenarray(tri, copy=False)


class TransformPoints(Points):
    '''Affinely transformed Points.'''

    def __init__(self, points: Points, trans: transform.TransformItem):
        assert isinstance(points, Points), f'points={points!r}'
        assert isinstance(trans, transform.TransformItem), f'trans={trans!r}'
        self.points = points
        self.trans = trans
        super().__init__(points.npoints, points.ndims)

    @cached_property
    def coords(self):
        return self.trans.apply(self.points.coords)

    @cached_property
    def weights(self):
        return types.frozenarray(self.points.weights * abs(float(self.trans.det)), copy=False)

    @property
    def tri(self):
        return self.points.tri

    @property
    def hull(self):
        return self.points.hull


class ConcatPoints(Points):
    '''Concatenation of several Points objects.

    An optional ``duplicates`` argument lists all points that are equal,
    triggering deduplication and resulting in a smaller total point count.
    '''

    def __init__(self, allpoints: Tuple[Points,...], duplicates: FrozenSet[Tuple[Tuple[Integral,Integral],...]] = frozenset()):
        assert isinstance(allpoints, tuple) and all(isinstance(p, Points) for p in allpoints), 'allpoints={allpoints!r}'
        assert isinstance(duplicates, frozenset) and all(isinstance(d, tuple) and all(isinstance(n, tuple) and len(n) == 2 and all(isinstance(ni, Integral) for ni in n) for n in d) for d in duplicates), f'duplicates={duplicates!r}'
        self.allpoints = allpoints
        self.duplicates = duplicates
        super().__init__(sum(points.npoints for points in allpoints) - sum(len(d)-1 for d in duplicates), allpoints[0].ndims)

    @cached_property
    def masks(self):
        masks = [numpy.ones(points.npoints, dtype=bool) for points in self.allpoints]
        for pairs in self.duplicates:
            for i, j in pairs[1:]:
                masks[i][j] = False
        return tuple(types.frozenarray(m, copy=False) for m in masks)

    @cached_property
    def coords(self):
        return types.frozenarray(numpy.concatenate([points.coords[mask] for mask, points in zip(self.masks, self.allpoints)] if self.duplicates else [points.coords for points in self.allpoints]), copy=False)

    @cached_property
    def weights(self):
        if not self.duplicates:
            return types.frozenarray(numpy.concatenate([points.weights for points in self.allpoints]), copy=False)
        weights = [points.weights[mask] for mask, points in zip(self.masks, self.allpoints)]
        for pairs in self.duplicates:
            I, J = pairs[0]
            weights[I][self.masks[I][:J].sum()] += sum(self.allpoints[i].weights[j] for i, j in pairs[1:])
        return types.frozenarray(numpy.concatenate(weights), copy=False)

    @cached_property
    def tri(self):
        if not self.duplicates:
            offsets = util.cumsum(points.npoints for points in self.allpoints)
            return types.frozenarray(numpy.concatenate([points.tri + offset for offset, points in zip(offsets, self.allpoints)]), copy=False)
        renumber = []
        n = 0
        for mask in self.masks:
            cumsum = mask.cumsum()
            renumber.append(cumsum+(n-1))
            n += cumsum[-1]
        assert n == self.npoints
        for pairs in self.duplicates:
            I, J = pairs[0]
            for i, j in pairs[1:]:
                renumber[i][j] = renumber[I][J]
        return types.frozenarray(numpy.concatenate([renum.take(points.tri) for renum, points in zip(renumber, self.allpoints)]), copy=False)


# UTILITY FUNCTIONS


@functools.lru_cache(8)
def gauss(n):
    k = numpy.arange(n) + 1
    d = k / numpy.sqrt(4*k**2-1)
    x, w = numpy.linalg.eigh(numpy.diagflat(d, -1))  # eigh operates (by default) on lower triangle
    return types.arraydata((x+1) * .5), types.arraydata(w[0]**2)


def gauss1(degree):
    '''Gauss quadrature for line.'''

    x, w = gauss(degree//2)
    return x.reshape(*x.shape, 1), w


@functools.lru_cache(8)
def gauss2(degree):
    '''Gauss quadrature for triangle.

    Reference: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''

    assert isinstance(degree, int) and degree >= 0

    I = [0, 0],
    J = [1, 1], [0, 1], [1, 0]
    K = [1, 2], [2, 0], [0, 1], [2, 1], [1, 0], [0, 2]

    icw = [
        (I, [1/3], 1)
    ] if degree <= 1 else [
        (J, [2/3, 1/6], 1/3)
    ] if degree == 2 else [
        (I, [1/3], -9/16),
        (J, [3/5, 1/5], 25/48),
    ] if degree == 3 else [
        (J, [0.816847572980458, 0.091576213509771], 0.109951743655322),
        (J, [0.108103018168070, 0.445948490915965], 0.223381589678011),
    ] if degree == 4 else [
        (I, [1/3], 0.225),
        (J, [0.797426985353088, 0.101286507323456], 0.125939180544827),
        (J, [0.059715871789770, 0.470142064105115], 0.132394152788506),
    ] if degree == 5 else [
        (J, [0.873821971016996, 0.063089014491502], 0.050844906370207),
        (J, [0.501426509658180, 0.249286745170910], 0.116786275726379),
        (K, [0.636502499121399, 0.310352451033785, 0.053145049844816], 0.082851075618374),
    ] if degree == 6 else [
        (I, [1/3.], -0.149570044467671),
        (J, [0.479308067841924, 0.260345966079038], 0.175615257433204),
        (J, [0.869739794195568, 0.065130102902216], 0.053347235608839),
        (K, [0.638444188569809, 0.312865496004875, 0.048690315425316], 0.077113760890257),
    ]

    if degree > 6:
        warnings.warn('inexact integration for polynomial of degree {}'.format(degree))

    return types.arraydata(numpy.concatenate([numpy.take(c, i) for i, c, w in icw])), \
        types.arraydata(numpy.concatenate([[w/2] * len(i) for i, c, w in icw]))


@functools.lru_cache(8)
def gauss3(degree):
    '''Gauss quadrature for tetrahedron.

    Reference http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''

    assert isinstance(degree, int) and degree >= 0

    I = [0, 0, 0],
    J = [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]
    K = [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    L = [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [0, 1, 2], [2, 0, 1]

    icw = [
        (I, [1/4], 1),
    ] if degree <= 1 else [
        (J, [0.5854101966249685, 0.1381966011250105], 1/4),
    ] if degree == 2 else [
        (I, [.25], -.8),
        (J, [.5, 1/6], .45),
    ] if degree == 3 else [
        (I, [.25], -.2368/3),
        (J, [0.7857142857142857, 0.0714285714285714], .1372/3),
        (K, [0.1005964238332008, 0.3994035761667992], .448/3),
    ] if degree == 4 else [
        (I, [.25], 0.1817020685825351),
        (J, [0, 1/3.], 0.0361607142857143),
        (J, [8/11., 1/11.], 0.0698714945161738),
        (K, [0.4334498464263357, 0.0665501535736643], 0.0656948493683187),
    ] if degree == 5 else [
        (J, [0.3561913862225449, 0.2146028712591517], 0.0399227502581679),
        (J, [0.8779781243961660, 0.0406739585346113], 0.0100772110553207),
        (J, [0.0329863295731731, 0.3223378901422757], 0.0553571815436544),
        (L, [0.2696723314583159, 0.0636610018750175, 0.6030056647916491], 0.0482142857142857),
    ] if degree == 6 else [
        (I, [.25], 0.1095853407966528),
        (J, [0.7653604230090441, 0.0782131923303186],  0.0635996491464850),
        (J, [0.6344703500082868, 0.1218432166639044], -0.3751064406859797),
        (J, [0.0023825066607383, 0.3325391644464206],  0.0293485515784412),
        (K, [0, .5], 0.0058201058201058),
        (L, [.2, .1, .6], 0.1653439153439105)
    ] if degree == 7 else [
        (I, [.25], -0.2359620398477557),
        (J, [0.6175871903000830, 0.1274709365666390], 0.0244878963560562),
        (J, [0.9037635088221031, 0.0320788303926323], 0.0039485206398261),
        (K, [0.4502229043567190, 0.0497770956432810], 0.0263055529507371),
        (K, [0.3162695526014501, 0.1837304473985499], 0.0829803830550589),
        (L, [0.0229177878448171, 0.2319010893971509, 0.5132800333608811], 0.0254426245481023),
        (L, [0.7303134278075384, 0.0379700484718286, 0.1937464752488044], 0.0134324384376852),
    ]

    if degree > 7:
        warnings.warn('inexact integration for polynomial of degree {}'.format(degree))

    return types.arraydata(numpy.concatenate([numpy.take(c, i) for i, c, w in icw])), \
        types.arraydata(numpy.concatenate([[w/6] * len(i) for i, c, w in icw]))


gaussn = None, gauss1, gauss2, gauss3


def find_duplicates(allpoints):
    coords = {}
    for i, points in enumerate(allpoints):
        for j in points.onhull.nonzero()[0]:
            coords.setdefault(tuple(points.coords[j]), []).append((i, j))
    return frozenset(tuple(pairs) for pairs in coords.values() if len(pairs) > 1)

# vim:sw=4:sts=4:et
