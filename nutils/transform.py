"""
The transform module.
"""

from typing import Tuple, Dict
from numbers import Integral
from . import cache, numeric, _util as util, types
from ._backports import cached_property
import nutils_poly as poly
import numpy
import collections
import itertools
import functools
import operator
_ = numpy.newaxis

TransformChain = Tuple['TransformItem']

# TRANSFORM CHAIN OPERATIONS


def apply(chain, points):
    # NOTE: we explicitly do not lru_cache apply, as doing so would create a
    # cyclic reference when chain is empty or contains only Identity transforms.
    # Instead we rely on the caching of individual transform items.
    for trans in reversed(chain):
        points = trans.apply(points)
    return points


def canonical(chain):
    # keep at lowest ndims possible; this is the required form for bisection
    n = len(chain)
    if n < 2:
        return tuple(chain)
    items = list(chain)
    i = 0
    while items[i].fromdims > items[n-1].fromdims:
        swapped = items[i+1].swapdown(items[i])
        if swapped:
            items[i:i+2] = swapped
            i -= i > 0
        else:
            i += 1
    return tuple(items)


def iscanonical(chain):
    return all(b.swapdown(a) == None for a, b in util.pairwise(chain))


def uppermost(chain):
    # bring to highest ndims possible
    n = len(chain)
    if n < 2:
        return tuple(chain)
    items = list(chain)
    i = n
    while items[i-1].todims < items[0].todims:
        swapped = items[i-2].swapup(items[i-1])
        if swapped:
            items[i-2:i] = swapped
            i += i < n
        else:
            i -= 1
    return tuple(items)


def promote(chain, ndims):
    # swap transformations such that ndims is reached as soon as possible, and
    # then maintained as long as possible (i.e. proceeds as canonical).
    for i, item in enumerate(chain):  # NOTE possible efficiency gain using bisection
        if item.fromdims == ndims:
            return canonical(chain[:i+1]) + uppermost(chain[i+1:])
    return chain  # NOTE at this point promotion essentially failed, maybe it's better to raise an exception

# TRANSFORM ITEMS


class TransformItem(types.Singleton):
    '''Affine transformation.

    Base class for transformations of the type :math:`x ↦ A x + b`.

    Args
    ----
    todims : :class:`int`
        Dimension of the affine transformation domain.
    fromdims : :class:`int`
        Dimension of the affine transformation range.
    '''

    def __init__(self, todims: Integral, fromdims: Integral):
        assert isinstance(todims, Integral), f'todims={todims!r}'
        assert isinstance(fromdims, Integral), f'fromdims={fromdims!r}'
        super().__init__()
        self.todims = todims
        self.fromdims = fromdims

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self)

    def swapup(self, other):
        return None

    def swapdown(self, other):
        return None


class Matrix(TransformItem):
    '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×m` matrix, :math:`n≥m`

    Parameters
    ----------
    linear : :class:`numpy.ndarray`
        The transformation matrix :math:`A`.
    offset : :class:`numpy.ndarray`
        The offset :math:`b`.
    '''

    def __init__(self, linear, offset):
        # we don't worry about mutability here as the meta class prevents
        # direct instantiation from mutable arguments, and derived classes are
        # trusted to not mutate arguments after construction.
        self.linear = numpy.asarray(linear)
        self.offset = numpy.asarray(offset)
        assert self.linear.ndim == 2 and self.linear.dtype == float
        assert self.offset.ndim == 1 and self.offset.dtype == float
        assert self.offset.shape[0] == self.linear.shape[0]
        super().__init__(*self.linear.shape)

    @types.lru_cache
    def apply(self, points):
        assert points.shape[-1] == self.fromdims
        return types.frozenarray(numpy.dot(points, self.linear.T) + self.offset, copy=False)

    def __mul__(self, other):
        assert isinstance(other, Matrix) and self.fromdims == other.todims
        linear = types.arraydata(self.linear @ other.linear)
        offset = types.arraydata(self.apply(other.offset))
        return Square(linear, offset) if self.todims == other.fromdims \
            else Updim(linear, offset, self.isflipped ^ other.isflipped) if self.todims == other.fromdims+1 \
            else Matrix(linear, offset)

    def __str__(self):
        if not hasattr(self, 'offset') or not hasattr(self, 'linear'):
            return '<uninitialized>'
        return util.obj2str(self.offset) + ''.join('+{}*x{}'.format(util.obj2str(v), i) for i, v in enumerate(self.linear.T))


class Square(Matrix):
    '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` square

    Parameters
    ----------
    linear : :class:`numpy.ndarray`
        The transformation matrix :math:`A`.
    offset : :class:`numpy.ndarray`
        The offset :math:`b`.
    '''

    def __init__(self, linear, offset):
        self._transform_matrix = {}
        super().__init__(linear, offset)
        assert self.fromdims == self.todims

    @types.lru_cache
    def invapply(self, points):
        return types.frozenarray(numpy.linalg.solve(self.linear, (points - self.offset).T).T, copy=False)

    @cached_property
    def det(self):
        return numpy.linalg.det(self.linear)

    @property
    def isflipped(self):
        return bool(self.det < 0)

    @types.lru_cache
    def transform_poly(self, coeffs):
        degree = poly.degree(self.fromdims, coeffs.shape[-1])
        try:
            M = self._transform_matrix[degree]
        except KeyError:
            self._transform_matrix[degree] = M = poly.composition_with_inner_matrix(numpy.concatenate([self.offset[:,None], self.linear], axis=1)[:,::-1], self.fromdims, self.fromdims, degree)
        return types.frozenarray(numpy.einsum('ij,...j->...i', M, coeffs), copy=False)


class Identity(Square):
    '''Identity transformation :math:`x ↦ x`

    Parameters
    ----------
    ndims : :class:`int`
        Dimension of :math:`x`.
    '''

    det = 1.

    def __init__(self, ndims: Integral):
        assert isinstance(ndims, Integral) and ndims >= 0, f'ndims={ndims!r}'
        super().__init__(numpy.eye(ndims), numpy.zeros(ndims))

    def apply(self, points):
        return points

    def invapply(self, points):
        return points

    def __str__(self):
        return 'x'


class Index(Identity):
    '''Identity transform with index

    This transformation serves as an element-specific or topology-specific index
    to form the basis of transformation lookups. Otherwise, the transform behaves
    like an identity.
    '''

    def __init__(self, ndims: Integral, index: Integral):
        assert isinstance(ndims, Integral) and ndims >= 0, f'ndims={ndims!r}'
        assert isinstance(index, Integral), f'index={index!r}'
        self.index = index
        super().__init__(ndims)

    def __repr__(self):
        return 'Index({}, {})'.format(self.todims, self.index)


class Updim(Matrix):
    '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×(n-1)` matrix

    Parameters
    ----------
    linear : :class:`numpy.ndarray`
        The transformation matrix :math:`A`.
    offset : :class:`numpy.ndarray`
        The offset :math:`b`.
    '''

    def __init__(self, linear, offset, isflipped: bool):
        assert isinstance(isflipped, bool), f'isflipped={isflipped!r}'
        self._affine = linear, offset
        self.isflipped = isflipped
        super().__init__(linear, offset)
        assert self.todims == self.fromdims + 1

    @cached_property
    def ext(self):
        ext = numeric.ext(self.linear)
        return types.frozenarray(-ext if self.isflipped else ext, copy=False)

    @property
    def flipped(self):
        assert type(self) == Updim
        return Updim(*self._affine, not self.isflipped)

    def swapdown(self, other):
        if isinstance(other, TensorChild):
            return ScaledUpdim(other, self), Identity(self.fromdims)


class SimplexEdge(Updim):

    swap = (
        ((1, 0), (2, 0), (3, 0), (7, 1)),
        ((0, 1), (2, 1), (3, 1), (6, 1)),
        ((0, 2), (1, 2), (3, 2), (5, 1)),
        ((0, 3), (1, 3), (2, 3), (4, 3)),
    )

    def __init__(self, ndims: Integral, iedge: Integral, inverted: bool = False):
        assert isinstance(ndims, Integral) and ndims >= 0, f'ndims={ndims!r}'
        assert isinstance(iedge, Integral) and iedge >= 0, f'iedge={iedge!r}'
        assert isinstance(inverted, bool), f'inverted={inverted!r}'
        assert ndims >= iedge >= 0
        self.iedge = iedge
        self.inverted = inverted
        vertices = numpy.concatenate([numpy.zeros(ndims)[_, :], numpy.eye(ndims)], axis=0)
        coords = vertices[list(range(iedge))+list(range(iedge+1, ndims+1))]
        super().__init__((coords[1:]-coords[0]).T, coords[0], inverted ^ bool(iedge % 2))

    @property
    def flipped(self):
        assert type(self) == SimplexEdge
        return SimplexEdge(self.todims, self.iedge, not self.inverted)

    def swapup(self, other):
        # prioritize ascending transformations, i.e. change updim << scale to scale << updim
        if isinstance(other, SimplexChild):
            ichild, iedge = self.swap[self.iedge][other.ichild]
            return SimplexChild(self.todims, ichild), SimplexEdge(self.todims, iedge, self.inverted)

    def swapdown(self, other):
        # prioritize decending transformations, i.e. change scale << updim to updim << scale
        if isinstance(other, SimplexChild):
            key = other.ichild, self.iedge
            for iedge, children in enumerate(self.swap[:self.todims+1]):
                try:
                    ichild = children[:2**self.fromdims].index(key)
                except ValueError:
                    pass
                else:
                    return SimplexEdge(self.todims, iedge, self.inverted), SimplexChild(self.fromdims, ichild)


class SimplexChild(Square):

    def __init__(self, ndims: Integral, ichild: Integral):
        assert isinstance(ndims, Integral) and ndims >= 0, f'ndims={ndims!r}'
        assert isinstance(ichild, Integral) and ichild >= 0, f'ichild={ichild!r}'
        self.ichild = ichild
        if ichild <= ndims:
            linear = numpy.eye(ndims) * .5
            offset = linear[ichild-1] if ichild else numpy.zeros(ndims)
        elif ndims == 2 and ichild == 3:
            linear = (-.5, 0), (.5, .5)
            offset = .5, 0
        elif ndims == 3 and ichild == 4:
            linear = (-.5, 0, -.5), (.5, .5, 0), (0, 0, .5)
            offset = .5, 0, 0
        elif ndims == 3 and ichild == 5:
            linear = (0, -.5, 0), (.5, 0, 0), (0, .5, .5)
            offset = .5, 0, 0
        elif ndims == 3 and ichild == 6:
            linear = (.5, 0, 0), (0, -.5, 0), (0, .5, .5)
            offset = 0, .5, 0
        elif ndims == 3 and ichild == 7:
            linear = (-.5, 0, -.5), (-.5, -.5, 0), (.5, .5, .5)
            offset = .5, .5, 0
        else:
            raise NotImplementedError('SimplexChild(ndims={}, ichild={})'.format(ndims, ichild))
        super().__init__(linear, offset)


class ScaledUpdim(Updim):

    def __init__(self, trans1: Square, trans2: Updim):
        assert isinstance(trans1, Square), f'trans1={trans1!r}'
        assert isinstance(trans2, Updim), f'trans2={trans2!r}'
        assert trans1.fromdims == trans2.todims
        self.trans1 = trans1
        self.trans2 = trans2
        super().__init__(numpy.dot(trans1.linear, trans2.linear), trans1.apply(trans2.offset), trans1.isflipped ^ trans2.isflipped)

    def swapup(self, other):
        if type(other) is Identity:
            return self.trans1, self.trans2

    @property
    def flipped(self):
        assert type(self) == ScaledUpdim
        return ScaledUpdim(self.trans1, self.trans2.flipped)


class TensorEdge1(Updim):

    def __init__(self, trans1: Updim, ndims2: Integral):
        assert isinstance(trans1, Updim), f'trans1={trans1!r}'
        assert isinstance(ndims2, Integral), f'trans2={trans2!r}'
        self.trans = trans1
        super().__init__(linear=numeric.blockdiag([trans1.linear, numpy.eye(ndims2)]), offset=numpy.concatenate([trans1.offset, numpy.zeros(ndims2)]), isflipped=trans1.isflipped)

    def swapup(self, other):
        # prioritize ascending transformations, i.e. change updim << scale to scale << updim
        if isinstance(other, TensorChild) and self.trans.fromdims == other.trans1.todims:
            swapped = self.trans.swapup(other.trans1)
            trans2 = other.trans2
        elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdims == other.todims and not self.trans.fromdims:
            swapped = self.trans.swapup(SimplexChild(0, 0))
            trans2 = other
        else:
            swapped = None
        if swapped:
            child, edge = swapped
            return TensorChild(child, trans2), TensorEdge1(edge, trans2.fromdims)

    def swapdown(self, other):
        # prioritize ascending transformations, i.e. change scale << updim to updim << scale
        if isinstance(other, TensorChild) and other.trans1.fromdims == self.trans.todims:
            swapped = self.trans.swapdown(other.trans1)
            if swapped:
                edge, child = swapped
                return TensorEdge1(edge, other.trans2.todims), TensorChild(child, other.trans2) if child.fromdims else other.trans2
            return ScaledUpdim(other, self), Identity(self.fromdims)

    @property
    def flipped(self):
        assert type(self) == TensorEdge1
        return TensorEdge1(self.trans.flipped, self.fromdims-self.trans.fromdims)


class TensorEdge2(Updim):

    def __init__(self, ndims1: Integral, trans2: Updim):
        assert isinstance(ndims1, Integral) and ndims1 >= 0, f'ndims1={ndims1!r}'
        assert isinstance(trans2, Updim), f'trans2={trans2!r}'
        self.trans = trans2
        super().__init__(linear=numeric.blockdiag([numpy.eye(ndims1), trans2.linear]), offset=numpy.concatenate([numpy.zeros(ndims1), trans2.offset]), isflipped=trans2.isflipped ^ bool(ndims1 % 2))

    def swapup(self, other):
        # prioritize ascending transformations, i.e. change updim << scale to scale << updim
        if isinstance(other, TensorChild) and self.trans.fromdims == other.trans2.todims:
            swapped = self.trans.swapup(other.trans2)
            trans1 = other.trans1
        elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdims == other.todims and not self.trans.fromdims:
            swapped = self.trans.swapup(SimplexChild(0, 0))
            trans1 = other
        else:
            swapped = None
        if swapped:
            child, edge = swapped
            return TensorChild(trans1, child), TensorEdge2(trans1.fromdims, edge)

    def swapdown(self, other):
        # prioritize ascending transformations, i.e. change scale << updim to updim << scale
        if isinstance(other, TensorChild) and other.trans2.fromdims == self.trans.todims:
            swapped = self.trans.swapdown(other.trans2)
            if swapped:
                edge, child = swapped
                return TensorEdge2(other.trans1.todims, edge), TensorChild(other.trans1, child) if child.fromdims else other.trans1
            return ScaledUpdim(other, self), Identity(self.fromdims)

    @property
    def flipped(self):
        assert type(self) == TensorEdge2
        return TensorEdge2(self.fromdims-self.trans.fromdims, self.trans.flipped)


class TensorChild(Square):

    def __init__(self, trans1: Square, trans2: Square):
        assert isinstance(trans1, Square), f'trans1={trans1!r}'
        assert isinstance(trans2, Square), f'trans2={trans2!r}'
        self.trans1 = trans1
        self.trans2 = trans2
        linear = numeric.blockdiag([trans1.linear, trans2.linear])
        offset = numpy.concatenate([trans1.offset, trans2.offset])
        super().__init__(linear, offset)

    @cached_property
    def det(self):
        return self.trans1.det * self.trans2.det


class Point(Matrix):

    def __init__(self, offset):
        offset = numpy.asarray(offset)
        super().__init__(numpy.zeros((offset.shape[0], 0)), offset)


def simplex(vertices, isflipped = None):
    '''Create transform item from simplex vertices.'''

    linear = types.arraydata((vertices[1:] - vertices[0]).T)
    offset = types.arraydata(vertices[0])
    if isflipped is None:
        return Square(linear, offset)
    else:
        return Updim(linear, offset, isflipped)


# vim:sw=4:sts=4:et
