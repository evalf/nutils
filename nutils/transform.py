"""
The transform module.
"""

from typing import Tuple, Dict
from . import cache, numeric, _util as util, types, evaluable
from .evaluable import Evaluable, Array
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

    __slots__ = 'todims', 'fromdims'

    @types.apply_annotations
    def __init__(self, todims, fromdims: int):
        super().__init__()
        self.todims = todims
        self.fromdims = fromdims

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self)

    def swapup(self, other):
        return None

    def swapdown(self, other):
        return None


stricttransformitem = types.strict[TransformItem]
stricttransform = types.tuple[stricttransformitem]


class Matrix(TransformItem):
    '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×m` matrix, :math:`n≥m`

    Parameters
    ----------
    linear : :class:`numpy.ndarray`
        The transformation matrix :math:`A`.
    offset : :class:`numpy.ndarray`
        The offset :math:`b`.
    '''

    __slots__ = 'linear', 'offset'

    @types.apply_annotations
    def __init__(self, linear: types.arraydata, offset: types.arraydata):
        assert linear.ndim == 2 and linear.dtype == float
        assert offset.ndim == 1 and offset.dtype == float
        assert offset.shape[0] == linear.shape[0]
        self.linear = numpy.asarray(linear)
        self.offset = numpy.asarray(offset)
        super().__init__(linear.shape[0], linear.shape[1])

    @types.lru_cache
    def apply(self, points):
        assert points.shape[-1] == self.fromdims
        return types.frozenarray(numpy.dot(points, self.linear.T) + self.offset, copy=False)

    def __mul__(self, other):
        assert isinstance(other, Matrix) and self.fromdims == other.todims
        linear = numpy.dot(self.linear, other.linear)
        offset = self.apply(other.offset)
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

    __slots__ = '_transform_matrix',
    __cache__ = 'det',

    @types.apply_annotations
    def __init__(self, linear: types.arraydata, offset: types.arraydata):
        assert linear.shape[0] == linear.shape[1]
        self._transform_matrix = {}
        super().__init__(linear, offset)

    @types.lru_cache
    def invapply(self, points):
        return types.frozenarray(numpy.linalg.solve(self.linear, (points - self.offset).T).T, copy=False)

    @property
    def det(self):
        return numpy.linalg.det(self.linear)

    @property
    def isflipped(self):
        return self.fromdims > 0 and self.det < 0

    @types.lru_cache
    def transform_poly(self, coeffs):
        degree = poly.degree(coeffs.shape[-1], self.fromdims)
        try:
            M = self._transform_matrix[degree]
        except KeyError:
            self._transform_matrix[degree] = M = poly.transform_matrix(numpy.concatenate([self.offset[:,None], self.linear], axis=1)[:,::-1], self.fromdims, degree, self.fromdims).T
        return types.frozenarray(numpy.einsum('ij,...j->...i', M, coeffs), copy=False)


class Identity(Square):
    '''Identity transformation :math:`x ↦ x`

    Parameters
    ----------
    ndims : :class:`int`
        Dimension of :math:`x`.
    '''

    __slots__ = ()

    det = 1.

    def __init__(self, ndims):
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

    __slots__ = 'index'

    @types.apply_annotations
    def __init__(self, ndims: int, index: int):
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

    __slots__ = 'isflipped',
    __cache__ = 'ext',

    @types.apply_annotations
    def __init__(self, linear: types.arraydata, offset: types.arraydata, isflipped: bool):
        assert linear.shape[0] == linear.shape[1] + 1
        self.isflipped = isflipped
        super().__init__(linear, offset)

    @property
    def ext(self):
        ext = numeric.ext(self.linear)
        return types.frozenarray(-ext if self.isflipped else ext, copy=False)

    @property
    def flipped(self):
        return Updim(self.linear, self.offset, not self.isflipped)

    def swapdown(self, other):
        if isinstance(other, TensorChild):
            return ScaledUpdim(other, self), Identity(self.fromdims)


class SimplexEdge(Updim):

    __slots__ = 'iedge', 'inverted'

    swap = (
        ((1, 0), (2, 0), (3, 0), (7, 1)),
        ((0, 1), (2, 1), (3, 1), (6, 1)),
        ((0, 2), (1, 2), (3, 2), (5, 1)),
        ((0, 3), (1, 3), (2, 3), (4, 3)),
    )

    @types.apply_annotations
    def __init__(self, ndims: types.strictint, iedge: types.strictint, inverted: bool = False):
        assert ndims >= iedge >= 0
        self.iedge = iedge
        self.inverted = inverted
        vertices = numpy.concatenate([numpy.zeros(ndims)[_, :], numpy.eye(ndims)], axis=0)
        coords = vertices[list(range(iedge))+list(range(iedge+1, ndims+1))]
        super().__init__((coords[1:]-coords[0]).T, coords[0], inverted ^ (iedge % 2))

    @property
    def flipped(self):
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

    __slots__ = 'ichild',

    def __init__(self, ndims, ichild):
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

    __slots__ = 'trans1', 'trans2'

    def __init__(self, trans1, trans2):
        assert trans1.todims == trans1.fromdims == trans2.todims == trans2.fromdims + 1
        self.trans1 = trans1
        self.trans2 = trans2
        super().__init__(numpy.dot(trans1.linear, trans2.linear), trans1.apply(trans2.offset), trans1.isflipped ^ trans2.isflipped)

    def swapup(self, other):
        if type(other) is Identity:
            return self.trans1, self.trans2

    @property
    def flipped(self):
        return ScaledUpdim(self.trans1, self.trans2.flipped)


class TensorEdge1(Updim):

    __slots__ = 'trans',

    def __init__(self, trans1, ndims2):
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
        return TensorEdge1(self.trans.flipped, self.fromdims-self.trans.fromdims)


class TensorEdge2(Updim):

    __slots__ = 'trans'

    def __init__(self, ndims1, trans2):
        self.trans = trans2
        super().__init__(linear=numeric.blockdiag([numpy.eye(ndims1), trans2.linear]), offset=numpy.concatenate([numpy.zeros(ndims1), trans2.offset]), isflipped=trans2.isflipped ^ (ndims1 % 2))

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
        return TensorEdge2(self.fromdims-self.trans.fromdims, self.trans.flipped)


class TensorChild(Square):

    __slots__ = 'trans1', 'trans2'
    __cache__ = 'det',

    def __init__(self, trans1, trans2):
        assert trans1.fromdims and trans2.fromdims
        self.trans1 = trans1
        self.trans2 = trans2
        linear = numeric.blockdiag([trans1.linear, trans2.linear])
        offset = numpy.concatenate([trans1.offset, trans2.offset])
        super().__init__(linear, offset)

    @property
    def det(self):
        return self.trans1.det * self.trans2.det


class Point(Matrix):

    @types.apply_annotations
    def __init__(self, offset: types.arraydata):
        super().__init__(numpy.zeros((offset.shape[0], 0)), offset)

# EVALUABLE TRANSFORM CHAIN


class EvaluableTransformChain(Evaluable):
    '''The :class:`~nutils.evaluable.Evaluable` equivalent of a transform chain.

    Attributes
    ----------
    todims : :class:`int`
        The to dimension of the transform chain.
    fromdims : :class:`int`
        The from dimension of the transform chain.
    '''

    __slots__ = 'todims', 'fromdims'

    @staticmethod
    def empty(__dim: int) -> 'EvaluableTransformChain':
        '''Return an empty evaluable transform chain with the given dimension.

        Parameters
        ----------
        dim : :class:`int`
            The to and from dimensions of the empty transform chain.

        Returns
        -------
        :class:`EvaluableTransformChain`
            The empty evaluable transform chain.
        '''

        return _EmptyTransformChain(__dim)

    @staticmethod
    def from_argument(name: str, todims: int, fromdims: int) -> 'EvaluableTransformChain':
        '''Return an evaluable transform chain that evaluates to the given argument.

        Parameters
        ----------
        name : :class:`str`
            The name of the argument.
        todims : :class:`int`
            The to dimension of the transform chain.
        fromdims: :class:`int`
            The from dimension of the transform chain.

        Returns
        -------
        :class:`EvaluableTransformChain`
            The transform chain that evaluates to the given argument.
        '''

        return _TransformChainArgument(name, todims, fromdims)

    def __init__(self, args: Tuple[Evaluable, ...], todims: int, fromdims: int) -> None:
        if fromdims > todims:
            raise ValueError('The dimension of the tip cannot be larger than the dimension of the root.')
        self.todims = todims
        self.fromdims = fromdims
        super().__init__(args)

    @property
    def linear(self) -> Array:
        ':class:`nutils.evaluable.Array`: The linear transformation matrix of the entire transform chain. Shape ``(todims,fromdims)``.'

        return _Linear(self)

    @property
    def basis(self) -> Array:
        ':class:`nutils.evaluable.Array`: A basis for the root coordinate system such that the first :attr:`fromdims` vectors span the tangent space. Shape ``(todims,todims)``.'

        if self.fromdims == self.todims:
            return evaluable.diagonalize(evaluable.ones((self.todims,)))
        else:
            return _Basis(self)

    def apply(self, __coords: Array) -> Array:
        '''Apply this transform chain to the last axis given coordinates.

        Parameters
        ----------
        coords : :class:`nutils.evaluable.Array`
            The coordinates to transform with shape ``(...,fromdims)``.

        Returns
        -------
        :class:`nutils.evaluable.Array`
            The transformed coordinates with shape ``(...,todims)``.
        '''

        return _Apply(self, __coords)

    def index_with_tail_in(self, __sequence: 'Transforms') -> Tuple[Array, 'EvaluableTransformChain']:
        '''Return the evaluable index of this transform chain in the given sequence.

        Parameters
        ----------
        sequence : :class:`nutils.transformseq.Transforms`
            The sequence of transform chains.

        Returns
        -------
        :class:`nutils.evaluable.Array`
            The index of this transform chain in the given sequence.
        :class:`EvaluableTransformChain`
            The tail.

        See also
        --------
        :meth:`nutils.transformseq.Transforms.index_with_tail` : the unevaluable version of this method
        '''

        index_tail = _EvaluableIndexWithTail(__sequence, self)
        index = evaluable.ArrayFromTuple(index_tail, 0, (), int, _lower=0, _upper=len(__sequence) - 1)
        tails = _EvaluableTransformChainFromTuple(index_tail, 1, __sequence.fromdims, self.fromdims)
        return index, tails


class _Linear(Array):

    __slots__ = '_fromdims'

    def __init__(self, chain: EvaluableTransformChain) -> None:
        self._fromdims = chain.fromdims
        super().__init__(args=(chain,), shape=(chain.todims, chain.fromdims), dtype=float)

    def evalf(self, chain: TransformChain) -> numpy.ndarray:
        return functools.reduce(lambda r, i: i @ r, (item.linear for item in reversed(chain)), numpy.eye(self._fromdims))

    def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
        return evaluable.zeros(self.shape + var.shape, dtype=float)


class _Basis(Array):

    __slots__ = '_todims', '_fromdims'

    def __init__(self, chain: EvaluableTransformChain) -> None:
        self._todims = chain.todims
        self._fromdims = chain.fromdims
        super().__init__(args=(chain,), shape=(chain.todims, chain.todims), dtype=float)

    def evalf(self, chain: TransformChain) -> numpy.ndarray:
        linear = numpy.eye(self._fromdims)
        for item in reversed(chain):
            linear = item.linear @ linear
            assert item.fromdims <= item.todims <= item.fromdims + 1
            if item.todims == item.fromdims + 1:
                linear = numpy.concatenate([linear, item.ext[:, _]], axis=1)
        assert linear.shape == (self._todims, self._todims)
        return linear

    def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
        return evaluable.zeros(self.shape + var.shape, dtype=float)


class _Apply(Array):

    __slots__ = '_chain', '_coords'

    def __init__(self, chain: EvaluableTransformChain, coords: Array) -> None:
        if coords.ndim == 0:
            raise ValueError('expected a coords array with at least one axis but got {}'.format(coords))
        if not evaluable.equalindex(chain.fromdims, coords.shape[-1]):
            raise ValueError('the last axis of coords does not match the from dimension of the transform chain')
        self._chain = chain
        self._coords = coords
        super().__init__(args=(chain, coords), shape=(*coords.shape[:-1], chain.todims), dtype=float)

    def evalf(self, chain: TransformChain, coords: numpy.ndarray) -> numpy.ndarray:
        return apply(chain, coords)

    def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
        axis = self._coords.ndim - 1
        linear = evaluable.appendaxes(evaluable.prependaxes(self._chain.linear, self._coords.shape[:-1]), var.shape)
        dcoords = evaluable.insertaxis(evaluable.derivative(self._coords, var, seen), axis, linear.shape[axis])
        return evaluable.dot(linear, dcoords, axis+1)


class _EmptyTransformChain(EvaluableTransformChain):

    __slots__ = ()

    def __init__(self, dim: int) -> None:
        super().__init__((), dim, dim)

    def evalf(self) -> TransformChain:
        return ()

    def apply(self, points: Array) -> Array:
        return points

    @property
    def linear(self):
        return evaluable.diagonalize(evaluable.ones((self.todims,)))


class _TransformChainArgument(EvaluableTransformChain):

    __slots__ = '_name'

    def __init__(self, name: str, todims: int, fromdims: int) -> None:
        self._name = name
        super().__init__((evaluable.EVALARGS,), todims, fromdims)

    def evalf(self, evalargs) -> TransformChain:
        chain = evalargs[self._name]
        assert isinstance(chain, tuple) and all(isinstance(item, TransformItem) for item in chain)
        assert not chain or chain[0].todims == self.todims and chain[-1].fromdims == self.fromdims
        return chain

    @property
    def arguments(self):
        return frozenset({self})


class _EvaluableIndexWithTail(evaluable.Evaluable):

    __slots__ = '_sequence'

    def __init__(self, sequence: 'Transforms', chain: EvaluableTransformChain) -> None:
        self._sequence = sequence
        super().__init__((chain,))

    def evalf(self, chain: TransformChain) -> Tuple[numpy.ndarray, TransformChain]:
        index, tails = self._sequence.index_with_tail(chain)
        return numpy.array(index), tails


class _EvaluableTransformChainFromTuple(EvaluableTransformChain):

    __slots__ = '_index'

    def __init__(self, items: evaluable.Evaluable, index: int, todims: int, fromdims: int) -> None:
        self._index = index
        super().__init__((items,), todims, fromdims)

    def evalf(self, items: tuple) -> TransformChain:
        return items[self._index]

# vim:sw=4:sts=4:et
