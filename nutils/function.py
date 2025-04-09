import typing
if typing.TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    Protocol = object

from typing import Tuple, Union, Type, Callable, Sequence, Any, Optional, Iterator, Iterable, Dict, Mapping, List, FrozenSet, NamedTuple
from . import evaluable, numeric, _util as util, types, warnings, debug_flags
from ._util import nutils_dispatch
from functools import cached_property
from .transformseq import Transforms
import nutils_poly as poly
import builtins
import numpy
import functools
import operator
import numbers
import inspect
import fractions
import treelog
import dataclasses

IntoArray = Union['Array', numpy.ndarray, bool, int, float, complex]
Shape = Sequence[int]
DType = Type[Union[bool, int, float, complex]]
_dtypes = bool, int, float, complex

@dataclasses.dataclass(frozen=True)
class LowerArg:
    '''Argument for :meth:`Lowerable.lower`.

    Attributes
    ----------
    space : :class:`str`
    transforms : :class:`nutils.transformseq.Transforms`
    index : :class:`nutils.evaluable.Array`
    coordinates : :class:`nutils.evaluable.Array`, optional
    '''

    space: str
    transforms: Transforms
    index: evaluable.Array
    coordinates: Optional[evaluable.Array]

    def __post_init__(self):
        assert isinstance(self.space, str)
        assert isinstance(self.transforms, Transforms)
        assert isinstance(self.index, evaluable.Array) and self.index.dtype == int and self.index.ndim == 0
        assert self.coordinates is None or isinstance(self.coordinates, evaluable.Array)

    replace = dataclasses.replace

    @property
    def without_points(self) -> 'LowerArg':
        'A copy of the :class:`LowerArg` with :attr:`coordinates` set to None.'

        return self if self.coordinates is None else self.replace(coordinates=None)

    def map_coordinates(self, map) -> 'LowerArg':
        'Return a copy of the :class:`LowerArg` with :attr:`coordinates` mapped using ``map``.'

        if self.coordinates is None:
            return self
        coordinates = map(self.coordinates)
        return self if coordinates is self.coordinates else self.replace(coordinates=coordinates)

    def rename_spaces(self, map: Mapping[str, str]) -> 'LowerArg':
        '''Return a copy of the :class:`LowerArg` with :attr:`space` renamed using ``map``.

        If the old space is not found in ``map``, the :class:`LowerArg` is returned unchanged.
        '''

        space = map.get(self.space, self.space)
        return self if space == self.space else self.replace(space=space)


@dataclasses.dataclass(frozen=True)
class LowerArgs:
    '''An ordered sequence of :class:`LowerArg`\\s with common :attr:`points_shape`.

    Attributes
    ----------
    points_shape : :class:`tuple` of scalar, integer :class:`nutils.evaluable.Array`
        The shape of the leading points axes that are to be added to the
        lowered :class:`nutils.evaluable.Array`.
    args : :class:`tuple` of :class:`LowerArg`
        Tuple of transforms, index and coordinates per space.
    '''

    points_shape: Tuple[evaluable.Array, ...]
    args: Tuple[LowerArg, ...]

    @classmethod
    def empty(cls, points_shape=()) -> 'LowerArgs':
        'Returns an empty instance of :class:`LowerArgs`, optionally with the provided ``points_shape``.'

        return cls(points_shape, ())

    @classmethod
    def for_space(cls, space: str, transforms: Tuple[Transforms, ...], index: evaluable.Array, coordinates: evaluable.Array) -> 'LowerArgs':
        if index.dtype != int or index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        args = LowerArg(space, transforms[0], index, coordinates),
        if len(transforms) > 1:
            args += LowerArg('~' + space, transforms[-1], index, coordinates),
        return cls(coordinates.shape[:-1], args)

    def __post_init__(self):
        assert isinstance(self.points_shape, tuple) and all(map(evaluable._isindex, self.points_shape))
        assert isinstance(self.args, tuple)
        assert all(arg.coordinates is None or evaluable._all_certainly_equal(arg.coordinates.shape[:-1], self.points_shape) for arg in self.args)

    @property
    def without_points(self) -> 'LowerArgs':
        'A copy of the :class:`LowerArgs` without points (empty :attr:`points_shape` and :attr:`LowerArg.coordinates`).'

        return LowerArgs((), tuple(arg.without_points for arg in self.args))

    def map_coordinates(self, points_shape, map: Callable[[evaluable.Array], Optional[evaluable.Array]]) -> 'LowerArgs':
        'Return a copy of the :class:`LowerArgs` with the given ``points_shape`` and :attr:`LowerArg.coordinates` mapped using ``map``.'

        return LowerArgs(points_shape, tuple(arg.map_coordinates(map) for arg in self.args))

    def rename_spaces(self, map: Mapping[str, str]) -> 'LowerArgs':
        '''Return a copy of the :class:`LowerArgs` with spaces renamed using ``map``.

        Spaces that are not found in ``map`` are unchanged. The order of the
        :attr:`args` remains the same.

        It is allowed to map different old spaces to the same new space. Note
        that :meth:`LowerArgs.__getitem__` only returns the last entry of
        :attr:`args` that matches the given space.
        '''

        return LowerArgs(self.points_shape, tuple(arg.rename_spaces(map) for arg in self.args))

    def __or__(self, other: 'LowerArgs') -> 'LowerArgs':
        warnings.deprecation('`LowerArgs.__or__()` is deprecated; use `LowerArgs.__mul__()` instead')
        return self * other

    def __mul__(self, other: 'LowerArgs') -> 'LowerArgs':
        '''Return the outer product of two :class:`LowerArgs`.

        The :attr:`points_shape` of the product is the concatenation of the
        :attr:`points_shape`\\s of the factors; likewise for the :attr:`args`. If
        both factors contain :class:`LowerArg`\\s for a certain space, the
        :class:`LowerArg` of the right factor will be :attr:`exposed`.
        '''

        args = tuple(arg.map_coordinates(lambda coords: evaluable.Transpose.to_end(evaluable.appendaxes(coords, other.points_shape), coords.ndim - 1)) for arg in self.args)
        args += tuple(arg.map_coordinates(lambda coords: evaluable.prependaxes(coords, self.points_shape)) for arg in other.args)
        return LowerArgs(self.points_shape + other.points_shape, args)

    def __add__(self, other: 'LowerArgs') -> 'LowerArgs':
        '''Join two :class:`LowerArgs` with the same :attr:`points_shape`.

        The :attr:`args` is the concatenation of the :attr:`args` of the terms.
        If both terms contain :class:`LowerArg`\\s for a certain space, the
        :class:`LowerArg` of the right term will be :attr:`exposed`.
        '''

        points_shape = evaluable.assert_equal_tuple(self.points_shape, other.points_shape)
        return LowerArgs(points_shape, self.args + other.args)

    def __getitem__(self, space: str) -> LowerArg:
        '''Return the :attr:`exposed` :class:`LowerArg` with the given ``space``.

        Raises
        ------
        KeyError
            If the ``space`` is unknown.
        '''

        for arg in reversed(self.args):
            if arg.space == space:
                return arg
        raise KeyError(f'no such space: {space} - did you forget integral or sample?')

    @cached_property
    def exposed(self):
        '''Subset of :attr:`args` that are exposed.

        An item `a` of :attr:`args` is exposed if there exists no item `b` in
        :attr:`args` to the right of `a` with the same :attr:`LowerArg.space`
        as `a`.
        '''

        spaces = set()
        args = []
        for arg in reversed(self.args):
            if arg.space not in spaces:
                spaces.add(arg.space)
                args.append(arg)
        return tuple(reversed(args))

    @cached_property
    def spaces(self) -> FrozenSet[str]:
        'The set of spaces of the :attr:`args`'

        return frozenset(arg.space for arg in self.args)


class Lowerable(Protocol):
    'Protocol for lowering to :class:`nutils.evaluable.Array`.'

    @property
    def spaces(self) -> FrozenSet[str]: ...

    @property
    def arguments(self) -> Mapping[str, Tuple[Shape, DType]]: ...

    def lower(self, args: LowerArgs) -> evaluable.Array:
        '''Lower this object to a :class:`nutils.evaluable.Array`.

        Parameters
        ----------
        args : :class:`LowerArgs`
        '''


_ArrayMeta = type

if debug_flags.lower:
    def _debug_lower(self, args: LowerArgs) -> evaluable.Array:
        result = self._ArrayMeta__debug_lower_orig(args)
        assert isinstance(result, evaluable.Array)
        offset = 0 if type(self) == _WithoutPoints else len(args.points_shape)
        assert result.ndim == self.ndim + offset
        assert tuple(sh.__index__() for sh in result.shape[offset:]) == self.shape, 'shape mismatch'
        assert result.dtype == self.dtype, ('dtype mismatch', self.__class__)
        return result

    class _ArrayMeta(_ArrayMeta):
        def __new__(mcls, name, bases, namespace):
            if 'lower' in namespace:
                namespace['_ArrayMeta__debug_lower_orig'] = namespace.pop('lower')
                namespace['lower'] = _debug_lower
            return super().__new__(mcls, name, bases, namespace)

# The lower cache introduced below should stay below the debug wrapper added
# above. Otherwise the cached results are debugge again and again.


def _cache_lower(self, args: LowerArgs) -> evaluable.Array:
    cached_args, cached_result = getattr(self, '_ArrayMeta__cached_lower', (None, None))
    if cached_args == args:
        return cached_result
    result = self._ArrayMeta__cache_lower_orig(args)
    self._ArrayMeta__cached_lower = args, result
    return result


class _ArrayMeta(_ArrayMeta):
    def __new__(mcls, name, bases, namespace):
        if 'lower' in namespace:
            namespace['_ArrayMeta__cache_lower_orig'] = namespace.pop('lower')
            namespace['lower'] = _cache_lower
        return super().__new__(mcls, name, bases, namespace)


class Array(numpy.lib.mixins.NDArrayOperatorsMixin, metaclass=_ArrayMeta):
    '''Base class for array valued functions.

    Parameters
    ----------
    shape : :class:`tuple` of :class:`int`
        The shape of the array function.
    dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
        The dtype of the array elements.
    spaces : :class:`frozenset` of :class:`str`
        The spaces this array function is defined on.
    arguments : mapping of :class:`str`
        The mapping of argument names to their shapes and dtypes for all
        arguments of this array function.

    Attributes
    ----------
    shape : :class:`tuple` of :class:`int`
        The shape of this array function.
    ndim : :class:`int`
        The dimension of this array function.
    dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
        The dtype of the array elements.
    spaces : :class:`frozenset` of :class:`str`
        The spaces this array function is defined on.
    arguments : mapping of :class:`str`
        The mapping of argument names to their shapes and dtypes for all
        arguments of this array function.
    '''

    __array_priority__ = 1.  # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if method != '__call__' or ufunc not in HANDLED_FUNCTIONS:
            return NotImplemented
        try:
            arrays = [Array.cast(v) for v in inputs]
        except ValueError:
            return NotImplemented
        return HANDLED_FUNCTIONS[ufunc](*arrays, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def cast(cls, __value: IntoArray, dtype: Optional[DType] = None, ndim: Optional[int] = None) -> 'Array':
        '''Cast a value to an :class:`Array`.

        Parameters
        ----------
        value : :class:`Array`, or a :class:`numpy.ndarray` or similar
            The value to cast.
        '''

        # We use deep_reduce to stack nested structures depth-first.
        # Objects that are already Array are ignored. Lists or tuples
        # containing at least one Array will cause all other items to be
        # cast as well, before being stacked into a new Array, setting off
        # a cascade down to the outermost list. Those that do not contain
        # Array objects will bypass Nutils' array dispatches entirely.

        # NOTE: it may seem that the same can be achieved without deep_reduce
        # by simply stacking __value if it is a list or tuple: if any of its
        # items are Array then it will be nutils-stacked; if all are numeric
        # then it will be numpy-stacked. However, if no items are Array, but
        # some are lists that contains Arrays, then Numpy's dispatch mechanism
        # will not pick up on this and an object array will be formed instead.
        # Hence the need for depth-first.
        try:
            value = util.deep_reduce(numpy.stack, __value)
        except Exception as e: # something went wrong, e.g. incompatible shapes
            raise ValueError(f'cannot convert {__value!r} to Array: {e}')

        # If __value does not contain any Array then the result will not be
        # Array either, in which case we wrap it in a _Constant. Since stack
        # did most of the work already, we have only a handful of object types
        # left that we know can be wrapped. We test for those here rather than
        # have _Constant figure things out at potentially higher cost.
        if isinstance(value, fractions.Fraction):
            value = float(value)
        if isinstance(value, (numpy.ndarray, bool, int, float, complex)):
            value = _Constant(value)
        elif not isinstance(value, Array):
            raise ValueError(f'cannot convert {__value!r} to Array: unsupported data type')

        if dtype is not None and _dtypes.index(value.dtype) > _dtypes.index(dtype):
            raise ValueError('expected an array with dtype `{}` but got `{}`'.format(dtype.__name__, value.dtype.__name__))
        if ndim is not None and value.ndim != ndim:
            raise ValueError('expected an array with dimension `{}` but got `{}`'.format(ndim, value.ndim))
        return value

    def __init__(self, shape: Shape, dtype: DType, spaces: FrozenSet[str], arguments: Mapping[str, Tuple[Shape, DType]]) -> None:
        self.shape = tuple(sh.__index__() for sh in shape)
        self.dtype = dtype
        self.spaces = frozenset(spaces)
        self.arguments = types.frozendict(arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        raise NotImplementedError

    @cached_property
    def as_evaluable_array(self) -> evaluable.Array:
        return self.lower(LowerArgs.empty())

    def __index__(self):
        if self.arguments or self.spaces:
            raise ValueError('cannot convert non-constant array to index: arguments={}'.format(','.join(self.arguments)))
        elif self.ndim:
            raise ValueError('cannot convert non-scalar array to index: shape={}'.format(self.shape))
        elif self.dtype != int:
            raise ValueError('cannot convert non-integer array to index: dtype={}'.format(self.dtype.__name__))
        else:
            return self.as_evaluable_array.__index__()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, item: Any) -> 'Array':
        if not isinstance(item, tuple):
            item = item,
        iell = None
        nx = self.ndim - len(item)
        for i, it in enumerate(item):
            if it is ...:
                assert iell is None, 'at most one ellipsis allowed'
                iell = i
            elif it is numpy.newaxis:
                nx += 1
        array = self
        axis = 0
        for it in item + (slice(None),)*nx if iell is None else item[:iell] + (slice(None),)*(nx+1) + item[iell+1:]:
            if it is numpy.newaxis:
                array = expand_dims(array, axis)
                axis += 1
            elif isinstance(it, slice):
                array = _takeslice(array, it, axis)
                axis += 1
            else:
                array = numpy.take(array, it, axis)
                axis += numpy.ndim(it)
        assert axis == array.ndim
        return array

    def __bool__(self) -> bool:
        raise ValueError('The truth value of a nutils Array is ambiguous')

    def __len__(self) -> int:
        'Length of the first axis.'

        if self.ndim == 0:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    def __iter__(self) -> Iterator['Array']:
        'Iterator over the first axis.'

        if self.ndim == 0:
            raise TypeError('iteration over a 0-D array')
        return (self[i, ...] for i in range(self.shape[0]))

    @property
    def size(self) -> Union[int, 'Array']:
        'The total number of elements in this array.'

        return util.product(self.shape, 1)

    @property
    def T(self) -> 'Array':
        'The transposed array.'

        return numpy.transpose(self)

    def astype(self, dtype):
        if dtype == self.dtype:
            return self
        else:
            return _Wrapper(functools.partial(evaluable.astype, dtype=dtype), self, shape=self.shape, dtype=dtype)

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
        '''Return the sum of array elements over the given axes.

        .. warning::

           This method will change in future to match Numpy's equivalent
           method, which sums over all axes by default. During transition, use
           of this method without an axis argument will raise an error if the
           input array is of ndim >= 2.

        Parameters
        ----------
        arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
        axis : :class:`int`, a sequence of :class:`int`, or ``None``
            The axis or axes to sum. ``None``, the default, implies summation
            over the last axis.

        Returns
        -------
        :class:`Array`
        '''
        if axis is None and self.ndim != 1:
            raise ValueError(
                "The default summation axis will change from being the last axis (old behaviour)\n"
                "to being ALL axes (numpy's behaviour). To facilitate the transition, the axis\n"
                "argument has been made MANDATORY for arrays with dimension > 1, for the duration\n"
                "of at least one release cycle.")

        return numpy.sum(self, axis)

    def prod(self, __axis: int) -> 'Array':
        '''Return the product of array elements over the given axes.

        Parameters
        ----------
        axis : :class:`int`, a sequence of :class:`int`, or ``None``
            The axis or axes along which the product is performed. ``None``, the
            default, implies all axes.

        Returns
        -------
        :class:`Array`
        '''

        return numpy.prod(self, __axis)

    def dot(self, __other: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
        '''Return the inner product of the arguments over the given axes, elementwise over the remanining axes.

        .. warning::

           This method will change in future to match Numpy's equivalent
           method, which does not support an axis argument and has different
           behaviour in case of higher dimensional input. During transition,
           use of this method for any situation other than the contraction of
           two vectors will raise a warning, and later an error. For
           continuity, use numpy.dot, numpy.matmul, or the @ operator instead.

        Parameters
        ----------
        arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
        axis : :class:`int`, a sequence of :class:`int`, or ``None``
            The axis or axes along which the inner product is performed. If the
            second argument has one dimension and axes is ``None``, the default, the
            inner product of the second argument with the first axis of the first
            argument is computed. Otherwise ``axes=None`` is not allowed.

        Returns
        -------
        :class:`Array`
        '''
        other = Array.cast(__other)
        if axes is None and self.ndim == other.ndim == 1:
            # this is the only scenario in which the old implementation of dot
            # was compatible with that of numpy
            return numpy.dot(self, other)
        warnings.warn(
            'The implementation of Array.dot will change in the next release cycle to make\n'
            'it equal to that of Numpy: the axis argument will be removed and contraction\n'
            'will happen over the last axis of the first argument, rather than the first.\n'
            'To prepare for this transition, please update your code to use numpy.dot,\n'
            'numpy.matmul, the @ operator, or a combination of multiply and sum instead.')
        if axes is None:
            assert other.ndim == 1 and other.shape[0] == self.shape[0]
            other = _append_axes(other, self.shape[1:])
            axes = 0,
        return numpy.sum(self * other, axes)

    def normalized(self, __axis: int = -1) -> 'Array':
        'See :func:`normalized`.'
        return normalized(self, __axis)

    def normal(self, refgeom: Optional['Array'] = None, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`normal`.'
        return normal(self, refgeom, spaces=spaces)

    def curvature(self, ndims: int = -1, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`curvature`.'
        return curvature(self, ndims, spaces=spaces)

    def swapaxes(self, __axis1: int, __axis2: int) -> 'Array':
        return numpy.swapaxes(self, __axis1, __axis2)

    def transpose(self, __axes: Optional[Sequence[int]]) -> 'Array':
        return numpy.transpose(self, __axes)

    def add_T(self, axes: Tuple[int, int]) -> 'Array':
        'See :func:`add_T`.'
        return add_T(self, axes)

    def grad(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`grad`.'
        return grad(self, geom, ndims, spaces=spaces)

    def laplace(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`laplace`.'
        return laplace(self, geom, ndims, spaces=spaces)

    def symgrad(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`symgrad`.'
        return symgrad(self, geom, ndims, spaces=spaces)

    def div(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`div`.'
        return div(self, geom, ndims, spaces=spaces)

    def curl(self, geom: IntoArray, /, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`curl`.'
        return curl(self, geom, spaces=spaces)

    def dotnorm(self, geom: IntoArray, /, axis: int = -1, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`dotnorm`.'
        return dotnorm(self, geom, axis, spaces=spaces)

    def tangent(self, vec: IntoArray, /, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`tangent`.'
        return tangent(self, vec, spaces=spaces)

    def ngrad(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`ngrad`.'
        return ngrad(self, geom, ndims, spaces=spaces)

    def nsymgrad(self, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> 'Array':
        'See :func:`nsymgrad`.'
        return nsymgrad(self, geom, ndims, spaces=spaces)

    def choose(self, __choices: Sequence[IntoArray]) -> 'Array':
        'See :func:`choose`.'
        return numpy.choose(self, __choices)

    def vector(self, ndims):
        if not self.ndim:
            raise Exception('a scalar function cannot be vectorized')
        return numpy.reshape(diagonalize(insertaxis(self, 1, ndims), 1), (self.shape[0] * ndims, ndims, *self.shape[1:]))

    def __repr__(self) -> str:
        return 'Array<{}>'.format(','.join(str(n) for n in self.shape))

    def eval(self, /, arguments=None, *, legacy=None, **kwargs) -> numpy.ndarray:
        'Evaluate this function.'

        if self.ndim > 1 and legacy is None:
            warnings.deprecation(
                'Evaluation of an array of dimension 2 or higher is going to '
                'change in Nutils 10. Instead of evaluating a 2D array as a '
                'sparse matrix, and 3D and higher as a sparse array object, '
                'evaluation will be dense by default, with sparse evaluation '
                'available via the function.as_csr and function.as_coo '
                'modifiers. To make this transition, a new "legacy" argument is '
                'introduced that can be set to True to explicitly request the '
                'old behaviour (and suppress this warning), and to False to '
                'switch to dense evaluation.')
            legacy = True

        if arguments is None:
            if kwargs:
                warnings.deprecation(
                    'providing evaluation arguments as keyword arguments is '
                    'deprecated, please use the "arguments" parameter instead')
            arguments = kwargs
        elif kwargs:
            raise ValueError('invalid argument {list(kwargs)[0]!r}')

        data = eval(self if not legacy or self.ndim < 2 else as_csr(self) if self.ndim == 2 else as_coo(self), arguments)
        if not legacy or not self.ndim > 1:
            return data
        elif self.ndim == 2:
            values, rowptr, colidx = data
            from . import matrix
            return matrix.assemble_csr(values, rowptr, colidx, self.shape[1])
        else:
            values, *indices = data
            from . import sparse
            return sparse.compose(indices, values, func.shape)

    def derivative(self, __var: Union[str, 'Argument']) -> 'Array':
        'See :func:`derivative`.'
        return derivative(self, __var)

    def replace(self, __arguments: Mapping[str, IntoArray]) -> 'Array':
        'Return a copy with arguments applied.'
        return replace_arguments(self, __arguments)

    def contains(self, __name: str) -> bool:
        'Test if target occurs in this function.'
        return __name in self.arguments

    @property
    def argshapes(self) -> Mapping[str, Tuple[int, ...]]:
        warnings.deprecation("array.argshapes[...] is deprecated and will be removed in Nutils 10, please use function.arguments_for(array)[...].shape instead")
        return {name: shape for name, (shape, dtype) in self.arguments.items()}

    def conjugate(self):
        '''Return the complex conjugate, elementwise.

        Returns
        -------
        :class:`Array`
            The complex conjugate.
        '''
        return numpy.conjugate(self)

    conj = conjugate

    @property
    def real(self):
        '''Return the real part of the complex argument.

        Returns
        -------
        :class:`Array`
            The real part of the complex argument.
        '''
        return numpy.real(self)

    @property
    def imag(self):
        '''Return the imaginary part of the complex argument.

        Returns
        -------
        :class:`Array`
            The imaginary part of the complex argument.
        '''
        return numpy.imag(self)


class _Unlower(Array):

    def __init__(self, array: evaluable.Array, spaces: FrozenSet[str], arguments: Mapping[str, Tuple[Shape, DType]], lower_args: LowerArgs) -> None:
        self._array = array
        self._lower_args = lower_args
        shape = tuple(n.__index__() for n in array.shape[len(lower_args.points_shape):])
        super().__init__(shape=shape, dtype=array.dtype, spaces=spaces, arguments=arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        if args != self._lower_args:
            raise ValueError('_Unlower must be lowered with the same arguments as those with which it is instantiated.')
        return self._array


class Custom(Array):
    '''Combined :mod:`nutils.function` and :mod:`nutils.evaluable` array base class.

    Ordinary :class:`Array` subclasses should define the ``Array.lower`` method,
    which returns the corresponding :class:`nutils.evaluable.Array` with the
    proper amount of points axes. In many cases the :class:`Array` subclass is
    trivial and the corresponding :class:`nutils.evaluable.Array` contains all
    the specifics. For those situations the :class:`Custom` base class exists.
    Rather than defining the ``Array.lower`` method, this base class allows you
    to define a :meth:`Custom.evalf` and optionally a
    :meth:`Custom.partial_derivative`, which are used to instantiate a generic
    :class:`nutils.evaluable.Array` automatically during lowering.

    By default the :class:`Array` arguments passed to the constructor are
    unmodified. Broadcasting and singleton expansion, if required, should be
    applied before passing the arguments to the constructor of :class:`Custom`.
    It is possible to declare ``npointwise`` leading axes as being pointwise. In
    that case :class:`Custom` applies singleton expansion to the leading
    pointwise axes and the shape of the result passed to :class:`Custom` should
    not include the pointwise axes.

    For internal reasons, both ``evalf`` and ``partial_derivative`` must be
    static methods, meaning that they will not receive a reference to ``self``
    when called. Instead, all relevant data should be passed to ``evalf`` via
    the constructor argument ``args``. The constructor will automatically
    distinguish between Array and non-Array arguments, and pass the latter on
    to ``evalf`` unchanged. The ``partial_derivative`` will not be called for
    those arguments. Furthermore, ``evalf`` and ``partial_derivative`` must be
    hashable. The :func:`nutils.types.hashable_function` decorator both defines
    a hash for the decorated function and makes the decorated function static.

    Parameters
    ----------
    args : iterable of :class:`Array` objects or immutable and hashable objects
        The arguments of this array function.
    shape : :class:`tuple` of :class:`int` or :class:`Array`
        The shape of the array function without leading pointwise axes.
    dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
        The dtype of the array elements.
    npointwise : :class:`int`
        The number of leading pointwise axis.

    Example
    -------

    The following class implements multiplication using :class:`Custom` without
    broadcasting and for :class:`float` arrays only.

    >>> from nutils.types import hashable_function
    >>> class Multiply(Custom):
    ...
    ...   def __init__(self, left: IntoArray, right: IntoArray) -> None:
    ...     # Broadcast the arrays. `broadcast_arrays` automatically casts the
    ...     # arguments to `Array`.
    ...     left, right = broadcast_arrays(left, right)
    ...     # Dtype coercion is beyond the scope of this example.
    ...     if left.dtype != float or right.dtype != float:
    ...       raise ValueError('left and right arguments should have dtype float')
    ...     # We treat all axes as pointwise, hence parameter `shape`, the shape
    ...     # of the remainder, is empty and `npointwise` is the dimension of the
    ...     # arrays.
    ...     super().__init__(args=(left, right), shape=(), dtype=float, npointwise=left.ndim)
    ...
    ...   @hashable_function
    ...   def evalf(left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
    ...     # Because all axes are pointwise, the evaluated `left` and `right`
    ...     # arrays are 1d.
    ...     return left * right
    ...
    ...   @hashable_function
    ...   def partial_derivative(iarg: int, left: Array, right: Array) -> IntoArray:
    ...     # The arguments passed to this function are of type `Array` and the
    ...     # pointwise axes are omitted, hence `left` and `right` are 0d.
    ...     if iarg == 0:
    ...       return right
    ...     elif iarg == 1:
    ...       return left
    ...     else:
    ...       raise NotImplementedError
    ...
    >>> Multiply([1., 2.], [3., 4.]).eval()
    array([ 3.,  8.])
    >>> a = Argument('a', (2,))
    >>> Multiply(a, [3., 4.]).derivative(a).eval(a=numpy.array([1., 2.])).export('dense')
    array([[ 3.,  0.],
           [ 0.,  4.]])

    The following class wraps :func:`numpy.roll`, applied to the last axis of the
    array argument, with constant shift.

    >>> class Roll(Custom):
    ...
    ...   def __init__(self, array: IntoArray, shift: int) -> None:
    ...     array = asarray(array)
    ...     # We are being nit-picky here and cast `exponent` to an `int` without
    ...     # truncation.
    ...     shift = shift.__index__()
    ...     # We treat all but the last axis of `array` as pointwise.
    ...     super().__init__(args=(array, shift), shape=array.shape[-1:], dtype=array.dtype, npointwise=array.ndim-1)
    ...
    ...   @hashable_function
    ...   def evalf(array: numpy.ndarray, shift: int) -> numpy.ndarray:
    ...     # `array` is evaluated to a `numpy.ndarray` because we passed `array`
    ...     # as an `Array` to the constructor. `shift`, however, is untouched
    ...     # because it is not an `Array`. The `array` has two axes: a points
    ...     # axis and the axis to be rolled.
    ...     return numpy.roll(array, shift, 1)
    ...
    ...   @hashable_function
    ...   def partial_derivative(iarg, array: Array, shift: int) -> IntoArray:
    ...     if iarg == 0:
    ...       return Roll(eye(array.shape[0]), shift).T
    ...     else:
    ...       # We don't implement the derivative to `shift`, because this is
    ...       # a constant `int`.
    ...       raise NotImplementedError
    ...
    >>> Roll([1, 2, 3], 1).eval()
    array([3, 1, 2])
    >>> b = Argument('b', (3,))
    >>> Roll(b, 1).derivative(b).eval().export('dense')
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.]])
    '''

    def __init__(self, args: Iterable[Any], shape: Tuple[int], dtype: DType, npointwise: int = 0):
        args = tuple(args)
        if any(isinstance(arg, evaluable.Evaluable) for arg in args):
            raise ValueError('It is not allowed to call this function with a `nutils.evaluable.Evaluable` argument.')
        if npointwise:
            # Apply singleton expansion to the leading points axes.
            points_shapes = tuple(arg.shape[:npointwise] for arg in args if isinstance(arg, Array))
            if not all(len(points_shape) == npointwise for points_shape in points_shapes):
                raise ValueError('All arrays must have at least {} axes.'.format(npointwise))
            if len(points_shapes) == 0:
                raise ValueError('Pointwise axes can only be used in combination with at least one `function.Array` argument.')
            points_shape = broadcast_shapes(*points_shapes)
            args = tuple(numpy.broadcast_to(arg, points_shape + arg.shape[npointwise:]) if isinstance(arg, Array) else arg for arg in args)
        else:
            points_shape = ()
        self._args = args
        self._npointwise = npointwise
        spaces = frozenset(space for arg in args if isinstance(arg, Array) for space in arg.spaces)
        arguments = _join_arguments(arg.arguments for arg in args if isinstance(arg, Array))
        super().__init__(shape=(*points_shape, *shape), dtype=dtype, spaces=spaces, arguments=arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        evalargs = tuple(arg.lower(args) if isinstance(arg, Array) else arg for arg in self._args)
        add_points_shape = tuple(map(evaluable.asarray, self.shape[:self._npointwise]))
        args = args.map_coordinates(
            args.points_shape + add_points_shape,
            lambda coords: evaluable.Transpose.to_end(evaluable.appendaxes(coords, add_points_shape), coords.ndim-1),
        )
        evalf = self.evalf
        partial_derivative = self.partial_derivative
        if getattr(self.evalf, '__nutils_hash__', None) is None:
            warnings.deprecation(f'`{type(self).__name__}.evalf()` does not implement `nutils.types.nutils_hash()`. This will be mandatory in Nutils 10. Try using `nutils.types.hashable_function()`.')
            evalf = types.hashable_function(f'{type(self).__name__}.evalf')(evalf)
        if getattr(self.partial_derivative, '__nutils_hash__', None) is None:
            warnings.deprecation(f'`{type(self).__name__}.partial_derivative()` does not implement `nutils.types.nutils_hash()`. This will be mandatory in Nutils 10. Try using `nutils.types.hashable_function()`.')
            partial_derivative = types.hashable_function(f'{type(self).__name__}.partial_derivative')(partial_derivative)
        return _CustomEvaluable(
            type(self).__name__,
            evalf,
            partial_derivative,
            evalargs,
            self.shape[self._npointwise:],
            self.dtype,
            self.spaces,
            self.arguments,
            args,
        )

    @types.hashable_function('NotImplemented')
    def evalf(*args: Any) -> numpy.ndarray:
        '''Evaluate this function for the given evaluated arguments.

        This function is called with arguments that correspond to the arguments
        that are passed to the constructor of :class:`Custom`: every instance of
        :class:`Array` is evaluated to a :class:`numpy.ndarray` with one leading
        axis compared to the :class:`Array` and all other instances are passed as
        is. The return value of this method should also include a leading axis with
        the same length as the other array arguments have, or length one if there
        are no array arguments. If constructor argument ``npointwise`` is nonzero,
        the pointwise axes of the :class:`Array` arguments are raveled and included
        in the single leading axis of the evaluated array arguments as well.

        If possible this method should not use ``self``, e.g. by decorating this
        method with :func:`staticmethod`. The result of this function must only
        depend on the arguments and must not mutate the arguments.

        This method is equivalent to ``nutils.evaluable.Array.evalf`` up to
        the treatment of the leading axis.

        Parameters
        ----------
        *args
            The evaluated arguments corresponding to the ``args`` parameter of the
            :class:`Custom` constructor.

        Returns
        -------
        :class:`numpy.ndarray`
            The result of this function with one leading points axis.
        '''

        raise NotImplementedError

    @types.hashable_function('NotImplemented')
    def partial_derivative(iarg: int, *args: Any) -> IntoArray:
        '''Return the partial derivative of this function to :class:`Custom` constructor argument number ``iarg``.

        This method is only called for those arguments that are instances of
        :class:`Array` with dtype :class:`float` and have the derivative target as
        a dependency. It is therefor allowed to omit an implementation for some or
        all arguments if the above conditions are not met.

        Axes that are declared pointwise via the ``npointwise`` constructor
        argument are omitted.

        Parameters
        ----------
        iarg : :class:`int`
            The index of the argument to compute the derivative for.
        *args
            The arguments as passed to the constructor of :class:`Custom`.

        Returns
        -------
        :class:`Array` or similar
            The partial derivative of this function to the given argument.
        '''

        raise NotImplementedError(f'The partial derivative to argument {iarg} (counting from 0) is not defined.')


class _CustomEvaluable(evaluable.Array):

    name: str
    custom_evalf: callable
    custom_partial_derivative: callable
    args: Tuple[evaluable.Array, ...]
    argshape: Tuple[int, ...]
    dtype: DType
    spaces: FrozenSet[str]
    function_arguments: types.frozendict
    lower_args: LowerArgs

    @property
    def points_dim(self):
        return len(self.lower_args.points_shape)

    @property
    def dependencies(self):
        return *self.lower_args.points_shape, *(arg for arg in self.args if isinstance(arg, evaluable.Array))

    @cached_property
    def shape(self):
        return *self.lower_args.points_shape, *map(evaluable.constant, self.argshape)

    @property
    def _node_details(self) -> str:
        return self.name

    def evalf(self, *args: Any) -> numpy.ndarray:
        points_shape = tuple(n.__index__() for n in args[:self.points_dim])
        npoints = util.product(points_shape, 1)
        # Flatten the points axes of the evaluable arguments, merge with the
        # unevaluable arguments and call `custom_evalf`.
        flattened = []
        args = iter(args[self.points_dim:])
        for arg in self.args:
            if isinstance(arg, evaluable.Array):
                arg = next(args)
                arg = arg.reshape(npoints, *arg.shape[self.points_dim:])
            flattened.append(arg)
        result = self.custom_evalf(*flattened)
        assert result.ndim == self.ndim + 1 - self.points_dim
        # Unflatten the points axes of the result. If there are no arguments,
        # the points axis must have length one. Otherwise the length must be
        # `npoints` (checked by `reshape`).
        if not any(isinstance(origarg, evaluable.Array) for origarg in self.args):
            if result.shape[0] != 1:
                raise ValueError('Expected a points axis of length one but got {}.'.format(result.shape[0]))
            return numpy.broadcast_to(result[0], points_shape + result.shape[1:])
        else:
            return result.reshape(points_shape + result.shape[1:])

    def _compile(self, builder):
        args = builder.compile(self.dependencies)
        evalf = builder.add_constant(self.evalf)
        out = builder.get_variable_for_evaluable(self)
        builder.get_block_for_evaluable(self).assign_to(out, evalf.call(*args))
        return out

    def _derivative(self, var: evaluable.Array, seen: Dict[evaluable.Array, evaluable.Array]) -> evaluable.Array:
        if self.dtype in (bool, int):
            return super()._derivative(var, seen)
        result = evaluable.Zeros(self.shape + var.shape, dtype=self.dtype)
        unlowered_args = tuple(_Unlower(arg, self.spaces, self.function_arguments, self.lower_args) if isinstance(arg, evaluable.Array) else arg for arg in self.args)
        for iarg, arg in enumerate(self.args):
            if not isinstance(arg, evaluable.Array) or arg.dtype in (bool, int) or var not in arg.arguments and var != arg:
                continue
            fpd = Array.cast(self.custom_partial_derivative(iarg, *unlowered_args))
            fpd_expected_shape = tuple(n.__index__() for n in self.shape[self.points_dim:] + arg.shape[self.points_dim:])
            if fpd.shape != fpd_expected_shape:
                raise ValueError('`partial_derivative` to argument {} returned an array with shape {} but {} was expected.'.format(iarg, fpd.shape, fpd_expected_shape))
            epd = evaluable.astype(evaluable.appendaxes(fpd.lower(self.lower_args), var.shape), self.dtype)
            eda = evaluable.derivative(arg, var, seen)
            eda = evaluable.Transpose.from_end(evaluable.appendaxes(eda, self.shape[self.points_dim:]), *range(self.points_dim, self.ndim))
            result += (epd * eda).sum(range(self.ndim, self.ndim + arg.ndim - self.points_dim))
        return result


class _WithoutPoints:

    def __init__(self, __arg: Array) -> None:
        self._arg = __arg
        self.spaces = __arg.spaces
        self.arguments = __arg.arguments

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return self._arg.lower(args.without_points)


class _Wrapper(Array):

    @classmethod
    def broadcasted_arrays(cls, lower: Callable[..., evaluable.Array], *args: IntoArray, min_dtype: DType = bool, force_dtype: Optional[DType] = None) -> '_Wrapper':
        broadcasted = broadcast_arrays(*typecast_arrays(*args, min_dtype=min_dtype))
        return cls(lower, *broadcasted, shape=broadcasted[0].shape, dtype=force_dtype or broadcasted[0].dtype)

    def __init__(self, lower: Callable[..., evaluable.Array], *args: Lowerable, shape: Shape, dtype: DType) -> None:
        self._lower = lower
        self._args = args
        assert all(hasattr(arg, 'lower') for arg in self._args)
        spaces = frozenset(space for arg in args for space in arg.spaces)
        arguments = _join_arguments(arg.arguments for arg in self._args)
        super().__init__(shape, dtype, spaces, arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return self._lower(*(arg.lower(args) for arg in self._args))


class _Zeros(Array):

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return evaluable.Zeros((*args.points_shape, *map(evaluable.constant, self.shape)), self.dtype)


class _Ones(Array):

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return evaluable.ones(args.points_shape + tuple(evaluable.constant(n) for n in self.shape), self.dtype)


class _Constant(Array):

    def __init__(self, value: Any) -> None:
        self._value = types.arraydata(value)
        super().__init__(self._value.shape, self._value.dtype, frozenset(()), {})

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return evaluable.prependaxes(evaluable.Constant(self._value), args.points_shape)


class Argument(Array):
    '''Array valued function argument.

    Parameters
    ----------
    name : str
        The name of this argument.
    shape : :class:`tuple` of :class:`int`
        The shape of this argument.
    dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
        The dtype of the array elements.

    Attributes
    ----------
    name : str
        The name of this argument.
    '''

    def __init__(self, name: str, shape: Shape, dtype: DType = float) -> None:
        self.name = name
        super().__init__(shape, dtype, frozenset(()), {name: (tuple(shape), dtype)})

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return evaluable.prependaxes(evaluable.Argument(self.name, tuple(evaluable.constant(n) for n in self.shape), self.dtype), args.points_shape)


class _Replace(Array):

    def __init__(self, arg: Array, replacements: Dict[str, Array]) -> None:
        self._arg = arg
        self._replacements = {}
        for old, new in _argument_to_array(replacements, arg):
            if new.spaces:
                raise ValueError(f'replacement functions cannot be bound to a space, but replacement for Argument {old.name!r} is bound to {", ".join(new.spaces)}.')
            self._replacements[old.name] = new
        # Build arguments map with replacements.
        unreplaced = {name: shape_dtype for name, shape_dtype in arg.arguments.items() if name not in replacements}
        arguments = _join_arguments([unreplaced] + [replacement.arguments for replacement in self._replacements.values()])
        super().__init__(arg.shape, arg.dtype, arg.spaces, arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        arg = self._arg.lower(args)
        replacements = {name: value.lower(args.without_points) for name, value in self._replacements.items()}
        return evaluable.replace_arguments(arg, replacements)


class _Transpose(Array):

    @classmethod
    def _end(cls, array: Array, axes: Tuple[int, ...], invert: bool = False) -> Array:
        axes = tuple(numeric.normdim(array.ndim, axis) for axis in axes)
        if all(a == b for a, b in enumerate(axes, start=array.ndim-len(axes))):
            return array
        trans = [i for i in range(array.ndim) if i not in axes]
        trans.extend(axes)
        if len(trans) != array.ndim:
            raise Exception('duplicate axes')
        return cls(array, tuple(numpy.argsort(trans) if invert else trans))

    @classmethod
    def from_end(cls, array: Array, *axes: int) -> Array:
        return cls._end(array, axes, invert=True)

    @classmethod
    def to_end(cls, array: Array, *axes: int) -> Array:
        return cls._end(array, axes, invert=False)

    def __init__(self, arg: Array, axes: Tuple[int, ...]) -> None:
        self._arg = arg
        self._axes = tuple(n.__index__() for n in axes)
        super().__init__(tuple(arg.shape[axis] for axis in axes), arg.dtype, arg.spaces, arg.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        arg = self._arg.lower(args)
        offset = len(args.points_shape)
        axes = (*range(offset), *(i+offset for i in self._axes))
        return evaluable.transpose(arg, axes)


class _SwapSpaces(Array):

    def __init__(self, arg: Array, space0: str, space1: str) -> None:
        self._arg = arg
        self._map = {space0: space1, space1: space0}
        spaces = tuple(self._map.get(space, space) for space in arg.spaces)
        super().__init__(arg.shape, arg.dtype, spaces, arg.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return self._arg.lower(args.rename_spaces(self._map))


class _Opposite(Array):

    def __init__(self, arg: Array, space: str) -> None:
        self._arg = arg
        self._space = space
        super().__init__(arg.shape, arg.dtype, arg.spaces, arg.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        opp_space = '~' + self._space
        if self._space in args.spaces and opp_space in args.spaces:
            args = args.rename_spaces({self._space: opp_space, opp_space: self._space})
        return self._arg.lower(args)


class _RootCoords(Array):

    def __init__(self, space: str, ndims: int) -> None:
        self._space = space
        super().__init__((ndims,), float, frozenset({space}), {})

    def lower(self, args: LowerArgs) -> evaluable.Array:
        inv_linear = evaluable.diagonalize(evaluable.ones(tuple(evaluable.constant(n) for n in self.shape)))
        inv_linear = evaluable.prependaxes(inv_linear, args.points_shape)
        arg = args[space]
        tip_coords = evaluable.WithDerivative(arg.coordinates, _tip_derivative_target(self._space, tip_coords.shape[-1]), evaluable.Diagonalize(evaluable.ones(tip_coords.shape)))
        coords = evaluable.TransformCoords(None, arg.transforms, arg.index, tip_coords)
        return evaluable.WithDerivative(coords, _root_derivative_target(self._space, evaluable.constant(self.shape[0])), inv_linear)


class _TransformsIndex(Array):

    def __init__(self, space: str, transforms: Transforms) -> None:
        self._space = space
        self._transforms = transforms
        super().__init__((), int, frozenset({space}), {})

    def lower(self, args: LowerArgs) -> evaluable.Array:
        arg = args[self._space]
        return evaluable.prependaxes(evaluable.TransformIndex(self._transforms, arg.transforms, arg.index), args.points_shape)


class _TransformsCoords(Array):

    def __init__(self, space: str, transforms: Transforms) -> None:
        self._space = space
        self._transforms = transforms
        super().__init__((transforms.fromdims,), float, frozenset({space}), {})

    def lower(self, args: LowerArgs) -> evaluable.Array:
        arg = args[self._space]
        index = evaluable.TransformIndex(self._transforms, arg.transforms, arg.index)
        L = evaluable.TransformLinear(None, self._transforms, index)
        if self._transforms.todims > self._transforms.fromdims:
            LTL = evaluable.einsum('ki,kj->ij', L, L)
            Linv = evaluable.einsum('ik,jk->ij', evaluable.inverse(LTL), L)
        else:
            Linv = evaluable.inverse(L)
        Linv = evaluable.prependaxes(Linv, args.points_shape)
        tip_coords = evaluable.WithDerivative(arg.coordinates, _tip_derivative_target(self._space, arg.coordinates.shape[-1]), evaluable.Diagonalize(evaluable.ones(arg.coordinates.shape)))
        coords = evaluable.TransformCoords(self._transforms, arg.transforms, arg.index, tip_coords)
        return evaluable.WithDerivative(coords, _root_derivative_target(self._space, evaluable.constant(self._transforms.todims)), Linv)


class _Derivative(Array):

    def __init__(self, arg: Array, var: Argument) -> None:
        assert isinstance(var, Argument)
        self._arg = arg
        self._var = var
        self._eval_var = evaluable.Argument(var.name, tuple(evaluable.constant(n) for n in var.shape), var.dtype)
        arguments = _join_arguments((arg.arguments, var.arguments))
        super().__init__(arg.shape+var.shape, complex if var.dtype == complex else arg.dtype, arg.spaces | var.spaces, arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        arg = self._arg.lower(args)
        return evaluable.derivative(arg, self._eval_var)


def _tip_derivative_target(space: str, dim) -> evaluable.DerivativeTargetBase:
    return evaluable.IdentifierDerivativeTarget((space, 'tip'), (dim,))


def _root_derivative_target(space: str, dim: int) -> evaluable.DerivativeTargetBase:
    return evaluable.IdentifierDerivativeTarget((space, 'root'), (dim,))


class _Gradient(Array):
    # Derivative of `func` to `geom` using the root coords as reference.

    def __init__(self, func: Array, geom: Array, spaces: FrozenSet[str]) -> None:
        assert spaces, '0d array'
        assert spaces <= geom.spaces, 'singular'
        assert geom.dtype == float
        common_shape = broadcast_shapes(func.shape, geom.shape[:-1])
        self._func = numpy.broadcast_to(func, common_shape)
        self._geom = numpy.broadcast_to(geom, (*common_shape, geom.shape[-1]))
        self._spaces = spaces
        arguments = _join_arguments((func.arguments, geom.arguments))
        super().__init__(self._geom.shape, complex if func.dtype == complex else float, func.spaces | geom.spaces, arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        func = self._func.lower(args)
        geom = self._geom.lower(args)
        ref_dim = builtins.sum(args[space].transforms.todims for space in self._spaces)
        if self._geom.shape[-1] != ref_dim:
            raise Exception('cannot invert {}x{} jacobian'.format(self._geom.shape[-1], ref_dim))
        refs = tuple(_root_derivative_target(arg.space, evaluable.constant(arg.transforms.todims)) for arg in args.exposed if arg.space in self._spaces)
        dfunc_dref = evaluable.concatenate([evaluable.derivative(func, ref) for ref in refs], axis=-1)
        dgeom_dref = evaluable.concatenate([evaluable.derivative(geom, ref) for ref in refs], axis=-1)
        dref_dgeom = evaluable.inverse(dgeom_dref)
        return evaluable.einsum('Ai,Aij->Aj', dfunc_dref, evaluable.astype(dref_dgeom, dfunc_dref.dtype))


class _SurfaceGradient(Array):
    # Surface gradient of `func` to `geom` using the tip coordinates as
    # reference.

    def __init__(self, func: Array, geom: Array, spaces: FrozenSet[str]) -> None:
        assert spaces, '0d array'
        assert spaces <= geom.spaces, 'singular'
        assert geom.dtype == float
        common_shape = broadcast_shapes(func.shape, geom.shape[:-1])
        self._func = numpy.broadcast_to(func, common_shape)
        self._geom = numpy.broadcast_to(geom, (*common_shape, geom.shape[-1]))
        self._spaces = spaces
        arguments = _join_arguments((func.arguments, geom.arguments))
        super().__init__(self._geom.shape, complex if func.dtype == complex else float, func.spaces | geom.spaces, arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        func = self._func.lower(args)
        geom = self._geom.lower(args)
        ref_dim = builtins.sum(args[space].transforms.fromdims for space in self._spaces)
        if self._geom.shape[-1] != ref_dim + 1:
            raise ValueError('expected a {}d geometry but got a {}d geometry'.format(ref_dim + 1, self._geom.shape[-1]))
        refs = []
        for arg in args.exposed:
            if arg.space not in self._spaces:
                continue
            refs.append((_root_derivative_target if arg.transforms.todims == arg.transforms.fromdims else _tip_derivative_target)(arg.space, evaluable.constant(arg.transforms.fromdims)))
        dfunc_dref = evaluable.concatenate([evaluable.derivative(func, ref) for ref in refs], axis=-1)
        dgeom_dref = evaluable.concatenate([evaluable.derivative(geom, ref) for ref in refs], axis=-1)
        dref_dgeom = evaluable.einsum('Ajk,Aik->Aij', dgeom_dref, evaluable.inverse(evaluable.grammium(dgeom_dref)))
        return evaluable.einsum('Ai,Aij->Aj', dfunc_dref, evaluable.astype(dref_dgeom, dfunc_dref.dtype))


class _Jacobian(Array):
    # The jacobian determinant of `geom` to the tip coordinates of the spaces of
    # `geom`. The last axis of `geom` is the coordinate axis.

    def __init__(self, geom: Array, spaces: FrozenSet[str], tip_dim: Optional[int] = None) -> None:
        assert geom.ndim >= 1
        assert geom.dtype == float
        assert spaces <= geom.spaces
        if not spaces and geom.shape[-1] != 0:
            raise ValueError('The jacobian of a constant (in space) geometry must have dimension zero.')
        if tip_dim is not None and tip_dim > geom.shape[-1]:
            raise ValueError('Expected a dimension of the tip coordinate system '
                             'not greater than the dimension of the geometry.')
        self._tip_dim = tip_dim
        self._geom = geom
        self._spaces = spaces
        super().__init__((), float, geom.spaces, geom.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        geom = self._geom.lower(args)
        tip_dim = builtins.sum(args[space].transforms.fromdims for space in self._spaces)
        if self._tip_dim is not None and self._tip_dim != tip_dim:
            raise ValueError('Expected a tip dimension of {} but got {}.'.format(self._tip_dim, tip_dim))
        if self._geom.shape[-1] < tip_dim:
            raise ValueError('the dimension of the geometry cannot be lower than the dimension of the tip coords')
        if not self._spaces:
            return evaluable.ones(geom.shape[:-1])
        tips = [_tip_derivative_target(arg.space, evaluable.constant(arg.transforms.fromdims)) for arg in args.exposed if arg.space in self._spaces]
        J = evaluable.concatenate([evaluable.derivative(geom, tip) for tip in tips], axis=-1)
        return evaluable.sqrt_abs_det_gram(J)


class _Normal(Array):

    def __init__(self, geom: Array, spaces: FrozenSet[str]) -> None:
        assert spaces <= geom.spaces
        self._geom = geom
        self._spaces = spaces
        assert geom.dtype == float
        super().__init__(geom.shape, float, geom.spaces, geom.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        geom = self._geom.lower(args)
        spaces_dim = builtins.sum(args[space].transforms.todims for space in self._spaces)
        normal_dim = spaces_dim - builtins.sum(args[space].transforms.fromdims for space in self._spaces)
        if self._geom.shape[-1] < spaces_dim:
            raise ValueError('The dimension of geometry must equal or larger than the sum of the dimensions of the given spaces.')
        if normal_dim == 0:
            raise ValueError('Cannot compute the normal because the dimension of the normal space is zero.')
        elif normal_dim > 1:
            raise ValueError('Cannot unambiguously compute the normal because the dimension of the normal space is larger than one.')
        tangents = []
        normal = None
        for arg in args.exposed:
            if arg.space not in self._spaces:
                continue
            chain = arg.transforms
            rgrad = evaluable.derivative(geom, _root_derivative_target(arg.space, evaluable.constant(chain.todims)))
            if chain.todims == chain.fromdims:
                # `chain.basis` is `eye(chain.todims)`
                tangents.append(rgrad)
            else:
                assert normal is None and chain.todims == chain.fromdims + 1
                basis = evaluable.einsum('Aij,jk->Aik', rgrad, evaluable.TransformBasis(chain, arg.index))
                tangents.append(basis[..., :chain.fromdims])
                normal = basis[..., chain.fromdims]
        assert normal is not None
        return evaluable.Orthonormal(evaluable.concatenate(tangents, axis=-1), normal)


class _ExteriorNormal(Array):

    def __init__(self, rgrad: Array) -> None:
        assert rgrad.dtype == float and rgrad.shape[-2] == rgrad.shape[-1] + 1
        self._rgrad = rgrad
        super().__init__(rgrad.shape[:-1], float, rgrad.spaces, rgrad.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        rgrad = self._rgrad.lower(args)
        if self._rgrad.shape[-2] == 2:
            normal = evaluable.stack([rgrad[..., 1, 0], -rgrad[..., 0, 0]], axis=-1)
        elif self._rgrad.shape[-2] == 3:
            i = evaluable.asarray([1, 2, 0])
            j = evaluable.asarray([2, 0, 1])
            normal = evaluable.Take(rgrad[..., 0], i) * evaluable.Take(rgrad[..., 1], j) - evaluable.Take(rgrad[..., 1], i) * evaluable.Take(rgrad[..., 0], j)
        else:
            raise NotImplementedError
        return normal / evaluable.InsertAxis(evaluable.sqrt(evaluable.Sum(normal**2.)), normal.shape[-1])


class _Concatenate(Array):

    def __init__(self, __arrays: Sequence[IntoArray], axis: int) -> None:
        self.arrays = typecast_arrays(*__arrays)
        shape0 = self.arrays[0].shape
        self.axis = numeric.normdim(len(shape0), axis)
        if any(array.shape[:self.axis] != shape0[:self.axis] or array.shape[self.axis+1:] != shape0[self.axis+1:] for array in self.arrays[1:]):
            raise ValueError('all the input array dimensions except for the concatenation axis must match exactly')
        super().__init__(
            shape=(*shape0[:self.axis], builtins.sum(array.shape[self.axis] for array in self.arrays), *shape0[self.axis+1:]),
            dtype=self.arrays[0].dtype,
            spaces=functools.reduce(operator.or_, (array.spaces for array in self.arrays)),
            arguments=_join_arguments(array.arguments for array in self.arrays))

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return util.sum(evaluable._inflate(array.lower(args), evaluable.Range(evaluable.constant(array.shape[self.axis])) + offset, evaluable.constant(self.shape[self.axis]), self.axis-self.ndim)
                        for array, offset in zip(self.arrays, util.cumsum(array.shape[self.axis] for array in self.arrays)))


def _join_arguments(args_list: Iterable[Mapping[str, Argument]]) -> Dict[str, Argument]:
    joined = {}  # type: Dict[str, Argument]
    for arguments in args_list:
        for name, (shape1, dtype1) in arguments.items():
            if name not in joined:
                joined[name] = shape1, dtype1
            else:
                shape2, dtype2 = joined[name]
                if shape1 != shape2:
                    raise ValueError('Argument {!r} has two different shapes: {}, {}.'.format(name, shape1, shape2))
                elif dtype1 != dtype2:
                    raise ValueError('Argument {!r} has two different dtypes: {}, {}.'.format(name, dtype1.__name__ if dtype1 in _dtypes else dtype1, dtype2.__name__ if dtype2 in _dtypes else dtype2))
    return joined


def asarray(__arg: IntoArray) -> Array:
    '''Cast a value to an :class:`Array`.

    Parameters
    ----------
    value : :class:`Array`, or a :class:`numpy.ndarray` or similar
        The value to cast.

    Returns
    -------
    :class:`Array`
    '''

    return Array.cast(__arg)


def zeros(shape: Shape, dtype: DType = float) -> Array:
    '''Create a new :class:`Array` of given shape and dtype, filled with zeros.

    Parameters
    ----------
    shape : :class:`tuple` of :class:`int` or :class:`Array`
        The shape of the new array.
    dtype : :class:`bool`, :class:`int` or :class:`float`
        The dtype of the array elements.

    Returns
    -------
    :class:`Array`
    '''

    return _Zeros(shape, dtype, frozenset(()), {})


def ones(shape: Shape, dtype: DType = float) -> Array:
    '''Create a new :class:`Array` of given shape and dtype, filled with ones.

    Parameters
    ----------
    shape : :class:`tuple` of :class:`int` or :class:`Array`
        The shape of the new array.
    dtype : :class:`bool`, :class:`int` or :class:`float`
        The dtype of the array elements.

    Returns
    -------
    :class:`Array`
    '''

    return _Ones(shape, dtype, frozenset(()), {})


def eye(__n, dtype=float):
    '''Create a 2-D :class:`Array` with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : :class:`int`
        The number of rows and columns.
    dtype : :class:`bool`, :class:`int` or :class:`float`
        The dtype of the array elements.

    Returns
    -------
    :class:`Array`
    '''

    return diagonalize(ones([__n], dtype=dtype))


def levicivita(__n: int, dtype: DType = float) -> Array:
    '''Create an n-D Levi-Civita symbol.

    Parameters
    ----------
    n : :class:`int`
        The dimension of the Levi-Civita symbol.
    dtype : :class:`bool`, :class:`int` or :class:`float`
        The dtype of the array elements.

    Returns
    -------
    :class:`Array`
    '''

    return _Constant(numeric.levicivita(__n))


@nutils_dispatch
def swap_spaces(arg: IntoArray, space0: str, space1: str, /) -> Array:
    '''Swap the two :attr:`~Array.spaces` of ``arg``.

    If ``arg`` is invariant to the spaces :func:`swap_spaces` does nothing.
    Also, swapping ``arg`` twice with the same set of spaces results in the
    original ``arg``.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
    space0 : :class:`str`
        Space to swap.
    space1 : :class:`str`
        Space to swap.
    '''

    arg = Array.cast(arg)
    return arg if space0 == space1 else _SwapSpaces(arg, space0, space1)


@nutils_dispatch
def opposite(__arg: IntoArray) -> Array:
    '''Evaluate this function at the opposite side.

    When evaluating a function ``arg`` at an interface, the function will be
    evaluated at one side of the interface. :func:`opposite` selects the opposite
    side.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

    Returns
    -------
    :class:`Array`

    Example
    -------

    We create a one dimensional topology with two elements and a discontinuous
    function ``f`` that is 1 on the first element and 2 on the second:

    >>> from nutils import mesh, function
    >>> topo, geom = mesh.rectilinear([2])
    >>> f = topo.basis('discont', 0).dot([1, 2])

    Evaluating this function at the interface gives (for this particular
    topology) the value at the side of the first element:

    >>> topo.interfaces.sample('bezier', 1).eval(f)
    array([ 1.])

    Using :func:`opposite` we obtain the value at the side of second element:

    >>> topo.interfaces.sample('bezier', 1).eval(function.opposite(f))
    array([ 2.])

    It is allowed to nest opposites:

    >>> topo.interfaces.sample('bezier', 1).eval(function.opposite(function.opposite(f)))
    array([ 1.])

    See Also
    --------

    :func:`mean` : the mean at an interface
    :func:`jump` : the jump at an interface
    '''

    arg = Array.cast(__arg)
    for space in sorted(arg.spaces):
        arg = _Opposite(arg, space)
    return arg


@nutils_dispatch
def mean(__arg: IntoArray) -> Array:
    '''Return the mean of the argument at an interface.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

    Returns
    -------
    :class:`Array`

    Example
    -------

    We create a one dimensional topology with two elements and a discontinuous
    function ``f`` that is 1 on the first element and 2 on the second:

    >>> from nutils import mesh, function
    >>> topo, geom = mesh.rectilinear([2])
    >>> f = topo.basis('discont', 0).dot([1, 2])

    Evaluating the mean of this function at the interface gives:

    >>> topo.interfaces.sample('bezier', 1).eval(function.mean(f))
    array([ 1.5])
    '''

    return .5 * (__arg + opposite(__arg))


@nutils_dispatch
def jump(__arg: IntoArray) -> Array:
    '''Return the jump of the argument at an interface.

    The sign of the jump depends on the orientation of the interfaces in a
    :class:`~nutils.topology.Topology`. Usually the jump is used as part of an
    inner product with the :func:`normal` of the geometry is used, which is
    independent of the orientation of the interfaces.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

    Returns
    -------
    :class:`Array`

    Example
    -------

    We create a one dimensional topology with two elements and a discontinuous
    function ``f`` that is 1 on the first element and 2 on the second:

    >>> from nutils import mesh, function
    >>> topo, geom = mesh.rectilinear([2])
    >>> f = topo.basis('discont', 0).dot([1, 2])

    Evaluating the jump of this function at the interface gives (for this
    particular topology):

    >>> topo.interfaces.sample('bezier', 1).eval(function.jump(f))
    array([ 1.])
    '''

    return opposite(__arg) - __arg


@nutils_dispatch
def normalized(__arg: IntoArray, axis: int = -1) -> Array:
    '''Return the argument normalized over the given axis, elementwise over the remanining axes.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : :class:`int`
        The axis along which the norm is computed. Defaults to the last axis.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(__arg)
    return arg / insertaxis(numpy.linalg.norm(arg, axis=axis), axis, 1)


def matmat(__arg0: IntoArray, *args: IntoArray) -> Array:
    'helper function, contracts last axis of arg0 with first axis of arg1, etc'
    retval = Array.cast(__arg0)
    for arg in map(Array.cast, args):
        if retval.shape[-1] != arg.shape[0]:
            raise ValueError('incompatible shapes')
        retval = numpy.sum(_append_axes(retval, arg.shape[1:]) * arg, retval.ndim-1)
    return retval


def diagonalize(__arg: IntoArray, __axis: int = -1, __newaxis: int = -1) -> Array:
    '''Return argument with ``newaxis`` such that ``axis`` and `newaxis`` is diagonal.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : :class:`int`
        The axis to diagonalize. Defaults to the last axis w.r.t. the argument.
    newaxis : :class:`int`
        The axis to add. Defaults to the last axis w.r.t. the return value.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(__arg)
    axis = numeric.normdim(arg.ndim, __axis)
    newaxis = numeric.normdim(arg.ndim+1, __newaxis)
    assert axis < newaxis
    transposed = _Transpose.to_end(arg, axis)
    diagonalized = _Wrapper(evaluable.Diagonalize, transposed, shape=(*transposed.shape, transposed.shape[-1]), dtype=transposed.dtype)
    return _Transpose.from_end(diagonalized, axis, newaxis)


def outer(arg1, arg2=None, axis=0):
    'outer product'

    warnings.deprecation('function.outer is deprecated and will be repurposed in Nutils 10, please use alternatives like numpy.einsum instead')
    if arg2 is None:
        arg2 = arg1
    elif arg1.ndim != arg2.ndim:
        raise ValueError('arg1 and arg2 have different dimensions')
    axis = numeric.normdim(arg1.ndim, axis)
    return expand_dims(arg1, axis+1) * expand_dims(arg2, axis)


def _append_axes(__array: IntoArray, __shape: Shape) -> Array:
    array = Array.cast(__array)
    for n in __shape:
        array = _Wrapper(evaluable.InsertAxis, array, _WithoutPoints(Array.cast(n)), shape=(*array.shape, n), dtype=array.dtype)
    return array


def _prepend_axes(__array: IntoArray, __shape: Shape) -> Array:
    array = Array.cast(__array)
    appended = _append_axes(array, __shape)
    return _Transpose.from_end(appended, *range(len(__shape)))


def insertaxis(__array: IntoArray, axis: int, length: int) -> Array:
    '''Insert an axis with given length.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : class:`int`
        The position of the inserted axis. Negative values count from the end of
        the resulting array.
    length : :class:`int` or :class:`Array`
        The length of the inserted axis.

    Returns
    -------
    :class:`Array`
    '''

    appended = _append_axes(__array, (length,))
    return _Transpose.from_end(appended, axis)


def expand_dims(__array: IntoArray, axis: int) -> Array:
    '''Insert a singleton axis.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : class:`int`
        The position of the inserted axis. Negative values count from the end of
        the resulting array.

    Returns
    -------
    :class:`Array`
    '''

    return insertaxis(__array, axis, 1)


def unravel(__array: IntoArray, axis: int, shape: Tuple[int, int]) -> Array:
    '''Unravel an axis to the given shape.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : :class:`int`
        The axis to unravel.
    shape : two-:class:`tuple` of :class:`int`
        The shape of the unraveled axes.

    Returns
    -------
    :class:`Array`
        The resulting array with unraveled axes ``axis`` and ``axis+1``.
    '''

    assert len(shape) == 2 and all(isinstance(sh, int) for sh in shape), 'function.unravel: invalid shape: expected two integers, received {}'.format(shape)
    transposed = _Transpose.to_end(Array.cast(__array), axis)
    unraveled = _Wrapper(evaluable.Unravel,
                         transposed,
                         _WithoutPoints(Array.cast(shape[0])),
                         _WithoutPoints(Array.cast(shape[1])),
                         shape=(*transposed.shape[:-1], *shape),
                         dtype=transposed.dtype)
    return _Transpose.from_end(unraveled, axis, axis+1)


def get(__array: IntoArray, __axis: int, __index: IntoArray) -> Array:
    '''Get one element from an array along an axis.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : :class:`int`
        The axis to get an element from.
    index : :class:`int` or :class:`Array`
        The index of the element to get.

    Returns
    -------
    :class:`Array`

    See Also
    --------
    :func:`kronecker` : The complement operation.
    '''

    return numpy.take(Array.cast(__array), Array.cast(__index, dtype=int, ndim=0), __axis)


def _takeslice(__array: IntoArray, __s: slice, __axis: int) -> Array:
    array = Array.cast(__array)
    s = __s
    axis = __axis
    n = array.shape[axis]
    if s.step == None or s.step == 1:
        start = 0 if s.start is None else s.start if s.start >= 0 else s.start + n
        stop = n if s.stop is None else s.stop if s.stop >= 0 else s.stop + n
        if start == 0 and stop == n:
            return array
        length = stop - start
        index = _Wrapper(evaluable.Range, _WithoutPoints(_Constant(length)), shape=(length,), dtype=int) + start
    elif isinstance(n, numbers.Integral):
        index = Array.cast(numpy.arange(*s.indices(int(n))))
    else:
        raise Exception('a non-unit slice requires a constant-length axis')
    return numpy.take(array, index, axis)


@nutils_dispatch
def scatter(__array: IntoArray, length: int, indices: IntoArray) -> Array:
    '''Distribute the last dimensions of an array over a new axis.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    length : :class:`int`
        The target length of the scattered axis.
    indices : :class:`Array`
        The indices of the elements in the resulting array.

    Returns
    -------
    :class:`Array`

    Notes
    -----
    Scatter strictly reorganizes array entries, it cannot assign multiple
    entries to the same position. In other words, the provided indices must be
    unique.
    '''

    array = Array.cast(__array)
    indices = Array.cast(indices)
    return _Wrapper(evaluable.Inflate,
                    array,
                    _WithoutPoints(indices),
                    _WithoutPoints(Array.cast(length)),
                    shape=array.shape[:array.ndim-indices.ndim] + (length,),
                    dtype=array.dtype)


@nutils_dispatch
def kronecker(__array: IntoArray, axis: int, length: int, pos: IntoArray) -> Array:
    '''Position an element in an axis of given length.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    axis : :class:`int`
        The axis to inflate. The elements are inflated from axes ``axis`` to
        ``axis+indices.ndim`` of the input array.
    length : :class:`int`
        The length of the inflated axis.
    pos : :class:`int` or :class:`Array`
        The index of the element in the resulting array.

    Returns
    -------
    :class:`Array`

    See Also
    --------
    :func:`get` : The complement operation.
    '''

    return _Transpose.from_end(scatter(__array, length, pos), axis)


def _argument_to_array(d: Any, array: Array) -> Iterable[Tuple[Argument, Array]]:
    '''Helper function for argument replacement.

    Given dictionary-like input, along with an array to look up argument names,
    yield ``(arg, new)`` tuples where ``arg`` is an ``Argument`` and ``new`` an
    ``Array`` of equal shape and dtype. A ``ValueError`` is raised if any key
    cannot be made a valid argument for the provided array or any value cannot
    be made an array of the same shape and dtype.

    Dictionary-like input is either an actual dictionary, a sequence of (key,
    value) tuples or strings, or a string, where any string is replaced by the
    relevant Argument object. The following inputs are all equivalent:

        d = {'u': 'v', 'p': 'q'}
        d = 'u:v,p:q'
        d = ('u:v', 'p:q')
        d = [('u', 'v'), ('p', 'q')]
        d = {'u': Argument('v', ...), 'p': Argument('q', ...)}
        d = [(Argument('u', ...), Argument('v', ...)), 'p:q']
    '''

    for item in d.split(',') if isinstance(d, str) else d.items() if isinstance(d, dict) else d:
        arg, new = item.split(':', 1) if isinstance(item, str) else item

        if isinstance(arg, str):
            if arg not in array.arguments:
                continue
            arg = Argument(arg, *array.arguments[arg])
        elif not isinstance(arg, Argument):
            raise ValueError('Key must be string or argument')
        elif arg.name not in arguments:
            continue
        elif array.arguments[arg.name] != (arg.shape, arg.dtype):
            raise ValueError(f'Argument {arg.name!r} has wrong shape or dtype')

        if isinstance(new, str):
            new = Argument(new, arg.shape, arg.dtype)
        else:
            new = Array.cast(new)
            if new.shape != arg.shape:
                raise ValueError(f'Argument {arg.name!r} has shape {arg.shape} but the replacement has shape {new.shape}.')
            elif new.dtype != arg.dtype:
                raise ValueError(f'Argument {arg.name!r} has dtype {arg.dtype.__name__} but the replacement has dtype {new.dtype.__name__}.')

        yield arg, new


@nutils_dispatch
def replace_arguments(__array: IntoArray, __arguments: Mapping[str, Union[IntoArray, str]]) -> Array:
    '''Replace arguments with :class:`Array` objects.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    arguments : :class:`dict` of :class:`str` and :class:`Array`
        The argument name to array mapping.

    Returns
    -------
    :class:`Array`
    '''

    return _Replace(Array.cast(__array), __arguments)


@nutils_dispatch
def linearize(__array: IntoArray, __arguments: Union[str, Dict[str, str], Iterable[str], Iterable[Tuple[str, str]]]):
    '''Linearize functional.

    Similar to :func:`derivative`, linearize takes the derivative of an array
    to one or more arguments, but with the derivative directions represented by
    arguments rather than array axes. The result is by definition linear in the
    new arguments.

    Parameters
    ----------
    array : :class:`Array` or something that can be :meth:`~Array.cast` into one
    arguments : :class:`str`, :class:`dict` or iterable of strings

    Example
    -------

    The following example demonstrates the use of linearize with four
    equivalent argument specifications:

    >>> u, v, p, q = [Argument(s, (), float) for s in 'uvpq']
    >>> f = u**2 + p
    >>> lin1 = linearize(f, 'u:v,p:q')
    >>> lin2 = linearize(f, dict(u='v', p='q'))
    >>> lin3 = linearize(f, ('u:v', 'p:q'))
    >>> lin4 = linearize(f, (('u', 'v'), ('p', 'q')))
    >>> # lin1 = lin2 == lin3 == lin4 == 2 * u * v + q
    '''

    array = Array.cast(__array)
    return util.sum(numpy.sum(derivative(array, arg) * lin, array.ndim + numpy.arange(arg.ndim))
        for arg, lin in _argument_to_array(__arguments, array))


def broadcast_arrays(*arrays: IntoArray) -> Tuple[Array, ...]:
    '''Broadcast the given arrays.

    Parameters
    ----------
    *arrays : :class:`Array` or similar

    Returns
    -------
    :class:`tuple` of :class:`Array`
        The broadcasted arrays.
    '''

    arrays_ = tuple(map(Array.cast, arrays))
    shape = broadcast_shapes(*(arg.shape for arg in arrays_))
    return tuple(numpy.broadcast_to(arg, shape) for arg in arrays_)


def typecast_arrays(*arrays: IntoArray, min_dtype: DType = bool):
    '''Cast the given arrays to the same dtype.

    Parameters
    ----------
    *arrays : :class:`Array` or similar

    Returns
    -------
    :class:`tuple` of :class:`Array`
        The typecasted arrays.
    '''

    arrays_ = tuple(map(Array.cast, arrays))
    dtype = builtins.max(min_dtype, *(arg.dtype for arg in arrays_), key=_dtypes.index)
    return tuple(arg.astype(dtype) for arg in arrays_)


def broadcast_shapes(*shapes: Shape) -> Tuple[int, ...]:
    '''Broadcast the given shapes into a single shape.

    Parameters
    ----------
    *shapes : :class:`tuple` or :class:`int`

    Returns
    -------
    :class:`tuple` of :class:`int`
        The broadcasted shape.
    '''

    if not shapes:
        raise ValueError('expected at least one shape but got none')
    broadcasted = []
    naxes = builtins.max(map(len, shapes))
    aligned_shapes = ((*(1,) * (naxes - len(shape)), *shape) for shape in shapes)
    for lengths in map(set, zip(*aligned_shapes)):
        if len(lengths) > 1:
            lengths.discard(1)
        if len(lengths) != 1:
            raise ValueError('cannot broadcast shapes {} because at least one or more axes have multiple lengths (excluding singletons)'.format(', '.join(map(str, shapes))))
        broadcasted.append(next(iter(lengths)))
    return tuple(broadcasted)


@nutils_dispatch
def derivative(__arg: IntoArray, __var: Union[str, 'Argument']) -> Array:
    '''Differentiate `arg` to `var`.

    Parameters
    ----------
    arg, var : :class:`Array` or something that can be :meth:`~Array.cast` into one

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(__arg)
    if isinstance(__var, str):
        if __var not in arg.arguments:
            raise ValueError('no such argument: {}'.format(__var))
        shape, dtype = arg.arguments[__var]
        __var = Argument(__var, shape, dtype=dtype)
    elif not isinstance(__var, Argument):
        raise ValueError('Expected an instance of `Argument` as second argument of `derivative` but got a `{}.{}`.'.format(type(__var).__module__, type(__var).__qualname__))
    if __var.name in arg.arguments:
        shape, dtype = arg.arguments[__var.name]
        if __var.shape != shape:
            raise ValueError('Argument {!r} has shape {} in the function, but the derivative to {!r} with shape {} was requested.'.format(__var.name, shape, __var.name, __var.shape))
        if __var.dtype != dtype:
            raise ValueError('Argument {!r} has dtype {} in the function, but the derivative to {!r} with dtype {} was requested.'.format(__var.name, dtype.__name__ if dtype in _dtypes else dtype, __var.name, __var.dtype.__name__ if __var.dtype in _dtypes else __var.dtype))
    return _Derivative(arg, __var)


@nutils_dispatch
def grad(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the gradient of the argument to the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
        The dimension of the local coordinate system.
    spaces : iterable of :class:`str`, optional
        Compute the gradient in ``spaces``. If absent all spaces of ``geom``
        are used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom)
    if geom.dtype != float:
        raise ValueError('The geometry must be real-valued.')
    if spaces is None:
        spaces = geom.spaces
    else:
        spaces = frozenset(spaces)
        if invariant_spaces := spaces - geom.spaces:
            invariant_spaces = ', '.join(sorted(invariant_spaces))
            raise ValueError(f'Gradient is singular because the geometry is invariant in the following space(s): {invariant_spaces}')
    if ndims == 0 or ndims == geom.size:
        op = _Gradient
    elif ndims == -1 or ndims == geom.size - 1:
        op = _SurfaceGradient
    else:
        raise NotImplementedError
    return numpy.reshape(op(arg, numpy.ravel(geom), spaces), arg.shape + geom.shape)


@nutils_dispatch
def curl(arg: IntoArray, geom: IntoArray, /, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the curl of the argument w.r.t. the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    spaces : iterable of :class:`str`, optional
        Compute the curl in ``spaces``. If absent all spaces of ``geom`` are
        used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom)
    if geom.dtype != float:
        raise ValueError('The geometry must be real-valued.')
    if geom.shape != (3,):
        raise ValueError('Expected a geometry with shape (3,) but got {}.'.format(geom.shape))
    if not arg.ndim:
        raise ValueError('Expected a function with at least 1 axis but got 0.')
    if arg.shape[-1] != 3:
        raise ValueError('Expected a function with a trailing axis of length 3 but got {}.'.format(arg.shape[-1]))
    return (levicivita(3).T * _append_axes(grad(arg, geom, spaces=spaces), (3,))).sum((-3, -2))


@nutils_dispatch
def normal(geom: IntoArray, /, refgeom: Optional[Array] = None, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the normal of the geometry.

    Parameters
    ----------
    geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    refgeom : :class:`Array`, optional`
        The reference geometry. If ``None``, the reference geometry is the tip
        coordinate system of the spaces on which ``geom`` is defined. The
        dimension of the reference geometry must be exactly one smaller than the
        dimension of the geometry.
    spaces : iterable of :class:`str`, optional
        Compute the normal in ``spaces``. If absent all spaces of ``geom`` are
        used.

    Returns
    -------
    :class:`Array`
    '''

    geom = Array.cast(geom)
    if geom.dtype != float:
        raise ValueError('The geometry must be real-valued.')
    if spaces is None:
        spaces = geom.spaces
    else:
        spaces = frozenset(spaces)
        if invariant_spaces := spaces - geom.spaces:
            invariant_spaces = ', '.join(sorted(invariant_spaces))
            raise ValueError(f'Normal is singular because the geometry is invariant in the following space(s): {invariant_spaces}')
    if refgeom is None:
        normal = _Normal(numpy.ravel(geom), spaces)
    else:
        if refgeom.dtype != float:
            raise ValueError('The reference geometry must be real-valued.')
        if refgeom.size != geom.size-1:
            raise ValueError(f'The reference geometry must have size {geom.size-1}, but got {refgeom.size}.')
        normal = _ExteriorNormal(grad(numpy.ravel(geom), numpy.ravel(refgeom), spaces=spaces))
    return numpy.reshape(normal, geom.shape)


def dotnorm(arg: IntoArray, geom: IntoArray, /, axis: int = -1, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the inner product of an array with the normal of the given geometry.

    Parameters
    ----------
    arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
        The array.
    geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
        The geometry. This must be a 1-D array.
    axis : :class:`int`
        The axis of ``arg`` along which the inner product should be performed.
        Defaults to the last axis.
    spaces : iterable of :class:`str`, optional
        Compute the inner product with the normal in ``spaces``. If absent all
        spaces of ``geom`` are used.

    Returns
    -------
    :class:`Array`
    '''

    return _Transpose.to_end(Array.cast(arg), axis) @ normal(geom, spaces=spaces)


def tangent(geom: IntoArray, vec: IntoArray, /, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the tangent.

    Parameters
    ----------
    geom, vec : :class:`Array` or something that can be :meth:`~Array.cast` into one
    spaces : iterable of :class:`str`, optional
        Compute the tangent in ``spaces``. If absent all spaces of ``geom`` are
        used.

    Returns
    -------
    :class:`Array`
    '''

    norm = normal(geom, spaces=spaces)
    vec = Array.cast(vec)
    return vec - (vec @ norm)[..., None] * norm


@nutils_dispatch
def jacobian(geom: IntoArray, ndims: Optional[int] = None, /, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the absolute value of the determinant of the Jacobian matrix of the given geometry.

    Parameters
    ----------
    geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
        The geometry.
    spaces : iterable of :class:`str`, optional
        Compute the jacobian in ``spaces``. If absent all spaces of ``geom``
        are used.

    Returns
    -------
    :class:`Array`
    '''

    geom = Array.cast(geom)
    if geom.dtype != float:
        raise ValueError('The geometry must be real-valued.')
    if spaces is None:
        spaces = geom.spaces
    else:
        spaces = frozenset(spaces)
        if invariant_spaces := spaces - geom.spaces:
            invariant_spaces = ', '.join(sorted(invariant_spaces))
            raise ValueError(f'Jacobian is singular because the geometry is invariant in the following space(s): {invariant_spaces}')
    return _Jacobian(numpy.ravel(geom), spaces, ndims)


def J(geom: IntoArray, ndims: Optional[int] = None, /, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the absolute value of the determinant of the Jacobian matrix of the given geometry.

    Alias of :func:`jacobian`.
    '''

    return jacobian(geom, ndims, spaces=spaces)


def _d1(arg: IntoArray, var: IntoArray) -> Array:
    return derivative(arg, var) if isinstance(var, Argument) else grad(arg, var)


def d(__arg: IntoArray, *vars: IntoArray) -> Array:
    return functools.reduce(_d1, vars, Array.cast(__arg))


@nutils_dispatch
def surfgrad(arg: IntoArray, /, geom: IntoArray, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the surface gradient of the argument to the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    spaces : iterable of :class:`str`, optional
        Compute the surface gradient in ``spaces``. If absent all spaces of
        ``geom`` are used.

    Returns
    -------
    :class:`Array`
    '''

    return grad(arg, geom, -1, spaces=spaces)


@nutils_dispatch
def curvature(geom: IntoArray, /, ndims: int = -1, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the curvature of the given geometry.

    Parameters
    ----------
    geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the curvature in ``spaces``. If absent all spaces of ``geom``
        are used.

    Returns
    -------
    :class:`Array`
    '''

    geom = Array.cast(geom)
    if spaces is not None:
        spaces = frozenset(spaces)
    return geom.normal(spaces=spaces).div(geom, ndims=ndims, spaces=spaces)


@nutils_dispatch
def div(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the divergence of ``arg`` w.r.t. the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the divergence in ``spaces``. If absent all spaces of ``geom``
        are used.

    Returns
    -------
    :class:`Array`
    '''

    geom = Array.cast(geom, ndim=1)
    return numpy.trace(grad(arg, geom, ndims, spaces=spaces), axis1=-2, axis2=-1)


@nutils_dispatch
def laplace(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the Laplacian of ``arg`` w.r.t. the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the Laplacian in ``spaces``. If absent all spaces of ``geom``
        are used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom, ndim=1)
    if spaces is not None:
        spaces = frozenset(spaces)
    return arg.grad(geom, ndims, spaces=spaces).div(geom, ndims, spaces=spaces)


def symgrad(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the symmetric gradient of ``arg`` w.r.t. the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the symmetric gradient in ``spaces``. If absent all spaces of
        ``geom`` are used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom)
    return .5 * add_T(arg.grad(geom, ndims, spaces=spaces))


def ngrad(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the inner product of the gradient of ``arg`` with the normal of the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the inner product of the gradient with the normal in
        ``spaces``. If absent all spaces of ``geom`` are used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom)
    if spaces is not None:
        spaces = frozenset(spaces)
    return dotnorm(grad(arg, geom, ndims, spaces=spaces), geom, spaces=spaces)


def nsymgrad(arg: IntoArray, geom: IntoArray, /, ndims: int = 0, *, spaces: Optional[Iterable[str]] = None) -> Array:
    '''Return the inner product of the symmetric gradient of ``arg`` with the normal of the given geometry.

    Parameters
    ----------
    arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
    ndims : :class:`int`
    spaces : iterable of :class:`str`, optional
        Compute the inner product of the symmetric gradient with the normal in
        ``spaces``. If absent all spaces of ``geom`` are used.

    Returns
    -------
    :class:`Array`
    '''

    arg = Array.cast(arg)
    geom = Array.cast(geom)
    if spaces is not None:
        spaces = frozenset(spaces)
    return dotnorm(symgrad(arg, geom, ndims, spaces=spaces), geom, spaces=spaces)

# MISC


@util.single_or_multiple
def eval(funcs: evaluable.AsEvaluableArray, /, arguments=None, **kwargs: numpy.ndarray) -> Tuple[numpy.ndarray, ...]:
    '''Evaluate one or several Array objects.

    Args
    ----
    funcs : :class:`tuple` of Array objects
        Arrays to be evaluated.
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.

    Returns
    -------
    results : :class:`tuple` of arrays
    '''

    if arguments is None:
        if kwargs:
            warnings.deprecation(
                'providing evaluation arguments as keyword arguments is '
                'deprecated, please use the "arguments" parameter instead',
                stacklevel=3)
        arguments = kwargs
    elif kwargs:
        raise ValueError('invalid argument {list(kwargs)[0]!r}')

    return evaluate(*funcs, arguments=arguments)


@nutils_dispatch
def evaluate(*arrays, arguments={}):
    return evaluable.eval_once(tuple(map(evaluable.asarray, arrays)), arguments=arguments)


def as_coo(array):
    '''Convert any array to an evaluable tuple of sparse COO data.

    The tuple consists of the array values, followed by the corresponding
    indices in all axes. Indices are lexicographically ordered and unique, but
    values are not guaranteed to be nonzero.'''

    values, indices, shape = array.as_evaluable_array.simplified.assparse
    return values, *indices


def as_csr(array):
    '''Convert a 2D array to an evaluable tuple of sparse CSR data.

    The tuple consists of the array values, row pointers, and column
    indices.'''

    if array.ndim != 2:
        raise ValueError('as_csr requires a 2D argument')
    values, rowptr, colidx, ncols = evaluable.as_csr(array.as_evaluable_array)
    return values, rowptr, colidx


def integral(func: IntoArray, sample) -> Array:
    '''Integrate a function over a sample.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    sample
        The integration sample.
    '''

    warnings.deprecation("function.integral is deprecated and will be removed in Nutils 10, please use the sample's .integral method instead")
    return sample.integral(func)


def sample(func: IntoArray, sample) -> Array:
    '''Evaluate a function in all sample points.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    sample
        The integration sample.
    '''

    warnings.deprecation("function.sample is deprecated and will be removed in Nutils 10, please use the sample's .bind method instead")
    return sample.bind(func)


def isarray(__arg: Any) -> bool:
    'Test if the argument is an instance of :class:`Array`.'
    return isinstance(__arg, Array)


def rootcoords(space: str, __dim: int) -> Array:
    'Return the root coordinates.'
    warnings.deprecation('function.rootcoords is deprecated and will be removed in Nutils 10')
    return _RootCoords(space, __dim)


def transforms_index(space: str, transforms: Transforms) -> Array:
    return _TransformsIndex(space, transforms)


def transforms_coords(space: str, transforms: Transforms) -> Array:
    return _TransformsCoords(space, transforms)


def piecewise(level: IntoArray, intervals: Sequence[IntoArray], *funcs: IntoArray) -> Array:
    'piecewise'
    level = Array.cast(level)
    return util.sum((level > interval).astype(int) for interval in intervals).choose(funcs)


def partition(f: IntoArray, *levels: float) -> Sequence[Array]:
    '''Create a partition of unity for a scalar function f.

    When ``n`` levels are specified, ``n+1`` indicator functions are formed that
    evaluate to one if and only if the following condition holds::

        indicator 0: f < levels[0]
        indicator 1: levels[0] < f < levels[1]
        ...
        indicator n-1: levels[n-2] < f < levels[n-1]
        indicator n: f > levels[n-1]

    At the interval boundaries the indicators evaluate to one half, in the
    remainder of the domain they evaluate to zero such that the whole forms a
    partition of unity. The partitions can be used to create a piecewise
    continuous function by means of multiplication and addition.

    The following example creates a topology consiting of three elements, and a
    function ``f`` that is zero in the first element, parabolic in the second,
    and zero again in the third element.

    >>> from nutils import mesh
    >>> domain, x = mesh.rectilinear([3])
    >>> left, center, right = partition(x[0], 1, 2)
    >>> f = (1 - (2*x[0]-3)**2) * center

    Args
    ----
    f : :class:`Array`
        Scalar-valued function
    levels : scalar constants or :class:`Array`\\s
        The interval endpoints.

    Returns
    -------
    :class:`list` of scalar :class:`Array`\\s
        The indicator functions.
    '''

    f = Array.cast(f)
    signs = [numpy.sign(f - level) for level in levels]
    return [.5 - .5 * signs[0]] + [.5 * (a - b) for a, b in zip(signs[:-1], signs[1:])] + [.5 + .5 * signs[-1]]


def heaviside(f: IntoArray):
    '''Create a heaviside step-function based on a scalar function f.

    .. math:: H(f) &= 0     && f < 0

              H(f) &= 0.5   && f = 0

              H(f) &= 1     && f > 0

    Args
    ----
    f : :class:`Array`
        Scalar-valued function

    Returns
    -------
    :class:`Array`
        The heaviside function.

    See Also
    --------

    :func:`partition`: generalized version of :func:`heaviside`
    '''

    return Array.cast(numpy.sign(f) * .5 + .5)


def chain(_funcs: Sequence[IntoArray]) -> Sequence[Array]:
    'chain'

    funcs = tuple(map(Array.cast, _funcs))
    shapes = [func.shape[0] for func in funcs]
    return [numpy.concatenate([func if i == j else zeros((sh,) + func.shape[1:])
        for j, sh in enumerate(shapes)], axis=0)
            for i, func in enumerate(funcs)]


def vectorize(args: Sequence[IntoArray]) -> Array:
    '''
    Combine scalar-valued bases into a vector-valued basis.

    Parameters
    ----
    args : iterable of 1-dimensional :class:`nutils.function.Array` objects

    Returns
    -------
    :class:`Array`
    '''

    args = tuple(args)
    return numpy.concatenate([kronecker(arg, axis=-1, length=len(args), pos=iarg) for iarg, arg in enumerate(args)])


def add_T(__arg: IntoArray, axes: Tuple[int, int] = (-2, -1)) -> Array:
    'add transposed'
    arg = Array.cast(__arg)
    return numpy.swapaxes(arg, *axes) + arg


def trignormal(_angle: IntoArray) -> Array:
    return Array.cast(numpy.stack([numpy.cos(_angle), numpy.sin(_angle)], axis=-1))


def trigtangent(_angle: IntoArray) -> Array:
    return Array.cast(numpy.stack([-numpy.sin(_angle), numpy.cos(_angle)], axis=-1))


def rotmat(__arg: IntoArray) -> Array:
    return Array.cast(numpy.stack([trignormal(__arg), trigtangent(__arg)], 0))


def dotarg(*args, **kwargs):
    '''Alias for :func:`field`.'''

    return field(*args, **kwargs)


@nutils_dispatch
def field(name: str, /, *arrays: IntoArray, shape: Tuple[int, ...] = (), dtype: DType = float) -> Array:
    '''Return the inner product of the first axes of the given arrays with an argument with the given name.

    An argument with shape ``(arrays[0].shape[0], ..., arrays[-1].shape[0]) +
    shape`` will be created. Repeatedly the inner product of the result, starting
    with the argument, with every array from ``arrays`` is taken, where all but
    the first axis are treated as an outer product.

    Parameters
    ----------
    name : :class:`str`
        The name of the argument.
    *arrays : :class:`Array` or something that can be :meth:`~Array.cast` into one
        The arrays to take inner products with.
    shape : :class:`tuple` of :class:`int`, optional
        The shape to be appended to the argument.
    dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
        The dtype of the argument.

    Returns
    -------
    :class:`Array`
        The inner product with shape ``shape + arrays[0].shape[1:] + ... + arrays[-1].shape[1:]``.
    '''

    result = Argument(name, tuple(array.shape[0] for array in arrays) + tuple(shape), dtype=dtype)
    for array in arrays:
        result = numpy.sum(_append_axes(result.transpose((*range(1, result.ndim), 0)), array.shape[1:]) * array, result.ndim-1)
    return result


@nutils_dispatch
def factor(array: Array) -> None:
    return _Factor(array)


class _Factor(Array):

    def __init__(self, array: Array) -> None:
        self._array = evaluable.factor(array)
        super().__init__(shape=array.shape, dtype=array.dtype, spaces=set(), arguments=array.arguments)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        return evaluable.prependaxes(self._array, args.points_shape)


@nutils_dispatch
def arguments_for(*arrays) -> Dict[str, Argument]:
    '''Get all arguments that array(s) depend on.

    Given any number of arrays, return a dictionary of all arguments involved,
    mapping the name to the :class:`Argument` object. Raise a ``ValueError`` if
    arrays have conflicting arguments, i.e. sharing a name but differing in
    shape and/or dtype.
    '''

    arguments = {}
    for array in arrays:
        if isinstance(array, Array):
            for name, (shape, dtype) in array.arguments.items():
                argument = arguments.get(name)
                if argument is None:
                    arguments[name] = Argument(name, shape, dtype)
                elif argument.shape != shape:
                    raise ValueError(f'inconsistent shapes for argument {name!r}')
                elif argument.dtype != dtype:
                    raise ValueError(f'inconsistent dtypes for argument {name!r}')
    return arguments


# BASES


def _int_or_vec(f, arg, argname, nargs, nvals):
    if isinstance(arg, numbers.Integral):
        return f(int(numeric.normdim(nargs, arg)))
    if numeric.isboolarray(arg):
        if arg.shape != (nargs,):
            raise IndexError('{} has invalid shape'.format(argname))
        arg, = arg.nonzero()
    if numeric.isintarray(arg):
        if arg.ndim != 1:
            raise IndexError('{} has invalid number of dimensions'.format(argname))
        if len(arg) == 0:
            return numpy.array([], dtype=int)
        arg = numpy.unique(arg)
        if arg[0] < 0 or arg[-1] >= nargs:
            raise IndexError('{} out of bounds'.format(argname))
        return functools.reduce(numpy.union1d, map(f, arg))
    raise IndexError('invalid {}'.format(argname))


def _int_or_vec_dof(f):
    @functools.wraps(f)
    def wrapped(self, dof: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
        return _int_or_vec(f.__get__(self), arg=dof, argname='dof', nargs=self.ndofs, nvals=self.nelems)
    return wrapped


def _int_or_vec_ielem(f):
    @functools.wraps(f)
    def wrapped(self, ielem: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
        return _int_or_vec(f.__get__(self), arg=ielem, argname='ielem', nargs=self.nelems, nvals=self.ndofs)
    return wrapped


class Basis(Array):
    '''Abstract base class for bases.

    A basis is a sequence of elementwise polynomial functions.

    Parameters
    ----------
    ndofs : :class:`int`
        The number of functions in this basis.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.

    Notes
    -----
    Subclasses must implement :meth:`get_dofs` and :meth:`get_coefficients` and
    if possible should redefine :meth:`get_support`.
    '''

    def __init__(self, ndofs: int, nelems: int, index: Array, coords: Array) -> None:
        self.ndofs = ndofs
        self.nelems = nelems
        self.index = Array.cast(index, dtype=int, ndim=0)
        self.coords = coords
        arguments = _join_arguments((index.arguments, coords.arguments))
        super().__init__((ndofs,), float, spaces=index.spaces | coords.spaces, arguments=arguments)

        _index = evaluable.InRange(evaluable.Argument('_index', shape=(), dtype=int), evaluable.constant(self.nelems))
        self._arg_dofs_evaluable, self._arg_coeffs_evaluable = self.f_dofs_coeffs(_index)
        self._arg_ndofs_evaluable = evaluable.asarray(self._arg_dofs_evaluable.shape[0])
        assert self._arg_dofs_evaluable.ndim == 1
        assert self._arg_coeffs_evaluable.ndim == 2
        assert not evaluable._certainly_different(self._arg_dofs_evaluable.shape[0], self._arg_coeffs_evaluable.shape[0])

    @cached_property
    def _arg_dofs(self):
        return evaluable.compile(self._arg_dofs_evaluable)

    @cached_property
    def _arg_coeffs(self):
        return evaluable.compile(self._arg_coeffs_evaluable)

    @cached_property
    def _arg_ndofs(self):
        return evaluable.compile(self._arg_ndofs_evaluable)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        index = self.index.lower(args.without_points)
        dofs, coeffs = self.f_dofs_coeffs(index)
        coords = self.coords.lower(args)
        return evaluable.Inflate(evaluable.Polyval(coeffs, coords), dofs, evaluable.constant(self.ndofs))

    @cached_property
    def _computed_support(self) -> Tuple[numpy.ndarray, ...]:
        support = [[] for i in range(self.ndofs)]  # type: List[List[int]]
        for ielem in range(self.nelems):
            for dof in numpy.unique(self.get_dofs(ielem)):
                support[dof].append(ielem)
        return tuple(types.frozenarray(ielems, dtype=int) for ielems in support)

    @_int_or_vec_dof
    def get_support(self, dof: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
        '''Return the support of basis function ``dof``.

        If ``dof`` is an :class:`int`, return the indices of elements that form the
        support of ``dof``.  If ``dof`` is an array, return the union of supports
        of the selected dofs as a unique array.  The returned array is always
        unique, i.e. strict monotonic increasing.

        Parameters
        ----------
        dof : :class:`int` or array of :class:`int` or :class:`bool`
            Index or indices of basis function or a mask.

        Returns
        -------
        support : sorted and unique :class:`numpy.ndarray`
            The elements (as indices) where function ``dof`` has support.
        '''

        return self._computed_support[dof]

    @_int_or_vec_ielem
    def get_dofs(self, ielem: Union[int, numpy.ndarray]) -> numpy.ndarray:
        '''Return an array of indices of basis functions with support on element ``ielem``.

        If ``ielem`` is an :class:`int`, return the dofs on element ``ielem``
        matching the coefficients array as returned by :meth:`get_coefficients`.
        If ``ielem`` is an array, return the union of dofs on the selected elements
        as a unique array, i.e. a strict monotonic increasing array.

        Parameters
        ----------
        ielem : :class:`int` or array of :class:`int` or :class:`bool`
            Element number(s) or mask.

        Returns
        -------
        dofs : :class:`numpy.ndarray`
            A 1D Array of indices.
        '''

        return self._arg_dofs(dict(_index=ielem))

    def get_ndofs(self, ielem: int) -> int:
        '''Return the number of basis functions with support on element ``ielem``.'''

        return int(self._arg_ndofs(dict(_index=numeric.normdim(self.nelems, ielem))))

    def get_coefficients(self, ielem: int) -> numpy.ndarray:
        '''Return an array of coefficients for all basis functions with support on element ``ielem``.

        Parameters
        ----------
        ielem : :class:`int`
            Element number.

        Returns
        -------
        coefficients : :class:`numpy.ndarray`
            Array of coefficients with shape ``(nlocaldofs,)+(degree,)*ndims``,
            where the first axis corresponds to the dofs returned by
            :meth:`get_dofs`.
        '''

        return self._arg_coeffs(dict(_index=numeric.normdim(self.nelems, ielem)))

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        raise NotImplementedError('{} must implement f_dofs_coeffs'.format(self.__class__.__name__))

    def __getitem__(self, index: Any) -> Array:
        if numeric.isintarray(index) and index.ndim == 1 and numpy.all(numpy.greater(numpy.diff(index), 0)):
            return MaskedBasis(self, index)
        elif numeric.isboolarray(index) and index.shape == (self.ndofs,):
            return MaskedBasis(self, numpy.where(index)[0])
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.ndofs)
            if step == 1 and start == 0 and stop == self.ndofs:
                return self
            else:
                return MaskedBasis(self, numpy.arange(start, stop, step))
        else:
            return super().__getitem__(index)

    @treelog.withcontext
    def discontinuous_at_partition_interfaces(self, part_indices: Sequence[int]):
        '''Returns a basis that is discontinuous at element partition interfaces.

        Given a partition of elements, this basis is made discontinuous at the
        partition interfaces. All elements that have the same part index belong
        to the same part.

        The returned basis is formed by clipping each function of the basis to
        each part individually and stacking all nonzero clipped functions. As a
        consequence, if a basis function has support on three topologically
        adjacent elements of which the first and the last element belong to one
        part and the middle to another, this function will not be clipped to
        each of the three elements individually, but to the first and the last
        element and to the middle element.

        Parameters
        ----------
        part_indices : sequence or :class:`numpy.ndarray` of :class:`int`
            For each element the index of the part the element belongs to.
        '''

        return _DiscontinuousPartitionBasis(self, part_indices)


class PlainBasis(Basis):
    '''A general purpose implementation of a :class:`Basis`.

    Use this class only if there exists no specific implementation of
    :class:`Basis` for the basis at hand.

    Parameters
    ----------
    coefficients : :class:`tuple` of :class:`numpy.ndarray` objects
        The coefficients of the basis functions per transform.  The order should
        match the ``transforms`` argument.
    dofs : :class:`tuple` of :class:`numpy.ndarray` objects
        The dofs corresponding to the ``coefficients`` argument.
    ndofs : :class:`int`
        The number of basis functions.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.
    '''

    def __init__(self, coefficients: Sequence[numpy.ndarray], dofs: Sequence[numpy.ndarray], ndofs: int, index: Array, coords: Array) -> None:
        self._coeffs = tuple(types.arraydata(numpy.asarray(c, dtype=float)) for c in coefficients)
        self._dofs = tuple(map(types.arraydata, dofs))
        assert len(self._coeffs) == len(self._dofs)
        assert all(c.ndim == 2 for c in self._coeffs)
        assert all(c.shape[0] == d.shape[0] for c, d in zip(self._coeffs, self._dofs))
        super().__init__(ndofs, len(coefficients), index, coords)

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        dofs = evaluable.Elemwise(self._dofs, index, dtype=int)
        coeffs = evaluable.Elemwise(self._coeffs, index, dtype=float)
        return dofs, coeffs


class DiscontBasis(Basis):
    '''A discontinuous basis with monotonic increasing dofs.

    Parameters
    ----------
    coefficients : :class:`tuple` of :class:`numpy.ndarray` objects
        The coefficients of the basis functions per transform.  The order should
        match the ``transforms`` argument.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.
    '''

    def __init__(self, coefficients: Sequence[numpy.ndarray], index: Array, coords: Array) -> None:
        self._coeffs = tuple(types.arraydata(c) for c in coefficients)
        assert all(c.ndim == 2 for c in self._coeffs)
        self._offsets = numpy.cumsum([0] + [c.shape[0] for c in self._coeffs])
        super().__init__(self._offsets[-1], len(coefficients), index, coords)

    @_int_or_vec_dof
    def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
        ielem = numpy.searchsorted(self._offsets[:-1], numeric.normdim(self.ndofs, dof), side='right')-1
        return numpy.array([ielem], dtype=int)

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        coeffs = evaluable.Elemwise(self._coeffs, index, dtype=float)
        dofs = evaluable.Range(coeffs.shape[0]) + evaluable.get(evaluable.constant(self._offsets), 0, index)
        return dofs, coeffs


class LegendreBasis(Basis):
    '''A discontinuous Legendre basis.

    Parameters
    ----------
    degree : :class:`int`
        The degree of the basis.
    nelems : :class:`int`
        The number of elements.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.
    '''

    def __init__(self, degree: int, nelems: int, index: Array, coords: Array) -> None:
        if coords.shape[-1] != 1:
            raise NotImplementedError('The Legendre basis is only implemented for dimension 1.')
        self._degree = degree
        super().__init__(nelems * (degree+1), nelems, index, coords)

    @_int_or_vec_dof
    def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
        if isinstance(dof, int):
            dof = numpy.array([dof])
        return numpy.unique(dof // (self._degree+1))

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        dofs = evaluable.Range(evaluable.constant(self._degree+1)) + index * (self._degree+1)
        coeffs = numpy.zeros((self._degree+1,)*2, dtype=int)
        for n in range(self._degree+1):
            for k in range(n+1):
                coeffs[n, self._degree - k] = (-1 if (n+k) % 2 else 1) * numeric.binom(n, k) * numeric.binom(n+k, k)
        return dofs, evaluable.astype(evaluable.asarray(coeffs), float)

    def lower(self, args: LowerArgs) -> evaluable.Array:
        index = self.index.lower(args.without_points)
        coords = self.coords.lower(args)
        leg = evaluable.Legendre(evaluable.get(coords, coords.ndim-1, evaluable.constant(0)) * 2. - 1., self._degree)
        dofs = evaluable.Range(evaluable.constant(self._degree+1)) + index * (self._degree+1)
        return evaluable.Inflate(leg, dofs, evaluable.constant(self.ndofs))


class MaskedBasis(Basis):
    '''An order preserving subset of another :class:`Basis`.

    Parameters
    ----------
    parent : :class:`Basis`
        The basis to mask.
    indices : array of :class:`int`\\s
        The strict monotonic increasing indices of ``parent`` basis functions to
        keep.
    '''

    def __init__(self, parent: Basis, indices: numpy.ndarray) -> None:
        indices = types.frozenarray(indices)
        if indices.ndim != 1:
            raise ValueError('`indices` should have one dimension but got {}'.format(indices.ndim))
        if len(indices) and not numpy.all(numpy.greater(numpy.diff(indices), 0)):
            raise ValueError('`indices` should be strictly monotonic increasing')
        if len(indices) and (indices[0] < 0 or indices[-1] >= len(parent)):
            raise ValueError('`indices` out of range \x5b0,{}\x29'.format(len(parent)))
        self._parent = parent
        self._indices = indices
        self._renumber = evaluable.constant(numeric.invmap(indices, length=parent.ndofs, missing=len(indices)))
        super().__init__(len(indices), parent.nelems, parent.index, parent.coords)

    def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
        if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
            raise IndexError('dof out of bounds')
        return self._parent.get_support(self._indices[dof])

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        p_dofs, p_coeffs = self._parent.f_dofs_coeffs(index)
        renumber = evaluable.Take(self._renumber, p_dofs)
        selection = evaluable.Find(evaluable.Less(renumber, evaluable.InsertAxis(evaluable.constant(self.ndofs), p_dofs.shape[0])))
        dofs = evaluable.take(renumber, selection, axis=0)
        coeffs = evaluable.take(p_coeffs, selection, axis=0)
        return dofs, coeffs


class StructuredBasis(Basis):
    '''A basis for class:`nutils.transformseq.StructuredTransforms`.

    Parameters
    ----------
    coeffs : :class:`tuple` of :class:`tuple`\\s of arrays
        Per dimension the coefficients of the basis functions per transform.
    start_dofs : :class:`tuple` of arrays of :class:`int`\\s
        Per dimension the dof of the first entry in ``coeffs`` per transform.
    stop_dofs : :class:`tuple` of arrays of :class:`int`\\s
        Per dimension one plus the dof of the last entry  in ``coeffs`` per
        transform.
    dofs_shape : :class:`tuple` of :class:`int`\\s
        The tensor shape of the dofs.
    transforms_shape : :class:`tuple` of :class:`int`\\s
        The tensor shape of the transforms.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.
    '''

    def __init__(self, coeffs: Sequence[Sequence[numpy.ndarray]], start_dofs: Sequence[numpy.ndarray], stop_dofs: Sequence[numpy.ndarray], dofs_shape: Sequence[int], transforms_shape: Sequence[int], index: Array, coords: Array) -> None:
        self._coeffs = tuple(tuple(map(types.arraydata, c)) for c in coeffs)
        self._start_dofs = tuple(map(types.frozenarray, start_dofs))
        self._stop_dofs = tuple(map(types.frozenarray, stop_dofs))
        self._ndofs = tuple(types.frozenarray(b-a) for a, b in zip(self._start_dofs, self._stop_dofs))
        self._dofs_shape = tuple(map(int, dofs_shape))
        self._transforms_shape = tuple(map(int, transforms_shape))
        super().__init__(util.product(dofs_shape), util.product(transforms_shape), index, coords)

    @_int_or_vec_dof
    def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
        dof = numeric.normdim(self.ndofs, dof)
        ndofs = 1
        ntrans = 1
        supports = []
        for start_dofs_i, stop_dofs_i, ndofs_i, ntrans_i in zip(reversed(self._start_dofs), reversed(self._stop_dofs), reversed(self._dofs_shape), reversed(self._transforms_shape)):
            dof, dof_i = builtins.divmod(dof, ndofs_i)
            supports_i = []
            while dof_i < stop_dofs_i[-1]:
                stop_ielem = numpy.searchsorted(start_dofs_i, dof_i, side='right')
                start_ielem = numpy.searchsorted(stop_dofs_i, dof_i, side='right')
                supports_i.append(numpy.arange(start_ielem, stop_ielem, dtype=int))
                dof_i += ndofs_i
            supports.append(numpy.unique(numpy.concatenate(supports_i)) * ntrans)
            ndofs *= ndofs_i
            ntrans *= ntrans_i
        assert dof == 0
        return numpy.asarray(functools.reduce(numpy.add.outer, reversed(supports)).ravel(), dtype=int)

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        indices = []
        for n in reversed(self._transforms_shape[1:]):
            index, ielem = evaluable.divmod(index, n)
            indices.append(ielem)
        indices.append(index)
        indices.reverse()
        ranges = [evaluable.Range(evaluable.get(evaluable.constant(lengths_i), 0, index_i)) + evaluable.get(evaluable.constant(offsets_i), 0, index_i)
                  for lengths_i, offsets_i, index_i in zip(self._ndofs, self._start_dofs, indices)]
        ndofs = self._dofs_shape[0]
        dofs = ranges[0] % ndofs
        for range_i, ndofs_i in zip(ranges[1:], self._dofs_shape[1:]):
            dofs = evaluable.Ravel(evaluable.RavelIndex(dofs, range_i % ndofs_i, evaluable.constant(ndofs), evaluable.constant(ndofs_i)))
            ndofs = ndofs * ndofs_i
        coeffs_per_dim = iter(evaluable.Elemwise(coeffs_i, index_i, float) for coeffs_i, index_i in zip(self._coeffs, indices))
        coeffs = next(coeffs_per_dim)
        for i, c in enumerate(coeffs_per_dim, 1):
            coeffs, c = evaluable.insertaxis(coeffs, 1, c.shape[0]), evaluable.insertaxis(c, 0, coeffs.shape[0])
            coeffs = evaluable.PolyMul(coeffs, c, (poly.MulVar.Left,) * i + (poly.MulVar.Right,))
            coeffs = evaluable.ravel(coeffs, 0)
        return dofs, coeffs


class PrunedBasis(Basis):
    '''A subset of another :class:`Basis`.

    Parameters
    ----------
    parent : :class:`Basis`
        The basis to prune.
    transmap : one-dimensional array of :class:`int`\\s
        The indices of transforms in ``parent`` that form this subset.
    index : :class:`Array`
        The element index.
    coords : :class:`Array`
        The element local coordinates.
    '''

    def __init__(self, parent: Basis, transmap: numpy.ndarray, index: Array, coords: Array) -> None:
        self._parent = parent
        self._transmap = types.frozenarray(transmap)
        self._dofmap = parent.get_dofs(self._transmap)
        self._renumber = types.frozenarray(numeric.invmap(self._dofmap, length=parent.ndofs, missing=len(self._dofmap)), copy=False)
        super().__init__(len(self._dofmap), len(transmap), index, coords)

    def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
        if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
            raise IndexError('dof out of bounds')
        return numeric.sorted_index(self._transmap, self._parent.get_support(self._dofmap[dof]), missing='mask')

    def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array, evaluable.Array]:
        p_dofs, p_coeffs = self._parent.f_dofs_coeffs(evaluable.get(evaluable.constant(self._transmap), 0, index))
        dofs = evaluable.take(evaluable.constant(self._renumber), p_dofs, axis=0)
        return dofs, p_coeffs


class _DiscontinuousPartitionBasis(Basis):

    def __init__(self, parent: Basis, part_indices: Sequence[int]):
        self._parent = parent

        part_indices = numpy.array(part_indices).astype(int, casting='safe', copy=False)
        if part_indices.shape != (parent.nelems,):
            raise ValueError(f'expected a sequence of {self.nelems} integers but got an array with shape {part_indices.shape}')

        # For each element we pair the parent dofs with the part indices and
        # use that as partitioned dof. Then we renumber the partitioned dofs
        # starting at 0, ordered by part index, then by parent dof.

        ielem = evaluable.loop_index('ielem', parent.nelems)
        parent_dofs, _ = parent.f_dofs_coeffs(ielem)
        # Concatenate all parent dofs and stack all ndofs per element.
        cc_parent_dofs = evaluable.loop_concatenate(parent_dofs, ielem)
        cc_ndofs = evaluable.loop_concatenate(evaluable.insertaxis(parent_dofs.shape[0], 0, evaluable.asarray(1)), ielem)
        cc_parent_dofs, cc_ndofs = evaluable.eval_once((cc_parent_dofs, cc_ndofs))
        # Stack the part index for each element for each dof.
        cc_part_indices = numpy.repeat(part_indices, cc_ndofs)
        # Renumber and count all unique dofs.
        unique_dofs, dofs = numpy.unique(numpy.stack([cc_part_indices, cc_parent_dofs], axis=1), axis=0, return_inverse=True)

        self._dofs = evaluable.asarray(dofs)
        self._ndofs = evaluable.asarray(cc_ndofs)
        self._offsets = evaluable._SizesToOffsets(self._ndofs)

        super().__init__(len(unique_dofs), parent.nelems, parent.index, parent.coords)

    def f_dofs_coeffs(self, index):
        dofs = evaluable.Take(self._dofs, evaluable.Range(evaluable.Take(self._ndofs, index)) + evaluable.Take(self._offsets, index))
        _, coeffs = self._parent.f_dofs_coeffs(index)
        return dofs, coeffs


def Namespace(*args, **kwargs):
    from .expression_v1 import Namespace
    return Namespace(*args, **kwargs)


HANDLED_FUNCTIONS = {}

class __implementations__:

    def implements(np_function):
        'Register an ``__array_function__`` or ``__array_ufunc__`` implementation for Array objects.'
        def decorator(func):
            HANDLED_FUNCTIONS[np_function] = func
            return func
        return decorator

    @implements(numpy.shape)
    def shape(arg: Array) -> Tuple[int, ...]:
        return arg.shape

    @implements(numpy.ndim)
    def ndim(arg: Array) -> int:
        return arg.ndim
    
    @implements(numpy.size)
    def size(arg: Array) -> int:
        return arg.size

    @implements(numpy.add)
    def add(left: IntoArray, right: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.add, left, right)
    
    @implements(numpy.subtract)
    def subtract(left: IntoArray, right: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.subtract, left, right)

    @implements(numpy.positive)
    def positive(arg: Array) -> Array:
        return arg

    @implements(numpy.negative)
    def negative(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.negative, arg)

    @implements(numpy.multiply)
    def multiply(left: IntoArray, right: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.multiply, left, right)
    
    @implements(numpy.true_divide)
    def divide(dividend: IntoArray, divisor: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.divide, dividend, divisor, min_dtype=float)

    @implements(numpy.reciprocal)
    def reciprocal(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.reciprocal, arg)

    @implements(numpy.floor_divide)
    def floor_divide(dividend: IntoArray, divisor: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.FloorDivide, dividend, divisor)

    @implements(numpy.mod)
    def mod(dividend: IntoArray, divisor: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Mod, dividend, divisor)

    @implements(numpy.divmod)
    def divmod(dividend: IntoArray, divisor: IntoArray) -> Tuple[Array, Array]:
        return numpy.floor_divide(dividend, divisor), numpy.mod(dividend, divisor)

    @implements(numpy.power)
    def power(base: IntoArray, exponent: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.power, base, exponent, min_dtype=int)

    @implements(numpy.sqrt)
    def sqrt(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.sqrt, arg, min_dtype=float)

    @implements(numpy.square)
    def square(arg: Array) -> Array:
        return numpy.power(arg, 2)

    @implements(numpy.hypot)
    def hypot(array1: IntoArray, array2: IntoArray) -> Array:
        return numpy.sqrt(numpy.square(array1) + numpy.square(array2))

    @implements(numpy.absolute)
    def abs(arg: IntoArray) -> Array:
        arg = Array.cast(arg)
        return _Wrapper(evaluable.abs, arg, shape=arg.shape, dtype=float if arg.dtype == complex else arg.dtype)

    @implements(numpy.sign)
    def sign(arg: IntoArray) -> Array:
        arg = Array.cast(arg)
        if arg.dtype == complex:
            raise ValueError('sign is not defined for complex numbers')
        return _Wrapper.broadcasted_arrays(evaluable.Sign, arg)

    @implements(numpy.matmul)
    def matmul(arg1: IntoArray, arg2: IntoArray) -> Array:
        arg1 = Array.cast(arg1)
        arg2 = Array.cast(arg2)
        if not arg1.ndim or not arg2.ndim:
            raise ValueError('cannot contract zero-dimensional array')
        if arg2.ndim == 1:
            return (arg1 * arg2).sum(-1)
        elif arg1.ndim == 1:
            return (arg1[:, numpy.newaxis] * arg2).sum(-2)
        else:
            return (arg1[..., :, :, numpy.newaxis] * arg2[..., numpy.newaxis, :, :]).sum(-2)

    @implements(numpy.sin)
    def sin(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Sin, arg, min_dtype=float)

    @implements(numpy.cos)
    def cos(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Cos, arg, min_dtype=float)

    @implements(numpy.tan)
    def tan(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Tan, arg, min_dtype=float)

    @implements(numpy.arcsin)
    def arcsin(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.ArcSin, arg, min_dtype=float)

    @implements(numpy.arccos)
    def arccos(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.ArcCos, arg, min_dtype=float)

    @implements(numpy.arctan)
    def arctan(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.ArcTan, arg, min_dtype=float)

    @implements(numpy.arctan2)
    def arctan2(dividend: IntoArray, divisor: IntoArray) -> Array:
        dividend, divisor = broadcast_arrays(*typecast_arrays(dividend, divisor, min_dtype=float))
        if dividend.dtype == complex:
            raise ValueError('arctan2 is not defined for complex numbers')
        return _Wrapper(evaluable.ArcTan2, dividend, divisor, shape=dividend.shape, dtype=float)

    @implements(numpy.sinc)
    def sinc(arg: Array) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.sinc, arg * numpy.pi, min_dtype=float)

    @implements(numpy.cosh)
    def cosh(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.cosh, arg, min_dtype=float)

    @implements(numpy.sinh)
    def sinh(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.sinh, arg, min_dtype=float)

    @implements(numpy.tanh)
    def tanh(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.tanh, arg, min_dtype=float)

    @implements(numpy.arctanh)
    def arctanh(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.arctanh, arg, min_dtype=float)

    @implements(numpy.exp)
    def exp(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Exp, arg, min_dtype=float)

    @implements(numpy.log)
    def log(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Log, arg, min_dtype=float)

    @implements(numpy.log2)
    def log2(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.log2, arg, min_dtype=float)

    @implements(numpy.log10)
    def log10(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.log10, arg, min_dtype=float)

    @implements(numpy.greater)
    def greater(left: IntoArray, right: IntoArray) -> Array:
        left, right = map(Array.cast, (left, right))
        if left.dtype == complex or right.dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        return _Wrapper.broadcasted_arrays(evaluable.Greater, left, right, force_dtype=bool)

    @implements(numpy.equal)
    def equal(left: IntoArray, right: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.Equal, left, right, force_dtype=bool)

    @implements(numpy.less)
    def less(left: IntoArray, right: IntoArray) -> Array:
        left, right = map(Array.cast, (left, right))
        if left.dtype == complex or right.dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        return _Wrapper.broadcasted_arrays(evaluable.Less, left, right, force_dtype=bool)

    @implements(numpy.minimum)
    def minimum(a: IntoArray, b: IntoArray) -> Array:
        a, b = map(Array.cast, (a, b))
        if a.dtype == complex or b.dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        return _Wrapper.broadcasted_arrays(evaluable.Minimum, a, b)

    @implements(numpy.maximum)
    def maximum(a: IntoArray, b: IntoArray) -> Array:
        a, b = map(Array.cast, (a, b))
        if a.dtype == complex or b.dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        return _Wrapper.broadcasted_arrays(evaluable.Maximum, a, b)

    @implements(numpy.logical_and)
    @implements(numpy.bitwise_and)
    def logical_and(a: IntoArray, b: IntoArray) -> Array:
        a, b = map(Array.cast, (a, b))
        if a.dtype != bool or b.dtype != bool:
            return NotImplemented
        return _Wrapper.broadcasted_arrays(evaluable.multiply, a, b)

    @implements(numpy.logical_or)
    @implements(numpy.bitwise_or)
    def logical_or(a: IntoArray, b: IntoArray) -> Array:
        a, b = map(Array.cast, (a, b))
        if a.dtype != bool or b.dtype != bool:
            return NotImplemented
        return _Wrapper.broadcasted_arrays(evaluable.add, a, b)

    @implements(numpy.logical_not)
    @implements(numpy.invert)
    def logical_not(a: IntoArray) -> Array:
        a = Array.cast(a)
        if a.dtype != bool:
            return NotImplemented
        return _Wrapper.broadcasted_arrays(evaluable.LogicalNot, a, force_dtype=bool)

    @implements(numpy.all)
    def all(a: IntoArray, axis = None) -> Array:
        a = Array.cast(a)
        if a.dtype != bool:
            return NotImplemented
        if axis is None:
            a = numpy.ravel(a)
        elif isinstance(axis, int):
            a = _Transpose.to_end(a, axis)
        else:
            return NotImplemented
        return _Wrapper(evaluable.Product, a, shape=a.shape[:-1], dtype=bool)

    @implements(numpy.any)
    def any(a: IntoArray, axis = None) -> Array:
        a = Array.cast(a)
        if a.dtype != bool:
            return NotImplemented
        if axis is None:
            a = numpy.ravel(a)
        elif isinstance(axis, int):
            a = _Transpose.to_end(a, axis)
        else:
            return NotImplemented
        return _Wrapper(evaluable.Sum, a, shape=a.shape[:-1], dtype=bool)

    @implements(numpy.sum)
    def sum(arg: IntoArray, axis: Optional[Union[int, Sequence[int]]] = None) -> Array:
        arg = Array.cast(arg)
        if arg.dtype == bool:
            arg = arg.astype(int)
        axes = range(arg.ndim) if axis is None else [axis] if isinstance(axis, numbers.Integral) else axis
        summed = _Transpose.to_end(arg, *axes)
        for i in range(len(axes)):
            summed = _Wrapper(evaluable.Sum, summed, shape=summed.shape[:-1], dtype=summed.dtype)
        return summed

    @implements(numpy.prod)
    def prod(arg: IntoArray, axis: int) -> Array:
        arg = Array.cast(arg)
        if arg.dtype == bool:
            arg = arg.astype(int)
        axes = range(arg.ndim) if axis is None else [axis] if isinstance(axis, numbers.Integral) else axis
        multiplied = _Transpose.to_end(arg, *axes)
        for i in range(len(axes)):
            multiplied = _Wrapper(evaluable.Product, multiplied, shape=multiplied.shape[:-1], dtype=multiplied.dtype)
        return multiplied

    @implements(numpy.conjugate)
    def conjugate(arg: IntoArray) -> Array:
        return _Wrapper.broadcasted_arrays(evaluable.conjugate, arg)

    @implements(numpy.real)
    def real(arg: IntoArray) -> Array:
        arg = Array.cast(arg)
        return _Wrapper(evaluable.real, arg, shape=arg.shape, dtype=float if arg.dtype == complex else arg.dtype)

    @implements(numpy.imag)
    def imag(arg: IntoArray) -> Array:
        arg = Array.cast(arg)
        return _Wrapper(evaluable.imag, arg, shape=arg.shape, dtype=float if arg.dtype == complex else arg.dtype)

    @implements(numpy.vdot)
    def vdot(a: IntoArray, b: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> Array:
        a, b = broadcast_arrays(a, b)
        return numpy.sum(numpy.conjugate(a) * b, range(a.ndim))

    @implements(numpy.dot)
    def dot(a: IntoArray, b: IntoArray) -> Array:
        a = Array.cast(a)
        b = Array.cast(b)
        if a.ndim == 0 or b.ndim == 0:
            return (a * b)
        if a.shape[-1] != b.shape[-1 if b.ndim == 1 else -2]:
            raise ValueError(f'shapes {a.shape} and {b.shape} are not aligned')
        if b.ndim > 1:
            b = _Transpose.to_end(b, -2)
            a = _Transpose.to_end(_append_axes(a, b.shape[:-1]), a.ndim-1)
            assert a.shape[-b.ndim:] == b.shape
        return numpy.sum(a * b, -1)

    @implements(numpy.reshape)
    def reshape(arg: Array, newshape):
        if isinstance(newshape, numbers.Integral):
            newshape = newshape,
        if -1 in newshape:
            i = newshape.index(-1)
            if -1 in newshape[i+1:]:
                raise ValueError('can only specify one unknown dimension')
            length, remainder = builtins.divmod(arg.size, numpy.prod(newshape, initial=-1))
            if remainder:
                raise ValueError(f'cannot reshape array of size {arg.size} into shape {newshape}')
            newshape = (*newshape[:i], length, *newshape[i+1:])
        elif numpy.prod(newshape, initial=1) != arg.size:
            raise ValueError(f'cannot reshape array of size {arg.size} into shape {newshape}')
        ncommon = 0
        while arg.ndim > ncommon and len(newshape) > ncommon and arg.shape[ncommon] == newshape[ncommon]:
            ncommon += 1
        # The first ncommon axes are already of the right shape, so these we
        # will not touch. The remaining axes will be ravelled and unravelled
        # until an axis of the desired length is formed, working from end to
        # beginning, and rolling finished axes to the front to reduce the
        # number of transposes.
        for i, s in enumerate(reversed(newshape[ncommon:])):
            if arg.ndim == ncommon + i: # the first i axes are finished so we need to append a new singleton axis to continue
                assert s == 1
                arg = _Wrapper(evaluable.InsertAxis, arg, _WithoutPoints(Array.cast(1)), shape=(*arg.shape, 1), dtype=arg.dtype)
            else:
                while arg.shape[-1] % s:
                    assert arg.ndim > ncommon + i + 1
                    arg = _Wrapper(evaluable.Ravel, arg, shape=(*arg.shape[:-2], arg.shape[-2]*arg.shape[-1]), dtype=arg.dtype)
                if arg.shape[-1] != s:
                    n = arg.shape[-1] // s
                    arg = _Wrapper(evaluable.Unravel, arg, _WithoutPoints(Array.cast(n)), _WithoutPoints(Array.cast(s)), shape=(*arg.shape[:-1], n, s), dtype=arg.dtype)
            arg = _Transpose.from_end(arg, ncommon) # move axis to front so that we can continue to operate on the end
        # If the original array had a surplus of singleton axes, these may
        # still be present in the tail. We take them away one by one.
        while arg.ndim > len(newshape):
            assert arg.shape[-1] == 1
            arg = _Wrapper(evaluable.Take, arg, _WithoutPoints(Array.cast(0)), shape=arg.shape[:-1], dtype=arg.dtype)
        assert arg.shape == newshape
        return arg

    @implements(numpy.ravel)
    def ravel(arg: Array):
        return numpy.reshape(arg, -1)

    @implements(numpy.trace)
    def trace(arg: Array, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
        return numpy.sum(numpy.diagonal(arg, offset, axis1, axis2), -1)

    @implements(numpy.transpose)
    def transpose(array: Array, axes: Optional[Sequence[int]] = None) -> Array:
        return _Transpose(array, tuple(reversed(range(array.ndim)) if axes is None else axes))

    @implements(numpy.repeat)
    def repeat(array: IntoArray, n: IntoArray, axis: int) -> Array:
        array = Array.cast(array)
        if array.shape[axis] != 1:
            raise NotImplementedError('only axes with length 1 can be repeated')
        return insertaxis(get(array, axis, 0), axis, n)

    @implements(numpy.swapaxes)
    def swapaxes(array: Array, axis1: int, axis2: int) -> Array:
        trans = list(range(array.ndim))
        trans[axis1], trans[axis2] = trans[axis2], trans[axis1]
        return numpy.transpose(array, trans)

    @implements(numpy.take)
    def take(array: IntoArray, indices: IntoArray, axis: Optional[int] = None) -> Array:
        array = Array.cast(array)
        if axis is None:
            array = numpy.ravel(array)
            axis = 0
        else:
            axis = numeric.normdim(array.ndim, axis)
        length = array.shape[axis]
        indices = util.deep_reduce(numpy.stack, indices)
        if isinstance(indices, Array):
            indices = _Wrapper.broadcasted_arrays(evaluable.NormDim, length, indices)
        else:
            indices = numpy.array(indices)
            indices[indices < 0] += length
            if (indices < 0).any() or (indices >= length).any():
                raise ValueError('indices out of bounds')
            indices = _Constant(indices)
        transposed = _Transpose.to_end(array, axis)
        taken = _Wrapper(evaluable.Take, transposed, _WithoutPoints(indices), shape=(*transposed.shape[:-1], *indices.shape), dtype=array.dtype)
        return _Transpose.from_end(taken, *range(axis, axis+indices.ndim))

    @implements(numpy.compress)
    def compress(condition: Sequence[bool], array: Array, axis: Optional[int] = None) -> Array:
        length = array.size if axis is None else array.shape[axis]
        if len(condition) != length:
            # NOTE: we are a bit stricter here than numpy, which does not check
            # the length of the condition but only whether the selected indices
            # are within bounds.
            raise ValueError(f'expected a condition of length {length} but received {len(condition)}')
        indices, = numpy.nonzero(condition)
        return numpy.take(array, indices, axis)

    @implements(numpy.concatenate)
    def concatenate(__arrays: Sequence[IntoArray], axis: int = 0) -> Array:
        return _Concatenate(__arrays, axis)

    @implements(numpy.stack)
    def stack(__arrays: Sequence[IntoArray], axis: int = 0) -> Array:
        aligned = broadcast_arrays(*typecast_arrays(*__arrays))
        return util.sum(kronecker(array, axis, len(aligned), i) for i, array in enumerate(aligned))

    @implements(numpy.broadcast_to)
    def broadcast_to(array: IntoArray, shape: Shape) -> Array:
        broadcasted = Array.cast(array)
        orig_shape = broadcasted.shape
        if broadcasted.ndim > len(shape):
            raise ValueError('cannot broadcast array with shape {} to {} because the dimension decreases'.format(orig_shape, shape))
        nnew = len(shape) - broadcasted.ndim
        broadcasted = _prepend_axes(broadcasted, shape[:nnew])
        for axis, (actual, desired) in enumerate(zip(broadcasted.shape[nnew:], shape[nnew:])):
            if actual == desired:
                continue
            elif actual == 1:
                broadcasted = numpy.repeat(broadcasted, desired, axis + nnew)
            else:
                raise ValueError('cannot broadcast array with shape {} to {} because input axis {} is neither singleton nor has the desired length'.format(orig_shape, shape, axis))
        return broadcasted

    @implements(numpy.searchsorted)
    def searchsorted(a, v: IntoArray, side='left', sorter=None):
        values = Array.cast(v)
        array = Array.cast(a)
        if side not in ('left', 'right'):
            raise ValueError(f'expected "left" or "right", got {side}')
        if sorter is not None:
            sorter = Array.cast(sorter)
            if sorter.shape != array.shape or sorter.dtype != int:
                raise ValueError('invalid sorter array')
            lower = functools.partial(evaluable.SearchSorted, side=side)
            return _Wrapper(lower, values, _WithoutPoints(array), _WithoutPoints(sorter), shape=values.shape, dtype=int)
        else:
            lower = functools.partial(evaluable.SearchSorted, sorter=None, side=side)
            return _Wrapper(lower, values, _WithoutPoints(array), shape=values.shape, dtype=int)

    @implements(numpy.interp)
    def interp(x, xp, fp, left=None, right=None):
        index = numpy.searchsorted(xp, x)
        _xp = numpy.concatenate([[xp[0]], xp])
        _fp = numpy.concatenate([[fp[0]], fp])
        _gp = numpy.concatenate([[0.], numpy.diff(fp) / numpy.diff(xp), [0.]])
        if left is not None:
            _fp[0] = left
        if right is not None:
            _fp[-1] = right
        def take_index(a):
            a = _Constant(a)
            return _Wrapper(evaluable.Take, _WithoutPoints(a), index, shape=index.shape, dtype=a.dtype)
        return take_index(_fp) + take_index(_gp) * (x - take_index(_xp))

    @implements(numpy.choose)
    def choose(a, choices):
        a, *choices = broadcast_arrays(a, *typecast_arrays(*choices))
        return _Wrapper(evaluable.Choose, a, numpy.stack(choices, -1), shape=a.shape, dtype=choices[0].dtype)

    @implements(numpy.linalg.norm)
    def norm(x, ord=None, axis=None):
        if ord is not None:
            raise NotImplementedError('only "ord" values of None are supported for now')
        if axis is None:
            axis = range(x.ndim)
        # NOTE while the sum of squares is always positive, we wrap it in
        # maximum(0, ..) to guard against the situation that the function
        # simplifies to a form in which round-off errors may nudge it below
        # zero (e.g. (a-b)^2 -> a^2 - 2 a b + b^2).
        return numpy.sqrt(numpy.maximum(0, numpy.sum(numpy.real(x)**2 + numpy.imag(x)**2, axis)))

    def _eig(symmetric, index, a):
        return evaluable.Eig(a, symmetric)[index]

    @implements(numpy.linalg.eig)
    def eig(a):
        return _Wrapper(functools.partial(__implementations__._eig, False, 0), a, shape=a.shape[:-1], dtype=complex), \
               _Wrapper(functools.partial(__implementations__._eig, False, 1), a, shape=a.shape, dtype=complex)

    @implements(numpy.linalg.eigh)
    def eigh(a):
        return _Wrapper(functools.partial(__implementations__._eig, True, 0), a, shape=a.shape[:-1], dtype=float), \
               _Wrapper(functools.partial(__implementations__._eig, True, 1), a, shape=a.shape, dtype=float if a.dtype != complex else complex)

    @implements(numpy.linalg.det)
    def det(a):
        if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
            raise ValueError('Last 2 dimensions of the array must be square')
        return _Wrapper(evaluable.Determinant, a, shape=a.shape[:-2], dtype=complex if a.dtype == complex else float)

    @implements(numpy.linalg.inv)
    def inv(a):
        if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
            raise ValueError('Last 2 dimensions of the array must be square')
        return _Wrapper(evaluable.Inverse, a, shape=a.shape, dtype=complex if a.dtype == complex else float)

    @implements(numpy.ndim)
    def ndim(a):
        return a.ndim

    @implements(numpy.size)
    def size(a):
        return a.size

    @implements(numpy.shape)
    def shape(a):
        return a.shape

    @implements(numpy.diagonal)
    def diagonal(a, offset=0, axis1=0, axis2=1):
        if a.shape[axis1] != a.shape[axis2]:
            raise ValueError('axis lengths do not match')
        arg = _Transpose.to_end(a, axis1, axis2)
        if offset > 0:
            arg = arg[...,:-offset,offset:]
        elif offset < 0:
            arg = arg[...,-offset:,:offset]
        return _Wrapper(evaluable.TakeDiag, arg, shape=arg.shape[:-1], dtype=a.dtype)

    @implements(numpy.einsum)
    def einsum(subscripts, *operands):
        *in_, out = numeric.sanitize_einsum_subscripts(subscripts, *map(numpy.shape, operands))
        axes = list(out)
        factors = []
        for s, operand in zip(in_, operands):
            for i, c in reversed(list(enumerate(s))):
                if c in s[i+1:]: # duplicate label -> diagonal
                    j = i + 1 + s[i+1:].index(c)
                    operand = numpy.diagonal(operand, 0, i, j)
                    s = s[:i] + s[i+1:j] + s[j+1:] + c
                elif c not in axes:
                    axes.insert(0, c) # prepended output axes will be summed
            transpose = sorted(range(numpy.ndim(operand)), key=lambda i: axes.index(s[i]))
            insert = tuple(slice(None) if c in s else numpy.newaxis for c in axes)
            factors.append(numpy.transpose(operand, transpose)[insert])
        return numpy.sum(util.product(factors), range(len(axes)-len(out)))

    @implements(numpy.cross)
    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        if axis is not None:
            axisa = axisb = axisc = axis
        a = _Transpose.to_end(a, axisa)
        b = _Transpose.to_end(b, axisb)
        if a.shape[-1] == b.shape[-1] == 2:
            return numpy.einsum('ij,...i,...j', levicivita(2), a, b)
        elif a.shape[-1] == b.shape[-1] == 3:
            return _Transpose.from_end(numpy.einsum('ijk,...j,...k', levicivita(3), a, b), axisc)
        else:
            raise ValueError('dimension must be 2 or 3')
