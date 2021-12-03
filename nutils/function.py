# Copyright (c) 2020 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import typing
if typing.TYPE_CHECKING:
  from typing_extensions import Protocol
else:
  Protocol = object

from typing import Tuple, Union, Type, Callable, Sequence, Any, Optional, Iterator, Iterable, Dict, Mapping, List, FrozenSet
from . import evaluable, numeric, util, types, warnings, debug_flags, sparse
from .transform import EvaluableTransformChain
from .transformseq import Transforms
import builtins, numpy, functools, operator, numbers

IntoArray = Union['Array', numpy.ndarray, bool, int, float]
Shape = Sequence[int]
DType = Type[Union[bool, int, float]]
_dtypes = bool, int, float

_PointsShape = Tuple[evaluable.Array, ...]
_TransformChainsMap = Mapping[str, Tuple[EvaluableTransformChain, EvaluableTransformChain]]
_CoordinatesMap = Mapping[str, evaluable.Array]

class Lowerable(Protocol):
  'Protocol for lowering to :class:`nutils.evaluable.Array`.'

  @property
  def spaces(self) -> FrozenSet[str]: ...

  @property
  def arguments(self) -> Mapping[str, Tuple[Shape, DType]]: ...

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    '''Lower this object to a :class:`nutils.evaluable.Array`.

    Parameters
    ----------
    points_shape : :class:`tuple` of scalar, integer :class:`nutils.evaluable.Array`
        The shape of the leading points axes that are to be added to the
        lowered :class:`nutils.evaluable.Array`.
    transform_chains : mapping of :class:`str` to :class:`nutils.transform.EvaluableTransformChain` pairs
    coordinates : mapping of :class:`str` to :class:`nutils.evaluable.Array` objects
        The coordinates at which the function will be evaluated.
    '''

_ArrayMeta = type

if debug_flags.lower:
  def _debug_lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    result = self._ArrayMeta__debug_lower_orig(points_shape, transform_chains, coordinates)
    assert isinstance(result, evaluable.Array)
    assert all(evaluable.equalshape(coords.shape[:-1], points_shape) for coords in coordinates.values())
    assert all(space in transform_chains for space in coordinates)
    offset = 0 if type(self) == _WithoutPoints else len(points_shape)
    assert result.ndim == self.ndim + offset
    assert tuple(int(sh) for sh in result.shape[offset:]) == self.shape, 'shape mismatch'
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

def _cache_lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
  key = points_shape, transform_chains, coordinates
  cached_key, cached_result = getattr(self, '_ArrayMeta__cached_lower', (None, None))
  if cached_key == key:
    return cached_result
  missing_spaces = self.spaces - set(transform_chains)
  if missing_spaces:
    raise ValueError('Cannot lower {} because the following spaces are unspecified: {}.'.format(self, missing_spaces))
  result = self._ArrayMeta__cache_lower_orig(points_shape, transform_chains, coordinates)
  self._ArrayMeta__cached_lower = key, result
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
  dtype : :class:`bool`, :class:`int` or :class:`float`
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
  dtype : :class:`bool`, :class:`int` or :class:`float`
      The dtype of the array elements.
  spaces : :class:`frozenset` of :class:`str`
      The spaces this array function is defined on.
  arguments : mapping of :class:`str`
      The mapping of argument names to their shapes and dtypes for all
      arguments of this array function.
  '''

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

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

    if isinstance(__value, Array):
      value = __value
    else:
      try:
        value = _Constant(__value)
      except:
        if isinstance(__value, (list, tuple)):
          value = stack(__value, axis=0)
        else:
          raise ValueError('cannot convert {}.{} to Array'.format(type(__value).__module__, type(__value).__qualname__))
    if dtype is not None and _dtypes.index(value.dtype) > _dtypes.index(dtype):
      raise ValueError('expected an array with dtype `{}` but got `{}`'.format(dtype.__name__, value.dtype.__name__))
    if ndim is not None and value.ndim != ndim:
      raise ValueError('expected an array with dimension `{}` but got `{}`'.format(ndim, value.ndim))
    return value

  def __init__(self, shape: Shape, dtype: DType, spaces: FrozenSet[str], arguments: Mapping[str, Tuple[Shape, DType]]) -> None:
    self.shape = tuple(sh.__index__() for sh in shape)
    self.dtype = dtype
    self.spaces = frozenset(spaces)
    self.arguments = dict(arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    raise NotImplementedError

  @util.cached_property
  def as_evaluable_array(self) -> evaluable.Array:
    return self.lower((), {}, {})

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
      if isinstance(it, numbers.Integral):
        array = get(array, axis, it)
      else:
        array = expand_dims(array, axis) if it is numpy.newaxis \
           else _takeslice(array, it, axis) if isinstance(it, slice) \
           else take(array, it, axis)
        axis += 1
    assert axis == array.ndim
    return array

  def __bool__(self) -> bool:
    return True

  def __len__(self) -> int:
    'Length of the first axis.'

    if self.ndim == 0:
      raise TypeError('len() of unsized object')
    return self.shape[0]

  def __iter__(self) -> Iterator['Array']:
    'Iterator over the first axis.'

    if self.ndim == 0:
      raise TypeError('iteration over a 0-D array')
    return (self[i,...] for i in range(self.shape[0]))

  @property
  def size(self) -> Union[int, 'Array']:
    'The total number of elements in this array.'

    return util.product(self.shape, 1)

  @property
  def T(self) -> 'Array':
    'The transposed array.'

    return transpose(self)

  def astype(self, dtype):
    if dtype == self.dtype:
      return self
    else:
      return _Wrapper(functools.partial(evaluable.astype, dtype=dtype), self, shape=self.shape, dtype=dtype)

  def sum(self, axis: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
    'See :func:`sum`.'
    return sum(self, axis)

  def prod(self, __axis: int) -> 'Array':
    'See :func:`prod`.'
    return product(self, __axis)

  def dot(self, __other: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> 'Array':
    'See :func:`dot`.'
    return dot(self, __other, axes)

  def normalized(self, __axis: int = -1) -> 'Array':
    'See :func:`normalized`.'
    return normalized(self, __axis)

  def normal(self, exterior: bool = False) -> 'Array':
    'See :func:`normal`.'
    return normal(self, exterior)

  def curvature(self, ndims: int = -1) -> 'Array':
    'See :func:`curvature`.'
    return curvature(self, ndims)

  def swapaxes(self, __axis1: int, __axis2: int) -> 'Array':
    'See :func:`swapaxes`.'
    return swapaxes(self, __axis1, __axis2)

  def transpose(self, __axes: Optional[Sequence[int]]) -> 'Array':
    'See :func:`transpose`.'
    return transpose(self, __axes)

  def add_T(self, axes: Tuple[int, int]) -> 'Array':
    'See :func:`add_T`.'
    return add_T(self, axes)

  def grad(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`grad`.'
    return grad(self, __geom, ndims)

  def laplace(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`laplace`.'
    return laplace(self, __geom, ndims)

  def symgrad(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`symgrad`.'
    return symgrad(self, __geom, ndims)

  def div(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`div`.'
    return div(self, __geom, ndims)

  def curl(self, __geom: IntoArray) -> 'Array':
    'See :func:`curl`.'
    return curl(self, __geom)

  def dotnorm(self, __geom: IntoArray, axis: int = -1) -> 'Array':
    'See :func:`dotnorm`.'
    return dotnorm(self, __geom, axis)

  def tangent(self, __vec: IntoArray) -> 'Array':
    'See :func:`tangent`.'
    return tangent(self, __vec)

  def ngrad(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`ngrad`.'
    return ngrad(self, __geom, ndims)

  def nsymgrad(self, __geom: IntoArray, ndims: int = 0) -> 'Array':
    'See :func:`nsymgrad`.'
    return nsymgrad(self, __geom, ndims)

  def choose(self, __choices: Sequence[IntoArray]) -> 'Array':
    'See :func:`choose`.'
    return choose(self, __choices)

  def vector(self, ndims):
    if not self.ndim:
      raise Exception('a scalar function cannot be vectorized')
    return ravel(diagonalize(insertaxis(self, 1, ndims), 1), 0)

  def __repr__(self) -> str:
    return 'Array<{}>'.format(','.join(str(n) for n in self.shape))

  @property
  def simplified(self):
    warnings.deprecation('`nutils.function.Array.simplified` is deprecated. This property returns the array unmodified and can safely be omitted.')
    return self

  def eval(self, **arguments: Any) -> numpy.ndarray:
    'Evaluate this function.'

    from .sample import eval_integrals
    return eval_integrals(self, **arguments)[0]

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
    return {name: shape for name, (shape, dtype) in self.arguments.items()}

class _Unlower(Array):

  def __init__(self, array: evaluable.Array, spaces: FrozenSet[str], arguments: Mapping[str, Tuple[Shape, DType]], points_shape: Tuple[evaluable.Array, ...], transform_chains: Tuple[EvaluableTransformChain, ...], coordinates: Tuple[evaluable.Array, ...]) -> None:
    self._array = array
    self._points_shape = points_shape
    self._transform_chains = transform_chains
    self._coordinates = coordinates
    shape = tuple(n.__index__() for n in array.shape[len(points_shape):])
    super().__init__(shape=shape, dtype=array.dtype, spaces=spaces, arguments=arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    if self._points_shape != points_shape or self._transform_chains != transform_chains or self._coordinates != coordinates:
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
  decorated as ``classmethod`` or ``staticmethod``, meaning that they will not
  receive a reference to ``self`` when called. Instead, all relevant data
  should be passed to ``evalf`` via the constructor argument ``args``. The
  constructor will automatically distinguish between Array and non-Array
  arguments, and pass the latter on to ``evalf`` unchanged. The
  ``partial_derivative`` will not be called for those arguments.

  The lowered array does not have a Nutils hash by default. If this is desired,
  the methods :meth:`evalf` and :meth:`partial_derivative` can be decorated
  with :func:`nutils.types.hashable_function` in addition to ``classmethod`` or
  ``staticmethod``.

  Parameters
  ----------
  args : iterable of :class:`Array` objects or immutable and hashable objects
      The arguments of this array function.
  shape : :class:`tuple` of :class:`int` or :class:`Array`
      The shape of the array function without leading pointwise axes.
  dtype : :class:`bool`, :class:`int` or :class:`float`
      The dtype of the array elements.
  npointwise : :class:`int`
      The number of leading pointwise axis.

  Example
  -------

  The following class implements :func:`multiply` using :class:`Custom`
  without broadcasting and for :class:`float` arrays only.

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
  ...   @staticmethod
  ...   def evalf(left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
  ...     # Because all axes are pointwise, the evaluated `left` and `right`
  ...     # arrays are 1d.
  ...     return left * right
  ...
  ...   @staticmethod
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
  ...   @staticmethod
  ...   def evalf(array: numpy.ndarray, shift: int) -> numpy.ndarray:
  ...     # `array` is evaluated to a `numpy.ndarray` because we passed `array`
  ...     # as an `Array` to the constructor. `shift`, however, is untouched
  ...     # because it is not an `Array`. The `array` has two axes: a points
  ...     # axis and the axis to be rolled.
  ...     return numpy.roll(array, shift, 1)
  ...
  ...   @staticmethod
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
      args = tuple(broadcast_to(arg, points_shape + arg.shape[npointwise:]) if isinstance(arg, Array) else arg for arg in args)
    else:
      points_shape = ()
    self._args = args
    self._npointwise = npointwise
    spaces = frozenset(space for arg in args if isinstance(arg, Array) for space in arg.spaces)
    arguments = _join_arguments(arg.arguments for arg in args if isinstance(arg, Array))
    super().__init__(shape=(*points_shape, *shape), dtype=dtype, spaces=spaces, arguments=arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    args = tuple(arg.lower(points_shape, transform_chains, coordinates) if isinstance(arg, Array) else evaluable.EvaluableConstant(arg) for arg in self._args) # type: Tuple[Union[evaluable.Array, evaluable.EvaluableConstant], ...]
    add_points_shape = tuple(map(evaluable.asarray, self.shape[:self._npointwise]))
    points_shape += add_points_shape
    coordinates = {space: evaluable.Transpose.to_end(evaluable.appendaxes(coords, add_points_shape), coords.ndim-1) for space, coords in coordinates.items()}
    return _CustomEvaluable(type(self).__name__, self.evalf, self.partial_derivative, args, self.shape[self._npointwise:], self.dtype, self.spaces, types.frozendict(self.arguments), points_shape, tuple(transform_chains.items()), tuple(coordinates.items()))

  @classmethod
  def evalf(cls, *args: Any) -> numpy.ndarray:
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

    raise NotImplementedError # pragma: nocover

  @classmethod
  def partial_derivative(cls, iarg: int, *args: Any) -> IntoArray:
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

    raise NotImplementedError('The partial derivative of {} to argument {} (counting from 0) is not defined.'.format(cls.__name__, iarg)) # pragma: nocover

class _CustomEvaluable(evaluable.Array):

  def __init__(self, name, evalf, partial_derivative, args: Tuple[Union[evaluable.Array, evaluable.EvaluableConstant], ...], shape: Tuple[int, ...], dtype: DType, spaces: FrozenSet[str], arguments: types.frozendict, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> None:
    assert all(isinstance(arg, (evaluable.Array, evaluable.EvaluableConstant)) for arg in args)
    self.name = name
    self.custom_evalf = evalf
    self.custom_partial_derivative = partial_derivative
    self.args = args
    self.points_dim = len(points_shape)
    self.lower_args = points_shape, dict(transform_chains), dict(coordinates)
    self.spaces = spaces
    self.function_arguments = arguments
    super().__init__((evaluable.Tuple(points_shape), *args), shape=points_shape+shape, dtype=dtype)

  @property
  def _node_details(self) -> str:
    return self.name

  def evalf(self, points_shape: Tuple[numpy.ndarray, ...], *args: Any) -> numpy.ndarray:
    points_shape = tuple(n.__index__() for n in points_shape)
    npoints = util.product(points_shape, 1)
    # Flatten the points axes of the array arguments and call `custom_evalf`.
    flattened = (arg.reshape(npoints, *arg.shape[self.points_dim:]) if isinstance(origarg, evaluable.Array) else arg for arg, origarg in zip(args, self.args))
    result = self.custom_evalf(*flattened)
    assert result.ndim == self.ndim + 1 - self.points_dim
    # Unflatten the points axes of the result. If there are no array arguments,
    # the points axis must have length one. Otherwise the length must be
    # `npoints` (checked by `reshape`).
    if not any(isinstance(origarg, evaluable.Array) for origarg in self.args):
      if result.shape[0] != 1:
        raise ValueError('Expected a points axis of length one but got {}.'.format(result.shape[0]))
      return numpy.broadcast_to(result[0], points_shape + result.shape[1:])
    else:
      return result.reshape(points_shape + result.shape[1:])

  def _derivative(self, var: evaluable.Array, seen: Dict[evaluable.Array, evaluable.Array]) -> evaluable.Array:
    if self.dtype != float:
      return super()._derivative(var, seen)
    result = evaluable.Zeros(self.shape + var.shape, dtype=self.dtype)
    unlowered_args = tuple(_Unlower(arg, self.spaces, self.function_arguments, *self.lower_args) if isinstance(arg, evaluable.Array) else arg.value for arg in self.args)
    for iarg, arg in enumerate(self.args):
      if not isinstance(arg, evaluable.Array) or arg.dtype != float or var not in arg.dependencies and var != arg:
        continue
      fpd = Array.cast(self.custom_partial_derivative(iarg, *unlowered_args))
      fpd_expected_shape = tuple(n.__index__() for n in self.shape[self.points_dim:] + arg.shape[self.points_dim:])
      if fpd.shape != fpd_expected_shape:
        raise ValueError('`partial_derivative` to argument {} returned an array with shape {} but {} was expected.'.format(iarg, fpd.shape, fpd_expected_shape))
      epd = evaluable.appendaxes(fpd.lower(*self.lower_args), var.shape)
      eda = evaluable.derivative(arg, var, seen)
      eda = evaluable.Transpose.from_end(evaluable.appendaxes(eda, self.shape[self.points_dim:]), *range(self.points_dim, self.ndim))
      result += (epd * eda).sum(range(self.ndim, self.ndim + arg.ndim - self.points_dim))
    return result

class _WithoutPoints:

  def __init__(self, __arg: Array) -> None:
    self._arg = __arg
    self.spaces = __arg.spaces
    self.arguments = __arg.arguments

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return self._arg.lower((), transform_chains, {})

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

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return self._lower(*(arg.lower(points_shape, transform_chains, coordinates) for arg in self._args))

class _Zeros(Array):

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return evaluable.Zeros((*points_shape, *self.shape), self.dtype)

class _Ones(Array):

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return evaluable.ones((*points_shape, *self.shape), self.dtype)

class _Constant(Array):

  def __init__(self, value: Any) -> None:
    self._value = types.arraydata(value)
    super().__init__(self._value.shape, self._value.dtype, frozenset(()), {})

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return evaluable.prependaxes(evaluable.Constant(self._value), points_shape)

class Argument(Array):
  '''Array valued function argument.

  Parameters
  ----------
  name : str
      The name of this argument.
  shape : :class:`tuple` of :class:`int`
      The shape of this argument.
  dtype : :class:`bool`, :class:`int` or :class:`float`
      The dtype of the array elements.

  Attributes
  ----------
  name : str
      The name of this argument.
  '''

  def __init__(self, name: str, shape: Shape, *, dtype: DType = float) -> None:
    self.name = name
    super().__init__(shape, dtype, frozenset(()), {name: (tuple(shape), dtype)})

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return evaluable.prependaxes(evaluable.Argument(self.name, self.shape, self.dtype), points_shape)

class _Replace(Array):

  def __init__(self, arg: Array, replacements: Dict[str, Array]) -> None:
    self._arg = arg
    # TODO: verify that the replacements have empty spaces
    self._replacements = replacements
    # Build arguments map with replacements.
    unreplaced = {}
    arguments = [unreplaced]
    for name, (shape, dtype) in arg.arguments.items():
      replacement = replacements.get(name, None)
      if replacement is None:
        unreplaced[name] = shape, dtype
      elif replacement.shape != shape:
        raise ValueError('Argument {!r} has shape {} but the replacement has shape {}.'.format(name, shape, replacement.shape))
      elif replacement.dtype != dtype:
        raise ValueError('Argument {!r} has dtype {} but the replacement has dtype {}.'.format(name, dtype.__name__ if dtype in _dtypes else dtype, replacement.dtype.__name__ if replacement.dtype in _dtypes else replacement.dtype))
      else:
        arguments.append(replacement.arguments)
    super().__init__(arg.shape, arg.dtype, arg.spaces, _join_arguments(arguments))

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    arg = self._arg.lower(points_shape, transform_chains, coordinates)
    replacements = {name: _WithoutPoints(value).lower(points_shape, transform_chains, coordinates) for name, value in self._replacements.items()}
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
    self._axes = axes
    super().__init__(tuple(arg.shape[axis] for axis in axes), arg.dtype, arg.spaces, arg.arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    arg = self._arg.lower(points_shape, transform_chains, coordinates)
    offset = len(points_shape)
    axes = (*range(offset), *(i+offset for i in self._axes))
    return evaluable.Transpose(arg, axes)

class _Opposite(Array):

  def __init__(self, arg: Array, space: str) -> None:
    self._arg = arg
    self._space = space
    super().__init__(arg.shape, arg.dtype, arg.spaces, arg.arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    transform_chains = dict(transform_chains)
    transform_chains[self._space] = transform_chains[self._space][::-1]
    return self._arg.lower(points_shape, transform_chains, coordinates)

class _RootCoords(Array):

  def __init__(self, space: str, ndims: int) -> None:
    self._space = space
    super().__init__((ndims,), float, frozenset({space}), {})

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    inv_linear = evaluable.diagonalize(evaluable.ones(self.shape))
    inv_linear = evaluable.prependaxes(inv_linear, points_shape)
    tip_coords = coordinates[self._space]
    tip_coords = evaluable.WithDerivative(tip_coords, _tip_derivative_target(self._space, tip_coords.shape[-1]), evaluable.Diagonalize(evaluable.ones(tip_coords.shape)))
    coords = transform_chains[self._space][0].apply(tip_coords)
    return evaluable.WithDerivative(coords, _root_derivative_target(self._space, self.shape[0]), inv_linear)

class _TransformsIndex(Array):

  def __init__(self, space: str, transforms: Transforms) -> None:
    self._space = space
    self._transforms = transforms
    super().__init__((), int, frozenset({space}), {})

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    index, tail = transform_chains[self._space][0].index_with_tail_in(self._transforms)
    return evaluable.prependaxes(index, points_shape)

class _TransformsCoords(Array):

  def __init__(self, space: str, transforms: Transforms) -> None:
    self._space = space
    self._transforms = transforms
    super().__init__((transforms.fromdims,), float, frozenset({space}), {})

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    index, tail = transform_chains[self._space][0].index_with_tail_in(self._transforms)
    head = self._transforms.get_evaluable(index)
    L = head.linear
    if self._transforms.todims > self._transforms.fromdims:
      LTL = evaluable.einsum('ki,kj->ij', L, L)
      Linv = evaluable.einsum('ik,jk->ij', evaluable.inverse(LTL), L)
    else:
      Linv = evaluable.inverse(L)
    Linv = evaluable.prependaxes(Linv, points_shape)
    tip_coords = coordinates[self._space]
    tip_coords = evaluable.WithDerivative(tip_coords, _tip_derivative_target(self._space, tip_coords.shape[-1]), evaluable.Diagonalize(evaluable.ones(tip_coords.shape)))
    coords = tail.apply(tip_coords)
    return evaluable.WithDerivative(coords, _root_derivative_target(self._space, self._transforms.todims), Linv)

class _Derivative(Array):

  def __init__(self, arg: Array, var: Argument) -> None:
    assert isinstance(var, Argument)
    self._arg = arg
    self._var = var
    self._eval_var = evaluable.Argument(var.name, var.shape, var.dtype)
    arguments = _join_arguments((arg.arguments, var.arguments))
    super().__init__(arg.shape+var.shape, arg.dtype, arg.spaces | var.spaces, arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    arg = self._arg.lower(points_shape, transform_chains, coordinates)
    return evaluable.derivative(arg, self._eval_var)

def _tip_derivative_target(space: str, dim: int) -> evaluable.DerivativeTargetBase:
  return evaluable.IdentifierDerivativeTarget((space, 'tip'), (dim,))

def _root_derivative_target(space: str, dim: int) -> evaluable.DerivativeTargetBase:
  return evaluable.IdentifierDerivativeTarget((space, 'root'), (dim,))

class _Gradient(Array):
  # Derivative of `func` to `geom` using the root coords as reference.

  def __init__(self, func: Array, geom: Array) -> None:
    assert geom.spaces, '0d array'
    common_shape = broadcast_shapes(func.shape, geom.shape[:-1])
    self._func = broadcast_to(func, common_shape)
    self._geom = broadcast_to(geom, (*common_shape, geom.shape[-1]))
    arguments = _join_arguments((func.arguments, geom.arguments))
    super().__init__(self._geom.shape, float, func.spaces | geom.spaces, arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    func = self._func.lower(points_shape, transform_chains, coordinates)
    geom = self._geom.lower(points_shape, transform_chains, coordinates)
    ref_dim = builtins.sum(transform_chains[space][0].todims for space in self._geom.spaces)
    if self._geom.shape[-1] != ref_dim:
      raise Exception('cannot invert {}x{} jacobian'.format(self._geom.shape[-1], ref_dim))
    refs = tuple(_root_derivative_target(space, chain.todims) for space, (chain, opposite) in transform_chains.items() if space in self._geom.spaces)
    dfunc_dref = evaluable.concatenate([evaluable.derivative(func, ref) for ref in refs], axis=-1)
    dgeom_dref = evaluable.concatenate([evaluable.derivative(geom, ref) for ref in refs], axis=-1)
    dref_dgeom = evaluable.inverse(dgeom_dref)
    return evaluable.einsum('Ai,Aij->Aj', dfunc_dref, dref_dgeom)

class _SurfaceGradient(Array):
  # Surface gradient of `func` to `geom` using the tip coordinates as
  # reference.

  def __init__(self, func: Array, geom: Array) -> None:
    assert geom.spaces, '0d array'
    common_shape = broadcast_shapes(func.shape, geom.shape[:-1])
    self._func = broadcast_to(func, common_shape)
    self._geom = broadcast_to(geom, (*common_shape, geom.shape[-1]))
    arguments = _join_arguments((func.arguments, geom.arguments))
    super().__init__(self._geom.shape, float, func.spaces | geom.spaces, arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    func = self._func.lower(points_shape, transform_chains, coordinates)
    geom = self._geom.lower(points_shape, transform_chains, coordinates)
    ref_dim = builtins.sum(transform_chains[space][0].fromdims for space in self._geom.spaces)
    if self._geom.shape[-1] != ref_dim + 1:
      raise ValueError('expected a {}d geometry but got a {}d geometry'.format(ref_dim + 1, self._geom.shape[-1]))
    refs = tuple((_root_derivative_target if chain.todims == chain.fromdims else _tip_derivative_target)(space, chain.fromdims) for space, (chain, opposite) in transform_chains.items() if space in self._geom.spaces)
    dfunc_dref = evaluable.concatenate([evaluable.derivative(func, ref) for ref in refs], axis=-1)
    dgeom_dref = evaluable.concatenate([evaluable.derivative(geom, ref) for ref in refs], axis=-1)
    dref_dgeom = evaluable.einsum('Ajk,Aik->Aij', dgeom_dref, evaluable.inverse(evaluable.grammium(dgeom_dref)))
    return evaluable.einsum('Ai,Aij->Aj', dfunc_dref, dref_dgeom)

class _Jacobian(Array):
  # The jacobian determinant of `geom` to the tip coordinates of the spaces of
  # `geom`. The last axis of `geom` is the coordinate axis.

  def __init__(self, geom: Array, tip_dim: Optional[int] = None) -> None:
    assert geom.ndim >= 1
    if not geom.spaces and geom.shape[-1] != 0:
      raise ValueError('The jacobian of a constant (in space) geometry must have dimension zero.')
    if tip_dim is not None and tip_dim > geom.shape[-1]:
      raise ValueError('Expected a dimension of the tip coordinate system '
                       'not greater than the dimension of the geometry.')
    self._tip_dim = tip_dim
    self._geom = geom
    super().__init__((), float, geom.spaces, geom.arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    geom = self._geom.lower(points_shape, transform_chains, coordinates)
    tip_dim = builtins.sum(transform_chains[space][0].fromdims for space in self._geom.spaces)
    if self._tip_dim is not None and self._tip_dim != tip_dim:
      raise ValueError('Expected a tip dimension of {} but got {}.'.format(self._tip_dim, tip_dim))
    if self._geom.shape[-1] < tip_dim:
      raise ValueError('the dimension of the geometry cannot be lower than the dimension of the tip coords')
    if not self._geom.spaces:
      return evaluable.ones(geom.shape[:-1])
    tips = [_tip_derivative_target(space, chain.fromdims) for space, (chain, opposite) in transform_chains.items() if space in self._geom.spaces]
    J = evaluable.concatenate([evaluable.derivative(geom, tip) for tip in tips], axis=-1)
    return evaluable.sqrt_abs_det_gram(J)

class _Normal(Array):

  def __init__(self, geom: Array) -> None:
    self._geom = geom
    super().__init__(geom.shape, float, geom.spaces, geom.arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    geom = self._geom.lower(points_shape, transform_chains, coordinates)
    spaces_dim = builtins.sum(transform_chains[space][0].todims for space in self._geom.spaces)
    normal_dim = spaces_dim - builtins.sum(transform_chains[space][0].fromdims for space in self._geom.spaces)
    if self._geom.shape[-1] != spaces_dim:
      raise ValueError('The dimension of geometry must equal the sum of the dimensions of the given spaces.')
    if normal_dim == 0:
      raise ValueError('Cannot compute the normal because the dimension of the normal space is zero.')
    elif normal_dim > 1:
      raise ValueError('Cannot unambiguously compute the normal because the dimension of the normal space is larger than one.')
    tangents = []
    normal = None
    for space, (chain, opposite) in transform_chains.items():
      if space not in self._geom.spaces:
        continue
      rgrad = evaluable.derivative(geom, _root_derivative_target(space, chain.todims))
      if chain.todims == chain.fromdims:
        # `chain.basis` is `eye(chain.todims)`
        tangents.append(rgrad)
      else:
        assert normal is None and chain.todims == chain.fromdims + 1
        basis = evaluable.einsum('Aij,jk->Aik', rgrad, chain.basis)
        tangents.append(basis[...,:chain.fromdims])
        normal = basis[...,chain.fromdims:]
    assert normal is not None
    return evaluable.Normal(evaluable.concatenate((*tangents, normal), axis=-1))

class _ExteriorNormal(Array):

  def __init__(self, geom: Array) -> None:
    self._geom = geom
    super().__init__(geom.shape, float, geom.spaces, geom.arguments)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    geom = self._geom.lower(points_shape, transform_chains, coordinates)
    ref_dim = builtins.sum(transform_chains[space][0].fromdims for space in self._geom.spaces)
    if self._geom.shape[-1] != ref_dim + 1:
      raise ValueError('For the exterior normal the dimension of the geometry must be one larger than that of the tip coordinate system, but got {} and {} respectively.'.format(self._geom.shape[-1], ref_dim))
    refs = tuple((_root_derivative_target if chain.todims == chain.fromdims else _tip_derivative_target)(space, chain.fromdims) for space, (chain, opposite) in transform_chains.items() if space in self._geom.spaces)
    rgrad = evaluable.concatenate([evaluable.derivative(geom, ref) for ref in refs], axis=-1)
    if self._geom.shape[-1] == 2:
      normal = evaluable.stack([rgrad[...,1,0], -rgrad[...,0,0]], axis=-1)
    elif self._geom.shape[-1] == 3:
      i = evaluable.asarray([1, 2, 0])
      j = evaluable.asarray([2, 0, 1])
      normal = evaluable.Take(rgrad[...,0], i) * evaluable.Take(rgrad[...,1], j) - evaluable.Take(rgrad[...,1], i) * evaluable.Take(rgrad[...,0], j)
    else:
      raise NotImplementedError
    return normal / evaluable.InsertAxis(evaluable.sqrt(evaluable.Sum(normal**2)), normal.shape[-1])

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

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    return util.sum(evaluable._inflate(array.lower(points_shape, transform_chains, coordinates), evaluable.Range(array.shape[self.axis]) + offset, self.shape[self.axis], self.axis-self.ndim)
      for array, offset in zip(self.arrays, util.cumsum(array.shape[self.axis] for array in self.arrays)))

def _join_arguments(args_list: Iterable[Mapping[str, Argument]]) -> Dict[str, Argument]:
  joined = {} # type: Dict[str, Argument]
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

# CONSTRUCTORS

HANDLED_FUNCTIONS = {}

def implements(np_function):
  'Register an ``__array_function__`` or ``__array_ufunc__`` implementation for Array objects.'
  def decorator(func):
    HANDLED_FUNCTIONS[np_function] = func
    return func
  return decorator

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

# ARITHMETIC

@implements(numpy.add)
def add(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return the sum of the arguments, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.add, __left, __right)

@implements(numpy.subtract)
def subtract(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return the difference of the arguments, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return add(__left, negative(__right))

@implements(numpy.positive)
def positive(__arg: IntoArray) -> Array:
  '''Return the argument unchanged.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return Array.cast(__arg)

@implements(numpy.negative)
def negative(__arg: IntoArray) -> Array:
  '''Return the negation of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return multiply(__arg, -1)

@implements(numpy.multiply)
def multiply(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return the product of the arguments, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.multiply, __left, __right)

@implements(numpy.true_divide)
def divide(__dividend: IntoArray, __divisor: IntoArray) -> Array:
  '''Return the true-division of the arguments, elementwise.

  Parameters
  ----------
  dividend, divisor : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return multiply(__dividend, reciprocal(__divisor))

@implements(numpy.floor_divide)
def floor_divide(__dividend: IntoArray, __divisor: IntoArray) -> Array:
  '''Return the floor-division of the arguments, elementwise.

  Parameters
  ----------
  dividend, divisor : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.FloorDivide, __dividend, __divisor)

@implements(numpy.reciprocal)
def reciprocal(__arg: IntoArray) -> Array:
  '''Return the reciprocal of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return power(__arg, -1.)

@implements(numpy.power)
def power(__base: IntoArray, __exponent: IntoArray) -> Array:
  '''Return the exponentiation of the arguments, elementwise.

  Parameters
  ----------
  base, exponent : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.power, __base, __exponent, min_dtype=int)

@implements(numpy.sqrt)
def sqrt(__arg: IntoArray) -> Array:
  '''Return the square root of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return power(__arg, .5)

@implements(numpy.square)
def square(__arg: IntoArray) -> Array:
  return power(__arg, 2)

@implements(numpy.hypot)
def hypot(__array1: IntoArray, __array2: IntoArray) -> Array:
  return sqrt(square(__array1) + square(__array2))

@implements(numpy.absolute)
def abs(__arg: IntoArray) -> Array:
  '''Return the absolute value of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  return arg * sign(arg)

@implements(numpy.matmul)
def matmul(__arg1: IntoArray, __arg2: IntoArray) -> Array:
  '''Return the matrix product of two arrays.

  Parameters
  ----------
  arg1, arg2 : :class:`Array` or something that can be :meth:`~Array.cast` into one
      Input arrays.

  Returns
  -------
  :class:`Array`

  See Also
  --------

  :any:`numpy.matmul` : the equivalent Numpy function.
  '''

  arg1 = Array.cast(__arg1)
  arg2 = Array.cast(__arg2)
  if not arg1.ndim or not arg2.ndim:
    raise ValueError('cannot contract zero-dimensional array')
  if arg2.ndim == 1:
    return (arg1 * arg2).sum(-1)
  elif arg1.ndim == 1:
    return (arg1[:,numpy.newaxis] * arg2).sum(-2)
  else:
    return (arg1[...,:,:,numpy.newaxis] * arg2[...,numpy.newaxis,:,:]).sum(-2)

@implements(numpy.sign)
def sign(__arg: IntoArray) -> Array:
  '''Return the sign of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Sign, __arg)

@implements(numpy.mod)
def mod(__dividend: IntoArray, __divisor: IntoArray) -> Array:
  '''Return the remainder of the floored division, elementwise.

  Parameters
  ----------
  dividend, divisor : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Mod, __dividend, __divisor)

@implements(numpy.divmod)
def divmod(__dividend: IntoArray, __divisor: IntoArray) -> Tuple[Array, Array]:
  '''Return the floor-division and remainder, elementwise.

  Parameters
  ----------
  dividend, divisor : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`tuple` of :class:`Array` and :class:`Array`
  '''

  return floor_divide(__dividend, __divisor), mod(__dividend, __divisor)

# TRIGONOMETRIC

@implements(numpy.cos)
def cos(__arg: IntoArray) -> Array:
  '''Return the trigonometric cosine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Cos, __arg, min_dtype=float)

@implements(numpy.sin)
def sin(__arg: IntoArray) -> Array:
  '''Return the trigonometric sine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Sin, __arg, min_dtype=float)

@implements(numpy.tan)
def tan(__arg: IntoArray) -> Array:
  '''Return the trigonometric tangent of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Tan, __arg, min_dtype=float)

@implements(numpy.arccos)
def arccos(__arg: IntoArray) -> Array:
  '''Return the trigonometric inverse cosine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.ArcCos, __arg, min_dtype=float)

@implements(numpy.arcsin)
def arcsin(__arg: IntoArray) -> Array:
  '''Return the trigonometric inverse sine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.ArcSin, __arg, min_dtype=float)

@implements(numpy.arctan)
def arctan(__arg: IntoArray) -> Array:
  '''Return the trigonometric inverse tangent of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.ArcTan, __arg, min_dtype=float)

@implements(numpy.arctan2)
def arctan2(__dividend: IntoArray, __divisor: IntoArray) -> Array:
  '''Return the trigonometric inverse tangent of the ``dividend / divisor``, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.ArcTan2, __dividend, __divisor, min_dtype=float)

@implements(numpy.cosh)
def cosh(__arg: IntoArray) -> Array:
  '''Return the hyperbolic cosine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  return .5 * (exp(arg) + exp(-arg))

@implements(numpy.sinh)
def sinh(__arg: IntoArray) -> Array:
  '''Return the hyperbolic sine of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  return .5 * (exp(arg) - exp(-arg))

@implements(numpy.tanh)
def tanh(__arg: IntoArray) -> Array:
  '''Return the hyperbolic tangent of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  return 1 - 2. / (exp(2*arg) + 1)

@implements(numpy.arctanh)
def arctanh(__arg: IntoArray) -> Array:
  '''Return the hyperbolic inverse tangent of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  return .5 * (ln(1+arg) - ln(1-arg))

@implements(numpy.exp)
def exp(__arg: IntoArray) -> Array:
  '''Return the exponential of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Exp, __arg, min_dtype=float)

@implements(numpy.log)
def log(__arg: IntoArray) -> Array:
  '''Return the natural logarithm of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Log, __arg, min_dtype=float)

ln = log

@implements(numpy.log2)
def log2(__arg: IntoArray) -> Array:
  '''Return the base 2 logarithm of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return log(__arg) / log(2)

@implements(numpy.log10)
def log10(__arg: IntoArray) -> Array:
  '''Return the base 10 logarithm of the argument, elementwise.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return log(__arg) / log(10)

# COMPARISON

@implements(numpy.greater)
def greater(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return if the first argument is greater than the second, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Greater, __left, __right, force_dtype=bool)

@implements(numpy.equal)
def equal(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return if the first argument equals the second, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Equal, __left, __right, force_dtype=bool)

@implements(numpy.less)
def less(__left: IntoArray, __right: IntoArray) -> Array:
  '''Return if the first argument is less than the second, elementwise.

  Parameters
  ----------
  left, right : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Less, __left, __right, force_dtype=bool)

@implements(numpy.min)
def min(__a: IntoArray, __b: IntoArray) -> Array:
  '''Return the minimum of the arguments, elementwise.

  Parameters
  ----------
  a, b : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Minimum, __a, __b)

@implements(numpy.max)
def max(__a: IntoArray, __b: IntoArray) -> Array:
  '''Return the maximum of the arguments, elementwise.

  Parameters
  ----------
  a, b : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return _Wrapper.broadcasted_arrays(evaluable.Maximum, __a, __b)

# OPPOSITE

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

  arg = Array.cast(__arg)
  return .5 * (arg + opposite(arg))

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

  arg = Array.cast(__arg)
  return opposite(arg) - arg

# REDUCTION

@implements(numpy.sum)
def sum(__arg: IntoArray, axis: Optional[Union[int, Sequence[int]]] = None) -> Array:
  '''Return the sum of array elements over the given axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`, a sequence of :class:`int`, or ``None``
      The axis or axes to sum. ``None``, the default, implies all axes.

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  if arg.dtype == bool:
    arg = arg.astype(int)
  if axis is None:
    if arg.ndim == 0:
      raise ValueError('Cannot sum last axis of 0-D array.')
    return _Wrapper(evaluable.Sum, arg, shape=arg.shape[:-1], dtype=arg.dtype)
  axes = typing.cast(Sequence[int], (axis,) if isinstance(axis, numbers.Integral) else axis)
  summed = _Transpose.to_end(arg, *axes)
  for i in range(len(axes)):
    summed = _Wrapper(evaluable.Sum, summed, shape=summed.shape[:-1], dtype=summed.dtype)
  return summed

@implements(numpy.product)
def product(__arg: IntoArray, axis: int) -> Array:
  '''Return the product of array elements over the given axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`, a sequence of :class:`int`, or ``None``
      The axis or axes along which the product is performed. ``None``, the
      default, implies all axes.

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  if arg.dtype == bool:
    arg = arg.astype(int)
  transposed = _Transpose.to_end(arg, axis)
  return _Wrapper(evaluable.Product, transposed, shape=transposed.shape[:-1], dtype=arg.dtype)

# LINEAR ALGEBRA

def dot(__a: IntoArray, __b: IntoArray, axes: Optional[Union[int, Sequence[int]]] = None) -> Array:
  '''Return the inner product of the arguments over the given axes, elementwise over the remanining axes.

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

  a = Array.cast(__a)
  b = Array.cast(__b)
  if axes is None:
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    b = _append_axes(b, a.shape[1:])
    axes = 0,
  return sum(multiply(a, b), axes)

@implements(numpy.trace)
def trace(__arg: IntoArray, axis1: int = -2, axis2: int = -1) -> Array:
  '''Return the trace, the sum of the diagonal, of an array over the two given axes, elementwise over the remanining axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis1 : :class:`int`
      The first axis. Defaults to the next to last axis.
  axis2 : :class:`int`
      The second axis. Defaults to the last axis.

  Returns
  -------
  :class:`Array`
  '''

  return sum(_takediag(__arg, axis1, axis2), -1)

def norm2(__arg: IntoArray, axis: Union[int, Sequence[int]] = -1) -> Array:
  '''Return the 2-norm of the argument over the given axis, elementwise over the remanining axes.

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
  return sqrt(sum(multiply(arg, arg), axis))

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

  See Also
  --------
  :func:`norm2` : The 2-norm.
  '''

  arg = Array.cast(__arg)
  return divide(arg, insertaxis(norm2(arg, axis), axis, 1))

def matmat(__arg0: IntoArray, *args: IntoArray) -> Array:
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = Array.cast(__arg0)
  for arg in map(Array.cast, args):
    if retval.shape[-1] != arg.shape[0]:
      raise ValueError('incompatible shapes')
    retval = dot(retval[(...,)+(numpy.newaxis,)*(arg.ndim-1)], arg[(numpy.newaxis,)*(retval.ndim-1)], retval.ndim-1)
  return retval

def inverse(__arg: IntoArray, __axes: Tuple[int, int] = (-2,-1)) -> Array:
  '''Return the inverse of the argument along the given axes, elementwise over the remaining axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axes : :class:`tuple` of two :class:`int`
      The two axes along which the inverse is computed. Defaults to the last
      two axes.

  Returns
  -------
  :class:`Array`
  '''

  transposed = _Transpose.to_end(Array.cast(__arg), *__axes)
  if transposed.shape[-2] != transposed.shape[-1]:
    raise ValueError('cannot compute the inverse along two axes with different lengths')
  inverted = _Wrapper(evaluable.Inverse, transposed, shape=transposed.shape, dtype=float)
  return _Transpose.from_end(inverted, *__axes)

def determinant(__arg: IntoArray, __axes: Tuple[int, int] = (-2,-1)) -> Array:
  '''Return the determinant of the argument along the given axes, elementwise over the remaining axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axes : :class:`tuple` of two :class:`int`
      The two axes along which the determinant is computed. Defaults to the last
      two axes.

  Returns
  -------
  :class:`Array`
  '''

  transposed = _Transpose.to_end(Array.cast(__arg), *__axes)
  return _Wrapper(evaluable.Determinant, transposed, shape=transposed.shape[:-2], dtype=float)

def _eval_eigval(arg: evaluable.Array, symmetric: bool) -> evaluable.Array:
  val, vec = evaluable.Eig(arg, symmetric)
  return val

def _eval_eigvec(arg: evaluable.Array, symmetric: bool) -> evaluable.Array:
  val, vec = evaluable.Eig(arg, symmetric)
  return vec

def eig(__arg: IntoArray, __axes: Tuple[int, int] = (-2,-1), symmetric: bool = False) -> Tuple[Array, Array]:
  '''Return the eigenvalues and right eigenvectors of the argument along the given axes, elementwise over the remaining axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axes : :class:`tuple` of two :class:`int`
      The two axes along which the determinant is computed. Defaults to the last
      two axes.
  symmetric : :class:`bool`
      Indicates if the argument is Hermitian.

  Returns
  -------
  eigval : :class:`Array`
      The diagonalized eigenvalues.
  eigvec : :class:`Array`
      The right eigenvectors.
  '''

  arg = Array.cast(__arg)
  transposed = _Transpose.to_end(arg, *__axes)
  eigval = _Wrapper(functools.partial(_eval_eigval, symmetric=symmetric), arg, shape=arg.shape[:-1], dtype=float if symmetric else complex)
  eigvec = _Wrapper(functools.partial(_eval_eigvec, symmetric=symmetric), arg, shape=arg.shape, dtype=float if symmetric and arg.dtype != complex else complex)
  return diagonalize(eigval), eigvec

def _takediag(__arg: IntoArray, _axis1: int = -2, _axis2: int =-1) -> Array:
  arg = Array.cast(__arg)
  transposed = _Transpose.to_end(arg, _axis1, _axis2)
  if transposed.shape[-2] != transposed.shape[-1]:
    raise ValueError('cannot take the diagonal along two axes with different lengths')
  return _Wrapper(evaluable.TakeDiag, transposed, shape=transposed.shape[:-1], dtype=transposed.dtype)

def takediag(__arg: IntoArray, __axis: int = -2, __rmaxis: int = -1) -> Array:
  '''Return the diagonal of the argument along the given axes.

  Parameters
  ----------
  arg : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`
      The axis to keep. Defaults to the next to last axis.
  rmaxis : :class:`int`
      The axis to remove. Defaults to the last axis.

  Returns
  -------
  :class:`Array`

  See Also
  --------
  :func:`diagonalize` : The complement operation.
  '''
  arg = Array.cast(__arg)
  axis = numeric.normdim(arg.ndim, __axis)
  rmaxis = numeric.normdim(arg.ndim, __rmaxis)
  assert axis < rmaxis
  return _Transpose.from_end(_takediag(arg, axis, rmaxis), axis)

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

  See Also
  --------
  :func:`takediag` : The complement operation.
  '''

  arg = Array.cast(__arg)
  axis = numeric.normdim(arg.ndim, __axis)
  newaxis = numeric.normdim(arg.ndim+1, __newaxis)
  assert axis < newaxis
  transposed = _Transpose.to_end(arg, axis)
  diagonalized = _Wrapper(evaluable.Diagonalize, transposed, shape=(*transposed.shape, transposed.shape[-1]), dtype=transposed.dtype)
  return _Transpose.from_end(diagonalized, axis, newaxis)

def cross(__arg1: IntoArray, __arg2: IntoArray, axis: int = -1) -> Array:
  '''Return the cross product of the arguments over the given axis, elementwise over the remaining axes.

  Parameters
  ----------
  arg1, arg2 : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`
      The axis along which the cross product is computed. Defaults to the last axis.

  Returns
  -------
  :class:`Array`

  See Also
  --------
  :func:`takediag` : The inverse operation.
  '''

  arg1, arg2 = broadcast_arrays(*typecast_arrays(__arg1, __arg2, min_dtype=int))
  axis = numeric.normdim(arg1.ndim, axis)
  assert arg1.shape[axis] == 3
  i = Array.cast(types.frozenarray([1, 2, 0]))
  j = Array.cast(types.frozenarray([2, 0, 1]))
  return take(arg1, i, axis) * take(arg2, j, axis) - take(arg2, i, axis) * take(arg1, j, axis)

def outer(arg1, arg2=None, axis=0):
  'outer product'

  if arg2 is None:
    arg2 = arg1
  elif arg1.ndim != arg2.ndim:
    raise ValueError('arg1 and arg2 have different dimensions')
  axis = numeric.normdim(arg1.ndim, axis)
  return expand_dims(arg1,axis+1) * expand_dims(arg2,axis)

# ARRAY OPS

@implements(numpy.transpose)
def transpose(__array: IntoArray, __axes: Optional[Sequence[int]] = None) -> Array:
  '''Permute the axes of an array.

  Parameters
  ----------
  array : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axes : sequence of :class:`int`
      Permutation of ``range(array.ndim)``. Defaults to reversing the order of
      the axes, ``reversed(range(array.ndim))``.

  Returns
  -------
  :class:`Array`
      The transposed array. Axis ``i`` of the resulting array corresponds to
      axis ``axes[i]`` of the argument.
  '''

  array = Array.cast(__array)
  axes = tuple(reversed(range(array.ndim)) if __axes is None else __axes)
  return _Transpose(array, axes)

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

@implements(numpy.repeat)
def repeat(__array: IntoArray, __n: IntoArray, axis: int) -> Array:
  '''Repeat the given axis of an array `n` times.

  Parameters
  ----------
  array : :class:`Array` or something that can be :meth:`~Array.cast` into one
  n : :class:`int` or :class:`Array`
      The number of repetitions.
  axis : class:`int`
      The position of the axis to be repeated.

  Returns
  -------
  :class:`Array`
  '''

  array = Array.cast(__array)
  if array.shape[axis] != 1:
    raise NotImplementedError('only axes with length 1 can be repeated')
  return insertaxis(get(array, axis, 0), axis, __n)

@implements(numpy.swapaxes)
def swapaxes(__array: IntoArray, __axis1: int, __axis2: int) -> Array:
  '''Swap two axes of an array.

  Parameters
  ----------
  array : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis1, axis2 : :class:`int`
      The axes to be swapped.

  Returns
  -------
  :class:`Array`
  '''

  array = Array.cast(__array)
  axis1 = __axis1
  axis2 = __axis2
  trans = list(range(array.ndim))
  trans[axis1], trans[axis2] = trans[axis2], trans[axis1]
  return transpose(array, trans)

def ravel(__array: IntoArray, axis: int) -> Array:
  '''Ravel two consecutive axes of an array.

  Parameters
  ----------
  array : :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`
      The first of the two consecutive axes to ravel.

  Returns
  -------
  :class:`Array`

  See Also
  --------
  :func:`unravel` : The reverse operation.
  '''

  array = Array.cast(__array)
  axis = numeric.normdim(array.ndim-1, axis)
  transposed = _Transpose.to_end(array, axis, axis+1)
  raveled = _Wrapper(evaluable.Ravel, transposed, shape=(*transposed.shape[:-2], transposed.shape[-2]*transposed.shape[-1]), dtype=transposed.dtype)
  return _Transpose.from_end(raveled, axis)

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

  See Also
  --------
  :func:`ravel` : The reverse operation.
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

@implements(numpy.take)
def take(__array: IntoArray, __indices: IntoArray, axis: int) -> Array:
  '''Take elements from an array along an axis.

  Parameters
  ----------
  array : :class:`Array` or something that can be :meth:`~Array.cast` into one
  indices : :class:`Array` with dtype :class:`int` or :class:`bool` or something that can be :meth:`~Array.cast` into one
      The indices of elements to take. The array of indices may have any dimension, including zero.
      However, if the array is boolean, the array must 1-D.
  axis : :class:`int`
      The axis to take elements from or, if ``indices`` has more than one
      dimension, the first axis of a range of ``indices.ndim`` axes to take
      elements from.

  Returns
  -------
  :class:`Array`
      The array with the taken elements. The original ``axis`` is replaced by
      ``indices.ndim`` axes.

  See Also
  --------
  :func:`get` : Special case of :func:`take` with scalar index.
  '''

  array = Array.cast(__array)
  axis = numeric.normdim(array.ndim, axis)
  if isinstance(__indices, numpy.ndarray) and __indices.ndim == 1 and __indices.dtype == bool \
      or isinstance(__indices, (list, tuple)) and all(isinstance(index, bool) for index in __indices):
    if len(__indices) != array.shape[axis]:
      raise ValueError('The length of the mask differs from the length of the given axis.')
    indices = Array.cast(numpy.nonzero(__indices)[0])
  else:
    indices = _Wrapper.broadcasted_arrays(evaluable.NormDim, array.shape[axis], Array.cast(__indices, dtype=int))
  transposed = _Transpose.to_end(array, axis)
  taken = _Wrapper(evaluable.Take, transposed, _WithoutPoints(indices), shape=(*transposed.shape[:-1], *indices.shape), dtype=array.dtype)
  return _Transpose.from_end(taken, *range(axis, axis+indices.ndim))

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
  :func:`take` : Take elements from an array along an axis.
  :func:`kronecker` : The complement operation.
  '''

  array = Array.cast(__array)
  axis = __axis
  index = Array.cast(__index, dtype=int, ndim=0)
  return take(array, index, axis)

def _range(__length: int, __offset: int) -> Array:
  length = Array.cast(__length, dtype=int, ndim=0)
  offset = Array.cast(__offset, dtype=int, ndim=0)
  return _Wrapper(lambda l, o: evaluable.Range(l) + o, _WithoutPoints(length), _WithoutPoints(offset), shape=(__length,), dtype=int)

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
    index = _range(stop-start, start)
  elif isinstance(n, numbers.Integral):
    index = Array.cast(numpy.arange(*s.indices(int(n))))
  else:
    raise Exception('a non-unit slice requires a constant-length axis')
  return take(array, index, axis)

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

  See Also
  --------
  :func:`take` : The complement operation.
  '''

  array = Array.cast(__array)
  indices = Array.cast(indices)
  return _Wrapper(evaluable.Inflate,
    array,
    _WithoutPoints(indices),
    _WithoutPoints(Array.cast(length)),
    shape=array.shape[:array.ndim-indices.ndim] + (length,),
    dtype=array.dtype)

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

@implements(numpy.concatenate)
def concatenate(__arrays: Sequence[IntoArray], axis: int = 0) -> Array:
  '''Join arrays along an existing axis.

  Parameters
  ----------
  arrays : sequence of :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`
      The existing axis along which the arrays are joined.

  Returns
  -------
  :class:`Array`

  See Also
  --------
  :func:`stack` : Join arrays along an new axis.
  '''

  return _Concatenate(__arrays, axis)

@implements(numpy.stack)
def stack(__arrays: Sequence[IntoArray], axis: int = 0) -> Array:
  '''Join arrays along a new axis.

  Parameters
  ----------
  arrays : sequence of :class:`Array` or something that can be :meth:`~Array.cast` into one
  axis : :class:`int`
      The axis in the resulting array along which the arrays are joined.

  Returns
  -------
  :class:`Array`

  See Also
  --------
  :func:`stack` : Join arrays along an new axis.
  '''

  aligned = broadcast_arrays(*typecast_arrays(*__arrays))
  return util.sum(kronecker(array, axis, len(aligned), i) for i, array in enumerate(aligned))

def replace_arguments(__array: IntoArray, __arguments: Mapping[str, IntoArray]) -> Array:
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

  return _Replace(Array.cast(__array), {k: Array.cast(v) for k, v in __arguments.items()})

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
  return tuple(broadcast_to(arg, shape) for arg in arrays_)

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

@implements(numpy.broadcast_to)
def broadcast_to(array: IntoArray, shape: Shape) -> Array:
  '''Broadcast an array to a new shape.

  Parameters
  ----------
  array : :class:`Array` or similar
      The array to broadcast.
  shape : :class:`tuple` of :class:`int`
      The desired shape.

  Returns
  -------
  :class:`Array`
      The broadcasted array.
  '''

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
      broadcasted = repeat(broadcasted, desired, axis + nnew)
    else:
      raise ValueError('cannot broadcast array with shape {} to {} because input axis {} is neither singleton nor has the desired length'.format(orig_shape, shape, axis))
  return broadcasted

# DERIVATIVES

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

def grad(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the gradient of the argument to the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`
      The dimension of the local coordinate system.

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom)
  if geom.ndim == 0:
    return grad(arg, _append_axes(geom, (1,)))[...,0]
  elif geom.ndim > 1:
    sh = geom.shape[-2:]
    return unravel(grad(arg, ravel(geom, geom.ndim-2)), arg.ndim+geom.ndim-2, sh)
  elif ndims == 0 or ndims == geom.shape[0]:
    return _Gradient(arg, geom)
  elif ndims == -1 or ndims == geom.shape[0] - 1:
    return _SurfaceGradient(arg, geom)
  else:
    raise NotImplementedError

def curl(__arg: IntoArray, __geom: IntoArray) -> Array:
  '''Return the curl of the argument w.r.t. the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom)
  if geom.shape != (3,):
    raise ValueError('Expected a geometry with shape (3,) but got {}.'.format(geom.shape))
  if not arg.ndim:
    raise ValueError('Expected a function with at least 1 axis but got 0.')
  if arg.shape[-1] != 3:
    raise ValueError('Expected a function with a trailing axis of length 3 but got {}.'.format(arg.shape[-1]))
  return (levicivita(3).T * _append_axes(grad(arg, geom), (3,))).sum((-3, -2))

def normal(__geom: IntoArray, exterior: bool = False) -> Array:
  '''Return the normal of the geometry.

  Parameters
  ----------
  geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  exterior : :class:`bool`

  Returns
  -------
  :class:`Array`
  '''

  geom = Array.cast(__geom)
  if geom.ndim == 0:
    return normal(insertaxis(geom, 0, 1), exterior)[...,0]
  elif geom.ndim > 1:
    sh = geom.shape[-2:]
    return unravel(normal(ravel(geom, geom.ndim-2), exterior), geom.ndim-2, sh)
  elif not exterior:
    return _Normal(geom)
  else:
    return _ExteriorNormal(geom)

def dotnorm(__arg: IntoArray, __geom: IntoArray, axis: int = -1) -> Array:
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

  Returns
  -------
  :class:`Array`
  '''

  arg = _Transpose.to_end(Array.cast(__arg), axis)
  geom = Array.cast(__geom, ndim=1)
  assert geom.shape[0] == arg.shape[-1]
  return dot(arg, _prepend_axes(normal(geom), arg.shape[:-1]), -1)

def tangent(__geom: IntoArray, __vec: IntoArray) -> Array:
  '''Return the tangent.

  Parameters
  ----------
  geom, vec : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  geom = Array.cast(__geom)
  vec = Array.cast(__vec)
  return subtract(vec, multiply(dot(vec, normal(geom), -1)[...,None], normal(geom)))

def jacobian(__geom: IntoArray, __ndims: Optional[int] = None) -> Array:
  '''Return the absolute value of the determinant of the Jacobian matrix of the given geometry.

  Parameters
  ----------
  geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
      The geometry.

  Returns
  -------
  :class:`Array`
  '''

  geom = Array.cast(__geom)
  if geom.ndim == 0:
    return jacobian(insertaxis(geom, 0, 1), __ndims)
  elif geom.ndim > 1:
    return jacobian(ravel(geom, geom.ndim-2), __ndims)
  else:
    return _Jacobian(geom, __ndims)

def J(__geom: IntoArray, __ndims: Optional[int] = None) -> Array:
  '''Return the absolute value of the determinant of the Jacobian matrix of the given geometry.

  Alias of :func:`jacobian`.
  '''

  return jacobian(__geom, __ndims)

def _d1(arg: IntoArray, var: IntoArray) -> Array:
  return derivative(arg, var) if isinstance(var, Argument) else grad(arg, var)

def d(__arg: IntoArray, *vars: IntoArray) -> Array:
  return functools.reduce(_d1, vars, Array.cast(__arg))

def surfgrad(__arg: IntoArray, geom: IntoArray) -> Array:
  '''Return the surface gradient of the argument to the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one

  Returns
  -------
  :class:`Array`
  '''

  return grad(__arg, geom, -1)

def curvature(__geom: IntoArray, ndims: int = -1) -> Array:
  '''Return the curvature of the given geometry.

  Parameters
  ----------
  geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  geom = Array.cast(__geom)
  return geom.normal().div(geom, ndims=ndims)

def div(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the divergence of ``arg`` w.r.t. the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  geom = Array.cast(__geom, ndim=1)
  return trace(grad(__arg, geom, ndims))

def laplace(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the Laplacian of ``arg`` w.r.t. the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom, ndim=1)
  return arg.grad(geom, ndims).div(geom, ndims)

def symgrad(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the symmetric gradient of ``arg`` w.r.t. the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom)
  return multiply(.5, add_T(arg.grad(geom, ndims)))

def ngrad(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the inner product of the gradient of ``arg`` with the normal of the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom)
  return dotnorm(grad(arg, geom, ndims), geom)

def nsymgrad(__arg: IntoArray, __geom: IntoArray, ndims: int = 0) -> Array:
  '''Return the inner product of the symmetric gradient of ``arg`` with the normal of the given geometry.

  Parameters
  ----------
  arg, geom : :class:`Array` or something that can be :meth:`~Array.cast` into one
  ndims : :class:`int`

  Returns
  -------
  :class:`Array`
  '''

  arg = Array.cast(__arg)
  geom = Array.cast(__geom)
  return dotnorm(symgrad(arg, geom, ndims), geom)

# MISC

@util.single_or_multiple
def eval(funcs: evaluable.AsEvaluableArray, **arguments: Mapping[str, numpy.ndarray]) -> Tuple[numpy.ndarray, ...]:
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

  return map(sparse.toarray, evaluable.eval_sparse(funcs, **arguments))

def isarray(__arg: Any) -> bool:
  'Test if the argument is an instance of :class:`Array`.'
  return isinstance(__arg, Array)

def rootcoords(space: str, __dim: int) -> Array:
  'Return the root coordinates.'
  return _RootCoords(space, __dim)

def transforms_index(space: str, transforms: Transforms) -> Array:
  return _TransformsIndex(space, transforms)

def transforms_coords(space: str, transforms: Transforms) -> Array:
  return _TransformsCoords(space, transforms)

def Elemwise(__data: Sequence[numpy.ndarray], __index: IntoArray, dtype: DType) -> Array:
  'elemwise'

  warnings.deprecation('function.Elemwise is deprecated; use function.get instead')
  return get(numpy.asarray(__data), 0, __index)

def piecewise(level: IntoArray, intervals: Sequence[IntoArray], *funcs: IntoArray) -> Array:
  'piecewise'
  level = Array.cast(level)
  return util.sum(greater(level, interval).astype(int) for interval in intervals).choose(funcs)

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
  signs = [sign(f - level) for level in levels]
  steps = map(subtract, signs[:-1], signs[1:])
  return [.5 - .5 * signs[0]] + [.5 * step for step in steps] + [.5 + .5 * signs[-1]]

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
  :func:`sign`: like :func:`heaviside` but with different levels
  '''

  return sign(f) * .5 + .5

def _eval_choose(_index: evaluable.Array, *_choices: evaluable.Array) -> evaluable.Array:
  return evaluable.Choose(_index, _choices)

def choose(__index: IntoArray, __choices: Sequence[IntoArray]) -> Array:
  'Function equivalent of :func:`numpy.choose`.'
  index = Array.cast(__index)
  if index.ndim != 0:
    raise ValueError
  choices = broadcast_arrays(*typecast_arrays(*__choices))
  shape = choices[0].shape
  dtype = choices[0].dtype
  index = _append_axes(index, shape)
  spaces = functools.reduce(operator.or_, (arg.spaces for arg in choices), index.spaces)
  return _Wrapper(_eval_choose, index, *choices, shape=shape, dtype=dtype)

def chain(_funcs: Sequence[IntoArray]) -> Sequence[Array]:
  'chain'

  funcs = tuple(map(Array.cast, _funcs))
  shapes = [func.shape[0] for func in funcs]
  return [concatenate([func if i==j else zeros((sh,) + func.shape[1:])
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

  return concatenate([kronecker(arg, axis=-1, length=len(args), pos=iarg) for iarg, arg in enumerate(args)])

def simplified(__arg: IntoArray) -> Array:
  warnings.deprecation('`nutils.function.simplified` is deprecated. This function returns the argument unmodified and can safely be omitted.')
  return Array.cast(__arg)

def iszero(__arg: IntoArray) -> bool:
  warnings.deprecation('`nutils.function.iszero` is deprecated. Use `evaluable.iszero` on the lowered function instead.')
  return False

def add_T(__arg: IntoArray, axes: Tuple[int, int] = (-2,-1)) -> Array:
  'add transposed'
  arg = Array.cast(__arg)
  return swapaxes(arg, *axes) + arg

def trignormal(_angle: IntoArray) -> Array:
  angle = Array.cast(_angle)
  return stack([cos(angle), sin(angle)], axis=-1)

def trigtangent(_angle: IntoArray) -> Array:
  angle = Array.cast(_angle)
  return stack([-sin(angle), cos(angle)], axis=-1)

def rotmat(__arg: IntoArray) -> Array:
  arg = Array.cast(__arg)
  return stack([trignormal(arg), trigtangent(arg)], 0)

def dotarg(__argname: str, *arrays: IntoArray, shape: Tuple[int, ...] = ()) -> Array:
  '''Return the inner product of the first axes of the given arrays with an argument with the given name.

  An argument with shape ``(arrays[0].shape[0], ..., arrays[-1].shape[0]) +
  shape`` will be created. Repeatedly the inner product of the result, starting
  with the argument, with every array from ``arrays`` is taken, where all but
  the first axis are treated as an outer product.

  Parameters
  ----------
  argname : :class:`str`
      The name of the argument.
  *arrays : :class:`Array` or something that can be :meth:`~Array.cast` into one
      The arrays to take inner products with.
  shape : :class:`tuple` of :class:`int`, optional
      The shape to be appended to the argument.

  Returns
  -------
  :class:`Array`
      The inner product with shape ``shape + arrays[0].shape[1:] + ... + arrays[-1].shape[1:]``.
  '''

  result = Argument(__argname, tuple(array.shape[0] for array in arrays) + tuple(shape), dtype=float)
  for array in arrays:
    result = numpy.sum(_append_axes(result.transpose((*range(1, result.ndim), 0)), array.shape[1:]) * array, result.ndim-1)
  return result

# BASES

def _int_or_vec(f, self, arg, argname, nargs, nvals):
  if isinstance(arg, numbers.Integral):
    return f(self, int(numeric.normdim(nargs, arg)))
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
    mask = numpy.zeros(nvals, dtype=bool)
    for d in arg:
      mask[numpy.asarray(f(self, d))] = True
    return mask.nonzero()[0]
  raise IndexError('invalid {}'.format(argname))

def _int_or_vec_dof(f):
  @functools.wraps(f)
  def wrapped(self, dof: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
    return _int_or_vec(f, self, arg=dof, argname='dof', nargs=self.ndofs, nvals=self.nelems)
  return wrapped

def _int_or_vec_ielem(f):
  @functools.wraps(f)
  def wrapped(self, ielem: Union[numbers.Integral, numpy.ndarray]) -> numpy.ndarray:
    return _int_or_vec(f, self, arg=ielem, argname='ielem', nargs=self.nelems, nvals=self.ndofs)
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

    _index = evaluable.Argument('_index', shape=(), dtype=int)
    self._arg_dofs, self._arg_coeffs = [f.optimized_for_numpy for f in self.f_dofs_coeffs(_index)]
    assert self._arg_dofs.ndim == 1
    assert self._arg_coeffs.ndim == 1 + coords.shape[0]
    assert evaluable.equalindex(self._arg_dofs.shape[0], self._arg_coeffs.shape[0])
    self._arg_ndofs = evaluable.asarray(self._arg_dofs.shape[0])

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    index = _WithoutPoints(self.index).lower(points_shape, transform_chains, coordinates)
    dofs, coeffs = self.f_dofs_coeffs(index)
    coords = self.coords.lower(points_shape, transform_chains, coordinates)
    return evaluable.Inflate(evaluable.Polyval(coeffs, coords), dofs, self.ndofs)

  @util.cached_property
  def _computed_support(self) -> Tuple[numpy.ndarray, ...]:
    support = [[] for i in range(self.ndofs)] # type: List[List[int]]
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

    return self._arg_dofs.eval(_index=ielem)

  def get_ndofs(self, ielem: int) -> int:
    '''Return the number of basis functions with support on element ``ielem``.'''

    return int(self._arg_ndofs.eval(_index=numeric.normdim(self.nelems, ielem)))

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

    return self._arg_coeffs.eval(_index=numeric.normdim(self.nelems, ielem))

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
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
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    assert all(c.shape[0] == d.shape[0] for c, d in zip(self._coeffs, self._dofs))
    super().__init__(ndofs, len(coefficients), index, coords)

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
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
    assert all(c.ndim == 1+coords.shape[0] for c in self._coeffs)
    self._offsets = numpy.cumsum([0] + [c.shape[0] for c in self._coeffs])
    super().__init__(self._offsets[-1], len(coefficients), index, coords)

  @_int_or_vec_dof
  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    ielem = numpy.searchsorted(self._offsets[:-1], numeric.normdim(self.ndofs, dof), side='right')-1
    return numpy.array([ielem], dtype=int)

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
    coeffs = evaluable.Elemwise(self._coeffs, index, dtype=float)
    dofs = evaluable.Range(coeffs.shape[0]) + evaluable.get(self._offsets, 0, index)
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

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
    dofs = evaluable.Range(self._degree+1) + index * (self._degree+1)
    coeffs = numpy.zeros((self._degree+1,)*2, dtype=int)
    for n in range(self._degree+1):
      for k in range(n+1):
        coeffs[n,k] = (-1 if (n+k) % 2 else 1) * numeric.binom(n, k) * numeric.binom(n+k, k)
    return dofs, evaluable.astype(evaluable.asarray(coeffs), float)

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    index = _WithoutPoints(self.index).lower(points_shape, transform_chains, coordinates)
    coords = self.coords.lower(points_shape, transform_chains, coordinates)
    assert evaluable.equalindex(coords.shape[-1], 1)
    leg = evaluable.Legendre(evaluable.get(coords, coords.ndim-1, 0) * 2 - 1, self._degree)
    dofs = evaluable.Range(self._degree+1) + index * (self._degree+1)
    return evaluable.Inflate(leg, dofs, self.ndofs)

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
    self._renumber = types.frozenarray(numeric.invmap(indices, length=parent.ndofs, missing=len(indices)), copy=False)
    super().__init__(len(indices), parent.nelems, parent.index, parent.coords)

  def get_support(self, dof: Union[int, numpy.ndarray]) -> numpy.ndarray:
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return self._parent.get_support(self._indices[dof])

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
    p_dofs, p_coeffs = self._parent.f_dofs_coeffs(index)
    renumber = evaluable.Take(self._renumber, p_dofs)
    selection = evaluable.Find(evaluable.Less(renumber, evaluable.InsertAxis(self.ndofs, p_dofs.shape[0])))
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

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
    indices = []
    for n in reversed(self._transforms_shape[1:]):
      index, ielem = evaluable.divmod(index, n)
      indices.append(ielem)
    indices.append(index)
    indices.reverse()
    ranges = [evaluable.Range(evaluable.get(lengths_i, 0, index_i)) + evaluable.get(offsets_i, 0, index_i)
      for lengths_i, offsets_i, index_i in zip(self._ndofs, self._start_dofs, indices)]
    ndofs = self._dofs_shape[0]
    dofs = ranges[0] % ndofs
    for range_i, ndofs_i in zip(ranges[1:], self._dofs_shape[1:]):
      dofs = evaluable.Ravel(evaluable.RavelIndex(dofs, range_i % ndofs_i, ndofs, ndofs_i))
      ndofs = ndofs * ndofs_i
    coeffs = functools.reduce(evaluable.PolyOuterProduct,
      [evaluable.Elemwise(coeffs_i, index_i, float) for coeffs_i, index_i in zip(self._coeffs, indices)])
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

  def f_dofs_coeffs(self, index: evaluable.Array) -> Tuple[evaluable.Array,evaluable.Array]:
    p_dofs, p_coeffs = self._parent.f_dofs_coeffs(evaluable.get(self._transmap, 0, index))
    dofs = evaluable.take(self._renumber, p_dofs, axis=0)
    return dofs, p_coeffs

def Namespace(*args, **kwargs):
  from .expression_v1 import Namespace
  return Namespace(*args, **kwargs)
