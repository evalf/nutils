# Copyright (c) 2014 Evalf
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

'''
The sample module defines the :class:`Sample` class, which represents a
collection of discrete points on a topology and is typically formed via
:func:`nutils.topology.Topology.sample`. Any function evaluation starts from
this sampling step, which drops element information and other topological
properties such as boundaries and groups, but retains point positions and
(optionally) integration weights. Evaluation is performed by subsequent calls
to :func:`Sample.integrate`, :func:`Sample.integral` or :func:`Sample.eval`.

Besides the location of points, :class:`Sample` also keeps track of point
connectivity through its :attr:`Sample.tri` and :attr:`Sample.hull`
properties, representing a (n-dimensional) triangulation of the interior and
boundary, respectively. Availability of these properties depends on the
selected sample points, and is typically used in combination with the "bezier"
set.
'''

from . import types, points, util, function, evaluable, parallel, numeric, matrix, sparse
from .pointsseq import PointsSequence
from .transformseq import Transforms
from .transform import EvaluableTransformChain
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union
import numpy, numbers, collections.abc, os, treelog as log, abc

_PointsShape = Tuple[evaluable.Array, ...]
_TransformChainsMap = Mapping[str, Tuple[EvaluableTransformChain, EvaluableTransformChain]]
_CoordinatesMap = Mapping[str, evaluable.Array]

graphviz = os.environ.get('NUTILS_GRAPHVIZ')

def argdict(arguments) -> Mapping[str, numpy.ndarray]:
  if len(arguments) == 1 and 'arguments' in arguments and isinstance(arguments['arguments'], collections.abc.Mapping):
    return arguments['arguments']
  return arguments

class Sample(types.Singleton):
  '''Collection of points on a topology.

  The :class:`Sample` class represents a collection of discrete points on a
  topology and is typically formed via :func:`nutils.topology.Topology.sample`.
  Any function evaluation starts from this sampling step, which drops element
  information and other topological properties such as boundaries and groups,
  but retains point positions and (optionally) integration weights. Evaluation
  is performed by subsequent calls to :func:`integrate`, :func:`integral` or
  :func:`eval`.

  Besides the location of points, ``Sample`` also keeps track of point
  connectivity through its :attr:`tri` and :attr:`hull` properties,
  representing a (n-dimensional) triangulation of the interior and boundary,
  respectively. Availability of these properties depends on the selected sample
  points, and is typically used in combination with the "bezier" set.
  '''

  __slots__ = 'spaces', 'ndims', 'nelems', 'npoints'

  @staticmethod
  def new(space: str, transforms: Iterable[Transforms], points: PointsSequence, index: Optional[Union[numpy.ndarray, Sequence[numpy.ndarray]]] = None) -> 'Sample':
    '''Create a new :class:`Sample`.

    Parameters
    ----------
    transforms : :class:`tuple` or transformation chains
        List of transformation chains leading to local coordinate systems that
        contain points.
    points : :class:`~nutils.pointsseq.PointsSequence`
        Points sequence.
    index : integer array or :class:`tuple` of integer arrays, optional
        Indices defining the order in which points show up in the evaluation.
        If absent the indices will be strictly increasing.
    '''

    sample = _DefaultIndex(space, tuple(transforms), points)
    if index is not None:
      if isinstance(index, (tuple, list)):
        assert all(ind.shape == (pnt.npoints,) for ind, pnt in zip(index, points))
        index = numpy.concatenate(index)
      sample = _CustomIndex(sample, types.arraydata(index))
    return sample

  def __init__(self, spaces: Tuple[str, ...], ndims: int, nelems: int, npoints: int) -> None:
    '''
    parameters
    ----------
    spaces : :class:`tuple` of :class:`str`
        The names of the spaces on which this sample is defined.
    ndims : :class:`int`
        The dimension of the coordinates.
    nelems : :class:`int`
        The number of elements.
    npoints : :class:`int`
        The number of points.
    '''

    self.spaces = spaces
    self.ndims = ndims
    self.nelems = nelems
    self.npoints = npoints

  def __repr__(self) -> str:
    return '{}.{}<{}D, {} elems, {} points>'.format(type(self).__module__, type(self).__qualname__, self.ndims, self.nelems, self.npoints)

  @property
  def index(self) -> Tuple[numpy.ndarray, ...]:
    return tuple(map(self.getindex, range(self.nelems)))

  @abc.abstractmethod
  def getindex(self, __ielem: int) -> numpy.ndarray:
    '''Return the indices of `Sample.points[ielem]` in results of `Sample.eval`.'''

    raise NotImplementedError

  @abc.abstractmethod
  def get_evaluable_indices(self, __ielem: evaluable.Array) -> evaluable.Array:
    '''Return the evaluable indices for the given evaluable element index.

    Parameters
    ----------
    ielem : :class:`nutils.evaluable.Array`, ndim: 0, dtype: :class:`int`
        The element index.

    Returns
    -------
    indices : :class:`nutils.evaluable.Array`
        The indices of the points belonging to the given element as a 1D array.

    See Also
    --------
    :meth:`getindex` : the non-evaluable equivalent
    '''

    raise NotImplementedError

  def get_evaluable_weights(self, __ielem: evaluable.Array) -> evaluable.Array:
    raise NotImplementedError

  def update_lower_args(self, __ielem: evaluable.Array, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> Tuple[_PointsShape, _TransformChainsMap, _CoordinatesMap]:
    raise NotImplementedError

  @util.positional_only
  @util.single_or_multiple
  def integrate(self, funcs: Iterable[function.IntoArray], arguments: Mapping[str, numpy.ndarray] = ...) -> Tuple[numpy.ndarray, ...]:
    '''Integrate functions.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    datas = self.integrate_sparse(funcs, argdict(arguments))
    with log.iter.fraction('assembling', datas) as items:
      return tuple(_convert(data, inplace=True) for data in items)

  @util.single_or_multiple
  def integrate_sparse(self, funcs: Iterable[function.IntoArray], arguments: Optional[Mapping[str, numpy.ndarray]] = None) -> Tuple[numpy.ndarray, ...]:
    '''Integrate functions into sparse data.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    return eval_integrals_sparse(*map(self.integral, funcs), **(arguments or {}))

  def integral(self, __func: function.IntoArray) -> function.Array:
    '''Create Integral object for postponed integration.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    '''

    return _Integral(function.Array.cast(__func), self)

  @util.positional_only
  @util.single_or_multiple
  def eval(self, funcs: Iterable[function.IntoArray], arguments: Mapping[str, numpy.ndarray] = ...) -> Tuple[numpy.ndarray, ...]:
    '''Evaluate function.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    datas = self.eval_sparse(funcs, arguments)
    with log.iter.fraction('assembling', datas) as items:
      return tuple(map(sparse.toarray, items))

  @util.positional_only
  @util.single_or_multiple
  def eval_sparse(self, funcs: Iterable[function.IntoArray], arguments: Optional[Mapping[str, numpy.ndarray]] = None) -> Tuple[numpy.ndarray, ...]:
    '''Evaluate function.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    return eval_integrals_sparse(*map(self.__rmatmul__, funcs), **(arguments or {}))

  def __rmatmul__(self, __func: function.IntoArray) -> function.Array:
    return _AtSample(function.Array.cast(__func), self)

  def basis(self) -> function.Array:
    '''Basis-like function that for every point in the sample evaluates to the
    unit vector corresponding to its index.'''

    raise NotImplementedError

  def asfunction(self, array: numpy.ndarray) -> function.Array:
    '''Convert sampled data to evaluable array.

    Using the result of :func:`Sample.eval`, create a
    :class:`nutils.function.Sampled` array that upon evaluation recovers the
    original function in the set of points matching the original sampling.

    >>> from nutils import mesh
    >>> domain, geom = mesh.rectilinear([1,2])
    >>> gauss = domain.sample('gauss', 2)
    >>> data = gauss.eval(geom)
    >>> sampled = gauss.asfunction(data)
    >>> domain.integrate(sampled, degree=2)
    array([ 1.,  2.])

    Args
    ----
    array :
        The sampled data.
    '''

    return function.matmat(self.basis(), array)

  @property
  @abc.abstractmethod
  def tri(self) -> numpy.ndarray:
    '''Triangulation of interior.

    A two-dimensional integer array with ``ndims+1`` columns, of which every
    row defines a simplex by mapping vertices into the list of points.
    '''

    raise NotImplementedError

  @property
  @abc.abstractmethod
  def hull(self) -> numpy.ndarray:
    '''Triangulation of the exterior hull.

    A two-dimensional integer array with ``ndims`` columns, of which every row
    defines a simplex by mapping vertices into the list of points. Note that
    the hull often does contain internal element boundaries as the
    triangulations originating from separate elements are disconnected.
    '''

    raise NotImplementedError

  def subset(self, __mask: numpy.ndarray) -> 'Sample':
    '''Reduce the number of points.

    Simple selection mechanism that returns a reduced Sample based on a
    selection mask. Points that are marked True will still be part of the new
    subset; points marked False may be dropped but this is not guaranteed. The
    point order of the original Sample is preserved.

    Args
    ----
    mask : :class:`bool` array.
        Boolean mask that selects all points that should remain. The resulting
        Sample may contain more points than this, but not less.

    Returns
    -------
    subset : :class:`Sample`
    '''

    raise NotImplementedError

class _TransformChainsSample(Sample):

  __slots__ = 'space', 'transforms', 'points'

  def __init__(self, space: str, transforms: Tuple[Transforms, ...], points: PointsSequence) -> None:
    '''
    parameters
    ----------
    space : ::class:`str`
        The name of the space on which this sample is defined.
    transforms : :class:`tuple` or transformation chains
        List of transformation chains leading to local coordinate systems that
        contain points.
    points : :class:`~nutils.pointsseq.PointsSequence`
        Points sequence.
    '''

    assert len(transforms) >= 1
    assert all(len(t) == len(points) for t in transforms)
    self.space = space
    self.transforms = transforms
    self.points = points
    super().__init__((space,), transforms[0].fromdims, len(points), points.npoints)

  def get_evaluable_weights(self, __ielem: evaluable.Array) -> evaluable.Array:
    return self.points.get_evaluable_weights(__ielem)

  def update_lower_args(self, __ielem: evaluable.Array, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> Tuple[_PointsShape, _TransformChainsMap, _CoordinatesMap]:
    if self.space in transform_chains or self.space in coordinates:
      raise ValueError('nested integrals or samples in the same space are not supported')

    transform_chains = dict(transform_chains)
    transform_chains[self.space] = space_transform_chains = tuple(t.get_evaluable(__ielem) for t in (self.transforms*2)[:2])

    space_coordinates = self.points.get_evaluable_coords(__ielem)
    assert space_coordinates.ndim == 2 # axes: points, coord dim
    coordinates = {space: evaluable.Transpose.to_end(evaluable.appendaxes(coords, space_coordinates.shape[:-1]), coords.ndim - 1) for space, coords in coordinates.items()}
    coordinates[self.space] = evaluable.prependaxes(space_coordinates, points_shape)

    points_shape = points_shape + space_coordinates.shape[:-1]

    return points_shape, transform_chains, coordinates

  def basis(self) -> function.Array:
    return _Basis(self)

  def subset(self, mask: numpy.ndarray) -> Sample:
    selection = types.frozenarray([ielem for ielem in range(self.nelems) if mask[self.getindex(ielem)].any()])
    transforms = tuple(transform[selection] for transform in self.transforms)
    return Sample.new(self.space, transforms, self.points.take(selection))

class _DefaultIndex(_TransformChainsSample):

  __slots__ = ()
  __cache__ = 'offsets'

  @property
  def offsets(self) -> numpy.ndarray:
    return types.frozenarray(numpy.cumsum([0]+[p.npoints for p in self.points]), copy=False)

  def getindex(self, ielem: int) -> numpy.ndarray:
    return types.frozenarray(numpy.arange(*self.offsets[ielem:ielem+2]), copy=False)

  @property
  def tri(self) -> numpy.ndarray:
    return self.points.tri

  @property
  def hull(self) -> numpy.ndarray:
    return self.points.hull

  def get_evaluable_indices(self, ielem: evaluable.Array) -> evaluable.Array:
    npoints = self.points.get_evaluable_coords(ielem).shape[0]
    offset = evaluable.get(_offsets(self.points), 0, ielem)
    return evaluable.Range(npoints) + offset

class _CustomIndex(_TransformChainsSample):

  __slots__ = '_parent', '_index'

  def __init__(self, parent: Sample, index: numpy.ndarray) -> None:
    assert index.shape == (parent.npoints,)
    self._parent = parent
    self._index = index
    super().__init__(parent.space, parent.transforms, parent.points)

  def getindex(self, ielem: int) -> numpy.ndarray:
    return numpy.take(self._index, self._parent.getindex(ielem))

  def get_evaluable_indices(self, ielem: evaluable.Array) -> evaluable.Array:
    return evaluable.Take(self._index, self._parent.get_evaluable_indices(ielem))

  @property
  def tri(self) -> numpy.ndarray:
    return numpy.take(self._index, self._parent.tri)

  @property
  def hull(self) -> numpy.ndarray:
    return numpy.take(self._index, self._parent.hull)

def eval_integrals(*integrals: evaluable.AsEvaluableArray, **arguments: Mapping[str, numpy.ndarray]) -> Tuple[Union[numpy.ndarray, matrix.Matrix], ...]:
  '''Evaluate integrals.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :meth:`nutils.function.Array.eval` on each
  integral individually, integrations will be grouped per Sample and jointly
  executed, potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  with log.iter.fraction('assembling', eval_integrals_sparse(*integrals, **argdict(arguments))) as retvals:
    return tuple(_convert(retval, inplace=True) for retval in retvals)

def eval_integrals_sparse(*integrals: evaluable.AsEvaluableArray, **arguments: Mapping[str, numpy.ndarray]) -> Tuple[numpy.ndarray, ...]:
  '''Evaluate integrals into sparse data.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :meth:`nutils.function.Array.eval` on each
  integral individually, integrations will be grouped per Sample and jointly
  executed, potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  integrals = tuple(integral.as_evaluable_array.assparse for integral in integrals)
  with evaluable.Tuple(tuple(integrals)).optimized_for_numpy.session(graphviz=graphviz) as eval:
    return eval(**arguments)

def _convert(data: numpy.ndarray, inplace: bool = False) -> Union[numpy.ndarray, matrix.Matrix]:
  '''Convert a two-dimensional sparse object to an appropriate object.

  The return type is determined based on dimension: a zero-dimensional object
  becomes a scalar, a one-dimensional object a (dense) Numpy vector, a
  two-dimensional object a Nutils matrix, and any higher dimensional object a
  deduplicated and pruned sparse object.
  '''

  ndim = sparse.ndim(data)
  return sparse.toarray(data) if ndim < 2 \
    else matrix.fromsparse(data, inplace=inplace) if ndim == 2 \
    else sparse.prune(sparse.dedup(data, inplace=inplace), inplace=True)

class _Integral(function.Array):

  def __init__(self, integrand: function.Array, sample: Sample) -> None:
    self._integrand = integrand
    self._sample = sample
    super().__init__(shape=integrand.shape, dtype=float if integrand.dtype in (bool, int) else integrand.dtype, spaces=integrand.spaces - frozenset(sample.spaces))

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    ielem = evaluable.loop_index('_sample_' + '_'.join(self._sample.spaces), self._sample.nelems)
    points_shape, transform_chains, coordinates = self._sample.update_lower_args(ielem, points_shape, transform_chains, coordinates)
    jacobian = util.product(evaluable.sqrt_abs_det_gram(transform_chains[space][0].linear) for space in self._sample.spaces)
    weights = self._sample.get_evaluable_weights(ielem)
    integrand = self._integrand.lower(points_shape, transform_chains, coordinates)
    elem_integral = evaluable.einsum(',B,ABC->AC', jacobian, weights, integrand, B=weights.ndim, C=self.ndim)
    return evaluable.loop_sum(elem_integral, ielem)

class _AtSample(function.Array):

  def __init__(self, func: function.Array, sample: _TransformChainsSample) -> None:
    self._func = func
    self._sample = sample
    super().__init__(shape=(sample.npoints, *func.shape), dtype=func.dtype, spaces=func.spaces - frozenset(sample.spaces))

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    axis = len(points_shape)
    ielem = evaluable.loop_index('_sample_' + '_'.join(self._sample.spaces), self._sample.nelems)
    points_shape, transform_chains, coordinates = self._sample.update_lower_args(ielem, points_shape, transform_chains, coordinates)
    indices = self._sample.get_evaluable_indices(ielem)
    axes = range(axis, axis + indices.ndim)
    func = self._func.lower(points_shape, transform_chains, coordinates)
    inflated = evaluable.Transpose.from_end(evaluable.Inflate(evaluable.Transpose.to_end(func, *axes), indices, self._sample.npoints), axis)
    return evaluable.loop_sum(inflated, ielem)

class _Basis(function.Array):

  def __init__(self, sample: _TransformChainsSample) -> None:
    self._sample = sample
    super().__init__(shape=(sample.npoints,), dtype=float, spaces=frozenset({sample.space}))

  def lower(self, points_shape: _PointsShape, transform_chains: _TransformChainsMap, coordinates: _CoordinatesMap) -> evaluable.Array:
    aligned_space_coords = coordinates[self._sample.space]
    assert aligned_space_coords.ndim == len(points_shape) + 1
    space_coords, where = evaluable.unalign(aligned_space_coords)
    # Reinsert the coordinate axis, the last axis of `aligned_space_coords`, or
    # make sure this is the last axis of `space_coords`.
    if len(points_shape) not in where:
      space_coords = evaluable.InsertAxis(space_coords, aligned_space_coords.shape[-1])
      where += len(points_shape),
    elif where[-1] != len(points_shape):
      space_coords = evaluable.Transpose(space_coords, numpy.argsort(where))
      where = tuple(sorted(where))

    chain = transform_chains[self._sample.space][0]
    index, tail = self._sample.transforms[0].evaluable_index_with_tail(chain)
    coords = tail.apply(space_coords)
    expect = self._sample.points.get_evaluable_coords(index)
    sampled = evaluable.Sampled(coords, expect)
    indices = self._sample.get_evaluable_indices(index)
    basis = evaluable.Inflate(sampled, dofmap=indices, length=self._sample.npoints)

    # Realign the points axes. The coordinate axis of `aligned_space_coords` is
    # replaced by a dofs axis in the aligned basis, hence we can reuse `where`.
    return evaluable.align(basis, where, (*points_shape, self._sample.npoints))

def _offsets(pointsseq: PointsSequence) -> evaluable.Array:
  ielem = evaluable.loop_index('_ielem', len(pointsseq))
  npoints, ndims = pointsseq.get_evaluable_coords(ielem).shape
  return evaluable._SizesToOffsets(evaluable.loop_concatenate(evaluable.InsertAxis(npoints, 1), ielem))

# vim:sw=2:sts=2:et
