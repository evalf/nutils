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

  __slots__ = 'nelems', 'space', 'transforms', 'points', 'ndims'

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
    self.nelems = len(transforms[0])
    self.space = space
    self.transforms = transforms
    self.points = points
    self.ndims = transforms[0].fromdims

  def __repr__(self) -> str:
    return '{}.{}<{}D, {} elems, {} points>'.format(type(self).__module__, type(self).__qualname__, self.ndims, self.nelems, self.npoints)

  @property
  def npoints(self) -> int:
    return self.points.npoints

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

  def _lower_for_loop(self, func: function.Array, points_shape: Tuple[evaluable.Array, ...] = (), transform_chains: Tuple[EvaluableTransformChain, ...] = (), coordinates: Tuple[evaluable.Array, ...] = ()) -> Tuple[evaluable.Array, Tuple[evaluable.Array, ...], Tuple[EvaluableTransformChain, ...], Tuple[evaluable.Array, ...], evaluable.Array]:
    if transform_chains or coordinates:
      raise ValueError('nested integrals or samples are not yet supported')
    ielem = evaluable.loop_index('_ielem', self.nelems)
    transform_chains = tuple(t.get_evaluable(ielem) for t in self.transforms)
    coordinates = (evaluable.prependaxes(self.points.get_evaluable_coords(ielem), points_shape),)*len(transform_chains)
    points_shape = points_shape + coordinates[0].shape[:-1]
    lowered = func.lower(points_shape=points_shape, transform_chains=transform_chains, coordinates=coordinates)
    return ielem, points_shape, transform_chains, coordinates, lowered

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

    return _Basis(self)

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

    selection = types.frozenarray([ielem for ielem in range(self.nelems) if __mask[self.getindex(ielem)].any()])
    transforms = tuple(transform[selection] for transform in self.transforms)
    return Sample.new(self.space, transforms, self.points.take(selection))

class _DefaultIndex(Sample):

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

class _CustomIndex(Sample):

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
    if len(integrand.spaces) > 1:
      raise NotImplementedError
    self._integrand = integrand
    self._sample = sample
    super().__init__(shape=integrand.shape, dtype=float if integrand.dtype in (bool, int) else integrand.dtype, spaces=integrand.spaces - frozenset({sample.space}))

  def lower(self, **kwargs):
    ielem, points_shape, transform_chains, coordinates, integrand = self._sample._lower_for_loop(self._integrand, **kwargs)
    rootdim = self._sample.transforms[0].todims
    manifolddim = self._sample.transforms[0].fromdims
    assert manifolddim <= rootdim
    weights = self._sample.points.get_evaluable_weights(ielem)
    elem_integral = evaluable.einsum('B,ABC->AC', weights, integrand, B=weights.ndim, C=self.ndim)
    return evaluable.loop_sum(elem_integral, ielem)

class _AtSample(function.Array):

  def __init__(self, func: function.Array, sample: Sample) -> None:
    if len(func.spaces) > 1:
      raise NotImplementedError
    self._func = func
    self._sample = sample
    super().__init__(shape=(sample.points.npoints, *func.shape), dtype=func.dtype, spaces=func.spaces - frozenset({sample.space}))

  def lower(self, **kwargs) -> evaluable.Array:
    ielem, points_shape, transform_chains, coordinates, func = self._sample._lower_for_loop(self._func, **kwargs)
    indices = self._sample.get_evaluable_indices(ielem)
    inflated = evaluable.Transpose.from_end(evaluable.Inflate(evaluable.Transpose.to_end(func, 0), indices, self._sample.npoints), 0)
    return evaluable.loop_sum(inflated, ielem)

class _Basis(function.Array):

  def __init__(self, sample: Sample) -> None:
    self._sample = sample
    super().__init__(shape=(sample.npoints,), dtype=float, spaces=frozenset({sample.space}))

  def lower(self, *, transform_chains=(), coordinates=(), **kwargs) -> evaluable.Array:
    assert transform_chains and coordinates and len(transform_chains) == len(coordinates)
    index, tail = transform_chains[0].index_with_tail_in(self._sample.transforms[0])
    coords = tail.apply(coordinates[0])
    expect = self._sample.points.get_evaluable_coords(index)
    sampled = evaluable.Sampled(coords, expect)
    indices = self._sample.get_evaluable_indices(index)
    return evaluable.Inflate(sampled, dofmap=indices, length=self._sample.npoints)

def _offsets(pointsseq: PointsSequence) -> evaluable.Array:
  ielem = evaluable.loop_index('_ielem', len(pointsseq))
  npoints, ndims = pointsseq.get_evaluable_coords(ielem).shape
  return evaluable._SizesToOffsets(evaluable.loop_concatenate(evaluable.InsertAxis(npoints, 1), ielem))

# vim:sw=2:sts=2:et
