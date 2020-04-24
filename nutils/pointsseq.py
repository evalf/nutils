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

'''Sequences of :class:`~nutils.points.Points`.'''

from . import types, numeric
from .points import Points
from typing import Tuple, Sequence, Iterable, Iterator, Optional, Union, overload
import abc, itertools, numpy

class PointsSequence(types.Singleton):
  '''Abstract base class for a sequence of :class:`~nutils.points.Points`.

  Parameters
  ----------
  ndims : :class:`int`
      The dimension of the point coordinates.

  Attributes
  ----------
  ndims : :class:`int`
      The dimension of the point coordinates.

  Notes
  -----
  Subclasses must implement :meth:`__len__` and :meth:`get`.
  '''

  __slots__ = 'ndims'
  __cache__ = 'npoints', 'tri', 'hull'

  @staticmethod
  def from_iter(value: Iterable[Points], ndims: int) -> 'PointsSequence':
    '''Create a :class:`PointsSequence` from an iterator.

    Parameters
    ----------
    value : iterable of :class:`~nutils.points.Points` objects
    ndims : :class:`int`

    Returns
    -------
    sequence : :class:`PointsSequence`
    '''

    value = tuple(value)
    if not all(item.ndims == ndims for item in value):
      raise ValueError('not all `Points` in the sequence have ndims equal to {}'.format(ndims))
    if len(value) == 0:
      return _Empty(ndims)
    elif all(item == value[0] for item in value[1:]):
      return _Uniform(value[0], len(value))
    else:
      return _Plain(value, ndims)

  @staticmethod
  def uniform(value: Points, length: int) -> 'PointsSequence':
    '''Create a uniform :class:`PointsSequence`.

    Parameters
    ----------
    value : :class:`~nutils.points.Points`
    length : :class:`int`

    Returns
    -------
    sequence : :class:`PointsSequence`
    '''

    if length < 0:
      raise ValueError('expected nonnegative `length` but got {}'.format(length))
    elif length == 0:
      return _Empty(value.ndims)
    else:
      return _Uniform(value, length)

  @staticmethod
  def empty(ndims: int) -> 'PointsSequence':
    '''Create an empty :class:`PointsSequence`.

    Parameters
    ----------
    ndims : :class:`int`

    Returns
    -------
    sequence : :class:`PointsSequence`
    '''

    if ndims < 0:
      raise ValueError('expected nonnegative `ndims` but got {}'.format(ndims))
    else:
      return _Empty(ndims)

  def __init__(self, ndims: int) -> None:
    self.ndims = ndims
    super().__init__()

  @property
  def npoints(self) -> int:
    '''The total number of points in this sequence.'''

    return sum(p.npoints for p in self)

  def __bool__(self) -> bool:
    '''Return ``bool(self)``.'''

    return bool(len(self))

  @abc.abstractmethod
  def __len__(self) -> int:
    '''Return ``len(self)``.'''

    raise NotImplementedError

  def __iter__(self) -> Iterator[Points]:
    '''Implement ``iter(self)``.'''

    return map(self.get, range(len(self)))

  @overload
  def __getitem__(self, index: int) -> Points:
    ...
  @overload
  def __getitem__(self, index: Union[slice, numpy.ndarray]) -> 'PointsSequence':
    ...
  def __getitem__(self, index):
    '''Return ``self[index]``.'''

    if numeric.isint(index):
      return self.get(index)
    elif isinstance(index, slice):
      index = range(len(self))[index]
      if index == range(len(self)):
        return self
      return self.take(numpy.arange(index.start, index.stop, index.step))
    elif numeric.isintarray(index):
      return self.take(index)
    elif numeric.isboolarray(index):
      return self.compress(index)
    else:
      raise IndexError('invalid index: {}'.format(index))

  def __add__(self, other: 'PointsSequence') -> 'PointsSequence':
    '''Return ``self+other``.'''

    if isinstance(other, PointsSequence):
      return self.chain(other)
    else:
      return NotImplemented

  @overload
  def __mul__(self, other: int) -> Points:
    ...
  @overload
  def __mul__(self, other: 'PointsSequence') -> 'PointsSequence':
    ...
  def __mul__(self, other):
    '''Return ``self*other``.'''

    if numeric.isint(other):
      return self.repeat(other)
    elif isinstance(other, PointsSequence):
      return self.product(other)
    else:
      return NotImplemented

  @abc.abstractmethod
  def get(self, index: int) -> Points:
    '''Return the points at ``index``.

    Parameters
    ----------
    index : :class:`int`

    Returns
    -------
    points: :class:`~nutils.points.Points`
        The points at ``index``.
    '''

    raise NotImplementedError

  def take(self, indices: numpy.ndarray) -> 'PointsSequence':
    '''Return a selection of this sequence.

    Parameters
    ----------
    indices : :class:`numpy.ndarray`, ndim: 1, dtype: int
        The indices of points of this sequence to select.

    Returns
    -------
    points: :class:`PointsSequence`
        The sequence of selected points.
    '''

    _check_take(len(self), indices)
    if len(indices) == 0:
      return _Empty(self.ndims)
    elif len(indices) == 1:
      return _Uniform(self.get(indices[0]), 1)
    else:
      return _Take(self, types.frozenarray(indices))

  def compress(self, mask: numpy.ndarray) -> 'PointsSequence':
    '''Return a selection of this sequence.

    Parameters
    ----------
    mask : :class:`numpy.ndarray`, ndim: 1, dtype: bool
        A boolean mask of points of this sequence to select.

    Returns
    -------
    sequence: :class:`PointsSequence`
        The sequence of selected points.
    '''

    _check_compress(len(self), mask)
    return self.take(numpy.nonzero(mask)[0])

  def repeat(self, count: int) -> 'PointsSequence':
    '''Return this sequence repeated ``count`` times.

    Parameters
    ----------
    count : :class:`int`

    Returns
    -------
    sequence : :class:`PointsSequence`
        This sequence repeated ``count`` times.
    '''

    _check_repeat(count)
    if count == 0:
      return _Empty(self.ndims)
    elif count == 1:
      return self
    else:
      return _Repeat(self, count)

  def product(self, other: 'PointsSequence') -> 'PointsSequence':
    '''Return the product of this sequence with another sequence.

    Parameters
    ----------
    other : :class:`PointsSequence`

    Returns
    -------
    sequence : :class:`PointsSequence`
        This product sequence.
    '''

    return _Product(self, other)

  def chain(self, other: 'PointsSequence') -> 'PointsSequence':
    '''Return the chained sequence of this sequence with ``other``.

    Parameters
    ----------
    other : :class:`PointsSequence`

    Returns
    -------
    sequence : :class:`PointsSequence`
        The chained sequence.
    '''

    if other.ndims != self.ndims:
      raise ValueError('expected a `PointsSequence` with ndims={} but got {}'.format(self.ndims, other.ndims))
    if not other:
      return self
    elif not self:
      return other
    else:
      selfitems = list(_unchain(self))
      otheritems = list(_unchain(other))
      # Since `self` and `other` are already properly merged, it suffices to
      # merge the tail of `self` with the head of `other`. Both `selfitems` and
      # `otheritems` cannot be empty by the above tests.
      merged = _merge_chain(selfitems[-1], otheritems[0])
      if merged:
        return _balanced_chain(selfitems[:-1] + [merged] + otheritems[1:])
      else:
        return _balanced_chain(selfitems + otheritems)

  @property
  def tri(self) -> types.frozenarray:
    '''Triangulation of interior.

    A two-dimensional integer array with ``ndims+1`` columns, of which every
    row defines a simplex by mapping vertices into the list of points.
    '''

    tri = []
    offset = 0
    for points in self:
      tri.append(points.tri + offset)
      offset += points.npoints
    return types.frozenarray(numpy.concatenate(tri) if tri else numpy.zeros((0,self.ndims+1), int), copy=False)

  @property
  def hull(self) -> types.frozenarray:
    '''Triangulation of the exterior hull.

    A two-dimensional integer array with ``ndims`` columns, of which every row
    defines a simplex by mapping vertices into the list of points. Note that
    the hull often does contain internal element boundaries as the
    triangulations originating from separate elements are disconnected.
    '''

    hull = []
    offset = 0
    for points in self:
      hull.append(points.hull + offset)
      offset += points.npoints
    return types.frozenarray(numpy.concatenate(hull) if hull else numpy.zeros((0,self.ndims), int), copy=False)

class _Empty(PointsSequence):

  __slots__ = ()

  def __len__(self) -> int:
    return 0

  def get(self, index: int) -> Points:
    raise IndexError('sequence index out of range')

class _Plain(PointsSequence):

  __slots__ = 'items'

  def __init__(self, items: Tuple[Points, ...], ndims: int) -> None:
    assert len(items), 'inefficient; this should have been `_Empty`'
    assert not all(item == items[0] for item in items), 'inefficient; this should have been `_Uniform`'
    assert all(item.ndims == ndims for item in items), 'not all items have ndims equal to {}'.format(ndims)
    self.items = items
    super().__init__(ndims)

  def __len__(self) -> int:
    return len(self.items)

  def __iter__(self) -> Iterator[Points]:
    return iter(self.items)

  def get(self, index: int) -> Points:
    return self.items[index]

class _Uniform(PointsSequence):

  __slots__ = 'item', 'length'
  __cache__ = 'tri', 'hull'

  def __init__(self, item, length):
    assert length >= 0, 'length should be nonnegative'
    assert length > 0, 'inefficient; this should have been `_Empty`'
    self.item = item
    self.length = length
    super().__init__(item.ndims)

  @property
  def npoints(self) -> int:
    return self.item.npoints * self.length

  def __len__(self) -> int:
    return self.length

  def __iter__(self) -> Iterator[Points]:
    return itertools.repeat(self.item, len(self))

  def get(self, index: int) -> Points:
    numeric.normdim(len(self), index)
    return self.item

  def take(self, indices: numpy.ndarray) -> PointsSequence:
    _check_take(len(self), indices)
    return PointsSequence.uniform(self.item, len(indices))

  def compress(self, mask: numpy.ndarray) -> PointsSequence:
    _check_compress(len(self), mask)
    return PointsSequence.uniform(self.item, mask.sum())

  def repeat(self, count: int) -> PointsSequence:
    _check_repeat(count)
    if count == 0:
      return _Empty(self.ndims)
    else:
      return PointsSequence.uniform(self.item, len(self) * count)

  def product(self, other: PointsSequence) -> PointsSequence:
    if isinstance(other, _Uniform):
      return PointsSequence.uniform(self.item * other.item, len(self) * len(other))
    else:
      return super().product(other)

  def _mk_indices(self, item: numpy.ndarray) -> types.frozenarray:
    npoints = self.item.npoints
    ind = item[None] + numpy.arange(0, len(self)*npoints, npoints)[:,None,None]
    ind = ind.reshape(len(self)*item.shape[0], item.shape[1])
    return types.frozenarray(ind, copy=False)

  @property
  def tri(self) -> types.frozenarray:
    return self._mk_indices(self.item.tri)

  @property
  def hull(self) -> types.frozenarray:
    return self._mk_indices(self.item.hull)

class _Take(PointsSequence):

  __slots__ = 'parent', 'indices'

  def __init__(self, parent, indices):
    _check_take(len(parent), indices)
    assert len(indices) > 1, 'inefficient; this should have been `_Empty` or `_Uniform`'
    assert not isinstance(parent, _Uniform), 'inefficient; this should have been `_Uniform`'
    self.parent = parent
    self.indices = indices
    super().__init__(parent.ndims)

  def __len__(self) -> int:
    return len(self.indices)

  def __iter__(self) -> Iterator[Points]:
    return map(self.parent.get, self.indices)

  def get(self, index: int) -> Points:
    return self.parent.get(self.indices[index])

  def take(self, indices: numpy.ndarray) -> PointsSequence:
    _check_take(len(self), indices)
    return self.parent.take(numpy.take(self.indices, indices))

  def compress(self, mask: numpy.ndarray) -> PointsSequence:
    _check_compress(len(self), mask)
    return self.parent.take(numpy.compress(mask, self.indices))

class _Repeat(PointsSequence):

  __slots__ = 'parent', 'count'
  __cache__ = 'tri', 'hull'

  def __init__(self, parent, count):
    assert count >= 0, 'count should be nonnegative'
    assert count > 0, 'inefficient; this should have been `_Empty`'
    assert not isinstance(parent, _Uniform), 'inefficient; this should have been `_Uniform`'
    self.parent = parent
    self.count = count
    super().__init__(parent.ndims)

  @property
  def npoints(self) -> int:
    return self.parent.npoints * self.count

  def __len__(self) -> int:
    return len(self.parent) * self.count

  def __iter__(self) -> Iterator[Points]:
    for i in range(self.count):
      yield from self.parent

  def get(self, index: int) -> Points:
    return self.parent.get(numeric.normdim(len(self), index) % len(self.parent))

  def repeat(self, count: int) -> PointsSequence:
    _check_repeat(count)
    if count == 0:
      return _Empty(self.ndims)
    else:
      return _Repeat(self.parent, self.count * count)

  def _mk_indices(self, parent: numpy.ndarray) -> types.frozenarray:
    npoints = self.parent.npoints
    ind = parent[None] + numpy.arange(0, self.count*npoints, npoints)[:,None,None]
    ind = ind.reshape(self.count*parent.shape[0], parent.shape[1])
    return types.frozenarray(ind, copy=False)

  @property
  def tri(self) -> types.frozenarray:
    return self._mk_indices(self.parent.tri)

  @property
  def hull(self) -> types.frozenarray:
    return self._mk_indices(self.parent.hull)

class _Product(PointsSequence):

  __slots__ = 'sequence1', 'sequence2'

  @types.apply_annotations
  def __init__(self, sequence1, sequence2):
    assert not (isinstance(sequence1, _Uniform) and isinstance(sequence2, _Uniform)), 'inefficient; this should have been `_Uniform`'
    self.sequence1 = sequence1
    self.sequence2 = sequence2
    super().__init__(sequence1.ndims + sequence2.ndims)

  @property
  def npoints(self) -> int:
    return self.sequence1.npoints * self.sequence2.npoints

  def __len__(self) -> int:
    return len(self.sequence1) * len(self.sequence2)

  def __iter__(self) -> Iterator[Points]:
    return (item1.product(item2) for item1 in self.sequence1 for item2 in self.sequence2)

  def get(self, index: int) -> Points:
    index1, index2 = divmod(numeric.normdim(len(self), index), len(self.sequence2))
    return self.sequence1.get(index1).product(self.sequence2.get(index2))

  def product(self, other: PointsSequence) -> PointsSequence:
    return self.sequence1.product(self.sequence2.product(other))

class _Chain(PointsSequence):

  __slots__ = 'sequence1', 'sequence2'
  __cache__ = 'tri', 'hull'

  def __init__(self, sequence1, sequence2):
    assert sequence1.ndims == sequence2.ndims, 'cannot chain sequences with different ndims'
    assert sequence1 and sequence2, 'inefficient; at least one of the sequences is empty'
    assert not _merge_chain(sequence1, sequence2), 'inefficient; this should have been `_Uniform` or `_Repeat`'
    self.sequence1 = sequence1
    self.sequence2 = sequence2
    super().__init__(sequence1.ndims)

  @property
  def npoints(self) -> int:
    return self.sequence1.npoints + self.sequence2.npoints

  def __len__(self) -> int:
    return len(self.sequence1) + len(self.sequence2)

  def __iter__(self) -> Iterator[Points]:
    return itertools.chain(self.sequence1, self.sequence2)

  def get(self, index: int) -> Points:
    index = numeric.normdim(len(self), index)
    n = len(self.sequence1)
    if index < n:
      return self.sequence1.get(index)
    else:
      return self.sequence2.get(index - n)

  def take(self, indices: numpy.ndarray) -> PointsSequence:
    _check_take(len(self), indices)
    n = len(self.sequence1)
    mask = numpy.less(indices, n)
    return self.sequence1.take(numpy.compress(mask, indices)).chain(self.sequence2.take(numpy.compress(~mask, indices) - n))

  def compress(self, mask: numpy.ndarray) -> PointsSequence:
    _check_compress(len(self), mask)
    n = len(self.sequence1)
    return self.sequence1.compress(mask[:n]).chain(self.sequence2.compress(mask[n:]))

  @property
  def tri(self) -> types.frozenarray:
    tri1 = self.sequence1.tri
    tri2 = self.sequence2.tri
    return types.frozenarray(numpy.concatenate([tri1, tri2 + self.sequence1.npoints]), copy=False)

  @property
  def hull(self) -> types.frozenarray:
    hull1 = self.sequence1.hull
    hull2 = self.sequence2.hull
    return types.frozenarray(numpy.concatenate([hull1, hull2 + self.sequence1.npoints]), copy=False)

def _unchain(seq: PointsSequence) -> Iterator[PointsSequence]:
  if isinstance(seq, _Chain):
    yield from _unchain(seq.sequence1)
    yield from _unchain(seq.sequence2)
  elif seq: # skip empty sequences
    yield seq

def _balanced_chain(items: Sequence[PointsSequence]) -> PointsSequence:
  assert items
  if len(items) == 1:
    return items[0]
  else:
    c = numpy.cumsum([0]+list(map(len, items)))
    i = numpy.argmin(abs(c[1:-1] - c[-1]/2)) + 1
    a = _balanced_chain(items[:i])
    b = _balanced_chain(items[i:])
    return _merge_chain(a, b) or _Chain(a, b)

def _merge_chain(a: PointsSequence, b: PointsSequence) -> Optional[PointsSequence]: # type: ignore[return]
  if a == b:
    return a.repeat(2)
  if isinstance(a, _Uniform) and isinstance(b, _Uniform) and a.item == b.item:
    return _Uniform(a.item, len(a) + len(b))
  if isinstance(a, _Repeat):
    if isinstance(b, _Repeat) and a.parent == b.parent:
      return a.parent.repeat(a.count + b.count)
    elif a.parent == b:
      return a.parent.repeat(a.count + 1)
  elif isinstance(b, _Repeat) and b.parent == a:
    return b.parent.repeat(b.count + 1)

def _check_repeat(count):
  if count < 0:
    raise ValueError('expected nonnegative `count` but got {}'.format(count))

def _check_take(length, indices):
  if not numeric.isintarray(indices):
    raise IndexError('expected an array of integers')
  if not indices.ndim == 1:
    raise IndexError('expected an array with dimension 1 but got {}'.format(indices.ndim))
  if len(indices) and not (0 <= indices.min() and indices.max() < length):
    raise IndexError('`indices` out of range')

def _check_compress(length, mask):
  if not numeric.isboolarray(mask):
    raise IndexError('expected an array of booleans')
  if not mask.ndim == 1:
    raise IndexError('expected an array with dimension 1 but got {}'.format(mask.ndim))
  if len(mask) != length:
    raise IndexError('expected an array with length {} but got {}'.format(length, len(mask)))

# vim:sw=2:sts=2:et
