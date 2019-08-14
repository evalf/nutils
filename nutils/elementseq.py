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

"""The elementseq module."""

from . import types, numeric, element, util
import abc, collections.abc, itertools, operator, numpy

class References(types.Singleton):
  '''Abstract base class for a sequence of :class:`~nutils.element.Reference` objects.

  Parameters
  ----------
  ndims : :class:`int`
      The number of dimensions of the references in this sequence.

  Notes
  -----
  Subclasses must implement :meth:`__len__` and :meth:`__getitem__`.
  '''

  __slots__ = 'ndims'

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    self.ndims = ndims
    super().__init__()

  @abc.abstractmethod
  def __len__(self):
    '''Return ``len(self)``.'''

    raise NotImplementedError

  def __getitem__(self, index):
    '''Return ``self[index]``.'''

    if numeric.isint(index):
      raise NotImplementedError
    elif isinstance(index, slice):
      index = range(len(self))[index]
      if index == range(len(self)):
        return self
      return SelectedReferences(self, numpy.arange(index.start, index.stop, index.step))
    elif numeric.isintarray(index):
      if index.ndim != 1:
        raise IndexError('invalid index')
      if numpy.any(numpy.less(index, 0)) or numpy.any(numpy.greater_equal(index, len(self))):
        raise IndexError('index out of range')
      if len(index) == 0:
        return EmptyReferences(self.ndims)
      if numpy.all(numpy.equal(numpy.diff(index), 1)) and len(index) == len(self):
        return self
      return SelectedReferences(self, index)
    elif numeric.isboolarray(index):
      if index.shape != (len(self),):
        raise IndexError('mask has invalid shape')
      if not numpy.any(index):
        return EmptyReferences(self.ndims)
      if numpy.all(index):
        return self
      index, = numpy.where(index)
      return SelectedReferences(self, index)
    else:
      raise IndexError('invalid index')

  def __iter__(self):
    '''Implement ``iter(self)``.'''

    for i in range(len(self)):
      yield self[i]

  @property
  def children(self):
    '''Return the sequence of child references.

    Returns
    -------
    :class:`References`
        The sequence of child references::

            (cref for ref in self for cref in ref.child_refs)
    '''

    return DerivedReferences(self, 'child_refs', self.ndims)

  @property
  def edges(self):
    '''Return the sequence of edge references.

    Returns
    -------
    :class:`References`
        The sequence of edge references::

            (eref for ref in self for eref in ref.edge_refs)
    '''

    return DerivedReferences(self, 'edge_refs', self.ndims-1)

  def getpoints(self, ischeme, degree):
    '''Return a sequence of :class:`~nutils.points.Points`.'''

    return tuple(reference.getpoints(ischeme, degree) for reference in self)

  def __add__(self, other):
    '''Return ``self+other``.'''

    if not isinstance(other, References):
      return NotImplemented
    return chain((self, other), self.ndims)

  def __mul__(self, other):
    '''Return ``self*other``.'''

    if numeric.isint(other):
      if other == 0:
        return EmptyReferences(self.ndims)
      elif other == 1:
        return self
      else:
        return RepeatedReferences(self, other)
    elif isinstance(other, References):
      return ProductReferences(self, other)
    else:
      return NotImplemented

  def unchain(self):
    '''Iterator of unchained :class:`References` items.

    Yields
    ------
    :class:`References`
        Unchained items.
    '''

    yield self

  @property
  def isuniform(self):
    '''``True`` if all reference in this sequence are equal.'''

    return False

strictreferences = types.strict[References]

class EmptyReferences(References):
  '''An empty sequence of references.'''

  def __len__(self):
    return 0

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    raise IndexError('index out of range')

  def __iter__(self):
    return iter(())

  @property
  def children(self):
    return self

  @property
  def edges(self):
    return EmptyReferences(self.ndims-1)

class PlainReferences(References):
  '''A general purpose implementation of :class:`References`.

  Use this class only if there exists no specific implementation of
  :class:`References` for the references at hand.

  Parameters
  ----------
  references : :class:`tuple` of :class:`~nutils.element.Reference` objects
      The sequence of references.
  ndims : :class:`int`
      The number of dimensions of the ``references``.
  '''

  __slots__ = '_references'

  @types.apply_annotations
  def __init__(self, references:types.tuple[element.strictreference], ndims:types.strictint):
    refs_ndims = set(ref.ndims for ref in references)
    if not (refs_ndims <= {ndims}):
      raise ValueError('expected references with ndims={}, but got {}'.format(ndims, refs_ndims))
    self._references = references
    super().__init__(ndims)

  def __len__(self):
    return len(self._references)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    return self._references[numeric.normdim(len(self), index)]

  def __iter__(self):
    return iter(self._references)

class UniformReferences(References):
  '''A uniform sequence.

  Parameters
  ----------
  reference : :class:`~nutils.element.Reference`
      The reference.
  length : :class:`int`
      The length of the sequence.
  '''

  __slots__ = '_reference', '_length'
  __cache__ = 'children', 'edges'

  @types.apply_annotations
  def __init__(self, reference:element.strictreference, length:types.strictint):
    if length <= 0:
      raise ValueError('length should be strict positive, but got {}'.format(length))
    self._reference = reference
    self._length = length
    super().__init__(reference.ndims)

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if numeric.isint(index):
      numeric.normdim(len(self), index)
      return self._reference
    elif isinstance(index, slice):
      return asreferences([self._reference], self.ndims) * len(range(len(self))[index])
    elif numeric.isintarray(index) and index.ndim == 1:
      if numpy.any(numpy.less(index, 0)) or numpy.any(numpy.greater_equal(index, len(self))):
        raise IndexError('index out of range')
      return asreferences([self._reference], self.ndims) * len(index)
    elif numeric.isboolarray(index) and index.shape == (len(self),):
      return asreferences([self._reference], self.ndims) * numpy.sum(index)
    else:
      return super().__getitem__(index)

  @property
  def children(self):
    return asreferences(self._reference.child_refs, self.ndims) * len(self)

  @property
  def edges(self):
    return asreferences(self._reference.edge_refs, self.ndims-1) * len(self)

  def getpoints(self, ischeme, degree):
    return (self._reference.getpoints(ischeme, degree),)*len(self)

  def __mul__(self, other):
    if numeric.isint(other):
      if other == 0:
        return EmptyReferences(self.ndims)
      else:
        return UniformReferences(self._reference, len(self)*other)
    else:
      return super().__mul__(other)

  @property
  def isuniform(self):
    return True

class SelectedReferences(References):
  '''A selection of references.  Duplication and reordering is allowed.

  Parameters
  ----------
  parent : :class:`References`
      The transforms to subset.
  indices : one-dimensional array of :class:`int`\\s
      Indices of ``parent`` that form this selection.
  '''

  __slots__ = '_parent', '_indices'

  @types.apply_annotations
  def __init__(self, parent:strictreferences, indices:types.frozenarray[types.strictint]):
    if not numpy.all(numpy.greater_equal(indices, 0) & numpy.less(indices, len(parent))):
      raise IndexError('`indices` out of range')
    self._parent = parent
    self._indices = indices
    super().__init__(parent.ndims)

  def __len__(self):
    return len(self._indices)

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.any(numpy.less(index, 0)):
      raise IndexError('index out of bounds')
    return self._parent[self._indices[index]]

class ChainedReferences(References):
  '''A sequence of chained :class:`References` objects.

  Parameters
  ----------
  items: :class:`tuple` of :class:`References` objects
      The :class:`References` objects to chain.
  '''

  __slots__ = '_items'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, items:types.tuple[strictreferences]):
    if len(items) == 0:
      raise ValueError('Empty chain.')
    if len(set(item.ndims for item in items)) != 1:
      raise ValueError('Cannot chain References with different ndims.')
    self._items = items
    super().__init__(self._items[0].ndims)

  @property
  def _offsets(self):
    return types.frozenarray(numpy.cumsum([0, *map(len, self._items)]), copy=False)

  def __len__(self):
    return self._offsets[-1]

  def __getitem__(self, index):
    if numeric.isint(index):
      index = numeric.normdim(len(self), index)
      outer = numpy.searchsorted(self._offsets, index, side='right') - 1
      assert outer >= 0 and outer < len(self._items)
      return self._items[outer][index-self._offsets[outer]]
    elif isinstance(index, slice) and index.step in (1, None):
      index = range(len(self))[index]
      if index == range(len(self)):
        return self
      elif index.start == index.stop:
        return EmptyReferences(self.ndims)
      ostart = numpy.searchsorted(self._offsets, index.start, side='right') - 1
      ostop = numpy.searchsorted(self._offsets, index.stop, side='left')
      return chain((item[max(0,index.start-istart):min(istop-istart,index.stop-istart)] for item, (istart, istop) in zip(self._items[ostart:ostop], util.pairwise(self._offsets[ostart:ostop+1]))), self.ndims)
    elif numeric.isintarray(index) and index.ndim == 1 and len(index) and numpy.all(numpy.greater(numpy.diff(index), 0)):
      if index[0] < 0 or index[-1] >= len(self):
        raise IndexError('index out of bounds')
      split = numpy.searchsorted(index, self._offsets, side='left')
      return chain((item[index[start:stop]-offset] for item, offset, (start, stop) in zip(self._items, self._offsets, util.pairwise(split)) if stop > start), self.ndims)
    elif numeric.isboolarray(index) and index.shape == (len(self),):
      return chain((item[index[start:stop]] for item, (start, stop) in zip(self._items, util.pairwise(self._offsets))), self.ndims)
    else:
      return super().__getitem__(index)

  def __iter__(self):
    return itertools.chain.from_iterable(self._items)

  @property
  def children(self):
    return chain((item.children for item in self._items), self.ndims)

  @property
  def edges(self):
    return chain((item.edges for item in self._items), self.ndims-1)

  def getpoints(self, ischeme, degree):
    return tuple(itertools.chain.from_iterable(item.getpoints(ischeme, degree) for item in self._items))

  def unchain(self):
    yield from self._items

class RepeatedReferences(References):
  '''An n-times repeated sequence of references.

  Parameters
  ----------
  parent : :class:`References`
      The references to repeat.
  count : :class:`int`
      The number of repetitions.
  '''

  __slots__ = '_parent', '_count'

  @types.apply_annotations
  def __init__(self, parent:strictreferences, count:types.strictint):
    if count <= 0:
      raise ValueError('count should be strict positive, but got {}'.format(count))
    self._parent = parent
    self._count = count
    super().__init__(parent.ndims)

  def __len__(self):
    return len(self._parent)*self._count

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    return self._parent[numeric.normdim(len(self), index) % len(self._parent)]

  def __iter__(self):
    for i in range(self._count):
      yield from self._parent

  def __mul__(self, other):
    if numeric.isint(other):
      if other == 0:
        return EmptyReferences(self.ndims)
      elif other == 1:
        return self
      else:
        return RepeatedReferences(self._parent, self._count*other)
    else:
      return super().__mul__(other)

  @property
  def children(self):
    return self._parent.children * self._count

  @property
  def edges(self):
    return self._parent.edges * self._count

  def getpoints(self, ischeme, degree):
    return self._parent.getpoints(ischeme, degree) * self._count

class ProductReferences(References):
  '''A sequence of products of two other sequences.

  Parameters
  ----------
  left : :class:`References`
  right : :class:`References`
  '''

  __slots__ = '_left', '_right'

  @types.apply_annotations
  def __init__(self, left:strictreferences, right:strictreferences):
    self._left = left
    self._right = right
    super().__init__(left.ndims+right.ndims)

  def __len__(self):
    return len(self._left)*len(self._right)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    li, ri = divmod(numeric.normdim(len(self), index), len(self._right))
    return self._left[li]*self._right[ri]

  def __iter__(self):
    for lref in self._left:
      for rref in self._right:
        yield lref * rref

class DerivedReferences(References):
  '''Abstract base class for references based on parent references.

  The derived references are ordered first by parent references, then by derived
  references::

      (dref for ref in parent for dref in getattr(ref, derived_attribute))

  Parameters
  ----------
  parent : :class:`References`
      The parent references.
  derived_attribute : :class:`str`
      The name of the attribute of a :class:`nutils.element.Reference` that
      contains the derived references.
  ndims : :class:`int`
      The number of dimensions of the references in this sequence.
  '''

  __slots__ = '_parent', '_derived_refs'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, parent:strictreferences, derived_attribute:types.strictstr, ndims:types.strictint):
    self._parent = parent
    self._derived_refs = operator.attrgetter(derived_attribute)
    super().__init__(ndims)

  @property
  def _offsets(self):
    return types.frozenarray(numpy.cumsum([0, *(len(self._derived_refs(ref)) for ref in self._parent)]), copy=False)

  def __len__(self):
    return self._offsets[-1]

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = numeric.normdim(len(self), index)
    parent_index = numpy.searchsorted(self._offsets, index, side='right')-1
    derived_index = index - self._offsets[parent_index]
    return self._derived_refs(self._parent[parent_index])[derived_index]

  def __iter__(self):
    for ref in self._parent:
      yield from self._derived_refs(ref)

def asreferences(value, ndims):
  '''Convert ``value`` to a :class:`References` object.'''

  if isinstance(value, References):
    if value.ndims != ndims:
      raise ValueError('expected References object with ndims={}, but got {}'.format(ndims, value.ndims))
    return value
  elif isinstance(value, collections.abc.Iterable):
    value = tuple(value)
    if len(value) == 0:
      return EmptyReferences(ndims)
    elif all(item == value[0] for item in value[1:]):
      return UniformReferences(value[0], len(value))
    else:
      return PlainReferences(value, ndims)
  else:
    raise ValueError('cannot convert {!r} to a References object'.format(value))

def chain(items, ndims):
  '''Return the chained references sequence of ``items``.

  Parameters
  ----------
  items : iterable of :class:`References` objects
      The :class:`References` objects to chain.
  ndims : :class:`int`
      the number of dimensions all references.

  Returns
  -------
  :class:`References`
      The chained references.
  '''

  unchained = tuple(filter(len, itertools.chain.from_iterable(item.unchain() for item in items)))
  items_ndims = set(item.ndims for item in unchained)
  if not (items_ndims <= {ndims}):
    raise ValueError('expected references with ndims={}, but got {}'.format(ndims, items_ndims))
  if len(unchained) == 0:
    return EmptyReferences(ndims)
  elif len(unchained) == 1:
    return unchained[0]
  elif all(item.isuniform for item in unchained) and len(set(item[0] for item in unchained)) == 1:
    return UniformReferences(unchained[0][0], sum(map(len, unchained)))
  else:
    return ChainedReferences(unchained)

# vim:sw=2:sts=2:et
