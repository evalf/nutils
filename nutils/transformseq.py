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

"""The transformseq module."""

from . import types, numeric, util, transform, element, elementseq
import abc, itertools, operator, numpy

class Transforms(types.Singleton):
  '''Abstract base class for a sequence of :class:`~nutils.transform.TransformItem` tuples.

  This class resembles to some extent a plain :class:`tuple`: the
  class supports indexing, iterating and has an :meth:`index` method.  In
  addition the class supports the :meth:`index_with_tail` method which can be
  used to find the index of a transform given the transform plus any number of
  child transformations.

  The transforms in this sequence must satisfy the following condition: any
  transform must not start with any other transform in the same sequence.

  Parameters
  ----------
  fromdims : :class:`int`
      The number of dimensions all transforms in this sequence map from.

  Attributes
  ----------
  fromdims : :class:`int`
      The number of dimensions all transforms in this sequence map from.

  Notes
  -----
  Subclasses must implement :meth:`__getitem__`, :meth:`__len__` and
  :meth:`index_with_tail`.
  '''

  __slots__ = 'fromdims'

  @types.apply_annotations
  def __init__(self, fromdims:types.strictint):
    self.fromdims = fromdims
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
      if index.step < 0:
        raise NotImplementedError('reordering the sequence is not yet implemented')
      return MaskedTransforms(self, numpy.arange(index.start, index.stop, index.step))
    elif numeric.isintarray(index):
      if index.ndim != 1:
        raise IndexError('invalid index')
      if numpy.any(numpy.less(index, 0)) or numpy.any(numpy.greater_equal(index, len(self))):
        raise IndexError('index out of range')
      dindex = numpy.diff(index)
      if len(index) == len(self) and (len(self) == 0 or (index[0] == 0 and numpy.all(numpy.equal(dindex, 1)))):
        return self
      if numpy.any(numpy.equal(dindex, 0)):
        raise ValueError('repeating an element is not allowed')
      if not numpy.all(numpy.greater(dindex, 0)):
        s = numpy.argsort(index)
        return ReorderedTransforms(self[index[s]], numpy.argsort(s))
      if len(index) == 0:
        return EmptyTransforms(self.fromdims)
      if len(index) == len(self):
        return self
      return MaskedTransforms(self, index)
    elif numeric.isboolarray(index):
      if index.shape != (len(self),):
        raise IndexError('mask has invalid shape')
      if not numpy.any(index):
        return EmptyTransforms(self.fromdims)
      if numpy.all(index):
        return self
      index, = numpy.where(index)
      return MaskedTransforms(self, index)
    else:
      raise IndexError('invalid index')

  @abc.abstractmethod
  def index_with_tail(self, trans):
    '''Return the index of ``trans[:n]`` and the tail ``trans[n:]``.

    Find the index of a transform in this sequence given the transform plus any
    number of child transforms.  In other words: find ``index`` such that
    ``self[index] == trans[:n]`` for some ``n``.  Note that there is either
    exactly one ``index`` satisfying this condition, or none, due to the
    restrictions of the transforms in a :class:`Transforms` object.

    Parameters
    ----------
    trans : :class:`tuple` of :class:`nutils.transform.TransformItem` objects
        The transform to find up to a possibly empty tail.

    Returns
    -------
    index : :class:`int`
        The index of ``trans`` without tail in this sequence.
    tail : :class:`tuple` of :class:`nutils.transform.TransformItem` objects
        The tail: ``trans[len(self[index]):]``.

    Raises
    ------
    :class:`ValueError`
        if ``trans`` is not found.

    Example
    -------

    Consider the following plain sequence of two shift transforms:

    >>> from nutils.transform import Shift, Scale
    >>> transforms = PlainTransforms([(Shift([0.]),), (Shift([1.]),)], fromdims=1)

    Calling :meth:`index_with_tail` with the first transform gives index ``0``
    and no tail:

    >>> transforms.index_with_tail((Shift([0.]),))
    (0, ())

    Calling with an additional scale gives:

    >>> transforms.index_with_tail((Shift([0.]), Scale(0.5, [0.])))
    (0, (Scale([0]+0.5*x),))
    '''

    raise NotImplementedError

  def __iter__(self):
    '''Implement ``iter(self)``.'''

    for i in range(len(self)):
      yield self[i]

  def index(self, trans):
    '''Return the index of ``trans``.

    Parameters
    ----------
    trans : :class:`tuple` of :class:`nutils.transform.TransformItem` objects

    Returns
    -------
    index : :class:`int`
        The index of ``trans`` in this sequence.

    Raises
    ------
    :class:`ValueError`
        if ``trans`` is not found.

    Example
    -------

    Consider the following plain sequence of two shift transforms:

    >>> from nutils.transform import Shift, Scale
    >>> transforms = PlainTransforms([(Shift([0.]),), (Shift([1.]),)], fromdims=1)

    Calling :meth:`index` with the first transform gives index ``0``:

    >>> transforms.index((Shift([0.]),))
    0

    Calling with an additional scale raises an exception, because the transform
    is not present in ``transforms``.

    >>> transforms.index((Shift([0.]), Scale(0.5, [0.])))
    Traceback (most recent call last):
      ...
    ValueError: (Shift([0]+x), Scale([0]+0.5*x)) not in sequence of transforms
    '''

    index, tail = self.index_with_tail(trans)
    if tail:
      raise ValueError('{!r} not in sequence of transforms'.format(trans))
    return index

  def contains(self, trans):
    '''Return ``trans`` in ``self``.

    Parameters
    ----------
    trans : :class:`tuple` of :class:`nutils.transform.TransformItem` objects

    Returns
    -------
    :class:`bool`
        ``True`` if ``trans`` is contained in this sequence of transforms, i.e.
        if :meth:`index` returns without :class:`ValueError`, otherwise
        ``False``.
    '''

    try:
      self.index(trans)
    except ValueError:
      return False
    else:
      return True

  __contains__ = contains

  def contains_with_tail(self, trans):
    '''Return ``trans[:n]`` in ``self`` for some ``n``.

    Parameters
    ----------
    trans : :class:`tuple` of :class:`nutils.transform.TransformItem` objects

    Returns
    -------
    :class:`bool`
        ``True`` if a head of ``trans`` is contained in this sequence
        of transforms, i.e. if :meth:`index_with_tail` returns without
        :class:`ValueError`, otherwise ``False``.
    '''

    try:
      self.index_with_tail(trans)
    except ValueError:
      return False
    else:
      return True

  def refined(self, references):
    '''Return the sequence of refined transforms given ``references``.

    Parameters
    ----------
    references : :class:`~nutils.elementseq.References`
        A sequence of references matching this sequence of transforms.

    Returns
    -------
    :class:`Transforms`
        The sequence of refined transforms::

            (trans+(ctrans,) for trans, ref in zip(self, references) for ctrans in ref.child_transforms)
    '''

    if references.isuniform:
      return UniformDerivedTransforms(self, references[0], 'child_transforms', self.fromdims)
    else:
      return DerivedTransforms(self, references, 'child_transforms', self.fromdims)

  def edges(self, references):
    '''Return the sequence of edge transforms given ``references``.

    Parameters
    ----------
    references : :class:`~nutils.elementseq.References`
        A sequence of references matching this sequence of transforms.

    Returns
    -------
    :class:`Transforms`
        The sequence of edge transforms::

            (trans+(etrans,) for trans, ref in zip(self, references) for etrans in ref.edge_transforms)
    '''

    if references.isuniform:
      return UniformDerivedTransforms(self, references[0], 'edge_transforms', self.fromdims-1)
    else:
      return DerivedTransforms(self, references, 'edge_transforms', self.fromdims-1)

  def __add__(self, other):
    '''Return ``self+other``.'''

    if not isinstance(other, Transforms) or self.fromdims != other.fromdims:
      return NotImplemented
    return chain((self, other), self.fromdims)

  def unchain(self):
    '''Iterator of unchained :class:`Transforms` items.

    Yields
    ------
    :class:`Transforms`
        Unchained items.
    '''

    yield self

stricttransforms = types.strict[Transforms]

class EmptyTransforms(Transforms):
  '''An empty sequence.'''

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    raise IndexError('index out of range')

  def __len__(self):
    return 0

  def index_with_tail(self, trans):
    raise ValueError

  def index(self, trans):
    raise ValueError

  def contains_with_tail(self, trans):
    return False

  def contains(self, trans):
    return False

  __contains__ = contains

class PlainTransforms(Transforms):
  '''A general purpose implementation of :class:`Transforms`.

  Use this class only if there exists no specific implementation of
  :class:`Transforms` for the transforms at hand.

  Parameters
  ----------
  transforms : :class:`tuple` of :class:`~nutils.transform.TransformItem` objects
      The sequence of transforms.
  fromdims : :class:`int`
      The number of dimensions all ``transforms`` map from.
  '''

  __slots__ = '_transforms', '_sorted', '_indices'

  @types.apply_annotations
  def __init__(self, transforms:types.tuple[transform.canonical], fromdims:types.strictint):
    transforms_fromdims = set(trans[-1].fromdims for trans in transforms)
    if not (transforms_fromdims <= {fromdims}):
      raise ValueError('expected transforms with fromdims={}, but got {}'.format(fromdims, transforms_fromdims))
    self._transforms = transforms
    self._sorted = numpy.empty([len(self._transforms)], dtype=object)
    for i, trans in enumerate(self._transforms):
      self._sorted[i] = tuple(map(id, trans))
    self._indices = numpy.argsort(self._sorted)
    self._sorted = self._sorted[self._indices]
    super().__init__(fromdims)

  def __iter__(self):
    return iter(self._transforms)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    return self._transforms[numeric.normdim(len(self), index)]

  def __len__(self):
    return len(self._transforms)

  def index_with_tail(self, trans):
    trans, orig_trans = transform.promote(trans, self.fromdims), trans
    transid_array = numpy.empty((), dtype=object)
    transid_array[()] = transid = tuple(map(id, trans))
    i = numpy.searchsorted(self._sorted, transid_array, side='right') - 1
    if i < 0:
      raise ValueError('{!r} not in sequence of transforms'.format(orig_trans))
    match = self._sorted[i]
    if transid[:len(match)] != match:
      raise ValueError('{!r} not in sequence of transforms'.format(orig_trans))
    return self._indices[i], trans[len(match):]

class IdentifierTransforms(Transforms):
  '''A sequence of :class:`nutils.transform.Identifier` singletons.

  Every identifier is instantiated with three arguments: the dimension, the
  name string, and an integer index matching its position in the sequence.

  Parameters
  ----------
  ndims : :class:`int`
      Dimension of the transformation.
  name : :class:`str`
      Identifying name string.
  length : :class:`int`
      Length of the sequence.
  '''

  __slots__ = '_name', '_length'

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, name:str, length:int):
    self._name = name
    self._length = length
    super().__init__(ndims)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = int(index) # make sure that index is a Python integer rather than numpy.intxx
    return transform.Identifier(self.fromdims, (self._name, numeric.normdim(self._length, index))),

  def __len__(self):
    return self._length

  def index_with_tail(self, trans):
    root = trans[0]
    if root.fromdims == self.fromdims and isinstance(root, transform.Identifier) and isinstance(root.token, tuple) and len(root.token) == 2 and root.token[0] == self._name and 0 <= root.token[1] < self._length:
      return root.token[1], trans[1:]
    raise ValueError

class Axis(types.Singleton):
  '''Abstract base class for axes of :class:`~nutils.topology.StructuredTopology`.'''

  __slots__ = ()

class DimAxis(Axis):

  __slots__ = 'i', 'j', 'isperiodic'
  isdim = True

  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, isperiodic:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.isperiodic = isperiodic

  def __len__(self):
    return self.j - self.i

  def unmap(self, index):
    if not self.i <= index < self.j:
      raise ValueError
    return index-self.i

  def map(self, index):
    return self.i+index

class EdgeAxis(Axis):

  __slots__ = 'i', 'j', 'ibound', 'side'
  isdim = False

  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, ibound:types.strictint, side:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.ibound = ibound
    self.side = side

class BndAxis(EdgeAxis):

  __slots__ = ()

  def __len__(self):
    return 1

  def unmap(self, index):
    if index != (self.i-1 if self.side else self.j):
      raise ValueError
    return 0

  def map(self, index):
    return self.i-1 if self.side else self.j

class IntAxis(EdgeAxis):

  __slots__ = ()

  def __len__(self):
    return self.j - self.i - 1

  def unmap(self, index):
    if not self.i < index + self.side < self.j:
      raise ValueError
    return index - self.i - (not self.side)

  def map(self, index):
    return index + self.i + (not self.side)

class PIntAxis(EdgeAxis):

  __slots__ = ()

  def __len__(self):
    return self.j - self.i

  def unmap(self, index):
    if not self.i <= index < self.j:
      raise ValueError
    return (index - self.i - (not self.side)) % len(self)

  def map(self, index):
    return self.i + ((index + (not self.side)) % len(self))

class StructuredTransforms(Transforms):
  '''Transforms sequence for :class:`~nutils.topology.StructuredTopology`.

  Parameters
  ----------
  root : :class:`~nutils.transform.TransformItem`
      Root transform of the :class:`~nutils.topology.StructuredTopology`.
  axes : :class:`tuple` of :class:`Axis` objects
      The axes defining the :class:`~nutils.topology.StructuredTopology`.
  nrefine : :class:`int`
      Number of structured refinements.
  '''

  __slots__ = '_root', '_axes', '_nrefine', '_etransforms', '_ctransforms', '_cindices'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, axes:types.tuple[types.strict[Axis]], nrefine:types.strictint):
    self._root = root
    self._axes = axes
    self._nrefine = nrefine

    ref = element.LineReference()**len(self._axes)
    self._ctransforms = numeric.asobjvector(ref.child_transforms).reshape((2,)*len(self._axes))
    self._cindices = {t: numpy.array(i, dtype=int) for i, t in numpy.ndenumerate(self._ctransforms)}

    etransforms = []
    rmdims = numpy.zeros(len(axes), dtype=bool)
    for order, side, idim in sorted((axis.ibound, axis.side, idim) for idim, axis in enumerate(axes) if not axis.isdim):
      ref = util.product(element.getsimplex(0 if rmdim else 1) for rmdim in rmdims)
      iedge = (idim - rmdims[:idim].sum()) * 2 + 1 - side
      etransforms.append(ref.edge_transforms[iedge])
      rmdims[idim] = True
    self._etransforms = tuple(etransforms)

    super().__init__(sum(axis.isdim for axis in self._axes))

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = numeric.normdim(len(self), index)
    # Decompose index into indices per dimension on the nrefined level.
    indices = []
    for axis in reversed(self._axes):
      index, rem = divmod(index, len(axis))
      indices.insert(0, axis.map(rem))
    assert index == 0
    # Create transform.
    ctransforms = []
    indices = numpy.asarray(indices, dtype=int)
    for i in range(self._nrefine):
      indices, r = divmod(indices, self._ctransforms.shape)
      ctransforms.insert(0, self._ctransforms[tuple(r)])
    trans0 = transform.Shift(types.frozenarray(indices, dtype=float, copy=False))
    return (self._root, trans0, *ctransforms, *self._etransforms)

  def __len__(self):
    return util.product(map(len, self._axes))

  def index_with_tail(self, trans):
    if len(trans) < 2 + self._nrefine + len(self._etransforms):
      raise ValueError

    root, shift, tail = trans[0], trans[1], transform.uppermost(trans[2:])
    if root != self._root:
      raise ValueError

    if not isinstance(shift, transform.Shift) or len(shift.offset) != len(self._axes) or not numpy.equal(shift.offset.astype(int), shift.offset).all():
      raise ValueError
    indices = numpy.array(shift.offset, dtype=int)

    # Match child transforms.
    for item in tail[:self._nrefine]:
      try:
        indices = indices*2 + self._cindices[item]
      except KeyError:
        raise ValueError

    # Check index boundaries and flatten.
    flatindex = 0
    for index, axis in zip(indices, self._axes):
      flatindex = flatindex*len(axis) + axis.unmap(index)

    # Promote the remainder and match the edge transforms.
    tail = transform.promote(tail[self._nrefine:], self.fromdims)
    if tail[:len(self._etransforms)] != self._etransforms:
      raise ValueError
    tail = tail[len(self._etransforms):]

    return flatindex, tail

class MaskedTransforms(Transforms):
  '''An order preserving subset of another :class:`Transforms` object.

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to subset.
  indices : one-dimensional array of :class:`int`\\s
      The strict monotonic increasing indices of ``parent`` transforms to keep.
  '''

  __slots__ = '_parent', '_mask', '_indices'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, indices:types.frozenarray[types.strictint]):
    self._parent = parent
    self._indices = indices
    super().__init__(parent.fromdims)

  def __iter__(self):
    for itrans in self._indices:
      yield self._parent[int(itrans)]

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.any(numpy.less(index, 0)):
      raise IndexError('index out of bounds')
    return self._parent[self._indices[index]]

  def __len__(self):
    return len(self._indices)

  def index_with_tail(self, trans):
    parent_index, tail = self._parent.index_with_tail(trans)
    index = numpy.searchsorted(self._indices, parent_index)
    if index == len(self._indices) or self._indices[index] != parent_index:
      raise ValueError
    else:
      return int(index), tail

class ReorderedTransforms(Transforms):
  '''A reordered :class:`Transforms` object.

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to reorder.
  indices : one-dimensional array of :class:`int`\\s
      The new order of the transforms.
  '''

  __slots__ = '_parent', '_mask', '_indices'
  __cache__ = '_rindices'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, indices:types.frozenarray[types.strictint]):
    self._parent = parent
    self._indices = indices
    super().__init__(parent.fromdims)

  @property
  def _rindices(self):
    return numpy.argsort(self._indices)

  def __iter__(self):
    for itrans in self._indices:
      yield self._parent[int(itrans)]

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.any(numpy.less(index, 0)):
      raise IndexError('index out of bounds')
    return self._parent[self._indices[index]]

  def __len__(self):
    return len(self._parent)

  def index_with_tail(self, trans):
    parent_index, tail = self._parent.index_with_tail(trans)
    return int(self._rindices[parent_index]), tail

class DerivedTransforms(Transforms):
  '''A sequence of derived transforms.

  The derived transforms are ordered first by parent transforms, then by derived
  transforms, as returned by the reference::

      (trans+(ctrans,) for trans, ref in zip(parent, parent_references) for ctrans in getattr(ref, derived_attribute))

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to refine.
  parent_references: :class:`~nutils.elementseq.References`
      The references to use for the refinement.
  derived_attribute : :class:`str`
      The name of the attribute of a :class:`nutils.element.Reference` that
      contains the derived references.
  fromdims : :class:`int`
      The number of dimensions all transforms in this sequence map from.
  '''

  __slots__ = '_parent', '_parent_references', '_derived_transforms'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, parent_references:elementseq.strictreferences, derived_attribute:types.strictstr, fromdims:types.strictint):
    if len(parent) != len(parent_references):
      raise ValueError('`parent` and `parent_references` should have the same length')
    if parent.fromdims != parent_references.ndims:
      raise ValueError('`parent` and `parent_references` have different dimensions')
    self._parent = parent
    self._parent_references = parent_references
    self._derived_transforms = operator.attrgetter(derived_attribute)
    super().__init__(fromdims)

  @property
  def _offsets(self):
    return types.frozenarray(numpy.cumsum([0, *(len(self._derived_transforms(ref)) for ref in self._parent_references)]), copy=False)

  def __len__(self):
    return self._offsets[-1]

  def __iter__(self):
    for reference, trans in zip(self._parent_references, self._parent):
      for dtrans in self._derived_transforms(reference):
        yield trans+(dtrans,)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = numeric.normdim(len(self), index)
    iparent = numpy.searchsorted(self._offsets, index, side='right')-1
    assert 0 <= iparent < len(self._offsets)-1
    iderived = index - self._offsets[iparent]
    return self._parent[iparent] + (self._derived_transforms(self._parent_references[iparent])[iderived],)

  def index_with_tail(self, trans):
    iparent, tail = self._parent.index_with_tail(trans)
    if not tail:
      raise ValueError
    if self.fromdims == self._parent.fromdims:
      tail = transform.uppermost(tail)
    else:
      tail = transform.canonical(tail)
    iderived = self._derived_transforms(self._parent_references[iparent]).index(tail[0])
    return self._offsets[iparent]+iderived, tail[1:]

class UniformDerivedTransforms(Transforms):
  '''A sequence of refined transforms from a uniform sequence of references.

  The refined transforms are ordered first by parent transforms, then by
  derived transforms, as returned by the reference::

      (trans+(ctrans,) for trans in parent for ctrans in getattr(parent_reference, derived_attribute))

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to refine.
  parent_reference: :class:`~nutils.element.Reference`
      The reference to use for the refinement.
  derived_attribute : :class:`str`
      The name of the attribute of a :class:`nutils.element.Reference` that
      contains the derived references.
  fromdims : :class:`int`
      The number of dimensions all transforms in this sequence map from.
  '''

  __slots__ = '_parent', '_derived_transforms'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, parent_reference:element.strictreference, derived_attribute:types.strictstr, fromdims:types.strictint):
    if parent.fromdims != parent_reference.ndims:
      raise ValueError('`parent` and `parent_reference` have different dimensions')
    self._parent = parent
    self._derived_transforms = getattr(parent_reference, derived_attribute)
    super().__init__(fromdims)

  def __len__(self):
    return len(self._parent)*len(self._derived_transforms)

  def __iter__(self):
    for trans in self._parent:
      for dtrans in self._derived_transforms:
        yield trans+(dtrans,)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    iparent, iderived = divmod(numeric.normdim(len(self), index), len(self._derived_transforms))
    return self._parent[iparent] + (self._derived_transforms[iderived],)

  def index_with_tail(self, trans):
    iparent, tail = self._parent.index_with_tail(trans)
    if not tail:
      raise ValueError
    if self.fromdims == self._parent.fromdims:
      tail = transform.uppermost(tail)
    else:
      tail = transform.canonical(tail)
    iderived = self._derived_transforms.index(tail[0])
    return iparent*len(self._derived_transforms) + iderived, tail[1:]

class ProductTransforms(Transforms):
  '''The product of two :class:`Transforms` objects.

  The order of the resulting transforms is: ``transforms1[0]*transforms2[0],
  transforms1[0]*transforms2[1], ..., transforms1[1]*transforms2[0],
  transforms1[1]*transforms2[1], ...``.

  Parameters
  ----------
  transforms1 : :class:`Transforms`
      The first sequence of transforms.
  transforms2 : :class:`Transforms`
      The second sequence of transforms.
  '''

  __slots__ = '_transforms1', '_transforms2'

  @types.apply_annotations
  def __init__(self, transforms1:stricttransforms, transforms2:stricttransforms):
    self._transforms1 = transforms1
    self._transforms2 = transforms2
    super().__init__(transforms1.fromdims+transforms2.fromdims)

  def __iter__(self):
    for trans1 in self._transforms1:
      for trans2 in self._transforms2:
        yield transform.Bifurcate(trans1, trans2),

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index1, index2 = divmod(numeric.normdim(len(self), index), len(self._transforms2))
    return transform.Bifurcate(self._transforms1[index1], self._transforms2[index2]),

  def __len__(self):
    return len(self._transforms1) * len(self._transforms2)

  def index_with_tail(self, trans):
    bf = trans[0]
    assert isinstance(bf, transform.Bifurcate)
    index1, tail1 = self._transforms1.index_with_tail(bf.trans1[:-1])
    index2, tail2 = self._transforms2.index_with_tail(bf.trans2[:-1])
    return index1*len(self._transforms2)+index2, None # FIXME

class ChainedTransforms(Transforms):
  '''A sequence of chained :class:`Transforms` objects.

  Parameters
  ----------
  items: :class:`tuple` of :class:`Transforms` objects
      The :class:`Transforms` objects to chain.
  '''

  __slots__ = '_items'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, items:types.tuple[stricttransforms]):
    if len(items) == 0:
      raise ValueError('Empty chain.')
    if len(set(item.fromdims for item in items)) != 1:
      raise ValueError('Cannot chain Transforms with different fromdims.')
    self._items = items
    super().__init__(self._items[0].fromdims)

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
        return EmptyTransforms(self.fromdims)
      ostart = numpy.searchsorted(self._offsets, index.start, side='right') - 1
      ostop = numpy.searchsorted(self._offsets, index.stop, side='left')
      return chain((item[max(0,index.start-istart):min(istop-istart,index.stop-istart)] for item, (istart, istop) in zip(self._items[ostart:ostop], util.pairwise(self._offsets[ostart:ostop+1]))), self.fromdims)
    elif numeric.isintarray(index) and index.ndim == 1 and len(index) and numpy.all(numpy.greater(numpy.diff(index), 0)):
      if index[0] < 0 or index[-1] >= len(self):
        raise IndexError('index out of bounds')
      split = numpy.searchsorted(index, self._offsets, side='left')
      return chain((item[index[start:stop]-offset] for item, offset, (start, stop) in zip(self._items, self._offsets, util.pairwise(split)) if stop > start), self.fromdims)
    elif numeric.isboolarray(index) and index.shape == (len(self),):
      return chain((item[index[start:stop]] for item, (start, stop) in zip(self._items, util.pairwise(self._offsets))), self.fromdims)
    else:
      return super().__getitem__(index)

  def __iter__(self):
    return itertools.chain.from_iterable(self._items)

  def index_with_tail(self, trans):
    offset = 0
    for item in self._items:
      try:
        index, tail = item.index_with_tail(trans)
        return index + offset, tail
      except ValueError:
        pass
      offset += len(item)
    raise ValueError

  def refined(self, references):
    return chain((item.refined(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.fromdims)

  def edges(self, references):
    return chain((item.edges(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.fromdims-1)

  def unchain(self):
    yield from self._items

def chain(items, fromdims):
  '''Return the chained transforms sequence of ``items``.

  Parameters
  ----------
  items : iterable of :class:`Transforms` objects
      The :class:`Transforms` objects to chain.
  fromdims : :class:`int`
      The number of dimensions all transforms in this sequence map from.

  Returns
  -------
  :class:`Transforms`
      The chained transforms.
  '''

  unchained = tuple(filter(len, itertools.chain.from_iterable(item.unchain() for item in items)))
  items_fromdims = set(item.fromdims for item in unchained)
  if not (items_fromdims <= {fromdims}):
    raise ValueError('expected transforms with fromdims={}, but got {}'.format(fromdims, items_fromdims))
  if len(unchained) == 0:
    return EmptyTransforms(fromdims)
  elif len(unchained) == 1:
    return unchained[0]
  else:
    return ChainedTransforms(unchained)

# vim:sw=2:sts=2:et
