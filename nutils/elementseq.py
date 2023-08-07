"""The elementseq module."""

from . import types, numeric, _util as util
from .element import Reference
from .pointsseq import PointsSequence
from functools import cached_property
from typing import Tuple, Sequence, Iterable, Iterator, Optional, Union, overload
import abc
import itertools
import operator
import numpy


class References(types.Singleton):
    '''Abstract base class for a sequence of :class:`~nutils.element.Reference` objects.

    Parameters
    ----------
    ndims : :class:`int`
        The number of dimensions of the references in this sequence.

    Notes
    -----
    Subclasses must implement :meth:`__len__` and :meth:`get`.
    '''

    @staticmethod
    def from_iter(value: Iterable[Reference], ndims: int) -> 'References':
        '''Create a :class:`References` sequence from an iterator.

        Parameters
        ----------
        value : iterable of :class:`~nutils.element.Reference` objects
        ndims : :class:`int`

        Returns
        -------
        sequence : :class:`References`
        '''

        value = tuple(value)
        if not all(item.ndims == ndims for item in value):
            raise ValueError('not all `Reference` objects in the sequence have ndims equal to {}'.format(ndims))
        if len(value) == 0:
            return _Empty(ndims)
        elif all(item == value[0] for item in value[1:]):
            return _Uniform(value[0], len(value))
        else:
            return _Plain(value, ndims)

    @staticmethod
    def uniform(value: Reference, length: int) -> 'References':
        '''Create a uniform :class:`References`.

        Parameters
        ----------
        value : :class:`~nutils.element.Reference`
        length : :class:`int`

        Returns
        -------
        sequence : :class:`References`
        '''

        if length < 0:
            raise ValueError('expected nonnegative `length` but got {}'.format(length))
        elif length == 0:
            return _Empty(value.ndims)
        else:
            return _Uniform(value, length)

    @staticmethod
    def empty(ndims: int) -> 'References':
        '''Create an empty :class:`References` sequence.

        Parameters
        ----------
        ndims : :class:`int`

        Returns
        -------
        sequence : :class:`References`
        '''

        if ndims < 0:
            raise ValueError('expected nonnegative `ndims` but got {}'.format(ndims))
        else:
            return _Empty(ndims)

    def __init__(self, ndims: int) -> None:
        self.ndims = ndims
        super().__init__()

    def __bool__(self) -> bool:
        '''Return ``bool(self)``.'''

        return bool(len(self))

    @abc.abstractmethod
    def __len__(self) -> int:
        '''Return ``len(self)``.'''

        raise NotImplementedError

    def __iter__(self) -> Iterator[Reference]:
        '''Implement ``iter(self)``.'''

        return map(self.get, range(len(self)))

    @overload
    def __getitem__(self, index: int) -> Reference:
        ...

    @overload
    def __getitem__(self, index: Union[slice, numpy.ndarray]) -> 'References':
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

    def __add__(self, other: 'References') -> 'References':
        '''Return ``self+other``.'''

        if isinstance(other, References):
            return self.chain(other)
        else:
            return NotImplemented

    @overload
    def __mul__(self, other: int) -> Reference:
        ...

    @overload
    def __mul__(self, other: 'References') -> 'References':
        ...

    def __mul__(self, other):
        '''Return ``self*other``.'''

        if numeric.isint(other):
            return self.repeat(other)
        elif isinstance(other, References):
            return self.product(other)
        else:
            return NotImplemented

    @abc.abstractmethod
    def get(self, index: int) -> Reference:
        '''Return the reference at ``index``.

        Parameters
        ----------
        index : :class:`int`

        Returns
        -------
        reference: :class:`~nutils.element.Reference`
            The reference at ``index``.
        '''

        raise NotImplementedError

    def take(self, indices: numpy.ndarray) -> 'References':
        '''Return a selection of this sequence.

        Parameters
        ----------
        indices : :class:`numpy.ndarray`, ndim: 1, dtype: int
            The indices of references of this sequence to select.

        Returns
        -------
        references: :class:`References`
            The sequence of selected references.
        '''

        _check_take(len(self), indices)
        if len(indices) == 0:
            return _Empty(self.ndims)
        elif len(indices) == 1:
            return _Uniform(self.get(indices[0]), 1)
        else:
            return _Take(self, types.arraydata(indices))

    def compress(self, mask: numpy.ndarray) -> 'References':
        '''Return a selection of this sequence.

        Parameters
        ----------
        mask : :class:`numpy.ndarray`, ndim: 1, dtype: bool
            A boolean mask of references of this sequence to select.

        Returns
        -------
        sequence: :class:`References`
            The sequence of selected references.
        '''

        _check_compress(len(self), mask)
        return self.take(numpy.nonzero(mask)[0])

    def repeat(self, count: int) -> 'References':
        '''Return this sequence repeated ``count`` times.

        Parameters
        ----------
        count : :class:`int`

        Returns
        -------
        sequence : :class:`References`
            This sequence repeated ``count`` times.
        '''

        _check_repeat(count)
        if count == 0:
            return _Empty(self.ndims)
        elif count == 1:
            return self
        else:
            return _Repeat(self, count)

    def product(self, other: 'References') -> 'References':
        '''Return the product of this sequence with another sequence.

        Parameters
        ----------
        other : :class:`References`

        Returns
        -------
        sequence : :class:`References`
            The product sequence.
        '''

        return _Product(self, other)

    def chain(self, other: 'References') -> 'References':
        '''Return the chained sequence of this sequence with ``other``.

        Parameters
        ----------
        other : :class:`References`

        Returns
        -------
        sequence : :class:`References`
            The chained sequence.
        '''

        if other.ndims != self.ndims:
            raise ValueError('expected a `References` sequence with ndims={} but got {}'.format(self.ndims, other.ndims))
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
    def children(self) -> 'References':
        '''Return the sequence of child references.

        Returns
        -------
        :class:`References`
            The sequence of child references::

                (cref for ref in self for cref in ref.child_refs)
        '''

        return _Derived(self, 'child_refs', self.ndims)

    @property
    def edges(self) -> 'References':
        '''Return the sequence of edge references.

        Returns
        -------
        :class:`References`
            The sequence of edge references::

                (eref for ref in self for eref in ref.edge_refs)
        '''

        return _Derived(self, 'edge_refs', self.ndims-1)

    @property
    def isuniform(self) -> 'bool':
        '''``True`` if all reference in this sequence are equal.'''

        return len(self) == 1

    def getpoints(self, ischeme: str, degree: int) -> PointsSequence:
        '''Return a sequence of :class:`~nutils.points.Points`.'''

        return PointsSequence.from_iter((reference.getpoints(ischeme, degree) for reference in self), self.ndims)


class _Empty(References):

    def __len__(self) -> int:
        return 0

    def get(self, index: int) -> Reference:
        raise IndexError('sequence index out of range')

    @property
    def children(self) -> References:
        return self

    @property
    def edges(self) -> References:
        return _Empty(self.ndims-1)

    def getpoints(self, ischeme, degree) -> PointsSequence:
        return PointsSequence.empty(self.ndims)


class _Plain(References):

    def __init__(self, items: Tuple[Reference, ...], ndims: int) -> None:
        assert len(items), 'inefficient; this should have been `_Empty`'
        assert not all(item == items[0] for item in items), 'inefficient; this should have been `_Uniform`'
        assert all(item.ndims == ndims for item in items), 'not all items have ndims equal to {}'.format(ndims)
        self.items = items
        super().__init__(ndims)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Reference]:
        return iter(self.items)

    def get(self, index) -> Reference:
        return self.items[index]


class _Uniform(References):

    def __init__(self, item: Reference, length: int) -> None:
        assert length >= 0, 'length should be nonnegative'
        assert length > 0, 'inefficient; this should have been `_Empty`'
        self.item = item
        self.length = length
        super().__init__(item.ndims)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Reference]:
        return itertools.repeat(self.item, len(self))

    def get(self, index: int) -> Reference:
        numeric.normdim(len(self), index)
        return self.item

    def take(self, indices: numpy.ndarray) -> References:
        _check_take(len(self), indices)
        return References.uniform(self.item, len(indices))

    def compress(self, mask: numpy.ndarray) -> References:
        _check_compress(len(self), mask)
        return References.uniform(self.item, mask.sum())

    def repeat(self, count: int) -> References:
        _check_repeat(count)
        if count == 0:
            return _Empty(self.ndims)
        else:
            return References.uniform(self.item, len(self) * count)

    def product(self, other: References) -> References:
        if isinstance(other, _Uniform):
            return References.uniform(self.item.product(other.item), len(self) * len(other))
        else:
            return super().product(other)

    @cached_property
    def children(self) -> References:
        return References.from_iter(self.item.child_refs, self.ndims).repeat(len(self))

    @cached_property
    def edges(self) -> References:
        return References.from_iter(self.item.edge_refs, self.ndims-1).repeat(len(self))

    @property
    def isuniform(self) -> bool:
        return True

    def getpoints(self, ischeme: str, degree: int) -> PointsSequence:
        return PointsSequence.uniform(self.item.getpoints(ischeme, degree), len(self))


class _Take(References):

    def __init__(self, parent: References, indices: types.arraydata) -> None:
        assert indices.shape[0] > 1, 'inefficient; this should have been `_Empty` or `_Uniform`'
        assert not isinstance(parent, _Uniform), 'inefficient; this should have been `_Uniform`'
        self.parent = parent
        self.indices = numpy.asarray(indices)
        _check_take(len(parent), self.indices)
        super().__init__(parent.ndims)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[Reference]:
        return map(self.parent.get, self.indices)

    def get(self, index: int) -> Reference:
        return self.parent.get(self.indices[index])

    def take(self, indices: numpy.ndarray) -> References:
        _check_take(len(self), indices)
        return self.parent.take(numpy.take(self.indices, indices))

    def compress(self, mask: numpy.ndarray) -> References:
        _check_compress(len(self), mask)
        return self.parent.take(numpy.compress(mask, self.indices))


class _Repeat(References):

    def __init__(self, parent: References, count: int) -> None:
        assert count >= 0, 'count should be nonnegative'
        assert count > 0, 'inefficient; this should have been `_Empty`'
        assert not isinstance(parent, _Uniform), 'inefficient; this should have been `_Uniform`'
        self.parent = parent
        self.count = count
        super().__init__(parent.ndims)

    def __len__(self) -> int:
        return len(self.parent) * self.count

    def __iter__(self) -> Iterator[Reference]:
        for i in range(self.count):
            yield from self.parent

    def get(self, index: int) -> Reference:
        return self.parent.get(numeric.normdim(len(self), index) % len(self.parent))

    def repeat(self, count: int) -> References:
        _check_repeat(count)
        if count == 0:
            return _Empty(self.ndims)
        else:
            return _Repeat(self.parent, self.count * count)

    @property
    def children(self) -> References:
        return self.parent.children.repeat(self.count)

    @property
    def edges(self) -> References:
        return self.parent.edges.repeat(self.count)

    def getpoints(self, ischeme: str, degree: int) -> PointsSequence:
        return self.parent.getpoints(ischeme, degree).repeat(self.count)


class _Product(References):

    def __init__(self, sequence1: References, sequence2: References) -> None:
        assert not (isinstance(sequence1, _Uniform) and isinstance(sequence2, _Uniform)), 'inefficient; this should have been `_Uniform`'
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        super().__init__(sequence1.ndims + sequence2.ndims)

    def __len__(self) -> int:
        return len(self.sequence1) * len(self.sequence2)

    def __iter__(self) -> Iterator[Reference]:
        return (item1.product(item2) for item1 in self.sequence1 for item2 in self.sequence2)

    def get(self, index: int) -> Reference:
        index1, index2 = divmod(numeric.normdim(len(self), index), len(self.sequence2))
        return self.sequence1.get(index1).product(self.sequence2.get(index2))

    def product(self, other: References) -> References:
        return self.sequence1.product(self.sequence2.product(other))

    def getpoints(self, ischeme: str, degree: int) -> PointsSequence:
        return self.sequence1.getpoints(ischeme, degree).product(self.sequence2.getpoints(ischeme, degree))


class _Chain(References):

    def __init__(self, sequence1: References, sequence2: References) -> None:
        assert sequence1.ndims == sequence2.ndims, 'cannot chain sequences with different ndims'
        assert sequence1 and sequence2, 'inefficient; at least one of the sequences is empty'
        assert not _merge_chain(sequence1, sequence2), 'inefficient; this should have been `_Uniform` or `_Repeat`'
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        super().__init__(sequence1.ndims)

    def __len__(self) -> int:
        return len(self.sequence1) + len(self.sequence2)

    def __iter__(self) -> Iterator[Reference]:
        return itertools.chain(self.sequence1, self.sequence2)

    def get(self, index: int) -> Reference:
        index = numeric.normdim(len(self), index)
        n = len(self.sequence1)
        if index < n:
            return self.sequence1.get(index)
        else:
            return self.sequence2.get(index - n)

    def take(self, indices: numpy.ndarray) -> References:
        _check_take(len(self), indices)
        n = len(self.sequence1)
        mask = numpy.less(indices, n)
        return self.sequence1.take(numpy.compress(mask, indices)).chain(self.sequence2.take(numpy.compress(~mask, indices) - n))

    def compress(self, mask: numpy.ndarray) -> References:
        _check_compress(len(self), mask)
        n = len(self.sequence1)
        return self.sequence1.compress(mask[:n]).chain(self.sequence2.compress(mask[n:]))

    @property
    def children(self) -> References:
        return self.sequence1.children.chain(self.sequence2.children)

    @property
    def edges(self) -> References:
        return self.sequence1.edges.chain(self.sequence2.edges)

    def getpoints(self, ischeme: str, degree: int) -> PointsSequence:
        return self.sequence1.getpoints(ischeme, degree).chain(self.sequence2.getpoints(ischeme, degree))


class _Derived(References):

    def __init__(self, parent: References, derived_attribute: str, ndims: int) -> None:
        self.parent = parent
        self.derived_refs = operator.attrgetter(derived_attribute)
        super().__init__(ndims)

    @cached_property
    def offsets(self) -> numpy.ndarray:
        return types.frozenarray(numpy.cumsum([0, *(len(self.derived_refs(ref)) for ref in self.parent)]), copy=False)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __iter__(self) -> Iterator[Reference]:
        for ref in self.parent:
            yield from self.derived_refs(ref)

    def get(self, index: int) -> Reference:
        index = numeric.normdim(len(self), index)
        parent_index = numpy.searchsorted(self.offsets, index, side='right')-1
        derived_index = index - self.offsets[parent_index]
        return self.derived_refs(self.parent.get(parent_index))[derived_index]


def _unchain(seq: References) -> Iterator[References]:
    if isinstance(seq, _Chain):
        yield from _unchain(seq.sequence1)
        yield from _unchain(seq.sequence2)
    elif seq:  # skip empty sequences
        yield seq


def _balanced_chain(items: Sequence[References]) -> References:
    assert items
    if len(items) == 1:
        return items[0]
    else:
        c = numpy.cumsum([0]+list(map(len, items)))
        i = numpy.argmin(abs(c[1:-1] - c[-1]/2)) + 1
        a = _balanced_chain(items[:i])
        b = _balanced_chain(items[i:])
        return _merge_chain(a, b) or _Chain(a, b)


def _merge_chain(a: References, b: References) -> Optional[References]:  # type: ignore[return]
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


def _check_repeat(count: int) -> None:
    if count < 0:
        raise ValueError('expected nonnegative `count` but got {}'.format(count))


def _check_take(length: int, indices: numpy.ndarray) -> None:
    if not numeric.isintarray(indices):
        raise IndexError('expected an array of integers')
    if not indices.ndim == 1:
        raise IndexError('expected an array with dimension 1 but got {}'.format(indices.ndim))
    if len(indices) and not (0 <= indices.min() and indices.max() < length):
        raise IndexError('`indices` out of range')


def _check_compress(length: int, mask: numpy.ndarray) -> None:
    if not numeric.isboolarray(mask):
        raise IndexError('expected an array of booleans')
    if not mask.ndim == 1:
        raise IndexError('expected an array with dimension 1 but got {}'.format(mask.ndim))
    if len(mask) != length:
        raise IndexError('expected an array with length {} but got {}'.format(length, len(mask)))

# vim:sw=4:sts=4:et
