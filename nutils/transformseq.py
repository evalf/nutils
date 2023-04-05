"""The transformseq module."""

from typing import Tuple
from numbers import Integral
from . import types, numeric, _util as util, transform, element
from ._backports import cached_property
from .elementseq import References
from .transform import TransformChain
import abc
import itertools
import operator
import numpy


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
    todims : :class:`int`
        The dimension all transforms in this sequence map to.
    fromdims : :class:`int`
        The dimension all transforms in this sequence map from.

    Attributes
    ----------
    todims : :class:`int`
        The dimension all transforms in this sequence map to.
    fromdims : :class:`int`
        The dimension all transforms in this sequence map from.

    Notes
    -----
    Subclasses must implement :meth:`__getitem__`, :meth:`__len__` and
    :meth:`index_with_tail`.
    '''

    def __init__(self, todims: Integral, fromdims: Integral):
        assert isinstance(todims, Integral), f'todims={todims!r}'
        assert isinstance(fromdims, Integral), f'fromdims={fromdims!r}'
        if not 0 <= fromdims <= todims:
            raise ValueError('invalid dimensions')
        self.todims = todims
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
            return MaskedTransforms(self, types.arraydata(numpy.arange(index.start, index.stop, index.step)))
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
                return ReorderedTransforms(self[index[s]], types.arraydata(numpy.argsort(s)))
            if len(index) == 0:
                return EmptyTransforms(self.todims, self.fromdims)
            if len(index) == len(self):
                return self
            return MaskedTransforms(self, types.arraydata(index))
        elif numeric.isboolarray(index):
            if index.shape != (len(self),):
                raise IndexError('mask has invalid shape')
            if not numpy.any(index):
                return EmptyTransforms(self.todims, self.fromdims)
            if numpy.all(index):
                return self
            index, = numpy.where(index)
            return MaskedTransforms(self, types.arraydata(index))
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

        Consider the following plain sequence of two index transforms:

        >>> from nutils.transform import Index, SimplexChild
        >>> transforms = PlainTransforms(((Index(1, 0),), (Index(1, 1),)), 1, 1)

        Calling :meth:`index_with_tail` with the first transform gives index ``0``
        and no tail:

        >>> transforms.index_with_tail((Index(1, 0),))
        (0, ())

        Calling with an additional scale gives:

        >>> transforms.index_with_tail((Index(1, 0), SimplexChild(1, 0)))
        (0, (SimplexChild([0]+[.5]*x0),))
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

        Consider the following plain sequence of two index transforms:

        >>> from nutils.transform import Index, SimplexChild
        >>> transforms = PlainTransforms(((Index(1, 0),), (Index(1, 1),)), 1, 1)

        Calling :meth:`index` with the first transform gives index ``0``:

        >>> transforms.index((Index(1, 0),))
        0

        Calling with an additional scale raises an exception, because the transform
        is not present in ``transforms``.

        >>> transforms.index((Index(1, 0), SimplexChild(1, 0)))
        Traceback (most recent call last):
          ...
        ValueError: (Index(1, 0), SimplexChild([0]+[.5]*x0)) not in sequence of transforms
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
        return chain((self, other), self.todims, self.fromdims)

    def unchain(self):
        '''Iterator of unchained :class:`Transforms` items.

        Yields
        ------
        :class:`Transforms`
            Unchained items.
        '''

        yield self


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
        The sequence of transforms in canonical order.
    fromdims : :class:`int`
        The number of dimensions all ``transforms`` map from.
    '''

    def __init__(self, transforms: Tuple[Tuple[transform.TransformItem, ...], ...], todims: Integral, fromdims: Integral):
        assert isinstance(transforms, tuple) and all(isinstance(items, tuple) and all(isinstance(item, transform.TransformItem) for item in items) and transform.iscanonical(items) for items in transforms), f'transforms={transforms!r}'
        assert isinstance(todims, Integral), f'todims={todims!r}'
        assert isinstance(fromdims, Integral), f'fromdims={fromdims!r}'
        transforms_todims = set(trans[0].todims for trans in transforms)
        transforms_fromdims = set(trans[-1].fromdims for trans in transforms)
        if not (transforms_todims <= {todims}):
            raise ValueError('expected transforms with todims={}, but got {}'.format(todims, transforms_todims))
        if not (transforms_fromdims <= {fromdims}):
            raise ValueError('expected transforms with fromdims={}, but got {}'.format(fromdims, transforms_fromdims))
        self._transforms = transforms
        self._sorted = numpy.empty([len(self._transforms)], dtype=object)
        for i, trans in enumerate(self._transforms):
            self._sorted[i] = tuple(map(id, trans))
        self._indices = numpy.argsort(self._sorted)
        self._sorted = self._sorted[self._indices]
        super().__init__(todims, fromdims)

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


class IndexTransforms(Transforms):
    '''A sequence of :class:`nutils.transform.Index` singletons.

    Parameters
    ----------
    ndims : :class:`int`
        Dimension of the transformation.
    length : :class:`int`
        Length of the sequence.
    offset : :class:`int`
        The index of the first :class:`nutils.transform.Index` in this sequence.
    '''

    def __init__(self, ndims: Integral, length: Integral, offset: Integral = 0):
        assert isinstance(ndims, Integral), f'ndims={ndims!r}'
        assert isinstance(length, Integral), f'length={length!r}'
        assert isinstance(offset, Integral), f'offset={offset!r}'
        self._length = length
        self._offset = offset
        super().__init__(ndims, ndims)

    def __getitem__(self, index):
        if not numeric.isint(index):
            return super().__getitem__(index)
        return transform.Index(self.fromdims, self._offset + numeric.normdim(self._length, index.__index__())),

    def __len__(self):
        return self._length

    def index_with_tail(self, trans):
        root = trans[0]
        if root.fromdims == self.fromdims and isinstance(root, transform.Index) and 0 <= root.index - self._offset < self._length:
            return root.index - self._offset, trans[1:]
        raise ValueError


class Axis(types.Singleton):
    '''Base class for axes of :class:`~nutils.topology.StructuredTopology`.'''

    def __init__(self, i: Integral, j: Integral, mod: Integral):
        assert isinstance(i, Integral), f'i={i!r}'
        assert isinstance(j, Integral), f'j={j!r}'
        assert isinstance(mod, Integral), f'mod={mod!r}'
        assert i <= j
        self.i = i
        self.j = j
        self.mod = mod

    def __len__(self):
        return self.j - self.i

    def unmap(self, index):
        ielem = index - self.i
        if self.mod:
            ielem %= self.mod
        if not 0 <= ielem < len(self):
            raise ValueError
        return ielem

    def map(self, ielem):
        assert 0 <= ielem < len(self)
        index = self.i + ielem
        if self.mod:
            index %= self.mod
        return index


class DimAxis(Axis):

    isdim = True

    def __init__(self, i: Integral, j: Integral, mod: Integral, isperiodic: bool):
        assert isinstance(isperiodic, bool), f'isperiodic={isperiodic!r}'
        super().__init__(i, j, mod)
        self.isperiodic = isperiodic

    @property
    def refined(self):
        return DimAxis(self.i*2, self.j*2, self.mod*2, self.isperiodic)

    def opposite(self, ibound):
        return self

    def getitem(self, s):
        if not isinstance(s, slice):
            raise NotImplementedError
        if s == slice(None):
            return self
        start, stop, stride = s.indices(self.j - self.i)
        assert stride == 1
        assert stop > start
        return DimAxis(self.i+start, self.i+stop, mod=self.mod, isperiodic=False)

    def boundaries(self, ibound):
        if not self.isperiodic:
            yield IntAxis(self.i, self.i+1, self.mod, ibound, side=False)
            yield IntAxis(self.j-1, self.j, self.mod, ibound, side=True)

    def intaxis(self, ibound, side):
        return IntAxis(self.i-side+1-self.isperiodic, self.j-side, self.mod, ibound, side)


class IntAxis(Axis):

    isdim = False

    def __init__(self, i: Integral, j: Integral, mod: Integral, ibound: Integral, side: bool):
        assert isinstance(ibound, Integral), f'ibound={ibound!r}'
        assert isinstance(side, Integral), f'side={side!r}'
        super().__init__(i, j, mod)
        self.ibound = ibound
        self.side = side

    @property
    def refined(self):
        return IntAxis(self.i*2+self.side, self.j*2+self.side-1, self.mod*2, self.ibound, self.side)

    def opposite(self, ibound):
        return IntAxis(self.i+2*self.side-1, self.j+2*self.side-1, self.mod, self.ibound, not self.side) if ibound == self.ibound else self

    def boundaries(self, ibound):
        return ()


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

    def __init__(self, root: transform.TransformItem, axes: Tuple[Axis, ...], nrefine: Integral):
        assert isinstance(root, transform.TransformItem), f'root={root!r}'
        assert isinstance(axes, tuple) and all(isinstance(axis, Axis) for axis in axes), f'axes={axes!r}'
        assert isinstance(nrefine, Integral), f'nrefine={nrefine!r}'

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

        super().__init__(root.todims, sum(axis.isdim for axis in self._axes))

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
        trans0 = (transform.Index(len(self._axes), index) for index in indices)
        return (self._root, *trans0, *ctransforms, *self._etransforms)

    def __len__(self):
        return util.product(map(len, self._axes))

    def index_with_tail(self, trans):
        if len(trans) < 1 + len(self._axes) + self._nrefine + len(self._etransforms):
            raise ValueError

        root, indices, tail = trans[0], trans[1:1+len(self._axes)], transform.uppermost(trans[1+len(self._axes):])
        if root != self._root:
            raise ValueError

        if not all(isinstance(index, transform.Index) and index.todims == len(self._axes) for index in indices):
            raise ValueError
        indices = numpy.array([index.index for index in indices], dtype=int)

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

    def __init__(self, parent: Transforms, indices: types.arraydata):
        assert isinstance(parent, Transforms), f'parent={parent!r}'
        assert isinstance(indices, types.arraydata) and indices.dtype == int, f'indices={indices!r}'
        self._parent = parent
        self._indices = numpy.asarray(indices)
        super().__init__(parent.todims, parent.fromdims)

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

    def __init__(self, parent: Transforms, indices: types.arraydata):
        assert isinstance(parent, Transforms), f'parent={parent!r}'
        assert isinstance(indices, types.arraydata) and indices.dtype == int, f'indices={indices!r}'
        self._parent = parent
        self._indices = numpy.asarray(indices)
        super().__init__(parent.todims, parent.fromdims)

    @cached_property
    def _rindices(self):
        return types.frozenarray(numpy.argsort(self._indices), copy=False)

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

    def __init__(self, parent: Transforms, parent_references: References, derived_attribute: str, fromdims: Integral):
        assert isinstance(parent, Transforms), f'parent={parent!r}'
        assert isinstance(parent_references, References), f'parent_references={parent_references!r}'
        assert isinstance(derived_attribute, str), f'derived_attribute={derived_attribute!r}'
        assert isinstance(fromdims, Integral), f'fromdims={fromdims!r}'
        if len(parent) != len(parent_references):
            raise ValueError('`parent` and `parent_references` should have the same length')
        if parent.fromdims != parent_references.ndims:
            raise ValueError('`parent` and `parent_references` have different dimensions')
        self._parent = parent
        self._parent_references = parent_references
        self._derived_transforms = operator.attrgetter(derived_attribute)
        super().__init__(parent.todims, fromdims)

    @cached_property
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

    def __init__(self, parent: Transforms, parent_reference: element.Reference, derived_attribute: str, fromdims: Integral):
        assert isinstance(parent, Transforms), f'parent={parent!r}'
        assert isinstance(parent_reference, element.Reference), f'parent_reference={parent_reference!r}'
        assert isinstance(derived_attribute, str), f'derived_attribute={derived_attribute!r}'
        assert isinstance(fromdims, Integral), f'fromdims={fromdims!r}'
        if parent.fromdims != parent_reference.ndims:
            raise ValueError('`parent` and `parent_reference` have different dimensions')
        self._parent = parent
        self._derived_transforms = getattr(parent_reference, derived_attribute)
        super().__init__(parent.todims, fromdims)

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


class ChainedTransforms(Transforms):
    '''A sequence of chained :class:`Transforms` objects.

    Parameters
    ----------
    items: :class:`tuple` of :class:`Transforms` objects
        The :class:`Transforms` objects to chain.
    '''

    def __init__(self, items: Tuple[Transforms, ...]):
        assert isinstance(items, tuple) and all(isinstance(item, Transforms) for item in items), f'items={items!r}'
        if len(items) == 0:
            raise ValueError('Empty chain.')
        if len(set(item.todims for item in items)) != 1:
            raise ValueError('Cannot chain Transforms with different todims.')
        if len(set(item.fromdims for item in items)) != 1:
            raise ValueError('Cannot chain Transforms with different fromdims.')
        self._items = items
        super().__init__(self._items[0].todims, self._items[0].fromdims)

    @cached_property
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
                return EmptyTransforms(self.todims, self.fromdims)
            ostart = numpy.searchsorted(self._offsets, index.start, side='right') - 1
            ostop = numpy.searchsorted(self._offsets, index.stop, side='left')
            return chain((item[max(0, index.start-istart):min(istop-istart, index.stop-istart)] for item, (istart, istop) in zip(self._items[ostart:ostop], util.pairwise(self._offsets[ostart:ostop+1]))), self.todims, self.fromdims)
        elif numeric.isintarray(index) and index.ndim == 1 and len(index) and numpy.all(numpy.greater(numpy.diff(index), 0)):
            if index[0] < 0 or index[-1] >= len(self):
                raise IndexError('index out of bounds')
            split = numpy.searchsorted(index, self._offsets, side='left')
            return chain((item[index[start:stop]-offset] for item, offset, (start, stop) in zip(self._items, self._offsets, util.pairwise(split)) if stop > start), self.todims, self.fromdims)
        elif numeric.isboolarray(index) and index.shape == (len(self),):
            return chain((item[index[start:stop]] for item, (start, stop) in zip(self._items, util.pairwise(self._offsets))), self.todims, self.fromdims)
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
        return chain((item.refined(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.todims, self.fromdims)

    def edges(self, references):
        return chain((item.edges(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.todims, self.fromdims-1)

    def unchain(self):
        yield from self._items


def chain(items, todims, fromdims):
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
    items_todims = set(item.todims for item in unchained)
    if not (items_todims <= {todims}):
        raise ValueError('expected transforms with todims={}, but got {}'.format(todims, items_todims))
    items_fromdims = set(item.fromdims for item in unchained)
    if not (items_fromdims <= {fromdims}):
        raise ValueError('expected transforms with fromdims={}, but got {}'.format(fromdims, items_fromdims))
    if len(unchained) == 0:
        return EmptyTransforms(todims, fromdims)
    elif len(unchained) == 1:
        return unchained[0]
    else:
        return ChainedTransforms(unchained)


# vim:sw=4:sts=4:et
