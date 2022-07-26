"""
The topology module defines the topology objects, notably the
:class:`StructuredTopology`. Maintaining strict separation of topological and
geometrical information, the topology represents a set of elements and their
interconnectivity, boundaries, refinements, subtopologies etc, but not their
positioning in physical space. The dimension of the topology represents the
dimension of its elements, not that of the the space they are embedded in.

The primary role of topologies is to form a domain for :mod:`nutils.function`
objects, like the geometry function and function bases for analysis, as well as
provide tools for their construction. It also offers methods for integration and
sampling, thus providing a high level interface to operations otherwise written
out in element loops. For lower level operations topologies can be used as
:mod:`nutils.element` iterators.
"""

from . import element, function, evaluable, util, parallel, numeric, cache, transform, warnings, matrix, types, points, sparse
from .sample import Sample
from .element import Reference
from .elementseq import References
from .pointsseq import PointsSequence
from ._rust import CoordSystem, Simplex
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import typing
import numpy
import functools
import collections.abc
from collections import OrderedDict
import itertools
import functools
import operator
import numbers
import pathlib
import abc
import treelog as log
import os
_ = numpy.newaxis

_identity = lambda x: x
_strictspace = types.strictstr
_ArgDict = Mapping[str, numpy.ndarray]


class Topology:
    '''topology base class

    Parameters
    ----------
    ref_coord_system : :class:`dict` of :class:`str` to :class:`~nutils._rust.CoordSystem`
    references : :class:`nutils.elementseq.References`
        The references.

    Attributes
    ----------
    spaces : :class:`tuple` of :class:`str`
        The unique, ordered list of spaces on which this topology is defined.
    space_dims : :class:`tuple` of :class:`int`
        The dimension of each space in :attr:`spaces`.
    ref_coord_system : :class:`dict` of :class:`str` to :class:`~nutils._rust.CoordSystem`
    references : :class:`nutils.elementseq.References`
        The references.
    ndims : :class:`int`
        The dimension of this topology.
    '''

    __slots__ = 'ref_coord_system', 'spaces', 'space_dims', 'references', 'coord_system', 'opposite', 'ndims'

    @staticmethod
    def empty(ref_coord_system: typing.OrderedDict[str, CoordSystem], ndims: int) -> 'Topology':
        '''Return an empty topology.

        Parameters
        ----------
        ref_coord_system : :class:`collections.OrderedDict` of :class:`str` to :class:`~nutils._rust.CoordSystem`
        ndims : :class:`int`
            The dimension of the empty topology.

        Returns
        -------
        :class:`Topology`
            The empty topology.

        See Also
        --------
        :meth:`empty_like` : create an empty topology with spaces and dimension copied from another topology
        '''

        return _Empty(ref_coord_system, ndims)

    def empty_like(self) -> 'Topology':
        '''Return an empty topology with the same spaces and dimensions as this topology.

        Returns
        -------
        :class:`Topology`
            The empty topology.

        See Also
        --------
        :meth:`empty_like` : create an empty topology with custom spaces and dimension
        '''

        return Topology.empty(self.ref_coord_system, self.ndims)

    def disjoint_union(*topos: 'Topology') -> 'Topology':
        '''Return the union of the given disjoint topologies.

        Parameters
        ----------
        *topos : :class:`Topology`
            The disjoint parts.

        Returns
        -------
        :class:`Topology`
            The union.
        '''

        if len(topos) == 0:
            raise ValueError('Cannot take the disjoint union of zero topologies. \
                        Suggestion: include an empty topology (see \
                        `Topology.empty_like`).')
        empty = Topology.empty_like(topos[0])
        if not all(topo.spaces == empty.spaces and topo.space_dims == empty.space_dims and topo.ndims == empty.ndims for topo in topos):
            raise ValueError('The topologies must have the same spaces and dimensions.')
        unempty = tuple(filter(None, topos))
        if unempty:
            return functools.reduce(_DisjointUnion, unempty)
        else:
            return empty

    @staticmethod
    def line(space: str, nelems: int, bnames: Optional[Iterable[str]] = None, periodic: bool = False) -> 'Topology':
        if not isinstance(space, str):
            raise ValueError('argument `space`: expected a `str`')
        if not isinstance(nelems, int) or nelems < 0:
            raise ValueError('argument `nelems`: expected a non-negative `int`')
        if bnames is None:
            bnames = f'{space}-left', f'{space}-right'
        else:
            bnames = tuple(bnames)
            if len(bnames) != 2 or not all(isinstance(name, str) for name in bnames):
                raise ValueError('argument `bnames`: expected an iterable of two `str`')
        coord_system = CoordSystem(1, nelems)
        ref_coord_system = OrderedDict({space: coord_system})
        return _Line(ref_coord_system, coord_system, coord_system, bnames, bool(periodic))

    @staticmethod
    def simplex(space: str, simplices: numpy.array) -> 'Topology':
        if not isinstance(space, str):
            raise ValueError('argument `space`: expected a `str`')

        simplices = numpy.asarray(simplices)
        keep = numpy.zeros(simplices.max()+1, dtype=bool)
        keep[simplices.flat] = True
        simplices = types.arraydata(simplices if keep.all() else (numpy.cumsum(keep)-1)[simplices])

        coord_system = CoordSystem(simplices.shape[1] - 1, simplices.shape[0])
        ref_coord_system = OrderedDict({space: coord_system})
        return _SimplexTopology(ref_coord_system, simplices, coord_system, coord_system)

    def __init__(self, ref_coord_system: typing.OrderedDict[str, CoordSystem], references: References, coord_system: CoordSystem, opposite: CoordSystem):
        if not isinstance(ref_coord_system, OrderedDict):
            raise ValueError('argument `ref_coord_system`: expected an `collections.OrderedDict`')
        if not (references.ndims == coord_system.dim == opposite.dim and len(references) == len(coord_system) == len(opposite)):
            raise ValueError('`references`, `coord_system` and `opposite` have different dimensions')
        self.ref_coord_system = ref_coord_system
        self.spaces = tuple(ref_coord_system)
        self.space_dims = tuple(ref.dim for ref in ref_coord_system.values())
        self.references = references
        self.coord_system = coord_system
        self.opposite = opposite
        self.ndims = references.ndims
        super().__init__()

    def __str__(self) -> str:
        'string representation'

        return '{}(#{})'.format(self.__class__.__name__, len(self))

    def __len__(self) -> int:
        return len(self.references)

    def get_groups(self, *groups: str) -> 'Topology':
        '''Return the union of the given groups.

        Parameters
        ----------
        *groups : :class:`str`
            The identifiers of the groups.

        Returns
        -------
        :class:`Topology`
            The union of the given groups.
        '''

        return self.empty_like()

    def take(self, __indices: Union[numpy.ndarray, Sequence[int]]) -> 'Topology':
        '''Return the selected elements as a disconnected topology.

        The indices refer to the raveled list of elements in this topology. The
        indices are treated as a set: duplicate indices are silently ignored and
        the returned elements have the same order as in this topology.

        Parameters
        ----------
        indices : integer :class:`numpy.ndarray` or similar
            The one-dimensional array of element indices.

        Returns
        -------
        :class:`Topology`
            The selected elements.

        See Also
        --------
        :meth:`compress` : select elements using a mask
        '''

        indices = numpy.asarray(__indices)
        if indices.ndim != 1:
            raise ValueError('expected a one-dimensional array')
        if not indices.size:
            return self.empty_like()
        indices = numpy.unique(indices.astype(int, casting='same_kind'))
        if indices[0] < 0 or indices[-1] >= len(self):
            raise IndexError('element index out of range')
        return self.take_unchecked(indices)

    def take_unchecked(self, __indices: numpy.ndarray) -> 'Topology':
        return _Take(self, types.arraydata(__indices))

    def compress(self, __mask: Union[numpy.ndarray, Sequence[bool]]) -> 'Topology':
        '''Return the selected elements as a disconnected topology.

        The mask refers to the raveled list of elements in this topology.

        Parameters
        ----------
        mask : boolean :class:`numpy.ndarray` or similar
            The one-dimensional array of elements to select.

        Returns
        -------
        :class:`Topology`
            The selected elements.

        See Also
        --------
        :meth:`take` : select elements by index
        '''

        mask = numpy.asarray(__mask)
        if mask.ndim != 1:
            raise ValueError('expected a one-dimensional array')
        if len(mask) != len(self):
            raise ValueError('length of mask does not match number of elements')
        indices, = numpy.where(__mask)
        if len(indices):
            return self.take_unchecked(indices)
        else:
            return self.empty_like()

    def slice(self, __s: slice, __idim: int) -> 'Topology':
        '''Return a slice of the given dimension index.

        Parameters
        ----------
        s : :class:`slice`
            The slice.
        idim : :class:`int`
            The dimension index.

        Returns
        -------
        :class:`Topology`
            The slice.
        '''

        if not 0 <= __idim < self.ndims:
            raise IndexError('dimension index out of range')
        return self.slice_unchecked(__s, __idim)

    def slice_unchecked(self, __s: slice, __idim: int) -> 'Topology':
        raise ValueError('cannot slice an unstructured topology')

    def __getitem__(self, item: Any) -> 'Topology':
        if isinstance(item, str):
            topo = self.get_groups(*item.split(','))
        elif isinstance(item, Sequence) and all(isinstance(i, str) for i in item):
            topo = self.get_groups(*item) if item else self
        elif isinstance(item, slice):
            if item == slice(None):
                return self
            else:
                return self.slice(item, 0)
        elif isinstance(item, Sequence) and all(i == ... or isinstance(i, slice) for i in item):
            if ... in item:
                item = list(item)
                i = item.index(...)
                if ... in item[i+1:]:
                    raise ValueError('only one ellipsis is allowed')
                item[i:i+1] = [slice(None)] * max(0, self.ndims - len(item) + 1)
            if len(item) > self.ndims:
                raise ValueError('too many indices: topology is {}-dimension, but {} were indexed'.format(self.ndims, len(item)))
            topo = self
            for idim, indices in enumerate(item):
                if indices != slice(None):
                    topo = topo.slice(indices, idim)
            return topo
        elif numeric.isintarray(item) and item.ndim == 1 or isinstance(item, Sequence) and all(isinstance(i, int) for i in item):
            return self.take(item)
        else:
            raise NotImplementedError
        if not topo:
            raise KeyError(item)
        return topo

    def __invert__(self):
        return OppositeTopology(self)

    def __mul__(self, other: Any) -> 'Topology':
        if isinstance(other, Topology):
            left = self._disjoint_topos
            right = other._disjoint_topos
            empty = Topology.empty(OrderedDict(**self.ref_coord_system, **other.ref_coord_system), self.ndims + other.ndims)
            return Topology.disjoint_union(empty, *(_Mul(l, r) for l in left for r in right if len(l) and len(r)))
        else:
            return NotImplemented

    def __and__(self, other: Any) -> 'Topology':
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.ref_coord_system != other.ref_coord_system:
            raise ValueError('The topologies must have the same (order of) spaces and reference coordinate system.')
        elif not self or not other:
            return self.empty_like()
        else:
            raise NotImplementedError

    __rand__ = __and__

    def __or__(self, other: Any) -> 'Topology':
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.ref_coord_system != other.ref_coord_system:
            raise ValueError('The topologies must have the same (order of) spaces and dimensions.')
        elif not self:
            return other
        elif not other:
            return self
        else:
            return UnionTopology((self, other))

    __ror__ = __or__

    @property
    def refine_iter(self) -> 'Topology':
        topo = self
        while True:
            yield topo
            topo = topo.refined

    @property
    def _index_coords(self):
        index = function.transforms_index(self.coord_system, *self.ref_coord_system)
        coords = function.transforms_coords(self.coord_system, *self.ref_coord_system)
        return index, coords

    @property
    def f_index(self) -> function.Array:
        '''The evaluable index of the element in this topology.'''

        return self._index_coords[0]

    @property
    def f_coords(self) -> function.Array:
        '''The evaluable element local coordinates.'''

        return self._index_coords[1]

    def basis(self, name: str, *args, **kwargs) -> function.Basis:
        '''
        Create a basis.
        '''
        if self.ndims == 0:
            return function.PlainBasis([[1]], [[0]], 1, self.f_index, self.f_coords)
        split = name.split('-', 1)
        if len(split) == 2 and split[0] in ('h', 'th'):
            name = split[1]  # default to non-hierarchical bases
            if split[0] == 'th':
                kwargs.pop('truncation_tolerance', None)
        f = getattr(self, 'basis_' + name)
        return f(*args, **kwargs)

    def sample(self, ischeme: str, degree: int) -> Sample:
        'Create sample.'

        points = PointsSequence.from_iter((ischeme(reference, degree) for reference in self.references), self.ndims) if callable(ischeme) \
            else self.references.getpoints(ischeme, degree)
        coord_systems = self.coord_system,
        if len(self.coord_system) == 0 or self.opposite != self.coord_system:
            coord_systems += self.opposite,
        return Sample.new(self.ref_coord_system, coord_systems, points)

    @util.single_or_multiple
    def integrate_elementwise(self, funcs: Iterable[function.Array], *, degree: int, asfunction: bool = False, ischeme: str = 'gauss', arguments: Optional[_ArgDict] = None) -> Union[List[numpy.ndarray], List[function.Array]]:
        'element-wise integration'

        retvals = [sparse.toarray(retval) for retval in self.sample(ischeme, degree).integrate_sparse(
            [function.kronecker(func, pos=self.f_index, length=len(self), axis=0) for func in funcs], arguments=arguments)]
        if asfunction:
            return [function.get(retval, 0, self.f_index) for retval in retvals]
        else:
            return retvals

    @util.single_or_multiple
    def elem_mean(self, funcs: Iterable[function.Array], geometry: Optional[function.Array] = None, ischeme: str = 'gauss', degree: Optional[int] = None, **kwargs) -> List[numpy.ndarray]:
        ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
        funcs = (1,)+funcs
        if geometry is not None:
            funcs = [func * function.J(geometry) for func in funcs]
        area, *integrals = self.integrate_elementwise(funcs, ischeme=ischeme, degree=degree, **kwargs)
        return [integral / area[(slice(None),)+(_,)*(integral.ndim-1)] for integral in integrals]

    @util.single_or_multiple
    def integrate(self, funcs: Iterable[function.IntoArray], ischeme: str = 'gauss', degree: Optional[int] = None, edit=None, *, arguments: Optional[_ArgDict] = None) -> Tuple[numpy.ndarray, ...]:
        'integrate functions'

        ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
        if edit is not None:
            funcs = [edit(func) for func in funcs]
        return self.sample(ischeme, degree).integrate(funcs, **arguments or {})

    def integral(self, func: function.IntoArray, ischeme: str = 'gauss', degree: Optional[int] = None, edit=None) -> function.Array:
        'integral'

        ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
        if edit is not None:
            funcs = edit(func)
        return self.sample(ischeme, degree).integral(func)

    def projection(self, fun: function.Array, onto: function.Array, geometry: function.Array, **kwargs) -> function.Array:
        'project and return as function'

        return self.project(fun, onto, geometry, **kwargs) @ onto

    @log.withcontext
    def project(self, fun: function.Array, onto: function.Array, geometry: function.Array, ischeme: str = 'gauss', degree: Optional[int] = None, droptol: float = 1e-12, exact_boundaries: bool = False, constrain=None, verify=None, ptype='lsqr', edit=None, *, arguments: Optional[_ArgDict] = None, **solverargs) -> numpy.ndarray:
        'L2 projection of function onto function space'

        log.debug('projection type:', ptype)

        if degree is not None:
            ischeme += str(degree)
        if constrain is None:
            constrain = util.NanVec(onto.shape[0])
        else:
            constrain = constrain.copy()
        if exact_boundaries:
            constrain |= self.boundary.project(fun, onto, geometry, constrain=constrain, ischeme=ischeme, droptol=droptol, ptype=ptype, edit=edit, arguments=arguments)
        assert isinstance(constrain, util.NanVec)
        assert constrain.shape == onto.shape[:1]

        avg_error = None  # setting this depends on projection type

        if ptype == 'lsqr':
            assert ischeme is not None, 'please specify an integration scheme for lsqr-projection'
            fun2 = function.asarray(fun)**2
            if len(onto.shape) == 1:
                Afun = function.outer(onto)
                bfun = onto * fun
            elif len(onto.shape) == 2:
                Afun = function.outer(onto).sum(2)
                bfun = function.sum(onto * fun, -1)
                if fun2.ndim:
                    fun2 = fun2.sum(-1)
            else:
                raise Exception
            assert fun2.ndim == 0
            J = function.J(geometry)
            A, b, f2, area = self.integrate([Afun*J, bfun*J, fun2*J, J], ischeme=ischeme, edit=edit, arguments=arguments)
            N = A.rowsupp(droptol)
            if numpy.equal(b, 0).all():
                constrain[~constrain.where & N] = 0
                avg_error = 0.
            else:
                solvecons = constrain.copy()
                solvecons[~(constrain.where | N)] = 0
                u = A.solve(b, constrain=solvecons, **solverargs)
                constrain[N] = u[N]
                err2 = f2 - numpy.dot(2 * b - A @ u, u)  # can be negative ~zero due to rounding errors
                avg_error = numpy.sqrt(err2) / area if err2 > 0 else 0

        elif ptype == 'convolute':
            assert ischeme is not None, 'please specify an integration scheme for convolute-projection'
            if len(onto.shape) == 1:
                ufun = onto * fun
                afun = onto
            elif len(onto.shape) == 2:
                ufun = function.sum(onto * fun, axis=-1)
                afun = function.norm2(onto)
            else:
                raise Exception
            J = function.J(geometry)
            u, scale = self.integrate([ufun*J, afun*J], ischeme=ischeme, edit=edit, arguments=arguments)
            N = ~constrain.where & (scale > droptol)
            constrain[N] = u[N] / scale[N]

        elif ptype == 'nodal':
            bezier = self.sample('bezier', 2)
            W, F = bezier.integrate([onto, fun * onto])
            I = ~constrain.where
            constrain[I] = F[I] / W[I]

        else:
            raise Exception('invalid projection {!r}'.format(ptype))

        numcons = constrain.where.sum()
        info = 'constrained {}/{} dofs'.format(numcons, constrain.size)
        if avg_error is not None:
            info += ', error {:.2e}/area'.format(avg_error)
        log.info(info)
        if verify is not None:
            assert numcons == verify, 'number of constraints does not meet expectation: {} != {}'.format(numcons, verify)

        return constrain

    def refined_by(self, refine: Iterable[int]) -> 'Topology':
        'create refined space by refining dofs in existing one'

        return HierarchicalTopology(self, [numpy.arange(len(self))]).refined_by(refine)

    @property
    def refined(self) -> 'Topology':
        return self.refine_spaces(self.spaces)

    def refine(self, __arg: Union[int, Iterable[str], Mapping[str, int]]) -> 'Topology':
        '''Return the refined topology.

        If the argument is an :class:`int`, then this method behaves like
        :meth:`refine_count`. If the argument is a sequence of :class:`str`, then
        this method behaves like :meth:`refine_spaces`. If the argument is a
        dictionary of :class:`str` and :class:`int`, then this method behaves like
        :meth:`refine_spaces_count`.

        Returns
        -------
        :class:`Topology`
            The refined topology.

        See Also
        --------
        :meth:`refine_count` : refine a topology the given amount times
        :meth:`refine_spaces` : refine the given spaces of the topology
        :meth:`refine_spaces_count` : refine the given spaces the given amount times
        '''

        if isinstance(__arg, int):
            return self.refine_count(__arg)
        elif isinstance(__arg, Mapping):
            return self.refine_spaces_count(__arg)
        elif isinstance(__arg, Iterable):
            return self.refine_spaces(__arg)
        else:
            raise ValueError

    def refine_count(self, count: int) -> 'Topology':
        '''Return the topology refined `count` times.

        Parameters
        ----------
        count : :class:`int`
            The number of times to refine. `count` is allowed to be zero, in which
            case the original topology is returned, but not negative.

        Returns
        -------
        :class:`Topology`
            The refined topology.

        See Also
        --------
        :meth:`refine_spaces` : refine the given spaces of the topology
        :meth:`refine_spaces_count` : refine the given spaces the given amount times
        '''

        if count < 0:
            raise ValueError('Negative counts are invalid.')
        topo = self
        for i in range(count):
            topo = topo.refined
        return topo

    def refine_spaces(self, __spaces: Iterable[str]) -> 'Topology':
        '''Return the topology with the given spaces refined once.

        Parameters
        ----------
        spaces : iterable of :class:`str`
            The spaces to refine. It is an error to specify spaces that do not
            exist in this topology. It is allowed to specify no spaces, in which
            case the original topology is returned.

        Returns
        -------
        :class:`Topology`
            The refined topology.

        See Also
        --------
        :meth:`refine_count` : refine a topology the given amount times
        :meth:`refine_spaces_count` : refine the given spaces the given amount times
        '''

        spaces = frozenset(__spaces)
        for space in sorted(spaces):
            if space not in self.spaces:
                raise ValueError('This topology does not have space {}.'.format(space))
        return self.refine_spaces_unchecked(spaces)

    def refine_spaces_unchecked(self, __spaces: FrozenSet[str]) -> 'Topology':
        '''Return the topology with the given spaces refined once.

        Parameters
        ----------
        spaces : iterable of :class:`str`
            The spaces to refine. It is an error to specify spaces that do not
            exist in this topology. It is allowed to specify no spaces, in which
            case the original topology is returned.

        Returns
        -------
        :class:`Topology`
            The refined topology.

        Notes
        -----
        This method does not check the validity of the arguments. Use
        :meth:`refine_spaces` instead unless you're absolutely sure what you are
        doing.
        '''

        return RefinedTopology(self)

    def refine_spaces_count(self, count: Mapping[str, int]) -> 'Topology':
        '''Return the topology with the given spaces refined the given amount times.

        Parameters
        ----------
        spaces : mapping of :class:`str` to :class:`int`
            The spaces to refine together with the count. It is an error to specify
            spaces that do not exist in this topology. It is allowed to specify no
            spaces, in which case the original topology is returned.

        Returns
        -------
        :class:`Topology`
            The refined topology.

        See Also
        --------
        :meth:`refine_count` : refine a topology the given amount times
        :meth:`refine_spaces` : refine the given spaces of the topology
        '''

        if not all(n >= 0 for n in count.values()):
            raise ValueError('Negative counts are invalid.')
        topo = self
        for i in itertools.count():
            spaces = tuple(space for space, n in count.items() if n > i)
            if not spaces:
                break
            topo = topo.refine_spaces(spaces)
        return topo

    def trim(self, levelset: function.Array, maxrefine: int, ndivisions: int = 8, name: str = 'trimmed', leveltopo: Optional['Topology'] = None, *, arguments: Optional[_ArgDict] = None) -> 'Topology':
        'trim element along levelset'

        raise NotImplementedError

        if arguments is None:
            arguments = {}

        refs = []
        if leveltopo is None:
            ielem_arg = evaluable.Argument('_trim_index', (), dtype=int)
            coordinates = self.references.getpoints('vertex', maxrefine).get_evaluable_coords(ielem_arg)
            levelset = levelset.lower(function.LowerArgs.for_space(self.space, (self.transforms, self.opposites), ielem_arg, coordinates)).optimized_for_numpy
            with log.iter.percentage('trimming', range(len(self)), self.references) as items:
                for ielem, ref in items:
                    levels = levelset.eval(_trim_index=ielem, **arguments)
                    refs.append(ref.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
        else:
            log.info('collecting leveltopo elements')
            coordinates = evaluable.Points(evaluable.NPoints(), self.ndims)
            ielem = evaluable.Argument('_leveltopo_ielem', (), int)
            levelset = levelset.lower(function.LowerArgs.for_space(self.space, (leveltopo.transforms, leveltopo.opposites), ielem, coordinates)).optimized_for_numpy
            bins = [set() for ielem in range(len(self))]
            for trans in leveltopo.transforms:
                ielem, tail = self.transforms.index_with_tail(trans)
                bins[ielem].add(tail)
            fcache = cache.WrapperCache()
            with log.iter.percentage('trimming', self.references, self.transforms, bins) as items:
                for ref, trans, ctransforms in items:
                    levels = numpy.empty(ref.nvertices_by_level(maxrefine))
                    cover = list(fcache[ref.vertex_cover](frozenset(ctransforms), maxrefine))
                    # confirm cover and greedily optimize order
                    mask = numpy.ones(len(levels), dtype=bool)
                    while mask.any():
                        imax = numpy.argmax([mask[indices].sum() for tail, points, indices in cover])
                        tail, points, indices = cover.pop(imax)
                        ielem = leveltopo.transforms.index(trans + tail)
                        levels[indices] = levelset.eval(_leveltopo_ielem=ielem, _points=points, **arguments)
                        mask[indices] = False
                    refs.append(ref.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
            log.debug('cache', fcache.stats)
        return SubsetTopology(self, refs, newboundary=name)

    def subset(self, topo: 'Topology', newboundary: Optional[Union[str, 'Topology']] = None, strict: bool = False) -> 'Topology':
        'intersection'

        raise NotImplementedError

        refs = [ref.empty for ref in self.references]
        for ref, trans in zip(topo.references, topo.transforms):
            try:
                ielem = self.transforms.index(trans)
            except ValueError:
                assert not strict, 'elements do not form a strict subset'
            else:
                subref = self.references[ielem] & ref
                if strict:
                    assert subref == ref, 'elements do not form a strict subset'
                refs[ielem] = subref
        if not any(refs):
            return self.empty_like()
        return SubsetTopology(self, refs, newboundary)

    def withgroups(self, vgroups: Mapping[str, Union[str, 'Topology']] = {}, bgroups: Mapping[str, Union[str, 'Topology']] = {}, igroups: Mapping[str, Union[str, 'Topology']] = {}, pgroups: Mapping[str, Union[str, 'Topology']] = {}) -> 'Topology':
        return _WithGroupsTopology(self, vgroups, bgroups, igroups, pgroups) if vgroups or bgroups or igroups or pgroups else self

    def withsubdomain(self, **kwargs: 'Topology') -> 'Topology':
        return self.withgroups(vgroups=kwargs)

    def withboundary(self, **kwargs: 'Topology') -> 'Topology':
        return self.withgroups(bgroups=kwargs)

    def withinterfaces(self, **kwargs: 'Topology') -> 'Topology':
        return self.withgroups(igroups=kwargs)

    def withpoints(self, **kwargs: 'Topology') -> 'Topology':
        return self.withgroups(pgroups=kwargs)

    @log.withcontext
    def volume(self, geometry: function.Array, ischeme: str = 'gauss', degree: int = 1, *, arguments: Optional[_ArgDict] = None) -> numpy.ndarray:
        return self.integrate(function.J(geometry), ischeme=ischeme, degree=degree, arguments=arguments)

    @log.withcontext
    def check_boundary(self, geometry: function.Array, elemwise: bool = False, ischeme: str = 'gauss', degree: int = 1, tol: float = 1e-15, print=print, *, arguments: Optional[_ArgDict] = None) -> None:
        if elemwise:
            for ref in self.references:
                ref.check_edges(tol=tol, print=print)
        volume = self.volume(geometry, ischeme=ischeme, degree=degree, arguments=arguments)
        J = function.J(geometry)
        zeros, volumes = self.boundary.integrate([geometry.normal()*J, geometry*geometry.normal()*J], ischeme=ischeme, degree=degree, arguments=arguments)
        if numpy.greater(abs(zeros), tol).any():
            print('divergence check failed: {} != 0'.format(zeros))
        if numpy.greater(abs(volumes - volume), tol).any():
            print('divergence check failed: {} != {}'.format(volumes, volume))

    def indicator(self, subtopo):
        if isinstance(subtopo, str):
            subtopo = self.get_groups(*subtopo.split(','))
        missing = frozenset(subtopo.spaces) - frozenset(self.spaces)
        if missing:
            raise ValueError('The following spaces of the sub topology are not present in the super topology: {}'.format(','.join(missing)))

        sub_coord_system = subtopo.coord_system
        for i in range(len(self.spaces) - len(subtopo.spaces) + 1):
            if self.spaces[i:i + len(subtopo.spaces)] == subtopo.spaces:
                break
        else:
            raise ValueError('Cannot create an indicator for a sub topology defined on a non-contiguous subset of spaces of the super topology.')
        for space in reversed(self.spaces[:i]):
            sub_coord_system = self.ref_coord_system[space] * sub_coord_system
        for space in self.spaces[i + len(subtopo.spaces):]:
            coord_system *= self.ref_coord_system[space]

        trans = sub_coord_system.trans_to(self.coord_system)
        values = numpy.zeros([len(self)], dtype=int)
        values[numpy.unique(trans.apply_indices(numpy.arange(len(subtopo))))] = 1
        return function.get(values, 0, self.f_index)

    def select(self, indicator: function.Array, ischeme: str = 'bezier2', **kwargs: numpy.ndarray) -> 'Topology':
        # Select elements where `indicator` is strict positive at any of the
        # integration points defined by `ischeme`. We sample `indicator > 0`
        # together with the element index (`self.f_index`) and keep all indices
        # with at least one positive result.
        sample = self.sample(*element.parse_legacy_ischeme(ischeme))
        isactive, ielem = sample.eval([indicator > 0, self.f_index], **kwargs)
        selected = types.frozenarray(numpy.unique(ielem[isactive]))
        return self[selected]

    def locate(self, geom, coords, *, tol=0, eps=0, maxiter=0, arguments=None, weights=None, maxdist=None, ischeme=None, scale=None, skip_missing=False) -> Sample:
        '''Create a sample based on physical coordinates.

        In a finite element application, functions are commonly evaluated in points
        that are defined on the topology. The reverse, finding a point on the
        topology based on a function value, is often a nonlinear process and as
        such involves Newton iterations. The ``locate`` function facilitates this
        search process and produces a :class:`nutils.sample.Sample` instance that
        can be used for the subsequent evaluation of any function in the given
        physical points.

        Example:

        >>> from . import mesh
        >>> domain, geom = mesh.unitsquare(nelems=3, etype='mixed')
        >>> sample = domain.locate(geom, [[.9, .4]], tol=1e-12)
        >>> sample.eval(geom).round(5).tolist()
        [[0.9, 0.4]]

        Locate requires a geometry function, an array of coordinates, and at least
        one of ``tol`` and ``eps`` to set the tolerance in physical of element
        space, respectively; if both are specified the least restrictive takes
        precedence.

        Args
        ----
        geom : 1-dimensional :class:`nutils.function.Array`
            Geometry function of length ``ndims``.
        coords : 2-dimensional :class:`float` array
            Array of coordinates with ``ndims`` columns.
        tol : :class:`float` (default: 0)
            Maximum allowed distance in physical coordinates between target and
            located point.
        eps : :class:`float` (default: 0)
            Maximum allowed distance in element coordinates between target and
            located point.
        maxiter : :class:`int` (default: 0)
            Maximum allowed number of Newton iterations, or 0 for unlimited.
        arguments : :class:`dict` (default: None)
            Arguments for function evaluation.
        weights : :class:`float` array (default: None)
            Optional weights, in case ``coords`` are quadrature points, making the
            resulting sample suitable for integration.
        maxdist : :class:`float` (default: None)
            Speed up failure by setting a physical distance between point and
            element centroid above which the element is rejected immediately. If
            all points are expected to be located then this can safely be left
            unspecified.
        skip_missing : :class:`bool` (default: False)
            When set to true, skip points that are not found (for instance because
            they fall outside the domain) in the returned sample. When set to false
            (the default) missing points raise a ``LocateError``.

        Returns
        -------
        located : :class:`nutils.sample.Sample`
        '''

        if ischeme is not None:
            warnings.deprecation('the ischeme argument is deprecated and will be removed in future')
        if scale is not None:
            warnings.deprecation('the scale argument is deprecated and will be removed in future')
        if max(tol, eps) <= 0:
            raise ValueError('locate requires either tol or eps to be strictly positive')
        coords = numpy.asarray(coords, dtype=float)
        if geom.ndim == 0:
            geom = geom[_]
            coords = coords[..., _]
        if not geom.shape == coords.shape[1:] == (self.ndims,):
            raise ValueError('invalid geometry or point shape for {}D topology'.format(self.ndims))
        arguments = dict(arguments or ())
        centroids = self.sample('_centroid', None).eval(geom, **arguments)
        assert len(centroids) == len(self)
        ielems = parallel.shempty(len(coords), dtype=int)
        points = parallel.shempty((len(coords), len(geom)), dtype=float)
        _ielem = evaluable.InRange(evaluable.Argument('_locate_ielem', shape=(), dtype=int), len(self))
        _point = evaluable.Argument('_locate_point', shape=(self.ndims,))
        lower_args = function.Bound(self.ref_coord_system, (self.coord_system, self.opposite), _ielem, _point).into_lower_args()
        egeom = geom.lower(lower_args)
        xJ = evaluable.Tuple((egeom, evaluable.derivative(egeom, _point))).simplified
        if skip_missing:
            if weights is not None:
                raise ValueError('weights and skip_missing are mutually exclusive')
            missing = parallel.shzeros(len(coords), dtype=bool)
        with parallel.ctxrange('locating', len(coords)) as ipoints:
            for ipoint in ipoints:
                xt = coords[ipoint]  # target
                dist = numpy.linalg.norm(centroids - xt, axis=1)
                for ielem in numpy.argsort(dist) if maxdist is None \
                        else sorted((dist < maxdist).nonzero()[0], key=dist.__getitem__):
                    ref = self.references[ielem]
                    arguments['_locate_ielem'] = ielem
                    arguments['_locate_point'] = p = numpy.array(ref.centroid)
                    ex = ep = numpy.inf
                    iiter = 0
                    while ex > tol and ep > eps:  # newton loop
                        if iiter > maxiter > 0:
                            break  # maximum number of iterations reached
                        iiter += 1
                        xp, Jp = xJ.eval(**arguments)
                        dx = xt - xp
                        ex0 = ex
                        ex = numpy.linalg.norm(dx)
                        if ex >= ex0:
                            break  # newton is diverging
                        try:
                            dp = numpy.linalg.solve(Jp, dx)
                        except numpy.linalg.LinAlgError:
                            break  # jacobian is singular
                        ep = numpy.linalg.norm(dp)
                        p += dp  # NOTE: modifies arguments['_locate_point'] in place
                    else:
                        if ref.inside(p, max(eps, ep)):
                            ielems[ipoint] = ielem
                            points[ipoint] = p
                            break
                else:
                    if skip_missing:
                        missing[ipoint] = True
                    else:
                        raise LocateError('failed to locate point: {}'.format(xt))
        if skip_missing:
            ielems = ielems[~missing]
            points = points[~missing]
        return self._sample(ielems, points, weights)

    def _sample(self, ielems, coords, weights=None):
        index = numpy.argsort(ielems, kind='stable')
        sorted_ielems = ielems[index]
        offsets = [0, *(sorted_ielems[:-1] != sorted_ielems[1:]).nonzero()[0]+1, len(index)]

        unique_ielems = sorted_ielems[offsets[:-1]]
        coord_systems = self.coord_system.take(unique_ielems),
        if len(self.coord_system) == 0 or self.opposite != self.coord_system:
            coord_systems += self.opposite.take(unique_ielems),

        slices = [index[n:m] for n, m in zip(offsets[:-1], offsets[1:])]
        points_ = PointsSequence.from_iter([points.CoordsPoints(coords[s]) for s in slices] if weights is None
                                           else [points.CoordsWeightsPoints(coords[s], weights[s]) for s in slices], self.ndims)

        return Sample.new(self.ref_coord_system, coord_systems, points_, index)

    @property
    def boundary(self) -> 'Topology':
        '''
        :class:`Topology`:
          The boundary of this topology.
        '''

        return self.boundary_spaces(self.spaces)

    def boundary_spaces(self, __spaces: Iterable[str]) -> 'Topology':
        '''Return the boundary in the given spaces.

        Parameters
        ----------
        spaces : iterable of :class:`str`
            Nonstrict subset of :attr:`spaces`. Duplicates are silently ignored.

        Returns
        -------
        :class:`Topology`
            The boundary in the given spaces.

        Raises
        ------
        :class:`ValueError`
            If the topology is 0D or the set of spaces is empty or not a subset of :attr:`spaces`.
        '''

        spaces = frozenset(__spaces)
        for space in sorted(spaces):
            if space not in self.spaces:
                raise ValueError('This topology does not have space {}.'.format(space))
        if self.ndims == 0 or sum(self.space_dims[self.spaces.index(space)] for space in spaces) == 0:
            raise ValueError('A 0D topology has no boundary.')
        return self.boundary_spaces_unchecked(spaces)

    def boundary_spaces_unchecked(self, __spaces: FrozenSet[str]) -> 'Topology':
        '''Return the boundary in the given spaces.

        The topology must be at least one-dimensional.

        Parameters
        ----------
        spaces : :class:`frozenset` of :class:`str`
            Unempty, nonstrict subset of :attr:`spaces`.

        Returns
        -------
        :class:`Topology`
            The boundary in the given spaces.

        Notes
        -----
        This method does not check the validity of the arguments or the dimension
        of the topology. Use :meth:`boundary_spaces` instead unless you're
        absolutely sure what you are doing.
        '''

        raise NotImplementedError

    @property
    def interfaces(self) -> 'Topology':
        return self.interfaces_spaces(self.spaces)

    def interfaces_spaces(self, __spaces: Iterable[str]) -> 'Topology':
        '''Return the interfaces in the given spaces.

        Parameters
        ----------
        spaces : iterable of :class:`str`
            Nonstrict subset of :attr:`spaces`. Duplicates are silently ignored.

        Returns
        -------
        :class:`Topology`
            The interfaces in the given spaces.

        Raises
        ------
        :class:`ValueError`
            If the topology is 0D or the set of spaces is empty or not a subset of :attr:`spaces`.
        '''

        spaces = frozenset(__spaces)
        for space in sorted(spaces):
            if space not in self.spaces:
                raise ValueError('This topology does not have space {}.'.format(space))
        if self.ndims == 0 or sum(self.space_dims[self.spaces.index(space)] for space in spaces) == 0:
            raise ValueError('A 0D topology has no interfaces.')
        return self.interfaces_spaces_unchecked(spaces)

    def interfaces_spaces_unchecked(self, __spaces: FrozenSet[str]) -> 'Topology':
        '''Return the interfaces in the given spaces.

        The topology must be at least one-dimensional.

        Parameters
        ----------
        spaces : :class:`frozenset` of :class:`str`
            Unempty, nonstrict subset of :attr:`spaces`.

        Returns
        -------
        :class:`Topology`
            The interfaces in the given spaces.

        Notes
        -----
        This method does not check the validity of the arguments or the dimension
        of the topology. Use :meth:`interfaces_spaces` instead unless you're
        absolutely sure what you are doing.
        '''

        raise NotImplementedError

    def basis_discont(self, degree: int) -> function.Basis:
        'discontinuous shape functions'

        assert numeric.isint(degree) and degree >= 0
        if self.references.isuniform:
            coeffs = [self.references[0].get_poly_coeffs('bernstein', degree=degree)]*len(self.references)
        else:
            coeffs = [ref.get_poly_coeffs('bernstein', degree=degree) for ref in self.references]
        return function.DiscontBasis(coeffs, self.f_index, self.f_coords)

    @property
    def _disjoint_topos(self):
        return self,


class _Empty(Topology):

    def __init__(self, ref_coord_system: typing.OrderedDict[str, CoordSystem], ndims: int) -> None:
        super().__init__(ref_coord_system, References.empty(ndims), CoordSystem(ndims, 0), CoordSystem(ndims, 0))

    def __invert__(self) -> Topology:
        return self

    @property
    def connectivity(self) -> Sequence[Sequence[int]]:
        return tuple()

    def indicator(self, subtopo: Union[str, Topology]) -> Topology:
        return function.zeros((), int)

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return self

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _Empty(self.ref_coord_system, self.ndims - 1)

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _Empty(self.ref_coord_system, self.ndims - 1)

    def basis_std(self, degree: int, *args, **kwargs) -> function.Array:
        return function.zeros((0,))

    basis_spline = basis_std

    def sample(self, ischeme: str, degree: int) -> Sample:
        return Sample.empty(self.ref_coord_system, self.ndims)


class _WithName(Topology):

    def __init__(self, topo: Topology, name: str):
        self.topo = topo
        self.name = name
        super().__init__(topo.ref_coord_system, topo.references, topo.coord_system, topo.opposite)

    def get_groups(self, *groups: str) -> Topology:
        if self.name in groups:
            return self.topo
        else:
            return self.topo.get_groups(*groups)

    def take_unchecked(self, __indices: numpy.ndarray) -> Topology:
        return _WithName(self.topo.take_unchecked(__indices), self.name)

    def slice_unchecked(self, __s: slice, __idim: int) -> 'Topology':
        return _WithName(self.topo.slice_unchecked(__s, __idim), self.name)

    def __invert__(self):
        return _WithName(~self.topo, self.name)

    @property
    def f_index(self) -> function.Array:
        return self.topo.f_index

    @property
    def f_coords(self) -> function.Array:
        return self.topo.f_coords

    def basis(self, *args, **kwargs) -> function.Basis:
        return self.topo.basis(*args, **kwargs)

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.topo.sample(ischeme, degree)

    def refine_spaces_unchecked(self, __spaces: FrozenSet[str]) -> Topology:
        return _WithName(self.topo.refine_spaces_unchecked(__spaces), self.name)

    def trim(self, *args, **kwargs) -> Topology:
        return _WithName(self.topo.trim(*args, **kwargs), self.name)

    def subset(self, *args, **kwargs) -> Topology:
        return _WithName(self.topo.subset(*args, **kwargs), self.name)

    def indicator(self, subtopo):
        if isinstance(subtopo, str) and self.name in subtopo.split(','):
            return function.ones(())
        else:
            return self.topo.indicator(subtopo)

    def select(self, *args, **kwargs) -> Topology:
        return _WithName(self.topo.select(*args, **kwargs), self.name)

    def locate(self, *args, **kwargs):
        return self.topo.locate(*args, **kwargs)

    def boundary_spaces_unchecked(self, __spaces: FrozenSet[str]) -> Topology:
        return self.topo.boundary_spaces_unchecked(__spaces)

    def interfaces_spaces_unchecked(self, __spaces: FrozenSet[str]) -> 'Topology':
        return self.topo.interfaces_spaces_unchecked(__spaces)

    @property
    def _disjoint_topos(self):
        return tuple(_WithName(part, self.name) for part in self.topo._disjoint_topos)


class _DisjointUnion(Topology):

    def __init__(self, topo1: Topology, topo2: Topology) -> None:
        if topo1.ref_coord_system != topo2.ref_coord_system:
            raise ValueError('The topologies must have the same (order of) spaces and reference coordinate system.')
        self.topo1 = topo1
        self.topo2 = topo2
        references = topo1.references + topo2.references
        coord_system = topo1.coord_system.concat(topo2.coord_system)
        opposite = topo1.opposite.concat(topo2.opposite)
        super().__init__(topo1.ref_coord_system, references, coord_system, opposite)

    def __invert__(self) -> Topology:
        return Topology.disjoint_union(~self.topo1, ~self.topo2)

    def __and__(self, other: Any) -> Topology:
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.ref_coord_system != other.ref_coord_system:
            raise ValueError('The topologies must have the same (order of) spaces and reference coordinate system.')
        else:
            return Topology.disjoint_union(self.topo1 & other, self.topo2 & other)

    __rand__ = __and__

    @property
    def connectivity(self) -> Sequence[Sequence[int]]:
        o = len(self.topo1)
        return tuple(self.topo1.connectivity) + tuple(tuple(-1 if n < 0 else n + o for n in N) for N in self.topo2.connectivity)

    def get_groups(self, *groups: str) -> Topology:
        return Topology.disjoint_union(self.topo1.get_groups(*groups), self.topo2.get_groups(*groups))

    def take_unchecked(self, __indices: numpy.ndarray) -> Topology:
        nelems1 = len(self.topo1)
        split = numpy.searchsorted(__indices, nelems1)
        topo1 = self.topo1.take_unchecked(__indices[:split]) if split else self.topo1.empty_like()
        topo2 = self.topo2.take_unchecked(__indices[split:] - nelems1) if split < len(__indices) else self.topo2.empty_like()
        return Topology.disjoint_union(topo1, topo2)

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return Topology.disjoint_union(self.topo1.refine_spaces(spaces), self.topo2.refine_spaces(spaces))

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return Topology.disjoint_union(self.topo1.boundary_spaces(spaces), self.topo2.boundary_spaces(spaces))

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return Topology.disjoint_union(self.topo1.interfaces_spaces(spaces), self.topo2.interfaces_spaces(spaces))

    @util.single_or_multiple
    def integrate_elementwise(self, funcs: Iterable[function.Array], *, degree: int, asfunction: bool = False, ischeme: str = 'gauss', arguments: Optional[_ArgDict] = None) -> Union[List[numpy.ndarray], List[function.Array]]:
        return list(map(numpy.concatenate, zip(*(topo.integrate_elementwise(funcs, degree=degree, ischeme=ischeme, arguments=arguments) for topo in (self.topo1, self.topo2)))))

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.topo1.sample(ischeme, degree) + self.topo2.sample(ischeme, degree)

    def trim(self, levelset: function.Array, maxrefine: int, ndivisions: int = 8, name: str = 'trimmed', leveltopo: Optional[Topology] = None, *, arguments: Optional[_ArgDict] = None) -> Topology:
        if leveltopo is not None:
            # TODO
            return super().trim(levelset, maxrefine, ndivisions, name, leveltopo, arguments=arguments)
        else:
            topo1 = self.topo1.trim(levelset, maxrefine, ndivisions, name, arguments=arguments)
            topo2 = self.topo2.trim(levelset, maxrefine, ndivisions, name, arguments=arguments)
            return Topology.disjoint_union(topo1, topo2)

    def select(self, indicator: function.Array, ischeme: str = 'bezier2', **kwargs: numpy.ndarray) -> Topology:
        topo1 = self.topo1.select(indicator, ischeme, **kwargs)
        topo2 = self.topo2.select(indicator, ischeme, **kwargs)
        return Topology.disjoint_union(topo1, topo2)

    @property
    def _disjoint_topos(self):
        return self.topo1._disjoint_topos + self.topo2._disjoint_topos


class _Mul(Topology):

    def __init__(self, topo1: Topology, topo2: Topology) -> None:
        if not set(topo1.spaces).isdisjoint(topo2.spaces):
            raise ValueError('Cannot multiply two topologies (partially) defined on the same spaces.')
        self.topo1 = topo1
        self.topo2 = topo2
        ref_coord_system = OrderedDict(**topo1.ref_coord_system, **topo2.ref_coord_system)
        references = topo1.references * topo2.references
        coord_system = topo1.coord_system * topo2.coord_system
        opposite = topo1.opposite * topo2.opposite
        super().__init__(ref_coord_system, references, coord_system, opposite)

    def __invert__(self) -> Topology:
        return ~self.topo1 * ~self.topo2

    @property
    def f_index(self) -> function.Array:
        return self.topo1.f_index * len(self.topo2) + self.topo2.f_index

    @property
    def f_coords(self) -> function.Array:
        return numpy.concatenate([self.topo1.f_coords, self.topo2.f_coords])

    @property
    def connectivity(self) -> Sequence[Sequence[int]]:
        connectivity1 = self.topo1.connectivity
        connectivity2 = self.topo2.connectivity
        s = len(self.topo2)
        return tuple(tuple(-1 if n1 < 0 else n1 * s + i2 for n1 in N1) + tuple(-1 if n2 < 0 else i1 * s + n2 for n2 in N2) for i1, N1 in enumerate(connectivity1) for i2, N2 in enumerate(connectivity2))

    def get_groups(self, *groups: str) -> Topology:
        subtopo1 = self.topo1.get_groups(*groups)
        subtopo2 = self.topo2.get_groups(*groups)
        if subtopo1 and subtopo2:
            raise NotImplementedError
        elif subtopo1:
            return subtopo1 * self.topo2
        elif subtopo2:
            return self.topo1 * subtopo2
        else:
            return self.empty_like()

    def slice_unchecked(self, indices: slice, idim: int) -> Topology:
        if idim < self.topo1.ndims:
            return self.topo1.slice_unchecked(indices, idim) * self.topo2
        else:
            return self.topo1 * self.topo2.slice_unchecked(indices, idim - self.topo1.ndims)

    def indicator(self, subtopo: Union[str, Topology]) -> Topology:
        if isinstance(subtopo, str):
            subtopo = self.get_groups(*subtopo.split(','))
        missing = frozenset(subtopo.spaces) - frozenset(self.spaces)
        if missing:
            raise ValueError('The following spaces of the sub topology are not present in the super topology: {}'.format(','.join(missing)))

        if frozenset(subtopo.spaces) <= frozenset(self.topo1.spaces):
            return self.topo1.indicator(subtopo)
        elif frozenset(subtopo.spaces) <= frozenset(self.topo2.spaces):
            return self.topo2.indicator(subtopo)
        else:
            return super().indicator(subtopo)

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return self.topo1.refine_spaces(spaces & frozenset(self.topo1.spaces)) * self.topo2.refine_spaces(spaces & frozenset(self.topo2.spaces))

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        spaces1 = spaces & frozenset(self.topo1.spaces)
        spaces2 = spaces & frozenset(self.topo2.spaces)
        boundaries = []
        if self.topo2.ndims and spaces2 or not spaces1:
            boundaries.append(self.topo1 * self.topo2.boundary_spaces(spaces2))
        if self.topo1.ndims and spaces1 or not spaces2:
            boundaries.append(self.topo1.boundary_spaces(spaces1) * self.topo2)
        return Topology.disjoint_union(*boundaries)

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        spaces1 = spaces & frozenset(self.topo1.spaces)
        spaces2 = spaces & frozenset(self.topo2.spaces)
        interfaces = []
        if self.topo2.ndims and spaces2 or not spaces1:
            interfaces.append(self.topo1 * self.topo2.interfaces_spaces(spaces2))
        if self.topo1.ndims and spaces1 or not spaces2:
            interfaces.append(self.topo1.interfaces_spaces(spaces1) * self.topo2)
        return Topology.disjoint_union(*interfaces)

    def basis(self, name: str, degree: Union[int, Sequence[int]], **kwargs) -> function.Basis:
        kwargs['degree'] = degree
        kwargs1 = {}
        kwargs2 = {}

        for attr in 'degree', 'continuity':
            val = kwargs.pop(attr, None)
            if val is None:
                pass
            elif isinstance(val, int):
                kwargs1[attr] = kwargs2[attr] = val
            elif isinstance(val, Sequence) and all(isinstance(v, int) for v in val):
                if len(val) != self.ndims:
                    raise ValueError('argument `{}` must have length {} but got {}'.format(attr, self.ndims, len(val)))
                kwargs1[attr] = val[:self.topo1.ndims]
                kwargs2[attr] = val[self.topo1.ndims:]
            else:
                raise ValueError('argument `{}` must be `None`, an `int` or sequence of `int`'.format(attr))

        periodic = kwargs.pop('periodic', None)
        if periodic is None:
            pass
        elif isinstance(periodic, Sequence) and all(isinstance(p, int) for p in periodic):
            kwargs1['periodic'] = tuple(p for p in periodic if p < self.topo1.ndims)
            kwargs2['periodic'] = tuple(p - self.topo1.ndims for p in periodic if p >= self.topo1.ndims)
        else:
            raise ValueError('argument `periodic` must be `None` or a sequence of `int`')

        for attr, typ in ('knotvalues', (int, float)), ('knotmultiplicities', int), ('removedofs', int):
            val = kwargs.pop(attr, None)
            if val is None:
                pass
            elif isinstance(val, Sequence) and all(v is None or isinstance(v, Sequence) and all(isinstance(w, typ) for w in v) for v in val):
                if len(val) != self.ndims:
                    raise ValueError('argument `{}` must have length {} but got {}'.format(attr, self.ndims, len(val)))
                kwargs1[attr] = val[:self.topo1.ndims]
                kwargs2[attr] = val[self.topo1.ndims:]
            else:
                raise ValueError('argument `{}` must be `None`, a sequence or a sequence of sequence'.format(attr))

        kwargs1.update(kwargs)
        kwargs2.update(kwargs)

        basis1 = self.topo1.basis(name, **kwargs1)
        basis2 = self.topo2.basis(name, **kwargs2)
        assert basis1.ndim == basis2.ndim == 1
        return numpy.ravel(basis1[:,None] * basis2[None,:])

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.topo1.sample(ischeme, degree) * self.topo2.sample(ischeme, degree)


class _Take(Topology):

    def __init__(self, parent: Topology, indices: types.arraydata) -> None:
        self.parent = parent
        self.indices = indices = numpy.asarray(indices)
        assert indices.ndim == 1 and indices.size
        assert numpy.greater(indices[1:], indices[:-1]).all()
        assert 0 <= indices[0] and indices[-1] < len(self.parent)
        references = parent.references.take(self.indices)
        coord_system = parent.coord_system.take(self.indices)
        opposite = parent.opposite.take(self.indices)
        super().__init__(parent.ref_coord_system, references, coord_system, opposite)

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.parent.sample(ischeme, degree).take_elements(self.indices)


class _ConformingTopology(Topology):

    def __init__(self, ref_coord_system: OrderedDict[str, CoordSystem], references: References, coord_system: CoordSystem, opposite: CoordSystem, connectivity):
        self.connectivity = connectivity
        super().__init__(ref_coord_system, references, coord_system, opposite)

    @log.withcontext
    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        if spaces != frozenset(self.spaces):
            return ValueError('Cannot create the boundary for a subset of spaces.')
        references = []
        selection = []
        iglobaledgeiter = itertools.count()
        refs_touched = False
        for ielem, (ioppelems, elemref) in enumerate(zip(self.connectivity, self.references)):
            for edgeref, ioppelem, iglobaledge in zip(elemref.edge_refs, ioppelems, iglobaledgeiter):
                if edgeref:
                    if ioppelem == -1:
                        references.append(edgeref)
                        selection.append(iglobaledge)
                    else:
                        ioppedge = util.index(self.connectivity[ioppelem], ielem)
                        ref = edgeref - self.references[ioppelem].edge_refs[ioppedge]
                        if ref:
                            references.append(ref)
                            selection.append(iglobaledge)
                            refs_touched = True
        selection = types.frozenarray(selection, dtype=int)
        if refs_touched:
            references = References.from_iter(references, self.ndims-1)
        else:
            references = self.references.edges[selection]
        coord_system = self.references.edges_coord_system(self.coord_system).take(selection)
        return Topology(self.space, references, coord_system, coord_system)

    @log.withcontext
    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        if spaces != frozenset(self.spaces):
            return ValueError('Cannot create the boundary for a subset of spaces.')
        raise NotImplementedError
        references = []
        selection = []
        oppselection = []
        iglobaledgeiter = itertools.count()
        refs_touched = False
        edges = self.transforms.edges(self.references)
        if self.references.isuniform:
            _nedges = self.references[0].nedges
            offset = lambda ielem: ielem * _nedges
        else:
            offset = numpy.cumsum([0]+list(ref.nedges for ref in self.references)).__getitem__
        for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
            for (edgetrans, edgeref), ioppelem, iglobaledge in zip(elemref.edges, ioppelems, iglobaledgeiter):
                if edgeref and -1 < ioppelem < ielem:
                    ioppedge = util.index(self.connectivity[ioppelem], ielem)
                    oppedgetrans, oppedgeref = self.references[ioppelem].edges[ioppedge]
                    ref = oppedgeref and edgeref & oppedgeref
                    if ref:
                        references.append(ref)
                        selection.append(iglobaledge)
                        oppselection.append(offset(ioppelem)+ioppedge)
                        if ref != edgeref:
                            refs_touched = True
        selection = types.frozenarray(selection, dtype=int)
        oppselection = types.frozenarray(oppselection, dtype=int)
        if refs_touched:
            references = References.from_iter(references, self.ndims-1)
        else:
            references = self.references.edges[selection]
        return TransformChainsTopology(self.space, references, edges[selection], edges[oppselection])

    def basis_spline(self, degree):
        assert degree == 1
        return self.basis('std', degree)

    def _basis_c0_structured(self, name, degree):
        'C^0-continuous shape functions with lagrange stucture'

        assert numeric.isint(degree) and degree >= 0

        if degree == 0:
            raise ValueError('Cannot build a C^0-continuous basis of degree 0.  Use basis \'discont\' instead.')

        coeffs = [ref.get_poly_coeffs(name, degree=degree) for ref in self.references]
        offsets = numpy.cumsum([0] + [len(c) for c in coeffs])
        dofmap = numpy.repeat(-1, offsets[-1])
        for ielem, ioppelems in enumerate(self.connectivity):
            for iedge, jelem in enumerate(ioppelems):  # loop over element neighbors and merge dofs
                if jelem < ielem:
                    continue  # either there is no neighbor along iedge or situation will be inspected from the other side
                jedge = util.index(self.connectivity[jelem], ielem)
                idofs = offsets[ielem] + self.references[ielem].get_edge_dofs(degree, iedge)
                jdofs = offsets[jelem] + self.references[jelem].get_edge_dofs(degree, jedge)
                for idof, jdof in zip(idofs, jdofs):
                    while dofmap[idof] != -1:
                        idof = dofmap[idof]
                    while dofmap[jdof] != -1:
                        jdof = dofmap[jdof]
                    if idof != jdof:
                        dofmap[max(idof, jdof)] = min(idof, jdof)  # create left-looking pointer
        # assign dof numbers left-to-right
        ndofs = 0
        for i, n in enumerate(dofmap):
            if n == -1:
                dofmap[i] = ndofs
                ndofs += 1
            else:
                dofmap[i] = dofmap[n]

        elem_slices = map(slice, offsets[:-1], offsets[1:])
        dofs = tuple(types.frozenarray(dofmap[s]) for s in elem_slices)
        return function.PlainBasis(coeffs, dofs, ndofs, self.f_index, self.f_coords)

    def basis_lagrange(self, degree):
        'lagrange shape functions'
        return self._basis_c0_structured('lagrange', degree)

    def basis_bernstein(self, degree):
        'bernstein shape functions'
        return self._basis_c0_structured('bernstein', degree)

    basis_std = basis_bernstein


class LocateError(Exception):
    pass


class _WithGroupsTopology(Topology):
    'item topology'

    __slots__ = 'basetopo', 'vgroups', 'bgroups', 'igroups', 'pgroups'
    __cache__ = 'refined',

    def __init__(self, basetopo: Topology, vgroups: Optional[Dict[str, Union[str, Topology]]] = None, bgroups: Optional[Dict[str, Union[str, Topology]]] = None, igroups: Optional[Dict[str, Union[str, Topology]]] = None, pgroups: Optional[Dict[str, Union[str, Topology]]] = None):
        assert vgroups or bgroups or igroups or pgroups
        self.basetopo = basetopo
        self.vgroups = vgroups or {}
        self.bgroups = bgroups or {}
        self.igroups = igroups or {}
        self.pgroups = pgroups or {}
        super().__init__(basetopo.ref_coord_system, basetopo.references, basetopo.coord_system, basetopo.opposite)
        assert all(topo is Ellipsis or isinstance(topo, str) or isinstance(topo, Topology) and topo.ndims == basetopo.ndims for topo in self.vgroups.values())

    def get_groups(self, *groups: str) -> Topology:
        topos = []
        basegroups = []
        for group in groups:
            if group in self.vgroups:
                item = self.vgroups[group]
                assert isinstance(item, (Topology, str))
                if isinstance(item, Topology):
                    topos.append(item)
                else:
                    basegroups.extend(item.split(','))
            else:
                basegroups.append(group)
        if basegroups:
            topos.append(self.basetopo.get_groups(*basegroups))
        return functools.reduce(operator.or_, topos, self.empty_like())

    def take_unchecked(self, __indices: numpy.ndarray) -> Topology:
        return self.basetopo.take_unchecked(__indices)

    def slice_unchecked(self, __s: slice, __idim: int) -> Topology:
        return self.basetopo.slice_unchecked(__s, __idim)

    @property
    def connectivity(self):
        return self.basetopo.connectivity

    @property
    def boundary(self):
        return self.basetopo.boundary.withgroups(self.bgroups)

    @property
    def interfaces(self):
        baseitopo = self.basetopo.interfaces
        igroups = self.igroups.copy()
        for name, topo in self.igroups.items():
            if isinstance(topo, Topology):
                raise NotImplementedError
                # last minute orientation fix
                s = []
                for transs in zip(topo.transforms, topo.opposites):
                    for trans in transs:
                        try:
                            s.append(baseitopo.transforms.index(trans))
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError('group is not a subset of topology')
                s = types.frozenarray(tuple(sorted(s)), dtype=int)
                igroups[name] = TransformChainsTopology(self.space, baseitopo.references[s], baseitopo.transforms[s], baseitopo.opposites[s])
        return baseitopo.withgroups(igroups)

    @property
    def points(self):
        ptopos = []
        pnames = []
        topo = self
        while isinstance(topo, _WithGroupsTopology):
            for pname, ptopo in topo.pgroups.items():
                if pname not in pnames:
                    pnames.append(pname)
                    ptopos.append(ptopo)
            topo = topo.basetopo
        return UnionTopology(ptopos, pnames)

    def basis(self, name, *args, **kwargs):
        return self.basetopo.basis(name, *args, **kwargs)

    @property
    def refined(self):
        groups = [{name: topo.refined if isinstance(topo, Topology) else topo for name, topo in groups.items()} for groups in (self.vgroups, self.bgroups, self.igroups, self.pgroups)]
        return self.basetopo.refined.withgroups(*groups)

    def locate(self, geom, coords, **kwargs):
        return self.basetopo.locate(geom, coords, **kwargs)

    def sample(self, *args, **kwargs):
        return self.basetopo.sample(*args, **kwargs)


class OppositeTopology(Topology):
    'opposite topology'

    __slots__ = 'basetopo',

    def __init__(self, basetopo):
        self.basetopo = basetopo
        super().__init__(basetopo.ref_coord_system, basetopo.references, basetopo.opposite, basetopo.coord_system)

    def get_groups(self, *groups: str) -> Topology:
        return ~(self.basetopo.get_groups(*groups))

    def take_unchecked(self, __indices: numpy.ndarray) -> Topology:
        return ~(self.basetopo.take_unchecked(__indices))

    def slice_unchecked(self, __s: slice, __idim: int) -> Topology:
        return ~(self.basetopo.slice_unchecked(__s, __idim))

    def __invert__(self):
        return self.basetopo


class _Line(_ConformingTopology):
    'structured line topology'

    __slots__ = '_bnames', '_periodic', '_asaffine_geom', '_asaffine_retval'
    __cache__ = 'connectivity', 'boundary', 'interfaces'

    def __init__(self, ref_coord_system: OrderedDict[str, CoordSystem], coord_system: CoordSystem, opposite: CoordSystem, bnames: Tuple[str, str], periodic: bool):
        'constructor'

        self._bnames = bnames
        self._periodic = periodic
        references = References.uniform(element.getsimplex(1), len(coord_system))

        connectivity = numpy.stack([numpy.arange(1, len(references) + 1), numpy.arange(-1, len(references) - 1)], axis=1)
        if len(references) == 0:
            pass
        elif self._periodic:
            connectivity[0, 1] = len(references) - 1
            connectivity[-1, 0] = 0
        else:
            connectivity[0, 1] = -1
            connectivity[-1, 0] = -1
        connectivity = types.frozenarray(connectivity, copy=False)

        super().__init__(ref_coord_system, references, coord_system, opposite, connectivity)

    def __repr__(self):
        return '{}<{}{}>'.format(type(self).__qualname__, len(self), 'p' if self._periodic else '')

    def slice_unchecked(self, indices: slice, idim: int) -> Topology:
        if indices == slice(None):
            return self
        return _Line(
            self.ref_coord_system,
            self.coord_system.slice(indices),
            self.opposite.slice(indices),
            self._bnames,
            False)

    @property
    def periodic(self):
        return (0,) if self._periodic else ()

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        if self._periodic:
            return Topology.empty(self.ref_coord_system, 0)
        references = References.uniform(element.getsimplex(0), 1)
        coord_system_left = self.coord_system.edges(Simplex.line, 0).take([1])
        coord_system_right = self.coord_system.edges(Simplex.line, 0).take([2 * len(self) - 2])
        left = _WithName(Topology(self.ref_coord_system, references, coord_system_left, coord_system_left), self._bnames[0])
        right = _WithName(Topology(self.ref_coord_system, references, coord_system_right, coord_system_right), self._bnames[1])
        return Topology.disjoint_union(left, right)

    @property
    def interfaces(self):
        'interfaces'

        raise NotImplementedError

    def basis_spline(self, degree, removedofs=None, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
        'spline basis'

        if removedofs is not None and not isinstance(removedofs[0], int):
            removedofs, = removedofs

        if periodic is None:
            periodic = self._periodic
        else:
            periodic = 0 in periodic

        if not numeric.isint(degree):
            degree, = degree

        if knotvalues is not None and not isinstance(knotvalues[0], (int, float)):
            knotvalues, = knotvalues

        if knotmultiplicities is not None and not isinstance(knotmultiplicities[0], int):
            knotmultiplicities, = knotmultiplicities

        if numpy.iterable(continuity):
            continuity, = continuity

        start_dofs = []
        stop_dofs = []
        dofshape = []
        coeffs = []
        cache = {}

        p = degree
        n = len(self)

        c = continuity
        if c < 0:
            c += p
        assert -1 <= c < p

        k = knotvalues
        if k is None:
            k = numpy.arange(n+1)  # default to uniform spacing
        else:
            k = numpy.array(k)
            while len(k) < n+1:
                k_ = numpy.empty(len(k)*2-1)
                k_[::2] = k
                k_[1::2] = (k[:-1] + k[1:]) / 2
                k = k_
            assert len(k) == n+1, 'knot values do not match the topology size'

        m = knotmultiplicities
        if m is None:
            m = numpy.repeat(p-c, n+1)  # default to open spline without internal repetitions
        else:
            m = numpy.array(m)
            assert min(m) > 0 and max(m) <= p+1, 'incorrect multiplicity encountered'
            while len(m) < n+1:
                m_ = numpy.empty(len(m)*2-1, dtype=int)
                m_[::2] = m
                m_[1::2] = p-c
                m = m_
            assert len(m) == n+1, 'knot multiplicity do not match the topology size'

        if periodic and not m[0] == m[n] == p+1:  # if m[0] == m[n] == p+1 the spline is discontinuous at the boundary
            assert m[0] == m[n], 'periodic spline multiplicity expected'
            dk = k[n] - k[0]
            m = m[:n]
            k = k[:n]
            nd = m.sum()
            while m[n:].sum() < p - m[0] + 2:
                k = numpy.concatenate([k, k+dk])
                m = numpy.concatenate([m, m])
                dk *= 2
            km = numpy.array([ki for ki, mi in zip(k, m) for cnt in range(mi)], dtype=float)
            if p > m[0]:
                km = numpy.concatenate([km[-p+m[0]:] - dk, km])
        else:
            m[0] = m[-1] = p
            nd = m[:n].sum()+1
            km = numpy.array([ki for ki, mi in zip(k, m) for cnt in range(mi)], dtype=float)

        offsets = numpy.cumsum(m[:n]) - m[0]
        start_dofs.append(offsets)
        stop_dofs.append(offsets+p+1)
        dofshape.append(nd)

        coeffs_i = []
        for offset in offsets:
            lknots = km[offset:offset+2*p]
            key = tuple(numeric.round((lknots[1:-1]-lknots[0])/(lknots[-1]-lknots[0])*numpy.iinfo(numpy.int32).max)) if lknots.size else (), p
            try:
                local_coeffs = cache[key]
            except KeyError:
                local_coeffs = cache[key] = self._localsplinebasis(lknots)
            coeffs_i.append(local_coeffs)
        coeffs.append(tuple(coeffs_i))

        func = function.StructuredBasis(coeffs, start_dofs, stop_dofs, dofshape, (n,), self.f_index, self.f_coords)
        if not removedofs:
            return func

        mask = numpy.ones((), dtype=bool)
        for idofs, ndofs in zip([removedofs], dofshape):
            mask = mask[..., _].repeat(ndofs, axis=-1)
            if idofs:
                mask[..., [numeric.normdim(ndofs, idof) for idof in idofs]] = False
        assert mask.shape == tuple(dofshape)
        return func[mask.ravel()]

    @staticmethod
    def _localsplinebasis(lknots):

        assert numeric.isarray(lknots), 'Local knot vector should be numpy array'
        p, rem = divmod(len(lknots), 2)
        assert rem == 0

        # Based on Algorithm A2.2 Piegl and Tiller
        N = [None]*(p+1)
        N[0] = numpy.poly1d([1.])

        if p > 0:

            assert numpy.less(lknots[:-1]-lknots[1:], numpy.spacing(1)).all(), 'Local knot vector should be non-decreasing'
            assert lknots[p]-lknots[p-1] > numpy.spacing(1), 'Element size should be positive'

            lknots = lknots.astype(float)

            xi = numpy.poly1d([lknots[p]-lknots[p-1], lknots[p-1]])

            left = [None]*p
            right = [None]*p

            for i in range(p):
                left[i] = xi - lknots[p-i-1]
                right[i] = -xi + lknots[p+i]
                saved = 0.
                for r in range(i+1):
                    temp = N[r]/(lknots[p+r]-lknots[p+r-i-1])
                    N[r] = saved+right[r]*temp
                    saved = left[i-r]*temp
                N[i+1] = saved

        assert all(Ni.order == p for Ni in N)

        return types.frozenarray([Ni.coeffs[::-1] for Ni in N])

    def basis_std(self, *args, **kwargs):
        return __class__.basis_spline(self, *args, continuity=0, **kwargs)

    def basis_legendre(self, degree: int):
        if self.ndims != 1:
            raise NotImplementedError('legendre is only implemented for 1D topologies')
        return function.LegendreBasis(degree, len(self), self.f_index, self.f_coords)

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]):
        if not spaces:
            return self
        return _Line(
            self.ref_coord_system,
            self.coord_system.children(Simplex.line, 0),
            self.opposite.children(Simplex.line, 0),
            self._bnames,
            self._periodic)

    def locate(self, geom, coords, *, tol=0, eps=0, weights=None, skip_missing=False, arguments=None, **kwargs):
        raise NotImplementedError
        coords = numpy.asarray(coords, dtype=float)
        if geom.ndim == 0:
            geom = geom[_]
            coords = coords[..., _]
        if not geom.shape == coords.shape[1:] == (self.ndims,):
            raise Exception('invalid geometry or point shape for {}D topology'.format(self.ndims))
        if tol or eps:
            if arguments:
                geom0, scale, error = self._asaffine(geom, arguments)
            elif geom is getattr(self, '_asaffine_geom', None):
                log.debug('locate found previously computed affine values')
                geom0, scale, error = self._asaffine_retval
            else:
                self._asaffine_geom = geom
                geom0, scale, error = self._asaffine_retval = self._asaffine(geom, {})
            if all(error <= numpy.maximum(tol, eps * scale)):
                log.debug('locate detected linear geometry: x = {} + {} xi ~{}'.format(geom0, scale, error))
                return self._locate(geom0, scale, coords, eps=eps, weights=weights, skip_missing=skip_missing)
        return super().locate(geom, coords, eps=eps, tol=tol, weights=weights, skip_missing=skip_missing, arguments=arguments, **kwargs)

    def _asaffine(self, geom, arguments):
        # determine geom0, scale, error such that geom ~= geom0 + index * scale + error
        n = 2 + (1 in self.shape) # number of sample points required to establish nonlinearity
        sampleshape = numpy.multiply(self.shape, n) # shape of uniform sample
        geom_ = self.sample('uniform', n).eval(geom, **arguments) \
            .reshape(*self.shape, *[n] * self.ndims, self.ndims) \
            .transpose(*(i+j for i in range(self.ndims) for j in (0, self.ndims)), self.ndims*2) \
            .reshape(*sampleshape, self.ndims)
        # strategy: fit an affine plane through the minima and maxima of a uniform sample,
        # and evaluate the error as the largest difference on the remaining sample points
        xmin, xmax = geom_.reshape(-1, self.ndims)[[0, -1]]
        dx = (xmax - xmin) / (sampleshape-1) # x = x0 + dx * (i + .5) => xmax - xmin = dx * (sampleshape-1)
        for idim in range(self.ndims):
            geom_[...,idim] -= xmin[idim] + dx[idim] * numpy.arange(sampleshape[idim]).reshape([-1 if i == idim else 1 for i in range(self.ndims)])
        return xmin - dx/2, dx * n, numpy.abs(geom_).reshape(-1, self.ndims).max(axis=0)

    def _locate(self, geom0, scale, coords, *, eps=0, weights=None, skip_missing=False):
        mincoords, maxcoords = numpy.sort([geom0, geom0 + scale * self.shape], axis=0)
        missing = numpy.any(numpy.less(coords, mincoords - eps) | numpy.greater(coords, maxcoords + eps), axis=1)
        if not skip_missing and missing.any():
            raise LocateError('failed to locate {}/{} points'.format(missing.sum(), len(coords)))
        xi = (coords - geom0) / scale
        ielem = numpy.minimum(numpy.maximum(xi.astype(int), 0), numpy.array(self.shape)-1)
        ielems = numpy.ravel_multi_index(ielem.T, self.shape)
        points = xi - ielem
        if skip_missing:
            ielems = ielems[~missing]
            points = points[~missing]
        return self._sample(ielems, points, weights)


class _SimplexTopology(_ConformingTopology):
    'simpex topology'

    __slots__ = 'simplices'
    __cache__ = 'connectivity'

    def _renumber(simplices):
        simplices = numpy.asarray(simplices)
        keep = numpy.zeros(simplices.max()+1, dtype=bool)
        keep[simplices.flat] = True
        return types.arraydata(simplices if keep.all() else (numpy.cumsum(keep)-1)[simplices])

    def __init__(self, ref_coord_system: OrderedDict[str, CoordSystem], simplices, coord_system: CoordSystem, opposite: CoordSystem):
        assert simplices.shape == (len(coord_system), coord_system.dim + 1)
        self.simplices = numpy.asarray(simplices)
        assert numpy.greater(self.simplices[:, 1:], self.simplices[:, :-1]).all(), 'nodes should be sorted'
        assert not numpy.equal(self.simplices[:, 1:], self.simplices[:, :-1]).all(), 'duplicate nodes'
        references = References.uniform(element.getsimplex(coord_system.dim), len(coord_system))

        ndims = references.ndims
        nverts = ndims + 1
        edge_vertices = numpy.arange(nverts).repeat(ndims).reshape(ndims, nverts)[:, ::-1].T  # nverts x ndims
        simplices_edges = numpy.take(simplices, edge_vertices, axis=1)  # nelems x nverts x ndims
        elems, edges = divmod(numpy.lexsort(simplices_edges.reshape(-1, ndims).T), nverts)
        sorted_simplices_edges = simplices_edges[elems, edges]  # (nelems x nverts) x ndims; matching edges are now adjacent
        i, = numpy.equal(sorted_simplices_edges[1:], sorted_simplices_edges[:-1]).all(axis=1).nonzero()
        j = i + 1
        assert numpy.greater(i[1:], j[:-1]).all(), 'single edge is shared by three or more simplices'
        connectivity = numpy.full((len(self.simplices), ndims+1), fill_value=-1, dtype=int)
        connectivity[elems[i], edges[i]] = elems[j]
        connectivity[elems[j], edges[j]] = elems[i]
        connectivity = types.frozenarray(connectivity, copy=False)

        super().__init__(ref_coord_system, references, coord_system, opposite, connectivity)

    def basis_std(self, degree):
        if degree == 1:
            coeffs = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
            return function.PlainBasis([coeffs] * len(self), self.simplices, self.simplices.max()+1, self.f_index, self.f_coords)
        return super().basis_std(degree)

    def basis_bubble(self):
        'bubble from vertices'

        bernstein = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
        bubble = functools.reduce(numeric.poly_mul, bernstein)
        coeffs = numpy.zeros((len(bernstein)+1,) + bubble.shape)
        coeffs[(slice(-1),)+(slice(2),)*self.ndims] = bernstein
        coeffs[-1] = bubble
        coeffs[:-1] -= bubble / (self.ndims+1)
        coeffs = types.frozenarray(coeffs, copy=False)
        nverts = self.simplices.max() + 1
        ndofs = nverts + len(self)
        nmap = [types.frozenarray(numpy.hstack([idofs, nverts+ielem]), copy=False) for ielem, idofs in enumerate(self.simplices)]
        return function.PlainBasis([coeffs] * len(self), nmap, ndofs, self.f_index, self.f_coords)


class UnionTopology(Topology):
    'grouped topology'

    __slots__ = '_topos', '_names', 'references', 'transforms', 'opposites'

    def __init__(self, topos: Tuple[Topology, ...], names: Tuple[str, ...] = ()):
        self._topos = topos
        self._names = tuple(names)[:len(self._topos)]
        assert len(set(self._names)) == len(self._names), 'duplicate name'
        ndims = self._topos[0].ndims
        assert all(topo.ndims == ndims for topo in self._topos)
        space = self._topos[0].space
        assert all(topo.space == space for topo in self._topos)

        references = []
        selections = [[] for topo in topos]
        for trans, indices in util.gather((trans, (itopo, itrans)) for itopo, topo in enumerate(self._topos) for itrans, trans in enumerate(topo.transforms)):
            itopo0, itrans0 = indices[0]
            selections[itopo0].append(itrans0)
            if len(indices) == 1:
                references.append(self._topos[itopo0].references[itrans0])
            else:
                refs = [self._topos[itopo].references[itrans] for itopo, itrans in indices]
                while len(refs) > 1:  # sweep all possible unions until a single reference is left
                    nrefs = len(refs)
                    iref = 0
                    while iref < len(refs)-1:
                        for jref in range(iref+1, len(refs)):
                            try:
                                unionref = refs[iref] | refs[jref]
                            except TypeError:
                                pass
                            else:
                                refs[iref] = unionref
                                del refs[jref]
                                break
                        iref += 1
                    assert len(refs) < nrefs, 'incompatible elements in union'
                references.append(refs[0])
                assert len(set(self._topos[itopo].opposites[itrans] for itopo, itrans in indices)) == 1
        selections = tuple(types.frozenarray(s, dtype=int) for s in selections)

        super().__init__(
            space,
            References.from_iter(references, ndims),
            transformseq.chain((topo.transforms[selection] for topo, selection in zip(topos, selections)), topos[0].transforms.todims, ndims),
            transformseq.chain((topo.opposites[selection] for topo, selection in zip(topos, selections)), topos[0].transforms.todims, ndims))

    def get_groups(self, *groups: str) -> Topology:
        topos = (topo if name in groups else topo.get_groups(*groups) for topo, name in itertools.zip_longest(self._topos, self._names))
        return functools.reduce(operator.or_, filter(None, topos), self.empty_like())

    def __or__(self, other):
        if not isinstance(other, Topology):
            return super().__or__(other)
        if not isinstance(other, UnionTopology):
            return UnionTopology(self._topos + (other,), self._names)
        return UnionTopology(self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names)

    @property
    def refined(self):
        return UnionTopology([topo.refined for topo in self._topos], self._names)


class DisjointUnionTopology(Topology):
    'grouped topology'

    __slots__ = '_topos', '_names'

    def __init__(self, topos: Tuple[Topology, ...], names: Tuple[str, ...] = ()):
        self._topos = topos
        self._names = tuple(names)[:len(self._topos)]
        assert len(set(self._names)) == len(self._names), 'duplicate name'
        ndims = self._topos[0].ndims
        assert all(topo.ndims == ndims for topo in self._topos)
        space = self._topos[0].space
        assert all(topo.space == space for topo in self._topos)
        super().__init__(
            space,
            util.sum(topo.references for topo in self._topos),
            transformseq.chain((topo.transforms for topo in self._topos), topos[0].transforms.todims, ndims),
            transformseq.chain((topo.opposites for topo in self._topos), topos[0].transforms.todims, ndims))

    def get_groups(self, *groups: str) -> Topology:
        topos = (topo if name in groups else topo.get_groups(*groups) for topo, name in itertools.zip_longest(self._topos, self._names))
        topos = tuple(filter(None, topos))
        if len(topos) == 0:
            return self.empty_like()
        elif len(topos) == 1:
            return topos[0]
        else:
            return DisjointUnionTopology(topos)

    @property
    def refined(self):
        return DisjointUnionTopology([topo.refined for topo in self._topos], self._names)


class SubsetTopology(Topology):
    'trimmed'

    __slots__ = 'refs', 'basetopo', 'newboundary', '_indices'
    __cache__ = 'connectivity', 'boundary', 'interfaces', 'refined'

    def __init__(self, basetopo: Topology, refs: Tuple[Reference, ...], newboundary=None):
        if newboundary is not None:
            assert isinstance(newboundary, str) or isinstance(newboundary, Topology) and newboundary.ndims == basetopo.ndims-1
        assert len(refs) == len(basetopo)
        self.refs = refs
        self.basetopo = basetopo
        self.newboundary = newboundary

        self._indices = types.frozenarray(numpy.array([i for i, ref in enumerate(self.refs) if ref], dtype=int), copy=False)
        references = References.from_iter(self.refs, self.basetopo.ndims).take(self._indices)
        transforms = self.basetopo.transforms[self._indices]
        opposites = self.basetopo.opposites[self._indices]
        super().__init__(basetopo.space, references, transforms, opposites)

    def get_groups(self, *groups: str) -> Topology:
        return self.basetopo.get_groups(*groups).subset(self, strict=False)

    def __rsub__(self, other):
        if self.basetopo == other:
            refs = [baseref - ref for baseref, ref in zip(self.basetopo.references, self.refs)]
            return SubsetTopology(self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary, Topology) else self.newboundary)
        return super().__rsub__(other)

    def __or__(self, other):
        if not isinstance(other, SubsetTopology) or self.basetopo != other.basetopo:
            return super().__or__(other)
        refs = [ref1 | ref2 for ref1, ref2 in zip(self.refs, other.refs)]
        if all(baseref == ref for baseref, ref in zip(self.basetopo.references, refs)):
            return self.basetopo
        return SubsetTopology(self.basetopo, refs)  # TODO boundary

    @property
    def connectivity(self):
        renumber = numeric.invmap([i for i, ref in enumerate(self.refs) if ref], length=len(self.refs)+1, missing=-1)  # length=..+1 serves to map -1 to -1
        return tuple(types.frozenarray(numpy.concatenate([renumber.take(ioppelems), numpy.repeat(-1, ref.nedges-len(ioppelems))]), copy=False)
                     for ref, ioppelems in zip(self.refs, self.basetopo.connectivity) if ref)

    @property
    def refined(self):
        child_refs = self.references.children
        indices = types.frozenarray(numpy.array([i for i, ref in enumerate(child_refs) if ref], dtype=int), copy=False)
        refined_transforms = self.transforms.refined(self.references)[indices]
        self_refined = TransformChainsTopology(self.space, child_refs[indices], refined_transforms, refined_transforms)
        return self.basetopo.refined.subset(self_refined, self.newboundary.refined if isinstance(self.newboundary, Topology) else self.newboundary, strict=True)

    @property
    def boundary(self):
        baseboundary = self.basetopo.boundary
        baseconnectivity = self.basetopo.connectivity
        brefs = [ref.empty for ref in baseboundary.references]
        trimmedreferences = []
        trimmedtransforms = []
        trimmedopposites = []
        for ielem, newref in enumerate(self.refs):
            if not newref:
                continue
            elemtrans = self.basetopo.transforms[ielem]
            # The first edges of newref by convention share location with the edges
            # of the original reference. We can therefore use baseconnectivity to
            # locate opposing edges.
            ioppelems = baseconnectivity[ielem]
            for (edgetrans, edgeref), ioppelem in zip(newref.edges, ioppelems):
                if not edgeref:
                    continue
                if ioppelem == -1:
                    # If the edge had no opposite in basetopology then it must already by
                    # in baseboundary, so we can use index to locate it.
                    brefs[baseboundary.transforms.index(elemtrans+(edgetrans,))] = edgeref
                else:
                    # If the edge did have an opposite in basetopology then there is a
                    # possibility this opposite (partially) disappeared, in which case
                    # the exposed part is added to the trimmed group.
                    ioppedge = util.index(baseconnectivity[ioppelem], ielem)
                    oppref = self.refs[ioppelem]
                    edgeref -= oppref.edge_refs[ioppedge]
                    if edgeref:
                        trimmedreferences.append(edgeref)
                        trimmedtransforms.append(elemtrans+(edgetrans,))
                        trimmedopposites.append(self.basetopo.transforms[ioppelem]+(oppref.edge_transforms[ioppedge],))
            # The last edges of newref (beyond the number of edges of the original)
            # cannot have opposites and are added to the trimmed group directly.
            for edgetrans, edgeref in newref.edges[len(ioppelems):]:
                trimmedreferences.append(edgeref)
                trimmedtransforms.append(elemtrans+(edgetrans,))
                trimmedopposites.append(elemtrans+(edgetrans.flipped,))
        origboundary = SubsetTopology(baseboundary, brefs)
        if isinstance(self.newboundary, Topology):
            trimmedbrefs = [ref.empty for ref in self.newboundary.references]
            for ref, trans in zip(trimmedreferences, trimmedtransforms):
                trimmedbrefs[self.newboundary.transforms.index(trans)] = ref
            trimboundary = SubsetTopology(self.newboundary, trimmedbrefs)
        else:
            trimboundary = TransformChainsTopology(self.space, References.from_iter(trimmedreferences, self.ndims-1), transformseq.PlainTransforms(trimmedtransforms, self.transforms.todims, self.ndims-1), transformseq.PlainTransforms(trimmedopposites, self.transforms.todims, self.ndims-1))
        return DisjointUnionTopology([trimboundary, origboundary], names=[self.newboundary] if isinstance(self.newboundary, str) else [])

    @property
    def interfaces(self):
        baseinterfaces = self.basetopo.interfaces
        superinterfaces = super().interfaces
        irefs = [ref.empty for ref in baseinterfaces.references]
        for ref, trans, opp in zip(superinterfaces.references, superinterfaces.transforms, superinterfaces.opposites):
            try:
                iielem = baseinterfaces.transforms.index(trans)
            except ValueError:
                iielem = baseinterfaces.transforms.index(opp)
            irefs[iielem] = ref
        return SubsetTopology(baseinterfaces, irefs)

    @log.withcontext
    def basis(self, name, *args, **kwargs):
        if isinstance(self.basetopo, HierarchicalTopology):
            warnings.warn('basis may be linearly dependent; a linearly indepent basis is obtained by trimming first, then creating hierarchical refinements')
        basis = self.basetopo.basis(name, *args, **kwargs)
        return function.PrunedBasis(basis, self._indices, self.f_index, self.f_coords)

    def locate(self, geom, coords, *, eps=0, **kwargs):
        sample = self.basetopo.locate(geom, coords, eps=eps, **kwargs)
        for isampleelem, (transforms, points) in enumerate(zip(sample.transforms[0], sample.points)):
            ielem = self.basetopo.transforms.index(transforms)
            ref = self.refs[ielem]
            if ref != self.basetopo.references[ielem]:
                for i, coord in enumerate(points.coords):
                    if not ref.inside(coord, eps):
                        raise LocateError('failed to locate point: {}'.format(coords[sample.getindex(isampleelem)[i]]))
        return sample


class RefinedTopology(Topology):
    'refinement'

    __slots__ = 'basetopo',
    __cache__ = 'boundary', 'connectivity'

    def __init__(self, basetopo: Topology):
        self.basetopo = basetopo
        super().__init__(
            self.basetopo.space,
            self.basetopo.references.children,
            self.basetopo.transforms.refined(self.basetopo.references),
            self.basetopo.opposites.refined(self.basetopo.references))

    def get_groups(self, *groups: str) -> Topology:
        return self.basetopo.get_groups(*groups).refined

    @property
    def boundary(self):
        return self.basetopo.boundary.refined

    @property
    def connectivity(self):
        offsets = numpy.cumsum([0] + [ref.nchildren for ref in self.basetopo.references])
        connectivity = [offset + edges for offset, ref in zip(offsets, self.basetopo.references) for edges in ref.connectivity]
        for ielem, edges in enumerate(self.basetopo.connectivity):
            for iedge, jelem in enumerate(edges):
                if jelem == -1:
                    for ichild, ichildedge in self.basetopo.references[ielem].edgechildren[iedge]:
                        connectivity[offsets[ielem]+ichild][ichildedge] = -1
                elif jelem < ielem:
                    jedge = util.index(self.basetopo.connectivity[jelem], ielem)
                    for (ichild, ichildedge), (jchild, jchildedge) in zip(self.basetopo.references[ielem].edgechildren[iedge], self.basetopo.references[jelem].edgechildren[jedge]):
                        connectivity[offsets[ielem]+ichild][ichildedge] = offsets[jelem]+jchild
                        connectivity[offsets[jelem]+jchild][jchildedge] = offsets[ielem]+ichild
        return tuple(types.frozenarray(c, copy=False) for c in connectivity)


class HierarchicalTopology(Topology):
    'collection of nested topology elments'

    __slots__ = 'basetopo', 'levels', '_indices_per_level', '_offsets'
    __cache__ = 'refined', 'boundary', 'interfaces'

    def __init__(self, basetopo: Topology, indices_per_level: types.tuple[types.arraydata]):
        'constructor'

        assert all(ind.dtype == int for ind in indices_per_level)
        assert not isinstance(basetopo, HierarchicalTopology)
        self.basetopo = basetopo
        self._indices_per_level = tuple(map(numpy.asarray, indices_per_level))
        self._offsets = numpy.cumsum([0, *map(len, self._indices_per_level)], dtype=int)

        level = None
        levels = []
        references = References.empty(basetopo.ndims)
        transforms = []
        opposites = []
        for indices in self._indices_per_level:
            level = self.basetopo if level is None else level.refined
            levels.append(level)
            if len(indices):
                references = references.chain(level.references.take(indices))
                transforms.append(level.transforms[indices])
                opposites.append(level.opposites[indices])
        self.levels = tuple(levels)

        super().__init__(basetopo.space, references, transformseq.chain(transforms, basetopo.transforms.todims, basetopo.ndims), transformseq.chain(opposites, basetopo.transforms.todims, basetopo.ndims))

    def __and__(self, other):
        if not isinstance(other, HierarchicalTopology) or self.basetopo != other.basetopo:
            return super().__and__(other)
        indices_per_level = []
        levels = max(self.levels, other.levels, key=len)
        for level, self_indices, other_indices in itertools.zip_longest(levels, self._indices_per_level, other._indices_per_level, fillvalue=()):
            keep = numpy.zeros(len(level), dtype=bool)
            for topo, topo_indices, indices in (other, other_indices, self_indices), (self, self_indices, other_indices):
                mask = numeric.asboolean(topo_indices, len(level))
                for index in indices:  # keep common elements or elements which are finer than conterpart
                    keep[index] = mask[index] or topo.transforms.contains_with_tail(level.transforms[index])
            indices, = keep.nonzero()
            indices_per_level.append(indices)
        return HierarchicalTopology(self.basetopo, indices_per_level)

    def _rebase(self, newbasetopo: Topology) -> 'HierarchicalTopology':
        itemindices_per_level = []
        for baseindices, baselevel, itemlevel in zip(self._indices_per_level, self.basetopo.refine_iter, newbasetopo.refine_iter):
            itemindices = []
            itemindex = itemlevel.transforms.index
            for basetrans in map(baselevel.transforms.__getitem__, baseindices):
                try:
                    itemindices.append(itemindex(basetrans))
                except ValueError:
                    pass
            itemindices_per_level.append(numpy.unique(numpy.array(itemindices, dtype=int)))
        return HierarchicalTopology(newbasetopo, itemindices_per_level)

    def slice_unchecked(self, __s: slice, __idim: int) -> 'HierarchicalTopology':
        return self._rebase(self.basetopo.slice_unchecked(__s, __idim))

    def get_groups(self, *groups: str) -> 'HierarchicalTopology':
        return self._rebase(self.basetopo.get_groups(*groups))

    def refined_by(self, refine):
        refine = tuple(refine)
        if not all(map(numeric.isint, refine)):
            refine = tuple(self.transforms.index_with_tail(item)[0] for item in refine)
        refine = numpy.unique(numpy.array(refine, dtype=int))
        splits = numpy.searchsorted(refine, self._offsets, side='left')
        indices_per_level = list(map(list, self._indices_per_level))+[[]]
        fine = self.basetopo
        for ilevel, (start, stop) in enumerate(zip(splits[:-1], splits[1:])):
            coarse, fine = fine, fine.refined
            coarse_indices = tuple(map(indices_per_level[ilevel].pop, reversed(refine[start:stop]-self._offsets[ilevel])))
            coarse_transforms = map(coarse.transforms.__getitem__, coarse_indices)
            coarse_references = map(coarse.references.__getitem__, coarse_indices)
            fine_transforms = (trans+(ctrans,) for trans, ref in zip(coarse_transforms, coarse_references) for ctrans, cref in ref.children if cref)
            indices_per_level[ilevel+1].extend(map(fine.transforms.index, fine_transforms))
        if not indices_per_level[-1]:
            indices_per_level.pop(-1)
        return HierarchicalTopology(self.basetopo, ([numpy.unique(numpy.array(i, dtype=int)) for i in indices_per_level]))

    @property
    def refined(self):
        refined_indices_per_level = [numpy.array([], dtype=int)]
        fine = self.basetopo
        for coarse_indices in self._indices_per_level:
            coarse, fine = fine, fine.refined
            coarse_transforms = map(coarse.transforms.__getitem__, coarse_indices)
            coarse_references = map(coarse.references.__getitem__, coarse_indices)
            fine_transforms = (trans+(ctrans,) for trans, ref in zip(coarse_transforms, coarse_references) for ctrans, cref in ref.children if cref)
            refined_indices_per_level.append(numpy.unique(numpy.fromiter(map(fine.transforms.index, fine_transforms), dtype=int)))
        return HierarchicalTopology(self.basetopo, refined_indices_per_level)

    @property
    @log.withcontext
    def boundary(self):
        'boundary elements'

        basebtopo = self.basetopo.boundary
        bindices_per_level = []
        for indices, level, blevel in zip(self._indices_per_level, self.basetopo.refine_iter, basebtopo.refine_iter):
            bindex = blevel.transforms.index
            bindices = []
            for index in indices:
                for etrans, eref in level.references[index].edges:
                    if eref:
                        trans = level.transforms[index]+(etrans,)
                        try:
                            bindices.append(bindex(trans))
                        except ValueError:
                            pass
            bindices = numpy.array(bindices, dtype=int)
            if len(bindices) > 1:
                bindices.sort()
                assert not numpy.equal(bindices[1:], bindices[:-1]).any()
            bindices_per_level.append(bindices)
        return HierarchicalTopology(basebtopo, bindices_per_level)

    @property
    @log.withcontext
    def interfaces(self):
        'interfaces'

        hreferences = References.empty(self.ndims-1)
        htransforms = []
        hopposites = []
        for level, indices in zip(self.levels, self._indices_per_level):
            selection = []
            to = level.interfaces.transforms, level.interfaces.opposites
            for trans, ref in zip(map(level.transforms.__getitem__, indices), map(level.references.__getitem__, indices)):
                for etrans, eref in ref.edges:
                    if not eref:
                        continue
                    for transforms, opposites in to, to[::-1]:
                        try:
                            i = transforms.index(trans+(etrans,))
                        except ValueError:
                            continue
                        if self.transforms.contains_with_tail(opposites[i]):
                            selection.append(i)
                        break
            if selection:
                selection = types.frozenarray(numpy.unique(selection))
                hreferences = hreferences.chain(level.interfaces.references.take(selection))
                htransforms.append(level.interfaces.transforms[selection])
                hopposites.append(level.interfaces.opposites[selection])
        return TransformChainsTopology(self.space, hreferences, transformseq.chain(htransforms, self.transforms.todims, self.ndims-1), transformseq.chain(hopposites, self.transforms.todims, self.ndims-1))

    @log.withcontext
    def basis(self, name, *args, truncation_tolerance=1e-15, **kwargs):
        '''Create hierarchical basis.

        A hierarchical basis is constructed from bases on different levels of
        uniform refinement. Two different types of hierarchical bases are
        supported:

        1. Classical -- Starting from the set of all basis functions originating
        from all levels of uniform refinement, only those basis functions are
        selected for which at least one supporting element is part of the
        hierarchical topology.

        2. Truncated -- Like classical, but with basis functions modified such that
        the area of support is reduced. An additional effect of this procedure is
        that it restores partition of unity. The spanned function space remains
        unchanged.

        Truncation is based on linear combinations of basis functions, where fine
        level basis functions are used to reduce the support of coarser level basis
        functions. See `Giannelli et al. 2012`_ for more information on truncated
        hierarchical refinement.

        .. _`Giannelli et al. 2012`: https://pdfs.semanticscholar.org/a858/aa68da617ad9d41de021f6807cc422002258.pdf

        Args
        ----
        name : :class:`str`
          Type of basis function as provided by the base topology, with prefix
          ``h-`` (``h-std``, ``h-spline``) for a classical hierarchical basis and
          prefix ``th-`` (``th-std``, ``th-spline``) for a truncated hierarchical
          basis.
        truncation_tolerance : :class:`float` (default 1e-15)
          In order to benefit from the extra sparsity resulting from truncation,
          vanishing polynomials need to be actively identified and removed from the
          basis. The ``trunctation_tolerance`` offers control over this threshold.

        Returns
        -------
        basis : :class:`nutils.function.Array`
        '''

        if name.startswith('h-'):
            truncated = False
            name = name[2:]
        elif name.startswith('th-'):
            truncated = True
            name = name[3:]
        else:
            return super().basis(name, *args, **kwargs)

        # 1. identify active (supported) and passive (unsupported) basis functions
        ubases = []
        ubasis_active = []
        ubasis_passive = []
        prev_transforms = None
        prev_ielems = []
        map_indices = []
        with log.iter.fraction('level', self.levels[::-1], self._indices_per_level[::-1]) as items:
            for topo, touchielems_i in items:

                topo_index_with_tail = topo.transforms.index_with_tail
                mapped_prev_ielems = [topo_index_with_tail(prev_transforms[j])[0] for j in prev_ielems]
                map_indices.insert(0, dict(zip(prev_ielems, mapped_prev_ielems)))
                nontouchielems_i = numpy.unique(numpy.array(mapped_prev_ielems, dtype=int))
                prev_ielems = ielems_i = numpy.unique(numpy.concatenate([numpy.asarray(touchielems_i, dtype=int), nontouchielems_i], axis=0))
                prev_transforms = topo.transforms

                basis_i = topo.basis(name, *args, **kwargs)
                assert isinstance(basis_i, function.Basis)
                ubases.insert(0, basis_i)
                # Basis functions that have at least one touchelem in their support.
                touchdofs_i = basis_i.get_dofs(touchielems_i)
                # Basis functions with (partial) support in this hierarchical topology.
                partsuppdofs_i = numpy.union1d(touchdofs_i, basis_i.get_dofs(numpy.setdiff1d(ielems_i, touchielems_i, assume_unique=True)))
                # Mask of basis functions in `partsuppdofs_i` with strict support in this hierarchical topology.
                partsuppdofs_supported_i = numpy.array([numeric.sorted_contains(ielems_i, basis_i.get_support(dof)).all() for dof in partsuppdofs_i], dtype=bool)
                ubasis_active.insert(0, numpy.intersect1d(touchdofs_i, partsuppdofs_i[partsuppdofs_supported_i], assume_unique=True))
                ubasis_passive.insert(0, partsuppdofs_i[~partsuppdofs_supported_i])

        *offsets, ndofs = numpy.cumsum([0, *map(len, ubasis_active)])

        # 2. construct hierarchical polynomials
        hbasis_dofs = []
        hbasis_coeffs = []
        projectcache = {}

        for ilevel, (level, indices) in enumerate(zip(self.levels, self._indices_per_level)):
            for ilocal in indices:

                hbasis_trans = transform.canonical(level.transforms[ilocal])
                tail = hbasis_trans[len(hbasis_trans)-ilevel:]
                trans_dofs = []
                trans_coeffs = []

                local_indices = [ilocal]
                for m in reversed(map_indices[:ilevel]):
                    ilocal = m[ilocal]
                    local_indices.insert(0, ilocal)

                if not truncated:  # classical hierarchical basis

                    for h, ilocal in enumerate(local_indices):  # loop from coarse to fine
                        mydofs = ubases[h].get_dofs(ilocal)

                        imyactive = numeric.sorted_index(ubasis_active[h], mydofs, missing=-1)
                        myactive = numpy.greater_equal(imyactive, 0)
                        if myactive.any():
                            trans_dofs.append(offsets[h]+imyactive[myactive])
                            mypoly = ubases[h].get_coefficients(ilocal)
                            trans_coeffs.append(mypoly[myactive])

                        if h < len(tail):
                            trans_coeffs = [tail[h].transform_poly(c) for c in trans_coeffs]

                else:  # truncated hierarchical basis

                    for h, ilocal in reversed(tuple(enumerate(local_indices))):  # loop from fine to coarse
                        mydofs = ubases[h].get_dofs(ilocal)
                        mypoly = ubases[h].get_coefficients(ilocal)

                        truncpoly = mypoly if h == len(tail) \
                            else numpy.tensordot(numpy.tensordot(tail[h].transform_poly(mypoly), project[..., mypassive], self.ndims), truncpoly[mypassive], 1)

                        imyactive = numeric.sorted_index(ubasis_active[h], mydofs, missing=-1)
                        myactive = numpy.greater_equal(imyactive, 0) & numpy.greater(abs(truncpoly), truncation_tolerance).any(axis=tuple(range(1, truncpoly.ndim)))
                        if myactive.any():
                            trans_dofs.append(offsets[h]+imyactive[myactive])
                            trans_coeffs.append(truncpoly[myactive])

                        mypassive = numeric.sorted_contains(ubasis_passive[h], mydofs)
                        if not mypassive.any():
                            break

                        try:  # construct least-squares projection matrix
                            project = projectcache[id(mypoly)][0]
                        except KeyError:
                            P = mypoly.reshape(len(mypoly), -1)
                            U, S, V = numpy.linalg.svd(P)  # (U * S).dot(V[:len(S)]) == P
                            project = (V.T[:, :len(S)] / S).dot(U.T).reshape(mypoly.shape[1:]+mypoly.shape[:1])
                            projectcache[id(mypoly)] = project, mypoly  # NOTE: mypoly serves to keep array alive

                # add the dofs and coefficients to the hierarchical basis
                hbasis_dofs.append(numpy.concatenate(trans_dofs))
                hbasis_coeffs.append(numeric.poly_concatenate(*trans_coeffs))

        return function.PlainBasis(hbasis_coeffs, hbasis_dofs, ndofs, self.f_index, self.f_coords)


class PatchBoundary(types.Singleton):

    __slots__ = 'id', 'dim', 'side', 'reverse', 'transpose'

    @types.apply_annotations
    def __init__(self, id: types.tuple[types.strictint], dim, side, reverse: types.tuple[bool], transpose: types.tuple[types.strictint]):
        super().__init__()
        self.id = id
        self.dim = dim
        self.side = side
        self.reverse = reverse
        self.transpose = transpose

    def apply_transform(self, array):
        return array[tuple(slice(None, None, -1) if i else slice(None) for i in self.reverse)].transpose(self.transpose)


class Patch(types.Singleton):

    __slots__ = 'topo', 'verts', 'boundaries'

    def __init__(self, topo: Topology, verts: types.arraydata, boundaries: types.tuple[types.strict[PatchBoundary]]):
        super().__init__()
        self.topo = topo
        self.verts = numpy.asarray(verts)
        self.boundaries = boundaries


class MultipatchTopology(Topology):
    'multipatch topology'

    __slots__ = 'patches',
    __cache__ = '_patchinterfaces', 'boundary', 'interfaces', 'refined', 'connectivity'

    @staticmethod
    def build_boundarydata(connectivity):
        'build boundary data based on connectivity'

        boundarydata = []
        for patch in connectivity:
            ndims = len(patch.shape)
            patchboundarydata = []
            for dim, side in itertools.product(range(ndims), [-1, 0]):
                # ignore vertices at opposite face
                verts = numpy.array(patch)
                opposite = tuple({0: -1, -1: 0}[side] if i == dim else slice(None) for i in range(ndims))
                verts[opposite] = verts.max()+1
                if len(set(verts.flat)) != 2**(ndims-1)+1:
                    raise NotImplementedError('Cannot compute canonical boundary if vertices are used more than once.')
                # reverse axes such that lowest vertex index is at first position
                reverse = tuple(map(bool, numpy.unravel_index(verts.argmin(), verts.shape)))
                verts = verts[tuple(slice(None, None, -1) if i else slice(None) for i in reverse)]
                # transpose such that second lowest vertex connects to lowest vertex in first dimension, third in second dimension, et cetera
                k = [verts[tuple(1 if i == j else 0 for j in range(ndims))] for i in range(ndims)]
                transpose = tuple(sorted(range(ndims), key=k.__getitem__))
                verts = verts.transpose(transpose)
                # boundarid
                boundaryid = tuple(verts[..., 0].flat)
                patchboundarydata.append(PatchBoundary(boundaryid, dim, side, reverse, transpose))
            boundarydata.append(tuple(patchboundarydata))

        return boundarydata

    @types.apply_annotations
    def __init__(self, patches: types.tuple[types.strict[Patch]]):
        'constructor'

        self.patches = patches

        space = patches[0].topo.space
        assert all(patch.topo.space == space for patch in patches)

        for boundaryid, patchdata in self._patchinterfaces.items():
            if len(patchdata) == 1:
                continue
            transposes = set()
            reverses = set()
            for topo, boundary in patchdata:
                assert boundary.transpose[-1] == boundary.dim
                transposes.add(tuple(i-1 if i > boundary.dim else i for i in boundary.transpose[:-1]))
                reverses.add(boundary.reverse[:boundary.dim]+boundary.reverse[boundary.dim+1:])
            if len(transposes) != 1 or len(reverses) != 1:
                raise NotImplementedError('patch interfaces must have the same order of axes and the same orientation per axis')

        super().__init__(
            space,
            util.sum(patch.topo.references for patch in self.patches),
            transformseq.chain([patch.topo.transforms for patch in self.patches], self.patches[0].topo.transforms.todims, self.patches[0].topo.ndims),
            transformseq.chain([patch.topo.opposites for patch in self.patches], self.patches[0].topo.transforms.todims, self.patches[0].topo.ndims))

    @property
    def _patchinterfaces(self):
        patchinterfaces = {}
        for patch in self.patches:
            for boundary in patch.boundaries:
                patchinterfaces.setdefault(boundary.id, []).append((patch.topo, boundary))
        return types.frozendict({
            boundaryid: tuple(data)
            for boundaryid, data in patchinterfaces.items()
            if len(data) > 1
        })

    def get_groups(self, *groups: str) -> Topology:
        topos = (patch.topo if 'patch{}'.format(i) in groups else patch.topo.get_groups(*groups) for i, patch in enumerate(self.patches))
        topos = tuple(filter(None, topos))
        if len(topos) == 0:
            return self.empty_like()
        elif len(topos) == 1:
            return topos[0]
        else:
            return DisjointUnionTopology(topos)

    def basis_spline(self, degree, patchcontinuous=True, knotvalues=None, knotmultiplicities=None, *, continuity=-1):
        '''spline from vertices

        Create a spline basis with degree ``degree`` per patch.  If
        ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
        interfaces.
        '''

        if knotvalues is None:
            knotvalues = {None: None}
        else:
            knotvalues, _knotvalues = {}, knotvalues
            for edge, k in _knotvalues.items():
                if k is None:
                    rk = None
                else:
                    k = tuple(k)
                    rk = k[::-1]
                if edge is None:
                    knotvalues[edge] = k
                else:
                    l, r = edge
                    assert (l, r) not in knotvalues
                    assert (r, l) not in knotvalues
                    knotvalues[(l, r)] = k
                    knotvalues[(r, l)] = rk

        if knotmultiplicities is None:
            knotmultiplicities = {None: None}
        else:
            knotmultiplicities, _knotmultiplicities = {}, knotmultiplicities
            for edge, k in _knotmultiplicities.items():
                if k is None:
                    rk = None
                else:
                    k = tuple(k)
                    rk = k[::-1]
                if edge is None:
                    knotmultiplicities[edge] = k
                else:
                    l, r = edge
                    assert (l, r) not in knotmultiplicities
                    assert (r, l) not in knotmultiplicities
                    knotmultiplicities[(l, r)] = k
                    knotmultiplicities[(r, l)] = rk

        missing = object()

        coeffs = []
        dofmap = []
        dofcount = 0
        commonboundarydofs = {}
        for ipatch, patch in enumerate(self.patches):
            # build structured spline basis on patch `patch.topo`
            patchknotvalues = []
            patchknotmultiplicities = []
            for idim in range(self.ndims):
                left = tuple(0 if j == idim else slice(None) for j in range(self.ndims))
                right = tuple(1 if j == idim else slice(None) for j in range(self.ndims))
                dimknotvalues = set()
                dimknotmultiplicities = set()
                for edge in zip(patch.verts[left].flat, patch.verts[right].flat):
                    v = knotvalues.get(edge, knotvalues.get(None, missing))
                    m = knotmultiplicities.get(edge, knotmultiplicities.get(None, missing))
                    if v is missing:
                        raise 'missing edge'
                    dimknotvalues.add(v)
                    if m is missing:
                        raise 'missing edge'
                    dimknotmultiplicities.add(m)
                if len(dimknotvalues) != 1:
                    raise 'ambiguous knot values for patch {}, dimension {}'.format(ipatch, idim)
                if len(dimknotmultiplicities) != 1:
                    raise 'ambiguous knot multiplicities for patch {}, dimension {}'.format(ipatch, idim)
                patchknotvalues.extend(dimknotvalues)
                patchknotmultiplicities.extend(dimknotmultiplicities)
            patchcoeffs, patchdofmap, patchdofcount = patch.topo._basis_spline(degree, knotvalues=patchknotvalues, knotmultiplicities=patchknotmultiplicities, continuity=continuity)
            coeffs.extend(patchcoeffs)
            dofmap.extend(types.frozenarray(dofs+dofcount, copy=False) for dofs in patchdofmap)
            if patchcontinuous:
                # reconstruct multidimensional dof structure
                dofs = dofcount + numpy.arange(numpy.prod(patchdofcount), dtype=int).reshape(patchdofcount)
                for boundary in patch.boundaries:
                    # get patch boundary dofs and reorder to canonical form
                    boundarydofs = boundary.apply_transform(dofs)[..., 0].ravel()
                    # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
                    commonboundarydofs.setdefault(boundary.id, []).append(boundarydofs)
            dofcount += numpy.prod(patchdofcount)

        if patchcontinuous:
            # build merge mapping: merge common boundary dofs (from low to high)
            pairs = itertools.chain(*(zip(*dofs) for dofs in commonboundarydofs.values() if len(dofs) > 1))
            merge = numpy.arange(dofcount)
            for dofs in sorted(pairs):
                merge[list(dofs)] = merge[list(dofs)].min()
            assert all(numpy.all(merge[a] == merge[b]) for a, *B in commonboundarydofs.values() for b in B), 'something went wrong is merging interface dofs; this should not have happened'
            # build renumber mapping: renumber remaining dofs consecutively, starting at 0
            remainder, renumber = numpy.unique(merge, return_inverse=True)
            # apply mappings
            dofmap = tuple(types.frozenarray(renumber[v], copy=False) for v in dofmap)
            dofcount = len(remainder)

        return function.PlainBasis(coeffs, dofmap, dofcount, self.f_index, self.f_coords)

    def basis_patch(self):
        'degree zero patchwise discontinuous basis'

        transforms = transformseq.PlainTransforms(tuple((patch.topo.root,) for patch in self.patches), self.ndims, self.ndims)
        index = function.transforms_index(self.space, transforms)
        coords = function.transforms_coords(self.space, transforms)
        return function.DiscontBasis([types.frozenarray(1, dtype=float).reshape(1, *(1,)*self.ndims)]*len(self.patches), index, coords)

    @property
    def boundary(self):
        'boundary'

        subtopos = []
        subnames = []
        for i, patch in enumerate(self.patches):
            for boundary in patch.boundaries:
                if boundary.id in self._patchinterfaces:
                    continue
                name = patch.topo._bnames[boundary.dim][boundary.side]
                subtopos.append(patch.topo.boundary[name])
                subnames.append('patch{}-{}'.format(i, name))
        if len(subtopos) == 0:
            return Topology.empty(self.ref_coord_system, self.ndims-1)
        else:
            return DisjointUnionTopology(subtopos, subnames)

    @property
    def interfaces(self):
        '''interfaces

        Return a topology with all element interfaces.  The patch interfaces are
        accessible via the group ``'interpatch'`` and the interfaces *inside* a
        patch via ``'intrapatch'``.
        '''

        intrapatchtopo = Topology.empty(self.ref_coord_system, self.ndims-1) if not self.patches else \
            DisjointUnionTopology(patch.topo.interfaces for patch in self.patches)

        btopos = []
        bconnectivity = []
        for boundaryid, patchdata in self._patchinterfaces.items():
            if len(patchdata) > 2:
                raise ValueError('Cannot create interfaces of multipatch topologies with more than two interface connections.')
            pairs = []
            references = None
            for topo, boundary in patchdata:
                btopo = topo.boundary[topo._bnames[boundary.dim][boundary.side]]
                if references is None:
                    references = numeric.asobjvector(btopo.references).reshape(btopo.shape)
                    references = references[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
                    references = boundary.apply_transform(references)[..., 0]
                    references = tuple(references.flat)
                transforms = numeric.asobjvector(btopo.transforms).reshape(btopo.shape)
                transforms = transforms[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
                transforms = boundary.apply_transform(transforms)[..., 0]
                pairs.append(tuple(transforms.flat))
            # create structured topology of joined element pairs
            references = References.from_iter(references, self.ndims-1)
            transforms, opposites = pairs
            transforms = transformseq.PlainTransforms(transforms, self.transforms.todims, self.ndims-1)
            opposites = transformseq.PlainTransforms(opposites, self.transforms.todims, self.ndims-1)
            btopos.append(TransformChainsTopology(self.space, references, transforms, opposites))
            bconnectivity.append(numpy.array(boundaryid).reshape((2,)*(self.ndims-1)))
        # create multipatch topology of interpatch boundaries
        interpatchtopo = MultipatchTopology(tuple(map(Patch, btopos, bconnectivity, self.build_boundarydata(bconnectivity))))

        return DisjointUnionTopology((intrapatchtopo, interpatchtopo), ('intrapatch', 'interpatch'))

    @property
    def connectivity(self):
        connectivity = []
        patchinterfaces = {}
        for patch in self.patches:  # len(connectivity) represents the element offset for the current patch
            ielems = numpy.arange(len(patch.topo)).reshape(patch.topo.shape) + len(connectivity)
            for boundary in patch.boundaries:
                patchinterfaces.setdefault(boundary.id, []).append((boundary.apply_transform(ielems)[..., 0], boundary.dim * 2 + (boundary.side == 0)))
            connectivity.extend(patch.topo.connectivity + len(connectivity) * numpy.not_equal(patch.topo.connectivity, -1))
        connectivity = numpy.array(connectivity)
        for patchdata in patchinterfaces.values():
            if len(patchdata) > 2:
                raise ValueError('Cannot create connectivity of multipatch topologies with more than two interface connections.')
            if len(patchdata) == 2:
                (ielem, iedge), (jelem, jedge) = patchdata
                assert ielem.shape == jelem.shape
                assert numpy.equal(connectivity[ielem, iedge], -1).all()
                assert numpy.equal(connectivity[jelem, jedge], -1).all()
                connectivity[ielem, iedge] = jelem
                connectivity[jelem, jedge] = ielem
        return types.frozenarray(connectivity, copy=False)

    @property
    def refined(self):
        'refine'

        return MultipatchTopology(Patch(patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches)

# vim:sw=2:sts=2:et
