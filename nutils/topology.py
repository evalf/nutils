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

from . import element, function, evaluable, _util as util, parallel, numeric, cache, transform, transformseq, warnings, types, points, sparse
from ._util import single_or_multiple
from ._backports import cached_property
from .elementseq import References
from .pointsseq import PointsSequence
from .sample import Sample

from dataclasses import dataclass
from functools import reduce
from os import environ
from typing import Any, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Sequence

import itertools
import numpy
import nutils_poly as poly
import operator
import treelog as log


_ = numpy.newaxis
_identity = lambda x: x
_ArgDict = Mapping[str, numpy.ndarray]


class Topology:
    '''topology base class

    Parameters
    ----------
    spaces : :class:`tuple` of :class:`str`
        The unique, ordered list of spaces on which this topology is defined.
    space_dims : :class:`tuple` of :class:`int`
        The dimension of each space in :attr:`spaces`.
    references : :class:`nutils.elementseq.References`
        The references.

    Attributes
    ----------
    spaces : :class:`tuple` of :class:`str`
        The unique, ordered list of spaces on which this topology is defined.
    space_dims : :class:`tuple` of :class:`int`
        The dimension of each space in :attr:`spaces`.
    references : :class:`nutils.elementseq.References`
        The references.
    ndims : :class:`int`
        The dimension of this topology.
    '''

    @staticmethod
    def empty(spaces: Iterable[str], space_dims: Iterable[int], ndims: int) -> 'Topology':
        '''Return an empty topology.

        Parameters
        ----------
        spaces : :class:`tuple` of :class:`str`
            The unique, ordered list of spaces on which the empty topology is defined.
        space_dims : :class:`tuple` of :class:`int`
            The dimension of each space in :attr:`spaces`.
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

        return _Empty(tuple(spaces), tuple(space_dims), ndims)

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

        return Topology.empty(self.spaces, self.space_dims, self.ndims)

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
            return reduce(_DisjointUnion, unempty)
        else:
            return empty

    def __init__(self, spaces: Sequence[str], space_dims: Sequence[int], references: References) -> None:
        assert isinstance(spaces, Sequence) and all(isinstance(space, str) for space in spaces), f'spaces={spaces!r}'
        assert isinstance(space_dims, Sequence) and all(isinstance(space_dim, int) for space_dim in space_dims), f'space_dims={space_dims!r}'
        assert isinstance(references, References), f'references={references!r}'
        self.spaces = tuple(spaces)
        self.space_dims = tuple(space_dims)
        self.references = references
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
        return _Take(self, __indices)

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
        elif numeric.isboolarray(item) and item.ndim == 1 and len(item) == len(self):
            return self.compress(item)
        else:
            raise NotImplementedError
        if not topo:
            raise KeyError(item)
        return topo

    def __mul__(self, other: Any) -> 'Topology':
        if isinstance(other, Topology):
            return _Mul(self, other)
        else:
            return NotImplemented

    def __and__(self, other: Any) -> 'Topology':
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.spaces != other.spaces or self.space_dims != other.space_dims or self.ndims != other.ndims:
            raise ValueError('The topologies must have the same spaces and dimensions.')
        elif not self or not other:
            return self.empty_like()
        else:
            return NotImplemented

    __rand__ = __and__

    def __or__(self, other: Any) -> 'Topology':
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.spaces != other.spaces or self.space_dims != other.space_dims or self.ndims != other.ndims:
            raise ValueError('The topologies must have the same spaces and dimensions.')
        elif not self:
            return other
        elif not other:
            return self
        else:
            return NotImplemented

    __ror__ = __or__

    @property
    def border_transforms(self) -> transformseq.Transforms:
        raise NotImplementedError

    @property
    def refine_iter(self) -> 'Topology':
        topo = self
        while True:
            yield topo
            topo = topo.refined

    @property
    def f_index(self) -> function.Array:
        '''The evaluable index of the element in this topology.'''

        raise NotImplementedError

    @property
    def f_coords(self) -> function.Array:
        '''The evaluable element local coordinates.'''

        raise NotImplementedError

    def basis(self, name: str, *args, **kwargs) -> function.Basis:
        '''
        Create a basis.
        '''
        if self.ndims == 0:
            return function.PlainBasis([[[1]]], [[0]], 1, self.f_index, self.f_coords)
        split = name.split('-', 1)
        if len(split) == 2 and split[0] in ('h', 'th'):
            name = split[1]  # default to non-hierarchical bases
            if split[0] == 'th':
                kwargs.pop('truncation_tolerance', None)
        f = getattr(self, 'basis_' + name)
        return f(*args, **kwargs)

    def sample(self, ischeme: str, degree: int) -> Sample:
        'Create sample.'

        raise NotImplementedError

    @single_or_multiple
    def integrate_elementwise(self, funcs: Iterable[function.Array], *, degree: int, asfunction: bool = False, ischeme: str = 'gauss', arguments: Optional[_ArgDict] = None) -> Union[List[numpy.ndarray], List[function.Array]]:
        'element-wise integration'

        retvals = [sparse.toarray(retval) for retval in self.sample(ischeme, degree).integrate_sparse(
            [function.kronecker(func, pos=self.f_index, length=len(self), axis=0) for func in funcs], arguments=arguments)]
        if asfunction:
            return [function.get(retval, 0, self.f_index) for retval in retvals]
        else:
            return retvals

    @single_or_multiple
    def elem_mean(self, funcs: Iterable[function.Array], geometry: Optional[function.Array] = None, ischeme: str = 'gauss', degree: Optional[int] = None, **kwargs) -> List[numpy.ndarray]:
        ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
        funcs = (1,)+funcs
        if geometry is not None:
            funcs = [func * function.J(geometry) for func in funcs]
        area, *integrals = self.integrate_elementwise(funcs, ischeme=ischeme, degree=degree, **kwargs)
        return [integral / area[(slice(None),)+(_,)*(integral.ndim-1)] for integral in integrals]

    @single_or_multiple
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

    def refined_by(self, refine: Union['Topology', Iterable[int], Iterable[Tuple[transform.TransformItem,...]]]) -> 'Topology':
        '''Create hierarchically refined topology by selectively refining
        elements.

        Parameters
        ----------
        refine : :class:`Topology` or iterable of :class:`int` or transformation chains
            The elements to refine, specified either as a subtopology or by
            their indices or locations in the topology.

        Returns
        -------
        :class:`Topology`
            The refined topology.
        '''

        if isinstance(refine, Topology):
            refine = refine.transforms
        elif not isinstance(refine, numpy.ndarray):
            # We convert refine to a tuple below both as a test for iterability
            # and to account for the possibility that it is a generator
            try:
                refine = tuple(refine)
            except:
                raise ValueError('refined_by expects an iterable argument') from None
        if len(refine) == 0:
            return self
        if isinstance(refine[0], tuple): # use first element for detection
            try:
                transforms = self.transforms
            except:
                raise TypeError('topology supports only refinement by element indices') from None
            refine = [transforms.index_with_tail(item)[0] for item in refine]
        refine = numpy.asarray(refine)
        if refine.dtype != int:
            raise ValueError(f'expected an array of dtype int, got {refine.dtype}')
        return self._refined_by(numpy.unique(refine))

    def _refined_by(self, refine: Iterable[int]) -> 'Topology':
        raise NotImplementedError

    @cached_property
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

        raise NotImplementedError

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

    def subset(self, topo: 'Topology', newboundary: Optional[Union[str, 'Topology']] = None, strict: bool = False) -> 'Topology':
        'intersection'

        raise NotImplementedError

    def withgroups(self, vgroups: Mapping[str, Union[str, 'Topology']] = {}, bgroups: Mapping[str, Union[str, 'Topology']] = {}, igroups: Mapping[str, Union[str, 'Topology']] = {}, pgroups: Mapping[str, Union[str, 'Topology']] = {}) -> 'Topology':
        if all(isinstance(v, str) for g in (vgroups, bgroups, igroups) for v in g.values()) and not pgroups:
            return _WithGroupAliases(self, types.frozendict(vgroups), types.frozendict(bgroups), types.frozendict(igroups))
        else:
            raise NotImplementedError

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

    def indicator(self, subtopo: Union[str, 'Topology']) -> 'Topology':
        '''Create an indicator function for a subtopology.'''

        raise NotImplementedError

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

        raise NotImplementedError

    @cached_property
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

    @cached_property
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


if environ.get('NUTILS_TENSORIAL', None) == 'test':  # pragma: nocover

    from unittest import SkipTest

    class _TensorialTopology(Topology):

        def __and__(self, other: Any) -> Topology:
            result = super().__and__(other)
            if type(self) == type(other) and result is NotImplemented:
                raise SkipTest('`{}` does not implement `Topology.__and__`'.format(type(self).__qualname__))
            return result

        def __rand__(self, other: Any) -> Topology:
            result = super().__and__(other)
            if result is NotImplemented:
                raise SkipTest('`{}` does not implement `Topology.__and__`'.format(type(self).__qualname__))
            return result

        def __sub__(self, other: Any) -> Topology:
            if type(self) == type(other):
                raise SkipTest('`{}` does not implement `Topology.__sub__`'.format(type(self).__qualname__))
            else:
                return NotImplemented

        def __rsub__(self, other: Any) -> Topology:
            if isinstance(other, Topology):
                raise SkipTest('`{}` does not implement `Topology.__sub__`'.format(type(self).__qualname__))
            else:
                return NotImplemented

        @property
        def space(self) -> str:
            raise SkipTest('`{}` does not implement `Topology.space`'.format(type(self).__qualname__))

        @property
        def transforms(self) -> transformseq.Transforms:
            raise SkipTest('`{}` does not implement `Topology.transforms`'.format(type(self).__qualname__))

        @property
        def opposites(self) -> transformseq.Transforms:
            raise SkipTest('`{}` does not implement `Topology.opposites`'.format(type(self).__qualname__))

        @property
        def border_transforms(self) -> transformseq.Transforms:
            raise SkipTest('`{}` does not implement `Topology.border_transforms`'.format(type(self).__qualname__))

        @property
        def f_index(self) -> function.Array:
            raise SkipTest('`{}` does not implement `Topology.f_index`'.format(type(self).__qualname__))

        @property
        def f_coords(self) -> function.Array:
            raise SkipTest('`{}` does not implement `Topology.f_coords`'.format(type(self).__qualname__))

        def refined_by(self, refine: Iterable[int]) -> Topology:
            raise SkipTest('`{}` does not implement `Topology.refined_by`'.format(type(self).__qualname__))

        def trim(self, levelset: function.Array, maxrefine: int, ndivisions: int = 8, name: str = 'trimmed', leveltopo: Optional[Topology] = None, *, arguments: Optional[_ArgDict] = None) -> Topology:
            raise SkipTest('`{}` does not implement `Topology.trim`'.format(type(self).__qualname__))

        def subset(self, topo: Topology, newboundary: Optional[Union[str, Topology]] = None, strict: bool = False) -> Topology:
            raise SkipTest('`{}` does not implement `Topology.subset`'.format(type(self).__qualname__))

        def withgroups(self, vgroups: Mapping[str, Union[str, Topology]] = {}, bgroups: Mapping[str, Union[str, Topology]] = {}, igroups: Mapping[str, Union[str, Topology]] = {}, pgroups: Mapping[str, Union[str, Topology]] = {}) -> Topology:
            try:
                return super().withgroups(vgroups, bgroups, igroups, pgroups)
            except NotImplementedError:
                raise SkipTest('`{}` does not implement `Topology.withgroups`'.format(type(self).__qualname__))

        def indicator(self, subtopo: Union[str, Topology]) -> Topology:
            raise SkipTest('`{}` does not implement `Topology.indicator`'.format(type(self).__qualname__))

        def locate(self, geom, coords, *, tol=0, eps=0, maxiter=0, arguments=None, weights=None, maxdist=None, ischeme=None, scale=None, skip_missing=False) -> Sample:
            raise SkipTest('`{}` does not implement `Topology.locate`'.format(type(self).__qualname__))

else:
    _TensorialTopology = Topology


class _EmptyUnlowerable(function.Array):

    def lower(self, args: function.LowerArgs) -> evaluable.Array:
        raise ValueError('cannot lower')


class _Empty(_TensorialTopology):

    def __init__(self, spaces: Sequence[str], space_dims: Sequence[int], ndims: int) -> None:
        super().__init__(spaces, space_dims, References.empty(ndims))

    def __invert__(self) -> Topology:
        return self

    @property
    def connectivity(self) -> Sequence[Sequence[int]]:
        return tuple()

    def indicator(self, subtopo: Union[str, Topology]) -> Topology:
        return function.zeros((), int)

    @property
    def f_index(self) -> function.Array:
        return _EmptyUnlowerable((), int, self.spaces, {})

    @property
    def f_coords(self) -> function.Array:
        return _EmptyUnlowerable((self.ndims,), float, self.spaces, {})

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return self

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _Empty(self.spaces, self.space_dims, self.ndims - 1)

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _Empty(self.spaces, self.space_dims, self.ndims - 1)

    def basis_std(self, degree: int, *args, **kwargs) -> function.Array:
        return function.zeros((0,))

    basis_spline = basis_std

    def sample(self, ischeme: str, degree: int) -> Sample:
        return Sample.empty(self.spaces, self.ndims)


class _DisjointUnion(_TensorialTopology):

    def __init__(self, topo1: Topology, topo2: Topology) -> None:
        if topo1.spaces != topo2.spaces or topo1.space_dims != topo2.space_dims or topo1.ndims != topo2.ndims:
            raise ValueError('The topologies must have the same spaces and dimensions.')
        self.topo1 = topo1
        self.topo2 = topo2
        super().__init__(topo1.spaces, topo1.space_dims, topo1.references + topo2.references)

    def __invert__(self) -> Topology:
        return Topology.disjoint_union(~self.topo1, ~self.topo2)

    def __and__(self, other: Any) -> Topology:
        if not isinstance(other, Topology):
            return NotImplemented
        elif self.spaces != other.spaces or self.space_dims != other.space_dims or self.ndims != other.ndims:
            raise ValueError('The topologies must have the same spaces and dimensions.')
        else:
            return Topology.disjoint_union(self.topo1 & other, self.topo2 & other)

    __rand__ = __and__

    @cached_property
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

    @single_or_multiple
    def integrate_elementwise(self, funcs: Iterable[function.Array], *, degree: int, asfunction: bool = False, ischeme: str = 'gauss', arguments: Optional[_ArgDict] = None) -> Union[List[numpy.ndarray], List[function.Array]]:
        return list(map(numpy.concatenate, zip(*(topo.integrate_elementwise(funcs, degree=degree, ischeme=ischeme, arguments=arguments) for topo in (self.topo1, self.topo2)))))

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.topo1.sample(ischeme, degree) + self.topo2.sample(ischeme, degree)

    def trim(self, levelset: function.Array, maxrefine: int, ndivisions: int = 8, name: str = 'trimmed', leveltopo: Optional[Topology] = None, *, arguments: Optional[_ArgDict] = None) -> Topology:
        if leveltopo is not None:
            return super().trim(levelset, maxrefine, ndivisions, name, leveltopo, arguments=arguments)
        else:
            topo1 = self.topo1.trim(levelset, maxrefine, ndivisions, name, arguments=arguments)
            topo2 = self.topo2.trim(levelset, maxrefine, ndivisions, name, arguments=arguments)
            return Topology.disjoint_union(topo1, topo2)

    def select(self, indicator: function.Array, ischeme: str = 'bezier2', **kwargs: numpy.ndarray) -> Topology:
        topo1 = self.topo1.select(indicator, ischeme, **kwargs)
        topo2 = self.topo2.select(indicator, ischeme, **kwargs)
        return Topology.disjoint_union(topo1, topo2)


class _Mul(_TensorialTopology):

    def __init__(self, topo1: Topology, topo2: Topology) -> None:
        if not set(topo1.spaces).isdisjoint(topo2.spaces):
            raise ValueError('Cannot multiply two topologies (partially) defined on the same spaces.')
        self.topo1 = topo1
        self.topo2 = topo2
        super().__init__(topo1.spaces + topo2.spaces, topo1.space_dims + topo2.space_dims, topo1.references * topo2.references)

    def __invert__(self) -> Topology:
        return ~self.topo1 * ~self.topo2

    @property
    def f_index(self) -> function.Array:
        return self.topo1.f_index * len(self.topo2) + self.topo2.f_index

    @property
    def f_coords(self) -> function.Array:
        return numpy.concatenate([self.topo1.f_coords, self.topo2.f_coords])

    @cached_property
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
            groups = subtopo.split(',')
            hassub1 = bool(self.topo1.get_groups(*groups))
            hassub2 = bool(self.topo2.get_groups(*groups))
            if hassub1 and hassub2:
                raise NotImplementedError
            elif hassub1:
                return self.topo1.indicator(subtopo)
            elif hassub2:
                return self.topo2.indicator(subtopo)
            else:
                return function.zeros((), int)
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


class _Take(_TensorialTopology):

    def __init__(self, parent: Topology, indices: numpy.ndarray) -> None:
        assert isinstance(parent, Topology), f'parent={parent!r}'
        assert isinstance(indices, numpy.ndarray), f'indices={indices!r}'
        self.parent = parent
        self.indices = indices = types.frozenarray(indices)
        assert indices.ndim == 1 and indices.size
        assert numpy.greater(indices[1:], indices[:-1]).all()
        assert 0 <= indices[0] and indices[-1] < len(self.parent)
        super().__init__(parent.spaces, parent.space_dims, parent.references.take(self.indices))

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.parent.sample(ischeme, degree).take_elements(self.indices)


class _WithGroupAliases(_TensorialTopology):

    def __init__(self, parent: Topology, vgroups: Mapping[str, str] = {}, bgroups: Mapping[str, str] = {}, igroups: Mapping[str, str] = {}) -> None:
        self.parent = parent
        self.vgroups = vgroups
        self.bgroups = bgroups
        self.igroups = igroups
        super().__init__(parent.spaces, parent.space_dims, parent.references)

    def _rewrite_groups(self, groups: Iterable[str]) -> Iterator[str]:
        for group in groups:
            if group in self.vgroups:
                yield from self.vgroups[group].split(',')
            else:
                yield group

    def get_groups(self, *groups: str) -> Topology:
        return self.parent.get_groups(*self._rewrite_groups(groups))

    def take_unchecked(self, indices: numpy.ndarray) -> Topology:
        # NOTE: the groups are gone after take
        return self.parent.take_unchecked(indices)

    def slice_unchecked(self, indices: slice, idim: int) -> Topology:
        # NOTE: the groups are gone after take
        return self.parent.slice_unchecked(indices, idim)

    @property
    def f_index(self) -> function.Array:
        return self.parent.f_index

    @property
    def f_coords(self) -> function.Array:
        return self.parent.f_coords

    @property
    def connectivity(self) -> Sequence[Sequence[int]]:
        return self.parent.connectivity

    def basis(self, name: str, *args, **kwargs) -> function.Basis:
        return self.parent.basis(name, *args, **kwargs)

    def sample(self, ischeme: str, degree: int) -> Sample:
        return self.parent.sample(ischeme, degree)

    def refine_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _WithGroupAliases(self.parent.refine_spaces(spaces), self.vgroups, self.bgroups, self.igroups)

    def indicator(self, subtopo: Union[str, Topology]) -> Topology:
        if isinstance(subtopo, str):
            return self.parent.indicator(','.join(self._rewrite_groups(subtopo.split(','))))
        else:
            return super().indicator(subtopo)

    def locate(self, geom, coords, *, tol=0, eps=0, maxiter=0, arguments=None, weights=None, maxdist=None, ischeme=None, scale=None, skip_missing=False) -> Sample:
        return self.parent.locate(geom, coords, tol=tol, eps=eps, maxiter=maxiter, arguments=arguments, weights=weights, maxdist=maxdist, ischeme=ischeme, scale=scale, skip_missing=skip_missing)

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _WithGroupAliases(self.parent.boundary_spaces_unchecked(spaces), self.bgroups, types.frozendict({}), types.frozendict({}))

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> Topology:
        return _WithGroupAliases(self.parent.interfaces_spaces_unchecked(spaces), self.igroups, types.frozendict({}), types.frozendict({}))


class TransformChainsTopology(Topology):
    'base class for topologies with transform chains'

    def __init__(self, space: str, references: References, transforms: transformseq.Transforms, opposites: transformseq.Transforms):
        assert isinstance(space, str), f'space={space!r}'
        assert isinstance(references, References), f'references={references!r}'
        assert isinstance(transforms, transformseq.Transforms), f'transforms={transforms!r}'
        assert isinstance(opposites, transformseq.Transforms), f'opposites={opposites!r}'
        assert transforms.todims == opposites.todims
        assert references.ndims == opposites.fromdims == transforms.fromdims
        assert len(references) == len(transforms) == len(opposites)
        self.space = space
        self.transforms = transforms
        self.opposites = opposites
        super().__init__((space,), (transforms.todims,), references)

    def empty_like(self) -> 'TransformChainsTopology':
        return EmptyTopology(self.space, self.transforms.todims, self.ndims)

    def get_groups(self, *groups):
        return self.empty_like()

    def take_unchecked(self, indices: numpy.ndarray) -> 'TransformChainsTopology':
        indices = types.frozenarray(indices, dtype=int)
        return TransformChainsTopology(self.space, self.references.take(indices), self.transforms[indices], self.opposites[indices])

    def __invert__(self):
        return OppositeTopology(self)

    def __or__(self, other):
        if not isinstance(other, TransformChainsTopology) or other.space != self.space and other.ndims != self.ndims:
            return super().__or__(other)
        return other if not self \
            else self if not other \
            else NotImplemented if isinstance(other, UnionTopology) \
            else UnionTopology((self, other))

    __ror__ = lambda self, other: self.__or__(other)

    def __and__(self, other):
        if not isinstance(other, TransformChainsTopology) or other.space != self.space:
            return super().__and__(other)
        keep_self = numpy.array(list(map(other.transforms.contains_with_tail, self.transforms)), dtype=bool)
        if keep_self.all():
            return self
        keep_other = numpy.array(list(map(self.transforms.contains_with_tail, other.transforms)), dtype=bool)
        if keep_other.all():
            return other
        ind_self = types.frozenarray(keep_self.nonzero()[0], copy=False)
        ind_other = types.frozenarray([i for i, trans in enumerate(other.transforms) if keep_other[i] and not self.transforms.contains(trans)], dtype=int)
        # The last condition is to avoid duplicate elements. Note that we could
        # have reused the result of an earlier lookup to avoid a new (using index
        # instead of contains) but we choose to trade some speed for simplicity.
        references = self.references.take(ind_self).chain(other.references.take(ind_other))
        transforms = transformseq.chain([self.transforms[ind_self], other.transforms[ind_other]], self.transforms.todims, self.ndims)
        opposites = transformseq.chain([self.opposites[ind_self], other.opposites[ind_other]], self.transforms.todims, self.ndims)
        return TransformChainsTopology(self.space, references, transforms, opposites)

    __rand__ = lambda self, other: self.__and__(other)

    def __add__(self, other):
        return self | other

    def __sub__(self, other):
        assert isinstance(other, TransformChainsTopology) and other.space == self.space and other.ndims == self.ndims
        return other.__rsub__(self)

    def __rsub__(self, other):
        assert isinstance(other, TransformChainsTopology) and other.space == self.space and other.ndims == self.ndims
        return other - other.subset(self, newboundary=getattr(self, 'boundary', None))

    @cached_property
    def border_transforms(self):
        indices = set()
        for btrans in self.boundary.transforms:
            try:
                ielem, tail = self.transforms.index_with_tail(btrans)
            except ValueError:
                pass
            else:
                indices.add(ielem)
        return self.transforms[numpy.array(sorted(indices), dtype=int)]

    @property
    def _index_coords(self):
        index = function.transforms_index(self.space, self.transforms)
        coords = function.transforms_coords(self.space, self.transforms)
        return index, coords

    @property
    def f_index(self):
        return self._index_coords[0]

    @property
    def f_coords(self):
        return self._index_coords[1]

    def sample(self, ischeme, degree):
        'Create sample.'

        points = PointsSequence.from_iter((ischeme(reference, degree) for reference in self.references), self.ndims) if callable(ischeme) \
            else self.references.getpoints(ischeme, degree)
        transforms = self.transforms,
        if len(self.transforms) == 0 or self.opposites != self.transforms:
            transforms += self.opposites,
        return Sample.new(self.space, transforms, points)

    def _refined_by(self, refine):
        fine = self.refined.transforms
        indices0 = numpy.setdiff1d(numpy.arange(len(self)), refine, assume_unique=True)
        indices1 = numpy.array([fine.index((*self.transforms[i], ctrans))
            for i in refine for ctrans, cref in self.references[i].children if cref])
        indices1.sort()
        return HierarchicalTopology(self, (indices0, indices1))

    @cached_property
    def refined(self):
        return RefinedTopology(self)

    def refine_spaces_unchecked(self, spaces: Iterable[str]) -> 'TransformChainsTopology':
        # Since every `TransformChainsTopology` has exactly one space, we implement
        # `refine_spaces` here for all subclasses and return `self.refined` if the
        # space of this topology is in the given `spaces`. Subclasses can redefine
        # the `refined` property.
        if not spaces:
            return self
        return self.refined

    def refine(self, n):
        if numpy.iterable(n):
            assert len(n) == self.ndims
            assert all(ni == n[0] for ni in n)
            n = n[0]
        return self if n <= 0 else self.refined.refine(n-1)

    def trim(self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None, *, arguments=None):
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
            coordinates = evaluable.Points(evaluable.NPoints(), evaluable.constant(self.ndims))
            ielem = evaluable.Argument('_leveltopo_ielem', (), int)
            levelset = levelset.lower(function.LowerArgs.for_space(self.space, (leveltopo.transforms, leveltopo.opposites), ielem, coordinates)).optimized_for_numpy
            bins = [set() for ielem in range(len(self))]
            for trans in leveltopo.transforms:
                ielem, tail = self.transforms.index_with_tail(trans)
                bins[ielem].add(tail)
            fcache = cache.WrapperCache()
            with log.iter.percentage('trimming', self.references, self.transforms, bins) as items:
                for ref, trans, ctransforms in items:
                    levels = numpy.empty(ref._nlinear_by_level(maxrefine))
                    cover = list(fcache[ref._linear_cover](frozenset(ctransforms), maxrefine))
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

    def subset(self, topo, newboundary=None, strict=False):
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
            return EmptyTopology(self.space, self.transforms.todims, self.ndims)
        return SubsetTopology(self, refs, newboundary)

    def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
        return WithGroupsTopology(self, vgroups, bgroups, igroups, pgroups) if vgroups or bgroups or igroups or pgroups else self

    def indicator(self, subtopo):
        if isinstance(subtopo, str):
            subtopo = self[subtopo]
        values = numpy.zeros([len(self)], dtype=int)
        values[numpy.fromiter(map(self.transforms.index, subtopo.transforms), dtype=int)] = 1
        return function.get(values, 0, self.f_index)

    @log.withcontext
    def locate(self, geom, coords, *, tol=0, eps=0, maxiter=0, arguments=None, weights=None, maxdist=None, ischeme=None, scale=None, skip_missing=False):
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
        if skip_missing and weights is not None:
            raise ValueError('weights and skip_missing are mutually exclusive')
        arguments = dict(arguments or ())
        centroids = self.sample('_centroid', None).eval(geom, **arguments)
        assert len(centroids) == len(self)
        ielems = parallel.shempty(len(coords), dtype=int)
        points = parallel.shempty((len(coords), len(geom)), dtype=float)
        _ielem = evaluable.Argument('_locate_ielem', shape=(), dtype=int)
        _point = evaluable.Argument('_locate_point', shape=(evaluable.constant(self.ndims),))
        egeom = geom.lower(function.LowerArgs.for_space(self.space, (self.transforms, self.opposites), _ielem, _point))
        xJ = evaluable.Tuple((egeom, evaluable.derivative(egeom, _point))).simplified
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
                    ielems[ipoint] = -1 # mark point as missing
                    if not skip_missing:
                        # rather than raising LocateError here, which
                        # parallel.fork will reraise as a general Exception if
                        # ipoint was assigned to a child process, we fast
                        # forward through the remaining points to abandon the
                        # loop and subsequently raise from the main process.
                        for ipoint in ipoints:
                            pass
        if -1 not in ielems: # all points are found
            return self._sample(ielems, points, weights)
        elif skip_missing: # not all points are found and that's ok, we just leave those out
            return self._sample(ielems[ielems != -1], points[ielems != -1])
        else: # not all points are found and that's an error
            raise LocateError(f'failed to locate point: {coords[ielems==-1][0]}')

    def _sample(self, ielems, coords, weights=None):
        index = numpy.argsort(ielems, kind='stable')
        sorted_ielems = ielems[index]
        offsets = [0, *(sorted_ielems[:-1] != sorted_ielems[1:]).nonzero()[0]+1, len(index)]

        unique_ielems = sorted_ielems[offsets[:-1]]
        transforms = self.transforms[unique_ielems],
        if len(self.transforms) == 0 or self.opposites != self.transforms:
            transforms += self.opposites[unique_ielems],

        slices = [index[n:m] for n, m in zip(offsets[:-1], offsets[1:])]
        points_ = PointsSequence.from_iter([points.CoordsPoints(types.arraydata(coords[s])) for s in slices] if weights is None
            else [points.CoordsWeightsPoints(types.arraydata(coords[s]), types.arraydata(weights[s])) for s in slices], self.ndims)

        return Sample.new(self.space, transforms, points_, index)

    def boundary_spaces_unchecked(self, spaces: FrozenSet[str]) -> 'TransformChainsTopology':
        return self.boundary

    @cached_property
    @log.withcontext
    def boundary(self):
        references = []
        selection = []
        iglobaledgeiter = itertools.count()
        refs_touched = False
        for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
            for (edgetrans, edgeref), ioppelem, iglobaledge in zip(elemref.edges, ioppelems, iglobaledgeiter):
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
        transforms = self.transforms.edges(self.references)[selection]
        return TransformChainsTopology(self.space, references, transforms, transforms)

    def interfaces_spaces_unchecked(self, spaces: FrozenSet[str]) -> 'TransformChainsTopology':
        return self.interfaces

    @cached_property
    @log.withcontext
    def interfaces(self):
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
        # To merge matching dofs we loop over the connectivity table to find
        # neighbouring elements (limited to jelem > ielem to consider every
        # neighbour pair exactly once as well as ignore exterior boundaries)
        # and mark the degrees of freedom on both sides to be equal.
        dofmap, ndofs = util.merge_index_map(offsets[-1], (merge_set
            for ielem, ioppelems in enumerate(self.connectivity)
                for iedge, jelem in enumerate(ioppelems) if jelem >= ielem
                    for merge_set in zip(
                        offsets[ielem] + self.references[ielem].get_edge_dofs(degree, iedge),
                        offsets[jelem] + self.references[jelem].get_edge_dofs(degree, util.index(self.connectivity[jelem], ielem)))))

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


class WithGroupsTopology(TransformChainsTopology):
    'item topology'

    def __init__(self, basetopo: Topology, vgroups: Mapping[str, Union[str, Topology]] = {}, bgroups: Mapping[str, Union[str, Topology]] = {}, igroups: Mapping[str, Union[str, Topology]] = {}, pgroups: Mapping[str, Union[str, Topology]] = {}):
        assert isinstance(basetopo, Topology), f'basetopo={basetopo!r}'
        assert isinstance(vgroups, Mapping) and all(isinstance(key, str) and (isinstance(value, str) or isinstance(value, Topology) and value.ndims == basetopo.ndims) for key, value in vgroups.items()), f'vgroups={vgroups!r}'
        assert isinstance(bgroups, Mapping) and all(isinstance(key, str) and (isinstance(value, str) or isinstance(value, Topology) and value.ndims == basetopo.ndims-1) for key, value in bgroups.items()), f'bgroups={bgroups!r}'
        assert isinstance(igroups, Mapping) and all(isinstance(key, str) and (isinstance(value, str) or isinstance(value, Topology) and value.ndims == basetopo.ndims-1) for key, value in igroups.items()), f'igroups={igroups!r}'
        assert isinstance(pgroups, Mapping) and all(isinstance(key, str) and (isinstance(value, str) or isinstance(value, Topology) and value.ndims == 0) for key, value in pgroups.items()), f'pgroups={pgroups!r}'
        assert vgroups or bgroups or igroups or pgroups
        self.basetopo = basetopo
        self.vgroups = types.frozendict(vgroups)
        self.bgroups = types.frozendict(bgroups)
        self.igroups = types.frozendict(igroups)
        self.pgroups = types.frozendict(pgroups)
        super().__init__(basetopo.space, basetopo.references, basetopo.transforms, basetopo.opposites)
        assert all(topo is Ellipsis or isinstance(topo, str) or isinstance(topo, TransformChainsTopology) and topo.ndims == basetopo.ndims for topo in self.vgroups.values())

    def __len__(self):
        return len(self.basetopo)

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        topos = []
        basegroups = []
        for group in groups:
            if group in self.vgroups:
                item = self.vgroups[group]
                assert isinstance(item, (TransformChainsTopology, str))
                if isinstance(item, TransformChainsTopology):
                    topos.append(item)
                else:
                    basegroups.extend(item.split(','))
            else:
                basegroups.append(group)
        if basegroups:
            topos.append(self.basetopo.get_groups(*basegroups))
        return reduce(operator.or_, topos, self.empty_like())

    def take_unchecked(self, __indices: numpy.ndarray) -> TransformChainsTopology:
        return self.basetopo.take_unchecked(__indices)

    def slice_unchecked(self, __s: slice, __idim: int) -> TransformChainsTopology:
        return self.basetopo.slice_unchecked(__s, __idim)

    @property
    def border_transforms(self):
        return self.basetopo.border_transforms

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
            if isinstance(topo, TransformChainsTopology):
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
        while isinstance(topo, WithGroupsTopology):
            for pname, ptopo in topo.pgroups.items():
                if pname not in pnames:
                    pnames.append(pname)
                    ptopos.append(ptopo)
            topo = topo.basetopo
        return UnionTopology(ptopos, pnames)

    def basis(self, name, *args, **kwargs):
        return self.basetopo.basis(name, *args, **kwargs)

    @cached_property
    def refined(self):
        groups = [{name: topo.refined if isinstance(topo, TransformChainsTopology) else topo for name, topo in groups.items()} for groups in (self.vgroups, self.bgroups, self.igroups, self.pgroups)]
        return self.basetopo.refined.withgroups(*groups)

    def locate(self, geom, coords, **kwargs):
        return self.basetopo.locate(geom, coords, **kwargs)


class OppositeTopology(TransformChainsTopology):
    'opposite topology'

    def __init__(self, basetopo):
        self.basetopo = basetopo
        super().__init__(basetopo.space, basetopo.references, basetopo.opposites, basetopo.transforms)

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        return ~(self.basetopo.get_groups(*groups))

    def take_unchecked(self, __indices: numpy.ndarray) -> TransformChainsTopology:
        return ~(self.basetopo.take_unchecked(__indices))

    def slice_unchecked(self, __s: slice, __idim: int) -> TransformChainsTopology:
        return ~(self.basetopo.slice_unchecked(__s, __idim))

    def __len__(self):
        return len(self.basetopo)

    def __invert__(self):
        return self.basetopo


class EmptyTopology(TransformChainsTopology):
    'empty topology'

    def __init__(self, space: str, todims: int, fromdims: int):
        assert isinstance(space, str), f'space={space!r}'
        assert isinstance(todims, int), f'todims={todims!r}'
        assert isinstance(fromdims, int), f'fromdims={fromdims!r}'
        super().__init__(space, References.empty(fromdims), transformseq.EmptyTransforms(todims, fromdims), transformseq.EmptyTransforms(todims, fromdims))

    def __or__(self, other):
        if self.space != other.space or self.ndims != other.ndims:
            return NotImplemented
        return other

    def __rsub__(self, other):
        return other


def StructuredLine(space, root: transform.TransformItem, i: int, j: int, periodic: bool = False, bnames: Optional[Tuple[str, str]] = None):
    assert isinstance(i, int), f'i={i!r}'
    assert isinstance(j, int), f'j={j!r}'
    assert isinstance(periodic, bool), f'periodic={periodic!r}'
    assert bnames is None or isinstance(bnames, Sequence) and len(bnames) == 2 and all(isinstance(bname, str) for bname in bnames), f'bnames={bnames!r}'
    if bnames is None:
        bnames = '_structured_line_dummy_boundary_left', '_structured_line_dummy_boundary_right'
    return StructuredTopology(space, root, axes=(transformseq.DimAxis(i, j, j if periodic else 0, periodic),), nrefine=0, bnames=(tuple(bnames),))


class StructuredTopology(TransformChainsTopology):
    'structured topology'

    def __init__(self, space: str, root: transform.TransformItem, axes: Sequence[transformseq.Axis], nrefine: int = 0, bnames: Sequence[Tuple[str, str]] = (('left', 'right'), ('bottom', 'top'), ('front', 'back'))):
        assert isinstance(space, str), f'space={space!r}'
        assert isinstance(root, transform.TransformItem), f'root={root!r}'
        assert isinstance(axes, Sequence) and all(isinstance(axis, transformseq.Axis) for axis in axes), f'axes={axes!r}'
        assert isinstance(nrefine, int), f'nrefine={nrefine!r}'
        assert isinstance(bnames, Sequence) and all(isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(name, str) for name in pair) for pair in bnames), f'bnames={bnames!r}'

        self.root = root
        self.axes = tuple(axes)
        self.nrefine = nrefine.__index__()
        self.shape = tuple(axis.j - axis.i for axis in self.axes if axis.isdim)
        self._bnames = tuple(bnames)

        references = References.uniform(util.product(element.getsimplex(1 if axis.isdim else 0) for axis in self.axes), len(self))
        transforms = transformseq.StructuredTransforms(self.root, self.axes, self.nrefine)
        nbounds = len(self.axes) - len(self.shape)
        if nbounds == 0:
            opposites = transforms
        else:
            axes = tuple(axis.opposite(nbounds-1) for axis in self.axes)
            opposites = transformseq.StructuredTransforms(self.root, axes, self.nrefine)

        super().__init__(space, references, transforms, opposites)

    def __repr__(self):
        return '{}<{}>'.format(type(self).__qualname__, 'x'.join(str(axis.j-axis.i)+('p' if axis.isperiodic else '') for axis in self.axes if axis.isdim))

    def __len__(self):
        return numpy.prod(self.shape, dtype=int)

    def slice_unchecked(self, indices: slice, idim: int) -> TransformChainsTopology:
        if indices == slice(None):
            return self
        axes = []
        for axis in self.axes:
            if axis.isdim:
                if idim == 0:
                    axis = axis.getitem(indices)
                idim -= 1
            axes.append(axis)
        return StructuredTopology(self.space, self.root, axes, self.nrefine, bnames=self._bnames)

    @property
    def periodic(self):
        dimaxes = (axis for axis in self.axes if axis.isdim)
        return tuple(idim for idim, axis in enumerate(dimaxes) if axis.isdim and axis.isperiodic)

    @cached_property
    def connectivity(self):
        connectivity = numpy.empty(self.shape+(self.ndims, 2), dtype=int)
        connectivity[...] = -1
        ielems = numpy.arange(len(self)).reshape(self.shape)
        for idim in range(self.ndims):
            s = (slice(None),)*idim
            s1 = s + (slice(1, None),)
            s2 = s + (slice(0, -1),)
            connectivity[s2+(..., idim, 0)] = ielems[s1]
            connectivity[s1+(..., idim, 1)] = ielems[s2]
            if idim in self.periodic:
                connectivity[s+(-1, ..., idim, 0)] = ielems[s+(0,)]
                connectivity[s+(0, ..., idim, 1)] = ielems[s+(-1,)]
        return types.frozenarray(connectivity.reshape(len(self), self.ndims*2), copy=False)

    @cached_property
    def boundary(self):
        'boundary'

        nbounds = len(self.axes) - self.ndims
        btopos = [StructuredTopology(self.space, root=self.root, axes=self.axes[:idim] + (bndaxis,) + self.axes[idim+1:], nrefine=self.nrefine, bnames=self._bnames)
                  for idim, axis in enumerate(self.axes)
                  for bndaxis in axis.boundaries(nbounds)]
        if not btopos:
            return EmptyTopology(self.space, self.transforms.todims, self.ndims-1)
        bnames = [bname for bnames, axis in zip(self._bnames, self.axes) if axis.isdim and not axis.isperiodic for bname in bnames]
        return DisjointUnionTopology(btopos, bnames)

    @cached_property
    def interfaces(self):
        'interfaces'

        assert self.ndims > 0, 'zero-D topology has no interfaces'
        itopos = []
        nbounds = len(self.axes) - self.ndims
        for idim, axis in enumerate(self.axes):
            if not axis.isdim:
                continue
            axes = (*self.axes[:idim], axis.intaxis(nbounds, side=True), *self.axes[idim+1:])
            oppaxes = (*self.axes[:idim], axis.intaxis(nbounds, side=False), *self.axes[idim+1:])
            itransforms = transformseq.StructuredTransforms(self.root, axes, self.nrefine)
            iopposites = transformseq.StructuredTransforms(self.root, oppaxes, self.nrefine)
            ireferences = References.uniform(util.product(element.getsimplex(1 if a.isdim else 0) for a in axes), len(itransforms))
            itopos.append(TransformChainsTopology(self.space, ireferences, itransforms, iopposites))
        assert len(itopos) == self.ndims
        return DisjointUnionTopology(itopos, names=['dir{}'.format(idim) for idim in range(self.ndims)])

    def _basis_spline(self, degree, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
        'spline with structure information'

        if periodic is None:
            periodic = self.periodic

        if numeric.isint(degree):
            degree = [degree]*self.ndims

        assert len(degree) == self.ndims

        if knotvalues is None or isinstance(knotvalues[0], (int, float)):
            knotvalues = [knotvalues] * self.ndims
        else:
            assert len(knotvalues) == self.ndims

        if knotmultiplicities is None or isinstance(knotmultiplicities[0], int):
            knotmultiplicities = [knotmultiplicities] * self.ndims
        else:
            assert len(knotmultiplicities) == self.ndims

        if not numpy.iterable(continuity):
            continuity = [continuity] * self.ndims
        else:
            assert len(continuity) == self.ndims

        vertex_structure = numpy.array(0)
        stdelems = []
        dofshape = []
        slices = []
        cache = {}
        for idim in range(self.ndims):
            p = degree[idim]
            n = self.shape[idim]
            isperiodic = idim in periodic

            c = continuity[idim]
            if c < 0:
                c += p
            assert -1 <= c < p

            k = knotvalues[idim]
            if k is None:  # Defaults to uniform spacing
                k = numpy.arange(n+1)
            else:
                k = numpy.array(k)
                while len(k) < n+1:
                    k_ = numpy.empty(len(k)*2-1)
                    k_[::2] = k
                    k_[1::2] = (k[:-1] + k[1:]) / 2
                    k = k_
                assert len(k) == n+1, 'knot values do not match the topology size'

            m = knotmultiplicities[idim]
            if m is None:  # Defaults to open spline without internal repetitions
                m = numpy.repeat(p-c, n+1)
                if not isperiodic:
                    m[0] = m[-1] = p+1
            else:
                m = numpy.array(m)
                assert min(m) > 0 and max(m) <= p+1, 'incorrect multiplicity encountered'
                while len(m) < n+1:
                    m_ = numpy.empty(len(m)*2-1, dtype=int)
                    m_[::2] = m
                    m_[1::2] = p-c
                    m = m_
                assert len(m) == n+1, 'knot multiplicity do not match the topology size'

            if not isperiodic:
                nd = sum(m)-p-1
                npre = p+1-m[0]  # Number of knots to be appended to front
                npost = p+1-m[-1]  # Number of knots to be appended to rear
                m[0] = m[-1] = p+1
            else:
                assert m[0] == m[-1], 'Periodic spline multiplicity expected'
                assert m[0] < p+1, 'Endpoint multiplicity for periodic spline should be p or smaller'

                nd = sum(m[:-1])
                npre = npost = 0
                k = numpy.concatenate([k[-p-1:-1]+k[0]-k[-1], k, k[1:1+p]-k[0]+k[-1]])
                m = numpy.concatenate([m[-p-1:-1], m, m[1:1+p]])

            km = numpy.array([ki for ki, mi in zip(k, m) for cnt in range(mi)], dtype=float)
            assert len(km) == sum(m)
            assert nd > 0, 'No basis functions defined. Knot vector too short.'

            stdelems_i = []
            slices_i = []
            offsets = numpy.cumsum(m[:-1])-p
            if isperiodic:
                offsets = offsets[p:-p]
            offset0 = offsets[0]+npre

            for offset in offsets:
                start = max(offset0-offset, 0)  # Zero unless prepending influence
                stop = p+1-max(offset-offsets[-1]+npost, 0)  # Zero unless appending influence
                slices_i.append(slice(offset-offset0+start, offset-offset0+stop))
                lknots = km[offset:offset+2*p] - km[offset]  # Copy operation required
                if p:  # Normalize for optimized caching
                    lknots /= lknots[-1]
                key = (tuple(numeric.round(lknots*numpy.iinfo(numpy.int32).max)), p)
                try:
                    coeffs = cache[key]
                except KeyError:
                    coeffs = cache[key] = self._localsplinebasis(lknots)
                stdelems_i.append(coeffs[start:stop])
            stdelems.append(stdelems_i)

            numbers = numpy.arange(nd)
            if isperiodic:
                numbers = numpy.concatenate([numbers, numbers[:p]])
            vertex_structure = vertex_structure[..., _]*nd+numbers
            dofshape.append(nd)
            slices.append(slices_i)

        # Cache effectivity
        log.debug('Local knot vector cache effectivity: {}'.format(100*(1.-len(cache)/float(sum(self.shape)))))

        # deduplicate stdelems and compute tensorial products `unique` with indices `index`
        # such that unique[index[i,j]] == poly_outer_product(stdelems[0][i], stdelems[1][j])
        index = numpy.array(0)
        for dim, stdelems_i in enumerate(stdelems):
            unique_i, index_i = util.unique(stdelems_i, key=types.arraydata)
            unique = unique_i if not index.ndim \
                else [poly.mul_different_vars(a[:,None], b[None], dim, 1).reshape(a.shape[0] * b.shape[0], -1) for a in unique for b in unique_i]
            index = index[..., _] * len(unique_i) + index_i

        coeffs = [unique[i] for i in index.flat]
        dofmap = [types.frozenarray(vertex_structure[S].ravel(), copy=False) for S in itertools.product(*slices)]
        return coeffs, dofmap, dofshape

    def basis_spline(self, degree, removedofs=None, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
        'spline basis'

        if removedofs is None or isinstance(removedofs[0], int):
            removedofs = [removedofs] * self.ndims
        else:
            assert len(removedofs) == self.ndims

        if periodic is None:
            periodic = self.periodic

        if numeric.isint(degree):
            degree = [degree]*self.ndims

        assert len(degree) == self.ndims

        if knotvalues is None or isinstance(knotvalues[0], (int, float)):
            knotvalues = [knotvalues] * self.ndims
        else:
            assert len(knotvalues) == self.ndims

        if knotmultiplicities is None or isinstance(knotmultiplicities[0], int):
            knotmultiplicities = [knotmultiplicities] * self.ndims
        else:
            assert len(knotmultiplicities) == self.ndims

        if not numpy.iterable(continuity):
            continuity = [continuity] * self.ndims
        else:
            assert len(continuity) == self.ndims

        start_dofs = []
        stop_dofs = []
        dofshape = []
        coeffs = []
        cache = {}
        for idim in range(self.ndims):
            p = degree[idim]
            n = self.shape[idim]

            c = continuity[idim]
            if c < 0:
                c += p
            assert -1 <= c < p

            k = knotvalues[idim]
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

            m = knotmultiplicities[idim]
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

            if idim in periodic and not m[0] == m[n] == p+1:  # if m[0] == m[n] == p+1 the spline is discontinuous at the boundary
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

        transforms_shape = tuple(axis.j-axis.i for axis in self.axes if axis.isdim)
        func = function.StructuredBasis(coeffs, start_dofs, stop_dofs, dofshape, transforms_shape, self.f_index, self.f_coords)
        if not any(removedofs):
            return func

        mask = numpy.ones((), dtype=bool)
        for idofs, ndofs in zip(removedofs, dofshape):
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

        return types.frozenarray([Ni.coeffs for Ni in N])

    def basis_std(self, *args, **kwargs):
        return __class__.basis_spline(self, *args, continuity=0, **kwargs)

    def basis_legendre(self, degree: int):
        if self.ndims != 1:
            raise NotImplementedError('legendre is only implemented for 1D topologies')
        return function.LegendreBasis(degree, len(self), self.f_index, self.f_coords)

    @cached_property
    def refined(self):
        'refine non-uniformly'

        axes = [axis.refined for axis in self.axes]
        return StructuredTopology(self.space, self.root, axes, self.nrefine+1, bnames=self._bnames)

    def locate(self, geom, coords, *, tol=0, eps=0, weights=None, skip_missing=False, arguments=None, **kwargs):
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

    def __str__(self):
        'string representation'

        return '{}({})'.format(self.__class__.__name__, 'x'.join(str(n) for n in self.shape))


class ConnectedTopology(TransformChainsTopology):
    'unstructured topology with connectivity'

    def __init__(self, space: str, references: References, transforms: transformseq.Transforms, opposites: transformseq.Transforms, connectivity: Sequence[numpy.ndarray]):
        assert isinstance(space, str), f'space={space!r}'
        assert isinstance(references, References), f'references={references!r}'
        assert isinstance(transforms, transformseq.Transforms), f'transforms={transforms!r}'
        assert isinstance(opposites, transformseq.Transforms), f'opposites={opposites!r}'
        assert isinstance(connectivity, numpy.ndarray) and connectivity.ndim == 2 and connectivity.dtype.kind == 'i' or isinstance(connectivity, Sequence) and all(isinstance(item, numpy.ndarray) and item.ndim == 1 and item.dtype.kind == 'i' for item in connectivity), f'connectivity={connectivity!r}'
        assert len(connectivity) == len(references) and all(c.shape[0] == e.nedges for c, e in zip(connectivity, references))

        self.connectivity = types.frozenarray(connectivity) if isinstance(connectivity, numpy.ndarray) else tuple(map(types.frozenarray, connectivity))
        super().__init__(space, references, transforms, opposites)


class SimplexTopology(TransformChainsTopology):
    'simpex topology'

    def _renumber(simplices):
        simplices = numpy.asarray(simplices)
        keep = numpy.zeros(simplices.max()+1, dtype=bool)
        keep[simplices.flat] = True
        return types.arraydata(simplices if keep.all() else (numpy.cumsum(keep)-1)[simplices])

    def __init__(self, space: str, simplices: numpy.ndarray, transforms: transformseq.Transforms, opposites: transformseq.Transforms):
        assert isinstance(space, str), f'space={space!r}'
        assert isinstance(simplices, numpy.ndarray), f'simplices={simplices!r}'
        assert simplices.shape == (len(transforms), transforms.fromdims+1)
        self.simplices = numpy.asarray(simplices)
        assert numpy.greater(self.simplices[:, 1:], self.simplices[:, :-1]).all(), 'nodes should be sorted'
        assert not numpy.equal(self.simplices[:, 1:], self.simplices[:, :-1]).all(), 'duplicate nodes'
        references = References.uniform(element.getsimplex(transforms.fromdims), len(transforms))
        super().__init__(space, references, transforms, opposites)

    def take_unchecked(self, indices):
        space, = self.spaces
        return SimplexTopology(space, self.simplices[indices], self.transforms[indices], self.opposites[indices])

    @cached_property
    def boundary(self):
        space, = self.spaces
        ielem, iedge = (self.connectivity == -1).nonzero()
        nd = self.ndims
        edges = numpy.arange(nd+1).repeat(nd).reshape(nd,nd+1).T[::-1]
        simplices = self.simplices[ielem, edges[iedge].T].T
        transforms = self.transforms.edges(self.references)[ielem * (nd+1) + iedge]
        return SimplexTopology(space, simplices, transforms, transforms)

    @cached_property
    def connectivity(self):
        nverts = self.ndims + 1
        edge_vertices = numpy.arange(nverts).repeat(self.ndims).reshape(self.ndims, nverts)[:, ::-1].T  # nverts x ndims
        simplices_edges = self.simplices.take(edge_vertices, axis=1)  # nelems x nverts x ndims
        elems, edges = divmod(numpy.lexsort(simplices_edges.reshape(-1, self.ndims).T), nverts)
        sorted_simplices_edges = simplices_edges[elems, edges]  # (nelems x nverts) x ndims; matching edges are now adjacent
        i, = numpy.equal(sorted_simplices_edges[1:], sorted_simplices_edges[:-1]).all(axis=1).nonzero()
        j = i + 1
        assert numpy.greater(i[1:], j[:-1]).all(), 'single edge is shared by three or more simplices'
        connectivity = numpy.full((len(self.simplices), self.ndims+1), fill_value=-1, dtype=int)
        connectivity[elems[i], edges[i]] = elems[j]
        connectivity[elems[j], edges[j]] = elems[i]
        return types.frozenarray(connectivity, copy=False)

    def basis_std(self, degree):
        if degree == 1:
            coeffs = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
            return function.PlainBasis([coeffs] * len(self), self.simplices, self.simplices.max()+1, self.f_index, self.f_coords)
        return super().basis_std(degree)

    def basis_bubble(self):
        'bubble from vertices'

        bernstein = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
        bubble = reduce(lambda l, r: poly.mul_same_vars(l, r, self.ndims), bernstein)
        coeffs = numpy.zeros((len(bernstein)+1, poly.ncoeffs(self.ndims, 1 + self.ndims)))
        coeffs[:-1] = poly.change_degree(bernstein, self.ndims, 1 + self.ndims) - bubble[None] / (self.ndims+1)
        coeffs[-1] = bubble
        coeffs = types.frozenarray(coeffs, copy=False)
        nverts = self.simplices.max() + 1
        ndofs = nverts + len(self)
        nmap = [types.frozenarray(numpy.hstack([idofs, nverts+ielem]), copy=False) for ielem, idofs in enumerate(self.simplices)]
        return function.PlainBasis([coeffs] * len(self), nmap, ndofs, self.f_index, self.f_coords)


class UnionTopology(TransformChainsTopology):
    'grouped topology'

    def __init__(self, topos: Sequence[Topology], names: Sequence[str] = ()):
        assert isinstance(topos, Sequence) and all(isinstance(topo, Topology) for topo in topos), f'topos={topos!r}'
        assert isinstance(names, Sequence) and all(isinstance(name, str) for name in names), f'names={names!r}'
        self._topos = tuple(topos)
        self._names = tuple(names[:len(topos)])
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

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        topos = (topo if name in groups else topo.get_groups(*groups) for topo, name in itertools.zip_longest(self._topos, self._names))
        return reduce(operator.or_, filter(None, topos), self.empty_like())

    def __or__(self, other):
        if not isinstance(other, TransformChainsTopology):
            return super().__or__(other)
        if not isinstance(other, UnionTopology):
            return UnionTopology(self._topos + (other,), self._names)
        return UnionTopology(self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names)

    @cached_property
    def refined(self):
        return UnionTopology([topo.refined for topo in self._topos], self._names)


class DisjointUnionTopology(TransformChainsTopology):
    'grouped topology'

    def __init__(self, topos: Sequence[Topology], names: Sequence[str] = ()):
        assert isinstance(topos, Sequence) and all(isinstance(topo, Topology) for topo in topos), f'topos={topos!r}'
        assert isinstance(names, Sequence) and all(isinstance(name, str) for name in names), f'names={names!r}'
        self._topos = tuple(topos)
        self._names = tuple(names[:len(topos)])
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

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        topos = (topo if name in groups else topo.get_groups(*groups) for topo, name in itertools.zip_longest(self._topos, self._names))
        topos = tuple(filter(None, topos))
        if len(topos) == 0:
            return self.empty_like()
        elif len(topos) == 1:
            return topos[0]
        else:
            return DisjointUnionTopology(topos)

    @cached_property
    def refined(self):
        return DisjointUnionTopology([topo.refined for topo in self._topos], self._names)


class SubsetTopology(TransformChainsTopology):
    'trimmed'

    def __init__(self, basetopo: Topology, refs: Sequence[element.Reference], newboundary: Optional[Union[str,TransformChainsTopology]] = None):
        assert isinstance(basetopo, Topology), f'basetopo={basetopo!r}'
        assert isinstance(refs, Sequence) and all(isinstance(ref, element.Reference) for ref in refs), f'refs={refs!r}'
        assert newboundary is None or isinstance(newboundary, str) or isinstance(newboundary, TransformChainsTopology) and newboundary.ndims == basetopo.ndims-1, f'newboundary={newboundary!r}'
        assert len(refs) == len(basetopo)
        self.refs = tuple(refs)
        self.basetopo = basetopo
        self.newboundary = newboundary

        self._indices = types.frozenarray(numpy.array([i for i, ref in enumerate(self.refs) if ref], dtype=int), copy=False)
        references = References.from_iter(self.refs, self.basetopo.ndims).take(self._indices)
        transforms = self.basetopo.transforms[self._indices]
        opposites = self.basetopo.opposites[self._indices]
        super().__init__(basetopo.space, references, transforms, opposites)

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        return self.basetopo.get_groups(*groups).subset(self, strict=False)

    def __rsub__(self, other):
        if self.basetopo == other:
            refs = [baseref - ref for baseref, ref in zip(self.basetopo.references, self.refs)]
            return SubsetTopology(self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary, TransformChainsTopology) else self.newboundary)
        return super().__rsub__(other)

    def __or__(self, other):
        if not isinstance(other, SubsetTopology) or self.basetopo != other.basetopo:
            return super().__or__(other)
        refs = [ref1 | ref2 for ref1, ref2 in zip(self.refs, other.refs)]
        if all(baseref == ref for baseref, ref in zip(self.basetopo.references, refs)):
            return self.basetopo
        return SubsetTopology(self.basetopo, refs)  # TODO boundary

    @cached_property
    def connectivity(self):
        renumber = numeric.invmap([i for i, ref in enumerate(self.refs) if ref], length=len(self.refs)+1, missing=-1)  # length=..+1 serves to map -1 to -1
        return tuple(types.frozenarray(numpy.concatenate([renumber.take(ioppelems), numpy.repeat(-1, ref.nedges-len(ioppelems))]), copy=False)
                     for ref, ioppelems in zip(self.refs, self.basetopo.connectivity) if ref)

    @cached_property
    def refined(self):
        child_refs = self.references.children
        indices = types.frozenarray(numpy.array([i for i, ref in enumerate(child_refs) if ref], dtype=int), copy=False)
        refined_transforms = self.transforms.refined(self.references)[indices]
        self_refined = TransformChainsTopology(self.space, child_refs[indices], refined_transforms, refined_transforms)
        return self.basetopo.refined.subset(self_refined, self.newboundary.refined if isinstance(self.newboundary, TransformChainsTopology) else self.newboundary, strict=True)

    @cached_property
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
                        trimmedtransforms.append(transform.canonical((*elemtrans, edgetrans)))
                        trimmedopposites.append(transform.canonical((*self.basetopo.transforms[ioppelem], oppref.edge_transforms[ioppedge])))
            # The last edges of newref (beyond the number of edges of the original)
            # cannot have opposites and are added to the trimmed group directly.
            for edgetrans, edgeref in newref.edges[len(ioppelems):]:
                trimmedreferences.append(edgeref)
                trimmedtransforms.append(transform.canonical((*elemtrans, edgetrans)))
                trimmedopposites.append(transform.canonical((*elemtrans, edgetrans.flipped)))
        origboundary = SubsetTopology(baseboundary, brefs)
        if isinstance(self.newboundary, TransformChainsTopology):
            trimmedbrefs = [ref.empty for ref in self.newboundary.references]
            for ref, trans in zip(trimmedreferences, trimmedtransforms):
                trimmedbrefs[self.newboundary.transforms.index(trans)] = ref
            trimboundary = SubsetTopology(self.newboundary, trimmedbrefs)
        else:
            trimboundary = TransformChainsTopology(self.space, References.from_iter(trimmedreferences, self.ndims-1), transformseq.PlainTransforms(tuple(trimmedtransforms), self.transforms.todims, self.ndims-1), transformseq.PlainTransforms(tuple(trimmedopposites), self.transforms.todims, self.ndims-1))
        return DisjointUnionTopology([trimboundary, origboundary], names=[self.newboundary] if isinstance(self.newboundary, str) else [])

    @cached_property
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

    def locate(self, geom, coords, *, eps=0, skip_missing=False, **kwargs):
        sample = self.basetopo.locate(geom, coords, eps=eps, skip_missing=skip_missing, **kwargs)
        missing = []
        for isampleelem, (transforms, points_) in enumerate(zip(sample.transforms[0], sample.points)):
            ielem = self.basetopo.transforms.index(transforms)
            ref = self.refs[ielem]
            if ref != self.basetopo.references[ielem]:
                for i, coord in enumerate(points_.coords):
                    if not ref.inside(coord, eps):
                        if not skip_missing:
                            raise LocateError('failed to locate point: {}'.format(coords[sample.getindex(isampleelem)[i]]))
                        missing.append((isampleelem, i))
        if not missing:
            return sample
        selection = numpy.ones(len(sample.points), dtype=bool)
        newpoints = []
        for isampleelem, points_ in enumerate(sample.points):
            mymissing = [] # collect missing points for current element
            for isampleelem_, i in missing[:points_.npoints]:
                if isampleelem_ != isampleelem:
                    break
                mymissing.append(i)
            if not mymissing: # no points are missing -> keep existing points object
                newpoints.append(points_)
            elif len(mymissing) < points_.npoints: # some points are missing -> create new CoordsPoints object
                newpoints.append(points.CoordsPoints(points_.coords[~numeric.asboolean(mymissing, points_.npoints)]))
            else: # all points are missing -> remove element from return sample
                selection[isampleelem] = False
            del missing[:len(mymissing)]
        assert not missing
        return Sample.new(sample.space, [trans[selection] for trans in sample.transforms], PointsSequence.from_iter(newpoints, sample.ndims))


class RefinedTopology(TransformChainsTopology):
    'refinement'

    def __init__(self, basetopo: Topology):
        assert isinstance(basetopo, Topology), f'basetopo={basetopo!r}'
        self.basetopo = basetopo
        super().__init__(
            self.basetopo.space,
            self.basetopo.references.children,
            self.basetopo.transforms.refined(self.basetopo.references),
            self.basetopo.opposites.refined(self.basetopo.references))

    def get_groups(self, *groups: str) -> TransformChainsTopology:
        return self.basetopo.get_groups(*groups).refined

    @cached_property
    def boundary(self):
        return self.basetopo.boundary.refined

    @cached_property
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


class HierarchicalTopology(TransformChainsTopology):
    'collection of nested topology elments'

    def __init__(self, basetopo: Topology, indices_per_level: Sequence[numpy.ndarray]):
        assert isinstance(basetopo, Topology) and not isinstance(basetopo, HierarchicalTopology), f'basetopo={basetopo!r}'
        assert isinstance(indices_per_level, Sequence) and all(isinstance(indices, numpy.ndarray) and indices.ndim == 1 and indices.dtype.kind == 'i' for indices in indices_per_level), f'indices_per_level={indices_per_level!r}'
        self.basetopo = basetopo
        self._indices_per_level = tuple(map(types.frozenarray, indices_per_level))
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

    def _refined_by(self, refine):
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

    @cached_property
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

    @cached_property
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

    @cached_property
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
                            else tail[h].transform_poly(mypoly) @ project[..., mypassive] @ truncpoly[mypassive]

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
                degree = poly.degree(self.ndims, max(c.shape[-1] for c in trans_coeffs))
                hbasis_coeffs.append(numpy.concatenate([poly.change_degree(c, self.ndims, degree) for c in trans_coeffs], axis=0))

        return function.PlainBasis(hbasis_coeffs, hbasis_dofs, ndofs, self.f_index, self.f_coords)


@dataclass(eq=True, frozen=True)
class PatchBoundary:

    id: Tuple[int, ...]
    dim: int
    side: int
    reverse: Tuple[bool, ...]
    transpose: Tuple[int, ...]

    def apply_transform(self, array):
        return array[tuple(slice(None, None, -1) if i else slice(None) for i in self.reverse)].transpose(self.transpose)


@dataclass(eq=True, frozen=True)
class Patch:

    topo: Topology
    verts: types.arraydata
    boundaries: Tuple[PatchBoundary, ...]


class MultipatchTopology(TransformChainsTopology):
    'multipatch topology'

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

    def __init__(self, patches: Sequence[Patch]):
        assert isinstance(patches, Sequence) and all(isinstance(patch, Patch) for patch in patches), f'patches={patches!r}'
        self.patches = tuple(patches)

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

    @cached_property
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

    def get_groups(self, *groups: str) -> TransformChainsTopology:
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
            renumber, dofcount = util.merge_index_map(dofcount, pairs)
            # apply mappings
            dofmap = tuple(types.frozenarray(renumber[v], copy=False) for v in dofmap)

        return function.PlainBasis(coeffs, dofmap, dofcount, self.f_index, self.f_coords)

    def basis_patch(self):
        'degree zero patchwise discontinuous basis'

        transforms = transformseq.PlainTransforms(tuple((patch.topo.root,) for patch in self.patches), self.ndims, self.ndims)
        index = function.transforms_index(self.space, transforms)
        coords = function.transforms_coords(self.space, transforms)
        return function.DiscontBasis([types.frozenarray([[1]], dtype=float)]*len(self.patches), index, coords)

    @cached_property
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
            return EmptyTopology(self.space, self.transforms.todims, self.ndims-1)
        else:
            return DisjointUnionTopology(subtopos, subnames)

    @cached_property
    def interfaces(self):
        '''interfaces

        Return a topology with all element interfaces.  The patch interfaces are
        accessible via the group ``'interpatch'`` and the interfaces *inside* a
        patch via ``'intrapatch'``.
        '''

        intrapatchtopo = EmptyTopology(self.space, self.transforms.todims, self.ndims-1) if not self.patches else \
            DisjointUnionTopology([patch.topo.interfaces for patch in self.patches])

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
                pairs.append(tuple(map(transform.canonical, transforms.flat)))
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

    @cached_property
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

    @cached_property
    def refined(self):
        'refine'

        return MultipatchTopology(tuple(Patch(patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches))

# vim:sw=4:sts=4:et
