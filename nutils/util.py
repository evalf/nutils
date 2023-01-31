"""
The util module provides a collection of general purpose methods.
"""

from . import numeric
import sys
import os
import numpy
import collections.abc
import inspect
import itertools
import functools
import operator
import numbers
import pathlib
import ctypes
import io
import contextlib
from typing import Iterable, Sequence, Tuple

supports_outdirfd = os.open in os.supports_dir_fd and os.listdir in os.supports_fd

sum = functools.partial(functools.reduce, operator.add)
product = functools.partial(functools.reduce, operator.mul)


def cumsum(seq):
    offset = 0
    for i in seq:
        yield offset
        offset += i


def gather(items):
    gathered = []
    d = {}
    for key, value in items:
        try:
            values = d[key]
        except KeyError:
            d[key] = values = []
            gathered.append((key, values))
        values.append(value)
    return gathered


def pairwise(items, *, periodic=False):
    items = iter(items)
    try:
        first = a = next(items)
    except StopIteration:
        return
    for b in items:
        yield a, b
        a = b
    if periodic:
        yield a, first


def allequal(seq1, seq2):
    seq1 = iter(seq1)
    seq2 = iter(seq2)
    for item1, item2 in zip(seq1, seq2):
        if item1 != item2:
            return False
    if list(seq1) or list(seq2):
        return False
    return True


class NanVec(numpy.ndarray):
    'nan-initialized vector'

    def __new__(cls, length):
        vec = numpy.empty(length, dtype=float).view(cls)
        vec[:] = numpy.nan
        return vec

    @property
    def where(self):
        return ~numpy.isnan(self.view(numpy.ndarray))

    def __iand__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__iand__(other)
        where = self.where
        if numpy.isscalar(other):
            self[where] = other
        else:
            assert numeric.isarray(other) and other.shape == self.shape
            self[where] = other[where]
        return self

    def __and__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__and__(other)
        return self.copy().__iand__(other)

    def __ior__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__ior__(other)
        wherenot = ~self.where
        self[wherenot] = other if numpy.isscalar(other) else other[wherenot]
        return self

    def __or__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__or__(other)
        return self.copy().__ior__(other)

    def __invert__(self):
        if self.dtype != float:
            return self.view(numpy.ndarray).__invert__()
        nanvec = NanVec(len(self))
        nanvec[numpy.isnan(self)] = 0
        return nanvec


def regularize(bbox, spacing, xy=numpy.empty((0, 2))):
    xy = numpy.asarray(xy)
    index0 = numeric.floor(bbox[:, 0] / (2*spacing)) * 2 - 1
    shape = numeric.ceil(bbox[:, 1] / (2*spacing)) * 2 + 2 - index0
    index = numeric.round(xy / spacing) - index0
    keep = numpy.logical_and(numpy.greater_equal(index, 0), numpy.less(index, shape)).all(axis=1)
    mask = numpy.zeros(shape, dtype=bool)
    for i, ind in enumerate(index):
        if keep[i]:
            if not mask[tuple(ind)]:
                mask[tuple(ind)] = True
            else:
                keep[i] = False
    coursex = mask[0:-2:2] | mask[1:-1:2] | mask[2::2]
    coarsexy = coursex[:, 0:-2:2] | coursex[:, 1:-1:2] | coursex[:, 2::2]
    vacant, = (~coarsexy).ravel().nonzero()
    newindex = numpy.array(numpy.unravel_index(vacant, coarsexy.shape)).T * 2 + index0 + 1
    return numpy.concatenate([newindex * spacing, xy[keep]], axis=0)


def tri_merge(tri, x, mergetol=0):
    '''Create connected triangulation by connecting (near) identical points.

    Based on a set of coordinates ``x``, create a modified copy of ``tri`` with
    any occurrence of ``j`` replaced by ``i`` if ``x[i]`` equals ``x[j]`` within
    specified tolerance. The result is a triangulation that remains valid for any
    associated data vector that follows the same equality relations.

    Example:

    >>> x = [0,0], [1,0], [0,1], [1,0], [1,1] # note: x[1] == x[3])
    >>> tri = [0,1,2], [2,3,4]
    >>> tri_merge(tri, x)
    array([[0, 1, 2],
           [2, 1, 4]])

    .. requires:: scipy

    Args
    ----
    x : :class:`float` array
        Vertex coordinates.
    tri : :class:`int` array
        Triangulation.
    mergetol : :class:`float` (optional, default 0)
        Distance within which two points are considered equal. If mergetol == 0
        then points are considered equal if and only if their coordinates are
        identical. If mergetol > 0 (required scipy) then points are considered
        equal if they are within euclidian distance < mergetol. If mergetol < 0
        then tri is returned unchanged.

    Returns
    -------
    merged_tri : :class:`int` array
    '''

    tri = numpy.asarray(tri)
    x = numpy.asarray(x)
    assert tri.dtype == int
    assert x.ndim == tri.ndim == 2
    assert tri.shape[1] == x.shape[1] + 1
    if mergetol < 0:
        return tri
    if mergetol == 0:
        order = numpy.lexsort(x.T)
        keep = numpy.concatenate([[True], numpy.diff(x[order], axis=0).any(axis=1)])
        renumber = numpy.empty(len(x), dtype=int)
        renumber[order] = order[keep][keep.cumsum()-1]
    else:
        import scipy.spatial
        renumber = numpy.arange(len(x))
        for i, j in sorted(scipy.spatial.cKDTree(x).query_pairs(mergetol)):
            assert i < j
            renumber[j] = renumber[i]
    return renumber[tri]


class tri_interpolator:
    '''Interpolate function values defined in triangulation vertices.

    Convenience object that implements 2D interpolation on top of matplotlib's
    triangulation routines. Unlike matplotlib's own ``LinearTriInterpolator``,
    the ``tri_interpolator`` allows for interpolation of multi-dimensional
    arrays, as well as repeated interpolations of different vertex values.

    The arguments are identical to :func:`tri_merge`.

    After instantiation of the interpolator object, interpolation coordinates are
    specified via the object's getitem operator. The resulting callable performs
    the interpolation:

    >>> trix = [0,0], [1,0], [0,1], [1,1] # vertex coordinates
    >>> triu = 0, 0, 10, 0 # vertex values
    >>> interpolate = tri_interpolator([[0,1,2],[1,3,2]], trix)
    >>> x = [.1,.1], [.1,.9], [.9,.9] # interpolation coordinates
    >>> u = interpolate[x](triu) # interpolated values

    .. requires:: matplotlib
    '''

    def __init__(self, tri, x, mergetol=0):
        x = numpy.asarray(x)
        assert x.ndim == 2
        if x.shape[1] != 2:
            raise NotImplementedError('only 2D interpolation is supported for now')
        import matplotlib.tri
        self.mpltri = matplotlib.tri.Triangulation(x[:, 0], x[:, 1], tri_merge(tri, x, mergetol))

    def __getitem__(self, x):
        x = numpy.asarray(x)
        assert x.shape[-1] == 2
        itri = self.mpltri.get_trifinder()(x[..., 0].ravel(), x[..., 1].ravel())
        inside = itri != -1
        itri = itri[inside]
        plane_coords = numpy.concatenate([x.reshape(-1, 2)[inside], numpy.ones([len(itri), 1])], axis=1)

        def interpolate(vtri):
            vtri = numpy.asarray(vtri)
            assert vtri.shape[0] == len(self.mpltri.x)
            vx = numpy.empty(x.shape[:-1] + vtri.shape[1:])
            vx[...] = numpy.nan
            for vx_items, vtri_items in zip(vx.reshape(len(inside), -1).T, vtri.reshape(len(vtri), -1).T):
                plane_coeffs = self.mpltri.calculate_plane_coefficients(vtri_items)
                vx_items[inside] = numeric.contract(plane_coords, plane_coeffs[itri], axis=1)
            return vx
        return interpolate


class linear_regressor:
    def add(self, x, y, weight=.5):
        y = numpy.asarray(y)
        new = numpy.outer([1, x], [x] + list(y.flat))
        (x_, *y_), (xx_, *xy_) = self.avg = (1-weight) * getattr(self, 'avg', new) + weight * new
        return numpy.dot([[-x_, 1], [xx_, -x_]], [y_, xy_]).reshape((2,)+y.shape) / (xx_-x_**2 or numpy.nan)


def obj2str(obj):
    '''compact, lossy string representation of arbitrary object'''
    return '['+','.join(obj2str(item) for item in obj)+']' if isinstance(obj, collections.abc.Iterable) \
        else str(obj).strip('0').rstrip('.') or '0' if isinstance(obj, numbers.Real) \
        else str(obj)


class single_or_multiple:
    """
    Method wrapper, converts first positional argument to tuple: tuples/lists
    are passed on as tuples, other objects are turned into tuple singleton.
    Return values should match the length of the argument list, and are unpacked
    if the original argument was not a tuple/list.

    >>> class Test:
    ...   @single_or_multiple
    ...   def square(self, args):
    ...     return [v**2 for v in args]
    ...
    >>> T = Test()
    >>> T.square(2)
    4
    >>> T.square([2,3])
    (4, 9)

    Args
    ----
    f: :any:`callable`
        Method that expects a tuple as first positional argument, and that
        returns a list/tuple of the same length.

    Returns
    -------
    :
        Wrapped method.
    """

    def __init__(self, f):
        functools.update_wrapper(self, f)

    def __get__(self, instance, owner):
        return single_or_multiple(self.__wrapped__.__get__(instance, owner))

    def __call__(self, *args, **kwargs):
        if not args:
            raise TypeError('{} requires at least 1 positional argument'.format(self.__wrapped__.__name__))
        ismultiple = isinstance(args[0], (list, tuple, map))
        retvals = tuple(self.__wrapped__(tuple(args[0]) if ismultiple else args[:1], *args[1:], **kwargs))
        if not ismultiple:
            retvals, = retvals
        return retvals


class positional_only:
    '''Change all positional-or-keyword arguments to positional-only.

    Python introduces syntax to define positional-only parameters in version 3.8,
    but the same effect can be achieved in older versions by using a wrapper with
    a var-positional argument. The :func:`positional_only` decorator uses this
    technique to treat all positional-or-keyword arguments as positional-only. In
    order to avoid name clashes between the positional-only arguments and
    variable keyword arguments, the wrapper additionally introduces the
    convention that the last argument receives the variable keyword argument
    dictionary in case is has a default value of ... (ellipsis).

    Example:

    >>> @positional_only
    ... def f(x, *, y):
    ...   pass
    >>> inspect.signature(f)
    <Signature (x, /, *, y)>

    >>> @positional_only
    ... def f(x, *args, y, kwargs=...):
    ...   pass
    >>> inspect.signature(f)
    <Signature (x, /, *args, y, **kwargs)>

    Args
    ----
    f : :any:`callable`
        Function to be wrapped.
    '''

    def __init__(self, f):
        signature = inspect.signature(f)
        parameters = list(signature.parameters.values())
        keywords = []
        varkw = None
        for i, param in enumerate(parameters):
            if param.kind is param.VAR_KEYWORD:
                raise Exception('positional_only decorated function must use ellipses to mark a variable keyword argument')
            if i == len(parameters)-1 and param.default is ...:
                parameters[i] = param.replace(kind=inspect.Parameter.VAR_KEYWORD, default=inspect.Parameter.empty)
                varkw = param.name
            elif param.kind is param.POSITIONAL_OR_KEYWORD:
                parameters[i] = param.replace(kind=param.POSITIONAL_ONLY)
            elif param.kind is param.KEYWORD_ONLY:
                keywords.append(param.name)
        self.__keywords = tuple(keywords)
        self.__varkw = varkw
        self.__signature__ = signature.replace(parameters=parameters)
        functools.update_wrapper(self, f)

    def __get__(self, instance, owner):
        return positional_only(self.__wrapped__.__get__(instance, owner))

    def __call__(self, *args, **kwargs):
        wrappedkwargs = {name: kwargs.pop(name) for name in self.__keywords if name in kwargs}
        if self.__varkw:
            wrappedkwargs[self.__varkw] = kwargs
        elif kwargs:
            raise TypeError('{}() got an unexpected keyword argument {!r}'.format(self.__wrapped__.__name__, *kwargs))
        return self.__wrapped__(*args, **wrappedkwargs)


def loadlib(**libname):
    '''
    Find and load a dynamic library using :any:`ctypes.CDLL`.  For each
    (supported) platform the name of the library should be specified as a keyword
    argument, including the extension, where the keywords should match the
    possible values of :any:`sys.platform`.

    Example
    -------

    To load the Intel MKL runtime library, write::

        loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')
    '''

    try:
        return ctypes.CDLL(libname[sys.platform])
    except (OSError, KeyError):
        pass


def readtext(path):
    '''Read file and return contents

    Args
    ----
    path: :class:`os.PathLike`, :class:`str` or :class:`io.TextIOBase`
        Path-like or file-like object pointing to the data to be read.

    Returns
    -------
    :
        File data as :class:`str`.
    '''

    if isinstance(path, pathlib.Path):
        with path.open() as f:
            return f.read()

    if isinstance(path, str):
        with open(path) as f:
            return f.read()

    if isinstance(path, io.TextIOBase):
        return path.read()

    raise TypeError('readtext requires a path-like or file-like argument')


def binaryfile(path):
    '''Open file for binary reading

    Args
    ----
    path: :class:`os.PathLike`, :class:`str` or :class:`io.BufferedIOBase`
        Path-like or file-like object pointing to the data to be read.

    Returns
    -------
    :
        Context that returns a :class:`io.BufferedReader` upon entry.
    '''

    if isinstance(path, pathlib.Path):
        return path.open('rb')

    if isinstance(path, str):
        return open(path, 'rb')

    if isinstance(path, io.BufferedIOBase):
        return contextlib.nullcontext(path) if hasattr(contextlib, 'nullcontext') \
            else contextlib.contextmanager(iter)([path])  # Python <= 3.6

    raise TypeError('binaryfile requires a path-like or file-like argument')


class settable:
    '''Context-switchable data container.

    A mutable container for a general Python object, which can be changed by
    entering the ``sets`` context. The current value can be accessed via the
    ``value`` attribute.

    >>> myprop = settable(2)
    >>> myprop.value
    2
    >>> with myprop.sets(3):
    ...   myprop.value
    3
    >>> myprop.value
    2
    '''

    __slots__ = 'value'

    def __init__(self, value=None):
        self.value = value

    @contextlib.contextmanager
    def sets(self, value):
        oldvalue = self.value
        self.value = value
        try:
            yield
        finally:
            self.value = oldvalue


def index(sequence, item):
    '''Index of first occurrence.

    Generalization of `tuple.index`.
    '''

    if isinstance(sequence, (list, tuple)):
        return sequence.index(item)
    for i, v in enumerate(sequence):
        if v == item:
            return i
    raise ValueError('index(sequence, item): item not in sequence')


def unique(items, key=None):
    '''Deduplicate items in sequence.

    Return a tuple `(unique, indices)` such that `items[i] == unique[indices[i]]`
    and `unique` does not contain duplicate items. An optional `key` is applied
    to all items before testing for equality.
    '''

    seen = {}
    unique = []
    indices = []
    for item in items:
        k = item if key is None else key(item)
        try:
            index = seen[k]
        except KeyError:
            index = seen[k] = len(unique)
            unique.append(item)
        indices.append(index)
    return unique, indices


try:
    cached_property = functools.cached_property
except AttributeError:  # python < 3.8
    def cached_property(func):  # minimal backport
        @functools.wraps(func)
        def wrapped(self):
            try:
                val = self.__dict__[func.__name__]
            except KeyError:
                self.__dict__[func.__name__] = val = func(self)
            return val
        return property(wrapped)


def merge_index_map(nin: int, merge_sets: Iterable[Sequence[int]]) -> Tuple[numpy.ndarray, int]:
    '''Returns an index map relating ``nin`` unmerged elements to ``nout`` merged elements.

    The index map, an array of length ``nin``, satisfies the following conditions:

    *   For every merge set in ``merge_sets``: for every pair of indices ``i``
        and ``j`` in a merge set, ``index_map[i] == index_map[j]`` is true.

        In code, the following is true::

            all(index_map[i] == index_map[j] for i, *js in merge_sets for j in js)

    *   Selecting the first occurences of indices in ``index_map`` gives the
        sequence ``range(nout)``.

    Args
    ----
    nin : :class:`int`
        The number of elements before merging.
    merge_sets : iterable of sequences of at least one :class:`int`
        An iterable of merge sets, where each merge set lists the indices of
        input elements that should be merged. Every merge set should have at
        least one index.

    Returns
    -------
    index_map : :class:`numpy.ndarray`
        Index map with satisfying the above conditions.
    nout : :class:`int`
        The number of output indices.
    '''

    index_map = numpy.arange(nin)
    def resolve(index):
        parent = index_map[index]
        while index != parent:
            index = parent
            parent = index_map[index]
        return index
    for merge_set in merge_sets:
        resolved = list(map(resolve, merge_set))
        index_map[resolved] = min(resolved)
    new_indices = itertools.count()
    for iin, ptr in enumerate(index_map):
        index_map[iin] = next(new_indices) if iin == ptr else index_map[ptr]
    return index_map, next(new_indices)


# vim:sw=2:sts=2:et
