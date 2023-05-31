"""
Module with general purpose types.
"""

import inspect
import functools
import hashlib
import numbers
import collections.abc
import itertools
import abc
import sys
import weakref
import re
import io
import types
import numpy
import dataclasses
from ._backports import cached_property
from ctypes import byref, c_int, c_ssize_t, c_void_p, c_char_p, py_object, pythonapi, Structure, POINTER
c_ssize_p = POINTER(c_ssize_t)


def argument_canonicalizer(signature):
    '''
    Returns a function that converts arguments matching ``signature`` to
    canonical positional and keyword arguments.  If possible, an argument is
    added to the list of positional arguments, otherwise to the keyword arguments
    dictionary.  The returned arguments include default values.

    Parameters
    ----------
    signature : :class:`inspect.Signature`
        The signature of a function to generate canonical arguments for.

    Returns
    -------
    :any:`callable`
        A function that returns a :class:`tuple` of a :class:`tuple` of
        positional arguments and a :class:`dict` of keyword arguments.

    Examples
    --------

    Consider the following function.

    >>> def f(a, b=4, *, c): pass

    The ``argument_canonicalizer`` for ``f`` is generated as follows:

    >>> canon = argument_canonicalizer(inspect.signature(f))

    Calling ``canon`` with parameter ``b`` passed as keyword returns arguments
    with parameter ``b`` as positional argument:

    >>> canon(1, c=3, b=2)
    ((1, 2), {'c': 3})

    When calling ``canon`` without parameter ``b`` the default value is added to
    the positional arguments:

    >>> canon(1, c=3)
    ((1, 4), {'c': 3})
    '''

    def canonicalize(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.args, bound.kwargs

    return canonicalize


def nutils_hash(data):
    '''
    Compute a stable hash of immutable object ``data``.  The hash is not affected
    by Python's hash randomization (see :meth:`object.__hash__`).

    Parameters
    ----------
    data
        An immutable object of type :class:`bool`, :class:`int`, :class:`float`,
        :class:`complex`, :class:`str`, :class:`bytes`, :class:`tuple`,
        :class:`frozenset`, or :any:`Ellipsis` or :any:`None`, or the type
        itself, or an object with a ``__nutils_hash__`` attribute.

    Returns
    -------
    40 :class:`bytes`
        The hash of ``data``.
    '''

    try:
        return data.__nutils_hash__
    except AttributeError:
        pass

    if isinstance(data, numpy.generic):
        # normalize Numpy's scalar types so that their nutils_hash are equal to
        # that of Python's counterparts, similar to Python's builtin hash
        t = dict(b=bool, i=int, f=float, c=complex)[data.dtype.kind]
        data = t(data)

    t = type(data)
    h = hashlib.sha1(t.__name__.encode()+b'\0')
    if data is Ellipsis or data is None:
        pass
    elif t is type:
        h.update(hashlib.sha1(data.__name__.encode()).digest())
    elif t in (bool, int, float, complex):
        h.update(hashlib.sha1(repr(data).encode()).digest())
    elif t is str:
        h.update(hashlib.sha1(data.encode()).digest())
    elif t is bytes:
        h.update(hashlib.sha1(data).digest())
    elif t in (list, tuple):
        for item in data:
            h.update(nutils_hash(item))
    elif t is dict:
        for item in sorted(nutils_hash(k) + nutils_hash(v) for k, v in data.items()):
            h.update(item)
    elif t in (set, frozenset):
        for item in sorted(map(nutils_hash, data)):
            h.update(item)
    elif issubclass(t, io.BufferedIOBase) and data.seekable():
        pos = data.tell()
        h.update(str(pos).encode())
        data.seek(0)
        chunk = data.read(0x20000)
        while chunk:
            h.update(chunk)
            chunk = data.read(0x20000)
        data.seek(pos)
    elif t is types.MethodType:
        h.update(nutils_hash(data.__self__))
        h.update(nutils_hash(data.__name__))
    elif t is numpy.ndarray:
        h.update('{}{}\0'.format(','.join(map(str, data.shape)), data.dtype.str).encode())
        h.update(data.tobytes())
    elif dataclasses.is_dataclass(t):
        # Note: we cannot use dataclasses.asdict here as its built-in recursion
        # makes nested dataclass instances indistinguishable from dictionaries.
        for item in sorted(nutils_hash((field.name, getattr(data, field.name))) for field in dataclasses.fields(t)):
            h.update(item)
    elif hasattr(data, '__getnewargs__'):
        for arg in data.__getnewargs__():
            h.update(nutils_hash(arg))
    else:
        raise TypeError('unhashable type: {!r} {!r}'.format(data, t))
    return h.digest()


# While we do not use `abc.ABCMeta` in `ImmutableMeta` itself, we will use it
# in many classes having `ImmutableMeta` as a meta(super)class.  To avoid
# having to write `class MCls(ImmutableMeta, abc.ABCMeta): pass` everywhere, we
# simply derive from `abc.ABCMeta` here.


class ImmutableMeta(abc.ABCMeta):

    def __new__(mcls, name, bases, namespace, *, version=0, **kwargs):
        if not isinstance(version, int):
            raise ValueError("'version' should be of type 'int' but got {!r}".format(version))
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Since we redefine `__call__` here and `inspect.signature(cls)` looks at
        # `cls.__signature__` and if absent the signature of `__call__`, we
        # explicitly copy the signature of `<cls instance>.__init__` to `cls`.
        cls.__signature__ = inspect.signature(cls.__init__.__get__(object(), object))
        cls._canonicalize = argument_canonicalizer(inspect.signature(cls.__init__))
        cls._version = version
        return cls

    def __init__(cls, name, bases, namespace, *, version=0, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

    def __call__(*args, **kwargs):
        return args[0].__new__(*args, **kwargs)

    def _new(cls, *args):
        self = object.__new__(cls)
        self._args = args
        self._hash = hash(args)
        self.__init__(*args[:-1], **dict(args[-1]))
        return self


class Immutable(metaclass=ImmutableMeta):
    '''
    Base class for immutable types.  This class adds equality tests, traditional
    hashing (:func:`hash`), nutils hashing (:func:`nutils_hash`) and pickling,
    all based solely on the (positional) intialization arguments, ``args`` for
    future reference.  Keyword-only arguments are not supported.  All arguments
    should be hashable by :func:`nutils_hash`.

    Positional and keyword initialization arguments are canonicalized
    automatically (by :func:`argument_canonicalizer`).

    Examples
    --------

    Consider the following class.

    >>> class Plain(Immutable):
    ...   def __init__(self, a, b):
    ...     pass

    Calling ``Plain`` with equivalent positional or keyword arguments produces
    equal instances:

    >>> Plain(1, 2) == Plain(a=1, b=2)
    True

    Passing unhashable values to ``Plain`` will fail:

    >>> Plain([1, 2], [3, 4]) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'list'
    '''

    def __new__(*args, **kwargs):
        cls = args[0]
        args, kwargs = cls._canonicalize(*args, **kwargs)
        return cls._new(*args[1:], tuple(sorted(kwargs.items())))

    def __reduce__(self):
        return self.__class__._new, self._args

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other or type(self) is type(other) and self._args == other._args

    @cached_property
    def __nutils_hash__(self):
        h = hashlib.sha1('{}.{}:{}\0'.format(type(self).__module__, type(self).__qualname__, type(self)._version).encode())
        for arg in self._args:
            h.update(nutils_hash(arg))
        return h.digest()

    def __getstate__(self):
        raise Exception('getstate should never be called')

    def __setstate__(self, state):
        raise Exception('setstate should never be called')

    def __str__(self):
        *args, kwargs = self._args
        return '{}({})'.format(self.__class__.__name__, ','.join([*map(str, args), *map('{0[0]}={0[1]}'.format, kwargs)]))


class SingletonMeta(ImmutableMeta):

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._cache = weakref.WeakValueDictionary()
        return cls

    def _new(cls, *args):
        try:
            self = cls._cache[args]
        except KeyError:
            self = cls._cache[args] = super()._new(*args)
        return self


class Singleton(Immutable, metaclass=SingletonMeta):
    '''
    Subclass of :class:`Immutable` that creates a single instance per unique set
    of initialization arguments.

    Examples
    --------

    Consider the following class.

    >>> class Plain(Singleton):
    ...   def __init__(self, a, b):
    ...     pass

    Calling ``Plain`` with equivalent positional or keyword arguments produces
    one instance:

    >>> Plain(1, 2) is Plain(a=1, b=2)
    True
    '''

    __hash__ = Immutable.__hash__
    __eq__ = object.__eq__


class arraydata(Singleton):
    '''hashable array container.

    The container can be used for fast equality checks and for dictionary keys.
    Data is copied at construction and canonicalized by casting it to the
    platform's primary data representation (e.g. int64 i/o int32). It can be
    retrieved via :func:`numpy.asarray`. Additionally the ``arraydata`` object
    provides direct access to the array's shape, dtype and bytes.

    Example
    -------
    >>> a = numpy.array([1,2,3])
    >>> w = arraydata(a)
    >>> w == arraydata([1,2,4]) # NOTE: equality only if entire array matches
    False
    >>> numpy.asarray(w)
    array([1, 2, 3])
    '''

    def __new__(cls, arg):
        if isinstance(arg, cls):
            return arg
        array = numpy.asarray(arg)
        dtype = dict(b=bool, u=int, i=int, f=float, c=complex)[array.dtype.kind]
        return super().__new__(cls, dtype, array.shape, array.astype(dtype, copy=False).tobytes())

    def reshape(self, *shape):
        if numpy.prod(shape) != numpy.prod(self.shape):
            raise ValueError(f'cannot reshape arraydata of shape {self.shape} into shape {shape}')
        return super().__new__(self.__class__, self.dtype, shape, self.bytes)

    def __init__(self, dtype, shape, bytes):
        self.dtype = dtype
        self.shape = shape
        self.bytes = bytes
        self.ndim = len(shape)
        # Note: we define __array_interface__ rather that __array_struct__ to
        # achieve that asarray(self) has its base attribute set equal to self,
        # rather than self.bytes, so that lru_cache recognizes successive asarrays
        # to be equal via their common weak referenceable base.
        self.__array_interface__ = numpy.frombuffer(bytes, dtype).reshape(shape).__array_interface__


class frozendict(collections.abc.Mapping):
    '''
    An immutable version of :class:`dict`.  The :class:`frozendict` is hashable
    and both the keys and values should be hashable as well.

    Examples
    --------

    >>> d = frozendict({'spam': 0.0})
    >>> d['spam']
    0.0
    >>> d['spam'] = 1.0
    Traceback (most recent call last):
        ...
    TypeError: 'frozendict' object does not support item assignment
    '''

    def __new__(cls, base):
        if isinstance(base, frozendict):
            return base
        self = object.__new__(cls)
        self.__base = dict(base)
        self.__hash = hash(frozenset(self.__base.items()))  # check immutability and precompute hash
        return self

    @cached_property
    def __nutils_hash__(self):
        h = hashlib.sha1('{}.{}\0'.format(type(self).__module__, type(self).__qualname__).encode())
        for item in sorted(nutils_hash(k)+nutils_hash(v) for k, v in self.items()):
            h.update(item)
        return h.digest()

    def __reduce__(self):
        return frozendict, (self.__base,)

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return False
        if self.__base is other.__base:
            return True
        if self.__hash != other.__hash or self.__base != other.__base:
            return False
        # deduplicate
        self.__base = other.__base
        return True

    __getitem__ = lambda self, item: self.__base.__getitem__(item)
    __iter__ = lambda self: self.__base.__iter__()
    __len__ = lambda self: self.__base.__len__()
    __hash__ = lambda self: self.__hash
    __contains__ = lambda self, key: self.__base.__contains__(key)

    copy = lambda self: self.__base.copy()

    __repr__ = __str__ = lambda self: '{}({})'.format(type(self).__name__, self.__base)


class frozenmultiset(collections.abc.Container):
    '''
    An immutable multiset_.  A multiset is a generalization of a set: items may
    occur more than once.  Two mutlisets are equal if they have the same set of
    items and the same item multiplicities.

    A custom item constructor can be supplied via the notation
    ``frozenmultiset[I]``, with ``I`` the item constructor.  This is shorthand
    for ``lambda items: frozenmultiset(map(I, items))``.  The item constructor
    should be any callable that takes one argument.

    .. _multiset: https://en.wikipedia.org/wiki/Multiset

    Examples
    --------

    >>> a = frozenmultiset(['spam', 'bacon', 'spam'])
    >>> b = frozenmultiset(['sausage', 'spam'])

    The :class:`frozenmultiset` objects support ``+``, ``-`` and ``&`` operators:

    >>> a + b
    frozenmultiset(['spam', 'bacon', 'spam', 'sausage', 'spam'])
    >>> a - b
    frozenmultiset(['bacon', 'spam'])
    >>> a & b
    frozenmultiset(['spam'])

    The order of the items is irrelevant:

    >>> frozenmultiset(['spam', 'spam', 'eggs']) == frozenmultiset(['spam', 'eggs', 'spam'])
    True

    The multiplicities, however, are not:

    >>> frozenmultiset(['spam', 'spam', 'eggs']) == frozenmultiset(['spam', 'eggs'])
    False
    '''

    def __new__(cls, items):
        if isinstance(items, frozenmultiset):
            return items
        self = object.__new__(cls)
        self.__items = tuple(items)
        self.__key = frozenset((item, self.__items.count(item)) for item in self.__items)
        return self

    @cached_property
    def __nutils_hash__(self):
        h = hashlib.sha1('{}.{}\0'.format(type(self).__module__, type(self).__qualname__).encode())
        for item in sorted('{:04d}'.format(count).encode()+nutils_hash(item) for item, count in self.__key):
            h.update(item)
        return h.digest()

    def __and__(self, other):
        '''
        Return a :class:`frozenmultiset` with elements from the left and right hand
        sides with strict positive multiplicity, where the multiplicity is the
        minimum of multiplicitie of the left and right hand side.
        '''

        items = list(self.__items)
        isect = []
        for item in other:
            try:
                items.remove(item)
            except ValueError:
                pass
            else:
                isect.append(item)
        return frozenmultiset(isect)

    def __add__(self, other):
        '''
        Return a :class:`frozenmultiset` with elements from the left and right hand
        sides with a multiplicity equal to the sum of the left and right hand
        sides.
        '''

        return frozenmultiset(self.__items + tuple(other))

    def __sub__(self, other):
        '''
        Return a :class:`frozenmultiset` with elements from the left hand sides with
        a multiplicity equal to the difference of the multiplicity of the left and
        right hand sides, truncated to zero.  Elements with multiplicity zero are
        omitted.
        '''

        items = list(self.__items)
        for item in other:
            try:
                items.remove(item)
            except ValueError:
                pass
        return frozenmultiset(items)

    __reduce__ = lambda self: (frozenmultiset, (self.__items,))
    __hash__ = lambda self: hash(self.__key)
    __eq__ = lambda self, other: type(other) is type(self) and self.__key == other.__key
    __contains__ = lambda self, item: item in self.__items
    __iter__ = lambda self: iter(self.__items)
    __len__ = lambda self: len(self.__items)
    __bool__ = lambda self: bool(self.__items)

    isdisjoint = lambda self, other: not any(item in self.__items for item in other)

    __repr__ = __str__ = lambda self: '{}({})'.format(type(self).__name__, list(self.__items))


def frozenarray(arg, *, copy=True, dtype=None):
    '''
    Create read-only Numpy array.

    Args
    ----
    arg : :class:`numpy.ndarray` or array_like
        Input data.
    copy : :class:`bool`
        If True (the default), do not modify the argument in place. No copy is
        ever forced if the argument is already immutable.
    dtype : :class:`numpy.dtype` or dtype_like, optional
        The desired data-type for the array.

    Returns
    -------
    :class:`numpy.ndarray`
    '''

    if isinstance(arg, numpy.generic):
        return arg
    if isinstance(arg, numpy.ndarray) and dtype in (None, arg.dtype):
        for base in _array_bases(arg):
            if base.flags.writeable:
                if copy:
                    break
                base.flags.writeable = False
        else:
            return arg
    array = numpy.array(arg, dtype=dtype)
    if not array.ndim:
        return array[()]  # convert to generic
    array.flags.writeable = False
    return array


def lru_cache(func):
    '''Buffer-aware cache.

    Returns values from a cache for previously seen arguments. Arguments must be
    hasheable objects or immutable Numpy arrays, the latter identified by the
    underlying buffer. Destruction of the buffer triggers a callback that removes
    the corresponding cache entry.

    At present, any writeable array will silently disable caching. This bevaviour
    is transitional, with future versions requiring that all arrays be immutable.

    .. caution::

        When a decorated function returns an object that references its argument
        (for instance, by returning the argument itself), the cached value keeps
        an argument's reference count from falling to zero, causing the object to
        remain in cache indefinitely. For this reason, care must be taken that
        the decorator is only applied to functions that return objects with no
        references to its arguments.
    '''

    cache = {}

    @functools.wraps(func)
    def wrapped(*args):
        key = []
        bases = []
        for arg in args:
            if isinstance(arg, numpy.ndarray):
                for base in _array_bases(arg):
                    if base.flags.writeable:
                        return func(*args)
                bases.append(base if base.base is None else base.base)
                key.append(tuple(map(arg.__array_interface__.__getitem__, ['data', 'strides', 'shape', 'typestr'])))
            else:
                key.append((type(arg), arg))
        if not bases:
            raise ValueError('arguments must include at least one array')
        key = tuple(key)
        try:
            v, refs_ = cache[key]
        except KeyError:
            v = func(*args)
            assert _isimmutable(v)
            popkey = functools.partial(cache.pop, key)
            cache[key] = v, [weakref.ref(base, popkey) for base in bases]
        return v

    wrapped.cache = cache
    return wrapped


class attributes:
    '''
    Dictionary-like container with attributes instead of keys, instantiated using
    keyword arguments:

    >>> A = attributes(foo=10, bar=True)
    >>> A
    attributes(bar=True, foo=10)
    >>> A.foo
    10
    '''

    def __init__(self, **args):
        self.__dict__.update(args)

    def __eq__(self, other):
        return type(other) == type(self) and other.__dict__ == self.__dict__

    def __repr__(self):
        return 'attributes({})'.format(', '.join(map('{0[0]}={0[1]!r}'.format, sorted(self.__dict__.items()))))


class _deprecation_wrapper:
    def create(self, *args, **kwargs):
        from . import warnings, unit
        warnings.deprecation('nutils.types.unit is deprecated; use nutils.unit.create instead')
        return unit.create(*args, **kwargs)
    __call__ = create


unit = _deprecation_wrapper()
del _deprecation_wrapper


def _array_bases(obj):
    'all ndarray bases starting from and including `obj`'
    while isinstance(obj, numpy.ndarray):
        yield obj
        obj = obj.base


def _isimmutable(obj):
    return obj is None \
        or isinstance(obj, (Immutable, bool, int, float, complex, str, bytes, frozenset, numpy.generic)) \
        or isinstance(obj, tuple) and all(_isimmutable(item) for item in obj) \
        or isinstance(obj, frozendict) and all(_isimmutable(value) for value in obj.values()) \
        or isinstance(obj, numpy.ndarray) and not any(base.flags.writeable for base in _array_bases(obj))


class _hashable_function_wrapper:

    def __init__(self, wrapped, identifier):
        self.__nutils_hash__ = nutils_hash(('hashable_function', identifier))
        functools.update_wrapper(self, wrapped)

    def __call__(*args, **kwargs):
        return args[0].__wrapped__(*args[1:], **kwargs)

    def __get__(self, instance, owner):
        return self

    def __hash__(self):
        return hash(self.__nutils_hash__)

    def __eq__(self, other):
        return type(self) is type(other) and self.__nutils_hash__ == other.__nutils_hash__


def hashable_function(identifier):
    '''Decorator that wraps the decorated function and adds a Nutils hash.

    Return a decorator that wraps the decorated function and adds a Nutils hash
    based solely on the given ``identifier``. The identifier can be anything that has a
    Nutils hash. The identifier should represent the behavior of the function and
    should be changed when the behavior of the function changes.

    If used on methods, this decorator behaves like :func:`staticmethod`.

    Examples
    --------

    Make some function ``func`` hashable:

    >>> @hashable_function('func v1')
    ... def func(a, b):
    ...   return a + b
    ...

    The Nutils hash can be obtained by calling ``nutils_hash`` on ``func``:

    >>> nutils_hash(func).hex()
    'b7fed72647f6a88dd3ce3808b2710eede7d7b5a5'

    Note that the hash is based solely on the identifier passed to the decorator.
    If we create another function ``other`` with the same identifier as ``func``,
    then both have the same hash, despite returning different values.

    >>> @hashable_function('func v1')
    ... def other(a, b):
    ...   return a * b
    ...
    >>> nutils_hash(other) == nutils_hash(func)
    True
    >>> func(1, 2) == other(1, 2)
    False

    The decorator can also be applied on methods:

    >>> class Spam:
    ...   @hashable_function('Spam.eggs v1')
    ...   def eggs(a, b):
    ...     # NOTE: `self` is absent because `hashable_function` behaves like `staticmethod`.
    ...     return a + b
    ...

    The hash of ``eggs`` accessed via the class or an instance is the same:

    >>> spam = Spam()
    >>> nutils_hash(Spam.eggs).hex()
    'dfdbb0ce20b617b17c3b854c23b2b9f7deb94cc6'
    >>> nutils_hash(spam.eggs).hex()
    'dfdbb0ce20b617b17c3b854c23b2b9f7deb94cc6'
    '''

    return functools.partial(_hashable_function_wrapper, identifier=identifier)

# vim:sw=4:sts=4:et
