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

"""
Module with general purpose types.
"""

import inspect, functools, hashlib, builtins, numbers, collections.abc, itertools, abc, sys, weakref, re, io
import numpy

def aspreprocessor(apply):
  '''
  Convert ``apply`` into a preprocessor decorator.  When applied to a function,
  ``wrapped``, the returned decorator preprocesses the arguments with ``apply``
  before calling ``wrapped``.  The ``apply`` function should return a tuple of
  ``args`` (:class:`tuple` or :class:`list`) and ``kwargs`` (:class:`dict`).
  The decorated function ``wrapped`` will be called with ``wrapped(*args,
  **kwargs)``.  The ``apply`` function is allowed to change the signature of
  the decorated function.

  Examples
  --------

  The following preprocessor swaps two arguments.

  >>> @aspreprocessor
  ... def swapargs(a, b):
  ...   return (b, a), {}

  Decorating a function with ``swapargs`` will cause the arguments to be
  swapped before the wrapped function is called.

  >>> @swapargs
  ... def func(a, b):
  ...   return a, b
  >>> func(1, 2)
  (2, 1)
  '''
  def preprocessor(wrapped):
    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
      args, kwargs = apply(*args, **kwargs)
      return wrapped(*args, **kwargs)
    wrapper.__preprocess__ = apply
    wrapper.__signature__ = inspect.signature(apply)
    return wrapper
  return preprocessor

def _build_apply_annotations(signature):
  try:
    # Find a prefix for internal variables that is guaranteed to be
    # collision-free with the parameter names of `signature`.
    for i in itertools.count():
      internal_prefix = '__apply_annotations_internal{}_'.format(i)
      if not any(name.startswith(internal_prefix) for name in signature.parameters):
        break
    # The `l` dictionary is used as locals when compiling the `apply` function.
    l = {}
    # Function to add `obj` to the locals `l`.  Returns the name of the
    # variable (in `l`) that refers to `obj`.
    def add_local(obj):
      name = '{}{}'.format(internal_prefix, len(l))
      assert name not in l
      l[name] = obj
      return name
    # If there are positional-only parameters and there is no var-keyword
    # parameter, we can create an equivalent signature with positional-only
    # parameters converted to positional-or-keyword with unused names.
    if any(param.kind == param.POSITIONAL_ONLY for param in signature.parameters.values()) and not any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
      n_positional_args = 0
      new_params = []
      for param in signature.parameters.values():
        if param.kind == param.POSITIONAL_ONLY:
          param = param.replace(kind=param.POSITIONAL_OR_KEYWORD, name='{}pos{}'.format(internal_prefix, n_positional_args))
        new_params.append(param)
      equiv_signature = signature.replace(parameters=new_params)
    else:
      equiv_signature = signature
    # We build the following function
    #
    #   def apply(<params>):
    #     <body>
    #     return (<args>), {<kwargs>}
    #
    # `params`, `body`, `args` and `kwargs` are lists of valid python code (as `str`).
    params = []
    body = []
    args = []
    kwargs = []
    allow_positional = True
    for name, param in equiv_signature.parameters.items():
      if param.kind == param.KEYWORD_ONLY and allow_positional:
        allow_positional = False
        params.append('*')
      if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
        p = name
        if param.default is not param.empty:
          p = '{}={}'.format(p, add_local(param.default))
        params.append(p)
        if allow_positional:
          args.append(name)
        else:
          kwargs.append('{0!r}:{0}'.format(name))
      elif param.kind == param.VAR_POSITIONAL:
        allow_positional = False
        p = '*{}'.format(name)
        params.append(p)
        args.append(p)
      elif param.kind == param.VAR_KEYWORD:
        allow_positional = False
        p = '**{}'.format(name)
        params.append(p)
        kwargs.append(p)
      else:
        raise ValueError('Cannot create function definition with parameter {}.'.format(param))
      if param.annotation is param.empty:
        pass
      elif param.default is None:
        # Omit the annotation if the argument is the default is None.
        body.append('  {arg} = None if {arg} is None else {ann}({arg})\n'.format(arg=name, ann=add_local(param.annotation)))
      else:
        body.append('  {arg} = {ann}({arg})\n'.format(arg=name, ann=add_local(param.annotation)))
    f = 'def apply({params}):\n{body}  return ({args}), {{{kwargs}}}\n'.format(params=','.join(params), body=''.join(body), args=''.join(arg+',' for arg in args), kwargs=','.join(kwargs))
    exec(f, l)
    apply = l['apply']
  except ValueError:
    def apply(*args, **kwargs):
      bound = signature.bind(*args, **kwargs)
      bound.apply_defaults()
      for name, param in signature.parameters.items():
        if param.annotation is param.empty:
          continue
        if param.default is None and bound.arguments[name] is None:
          continue
        bound.arguments[name] = param.annotation(bound.arguments[name])
      return bound.args, bound.kwargs
    # Copy the signature of `wrapped` without annotations.  This matches the
    # behaviour of the compiled `apply` above.
    apply.__signature__ = inspect.Signature(parameters=[param.replace(annotation=param.empty) for param in signature.parameters.values()])
  apply.returns_canonical_arguments = True
  return apply

def apply_annotations(wrapped):
  '''
  Decorator that applies annotations to arguments.  All annotations should be
  :any:`callable`\\s taking one argument and returning a transformed argument.
  All annotations are strongly recommended to be idempotent_.

  .. _idempotent: https://en.wikipedia.org/wiki/Idempotence

  If a parameter of the decorated function has a default value ``None`` and the
  value of this parameter is ``None`` as well, the annotation is omitted.

  Examples
  --------

  Consider the following function.

  >>> @apply_annotations
  ... def f(a:tuple, b:int):
  ...   return a + (b,)

  When calling ``f`` with a :class:`list` and :class:`str` as arguments, the
  :func:`apply_annotations` decorator first applies :class:`tuple` and
  :class:`int` to the arguments before passing them to the decorated function.

  >>> f([1, 2], '3')
  (1, 2, 3)

  The following example illustrates the behavior of parameters with default
  value ``None``.

  >>> addone = lambda x: x+1
  >>> @apply_annotations
  ... def g(a:addone=None):
  ...   return a

  When calling ``g`` without arguments or with argument ``None``, the
  annotation ``addone`` is not applied.  Note that ``None + 1`` would raise an
  exception.

  >>> g() is None
  True
  >>> g(None) is None
  True

  When passing a different value, the annotation is applied:

  >>> g(1)
  2
  '''
  signature = inspect.signature(wrapped)
  if all(param.annotation is param.empty for param in signature.parameters.values()):
    return wrapped
  else:
    return aspreprocessor(_build_apply_annotations(signature))(wrapped)

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
  return _build_apply_annotations(inspect.Signature(parameters=[param.replace(annotation=param.empty) for param in signature.parameters.values()]))

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

  t = type(data)
  h = hashlib.sha1(t.__name__.encode()+b'\0')
  if data is Ellipsis:
    pass
  elif data is None:
    pass
  elif any(data is dtype for dtype in (bool, int, float, complex, str, bytes, builtins.tuple, frozenset, type(Ellipsis), type(None))):
    h.update(hashlib.sha1(data.__name__.encode()).digest())
  elif any(t is dtype for dtype in (bool, int, float, complex)):
    h.update(hashlib.sha1(repr(data).encode()).digest())
  elif t is str:
    h.update(hashlib.sha1(data.encode()).digest())
  elif t is bytes:
    h.update(hashlib.sha1(data).digest())
  elif t is builtins.tuple:
    for item in data:
      h.update(nutils_hash(item))
  elif t is frozenset:
    for item in sorted(map(nutils_hash, data)):
      h.update(item)
  elif issubclass(t, io.BufferedIOBase) and data.seekable() and not data.writable():
    pos = data.tell()
    h.update(str(pos).encode())
    data.seek(0)
    map(h.update, iter(lambda: data.read(0x20000), b''))
    data.seek(pos)
  else:
    raise TypeError('unhashable type: {!r} {!r}'.format(data, t))
  return h.digest()

class _CacheMeta_property:
  '''
  Memoizing property used by :class:`CacheMeta`.
  '''

  _self = object()

  def __init__(self, prop, cache_attr):
    assert isinstance(prop, property)
    self.fget = prop.fget
    self.cache_attr = cache_attr
    self.__doc__ = prop.__doc__

  def __get__(self, instance, owner):
    if instance is None:
      return self
    try:
      cached_value = getattr(instance, self.cache_attr)
    except AttributeError:
      value = self.fget(instance)
      setattr(instance, self.cache_attr, value if value is not instance else self._self)
      return value
    else:
      return cached_value if cached_value is not self._self else instance

  def __set__(self, instance, value):
    raise AttributeError("can't set attribute")

  def __delete__(self, instance):
    raise AttributeError("can't delete attribute")

def _CacheMeta_method(func, cache_attr):
  '''
  Memoizing method decorator used by :class:`CacheMeta`.
  '''

  _self = object()

  orig_func = func
  signature = inspect.signature(func)
  if not hasattr(func, '__preprocess__') and len(signature.parameters) == 1 and next(iter(signature.parameters.values())).kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):

    def wrapper(self):
      try:
        cached_value = getattr(self, cache_attr)
        value = self if cached_value is _self else cached_value
      except AttributeError:
        value = func(self)
        assert hash(value), 'cannot cache function because the return value is not hashable'
        setattr(self, cache_attr, _self if value is self else value)
      return value

  else:

    # Peel off the preprocessors (see `aspreprocessor`).
    preprocessors = []
    while hasattr(func, '__preprocess__'):
      preprocessors.append(func.__preprocess__)
      func = func.__wrapped__
    if not preprocessors or not getattr(preprocessors[-1], 'returns_canonical_arguments', False):
      preprocessors.append(argument_canonicalizer(inspect.signature(func)))

    def wrapper(*args, **kwargs):
      self = args[0]

      # Apply preprocessors.
      for preprocess in preprocessors:
        args, kwargs = preprocess(*args, **kwargs)
      key = args[1:], tuple(sorted(kwargs.items()))

      assert hash(key), 'cannot cache function because arguments are not hashable'

      # Fetch cached value, if any, and return cached value if args match.
      try:
        cached_key, cached_value = getattr(self, cache_attr)
      except AttributeError:
        pass
      else:
        if cached_key == key:
          return self if cached_value is _self else cached_value

      value = func(*args, **kwargs)

      assert hash(value), 'cannot cache function because the return value is not hashable'
      setattr(self, cache_attr, (key, _self if value is self else value))

      return value

  wrapper.__name__ = orig_func.__name__
  wrapper.__doc__ = orig_func.__doc__
  wrapper.__signature__ = signature
  return wrapper

# While we do not use `abc.ABCMeta` in `CacheMeta` itself, we will use it in
# many classes having `CacheMeta` as a meta(super)class.  To avoid having to
# write `class MCls(CacheMeta, abc.ABCMeta): pass` everywhere, we simply derive
# from `abc.ABCMeta` here.
class CacheMeta(abc.ABCMeta):
  '''
  Metaclass that adds caching functionality to properties and methods listed in
  the special attribute ``__cache__``.  If an attribute is of type
  :class:`property`, the value of the property will be computed at the first
  attribute access and served from cache subsequently.  If an attribute is a
  method, the arguments and return value are cached and the cached value will
  be used if a subsequent call is made with the same arguments; if not, the
  cache will be overwritten.  The cache lives in private attributes in the
  instance.  The metaclass supports the use of ``__slots__``.  If a subclass
  redefines a cached property or method (in the sense of this metaclass) of a
  base class, the property or method of the subclass is *not* automatically
  cached; ``__cache__`` should be used in the subclass explicitly.

  Examples
  --------

  An example of a class with a cached property:

  >>> class T(metaclass=CacheMeta):
  ...   __cache__ = 'x',
  ...   @property
  ...   def x(self):
  ...     print('uncached')
  ...     return 1

  The print statement is added to illustrate when method ``x`` (as defined
  above) is called:

  >>> t = T()
  >>> t.x
  uncached
  1
  >>> t.x
  1

  An example of a class with a cached method:

  >>> class U(metaclass=CacheMeta):
  ...   __cache__ = 'y',
  ...   def y(self, a):
  ...     print('uncached')
  ...     return a

  Again, the print statement is added to illustrate when the method ``y`` (as defined above) is
  called:

  >>> u = U()
  >>> u.y(1)
  uncached
  1
  >>> u.y(1)
  1
  >>> u.y(2)
  uncached
  2
  >>> u.y(2)
  2
  '''

  def __new__(mcls, name, bases, namespace, **kwargs):
    # Wrap all properties that should be cached and reserve slots.
    if '__cache__' in namespace:
      cache = namespace['__cache__']
      cache = (cache,) if isinstance(cache, str) else tuple(cache)
      cache_attrs = []
      for attr in cache:
        # Apply name mangling (see https://docs.python.org/3/tutorial/classes.html#private-variables).
        if attr.startswith('__') and not attr.endswith('__'):
          attr = '_{}{}'.format(name, attr)
        # Reserve an attribute for caching property values that is reasonably
        # unique, by combining the class and attribute names.  The following
        # artificial situation will fail though, because both the base class
        # and the subclass have the same name, hence the cached properties
        # point to the same attribute for caching:
        #
        #     Class A(metaclass=CacheMeta):
        #       __cache__ = 'x',
        #       @property
        #       def x(self):
        #         return 1
        #
        #     class A(A):
        #       __cache__ = 'x',
        #       @property
        #       def x(self):
        #         return super().x + 1
        #       @property
        #       def y(self):
        #         return super().x
        #
        # With `a = A()`, `a.x` first caches `1`, then `2` and `a.y` will
        # return `2`.  On the other hand, `a.y` calls property `x` of the base
        # class and caches `1` and subsequently `a.x` will return `1` from
        # cache.
        cache_attr = '_CacheMeta__cached_property_{}_{}'.format(name, attr)
        cache_attrs.append(cache_attr)
        if attr not in namespace:
          raise TypeError('Attribute listed in __cache__ is undefined: {}'.format(attr))
        value = namespace[attr]
        if isinstance(value, property):
          namespace[attr] = _CacheMeta_property(value, cache_attr)
        elif inspect.isfunction(value) and not inspect.isgeneratorfunction(value):
          namespace[attr] = _CacheMeta_method(value, cache_attr)
        else:
          raise TypeError("Don't know how to cache attribute {}: {!r}".format(attr, value))
      if '__slots__' in namespace and cache_attrs:
        # Add `cache_attrs` to the slots.
        slots = namespace['__slots__']
        slots = [slots] if isinstance(slots, str) else list(slots)
        for cache_attr in cache_attrs:
          assert cache_attr not in slots, 'Private attribute for caching is listed in __slots__: {}'.format(cache_attr)
          slots.append(cache_attr)
        namespace['__slots__'] = tuple(slots)
    return super().__new__(mcls, name, bases, namespace, **kwargs)

class ImmutableMeta(CacheMeta):

  def __new__(mcls, name, bases, namespace, *, version=0, **kwargs):
    if not isinstance(version, int):
      raise ValueError("'version' should be of type 'int' but got {!r}".format(version))
    cls = super().__new__(mcls, name, bases, namespace, **kwargs)
    # Since we redefine `__call__` here and `inspect.signature(cls)` looks at
    # `cls.__signature__` and if absent the signature of `__call__`, we
    # explicitly copy the signature of `<cls instance>.__init__` to `cls`.
    cls.__signature__ = inspect.signature(cls.__init__.__get__(object(), object))
    # Peel off the preprocessors (see `aspreprocessor`) and store the
    # preprocessors and the uncovered init separately.
    pre_init = []
    init = cls.__init__
    while hasattr(init, '__preprocess__'):
      pre_init.append(init.__preprocess__)
      init = init.__wrapped__
    if not pre_init or not getattr(pre_init[-1], 'returns_canonical_arguments', False):
      pre_init.append(argument_canonicalizer(inspect.signature(init)))
    cls._pre_init = tuple(pre_init)
    cls._init = init
    cls._version = version
    return cls

  def __init__(cls, name, bases, namespace, *, version=0, **kwargs):
    super().__init__(name, bases, namespace, **kwargs)

  def __call__(*args, **kwargs):
    cls = args[0]
    # Use `None` as temporary `self` argument, apply preprocessors and
    # remove the temporary `self`.
    args = None, *args[1:]
    for preprocess in cls._pre_init:
      args, kwargs = preprocess(*args, **kwargs)
    args = args[1:]
    return cls._new(*args, **kwargs)

  def _new(cls, *args, **kwargs):
    self = cls.__new__(cls)
    self._args = args
    self._kwargs = kwargs
    self._hash = hash(args + tuple((key, kwargs[key]) for key in sorted(kwargs)))
    self._init(*args, **kwargs)
    return self

class Immutable(metaclass=ImmutableMeta):
  '''
  Base class for immutable types.  This class adds equality tests, traditional
  hashing (:func:`hash`), nutils hashing (:func:`nutils_hash`) and pickling,
  all based solely on the (positional) intialization arguments, ``args`` for
  future reference.  Keyword-only arguments are not supported.  All arguments
  should be hashable by :func:`nutils_hash`.

  Positional and keyword initialization arguments are canonicalized
  automatically (by :func:`argument_canonicalizer`).  If the ``__init__``
  method of a subclass is decorated with preprocessors (see
  :func:`aspreprocessor`), the preprocessors are applied to the initialization
  arguments and ``args`` becomes the preprocessed positional part.

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

  This can be solved by adding and applying annotations to ``__init__``.  The
  following class converts its initialization arguments to :class:`tuple`
  automaticaly:

  >>> class Annotated(Immutable):
  ...   @apply_annotations
  ...   def __init__(self, a:tuple, b:tuple):
  ...     pass

  Calling ``Annotated`` with either :class:`list`\\s of ``1, 2`` and ``3, 4``
  or :class:`tuple`\\s gives equal instances:

  >>> Annotated([1, 2], [3, 4]) == Annotated((1, 2), (3, 4))
  True
  '''

  __slots__ = '__weakref__', '_args', '_kwargs', '_hash'
  __cache__ = '__nutils_hash__',

  def __reduce__(self):
    return self.__class__._new, self._args

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    return self.__class__ is other.__class__ and self._hash == other._hash and self._args == other._args and self._kwargs == other._kwargs

  @property
  def __nutils_hash__(self):
    h = hashlib.sha1('{}.{}:{}\0'.format(type(self).__module__, type(self).__qualname__, type(self)._version).encode())
    for arg in self._args:
      h.update(nutils_hash(arg))
    for name in sorted(self._kwargs):
      h.update(nutils_hash(name))
      h.update(nutils_hash(self._kwargs[name]))
    return h.digest()

  def __getstate__(self):
    raise Exception('getstate should never be called')

  def __setstate__(self, state):
    raise Exception('setstate should never be called')

  def __str__(self):
    return '{}({})'.format(self.__class__.__name__, ','.join(str(arg) for arg in self._args))

  def edit(self, op):
    return self.__class__(*[op(arg) for arg in self._args])

class SingletonMeta(ImmutableMeta):

  def __new__(mcls, name, bases, namespace, **kwargs):
    cls = super().__new__(mcls, name, bases, namespace, **kwargs)
    cls._cache = weakref.WeakValueDictionary()
    return cls

  def _new(cls, *args, **kwargs):
    key = args + tuple((key, kwargs[key]) for key in sorted(kwargs))
    try:
      self = cls._cache[key]
    except KeyError:
      cls._cache[key] = self = super()._new(*args, **kwargs)
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

  Consider the folling class with annotations.

  >>> class Annotated(Singleton):
  ...   @apply_annotations
  ...   def __init__(self, a:tuple, b:tuple):
  ...     pass

  Calling ``Annotated`` with either :class:`list`\\s of ``1, 2`` and ``3, 4``
  or :class:`tuple`\\s gives a single instance:

  >>> Annotated([1, 2], [3, 4]) is Annotated((1, 2), (3, 4))
  True
  '''

  __slots__ = ()

  __hash__ = Immutable.__hash__
  __eq__ = object.__eq__

def strictint(value):
  '''
  Converts any type that is a subclass of :class:`numbers.Integral` (e.g.
  :class:`int` and ``numpy.int64``) to :class:`int`, and fails otherwise.
  Notable differences with the behavior of :class:`int`:

  *   :func:`strictint` does not convert a :class:`str` to an :class:`int`.
  *   :func:`strictint` does not truncate :class:`float` to an :class:`int`.

  Examples
  --------

  >>> strictint(1), type(strictint(1))
  (1, <class 'int'>)
  >>> strictint(numpy.int64(1)), type(strictint(numpy.int64(1)))
  (1, <class 'int'>)
  >>> strictint(1.0)
  Traceback (most recent call last):
      ...
  ValueError: not an integer: 1.0
  >>> strictint('1')
  Traceback (most recent call last):
      ...
  ValueError: not an integer: '1'
  '''

  if not isinstance(value, numbers.Integral):
    raise ValueError('not an integer: {!r}'.format(value))
  return builtins.int(value)

def strictfloat(value):
  '''
  Converts any type that is a subclass of :class:`numbers.Real` (e.g.
  :class:`float` and ``numpy.float64``) to :class:`float`, and fails
  otherwise.  Notable difference with the behavior of :class:`float`:

  *   :func:`strictfloat` does not convert a :class:`str` to an :class:`float`.

  Examples
  --------

  >>> strictfloat(1), type(strictfloat(1))
  (1.0, <class 'float'>)
  >>> strictfloat(numpy.float64(1.2)), type(strictfloat(numpy.float64(1.2)))
  (1.2, <class 'float'>)
  >>> strictfloat(1.2+3.4j)
  Traceback (most recent call last):
      ...
  ValueError: not a real number: (1.2+3.4j)
  >>> strictfloat('1.2')
  Traceback (most recent call last):
      ...
  ValueError: not a real number: '1.2'
  '''

  if not isinstance(value, numbers.Real):
    raise ValueError('not a real number: {!r}'.format(value))
  return builtins.float(value)

def strictstr(value):
  '''
  Returns ``value`` unmodified if it is a :class:`str`, and fails otherwise.
  Notable difference with the behavior of :class:`str`:

  *   :func:`strictstr` does not call ``__str__`` methods of objects to
      automatically convert objects to :class:`str`\\s.

  Examples
  --------

  Passing a :class:`str` to :func:`strictstr` works:

  >>> strictstr('spam')
  'spam'

  Passing an :class:`int` will fail:

  >>> strictstr(1)
  Traceback (most recent call last):
      ...
  ValueError: not a 'str': 1
  '''

  if not isinstance(value, str):
    raise ValueError("not a 'str': {!r}".format(value))
  return value

def _getname(value):
  name = []
  if hasattr(value, '__module__'):
    name.append(value.__module__)
  if hasattr(value, '__qualname__'):
    name.append(value.__qualname__)
  elif hasattr(value, '__name__'):
    name.append(value.__name__)
  else:
    return str(value)
  return '.'.join(name)

def _copyname(dst=None, *, src, suffix=''):
  if dst is None:
    return functools.partial(_copyname, src=src, suffix=suffix)
  if hasattr(src, '__name__'):
    dst.__name__ = src.__name__+suffix
  if hasattr(src, '__qualname__'):
    dst.__qualname__ = src.__qualname__+suffix
  if hasattr(src, '__module__'):
    dst.__module__ = src.__module__
  return dst

class _strictmeta(type):
  def __getitem__(self, cls):
    def constructor(value):
      if not isinstance(value, cls):
        raise ValueError('expected an object of type {!r} but got {!r} with type {!r}'.format(cls.__qualname__, value, type(value).__qualname__))
      return value
    constructor.__qualname__ = constructor.__name__ = 'strict[{}]'.format(_getname(cls))
    return constructor
  def __call__(*args, **kwargs):
    raise TypeError("cannot create an instance of class 'strict'")

class strict(metaclass=_strictmeta):
  '''
  Type checker.  The function ``strict[cls](value)`` returns ``value``
  unmodified if ``value`` is an instance of ``cls``, otherwise a
  :class:`ValueError` is raised.

  Examples
  --------

  The ``strict[int]`` function passes integers unmodified:

  >>> strict[int](1)
  1

  Other types fail:

  >>> strict[int]('1')
  Traceback (most recent call last):
      ...
  ValueError: expected an object of type 'int' but got '1' with type 'str'
  '''

class _tuplemeta(type):
  def __getitem__(self, itemtype):
    @_copyname(src=self, suffix='[{}]'.format(_getname(itemtype)))
    def constructor(value):
      return builtins.tuple(map(itemtype, value))
    return constructor
  @staticmethod
  def __call__(*args, **kwargs):
    return builtins.tuple(*args, **kwargs)

class tuple(builtins.tuple, metaclass=_tuplemeta):
  '''
  Wrapper of :class:`tuple` that supports a user-defined item constructor via
  the notation ``tuple[I]``, with ``I`` the item constructor.  This is
  shorthand for ``lambda items: tuple(map(I, items))``.  The item constructor
  should be any callable that takes one argument.

  Examples
  --------

  A tuple with items processed with :func:`strictint`:

  >>> tuple[strictint]((False, 1, 2, numpy.int64(3)))
  (0, 1, 2, 3)

  If the item constructor raises an exception, the construction of the
  :class:`tuple` failes accordingly:

  >>> tuple[strictint]((1, 2, 3.4))
  Traceback (most recent call last):
      ...
  ValueError: not an integer: 3.4
  '''

  __slots__ = ()

class _frozendictmeta(CacheMeta):
  def __getitem__(self, keyvaluetype):
    if not isinstance(keyvaluetype, builtins.tuple) or len(keyvaluetype) != 2:
      raise RuntimeError("expected a 'tuple' of length 2 but got {!r}".format(keyvaluetype))
    keytype, valuetype = keyvaluetype
    @_copyname(src=self, suffix='[{},{}]'.format(_getname(keytype), _getname(valuetype)))
    def constructor(arg):
      if isinstance(arg, collections.abc.Mapping):
        items = arg.items()
      elif isinstance(arg, (collections.abc.MappingView, collections.abc.Iterable)):
        items = arg
      else:
        raise ValueError('expected a mapping or iterable but got {!r}'.format(arg))
      return self((keytype(key), valuetype(value)) for key, value in items)
    return constructor

class frozendict(collections.abc.Mapping, metaclass=_frozendictmeta):
  '''
  An immutable version of :class:`dict`.  The :class:`frozendict` is hashable
  and both the keys and values should be hashable as well.

  Custom key and value constructors can be supplied via the ``frozendict[K,V]``
  notation, with ``K`` the key constructor and ``V`` the value constructor,
  which is roughly equivalent to ``lambda *args, **kwargs: {K(k): V(v) for k, v
  in dict(*args, **kwargs).items()}``.

  Examples
  --------

  A :class:`frozendict` with :func:`strictstr` as key constructor and
  :func:`strictfloat` as value constructor:

  >>> frozendict[strictstr,strictfloat]({'spam': False})
  frozendict({'spam': 0.0})

  Similar but with non-strict constructors:

  >>> frozendict[str,float]({None: '1.2'})
  frozendict({'None': 1.2})

  Applying the strict constructors to invalid data raises an exception:

  >>> frozendict[strictstr,strictfloat]({None: '1.2'})
  Traceback (most recent call last):
      ...
  ValueError: not a 'str': None
  '''

  __slots__ = '__base', '__hash'
  __cache__ = '__nutils_hash__',

  def __new__(cls, base):
    if isinstance(base, frozendict):
      return base
    self = object.__new__(cls)
    self.__base = dict(base)
    self.__hash = hash(frozenset(self.__base.items())) # check immutability and precompute hash
    return self

  @property
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
    if not isinstance(other, frozendict):
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

class _frozenmultisetmeta(CacheMeta):
  def __getitem__(self, itemtype):
    @_copyname(src=self, suffix='[{}]'.format(_getname(itemtype)))
    def constructor(value):
      return self(map(itemtype, value))
    return constructor

class frozenmultiset(collections.abc.Container, metaclass=_frozenmultisetmeta):
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

  __slots__ = '__items', '__key'
  __cache__ = '__nutils_hash__',

  def __new__(cls, items):
    if isinstance(items, frozenmultiset):
      return items
    self = object.__new__(cls)
    self.__items = tuple(items)
    self.__key = frozenset((item, self.__items.count(item)) for item in self.__items)
    return self

  @property
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
  __eq__ = lambda self, other: isinstance(other, frozenmultiset) and self.__key == other.__key
  __contains__ = lambda self, item: item in self.__items
  __iter__ = lambda self: iter(self.__items)
  __len__ = lambda self: len(self.__items)
  __bool__ = lambda self: bool(self.__items)

  isdisjoint = lambda self, other: not any(item in self.__items for item in other)

  __repr__ = __str__ = lambda self: '{}({})'.format(type(self).__name__, list(self.__items))

class _frozenarraymeta(CacheMeta):
  def __getitem__(self, dtype):
    @_copyname(src=self, suffix='[{}]'.format(_getname(dtype)))
    def constructor(value):
      return self(value, dtype=dtype)
    return constructor

class frozenarray(collections.abc.Sequence, metaclass=_frozenarraymeta):
  '''
  An immutable version (and drop-in replacement) of :class:`numpy.ndarray`.

  Besides being immutable, the :class:`frozenarray` differs from
  :class:`numpy.ndarray` in (in)equality tests.  Given two :class:`frozenarray`
  objects ``a`` and ``b``, the test ``a == b`` returns ``True`` if both arrays
  are equal in its entirety, including dtype and shape, while the same test
  with :class:`numpy.ndarray` objects would give a boolean array with
  element-wise thruth values.

  The constructor with predefined ``dtype`` argument can generated via the
  notation ``frozenarray[dtype]``.  This is shorthand for ``lambda base:
  frozenarray(base, dtype=dtype)``.

  Parameters
  ----------
  base : :class:`numpy.ndarray` or array-like
      The array data.
  dtype
      The dtype of the array or ``None``.
  copy : :class:`bool`
      If ``base`` is a :class:`frozenarray` and the ``dtype`` matches or is
      ``None``, this argument is ignored.  If ``base`` is a
      :class:`numpy.ndarray` and the ``dtype`` matches or is ``None`` and
      ``copy`` is ``False``, ``base`` is stored as is.  Otherwise ``base`` is
      copied.
  '''

  __slots__ = '__base'
  __cache__ = '__nutils_hash__', '__hash__'

  @staticmethod
  def full(shape, fill_value):
    return frozenarray(numpy.lib.stride_tricks.as_strided(fill_value, shape, [0]*len(shape)), copy=False)

  def __new__(cls, base, dtype=None, copy=True):
    isstrict = dtype in (strictint, strictfloat)
    if dtype is None:
      pass
    elif dtype == bool:
      dtype = bool
    elif dtype in (int, strictint):
      dtype = int
    elif dtype in (float, strictfloat):
      dtype = float
    elif dtype == complex:
      dtype = complex
    else:
      raise ValueError('unsupported dtype: {!r}'.format(dtype))
    if isinstance(base, frozenarray):
      if dtype is None or dtype == base.dtype:
        return base
      base = base.__base
    if isstrict:
      if not isinstance(base, numpy.ndarray):
        base = numpy.array(base)
        if base.size == 0:
          base = base.astype(dtype)
        copy = False
      if base.dtype == complex or base.dtype == float and dtype == int:
        raise ValueError('downcasting {!r} to {!r} is forbidden'.format(base.dtype, dtype))
    self = object.__new__(cls)
    self.__base = numpy.array(base, dtype=dtype) if copy or not isinstance(base, numpy.ndarray) or dtype and dtype != base.dtype else base
    self.__base.flags.writeable = False
    return self

  def __hash__(self):
    return hash((self.__base.shape, self.__base.dtype, tuple(self.__base.flat[::self.__base.size//32+1]) if self.__base.size else ())) # NOTE special case self.__base.size == 0 necessary for numpy<1.12

  @property
  def __nutils_hash__(self):
    h = hashlib.sha1('{}.{}\0{} {}'.format(type(self).__module__, type(self).__qualname__, self.__base.shape, self.__base.dtype.str).encode())
    h.update(self.__base.tobytes())
    return h.digest()

  @property
  def __array_struct__(self):
    return self.__base.__array_struct__

  def __reduce__(self):
    return frozenarray, (self.__base, None, False)

  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, frozenarray):
      return False
    if self.__base is other.__base:
      return True
    if hash(self) != hash(other) or self.__base.dtype != other.__base.dtype or self.__base.shape != other.__base.shape or numpy.not_equal(self.__base, other.__base).any():
      return False
    # deduplicate
    self.__base = other.__base
    return True

  def __lt__(self, other):
    if not isinstance(other, frozenarray):
      return NotImplemented
    return self != other and (self.dtype < other.dtype
      or self.dtype == other.dtype and (self.shape < other.shape
        or self.shape == other.shape and self.__base.tolist() < other.__base.tolist()))

  def __le__(self, other):
    if not isinstance(other, frozenarray):
      return NotImplemented
    return self == other or (self.dtype < other.dtype
      or self.dtype == other.dtype and (self.shape < other.shape
        or self.shape == other.shape and self.__base.tolist() < other.__base.tolist()))

  def __gt__(self, other):
    if not isinstance(other, frozenarray):
      return NotImplemented
    return self != other and (self.dtype > other.dtype
      or self.dtype == other.dtype and (self.shape > other.shape
        or self.shape == other.shape and self.__base.tolist() > other.__base.tolist()))

  def __ge__(self, other):
    if not isinstance(other, frozenarray):
      return NotImplemented
    return self == other or (self.dtype > other.dtype
      or self.dtype == other.dtype and (self.shape > other.shape
        or self.shape == other.shape and self.__base.tolist() > other.__base.tolist()))

  def __getitem__(self, item):
    retval = self.__base.__getitem__(item)
    return frozenarray(retval, copy=False) if isinstance(retval, numpy.ndarray) else retval

  dtype = property(lambda self: self.__base.dtype)
  shape = property(lambda self: self.__base.shape)
  size = property(lambda self: self.__base.size)
  ndim = property(lambda self: self.__base.ndim)
  flat = property(lambda self: self.__base.flat)
  T = property(lambda self: frozenarray(self.__base.T, copy=False))

  __len__ = lambda self: self.__base.__len__()
  __repr__ = lambda self: 'frozenarray'+self.__base.__repr__()[5:]
  __str__ = lambda self: self.__base.__str__()
  __add__ = lambda self, other: self.__base.__add__(other)
  __radd__ = lambda self, other: self.__base.__radd__(other)
  __sub__ = lambda self, other: self.__base.__sub__(other)
  __rsub__ = lambda self, other: self.__base.__rsub__(other)
  __mul__ = lambda self, other: self.__base.__mul__(other)
  __rmul__ = lambda self, other: self.__base.__rmul__(other)
  __truediv__ = lambda self, other: self.__base.__truediv__(other)
  __rtruediv__ = lambda self, other: self.__base.__rtruediv__(other)
  __floordiv__ = lambda self, other: self.__base.__floordiv__(other)
  __rfloordiv__ = lambda self, other: self.__base.__rfloordiv__(other)
  __pow__ = lambda self, other: self.__base.__pow__(other)
  __int__ = lambda self: self.__base.__int__()
  __float__ = lambda self: self.__base.__float__()
  __abs__ = lambda self: self.__base.__abs__()
  __neg__ = lambda self: self.__base.__neg__()
  __invert__ = lambda self: self.__base.__invert__()
  __or__ = lambda self, other: self.__base.__or__(other)
  __ror__ = lambda self, other: self.__base.__ror__(other)
  __and__ = lambda self, other: self.__base.__and__(other)
  __rand__ = lambda self, other: self.__base.__rand__(other)
  __xor__ = lambda self, other: self.__base.__xor__(other)
  __rxor__ = lambda self, other: self.__base.__rxor__(other)

  tolist = lambda self, *args, **kwargs: self.__base.tolist(*args, **kwargs)
  copy = lambda self, *args, **kwargs: self.__base.copy(*args, **kwargs)
  astype = lambda self, *args, **kwargs: self.__base.astype(*args, **kwargs)
  take = lambda self, *args, **kwargs: self.__base.take(*args, **kwargs)
  any = lambda self, *args, **kwargs: self.__base.any(*args, **kwargs)
  all = lambda self, *args, **kwargs: self.__base.all(*args, **kwargs)
  sum = lambda self, *args, **kwargs: self.__base.sum(*args, **kwargs)
  min = lambda self, *args, **kwargs: self.__base.min(*args, **kwargs)
  max = lambda self, *args, **kwargs: self.__base.max(*args, **kwargs)
  prod = lambda self, *args, **kwargs: self.__base.prod(*args, **kwargs)
  dot = lambda self, *args, **kwargs: self.__base.dot(*args, **kwargs)
  argsort = lambda self, *args, **kwargs: self.__base.argsort(*args, **kwargs)
  swapaxes = lambda self, *args, **kwargs: frozenarray(self.__base.swapaxes(*args, **kwargs), copy=False)
  ravel = lambda self, *args, **kwargs: frozenarray(self.__base.ravel(*args, **kwargs), copy=False)
  reshape = lambda self, *args, **kwargs: frozenarray(self.__base.reshape(*args, **kwargs), copy=False)
  transpose = lambda self, *args, **kwargs: frozenarray(self.__base.transpose(*args, **kwargs), copy=False)
  cumsum = lambda self, *args, **kwargs: frozenarray(self.__base.cumsum(*args, **kwargs), copy=False)
  nonzero = lambda self, *args, **kwargs: frozenarray(self.__base.nonzero(*args, **kwargs), copy=False)

class _c_arraymeta(type):
  def __getitem__(self, dtype):
    def constructor(value):
      if isinstance(value, numpy.core._internal._ctypes):
        return value
      if not isinstance(value, numpy.ndarray):
        value = numpy.array(value, dtype=dtype)
      if not value.flags.c_contiguous:
        raise ValueError('Array is not contiguous.')
      if value.dtype != dtype:
        raise ValueError('Expected dtype {} but array has dtype {}.'.format(dtype, value.dtype))
      return value.ctypes
    constructor.__qualname__ = constructor.__name__ = 'c_array[{}]'.format(_getname(dtype))
    return constructor
  def __call__(*args, **kwargs):
    raise TypeError("cannot create an instance of class 'c_array'")

class c_array(metaclass=_c_arraymeta):
  '''
  Converts an array-like object to a ctypes array with a specific dtype.  The
  function ``c_array[dtype](array)`` returns ``array`` unmodified if ``array``
  is already a ctypes array.  If ``array`` is a :class:`numpy.ndarray`, the
  array is converted if the ``dtype`` is correct and the array is contiguous;
  otherwise :class:`ValueError` is raised.  Otherwise, ``array`` is first
  converted to a contiguous :class:`numpy.ndarray` and then converted to ctypes
  array.  In the first two cases changes made to the ctypes array are reflected
  by the ``array`` argument: both are essentially views of the same data.  In
  the third case, changes to either ``array`` or the returned ctypes array are
  not reflected by the other.
  '''

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
  def __repr__(self):
    return 'attributes({})'.format(', '.join(map('{0[0]}={0[1]!r}'.format, sorted(self.__dict__.items()))))

class unit:
  '''
  Framework for physical units.

  The unit class provides a basic framework for specifying values with physical
  units using readable notation such as ``2.5km/h``. The system ensures that
  values are consistent with a measurement system derived from base units, but
  it does impose or even preload one such system. Instead a derived class,
  created using either :func:`unit.create` or class arguments, should specify
  the units and scales relevant for the situation to which it is applied.

  Once units are defined, the formal syntax for instantiating a quantity is:

  .. code:: BNF

      <quantity> ::= <number> <units> | <number> <operator> <units>
      <number>   ::= "" | <integer> | <integer> "." <integer>
                  ; Numerical value, allowing for decimal fractions but not
                  ; scientific notation. An empty number is equivalent to 1.
      <units>    ::= <unit> | <unit> <operator> <units>
      <unit>     ::= <prefix> <name> <power>
      <prefix>   ::= "" | "h" | "k" | "M" | "G" | "T" | "P" | "E" | "Z" | "Y"
                  | "d" | "c" | "m" | "μ" | "n" | "p" | "f" | "a" | "z" | "y"
                  ; Single character prefix to indicate a multiple or fraction
                  ; of the unit. All SI prefixes are supported except for deca.
                  ; An empty prefix signifies no scaling.
      <name>     ::= <string>
                  ; One of the defined units, case sensitive, containing Latin
                  ; or Greek symbols.
      <power>    ::= "" | <integer>
                  ; Integer power to which to raise the unit. An empty power is
                  ; equivalent to 1.
      <operator> ::= "*" | "/"
                  ; Multiplication or division.

  With the prefix and unit name sharing an alphabet there is potential for
  ambiguities (is it mol or micro-ol?). These are resolved using the simple
  logic that the first character is considered part of the unit if this unit
  exists; otherwise it is considered a prefix.
  '''

  _words = re.compile('([a-zA-Zα-ωΑ-Ω]+)')
  _prefix = dict(Y=1e24, Z=1e21, E=1e18, P=1e15, T=1e12, G=1e9, M=1e6, k=1e3, h=1e2,
    d=1e-1, c=1e-2, m=1e-3, μ=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18, z=1e-21, y=1e-24)

  class _pdict(dict):
    '''
    Minimal helper class for sparse addition, multiplication
    '''
    def __iadd__(self, other):
      for key, value in other.items():
        value += self.pop(key, 0)
        if value:
          self[key] = value
      return self
    def __mul__(self, other):
      return {key: value*other for key, value in self.items()}
    def __str__(self):
      return ''.join(sorted('*/'[value<0] + key + (str(abs(value)) if abs(value) > 1 else '') for key, value in self.items())).lstrip('*') or 'dimensionless'

  @classmethod
  def create(*args, **units):
    '''
    Alternative constructor for backwards compatibility.
    '''
    return args[0](**units)

  def __init__(*args, **units):
    '''
    Create new unit type.

    The unit system is defined via variable keyword arguments, with every unit
    specified either as a direct numerical value or as a string referencing
    other units using the standard expression syntax. Ultimately every unit
    should be resolvable to a numerical value by tracing its dependencies.

    The following example defines a subset of the SI system. Note that we
    cannot use prefixes on the receiving end of a definition for reasons of
    ambiguity, hence the definition of a gram as 1/1000:

    >>> SI = unit(m=1, s=1, g=1e-3, N='kg*m/s2', Pa='N/m2')
    >>> SI('2km')
    2000.0
    >>> SI('2g')
    0.002

    Args
    ----
    name : :class:`str` (optional, positional only)
        Name of the new class object.
    **units :
        Unit definitions.

    Returns
    -------
    :
        The newly created (uninitiated) unit class.
    '''
    self, = args
    def depth(name, d={}):
      if name not in units:
        name = name[1:] # strip prefix
      if name not in d:
        value = units.get(name)
        d[name] = isinstance(value, str) and sum(map(depth, self._words.findall(value)), 1)
      return d[name]
    self._units = {}
    for name in sorted(units, key=depth): # sort by dependency depth to establish resolve order
      value = units[name]
      self._units[name] = self._parse(value) if isinstance(value, str) else (value, self._pdict({name: 1}))

  def __getitem__(self, unit):
    '''
    Create subclass of float with custom stringly loads, dumps methods.
    '''
    if unit[0] in '1234567890.*':
      raise ValueError('unit cannot start with a numeral')
    return type('unit:'+unit, (float,), dict(unit=unit, __stringly_loads__=classmethod(self._loads), __stringly_dumps__=classmethod(self._dumps)))

  def __call__(self, s):
    '''
    Create subclass of float and instantiate.
    '''
    value, powers = self._parse(s)
    return self[s.lstrip('1234567890.*')](value)

  def _parse(self, s):
    '''
    Parse string into a tuple of float, _pdict.
    '''
    parts = self._words.split(s)
    value = float(parts[0].rstrip('*/') or 1)
    powers = self._pdict()
    for i in range(1, len(parts), 2):
      s = int(parts[i+1].rstrip('*/') or 1)
      if parts[i-1].endswith('/'):
        s = -s
      name = parts[i]
      if name not in self._units:
        if name[0] not in self._prefix or name[1:] not in self._units:
          raise ValueError('unknown unit: {}'.format(name))
        v, p = self._units[name[1:]]
        v *= self._prefix[name[0]]
      else:
        v, p = self._units[name]
      value *= v**s
      powers += p*s
    return value, powers

  def _loads(self, U, s):
    uvalue, upowers = self._parse(U.unit)
    value, powers = self._parse(s)
    if powers != upowers:
      raise ValueError('invalid unit: expected {}, got {}'.format(upowers, powers))
    return value

  def _dumps(self, U, v):
    if not isinstance(v, (int,float)):
      raise ValueError('can only dump numerical values as unit, got {!r}'.format(type(v)))
    uvalue, upowers = self._parse(U.unit)
    return _f2s(v/uvalue) + U.unit

def _f2s(v):
  'convert float to string without scientific notation'
  s, sep, e = str(v).partition('e')
  a, sep, b = s.partition('.')
  pos = len(a) + int(e or 0)
  s = (a + b).rstrip('0')
  if pos >= len(s):
    return s.ljust(pos, '0')
  elif pos <= 0:
    return '0.' + '0' * -pos + s
  else:
    return s[:pos] + '.' + s[pos:]

# vim:sw=2:sts=2:et
