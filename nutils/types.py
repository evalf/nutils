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

import inspect, functools, hashlib, builtins, numbers, itertools, abc
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
  if data is Ellipsis:
    hargs = ()
  elif data is None:
    hargs = ()
  elif any(data is dtype for dtype in (bool, int, float, complex, str, bytes, tuple, frozenset, type(Ellipsis), type(None))):
    hargs = hashlib.sha1(data.__name__.encode()).digest(),
  elif any(t is dtype for dtype in (bool, int, float, complex)):
    hargs = hashlib.sha1(repr(data).encode()).digest(),
  elif t is str:
    hargs = hashlib.sha1(data.encode()).digest(),
  elif t is bytes:
    hargs = hashlib.sha1(data).digest(),
  elif t is tuple:
    hargs = map(nutils_hash, data)
  elif t is frozenset:
    hargs = sorted(map(nutils_hash, data))
  else:
    raise TypeError('unhashable type: {!r}'.format(data))

  h = hashlib.sha1(t.__name__.encode()+b'\0')
  for harg in hargs:
    h.update(harg)
  return h.digest()

class _CacheMeta_property:
  '''
  Memoizing property used by :class:`CacheMeta`.
  '''

  _self = object()

  def __init__(self, fget, cache_attr):
    self.fget = fget
    self.cache_attr = cache_attr

  def __get__(self, instance, owner):
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
          namespace[attr] = _CacheMeta_property(value.fget, cache_attr)
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

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
