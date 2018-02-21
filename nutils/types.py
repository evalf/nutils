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

import inspect, functools, hashlib, itertools

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

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
