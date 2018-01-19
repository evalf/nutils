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
The cache module.
"""

from . import config, log, numeric, util
import os, sys, numpy, functools, inspect, builtins

def property(f):
  _self = object()
  _temp = object()
  _name = f.__name__
  def property_getter(self):
    try:
      dictvalue = self.__dict__[_name]
    except KeyError:
      self.__dict__[_name] = _temp # placeholder for detection of cyclic dependencies
      value = f(self)
      self.__dict__[_name] = value if value is not self else _self
    else:
      assert dictvalue is not _temp, 'attribute {!r} requested during construction'.format(_name)
      value = dictvalue if dictvalue is not _self else self
    return value
  def property_setter(self, value):
    assert _name not in self.__dict__, 'attempting to set attribute {!r} twice'.format(_name)
    self.__dict__[_name] = value if value is not self else _self
  return builtins.property(fget=property_getter, fset=property_setter)

class Wrapper:
  'function decorator that caches results by arguments'

  def __init__( self, func ):
    self.func = func
    self.cache = {}
    self.count = 0
    self.signature = inspect.signature(func)

  def __call__( self, *args, **kwargs ):
    self.count += 1
    bound = self.signature.bind(*args, **kwargs)
    bound.apply_defaults()
    args = bound.args
    assert not bound.kwargs
    try:
      value = self.cache[args]
    except KeyError:
      value = self.func(*args)
      self.cache[args] = value
    return value

  @builtins.property
  def hits( self ):
    return self.count - len(self.cache)

class WrapperCache:
  'maintains a cache for Wrapper instances'

  def __init__( self ):
    self.cache = {}

  def __getitem__( self, func ):
    try:
      wrapper = self.cache[func]
    except KeyError:
      wrapper = Wrapper(func)
      self.cache[func] = wrapper
    return wrapper

  @builtins.property
  def stats( self ):
    hits = count = 0
    for wrapper in self.cache.values():
      hits += wrapper.hits
      count += wrapper.count
    return 'not used' if not count \
      else 'effectivity {}% (hit {}/{} calls over {} functions)'.format(100*hits/count, hits, count, len(self.cache))

class WrapperDummyCache:
  'placeholder object'

  stats = 'caching disabled'

  def __getitem__( self, func ):
    return func

class ImmutableMeta(type):

  _cleanup_threshold = 1000 # number of new instances until next cleanup

  def __init__(cls, *args, **kwargs):
    super().__init__(*args, **kwargs)
    signature = inspect.signature(cls.__init__)
    param0, *params = signature.parameters.values()
    cls._signature = inspect.Signature(params)
    cls._annotations = [(param.name, param.annotation) for param in params if param.annotation != param.empty]
    cls._cache = {}
    cls._init = cls.__init__
    if cls._annotations:
      cls.__init__ = util.enforcetypes(cls.__init__, signature)

  def __call__(cls, *args, **kwargs):
    bound = cls._signature.bind(*args, **kwargs)
    bound.apply_defaults()
    for name, op in cls._annotations:
      bound.arguments[name] = op(bound.arguments[name])
    assert not bound.kwargs
    return cls._new(*bound.args)

  def _new(cls, *args):
    try:
      self = cls._cache[args]
    except KeyError:
      self = cls.__new__(cls)
      self._args = args
      self._hash = hash(args)
      self._init(*args)
      cls._cache[args] = self
      if len(cls._cache) > cls._cleanup_threshold:
        cls._cache = {key: value for key, value in cls._cache.items() if sys.getrefcount(value) > 4}
        cls._cleanup_threshold = ImmutableMeta._cleanup_threshold + len(cls._cache)
    return self

class Immutable(metaclass=ImmutableMeta):

  def __init__( self ):
    pass

  def __reduce__( self ):
    return self.__class__._new, self._args

  def __hash__( self ):
    return self._hash

  def __lt__(self, other):
    return self is not other and (self.__class__.__name__,)+self._args < (other.__class__.__name__,)+other._args

  def __gt__(self, other):
    return self is not other and (self.__class__.__name__,)+self._args > (other.__class__.__name__,)+other._args

  def __le__(self, other):
    return self is other or (self.__class__.__name__,)+self._args < (other.__class__.__name__,)+other._args

  def __ge__(self, other):
    return self is other or (self.__class__.__name__,)+self._args > (other.__class__.__name__,)+other._args

  def __getstate__( self ):
    raise Exception( 'getstate should never be called' )

  def __setstate__( self, state ):
    raise Exception( 'setstate should never be called' )

  def __str__( self ):
    return '{}({})'.format( self.__class__.__name__, ','.join( str(arg) for arg in self._args ) )

  def edit(self, op):
    return self.__class__(*[op(arg) for arg in self._args])

class FileCache:
  'cache'

  def __init__( self, *args ):
    'constructor'

    import os, numpy, hashlib, pickle
    serial = pickle.dumps( args, -1 )
    self.myhash = hash( serial )
    hexhash = hashlib.md5(serial).hexdigest()
    cachedir = config.cachedir
    if not os.path.exists( cachedir ):
      os.makedirs( cachedir )
    path = os.path.join( cachedir, hexhash )
    if not os.path.isfile( path ) or config.recache:
      log.info( 'starting new cache:', hexhash )
      data = open( path, 'wb+' )
      data.write( serial )
      data.flush()
    else:
      log.info( 'continuing from cache:', hexhash )
      data = open( path, 'ab+' )
      data.seek(0)
      recovered_args = pickle.load( data )
      assert recovered_args == args, 'hash clash'
    self.data = data

  def __call__( self, func, *args, **kwargs ):
    'call'

    try:
      import cPickle as pickle
    except ImportError:
      import pickle
    name = func.__name__ + ''.join( ' {}'.format(arg) for arg in args ) + ''.join( ' {}={}'.format(*item) for item in kwargs.items() )
    pos = self.data.tell()
    try:
      data = pickle.load( self.data )
    except EOFError:
      data = func( *args, **kwargs)
      self.data.seek( pos )
      pickle.dump( data, self.data, -1 )
      self.data.flush()
      msg = 'written to'
    else:
      msg = 'loaded from'
    log.info( msg, 'cache:', name, '[{}b]'.format(self.data.tell()-pos) )
    return data

  def truncate( self ):
    log.info( 'truncating cache' )
    self.data.truncate()

  def __hash__( self ):
    return self.myhash

def replace(func):
  '''decorator for deep object replacement

  Generates a deep replacement method for Immutable objects based on a callable
  that is applied (recursively) on individual constructor arguments.

  Args
  ----
  func : callable which maps (obj, ...) onto replaced_obj

  Returns
  -------
  callable
      The method that searches the object to perform the replacements.
  '''

  @functools.wraps(func)
  def wrapped(target, *funcargs, **funckwargs):
    cache = {}
    def op(obj):
      try:
        replaced = cache[obj]
      except TypeError: # unhashable
        replaced = obj
      except KeyError:
        replaced = func(obj, *funcargs, **funckwargs)
        if replaced is None:
          replaced = obj.edit(op) if isinstance(obj, Immutable) else obj
        cache[obj] = replaced
      return replaced
    retval = op(target)
    del op
    return retval

  return wrapped

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
