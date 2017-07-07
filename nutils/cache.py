# -*- coding: utf8 -*-
#
# Module CACHE
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The cache module.
"""

from . import core, log
import os, sys, weakref, numpy, functools, inspect, builtins

_self = object()

def property(f):
  def property_getter(self, *, _name=f.__name__, _func=f, _temp=object()):
    try:
      dictvalue = self.__dict__[_name]
      assert dictvalue is not _temp, 'attribute requested during construction'
    except KeyError:
      self.__dict__[_name] = _temp # placeholder for detection of cyclic dependencies
      value = _func(self)
      self.__dict__[_name] = value if value is not self else _self
    else:
      value = dictvalue if dictvalue is not _self else self
    return value
  def property_setter(self, value, *, _name=f.__name__):
    assert _name not in self.__dict__, 'property can be set only once'
    self.__dict__[_name] = value if value is not self else _self
  assert not property_getter.__closure__ and not property_setter.__closure__
  return builtins.property(fget=property_getter, fset=property_setter)

def weakproperty( f ):
  def cache_property_wrapper( self, f=f ):
    value = self.__dict__.get( f.__name__ )
    if value:
      value = value()
    if not value:
      value = f( self )
      self.__dict__[f.__name__] = weakref.ref(value)
    return value
  assert not cache_property_wrapper.__closure__
  return builtins.property(cache_property_wrapper)

def argdict( f ):
  cache = {}
  @functools.wraps( f )
  def f_wrapped( *args ):
    key = _hashable( args )
    try:
      return cache[key]
    except KeyError:
      pass
    value = cache[key] = f( *args )
    return value
  return f_wrapped

class HashableBase( object ):

  pass

class HashableArray( HashableBase ):
  # FRAGILE: assumes contents are not going to be changed
  def __init__( self, array ):
    self.array = array
    self.quickdata = array.shape, array.dtype.kind, tuple( array.flat[::array.size//32+1] ) if array.size else None # required to prevent segfault
  def __hash__( self ):
    return hash( self.quickdata )
  def __eq__( self, other ):
    # check full array only if we really must
    return isinstance(other,HashableArray) and ( self.array is other.array
      or self.quickdata == other.quickdata and numpy.all( self.array == other.array ) )
  
class HashableList( tuple, HashableBase ):
  def __new__( cls, L ):
    return tuple.__new__( cls, map( _hashable, L ) )

class HashableDict( frozenset, HashableBase ):
  def __new__( cls, D ):
    return frozenset.__new__( cls, map( _hashable, D if isinstance( D, frozenset ) else dict( D ).items() ) )

class HashableAny( HashableBase ):
  def __init__( self, obj ):
    self.obj = obj
  def __hash__( self ):
    return hash( id(self.obj) )
  def __eq__( self, other ):
    return isinstance(other,HashableAny) and self.obj is other.obj

def _hashable( obj ):
  try:
    hash(obj)
  except:
    pass
  else:
    return obj
  return tuple( _hashable(o) for o in obj ) if isinstance( obj, tuple ) \
    else frozenset( _hashable(o) for o in obj ) if isinstance( obj, (set,frozenset) ) \
    else HashableArray( obj ) if isinstance( obj, numpy.ndarray ) \
    else HashableList( obj ) if isinstance( obj, list ) \
    else HashableDict( obj ) if isinstance( obj, dict ) \
    else HashableAny( obj )

def _position_args( func ):
  sig = inspect.signature( func )
  var = inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
  assert not any( parameter.kind in var for parameter in sig.parameters.values() ), 'var-arguments not allowed in {}'.format(func)
  return tuple( sig.parameters ), { parameter.name: parameter.default for parameter in sig.parameters.values() if parameter.default != sig.empty }

class Wrapper( object ):
  'function decorator that caches results by arguments'

  def __init__( self, func ):
    self.func = func
    self.cache = {}
    self.count = 0
    self.argnames, self.defaults = _position_args( func )

  def __call__( self, *args, **kwargs ):
    self.count += 1
    assert len(args) <= len(self.argnames), 'too many arguments for function {}'.format( self.func )
    for name in self.argnames[len(args):]:
      try:
        val = kwargs.pop(name)
      except KeyError:
        val = self.defaults[name]
      args += val,
    assert not kwargs, 'invalid arguments for function {}: {}'.format( self.func, ', '.join(kwargs) )
    key = tuple( _hashable(arg) for arg in args )
    value = self.cache.get( key )
    if value is None:
      value = self.func( *args )
      self.cache[ key ] = value
    return value

  @builtins.property
  def hits( self ):
    return self.count - len(self.cache)

class WrapperCache( object ):
  'maintains a cache for Wrapper instances'

  def __init__( self ):
    self.cache = {}

  def __getitem__( self, func ):
    try:
      return self.cache[func]
    except KeyError:
      pass
    wrapper = self.cache[func] = Wrapper( func )
    return wrapper

  @builtins.property
  def stats( self ):
    hits = count = 0
    for wrapper in self.cache.values():
      hits += wrapper.hits
      count += wrapper.count
    return 'not used' if not count \
      else 'effectivity %d%% (hit %d/%d calls over %d functions)' % ( 100*hits/count, hits, count, len(self.cache) )

class WrapperDummyCache( object ):
  'placeholder object'

  stats = 'caching disabled'

  def __getitem__( self, func ):
    return func

class CallDict( object ):
  'deprecated object'

  def __init__( self ):
    warnings.warn( 'CallDict will be removed in future, please use WrapperCache instead', DeprecationWarning, stacklevel=2 )
    self.wrappercache = WrapperCache()

  def __call__( self, func, *args, **kwargs ):
    return self.wrappercache[func]( *args, **kwargs )

  def summary( self ):
    return self.wrappercache.stats

class ImmutableMeta( type ):
  def __init__( cls, *args, **kwargs ):
    type.__init__( cls, *args, **kwargs )
    cls.argnames, cls.defaults = _position_args( cls.__init__ )
    cls.cache = weakref.WeakValueDictionary()
  def __call__( cls, *args, **kwargs ):
    assert len(args) <= len(cls.argnames), 'too many arguments for construction of {}'.format( cls )
    for name in cls.argnames[len(args)+1:]: # +1 to exclude 'self'
      try:
        val = kwargs.pop(name)
      except KeyError:
        val = cls.defaults[name]
      args += val,
    assert not kwargs, 'invalid arguments in construction of {}: {}'.format( cls, ', '.join(kwargs) )
    key = tuple( _hashable(arg) for arg in args )
    try:
      return cls.cache[key]
    except KeyError:
      pass
    self = type.__call__( cls, *args )
    self._args = args
    self._hash = hash(key)
    cls.cache[key] = self
    return self

class Immutable( object, metaclass=ImmutableMeta ):

  def __init__( self ):
    pass

  def __reduce__( self ):
    return self.__class__.__call__, self._args

  def __hash__( self ):
    return self._hash

  def __getstate__( self ):
    raise Exception( 'getstate should never be called' )

  def __setstate__( self, state ):
    raise Exception( 'setstate should never be called' )

  def __str__( self ):
    return '{}({})'.format( self.__class__.__name__, ','.join( str(arg) for arg in self._args ) )

class FileCache( object ):
  'cache'

  def __init__( self, *args ):
    'constructor'

    import os, numpy, hashlib, pickle
    serial = pickle.dumps( args, -1 )
    self.myhash = hash( serial )
    hexhash = hashlib.md5(serial).hexdigest()
    cachedir = core.getprop( 'cachedir', 'cache' )
    if not os.path.exists( cachedir ):
      os.makedirs( cachedir )
    path = os.path.join( cachedir, hexhash )
    if not os.path.isfile( path ) or core.getprop( 'recache', False ):
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
    name = func.__name__ + ''.join( ' %s' % arg for arg in args ) + ''.join( ' %s=%s' % item for item in kwargs.items() )
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
    log.info( msg, 'cache:', name, '[%db]' % (self.data.tell()-pos) )
    return data

  def truncate( self ):
    log.info( 'truncating cache' )
    self.data.truncate()

  def __hash__( self ):
    return self.myhash

class Tuple( object ):
  unknown = object()
  def __init__( self, items, getitem, start=0, stride=1 ):
    if isinstance( items, int ):
      assert items > 0
      items = numpy.array( [ self.unknown ] * items, dtype=object )
    self.__items = numpy.asarray( items, dtype=object )
    self.__getitem = getitem
    self.__start = start
    self.__stride = stride
  def __len__( self ):
    return len(self.__items)
  def __iter__( self ):
    for i, item in enumerate( self.__items ):
      if item is self.unknown:
        self.__items[i] = item = self.__getitem( self.__start + i * self.__stride )
      yield item
  def __getitem__( self, i ):
    if isinstance( i, slice ):
      items = self.__items[i]
      if self.unknown not in items:
        return tuple(items)
      start, stop, stride = i.indices( len(self) )
      return Tuple( items, self.__getitem, self.__start + start * self.__stride, stride * self.__stride )
    assert isinstance( i, int )
    item = self.__items[i]
    if item is self.unknown:
      self.__items[i] = item = self.__getitem( self.__start + i * self.__stride )
    return item

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
