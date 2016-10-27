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

from . import core, log, rational
import os, sys, weakref, numpy, functools, inspect


_property = property

def property( f ):
  name = f.__name__
  def property_getter( self, name=name, f=f, tmp=object() ):
    try:
      value = self.__dict__[name]
      assert value is not tmp, 'attribute requested during construction'
    except KeyError:
      self.__dict__[name] = tmp
      value = f( self )
      self.__dict__[name] = value
    return value
  def property_setter( self, value, name=name ):
    assert name not in self.__dict__, 'property can be set only once'
    self.__dict__[name] = value
  assert not property_getter.__closure__ and not property_setter.__closure__
  return _property( fget=property_getter, fset=property_setter )

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
  return _property(cache_property_wrapper)

def argdict( f ):
  cache = {}
  @functools.wraps( f )
  def f_wrapped( *args ):
    key = _hashable( args )
    try:
      value = cache[key]
    except KeyError:
      value = f( *args )
      cache[key] = value
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
    else ( obj.denom, HashableArray(obj.numer) ) if isinstance( obj, rational.Rational ) \
    else HashableArray( obj ) if isinstance( obj, numpy.ndarray ) \
    else HashableList( obj ) if isinstance( obj, list ) \
    else HashableDict( obj ) if isinstance( obj, dict ) \
    else HashableAny( obj )

def _position_args( func, *args, **kwargs ):
  sig = inspect.signature( func )
  invalid_kinds = inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD
  params = tuple( sig.parameters.values() )
  assert not any( param.kind in invalid_kinds for param in params )
  bound = sig.bind( *args, **kwargs )
  positional = tuple( bound.arguments.get( param.name, param.default ) for param in params )
  assert not any( arg is inspect.Parameter.empty for arg in positional )
  return positional

class Wrapper( object ):
  'function decorator that caches results by arguments'

  def __init__( self, func ):
    self.func = func
    self.cache = {}
    self.count = 0

  def __call__( self, *args, **kwargs ):
    self.count += 1
    key = _hashable( _position_args( self.func, *args, **kwargs ) )
    value = self.cache.get( key )
    if value is None:
      value = self.func( *args, **kwargs )
      self.cache[ key ] = value
    return value

  @_property
  def hits( self ):
    return self.count - len(self.cache)

class WrapperCache( object ):
  'maintains a cache for Wrapper instances'

  def __init__( self ):
    self.cache = {}

  def __getitem__( self, func ):
    try:
      wrapper = self.cache[func]
    except KeyError:
      wrapper = Wrapper( func )
      self.cache[func] = wrapper
    return wrapper

  @_property
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
    cls.cache = weakref.WeakValueDictionary()
  def __call__( cls, *args, **kwargs ):
    _args = _position_args( cls.__init__, None, *args, **kwargs )[1:]
    key = _hashable( _args )
    try:
      self = cls.cache[key]
    except KeyError:
      self = type.__call__( cls, *_args )
      self._args = _args
      cls.cache[key] = self
    return self

class Immutable( object, metaclass=ImmutableMeta ):

  def __reduce__( self ):
    return self.__class__.__call__, self._args

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
