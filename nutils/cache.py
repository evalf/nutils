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
import os, weakref


_property = property
def property( f ):
  def cache_property_wrapper( self, f=f ):
    try:
      value = self.__dict__[f.func_name]
    except KeyError:
      value = f( self )
      self.__dict__[f.func_name] = value
    return value
  assert not cache_property_wrapper.__closure__
  return _property(cache_property_wrapper)


def _keyfromargs( func, args, kwargs, offset=0 ):
  if getattr( func, '__self__' ): # bound instancemethod
    offset += 1
  code = func.func_code
  names = code.co_varnames[offset+len(args):code.co_argcount]
  for name in names:
    try:
      val = kwargs.pop(name)
    except KeyError:
      index = names.index(name)-len(names)
      try:
        val = func.func_defaults[index]
      except Exception as e:
        raise TypeError, '%s missing mandatory argument %r' % ( func.__name__, name )
    args += val,
  assert not kwargs, '%s got invalid arguments: %s' % ( func.__name__, ', '.join(kwargs) )
  mask = 0
  key = []
  for arg in args:
    try:
      hash(arg)
    except:
      mask |= 1 << len(key)
      arg = id(arg)
    key.append( arg )
  return mask, tuple(key)


class CallDict( dict ):
  'very simple cache object'

  hit = 0

  def __call__( self, func, *args, **kwargs ):
    '''cache(func,*args,**kwargs):
    Execute func(*args,**kwargs) and cache the result.'''

    key = func, _keyfromargs( func, args, kwargs )
    value = self.get( key )
    if value is None:
      value = func( *args, **kwargs )
      self[ key ] = value
    else:
      self.hit += 1

    return value

  def summary( self ):
    return 'not used' if not self \
      else 'effectivity %d%% (%d hits, %d misses)' % ( (100*self.hit)/(self.hit+len(self)), self.hit, len(self) )


class Immutable( object ):
  'weakly cache object instances based on init args'

  __slots__ = '__weakref__',

  class __metaclass__( type ):
    def __init__( cls, *args, **kwargs ):
      type.__init__( cls, *args, **kwargs )
      cls.cache = weakref.WeakValueDictionary()
    def __call__( cls, *args, **kwargs ):
      key = _keyfromargs( cls.__init__, args, kwargs, 1 )
      try:
        self = cls.cache[key]
      except KeyError:
        self = type.__call__( cls, *args, **kwargs )
        cls.cache[key] = self
      return self


class FileCache( object ):
  'cache'

  def __init__( self, *args ):
    'constructor'

    import os, numpy
    self.myhash = hash( args )
    hexhash = hex( self.myhash )[2:]
    cachedir = core.getprop( 'cachedir', 'cache' )
    if not os.path.exists( cachedir ):
      os.makedirs( cachedir )
    path = os.path.join( cachedir, hexhash )
    allhash = numpy.array( [ hash(arg) for arg in args ], 'uint' )
    if not os.path.isfile( path ) or core.getprop( 'recache', False ):
      log.info( 'starting new cache:', hexhash )
      data = open( path, 'wb+' )
      allhash.tofile( data )
    else:
      log.info( 'continuing from cache:', hexhash )
      data = open( path, 'ab+' )
      checkhash = numpy.fromfile( data, 'uint', len(allhash) )
      assert all( checkhash == allhash ), 'hash clash'
    self.data = data

  def __call__( self, func, *args, **kwargs ):
    'call'

    import cPickle
    name = func.__name__ + ''.join( ' %s' % arg for arg in args ) + ''.join( ' %s=%s' % item for item in kwargs.iteritems() )
    pos = self.data.tell()
    try:
      data = cPickle.load( self.data )
    except EOFError:
      data = func( *args, **kwargs)
      self.data.seek( pos )
      cPickle.dump( data, self.data, -1 )
      msg = 'written to'
    else:
      msg = 'loaded from'
    log.info( msg, 'cache:', name, '[%db]' % (self.data.tell()-pos) )
    return data

  def __hash__( self ):
    return self.myhash


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
