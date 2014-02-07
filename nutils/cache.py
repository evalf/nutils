import weakref

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

class CallDict( dict ):
  'very simple cache object'

  hit = 0

  def __call__( self, *key ):
    'cache(func,*args): execute func(args) and cache result'
    value = self.get( key )
    if value is None:
      value = key[0]( *key[1:] )
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
  __cache = weakref.WeakValueDictionary()

  def __new__( *args, **kwargs ):
    init = args[0].__init__.im_func
    names = init.func_code.co_varnames[len(args):init.func_code.co_argcount]
    for name in names:
      try:
        val = kwargs.pop(name)
      except KeyError:
        index = names.index(name)-len(names)
        try:
          val = init.func_defaults[ names.index(name)-len(names) ]
        except Exception as e:
          raise TypeError, '%s.__init__ missing mandatory argument %r' % ( args[0].__name__, name )
      args += val,
    assert not kwargs
    try:
      self = Immutable.__cache[args]
    except KeyError:
      self = object.__new__( args[0] )
      Immutable.__cache[args] = self
    return self

class FileCache( object ):
  'cache'

  def __init__( self, *args ):
    'constructor'

    import hashlib
    strhash = ','.join( str(arg) for arg in args )
    md5hash = hashlib.md5( strhash ).hexdigest()
    log.info( 'using cache:', md5hash )
    cachedir = getattr( prop, 'cachedir', 'cache' )
    if not os.path.exists( cachedir ):
      os.makedirs( cachedir )
    path = os.path.join( cachedir, md5hash )
    self.data = file( path, 'ab+' if not getattr( prop, 'recache', False ) else 'wb+' )

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

