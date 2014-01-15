import inspect, weakref, os, numpy, warnings

ALLCACHE = []

def _cache( func, cache ):
  argnames, varargsname, keywordsname, defaults = inspect.getargspec( func )
  allargnames = list(argnames)
  if varargsname:
    allargnames.append( '*' + varargsname )
  if keywordsname:
    allargnames.append( '**' + keywordsname )
  fname = '%s(%s) in %s:%d' % ( func.__name__, ','.join(allargnames), os.path.relpath(func.func_code.co_filename), func.func_code.co_firstlineno )
  ALLCACHE.append(( fname, cache ))
  def wrapped( *args, **kwargs ):
    if kwargs:
      for iarg in range( len(args), len(argnames) ):
        try:
          val = kwargs.pop( argnames[iarg] )
        except KeyError:
          val = defaults[ iarg-len(argnames) ]
        args += val,
      assert not kwargs, 'leftover keyword arguments'
    elif len(args) < len(argnames):
      args += defaults[ len(args)-len(argnames): ]
    try:
      value = cache[ args ]
    except TypeError:
      warnings.warn( 'unhashable item; skipping cache for %s' % fname )
      value = func( *args )
    except KeyError:
      value = func( *args )
      cache[ args ] = value
    return value
  return wrapped

def cache_info():
  from . import log
  items = []
  for fname, d in ALLCACHE:
    if not d:
      continue
    types = {}
    for v in d.values():
      types.setdefault( type(v), [] ).append( isinstance(v,numpy.ndarray) and v.nbytes )
    count = ', '.join( '%dx %s%s' % ( len(N), T.__name__, '' if N[0] is False else ' (%db)' % sum(N) ) for T, N in types.items() )
    cachename = 'weakly cached' if isinstance( d, weakref.WeakValueDictionary ) else 'cached'
    items.append(( len(d), '%s %s %s' % ( fname, cachename, count ) ))
  items.sort( reverse=True )
  log.info( 'cache usage:\n' +  '\n'.join( '  ' + s for i, s in items ) )

def cache_flush():
  for fname, d in ALLCACHE:
    d.clear()

def cache( func ):
  return _cache( func, {} )

def weakcache( func ):
  return _cache( func, weakref.WeakValueDictionary() )

def savelast( func ):
  saved = [ None, None ]
  def wrapped( *args ):
    if args != saved[0]:
      saved[:] = args, func( *args )
    return saved[1]
  return wrapped


#def weakcacheprop( func ):
#  'weakly cached property'
#
#  key = func.func_name
#  def wrapped( self ):
#    value = self.__dict__.get( key )
#    value = value and value()
#    if value is None:
#      value = func( self )
#      self.__dict__[ key ] = weakref.ref(value)
#    return value
#
#  return property( wrapped )
#
#def cacheprop( func ):
#  'cached property'
#
#  key = func.func_name
#  def wrapped( self ):
#    value = self.__dict__.get( key )
#    if value is None:
#      value = func( self )
#      self.__dict__[ key ] = value
#    return value
#
#  return property( wrapped )
#
#def cachefunc( func ):
#  'cached property'
#
#  def wrapped( self, *args, **kwargs ):
#    try:
#      hash( args + tuple(kwargs.values()) )
#    except TypeError: # unhashable arguments; skip cache
#      return func( self, *args, **kwargs )
#    funcache = self.__dict__.setdefault( '_funcache', {} )
#    argcount = func.func_code.co_argcount - (len(args)+1) # remaining after args
#    if not argcount:
#      assert not kwargs
#    else:
#      unspecified = object()
#      extra = [unspecified] * argcount
#      for kwarg, val in kwargs.items():
#        try:
#          i = func.func_code.co_varnames.index(kwarg) - (len(args)+1)
#        except ValueError:
#          raise TypeError, '%s() got an unexpected keyword argument %r' % ( func.func_name, kwarg )
#        assert i >= 0 and extra[i] is unspecified, 'repeated argument %d in %s' % ( kwarg, func.func_name )
#        extra[i] = val
#      defaults = func.func_defaults or ()
#      assert len(defaults) <= argcount
#      for i in range( argcount ):
#        if argcount-i > len(defaults):
#          assert extra[i] is not unspecified
#        elif extra[i] is unspecified:
#          extra[i] = defaults[len(defaults)-(argcount-i)]
#      args += tuple(extra)
#    key = (func.func_name,) + args
#    value = funcache.get( key )
#    if value is None:
#      value = func( self, *args )
#      funcache[ key ] = value
#    return value
#
#  return wrapped
#
#def classcache( fun ):
#  'wrapper to cache return values'
#
#  cache = {}
#  def wrapped_fun( cls, *args ):
#    data = cache.get( args )
#    if data is None:
#      data = fun( cls, *args )
#      cache[ args ] = data
#    return data
#  return wrapped_fun if fun.func_name == '__new__' \
#    else classmethod( wrapped_fun )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
