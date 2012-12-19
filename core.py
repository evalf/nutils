import sys, weakref

def setprop( name, value ):
  'set stack-global property'

  key = '__property__' + name
  frame = sys._getframe(1)
  frame.f_locals[key] = value

def delprop( name, value ):
  'set stack-global property'

  key = '__property__' + name
  frame = sys._getframe(1)
  frame.f_locals.pop( key, None )

getprop_nodefault = object()
def getprop( name, default=getprop_nodefault ):
  'get stack-global property'

  key = '__property__' + name
  frame = sys._getframe(1)
  while frame:
    if key in frame.f_locals:
      return frame.f_locals[key]
    frame = frame.f_back
  assert default is not getprop_nodefault, 'property %r not found' % name
  return default

def weakcacheprop( func ):
  'weakly cached property'

  key = func.func_name
  def wrapped( self ):
    value = self.__dict__.get( key )
    value = value and value()
    if value is None:
      value = func( self )
      self.__dict__[ key ] = weakref.ref(value)
    return value

  return property( wrapped )

def cacheprop( func ):
  'cached property'

  key = func.func_name
  def wrapped( self ):
    value = self.__dict__.get( key )
    if value is None:
      value = func( self )
      self.__dict__[ key ] = value
    return value

  return property( wrapped )

def cachefunc( func ):
  'cached property'

  def wrapped( self, *args, **kwargs ):
    funcache = self.__dict__.setdefault( '_funcache', {} )

    unspecified = object()
    argcount = func.func_code.co_argcount - 1 # minus self
    args = list(args) + [unspecified] * ( argcount - len(args) ) if func.func_defaults is None \
      else list(args) + list(func.func_defaults[ len(args) + len(func.func_defaults) - argcount: ]) if len(args) + len(func.func_defaults) > argcount \
      else list(args) + [unspecified] * ( argcount - len(func.func_defaults) - len(args) ) + list(func.func_defaults)
    try:
      for kwarg, val in kwargs.items():
        args[ func.func_code.co_varnames.index(kwarg)-1 ] = val
    except ValueError:
      raise TypeError, '%s() got an unexpected keyword argument %r' % ( func.func_name, kwarg )
    args = tuple( args )
    if unspecified in args:
      raise TypeError, '%s() not all arguments were specified' % func.func_name
    key = (func.func_name,) + args
    value = funcache.get( key )
    if value is None:
      value = func( self, *args )
      funcache[ key ] = value
    return value

  return wrapped

def classcache( fun ):
  'wrapper to cache return values'

  cache = {}
  def wrapped_fun( cls, *args ):
    data = cache.get( args )
    if data is None:
      data = fun( cls, *args )
      cache[ args ] = data
    return data
  return wrapped_fun if fun.func_name == '__new__' \
    else classmethod( wrapped_fun )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
