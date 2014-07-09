# -*- coding: utf8 -*-
#
# Module CORE
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The core module provides a collection of low level constructs that have no
dependencies on other nutils modules. Primarily for internal use.
"""

import inspect, weakref, os, numpy, warnings, collections, sys


ALLCACHE = []


class TruncDict( dict ):

  def __init__( self, maxlen ):
    self.__touched = collections.deque( maxlen=maxlen )
      # items in order that they were touched, most recent first

  @property
  def saturated( self ):
    return len( self.__touched ) == self.__touched.maxlen

  def __getitem__( self, item ):
    value = dict.__getitem__( self, item )
    # success, so we know touched has at least 1 item
    if self.__touched[0] != item:
      self.__touched.remove( item )
      self.__touched.appendleft( item )
    return value

  def __setitem__( self, item, value ):
    dict.__setitem__( self, item, value )
    if self.saturated:
      warnings.warn( 'truncdict reached maxlen, dropping oldest item.', stacklevel=3 )
      del self[ self.__touched[-1] ]
    self.__touched.appendleft( item )
    assert len(self.__touched) == len(self)


def _cache( func, cache ):
  argnames, varargsname, keywordsname, defaults = inspect.getargspec( func )
  allargnames = list(argnames)
  if varargsname:
    allargnames.append( '*' + varargsname )
  if keywordsname:
    allargnames.append( '**' + keywordsname )
  fname = 'In %s:%d %s(%s)' % ( os.path.relpath(func.func_code.co_filename), func.func_code.co_firstlineno, func.__name__, ','.join(allargnames) )
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
      warnings.warn( 'unhashable item; skipping cache\n  %s' % fname, stacklevel=2 )
      value = func( *args )
    except KeyError:
      value = func( *args )
      cache[ args ] = value
    return value
  return wrapped


def cache_info( brief=True ):
  items = []
  for fname, d in ALLCACHE:
    if not d or brief and ( not isinstance(d,TruncDict) or not d.saturated ):
      continue
    types = {}
    for v in d.values():
      types.setdefault( type(v), [] ).append( isinstance(v,numpy.ndarray) and v.nbytes )
    count = ', '.join( '%dx %s%s' % ( len(N), T.__name__, '' if N[0] is False else ' (%db)' % sum(N) ) for T, N in types.items() )
    cachename = 'weakly cached' if isinstance( d, weakref.WeakValueDictionary ) else 'cached'
    items.append(( len(d), '%s %s %s' % ( fname, cachename, count ) ))
  items.sort( reverse=True )
  return [ s for i, s in items ]


def cache_flush():
  for fname, d in ALLCACHE:
    d.clear()


def cache( func ):
  return _cache( func, TruncDict(1000) )


def weakcache( func ):
  return _cache( func, weakref.WeakValueDictionary() )


def savelast( func ):
  saved = [ None, None ]
  def wrapped( *args ):
    if args != saved[0]:
      saved[:] = args, func( *args )
    return saved[1]
  return wrapped


_nodefault = object()
def getprop( name, default=_nodefault ):
  """Access a semi-global property.

  The use of global variables is discouraged, as changes can go unnoticed and
  lead to abscure failure. The getprop mechanism makes local variables accesible
  (read only) from nested scopes, but not from the encompassing scope.

  >>> def f():
  >>>   print getprop('myval')
  >>> 
  >>> def main():
  >>>   __myval__ = 2
  >>>   f()

  Args:
      name (str): Property name, corresponds to __name__ local variable.
      default: Optional default value.

  Returns:
      The object corresponding to the first __name__ encountered in a higher
      scope. If none found, return default. If no default specified, raise
      NameError.
  """

  frame = sys._getframe(1)
  key = '__%s__' % name
  while frame:
    if key in frame.f_locals:
      return frame.f_locals[key]
    frame = frame.f_back
  if default is _nodefault:
    raise NameError, 'property %r is not defined' % name
  return default


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
