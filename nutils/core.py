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

from __future__ import print_function, division
import sys, functools, os

globalproperties = {
  'nprocs': 1,
  'outrootdir': '~/public_html',
  'outdir': '.',
  'verbose': 6,
  'richoutput': False,
  'tbexplore': False,
  'imagetype': 'png',
  'symlink': False,
  'recache': False,
  'dot': False,
  'profile': False,
}

for nutilsrc in ['~/.config/nutils/config', '~/.nutilsrc']:
  nutilsrc = os.path.expanduser( nutilsrc )
  if not os.path.isfile( nutilsrc ):
    continue
  try:
    exec( open(nutilsrc).read(), {}, globalproperties )
  except:
    exc_value, frames = sys.exc_info()
    exc_str = '\n'.join( [ repr(exc_value) ] + [ str(f) for f in frames ] )
    print( 'Skipping .nutilsrc: {}'.format(exc_str) )
  break

_nodefault = object()
def getprop( name, default=_nodefault, frame=None ):
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

  key = '__%s__' % name
  if frame is None:
    frame = sys._getframe(1)
  while frame:
    if key in frame.f_locals:
      return frame.f_locals[key]
    frame = frame.f_back
  if name in globalproperties:
    return globalproperties[name]
  if default is _nodefault:
    raise NameError( 'property %r is not defined' % name )
  return default

def index( items ):
  """Index of the first nonzero item.

  Args:
      items: Any iterable object

  Returns:
      The index of the first item for which bool(item) returns True.
  """

  for i, item in enumerate(items):
    if item:
      return i
  raise ValueError

def single_or_multiple( f ):
  """
  Method wrapper, converts first positional argument to tuple: tuples/lists
  are passed on as tuples, other objects are turned into tuple singleton.
  Return values should match the length of the argument list, and are unpacked
  if the original argument was not a tuple/list.

  >>> class Test:
  >>>   @single_or_multiple
  >>>   def square( args ):
  >>>     return [ v**2 for v in args ]
  >>>
  >>> T = Test()
  >>> a = T.square( 2 ) # 4
  >>> a, b = T.square( [2,3] ) # (4,9)

  Args:
      f: Method that expects a tuple as first positional argument, and that
      returns a list/tuple of the same length.

  Returns:
      Wrapped method.
  """

  @functools.wraps( f )
  def wrapped( self, arg0, *args, **kwargs ):
    ismultiple = isinstance( arg0, (list,tuple) )
    arg0mod = tuple(arg0) if ismultiple else (arg0,)
    retvals = f( self, arg0mod, *args, **kwargs )
    if not ismultiple:
      retvals, = retvals
    return retvals
  return wrapped


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
