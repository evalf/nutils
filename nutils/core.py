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

import sys

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
