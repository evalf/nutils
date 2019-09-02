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
The log module provides print methods :func:`debug`, :func:`info`,
:func:`user`, :func:`warning`, and :func:`error`, in increasing order of
priority. Output is sent to stdout as well as to an html formatted log file if
so configured.

This is a transitional wrapper around the external treelog module.
"""

import builtins, itertools, treelog, contextlib, sys, functools
from treelog import set, add, disable, withcontext, \
  Log, TeeLog, FilterLog, NullLog, DataLog, RecordLog, StdoutLog, RichOutputLog, LoggingLog
from . import warnings

def _len(iterable):
  try:
    return len(iterable)
  except:
    return 0

class iter:
  def __init__(self, title, iterable, length=None):
    self._log = treelog.current
    self._iter = builtins.iter(iterable)
    self._title = title
    self._length = length or _len(iterable)
    self._index = 0
    text = '{} 0'.format(self._title)
    if self._length:
      text += ' (0%)'
    self._log.pushcontext(text)
    self.closed = False
  def __iter__(self):
    return self
  def __next__(self):
    if self.closed:
      raise StopIteration
    try:
      value = next(self._iter)
    except:
      self.close()
      raise
    self._index += 1
    text = '{} {}'.format(self._title, self._index)
    if self._length:
      text += ' ({:.0f}%)'.format(100 * self._index / self._length)
    self._log.popcontext()
    self._log.pushcontext(text)
    return value
  def close(self):
    if not self.closed:
      self._log.popcontext()
      self.closed = True
  def __enter__(self):
    return self
  def __exit__(self, *args):
    self.close()
  def __del__(self):
    if not self.closed:
      warnings.warn('unclosed iterator {!r}'.format(self._title), ResourceWarning)
      self.close()

def range(title, *args):
  return iter(title, builtins.range(*args))

def enumerate(title, iterable):
  return iter(title, builtins.enumerate(iterable), length=_len(iterable))

def zip(title, *iterables):
  return iter(title, builtins.zip(*iterables), length=min(map(_len, iterables)))

def count(title, start=0, step=1):
  return iter(title, itertools.count(start, step))

@contextlib.contextmanager
def open(filename, mode, *, level='user', exists=None):
  if exists is not None:
    warnings.deprecation('the "exists" argument is deprecated and will be ignored')
  levels = 'debug', 'info', 'user', 'warning', 'error'
  if level not in levels:
    raise Exception('the "level" argument should be on of {}'.format(', '.join(levels)))
  with treelog.open(filename, mode, level=levels.index(level), id=None) as f:
    f.devnull = not f
    yield f

@contextlib.contextmanager
def context(title, *args):
  log = treelog.current
  log.pushcontext(title.format(*args))
  try:
    yield
  finally:
    log.popcontext()

def _print(level, *args, sep=' '):
  '''Write message to log.

  Args
  ----
  *args : tuple of :class:`str`
      Values to be printed to the log.
  sep : :class:`str`
      String inserted between values, default a space.
  '''
  treelog.current.write(sep.join(map(str, args)), level)

def _file(level, name, mode, *, id=None):
  '''Open file in logger-controlled directory.

  Args
  ----
  filename : :class:`str`
  mode : :class:`str`
      Should be either ``'w'`` (text) or ``'wb'`` (binary data).
  id :
      Bytes identifier that can be used to decide a priori that a file has
      already been constructed. Default: None.
  '''
  return treelog.current.open(name, mode, level, id)

debug, info, user, warning, error = [functools.partial(_print, level) for level in builtins.range(5)]
debugfile, infofile, userfile, warningfile, errorfile = [functools.partial(_file, level) for level in builtins.range(5)]

del _print, _file

# vim:sw=2:sts=2:et
