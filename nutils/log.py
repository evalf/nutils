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

This is a transitional wrapper around the external treelog module that will be
removed in version 6.
"""

import builtins, itertools, treelog, contextlib, sys, functools, inspect, treelog, distutils
from treelog import set, add, disable, withcontext, \
  Log, TeeLog, FilterLog, NullLog, DataLog, RecordLog, StdoutLog, RichOutputLog, LoggingLog, HtmlLog
from . import warnings

if distutils.version.StrictVersion(treelog.version) < distutils.version.StrictVersion('1.0b5'):
  warnings.deprecation('treelog 1.0b5 or higher is required, please update using pip install -U treelog')
else:
  sys.modules['nutils.log'] = treelog

def _len(iterable):
  try:
    return len(iterable)
  except:
    return 0

class iter:
  def __init__(self, title, iterable, length=None):
    warnings.deprecation('log.iter is deprecated; use log.iter.percentage instead')
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

  class wrap:

    def __init__(self, titles, iterable):
      self._titles = builtins.iter(titles)
      self._iterable = builtins.iter(iterable)
      self._log = None
      self._warn = False

    def __enter__(self):
      if self._log is not None:
        raise Exception('iter.wrap is not reentrant')
      self._log = treelog.current
      self._log.pushcontext(next(self._titles))
      return builtins.iter(self)

    def __iter__(self):
      if self._log is not None:
        cansend = inspect.isgenerator(self._titles)
        for value in self._iterable:
          self._log.popcontext()
          self._log.pushcontext(self._titles.send(value) if cansend else next(self._titles))
          yield value
      else:
        with self:
          self._warn = True
          yield from self

    def __exit__(self, exctype, excvalue, tb):
      if self._log is None:
        raise Exception('iter.wrap has not yet been entered')
      if self._warn and exctype is GeneratorExit:
        warnings.warn('unclosed iter.wrap', ResourceWarning)
      self._log.popcontext()
      self._log = False

  def plain(title, *args):
    titles = map((iter._escape(title) + ' {}').format, itertools.count())
    return iter.wrap(titles, builtins.zip(*args) if len(args) > 1 else args[0])

  def fraction(title, *args, length=None):
    if length is None:
      length = min(len(arg) for arg in args)
    titles = map((iter._escape(title) + ' {}/' + str(length)).format, itertools.count())
    return iter.wrap(titles, builtins.zip(*args) if len(args) > 1 else args[0])

  def percentage(title, *args, length=None):
    if length is None:
      length = min(len(arg) for arg in args)
    if length:
      titles = map((iter._escape(title) + ' {:.0f}%').format, itertools.count(step=100/length))
    else:
      titles = title + ' 100%',
    return iter.wrap(titles, builtins.zip(*args) if len(args) > 1 else args[0])

  def _escape(s):
    return s.replace('{', '{{').replace('}', '}}')

def range(title, *args):
  warnings.deprecation('log.range is deprecated; use log.iter.percentage instead')
  return iter(title, builtins.range(*args))

def enumerate(title, iterable):
  warnings.deprecation('log.enumerate is deprecated; use log.iter.percentage instead')
  return iter(title, builtins.enumerate(iterable), length=_len(iterable))

def zip(title, *iterables):
  warnings.deprecation('log.zip is deprecated; use log.iter.percentage instead')
  return iter(title, builtins.zip(*iterables), length=min(map(_len, iterables)))

def count(title, start=0, step=1):
  warnings.deprecation('log.count is deprecated; use log.iter.percentage instead')
  return iter(title, itertools.count(start, step))

@contextlib.contextmanager
def open(filename, mode, *, level='user'):
  warnings.deprecation('log.open is deprecated; use log.userfile instead')
  levels = 'debug', 'info', 'user', 'warning', 'error'
  if level not in levels:
    raise Exception('the "level" argument should be on of {}'.format(', '.join(levels)))
  with treelog.open(filename, mode, level=levels.index(level), id=None) as f:
    f.devnull = not f
    yield f

@contextlib.contextmanager
def context(title, *initargs, **initkwargs):
  log = treelog.current
  if initargs or initkwargs:
    log.pushcontext(title.format(*initargs, **initkwargs))
    def reformat(*args, **kwargs):
      log.popcontext()
      log.pushcontext(title.format(*args, **kwargs))
  else:
    log.pushcontext(title)
    reformat = None
  try:
    yield reformat
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

@contextlib.contextmanager
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
  with treelog.current.open(name, mode, level, id) as f, context(name):
    yield f

debug, info, user, warning, error = [functools.partial(_print, level) for level in builtins.range(5)]
debugfile, infofile, userfile, warningfile, errorfile = [functools.partial(_file, level) for level in builtins.range(5)]

del _print, _file

# vim:sw=2:sts=2:et
