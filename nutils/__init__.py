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

import sys, numpy, contextlib
from distutils.version import LooseVersion

assert sys.version_info >= (3, 3)
assert LooseVersion(numpy.version.version) >= LooseVersion('1.8'), 'nutils requires numpy 1.8 or higher, got %s' % numpy.version.version

version = '3beta'

class _Config(type):

  def __setattr__(self, k, v):
    raise AttributeError('readonly attribute: {}'.format(k))

  def __delattr__(self, k):
    raise AttributeError('readonly attribute: {}'.format(k))

  @contextlib.contextmanager
  def __call__(*args, **data):
    if len(args) != 1:
      raise TypeError('function takes 1 positional argument but {} were given'.format(len(args)))
    self, = args
    EMPTY = object()
    old = {}
    try:
      for k, new_v in data.items():
        old_v = getattr(self, k, EMPTY)
        if old_v is not new_v:
          old[k] = old_v
          type.__setattr__(self, k, new_v)
      yield
    finally:
      for k, old_v in old.items():
        if old_v is EMPTY:
          type.__delattr__(self, k)
        else:
          type.__setattr__(self, k, old_v)

  def __str__(self):
    return 'configuration: {}'.format(', '.join('{}={!r}'.format(k, v) for k, v in sorted(self.__dict__.items()) if not k.startswith('_')))

class config(metaclass=_Config):
  '''
  This object holds the Nutils global configuration, stored as (immutable)
  attributes.  To inspect the current configuration, use :func:`print` or
  :func:`vars` on this object.  The configuration can be changed temporarily by
  calling this object with the new settings passed as keyword arguments and
  entering the returned context.  The old settings are restored as soon as the
  context is exited.  Example:

  >>> config.verbose
  4
  >>> with config(verbose=2, nprocs=4):
  ...   # The configuration has been updated.
  ...   config.verbose
  2
  >>> # Exiting the context reverts the changes:
  >>> config.verbose
  4

  .. Note::
     The default entry point for Nutils scripts :func:`nutils.cli.run` (and
     :func:`nutils.cli.choose`) will read user configuration from disk.

  .. Important::
     The configuration is not thread-safe: changing the configuration inside a
     thread changes the process wide configuration.
  '''

  nprocs = 1
  outrootdir = '~/public_html'
  outdir = ''
  outdirfd = None
  verbose = 4
  richoutput = False
  htmloutput = True
  pdb = False
  imagetype = 'png'
  symlink = ''
  recache = False
  dot = False
  profile = False
  selfcheck = False
  cachedir = 'cache'

_ = numpy.newaxis
__all__ = ['_', 'numpy', 'core', 'numeric', 'element', 'function', 'expression',
  'mesh', 'plot', 'topology', 'util', 'matrix', 'parallel', 'log',
  'cache', 'transform', 'solver', 'cli', 'warnings', 'config']
