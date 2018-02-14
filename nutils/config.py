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

import types, contextlib, sys

def load_rcfile(path):
  settings = {}
  try:
    with open(path) as rc:
      exec(rc.read(), {}, settings)
  except Exception as e:
    raise Exception('error loading config from {}'.format(path)) from e
  return settings

class Config(types.ModuleType):
  '''
  This module holds the Nutils global configuration, stored as (immutable)
  attributes.  To inspect the current configuration, use :func:`print` or
  :func:`vars` on this module.  The configuration can be changed temporarily by
  calling this module with the new settings passed as keyword arguments and
  entering the returned context.  The old settings are restored as soon as the
  context is exited.  Example:

  >>> from nutils import config
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

  def __init__(*args, **data):
    self, name = args
    super(Config, self).__init__(name, self.__doc__)
    self.__dict__.update(data)

  def __setattr__(self, k, v):
    raise AttributeError('readonly attribute: {}'.format(k))

  def __delattr__(self, k):
    raise AttributeError('readonly attribute: {}'.format(k))

  @contextlib.contextmanager
  def __call__(*args, **data):
    if len(args) < 1:
      raise TypeError('__call__ takes at least 1 positional argument but none were given')
    self, *configs = args
    configs.append(data)
    old = self.__dict__.copy()
    try:
      for config in configs:
        self.__dict__.update(config if isinstance(config, dict) else load_rcfile(config))
      yield
    finally:
      self.__dict__.clear()
      self.__dict__.update(old)

  def __str__(self):
    return 'configuration: {}'.format(', '.join('{}={!r}'.format(k, v) for k, v in sorted(self.__dict__.items()) if not k.startswith('_')))

sys.modules[__name__] = Config(
  __name__,
  nprocs = 1,
  outrootdir = '~/public_html',
  outdir = '',
  outdirfd = None,
  verbose = 4,
  richoutput = False,
  htmloutput = True,
  pdb = False,
  imagetype = 'png',
  symlink = '',
  recache = False,
  dot = False,
  profile = False,
  selfcheck = False,
  cachedir = 'cache',
  matrix = 'scipy,numpy',
)
