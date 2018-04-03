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

  The following configuration properties are used in Nutils.

  .. attribute:: nprocs

     Controls the number of processes to use for computing integrals
     (:meth:`nutils.topology.Topology.integrate`) and a few other expensive and
     parallelizable functions.

     Defaults to ``1``.

  .. attribute:: verbose

     Controls the level of verbosity of loggers.  Log entries with a level
     higher than :attr:`verbose` are omitted.  The levels are ``1``: error,
     ``2``: warning, ``3``: user, ``4``: info and ``5``: debug.

     Defaults to ``4``: info.

  .. attribute:: outdir

     Defines the output directory for the HTML log
     (:class:`nutils.log.HtmlLog`) and plots.  Relative
     paths are relative with respect to the current working directory (see
     :func:`os.getcwd`).

     .. Note::

        If :attr:`outdir` is empty (the default), :func:`nutils.cli.run` and
        :func:`nutils.cli.choose` change this value.  See :attr:`outrootdir`.

     Defaults to ``''``.

  .. attribute:: outdirfd

     Used internally.  Do not touch.

     Defaults to ``None``.

  .. attribute:: dot

     If ``True``, :meth:`nutils.topology.Topology.elem_eval` and
     :meth:`nutils.topology.Topology.integrate` log a visualization of the
     function tree that is being evaluated or integrated.

     Defaults to ``False``.

  .. attribute:: imagetype

     A comma-separated list of (file extensions of) image types.
     :class:`nutils.plot.PyPlot` generates figures for all listed types.

     Defaults to ``'png'``.

  The following properties are only used in :func:`nutils.cli.run` and
  :func:`nutils.cli.choose`.

  .. attribute:: cache

     Controls on-disk caching.  If ``True``, functions decorated with
     :func:`nutils.cache.function` and subclasses of
     :class:`nutils.cache.Recursion` are automatically cached.  Examples are
     :meth:`nutils.topology.Topology.integrate` and
     :class:`nutils.solver.thetamethod`.

     The location of the on-disk cache is defined by :attr:`cachedir`.

     Defaults to ``False``.

  .. attribute:: cachedir

     Defines the location of the on-disk cache (see :attr:`cache`).  Relative
     paths are relative with respect to the current working directory (see
     :func:`os.getcwd`).

     .. Note::

        If :attr:`outdir` is empty (the default), :func:`nutils.cli.run` and
        :func:`nutils.cli.choose` change this value.  See :attr:`outrootdir`.

     Defaults to ``'cache'``.

  .. attribute:: outrootdir

     Defines the root directory for the HTML log and plots.  This property is
     used to override :attr:`outdir`, if empty, to
     ``'<outrootdir>/<scriptname>/<YY/MM/DD/HH-MM-SS>/'``.  Likewise,
     if :attr:`outdir` is empty, the :attr:`cachedir` property is redefined to
     ``'<outrootdir>/<scriptname>/cache/'``.

     Defaults to ``'~/public_html'``

  .. attribute:: symlink

     If not empty, but :attr:`outdir` is empty, the symlinks
     ``'<outrootdir>/<symlink>'`` and ``'<outrootdir>/<scriptname>/<symlink>'``
     will be created, both pointing to
     ``'<outrootdir>/<scriptname>/<YY/MM/DD/HH-MM-SS>/log.html'``.

     Defaults to ``''``.

  .. attribute:: richoutput

     Controls whether or not the console logger should output rich text
     (:class:`nutils.log.RichOutputLog`) or plain text
     (:class:`nutils.log.StdoutLog`).

     Defaults to ``True`` if ``sys.stdout`` is attached to a terminal (i.e.
     ``sys.stdout.isatty()`` returns true), otherwise ``False``.

  .. attribute:: htmloutput

     If ``True`` the HTML logger (:class:`nutils.log.HtmlLog`) is enabled and
     written to ``'log.html'`` inside the directory defined by :attr:`outdir`.
     With the default configuration the log will be generated in
     ``'~/public_html/<scriptname>/<YY/MM/DD/HH-MM-SS>/log.html'``

     Defaults to ``True``.

  .. attribute:: pdb

     If ``True`` the debugger will be invoked when an exception reaches
     :func:`nutils.cli.run` or :func:`nutils.cli.choose`.

     Defaults to ``False``.

  .. attribute:: matrix

     A comma-separated list of matrix backends.  The first one available is
     activated.  The names — the case is irrelevant – correspond to subclasses
     of :class:`nutils.matrix.Backend`.  Use
     ``nutils.matrix.Backend.__subclasses__()`` to list the available backends.

     Defauls to ``'mkl,scipy,numpy'``.
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
  dot = False,
  cachedir = 'cache',
  matrix = 'mkl,scipy,numpy',
  cache = False,
)
