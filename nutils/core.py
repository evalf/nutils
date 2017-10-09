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
The core module provides a collection of low level constructs that have no
dependencies on other nutils modules. Primarily for internal use.
"""

import sys, functools, os
from . import config

def open_in_outdir( file, *args, **kwargs ):
  '''open a file relative to the ``outdirfd`` or ``outdir`` property

  Wrapper around :func:`open` that opens a file relative to either the
  ``outdirfd`` property (if supported, see :func:`os.supports_dir_fd`) or
  ``outdir``.  Takes the same arguments as :func:`open`.
  '''

  assert 'opener' not in kwargs
  if config.outdirfd is not None:
    kwargs['opener'] = functools.partial(os.open, dir_fd=config.outdirfd)
  elif config.outdir:
    file = os.path.join(os.path.expanduser(config.outdir), file)
  return open( file, *args, **kwargs )

def listoutdir():
  '''list files in ``outdirfd`` or ``outdir`` property'''

  if config.outdirfd is not None:
    return os.listdir(config.outdirfd)
  elif config.outdir:
    return os.listdir(os.path.expanduser(config.outdir))
  else:
    return os.listdir()


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
