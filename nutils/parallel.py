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
The parallel module provides tools aimed at parallel computing. At this point
all parallel solutions use the ``fork`` system call and are supported on limited
platforms, notably excluding Windows. On unsupported platforms parallel features
will disable and a warning is printed.
"""

from . import log, numpy, numeric
import os, sys, multiprocessing, tempfile, mmap, traceback, signal, collections.abc

procid = None # current process id, None for unforked

def shempty(shape, dtype=float):
  '''create uninitialized array in shared memory'''

  if numeric.isint(shape):
    shape = shape,
  else:
    assert all(numeric.isint(sh) for sh in shape)
  dtype = numpy.dtype(dtype)
  size = (numpy.product(shape) if shape else 1) * dtype.itemsize
  if size == 0:
    return numpy.empty(shape, dtype)
  # `mmap(-1,...)` will allocate *anonymous* memory.  Although linux' man page
  # mmap(2) states that anonymous memory is initialized to zero, we can't rely
  # on this to be true for all platforms (see [SO-mmap]).  [SO-mmap]:
  # https://stackoverflow.com/a/17896084
  return numpy.frombuffer(mmap.mmap(-1, size), dtype).reshape(shape)

def shzeros(shape, dtype=float):
  '''create zero-initialized array in shared memory'''

  array = shempty(shape, dtype=dtype)
  array.fill(0)
  return array

def pariter(iterable, nprocs):
  '''iterate in parallel

  Fork into ``nprocs`` subprocesses, then yield items from iterable such that
  all processes receive a nonoverlapping subset of the total. It is up to the
  user to prepare shared memory and/or locks for inter-process communication.
  The following creates a data vector containing the first four quadratics::

     data = shzeros(shape=[4], dtype=int)
     for i in pariter(range(4), 2):
       data[i] = i**2
     data

  As a safety measure nested pariters are blocked by setting the global
  ``procid`` variable; all secundary pariters will be treated like normal
  serial iterators.

  Parameters
  ----------
  iterable : :class:`collections.abc.Iterable`
      The collection of items to be distributed over processors
  nprocs : :class:`int`
      Maximum number of processers to use

  Yields
  ------
      Items from iterable, distributed over at most nprocs processors.
  '''

  global procid

  if procid is not None:
    log.warning('ignoring pariter for already forked process')
    yield from iterable
    return

  if isinstance(iterable, collections.abc.Sized):
    nprocs = min(nprocs, len(iterable))

  if nprocs <= 1:
    yield from iterable
    return

  if not hasattr(os, 'fork'):
    raise NotImplementedError('pariter requires os.fork, which is unavailable on this platform')

  shared_iter = multiprocessing.RawValue('i', nprocs) # shared integer pointing at first unyielded item
  lock = multiprocessing.Lock() # lock to avoid race conditions in incrementing shared_iter
  children = [] # list of forked processes, non-empty only in primary process

  try:

    for procid in range(1, nprocs):
      child_pid = os.fork()
      if not child_pid:
        signal.signal(signal.SIGINT, signal.SIG_IGN) # disable sigint (ctrl+c) handler
        break
      children.append(child_pid)
    else:
      procid = 0

    iiter = procid # first index is 0 .. nprocs-1, with shared_iter at nprocs
    for n, it in enumerate(iterable):
      if n < iiter: # fast forward to iiter
        continue
      assert n == iiter
      yield it
      with lock:
        iiter = shared_iter.value # claim next value
        shared_iter.value = iiter + 1

  except:

    fail = 1
    if procid == 0:
      raise # reraise in main process

    # in child processes print traceback then exit
    excval = sys.exc_info()[1]
    if isinstance(excval, GeneratorExit):
      log.error('generator failed with unknown exception')
    elif not isinstance(excval, KeyboardInterrupt):
      log.error(traceback.format_exc())

  else:

    fail = 0

  finally:

    if procid != 0: # before anything else can fail:
      os._exit(fail) # cumminicate exit status to main process

    procid = None # unset global variable
    totalfail = fail
    while children:
      child_pid, child_status = os.wait()
      children.remove(child_pid)
      if child_status:
        totalfail += 1
    if fail: # failure in main process: exception has been reraised
      log.error('pariter failed in {} out of {} processes; reraising exception for main process'.format(totalfail, nprocs))
    elif totalfail: # failure in child process: raise exception
      raise Exception('pariter failed in {} out of {} processes'.format(totalfail, nprocs))

def parmap(func, iterable, nprocs, shape=(), dtype=float):
  '''parallel equivalent to builtin map function

  Produces an array of ``func(item)`` values for all items in ``iterable``.
  Because of shared memory restrictions ``func`` must yield numpy arrays of
  predetermined shape and type.

  Parameters
  ----------
  func : :any:`callable`
      Takes item from iterable, returns numpy array of ``shape`` and ``dtype``
  iterable : :class:`collections.abc.Iterable`
      Collection of items
  nprocs : :class:`int`
      Maximum number of processers to use
  shape : :class:`tuple`
      Return shape of ``func``, defaults to scalar
  dtype : :class:`tuple`
      Return dtype of ``func``, defaults to float

  Returns
  -------
      Array of shape ``len(iterable),+shape`` and dtype ``dtype``
  '''


  n = len(iterable)
  out = shzeros((n,)+shape, dtype=dtype)
  for i, item in pariter(enumerate(iterable), nprocs=min(n,nprocs)):
    out[i] = func(item)
  return out

# vim:sw=2:sts=2:et
