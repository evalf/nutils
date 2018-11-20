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

from . import numeric, warnings
import os, multiprocessing, mmap, signal, contextlib, builtins, numpy, treelog as log

procid = None # current process id, None for unforked

@contextlib.contextmanager
def fork(nprocs):
  '''continue as ``nprocs`` parallel processes by forking ``nprocs-1`` times

  It is up to the user to prepare shared memory and/or locks for inter-process
  communication. As a safety measure nested forks are blocked by setting the
  global ``procid`` variable; all secondary forks will be silently ignored.
  '''

  global procid
  if nprocs == 1 or procid is not None:
    yield 0
    return
  if not hasattr(os, 'fork'):
    log.warning('fork is unavailable on this platform')
    yield 0
    return
  child_pids = []
  try:
    fail = 1
    for procid in builtins.range(1, nprocs):
      pid = os.fork()
      if not pid:
        signal.signal(signal.SIGINT, signal.SIG_IGN) # disable sigint (ctrl+c) handler
        log.current = log.NullLog()
        break
      child_pids.append(pid)
    else:
      procid = 0
    yield procid
    fail = 0
  finally:
    if procid: # before anything else can fail:
      os._exit(fail) # communicate exit status to main process
    procid = None # unset global variable
    nfails = fail + sum(os.waitpid(pid, 0)[1] != 0 for pid in child_pids)
    if fail: # failure in main process: exception has been reraised
      log.error('fork failed in {} out of {} processes; reraising exception for main process'.format(nfails, nprocs))
    elif nfails: # failure in child process: raise exception
      raise Exception('fork failed in {} out of {} processes'.format(nfails, nprocs))

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

class range:
  '''a shared range-like iterable that yields every index exactly once'''

  def __init__(self, stop):
    self._stop = stop
    self._index = multiprocessing.RawValue('i', 0)
    self._lock = multiprocessing.Lock() # lock to avoid race conditions in incrementing index
  def __iter__(self):
    return self
  def __next__(self):
    with self._lock:
      iiter = self._index.value # claim next value
      if iiter >= self._stop:
        raise StopIteration
      self._index.value = iiter + 1
    return iiter

def pariter(items, nprocs):
  '''iterate in parallel

  Fork into ``nprocs`` subprocesses, then yield items from iterable such that
  all processes receive a nonoverlapping subset of the total.

  NOTE: Pariter is deprecated because a child proces may not be ended if it
  forcably breaks out of the loop. Instead a :class:`fork` context should be
  used in combination with a shared :class:`range` interator.

  Parameters
  ----------
  iterable :
      The collection of items to be distributed over processors
  nprocs : :class:`int`
      Maximum number of processers to use
 
  Yields
  ------
      Items from iterable, distributed over at most nprocs processors.
  '''

  warnings.deprecation('pariter is deprecated, use fork, range instead')
  if not hasattr(items, '__getitem__'):
    items = tuple(items)
  indices = range(len(items))
  with fork(nprocs):
    for index in indices:
      yield items[index]

# vim:sw=2:sts=2:et
