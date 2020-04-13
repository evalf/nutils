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

from . import numeric, warnings, util
import os, multiprocessing, mmap, signal, contextlib, builtins, numpy, treelog

_maxprocs = 1

@contextlib.contextmanager
@util.positional_only
def maxprocs(new: int):
  '''limit number of processes for fork.'''

  if not isinstance(new, int) or new < 1:
    raise ValueError('nprocs requires a positive integer argument')
  global _maxprocs
  old = _maxprocs
  _maxprocs = new
  try:
    yield
  finally:
    _maxprocs = old

@contextlib.contextmanager
def fork(nprocs=None):
  '''continue as ``nprocs`` parallel processes by forking ``nprocs-1`` times

  If ``nprocs`` exceeds the configured ``maxprocs`` than it will silently be
  capped. It is up to the user to prepare shared memory and/or locks for
  inter-process communication. As a safety measure nested forks are blocked by
  limiting nprocs to 1; all secondary forks will be silently ignored.
  '''

  if nprocs is None or nprocs > _maxprocs:
    nprocs = _maxprocs
  if nprocs == 1:
    yield 0
    return
  if not hasattr(os, 'fork'):
    warnings.warn('fork is unavailable on this platform')
    yield 0
    return
  amchild = False
  try:
    child_pids = []
    for procid in builtins.range(1, nprocs):
      pid = os.fork()
      if not pid: # pragma: no cover
        amchild = True
        signal.signal(signal.SIGINT, signal.SIG_IGN) # disable sigint (ctrl+c) handler
        treelog.current = treelog.NullLog() # silence treelog
        break
      child_pids.append(pid)
    else:
      procid = 0
    with maxprocs(1):
      yield procid
  except BaseException as e:
    if amchild: # pragma: no cover
      print('[parallel.fork] exception in child process:', e)
      os._exit(1) # communicate failure to main process
    for pid in child_pids: # kill all child processes
      os.kill(pid, signal.SIGKILL)
    raise
  else:
    if amchild: # pragma: no cover
      os._exit(0) # communicate success to main process
    with treelog.context('waiting for child processes'):
      nfails = sum(os.waitpid(pid, 0)[1] != 0 for pid in child_pids)
    if nfails: # failure in child process: raise exception
      raise Exception('fork failed in {} out of {} processes'.format(nfails, nprocs))
  finally:
    if amchild: # pragma: no cover
      os._exit(1) # failsafe

def shempty(shape, dtype=float):
  '''create uninitialized array in shared memory'''

  if numeric.isint(shape):
    shape = shape,
  else:
    assert all(numeric.isint(sh) for sh in shape)
  dtype = numpy.dtype(dtype)
  size = (numpy.product(shape) if shape else 1) * dtype.itemsize
  if size == 0 or _maxprocs == 1:
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

@contextlib.contextmanager
def ctxrange(name, nitems):
  '''fork and yield shared range-like counter with percentage-style logging'''

  rng = range(nitems) # shared range, must be created pre-fork
  with fork(nitems), treelog.iter.wrap(_pct(name, nitems), rng) as wrprng:
    yield wrprng

def _pct(name, n):
  '''helper function for ctxrange'''

  i = yield name + ' 0%'
  while True:
    i = yield name + ' {:.0f}%'.format(100*(i+1)/n)

# vim:sw=2:sts=2:et
