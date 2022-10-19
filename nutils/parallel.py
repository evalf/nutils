"""
The parallel module provides tools aimed at parallel computing. At this point
all parallel solutions use the ``fork`` system call and are supported on limited
platforms, notably excluding Windows. On unsupported platforms parallel features
will disable and a warning is printed.
"""

from . import numeric, warnings, _util as util
import os
import multiprocessing
import mmap
import signal
import contextlib
import builtins
import numpy
import treelog


@util.set_current
@util.defaults_from_env
def maxprocs(nprocs: int = 1):
    if not isinstance(nprocs, int) or nprocs < 1:
        raise ValueError('nprocs requires a positive integer argument')
    return nprocs


@contextlib.contextmanager
def fork(nprocs=None):
    '''continue as ``nprocs`` parallel processes by forking ``nprocs-1`` times

    If ``nprocs`` exceeds the configured ``maxprocs`` than it will silently be
    capped. It is up to the user to prepare shared memory and/or locks for
    inter-process communication. As a safety measure nested forks are blocked by
    limiting nprocs to 1; all secondary forks will be silently ignored.
    '''

    if nprocs is None or nprocs > maxprocs.current:
        nprocs = maxprocs.current
    if nprocs <= 1:
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
            if not pid:  # pragma: no cover
                amchild = True
                signal.signal(signal.SIGINT, signal.SIG_IGN)  # disable sigint (ctrl+c) handler
                setter = treelog.set(treelog.NullLog())
                setter.__enter__()  # silence treelog
                # NOTE for treelog >= 2.0 we must hold a reference to setter until
                # os.exit_ to save the formerly active logger from being destructed
                break
            child_pids.append(pid)
        else:
            procid = 0
        with maxprocs(1):
            yield procid
    except BaseException as e:
        if amchild:  # pragma: no cover
            try:
                print('[parallel.fork] exception in child process:', e)
            finally:
                os._exit(1)  # communicate failure to main process
        for pid in child_pids:  # kill all child processes
            os.kill(pid, signal.SIGKILL)
        raise
    else:
        if amchild:  # pragma: no cover
            os._exit(0)  # communicate success to main process
        with treelog.context('waiting for child processes'):
            nfails = sum(not _wait(pid) for pid in child_pids)
        if nfails:  # failure in child process: raise exception
            raise Exception('fork failed in {} out of {} processes'.format(nfails, nprocs))
    finally:
        if amchild:  # pragma: no cover
            os._exit(1)  # failsafe


def shempty(shape, dtype=float):
    '''create uninitialized array in shared memory'''

    if numeric.isint(shape):
        shape = shape,
    else:
        assert all(numeric.isint(sh) for sh in shape)
    dtype = numpy.dtype(dtype)
    size = util.product(map(int, shape), int(dtype.itemsize))
    if size == 0 or maxprocs.current == 1:
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
        self._lock = multiprocessing.Lock()  # lock to avoid race conditions in incrementing index

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            iiter = self._index.value  # claim next value
            if iiter >= self._stop:
                raise StopIteration
            self._index.value = iiter + 1
        return iiter


@contextlib.contextmanager
def ctxrange(name, nitems):
    '''fork and yield shared range-like counter with percentage-style logging'''

    rng = range(nitems)  # shared range, must be created pre-fork
    with fork(nitems), treelog.iter.wrap(_pct(name, nitems), rng) as wrprng:
        yield wrprng


def _pct(name, n):
    '''helper function for ctxrange'''

    i = yield name + ' 0%'
    while True:
        i = yield name + ' {:.0f}%'.format(100*(i+1)/n)


def _wait(pid):
    '''wait for process to finish and return True upon success,
    False upon failure while logging the reason.'''
    pid_, status = os.waitpid(pid, 0)
    assert pid_ == pid
    if os.WIFEXITED(status):
        s = os.WEXITSTATUS(status)
        if not s:
            return True
        msg = 'exited with status {}'.format(s)
    elif os.WIFSIGNALED(status):
        s = os.WTERMSIG(status)
        msg = 'was killed with signal {} ({})'.format(s, signal.Signals(s).name)
    elif os.WIFSTOPPED(status):
        s = os.WSTOPSIG(status)
        msg = 'was stopped with signal {} ({})'.format(s, signal.Signals(s).name)
    else:
        msg = 'died of unnatural causes'
    treelog.error('process {} {}'.format(pid, msg))
    return False

# vim:sw=4:sts=4:et
