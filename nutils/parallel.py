from . import util, log, debug
import os, sys, multiprocessing, thread, numpy

Lock = multiprocessing.Lock
cpu_count = multiprocessing.cpu_count

def waitpid_noerr( pid ):
  try:
    os.waitpid( pid, 0 )
  except:
    pass

def fork( func, nice=19 ):
  'fork and run (return value is lost)'

  if not hasattr( os, 'fork' ):
    log.warning( 'fork does not exist on this platform; running %s in serial' % func.__name__ )
    return func

  def wrapped( *args, **kwargs ):
    pid = os.fork()
    if pid:
      thread.start_new_thread( waitpid_noerr, (pid,) ) # kill the zombies
      # see: http://stackoverflow.com/a/13331632/445031
      # this didn't work: http://stackoverflow.com/a/6718735/445031
      return pid
    try:
      os.nice( nice )
      __nprocs__ = 1
      func( *args, **kwargs )
    except KeyboardInterrupt:
      pass
    except:
      log.stack( repr(sys.exc_value), debug.exception() )
    finally:
      os._exit( 0 )

  return wrapped

def shzeros( shape, dtype=float ):
  'create zero-initialized array in shared memory'

  if isinstance( shape, int ):
    shape = shape,
  else:
    assert all( isinstance(sh,int) for sh in shape )
  size = numpy.product( shape ) if shape else 1
  if dtype == float:
    typecode = 'd'
  elif dtype == int:
    typecode = 'i'
  else:
    raise Exception, 'invalid dtype: %r' % dtype
  buf = multiprocessing.RawArray( typecode, size )
  return numpy.frombuffer( buf, dtype ).reshape( shape )

def pariter( iterable ):
  'iterate parallel'

  nprocs = util.prop( 'nprocs', 1 )
  return iterable if nprocs <= 1 else _pariter( iterable, nprocs )

def _pariter( iterable, nprocs ):
  'iterate parallel, helper generator'

  # shared memory objects
  shared_iter = multiprocessing.RawValue( 'i', nprocs )
  lock = Lock()

  try:

    for iproc in range( nprocs-1 ):
      child_pid = os.fork()
      if child_pid:
        break
    else:
      child_pid = None
      iproc = nprocs-1

    iiter = iproc
    for n, it in enumerate( iterable ):
      if n < iiter:
        continue
      assert n == iiter
      yield it
      with lock:
        iiter = shared_iter.value
        shared_iter.value = iiter + 1

    if child_pid:
      pid, status = os.waitpid( child_pid, 0 )
      assert status == 0, 'child exited with nonzero status'
      assert pid == child_pid, 'pid failure; got %s, was waiting for %s' % (pid,child_pid)

  except:
    status = 1
    if iproc:
      etype, evalue, tb = sys.exc_info()
      if itype == KeyboardInterrupt:
        pass
      elif etype == GeneratorExit:
        log.stack( 'generator failed with unknown exception', debug.callstack( depth=2 ) )
      elif etype == AssertionError:
        log.error( 'error:', evalue )
      else:
        log.stack( repr(evalue), debug.exception() )
    else:
      raise

  else:
    status = 0

  finally:
    if iproc:
      os._exit( status )

def parmap( func, iterable, shape=(), dtype=float ):
  n = len(iterable)
  out = shzeros( (n,)+shape, dtype=dtype )
  for i, item in pariter( enumerate(iterable) ):
    out[i] = func( item )
  return out

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
