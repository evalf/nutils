from . import prop, log, numpy
import os, sys, multiprocessing

Lock = multiprocessing.Lock
cpu_count = multiprocessing.cpu_count

def fork( func, nice=19 ):
  'fork and run (return value is lost)'

  if not hasattr( os, 'fork' ):
    log.warning( 'fork does not exist on this platform; running %s in serial' % func.__name__ )
    return func

  def wrapped( *args, **kwargs ):
    pid = os.fork()
    if pid:
      return pid
    os.nice( nice )
    try:
      func( *args, **kwargs )
    except KeyboardInterrupt:
      pass
    except:
      log.traceback()
    os._exit( 0 )
  return wrapped

def shzeros( shape, dtype=float ):
  'create zero-initialized array in shared memory'

  if isinstance( shape, int ):
    shape = shape,
  else:
    assert all( isinstance(sh,int) for sh in shape )
  size = numpy.product( shape ) if shape else 1
  typecode = { int: 'i', float: 'd' }[ dtype ]
  buf = multiprocessing.RawArray( typecode, size )
  return numpy.frombuffer( buf, dtype ).reshape( shape )

def pariter( iterable ):
  'iterate parallel'

  nprocs = getattr( prop, 'nprocs', 1 )
  if nprocs <= 1:
    for it in iterable:
      yield it
    return

  shared_iter = multiprocessing.RawValue( 'i' )
  iterable = iter( iterable )
  lock = Lock()

  iproc = 0
  while iproc < nprocs-1 and os.fork() == 0:
    iproc += 1

  oldcontext = log.context( 'proc %d' % ( iproc+1 ), depth=1 )

  status = 1
  try:
    with lock:
      iiter = shared_iter.value
      shared_iter.value = iiter + 1
    for n, it in enumerate( iterable ):
      if n < iiter:
        continue
      assert n == iiter
      yield it
      with lock:
        iiter = shared_iter.value
        shared_iter.value = iiter + 1
    status = 0
  finally:
    if status:
      log.error( 'an exception occurred' )
    if iproc < nprocs-1:
      child_pid, child_status = os.wait()
      if child_status:
        status = 1
    if iproc:
      os._exit( status )
    log.restore( oldcontext, depth=1 )

  if status:
    raise Exception, 'one or more processes failed'

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
