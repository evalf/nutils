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

  shared_iter = multiprocessing.RawValue( 'i', nprocs )
  lock = Lock()

  for iproc in range( nprocs-1 ):
    child_pid = os.fork()
    if child_pid:
      break
  else:
    iproc = nprocs-1
    child_pid = None

  oldcontext = log.context( 'proc %d' % ( iproc+1 ), depth=1 )

  status = 1
  try:
    iiter = iproc
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
    try:
      if status:
        log.error( 'an exception occurred' )
      if child_pid is not None:
        check_child_pid, child_status = os.waitpid( child_pid, 0 )
        if check_child_pid != child_pid:
          log.error( 'pid failure! got %s, was waiting for %s' % (check_child_pid,child_pid) )
          status = 1
        elif child_status:
          status = 1
      log.restore( oldcontext, depth=1 )
    except:
      status = 1
    if iproc:
      os._exit( status )

  if status:
    raise Exception, 'one or more processes failed'

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
