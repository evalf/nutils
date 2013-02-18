from . import prop
from multiprocessing import Lock, cpu_count
import os

def fork( func, nice=19 ):
  'fork and run (return value is lost)'

  def wrapped( *args, **kwargs ):
    pid = os.fork()
    if pid:
      return pid
    try:
      os.nice( nice )
      func( *args, **kwargs )
    finally:
      os._exit( 0 )
  return wrapped

def shzeros( shape, dtype=float ):
  'create zero-initialized array in shared memory'

  from multiprocessing import RawArray
  from numpy import frombuffer, product, array

  try:
    shape = map(int,shape)
  except TypeError:
    shape = [int(shape)]
  size = product( shape ) if shape else 1
  typecode = {
    int: 'i',
    float: 'd' }
  buf = RawArray( typecode[dtype], size )
  return frombuffer( buf, dtype ).reshape( shape )

def oldpariter( iterable, verbose=False ):
  'fork and iterate, handing equal-sized chunks to all processors'

  nprocs = getattr( prop, 'nprocs', 1 )
  if nprocs == 1:
    log.debug( 'pariter: iterating in sequential mode (nprocs=1)' )
    for i in iterable:
      yield i
    return

  log.debug( 'pariter: iterating in parallel mode (nprocs=%d)' % nprocs )

  from os import fork, wait, _exit

  iterable = tuple( iterable )
  pids = set()
  for iproc in range( nprocs ):
    pid = fork()
    if pid:
      pids.add( pid )
      continue
    try:
      for i in range( iproc, len(iterable), nprocs ):
        yield iterable[ i ]
    except Exception, e:
      log.error( 'an error occured: %s' % e )
      _exit( 1 )
    else:
      _exit( 0 )

  while pids:
    pid, status = wait()
    assert status == 0, 'subprocess #%d failed'
    pids.remove( pid )
    log.debug( 'pariter: process #%d finished, %d pending' % ( pid, len(pids) ) )

def pariter( iterable ):
  'fork and iterate, handing equal-sized chunks to all processors'

  nprocs = getattr( prop, 'nprocs', 1 )
  if nprocs == 1:
    log.debug( 'pariter: iterating in sequential mode (nprocs=1)' )
    for item in iterable:
      yield item
    return

  log.debug( 'pariter: iterating in parallel mode (nprocs=%d)' % nprocs )

  from os import fork, wait, _exit

  pids = set()

  for item in iterable:
    pid = fork()
    if not pid: # child
      try:
        yield item
      except Exception, e:
        log.error( 'an error occured: %s' % e )
        _exit( 1 )
      _exit( 0 )

    pids.add( pid )
    if len(pids) >= nprocs:
      pid, status = wait()
      assert status == 0, 'subprocess #%d failed'
      pids.remove( pid )

  while pids:
    pid, status = wait()
    assert status == 0, 'subprocess #%d failed'
    pids.remove( pid )

def example():
  'simple example demonstrating a parallel loop'

  print 'parallel example'
  lock = Lock()
  A = shzeros( 4 )
  n = 0
  for i in pariter( [1,2,3,4], verbose=True ):
    with lock:
      A[n] += i
    n += 1
  print A

if __name__ == '__main__':
  example()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
