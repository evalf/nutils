from multiprocessing import Lock, cpu_count

nprocs = 1#cpu_count()

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

def pariter( iterable, verbose=False ):
  'fork and iterate, handing equal-sized chunks to all processors'

  if nprocs == 1:
    if verbose:
      print 'pariter: iterating in sequential mode (nprocs=1)'
    for i in iterable:
      yield i
    return

  if verbose:
    print 'pariter: iterating in parallel mode (nprocs=%d)' % nprocs

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
      print 'an error occured:', e
      _exit( 1 )
    else:
      _exit( 0 )

  while pids:
    pid, status = wait()
    assert status == 0, 'subprocess #%d failed'
    pids.remove( pid )
    if verbose:
      print 'pariter: process #%d finished, %d pending' % ( pid, len(pids) )

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
