from multiprocessing import Lock, cpu_count

def shzeros( shape, dtype=float ):
  'create zero-initialized array in shared memory'

  from multiprocessing import RawArray
  from numpy import frombuffer, product

  size = product( shape )
  typecode = {
    int: 'i',
    float: 'd' }
  buf = RawArray( typecode[dtype], size )
  return frombuffer( buf, dtype ).reshape( shape )

def pariter( iterable, nprocs=None, verbose=False ):
  'fork and iterate, handing equal-sized chunks to all processors'

  from os import fork, wait, _exit

  if nprocs is None:
    nprocs = cpu_count()
  iterable = tuple( iterable )
  pids = set()
  for iproc in range( nprocs ):
    pid = fork()
    if pid:
      pids.add( pid )
      continue
    for i in range( iproc, len(iterable), nprocs ):
      yield iterable[ i ]
    _exit( 0 )

  while pids:
    pid, status = wait()
    pids.remove( pid )
    print 'process #%d finished, %d pending' % ( pid, len(pids) )

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
