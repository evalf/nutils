from . import prop, log, numpy, debug
import os, sys, multiprocessing, thread

Lock = multiprocessing.Lock
cpu_count = multiprocessing.cpu_count

class Fork( object ):
  'nested fork context, unwinds at exit'

  def __init__( self, nprocs ):
    'constructor'

    self.nprocs = nprocs

  def __enter__( self ):
    'fork and return iproc'

    for self.iproc in range( self.nprocs-1 ):
      self.child_pid = os.fork()
      if self.child_pid:
        break
    else:
      self.child_pid = None
      self.iproc = self.nprocs-1
    self.logger = log.context( 'proc %d' % ( self.iproc+1 ), depth=2 )
    return self.iproc

  def __exit__( self, exctype, excvalue, tb ):
    'kill all processes but first one'

    status = 0
    try:
      if exctype == KeyboardInterrupt:
        status = 1
      elif exctype == GeneratorExit:
        if self.iproc:
          log.stack( 'generator failed with unknown exception', debug.callstack( depth=2 ) )
        status = 1
      elif exctype:
        if self.iproc:
          log.stack( repr(excvalue), debug.exception() )
        status = 1
      if self.child_pid:
        child_pid, child_status = os.waitpid( self.child_pid, 0 )
        if child_pid != self.child_pid:
          log.error( 'pid failure! got %s, was waiting for %s' % (child_pid,self.child_pid) )
          status = 1
        elif child_status:
          status = 1
      self.logger.disable()
    except: # should not happen.. but just to be sure
      status = 1
    if self.iproc:
      os._exit( status )
    if not exctype:
      assert status == 0, 'one or more subprocesses failed'

class AlternativeFork( object ):
  'single master, multiple slave fork context, unwinds at exit'

  def __init__( self, nprocs ):
    'constructor'

    self.nprocs = nprocs
    self.children = None

  def __enter__( self ):
    'fork and return iproc'

    children = []
    for self.iproc in range( 1, self.nprocs ):
      child_pid = os.fork()
      if not child_pid:
        break
      children.append( child_pid )
    else:
      self.children = children
      self.iproc = 0
    self.logger = log.context( 'proc %d' % ( self.iproc+1 ), depth=2 )
    return self.iproc

  def __exit__( self, exctype, excvalue, tb ):
    'kill all processes but first one'

    status = 0
    try:
      if exctype:
        log.stack( repr(excvalue), debug.exception() )
        status = 1
      while self.children:
        child_pid, child_status = os.wait()
        self.children.remove( child_pid )
        if child_status:
          status = 1
      self.logger.disable()
    except: # should not happen.. but just to be sure
      status = 1
    if self.iproc:
      os._exit( status )
    if not exctype:
      assert status == 0, 'one or more subprocesses failed'

def waitpid_noerr( pid ):
  try:
    os.waitpid( pid, 0 )
  except:
    pass

def fork( func, nice=19 ):
  'fork and run (return value is lost)'

  return func # disable fork

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

  # return numpy.zeros( shape, dtype=dtype ) # TODO: toggle to numpy for debugging
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

  nprocs = getattr( prop, 'nprocs', 1 )
  return iterable if nprocs <= 1 else _pariter( iterable, nprocs )

def _pariter( iterable, nprocs ):
  'iterate parallel, helper generator'

  shared_iter = multiprocessing.RawValue( 'i', nprocs )
  lock = Lock()
  with Fork( nprocs ) as iproc:
    iiter = iproc
    for n, it in enumerate( iterable ):
      if n < iiter:
        continue
      assert n == iiter
      yield it
      with lock:
        iiter = shared_iter.value
        shared_iter.value = iiter + 1

def parmap( func, iterable, shape=(), dtype=float ):
  n = len(iterable)
  out = shzeros( (n,)+shape, dtype=dtype )
  for i, item in pariter( enumerate(iterable) ):
    out[i] = func( item )
  return out

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
