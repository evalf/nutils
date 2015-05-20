from nutils import log, debug, core, numpy
import sys, time

selection = list(sys.argv[1:])
try:
  selection.remove( '--tbexplore' )
except ValueError:
  tbexplore = False
else:
  tbexplore = True
packages = []

def _run():
  __richoutput__ = True
  __log__ = log.clone()
  __results__ = []
  try:
    for package in packages:
      package()
  except KeyboardInterrupt:
    log.info( 'aborted.' )
    sys.exit( -1 )
  except:
    exc, frames = debug.exc_info()
    log.stack( 'error in unit testing framework: {}'.format(exc), frames )
    log.info( 'crashed.' )
    sys.exit( -2 )
  passed, failed, error, packagefail = results_by_status = [], [], [], []
  for name, status in __results__:
    results_by_status[status].append( name )
  log.info( '{}/{} tests passed.'.format( len(passed), len(__results__) ) )
  if failed:
    log.info( '* failures ({}):'.format(len(failed)), ', '.join( failed ) )
  if error:
    log.info( '* errors ({}):'.format(len(error)), ', '.join( error ) )
  if packagefail:
    log.info( '* package failures ({}):'.format(len(packagefail)), ', '.join( packagefail ) )
  sys.exit( len(__results__) - len(passed) )

def _package( f, __scope__ ):
  def wrapper():
    __log__ = log.clone()
    for item in __scope__.split('.'):
      __log__.append( item )
    results0 = core.getprop('results')
    __results__ = []
    t0 = time.time()
    try:
      f()
    except:
      exc, frames = debug.exc_info()
      log.stack( 'error: {}'.format(exc), frames )
      results0.append(( __scope__, 3 ))
      if tbexplore:
        debug.explore( repr(exc), frames, '''Test package
          {!r} failed. The traceback explorer allows you to examine the failure
          state. Closing the explorer will resume testing with the next
          package.'''.format(__scope__) )
    else:
      dt = time.time() - t0
      npassed = sum( status == 0 for name, status in __results__ )
      log.info( 'passed {}/{} tests in {:.2f} seconds'.format( npassed, len(__results__), dt ) )
      results0.extend( __results__ )
  wrapper.__module__ = f.__module__
  wrapper.__name__ = f.__name__
  return wrapper

def register( arg0, *args, **kwargs ):
  if not callable( arg0 ):
    return lambda f: register( _withattrs( lambda: f( *args, **kwargs ), __name__=f.__name__+':'+str(arg0), __module__=f.__module__, __wraps__=f ) )
  assert not args and not kwargs
  pkgname, scope = ( arg0.__module__ + '.' + arg0.__name__ ).split( '.', 1 )
  assert pkgname == __name__
  if not selection or any( scope == arg or arg.startswith(scope+'.') or scope.startswith(arg+'.') for arg in selection ):
    packages.append( _package(arg0,scope) )
  return getattr( arg0, '__wraps__', arg0 )

def _unittest( f ):
  infostream = log.getstream( 'info' )
  infostream.write( 'testing..' )
  output = log.CaptureStreamFactory()
  __log__ = log.Log( log.ProgressStreamFactory( infostream, output ) )
  try:
    f()
  except AssertionError:
    status = 1
    exc, frames = debug.exc_info()
    infostream.write( ' FAILED: {}\n'.format( str(exc).strip() ) )
  except:
    status = 2
    exc, frames = debug.exc_info()
    infostream.write( ' ERROR: {}\n'.format( str(exc).strip() ) )
  else:
    status = 0
    exc = frames = None
    infostream.write( ' OK\n' )
  infostream.close()
  return status, output.captured, exc, frames

def unittest( arg ):
  if not callable( arg ):
    return lambda f: unittest( _withattrs( f, __name__=f.__name__+':'+str(arg) ) )
  name = core.getprop('scope') + '.' + arg.__name__
  if selection and not any( name == arg or name.startswith( arg+'.' ) for arg in selection ):
    return
  __log__ = log.clone()
  __log__.append( arg.__name__ )
  status, captured, exc, frames = _unittest( arg )
  core.getprop('results').append(( name, status ))
  if status:
    if captured:
      log.error( 'captured output:\n-----\n{}-----'.format(captured) )
    if tbexplore:
      debug.explore( repr(exc), frames, '''Unit test {!r} failed. The traceback
        explorer allows you to examine the failure state. Closing the explorer
        will resume testing.'''.format( name ) )


## HELPER FUNCTIONS

def _withattrs( f, **attrs ):
  wrapped = lambda *args, **kwargs: f( *args, **kwargs )
  wrapped.__name__ = f.__name__
  wrapped.__module__ = f.__module__
  for attr, value in attrs.items():
    setattr( wrapped, attr, value )
  return wrapped
