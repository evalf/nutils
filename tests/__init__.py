# -*- coding: utf8 -*-

from nutils import log, debug, core
import sys, time, collections, functools


## INTERNAL VARIABLES

PACKAGES = collections.OrderedDict()
OK, FAILED, ERROR, PKGERROR = range(4)


## INTERNAL METHODS

def _runtests( pkg, whitelist ):
  __log__ = log._getlog()
  __results__ = {}
  if isinstance( pkg, dict ):
    for key in pkg:
      if whitelist and key != whitelist[0]:
        continue
      __log__.push( key )
      try:
        __results__[key] = _runtests( pkg[key], whitelist[1:] )
      finally:
        __log__.pop()
  else:
    t0 = time.time()
    try:
      pkg()
    except KeyboardInterrupt:
      raise
    except:
      exc, frames = debug.exc_info()
      log.stack( 'error: {}'.format(exc), frames )
      if core.getprop( 'tbexplore', False ):
        debug.explore( repr(exc), frames, '''Test package
          failed. The traceback explorer allows you to examine the failure
          state. Closing the explorer will resume testing with the next
          package.''' )
      __results__ = PKGERROR
    else:
      dt = time.time() - t0
      npassed = sum( status == OK for name, status in __results__.items() )
      __log__.write( 'info', 'passed {}/{} tests in {:.2f} seconds'.format( npassed, len(__results__), dt ) )
  return __results__

def _withattrs( f, **attrs ):
  wrapped = lambda *args, **kwargs: f( *args, **kwargs )
  wrapped.__name__ = f.__name__
  wrapped.__module__ = f.__module__
  for attr, value in attrs.items():
    setattr( wrapped, attr, value )
  return wrapped

def _summarize( pkg, name=() ):
  if not isinstance( pkg, dict ):
    return { pkg: ['.'.join(name)] }
  summary = {}
  for key, val in pkg.items():
    for status, tests in _summarize( val, name+(key,) ).items():
      summary.setdefault( status, [] ).extend( tests )
  return summary


## EXPOSED MODULE METHODS

def runtests():

  args = sys.argv[1:] # command line arguments

  if 'coverage' in sys.argv[0]:
    # coverage passes the complete commandline to `sys.argv`
    # find '-m tests' and keep the tail
    m = args.index( '-m' )
    assert args[m+1] == __name__
    args = args[m+2:]

  __tbexplore__ = '--tbexplore' in args
  if __tbexplore__:
    args.remove( '--tbexplore' )
  
  if args:
    assert len(args) == 1
    whitelist = args[0].split( '.' )
  else:
    whitelist = []

  __richoutput__ = True
  __selfcheck__ = True
  __log__ = log._mklog()
  try:
    results = _runtests( PACKAGES, whitelist )
  except KeyboardInterrupt:
    log.info( 'aborted.' )
    sys.exit( -1 )
  except:
    exc, frames = debug.exc_info()
    log.stack( 'error in unit testing framework: {}'.format(exc), frames )
    log.info( 'crashed.' )
    sys.exit( -2 )

  summary = _summarize(results)
  ntests = sum( len(tests) for tests in summary.values() )
  passed = summary.pop( OK, [] )
  failed = summary.pop( FAILED, [] )
  error = summary.pop( ERROR, [] )
  pkgerror = summary.pop( PKGERROR, [] )

  log.info( '{}/{} tests passed.'.format( len(passed), ntests ) )
  if failed:
    log.info( '* failures ({}):'.format(len(failed)), ', '.join( failed ) )
  if error:
    log.info( '* errors ({}):'.format(len(error)), ', '.join( error ) )
  if pkgerror:
    log.info( '* package failures ({}):'.format(len(pkgerror)), ', '.join( pkgerror ) )
  if summary:
    log.info( '* invalid status ({}) - this should not happen!'.format(len(summary)) )

  sys.exit( ntests - len(passed) )

def register( f, *args, **kwargs ):
  if not callable( f ):
    return functools.partial( register, name_suffix=':'+str(f), f_args=args, f_kwargs=kwargs )
  assert not args
  name = f.__name__ + kwargs.pop( 'name_suffix', '' )
  f_args = kwargs.pop( 'f_args', () )
  f_kwargs = kwargs.pop( 'f_kwargs', {} )
  assert not kwargs
  pkgname, scope = f.__module__.split( '.', 1 )
  assert pkgname == __name__
  pkg = PACKAGES
  for item in scope.split( '.' ):
    pkg = pkg.setdefault( item, collections.OrderedDict() )
    assert isinstance( pkg, dict )
  assert name not in pkg
  pkg[name] = lambda: f(*f_args, **f_kwargs)
  return f

class _NoException( Exception ): pass

def unittest( func=None, *, name=None, raises=_NoException ):
  if func is None:
    return functools.partial( unittest, name=name, raises=raises )
  fullname = func.__name__
  if name is not None:
    fullname += ':{}'.format(name)
  if core.getprop( 'filter', fullname ) != fullname:
    return
  parentlog = log._getlog()
  __log__ = log.CaptureLog()
  parentlog.push( fullname )
  try:
    parentlog.write( 'info', 'testing..', endl=False )
    func()
    assert raises is _NoException, 'exception not raised, expected {!r}'.format( raises )
  except raises:
    status = OK
    print( ' OK' )
  except AssertionError:
    status = FAILED
    exc, frames = debug.exc_info()
    print( ' FAILED:', str(exc).strip() )
  except KeyboardInterrupt:
    raise
  except:
    status = ERROR
    exc, frames = debug.exc_info()
    print( ' ERROR:', str(exc).strip() )
  else:
    status = OK
    print( ' OK' )
  finally:
    parentlog.pop()
  core.getprop('results')[fullname] = status
  if status != OK:
    parentlog.write( 'info', 'captured output:\n-----\n{}\n-----'.format(__log__.captured) )
    if core.getprop( 'tbexplore', False ):
      debug.explore( repr(exc), frames, '''Unit test {!r} failed. The traceback
        explorer allows you to examine the failure state. Closing the explorer
        will resume testing.'''.format( fullname ) )


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
