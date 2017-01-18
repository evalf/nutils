# -*- coding: utf8 -*-
#
# Module CLI
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The cli (command line interface) module provides the `cli.run` function that
can be used set up properties, initiate an output environment, and execute a
python function based arguments specified on the command line.
"""

from . import log, core, version, debug, util
import sys, inspect, os, time, argparse, traceback

def _githash( path, depth=0  ):
  abspath = os.path.abspath( path )
  for i in range( depth ):
    abspath = os.path.dirname( abspath )
  git = os.path.join( abspath, '.git' )
  with open( os.path.join( git, 'HEAD' ) ) as HEAD:
    head = HEAD.read()
  assert head.startswith( 'ref:' )
  ref = head[4:].strip()
  with open( os.path.join( git, ref ) ) as ref:
    githash, = ref.read().split()
  return githash

def _bool( s ):
  if s in ('true','True'):
    return True
  if s in ('false','False'):
    return False
  raise argparse.ArgumentTypeError( 'invalid boolean value: {!r}'.format(s) )

def _relative_paths( basepath, path ):
  baseparts = basepath.rstrip( os.path.sep ).split( os.path.sep )
  parts = path.rstrip( os.path.sep ).split( os.path.sep )
  if parts[:len(baseparts)] == baseparts:
    for i in range( len(baseparts), len(parts) ):
      yield os.path.sep.join(parts[:i]), os.path.sep.join(parts[i:])

def run( *functions ):
  '''parse command line arguments and call function'''

  assert functions

  # parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument( '--nprocs', type=int, metavar='INT', default=core.globalproperties['nprocs'], help='number of processors' )
  parser.add_argument( '--outrootdir', type=str, metavar='PATH', default=core.globalproperties['outrootdir'], help='root directory for output' )
  parser.add_argument( '--outdir', type=str, metavar='PATH', default=None, help='custom directory for output' )
  parser.add_argument( '--verbose', type=int, metavar='INT', default=core.globalproperties['verbose'], help='verbosity level' )
  parser.add_argument( '--richoutput', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['richoutput'], help='use rich output (colors, unicode)' )
  parser.add_argument( '--htmloutput', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['htmloutput'], help='generate a HTML log' )
  parser.add_argument( '--tbexplore', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['tbexplore'], help='start traceback explorer on error' )
  parser.add_argument( '--imagetype', type=str, metavar='STR', default=core.globalproperties['imagetype'], help='default image type' )
  parser.add_argument( '--symlink', type=str, metavar='STR', default=core.globalproperties['symlink'], help='create symlink to latest results' )
  parser.add_argument( '--recache', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['recache'], help='overwrite existing cache' )
  parser.add_argument( '--dot', type=str, metavar='STR', default=core.globalproperties['dot'], help='graphviz executable' )
  parser.add_argument( '--selfcheck', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['selfcheck'], help='active self checks (slow!)' )
  parser.add_argument( '--profile', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['profile'], help='show profile summary at exit' )
  subparsers = parser.add_subparsers( dest='command', help='command (add -h for command-specific help)' )
  subparsers.required = True
  for func in functions:
    subparser = subparsers.add_parser( func.__name__, formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    for parameter in inspect.signature( func ).parameters.values():
      subparser.add_argument( '--'+parameter.name,
        dest='='+parameter.name, # prefix with '=' to distinguish nutils/func args
        default=parameter.default,
        metavar=parameter.name[0].upper(),
        help=parameter.annotation if parameter.annotation is not inspect._empty else None,
        type=str )
  ns = parser.parse_args()

  # set properties
  __scriptname__ = os.path.basename(sys.argv[0])
  __nprocs__ = ns.nprocs
  __outrootdir__ = os.path.abspath(os.path.expanduser(ns.outrootdir))
  __cachedir__ = os.path.join( __outrootdir__, __scriptname__, 'cache' )
  __outdir__ = os.path.abspath(os.path.expanduser(ns.outdir)) if ns.outdir is not None \
          else os.path.join( __outrootdir__, __scriptname__, time.strftime( '%Y/%m/%d/%H-%M-%S/', time.localtime() ) )
  __verbose__ = ns.verbose
  __richoutput__ = ns.richoutput
  __htmloutput__ = ns.htmloutput
  __tbexplore__ = ns.tbexplore
  __imagetype__ = ns.imagetype
  __symlink__ = ns.symlink
  __recache__ = ns.recache
  __dot__ = ns.dot
  __selfcheck__ = ns.selfcheck
  __profile__ = ns.profile

  # call function
  func = { f.__name__: f for f in functions }[ ns.command ]
  kwargs = { key[1:]: val for key, val in vars(ns).items() if key[0] == '=' }
  status = call( func, **kwargs )
  sys.exit( status )

def call( func, **kwargs ):
  '''set up compute environment and call function'''

  outdir = core.getprop( 'outdir' )
  os.makedirs( outdir ) # asserts nonexistence

  symlink = core.getprop( 'symlink', None )
  if symlink:
    for base, relpath in _relative_paths( core.getprop('outrootdir'), outdir ):
      target = os.path.join( base, symlink )
      if os.path.islink( target ):
        os.remove( target )
      os.symlink( relpath, target )

  htmloutput = core.getprop( 'htmloutput', True )
  if htmloutput:
    redirect = '<html>\n<head>\n<meta http-equiv="cache-control" content="max-age=0" />\n' \
             + '<meta http-equiv="cache-control" content="no-cache" />\n' \
             + '<meta http-equiv="expires" content="0" />\n' \
             + '<meta http-equiv="expires" content="Tue, 01 Jan 1980 1:00:00 GMT" />\n' \
             + '<meta http-equiv="pragma" content="no-cache" />\n' \
             + '<meta http-equiv="refresh" content="0;URL={}" />\n</head>\n</html>\n'
    for base, relpath in _relative_paths( core.getprop('outrootdir'), outdir ):
      with open( os.path.join( base, 'log.html' ), 'w' ) as redirlog:
        print( redirect.format( os.path.join( relpath, 'log.html' ) ), file=redirlog )

  # `chdir` to `outdir`.  Since this function raises `SystemExit` at the end
  # don't bother restoring the current directory.
  if outdir != '.':
    os.chdir( outdir )

  scriptname = core.getprop( 'scriptname' )
  textlog = log._mklog()
  if htmloutput:
    title = '{} {}'.format( scriptname, time.strftime( '%Y/%m/%d %H:%M:%S', time.localtime() ) )
    htmllog = log.HtmlLog( os.path.join( outdir, 'log.html' ), title=title, scriptname=scriptname )
    __log__ = log.TeeLog( textlog, htmllog )
  else:
    __log__ = textlog

  with __log__:

    ctime = time.ctime()

    try:
      import signal
      signal.signal( signal.SIGTSTP, debug.signal_handler ) # start traceback explorer at ^Z
    except Exception as e:
      log.warning( 'failed to install signal handler:', e )

    try:
      gitversion = version + '.' + _githash(__file__,2)[:8]
    except:
      gitversion = version
    log.info( 'nutils v{}'.format( gitversion ) )
    log.info( '' )

    textlog.write( 'info', ' \\\n'.join( [ ' '.join([ scriptname, func.__name__ ]) ] + [ '  --{}={}'.format( *item ) for item in kwargs.items() ] ) )
    if htmloutput:
      htmllog.write( 'info', '{} {}'.format( scriptname, func.__name__ ) )
      for arg, value in kwargs.items():
        htmllog.write( 'info', '  --{}={}'.format( arg, value ) )

    log.info( '' )
    log.info( 'start {}'.format(ctime) )
    log.info( '' )

    t0 = time.time()

    if core.getprop( 'profile' ):
      import cProfile
      prof = cProfile.Profile()
      prof.enable()

    failed = 1
    frames = None
    try:
      func( **kwargs )
    except KeyboardInterrupt:
      log.error( 'killed by user' )
    except Exception:
      exc, frames = debug.exc_info()
      log.error( traceback.format_exc() )
    else:
      failed = 0

    if core.getprop( 'profile' ):
      prof.disable()

    dt = time.time() - t0
    hours = dt // 3600
    minutes = dt // 60 - 60 * hours
    seconds = dt // 1 - 60 * minutes - 3600 * hours

    log.info( '' )
    log.info( 'finish {}'.format( time.ctime() ) )
    log.info( 'elapsed %02.0f:%02.0f:%02.0f' % ( hours, minutes, seconds ) )

    if core.getprop( 'uncollected_summary', False ):
      debug.trace_uncollected()

    if core.getprop( 'profile' ):
      import pstats
      stream = BufferStream()
      stream.write( 'profile results:\n' )
      pstats.Stats( prof, stream=stream ).strip_dirs().sort_stats( 'time' ).print_stats()
      log.warning( str(stream) )

    if frames and htmloutput:
      htmllog.write_post_mortem( repr(exc), frames )

  if frames:
    if core.getprop( 'tbexplore', False ):
      debug.explore( repr(exc), frames, '''
        Your program has died. The traceback explorer allows you to
        examine its post-mortem state to figure out why this happened.
        Type 'help' for an overview of commands to get going.''' )

  return failed

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
