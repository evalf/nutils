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
import sys, collections, inspect, os, time

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

def getkwargdefaults( func ):
  'helper for run'

  kwargs = util.OrderedDict()
  signature = inspect.signature( func )
  for parameter in signature.parameters.values():
    if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
      continue
    if parameter.default is parameter.empty:
      raise ValueError( 'Function cannot be called without arguments.' )
    kwargs[parameter.name] = parameter.default
  return kwargs

def run( *functions ):
  'call function specified on command line'

  assert functions

  if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    print( 'Usage: %s [FUNC] [ARGS]' % sys.argv[0] )
    print( '''
  --help                  Display this help
  --nprocs=%(nprocs)-14s Select number of processors
  --outrootdir=%(outrootdir)-10s Define the root directory for output
  --outdir=               Define custom directory for output
  --verbose=%(verbose)-13s Set verbosity level, 9=all
  --richoutput=%(richoutput)-10s Use rich output (colors, unicode)
  --htmloutput=%(htmloutput)-10s Generate an HTML log
  --tbexplore=%(tbexplore)-11s Start traceback explorer on error
  --imagetype=%(imagetype)-11s Set image type
  --symlink=%(symlink)-13s Create symlink to latest results
  --recache=%(recache)-13s Overwrite existing cache
  --dot=%(dot)-17s Set graphviz executable
  --selfcheck=%(selfcheck)-11s Activate self checks (slow!)
  --profile=%(profile)-13s Show profile summary at exit''' % core.globalproperties )
    for i, func in enumerate( functions ):
      print()
      print( 'Arguments for %s%s' % ( func.__name__, '' if i else ' (default)' ) )
      print()
      for kwarg, default in getkwargdefaults( func ).items():
        print( '  --%s=%s' % ( kwarg, default ) )
    return

  func = functions[0]
  argv = sys.argv[1:]
  funcbyname = { func.__name__: func for func in functions }
  if argv and argv[0] in funcbyname:
    func = funcbyname[argv[0]]
    argv = argv[1:]

  kwargs = getkwargdefaults( func )
  properties = {}
  for arg in argv:
    arg = arg.lstrip('-')
    try:
      arg, val = arg.split( '=', 1 )
      val = eval( val, sys._getframe(1).f_globals )
    except ValueError: # split failed
      val = True
    except (SyntaxError,NameError): # eval failed
      pass
    arg = arg.replace( '-', '_' )
    if arg in kwargs:
      kwargs[ arg ] = val
    else:
      assert arg in core.globalproperties, 'invalid argument %r' % arg
      properties[arg] = val

  locals().update({ '__%s__' % name: value for name, value in properties.items() })

  scriptname = os.path.basename(sys.argv[0])
  outrootdir = os.path.expanduser( core.getprop( 'outrootdir' ) ).rstrip( os.sep ) + os.sep
  basedir = outrootdir + scriptname + os.sep
  localtime = time.localtime()
  timepath = time.strftime( '%Y/%m/%d/%H-%M-%S/', localtime )
  outdir = properties.get( 'outdir', None )
  htmloutput = core.getprop( 'htmloutput', True )

  if outdir is None:
    # `outdir` not specified on the commandline, use default directory layout

    outdir = basedir + timepath
    os.makedirs( outdir ) # asserts nonexistence

    if core.getprop( 'symlink' ):
      for i in range(2): # make two links
        target = outrootdir
        dest = ''
        if i: # global link
          target += scriptname + os.sep
        else: # script-local link
          dest += scriptname + os.sep
        target += core.getprop( 'symlink' )
        dest += timepath
        if os.path.islink( target ):
          os.remove( target )
        os.symlink( dest, target )

    if htmloutput:

      redirect = '<html>\n<head>\n<meta http-equiv="cache-control" content="max-age=0" />\n' \
               + '<meta http-equiv="cache-control" content="no-cache" />\n' \
               + '<meta http-equiv="expires" content="0" />\n' \
               + '<meta http-equiv="expires" content="Tue, 01 Jan 1980 1:00:00 GMT" />\n' \
               + '<meta http-equiv="pragma" content="no-cache" />\n' \
               + '<meta http-equiv="refresh" content="0;URL=%slog.html" />\n</head>\n</html>\n'

      with open( outrootdir+'log.html', 'w' ) as redirlog1:
        print( redirect % ( scriptname + '/' + timepath ), file=redirlog1 )

      with open( basedir+'log.html', 'w' ) as redirlog2:
        print( redirect % ( timepath ), file=redirlog2 )

  elif not os.path.isdir( outdir ):
    # use custom directory layout, skip creating symlinks, redirects
    os.makedirs( outdir )

  # `chdir` to `outdir`.  Since this function raises `SystemExit` at the end
  # don't bother restoring the current directory.
  if outdir != '.':
    os.chdir( outdir )

  textlog = log._mklog()
  if htmloutput:
    title = '{} {}'.format( scriptname, time.strftime( '%Y/%m/%d %H:%M:%S', localtime ) )
    htmllog = log.HtmlLog( os.path.join( outdir, 'log.html' ), title=title, scriptname=scriptname )
    __log__ = log.TeeLog( textlog, htmllog )
  else:
    __log__ = textlog

  __outdir__ = outdir
  __cachedir__ = basedir + 'cache'

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
      log.stack( repr(exc), frames )
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

  sys.exit( failed )

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
