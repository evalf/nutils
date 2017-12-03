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

from . import log, core, version
import sys, inspect, os, datetime, argparse, pdb, signal, subprocess, pathlib, contextlib

def _version():
  try:
    githash = subprocess.check_output( ['git','rev-parse','--short','HEAD'], universal_newlines=True, cwd=os.path.dirname(__file__) ).strip()
    if subprocess.check_output( ['git','status','--untracked-files=no','--porcelain'], cwd=os.path.dirname(__file__) ):
      githash += '+'
  except:
    return version
  else:
    return '{} (git:{})'.format( version, githash )

def _bool( s ):
  if s in ('true','True'):
    return True
  if s in ('false','False'):
    return False
  raise argparse.ArgumentTypeError( 'invalid boolean value: {!r}'.format(s) )

def _mkbox( *lines ):
  width = max( len(line) for line in lines )
  ul, ur, ll, lr, hh, vv = '┌┐└┘─│' if core.getprop('richoutput') else '++++-|'
  return '\n'.join( [ ul + hh * (width+2) + ur ]
                  + [ vv + (' '+line).ljust(width+2) + vv for line in lines ]
                  + [ ll + hh * (width+2) + lr ] )

def _sigint_handler( mysignal, frame ):
  _handler = signal.signal( mysignal, signal.SIG_IGN ) # temporarily disable handler
  try:
    while True:
      answer = input( 'interrupted. quit, continue or start debugger? [q/c/d]' )
      if answer == 'q':
        raise KeyboardInterrupt
      if answer == 'c' or answer == 'd':
        break
    if answer == 'd': # after break, to minimize code after set_trace
      print( _mkbox(
        'TRACING ACTIVATED. Use the Python debugger',
        'to step through the code at source line',
        'level, list source code, set breakpoints,',
        'and evaluate arbitrary Python code in the',
        'context of any stack frame. Type "h" for',
        'an overview of commands to get going, or',
        '"c" to continue uninterrupted execution.' ) )
      pdb.set_trace()
  finally:
    signal.signal( mysignal, _handler )

Path = pathlib.Path

def run(func, *, args=None, scriptname=None):
  '''parse command line arguments and call function'''

  return choose(func, cmd=False, args=args, scriptname=scriptname)

def choose(*functions, cmd=True, args=None, scriptname=None):
  '''parse command line arguments and call one of multiple functions'''

  assert functions, 'no functions specified'
  assert cmd or len(functions) == 1, 'multiple functions conflicting with cmd=False'

  # parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument( '--nprocs', type=int, metavar='INT', default=core.globalproperties['nprocs'], help='number of processors' )
  parser.add_argument( '--outrootdir', type=str, metavar='PATH', default=core.globalproperties['outrootdir'], help='root directory for output' )
  parser.add_argument( '--outdir', type=str, metavar='PATH', default=core.globalproperties['outdir'], help='custom directory for output' )
  parser.add_argument( '--verbose', type=int, metavar='INT', default=core.globalproperties['verbose'], help='verbosity level' )
  parser.add_argument( '--richoutput', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['richoutput'], help='use rich output (colors, unicode)' )
  parser.add_argument( '--htmloutput', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['htmloutput'], help='generate a HTML log' )
  parser.add_argument( '--pdb', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['pdb'], help='start python debugger on error' )
  parser.add_argument( '--imagetype', type=str, metavar='STR', default=core.globalproperties['imagetype'], help='default image type' )
  parser.add_argument( '--symlink', type=str, metavar='STR', default=core.globalproperties['symlink'], help='create symlink to latest results' )
  parser.add_argument( '--recache', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['recache'], help='overwrite existing cache' )
  parser.add_argument( '--dot', type=str, metavar='STR', default=core.globalproperties['dot'], help='graphviz executable' )
  parser.add_argument( '--selfcheck', type=_bool, nargs='?', const=True, metavar='BOOL', default=core.globalproperties['selfcheck'], help='active self checks (slow!)' )
  if cmd:
    subparsers = parser.add_subparsers( dest='command', help='command (add -h for command-specific help)' )
    subparsers.required = True
  for func in functions:
    subparser = subparsers.add_parser( func.__name__ ) if cmd else parser.add_argument_group( 'optional arguments for {}'.format(func.__name__) )
    for parameter in inspect.signature( func ).parameters.values():
      subparser.add_argument( '--'+parameter.name,
        dest='='+parameter.name, # prefix with '=' to distinguish nutils/func args
        default=parameter.default,
        metavar=type(parameter.default).__name__.upper(),
        help='{} (default: %(default)s)'.format(parameter.annotation) if parameter.annotation is not parameter.empty else 'default: %(default)s',
        **{'type':_bool,'nargs':'?','const':True} if isinstance( parameter.default, bool ) else {'type':type(parameter.default)} )
  ns = parser.parse_args( args )

  # set properties
  __scriptname__ = scriptname or os.path.basename(sys.argv[0])
  __nprocs__ = ns.nprocs
  __outrootdir__ = ns.outrootdir
  __outdir__ = ns.outdir
  __verbose__ = ns.verbose
  __richoutput__ = ns.richoutput
  __htmloutput__ = ns.htmloutput
  __pdb__ = ns.pdb
  __imagetype__ = ns.imagetype
  __symlink__ = ns.symlink
  __recache__ = ns.recache
  __dot__ = ns.dot
  __selfcheck__ = ns.selfcheck

  # call function
  func = { f.__name__: f for f in functions }[ ns.command ] if cmd else functions[0]
  kwargs = { key[1:]: val for key, val in vars(ns).items() if key[0] == '=' }
  status = call( func, **kwargs )
  sys.exit( status )

def call( func, **kwargs ):
  '''set up compute environment and call function'''

  starttime = datetime.datetime.now()
  scriptname = core.getprop('scriptname')

  with contextlib.ExitStack() as stack:

    stack.callback( signal.signal, signal.SIGINT, signal.signal( signal.SIGINT, _sigint_handler ) )

    outdir = os.path.expanduser(core.getprop('outdir'))
    if outdir:
      relpaths = ()
    else:
      outrootdir = os.path.expanduser(core.getprop('outrootdir'))
      ymdt = starttime.strftime('%Y/%m/%d/%H-%M-%S/')
      outdir = os.path.join(outrootdir, scriptname, ymdt)
      __outdir__ = outdir # set property
      __cachedir__ = os.path.join(outrootdir, scriptname, 'cache')
      relpaths = (outrootdir, os.path.join(scriptname, ymdt)), (os.path.join(outrootdir, scriptname), ymdt)
    os.makedirs(outdir) # asserts nonexistence

    if os.open in os.supports_dir_fd:
      __outdirfd__ = os.open( outdir, flags=os.O_RDONLY )
      stack.callback( os.close, __outdirfd__ )

    symlink = core.getprop( 'symlink', None )
    if symlink:
      for base, relpath in relpaths:
        target = os.path.join( base, symlink )
        if os.path.islink( target ):
          os.remove( target )
        os.symlink( relpath, target )

    htmloutput = core.getprop( 'htmloutput', True )
    if htmloutput:
      for base, relpath in relpaths:
        with open( os.path.join(base,'log.html'), 'w' ) as redirlog:
          print( '<html><head>', file=redirlog )
          print( '<meta http-equiv="cache-control" content="max-age=0" />', file=redirlog )
          print( '<meta http-equiv="cache-control" content="no-cache" />', file=redirlog )
          print( '<meta http-equiv="expires" content="0" />', file=redirlog )
          print( '<meta http-equiv="expires" content="Tue, 01 Jan 1980 1:00:00 GMT" />', file=redirlog )
          print( '<meta http-equiv="pragma" content="no-cache" />', file=redirlog )
          print( '<meta http-equiv="refresh" content="0;URL={}" />'.format(os.path.join(relpath,'log.html')), file=redirlog )
          print( '</head></html>', file=redirlog )

    __log__ = log._mklog() if not htmloutput \
         else log.TeeLog( log._mklog(), log.HtmlLog( 'log.html', title=scriptname, scriptname=scriptname ) )
    try:
      with __log__:

        log.info( 'nutils v{}'.format( _version() ) )
        log.info( '' )
        log.info( '{} {}'.format( scriptname, func.__name__ ) )
        for parameter in inspect.signature( func ).parameters.values():
          argstr = '  --{}={}'.format( parameter.name, kwargs.get(parameter.name,parameter.default) )
          if parameter.annotation is not parameter.empty:
            argstr += ' ({})'.format( parameter.annotation )
          log.info( argstr )

        log.info( '' )
        log.info( 'start {}'.format( starttime.ctime() ) )
        log.info( '' )

        func( **kwargs )

        endtime = datetime.datetime.now()
        minutes, seconds = divmod( (endtime-starttime).seconds, 60 )
        hours, minutes = divmod( minutes, 60 )

        log.info( '' )
        log.info( 'finish {}'.format( endtime.ctime() ) )
        log.info( 'elapsed {:.0f}:{:02.0f}:{:02.0f}'.format( hours, minutes, seconds ) )

    except (KeyboardInterrupt,SystemExit,pdb.bdb.BdbQuit):
      return 1
    except:
      if core.getprop( 'pdb', False ):
        try:
          del __log__
        except NameError:
          pass
        print( _mkbox(
          'YOUR PROGRAM HAS DIED. The Python debugger',
          'allows you to examine its post-mortem state',
          'to figure out why this happened. Type "h"',
          'for an overview of commands to get going.' ) )
        pdb.post_mortem()
      return 2
    else:
      return 0

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
