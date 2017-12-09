# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The cli (command line interface) module provides the `cli.run` function that
can be used set up properties, initiate an output environment, and execute a
python function based arguments specified on the command line.
"""

from . import log, core, version
import sys, inspect, os, datetime, pdb, signal, subprocess, contextlib

def _version():
  try:
    githash = subprocess.check_output( ['git','rev-parse','--short','HEAD'], universal_newlines=True, cwd=os.path.dirname(__file__) ).strip()
    if subprocess.check_output( ['git','status','--untracked-files=no','--porcelain'], cwd=os.path.dirname(__file__) ):
      githash += '+'
  except:
    return version
  else:
    return '{} (git:{})'.format( version, githash )

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

def run(func, *, skip=1):
  '''parse command line arguments and call function'''

  params = inspect.signature(func).parameters.values()

  if '-h' in sys.argv[skip:] or '--help' in sys.argv[skip:]:
    print('usage: {} (...)'.format(' '.join(sys.argv[:skip])))
    print()
    for param in params:
      cls = param.default.__class__
      print('  --{:<20}'.format(param.name + '=' + cls.__name__.upper() if cls != bool else '(no)' + param.name), end=' ')
      if param.annotation != param.empty:
        print(param.annotation, end=' ')
      print('[{}]'.format(param.default))
    sys.exit(1)

  kwargs = {param.name: param.default for param in params}

  for arg in sys.argv[skip:]:
    name, sep, value = arg.lstrip('-').partition('=')
    if not sep:
      value = not name.startswith('no')
      if not value:
        name = name[2:]
    if name in kwargs:
      default = kwargs[name]
      args = kwargs
    else:
      try:
        default = core.getprop(name)
      except NameError:
        print('invalid argument {!r}'.format(arg))
        sys.exit(2)
      name = '__{}__'.format(name)
      args = locals()
    try:
      if isinstance(default, bool) and not isinstance(value, bool):
        raise Exception('boolean value should be specifiec as --{0}/--no{0}'.format(name))
      args[name] = default.__class__(value)
    except Exception as e:
      print('invalid argument for {!r}: {}'.format(name, e))
      sys.exit(2)

  status = call(func, **kwargs)
  sys.exit(status)

def choose(*functions):
  '''parse command line arguments and call one of multiple functions'''

  assert functions, 'no functions specified'

  funcnames = [func.__name__ for func in functions]
  if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
    print('usage: {} [{}] (...)'.format(sys.argv[0], '|'.join(funcnames)))
    sys.exit(1)

  try:
    ifunc = funcnames.index(sys.argv[1])
  except ValueError:
    print('invalid argument {!r}; choose from {}'.format(sys.argv[1], ', '.join(funcnames)))
    sys.exit(2)

  run(functions[ifunc], skip=2)

def call( func, **kwargs ):
  '''set up compute environment and call function'''

  starttime = datetime.datetime.now()
  scriptname = os.path.basename(sys.argv[0])

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
