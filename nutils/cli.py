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

from . import long_version, warnings, matrix
import sys, inspect, os, time, signal, subprocess, contextlib, traceback, pathlib, html, functools, pdb, stringly, textwrap, typing, treelog, collections

try:
  Level = treelog.proto.Level
except AttributeError: # treelog version < 1.0b6
  Level = collections.namedtuple('Level', ['debug', 'info', 'user', 'warning', 'error'])(0,1,2,3,4)

def _version():
  try:
    githash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True, stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)).strip()
    if subprocess.check_output(['git', 'status', '--untracked-files=no', '--porcelain'], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)):
      githash += '+'
  except:
    return long_version
  else:
    return '{} (git:{})'.format(long_version, githash)

def _mkbox(*lines, richoutput=False):
  width = max(len(line) for line in lines)
  ul, ur, ll, lr, hh, vv = '┌┐└┘─│' if richoutput else '++++-|'
  return '\n'.join([ul + hh * (width+2) + ur]
                 + [vv + (' '+line).ljust(width+2) + vv for line in lines]
                 + [ll + hh * (width+2) + lr])

@contextlib.contextmanager
def _signal_handler(s, handler):
  oldhandler = signal.signal(s, handler)
  try:
    yield
  finally:
    signal.signal(s, oldhandler)

def _breakpoint(richoutput, mysignal, frame):
  with _signal_handler(mysignal, signal.SIG_IGN): # temporarily disable handler
    while True:
      answer = input('interrupted. quit, continue or start debugger? [q/c/d]')
      if answer == 'q':
        raise KeyboardInterrupt
      if answer == 'c' or answer == 'd':
        break
    if answer == 'd': # after break, to minimize code after set_trace
      print(_mkbox(
        'TRACING ACTIVATED. Use the Python debugger',
        'to step through the code at source line',
        'level, list source code, set breakpoints,',
        'and evaluate arbitrary Python code in the',
        'context of any stack frame. Type "h" for',
        'an overview of commands to get going, or',
        '"c" to continue uninterrupted execution.', richoutput=richoutput))
      pdb.set_trace()

@contextlib.contextmanager
def _traceback(richoutput, postmortem, exit):
  try:
    yield
  except (KeyboardInterrupt, SystemExit, pdb.bdb.BdbQuit):
    treelog.error('killed by user')
    if exit:
      raise SystemExit(1) from None
    raise
  except:
    exc = traceback.TracebackException(*sys.exc_info())
    prefix = ''
    while True:
      treelog.error(prefix + ''.join(exc.format_exception_only()).rstrip())
      treelog.debug('Traceback (most recent call first):\n' + ''.join(reversed(exc.stack.format())).rstrip())
      if exc.__cause__ is not None:
        exc = exc.__cause__
        prefix = '.. caused by '
      elif exc.__context__ is not None and not exc.__suppress_context__:
        exc = exc.__context__
        prefix = '.. while handling '
      else:
        break
    if postmortem:
      print(_mkbox(
        'YOUR PROGRAM HAS DIED. The Python debugger',
        'allows you to examine its post-mortem state',
        'to figure out why this happened. Type "h"',
        'for an overview of commands to get going.', richoutput=richoutput))
      pdb.post_mortem()
    if exit:
      raise SystemExit(2) from None
    raise
  else:
    if exit:
      raise SystemExit(0)

try:
  from bottombar import BottomBar
except ImportError:
  BottomBar = None

try:
  from resource import getrusage, RUSAGE_SELF
except ImportError:
  try:
    from psutil import Process
  except ImportError:
    _rss_memory = None
  else:
    _rss_memory = lambda: Process().memory_info().rss
else:
  _rss_memory = lambda: getrusage(RUSAGE_SELF).ru_maxrss << 10

def _format(uri, t0, width):
  if len(uri) + 8 <= width: # space for full uri, maybe stats
    minutes, seconds = divmod(int(time.perf_counter() - t0), 60)
    hours, minutes = divmod(minutes, 60)
    runtime = ' runtime: {}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    memory = ' memory: {:,}M |'.format(_rss_memory() >> 20) if _rss_memory else ''
    prefix = 'writing log to '
    if len(prefix) + len(uri) + len(memory) + len(runtime) > width:
      memory = memory[8:]
      runtime = runtime[9:]
    if len(prefix) + len(uri) + len(memory) + len(runtime) <= width:
      uri = prefix + uri
    elif len(uri) + len(memory) + len(runtime) > width:
      memory = ''
    if len(uri) + len(memory) + len(runtime) <= width:
      uri += (memory + runtime).rjust(width - len(uri))
  elif len(uri) > width: # no space for uri
    uri = '...' + uri[3-width:]
  return '\033[2m' + uri

@contextlib.contextmanager
def _status(uri, richoutput):
  try:
    if richoutput and BottomBar:
      with BottomBar(uri, time.perf_counter(), format=_format, interval=1):
        yield
    else:
      print('opened log at', uri)
      yield
  finally:
    print('log written to', uri)

def _load_rcfile(path):
  settings = {}
  try:
    with open(path) as rc:
      exec(rc.read(), {}, settings)
  except Exception as e:
    raise Exception('error loading config from {}'.format(path)) from e
  return settings

def _htmllog(outdir, scriptname, kwargs):
  htmllog = treelog.HtmlLog(outdir, title=scriptname, htmltitle='<a href="http://www.nutils.org">{}</a> {}'.format(SVGLOGO, html.escape(scriptname)), favicon=FAVICON)
  if kwargs:
    htmllog.write('<ul style="list-style-position: inside; padding-left: 0px; margin-top: 0px;">{}</ul>'.format(''.join(
      '<li>{}={} <span style="color: gray;">{}</span></li>'.format(name, value, doc) for name, value, doc in kwargs)), level=Level.info, escape=False)
  return htmllog

def run(func, *, args=None, loaduserconfig=True):
  '''parse command line arguments and call function'''

  if args is None:
    args = sys.argv.copy()

  scriptname = os.path.basename(args.pop(0))
  sig = inspect.signature(func)
  doc = stringly.util.DocString(func)

  argdocs = doc.argdocs
  for param in sig.parameters.values():
    if isinstance(param.annotation, str):
      argdocs[param.name] = param.annotation

  types = {param.name: param.annotation for param in inspect.signature(setup).parameters.values() if param.default is not param.empty}
  for param in sig.parameters.values():
    if param.annotation is not param.empty and not isinstance(param.annotation, str):
      types[param.name] = param.annotation
    elif param.default is not param.empty:
      types[param.name] = type(param.default)
    else:
      sys.exit('cannot infer type of argument {!r}'.format(param.name))

  if '-h' in args or '--help' in args:
    usage = []
    if doc.text:
      usage.append(doc.text)
      usage.append('\n\n')
    usage.append('USAGE: {}'.format(scriptname))
    if doc.presets:
      usage.append(' [{}]'.format('|'.join(doc.presets)))
    if sig.parameters:
      usage.append(' [arg=value] [...]\n')
      defaults = doc.defaults
      for param in sig.parameters.values():
        usage.append('\n  {}'.format(param.name))
        if param.name in defaults:
          usage.append(' [{}]'.format(defaults[param.name]))
        elif param.default is not param.empty:
          usage.append(' [{}]'.format(stringly.dumps(types[param.name], param.default)))
        if param.name in argdocs:
          usage.extend(textwrap.wrap(argdocs[param.name], initial_indent='\n    ', subsequent_indent='\n    '))
    print(''.join(usage))
    sys.exit(1)

  strargs = doc.defaults
  if args and args[0] in doc.presets:
    strargs.update(doc.presets[args.pop(0)])
  for arg in args:
    name, sep, value = arg.lstrip('-').partition('=')
    if not sep:
      if name in types:
        value = 'yes'
      elif name.startswith('no') and name[2:] in types:
        name = name[2:]
        value = 'no'
      else:
        print('argument {!r} requires a value'.format(name))
        sys.exit(2)
    strargs[name] = value

  funcargs = {}
  setupargs = {}

  if loaduserconfig:
    home = os.path.expanduser('~')
    for path in os.path.join(home, '.config', 'nutils', 'config'), os.path.join(home, '.nutilsrc'):
      if os.path.isfile(path):
        setupargs.update(_load_rcfile(path))

  for name, s in strargs.items():
    if name not in types:
      sys.exit('unexpected argument: {}'.format(name))
    try:
      value = stringly.loads(types[name], s)
    except stringly.error.StringlyError as e:
      print(e)
      sys.exit(2)
    (funcargs if name in sig.parameters else setupargs)[name] = value

  kwargs = [(param.name, stringly.dumps(types[param.name], funcargs.get(param.name, param.default)), argdocs.get(param.name)) for param in sig.parameters.values()]

  with setup(scriptname=scriptname, kwargs=kwargs, **setupargs):
    func(**funcargs)

def choose(*functions, args=None, loaduserconfig=True):
  '''parse command line arguments and call one of multiple functions'''

  if args is None:
    args = sys.argv.copy()

  assert functions, 'no functions specified'

  funcnames = {func.__name__: func for func in functions}
  if len(args) == 1 or args[1] in ('-h', '--help'):
    print('USAGE: {} [{}] (...)'.format(args[0], '|'.join(funcnames)))
    sys.exit(1)

  funcname = args.pop(1)
  if funcname not in funcnames:
    print('invalid argument {!r}; choose from {}'.format(funcname, ', '.join(funcnames)))
    sys.exit(2)

  run(funcnames[funcname], args=args, loaduserconfig=loaduserconfig)

@contextlib.contextmanager
def setup(scriptname: str,
          kwargs: typing.List[typing.Tuple[str,str,str]],
          outrootdir: str = '~/public_html',
          outdir: typing.Optional[str] = None,
          cachedir: str = 'cache',
          cache: bool = False,
          nprocs: int = 1,
          matrix: matrix.backend = matrix.auto,
          richoutput: typing.Optional[bool] = None,
          outrooturi: typing.Optional[str] = None,
          outuri: typing.Optional[str] = None,
          verbose: typing.Optional[int] = 4,
          pdb: bool = False,
          gracefulexit: bool = True,
          **unused):
  '''Set up compute environment.'''

  from . import cache as _cache, parallel as _parallel

  for name in unused:
    warnings.warn('ignoring unused configuration variable {!r}'.format(name))

  if outdir is None:
    outdir = os.path.join(os.path.expanduser(outrootdir), scriptname)
    if outrooturi is None:
      outrooturi = pathlib.Path(outrootdir).expanduser().resolve().as_uri()
    outuri = outrooturi.rstrip('/') + '/' + scriptname
  elif outuri is None:
    outuri = pathlib.Path(outdir).resolve().as_uri()

  if richoutput is None:
    richoutput = sys.stdout.isatty()

  consolellog = treelog.RichOutputLog() if richoutput else treelog.StdoutLog()
  if verbose is not None:
    consolellog = treelog.FilterLog(consolellog, minlevel=tuple(Level)[5-verbose])
  htmllog = _htmllog(outdir, scriptname, kwargs)

  with htmllog, \
       _status(outuri+'/'+htmllog.filename, richoutput), \
       treelog.set(treelog.TeeLog(consolellog, htmllog)), \
       _traceback(richoutput=richoutput, postmortem=pdb, exit=gracefulexit), \
       warnings.via(treelog.warning), \
       _cache.enable(os.path.join(outdir, cachedir)) if cache else _cache.disable(), \
       _parallel.maxprocs(nprocs), \
       matrix, \
       _signal_handler(signal.SIGINT, functools.partial(_breakpoint, richoutput)):

    treelog.info('nutils v{}'.format(_version()))
    treelog.info('start', time.ctime())
    yield
    treelog.info('finish', time.ctime())

SVGLOGO = '''\
<svg style="vertical-align: middle;" width="32" height="32" xmlns="http://www.w3.org/2000/svg">
  <path d="M7.5 19 v-6 a6 6 0 0 1 12 0 v6 M25.5 13 v6 a6 6 0 0 1 -12 0 v-6" fill="none" stroke-width="3" stroke-linecap="round"/>
</svg>'''

FAVICON = 'data:image/png;base64,' \
  'iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAQAAAD9CzEMAAACAElEQVRYw+2YS04bQRCGP2wJ' \
  'gbAimS07WABXGMLzAgiBcgICFwDEEiGiDCScggWPHVseC1AIZ8AIJBA2hg1kF5DiycLYqppp' \
  'M91j2KCp3rSq//7/VldPdfVAajHW0nAkywDjeHSTBx645IRdfvPvLWTbWeSewNDuWKC9Wfov' \
  '3BjJa+2aqWa2bInKq/QBARV8MknoM2zHktfaVhKJ79b0AQEr7nsfpthjml466KCPr+xHNmrS' \
  '7eTo0J4xFMEMUwiFu81eYFFNPSJvROU5Vrh5W/qsOvdnDegBOjkXyDJZO4Fhta7RV7FDCvvZ' \
  'TmBdhTbODgT6R9zJr9qA8G2LfiurlCji0yq8O6LvKT4zHlQEeoXfr3t94e1TUSAWDzyJKTnh' \
  'L9W9t8KbE+i/iieCr6XroEEKb9qfee8LJxVIBVKBjyRQqnuKavxZpTiZ1Ez4Typ9KoGN+sCG' \
  'Evgj+l2ib8ZLxCOhi8KnaLgoTkVino7Fzwr0L7st/Cmm7MeiDwV6zU5gUF3wYw6Fg2dbztyJ' \
  'SQWHcsb6fC6odR3T2YBeF2RzLiXltZpaYCSCGVWrD7hyKSlhKvJiOGCGfnLk6GdGhbZaFE+4' \
  'fo7fnMr65STf+5Y1/Way9PPOT6uqTYbCHW5X7nsftjbmKRvJy8yZT05Lgnh4jOPR8/JAv+CE' \
  'XU6ppH81Etp/wL7MKaEwo4sAAAAASUVORK5CYII='

# vim:sw=2:sts=2:et
