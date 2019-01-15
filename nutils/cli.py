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

from . import util, config, long_version, warnings, matrix, cache
import sys, inspect, os, io, time, pdb, signal, subprocess, contextlib, traceback, pathlib, html, treelog as log, stickybar

def _version():
  try:
    githash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True, stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)).strip()
    if subprocess.check_output(['git', 'status', '--untracked-files=no', '--porcelain'], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)):
      githash += '+'
  except:
    return long_version
  else:
    return '{} (git:{})'.format(long_version, githash)

def _mkbox(*lines):
  width = max(len(line) for line in lines)
  ul, ur, ll, lr, hh, vv = '┌┐└┘─│' if config.richoutput else '++++-|'
  return '\n'.join([ul + hh * (width+2) + ur]
                 + [vv + (' '+line).ljust(width+2) + vv for line in lines]
                 + [ll + hh * (width+2) + lr])

def _sigint_handler(mysignal, frame):
  _handler = signal.signal(mysignal, signal.SIG_IGN) # temporarily disable handler
  try:
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
        '"c" to continue uninterrupted execution.'))
      pdb.set_trace()
  finally:
    signal.signal(mysignal, _handler)

def _hms(dt):
  seconds = int(dt)
  minutes, seconds = divmod(seconds, 60)
  hours, minutes = divmod(minutes, 60)
  return hours, minutes, seconds

def run(func, *, skip=1, loaduserconfig=True):
  '''parse command line arguments and call function'''

  configs = []
  if loaduserconfig:
    home = os.path.expanduser('~')
    configs.append(dict(richoutput=sys.stdout.isatty()))
    configs.extend(path for path in (os.path.join(home, '.config', 'nutils', 'config'), os.path.join(home, '.nutilsrc')) if os.path.isfile(path))

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
  cli_config = {}

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
        default = getattr(config, name)
      except AttributeError:
        print('invalid argument {!r}'.format(arg))
        sys.exit(2)
      args = cli_config
    try:
      if isinstance(default, bool) and not isinstance(value, bool):
        raise Exception('boolean value should be specifiec as --{0}/--no{0}'.format(name))
      args[name] = default.__class__(value)
    except Exception as e:
      print('invalid argument for {!r}: {}'.format(name, e))
      sys.exit(2)

  with config(*configs, **cli_config):
    status = call(func, kwargs, scriptname=os.path.basename(sys.argv[0]), funcname=None if skip==1 else func.__name__)

  sys.exit(status)

def choose(*functions, loaduserconfig=True):
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

  run(functions[ifunc], skip=2, loaduserconfig=loaduserconfig)

def call(func, kwargs, scriptname, funcname=None):
  '''set up compute environment and call function'''

  outdir = config.outdir or os.path.join(os.path.expanduser(config.outrootdir), scriptname)

  with contextlib.ExitStack() as stack:

    stack.enter_context(cache.enable(os.path.join(outdir, config.cachedir)) if config.cache else cache.disable())
    stack.enter_context(matrix.backend(config.matrix))
    stack.enter_context(log.set(log.FilterLog(log.RichOutputLog() if config.richoutput else log.StdoutLog(), minlevel=5-config.verbose)))
    if config.htmloutput:
      htmllog = stack.enter_context(log.HtmlLog(outdir, title=scriptname, htmltitle='<a href="http://www.nutils.org">{}</a> {}'.format(SVGLOGO, html.escape(scriptname)), favicon=FAVICON))
      uri = (config.outrooturi.rstrip('/') + '/' + scriptname if config.outrooturi else pathlib.Path(outdir).resolve().as_uri()) + '/' + htmllog.filename
      if config.richoutput:
        t0 = time.perf_counter()
        bar = lambda running: '{0} [{1}] {2[0]}:{2[1]:02d}:{2[2]:02d}'.format(uri, 'RUNNING' if running else 'STOPPED', _hms(time.perf_counter()-t0))
        stack.enter_context(stickybar.activate(bar, update=1))
      else:
        log.info('opened log at', uri)
      htmllog.write('<ul style="list-style-position: inside; padding-left: 0px; margin-top: 0px;">{}</ul>'.format(''.join(
        '<li>{}={} <span style="color: gray;">{}</span></li>'.format(param.name, kwargs.get(param.name, param.default), param.annotation)
          for param in inspect.signature(func).parameters.values())), level=1, escape=False)
      stack.enter_context(log.add(htmllog))
    stack.enter_context(warnings.via(lambda msg: log.warning(msg)))
    stack.callback(signal.signal, signal.SIGINT, signal.signal(signal.SIGINT, _sigint_handler))

    log.info('nutils v{}'.format(_version()))
    log.info('start', time.ctime())
    try:
      func(**kwargs)
    except (KeyboardInterrupt, SystemExit, pdb.bdb.BdbQuit):
      log.error('killed by user')
      return 1
    except:
      log.error(traceback.format_exc())
      if config.pdb:
        print(_mkbox(
          'YOUR PROGRAM HAS DIED. The Python debugger',
          'allows you to examine its post-mortem state',
          'to figure out why this happened. Type "h"',
          'for an overview of commands to get going.'))
        pdb.post_mortem()
      return 2
    else:
      log.info('finish', time.ctime())
      return 0

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
