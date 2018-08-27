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
The log module provides print methods ``debug``, ``info``, ``user``,
``warning``, and ``error``, in increasing order of priority. Output is sent to
stdout as well as to an html formatted log file if so configured.
"""

import time, functools, itertools, io, abc, contextlib, html, urllib.parse, os, json, traceback, bdb, inspect, textwrap, builtins, hashlib, sys, tempfile
from . import core, config, warnings

LEVELS = 'error', 'warning', 'user', 'info', 'debug' # NOTE this should match the log levels defined in `nutils/_log/viewer.js`
HTMLHEAD = '''\
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, user-scalable=no"/>
<title>{title}</title>
<script src="{viewer_js}"></script>
<link rel="stylesheet" type="text/css" href="{viewer_css}"/>
<link rel="icon" sizes="48x48" type="image/png" href="{favicon_png}"/>
</head>'''

## LOG

class Log(metaclass=abc.ABCMeta):
  '''
  Base class for log objects.  A subclass should define a :meth:`context`
  method that returns a context manager which adds a contextual layer and a
  :meth:`write` method.
  '''

  def __enter__(self):
    if hasattr(self, '_stack'):
      raise RuntimeError('This context manager is not reentrant.')
    stack = contextlib.ExitStack()
    try:
      self._init_context(stack)
      # Replace the current log object with `self` and restore at exit.
      stack.callback(_set_current_log, _current_log)
      _set_current_log(self)
    except:
      stack.__exit__(*sys.exc_info())
      raise
    self._stack = stack
    return self

  def __exit__(self, etype, value, tb):
    if not hasattr(self, '_stack'):
      raise RuntimeError('This context manager is not yet entered.')
    try:
      return self._stack.__exit__(etype, value, tb)
    finally:
      del self._stack

  def _init_context(self, stack):
    pass

  @abc.abstractmethod
  def context(self, title, mayskip=False):
    '''Return a context manager that adds a contextual layer named ``title``.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def write(self, level, text):
    '''Write ``text`` with log level ``level`` to the log.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def open(self, filename, mode, level, exists):
    '''Create file object.'''

class DataLog(Log):
  '''Output only data.'''

  def __init__(self, outdir):
    self.outdir = outdir
    super().__init__()

  def _init_context(self, stack):
    self._open = stack.enter_context(_makedirs(self.outdir, exist_ok=True))
    super()._init_context(stack)

  @contextlib.contextmanager
  def context(self, title, mayskip=False):
    yield

  def write(self, level, text):
    pass

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    with self._open(filename, mode, exists) as f:
      yield f

class ContextLog(Log):
  '''Base class for loggers that keep track of the current list of contexts.

  The base class implements :meth:`context` which keeps the attribute
  :attr:`_context` up-to-date.

  .. attribute:: _context

     A :class:`list` of contexts (:class:`str`\\s) that are currently active.
  '''

  def __init__(self):
    self._context = []
    super().__init__()

  def _push_context(self, title, mayskip):
    self._context.append(title)

  def _pop_context(self):
    self._context.pop()

  @contextlib.contextmanager
  def context(self, title, mayskip=False):
    '''Return a context manager that adds a contextual layer named ``title``.

    The list of currently active contexts is stored in :attr:`_context`.'''
    self._push_context(title, mayskip)
    try:
      yield
    finally:
      self._pop_context()

class ContextTreeLog(ContextLog):
  '''Base class for loggers that display contexts as a tree.

  .. automethod:: _print_push_context
  .. automethod:: _print_pop_context
  .. automethod:: _print_item
  '''

  def __init__(self):
    super().__init__()
    self._printed_context = 0

  def _pop_context(self):
    super()._pop_context()
    if self._printed_context > len(self._context):
      self._printed_context -= 1
      self._print_pop_context()

  def write(self, level, text, **kwargs):
    '''Write ``text`` with log level ``level`` to the log.

    This method makes sure the current context is printed and calls
    :meth:`_print_item`.
    '''
    from . import parallel
    if parallel.procid:
      return
    for title in self._context[self._printed_context:]:
      self._print_push_context(title)
      self._printed_context += 1
    self._print_item(level, text, **kwargs)

  @abc.abstractmethod
  def _print_push_context(self, title):
    '''Push a context to the log.

    This method is called just before the first item of this context is added
    to the log.  If no items are added to the log within this context or
    children of this context this method nor :meth:`_print_pop_context` will be
    called.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def _print_pop_context(self):
    '''Pop a context from the log.

    This method is called whenever a context is exited, but only if
    :meth:`_print_push_context` has been called before for the same context.

    .. Note:: This function is abstract.
    '''

  @abc.abstractmethod
  def _print_item(self, level, text):
    '''Add an item to the log.

    .. Note:: This function is abstract.
    '''

class StdoutLog(ContextLog):
  '''Output plain text to stream.'''

  def __init__(self, stream=None):
    self.stream = stream
    super().__init__()

  def _write_post_mortem(self, etype, value, tb):
    if etype in (None, SystemExit):
      return
    elif etype in (KeyboardInterrupt, bdb.BdbQuit):
      self.write('error', 'killed by user')
    else:
      try:
        msg = ''.join(traceback.format_exception(etype, value, tb))
      except Exception as e:
        msg = '{} (traceback failed: {})'.format(value, e)
      self.write('error', msg)

  def _init_context(self, stack):
    stack.push(self._write_post_mortem)
    super()._init_context(stack)

  def _mkstr(self, level, text):
    return ' > '.join(self._context + ([text] if text is not None else []))

  def write(self, level, text, endl=True):
    verbose = config.verbose
    if level not in LEVELS[verbose:]:
      from . import parallel
      if parallel.procid is not None:
        text = '[{}] {}'.format(parallel.procid, text)
      s = self._mkstr(level, text)
      print(s, end='\n' if endl else '', file=self.stream)

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    with _devnull(filename, mode) as f:
      yield f
    self.write(level, filename)

class RichOutputLog(StdoutLog):
  '''Output rich (colored,unicode) text to stream.'''

  # color order: black, red, green, yellow, blue, purple, cyan, white

  cmap = { 'path': (2,1), 'error': (1,1), 'warning': (1,0), 'user': (3,0) }

  def __init__(self, stream=None, *, progressinterval=None):
    super().__init__(stream=stream)
    # Timestamp at which a new progress line may be written.
    self._progressupdate = 0
    # Progress update interval in seconds.
    self._progressinterval = progressinterval or getattr(config, 'progressinterval', 0.1)

  def _init_context(self, stack):
    stack.callback(print, end='\033[K', file=self.stream) # clear the progress line
    super()._init_context(stack)

  def _mkstr(self, level, text):
    if text is not None:
      string = ' · '.join(self._context + [text])
      n = len(string) - len(text)
      # This is not a progress line.  Reset the update timestamp.
      self._progressupdate = 0
    else:
      string = ' · '.join(self._context)
      n = len(string)
      # Don't touch `self._progressupdate` here.  Will be done in
      # `self._push_context`.
    try:
      colorid, boldid = self.cmap[level]
    except KeyError:
      return '\033[K\033[1;30m{}\033[0m{}'.format(string[:n], string[n:])
    else:
      return '\033[K\033[1;30m{}\033[{};3{}m{}\033[0m'.format(string[:n], boldid, colorid, string[n:])

  def _push_context(self, title, mayskip):
    super()._push_context(title, mayskip)
    from . import parallel
    if parallel.procid:
      return
    t = time.time()
    if not mayskip or t >= self._progressupdate:
      self._progressupdate = t + self._progressinterval
      print(self._mkstr('progress', None), end='\r', file=self.stream)

class HtmlLog(ContextTreeLog):
  '''Output html nested lists.'''

  def __init__(self, outdir, *, title='nutils', scriptname=None, funcname=None, funcargs=None):
    self._outdir = outdir
    self._title = title
    self._scriptname = scriptname
    self._funcname = funcname
    self._funcargs = funcargs
    super().__init__()

  def _init_context(self, stack):
    self._open = stack.enter_context(_makedirs(self._outdir, exist_ok=True))
    # Copy dependencies.
    paths = {}
    for filename in 'favicon.png', 'viewer.css', 'viewer.js':
      with builtins.open(os.path.join(os.path.dirname(__file__), '_log', filename), 'rb') as src:
        data = src.read()
      with self._open(hashlib.sha1(data).hexdigest() + '.' + filename.split('.')[1], 'wb', exists='skip') as dst:
        dst.write(data)
      paths[filename.replace('.', '_')] = dst.name
    # Write header.
    self._file = stack.enter_context(self._open('log.html', 'w', exists='rename'))
    self._print('<!DOCTYPE html>')
    self._print('<html>')
    self._print(HTMLHEAD.format(title=html.escape(self._title), **paths))
    body_attrs = []
    if self._scriptname:
      body_attrs.append(('data-scriptname', html.escape(self._scriptname)))
      body_attrs.append(('data-latest', '../../../../log.html'))
    if self._funcname:
      body_attrs.append(('data-funcname', html.escape(self._funcname)))
    self._print(''.join(['<body'] + [' {}="{}"'.format(*item) for item in body_attrs] + ['>']))
    self._print('<div id="log">')
    if self._funcargs:
      self._print('<ul class="cmdline">')
      for name, value, annotation in self._funcargs:
        self._print(('  <li>{}={}<span class="annotation">{}</span></li>' if annotation is not inspect.Parameter.empty else '<li>{}={}</li>').format(*(html.escape(str(v)) for v in (name, value, annotation))))
      self._print('</ul>')
    stack.callback(self._print, '</div></body></html>')
    stack.push(self._write_post_mortem)
    return super()._init_context(stack)

  def _print(self, *args, flush=False):
    print(*args, file=self._file)
    if flush:
      self._file.flush()

  def _print_push_context(self, title):
    self._print('<div class="context"><div class="title">{}</div><div class="children">'.format(html.escape(title)), flush=True)

  def _print_pop_context(self):
    self._print('</div><div class="end"></div></div>', flush=True)

  def _print_item(self, level, text, escape=True):
    if escape:
      text = html.escape(text)
    self._print('<div class="item" data-loglevel="{}">{}</div>'.format(LEVELS.index(level), text), flush=True)

  def _write_post_mortem(self, etype, value, tb):
    'write exception nfo to html log'

    if etype in (None, SystemExit):
      return
    if etype in (KeyboardInterrupt, bdb.BdbQuit):
      self.write('error', 'killed by user')
      return

    try:
      msg = ''.join(traceback.format_exception(etype, value, tb))
    except Exception as e:
      msg = '{} (traceback failed: {})'.format(value, e)
    self.write('error', msg)

    _fmt = lambda obj: '=' + ''.join(s.strip() for s in repr(obj).split('\n'))
    self._print('<div class="post-mortem">')
    self._print('EXHAUSTIVE STACK TRACE')
    self._print()
    for frame, filename, lineno, function, code_context, index in inspect.getinnerframes(tb):
      self._print('File "{}", line {}, in {}'.format(filename, lineno, function))
      self._print(html.escape(textwrap.fill(inspect.formatargvalues(*inspect.getargvalues(frame),formatvalue=_fmt), initial_indent=' ', subsequent_indent='  ', width=80)))
      if code_context:
        self._print()
        for line in code_context:
          self._print(html.escape(textwrap.fill(line.strip(), initial_indent='>>> ', subsequent_indent='    ', width=80)))
      self._print()
    self._print('</div>', flush=True)

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    with self._open(filename, mode, exists) as f:
      yield f
    self.write(level, '<a href="{href}">{name}</a>'.format(href=urllib.parse.quote(f.name), name=html.escape(filename)), escape=False)

class IndentLog(ContextTreeLog):
  '''Output indented html snippets.'''

  def __init__(self, outdir, *, progressinterval=None):
    self._outdir = outdir
    self._prefix = ''
    self._progressupdate = 0 # progress update interval in seconds
    self._progressinterval = progressinterval or getattr(config, 'progressinterval', 1)
    super().__init__()

  def _write_post_mortem(self, etype, value, tb):
    if etype in (None, SystemExit):
      return
    elif etype in (KeyboardInterrupt, bdb.BdbQuit):
      self.write('error', 'killed by user')
    else:
      try:
        msg = ''.join(traceback.format_exception(etype, value, tb))
      except Exception as e:
        msg = '{} (traceback failed: {})'.format(value, e)
      self.write('error', msg)

  def _init_context(self, stack):
    self._open = stack.enter_context(_makedirs(self._outdir, exist_ok=True))
    self._logfile = stack.enter_context(self._open('log.html', 'w', exists='overwrite'))
    self._progressfile = stack.enter_context(self._open('progress.json', 'w', exists='overwrite'))
    stack.push(self._write_post_mortem)
    super()._init_context(stack)

  def _print(self, *args, flush=False):
    print(*args, file=self._logfile)
    if flush:
      self._logfile.flush()

  def _print_push_context(self, title):
    title = title.replace('\n', '').replace('\r', '')
    self._print('{}c {}'.format(self._prefix, html.escape(title)), flush=True)
    self._prefix += ' '

  def _print_pop_context(self):
    self._prefix = self._prefix[:-1]

  def _print_item(self, level, text, escape=True):
    if escape:
      text = html.escape(text)
    level = html.escape(level[0])
    for line in text.splitlines():
      self._print('{}{} {}'.format(self._prefix, level, line), flush=True)
      level = '|'
    self._print_progress(level, text)
    self._progressupdate = 0

  def _push_context(self, title, mayskip):
    super()._push_context(title, mayskip)
    from . import parallel
    if parallel.procid:
      return
    t = time.time()
    if t < self._progressupdate:
      return
    self._print_progress(None, None)
    self._progressupdate = t + self._progressinterval

  def _print_progress(self, level, text):
    self._progressfile.seek(0)
    self._progressfile.truncate(0)
    json.dump(dict(logpos=self._logfile.tell(), context=self._context, text=text, level=level), self._progressfile)
    self._progressfile.write('\n')
    self._progressfile.flush()

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    with self._open(filename, mode, exists) as f:
      yield f
    self._print_item(level, '<a href="{href}">{name}</a>'.format(href=urllib.parse.quote(f.name), name=html.escape(filename)), escape=False)

class TeeLog(Log):
  '''Simultaneously interface multiple logs'''

  def __init__(self, *logs):
    self.logs = logs
    super().__init__()

  def _init_context(self, stack):
    for log in self.logs:
      stack.enter_context(log)
    super()._init_context(stack)

  @contextlib.contextmanager
  def context(self, title, mayskip=False):
    with contextlib.ExitStack() as stack:
      for log in self.logs:
        stack.enter_context(log.context(title, mayskip))
      yield

  def write(self, level, text):
    for log in self.logs:
      log.write(level, text)

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    if mode not in ('w', 'wb'):
      raise ValueError('invalid mode: {!r}'.format(mode))
    with contextlib.ExitStack() as stack:
      files = [stack.enter_context(log.open(filename, mode, level, exists)) for log in self.logs]
      files = list(filter(lambda file: not file.devnull, files))
      if len(files) == 0:
        yield stack.enter_context(_devnull(filename, mode))
      elif len(files) == 1:
        yield files[0]
      else:
        f = stack.enter_context(tempfile.NamedTemporaryFile(suffix=filename, mode=mode+'+'))
        f.devnull = False
        yield f
        f.seek(0)
        data = f.read()
        for file in files:
          file.write(data)

class RecordLog(Log):
  '''
  Log object that records log messages.  All messages are forwarded to the log
  that whas active before activating this log (e.g. by ``with RecordLog() as
  record:``).  The recorded messages can be replayed to the log that's
  currently active by :meth:`replay`.

  Typical usage is caching expensive operations::

      # compute
      with RecordLog() as record:
        result = compute_something_expensive()
      raw = pickle.dumps((record, result))
      # reuse
      record, result = pickle.loads(raw)
      record.replay()

  .. Note::
     Instead of using :class:`RecordLog` and :mod:`pickle` manually, as in
     above example, we advice to use :class:`nutils.cache.FileCache` instead.

  .. Note::
     Exceptions raised while in a :meth:`Log.context` are not recorded.

  .. Note::
     Messages dispatched from forks (e.g. inside
     :meth:`nutils.parallel.pariter`) are not recorded.
  '''

  def __init__(self):
    # Replayable log messages.  Each entry is a tuple of `(cmd, *args)`, where
    # `cmd` is either 'entercontext', 'exitcontext' or 'write'.  See
    # `self.replay` below.
    self._messages = []
    # `self._contexts` is a list of entered context titles.  We keep track of
    # the titles because we delay appending the 'entercontext' command until
    # something (nonzero) is written to the log.  This is to exclude progress
    # information.  The `self._appended_contexts` index tracks the number of
    # contexts that we have appended to `self._messages`.
    self._contexts = []
    self._appended_contexts = 0
    super().__init__()

  def __enter__(self):
    self._writethrough_log = _current_log
    return super().__enter__()

  def __exit__(self, *exc_info):
    try:
      del self._writethrough_log
    except AttributeError:
      pass
    return super().__exit__(*exc_info)

  @contextlib.contextmanager
  def context(self, title, mayskip=False):
    self._contexts.append(title)
    # We don't append 'entercontext' here.  See `self.__init__`.
    try:
      with self._writethrough_log.context(title, mayskip):
        yield
    finally:
      self._contexts.pop()
      if self._appended_contexts > len(self._contexts):
        self._appended_contexts -= 1
        self._messages.append(('exitcontext',))

  def write(self, level, text):
    self._writethrough_log.write(level, text)
    from . import parallel
    if not parallel.procid:
      # Append all currently entered contexts that have not been append yet
      # before appending the 'write' entry.
      for title in self._contexts[self._appended_contexts:]:
        self._messages.append(('entercontext', title))
      self._appended_contexts = len(self._contexts)
      self._messages.append(('write', level, text))

  @contextlib.contextmanager
  def open(self, filename, mode, level, exists):
    if mode not in ('w', 'wb'):
      raise ValueError('invalid mode: {!r}'.format(mode))
    for title in self._contexts[self._appended_contexts:]:
      self._messages.append(('entercontext', title))
    self._appended_contexts = len(self._contexts)
    with tempfile.NamedTemporaryFile(suffix=filename, mode=mode+'+') as f:
      f.devnull = False
      yield f
      f.seek(0)
      data = f.read()
    self._messages.append(('open', filename, mode, level, exists, data))
    with self._writethrough_log.open(filename, mode, level, exists) as f:
      if not f.devnull:
        f.write(data)

  def replay(self):
    '''
    Replay this recorded log in the log that's currently active.
    '''
    contexts = []
    for cmd, *args in self._messages:
      if cmd == 'entercontext':
        context = _current_log.context(*args)
        context.__enter__()
        contexts.append(context)
      elif cmd == 'exitcontext':
        contexts.pop().__exit__(None, None, None)
      elif cmd == 'write':
        _current_log.write(*args)
      elif cmd == 'open':
        filename, mode, level, exists, data = args
        with _current_log.open(filename, mode, level, exists) as f:
          f.write(data)

## INTERNAL FUNCTIONS

# Reference to the current log instance.  This is updated by the `Log`'s
# context manager, see `Log` base class.
_current_log = None

def _set_current_log(new_log):
  global _current_log
  _current_log = new_log

# Set a default log instance.
StdoutLog().__enter__()

def _len(iterable):
  '''Return length if available, otherwise None'''

  try:
    return len(iterable)
  except:
    return None

def _print(level, *args):
  return _current_log.write(level, ' '.join(str(arg) for arg in args))

def _devnull(name, mode):
  if mode not in ('w', 'wb'):
    raise ValueError('invalid mode: {!r}'.format(mode))
  f = builtins.open(name, mode, opener=lambda name_, flags: os.open(os.devnull, flags))
  f.devnull = True
  return f

class _makedirs:
  def __init__(self, path, exist_ok=False):
    self.path = path
    self.exist_ok = exist_ok
    super().__init__()
  def __enter__(self):
    os.makedirs(self.path, exist_ok=self.exist_ok)
    if os.open in os.supports_dir_fd and os.listdir in os.supports_fd:
      self.path = os.open(self.path, flags=os.O_RDONLY)
    return self.open
  def __exit__(self, etype, value, tb):
    if isinstance(self.path, int):
      os.close(self.path)
  def _open(self, name, *args):
    return os.open(name, *args, dir_fd=self.path) if isinstance(self.path, int) \
      else os.open(os.path.join(self.path, name), *args)
  def open(self, filename, mode, exists):
    if mode not in ('w', 'wb'):
      raise ValueError('invalid mode: {!r}'.format(mode))
    if exists not in ('overwrite', 'rename', 'skip'):
      raise ValueError('invalid exists: {!r}'.format(exists))
    if exists != 'overwrite':
      listdir = set(os.listdir(self.path))
      if filename in listdir:
        if exists == 'skip':
          return _devnull(filename, mode)
        for filename in map('-{}'.join(os.path.splitext(filename)).format, itertools.count(1)):
          if filename not in listdir:
            break
    f = builtins.open(filename, mode, opener=self._open)
    f.devnull = False
    return f

## MODULE-ONLY METHODS

locals().update({ name: functools.partial(_print, name) for name in LEVELS })

def path(*args):
  warnings.deprecation("log level 'path' will be removed in the future, please use any other log level instead")
  return _print('info', *args)

def range(title, *args):
  '''Progress logger identical to built in range'''

  items = builtins.range(*args)
  for index, item in builtins.enumerate(items):
    with _current_log.context('{} {} ({:.0f}%)'.format(title, item, index*100/len(items)), mayskip=index):
      yield item

def iter(title, iterable, length=None):
  '''Progress logger identical to built in iter'''

  if length is None:
    length = _len(iterable)
  it = builtins.iter(iterable)
  for index in itertools.count():
    text = '{} {}'.format(title, index)
    if length:
      text += ' ({:.0f}%)'.format(100 * index / length)
    with _current_log.context(text, mayskip=index):
      try:
        yield next(it)
      except StopIteration:
        break

def enumerate(title, iterable):
  '''Progress logger identical to built in enumerate'''

  return iter(title, builtins.enumerate(iterable), length=_len(iterable))

def zip(title, *iterables):
  '''Progress logger identical to built in enumerate'''

  lengths = [_len(iterable) for iterable in iterables]
  return iter(title, builtins.zip(*iterables), length=all(lengths) and min(lengths))

def count(title, start=0, step=1):
  '''Progress logger identical to itertools.count'''

  for item in itertools.count(start, step):
    with _current_log.context('{} {}'.format(title, item), mayskip=item!=start):
      yield item

def title(f): # decorator
  '''Decorator, adds title argument with default value equal to the name of the
  decorated function, unless argument already exists. The title value is used
  in a static log context that is destructed with the function frame.'''

  assert getattr(f, '__self__', None) is None, 'cannot decorate bound instance method'
  default = f.__name__
  argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
  if 'title' in argnames:
    index = argnames.index('title')
    if index >= len(argnames) - len(f.__defaults__ or []):
      default = f.__defaults__[index - len(argnames)]
    gettitle = lambda args, kwargs: args[index] if index < len(args) else kwargs.get('title', default)
  else:
    gettitle = lambda args, kwargs: kwargs.pop('title', default)
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    title = gettitle(args, kwargs)
    if title != default:
      warnings.deprecation('title argument will be removed in future')
    with _current_log.context(title):
      return f(*args, **kwargs)
  return wrapped

def context(title, mayskip=False):
  return _current_log.context(title, mayskip)

def withcontext(f):
  '''Decorator; executes the wrapped function in its own logging context.'''

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    with context(f.__name__):
      return f(*args, **kwargs)
  return wrapped

def open(filename, mode, *, level='user', exists='rename'):
  '''Open file in logger-controlled directory.

  Args
  ----
  filename : :class:`str`
  mode : :class:`str`
      Should be either ``'w'`` (text) or ``'wb'`` (binary data).
  level : :class:`str`
      Log level in which the filename is displayed. Default: ``'user'``.
  exists : :class:`str`
      Determines how existence of ``filename`` in the output directory should
      be handled. Valid values are:

      *   ``'overwrite'``: open the file and remove current contents.

      *   ``'rename'``: change the filename by adding the smallest positive
          suffix ``n`` for which ``filename-n.ext`` does not exist.

      *   ``'skip'``: return a dummy file object with attribute ``devnull`` set
          to ``False`` to allow content creation to be skipped altogether.
  '''

  return _current_log.open(filename, mode, level, exists)

# vim:sw=2:sts=2:et
