import io, tempfile, os
from . import *
import nutils.log, nutils.core, nutils.util, nutils.parallel

log_stdout = '''\
iterator > iter 0 (0%) > a
iterator > iter 1 (33%) > b
iterator > iter 2 (67%) > c
levels > error
levels > warning
levels > user
levels > info
forked > [1] error
forked > [1] warning
forked > [1] user
forked > [1] info
exception > ValueError('test',)
  File "??", line ??, in ??
    raise ValueError( 'test' )
test.png
nonexistent.png
'''

log_stdout3 = '''\
levels > error
levels > warning
levels > user
forked > [1] error
forked > [1] warning
forked > [1] user
exception > ValueError('test',)
  File "??", line ??, in ??
    raise ValueError( 'test' )
'''

log_rich_output = '''\
\033[K\033[1;30miterator\033[0m\r\
\033[K\033[1;30miterator · iter 0 (0%)\033[0m\r\
\033[K\033[1;30miterator · iter 0 (0%) · \033[0ma
\033[K\033[1;30miterator · iter 1 (33%)\033[0m\r\
\033[K\033[1;30miterator · iter 1 (33%) · \033[0mb
\033[K\033[1;30miterator · iter 2 (67%)\033[0m\r\
\033[K\033[1;30miterator · iter 2 (67%) · \033[0mc
\033[K\033[1;30miterator · iter 3 (100%)\033[0m\r\
\033[K\033[1;30mempty\033[0m\r\
\033[K\033[1;30mempty · empty\033[0m\r\
\033[K\033[1;30mlevels\033[0m\r\
\033[K\033[1;30mlevels · \033[1;31merror\033[0m
\033[K\033[1;30mlevels · \033[0;31mwarning\033[0m
\033[K\033[1;30mlevels · \033[0;33muser\033[0m
\033[K\033[1;30mlevels · \033[0minfo
\033[K\033[1;30mforked · \033[1;31m[1] error\033[0m
\033[K\033[1;30mforked · \033[0;31m[1] warning\033[0m
\033[K\033[1;30mforked · \033[0;33m[1] user\033[0m
\033[K\033[1;30mforked · \033[0m[1] info
\033[K\033[1;30mexception\033[0m\r\
\033[K\033[1;30mexception · \033[1;31mValueError(\'test\',)
  File "??", line ??, in ??
    raise ValueError( \'test\' )\033[0m
\033[K\033[1;30m\033[0mtest.png
\033[K\033[1;30m\033[0mnonexistent.png
\033[K'''

log_html = '''\
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, user-scalable=no"/>
<title>test</title>
<script src="viewer.js"></script>
<link rel="stylesheet" type="text/css" href="viewer.css"/>
</head><body>
<div id="log">
<div class="context"><div class="title">iterator</div><div class="children">
<div class="context"><div class="title">iter 0 (0%)</div><div class="children">
<div class="item" data-loglevel="3">a</div>
</div><div class="end"></div></div>
<div class="context"><div class="title">iter 1 (33%)</div><div class="children">
<div class="item" data-loglevel="3">b</div>
</div><div class="end"></div></div>
<div class="context"><div class="title">iter 2 (67%)</div><div class="children">
<div class="item" data-loglevel="3">c</div>
</div><div class="end"></div></div>
</div><div class="end"></div></div>
<div class="context"><div class="title">levels</div><div class="children">
<div class="item" data-loglevel="0">error</div>
<div class="item" data-loglevel="1">warning</div>
<div class="item" data-loglevel="2">user</div>
<div class="item" data-loglevel="3">info</div>
</div><div class="end"></div></div>
<div class="context"><div class="title">exception</div><div class="children">
<div class="item" data-loglevel="0">ValueError(&#x27;test&#x27;,)
  File &quot;??&quot;, line ??, in ??
    raise ValueError( &#x27;test&#x27; )</div>
</div><div class="end"></div></div>
<div class="item" data-loglevel="3"><a href="test.png" class="plot">test.png</a></div>
<div class="item" data-loglevel="3">nonexistent.png</div>
</div>
</body></html>
'''

log_indent = '''\
c iterator
 c iter 0 (0%)
  i a
 c iter 1 (33%)
  i b
 c iter 2 (67%)
  i c
c levels
 e error
 w warning
 u user
 i info
c exception
 e ValueError(&#x27;test&#x27;,)
 |   File &quot;??&quot;, line ??, in ??
 |     raise ValueError( &#x27;test&#x27; )
i <a href="test.png" class="plot">test.png</a>
i nonexistent.png
'''

def generate_log():
  with nutils.log.context( 'iterator' ):
    for i in nutils.log.iter( 'iter', 'abc' ):
      nutils.log.info( i )
  with nutils.log.context( 'empty' ):
    with nutils.log.context( 'empty' ):
      pass
  with nutils.log.context( 'levels' ):
    for level in ( 'error', 'warning', 'user', 'info' ):
      getattr( nutils.log, level )( level )
  nutils.parallel.procid = 1
  with nutils.log.context( 'forked' ):
    for level in ( 'error', 'warning', 'user', 'info' ):
      getattr( nutils.log, level )( level )
  nutils.parallel.procid = None
  with nutils.log.context( 'exception' ):
    nutils.log.error(
      "ValueError('test',)\n" \
      '  File "??", line ??, in ??\n' \
      "    raise ValueError( 'test' )")
  with nutils.core.open_in_outdir( 'test.png', 'w' ) as f:
    pass
  nutils.log.info( 'test.png' )
  nutils.log.info( 'nonexistent.png' )

@parametrize
class logoutput(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outdir = stack.enter_context(tempfile.TemporaryDirectory())
    if self.progressfile == 'seekable':
      self.progressfileobj = stack.enter_context(open(os.path.join(self.outdir, 'progress.json'), 'w' ))
    elif self.progressfile == 'stream':
      self.progressfileobj = progressfile=io.StringIO()
    else:
      self.progressfileobj = None
    stack.enter_context(nutils.config(
      outdir=self.outdir,
      verbose=self.verbose,
      # Make sure all progress information is written, regardless the speed of
      # this computer.
      progressinterval=-1,
    ))

  def test(self):
    stream = io.StringIO()
    kwargs = dict(title='test') if self.logcls == nutils.log.HtmlLog else {}
    if self.progressfileobj is not None:
      kwargs.update(progressfile=self.progressfileobj)
    with self.logcls(stream, **kwargs):
      generate_log()
    self.assertEqual(stream.getvalue(), self.logout)

_logoutput = lambda name, logcls, logout, verbose=len(nutils.log.LEVELS), progressfile=False: logoutput(name, logcls=logcls, logout=logout, verbose=verbose, progressfile=progressfile)
_logoutput('stdout', nutils.log.StdoutLog, log_stdout)
_logoutput('stdout-verbose3', nutils.log.StdoutLog, log_stdout3, verbose=3)
_logoutput('rich_output', nutils.log.RichOutputLog, log_rich_output)
_logoutput('html', nutils.log.HtmlLog, log_html)
_logoutput('indent', nutils.log.IndentLog, log_indent)
_logoutput('indent-progress-seekable', nutils.log.IndentLog, log_indent, progressfile='seekable')
_logoutput('indent-progress-stream', nutils.log.IndentLog, log_indent, progressfile='stream')

class tee_stdout_html(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outdir = stack.enter_context(tempfile.TemporaryDirectory())
    stack.enter_context(nutils.config(
      outdir=self.outdir,
      verbose=len(nutils.log.LEVELS),
    ))

  def test(self):
    stream_stdout = io.StringIO()
    stream_html = io.StringIO()
    with nutils.log.TeeLog(nutils.log.StdoutLog(stream_stdout), nutils.log.HtmlLog(stream_html, title='test')):
      generate_log()
    self.assertEqual(stream_stdout.getvalue(), log_stdout)
    self.assertEqual(stream_html.getvalue(), log_html)

class html_post_mortem(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outdir = stack.enter_context(tempfile.TemporaryDirectory())
    stack.enter_context(nutils.config(
      outdir=self.outdir,
      verbose=len(nutils.log.LEVELS),
    ))

  def test(self):
    class TestException(Exception): pass

    virtual_module = dict(TestException=TestException)
    exec('''\
def generate_exception(level=0):
  if level == 1:
    raise TestException
  else:
    generate_exception(level+1)
''', virtual_module)
    stream = io.StringIO()
    with self.assertRaises(TestException):
      with nutils.log.HtmlLog(stream, title='test'):
        virtual_module['generate_exception']()
    self.assertIn('<div class="post-mortem">', stream.getvalue())

class move_outdir(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
    self.outdira = os.path.join(tmpdir, 'a')
    self.outdirb = os.path.join(tmpdir, 'b')
    os.mkdir(self.outdira)
    self.outdirfd = os.open(self.outdira, flags=os.O_RDONLY)
    stack.callback(os.close, self.outdirfd)
    stack.enter_context(nutils.config(
      outdirfd=self.outdirfd,
      verbose=len(nutils.log.LEVELS),
    ))

  @unittest.skipIf(not nutils.util.supports_outdirfd, 'outdirfd is not supported on this platform')
  def test(self):
    os.rename(self.outdira, self.outdirb)
    stream = io.StringIO()
    with nutils.log.HtmlLog(stream, title='test'):
      generate_log()
    self.assertEqual(stream.getvalue(), log_html)

class log_context_manager(TestCase):

  def test_reenter(self):
    log = nutils.log.StdoutLog()
    with log:
      with self.assertRaises(RuntimeError):
        with log:
          pass

  def test_exit_before_enter(self):
    log = nutils.log.StdoutLog()
    with self.assertRaises(RuntimeError):
      log.__exit__(None, None, None)

class log_module_funcs(TestCase):

  @contextlib.contextmanager
  def assertLogs(self, desired):
    stream = io.StringIO()
    with nutils.log.StdoutLog(stream):
      yield
    self.assertEqual(stream.getvalue(), desired)

  def test_range_1(self):
    with self.assertLogs('x 0 (0%) > 0\nx 1 (50%) > 1\n'):
      for item in nutils.log.range('x', 2):
        nutils.log.user(str(item))

  def test_range_2(self):
    with self.assertLogs('x 1 (0%) > 1\nx 2 (50%) > 2\n'):
      for item in nutils.log.range('x', 1, 3):
        nutils.log.user(str(item))

  def test_range_3(self):
    with self.assertLogs('x 5 (0%) > 5\nx 3 (50%) > 3\n'):
      for item in nutils.log.range('x', 5, 1, -2):
        nutils.log.user(str(item))

  def test_iter_known_length(self):
    with self.assertLogs('x 0 (0%) > 0\nx 1 (50%) > 1\n'):
      for item in nutils.log.iter('x', [0, 1]):
        nutils.log.user(str(item))

  def test_iter_unknown_length(self):
    def items():
      yield 0
      yield 1
    with self.assertLogs('x 0 > 0\nx 1 > 1\n'):
      for item in nutils.log.iter('x', items()):
        nutils.log.user(str(item))

  def test_enumerate(self):
    with self.assertLogs('x 0 (0%) > a\nx 1 (50%) > b\n'):
      for i, v in nutils.log.enumerate('x', 'ab'):
        nutils.log.user(v)

  def test_zip_known_length(self):
    with self.assertLogs('x 0 (0%) > ax\nx 1 (50%) > by\n'):
      for v0, v1 in nutils.log.zip('x', 'ab', 'xyz'):
        nutils.log.user(v0+v1)

  def test_zip_unknown_length(self):
    def items():
      yield 'x'
      yield 'y'
      yield 'z'
    with self.assertLogs('x 0 > ax\nx 1 > by\n'):
      for v0, v1 in nutils.log.zip('x', 'ab', items()):
        nutils.log.user(v0+v1)

  def test_count(self):
    with self.assertLogs('x 0 > 0\nx 1 > 1\n'):
      count = nutils.log.count('x')
      j = 0
      for i in nutils.log.count('x'):
        nutils.log.user(str(i))
        if j == 1:
          break
        j += 1

  def test_title_noarg(self):
    @nutils.log.title
    def x():
      nutils.log.user('y')
    with self.assertLogs('x > y\n'):
      x()

  def test_title_arg_default(self):
    @nutils.log.title
    def x(title='default'):
      nutils.log.user('y')
    with self.assertLogs('default > y\n'):
      x()

  def test_title_arg_nodefault(self):
    @nutils.log.title
    def x(title):
      nutils.log.user('y')
    with self.assertLogs('arg > y\n'):
      x('arg')

  def test_title_varkw(self):
    @nutils.log.title
    def x(**kwargs):
      nutils.log.user('y')
    with self.assertLogs('arg > y\n'):
      x(title='arg')

  def test_context(self):
    with self.assertLogs('x > y\n'):
      with nutils.log.context('x'):
        nutils.log.user('y')
