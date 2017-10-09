import io, tempfile, os
from . import *
import nutils.log, nutils.core, nutils.parallel, nutils.debug

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
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">
<html><head>
<title>test</title>
<script type="text/javascript" src="viewer.js" ></script>
<link rel="stylesheet" type="text/css" href="style.css">
</head><body class="newstyle"><pre>
<ul>
<li class="context">iterator</li><ul>
<li class="context">iter 0 (0%)</li><ul>
<li class="info">a</li>
</ul>
<li class="context">iter 1 (33%)</li><ul>
<li class="info">b</li>
</ul>
<li class="context">iter 2 (67%)</li><ul>
<li class="info">c</li>
</ul>
</ul>
<li class="context">levels</li><ul>
<li class="error">error</li>
<li class="warning">warning</li>
<li class="user">user</li>
<li class="info">info</li>
</ul>
<li class="context">exception</li><ul>
<li class="error">ValueError(&#x27;test&#x27;,)
  File &quot;??&quot;, line ??, in ??
    raise ValueError( &#x27;test&#x27; )</li>
</ul>
<li class="info"><a href="test.png" class="plot">test.png</a></li>
<li class="info">nonexistent.png</li>
</ul>
</pre></body></html>
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

  def test(self):
    __outdir__ = self.outdir
    __verbose__ = self.verbose
    # Make sure all progress information is written, regardless the speed of
    # this computer.
    __progressinterval__ = -1
    stream = io.StringIO()
    kwargs = dict(title='test') if self.logcls == nutils.log.HtmlLog else {}
    if self.progressfileobj is not None:
      kwargs.update(progressfile=self.progressfileobj)
    with self.logcls(stream, **kwargs) as __log__:
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

  def test(self):
    __outdir__ = self.outdir
    __verbose__ = len(nutils.log.LEVELS)
    stream_stdout = io.StringIO()
    stream_html = io.StringIO()
    __log__ = nutils.log.TeeLog(
      nutils.log.StdoutLog(stream_stdout),
      nutils.log.HtmlLog(stream_html, title='test'))
    with __log__:
      generate_log()
    self.assertEqual(stream_stdout.getvalue(), log_stdout)
    self.assertEqual(stream_html.getvalue(), log_html)

class html_post_mortem(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outdir = stack.enter_context(tempfile.TemporaryDirectory())

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
    __outdir__ = self.outdir
    stream = io.StringIO()
    with self.assertRaises(TestException):
      with nutils.log.HtmlLog(stream, title='test') as __log__:
        virtual_module['generate_exception']()
    self.assertIn('<span class="post-mortem">', stream.getvalue())

class move_outdir(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
    self.outdira = os.path.join(tmpdir, 'a')
    self.outdirb = os.path.join(tmpdir, 'b')
    os.mkdir(self.outdira)
    self.outdirfd = os.open(self.outdira, flags=os.O_RDONLY)
    stack.callback(os.close, self.outdirfd)

  def test(self):
    __outdirfd__ = self.outdirfd
    os.rename(self.outdira, self.outdirb)
    __verbose__ = len(nutils.log.LEVELS)
    stream = io.StringIO()
    with nutils.log.HtmlLog(stream, title='test') as __log__:
      generate_log()
    self.assertEqual(stream.getvalue(), log_html)
