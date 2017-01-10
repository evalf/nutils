import io, tempfile, os, contextlib
from . import register, unittest
import nutils.log, nutils.core, nutils.parallel

log_stdout = '''\
iterator > iter 0 (17%) > a
iterator > iter 1 (50%) > b
iterator > iter 2 (83%) > c
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
\033[K\033[1;30miterator · iter 0 (17%)\033[0m\r\
\033[K\033[1;30miterator · iter 0 (17%) · \033[0ma
\033[K\033[1;30miterator · iter 1 (50%)\033[0m\r\
\033[K\033[1;30miterator · iter 1 (50%) · \033[0mb
\033[K\033[1;30miterator · iter 2 (83%)\033[0m\r\
\033[K\033[1;30miterator · iter 2 (83%) · \033[0mc
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
'''

log_html = '''\
<li class="context">iterator</li><ul>
<li class="context">iter 0 (17%)</li><ul>
<li class="info">a</li>
</ul>
<li class="context">iter 1 (50%)</li><ul>
<li class="info">b</li>
</ul>
<li class="context">iter 2 (83%)</li><ul>
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
'''

log_indent = '''\
c iterator
 c iter 0 (17%)
  i a
 c iter 1 (50%)
  i b
 c iter 2 (83%)
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
  with open( os.path.join( nutils.core.getoutdir(), 'test.png' ), 'w' ) as f:
    pass
  nutils.log.info( 'test.png' )
  nutils.log.info( 'nonexistent.png' )

@register( 'stdout', nutils.log.StdoutLog, log_stdout )
@register( 'stdout-verbose3', nutils.log.StdoutLog, log_stdout3, verbose=3 )
@register( 'rich_output', nutils.log.RichOutputLog, log_rich_output )
@register( 'html', nutils.log.HtmlLog, log_html )
@register( 'indent', nutils.log.IndentLog, log_indent )
@register( 'indent-progress-seekable', nutils.log.IndentLog, log_indent, progressfile='seekable' )
@register( 'indent-progress-stream', nutils.log.IndentLog, log_indent, progressfile='stream' )
def logoutput( logcls, logout, verbose=len( nutils.log.LEVELS ), progressfile=False ):

  @unittest
  def test():
    with contextlib.ExitStack() as stack:
      __outdir__ = stack.enter_context( tempfile.TemporaryDirectory() )
      __verbose__ = verbose
      # Make sure all progress information is written, regardless the speed of
      # this computer.
      __progressinterval__ = -1
      stream = io.StringIO()
      kwargs = {}
      if logcls in (nutils.log.HtmlLog, nutils.log.IndentLog):
        kwargs.update( logdir=__outdir__ )
      if progressfile == 'seekable':
        kwargs.update( progressfile=stack.enter_context( open( os.path.join( __outdir__, 'progress.json' ), 'w' ) ) )
      elif progressfile == 'stream':
        kwargs.update( progressfile=io.StringIO() )
      elif progressfile is False:
        pass
      else:
        raise ValueError
      __log__ = logcls( stream, **kwargs )
      generate_log()
      assert stream.getvalue() == logout

@register
def tee_stdout_html():

  @unittest
  def test():
    with tempfile.TemporaryDirectory() as __outdir__:
      __verbose__ = len( nutils.log.LEVELS )
      stream_stdout = io.StringIO()
      stream_html = io.StringIO()
      __log__ = nutils.log.TeeLog(
        nutils.log.StdoutLog( stream_stdout ),
        nutils.log.HtmlLog( stream_html, logdir=__outdir__ ))
      generate_log()
      assert stream_stdout.getvalue() == log_stdout
      assert stream_html.getvalue() == log_html
