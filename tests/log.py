import io
from . import register, unittest
import nutils.log

log_stdout = '''\
iterator > iter 0 (17%) > a
iterator > iter 1 (50%) > b
iterator > iter 2 (83%) > c
empty
levels > path
levels > error
levels > warning
levels > user
levels > info
levels > progress
exception > ValueError('test',)
  File "??", line ??, in ??
    raise ValueError( 'test' )
'''

log_stdout3 = '''\
levels > path
levels > error
levels > warning
exception > ValueError('test',)
  File "??", line ??, in ??
    raise ValueError( 'test' )
'''

log_rich_output = '''\
\033[1;30miterator · iter 0 (17%) · \033[0ma
\033[1;30miterator · iter 1 (50%) · \033[0mb
\033[1;30miterator · iter 2 (83%) · \033[0mc
\033[1;30mempty\033[0m
\033[1;30mlevels · \033[1;32mpath\033[0m
\033[1;30mlevels · \033[1;31merror\033[0m
\033[1;30mlevels · \033[0;31mwarning\033[0m
\033[1;30mlevels · \033[0;33muser\033[0m
\033[1;30mlevels · \033[0minfo
\033[1;30mlevels · \033[0mprogress
\033[1;30mexception · \033[1;31mValueError(\'test\',)
  File "??", line ??, in ??
    raise ValueError( \'test\' )\033[0m
'''

log_html = '''\
<span class="line">iterator &middot; iter 0 (17%) &middot; <span class="info">a</span></span>
<span class="line">iterator &middot; iter 1 (50%) &middot; <span class="info">b</span></span>
<span class="line">iterator &middot; iter 2 (83%) &middot; <span class="info">c</span></span>
<span class="line">levels &middot; <span class="path">path</span></span>
<span class="line">levels &middot; <span class="error">error</span></span>
<span class="line">levels &middot; <span class="warning">warning</span></span>
<span class="line">levels &middot; <span class="user">user</span></span>
<span class="line">levels &middot; <span class="info">info</span></span>
<span class="line">levels &middot; <span class="progress">progress</span></span>
<span class="line">exception &middot; <span class="error">ValueError('test',)
  File "??", line ??, in ??
    raise ValueError( 'test' )</span></span>
'''

def generate_log():
  with nutils.log.context( 'iterator' ):
    for i in nutils.log.iter( 'iter', 'abc' ):
      nutils.log.info( i )
  with nutils.log.context( 'empty' ):
    with nutils.log.context( 'empty' ):
      pass
    nutils.log._getlog().write( 'progress', None )
  with nutils.log.context( 'levels' ):
    for level in ( 'path', 'error', 'warning', 'user', 'info', 'progress' ):
      getattr( nutils.log, level )( level )
  with nutils.log.context( 'exception' ):
    nutils.log.error(
      "ValueError('test',)\n" \
      '  File "??", line ??, in ??\n' \
      "    raise ValueError( 'test' )")

@register( 'stdout', nutils.log.StdoutLog, log_stdout )
@register( 'stdout-verbose3', nutils.log.StdoutLog, log_stdout3, verbose=3 )
@register( 'rich_output', nutils.log.RichOutputLog, log_rich_output )
@register( 'html', nutils.log.HtmlLog, log_html )
def logoutput( logcls, logout, verbose=len( nutils.log.LEVELS ) ):

  @unittest
  def test():
    __verbose__ = verbose
    stream = io.StringIO()
    __log__ = logcls( stream )
    generate_log()
    assert stream.getvalue() == logout

@register
def tee_stdout_html():

  @unittest
  def test():
    __verbose__ = len( nutils.log.LEVELS )
    stream_stdout = io.StringIO()
    stream_html = io.StringIO()
    __log__ = nutils.log.TeeLog(
      nutils.log.StdoutLog( stream_stdout ),
      nutils.log.HtmlLog( stream_html ))
    generate_log()
    assert stream_stdout.getvalue() == log_stdout
    assert stream_html.getvalue() == log_html
