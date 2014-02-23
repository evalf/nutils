from __future__ import print_function
import sys, time, os, warnings, numeric, re, util

warnings.showwarning = lambda message, category, filename, lineno, *args: \
  warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )


## LOGGERS, STREAMS


def SimpleLog( level ): # just writes to stdout
  sys.stdout.writelines( '%s > ' % context for context in contexts )
  return sys.stdout

class HtmlLog( object ):
  'html log'

  def __init__( self, html ):
    self.html = html

  def __call__( self, level, *contexts ):
    return HtmlStream( level, contexts, self.html )

class HtmlStream( object ):
  'html line stream'

  def __init__( self, level, contexts, html ):
    'constructor'

    sys.stdout.writelines( '%s > ' % context for context in contexts )
    self.level = level
    self.head = ''.join( '%s &middot; ' % context for context in contexts )
    self.body = ''
    self.html = html

  def write( self, text ):
    'write to out and buffer for html'

    sys.stdout.write( text )
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  @staticmethod
  def _path2href( match ):
    whitelist = ['.jpg','.png','.svg','.txt'] + util.prop( 'plot_extensions', [] )
    filename = match.group(0)
    ext = match.group(1)
    return '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
      else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)

  def __del__( self ):
    'postprocess buffer and write to html'

    body = self.body
    if self.level == 'path':
      body = re.sub( r'\b\w+([.]\w+)\b', self._path2href, body )
    if self.level:
      body = '<span class="%s">%s</span>' % ( self.level, body )
    line = '<span class="line">%s</span>' % ( self.head + body )

    self.html.write( line )
    self.html.flush()

class ContextLog( object ):
  'static text with parent'

  def __init__( self, title, parent=None ):
    self.title = title
    self.parent = getlogger() if parent is None else parent

  def __call__( self, level, *contexts ):
    return self.parent( level, self.title, *contexts )

class IterLog( object ):
  'iterable context logger that updates progress info'

  def __init__( self, title, iterator, length=None, parent=None ):
    self.title = title
    self.parent = getlogger() if parent is None else parent
    self.length = length
    self.iterator = iterator
    self.index = -1

    # clock
    self.dt = util.prop( 'progress_interval', 1. )
    self.dtexp = util.prop( 'progress_interval_scale', 2 )
    self.dtmax = util.prop( 'progress_interval_max', numeric.inf )
    self.tnext = time.time() + self.dt

  def mktitle( self ):
    self.tnext = time.time() + self.dt
    return '%s %d' % ( self.title, self.index ) if self.length is None \
      else '%s %d/%d (%d%%)' % ( self.title, self.index, self.length, self.index * 100. / self.length )

  def __iter__( self ):
    self.index = 0
    return self

  def next( self ):
    if time.time() > self.tnext:
      if self.dtexp != 1:
        self.dt = min( self.dt * self.dtexp, self.dtmax )
      self.parent( 'progress' ).write( self.mktitle() + '\n' )
    self.index += 1
    try:
      return self.iterator.next()
    except:
      self.index = -1
      raise

  def __call__( self, level, *contexts ):
    return self.parent( level, self.mktitle(), *contexts ) if self.index >= 0 \
      else self.parent( level, *contexts )


## UTILITY FUNCTIONS


_range = range
def range( title, length, parent=None ):
  return IterLog( title, _iter(xrange(length)), length, parent )

_iter = iter
def iter( title, iterable, length=None, parent=None ):
  return IterLog( title, _iter(iterable), len(iterable) if length is None else length, parent )

_enumerate = enumerate
def enumerate( title, iterable, length=None, parent=None ):
  return IterLog( title, _enumerate(iterable), len(iterable) if length is None else length, parent )

def count( title, start=0, parent=None ):
  from itertools import count
  return IterLog( title, count(start), None, parent )
    
def stack( msg, frames=None ):
  'print stack trace'

  if frames is None:
    from . import debug
    frames = debug.callstack( depth=2 )
  print( msg, *reversed(frames), sep='\n', file=getstream( 'error' ) )

def title( f ): # decorator
  def wrapped( *args, **kwargs ):
    __logger__ = ContextLog( kwargs.pop( 'title', f.func_name ) )
    return f( *args, **kwargs )
  return wrapped

def getlogger():
  return util.prop( 'logger', SimpleLog )

def getstream( level ):
  __logger__ = getlogger()
  return __logger__( level )


path     = lambda *args: print( *args, file=getstream( 'path'     ) )
error    = lambda *args: print( *args, file=getstream( 'error'    ) )
warning  = lambda *args: print( *args, file=getstream( 'warning'  ) )
user     = lambda *args: print( *args, file=getstream( 'user'     ) )
info     = lambda *args: print( *args, file=getstream( 'info'     ) )
progress = lambda *args: print( *args, file=getstream( 'progress' ) )
debug    = lambda *args: print( *args, file=getstream( 'debug'    ) )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
