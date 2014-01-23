from __future__ import print_function
from . import prop, debug
import sys, time, os, warnings, numpy

_KEY = '__logger__'
_makestr = lambda args: ' '.join( str(arg) for arg in args )


def _backtrace( frame ):
  while frame:
    yield frame
    frame = frame.f_back


def _findlogger( frame=None ):
  'find logger in call stack'

  for frame in _backtrace( frame or sys._getframe(1) ):
    logger = frame.f_locals.get(_KEY)
    if logger:
      return logger
  return SimpleLog()


warnings.showwarning = lambda message, category, filename, lineno, *args: warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )


def stack( msg, frames=None ):
  'print stack trace'

  if frames is None:
    frames = debug.callstack( depth=2 )
  summary = '\n'.join( [ msg ] + [ str(f) for f in reversed(frames) ] )
  _findlogger( frames[-1].frame ).write( ('error',[summary ]) )


class BlackHoleStream( object ):
  @staticmethod
  def write( *args ):
    pass


class SimpleLog( object ):
  'simple log'

  def __init__( self, depth=1 ):
    'constructor'

    sys._getframe(depth).f_locals[_KEY] = self

  @staticmethod
  def getstream( *chunks ):
    
    _levels = 'path', 'error', 'warning', 'user', 'info', 'debug'
    level = _levels.index( chunks[-1] )
    if level >= getattr( prop, 'verbose', 9 ):
      return BlackHoleStream
    sys.stdout.write( ' > '.join( chunks[:-1] + ('',) ) )
    return sys.stdout


class HtmlStream( object ):
  'html line stream'

  def __init__( self, chunks, html ):
    'constructor'

    self.out = SimpleLog.getstream( *chunks )
    self.mtype = chunks[-1]
    self.head = ' &middot; '.join( chunks[:-1] + ('',) )
    self.body = ''
    self.html = html

  def write( self, text ):
    'write to out and buffer for html'

    self.out.write( text )
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  def __del__( self ):
    'postprocess buffer and write to html'

    body = self.body
    if self.mtype == 'path':
      whitelist = ['.jpg','.png','.svg','.txt'] + getattr( prop, 'plot_extensions', [] )
      hrefs = []
      for filename in body.split( ', ' ):
        root, ext = os.path.splitext( filename )
        href = '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
          else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)
        hrefs.append( href )
      body = ', '.join( hrefs )
    line = '<span class="line">%s<span class="%s">%s</span></span>' % ( self.head, self.mtype, body )

    self.html.write( line )
    self.html.flush()


class HtmlLog( object ):
  'html log'

  def __init__( self, fileobj, title, depth=1 ):
    'constructor'

    self.html = fileobj
    self.html.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
    self.html.write( '<html><head>\n' )
    self.html.write( '<title>%s</title>\n' % title )
    self.html.write( '<script type="text/javascript" src="../../../../../viewer.js" ></script>\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../style.css">\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../custom.css">\n' )
    self.html.write( '</head><body><pre>\n' )
    self.html.write( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html">latest %s</a> | <a class="nav_latestall" href="../../../../../log.html">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>\n\n' % title.split()[0] )
    self.html.flush()

    sys._getframe(depth).f_locals[_KEY] = self

  def getstream( self, *chunks ):
    return HtmlStream( chunks, self.html )

  def __del__( self ):
    'destructor'

    self.html.write( '</pre></body></html>\n' )
    self.html.close()
    

class ContextLog( object ):
  'base class'

  def __init__( self, depth=1 ):
    'constructor'

    frame = sys._getframe(depth)

    parent = _findlogger( frame )
    if isinstance( parent, ContextLog ) and not parent.__enabled:
      parent = parent.parent

    self.parent = parent
    self.__enabled = True

    frame.f_locals[_KEY] = self

  def getstream( self, *chunks ):
    return self.parent.getstream( self.text, *chunks ) if self.__enabled \
      else self.parent.getstream( *chunks )

  def disable( self ):
    'disable this logger'

    self.__enabled = False

  def __repr__( self ):
    return '%s(%s)' % ( self.__class__.__name__, self )

  def __str__( self ):
    return '%s > %s' % ( self.parent, self.text ) if self.__enabled else str(self.parent)


class StaticContextLog( ContextLog ):
  'simple text logger'

  def __init__( self, text, depth=1 ):
    'constructor'

    self.text = text
    ContextLog.__init__( self, depth=depth+1 )


class ProgressContextLog( ContextLog ):
  'progress bar'

  def __init__( self, text, iterable=None, target=None, showpct=True, depth=1 ):
    'constructor'

    self.msg = text
    self.showpct = showpct
    self.tint = getattr(prop,'progress_interval',1.)
    self.tmax = getattr(prop,'progress_interval_max',numpy.inf)
    self.texp = getattr(prop,'progress_interval_scale',2.)
    self.t0 = time.time()
    self.tnext = self.t0 + min( self.tint, self.tmax )
    self.iterable = iterable
    self.target = len(iterable) if target is None else target
    self.current = 0
    ContextLog.__init__( self, depth=depth+1 )

  def __iter__( self ):
    try:
      for i, item in enumerate( self.iterable ):
        self.update( i )
        yield item
    finally:
      self.disable()

  def update( self, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      self.write( ('progress',None) )

  @property
  def text( self ):
    'get text'

    self.tint = min( self.tint*self.texp, self.tmax )
    self.tnext = time.time() + self.tint
    pbar = self.msg + ' %.0f/%.0f' % ( self.current, self.target )
    if self.showpct:
      pct = 100 * self.current / float(self.target)
      pbar += ' (%.0f%%)' % pct
    return pbar


# historically grown
context = StaticContextLog
progress = iterate = ProgressContextLog
setup_html = HtmlLog


def getstream( level ):
  return _findlogger().getstream( level )

def _mklog( level ):
  return lambda *args: print( *args, file=getstream(level) )

path    = _mklog( 'path'    )
error   = _mklog( 'error'   )
warning = _mklog( 'warning' )
user    = _mklog( 'user'    )
info    = _mklog( 'info'    )
debug   = _mklog( 'debug'   )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
