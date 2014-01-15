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


class SimpleLog( object ):
  'simple log'

  def __init__( self, depth=1 ):
    'constructor'

    sys._getframe(depth).f_locals[_KEY] = self

  def write( self, *chunks ):
    'write'

    mtype, args = chunks[-1]
    s = (_makestr(args),) if args else ()
    print ' > '.join( chunks[:-1] + s )


class HtmlLog( object ):
  'html log'

  def __init__( self, maxlevel, fileobj, title, depth=1 ):
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
    self.maxlevel = maxlevel

    sys._getframe(depth).f_locals[_KEY] = self

  def __del__( self ):
    'destructor'

    self.html.write( '</pre></body></html>\n' )
    self.html.flush()

  def write( self, *chunks ):
    'write'

    mtype, args = chunks[-1]
    levels = 'error', 'warning', 'user', 'path', 'info', 'progress', 'debug'
    try:
      ilevel = levels.index(mtype)
    except:
      ilevel = -1
    if ilevel > self.maxlevel:
      return

    s = (_makestr(args),) if args else ()
    print ' > '.join( chunks[:-1] + s )

    if args:
      if mtype == 'path':
        whitelist = ['.jpg','.png','.svg','.txt'] + getattr( prop, 'plot_extensions', [] )
        args = [ '<a href="%s" name="%s" %s>%s</a>' % (args[0],args[0],'class="plot"' if any(args[0].endswith(ext) for ext in whitelist) else '',args[0]) ] \
             + [ '<a href="%s">%s</a>' % (arg,arg) for arg in args[1:] ]
        last = _makestr( args )
      else:
        last = _makestr( args ).replace( '<', '&lt;' ).replace( '>', '&gt;' )
      if '\n' in last:
        parts = last.split( '\n' )
        last = '\n'.join( [ '<b>%s</b>' % parts[0] ] + parts[1:] )
      s = '<span class="%s">%s</span>' % (mtype,last),
    else:
      s = ()

    self.html.write( '<span class="line">%s</span>' % ' &middot; '.join( chunks[:-1] + s ) + '\n' )
    self.html.flush()


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

  def disable( self ):
    'disable this logger'

    self.__enabled = False

  def write( self, *text ):
    'write'

    if self.__enabled:
      self.parent.write( self.text, *text )
    else:
      self.parent.write( *text )

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

debug   = lambda *args: _findlogger().write( ( 'debug',   args ) )
info    = lambda *args: _findlogger().write( ( 'info',    args ) )
user    = lambda *args: _findlogger().write( ( 'user',    args ) )
error   = lambda *args: _findlogger().write( ( 'error',   args ) )
warning = lambda *args: _findlogger().write( ( 'warning', args ) )
path    = lambda *args: _findlogger().write( ( 'path',    args ) )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
