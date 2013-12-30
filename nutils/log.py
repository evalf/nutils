from . import core, prop, debug
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

debug   = lambda *args: _findlogger().write( ( 'debug',   args ) )
info    = lambda *args: _findlogger().write( ( 'info',    args ) )
user    = lambda *args: _findlogger().write( ( 'user',    args ) )
error   = lambda *args: _findlogger().write( ( 'error',   args ) )
warning = lambda *args: _findlogger().write( ( 'warning', args ) )
path    = lambda *args: _findlogger().write( ( 'path',    args ) )

warnings.showwarning = lambda message, category, filename, lineno, *args: warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )

def context( *args, **kwargs ):
  'context'

  depth = kwargs.pop( 'depth', 0 )
  assert not kwargs
  frame = sys._getframe(depth+1)
  old = frame.f_locals.get(_KEY)
  new = ContextLog( _makestr(args) )
  frame.f_locals[_KEY] = new
  return new, old

def restore( (new,old) ):
  'pop context'

  for frame in _backtrace( sys._getframe(1) ):
    logger = frame.f_locals.get(_KEY)
    if logger:
      break
  else:
    warnings.warn( 'failed to restore log context: no log instance found' )
    return

  if logger != new:
    warnings.warn( 'failed to restore log context: unexpected log instance found' )
    return

  frame.f_locals[_KEY] = old

def iterate( text, iterable, target=None, **kwargs ):
  'iterate'
  
  logger = ProgressLog( text, target if target is not None else len(iterable), **kwargs )
  f_locals = sys._getframe(1).f_locals
  try:
    frame = f_locals[_KEY] = logger
    for i, item in enumerate( iterable ):
      logger.update( i )
      yield item
  finally:
    frame = f_locals[_KEY] = logger.parent

def stack( msg, frames=None ):
  'print stack trace'

  if frames is None:
    frames = debug.callstack( depth=2 )
  summary = '\n'.join( [ msg ] + [ str(f) for f in reversed(frames) ] )
  _findlogger( frames[-1].frame ).write( ('error',[summary ]) )

class SimpleLog( object ):
  'simple log'

  def __init__( self ):
    'constructor'

    self.out = getattr( prop, 'html', sys.stdout )

  def write( self, *chunks ):
    'write'

    mtype, args = chunks[-1]
    s = (_makestr(args),) if args else ()
    print ' > '.join( chunks[:-1] + s )

def setup_html( maxlevel, fileobj, title, depth=0 ):
  'setup html logging'

  sys._getframe(depth+1).f_locals[_KEY] = HtmlLog( maxlevel, fileobj, title )

class HtmlLog( object ):
  'html log'

  def __init__( self, maxlevel, fileobj, title ):
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
  'simple text logger'

  def __init__( self, text ):
    'constructor'

    self.text = text
    self.parent = _findlogger()

  def write( self, *text ):
    'write'

    self.parent.write( self.text, *text )

  def __repr__( self ):
    'string representation'

    return 'ContextLog(%s)' % self

  def __str__( self ):

    return '%s > %s' % ( self.parent, self.text )

class ProgressLog( object ):
  'progress bar'

  def __init__( self, text, target, showpct=True ):
    'constructor'

    self.text = text
    self.showpct = showpct
    self.tint = getattr(prop,'progress_interval',1.)
    self.tmax = getattr(prop,'progress_interval_max',numpy.inf)
    self.texp = getattr(prop,'progress_interval_scale',2.)
    self.t0 = time.time()
    self.tnext = self.t0 + min( self.tint, self.tmax )
    self.target = target
    self.current = 0
    self.parent = _findlogger()

  def update( self, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      self.write( ('progress',None) )

  def write( self, *text ):
    'write'

    self.tint = min( self.tint*self.texp, self.tmax )
    self.tnext = time.time() + self.tint
    pbar = self.text + ' %.0f/%.0f' % ( self.current, self.target )
    if self.showpct:
      pct = 100 * self.current / float(self.target)
      pbar += ' (%.0f%%)' % pct
    self.parent.write( pbar, *text )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
