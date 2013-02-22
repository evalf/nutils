from . import core, prop
import sys, time, os, traceback

_KEY = '__logger__'
_makestr = lambda args: ' '.join( str(arg) for arg in args )

def _findlogger( frame=None ):
  'find logger in call stack'

  if frame is None:
    frame = sys._getframe(1)
  while frame:
    if _KEY in frame.f_locals:
      return frame.f_locals[_KEY]
    frame = frame.f_back
  return SimpleLog()

info    = lambda *args: _findlogger().write( 'info',    _makestr(args) )
path    = lambda *args: _findlogger().write( 'path',    _makestr(args) )
error   = lambda *args: _findlogger().write( 'error',   _makestr(args) )
warning = lambda *args: _findlogger().write( 'warning', _makestr(args) )

def context( *args, **kwargs ):
  'context'

  depth = kwargs.pop( 'depth', 0 )
  assert not kwargs
  f_locals = sys._getframe(depth+1).f_locals
  old = f_locals.get(_KEY)
  f_locals[_KEY] = ContextLog( _makestr(args) )
  return old

def restore( logger, depth=0 ):
  'pop context'

  f_locals = sys._getframe(depth+1).f_locals
  if logger:
    f_locals[_KEY] = logger
  else:
    f_locals.pop(_KEY,None)

def iterate( text, iterable, target=None, **kwargs ):
  'iterate'
  
  logger = ProgressLog( text, target if target is not None else len(iterable), **kwargs )
  f_locals = sys._getframe(1).f_locals
  try:
    frame = f_locals[_KEY] = logger
    for i, item in enumerate( iterable ):
      yield item
      logger.update( level='progress', current=i )
  finally:
    frame = f_locals[_KEY] = logger.parent

def exception( exc_info=None ):
  'print traceback'

  exc_type, exc_value, exc_traceback = exc_info or sys.exc_info()
  while exc_traceback.tb_next:
    exc_traceback = exc_traceback.tb_next
  frame = exc_traceback.tb_frame
  parts = traceback.format_exception_only( exc_type, exc_value ) + traceback.format_stack( frame )
  _findlogger( frame ).write( 'error', ''.join( parts ) )

class SimpleLog( object ):
  'simple log'

  def __init__( self ):
    'constructor'

    self.out = getattr( prop, 'html', sys.stdout )

  def write( self, level, *text ):
    'write'

    print ' > '.join( text )

def setup_html( maxlevel, path, title, depth=0 ):
  'setup html logging'

  sys._getframe(depth+1).f_locals[_KEY] = HtmlLog( maxlevel, path, title )

class HtmlLog( object ):
  'html log'

  def __init__( self, maxlevel, path, title ):
    'constructor'

    self.html = open( path, 'w' )
    self.html.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
    self.html.write( '<html><head>\n' )
    self.html.write( '<title>{}</title>\n'.format(title) )
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

  def write( self, level, *chunks ):
    'write'

    if not chunks:
      return
    ilevel = { 'error':0, 'warning':1, 'path':2, 'info':3, 'progress':4, 'debug':5 }.get( level, -1 )
    if ilevel <= self.maxlevel:
      print ' > '.join( chunks )
    if level == 'path':
      chunks = chunks[:-1] + ( '<a href="%s" class="plot">%s</a>' % (chunks[-1],chunks[-1]), )
    else:
      chunks = chunks[:-1] + ( '<span class="%s">%s</span>' % (level,chunks[-1]), )
    self.html.write( ' &middot; '.join( chunks ) + '\n' )
    self.html.flush()

class ContextLog( object ):
  'simple text logger'

  def __init__( self, text ):
    'constructor'

    self.text = text
    self.parent = _findlogger()

  def write( self, level, *text ):
    'write'

    self.parent.write( level, self.text, *text )

class ProgressLog( object ):
  'progress bar'

  def __init__( self, text, target, showpct=True, tint=1, texp=2 ):
    'constructor'

    self.text = text
    self.showpct = showpct
    self.tint = tint
    self.texp = texp
    self.t0 = time.time()
    self.tnext = self.t0 + self.tint
    self.target = target
    self.current = 0
    self.parent = _findlogger()

  def update( self, level, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      self.write( level )

  def write( self, level, *text ):
    'write'

    self.tint *= self.texp
    self.tnext = time.time() + self.tint
    pbar = self.text + ' %.0f/%.0f' % ( self.current, self.target )
    if self.showpct:
      pct = 100 * self.current / float(self.target)
      pbar += ' (%.0f%%)' % pct
    self.parent.write( level, pbar, *text )

# DEPRECATED

progress = info

class ProgressBar( object ):
  'temporary construct for backwards compatibility'

  @core.deprecated( old='ProgressBar(n,text)', new='iterate(text,n)' )
  def __init__( self, n, title ):
    'constructor'

    self.text = title
    if isinstance( n, int ):
      self.iterable = None
      self.target = n
    else:
      self.iterable = n
      self.target = len(n)

  def __iter__( self ):
    'iterate'

    return iter( iterate( self.text, self.iterable, self.target ) )

  def add( self, text ):
    'add to text'

    self.text += ' %s' % text

  def close( self ):
    'does nothing'

    pass

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
